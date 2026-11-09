"""Source-conditioned role prediction and the role decoder.

At the 12 mm and 6 mm levels of the local encoder, the role head predicts for every source the logits of the three
roles (target error, other errors, correct tissue) at every position, from the source vector, the encoder feature
at that position and the position relative to the source (:class:`SourceRoleHead`)::

    k_v     = LN(Conv1^3(F_v))                                   F = level-2 (12 mm) or level-1 (6 mm) features
    h_jv    = GELU(W_k k_v + W_s s_j + W_g g_jv + b)             s_j source vector, g_jv relative geometry (8)
    l12_jv  = W_o h_jv + b_o + [s_j . k_v / sqrt(128), 0, 0]     (the dot product on the target-error logit only)
    l6_jv   = U(l12_j)(v) + W6_o h6_jv + b6_o                    U: trilinear upsampling, W6_o and b6_o start at zero

W_k, W_s, W_g and b are the column blocks of one ``nn.Linear(128 + 128 + 8, 64)`` shared by both levels; each level
has its own key projection and normalisation. The target margin of a source at a position is
``l_target - logsumexp(l_other, l_correct)``, its log-odds of the target role. At each position the valid source
with the largest margin passes its log-softmax role map into the decoder (:func:`select_source`; ties go to the
smallest index). All role arithmetic runs in FP32 with autocast disabled.

The role decoder (:class:`RoleDecoder`) upsamples the local features with transposed convolutions, concatenates
the encoder skip features, the whole-volume features (at 12 mm) and the selected role maps at each level, and
outputs decoder features and the coarse role logits on the 3 mm grid.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..data.geometry import MAX_SOURCES
from ..data.observations import DISTANCE_SCALE_MM
from ..data.roles import CORRECT_TISSUE, OTHER_ERRORS, ROLE_COUNT, TARGET_ERROR
from .common import (DepthwiseConvBlock3d, ResidualBlock3d, fix_correct_tissue_logit, fp32_region, run_checkpointed)
from .source_encoder import SOURCE_WIDTH

__all__ = ["GEOMETRY_WIDTH", "LEVELS", "RoleDecoder", "SourceRoleHead", "role_margin", "role_usage_statistics",
           "select_source", "selection_statistics", "relative_source_geometry"]

KEY_WIDTH = 128
HIDDEN_WIDTH = 64
#: Relative geometry of a source at a position: displacement / examination extent (3), normalised log distance (1),
#: source extent / examination extent (3), fraction of the source's voxels inside the scribble window (1).
GEOMETRY_WIDTH = 8
#: Column blocks of the shared hidden layer (keys, source vectors, geometry).
KEY_BLOCK = (0, KEY_WIDTH)
SOURCE_BLOCK = (KEY_WIDTH, KEY_WIDTH + SOURCE_WIDTH)
GEOMETRY_BLOCK = (KEY_WIDTH + SOURCE_WIDTH, KEY_WIDTH + SOURCE_WIDTH + GEOMETRY_WIDTH)
#: The two role levels: local-encoder level 2 (12 mm cells) and level 1 (6 mm cells).
LEVELS = ("12mm", "6mm")


def role_margin(logits: torch.Tensor) -> torch.Tensor:
    """Target margin ``l_target - logsumexp(l_other, l_correct)`` per source and position ([B,J,3,...] -> [B,J,...])."""
    return logits[:, :, TARGET_ERROR] - torch.logsumexp(logits[:, :, [OTHER_ERRORS, CORRECT_TISSUE]], dim=2)


def select_source(logits: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(log-softmax of the selected source's role logits [B,3,...], selected source index [B,...]).

    ``logits`` [B,J,3,...]; the selected source has the largest target margin among the VALID sources, equal margins
    go to the smallest index and an invalid source is never selected. Only the selected logits receive a gradient.
    """
    if logits.ndim < 4 or logits.shape[2] != ROLE_COUNT or valid.shape != logits.shape[:2]:
        raise ValueError(f"source role logits actual={tuple(logits.shape)}, valid={tuple(valid.shape)}")
    if not bool(valid.any(dim=1).all()):
        raise ValueError("source selection needs at least one valid source per batch item")
    spatial = logits.shape[3:]
    expand = (slice(None), slice(None)) + (None,) * len(spatial)
    margin = role_margin(logits)
    margin = margin.masked_fill(~valid[expand], -torch.inf)
    best = margin.max(dim=1, keepdim=True).values
    order = torch.arange(logits.shape[1], device=logits.device).view(1, -1, *(1,) * len(spatial))
    selected = torch.where(margin == best, order, torch.full_like(order, logits.shape[1])).amin(dim=1)
    index = selected[:, None, None].expand(-1, 1, ROLE_COUNT, *spatial)
    chosen = logits.gather(1, index)[:, 0]
    return F.log_softmax(chosen, dim=1), selected


def relative_source_geometry(cell_world: torch.Tensor, representative: torch.Tensor, examination_extent: torch.Tensor,
                             examination_diagonal: torch.Tensor, source_extent: torch.Tensor,
                             window_fraction: torch.Tensor) -> torch.Tensor:
    """The geometry g [B,J,8,N] (FP32) of every source at N cell centres.

    ``cell_world`` [B,N,3] float64 world centres of this window's cells; ``representative`` [B,J,3] float64;
    ``examination_extent`` [B,3]; ``examination_diagonal`` [B]; ``source_extent`` [B,J,3] (already divided by the
    examination extent); ``window_fraction`` [B,J]. The distance is normalised like the distance channels,
    log1p(d / 3 mm) / log1p(examination diagonal / 3 mm).
    """
    difference = cell_world[:, None, :, :] - representative[:, :, None, :]            # [B,J,N,3]
    displacement = difference / examination_extent[:, None, None, :]
    distance = torch.sqrt((difference * difference).sum(-1))                          # [B,J,N]
    log_distance = (torch.log1p(distance / DISTANCE_SCALE_MM)
                    / torch.log1p(examination_diagonal / DISTANCE_SCALE_MM)[:, None, None])
    n = cell_world.shape[1]
    extent = source_extent[:, :, None, :].expand(-1, -1, n, -1).to(torch.float64)
    fraction = window_fraction[:, :, None, None].expand(-1, -1, n, 1).to(torch.float64)
    geometry = torch.cat((displacement, log_distance[..., None], extent, fraction), dim=-1)
    return geometry.permute(0, 1, 3, 2).float()


class SourceRoleHead(nn.Module):
    """Source-conditioned role prediction at 12 mm and 6 mm (module docstring)."""

    def __init__(self, *, features_12mm: int = 96, features_6mm: int = 48, correct_tissue_role: bool = True) -> None:
        super().__init__()
        self.correct_tissue_role = bool(correct_tissue_role)
        self.key_projection_12mm = nn.Conv3d(features_12mm, KEY_WIDTH, 1, bias=False)
        self.key_norm_12mm = nn.LayerNorm(KEY_WIDTH)
        self.hidden = nn.Linear(KEY_WIDTH + SOURCE_WIDTH + GEOMETRY_WIDTH, HIDDEN_WIDTH)
        self.output_12mm = nn.Linear(HIDDEN_WIDTH, ROLE_COUNT)
        self.key_projection_6mm = nn.Conv3d(features_6mm, KEY_WIDTH, 1, bias=False)
        self.key_norm_6mm = nn.LayerNorm(KEY_WIDTH)
        self.output_6mm = nn.Linear(HIDDEN_WIDTH, ROLE_COUNT)
        # The 6 mm output starts at zero: initially the 6 mm logits equal the upsampled 12 mm logits.
        with torch.no_grad():
            self.output_6mm.weight.zero_()
            self.output_6mm.bias.zero_()

    def _block(self, block):
        return self.hidden.weight[:, block[0]:block[1]]

    def key_features(self, features: torch.Tensor, *, level: str) -> torch.Tensor:
        """k = LN(Conv1^3(F)) as [B,N,128] of one level's encoder features."""
        if level == LEVELS[0]:
            projection, norm = self.key_projection_12mm, self.key_norm_12mm
        elif level == LEVELS[1]:
            projection, norm = self.key_projection_6mm, self.key_norm_6mm
        else:
            raise ValueError(f"role level actual={level!r}, expected one of {LEVELS}")
        with fp32_region(features):
            return norm(projection(features.float()).flatten(2).transpose(1, 2))

    def key_term_from(self, keys: torch.Tensor) -> torch.Tensor:
        """W_k k as [B,64,N]."""
        with fp32_region(keys):
            return torch.einsum("bnk,hk->bhn", keys, self._block(KEY_BLOCK))

    def key_term(self, features: torch.Tensor, *, level: str) -> torch.Tensor:
        """W_k LN(Conv1^3(F)) as [B,64,N] (computed once per window and level)."""
        return self.key_term_from(self.key_features(features, level=level))

    def similarity(self, source_vectors: torch.Tensor, valid: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """s_j . k_v / sqrt(128) as [B,J,N] (no parameter)."""
        with fp32_region(source_vectors):
            safe = torch.where(valid[..., None], source_vectors.float(), torch.zeros_like(source_vectors.float()))
            return torch.einsum("bjq,bnq->bjn", safe, keys.float()) / math.sqrt(KEY_WIDTH)

    def source_term(self, source_vectors: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """W_s s_j + b as [B,J,64]; an invalid slot's source vector is zero (and it is never selected)."""
        with fp32_region(source_vectors):
            safe = torch.where(valid[..., None], source_vectors.float(), torch.zeros_like(source_vectors.float()))
            return torch.einsum("bjq,hq->bjh", safe, self._block(SOURCE_BLOCK)) + self.hidden.bias

    def _hidden(self, key_term, source_term, geometry):
        geometry_term = torch.einsum("bjgn,hg->bjhn", geometry, self._block(GEOMETRY_BLOCK))
        return F.gelu(key_term[:, None] + source_term[..., None] + geometry_term)

    def logits_12mm(self, key_term, source_term, geometry, *, similarity) -> torch.Tensor:
        """l12 [B,J,3,N] from the 12 mm key term, the source term, the geometry [B,J,8,N] and the similarity [B,J,N],
        which is added to the target-error logit only."""
        with fp32_region(key_term):
            hidden = self._hidden(key_term, source_term, geometry)
            logits = (torch.einsum("bjhn,ch->bjcn", hidden, self.output_12mm.weight)
                      + self.output_12mm.bias[None, None, :, None])
            target_only = torch.zeros(ROLE_COUNT, dtype=logits.dtype, device=logits.device)
            target_only[TARGET_ERROR] = 1
            logits = logits + similarity[:, :, None, :] * target_only[None, None, :, None]
        return logits if self.correct_tissue_role else fix_correct_tissue_logit(logits, 2)

    def residual_6mm(self, key_term, source_term, geometry) -> torch.Tensor:
        """W6_o h6 + b6_o [B,c,3,N] for one chunk of sources."""
        with fp32_region(key_term):
            hidden = self._hidden(key_term, source_term, geometry)
            return (torch.einsum("bjhn,ch->bjcn", hidden, self.output_6mm.weight)
                    + self.output_6mm.bias[None, None, :, None])

    def logits_6mm(self, key_term, source_term, upsampled_12mm, geometry_fn, *, chunk: int, recompute: bool):
        """(l6 = U(l12) + residual, residual), the residual computed ``chunk`` sources at a time.

        ``geometry_fn(start, stop)`` returns the 6 mm geometry [B,stop-start,8,N] of those sources. With ``recompute``
        (training with gradients) each chunk is rebuilt in the backward pass, so no chunk's hidden activation is
        kept. Chunking and recomputation change memory and time only.
        """
        if not 1 <= int(chunk) <= MAX_SOURCES:
            raise ValueError(f"role head source chunk actual={chunk}, expected 1..{MAX_SOURCES}")
        sources = source_term.shape[1]
        residuals = []

        def block(key, source, start, stop):
            return self.residual_6mm(key, source, geometry_fn(start, stop))

        for start in range(0, sources, int(chunk)):
            stop = min(start + int(chunk), sources)
            source = source_term[:, start:stop]
            if recompute and torch.is_grad_enabled() and self.training:
                residuals.append(checkpoint(block, key_term, source, start, stop, use_reentrant=False))
            else:
                residuals.append(block(key_term, source, start, stop))
        residual = torch.cat(residuals, dim=1)
        if residual.shape != upsampled_12mm.shape:
            raise ValueError(f"6 mm residual actual={tuple(residual.shape)}, expected={tuple(upsampled_12mm.shape)}")
        logits = upsampled_12mm + residual
        return (logits if self.correct_tissue_role else fix_correct_tissue_logit(logits, 2)), residual


class RoleDecoder(nn.Module):
    """Transposed-convolution decoder from local level 3 to the 3 mm grid; outputs 32 features and 3 role logits.

    At level l it concatenates the upsampled features, the encoder features of level l, the whole-volume features
    (level 2 only) and the selected role map of level l. The coarse role logits are a 1^3 projection of the features
    plus a 1^3 projection of the level-0 role map (initially 0.1 times the identity).
    """

    def __init__(self, image_widths: Sequence[int] = (24, 48, 96, 192), *, whole_volume_width: int = 64,
                 feature_width: int = 32, correct_tissue_role: bool = True) -> None:
        super().__init__()
        widths = tuple(int(w) for w in image_widths)
        if len(widths) != 4:
            raise ValueError(f"the role decoder expects four encoder levels; got {widths}")
        self.correct_tissue_role = bool(correct_tissue_role)
        self.upsampling = nn.ModuleList((
            nn.ConvTranspose3d(widths[3], widths[2], 2, stride=2, bias=False),
            nn.ConvTranspose3d(widths[2], widths[1], 2, stride=2, bias=False),
            nn.ConvTranspose3d(widths[1], widths[0], 2, stride=2, bias=False),
        ))
        self.stages = nn.ModuleList((ResidualBlock3d(2 * widths[2] + whole_volume_width + ROLE_COUNT, widths[2]),
                                     ResidualBlock3d(2 * widths[1] + ROLE_COUNT, widths[1]),
                                     ResidualBlock3d(2 * widths[0] + ROLE_COUNT, feature_width)))
        self.feature_refinement = DepthwiseConvBlock3d(feature_width, feature_width)
        self.logit_projection = nn.Conv3d(feature_width, ROLE_COUNT, 1, bias=False)
        self.role_projection = nn.Conv3d(ROLE_COUNT, ROLE_COUNT, 1, bias=True)
        self.activation_checkpointing = False
        with torch.no_grad():
            self.role_projection.weight.copy_(0.1 * torch.eye(ROLE_COUNT).reshape(ROLE_COUNT, ROLE_COUNT, 1, 1, 1))
            self.role_projection.bias.zero_()

    def forward(self, pyramid: list[torch.Tensor], whole_volume_local: torch.Tensor,
                role_maps: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """(decoder features [B,32,P,P,P], coarse role logits [B,3,P,P,P]); ``role_maps[l]`` is level l's map."""
        x = pyramid[3]
        for step, level in enumerate((2, 1, 0)):
            upsampled = self.upsampling[step](x)
            if upsampled.shape[2:] != pyramid[level].shape[2:]:
                raise ValueError(f"the decoder needs window axes divisible by eight: {tuple(upsampled.shape)} vs "
                                 f"{tuple(pyramid[level].shape)}")
            pieces = [upsampled, pyramid[level]]
            if level == 2:
                pieces.append(whole_volume_local)
            pieces.append(role_maps[level].to(upsampled.dtype))
            x = run_checkpointed(self, self.stages[step], torch.cat(pieces, dim=1))
        features = run_checkpointed(self, self.feature_refinement, x)
        logits = self.logit_projection(features) + self.role_projection(role_maps[0].to(features.dtype))
        return features, (logits if self.correct_tissue_role else fix_correct_tissue_logit(logits, 1))


# ------------------------------------------------------------------------------------------- monitoring readings
def selection_statistics(selected: torch.Tensor, sources: int) -> dict[str, torch.Tensor]:
    """Entropy (nats) of the selected-source distribution over positions and the leading source's share, per item."""
    counts = torch.stack([torch.bincount(row.flatten(), minlength=sources) for row in selected]).float()
    share = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
    entropy = -(share * torch.log(share.clamp(min=1e-30))).sum(dim=1)
    return {"entropy": entropy, "top_share": share.max(dim=1).values, "counts": counts}


def _residual_norm_ratio(residual: torch.Tensor, base: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    mask = valid[:, :, None, None].to(residual.dtype)
    flat_r = (residual.flatten(3) * mask).flatten(1)
    flat_b = (base.flatten(3) * mask).flatten(1)
    return flat_r.norm(dim=1) / flat_b.norm(dim=1).clamp(min=math.ulp(1.0))


def role_usage_statistics(residual: torch.Tensor, base: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    """How much the 6 mm residual can change role decisions, per batch item (monitoring only).

    ``residual`` and ``base`` = U(l12) are [B,J,3,N]. Reports the norm ratio of the residual, that of the
    class-centred residual (a common offset of the three logits changes nothing), the mean absolute change of the
    selected source's margin and the fraction of positions whose selected source differs from U(l12)'s.
    """
    fine = base + residual
    _, base_selected = select_source(base, valid)
    _, fine_selected = select_source(fine, valid)
    index = fine_selected[:, None]
    margin_fine = role_margin(fine).gather(1, index)[:, 0]
    margin_base = role_margin(base).gather(1, index)[:, 0]
    centred = residual - residual.mean(dim=2, keepdim=True)
    return {"raw_norm_ratio": _residual_norm_ratio(residual, base, valid),
            "centered_norm_ratio": _residual_norm_ratio(centred, base, valid),
            "selected_margin_change": (margin_fine - margin_base).abs().flatten(1).mean(dim=1),
            "selected_switch_fraction": (fine_selected != base_selected).float().flatten(1).mean(dim=1)}
