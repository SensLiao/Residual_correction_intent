#!/usr/bin/env python3
"""Deferred architecture ladder for the PET/CT intent-residual mainline (SCEP v3+).

Every class in this module is **inactive**: nothing here is registered in any
experiment config, no launcher imports it for training, and no checkpoint may
claim these architecture ids until the corresponding ablation is separately
preregistered (docs/05-MODELS-AND-FUSION.md section 5).  The module exists so
that the post-primary experiments — editor conditioning ladder, PET/CT fusion
ladder, 3D variants, stronger backbones — have one tested, hash-bound source
instead of ad-hoc copies.

Study families (frozen naming, do not rename):
  * ``FiLM / concat / gated / intent-image cross_attention``  — editor conditioning
  * ``dual_stem / modality_gate / PET-CT cross_attention``    — PET/CT fusion
  * ``misalignment_DA``                                       — robustness ablation
  * 3D editor/compiler variants
  * residual-block stronger backbones

Cross-attention families are deliberately separate interfaces with separate
architecture ids (causal-label boundary from the frozen contract): P2T
cue↔state, editor intent↔image, and PET↔CT modality attention may never share
a label or a result table.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# Deferred architecture ids — namespaced so they can never collide with the
# active v2/v3 ids ("matched_legal_component_program_editor_v1", ...).
EDITOR_FILM_ARCHITECTURE_ID = "deferred_editor_film_v1"
EDITOR_CONCAT_ARCHITECTURE_ID = "deferred_editor_concat_v1"
EDITOR_GATED_ARCHITECTURE_ID = "deferred_editor_gated_v1"
EDITOR_INTENT_CROSS_ATTN_ARCHITECTURE_ID = "deferred_editor_intent_cross_attention_v1"
FUSION_DUAL_STEM_ARCHITECTURE_ID = "deferred_fusion_dual_stem_v1"
FUSION_MODALITY_GATE_ARCHITECTURE_ID = "deferred_fusion_modality_gate_v1"
FUSION_PET_CT_CROSS_ATTN_ARCHITECTURE_ID = "deferred_fusion_petct_cross_attention_v1"
EDITOR_3D_ARCHITECTURE_ID = "deferred_editor_3d_v1"
COMPILER_3D_ARCHITECTURE_ID = "deferred_compiler_3d_v1"
EDITOR_RES_ARCHITECTURE_ID = "deferred_editor_residual_backbone_v1"

VISUAL_13 = 13  # PET5 + CT5 + central-M0 + signed cue + selected component
STATE_17 = 17  # PET5 + CT5 + M0_5 + FG cue + BG cue (frozen P2T visible input)
NULL_FAMILY_ID = -1


def _channel_norm(channels: int) -> nn.GroupNorm:
    """Largest divisor from (8, 4, 2, 1) — GroupNorm must divide the channel
    count exactly (e.g. 12 channels -> 4 groups, 13 -> 1)."""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise ValueError(f"no group size divides {channels}")  # unreachable: 1 always divides


# --------------------------------------------------------------------------
# Shared program embedding (reuse the frozen v3 one; import kept lazy so this
# module stays importable without the mainline package on path).
# --------------------------------------------------------------------------

def _program_embedding(dim: int) -> nn.Module:
    from common.petct_program_models import ProgramEmbedding  # local import

    return ProgramEmbedding(dim=dim)


# --------------------------------------------------------------------------
# Editor conditioning ladder (deferred study 1)
# --------------------------------------------------------------------------

class ProgramFiLMConditioning(nn.Module):
    """Multi-scale FiLM: per-channel gamma/beta modulation from the program
    embedding, applied at encoder and decoder blocks.  NULL (family_id == -1)
    keeps the exact identity bypass (gamma=1, beta=0 on the bypassed path).
    """

    def __init__(
        self,
        program_dim: int = 64,
        channels: int = 128,
        num_scales: int = 3,
        *,
        channels_per_scale: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.architecture_id = EDITOR_FILM_ARCHITECTURE_ID
        self.num_scales = num_scales
        widths = (
            list(channels_per_scale)
            if channels_per_scale is not None
            else [channels] * num_scales
        )
        if len(widths) != num_scales:
            raise ValueError(
                f"channels_per_scale needs {num_scales} entries, got {len(widths)}"
            )
        self.channels_per_scale = tuple(widths)
        self.gamma = nn.ModuleList(
            [nn.Linear(program_dim, width, bias=True) for width in widths]
        )
        self.beta = nn.ModuleList(
            [nn.Linear(program_dim, width, bias=True) for width in widths]
        )

    def forward(
        self, features: Sequence[Tensor], condition: Tensor, bypass_mask: Tensor
    ) -> list[Tensor]:
        if len(features) != self.num_scales:
            raise ValueError(f"FiLM expects {self.num_scales} feature maps")
        mask = bypass_mask.to(dtype=torch.bool)
        outputs: list[Tensor] = []
        for scale, feature in enumerate(features):
            if feature.shape[1] != self.channels_per_scale[scale]:
                raise ValueError(
                    f"FiLM scale {scale} has {feature.shape[1]} channels; "
                    f"expected {self.channels_per_scale[scale]}"
                )
            gamma = self.gamma[scale](condition)[:, :, None, None] + 1.0
            beta = self.beta[scale](condition)[:, :, None, None]
            modulated = feature * gamma + beta
            modulated = torch.where(mask[:, None, None, None], feature, modulated)
            outputs.append(modulated)
        return outputs

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ProgramConcatConditioning(nn.Module):
    """Bottleneck concat conditioning: program vector broadcast as a constant
    feature map and concatenated before the decoder's first block.  NULL
    bypass keeps the frozen additive identity semantics at the same injection
    location, so budgets stay comparable across the ladder.
    """

    def __init__(self, program_dim: int = 64, bottleneck_channels: int = 128):
        super().__init__()
        self.architecture_id = EDITOR_CONCAT_ARCHITECTURE_ID
        self.projection = nn.Sequential(
            nn.Linear(program_dim, bottleneck_channels, bias=False)
        )
        self.merge = nn.Conv2d(
            bottleneck_channels * 2, bottleneck_channels, 1, bias=False
        )

    def forward(self, bottleneck: Tensor, condition: Tensor, bypass_mask: Tensor) -> Tensor:
        broadcast = self.projection(condition)[:, :, None, None].expand(
            -1, -1, bottleneck.shape[-2], bottleneck.shape[-1]
        )
        merged = self.merge(torch.cat([bottleneck, broadcast], dim=1))
        mask = bypass_mask.to(dtype=torch.bool)
        return torch.where(mask[:, None, None, None], bottleneck, merged)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ProgramGatedConditioning(nn.Module):
    """Gated injection: element-wise gate over the program shift at the
    bottleneck, with NULL identity bypass (gate ignored on bypass).
    """

    def __init__(self, program_dim: int = 64, channels: int = 128):
        super().__init__()
        self.architecture_id = EDITOR_GATED_ARCHITECTURE_ID
        self.shift = nn.Linear(program_dim, channels, bias=False)
        self.gate = nn.Sequential(nn.Linear(program_dim, channels), nn.Sigmoid())

    def forward(self, feature: Tensor, condition: Tensor, bypass_mask: Tensor) -> Tensor:
        shift = self.shift(condition)[:, :, None, None]
        gate = self.gate(condition)[:, :, None, None]
        gated = feature + gate * shift
        mask = bypass_mask.to(dtype=torch.bool)
        return torch.where(mask[:, None, None, None], feature, gated)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class IntentImageCrossAttention(nn.Module):
    """Intent-token ↔ image cross-attention at one feature scale.

    A single learned intent query (from the program embedding) attends over
    the flattened spatial feature tokens; the attended summary is added back
    as a channel shift at the same location as the frozen additive injector.
    Budget-matched single-scale by construction.  Distinct architecture id
    from PET↔CT fusion cross-attention (causal-label boundary).
    """

    def __init__(self, program_dim: int = 64, channels: int = 128, heads: int = 4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.architecture_id = EDITOR_INTENT_CROSS_ATTN_ARCHITECTURE_ID
        self.heads = heads
        self.head_dim = channels // heads
        self.query = nn.Linear(program_dim, channels, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.output_projection = nn.Linear(channels, channels, bias=False)

    def forward(self, feature: Tensor, condition: Tensor, bypass_mask: Tensor) -> Tensor:
        batch, channels, height, width = feature.shape
        if channels != self.heads * self.head_dim:
            raise ValueError("feature channels do not match attention heads")
        query = self.query(condition).view(batch, self.heads, self.head_dim)  # [B, H, D]
        keys = (
            self.key(feature)
            .view(batch, self.heads, self.head_dim, height * width)
            .transpose(2, 3)
        )  # [B, H, HW, D]
        values = (
            self.value(feature)
            .view(batch, self.heads, self.head_dim, height * width)
            .transpose(2, 3)
        )  # [B, H, HW, D]
        attention = torch.einsum("bhd,bhnd->bhn", query, keys) / float(
            self.head_dim
        ) ** 0.5
        attention = attention.softmax(dim=-1)
        attended = torch.einsum("bhn,bhnd->bhd", attention, values)  # [B, H, D]
        shift = self.output_projection(
            attended.reshape(batch, channels)
        )[:, :, None, None]
        out = feature + shift
        mask = bypass_mask.to(dtype=torch.bool)
        return torch.where(mask[:, None, None, None], feature, out)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# PET/CT fusion ladder (deferred study 2)
# --------------------------------------------------------------------------

def _split_modalities(visual: Tensor, visual_channels: int) -> tuple[Tensor, Tensor]:
    """Slice a PET5+CT5+... visual into its two modality blocks."""
    if visual.ndim != 4 or visual.shape[1] < 10 or visual.shape[1] != visual_channels:
        raise ValueError(f"visual must be BCHW with {visual_channels} channels")
    return visual[:, :5], visual[:, 5:10]


class _ConvNormAct2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            _channel_norm(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            _channel_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DualShallowStemBottleneckFusion(nn.Module):
    """Separate shallow PET and CT stems fused at the bottleneck of a shared
    encoder; output replaces the frozen single-stem encoder head.  The first
    preregistered adapt step for the PET/CT fusion ladder.
    """

    def __init__(self, visual_channels: int = VISUAL_13, base: int = 32):
        super().__init__()
        if visual_channels != VISUAL_13:
            raise ValueError("fusion ladder assumes the 13-channel visual contract")
        self.architecture_id = FUSION_DUAL_STEM_ARCHITECTURE_ID
        self.visual_channels = visual_channels
        self.pet_stem = _ConvNormAct2D(5, base)
        self.ct_stem = _ConvNormAct2D(5, base)
        self.context = _ConvNormAct2D(visual_channels - 10, base)
        self.fuse = nn.Sequential(
            nn.Conv2d(base * 3, base * 2, 1, bias=False),
            _channel_norm(base * 2),
            nn.GELU(),
        )

    def forward(self, visual: Tensor) -> Tensor:
        pet, ct = _split_modalities(visual, self.visual_channels)
        pet_map = self.pet_stem(pet)
        ct_map = self.ct_stem(ct)
        context_map = self.context(visual[:, 10:])
        return self.fuse(torch.cat([pet_map, ct_map, context_map], dim=1))


class ModalityGatedFusion(nn.Module):
    """Modality gate: learned PET/CT channel gates over the two shallow stems
    before a shared fusion head.  Second preregistered adapt step.
    """

    def __init__(self, base: int = 32):
        super().__init__()
        self.architecture_id = FUSION_MODALITY_GATE_ARCHITECTURE_ID
        self.pet_stem = _ConvNormAct2D(5, base)
        self.ct_stem = _ConvNormAct2D(5, base)
        self.pet_gate = nn.Sequential(nn.Conv2d(base, base, 1), nn.Sigmoid())
        self.ct_gate = nn.Sequential(nn.Conv2d(base, base, 1), nn.Sigmoid())

    def forward(self, pet: Tensor, ct: Tensor) -> tuple[Tensor, Tensor]:
        # stems evaluated exactly once; gates consume the stem outputs
        pet_map = self.pet_stem(pet)
        ct_map = self.ct_stem(ct)
        return pet_map * self.pet_gate(pet_map), ct_map * self.ct_gate(ct_map)


# Full-resolution (H*W)^2 attention logits explode memory (~4 GiB fp32 per
# direction at 128x128); the guard keeps this scaffold on pooled /
# low-resolution / windowed features by contract.
MAX_PETCT_CROSS_ATTN_TOKENS = 4096  # 64x64


class PETCTBidirectionalCrossAttention(nn.Module):
    """Budget-matched bidirectional PET↔CT cross-attention at one feature
    scale.  Distinct architecture id from the editor intent-image attention
    and from the P2T cue-state attention (causal-label boundary).
    """

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.architecture_id = FUSION_PET_CT_CROSS_ATTN_ARCHITECTURE_ID
        self.heads = heads
        self.head_dim = channels // heads
        self.q_pet = nn.Conv2d(channels, channels, 1, bias=False)
        self.q_ct = nn.Conv2d(channels, channels, 1, bias=False)
        self.output_projection = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, pet_map: Tensor, ct_map: Tensor) -> Tensor:
        if pet_map.shape != ct_map.shape:
            raise ValueError("PET and CT maps must share shape")
        batch, channels, height, width = pet_map.shape
        if height * width > MAX_PETCT_CROSS_ATTN_TOKENS:
            raise ValueError(
                f"PET-CT cross-attention on {height}x{width}={height * width} "
                f"tokens exceeds the {MAX_PETCT_CROSS_ATTN_TOKENS}-token guard; "
                "run it on pooled / low-resolution / windowed features"
            )

        def attend(source: Tensor, target: Tensor) -> Tensor:
            query = target.view(batch, self.heads, self.head_dim, -1)
            key = source.view(batch, self.heads, self.head_dim, -1)
            value = source.view(batch, self.heads, self.head_dim, -1)
            attention = torch.einsum("bhdc,bhdt->bhct", query, key) / float(
                self.head_dim
            ) ** 0.5
            attention = attention.softmax(dim=-1)
            return torch.einsum("bhct,bhdt->bhdc", attention, value).reshape(
                batch, channels, height, width
            )

        # PET attends over CT, CT attends over PET.
        pet_refined = attend(self.q_ct(ct_map), pet_map)
        ct_refined = attend(self.q_pet(pet_map), ct_map)
        return self.output_projection(torch.cat([pet_refined, ct_refined], dim=1))


class MisalignmentDA(nn.Module):
    """Deferred misalignment robustness study utility: deterministic spatial
    jitter applied to ONE modality block of a 13-channel visual (never to the
    cue/component channels).  Pure data-plane transform; no training claims.
    """

    def __init__(self, max_shift: int = 2):
        super().__init__()
        self.max_shift = int(max_shift)

    def forward(self, visual: Tensor, shifts_xy: Tensor, modality: str = "ct") -> Tensor:
        if visual.ndim != 4 or visual.shape[1] != VISUAL_13:
            raise ValueError("misalignment DA assumes the 13-channel visual")
        if shifts_xy.shape[0] != visual.shape[0] or shifts_xy.shape[1] != 2:
            raise ValueError("shifts_xy must be [B, 2] integer pixel offsets")
        if modality not in ("pet", "ct"):
            raise ValueError("modality must be 'pet' or 'ct'")
        start = 0 if modality == "pet" else 5
        shifted = visual.clone()
        for batch_index, (dy, dx) in enumerate(shifts_xy.tolist()):
            dy, dx = int(dy), int(dx)
            if abs(dy) > self.max_shift or abs(dx) > self.max_shift:
                raise ValueError(f"shift {(dy, dx)} exceeds max_shift {self.max_shift}")
            block = visual[batch_index : batch_index + 1, start : start + 5]
            shifted[batch_index, start : start + 5] = self._shift_2d(
                block, dy, dx
            )[0]
        return shifted

    @staticmethod
    def _shift_2d(block: Tensor, dy: int, dx: int) -> Tensor:
        """Zero-padded shift: content moved off-canvas is dropped, never
        wrapped around (torch.roll would reappear on the opposite edge).

        For a +dx shift we pad on the LEFT and crop the leading window, so
        ``out[..., x] = in[..., x - dx]`` with zero fill at the vacated edge.
        """
        left, right = max(0, dx), max(0, -dx)
        top, bottom = max(0, dy), max(0, -dy)
        padded = F.pad(block, (left, right, top, bottom))
        start_y = max(0, -dy)
        start_x = max(0, -dx)
        return padded[
            ...,
            start_y : start_y + block.shape[-2],
            start_x : start_x + block.shape[-1],
        ]


# --------------------------------------------------------------------------
# 3D variants (deferred study 3)
# --------------------------------------------------------------------------

class _ConvNormAct3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            _channel_norm(out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            _channel_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ProgramEditorUNet3D(nn.Module):
    """3D residual editor: 13-channel volume (PET5+CT5+central-M0+signed cue+
    selected component) conditioned with the SAME program/continuous ladder as
    the frozen 2D editor.  Output is one 3D residual logit; operation algebra
    is applied by ``apply_program_operation_3d``.

    Native-3D contract note (2026-08-18 external review): the frozen 13-channel
    visual is the 2D/2.5D slice-stack bridge, NOT the final 3D channel
    contract.  The preregistered 3D ladder must define real volume channels
    (PET / CT / M / cue / component) with spacing-aware pooling before any
    training claim on this scaffold.
    """

    def __init__(
        self,
        visual_channels: int = VISUAL_13,
        base: int = 16,
        program_dim: int = 64,
        *,
        conditioner: str = "program",
        state_dim: int = 64,
    ):
        super().__init__()
        if conditioner not in ("program", "continuous"):
            raise ValueError("conditioner must be 'program' or 'continuous'")
        self.architecture_id = EDITOR_3D_ARCHITECTURE_ID
        self.visual_channels = visual_channels
        self.conditioner = conditioner
        if conditioner == "program":
            self.condition = _program_embedding(program_dim)
        else:
            self.condition = nn.Sequential(
                nn.Linear(state_dim, state_dim), nn.GELU(), nn.Linear(state_dim, program_dim)
            )
        self.enc1 = _ConvNormAct3D(visual_channels, base)
        self.enc2 = _ConvNormAct3D(base, base * 2)
        self.bottleneck = _ConvNormAct3D(base * 2, base * 4)
        self.dec2 = _ConvNormAct3D(base * 4 + base * 2, base * 2)
        self.dec1 = _ConvNormAct3D(base * 2 + base, base)
        self.condition_projection = nn.Linear(program_dim, base * 4, bias=False)
        self.output = nn.Conv3d(base, 1, 1)

    def forward(
        self,
        visual: Tensor,
        family_ids_t: Tensor,
        operand_mode: Tensor,
        support_mode: Tensor,
        state_embedding: Optional[Tensor] = None,
        active_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if visual.ndim != 5 or visual.shape[1] != self.visual_channels:
            raise ValueError("visual must be BCDHW with 13 channels")
        if self.conditioner == "program":
            condition = self.condition(family_ids_t, operand_mode, support_mode)
            neutral_mask = family_ids_t == NULL_FAMILY_ID
        else:
            if state_embedding is None or active_mask is None:
                raise ValueError("continuous conditioner requires state embedding and mask")
            condition = self.condition(state_embedding)
            neutral_mask = ~active_mask.to(dtype=torch.bool)
        e1 = self.enc1(visual)
        # ceiling: fixed factor-2 pooling ignores voxel anisotropy; a
        # spacing-aware planner belongs to the preregistered 3D ladder
        e2 = self.enc2(F.max_pool3d(e1, 2))
        b = self.bottleneck(F.max_pool3d(e2, 2))
        shift = self.condition_projection(condition)[:, :, None, None, None]
        b = torch.where(neutral_mask[:, None, None, None, None], b, b + shift)
        u2 = F.interpolate(b, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.output(d1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def apply_program_operation_3d(
    logits: Tensor,
    m0_volume: Tensor,
    selected_component: Tensor,
    operation_ids: Tensor,
    threshold: float = 0.5,
) -> Tuple[Tensor, Tensor]:
    """SCEP execution algebra on full 3D volumes (ADD union / REMOVE
    complement-protected subtraction), mirroring the frozen 2D algebra.
    """
    if logits.shape != m0_volume.shape or selected_component.shape != m0_volume.shape:
        raise ValueError("logits, M0 and selected component must share shape")
    if operation_ids.ndim != 1 or operation_ids.shape[0] != logits.shape[0]:
        raise ValueError("operation_ids must have shape [B]")
    if torch.any((operation_ids != 0) & (operation_ids != 1)):
        raise ValueError("editor application requires ADD(0) or REMOVE(1)")
    current = m0_volume > 0
    prediction = torch.sigmoid(logits) >= float(threshold)
    add_rows = operation_ids == 0
    delta = torch.where(
        add_rows[:, None, None, None, None],
        prediction & ~current,
        prediction & (selected_component > 0),
    )
    corrected = torch.where(
        add_rows[:, None, None, None, None],
        current | delta,
        current & ~delta,
    )
    return delta, corrected


# --------------------------------------------------------------------------
# Stronger backbones (deferred study 4)
# --------------------------------------------------------------------------

class ResBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _channel_norm(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            _channel_norm(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.block(x) + x)


class ProgramEditorUNet2DRes(nn.Module):
    """Residual-block 2D editor with the frozen conditioning ladder.

    Same 13-channel visual and program/continuous conditioning contract as
    ``ProgramEditorUNet2D``; the plain double-conv blocks are replaced by
    residual blocks and the base width can be raised for a stronger backbone
    sweep.  Inactive until a backbone ablation is preregistered.
    """

    def __init__(
        self,
        visual_channels: int = VISUAL_13,
        base: int = 32,
        program_dim: int = 64,
        *,
        conditioner: str = "program",
        state_dim: int = 64,
    ):
        super().__init__()
        if conditioner not in ("program", "continuous"):
            raise ValueError("conditioner must be 'program' or 'continuous'")
        self.architecture_id = EDITOR_RES_ARCHITECTURE_ID
        self.visual_channels = visual_channels
        self.conditioner = conditioner
        if conditioner == "program":
            self.condition = _program_embedding(program_dim)
        else:
            self.condition = nn.Sequential(
                nn.Linear(state_dim, state_dim), nn.GELU(), nn.Linear(state_dim, program_dim)
            )
        self.enc1 = nn.Sequential(
            nn.Conv2d(visual_channels, base, 3, padding=1, bias=False),
            _channel_norm(base),
            nn.GELU(),
        )
        self.res1 = ResBlock2D(base)
        self.enc2 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False)
        self.res2 = ResBlock2D(base * 2)
        self.enc3 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False)
        self.res3 = ResBlock2D(base * 4)
        self.condition_projection = nn.Linear(program_dim, base * 4, bias=False)
        self.up2 = nn.Conv2d(base * 4, base * 2, 1, bias=False)
        self.res4 = ResBlock2D(base * 2)
        self.up1 = nn.Conv2d(base * 2, base, 1, bias=False)
        self.res5 = ResBlock2D(base)
        self.output = nn.Conv2d(base, 1, 1)

    def forward(
        self,
        visual: Tensor,
        family_ids_t: Tensor,
        operand_mode: Tensor,
        support_mode: Tensor,
        state_embedding: Optional[Tensor] = None,
        active_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if visual.ndim != 4 or visual.shape[1] != self.visual_channels:
            raise ValueError("visual must be BCHW with 13 channels")
        if self.conditioner == "program":
            condition = self.condition(family_ids_t, operand_mode, support_mode)
            neutral_mask = family_ids_t == NULL_FAMILY_ID
        else:
            if state_embedding is None or active_mask is None:
                raise ValueError("continuous conditioner requires state embedding and mask")
            condition = self.condition(state_embedding)
            neutral_mask = ~active_mask.to(dtype=torch.bool)
        e1 = self.res1(self.enc1(visual))
        e2 = self.res2(self.enc2(e1))
        b = self.res3(self.enc3(e2))
        shift = self.condition_projection(condition)[:, :, None, None]
        b = torch.where(neutral_mask[:, None, None, None], b, b + shift)
        u2 = self.res4(self.up2(F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)) + e2)
        u1 = self.res5(self.up1(F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)) + e1)
        return self.output(u1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------
# 3D compiler variant
# --------------------------------------------------------------------------

class ProgramCompilerNet3D(nn.Module):
    """3D compiler: same frozen pooling contract as the 2D compiler (global /
    cue-support-pooled state summary + signed-cue map + relative geometry),
    lifted to 3D convolutions.  Family/pointer heads are identical in shape to
    ``ProgramCompilerNet`` so the deterministic ``LegalCallCompiler`` and the
    embedding ladder can be reused unchanged.
    """

    def __init__(
        self,
        state_channels: int = 15,
        width: int = 32,
    ):
        super().__init__()
        self.architecture_id = COMPILER_3D_ARCHITECTURE_ID
        self.state_channels = state_channels
        self.width = width
        self.state_encoder = _ConvNormAct3D(state_channels, width, stride=2)
        self.scribble_encoder = _ConvNormAct3D(2, width, stride=2)
        self.fusion = nn.Sequential(
            nn.Linear(width * 3, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.family_head_add = nn.Linear(width, 3)
        self.family_head_remove = nn.Linear(width, 3)
        self.pointer_query = nn.Linear(width, width)
        self.pointer_key = nn.Linear(7, width)
        self.pointer_distance_mlp = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(
        self,
        visual: Tensor,
        operation_ids: Tensor,
        component_vectors: Tensor,
        component_mask: Tensor,
    ) -> dict[str, Tensor]:
        if visual.ndim != 5 or visual.shape[1] != self.state_channels + 2:
            raise ValueError("visual must be BCDHW with 17 channels")
        state_map = self.state_encoder(visual[:, : self.state_channels])
        scribble_map = self.scribble_encoder(visual[:, self.state_channels :])
        binary_scribble = (visual[:, self.state_channels :] > 0).to(visual.dtype).amax(dim=1, keepdim=True)
        prompt_map = F.adaptive_max_pool3d(binary_scribble, output_size=state_map.shape[-3:])
        if torch.any(prompt_map.flatten(1).sum(dim=1) == 0):
            raise RuntimeError("scribble vanished during state-feature pooling")
        prompt_weight = prompt_map.flatten(2).transpose(1, 2)
        denominator = prompt_weight.sum(dim=1).clamp_min(1.0)
        state_tokens = state_map.flatten(2).transpose(1, 2)
        state_global = state_tokens.mean(dim=1)
        state_at_prompt = (state_tokens * prompt_weight).sum(dim=1) / denominator
        scribble_at_prompt = (
            scribble_map.flatten(2).transpose(1, 2) * prompt_weight
        ).sum(dim=1) / denominator
        embedding = self.fusion(
            torch.cat([state_global, state_at_prompt, scribble_at_prompt], dim=1)
        )
        family_logits = torch.zeros(
            visual.shape[0], 3, dtype=visual.dtype, device=visual.device
        )
        add_rows = operation_ids == 0
        remove_rows = operation_ids == 1
        if torch.any(~(add_rows | remove_rows)):
            raise ValueError("operation_ids must be ADD(0) or REMOVE(1)")
        if add_rows.any():
            family_logits[add_rows] = self.family_head_add(embedding[add_rows])
        if remove_rows.any():
            family_logits[remove_rows] = self.family_head_remove(embedding[remove_rows])
        query = self.pointer_query(embedding)
        keys = self.pointer_key(component_vectors)
        dot = torch.einsum("bd,bkd->bk", query, keys) / float(self.width) ** 0.5
        distance_features = torch.clamp(component_vectors[..., -2:], min=0.0, max=200.0) / 100.0
        distance_score = self.pointer_distance_mlp(distance_features).squeeze(-1)
        pointer_logits = dot + distance_score
        pointer_logits = torch.where(
            component_mask.to(dtype=torch.bool),
            pointer_logits,
            torch.full_like(pointer_logits, float("-inf")),
        )
        # Fail-safe: a row with no eligible component would softmax to NaN
        # downstream.  Fill with zeros (uniform posterior); NEW_CUE resolution
        # belongs to the deterministic legal-call layer, not this scaffold.
        no_candidate = ~component_mask.to(dtype=torch.bool).any(dim=1)
        pointer_logits = torch.where(
            no_candidate[:, None],
            torch.zeros_like(pointer_logits),
            pointer_logits,
        )
        return {
            "family_logits": family_logits,
            "pointer_logits": pointer_logits,
            "embedding": embedding,
        }

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
