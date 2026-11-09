"""Native refinement: the roles of every editable voxel on the native PET grid.

The native context of a voxel (63 values) is the decoder features (32) and coarse role logits (3) sampled at the
voxel, its 20 voxel values (its own image, mask and interaction values, offset inside its 3 mm cell and spacing) and
an 8-wide learned code of its 3^3 neighbourhood (:meth:`NativeRefinement.encode_neighbourhood`, from 108 values).

Two projections, each followed by layer normalisation without affine parameters, map the source vectors (a_j) and
the native context (h) to width 32. The source weights are a softmax over the valid sources of the scaled
dot-product score plus the target margin of each source at the voxel (from its 6 mm role logits); the scribble
evidence is the weighted sum of the projected source vectors::

    alpha_j = softmax_j( m_j + h . a_j / sqrt(32) ),     v = sum_j alpha_j a_j

The native head (159 -> 64 -> 32 -> 3, GELU) reads ``[context, v, h - v, h * v]`` and adds its three outputs to the
coarse role logits. Its last layer starts at zero, so training starts from the coarse prediction.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..data.geometry import sample_world_features
from ..data.observations import NEIGHBOURHOOD_WIDTH, VOXEL_WIDTH
from ..data.roles import ROLE_COUNT
from .common import KEPT_WITHOUT_SCRIBBLE, fix_correct_tissue_logit, zero_channels
from .source_encoder import SOURCE_WIDTH

__all__ = ["CONTEXT_WIDTH", "FEATURE_WIDTH", "HEAD_INPUT_WIDTH", "NEIGHBOURHOOD_CODE_WIDTH", "PROJECTION_WIDTH",
           "NativeRefinement", "sample_source_role_logits"]

FEATURE_WIDTH = 32
NEIGHBOURHOOD_HIDDEN_WIDTH = 16
NEIGHBOURHOOD_CODE_WIDTH = 8
#: Native context: decoder features, coarse role logits, voxel values, neighbourhood code.
CONTEXT_WIDTH = FEATURE_WIDTH + ROLE_COUNT + VOXEL_WIDTH + NEIGHBOURHOOD_CODE_WIDTH
PROJECTION_WIDTH = 32
LAYER_NORM_EPSILON = 1e-5
#: Native head input: context, scribble evidence, projected context minus evidence, their element-wise product.
HEAD_INPUT_WIDTH = CONTEXT_WIDTH + 3 * PROJECTION_WIDTH
HEAD_HIDDEN_WIDTHS = (64, 32)


class NativeRefinement(nn.Module):
    """Neighbourhood encoder, source and context projections, and native head (module docstring).

    ``scribble_conditioning=False`` (the No scribble conditioning variant) zeroes every scribble-derived voxel value
    and gives the head zeros instead of the evidence terms; ``correct_tissue_role=False`` (the No correct-tissue role
    variant) fixes the correct-tissue output logit.
    """

    def __init__(self, *, scribble_conditioning: bool = True, correct_tissue_role: bool = True) -> None:
        super().__init__()
        self.scribble_conditioning = bool(scribble_conditioning)
        self.correct_tissue_role = bool(correct_tissue_role)
        self.neighbourhood_encoder = nn.Sequential(nn.Linear(NEIGHBOURHOOD_WIDTH, NEIGHBOURHOOD_HIDDEN_WIDTH), nn.GELU(),
                                                   nn.Linear(NEIGHBOURHOOD_HIDDEN_WIDTH, NEIGHBOURHOOD_CODE_WIDTH))
        first_hidden, second_hidden = HEAD_HIDDEN_WIDTHS
        # The context columns of the first layer get the default initialisation of a context-only layer; the
        # evidence columns start at zero, so initially the head reads the native context alone.
        context_layer = nn.Linear(CONTEXT_WIDTH, first_hidden)
        first = nn.Linear(HEAD_INPUT_WIDTH, first_hidden)
        last = nn.Linear(second_hidden, ROLE_COUNT)
        with torch.no_grad():
            first.weight.zero_()
            first.weight[:, :CONTEXT_WIDTH].copy_(context_layer.weight)
            first.bias.copy_(context_layer.bias)
            last.weight.zero_()
            last.bias.zero_()
        self.native_head = nn.Sequential(first, nn.GELU(), nn.Linear(first_hidden, second_hidden), nn.GELU(), last)
        self.source_projection = nn.Linear(SOURCE_WIDTH, PROJECTION_WIDTH)
        self.context_projection = nn.Linear(CONTEXT_WIDTH, PROJECTION_WIDTH)

    def encode_neighbourhood(self, neighbourhood: torch.Tensor) -> torch.Tensor:
        """The 8-wide code of the 108 neighbourhood values ([..., 108] -> [..., 8])."""
        return self.neighbourhood_encoder(neighbourhood)

    def attend(self, context: torch.Tensor, source_vectors: torch.Tensor, source_role_logits: torch.Tensor,
               source_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(source weights [B,N,J], scribble evidence [B,N,32], projected context [B,N,32]) of N voxels.

        ``context`` [B,N,63]; ``source_vectors`` [B,J,128]; ``source_role_logits`` [B,N,J,3], the 6 mm role logits
        of every source sampled at the voxels; ``source_valid`` [B,J]. Invalid sources get zero weight.
        """
        if context.shape[-1] != CONTEXT_WIDTH or source_vectors.shape[-1] != SOURCE_WIDTH:
            raise ValueError(f"native refinement needs a {CONTEXT_WIDTH}-wide context and {SOURCE_WIDTH}-wide source "
                             f"vectors; got {tuple(context.shape)} and {tuple(source_vectors.shape)}")
        if source_role_logits.shape != (*context.shape[:-1], source_vectors.shape[-2], ROLE_COUNT):
            raise ValueError("source role logits must be [B,N,J,3] at the native voxels")
        if source_valid.shape != source_vectors.shape[:2]:
            raise ValueError("source validity shape differs from the source vectors")
        # Checked on the CPU only: on a GPU the empty scribble is refused before the model runs.
        if not source_valid.is_cuda and not bool(source_valid.any(-1).all()):
            raise ValueError("an empty scribble must leave the mask unchanged before native refinement")
        safe_sources = torch.where(source_valid[..., None], source_vectors.float(),
                                   torch.zeros_like(source_vectors.float()))
        safe_logits = torch.where(source_valid[:, None, :, None], source_role_logits.float(),
                                  torch.zeros_like(source_role_logits.float()))
        projected_sources = F.layer_norm(self.source_projection(safe_sources), (PROJECTION_WIDTH,),
                                         eps=LAYER_NORM_EPSILON)
        projected_context = F.layer_norm(self.context_projection(context.float()), (PROJECTION_WIDTH,),
                                         eps=LAYER_NORM_EPSILON)
        margin = safe_logits[..., 0] - torch.logsumexp(safe_logits[..., 1:], -1)
        score = margin + torch.einsum("bnr,bjr->bnj", projected_context, projected_sources) / math.sqrt(PROJECTION_WIDTH)
        score = score.masked_fill(~source_valid[:, None], -torch.inf)
        weights = score.softmax(-1)
        evidence = torch.einsum("bnj,bjr->bnr", weights, projected_sources)
        return weights, evidence, projected_context

    def forward(self, features: torch.Tensor, coarse_logits: torch.Tensor, voxel_values: torch.Tensor,
                neighbourhood_code: torch.Tensor, *, source_vectors: torch.Tensor, source_role_logits: torch.Tensor,
                source_valid: torch.Tensor) -> torch.Tensor:
        """Native role logits [B,N,3] of N voxels from their sampled features [B,N,32] and coarse role logits [B,N,3],
        voxel values [B,N,20] and neighbourhood code [B,N,8] (source arguments as in :meth:`attend`)."""
        if self.scribble_conditioning:
            context = torch.cat((features, coarse_logits, voxel_values, neighbourhood_code), -1)
            _, evidence, projected = self.attend(context, source_vectors, source_role_logits, source_valid)
            head_input = torch.cat((context.float(), evidence, projected - evidence, projected * evidence), -1)
        else:
            voxel_values = zero_channels(voxel_values, -1, KEPT_WITHOUT_SCRIBBLE["voxel"])
            context = torch.cat((features, coarse_logits, voxel_values, neighbourhood_code), -1).float()
            if context.shape[-1] != CONTEXT_WIDTH:
                raise ValueError(f"native context actual={context.shape[-1]}, expected={CONTEXT_WIDTH}")
            removed = context.new_zeros(*context.shape[:-1], HEAD_INPUT_WIDTH - CONTEXT_WIDTH)
            head_input = torch.cat((context, removed), -1)
        logits = coarse_logits.float() + self.native_head(head_input)
        return logits if self.correct_tissue_role else fix_correct_tissue_logit(logits, -1)


def sample_source_role_logits(field: torch.Tensor, field_affine_xyz, world_xyz, source_valid: torch.Tensor, *,
                              support_affine_xyz, support_shape_zyx, support_valid_zyx=None
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """The role logits of every source at world points: ([B,N,J,3] FP32, validity [B,N]).

    ``field`` [B,J,3,Z,Y,X] holds the 6 mm source role logits; invalid sources are zeroed before sampling.
    """
    b, j, _, z, y, x = field.shape
    safe = torch.where(source_valid[:, :, None, None, None, None], field, torch.zeros_like(field))
    sampled, inside = sample_world_features(safe.reshape(b, j * 3, z, y, x), field_affine_xyz, world_xyz,
                                            support_affine_xyz=support_affine_xyz, support_shape_zyx=support_shape_zyx,
                                            support_valid_zyx=support_valid_zyx)
    return sampled.reshape(b, -1, j, 3).float(), inside
