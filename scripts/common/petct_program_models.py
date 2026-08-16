#!/usr/bin/env python3
"""v3 program compiler and program-conditioned editor modules (SCEP).

The compiler reuses the frozen v2 simple state encoder shape (15-channel
PET5+CT5+M5 pooled state + 2 signed cue channels) so the same visible tensor
bundles feed both v2 and v3.  It adds:

  * an operation-conditioned legal-family scorer (never predicts operation);
  * a conditional component pointer that only fires for ADD existing-object
    families (SCEP 5.6); REMOVE uses deterministic cue-hit binding;
  * a deterministic legal-call compiler with an auditable typed trace;
  * ``ProgramEmbedding`` with an exact NULL bypass;
  * ``ContinuousReadoutConditioner``, the capacity-matched continuous-surface
    control arm (must keep the frozen naming ``continuous_surface_readout_control``);
  * ``ProgramEditorUNet2D``, the 13-channel editor variant whose 13th channel
    is the central selected-component mask (bitwise zero for NEW_CUE).

No pretrained or external text/image encoder is loaded by this module.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from common.petct_program_contract import (
    ARCHITECTURE_ID,
    GRAMMAR_VERSION,
    NEW_CUE_SENTINEL,
    ProgramContractError,
    family_ids,
    validate_legal_call,
)

PROGRAM_EDITOR_ARCHITECTURE_ID = "matched_legal_component_program_editor_v1"
COMPONENT_DESCRIPTOR_DIM = 7
# Descriptor order consumed by this module:
# [log_volume, z_span, prompted_slice_overlap, centroid_dx_mm, centroid_dy_mm,
#  cue_overlap_voxels, distance_from_cue_mm]
NULL_FAMILY_ID = -1


def _normalized_pointer_features(component_vectors: Tensor) -> Tensor:
    """Keep pointer features finite and roughly unit-scale without GT stats."""

    scale = torch.tensor(
        [3.0, 24.0, 24.0, 100.0, 100.0, 16.0, 100.0],
        dtype=component_vectors.dtype,
        device=component_vectors.device,
    )
    if not torch.isfinite(component_vectors).all():
        raise ProgramContractError("component descriptors must be finite")
    # Centroid offsets are signed.  Clipping every field at zero destroyed
    # left/right and anterior/posterior information and made two different
    # candidates observationally identical to the pointer.
    clamped = component_vectors.clone()
    nonnegative = (0, 1, 2, 5, 6)
    signed = (3, 4)
    positive_values = torch.maximum(
        clamped[..., list(nonnegative)],
        torch.zeros_like(clamped[..., list(nonnegative)]),
    )
    clamped[..., list(nonnegative)] = torch.minimum(
        positive_values, scale[list(nonnegative)] * 2.0
    )
    signed_limit = scale[list(signed)] * 2.0
    clamped[..., list(signed)] = torch.minimum(
        torch.maximum(clamped[..., list(signed)], -signed_limit), signed_limit
    )
    return clamped / scale.clamp_min(1.0)


class ProgramCompilerNet(nn.Module):
    """State-conditioned legal-program compiler for the v3 contract.

    Input: the frozen 17-channel visible crop plus per-sample component
    descriptor vectors.  Output: legal-family logits (per operation, size 3)
    and component-pointer logits (only meaningful for ADD existing-object
    families).  The operation itself comes from the signed cue and is never
    a learned output of this network.
    """

    def __init__(
        self,
        state_channels: int = 15,
        width: int = 64,
        *,
        include_repair: bool = True,
        use_relative_geometry: bool = True,
    ):
        super().__init__()
        self.state_channels = state_channels
        self.width = width
        self.architecture_id = ARCHITECTURE_ID
        self.include_repair = bool(include_repair)
        self.state_encoder = _ConvNormAct(state_channels, width, stride=2)
        self.scribble_encoder = nn.Sequential(
            nn.Conv2d(2, width, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8 if width >= 8 else 1, width),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(width * 3, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        # Operation-conditioned family scorer: one head per operation keeps
        # the logit spaces disjoint so no cross-operation family is legal.
        self.family_head_add = nn.Linear(width, 3)
        self.family_head_remove = nn.Linear(width, 3)
        # Conditional pointer scorer (SCEP 5.6): q(h)^T k(v) + MLP(d).
        self.pointer_query = nn.Linear(width, width)
        self.pointer_key = nn.Linear(COMPONENT_DESCRIPTOR_DIM, width)
        self.pointer_distance_mlp = nn.Sequential(
            nn.Linear(2, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.use_relative_geometry = use_relative_geometry
        self.relative_geometry_mlp = nn.Sequential(
            nn.Linear(4, width), nn.GELU(), nn.Linear(width, width)
        )

    def forward(
        self,
        visual: Tensor,
        spacing_xy: Tensor,
        operation_ids: Tensor,
        component_vectors: Tensor,
        component_mask: Tensor,
    ) -> Dict[str, Tensor]:
        if visual.ndim != 4 or visual.shape[1] != self.state_channels + 2:
            raise ValueError("visual must be BCHW with 17 channels")
        if operation_ids.shape[0] != visual.shape[0] or operation_ids.ndim != 1:
            raise ValueError("operation_ids must have shape [B]")
        if component_vectors.ndim != 3 or component_vectors.shape[2] != COMPONENT_DESCRIPTOR_DIM:
            raise ValueError(
                "component_vectors must have shape [B,K,%d]"
                % COMPONENT_DESCRIPTOR_DIM
            )
        if component_mask.shape != component_vectors.shape[:2]:
            raise ValueError("component_mask must have shape [B,K]")
        state_input = visual[:, : self.state_channels]
        scribble = visual[:, self.state_channels :]
        state_map = self.state_encoder(state_input)
        scribble_map = self.scribble_encoder(scribble)
        binary_scribble = (scribble > 0).to(dtype=visual.dtype).amax(dim=1, keepdim=True)
        prompt_map = F.adaptive_max_pool2d(
            binary_scribble, output_size=state_map.shape[-2:]
        )
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
        if self.use_relative_geometry:
            state_global = state_global + self.relative_geometry_mlp(
                self._relative_geometry_shift(scribble, prompt_map, spacing_xy)
            )
        embedding = self.fusion(
            torch.cat([state_global, state_at_prompt, scribble_at_prompt], dim=1)
        )
        family_logits = torch.zeros(
            visual.shape[0], 3, dtype=visual.dtype, device=visual.device
        )
        add_rows = operation_ids == 0
        remove_rows = operation_ids == 1
        if torch.any(~(add_rows | remove_rows)):
            raise ProgramContractError("operation_ids must be ADD(0) or REMOVE(1)")
        if add_rows.any():
            family_logits[add_rows] = self.family_head_add(embedding[add_rows])
        if remove_rows.any():
            family_logits[remove_rows] = self.family_head_remove(embedding[remove_rows])
        if not self.include_repair and remove_rows.any():
            family_logits[remove_rows, 2] = float("-inf")
        query = self.pointer_query(embedding)
        keys = self.pointer_key(_normalized_pointer_features(component_vectors))
        dot = torch.einsum("bd,bkd->bk", query, keys) / float(self.width) ** 0.5
        distance_features = component_vectors[..., -2:]
        distance_features = torch.clamp(distance_features, min=0.0, max=200.0) / 100.0
        distance_score = self.pointer_distance_mlp(distance_features).squeeze(-1)
        pointer_logits = dot + distance_score
        pointer_logits = torch.where(
            component_mask.to(dtype=torch.bool), pointer_logits, torch.full_like(pointer_logits, float("-inf"))
        )
        return {
            "family_logits": family_logits,
            "pointer_logits": pointer_logits,
            "embedding": embedding,
        }

    @staticmethod
    def _relative_geometry_shift(
        scribble: Tensor, prompt_map: Tensor, spacing_xy: Tensor
    ) -> Tensor:
        """Compact cue-relative geometry summary (state-relative, GT-free)."""

        binary = (scribble > 0).any(dim=1).to(dtype=scribble.dtype)
        mass = binary.sum(dim=(1, 2)).clamp_min(1.0)
        rows = torch.arange(binary.shape[1], dtype=scribble.dtype, device=scribble.device)
        cols = torch.arange(binary.shape[2], dtype=scribble.dtype, device=scribble.device)
        center_y = (binary * rows[None, :, None]).sum(dim=(1, 2)) / mass
        center_x = (binary * cols[None, None, :]).sum(dim=(1, 2)) / mass
        support = prompt_map.flatten(2).sum(dim=2).squeeze(1)
        return torch.stack(
            [
                center_x / float(binary.shape[2]),
                center_y / float(binary.shape[1]),
                support / prompt_map.flatten(1).sum(dim=1).clamp_min(1.0),
                torch.ones_like(center_x),
            ],
            dim=1,
        ).to(dtype=scribble.dtype)


class LegalCallCompiler(nn.Module):
    """Deterministic compiler: posterior -> legal call record + typed trace.

    This module has no parameters; it is kept as an nn.Module so its
    invocation is part of the checkpointed forward graph and its version is
    captured by ``source_sha256`` receipts.
    """

    def __init__(self, *, include_repair: bool = True):
        super().__init__()
        self.include_repair = bool(include_repair)

    def forward(
        self,
        operation: str,
        family_logits: Tensor,
        pointer_probs: Optional[Tensor],
        components: List[Mapping],
        cue_hit_component_position: Optional[int] = None,
    ) -> Dict[str, object]:
        """Compile one episode. ``family_logits`` is [3]; pointer optional."""

        legal = family_ids(operation, self.include_repair)
        if family_logits.ndim != 1 or family_logits.numel() != len(legal):
            raise ProgramContractError(
                "family logits must contain exactly the legal operation-conditioned families"
            )
        family_id = int(family_logits.argmax().item())
        family = legal[family_id]
        trace: List[Dict[str, object]] = [
            {"step": "OBSERVE", "operation_authority": operation, "source": "signed_cue"},
            {"step": "ENUMERATE", "component_count": len(components)},
            {"step": "SCORE", "family_scores": family_logits.tolist(), "legal_families": legal},
        ]
        if operation == "ADD" and family != "CREATE_NEW":
            if pointer_probs is None:
                raise ProgramContractError("ADD existing-object family requires pointer posterior")
            if len(components) == 0:
                raise ProgramContractError("ADD existing-object call has no component candidates")
            if pointer_probs.ndim != 1 or pointer_probs.numel() != len(components):
                raise ProgramContractError("pointer posterior and candidates disagree")
            operand_index = int(pointer_probs.argmax().item())
            operand = str(components[operand_index]["component_key"])
            trace.append(
                {
                    "step": "BIND",
                    "binding": "learned_pointer",
                    "operand_index": operand_index,
                    "pointer_probs": pointer_probs.tolist(),
                }
            )
        elif family == "CREATE_NEW":
            operand = NEW_CUE_SENTINEL
            trace.append({"step": "BIND", "binding": "NEW_CUE_sentinel"})
        else:
            if cue_hit_component_position is None:
                raise ProgramContractError(
                    "REMOVE requires the deterministic cue-hit component position"
                )
            operand_index = int(cue_hit_component_position)
            if not 0 <= operand_index < len(components):
                raise ProgramContractError("cue-hit component position is outside candidates")
            operand = str(components[operand_index]["component_key"])
            trace.append(
                {
                    "step": "BIND",
                    "binding": "deterministic_cue_hit",
                    "operand_index": operand_index,
                }
            )
        validate_legal_call(operation, family, operand)
        trace.append({"step": "TYPECHECK", "verdict": "LEGAL", "grammar_version": GRAMMAR_VERSION})
        call = {
            "grammar_version": GRAMMAR_VERSION,
            "operation": operation,
            "family": family,
            "operand": operand,
        }
        trace.append({"step": "EXECUTE", "call": call})
        return {"call": call, "typed_trace": trace}


class ProgramEmbedding(nn.Module):
    """Legal-call embedding with an exact NULL bypass.

    NULL is represented by ``family_id == NULL_FAMILY_ID``; its embedding is
    exactly zero so NULL-conditioned editors are identity-conditioned, not
    merely small.
    """

    def __init__(self, dim: int = 64, *, include_repair: bool = True):
        super().__init__()
        self.dim = dim
        self.include_repair = bool(include_repair)
        family_count = 3 + (3 if include_repair else 2)
        self.family = nn.Embedding(family_count + 1, dim, padding_idx=0)
        self.operand_mode = nn.Embedding(3, dim)  # component / NEW_CUE / NULL
        self.support_policy = nn.Embedding(2, dim)  # protect-complement / NULL
        with torch.no_grad():
            self.operand_mode.weight[2].zero_()
            self.support_policy.weight[1].zero_()
        self.mlp = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(
        self,
        family_ids_t: Tensor,
        operand_mode: Tensor,
        support_mode: Tensor,
    ) -> Tensor:
        null_mask = family_ids_t == NULL_FAMILY_ID
        if torch.any((~null_mask) & ((family_ids_t < 0) | (family_ids_t >= self.family.num_embeddings - 1))):
            raise ProgramContractError("global family id is outside the embedding vocabulary")
        # Row zero is reserved exclusively for NULL.  Legal global family
        # ids 0..5 are shifted by one so GROW_LOCAL is trainable rather than
        # silently mapped to the frozen padding vector.
        safe_family = torch.where(
            null_mask, torch.zeros_like(family_ids_t), family_ids_t + 1
        )
        joined = torch.cat(
            [
                self.family(safe_family),
                self.operand_mode(operand_mode),
                self.support_policy(support_mode),
            ],
            dim=-1,
        )
        encoded = self.mlp(joined)
        return torch.where(null_mask[:, None], torch.zeros_like(encoded), encoded)


class ContinuousReadoutConditioner(nn.Module):
    """Capacity-matched continuous-surface control arm.

    Frozen naming: this is the ``continuous_surface_readout_control``.  It
    maps the same state encoder hidden features to the same conditioning
    dimension as ``ProgramEmbedding`` with a comparable parameter budget;
    it is a capacity/readout control, NOT an ontology-free claim.
    """

    def __init__(self, state_dim: int, dim: int = 64):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(state_dim, state_dim), nn.GELU(), nn.Linear(state_dim, dim)
        )

    def forward(self, state_embedding: Tensor, active_mask: Tensor) -> Tensor:
        projected = self.readout(state_embedding)
        return torch.where(
            active_mask.to(dtype=torch.bool)[:, None],
            projected,
            torch.zeros_like(projected),
        )


class ProgramEditorUNet2D(nn.Module):
    """13-channel residual editor conditioned on a program embedding.

    Channel 12 (0-based) is the central selected-component mask; it must be
    bitwise zero for NEW_CUE episodes.  Conditioning uses the same additive
    bottleneck mechanism as the frozen v2 editor so the representation ladder
    (J6 spatial-only / J7 flat-action / J8 continuous / J9 typed) shares one
    injection location and comparable parameter budgets.
    """

    def __init__(
        self,
        visual_channels: int = 13,
        base: int = 32,
        program_dim: int = 64,
        *,
        conditioner: str = "program",
        state_dim: int = 64,
    ):
        super().__init__()
        if conditioner not in ("program", "continuous"):
            raise ValueError("conditioner must be 'program' or 'continuous'")
        self.visual_channels = visual_channels
        self.architecture_id = PROGRAM_EDITOR_ARCHITECTURE_ID
        self.conditioner = conditioner
        if conditioner == "program":
            self.condition = ProgramEmbedding(program_dim)
        else:
            self.condition = ContinuousReadoutConditioner(state_dim, program_dim)
        self.enc1 = _ConvNormAct(visual_channels, base)
        self.enc2 = _ConvNormAct(base, base * 2)
        self.bottleneck = _ConvNormAct(base * 2, base * 4)
        self.dec2 = _ConvNormAct(base * 4 + base * 2, base * 2)
        self.dec1 = _ConvNormAct(base * 2 + base, base)
        self.fusion = _SimpleConditioning(program_dim, base * 4)
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
            raise ValueError(
                "visual must be BCHW with %d channels" % self.visual_channels
            )
        if self.conditioner == "program":
            condition = self.condition(family_ids_t, operand_mode, support_mode)
            neutral_mask = family_ids_t == NULL_FAMILY_ID
        else:
            if state_embedding is None or active_mask is None:
                raise ValueError("continuous conditioner requires state embedding and mask")
            active_mask = active_mask.to(dtype=torch.bool)
            condition = self.condition(state_embedding, active_mask)
            neutral_mask = ~active_mask
        e1 = self.enc1(visual)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottleneck(F.max_pool2d(e2, 2))
        b = self.fusion(b, condition, neutral_mask)
        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.output(d1)

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def apply_program_operation(
        logits: Tensor,
        m0_center: Tensor,
        selected_component: Tensor,
        operation_ids: Tensor,
        threshold: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """SCEP execution algebra (2D central-slice application).

        ADD:  delta = prediction AND NOT current foreground (monotone union).
        REMOVE: delta = prediction AND selected component (hard complement
        protection); non-selected components cannot be edited.
        """

        if logits.shape != m0_center.shape or selected_component.shape != m0_center.shape:
            raise ValueError("logits, M0 and selected component must share shape")
        if operation_ids.ndim != 1 or operation_ids.shape[0] != logits.shape[0]:
            raise ValueError("operation_ids must have shape [B]")
        if torch.any((operation_ids != 0) & (operation_ids != 1)):
            raise ValueError("editor application requires ADD(0) or REMOVE(1)")
        current = m0_center > 0
        prediction = torch.sigmoid(logits) >= float(threshold)
        add_rows = operation_ids == 0
        delta = torch.where(
            add_rows[:, None, None, None],
            prediction & ~current,
            prediction & (selected_component > 0),
        )
        corrected = torch.where(
            add_rows[:, None, None, None],
            current | delta,
            current & ~delta,
        )
        return delta, corrected


# Private module-local helpers mirror the frozen v2 building blocks without
# importing them, so this module stays standalone (and the frozen file is
# never modified).
class _ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        groups = 8 if out_channels >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _SimpleConditioning(nn.Module):
    """Additive bottleneck projection with an exact identity NULL."""

    def __init__(self, condition_dim: int, channels: int):
        super().__init__()
        self.projection = nn.Linear(condition_dim, channels, bias=False)

    def forward(self, feature: Tensor, condition: Tensor, bypass_mask: Tensor) -> Tensor:
        shift = self.projection(condition)[:, :, None, None]
        output = feature + shift
        return torch.where(bypass_mask[:, None, None, None], feature, output)
