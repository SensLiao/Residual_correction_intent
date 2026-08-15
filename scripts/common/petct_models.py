#!/usr/bin/env python3
"""Professor-directed simple-first P2T and residual-editor modules.

The current P2T pools PET/CT/M0 features globally and at disjoint FG/BG cue
supports, then predicts the six legal joint classes plus three auxiliary slot
heads.  The current editor uses one additive structured-intent bottleneck
projection.  Cross-attention, FiLM, concat, and gated fusion remain explicitly
deferred references and are not selectable by current launchers.
No pretrained or external text/image encoder is loaded by this module.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


OPERATION_TO_ID = {"ADD": 0, "REMOVE": 1, "NULL": 2}
TARGET_TO_ID = {"SAME": 0, "NEW": 1, "NULL": 2}
SCOPE_TO_ID = {"LOCAL": 0, "COMPLETE": 1, "NULL": 2}
LEGAL_GOALS = (
    "ADD_SAME_LOCAL",
    "REMOVE_SAME_LOCAL",
    "ADD_SAME_COMPLETE",
    "REMOVE_SAME_COMPLETE",
    "ADD_NEW_COMPLETE",
    "REMOVE_NEW_COMPLETE",
)
LEGAL_GOAL_SLOTS = (
    (0, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
FUSION_MODES = ("simple",)
P2T_PRIMARY_ARCHITECTURE_ID = "simple_signed_scribble_state_pool_v2"
P2T_DEFERRED_CROSS_ATTENTION_ARCHITECTURE_ID = "scribble_query_cross_attention_v1"
P2T_ARCHITECTURE_IDS = (P2T_PRIMARY_ARCHITECTURE_ID,)
EDITOR_PRIMARY_ARCHITECTURE_ID = "simple_operation_conditioned_residual_unet_v2"


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        groups = 8 if out_channels >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, stride=stride, padding=1, bias=False
            ),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ScribbleIntentNet(nn.Module):
    """Deferred cross-attention reference; not a current selectable model.

    The input order is five PET slices, five CT slices, five M0 slices and one
    FG scribble on the prompted axial slice.  The scribble remains a separate
    query input rather than being concatenated into the state encoder.
    The scribble is not converted into text.  It supplies an attention query
    that reads spatial state tokens, which is the causal order required by P2T.
    """

    def __init__(
        self,
        state_channels: int = 15,
        width: int = 64,
        heads: int = 4,
        *,
        use_relative_geometry: bool = True,
        prompt_channels: int = 2,
    ):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by attention heads")
        self.state_channels = state_channels
        if prompt_channels != 2:
            raise ValueError("current P2T contract requires FG and BG prompt channels")
        self.prompt_channels = prompt_channels
        self.architecture_id = P2T_DEFERRED_CROSS_ATTENTION_ARCHITECTURE_ID
        self.use_relative_geometry = bool(use_relative_geometry)
        self.state_encoder = ConvNormAct(state_channels, width, stride=2)
        self.scribble_encoder = nn.Sequential(
            nn.Conv2d(prompt_channels, width, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8 if width >= 8 else 1, width),
            nn.GELU(),
        )
        self.learned_query = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.learned_query, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.relative_geometry_mlp = nn.Sequential(
            nn.Linear(4, width), nn.GELU(), nn.Linear(width, width)
        )
        self.norm = nn.LayerNorm(width)
        self.operation_head = nn.Linear(width, 2)
        self.target_head = nn.Linear(width, 2)
        self.scope_head = nn.Linear(width, 2)
        self.joint_head = nn.Linear(width, len(LEGAL_GOALS))

    @staticmethod
    def _relative_geometry(
        scribble: Tensor, prompt_map: Tensor, spacing_xy: Tensor
    ) -> Tensor:
        """Return per-token [x_rel_mm, y_rel_mm, distance_mm, inside]."""

        if spacing_xy.ndim != 2 or spacing_xy.shape != (scribble.shape[0], 2):
            raise ValueError("spacing_xy must have shape [B,2]")
        if not torch.isfinite(spacing_xy).all() or torch.any(spacing_xy <= 0):
            raise ValueError("spacing_xy must contain positive finite values")
        binary = (scribble > 0).any(dim=1).to(dtype=scribble.dtype)
        height, width = binary.shape[-2:]
        row = torch.arange(height, dtype=scribble.dtype, device=scribble.device)
        col = torch.arange(width, dtype=scribble.dtype, device=scribble.device)
        mass = binary.sum(dim=(1, 2)).clamp_min(1.0)
        center_x = (binary * row[None, :, None]).sum(dim=(1, 2)) / mass
        center_y = (binary * col[None, None, :]).sum(dim=(1, 2)) / mass

        token_height, token_width = prompt_map.shape[-2:]
        token_x = (
            torch.arange(token_height, dtype=scribble.dtype, device=scribble.device)
            + 0.5
        ) * (float(height) / float(token_height)) - 0.5
        token_y = (
            torch.arange(token_width, dtype=scribble.dtype, device=scribble.device)
            + 0.5
        ) * (float(width) / float(token_width)) - 0.5
        x_rel = (token_x[None, :, None] - center_x[:, None, None]) * spacing_xy[
            :, 0, None, None
        ]
        y_rel = (token_y[None, None, :] - center_y[:, None, None]) * spacing_xy[
            :, 1, None, None
        ]
        x_rel = x_rel.expand(-1, token_height, token_width)
        y_rel = y_rel.expand(-1, token_height, token_width)
        distance = torch.sqrt(x_rel.square() + y_rel.square())
        inside = (prompt_map[:, 0] > 0).to(dtype=scribble.dtype)
        return torch.stack([x_rel, y_rel, distance, inside], dim=-1).flatten(1, 2)

    def forward(self, visual: Tensor, spacing_xy: Tensor) -> Dict[str, Tensor]:
        if visual.ndim != 4 or visual.shape[1] != self.state_channels + self.prompt_channels:
            raise ValueError(
                "visual must be BCHW with %d channels"
                % (self.state_channels + self.prompt_channels)
            )
        state_input = visual[:, : self.state_channels]
        scribble = visual[:, self.state_channels :]
        state_map = self.state_encoder(state_input)
        scribble_features = self.scribble_encoder(scribble)
        tokens = state_map.flatten(2).transpose(1, 2)
        binary_channels = (scribble > 0).float()
        channel_mass = binary_channels.flatten(2).sum(dim=2)
        active_channels = channel_mass > 0
        single = active_channels.sum(dim=1) == 1
        polarity_blind = (active_channels.sum(dim=1) == 2) & torch.all(
            binary_channels[:, 0] == binary_channels[:, 1], dim=(1, 2)
        )
        if torch.any(~(single | polarity_blind)):
            raise ValueError(
                "P2T cue must be one polarity or the exact polarity-blind support ablation"
            )
        binary_scribble = binary_channels.amax(dim=1, keepdim=True)
        prompt_map = F.adaptive_max_pool2d(
            binary_scribble, output_size=state_map.shape[-2:]
        )
        if torch.any(prompt_map.flatten(1).sum(dim=1) == 0):
            raise RuntimeError("scribble vanished during state-token pooling")
        prompt_weight = prompt_map.flatten(2).transpose(1, 2)
        relative_geometry = self._relative_geometry(
            scribble,
            prompt_map,
            spacing_xy.to(device=visual.device, dtype=visual.dtype),
        )
        if self.use_relative_geometry:
            tokens = tokens + self.relative_geometry_mlp(relative_geometry)
        denom = prompt_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        prompt_pool = (tokens * prompt_weight).sum(dim=1, keepdim=True) / denom
        scribble_pool = scribble_features.mean(dim=(2, 3), keepdim=False).unsqueeze(1)
        query = self.learned_query.expand(visual.shape[0], -1, -1)
        query = query + prompt_pool + scribble_pool
        attended, weights = self.cross_attention(
            query, tokens, tokens, need_weights=True
        )
        embedding = self.norm(attended[:, 0])
        return {
            "joint_logits": self.joint_head(embedding),
            "operation_logits": self.operation_head(embedding),
            "target_logits": self.target_head(embedding),
            "scope_logits": self.scope_head(embedding),
            "embedding": embedding,
            "attention": weights[:, 0],
            "relative_geometry": relative_geometry,
        }


class SimpleSignedCueIntentNet(nn.Module):
    """Simple-first pooled-state P2T baseline for the six-class contract.

    It sees the 15-channel PET/CT/M0 state and independent FG/BG cue channels,
    pools state features globally and at the cue support, and combines those
    with cue and relative-geometry summaries.  It deliberately contains no
    cross-attention, pretrained encoder, or free-text generator.

    This professor-directed primary has no cross-attention.  It concatenates
    global state features, state features
    pooled at the scribble, scribble features pooled at the same support, and a
    prompt-relative physical-geometry summary before classification.
    """

    def __init__(
        self,
        state_channels: int = 15,
        width: int = 64,
        *,
        use_relative_geometry: bool = True,
        prompt_channels: int = 2,
    ):
        super().__init__()
        self.state_channels = state_channels
        if prompt_channels != 2:
            raise ValueError("current P2T contract requires FG and BG prompt channels")
        self.prompt_channels = prompt_channels
        self.architecture_id = P2T_PRIMARY_ARCHITECTURE_ID
        self.use_relative_geometry = bool(use_relative_geometry)
        self.state_encoder = ConvNormAct(state_channels, width, stride=2)
        self.scribble_encoder = nn.Sequential(
            nn.Conv2d(prompt_channels, width, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8 if width >= 8 else 1, width),
            nn.GELU(),
        )
        self.relative_geometry_mlp = nn.Sequential(
            nn.Linear(4, width), nn.GELU(), nn.Linear(width, width)
        )
        self.fusion = nn.Sequential(
            nn.Linear(width * 4, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.operation_head = nn.Linear(width, 2)
        self.target_head = nn.Linear(width, 2)
        self.scope_head = nn.Linear(width, 2)
        self.joint_head = nn.Linear(width, len(LEGAL_GOALS))

    def forward(self, visual: Tensor, spacing_xy: Tensor) -> Dict[str, Tensor]:
        if visual.ndim != 4 or visual.shape[1] != self.state_channels + self.prompt_channels:
            raise ValueError(
                "visual must be BCHW with %d channels"
                % (self.state_channels + self.prompt_channels)
            )
        state_input = visual[:, : self.state_channels]
        scribble = visual[:, self.state_channels :]
        binary_channels = (scribble > 0).to(dtype=visual.dtype)
        active_channels = binary_channels.flatten(2).sum(dim=2) > 0
        single = active_channels.sum(dim=1) == 1
        polarity_blind = (active_channels.sum(dim=1) == 2) & torch.all(
            binary_channels[:, 0] == binary_channels[:, 1], dim=(1, 2)
        )
        if torch.any(~(single | polarity_blind)):
            raise ValueError(
                "P2T cue must be one polarity or the exact polarity-blind support ablation"
            )
        binary_scribble = binary_channels.amax(dim=1, keepdim=True)

        state_map = self.state_encoder(state_input)
        scribble_map = self.scribble_encoder(scribble)
        prompt_map = F.adaptive_max_pool2d(
            binary_scribble, output_size=state_map.shape[-2:]
        )
        if torch.any(prompt_map.flatten(1).sum(dim=1) == 0):
            raise RuntimeError("scribble vanished during state-feature pooling")
        prompt_weight = prompt_map.flatten(2).transpose(1, 2)
        denominator = prompt_weight.sum(dim=1).clamp_min(1.0)

        state_tokens = state_map.flatten(2).transpose(1, 2)
        scribble_tokens = scribble_map.flatten(2).transpose(1, 2)
        state_global = state_tokens.mean(dim=1)
        state_at_prompt = (state_tokens * prompt_weight).sum(dim=1) / denominator
        scribble_at_prompt = (scribble_tokens * prompt_weight).sum(dim=1) / denominator

        relative_geometry = ScribbleIntentNet._relative_geometry(
            scribble,
            prompt_map,
            spacing_xy.to(device=visual.device, dtype=visual.dtype),
        )
        if self.use_relative_geometry:
            geometry_global = self.relative_geometry_mlp(relative_geometry).mean(dim=1)
        else:
            geometry_global = torch.zeros_like(state_global)

        embedding = self.fusion(
            torch.cat(
                [state_global, state_at_prompt, scribble_at_prompt, geometry_global],
                dim=1,
            )
        )
        return {
            "joint_logits": self.joint_head(embedding),
            "operation_logits": self.operation_head(embedding),
            "target_logits": self.target_head(embedding),
            "scope_logits": self.scope_head(embedding),
            "embedding": embedding,
            "relative_geometry": relative_geometry,
        }


def build_p2t_model(
    architecture_id: str = P2T_PRIMARY_ARCHITECTURE_ID,
    *,
    state_channels: int = 15,
    width: int = 64,
    heads: int = 4,
    use_relative_geometry: bool = True,
) -> nn.Module:
    """Build the sole current simple-first P2T architecture.

    Deferred cross-attention code is intentionally absent from
    ``P2T_ARCHITECTURE_IDS`` and cannot be selected by this factory.
    """

    if architecture_id == P2T_PRIMARY_ARCHITECTURE_ID:
        return SimpleSignedCueIntentNet(
            state_channels=state_channels,
            width=width,
            use_relative_geometry=use_relative_geometry,
        )
    raise ValueError("unsupported P2T architecture_id: %s" % architecture_id)


def p2t_architecture_contract(
    architecture_id: str, *, use_relative_geometry: bool
) -> Dict[str, object]:
    """Return receipt metadata that makes architecture comparisons explicit."""

    shared: Dict[str, object] = {
        "architecture_id": architecture_id,
        "state_channels": 15,
        "state_order": "PET5+CT5+M0_5",
        "prompt_channels": 2,
        "prompt_order": "FG_positive+BG_negative",
        "relative_geometry": bool(use_relative_geometry),
        "relative_geometry_features": [
            "x_rel_mm",
            "y_rel_mm",
            "distance_mm",
            "inside_scribble",
        ],
        "output_contract": (
            "six_legal_joint_goals_plus_operation_target_scope_auxiliary_heads"
        ),
        "free_text_generation": False,
        "external_text_or_image_encoder": False,
    }
    if architecture_id == P2T_PRIMARY_ARCHITECTURE_ID:
        shared.update(
            {
                "fusion": "global_and_prompt_support_feature_pooling_then_mlp",
                "comparison_role": "professor_directed_simple_first_primary",
                "cross_attention": False,
            }
        )
        return shared
    raise ValueError("unsupported P2T architecture_id: %s" % architecture_id)


def validate_p2t_architecture_selection(
    config: Mapping[str, object],
    architecture_id: str,
    input_ablation: str,
) -> str:
    """Validate a CLI architecture choice and return its receipt arm role."""

    p2t = config.get("p2t")
    if not isinstance(p2t, Mapping):
        raise ValueError("config.p2t must be a mapping")
    configured_primary = p2t.get("primary_architecture_id")
    if configured_primary != P2T_PRIMARY_ARCHITECTURE_ID:
        raise ValueError("the frozen primary P2T architecture_id changed")
    if architecture_id == P2T_PRIMARY_ARCHITECTURE_ID:
        return "primary" if input_ablation == "full" else "ablation"
    raise ValueError("deferred P2T architectures are not executable in this campaign")


def decode_joint_goal(
    joint_logits: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Decode the six-class joint head into three consistent binary slots."""

    if joint_logits.ndim != 2 or joint_logits.shape[1] != len(LEGAL_GOALS):
        raise ValueError("joint logits must have shape [B,6]")
    joint = joint_logits.argmax(dim=1)
    slots = torch.tensor(LEGAL_GOAL_SLOTS, device=joint.device)[joint]
    return joint, slots[:, 0], slots[:, 1], slots[:, 2]


class IntentEmbedding(nn.Module):
    """Three-slot six-class intent representation with an exact NULL bypass."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.operation = nn.Embedding(len(OPERATION_TO_ID), dim)
        self.target = nn.Embedding(len(TARGET_TO_ID), dim)
        self.scope = nn.Embedding(len(SCOPE_TO_ID), dim)
        with torch.no_grad():
            self.operation.weight[OPERATION_TO_ID["NULL"]].zero_()
            self.target.weight[TARGET_TO_ID["NULL"]].zero_()
            self.scope.weight[SCOPE_TO_ID["NULL"]].zero_()
        self.mlp = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))

    @staticmethod
    def _validate_ids(
        operation_id: Tensor, target_id: Tensor, scope_id: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if operation_id.ndim != 1 or target_id.ndim != 1 or scope_id.ndim != 1:
            raise ValueError("operation, target and scope ids must be one-dimensional")
        if operation_id.shape != target_id.shape or target_id.shape != scope_id.shape:
            raise ValueError("operation, target and scope ids must share shape")
        null_mask = (operation_id == OPERATION_TO_ID["NULL"]) & (
            target_id == TARGET_TO_ID["NULL"]
        ) & (
            scope_id == SCOPE_TO_ID["NULL"]
        )
        operation_only_mask = (
            ((operation_id == OPERATION_TO_ID["ADD"]) | (operation_id == OPERATION_TO_ID["REMOVE"]))
            & (target_id == TARGET_TO_ID["NULL"])
            & (scope_id == SCOPE_TO_ID["NULL"])
        )
        regular_mask = ~(null_mask | operation_only_mask)
        valid_regular = ((operation_id == 0) | (operation_id == 1)) & (
            ((target_id == 0) & ((scope_id == 0) | (scope_id == 1)))
            | ((target_id == 1) & (scope_id == 1))
        )
        if torch.any(regular_mask & ~valid_regular):
            raise ValueError("unsupported operation/target/scope condition id tuple")
        partial_null = (
            (operation_id == OPERATION_TO_ID["NULL"])
            | (target_id == TARGET_TO_ID["NULL"])
            | (scope_id == SCOPE_TO_ID["NULL"])
        ) & ~(null_mask | operation_only_mask)
        if torch.any(partial_null):
            raise ValueError("NULL must be used for all three intent slots")
        return null_mask, operation_only_mask

    def forward(
        self, operation_id: Tensor, target_id: Tensor, scope_id: Tensor
    ) -> Tensor:
        null_mask, operation_only_mask = self._validate_ids(
            operation_id, target_id, scope_id
        )

        visible_mask = ~null_mask
        safe_operation = torch.where(
            visible_mask,
            operation_id,
            torch.full_like(operation_id, OPERATION_TO_ID["ADD"]),
        )
        safe_target = torch.where(
            visible_mask,
            target_id,
            torch.full_like(target_id, TARGET_TO_ID["SAME"]),
        )
        safe_scope = torch.where(
            visible_mask,
            scope_id,
            torch.full_like(scope_id, SCOPE_TO_ID["LOCAL"]),
        )
        joined = torch.cat(
            [
                self.operation(safe_operation),
                self.target(safe_target),
                self.scope(safe_scope),
            ],
            dim=-1,
        )
        encoded = self.mlp(joined)
        return torch.where(null_mask[:, None], torch.zeros_like(encoded), encoded)

    def slot_tokens(
        self, operation_id: Tensor, target_id: Tensor, scope_id: Tensor
    ) -> Tensor:
        """Return explicit [operation, target, scope] tokens."""

        null_mask, operation_only_mask = self._validate_ids(
            operation_id, target_id, scope_id
        )
        visible_mask = ~null_mask
        safe_operation = torch.where(
            visible_mask,
            operation_id,
            torch.full_like(operation_id, OPERATION_TO_ID["ADD"]),
        )
        safe_target = torch.where(
            visible_mask,
            target_id,
            torch.full_like(target_id, TARGET_TO_ID["SAME"]),
        )
        safe_scope = torch.where(
            visible_mask,
            scope_id,
            torch.full_like(scope_id, SCOPE_TO_ID["LOCAL"]),
        )
        tokens = torch.stack(
            [
                self.operation(safe_operation),
                self.target(safe_target),
                self.scope(safe_scope),
            ],
            dim=1,
        )
        return torch.where(null_mask[:, None, None], torch.zeros_like(tokens), tokens)


class FiLM(nn.Module):
    def __init__(self, condition_dim: int, channels: int):
        super().__init__()
        self.affine = nn.Linear(condition_dim, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(
        self,
        feature: Tensor,
        condition: Tensor,
        bypass_mask: Optional[Tensor] = None,
    ) -> Tensor:
        gamma, beta = self.affine(condition).chunk(2, dim=-1)
        if bypass_mask is not None:
            if bypass_mask.ndim != 1 or bypass_mask.shape[0] != feature.shape[0]:
                raise ValueError("FiLM bypass mask must have shape [B]")
            active = (~bypass_mask).to(dtype=gamma.dtype)[:, None]
            gamma = gamma * active
            beta = beta * active
        return feature * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class SimpleConditioning(nn.Module):
    """One additive bottleneck projection; exact NULL is an identity."""

    def __init__(self, condition_dim: int, channels: int):
        super().__init__()
        self.projection = nn.Linear(condition_dim, channels, bias=False)

    def forward(
        self, feature: Tensor, condition: Tensor, bypass_mask: Tensor
    ) -> Tensor:
        shift = self.projection(condition)[:, :, None, None]
        output = feature + shift
        return torch.where(bypass_mask[:, None, None, None], feature, output)


class ConcatFusion(nn.Module):
    """Residual concatenation fusion with an exact zero-condition baseline."""

    def __init__(self, condition_dim: int, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels + condition_dim, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(
        self, feature: Tensor, condition: Tensor, bypass_mask: Tensor
    ) -> Tensor:
        condition_map = condition[:, :, None, None].expand(
            -1, -1, feature.shape[-2], feature.shape[-1]
        )
        zero_map = torch.zeros_like(condition_map)
        conditioned = self.block(torch.cat([feature, condition_map], dim=1))
        neutral = self.block(torch.cat([feature, zero_map], dim=1))
        output = feature + conditioned - neutral
        return torch.where(bypass_mask[:, None, None, None], feature, output)


class GatedFusion(nn.Module):
    """Intent-derived multiplicative/additive gate with an exact NULL identity."""

    def __init__(self, condition_dim: int, channels: int):
        super().__init__()
        self.affine = nn.Linear(condition_dim, channels * 2, bias=False)

    def forward(
        self, feature: Tensor, condition: Tensor, bypass_mask: Tensor
    ) -> Tensor:
        gate, shift = self.affine(condition).chunk(2, dim=-1)
        output = feature * (1.0 + torch.tanh(gate)[:, :, None, None])
        output = output + shift[:, :, None, None]
        return torch.where(bypass_mask[:, None, None, None], feature, output)


class CrossAttentionFusion(nn.Module):
    """Bidirectional structured-slot <-> bottleneck-image cross-attention."""

    def __init__(self, condition_dim: int, channels: int, heads: int = 4):
        super().__init__()
        if channels % heads:
            raise ValueError("bottleneck channels must be divisible by attention heads")
        self.slot_projection = nn.Linear(condition_dim, channels, bias=False)
        self.slot_to_image = nn.MultiheadAttention(
            channels, heads, dropout=0.0, batch_first=True
        )
        self.image_to_slot = nn.MultiheadAttention(
            channels, heads, dropout=0.0, batch_first=True
        )
        self.output_projection = nn.Linear(channels, channels, bias=False)

    def forward(
        self, feature: Tensor, slot_tokens: Tensor, bypass_mask: Tensor
    ) -> Tensor:
        image_tokens = feature.flatten(2).transpose(1, 2)
        slots = self.slot_projection(slot_tokens)
        slot_context, _ = self.slot_to_image(slots, image_tokens, image_tokens)
        enriched_slots = slots + slot_context
        image_context, _ = self.image_to_slot(
            image_tokens, enriched_slots, enriched_slots
        )
        output = image_tokens + self.output_projection(image_context)
        output = output.transpose(1, 2).reshape_as(feature)
        return torch.where(bypass_mask[:, None, None, None], feature, output)


class ResidualEditorUNet2D(nn.Module):
    """Polarity-aware 2.5D residual editor with explicit fusion mode."""

    def __init__(
        self,
        visual_channels: int = 12,
        base: int = 32,
        intent_dim: int = 64,
        *,
        fusion_mode: str = "simple",
    ):
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError("unsupported fusion_mode: %s" % fusion_mode)
        self.visual_channels = visual_channels
        self.fusion_mode = fusion_mode
        self.architecture_id = EDITOR_PRIMARY_ARCHITECTURE_ID
        self.intent = IntentEmbedding(intent_dim)
        self.enc1 = ConvNormAct(visual_channels, base)
        self.enc2 = ConvNormAct(base, base * 2)
        self.bottleneck = ConvNormAct(base * 2, base * 4)
        self.dec2 = ConvNormAct(base * 4 + base * 2, base * 2)
        self.dec1 = ConvNormAct(base * 2 + base, base)
        if fusion_mode == "simple":
            self.bottleneck_fusion = SimpleConditioning(intent_dim, base * 4)
        elif fusion_mode == "film":
            self.film1 = FiLM(intent_dim, base)
            self.film2 = FiLM(intent_dim, base * 2)
            self.filmb = FiLM(intent_dim, base * 4)
            self.filmd2 = FiLM(intent_dim, base * 2)
            self.filmd1 = FiLM(intent_dim, base)
        elif fusion_mode == "concat":
            self.bottleneck_fusion = ConcatFusion(intent_dim, base * 4)
        elif fusion_mode == "gated":
            self.bottleneck_fusion = GatedFusion(intent_dim, base * 4)
        else:
            self.bottleneck_fusion = CrossAttentionFusion(intent_dim, base * 4)
        self.output = nn.Conv2d(base, 1, 1)

    def forward(
        self,
        visual: Tensor,
        operation_id: Tensor,
        target_id: Tensor,
        scope_id: Tensor,
    ) -> Tensor:
        if visual.ndim != 4 or visual.shape[1] != self.visual_channels:
            raise ValueError(
                "visual must be BCHW with %d channels" % self.visual_channels
            )
        neutral_mask = (operation_id == OPERATION_TO_ID["NULL"]) & (
            target_id == TARGET_TO_ID["NULL"]
        ) & (
            scope_id == SCOPE_TO_ID["NULL"]
        )
        z = self.intent(operation_id, target_id, scope_id)
        e1 = self.enc1(visual)
        if self.fusion_mode == "film":
            e1 = self.film1(e1, z, neutral_mask)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        if self.fusion_mode == "film":
            e2 = self.film2(e2, z, neutral_mask)
        b = self.bottleneck(F.max_pool2d(e2, 2))
        if self.fusion_mode == "film":
            b = self.filmb(b, z, neutral_mask)
        elif self.fusion_mode == "simple":
            b = self.bottleneck_fusion(b, z, neutral_mask)
        elif self.fusion_mode == "cross_attention":
            b = self.bottleneck_fusion(
                b,
                self.intent.slot_tokens(operation_id, target_id, scope_id),
                neutral_mask,
            )
        else:
            b = self.bottleneck_fusion(b, z, neutral_mask)
        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        if self.fusion_mode == "film":
            d2 = self.filmd2(d2, z, neutral_mask)
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        if self.fusion_mode == "film":
            d1 = self.filmd1(d1, z, neutral_mask)
        return self.output(d1)

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def apply_operation(
        logits: Tensor,
        m0: Tensor,
        operation_id: Tensor,
        threshold: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        if logits.shape != m0.shape:
            raise ValueError("logits and M0 must share shape")
        if operation_id.ndim != 1 or operation_id.shape[0] != logits.shape[0]:
            raise ValueError("operation_id must have shape [B]")
        if torch.any((operation_id != OPERATION_TO_ID["ADD"]) & (operation_id != OPERATION_TO_ID["REMOVE"])):
            raise ValueError("editor application requires ADD or REMOVE operation")
        current = m0 > 0
        selected = torch.sigmoid(logits) >= float(threshold)
        add_mask = operation_id == OPERATION_TO_ID["ADD"]
        delta = torch.where(
            add_mask[:, None, None, None], selected & ~current, selected & current
        )
        corrected = torch.where(
            add_mask[:, None, None, None],
            torch.logical_or(current, delta),
            current & ~delta,
        )
        return delta, corrected


def editor_condition_ids(
    operation: str, target: str, scope: str, intent_mode: str
) -> Tuple[int, int, int]:
    if intent_mode == "NULL":
        return (
            OPERATION_TO_ID["NULL"],
            TARGET_TO_ID["NULL"],
            SCOPE_TO_ID["NULL"],
        )
    if intent_mode == "OPERATION_ONLY":
        if operation not in ("ADD", "REMOVE"):
            raise ValueError("operation-only intent requires ADD or REMOVE")
        return (
            OPERATION_TO_ID[operation],
            TARGET_TO_ID["NULL"],
            SCOPE_TO_ID["NULL"],
        )
    if intent_mode != "CORRECT":
        raise ValueError("intent_mode must be CORRECT, OPERATION_ONLY or NULL")
    if operation not in ("ADD", "REMOVE"):
        raise ValueError("non-null intent requires ADD or REMOVE")
    if target not in ("SAME", "NEW") or scope not in (
        "LOCAL",
        "COMPLETE",
    ):
        raise ValueError("non-null intent requires legal target and scope")
    if target == "NEW" and scope == "LOCAL":
        raise ValueError("NEW_LOCAL is structurally invalid")
    return OPERATION_TO_ID[operation], TARGET_TO_ID[target], SCOPE_TO_ID[scope]
