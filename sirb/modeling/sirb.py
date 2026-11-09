"""SIRB: the whole correction network, from the 14 input channels to native role logits.

One state is processed in three steps:

1. :meth:`SIRB.prepare_context`, once per state (and per flip): the whole-volume encoder on the whole-volume grid,
   the local encoder up to level 2 on the scribble window, and the source encoder on the scribble's sources.
2. :meth:`SIRB.forward_window`, per window of the 3 mm grid: the local encoder, source-conditioned role prediction
   at 12 mm and 6 mm and the role decoder, giving decoder features and coarse role logits on the window grid, and
   the 6 mm role logits of every source.
3. :meth:`SIRB.refine_native`, per editable native voxel of a window: native refinement with the source weights and
   the native head, giving the native role logits (channel order: target error, other errors, correct tissue).
   :meth:`SIRB.native_role_logits` samples a window's outputs at native voxels and refines them in one call.

Merging overlapping windows, the left-right flip and the update rule are the predictor's job.

The variants of the paper are switches of the same network; every variant keeps every parameter, so checkpoints of
all variants have the same keys, and parameters a variant does not use receive no gradient:

* ``scribble_conditioning=False`` (No scribble conditioning): the encoders read only PET, CT, the mask and its
  covered fraction, the source pathway (source encoder, role head, source weights) is not used, the decoder receives
  uniform role maps, and native voxels keep only their image, mask, offset and spacing values;
* ``correct_tissue_role=False`` (No correct-tissue role): the correct-tissue logit is the constant
  ``FIXED_CORRECT_TISSUE_LOGIT`` wherever role logits are produced (role head, decoder, native head), so correct
  tissue takes part in no selection, weight or probability;
* ``native_refinement=False`` (Coarse roles only): every native voxel takes the coarse role logits sampled at it;
* Merged non-target roles changes only the training objective and is a configuration key of the trainer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import nn

from ..data.geometry import MAX_SOURCES, GeometryError, grid_world_points, sample_world_features
from ..data.roles import ROLE_COUNT
from .common import KEPT_WITHOUT_SCRIBBLE, sample_to_grid, zero_channels
from .local_encoder import LocalEncoder
from .native_refinement import NativeRefinement, sample_source_role_logits
from .role_decoder import (LEVELS, RoleDecoder, SourceRoleHead, relative_source_geometry, role_usage_statistics,
                           select_source, selection_statistics)
from .source_encoder import SOURCE_INPUT_WIDTH, SOURCE_WIDTH, ScribbleSources, SourceEncoder, pool_source_inputs
from .whole_volume_encoder import WholeVolumeEncoder

__all__ = ["SIRB", "StateContext"]

#: Role maps given to the decoder without scribble conditioning: log(1/3) for every role.
UNIFORM_ROLE_LOG_PROBABILITY = math.log(1.0 / ROLE_COUNT)
#: The local-encoder level whose features the source encoder pools from the scribble window (12 mm).
SCRIBBLE_WINDOW_LEVEL = 2


@dataclass
class StateContext:
    """What :meth:`SIRB.prepare_context` computes once per state; every window of the state reuses it."""
    whole_volume_features: torch.Tensor          # [B,64,z,y,x]
    whole_volume_feature_affine: torch.Tensor    # affine of the whole-volume feature grid
    whole_volume_affine: torch.Tensor            # affine of the whole-volume input grid
    whole_volume_shape: tuple                    # ZYX shape of the whole-volume input grid
    whole_volume_valid: torch.Tensor | None      # [B,1,Z,Y,X] input cells that overlap valid native voxels
    source_vectors: torch.Tensor                 # [B,16,128]
    source_valid: torch.Tensor                   # [B,16]
    representative_xyz: torch.Tensor             # [B,16,3] float64
    source_extent: torch.Tensor                  # [B,16,3]
    window_fraction: torch.Tensor                # [B,16]
    examination_lower_xyz: torch.Tensor          # [B,3] float64
    examination_extent_xyz: torch.Tensor         # [B,3] float64
    examination_diagonal_mm: torch.Tensor        # [B] float64
    source_inputs: torch.Tensor                  # [B,16,340]


class SIRB(nn.Module):
    """The SIRB network (module docstring). Variant switches are required keywords; the last three settings change
    memory and time only (activation checkpointing in the encoders and decoder, sources per chunk of the 6 mm role
    head, recomputation of those chunks in the backward pass)."""

    def __init__(self, *, window_size: int, scribble_conditioning: bool, correct_tissue_role: bool,
                 native_refinement: bool, activation_checkpointing: bool = False, source_chunk: int = 4,
                 recompute_role_head: bool = False) -> None:
        super().__init__()
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 8 or window_size % 8:
            raise ValueError(f"window_size must be a multiple of eight, at least 8; got {window_size!r}")
        for name, value in (("scribble_conditioning", scribble_conditioning),
                            ("correct_tissue_role", correct_tissue_role), ("native_refinement", native_refinement),
                            ("activation_checkpointing", activation_checkpointing),
                            ("recompute_role_head", recompute_role_head)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} actual={value!r}, expected true/false")
        if isinstance(source_chunk, bool) or not isinstance(source_chunk, int) or not 1 <= source_chunk <= MAX_SOURCES:
            raise ValueError(f"source_chunk actual={source_chunk!r}, expected 1..{MAX_SOURCES}")
        self.window_size = window_size
        self.scribble_conditioning = scribble_conditioning
        self.correct_tissue_role = correct_tissue_role
        self.use_native_refinement = native_refinement
        self.source_chunk = source_chunk
        self.recompute_role_head = recompute_role_head
        self.local_encoder = LocalEncoder()
        self.whole_volume_encoder = WholeVolumeEncoder()
        self.source_encoder = SourceEncoder()
        self.role_head = SourceRoleHead(correct_tissue_role=correct_tissue_role)
        self.role_decoder = RoleDecoder(correct_tissue_role=correct_tissue_role)
        self.native_refinement = NativeRefinement(scribble_conditioning=scribble_conditioning,
                                                  correct_tissue_role=correct_tissue_role)
        self.activation_checkpointing = activation_checkpointing

    @property
    def activation_checkpointing(self) -> bool:
        return self.local_encoder.activation_checkpointing

    @activation_checkpointing.setter
    def activation_checkpointing(self, value: bool) -> None:
        for module in (self.local_encoder, self.whole_volume_encoder, self.role_decoder):
            module.activation_checkpointing = bool(value)

    # ------------------------------------------------------------------ reports
    def parameter_counts(self) -> dict[str, int]:
        counts = {name: sum(p.numel() for p in module.parameters()) for name, module in self.named_children()}
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def branch_gradient_norms(self) -> dict[str, torch.Tensor]:
        """Gradient norms of the source encoder, the role head per level, the native head, the neighbourhood encoder
        and the two projections of native refinement (after a backward pass; zero for a branch without gradient)."""
        def norm(parameters):
            values = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
            return torch.stack(values).sum().sqrt() if values else torch.zeros(())
        head, refinement = self.role_head, self.native_refinement
        level_12mm = [head.key_projection_12mm, head.key_norm_12mm, head.hidden, head.output_12mm]
        level_6mm = [head.key_projection_6mm, head.key_norm_6mm, head.output_6mm]
        return {"source_encoder": norm(self.source_encoder.parameters()),
                "role_head_12mm": norm([p for module in level_12mm for p in module.parameters()]),
                "role_head_6mm": norm([p for module in level_6mm for p in module.parameters()]),
                "native_head": norm(refinement.native_head.parameters()),
                "neighbourhood_encoder": norm(refinement.neighbourhood_encoder.parameters()),
                "source_projection": norm(refinement.source_projection.parameters()),
                "context_projection": norm(refinement.context_projection.parameters())}

    # ------------------------------------------------------------------ context
    def _validate_window(self, image: torch.Tensor, state: torch.Tensor) -> None:
        expected = (self.window_size,) * 3
        if image.ndim != 5 or tuple(image.shape[2:]) != expected:
            raise ValueError(f"window shape must be {expected}; got {tuple(image.shape)}")
        if state.shape[0] != image.shape[0] or state.shape[2:] != image.shape[2:]:
            raise ValueError(f"window image/state mismatch: {tuple(image.shape)} vs {tuple(state.shape)}")

    @staticmethod
    def _source_list(sources, batch: int) -> list[ScribbleSources]:
        items = [sources] * batch if isinstance(sources, ScribbleSources) else list(sources)
        if len(items) != batch or not all(isinstance(item, ScribbleSources) for item in items):
            raise ValueError(f"sources must be one ScribbleSources or one per batch item ({batch})")
        if not all(item.source_valid.any() for item in items):
            raise ValueError("an empty scribble must leave the mask unchanged before the model runs")
        return items

    @staticmethod
    def _examination_box(examination_bounds_xyz, batch: int, device):
        bounds = torch.as_tensor(examination_bounds_xyz, device=device, dtype=torch.float64)
        if bounds.ndim == 2:
            bounds = bounds[None].expand(batch, -1, -1)
        if bounds.shape != (batch, 2, 3) or not bool(torch.isfinite(bounds).all()):
            raise ValueError(f"examination bounds must be finite [B,2,3] lower/upper xyz; got {tuple(bounds.shape)}")
        lower, extent = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
        return lower, extent

    def prepare_context(self, whole_volume_input: torch.Tensor, whole_volume_affine_xyz, window_image: torch.Tensor,
                        window_state: torch.Tensor, window_affine_xyz, sources, correction_sign,
                        examination_bounds_xyz, *, whole_volume_valid: torch.Tensor | None = None,
                        window_valid: torch.Tensor | None = None) -> StateContext:
        """The state's :class:`StateContext`.

        ``whole_volume_input`` [B,14,Z,Y,X] on the whole-volume grid; ``window_image`` [B,2,P,P,P] and
        ``window_state`` [B,12,P,P,P] on the scribble window; ``sources`` one :class:`ScribbleSources` (shared by the
        batch) or one per item; ``correction_sign`` +1 (addition) or -1 (removal), a scalar or [B];
        ``examination_bounds_xyz`` [2,3] or [B,2,3] lower and upper world corners of the native voxel cells. The
        validity masks mark the cells that overlap valid native voxels.
        """
        if not self.scribble_conditioning:
            return self._prepare_context_without_scribble(
                whole_volume_input, whole_volume_affine_xyz, window_image, window_state, sources,
                examination_bounds_xyz, whole_volume_valid=whole_volume_valid)
        self._validate_window(window_image, window_state)
        batch = whole_volume_input.shape[0]
        if window_image.shape[0] != batch:
            raise ValueError(f"window/whole-volume batch mismatch: {window_image.shape[0]} vs {batch}")
        items = self._source_list(sources, batch)
        device = whole_volume_input.device
        whole_volume_affine = torch.as_tensor(whole_volume_affine_xyz, device=device, dtype=torch.float64)
        window_affine = torch.as_tensor(window_affine_xyz, device=device, dtype=torch.float64)
        lower, extent = self._examination_box(examination_bounds_xyz, batch, device)
        if not bool((extent > 0).all()):
            raise ValueError(f"examination bounds: the upper corner must exceed the lower corner: {extent}")
        diagonal = torch.linalg.vector_norm(extent, dim=-1)
        sign = torch.as_tensor(correction_sign, device=device)
        if sign.ndim == 0:
            sign = sign.expand(batch)
        if sign.shape != (batch,):
            raise ValueError(f"correction_sign must have {batch} entries; got {tuple(sign.shape)}")
        whole_volume_features = self.whole_volume_encoder(whole_volume_input)
        whole_volume_frame = self.whole_volume_encoder.feature_affine(whole_volume_affine)
        window_features = self.local_encoder.forward_to_level(window_image, window_state,
                                                              max_level=SCRIBBLE_WINDOW_LEVEL)[SCRIBBLE_WINDOW_LEVEL]
        window_frame = self.local_encoder.feature_affines(window_affine)[SCRIBBLE_WINDOW_LEVEL]
        correction_embedding = self.source_encoder.embed_correction_type(sign)
        pooled = pool_source_inputs(
            items, whole_volume_features=whole_volume_features, whole_volume_feature_affine_xyz=whole_volume_frame,
            whole_volume_affine_xyz=whole_volume_affine, whole_volume_shape_zyx=tuple(whole_volume_input.shape[2:]),
            whole_volume_valid_zyx=whole_volume_valid, window_features=window_features,
            window_feature_affine_xyz=window_frame, window_affine_xyz=window_affine,
            window_shape_zyx=tuple(window_image.shape[2:]), window_valid_zyx=window_valid,
            examination_lower_xyz=lower, examination_extent_xyz=extent, correction_embedding=correction_embedding)
        if not bool(pooled.source_valid.any(dim=1).all()):
            raise GeometryError("a non-empty scribble has no source inside the whole-volume grid")
        source_vectors = self.source_encoder(pooled.source_inputs, pooled.source_valid)
        return StateContext(whole_volume_features, whole_volume_frame, whole_volume_affine,
                            tuple(whole_volume_input.shape[2:]), whole_volume_valid, source_vectors,
                            pooled.source_valid, pooled.representative_xyz, pooled.source_extent,
                            pooled.window_fraction, lower, extent, diagonal, pooled.source_inputs)

    def _prepare_context_without_scribble(self, whole_volume_input, whole_volume_affine_xyz, window_image,
                                          window_state, sources, examination_bounds_xyz, *,
                                          whole_volume_valid) -> StateContext:
        """No scribble conditioning: no scribble channel, no scribble window, no source vectors (all zero)."""
        whole_volume_input = zero_channels(whole_volume_input, 1, KEPT_WITHOUT_SCRIBBLE["whole_volume"])
        self._validate_window(window_image, window_state)
        batch = whole_volume_input.shape[0]
        items = self._source_list(sources, batch)
        device = whole_volume_input.device
        whole_volume_affine = torch.as_tensor(whole_volume_affine_xyz, device=device, dtype=torch.float64)
        lower, extent = self._examination_box(examination_bounds_xyz, batch, device)
        if not bool((extent > 0).all()):
            raise GeometryError(f"examination bounds: the upper corner must exceed the lower corner: {extent}")
        whole_volume_features = self.whole_volume_encoder(whole_volume_input)
        whole_volume_frame = self.whole_volume_encoder.feature_affine(whole_volume_affine)
        valid = torch.as_tensor(np.stack([item.source_valid for item in items]), device=device, dtype=torch.bool)
        representative = torch.as_tensor(np.stack([item.representative_xyz for item in items]), device=device,
                                         dtype=torch.float64)
        zeros = whole_volume_features.new_zeros
        return StateContext(whole_volume_features, whole_volume_frame, whole_volume_affine,
                            tuple(whole_volume_input.shape[2:]), whole_volume_valid,
                            zeros(batch, MAX_SOURCES, SOURCE_WIDTH), valid, representative, zeros(batch, MAX_SOURCES, 3),
                            zeros(batch, MAX_SOURCES), lower, extent, torch.linalg.vector_norm(extent, dim=-1),
                            zeros(batch, MAX_SOURCES, SOURCE_INPUT_WIDTH))

    def expand_context(self, context: StateContext, batch: int) -> StateContext:
        """One state's context repeated for ``batch`` windows of that state (batched inference)."""
        if context.whole_volume_features.shape[0] != 1:
            raise ValueError(f"context batch actual={context.whole_volume_features.shape[0]}, expected=1 (one state)")

        def repeat(value):
            return None if value is None else value.expand(batch, *value.shape[1:]).contiguous()
        return replace(context, whole_volume_features=repeat(context.whole_volume_features),
                       whole_volume_valid=repeat(context.whole_volume_valid),
                       source_vectors=repeat(context.source_vectors), source_valid=repeat(context.source_valid),
                       representative_xyz=repeat(context.representative_xyz),
                       source_extent=repeat(context.source_extent), window_fraction=repeat(context.window_fraction),
                       examination_lower_xyz=repeat(context.examination_lower_xyz),
                       examination_extent_xyz=repeat(context.examination_extent_xyz),
                       examination_diagonal_mm=repeat(context.examination_diagonal_mm),
                       source_inputs=repeat(context.source_inputs))

    # ------------------------------------------------------------------ window forward
    @staticmethod
    def _cells(shape_zyx, frame, batch):
        points = grid_world_points(shape_zyx, frame, device=frame.device, dtype=torch.float64)
        return points.reshape(-1, 3)[None].expand(batch, -1, -1) if points.ndim == 4 \
            else points.reshape(points.shape[0], -1, 3)

    def forward_window(self, image: torch.Tensor, state: torch.Tensor, window_affine_xyz, context: StateContext, *,
                       diagnostics: bool = False) -> dict[str, Any]:
        """Decoder features and coarse role logits of a batch of windows, with the 6 mm source role logits.

        ``image`` [B,2,P,P,P], ``state`` [B,12,P,P,P]; ``window_affine_xyz`` (4,4) or one per item (B,4,4).
        Returns ``features`` [B,32,P,P,P], ``coarse_role_logits`` [B,3,P,P,P] and ``feature_affine`` (their grid),
        ``source_role_logits`` and ``source_role_affines`` per level ("12mm" [B,16,3,P/4,P/4,P/4] and "6mm"
        [B,16,3,P/2,P/2,P/2]), ``source_vectors`` and ``source_valid`` (from the context) and, with
        ``diagnostics``, detached monitoring readings.
        """
        if not self.scribble_conditioning:
            return self._forward_window_without_scribble(image, state, window_affine_xyz, context,
                                                         diagnostics=diagnostics)
        self._validate_window(image, state)
        batch = image.shape[0]
        if context.whole_volume_features.shape[0] != batch:
            raise ValueError(f"context/window batch mismatch: {context.whole_volume_features.shape[0]} vs {batch}")
        window_affine = torch.as_tensor(window_affine_xyz, device=image.device, dtype=torch.float64)
        window_shape = tuple(image.shape[2:])
        frames = self.local_encoder.feature_affines(window_affine)
        pyramid = self.local_encoder(image, state)
        level1, level2 = pyramid[1], pyramid[2]
        valid = context.source_valid
        head = self.role_head
        source_term = head.source_term(context.source_vectors, valid)

        def geometry(cells, start=0, stop=MAX_SOURCES):
            return relative_source_geometry(cells, context.representative_xyz[:, start:stop],
                                            context.examination_extent_xyz, context.examination_diagonal_mm,
                                            context.source_extent[:, start:stop], context.window_fraction[:, start:stop])

        # 12 mm: source role logits and the selected source.
        shape_12mm = tuple(level2.shape[2:])
        cells_12mm = self._cells(shape_12mm, frames[2], batch)
        keys_12mm = head.key_features(level2, level=LEVELS[0])
        similarity = head.similarity(context.source_vectors, valid, keys_12mm)
        logits_12mm = head.logits_12mm(head.key_term_from(keys_12mm), source_term, geometry(cells_12mm),
                                       similarity=similarity)
        logits_12mm = logits_12mm.reshape(batch, MAX_SOURCES, ROLE_COUNT, *shape_12mm)
        role_map_12mm, selected_12mm = select_source(logits_12mm, valid)

        # 6 mm: a residual on the upsampled 12 mm logits.
        shape_6mm = tuple(level1.shape[2:])
        upsampled = sample_to_grid(logits_12mm.reshape(batch, MAX_SOURCES * ROLE_COUNT, *shape_12mm), frames[2],
                                   shape_6mm, frames[1], support_affine_xyz=window_affine,
                                   support_shape_zyx=window_shape)[0]
        upsampled = upsampled.reshape(batch, MAX_SOURCES, ROLE_COUNT, -1)
        cells_6mm = self._cells(shape_6mm, frames[1], batch)
        key_term_6mm = head.key_term(level1, level=LEVELS[1])
        logits_6mm, residual = head.logits_6mm(key_term_6mm, source_term, upsampled,
                                               lambda start, stop: geometry(cells_6mm, start, stop),
                                               chunk=self.source_chunk, recompute=self.recompute_role_head)
        logits_6mm = logits_6mm.reshape(batch, MAX_SOURCES, ROLE_COUNT, *shape_6mm)
        role_map_6mm, selected_6mm = select_source(logits_6mm, valid)

        role_map_6mm_up = sample_to_grid(role_map_6mm, frames[1], tuple(pyramid[0].shape[2:]), frames[0],
                                         support_affine_xyz=window_affine, support_shape_zyx=window_shape)[0]
        whole_volume_local, whole_volume_coverage = sample_to_grid(
            context.whole_volume_features, context.whole_volume_feature_affine, shape_12mm, frames[2],
            support_affine_xyz=context.whole_volume_affine, support_shape_zyx=context.whole_volume_shape,
            support_valid_zyx=context.whole_volume_valid)
        features, coarse_logits = self.role_decoder(pyramid, whole_volume_local,
                                                    [role_map_6mm_up, role_map_6mm, role_map_12mm])
        output = self._window_output(features, coarse_logits, frames, batch, context,
                                     {LEVELS[0]: logits_12mm, LEVELS[1]: logits_6mm})
        if diagnostics:
            # Small detached readings for monitoring only; never part of a trainable path.
            with torch.no_grad():
                readings_12mm = selection_statistics(selected_12mm, MAX_SOURCES)
                readings_6mm = selection_statistics(selected_6mm, MAX_SOURCES)
                output["diagnostics"] = {
                    "source_valid": valid, "whole_volume_valid": whole_volume_coverage,
                    "selection_entropy": {LEVELS[0]: readings_12mm["entropy"], LEVELS[1]: readings_6mm["entropy"]},
                    "top_source_share": {LEVELS[0]: readings_12mm["top_share"], LEVELS[1]: readings_6mm["top_share"]},
                    "selected_source_counts": {LEVELS[0]: readings_12mm["counts"], LEVELS[1]: readings_6mm["counts"]},
                    "role_usage_6mm": role_usage_statistics(residual.detach(), upsampled.detach(), valid),
                    "role_shapes_zyx": {LEVELS[0]: shape_12mm, LEVELS[1]: shape_6mm}}
        return output

    def _forward_window_without_scribble(self, image, state, window_affine_xyz, context, *, diagnostics):
        """No scribble conditioning: the decoder receives uniform role maps; the source role logits are zero."""
        state = zero_channels(state, 1, KEPT_WITHOUT_SCRIBBLE["state"])
        self._validate_window(image, state)
        batch = image.shape[0]
        if context.whole_volume_features.shape[0] != batch:
            raise ValueError(f"context/window batch mismatch: {context.whole_volume_features.shape[0]} vs {batch}")
        window_affine = torch.as_tensor(window_affine_xyz, device=image.device, dtype=torch.float64)
        frames = self.local_encoder.feature_affines(window_affine)
        pyramid = self.local_encoder(image, state)
        shape_12mm, shape_6mm = tuple(pyramid[2].shape[2:]), tuple(pyramid[1].shape[2:])
        role_maps = [level.new_full((batch, ROLE_COUNT, *level.shape[2:]), UNIFORM_ROLE_LOG_PROBABILITY)
                     for level in (pyramid[0], pyramid[1], pyramid[2])]
        whole_volume_local, whole_volume_coverage = sample_to_grid(
            context.whole_volume_features, context.whole_volume_feature_affine, shape_12mm, frames[2],
            support_affine_xyz=context.whole_volume_affine, support_shape_zyx=context.whole_volume_shape,
            support_valid_zyx=context.whole_volume_valid)
        features, coarse_logits = self.role_decoder(pyramid, whole_volume_local, role_maps)
        placeholders = {LEVELS[0]: coarse_logits.new_zeros(batch, MAX_SOURCES, ROLE_COUNT, *shape_12mm),
                        LEVELS[1]: coarse_logits.new_zeros(batch, MAX_SOURCES, ROLE_COUNT, *shape_6mm)}
        output = self._window_output(features, coarse_logits, frames, batch, context, placeholders)
        if diagnostics:
            output["diagnostics"] = {"source_valid": context.source_valid, "whole_volume_valid": whole_volume_coverage}
        return output

    @staticmethod
    def _window_output(features, coarse_logits, frames, batch, context, source_role_logits):
        def batched(frame):
            return frame if frame.ndim == 3 else frame[None].expand(batch, -1, -1)
        return {"features": features, "coarse_role_logits": coarse_logits, "feature_affine": frames[0],
                "source_role_logits": source_role_logits,
                "source_role_affines": {LEVELS[0]: batched(frames[2]), LEVELS[1]: batched(frames[1])},
                "source_vectors": context.source_vectors, "source_valid": context.source_valid, "diagnostics": None}

    def forward(self, image, state, window_affine_xyz, context):
        return self.forward_window(image, state, window_affine_xyz, context)

    # ------------------------------------------------------------------ native refinement
    def encode_neighbourhood(self, neighbourhood: torch.Tensor) -> torch.Tensor:
        """The 8-wide code of each voxel's 108 neighbourhood values (computed once per voxel at inference)."""
        return self.native_refinement.encode_neighbourhood(neighbourhood)

    def refine_native(self, features: torch.Tensor, coarse_logits: torch.Tensor, voxel_values: torch.Tensor, *,
                      neighbourhood: torch.Tensor | None = None, neighbourhood_code: torch.Tensor | None = None,
                      source_vectors: torch.Tensor, source_role_logits: torch.Tensor,
                      source_valid: torch.Tensor) -> torch.Tensor:
        """Native role logits [B,N,3] of N native voxels of one window.

        ``features`` [B,N,32] and ``coarse_logits`` [B,N,3] are the window outputs sampled at the voxels,
        ``voxel_values`` [B,N,20]; give either the 108 ``neighbourhood`` values or their ``neighbourhood_code``;
        ``source_role_logits`` [B,N,16,3] are the 6 mm source role logits sampled at the voxels. Without native
        refinement the coarse role logits are returned.
        """
        if (neighbourhood is None) == (neighbourhood_code is None):
            raise ValueError("give exactly one of neighbourhood and neighbourhood_code")
        if not self.use_native_refinement:
            return coarse_logits.float()
        if neighbourhood_code is None:
            neighbourhood_code = self.native_refinement.encode_neighbourhood(neighbourhood)
        return self.native_refinement(features, coarse_logits, voxel_values, neighbourhood_code,
                                      source_vectors=source_vectors, source_role_logits=source_role_logits,
                                      source_valid=source_valid)

    def native_role_logits(self, output: dict[str, Any], world_xyz: torch.Tensor, voxel_values: torch.Tensor, *,
                           window_affine_xyz, window_valid: torch.Tensor | None = None,
                           neighbourhood: torch.Tensor | None = None,
                           neighbourhood_code: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(native role logits [B,N,3] FP32, validity [B,N]) of native voxels from a :meth:`forward_window` output.

        ``world_xyz`` [N,3] or [B,N,3] FP64 voxel centres; ``voxel_values`` [B,N,20]; the window extent
        (``window_affine_xyz``, its ``window_valid`` cells) decides which voxels are valid.
        """
        support = dict(support_affine_xyz=window_affine_xyz, support_shape_zyx=(self.window_size,) * 3,
                       support_valid_zyx=window_valid)
        features, valid_features = sample_world_features(output["features"], output["feature_affine"], world_xyz,
                                                         **support)
        coarse, valid_coarse = sample_world_features(output["coarse_role_logits"], output["feature_affine"], world_xyz,
                                                     **support)
        source_logits, valid_sources = sample_source_role_logits(
            output["source_role_logits"][LEVELS[1]], output["source_role_affines"][LEVELS[1]], world_xyz,
            output["source_valid"], **support)
        logits = self.refine_native(features, coarse, voxel_values.float(),
                                    neighbourhood=None if neighbourhood is None else neighbourhood.float(),
                                    neighbourhood_code=neighbourhood_code, source_vectors=output["source_vectors"],
                                    source_role_logits=source_logits, source_valid=output["source_valid"])
        return logits.float(), valid_features & valid_coarse & valid_sources
