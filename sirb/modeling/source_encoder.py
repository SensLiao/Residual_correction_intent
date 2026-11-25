"""Sources of a scribble and the source encoder.

A scribble is divided into at most 16 sources: the representatives come from
:func:`sirb.data.geometry.scribble_representatives`, and every scribble voxel joins its physically closest
representative (:func:`group_scribble`). A source is a feature-pooling group of scribble voxels; it does not
correspond to one error or one lesion.

The source encoder summarises each source by a 128-wide source vector (:class:`SourceEncoder`). Its 340 inputs
(:func:`pool_source_inputs`) are, per source: the mean and the maximum of the whole-volume features over the source
voxels inside the whole-volume grid (64 + 64), the mean and the maximum of the level-2 local features of the
scribble window over the source voxels inside that window (96 + 96, zero when none is inside), the representative's
position scaled to [-1, 1] by the examination box (3), the correction-type embedding (16) and the fraction of the
source's voxels inside the scribble window (1). Invalid source slots are all zero, and so is their source vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..data.geometry import (MAX_SOURCES, TIE_ATOL_MM2, TIE_RTOL, GeometryError, SourceRepresentatives,
                             index_xyz_to_world, sample_world_features)

__all__ = ["CORRECTION_TYPE_WIDTH", "SOURCE_INPUT_WIDTH", "SOURCE_WIDTH", "PooledSources", "ScribbleSources",
           "SourceEncoder", "group_scribble", "nearest_representative", "pool_source_inputs"]

SOURCE_WIDTH = 128
CORRECTION_TYPE_WIDTH = 16
WHOLE_VOLUME_FEATURE_WIDTH = 64
WINDOW_FEATURE_WIDTH = 96
#: [whole-volume mean 64, max 64, scribble-window mean 96, max 96, position 3, correction type 16, window fraction 1]
SOURCE_INPUT_WIDTH = 2 * WHOLE_VOLUME_FEATURE_WIDTH + 2 * WINDOW_FEATURE_WIDTH + 3 + CORRECTION_TYPE_WIDTH + 1
#: Scribble voxels per distance block when grouping (bounds the N x 16 float64 temporary).
GROUPING_BLOCK = 1 << 16


@dataclass(frozen=True)
class ScribbleSources:
    """The sources of one scribble (CPU arrays). Slot j of ``source_valid`` / ``representative_xyz`` is source j."""
    voxel_world_xyz: np.ndarray      # [N, 3] float64 world centres of the scribble voxels
    voxel_source: np.ndarray         # [N] int64 source slot of each scribble voxel
    representative_xyz: np.ndarray   # [16, 3] float64 (zeros on invalid slots)
    source_valid: np.ndarray         # [16] bool

    def __post_init__(self):
        world = np.asarray(self.voxel_world_xyz, dtype=np.float64)
        source = np.asarray(self.voxel_source, dtype=np.int64)
        valid = np.asarray(self.source_valid, dtype=bool)
        representative = np.asarray(self.representative_xyz, dtype=np.float64)
        if world.ndim != 2 or world.shape[1] != 3 or source.shape != (len(world),):
            raise GeometryError(f"source voxels actual={world.shape}/{source.shape}, expected=(N,3)/(N,)")
        if valid.shape != (MAX_SOURCES,) or representative.shape != (MAX_SOURCES, 3):
            raise GeometryError(f"source slots actual={valid.shape}/{representative.shape}, "
                                f"expected={MAX_SOURCES} slots")
        if len(source) and (source.min() < 0 or source.max() >= MAX_SOURCES or not valid[source].all()):
            raise GeometryError("a scribble voxel was assigned to an invalid source slot")
        counts = np.bincount(source, minlength=MAX_SOURCES)
        if np.any(counts[valid] == 0):
            raise GeometryError(f"every valid source must own scribble voxels; counts={counts.tolist()}")
        object.__setattr__(self, "voxel_world_xyz", world)
        object.__setattr__(self, "voxel_source", source)
        object.__setattr__(self, "source_valid", valid)
        object.__setattr__(self, "representative_xyz", representative)

    @property
    def source_count(self) -> int:
        return int(self.source_valid.sum())

    def counts(self) -> np.ndarray:
        return np.bincount(self.voxel_source, minlength=MAX_SOURCES)


def nearest_representative(world_xyz, representatives_xyz) -> np.ndarray:
    """Index of the physically closest representative; equal distances go to the smallest index.

    Squared distances are compared with the tolerance of :func:`sirb.data.geometry.scribble_representatives`, so a
    mirror-symmetric tie is not broken by floating-point rounding.
    """
    world = np.asarray(world_xyz, dtype=np.float64).reshape(-1, 3)
    reps = np.asarray(representatives_xyz, dtype=np.float64).reshape(-1, 3)
    if not len(reps):
        raise GeometryError("grouping needs at least one representative")
    result = np.empty(len(world), dtype=np.int64)
    for start in range(0, len(world), GROUPING_BLOCK):
        block = world[start:start + GROUPING_BLOCK]
        squared = ((block[:, None, :] - reps[None, :, :]) ** 2).sum(axis=-1)
        minimum = squared.min(axis=1, keepdims=True)
        tied = np.isclose(squared, minimum, atol=TIE_ATOL_MM2, rtol=TIE_RTOL)
        result[start:start + GROUPING_BLOCK] = np.argmax(tied, axis=1)   # first True = smallest index
    return result


def group_scribble(scribble_zyx, native_affine_xyz, representatives: SourceRepresentatives, *,
                   native_valid_zyx=None) -> ScribbleSources:
    """Assign every voxel of a non-empty scribble to its closest representative (ties: smallest index).

    Grouping uses world coordinates only, so a left-right flip of the arrays (whose affine flips too) keeps every
    source; nothing is regrouped after a flip.
    """
    scribble = np.asarray(scribble_zyx, dtype=bool)
    valid_slots = np.asarray(representatives.source_valid, dtype=bool)
    if valid_slots.shape != (MAX_SOURCES,):
        raise GeometryError(f"source slots actual={valid_slots.shape}, expected=({MAX_SOURCES},)")
    count = int(valid_slots.sum())
    if not count or not np.array_equal(valid_slots, np.arange(MAX_SOURCES) < count):
        raise GeometryError("representatives must occupy the first slots contiguously and be non-empty")
    representative_world = np.asarray(representatives.world_xyz_mm, dtype=np.float64)
    candidates = np.argwhere(scribble)
    if not len(candidates):
        raise GeometryError("grouping requires a non-empty scribble")
    if native_valid_zyx is not None and not np.asarray(native_valid_zyx, dtype=bool)[tuple(candidates.T)].all():
        raise GeometryError("the scribble includes invalid native voxels")
    world = index_xyz_to_world(candidates[:, ::-1], native_affine_xyz)
    source = nearest_representative(world, representative_world[:count])
    representative = np.where(valid_slots[:, None], representative_world, 0.0)
    return ScribbleSources(world, source, representative, valid_slots.copy())


class SourceEncoder(nn.Module):
    """Linear 340 -> 128, GELU, Linear 128 -> 128, LayerNorm (FP32), and the 2 x 16 correction-type embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.correction_type_embedding = nn.Embedding(2, CORRECTION_TYPE_WIDTH)
        self.layers = nn.Sequential(nn.Linear(SOURCE_INPUT_WIDTH, SOURCE_WIDTH), nn.GELU(),
                                    nn.Linear(SOURCE_WIDTH, SOURCE_WIDTH), nn.LayerNorm(SOURCE_WIDTH))

    def embed_correction_type(self, sign: torch.Tensor) -> torch.Tensor:
        """Embedding of a [B] tensor of correction signs (+1 addition, -1 removal)."""
        if sign.ndim != 1 or not bool(((sign == 1) | (sign == -1)).all()):
            raise ValueError(f"correction signs must be a [B] tensor of +1 (addition) / -1 (removal); got {sign}")
        return self.correction_type_embedding((sign == -1).long())

    def forward(self, source_inputs: torch.Tensor, source_valid: torch.Tensor) -> torch.Tensor:
        if source_inputs.shape[-1] != SOURCE_INPUT_WIDTH or source_inputs.shape[:2] != source_valid.shape:
            raise ValueError(f"source inputs actual={tuple(source_inputs.shape)}, valid={tuple(source_valid.shape)}")
        safe = torch.where(source_valid[..., None], source_inputs, torch.zeros_like(source_inputs))
        with torch.autocast(device_type=safe.device.type, enabled=False):
            encoded = self.layers(safe.float())
        return torch.where(source_valid[..., None], encoded, torch.zeros_like(encoded))


@dataclass
class PooledSources:
    source_inputs: torch.Tensor      # [B, 16, 340]
    source_valid: torch.Tensor       # [B, 16] valid slot with at least one voxel inside the whole-volume grid
    window_fraction: torch.Tensor    # [B, 16] fraction of the source voxels inside the scribble window
    source_extent: torch.Tensor      # [B, 16, 3] physical extent of the source voxels / examination extent
    representative_xyz: torch.Tensor  # [B, 16, 3] float64


def _segment_mean_max(values: torch.Tensor, inside: torch.Tensor, group: torch.Tensor):
    """Per-source mean and maximum of ``values`` [N, C] over rows with ``inside``; zero for a source with none."""
    channels = values.shape[1]
    weight = inside.to(values.dtype)
    sums = values.new_zeros((MAX_SOURCES, channels)).index_add(0, group, values * weight[:, None])
    counts = values.new_zeros(MAX_SOURCES).index_add(0, group, weight)
    mean = sums / counts.clamp(min=1)[:, None]
    masked = torch.where(inside[:, None], values, torch.full_like(values, -torch.inf))
    maximum = values.new_full((MAX_SOURCES, channels), -torch.inf).scatter_reduce(
        0, group[:, None].expand(-1, channels), masked, "amax", include_self=True)
    present = counts > 0
    maximum = torch.where(present[:, None], maximum, torch.zeros_like(maximum))
    mean = torch.where(present[:, None], mean, torch.zeros_like(mean))
    return mean, maximum, counts


def pool_source_inputs(sources, *, whole_volume_features, whole_volume_feature_affine_xyz, whole_volume_affine_xyz,
                       whole_volume_shape_zyx, whole_volume_valid_zyx, window_features, window_feature_affine_xyz,
                       window_affine_xyz, window_shape_zyx, window_valid_zyx, examination_lower_xyz,
                       examination_extent_xyz, correction_embedding) -> PooledSources:
    """The 340 source inputs of every batch item; ``sources`` holds one :class:`ScribbleSources` per item.

    Pooling is vectorised (index_add / scatter_reduce), so nothing is read back from the device per source.
    """
    batch = whole_volume_features.shape[0]
    if len(sources) != batch:
        raise ValueError(f"scribble sources actual={len(sources)}, expected one per batch item ({batch})")
    device = whole_volume_features.device
    inputs, effective, fractions, extents, representatives = [], [], [], [], []
    for b, item in enumerate(sources):
        points = torch.as_tensor(item.voxel_world_xyz, dtype=torch.float64, device=device)
        index = torch.as_tensor(item.voxel_source, dtype=torch.long, device=device)
        slot_valid = torch.as_tensor(item.source_valid, dtype=torch.bool, device=device)
        whole_valid = None if whole_volume_valid_zyx is None else whole_volume_valid_zyx[b:b + 1]
        window_valid = None if window_valid_zyx is None else window_valid_zyx[b:b + 1]
        whole_frame = whole_volume_feature_affine_xyz if whole_volume_feature_affine_xyz.ndim == 2 \
            else whole_volume_feature_affine_xyz[b]
        window_frame = window_feature_affine_xyz if window_feature_affine_xyz.ndim == 2 \
            else window_feature_affine_xyz[b]
        whole_support = whole_volume_affine_xyz if whole_volume_affine_xyz.ndim == 2 else whole_volume_affine_xyz[b]
        window_support = window_affine_xyz if window_affine_xyz.ndim == 2 else window_affine_xyz[b]
        window_inside_all = sample_world_features(
            window_features[b:b + 1, :1], window_frame, points, support_affine_xyz=window_support,
            support_shape_zyx=window_shape_zyx, support_valid_zyx=window_valid)[1][0]
        whole_samples, whole_inside = sample_world_features(
            whole_volume_features[b:b + 1], whole_frame, points, support_affine_xyz=whole_support,
            support_shape_zyx=whole_volume_shape_zyx, support_valid_zyx=whole_valid)
        window_samples, window_inside = sample_world_features(
            window_features[b:b + 1], window_frame, points, support_affine_xyz=window_support,
            support_shape_zyx=window_shape_zyx, support_valid_zyx=window_valid)
        with torch.autocast(device_type=device.type, enabled=False):
            whole_mean, whole_max, whole_count = _segment_mean_max(whole_samples[0].float(), whole_inside[0], index)
            window_mean, window_max, _ = _segment_mean_max(window_samples[0].float(), window_inside[0], index)
        window_count = torch.zeros(MAX_SOURCES, dtype=torch.float32, device=device).index_add(
            0, index, window_inside_all.to(torch.float32))
        members = torch.zeros(MAX_SOURCES, dtype=torch.float32, device=device).index_add(
            0, index, torch.ones(len(index), dtype=torch.float32, device=device))
        fraction = torch.where(members > 0, window_count / members.clamp(min=1), torch.zeros_like(members))
        # Physical extent of each source's voxel centres per world axis, divided by the examination box.
        low = torch.full((MAX_SOURCES, 3), torch.inf, dtype=torch.float64, device=device).scatter_reduce(
            0, index[:, None].expand(-1, 3), points, "amin", include_self=True)
        high = torch.full((MAX_SOURCES, 3), -torch.inf, dtype=torch.float64, device=device).scatter_reduce(
            0, index[:, None].expand(-1, 3), points, "amax", include_self=True)
        extent = torch.where((members > 0)[:, None], high - low, torch.zeros_like(high))
        extent = (extent / examination_extent_xyz[b][None, :]).float()
        representative = torch.as_tensor(item.representative_xyz, dtype=torch.float64, device=device)
        representative = torch.where(slot_valid[:, None], representative, torch.zeros_like(representative))
        position = (2 * (representative - examination_lower_xyz[b][None, :]) / examination_extent_xyz[b][None, :]
                    - 1).float()
        correction = correction_embedding[b][None, :].expand(MAX_SOURCES, -1).float()
        valid_b = slot_valid & (whole_count > 0)
        row = torch.cat((whole_mean, whole_max, window_mean, window_max, position, correction, fraction[:, None]),
                        dim=-1)
        if row.shape[-1] != SOURCE_INPUT_WIDTH or whole_mean.shape[-1] != WHOLE_VOLUME_FEATURE_WIDTH \
                or window_mean.shape[-1] != WINDOW_FEATURE_WIDTH:
            raise ValueError(f"source input width actual={row.shape[-1]}, expected={SOURCE_INPUT_WIDTH}")
        row = torch.where(valid_b[:, None], row, torch.zeros_like(row))
        inputs.append(row)
        effective.append(valid_b)
        fractions.append(torch.where(valid_b, fraction, torch.zeros_like(fraction)))
        extents.append(torch.where(valid_b[:, None], extent, torch.zeros_like(extent)))
        representatives.append(representative)
    return PooledSources(torch.stack(inputs), torch.stack(effective), torch.stack(fractions), torch.stack(extents),
                         torch.stack(representatives))
