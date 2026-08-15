#!/usr/bin/env python3
"""Deterministic 3D connected-component enumeration over the current mask.

Inference-visible only: every descriptor is computed from the current mask
``M_k``, the signed cue, and spacing.  No function in this module accepts a
GT parameter; the candidate builder must never be able to see ground truth.

Connectivity: 3D 18-connectivity (face + edge neighbours, no corner-only
neighbours), matching the MS-LCP blueprint and the SCEP redesign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

ENUMERATION_VERSION = "PETCT-COMPONENT-ENUMERATION-v1.0"

# 26-neighbour structure minus the eight corner-only positions.
_STRUCTURE_18 = np.ones((3, 3, 3), dtype=bool)
for _x, _y, _z in (
    (0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2),
    (2, 0, 0), (2, 0, 2), (2, 2, 0), (2, 2, 2),
):
    _STRUCTURE_18[_x, _y, _z] = False


@dataclass(frozen=True)
class ComponentDescriptor:
    """Inference-visible descriptor of one current-mask component."""

    index: int
    volume_voxels: int
    log_volume: float
    bbox_min: tuple
    bbox_max: tuple
    z_span: int
    centroid_voxel: tuple
    centroid_mm: tuple
    prompted_slice_mask: Optional[np.ndarray] = field(default=None, repr=False)
    prompted_slice_overlap: int = 0
    cue_overlap_voxels: int = 0
    distance_from_cue_mm: float = float("inf")

    def as_dict(self, *, include_mask: bool = False) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "index": self.index,
            "volume_voxels": self.volume_voxels,
            "log_volume": self.log_volume,
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
            "z_span": self.z_span,
            "centroid_voxel": list(self.centroid_voxel),
            "centroid_mm": list(self.centroid_mm),
            "prompted_slice_overlap": self.prompted_slice_overlap,
            "cue_overlap_voxels": self.cue_overlap_voxels,
            "distance_from_cue_mm": self.distance_from_cue_mm,
        }
        if include_mask and self.prompted_slice_mask is not None:
            record["prompted_slice_mask"] = self.prompted_slice_mask
        return record


@dataclass(frozen=True)
class ComponentEnumeration:
    """One full enumeration result, keyed to the mask it came from."""

    episode_id: str
    m_sha256: str
    enumeration_version: str = ENUMERATION_VERSION
    components: tuple = ()

    def key(self) -> str:
        return f"{self.episode_id}|{self.m_sha256[:16]}|{self.enumeration_version}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "m_sha256": self.m_sha256,
            "enumeration_version": self.enumeration_version,
            "component_count": len(self.components),
            "components": [component.as_dict() for component in self.components],
        }


def enumerate_components(
    full_mask: np.ndarray,
    *,
    episode_id: str,
    m_sha256: str,
    spacing_xyz: np.ndarray,
    cue_voxels: Optional[np.ndarray] = None,
    prompted_z: Optional[int] = None,
) -> ComponentEnumeration:
    """Enumerate 18-connected components of the full 3D current mask.

    Args:
        full_mask: binary 3D mask of the full volume, shape [D,H,W].
        episode_id: stable episode identifier used in the component key.
        m_sha256: sha256 of the mask provenance record.
        spacing_xyz: per-axis physical spacing in mm, shape (3,), positive.
        cue_voxels: optional Nx3 integer voxel coordinates of the signed cue
            (any polarity); used for overlap/distance descriptors only.
        prompted_z: optional z index of the prompted axial slice used for the
            central projection descriptor.

    Returns:
        A frozen ComponentEnumeration.  No GT is read or returned.
    """

    if full_mask.ndim != 3:
        raise ValueError("full_mask must be a 3D array")
    if not np.all(np.isin(full_mask, [0, 1])):
        raise ValueError("full_mask must be binary")
    if spacing_xyz.shape != (3,) or not np.all(np.isfinite(spacing_xyz)) or np.any(spacing_xyz <= 0):
        raise ValueError("spacing_xyz must contain three positive finite values")
    if prompted_z is not None and not 0 <= int(prompted_z) < full_mask.shape[0]:
        raise ValueError("prompted_z is outside the volume")
    if cue_voxels is not None:
        cue_voxels = np.asarray(cue_voxels, dtype=np.int64)
        if cue_voxels.ndim != 2 or cue_voxels.shape[1] != 3 or len(cue_voxels) == 0:
            raise ValueError("cue_voxels must be a non-empty Nx3 integer array")

    try:
        from scipy import ndimage
    except ImportError:
        raise ImportError(
            "scipy is required for 18-connectivity component labelling"
        ) from None

    labels, count = ndimage.label(full_mask.astype(bool), structure=_STRUCTURE_18)
    descriptors: List[ComponentDescriptor] = []
    zz, yy, xx = np.nonzero(labels)
    if count == 0:
        return ComponentEnumeration(
            episode_id=episode_id,
            m_sha256=m_sha256,
            components=tuple(descriptors),
        )
    coord_by_label: Dict[int, List[np.ndarray]] = {}
    for axis_values, axis_name in ((zz, "z"), (yy, "y"), (xx, "x")):
        del axis_name
    for label_id in range(1, count + 1):
        mask = labels == label_id
        voxels = np.argwhere(mask)  # [N,3] z,y,x
        coord_by_label[label_id] = voxels
    for label_id in range(1, count + 1):
        voxels = coord_by_label[label_id]
        volume = len(voxels)
        bbox_min = tuple(int(value) for value in voxels.min(axis=0))
        bbox_max = tuple(int(value) for value in voxels.max(axis=0))
        centroid_voxel = tuple(float(value) for value in voxels.mean(axis=0))
        centroid_mm = tuple(
            centroid_voxel[axis] * float(spacing_xyz[axis]) for axis in range(3)
        )
        prompted_mask = None
        prompted_overlap = 0
        if prompted_z is not None:
            prompted_mask = mask[int(prompted_z)].astype(np.uint8)
            prompted_overlap = int(prompted_mask.sum())
        cue_overlap = 0
        distance_mm = float("inf")
        if cue_voxels is not None:
            within = (cue_voxels[:, 0] >= bbox_min[0]) & (
                cue_voxels[:, 0] <= bbox_max[0]
            ) & (cue_voxels[:, 1] >= bbox_min[1]) & (
                cue_voxels[:, 1] <= bbox_max[1]
            ) & (cue_voxels[:, 2] >= bbox_min[2]) & (
                cue_voxels[:, 2] <= bbox_max[2]
            )
            cue_overlap = int(mask[tuple(cue_voxels[within].T)].sum())
            if within.any():
                nearest = voxels[
                    np.argmin(
                        np.abs(voxels[:, None, :] - cue_voxels[within][None, :, :]).sum(
                            axis=2
                        ),
                        axis=0,
                    )
                ]
                deltas_mm = np.abs(nearest - cue_voxels[within].astype(float)) * spacing_xyz
                distance_mm = float(np.min(np.linalg.norm(deltas_mm, axis=1)))
        descriptors.append(
            ComponentDescriptor(
                index=label_id,
                volume_voxels=volume,
                log_volume=float(np.log1p(volume)),
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                z_span=bbox_max[0] - bbox_min[0] + 1,
                centroid_voxel=centroid_voxel,
                centroid_mm=centroid_mm,
                prompted_slice_mask=prompted_mask,
                prompted_slice_overlap=prompted_overlap,
                cue_overlap_voxels=cue_overlap,
                distance_from_cue_mm=distance_mm,
            )
        )
    return ComponentEnumeration(
        episode_id=episode_id,
        m_sha256=m_sha256,
        components=tuple(descriptors),
    )


def cue_hit_component(
    enumeration: ComponentEnumeration,
    cue_voxels: np.ndarray,
    full_mask: np.ndarray,
) -> Optional[int]:
    """Deterministic cue-hit binding used for REMOVE.

    The component containing the largest number of cue voxels wins; ties are
    broken by the lowest component index (deterministic).  Returns the
    component index or None when the cue misses every component.
    """

    cue_voxels = np.asarray(cue_voxels, dtype=np.int64)
    if cue_voxels.ndim != 2 or cue_voxels.shape[1] != 3:
        raise ValueError("cue_voxels must be an Nx3 integer array")
    overlap: Dict[int, int] = {}
    for component in enumeration.components:
        bbox_min = component.bbox_min
        bbox_max = component.bbox_max
        within = (
            (cue_voxels[:, 0] >= bbox_min[0]) & (cue_voxels[:, 0] <= bbox_max[0])
            & (cue_voxels[:, 1] >= bbox_min[1]) & (cue_voxels[:, 1] <= bbox_max[1])
            & (cue_voxels[:, 2] >= bbox_min[2]) & (cue_voxels[:, 2] <= bbox_max[2])
        )
        if within.any():
            overlap[component.index] = int(full_mask[tuple(cue_voxels[within].T)].sum())
    if not overlap:
        return None
    return min(overlap, key=lambda index: (-overlap[index], index))
