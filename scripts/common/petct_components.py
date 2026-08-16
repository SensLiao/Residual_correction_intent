#!/usr/bin/env python3
"""Deterministic, inference-visible component enumeration for PET/CT v3.

All volume arrays and voxel coordinates in this module use the native NIfTI
``(x, y, z)`` axis order. Axial slices therefore index axis 2. The module
accepts only the current mask, signed-cue coordinates, and physical spacing;
there is deliberately no GT or authorized-residual argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ENUMERATION_VERSION = "PETCT-COMPONENT-ENUMERATION-v1.1"
COMPONENT_DESCRIPTOR_FIELDS: Tuple[str, ...] = (
    "log_volume",
    "z_span",
    "prompted_slice_overlap",
    "centroid_dx_mm",
    "centroid_dy_mm",
    "cue_overlap_voxels",
    "distance_from_cue_mm",
)

# Face and edge neighbours are connected; the eight corner-only neighbours
# are excluded. Array axes here are x, y, z, but the structure is symmetric.
_STRUCTURE_18 = np.ones((3, 3, 3), dtype=bool)
for _x, _y, _z in (
    (0, 0, 0),
    (0, 0, 2),
    (0, 2, 0),
    (0, 2, 2),
    (2, 0, 0),
    (2, 0, 2),
    (2, 2, 0),
    (2, 2, 2),
):
    _STRUCTURE_18[_x, _y, _z] = False


def component_key_for(
    episode_id: str,
    m_sha256: str,
    position: int,
    *,
    enumeration_version: str = ENUMERATION_VERSION,
) -> str:
    """Return the stable identity of one position in one frozen enumeration."""

    if not episode_id or not m_sha256:
        raise ValueError("episode_id and m_sha256 must be non-empty")
    if int(position) < 0:
        raise ValueError("component position must be 0-based and non-negative")
    return "%s|%s|%s|component_%04d" % (
        episode_id,
        m_sha256,
        enumeration_version,
        int(position),
    )


def label_components_18(full_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label a binary ``[X,Y,Z]`` mask with deterministic 18-connectivity."""

    mask = np.asarray(full_mask)
    if mask.ndim != 3:
        raise ValueError("full_mask must be a 3D [X,Y,Z] array")
    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError("full_mask must be binary")
    try:
        from scipy import ndimage
    except ImportError:
        raise ImportError(
            "scipy is required for 18-connectivity component labelling"
        ) from None
    labels, count = ndimage.label(mask.astype(bool), structure=_STRUCTURE_18)
    return labels.astype(np.int32, copy=False), int(count)


def _validated_cue(cue_voxels: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    cue = np.asarray(cue_voxels)
    if cue.ndim != 2 or cue.shape[1] != 3 or len(cue) == 0:
        raise ValueError("cue_voxels must be a non-empty Nx3 array")
    if not np.all(np.isfinite(cue)) or not np.all(cue == np.floor(cue)):
        raise ValueError("cue_voxels must contain finite integer coordinates")
    cue = np.unique(cue.astype(np.int64), axis=0)
    bounds = np.asarray(tuple(int(value) for value in shape), dtype=np.int64)
    if np.any(cue < 0) or np.any(cue >= bounds[None]):
        raise ValueError("cue_voxels contains a coordinate outside the volume")
    return cue


@dataclass(frozen=True)
class ComponentDescriptor:
    """Inference-visible descriptor of one current-mask component."""

    position: int
    component_key: str
    volume_voxels: int
    log_volume: float
    bbox_min_xyz: tuple
    bbox_max_xyz: tuple
    z_span: int
    centroid_voxel_xyz: tuple
    centroid_mm_xyz: tuple
    centroid_dx_mm: Optional[float]
    centroid_dy_mm: Optional[float]
    prompted_slice_mask: Optional[np.ndarray] = field(default=None, repr=False)
    prompted_slice_overlap: int = 0
    cue_overlap_voxels: int = 0
    distance_from_cue_mm: Optional[float] = None

    @property
    def index(self) -> int:
        """Legacy scipy-label view; persisted v3 artifacts use ``position``."""

        return self.position + 1

    @property
    def bbox_min(self) -> tuple:
        """Compatibility alias whose values remain in xyz order."""

        return self.bbox_min_xyz

    @property
    def bbox_max(self) -> tuple:
        """Compatibility alias whose values remain in xyz order."""

        return self.bbox_max_xyz

    @property
    def centroid_voxel(self) -> tuple:
        return self.centroid_voxel_xyz

    @property
    def centroid_mm(self) -> tuple:
        return self.centroid_mm_xyz

    def descriptor_vector(self) -> List[float]:
        values = [getattr(self, name) for name in COMPONENT_DESCRIPTOR_FIELDS]
        if any(value is None or not np.isfinite(float(value)) for value in values):
            raise ValueError(
                "component descriptor vector requires cue-conditioned finite values"
            )
        return [float(value) for value in values]

    def as_dict(self, *, include_mask: bool = False) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "position": int(self.position),
            "component_key": self.component_key,
            "volume_voxels": int(self.volume_voxels),
            "log_volume": float(self.log_volume),
            "bbox_min_xyz": [int(value) for value in self.bbox_min_xyz],
            "bbox_max_xyz": [int(value) for value in self.bbox_max_xyz],
            "z_span": int(self.z_span),
            "centroid_voxel_xyz": [
                float(value) for value in self.centroid_voxel_xyz
            ],
            "centroid_mm_xyz": [float(value) for value in self.centroid_mm_xyz],
            "centroid_dx_mm": (
                None if self.centroid_dx_mm is None else float(self.centroid_dx_mm)
            ),
            "centroid_dy_mm": (
                None if self.centroid_dy_mm is None else float(self.centroid_dy_mm)
            ),
            "prompted_slice_overlap": int(self.prompted_slice_overlap),
            "cue_overlap_voxels": int(self.cue_overlap_voxels),
            "distance_from_cue_mm": (
                None
                if self.distance_from_cue_mm is None
                else float(self.distance_from_cue_mm)
            ),
        }
        if include_mask:
            if self.prompted_slice_mask is None:
                raise ValueError("prompted slice mask was not materialized")
            record["prompted_slice_mask"] = self.prompted_slice_mask.tolist()
        return record


@dataclass(frozen=True)
class ComponentEnumeration:
    """One full enumeration result, cryptographically bound to its mask."""

    episode_id: str
    m_sha256: str
    enumeration_version: str = ENUMERATION_VERSION
    components: tuple = ()

    def key(self) -> str:
        return "%s|%s|%s" % (
            self.episode_id,
            self.m_sha256,
            self.enumeration_version,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "m_sha256": self.m_sha256,
            "enumeration_version": self.enumeration_version,
            "axis_order": "xyz",
            "axial_axis": 2,
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
    """Enumerate components of a full NIfTI-order ``[X,Y,Z]`` current mask.

    With a cue, every component receives a finite nearest-cue distance in
    physical millimetres. The distance is the minimum Euclidean norm over
    all component/cue voxel pairs after per-axis spacing is applied; it is
    not limited by the component bounding box and is not an L1 proxy.
    """

    mask = np.asarray(full_mask)
    spacing = np.asarray(spacing_xyz, dtype=np.float64)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)):
        raise ValueError("spacing_xyz must contain three finite values")
    if np.any(spacing <= 0):
        raise ValueError("spacing_xyz values must be positive")
    labels, count = label_components_18(mask)
    if prompted_z is not None and not 0 <= int(prompted_z) < mask.shape[2]:
        raise ValueError("prompted_z is outside axial axis 2")
    cue = None
    cue_centroid_mm = None
    cue_tree = None
    if cue_voxels is not None:
        cue = _validated_cue(cue_voxels, mask.shape)
        cue_mm = cue.astype(np.float64) * spacing[None]
        cue_centroid_mm = cue_mm.mean(axis=0)
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("scipy is required for physical cue distances") from None
        cue_tree = cKDTree(cue_mm)

    descriptors: List[ComponentDescriptor] = []
    for label_id in range(1, count + 1):
        component_mask = labels == label_id
        voxels = np.argwhere(component_mask)  # [N,3] x,y,z
        volume = int(len(voxels))
        bbox_min = tuple(int(value) for value in voxels.min(axis=0))
        bbox_max = tuple(int(value) for value in voxels.max(axis=0))
        centroid_voxel = tuple(float(value) for value in voxels.mean(axis=0))
        centroid_mm = tuple(
            centroid_voxel[axis] * float(spacing[axis]) for axis in range(3)
        )
        prompted_mask = None
        prompted_overlap = 0
        if prompted_z is not None:
            prompted_mask = component_mask[:, :, int(prompted_z)].astype(np.uint8)
            prompted_overlap = int(prompted_mask.sum())

        cue_overlap = 0
        distance_mm = None
        centroid_dx_mm = None
        centroid_dy_mm = None
        if cue is not None and cue_tree is not None and cue_centroid_mm is not None:
            cue_labels = labels[tuple(cue.T)]
            cue_overlap = int(np.count_nonzero(cue_labels == label_id))
            distances, _ = cue_tree.query(
                voxels.astype(np.float64) * spacing[None], k=1
            )
            distance_mm = float(np.min(np.asarray(distances, dtype=np.float64)))
            centroid_dx_mm = float(centroid_mm[0] - cue_centroid_mm[0])
            centroid_dy_mm = float(centroid_mm[1] - cue_centroid_mm[1])
            if not all(
                np.isfinite(value)
                for value in (distance_mm, centroid_dx_mm, centroid_dy_mm)
            ):
                raise RuntimeError("non-finite cue-conditioned component descriptor")

        position = label_id - 1
        descriptors.append(
            ComponentDescriptor(
                position=position,
                component_key=component_key_for(episode_id, m_sha256, position),
                volume_voxels=volume,
                log_volume=float(np.log1p(volume)),
                bbox_min_xyz=bbox_min,
                bbox_max_xyz=bbox_max,
                z_span=bbox_max[2] - bbox_min[2] + 1,
                centroid_voxel_xyz=centroid_voxel,
                centroid_mm_xyz=centroid_mm,
                centroid_dx_mm=centroid_dx_mm,
                centroid_dy_mm=centroid_dy_mm,
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


def cue_hit_component_position(
    enumeration: ComponentEnumeration,
    cue_voxels: np.ndarray,
    full_mask: np.ndarray,
) -> Optional[int]:
    """Bind a REMOVE cue by exact label membership and return a 0-based position."""

    labels, count = label_components_18(full_mask)
    if count != len(enumeration.components):
        raise ValueError("enumeration does not match the supplied full_mask")
    cue = _validated_cue(cue_voxels, labels.shape)
    hit_labels = labels[tuple(cue.T)]
    hit_labels = hit_labels[hit_labels > 0]
    if len(hit_labels) == 0:
        return None
    label_ids, hit_counts = np.unique(hit_labels, return_counts=True)
    best_label = min(
        zip(label_ids.tolist(), hit_counts.tolist()),
        key=lambda pair: (-pair[1], pair[0]),
    )[0]
    position = int(best_label) - 1
    expected_key = component_key_for(
        enumeration.episode_id,
        enumeration.m_sha256,
        position,
        enumeration_version=enumeration.enumeration_version,
    )
    if enumeration.components[position].component_key != expected_key:
        raise ValueError("enumeration component key/position mismatch")
    return position


def cue_hit_component_key(
    enumeration: ComponentEnumeration,
    cue_voxels: np.ndarray,
    full_mask: np.ndarray,
) -> Optional[str]:
    """Bind a REMOVE cue and return the stable component key."""

    position = cue_hit_component_position(enumeration, cue_voxels, full_mask)
    if position is None:
        return None
    return enumeration.components[position].component_key


def cue_hit_component(
    enumeration: ComponentEnumeration,
    cue_voxels: np.ndarray,
    full_mask: np.ndarray,
) -> Optional[int]:
    """Compatibility API returning the historical 1-based scipy label.

    New v3 artifacts and model targets must use
    :func:`cue_hit_component_position` or :func:`cue_hit_component_key`.
    Exact label membership is still used, so a cue inside a component's
    bounding-box hole correctly returns ``None``.
    """

    position = cue_hit_component_position(enumeration, cue_voxels, full_mask)
    return None if position is None else position + 1
