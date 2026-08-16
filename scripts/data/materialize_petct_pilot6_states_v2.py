#!/usr/bin/env python3
"""Euclidean scribble-anchored pilot6 state construction (v2, D-2026-08-16-01).

The v1 constructor partitioned components with graph distances (10%/65% of
the geodesic diameter) while the official re-derivation used the Euclidean
15 mm radius + 50 mm^2 area rules — the two radii disagreed systematically
and 72.6% of the corpus was excluded at the receipt.  This module replaces
the partition entirely: the official scribble is generated first, and every
one of the six state masks is anchored to that scribble's Euclidean
geometry, so construction and re-derivation share one source of truth.

All functions here operate on inference-visible material plus the scribble;
no GT-derived quantity beyond the target/false-positive masks is exposed.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
from scipy import ndimage

PILOT6_V2_SCHEMA = "PETCT-PILOT6-STATES-v2.0"
GOALS = (
    "ADD_SAME_LOCAL",
    "REMOVE_SAME_LOCAL",
    "ADD_SAME_COMPLETE",
    "REMOVE_SAME_COMPLETE",
    "ADD_NEW_COMPLETE",
    "REMOVE_NEW_COMPLETE",
)
STRUCTURE_18 = np.ones((3, 3, 3), dtype=bool)
for x, y, z in (
    (0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2),
    (2, 0, 0), (2, 0, 2), (2, 2, 0), (2, 2, 2),
):
    STRUCTURE_18[x, y, z] = False


class Pilot6V2Error(ValueError):
    """Raised when Euclidean-anchored state construction cannot proceed."""


def select_add_component(
    gt: np.ndarray, *, min_component_voxels: int
) -> Tuple[np.ndarray, str]:
    """Pick the ADD target: largest eligible GT component.

    Tie-break is the row-major first coordinate (deterministic).  Returns
    the full-volume component mask and an eligibility reason when there is
    no component above the size floor.
    """

    truth = np.asarray(gt) > 0
    labels, count = ndimage.label(truth, structure=STRUCTURE_18)
    if count == 0:
        raise Pilot6V2Error("no GT component")
    best_label = None
    best_size = 0
    best_first = None
    for label_id in range(1, count + 1):
        size = int(np.count_nonzero(labels == label_id))
        if size < int(min_component_voxels):
            continue
        first = _row_major_first(labels == label_id)
        if size > best_size or (size == best_size and (best_first is None or first < best_first)):
            best_size, best_label, best_first = size, label_id, first
    if best_label is None:
        raise Pilot6V2Error("no GT component reaches the minimum voxel floor")
    return labels == best_label, ""


def select_remove_component(
    gt: np.ndarray, selected_component: np.ndarray, *, shell_iterations: int, min_component_voxels: int
) -> np.ndarray:
    """Pick the REMOVE target: largest background shell adjacent to the
    selected ADD component (dilated ring outside GT, 18-connectivity)."""

    truth = np.asarray(gt) > 0
    shell = (
        ndimage.binary_dilation(
            selected_component, structure=STRUCTURE_18, iterations=int(shell_iterations)
        )
        & ~truth
    )
    labels, count = ndimage.label(shell, structure=STRUCTURE_18)
    candidates = []
    for label_id in range(1, count + 1):
        candidate = labels == label_id
        if int(candidate.sum()) < int(min_component_voxels):
            continue
        if not np.any(
            ndimage.binary_dilation(candidate, structure=STRUCTURE_18)
            & selected_component
        ):
            continue
        candidates.append(candidate)
    if not candidates:
        raise Pilot6V2Error("no eligible background shell adjacent to the selected component")
    return max(candidates, key=lambda mask: (int(mask.sum()), tuple(-v for v in _row_major_first(mask))))


def euclidean_local_mask(
    residual: np.ndarray,
    scribble_coords: Sequence[Sequence[int]],
    spacing_xy: Sequence[float],
    local_radius_mm: float,
) -> Tuple[np.ndarray, bool]:
    """Local subset of the residual on the scribble's axial slice.

    Mirrors the official re-derivation exactly: the Euclidean distance is
    computed on the scribble z-slice; ``local`` holds the residual voxels
    within ``local_radius_mm`` of any scribble voxel.  Returns the 2D local
    mask on that slice and whether any far (out-of-radius) voxel exists on
    the same slice.
    """

    residual_bin = np.asarray(residual) > 0
    coords = np.asarray(scribble_coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise Pilot6V2Error("scribble coordinates must be Nx3")
    z_values = {int(coord[2]) for coord in coords}
    if len(z_values) != 1:
        raise Pilot6V2Error("one-slice scribble required for the local partition")
    center_z = next(iter(z_values))
    scribble_2d = np.zeros(residual_bin.shape[:2], dtype=bool)
    for x, y, _ in coords:
        scribble_2d[int(x), int(y)] = True
    spacing = tuple(float(value) for value in spacing_xy)
    if len(spacing) != 2 or any(value <= 0 for value in spacing):
        raise Pilot6V2Error("positive in-plane spacing is required")
    distance = ndimage.distance_transform_edt(~scribble_2d, sampling=spacing)
    residual_slice = residual_bin[:, :, center_z]
    local = residual_slice & (distance <= float(local_radius_mm))
    far = residual_slice & (distance > float(local_radius_mm))
    return local, bool(far.any())


def construct_pilot6_states_v2(
    gt: np.ndarray,
    *,
    add_component: np.ndarray,
    remove_component: np.ndarray,
    scribble_add: Sequence[Sequence[int]],
    scribble_remove: Sequence[Sequence[int]],
    spacing_xy: Sequence[float],
    local_radius_mm: float = 15.0,
    minimum_local_area_mm2: float = 50.0,
) -> Dict[str, Any]:
    """Build the six state masks from the Euclidean scribble geometry.

    For each goal the mask is constructed so that the official re-derivation
    (``derive_goal_and_authorized_target``) provably returns the same goal:

      ADD_SAME_LOCAL      m0 = gt minus local(FN) on the scribble slice
      ADD_SAME_COMPLETE   m0 = gt minus full FN component
      ADD_NEW_COMPLETE    m0 = gt minus the whole component
      REMOVE_SAME_LOCAL   m0 = gt plus local(FP) on the scribble slice
      REMOVE_SAME_COMPLETE m0 = gt plus full FP shell
      REMOVE_NEW_COMPLETE m0 = FP shell only

    Returns the states dict keyed by goal; raises Pilot6V2Error when the
    LOCAL area floor or the COMPLETE far-voxel requirement fails (the
    builder converts that into a counted exclusion).
    """

    truth = np.asarray(gt) > 0
    if truth.shape != add_component.shape or truth.shape != remove_component.shape:
        raise Pilot6V2Error("GT and component masks must share shape")
    if truth.ndim != 3:
        raise Pilot6V2Error("masks must be 3D")
    if not np.all(add_component <= truth):
        raise Pilot6V2Error("ADD component must be a subset of GT")
    if np.any(remove_component & truth):
        raise Pilot6V2Error("REMOVE shell must be disjoint from GT")
    spacing = tuple(float(value) for value in spacing_xy)

    local_fn, far_fn = euclidean_local_mask(
        add_component, scribble_add, spacing, local_radius_mm
    )
    area_fn = float(local_fn.sum()) * spacing[0] * spacing[1]
    if area_fn < float(minimum_local_area_mm2):
        raise Pilot6V2Error(
            "ADD SAME_LOCAL candidate is below the minimum physical area"
        )
    if not far_fn:
        raise Pilot6V2Error("ADD COMPLETE requires far residual voxels on the scribble slice")

    local_fp, far_fp = euclidean_local_mask(
        remove_component, scribble_remove, spacing, local_radius_mm
    )
    area_fp = float(local_fp.sum()) * spacing[0] * spacing[1]
    if area_fp < float(minimum_local_area_mm2):
        raise Pilot6V2Error(
            "REMOVE SAME_LOCAL candidate is below the minimum physical area"
        )
    if not far_fp:
        raise Pilot6V2Error("REMOVE COMPLETE requires far residual voxels on the scribble slice")

    z_add = int(next(iter({int(c[2]) for c in scribble_add})))
    z_remove = int(next(iter({int(c[2]) for c in scribble_remove})))

    def minus_slice(base: np.ndarray, local_mask: np.ndarray, z: int) -> np.ndarray:
        out = base.copy()
        out[:, :, z] = out[:, :, z] & ~local_mask
        return out

    def plus_slice(base: np.ndarray, local_mask: np.ndarray, z: int) -> np.ndarray:
        out = base.copy()
        out[:, :, z] = out[:, :, z] | local_mask
        return out

    # SAME states must retain a non-empty part of the target component in m0
    # (the official counterpart check).  The retained cap sits on the slice
    # farthest from the scribble z, so the scribble-slice residual keeps its
    # far voxels and the COMPLETE derivation stays COMPLETE.
    def _retained_cap(component: np.ndarray, z: int) -> np.ndarray:
        slices_with_voxels = [
            int(z_index) for z_index in range(component.shape[2])
            if np.any(component[:, :, z_index])
        ]
        if not slices_with_voxels or slices_with_voxels == [z]:
            raise Pilot6V2Error(
                "ADD SAME_COMPLETE requires a retainable cap outside the scribble slice"
            )
        cap_z = max(slices_with_voxels, key=lambda value: abs(value - z))
        cap = np.zeros(component.shape, dtype=bool)
        cap[:, :, cap_z] = component[:, :, cap_z]
        return cap

    cap = _retained_cap(add_component, z_add)
    fn_complete = add_component & ~cap
    m0_add_local = minus_slice(truth, local_fn, z_add)
    m0_add_complete = truth & ~fn_complete
    m0_add_new = truth & ~add_component
    m0_remove_local = plus_slice(truth, local_fp, z_remove)
    m0_remove_complete = truth | remove_component
    m0_remove_new = remove_component.copy()

    states: Dict[str, Dict[str, Any]] = {}
    for goal, m0, residual, operation in (
        ("ADD_SAME_LOCAL", m0_add_local, truth & ~m0_add_local, "ADD"),
        ("ADD_SAME_COMPLETE", m0_add_complete, fn_complete, "ADD"),
        ("ADD_NEW_COMPLETE", m0_add_new, add_component, "ADD"),
        ("REMOVE_SAME_LOCAL", m0_remove_local, m0_remove_local & ~truth, "REMOVE"),
        ("REMOVE_SAME_COMPLETE", m0_remove_complete, remove_component, "REMOVE"),
        ("REMOVE_NEW_COMPLETE", m0_remove_new, remove_component, "REMOVE"),
    ):
        residual_bool = np.asarray(residual) > 0
        expected = (
            (truth & ~m0) if operation == "ADD" else (m0 & ~truth)
        )
        if not np.array_equal(expected, residual_bool):
            raise Pilot6V2Error("operation residual formula mismatch for %s" % goal)
        states[goal] = {
            "m0": np.ascontiguousarray(m0, dtype=np.uint8),
            "operation": operation,
            "operation_residual": np.ascontiguousarray(residual_bool, dtype=np.uint8),
            "authorized_target": np.ascontiguousarray(residual_bool, dtype=np.uint8),
        }
    return {
        "schema_version": PILOT6_V2_SCHEMA,
        "eligible": True,
        "states": states,
    }


def _row_major_first(mask: np.ndarray) -> Tuple[int, int, int]:
    array = np.asarray(mask)
    if not array.any():
        raise Pilot6V2Error("cannot select from an empty mask")
    flat = int(np.argmax(array.ravel()))
    return tuple(int(value) for value in np.unravel_index(flat, array.shape))
