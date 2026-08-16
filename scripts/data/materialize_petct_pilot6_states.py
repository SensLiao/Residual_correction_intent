#!/usr/bin/env python3
"""Construct six controlled state-relative ADD/REMOVE intent states.

This is the v2 replacement for the historical add-only Pilot-3 constructor.
It does not generate a scribble or run a model.  It creates two operation-
specific, binary cue supports and three counterfactual M0 states per support:
SAME_LOCAL, SAME_COMPLETE, and NEW_COMPLETE.  Target labels
are topological (18-connectivity in binary GT/M0), never pseudo-instance IDs.
The downstream matched-state builder must still re-derive every label from the
official simulator coordinates and reject any mismatch.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np
from scipy import ndimage


MATERIALIZER_VERSION = "PETCT-PILOT6-STATE-MATERIALIZER-v2.0"
ELIGIBILITY_RECEIPT_VERSION = "PETCT-PILOT6-ELIGIBILITY-v2.0"
DATASET_ID = "PSMA-PET-CT-Lesions-v3"
OPERATIONS = ("ADD", "REMOVE")
GOALS = (
    "ADD_SAME_LOCAL",
    "REMOVE_SAME_LOCAL",
    "ADD_SAME_COMPLETE",
    "REMOVE_SAME_COMPLETE",
    "ADD_NEW_COMPLETE",
    "REMOVE_NEW_COMPLETE",
)
GOALS_BY_OPERATION = {
    operation: tuple(goal for goal in GOALS if goal.startswith(operation + "_"))
    for operation in OPERATIONS
}
CONNECTIVITY = 18
_STRUCTURE_18 = ndimage.generate_binary_structure(3, 2)
_OFFSETS_18 = tuple(
    sorted(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if 0 < abs(dx) + abs(dy) + abs(dz) <= 2
    )
)
THRESHOLDS: dict[str, int | float] = {
    "min_component_voxels": 27,
    "min_geodesic_diameter_edges": 6,
    "local_radius_fraction": 0.20,
    "distal_start_fraction": 0.65,
    "min_support_voxels": 3,
    "max_support_fraction": 0.35,
    "min_prompt_distal_voxels": 3,
    "min_retained_core_voxels": 3,
    "min_graph_distance_gap": 2,
    "remove_shell_iterations": 3,
}


class Pilot6MaterializationError(RuntimeError):
    """Raised when the six-state controlled contract is violated."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _mask_sha256(mask: np.ndarray) -> str:
    binary = np.ascontiguousarray(np.asarray(mask) > 0, dtype=np.uint8)
    return hashlib.sha256(
        _canonical_bytes({"shape": list(binary.shape), "dtype": "uint8"})
        + b"\0"
        + binary.tobytes(order="C")
    ).hexdigest()


def _numeric_sha256(volume: np.ndarray) -> str:
    numeric = np.ascontiguousarray(volume, dtype="<f8")
    return hashlib.sha256(
        _canonical_bytes({"shape": list(numeric.shape), "dtype": "float64-le"})
        + b"\0"
        + numeric.tobytes(order="C")
    ).hexdigest()


def _validate_volumes(
    pet: np.ndarray, ct: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pet_array, ct_array, gt_array = map(np.asarray, (pet, ct, gt))
    if any(array.ndim != 3 for array in (pet_array, ct_array, gt_array)):
        raise Pilot6MaterializationError("PET, CT, and GT must all be 3D")
    if pet_array.shape != ct_array.shape or pet_array.shape != gt_array.shape:
        raise Pilot6MaterializationError("PET, CT, and GT shape mismatch")
    if not np.isfinite(pet_array).all() or not np.isfinite(ct_array).all():
        raise Pilot6MaterializationError("PET and CT must contain finite values")
    if not np.all(np.isin(np.unique(gt_array), (0, 1))):
        raise Pilot6MaterializationError("GT must be a strict binary mask")
    return pet_array, ct_array, np.ascontiguousarray(gt_array, dtype=np.uint8)


def label_components_18(mask: np.ndarray) -> tuple[np.ndarray, int]:
    array = np.asarray(mask)
    if array.ndim != 3:
        raise Pilot6MaterializationError("component mask must be 3D")
    labels, count = ndimage.label(array > 0, structure=_STRUCTURE_18)
    return labels.astype(np.int32, copy=False), int(count)


def _first_coordinate(mask: np.ndarray) -> tuple[int, int, int]:
    """First true voxel in row-major (z, y, x) order.

    Row-major first == lexsort((x, y, z)) first, so this C-level ``argmax``
    scan replaces the previous ``np.argwhere`` allocation (which built the
    full coordinate array for every component — a per-case multi-minute
    cost on whole-body masks).
    """

    array = np.asarray(mask)
    if not array.any():
        raise Pilot6MaterializationError("cannot select from an empty mask")
    flat = int(np.argmax(array.ravel()))
    return tuple(int(value) for value in np.unravel_index(flat, array.shape))


def _graph_distances(mask: np.ndarray, start: Sequence[int]) -> np.ndarray:
    """Geodesic (18-connectivity, on-mask) distances from ``start``.

    Vectorized layer-by-layer BFS confined to the mask bounding box.  Each
    layer uses the C-implemented ``ndimage.binary_dilation``, so a component
    with V voxels and geodesic diameter D costs O(D * bbox_volume) C-ops
    instead of the O(18 * V) pure-Python deque walk (the original
    implementation could not finish one whole-body case within hours).
    """
    array = np.asarray(mask)
    if array.ndim != 3:
        raise Pilot6MaterializationError("component mask must be 3D")
    binary = array > 0
    objects = ndimage.find_objects(binary)
    if objects is None or not objects or objects[0] is None:
        raise Pilot6MaterializationError("cannot traverse an empty mask")
    slices = objects[0]
    sub = binary[slices]
    local_start = tuple(
        int(value) - int(sl.start) for value, sl in zip(start, slices)
    )
    if (
        any(value < 0 or value >= sub.shape[index] for index, value in enumerate(local_start))
        or not sub[local_start]
    ):
        raise Pilot6MaterializationError("start coordinate outside the component")
    distances_sub = np.full(sub.shape, -1, dtype=np.int32)
    frontier = np.zeros(sub.shape, dtype=bool)
    frontier[local_start] = True
    step = 0
    voxel_budget = int(sub.sum()) + 2
    while frontier.any():
        distances_sub[frontier] = step
        frontier = (
            ndimage.binary_dilation(frontier, structure=_STRUCTURE_18)
            & sub
            & (distances_sub < 0)
        )
        step += 1
        if step > voxel_budget:
            raise Pilot6MaterializationError(
                "geodesic traversal exceeded the component size"
            )
    distances = np.full(array.shape, -1, dtype=np.int32)
    distances[slices] = distances_sub
    if np.any(binary & (distances < 0)):
        raise Pilot6MaterializationError("18-connected component traversal failed")
    return distances


def _partition_component(component: np.ndarray) -> dict[str, Any]:
    first = _first_coordinate(component)
    initial = _graph_distances(component, first)
    farthest = int(initial[component].max())
    endpoint = _first_coordinate(component & (initial == farthest))
    distances = _graph_distances(component, endpoint)
    diameter = int(distances[component].max())
    local_radius = int(np.floor(float(THRESHOLDS["local_radius_fraction"]) * diameter))
    distal_start = max(
        int(np.ceil(float(THRESHOLDS["distal_start_fraction"]) * diameter)),
        local_radius + int(THRESHOLDS["min_graph_distance_gap"]),
    )
    support = component & (distances <= local_radius)
    distal = component & (distances >= distal_start)
    core = component & ~support & ~distal
    reasons: list[str] = []
    voxel_count = int(component.sum())
    checks = (
        (voxel_count >= int(THRESHOLDS["min_component_voxels"]), "component_below_min_voxels"),
        (diameter >= int(THRESHOLDS["min_geodesic_diameter_edges"]), "diameter_below_min"),
        (int(support.sum()) >= int(THRESHOLDS["min_support_voxels"]), "support_below_min"),
        (int(distal.sum()) >= int(THRESHOLDS["min_prompt_distal_voxels"]), "distal_below_min"),
        (int(core.sum()) >= int(THRESHOLDS["min_retained_core_voxels"]), "core_below_min"),
        (float(support.sum()) / max(voxel_count, 1) <= float(THRESHOLDS["max_support_fraction"]), "support_above_max_fraction"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "component": component,
        "support": support,
        "distal": distal,
        "core": core,
        "diameter": diameter,
        "endpoint": list(endpoint),
    }


def _remove_component(gt: np.ndarray, selected_component: np.ndarray) -> dict[str, Any]:
    shell = ndimage.binary_dilation(
        selected_component,
        structure=_STRUCTURE_18,
        iterations=int(THRESHOLDS["remove_shell_iterations"]),
    ) & ~gt.astype(bool)
    labels, count = label_components_18(shell)
    candidates = [labels == label for label in range(1, count + 1)]
    if not candidates:
        return {"eligible": False, "reasons": ["no_background_shell"]}
    adjacent = [
        candidate
        for candidate in candidates
        if np.any(ndimage.binary_dilation(candidate, structure=_STRUCTURE_18) & selected_component)
    ]
    if not adjacent:
        return {"eligible": False, "reasons": ["background_shell_not_adjacent"]}
    component = max(adjacent, key=lambda value: (int(value.sum()), tuple(-v for v in _first_coordinate(value))))
    partition = _partition_component(component)
    if not partition["eligible"]:
        return partition
    # Use the endpoint closest to GT for the shared REMOVE cue support, so the
    # LOCAL state is connected to retained truth while COMPLETE has a distal tail.
    distance_to_gt = ndimage.distance_transform_edt(~gt.astype(bool))
    minimum = float(distance_to_gt[component].min())
    near = component & (distance_to_gt <= minimum + 1.0)
    anchor = _first_coordinate(near)
    graph_distance = _graph_distances(component, anchor)
    local_radius = max(1, int(np.floor(0.10 * int(graph_distance[component].max()))))
    support = component & (graph_distance <= local_radius)
    if int(support.sum()) < int(THRESHOLDS["min_support_voxels"]):
        support = component & (graph_distance <= local_radius + 1)
    distal = component & (
        graph_distance >= max(local_radius + 2, int(np.ceil(0.65 * graph_distance[component].max())))
    )
    if not distal.any():
        return {"eligible": False, "reasons": ["remove_distal_below_min"]}
    return {**partition, "support": support, "distal": distal}


def _state(gt: np.ndarray, residual: np.ndarray, *, operation: str, new: bool) -> dict[str, np.ndarray | str]:
    residual_bool = np.asarray(residual) > 0
    if operation == "ADD":
        m0 = gt.astype(bool) & ~residual_bool
        expected = gt.astype(bool) & ~m0
    elif operation == "REMOVE":
        base = np.zeros_like(gt, dtype=bool) if new else gt.astype(bool)
        m0 = base | residual_bool
        expected = m0 & ~gt.astype(bool)
    else:
        raise Pilot6MaterializationError("operation must be ADD or REMOVE")
    if not np.array_equal(expected, residual_bool):
        raise Pilot6MaterializationError("operation residual formula mismatch")
    return {
        "m0": np.ascontiguousarray(m0, dtype=np.uint8),
        "operation_residual": np.ascontiguousarray(residual_bool, dtype=np.uint8),
        "authorized_target": np.ascontiguousarray(residual_bool, dtype=np.uint8),
        "operation": operation,
    }


def construct_pilot6_states(
    pet: np.ndarray,
    ct: np.ndarray,
    gt: np.ndarray,
    *,
    split: str,
    dataset_id: str,
) -> dict[str, Any]:
    """Construct six candidates; downstream coordinates decide final admission."""
    if split != "development":
        raise Pilot6MaterializationError("controlled state materialization is development-only")
    if dataset_id != DATASET_ID:
        raise Pilot6MaterializationError(f"controlled materialization is PSMA-only ({DATASET_ID})")
    pet_array, ct_array, gt_array = _validate_volumes(pet, ct, gt)
    labels, count = label_components_18(gt_array)
    # Bbox-confine every per-component computation: `labels == label` over the
    # full volume plus full-volume argwhere used to cost seconds per component
    # and minutes per case on whole-body masks.  Sub-arrays keep the same
    # (z,y,x) lexicographic order as the full volume, so downstream ordering
    # and tie-breaking are unchanged.
    objects = ndimage.find_objects(labels)
    partitions: list[tuple[int, dict[str, Any]]] = []
    for label_id in range(1, count + 1):
        slices = objects[label_id - 1]
        component_sub = labels[slices] == label_id
        partitions.append((label_id, _partition_component(component_sub)))
    eligible = [(label_id, p) for label_id, p in partitions if p["eligible"]]
    receipt: dict[str, Any] = {
        "schema_version": ELIGIBILITY_RECEIPT_VERSION,
        "materializer_version": MATERIALIZER_VERSION,
        "dataset_id": DATASET_ID,
        "split": split,
        "connectivity": CONNECTIVITY,
        "instance_identity_source": "BINARY_MASK_18_CONNECTIVITY_NOT_INSTANCE_IDS",
        "thresholds": dict(THRESHOLDS),
        "input_content_sha256": {
            "pet_content_sha256": _numeric_sha256(pet_array),
            "ct_content_sha256": _numeric_sha256(ct_array),
            "gt_content_sha256": _mask_sha256(gt_array),
        },
        "operations": list(OPERATIONS),
        "legal_joint_goals": list(GOALS),
        "legacy_add_only_contract": "REJECTED",
        "training_run_count": 0,
        "inference_run_count": 0,
        "experiment_result_count": 0,
    }
    if not eligible:
        return {
            "eligible": False,
            "reason": "NO_ELIGIBLE_GT_COMPONENT",
            "states": {},
            "scribble_supports": {},
            "receipt": {**receipt, "status": "INELIGIBLE"},
        }
    selected_label_id, selected = max(
        eligible,
        key=lambda item: (
            -int(item[1]["component"].sum()),
            _first_coordinate(item[1]["component"]),
        ),
    )
    # Embed only the selected component's masks back to full-volume shape;
    # all other components stay as small bbox sub-arrays.
    def _embed(sub_mask: np.ndarray) -> np.ndarray:
        full = np.zeros(gt_array.shape, dtype=bool)
        full[objects[selected_label_id - 1]] = sub_mask
        return full

    selected_component_full = _embed(selected["component"])
    remove = _remove_component(gt_array, selected_component_full)
    if not remove.get("eligible"):
        return {
            "eligible": False,
            "reason": "NO_ELIGIBLE_REMOVE_BACKGROUND_COMPONENT",
            "states": {},
            "scribble_supports": {},
            "receipt": {**receipt, "status": "INELIGIBLE", "remove_reasons": remove.get("reasons", [])},
        }
    add_local = _embed(selected["support"])
    add_complete = _embed(selected["support"]) | _embed(selected["distal"])
    add_new = selected_component_full
    remove_local = remove["support"]
    remove_complete = remove["component"]
    states = {
        "ADD_SAME_LOCAL": _state(gt_array, add_local, operation="ADD", new=False),
        "REMOVE_SAME_LOCAL": _state(gt_array, remove_local, operation="REMOVE", new=False),
        "ADD_SAME_COMPLETE": _state(gt_array, add_complete, operation="ADD", new=False),
        "REMOVE_SAME_COMPLETE": _state(gt_array, remove_complete, operation="REMOVE", new=False),
        "ADD_NEW_COMPLETE": _state(gt_array, add_new, operation="ADD", new=True),
        "REMOVE_NEW_COMPLETE": _state(gt_array, remove_complete, operation="REMOVE", new=True),
    }
    if tuple(states) != GOALS:
        raise Pilot6MaterializationError("internal six-class order drift")
    supports = {
        "ADD": np.ascontiguousarray(add_local, dtype=np.uint8),
        "REMOVE": np.ascontiguousarray(remove_local, dtype=np.uint8),
    }
    coverage = {
        goal: {
            "operation": str(states[goal]["operation"]),
            "residual_voxels": int(np.asarray(states[goal]["operation_residual"]).sum()),
            "m0_sha256": _mask_sha256(np.asarray(states[goal]["m0"])),
        }
        for goal in GOALS
    }
    if set(coverage) != set(GOALS):
        raise Pilot6MaterializationError("six-class coverage receipt is incomplete")
    receipt.update(
        {
            "status": "ELIGIBLE",
            "selected_gt_component_sha256": _mask_sha256(selected["component"]),
            "remove_background_component_sha256": _mask_sha256(remove["component"]),
            "cue_support_sha256": {operation: _mask_sha256(mask) for operation, mask in supports.items()},
            "shared_physical_cue_within_operation": True,
            "shared_physical_cue_across_operations": False,
            "class_coverage": coverage,
            "class_coverage_complete": set(coverage) == set(GOALS),
        }
    )
    return {
        "eligible": True,
        "states": states,
        "scribble_supports": supports,
        "receipt": receipt,
    }
