#!/usr/bin/env python3
"""Pure operation-aware metrics for one PET/CT residual-correction episode."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import ndimage


class BidirectionalMetricError(RuntimeError):
    """An ADD/REMOVE correction contract was violated."""


def _mask(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in (2, 3) or not np.all(np.isin(array, (0, 1))):
        raise BidirectionalMetricError(f"{name} must be a binary 2D/3D mask")
    return array.astype(bool, copy=False)


def _dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(truth.sum())
    return (
        1.0
        if denominator == 0
        else float(2 * np.logical_and(prediction, truth).sum() / denominator)
    )


def _recall(changed: np.ndarray, target: np.ndarray) -> float | None:
    denominator = int(target.sum())
    return (
        None
        if denominator == 0
        else float(np.logical_and(changed, target).sum() / denominator)
    )


def bidirectional_correction_metrics(
    *,
    gt: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    authorized_target: np.ndarray,
    cue_support: np.ndarray,
    operation: str,
    execution_operation: str | None = None,
    scope: str,
    spacing_xyz: Sequence[float],
    distal_radius_mm: float,
) -> dict[str, Any]:
    """Evaluate a delta under the canonical operation-specific set algebra.

    ``authorized_target`` is an ADD delta in ``GT\\M0`` or a REMOVE delta in
    ``M0\\GT``.  The same delta head is therefore scored symmetrically while
    ADD and REMOVE retain different safety denominators.
    """

    truth = _mask(gt, "GT")
    current = _mask(m0, "M0")
    corrected = _mask(m1, "M1")
    authorized = _mask(authorized_target, "authorized target")
    cue = _mask(cue_support, "cue support")
    if len({truth.shape, current.shape, corrected.shape, authorized.shape, cue.shape}) != 1:
        raise BidirectionalMetricError("all masks must share one shape")
    if operation not in {"ADD", "REMOVE"}:
        raise BidirectionalMetricError("operation must be ADD or REMOVE")
    execution = operation if execution_operation is None else execution_operation
    if execution not in {"ADD", "REMOVE"}:
        raise BidirectionalMetricError("execution_operation must be ADD or REMOVE")
    if scope not in {"LOCAL", "COMPLETE"}:
        raise BidirectionalMetricError("scope must be LOCAL or COMPLETE")
    spacing = np.asarray(tuple(spacing_xyz), dtype=float)
    if (
        spacing.shape != (truth.ndim,)
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0)
    ):
        raise BidirectionalMetricError("spacing must be positive and match mask rank")
    if not np.isfinite(distal_radius_mm) or distal_radius_mm < 0:
        raise BidirectionalMetricError("distal radius must be finite and non-negative")
    if not authorized.any() or np.any(cue & ~authorized):
        raise BidirectionalMetricError(
            "authorized target must be non-empty and contain the cue support"
        )

    expected_domain = (truth & ~current) if operation == "ADD" else (current & ~truth)
    if np.any(authorized & ~expected_domain):
        raise BidirectionalMetricError(
            f"{operation} authorized target is outside its residual domain"
        )

    added = corrected & ~current
    removed = current & ~corrected
    if execution == "ADD" and removed.any():
        raise BidirectionalMetricError("ADD prediction violates M1=M0 union Delta")
    if execution == "REMOVE" and added.any():
        raise BidirectionalMetricError("REMOVE prediction violates M1=M0 minus Delta")
    changed = added if execution == "ADD" else removed
    authorized_changed = changed & authorized
    unauthorized_add = added & ~authorized
    unauthorized_remove = removed & ~authorized

    distance = ndimage.distance_transform_edt(~cue, sampling=tuple(spacing))
    distal = authorized & (distance > float(distal_radius_mm))
    voxel_measure = float(np.prod(spacing))
    # With 3-D spacing this is millilitres; the 2-D ROI diagnostic remains mm^2.
    volume_ml_factor = voxel_measure / 1000.0 if truth.ndim == 3 else None
    protected_tp = current & truth
    protected_tp_count = int(protected_tp.sum())
    preserved_tp_count = int((protected_tp & corrected).sum())
    changed_count = int(changed.sum())
    authorized_count = int(authorized.sum())
    unauthorized_add_count = int(unauthorized_add.sum())
    unauthorized_remove_count = int(unauthorized_remove.sum())
    authorized_recall = _recall(changed, authorized)
    distal_recall = _recall(changed, distal)
    dice_m0 = _dice(current, truth)
    dice_m1 = _dice(corrected, truth)
    false_positive = corrected & ~truth
    false_negative = truth & ~corrected
    complete_remove_defined = operation == "REMOVE" and scope == "COMPLETE"
    complete_remove = (
        float(not np.any(authorized & corrected))
        if complete_remove_defined
        else None
    )

    return {
        "operation": operation,
        "execution_operation": execution,
        "execution_operation_matches_target": execution == operation,
        "scope": scope,
        "dice_m0": dice_m0,
        "dice_m1": dice_m1,
        "dice_gain": dice_m1 - dice_m0,
        "delta_dice": dice_m1 - dice_m0,
        "authorized_delta_recall": authorized_recall,
        "authorized_add_recall": authorized_recall if operation == "ADD" else None,
        "authorized_add_recall_defined": operation == "ADD",
        "authorized_remove_recall": (
            authorized_recall if operation == "REMOVE" else None
        ),
        "authorized_remove_recall_defined": operation == "REMOVE",
        # Compatibility name used by the existing confirmatory analysis.
        "target_residual_recall": authorized_recall,
        "target_residual_recall_defined": True,
        "target_residual_recall_denominator_voxels": float(authorized_count),
        "prompt_distal_recall": distal_recall,
        "prompt_distal_recall_defined": bool(distal.any()),
        "prompt_distal_recall_denominator_voxels": float(distal.sum()),
        "authorized_delta_voxels": float(authorized_changed.sum()),
        "authorized_addition_voxels": (
            float(authorized_changed.sum()) if operation == "ADD" else 0.0
        ),
        "authorized_removal_voxels": (
            float(authorized_changed.sum()) if operation == "REMOVE" else 0.0
        ),
        "unauthorized_addition_voxels": float(unauthorized_add_count),
        "unauthorized_addition_physical_measure": float(
            unauthorized_add_count * voxel_measure
        ),
        "unauthorized_removal_voxels": float(unauthorized_remove_count),
        "unauthorized_removal_physical_measure": float(
            unauthorized_remove_count * voxel_measure
        ),
        "residual_precision": (
            None
            if changed_count == 0
            else float(authorized_changed.sum() / changed_count)
        ),
        "residual_precision_defined": changed_count > 0,
        "residual_precision_denominator_voxels": float(changed_count),
        "true_positive_preservation_rate": (
            None
            if protected_tp_count == 0
            else float(preserved_tp_count / protected_tp_count)
        ),
        "true_positive_preservation_rate_defined": protected_tp_count > 0,
        "true_positive_preservation_denominator_voxels": float(protected_tp_count),
        "complete_fp_removal": complete_remove,
        "complete_fp_removal_defined": complete_remove_defined,
        "complete_fp_removal_denominator_voxels": (
            float(authorized_count) if complete_remove_defined else 0.0
        ),
        "fpv_ml": (
            None
            if volume_ml_factor is None
            else float(false_positive.sum() * volume_ml_factor)
        ),
        "fnv_ml": (
            None
            if volume_ml_factor is None
            else float(false_negative.sum() * volume_ml_factor)
        ),
        "m0_voxels": float(current.sum()),
        "m0_preserved_voxels": float((current & corrected).sum()),
        "m0_removed_voxels": float(removed.sum()),
        "m0_preservation_rate": (
            None if not current.any() else float((current & corrected).sum() / current.sum())
        ),
        "physical_measure_unit": "mm^%d" % truth.ndim,
        "operation_algebra_safety_pass": 1.0,
        "target_operation_safety_pass": float(execution == operation),
        "unauthorized_change_voxels": float(
            unauthorized_add_count + unauthorized_remove_count
        ),
    }
