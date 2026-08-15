#!/usr/bin/env python3
"""Pure, CPU-testable contracts for the PET/CT Route-A pipeline.

This module deliberately contains no training launch, SSH, download or remote-write
logic.  GPU entrypoints consume these contracts, while unit tests exercise them on
small synthetic arrays.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy import ndimage


FOLDS = (0, 1, 2, 3, 4)
SCRIBBLE_STRATEGIES = ("centerline", "random", "boundary")
OPERATIONS = ("ADD", "REMOVE")
TARGETS = ("SAME", "NEW")
SCOPES = ("LOCAL", "COMPLETE")
LEGAL_JOINT_GOALS = (
    "ADD_SAME_LOCAL",
    "REMOVE_SAME_LOCAL",
    "ADD_SAME_COMPLETE",
    "REMOVE_SAME_COMPLETE",
    "ADD_NEW_COMPLETE",
    "REMOVE_NEW_COMPLETE",
)
LEGACY_ADD_ONLY_GOALS = ("SAME_LOCAL", "SAME_COMPLETE", "NEW_COMPLETE")
EDITOR_CONDITIONS = (
    "visual_state_only",
    "spatial_only",
    "intent_only",
    "scribble_plus_intent",
    "scribble_plus_operation",
    "same_weight_NULL",
    "same_weight_wrong_scope",
    "same_weight_shuffled",
    "oracle_slots",
    "predicted_slots",
    "wrong_operation_OOD",
)
EDITOR_TRAINING_CONDITIONS = (
    "visual_state_only",
    "spatial_only",
    "intent_only",
    "scribble_plus_intent",
)
EDITOR_CHECKPOINT_CONDITION_ALIASES = {
    "scribble_plus_operation": "scribble_plus_intent",
    "same_weight_NULL": "scribble_plus_intent",
    "same_weight_wrong_scope": "scribble_plus_intent",
    "same_weight_shuffled": "scribble_plus_intent",
    "oracle_slots": "scribble_plus_intent",
    "predicted_slots": "scribble_plus_intent",
    "wrong_operation_OOD": "scribble_plus_intent",
}


class ContractError(RuntimeError):
    """A frozen data, comparison or evaluation invariant was violated."""


def expected_editor_checkpoint_condition(condition: str) -> str:
    """Resolve an inference condition to one of four distinct trainable editors."""

    if condition not in EDITOR_CONDITIONS:
        raise ContractError("unknown editor condition: %s" % condition)
    return EDITOR_CHECKPOINT_CONDITION_ALIASES.get(condition, condition)


def validate_intent_slots(
    operation: str, target: str, scope: str
) -> tuple[str, str, str]:
    """Validate the factorized intent and its six-class structural support."""

    if operation not in OPERATIONS:
        raise ContractError("operation must be ADD or REMOVE")
    if target not in TARGETS:
        raise ContractError("target must be SAME or NEW")
    if scope not in SCOPES:
        raise ContractError("scope must be LOCAL or COMPLETE")
    if target == "NEW" and scope == "LOCAL":
        raise ContractError("NEW_LOCAL is structurally invalid")
    return operation, target, scope


def joint_goal(operation: str, target: str, scope: str) -> str:
    validate_intent_slots(operation, target, scope)
    goal = f"{operation}_{target}_{scope}"
    if goal not in LEGAL_JOINT_GOALS:
        raise ContractError("slot tuple is not in the frozen six-class ontology")
    return goal


def intent_slots_from_goal(goal: str) -> tuple[str, str, str]:
    """Decode a current goal; retired SAME/NEW labels fail closed.

    The old labels remain identifiable for provenance/migration reports but
    must never enter a current training manifest without an explicit offline
    migration that emits a new six-class goal.
    """

    if goal in LEGACY_ADD_ONLY_GOALS:
        raise ContractError(
            "legacy three-class ADD-only goal is not valid current supervision"
        )
    parts = str(goal).split("_")
    if len(parts) != 3:
        raise ContractError("intent goal must contain operation/target/scope")
    operation, target, scope = parts
    if joint_goal(operation, target, scope) != goal:
        raise ContractError("intent goal is not canonical")
    return operation, target, scope


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def binary_mask(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim not in (2, 3):
        raise ContractError("%s must be a 2D or 3D mask" % name)
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise ContractError("%s must be numeric or boolean" % name)
    if not np.all(np.isfinite(array)) or not np.all(np.isin(array, [0, 1])):
        raise ContractError("%s must contain only finite 0/1 values" % name)
    return array.astype(bool, copy=False)


def residual_masks(gt: np.ndarray, m0: np.ndarray) -> Dict[str, np.ndarray]:
    truth = binary_mask(gt, "GT")
    current = binary_mask(m0, "M0")
    if truth.shape != current.shape:
        raise ContractError("GT and M0 shapes differ")
    return {"fn": truth & ~current, "fp": current & ~truth}


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = binary_mask(prediction, "prediction")
    truth = binary_mask(target, "target")
    if pred.shape != truth.shape:
        raise ContractError("prediction and target shapes differ")
    denom = int(pred.sum()) + int(truth.sum())
    return 1.0 if denom == 0 else float(2 * np.logical_and(pred, truth).sum() / denom)


def safe_recall(prediction: np.ndarray, target: np.ndarray) -> Optional[float]:
    pred = binary_mask(prediction, "prediction")
    truth = binary_mask(target, "target")
    if pred.shape != truth.shape:
        raise ContractError("prediction and target shapes differ")
    return None if not truth.any() else float(np.logical_and(pred, truth).sum() / truth.sum())


def prompt_distal_mask(
    authorized_target: np.ndarray,
    scribble: np.ndarray,
    spacing_xyz: Sequence[float],
    radius_mm: float,
) -> np.ndarray:
    target = binary_mask(authorized_target, "authorized target")
    prompt = binary_mask(scribble, "scribble")
    if target.shape != prompt.shape:
        raise ContractError("authorized target and scribble shapes differ")
    if len(spacing_xyz) != target.ndim or any(float(v) <= 0 for v in spacing_xyz):
        raise ContractError("spacing must be positive and match the mask rank")
    if radius_mm < 0:
        raise ContractError("radius_mm must be non-negative")
    distance = ndimage.distance_transform_edt(~prompt, sampling=tuple(spacing_xyz))
    return target & (distance > float(radius_mm))


def correction_metrics(
    *,
    gt: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    operation: str,
    authorized_target: np.ndarray,
    scribble: np.ndarray,
    spacing_xyz: Sequence[float],
    distal_radius_mm: float,
    gt_lesion_identity: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    truth = binary_mask(gt, "GT")
    current = binary_mask(m0, "M0")
    corrected = binary_mask(m1, "M1")
    target = binary_mask(authorized_target, "authorized target")
    prompt = binary_mask(scribble, "scribble")
    shapes = {truth.shape, current.shape, corrected.shape, target.shape, prompt.shape}
    if len(shapes) != 1:
        raise ContractError("all correction masks must share one shape")
    if operation not in OPERATIONS:
        raise ContractError("operation must be ADD or REMOVE")
    legal_residual = (truth & ~current) if operation == "ADD" else (current & ~truth)
    if not target.any() or np.any(target & ~legal_residual):
        formula = "GT\\M0" if operation == "ADD" else "M0\\GT"
        raise ContractError(
            f"{operation} authorized target must be a non-empty subset of {formula}"
        )
    if not prompt.any() or np.any(prompt & ~target):
        raise ContractError("scribble must be non-empty and inside authorized target")
    removed = current & ~corrected
    added = corrected & ~current
    if operation == "ADD" and removed.any():
        raise ContractError("ADD operation violates M1=M0 union Delta")
    if operation == "REMOVE" and added.any():
        raise ContractError("REMOVE operation violates M1=M0 minus Delta")
    if operation == "REMOVE":
        changed = removed
        authorized_removed = changed & target
        unauthorized_removed = changed & ~target
        distal = prompt_distal_mask(target, prompt, spacing_xyz, distal_radius_mm)
        pixel_measure = float(np.prod(np.asarray(spacing_xyz, dtype=float)))
        target_count = int(target.sum())
        distal_count = int(distal.sum())
        changed_count = int(changed.sum())
        protected_truth = current & truth
        protected_count = int(protected_truth.sum())
        preserved_truth = int((protected_truth & corrected).sum())
        return {
            "operation": operation,
            "dice_m0": dice(current, truth),
            "dice_m1": dice(corrected, truth),
            "dice_gain": dice(corrected, truth) - dice(current, truth),
            "target_residual_recall": safe_recall(changed, target),
            "target_residual_recall_defined": target_count > 0,
            "target_residual_recall_denominator_voxels": float(target_count),
            "prompt_distal_recall": safe_recall(changed, distal),
            "prompt_distal_recall_defined": distal_count > 0,
            "prompt_distal_recall_denominator_voxels": float(distal_count),
            "authorized_delta_voxels": float(authorized_removed.sum()),
            "authorized_addition_voxels": 0.0,
            "authorized_removal_voxels": float(authorized_removed.sum()),
            "unauthorized_addition_voxels": 0.0,
            "unauthorized_removal_voxels": float(unauthorized_removed.sum()),
            "unauthorized_removal_physical_measure": float(
                unauthorized_removed.sum() * pixel_measure
            ),
            "unauthorized_change_voxels": float(unauthorized_removed.sum()),
            "residual_precision": (
                None
                if changed_count == 0
                else float(authorized_removed.sum() / changed_count)
            ),
            "residual_precision_defined": changed_count > 0,
            "residual_precision_denominator_voxels": float(changed_count),
            "true_positive_preservation_rate": (
                None if protected_count == 0 else float(preserved_truth / protected_count)
            ),
            "true_positive_preservation_rate_defined": protected_count > 0,
            "true_positive_preservation_denominator_voxels": float(protected_count),
            "m0_voxels": float(current.sum()),
            "m0_preserved_voxels": float((current & corrected).sum()),
            "m0_removed_voxels": float(removed.sum()),
            "m0_preservation_rate": (
                None
                if not current.any()
                else float((current & corrected).sum() / current.sum())
            ),
            "physical_measure_unit": "mm^%d" % truth.ndim,
            "operation_algebra_safety_pass": 1.0,
        }
    authorized_added = added & target
    unauthorized = added & ~target
    structure = ndimage.generate_binary_structure(truth.ndim, 2)
    if gt_lesion_identity is None:
        truth_labels, _ = ndimage.label(truth, structure=structure)
    else:
        truth_labels = np.asarray(gt_lesion_identity)
        if (
            truth_labels.shape != truth.shape
            or not np.issubdtype(truth_labels.dtype, np.integer)
            or np.any(truth_labels < 0)
            or np.any((truth_labels > 0) != truth)
        ):
            raise ContractError(
                "GT lesion identity must be a non-negative integer label for every GT voxel"
            )
    target_lesion_ids = {
        int(value) for value in np.unique(truth_labels[target]) if int(value) > 0
    }
    if len(target_lesion_ids) != 1:
        raise ContractError("authorized target must bind exactly one GT lesion")
    target_lesion_id = next(iter(target_lesion_ids))
    target_lesion = truth_labels == target_lesion_id
    target_lesion_residual = target_lesion & ~current
    target_scope_residual = target_lesion_residual & ~target
    target_scope_overreach = added & target_scope_residual
    other_lesion_residual = truth & ~current & ~target_lesion
    other_lesion_capture = added & other_lesion_residual
    background_false_addition = added & ~truth
    if not np.array_equal(
        unauthorized,
        target_scope_overreach | other_lesion_capture | background_false_addition,
    ):
        raise ContractError("unauthorized-addition lesion partition is inconsistent")
    distal = prompt_distal_mask(target, prompt, spacing_xyz, distal_radius_mm)
    addition_count = int(added.sum())
    current_labels, current_count = ndimage.label(current, structure=structure)
    corrected_labels, corrected_count = ndimage.label(corrected, structure=structure)
    current_component_gt_ids = {
        current_id: {
            int(value)
            for value in np.unique(truth_labels[current_labels == current_id])
            if int(value) > 0
        }
        for current_id in range(1, current_count + 1)
    }
    harmful_merged_existing_components = 0
    legal_same_lesion_reconnections = 0
    for corrected_id in range(1, corrected_count + 1):
        existing_ids = set(
            int(value)
            for value in np.unique(current_labels[corrected_labels == corrected_id])
            if int(value) > 0
        )
        if len(existing_ids) > 1:
            identity_sets = [current_component_gt_ids[value] for value in existing_ids]
            common_gt_identity = (
                set.intersection(*(set(value) for value in identity_sets))
                if all(identity_sets)
                else set()
            )
            if common_gt_identity:
                legal_same_lesion_reconnections += 1
            else:
                harmful_merged_existing_components += 1
    target_added_bridged_to_other_identity = False
    for corrected_id in {
        int(value)
        for value in np.unique(corrected_labels[authorized_added])
        if int(value) > 0
    }:
        for current_id in {
            int(value)
            for value in np.unique(current_labels[corrected_labels == corrected_id])
            if int(value) > 0
        }:
            if target_lesion_id not in current_component_gt_ids[current_id]:
                target_added_bridged_to_other_identity = True
                break
        if target_added_bridged_to_other_identity:
            break
    pixel_measure = float(np.prod(np.asarray(spacing_xyz, dtype=float)))
    authorized_count = int(authorized_added.sum())
    unauthorized_count = int(unauthorized.sum())
    target_scope_overreach_count = int(target_scope_overreach.sum())
    other_lesion_capture_count = int(other_lesion_capture.sum())
    background_false_addition_count = int(background_false_addition.sum())
    target_count = int(target.sum())
    distal_count = int(distal.sum())
    target_scope_residual_count = int(target_scope_residual.sum())
    other_lesion_residual_count = int(other_lesion_residual.sum())
    m0_count = int(current.sum())
    return {
        "operation": operation,
        "dice_m0": dice(current, truth),
        "dice_m1": dice(corrected, truth),
        "dice_gain": dice(corrected, truth) - dice(current, truth),
        "target_residual_recall": safe_recall(added, target),
        "target_residual_recall_defined": target_count > 0,
        "target_residual_recall_denominator_voxels": float(target_count),
        "prompt_distal_recall": safe_recall(added, distal),
        "prompt_distal_recall_defined": distal_count > 0,
        "prompt_distal_recall_denominator_voxels": float(distal_count),
        "authorized_addition_voxels": float(authorized_count),
        "authorized_addition_physical_measure": float(authorized_count * pixel_measure),
        "unauthorized_addition_voxels": float(unauthorized_count),
        "unauthorized_addition_ratio": (
            0.0 if addition_count == 0 else float(unauthorized_count / addition_count)
        ),
        "unauthorized_addition_ratio_denominator_voxels": float(addition_count),
        "residual_precision": (
            None if addition_count == 0 else float(authorized_count / addition_count)
        ),
        "residual_precision_defined": addition_count > 0,
        "residual_precision_denominator_voxels": float(addition_count),
        "unauthorized_addition_physical_measure": float(
            unauthorized_count * pixel_measure
        ),
        "physical_measure_unit": "mm^%d" % truth.ndim,
        "target_lesion_scope_overreach_voxels": float(target_scope_overreach_count),
        "target_lesion_scope_overreach_physical_measure": float(
            target_scope_overreach_count * pixel_measure
        ),
        "target_lesion_scope_overreach_ratio": (
            None
            if target_scope_residual_count == 0
            else float(target_scope_overreach_count / target_scope_residual_count)
        ),
        "target_lesion_scope_overreach_ratio_defined": target_scope_residual_count > 0,
        "target_lesion_scope_overreach_ratio_denominator_voxels": float(
            target_scope_residual_count
        ),
        "other_lesion_capture_voxels": float(other_lesion_capture_count),
        "other_lesion_capture_physical_measure": float(
            other_lesion_capture_count * pixel_measure
        ),
        "other_lesion_capture_ratio": (
            None
            if other_lesion_residual_count == 0
            else float(other_lesion_capture_count / other_lesion_residual_count)
        ),
        "other_lesion_capture_ratio_defined": other_lesion_residual_count > 0,
        "other_lesion_capture_ratio_denominator_voxels": float(
            other_lesion_residual_count
        ),
        "background_false_addition_voxels": float(background_false_addition_count),
        "background_false_addition_physical_measure": float(
            background_false_addition_count * pixel_measure
        ),
        "merged_existing_component_count": float(
            harmful_merged_existing_components
        ),
        "legal_same_lesion_reconnection_count": float(
            legal_same_lesion_reconnections
        ),
        "target_added_bridged_to_other_lesion_or_background": float(
            target_added_bridged_to_other_identity
        ),
        "new_target_bridged_to_m0": float(
            target_added_bridged_to_other_identity
        ),
        "unintended_bridge_or_merge": float(
            harmful_merged_existing_components > 0
            or target_added_bridged_to_other_identity
        ),
        "m0_voxels": float(m0_count),
        "m0_preserved_voxels": float(m0_count),
        "m0_removed_voxels": 0.0,
        "m0_preservation_rate": None if m0_count == 0 else 1.0,
        "m0_preservation_rate_defined": m0_count > 0,
        "m0_preservation_rate_denominator_voxels": float(m0_count),
        "operation_algebra_safety_pass": 1.0,
        "preserved_m0_loss": 0.0,
    }


def validate_patient_folds(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    patient_fold: Dict[str, int] = {}
    case_fold: Dict[str, int] = {}
    fold_cases: Dict[int, List[str]] = defaultdict(list)
    for row in rows:
        case_id = str(row.get("case_id") or "")
        patient_id = str(row.get("patient_id") or "")
        fold = row.get("held_out_fold")
        if not case_id or not patient_id or fold not in FOLDS:
            raise ContractError("each row needs case_id, patient_id and held_out_fold 0..4")
        if case_id in case_fold:
            raise ContractError("case %s appears more than once" % case_id)
        if patient_id in patient_fold and patient_fold[patient_id] != fold:
            raise ContractError("patient %s crosses held-out folds" % patient_id)
        patient_fold[patient_id] = int(fold)
        case_fold[case_id] = int(fold)
        fold_cases[int(fold)].append(case_id)
    if set(fold_cases) != set(FOLDS):
        raise ContractError("all five folds must be represented")
    return {
        "status": "PASS",
        "case_count": len(case_fold),
        "patient_count": len(patient_fold),
        "fold_case_counts": {str(k): len(fold_cases[k]) for k in FOLDS},
        "case_manifest_sha256": canonical_hash(case_fold),
    }


def plan_gpu_queues(
    *,
    running_folds: Sequence[int] = (),
    completed_folds: Sequence[int] = (),
    gpu_for_running: Mapping[int, int] = None,
) -> Dict[str, Any]:
    """Return a deterministic two-GPU plan without launching anything.

    The canonical fresh plan is GPU0: 0->2->4 and GPU1: 1->3.  When fold 0 is
    already running on GPU0, fold 1 may start immediately on GPU1; later folds
    stay serial within each GPU queue.
    """
    gpu_for_running = dict(gpu_for_running or {})
    running = set(int(v) for v in running_folds)
    completed = set(int(v) for v in completed_folds)
    if not running.issubset(FOLDS) or not completed.issubset(FOLDS):
        raise ContractError("folds must be 0..4")
    if running & completed:
        raise ContractError("a fold cannot be running and completed")
    for fold in running:
        if gpu_for_running.get(fold) not in (0, 1):
            raise ContractError("every running fold needs an assigned GPU")
    base = {0: [0, 2, 4], 1: [1, 3]}
    queues = {
        str(gpu): [f for f in folds if f not in running and f not in completed]
        for gpu, folds in base.items()
    }
    active = {str(gpu): None for gpu in (0, 1)}
    for fold, gpu in gpu_for_running.items():
        if fold in running:
            if active[str(gpu)] is not None:
                raise ContractError("one GPU cannot own two running folds")
            active[str(gpu)] = fold
    return {
        "status": "PLAN_ONLY",
        "active": active,
        "queues": queues,
        "completed": sorted(completed),
        "launch_performed": False,
    }


def condition_contract(condition: str) -> Dict[str, Any]:
    if condition not in EDITOR_CONDITIONS:
        raise ContractError("unknown editor condition: %s" % condition)
    table = {
        "visual_state_only": (False, "NULL", "separate_checkpoint"),
        "spatial_only": (True, "NULL", "separate_checkpoint"),
        "intent_only": (False, "CORRECT", "separate_checkpoint"),
        "scribble_plus_intent": (True, "CORRECT", "shared_structured_intent_checkpoint"),
        "scribble_plus_operation": (
            True,
            "OPERATION_ONLY",
            "shared_structured_intent_checkpoint",
        ),
        "same_weight_NULL": (True, "NULL", "shared_structured_intent_checkpoint"),
        "same_weight_wrong_scope": (True, "WRONG_SCOPE", "shared_structured_intent_checkpoint"),
        "same_weight_shuffled": (True, "SHUFFLED", "shared_structured_intent_checkpoint"),
        "oracle_slots": (True, "ORACLE", "shared_structured_intent_checkpoint"),
        "predicted_slots": (True, "PREDICTED", "shared_structured_intent_checkpoint"),
        "wrong_operation_OOD": (
            True,
            "WRONG_OPERATION_OOD",
            "shared_structured_intent_checkpoint",
        ),
    }
    use_scribble, intent, weights = table[condition]
    contract = {
        "condition": condition,
        "use_pet": True,
        "use_ct": True,
        "use_m0": True,
        "use_scribble": use_scribble,
        "intent": intent,
        "weight_scope": weights,
    }
    if condition == "visual_state_only":
        contract.update(
            {
                "roi_centering": "frozen_scribble_center",
                "model_visible_prompt": False,
                "interpretation": (
                    "PET5+CT5+central-M0 within a frozen-scribble-centered ROI; "
                    "scribble and intent are model-invisible, not prompt-free whole-image"
                ),
            }
        )
    if condition == "wrong_operation_OOD":
        contract.update(
            {
                "comparison_role": "safety_stress_only",
                "confirmatory_utility": False,
                "target_operation": "gold_cue_operation_not_flipped_token",
            }
        )
    return contract


def comparison_matrix() -> List[Dict[str, Any]]:
    rows = [condition_contract(name) for name in EDITOR_CONDITIONS]
    rows.extend(
        [
            {
                "condition": "official_autoPETV_4ch_nnunet",
                "status": "ADAPTER_ONLY",
                "prompt": "FG/BG heatmaps",
                "fairness": "retrain or evaluate separately; public PSMA-seen weights are non-headline",
            },
            {
                "condition": "ScribblePrompt",
                "status": "ADAPTER_ONLY",
                "prompt": "2D scribble",
                "fairness": "report 2D/pretraining/task mismatch; not a same-weight causal denominator",
            },
            {
                "condition": "SW-FastEdit_or_PRISM",
                "status": "ADAPTER_ONLY",
                "prompt": "interactive spatial prompt",
                "fairness": "use identical patient split and step budget when a compatible implementation exists",
            },
        ]
    )
    return rows


def patient_cluster_summary(rows: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    defined_episode_count = 0
    for row in rows:
        patient = str(row.get("patient_id") or "")
        value = row.get(metric)
        if not patient:
            raise ContractError("metric rows require patient_id")
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ContractError("metric rows require numeric/null %s" % metric)
        if not np.isfinite(float(value)):
            raise ContractError("metric rows forbid non-finite %s" % metric)
        grouped[patient].append(float(value))
        defined_episode_count += 1
    if not grouped:
        return {
            "defined": False,
            "episode_count": float(len(rows)),
            "defined_episode_count": 0.0,
            "patient_count": 0.0,
            "mean": None,
            "median": None,
            "std": None,
            "std_defined": False,
        }
    per_patient = np.asarray([np.mean(values) for values in grouped.values()], dtype=float)
    return {
        "defined": True,
        "episode_count": float(len(rows)),
        "defined_episode_count": float(defined_episode_count),
        "patient_count": float(len(per_patient)),
        "mean": float(per_patient.mean()),
        "median": float(np.median(per_patient)),
        "std": float(per_patient.std(ddof=1)) if len(per_patient) > 1 else None,
        "std_defined": len(per_patient) > 1,
    }
