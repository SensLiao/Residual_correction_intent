from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "common"))

from petct_route_a_core import (  # noqa: E402
    ContractError,
    EDITOR_TRAINING_CONDITIONS,
    comparison_matrix,
    condition_contract,
    correction_metrics,
    expected_editor_checkpoint_condition,
    plan_gpu_queues,
    residual_masks,
    validate_patient_folds,
)


def test_residual_masks_separate_fn_and_fp() -> None:
    gt = np.zeros((5, 5), dtype=np.uint8)
    m0 = np.zeros_like(gt)
    gt[1:3, 1:3] = 1
    m0[2:4, 2:4] = 1
    residual = residual_masks(gt, m0)
    assert residual["fn"].sum() == 3
    assert residual["fp"].sum() == 3


def test_patient_fold_contract_rejects_cross_fold_patient() -> None:
    rows = [
        {"case_id": "case-%d" % fold, "patient_id": "p-%d" % fold, "held_out_fold": fold}
        for fold in range(5)
    ]
    assert validate_patient_folds(rows)["case_count"] == 5
    rows.append({"case_id": "repeat", "patient_id": "p-0", "held_out_fold": 1})
    with pytest.raises(ContractError, match="crosses"):
        validate_patient_folds(rows)


def test_two_gpu_plan_starts_fold1_while_fold0_is_running() -> None:
    plan = plan_gpu_queues(running_folds=[0], gpu_for_running={0: 0})
    assert plan["active"] == {"0": 0, "1": None}
    assert plan["queues"]["0"] == [2, 4]
    assert plan["queues"]["1"] == [1, 3]
    assert plan["launch_performed"] is False


def test_condition_contract_distinguishes_scribble_and_intent() -> None:
    visual_state = condition_contract("visual_state_only")
    spatial = condition_contract("spatial_only")
    intent = condition_contract("intent_only")
    joint = condition_contract("scribble_plus_intent")
    add = condition_contract("scribble_plus_operation")
    assert visual_state == {
        "condition": "visual_state_only",
        "use_pet": True,
        "use_ct": True,
        "use_m0": True,
        "use_scribble": False,
        "intent": "NULL",
        "weight_scope": "separate_checkpoint",
        "roi_centering": "frozen_scribble_center",
        "model_visible_prompt": False,
        "interpretation": (
            "PET5+CT5+central-M0 within a frozen-scribble-centered ROI; "
            "scribble and intent are model-invisible, not prompt-free whole-image"
        ),
    }
    assert spatial["use_scribble"] is True and spatial["intent"] == "NULL"
    assert intent["use_scribble"] is False and intent["intent"] == "CORRECT"
    assert joint["use_scribble"] is True and joint["intent"] == "CORRECT"
    assert add["intent"] == "OPERATION_ONLY"
    assert add["weight_scope"] == joint["weight_scope"]
    assert add["weight_scope"] == "shared_structured_intent_checkpoint"
    assert len(comparison_matrix()) >= 12


def test_oracle_and_predicted_slots_reuse_the_rich_editor_checkpoint() -> None:
    assert EDITOR_TRAINING_CONDITIONS == (
        "visual_state_only",
        "spatial_only",
        "intent_only",
        "scribble_plus_intent",
    )
    assert expected_editor_checkpoint_condition("oracle_slots") == (
        "scribble_plus_intent"
    )
    assert expected_editor_checkpoint_condition("predicted_slots") == (
        "scribble_plus_intent"
    )
    assert condition_contract("oracle_slots")["weight_scope"] == (
        "shared_structured_intent_checkpoint"
    )
    assert condition_contract("predicted_slots")["weight_scope"] == (
        "shared_structured_intent_checkpoint"
    )
    config = json.loads(
        (PROJECT / "configs" / "petct_route_a_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(config["editor"]["training_conditions"]) == (
        EDITOR_TRAINING_CONDITIONS
    )


def test_add_only_metrics_reject_removed_m0() -> None:
    shape = (7, 7)
    gt = np.zeros(shape, dtype=np.uint8)
    m0 = np.zeros(shape, dtype=np.uint8)
    target = np.zeros(shape, dtype=np.uint8)
    scribble = np.zeros(shape, dtype=np.uint8)
    m0[1, 1] = 1
    gt[1, 1] = 1
    target[3:6, 3:6] = 1
    gt[target > 0] = 1
    scribble[4, 4] = 1
    m1 = np.maximum(m0, target)
    metrics = correction_metrics(
        gt=gt,
        m0=m0,
        m1=m1,
        operation="ADD",
        authorized_target=target,
        scribble=scribble,
        spacing_xyz=(1.0, 1.0),
        distal_radius_mm=1.0,
    )
    assert metrics["target_residual_recall"] == 1.0
    assert metrics["unauthorized_addition_ratio"] == 0.0
    broken = m1.copy()
    broken[1, 1] = 0
    with pytest.raises(ContractError, match="ADD operation violates"):
        correction_metrics(
            gt=gt,
            m0=m0,
            m1=broken,
            operation="ADD",
            authorized_target=target,
            scribble=scribble,
            spacing_xyz=(1.0, 1.0),
            distal_radius_mm=1.0,
        )


def test_remove_metrics_use_m0_minus_gt_and_preserve_true_positive() -> None:
    gt = np.zeros((9, 9), dtype=np.uint8)
    gt[2:5, 2:5] = 1
    m0 = gt.copy()
    m0[5:8, 2:5] = 1
    authorized = np.zeros_like(gt)
    authorized[6:8, 2:5] = 1
    cue = np.zeros_like(gt)
    cue[6, 3] = 1
    m1 = m0.copy()
    m1[authorized > 0] = 0
    metrics = correction_metrics(
        gt=gt,
        m0=m0,
        m1=m1,
        operation="REMOVE",
        authorized_target=authorized,
        scribble=cue,
        spacing_xyz=(1.0, 1.0),
        distal_radius_mm=1.0,
    )
    assert metrics["operation"] == "REMOVE"
    assert metrics["target_residual_recall"] == 1.0
    assert metrics["authorized_removal_voxels"] == float(authorized.sum())
    assert metrics["unauthorized_removal_voxels"] == 0.0
    assert metrics["true_positive_preservation_rate"] == 1.0


def test_same_gt_lesion_fragments_can_be_legally_reconnected() -> None:
    gt = np.zeros((7, 7), dtype=np.uint8)
    gt[3, 1:6] = 1
    m0 = np.zeros_like(gt)
    m0[3, 1:3] = 1
    m0[3, 4:6] = 1
    target = np.zeros_like(gt)
    target[3, 3] = 1
    metrics = correction_metrics(
        gt=gt,
        m0=m0,
        m1=m0 | target,
        operation="ADD",
        authorized_target=target,
        scribble=target,
        spacing_xyz=(1.0, 1.0),
        distal_radius_mm=0.0,
    )
    assert metrics["legal_same_lesion_reconnection_count"] == 1.0
    assert metrics["merged_existing_component_count"] == 0.0
    assert metrics["new_target_bridged_to_m0"] == 0.0
    assert metrics["unintended_bridge_or_merge"] == 0.0


def test_unauthorized_additions_split_scope_other_lesion_and_background() -> None:
    gt = np.zeros((9, 9), dtype=np.uint8)
    gt[2, 1:6] = 1
    gt[6, 5:7] = 1
    m0 = np.zeros_like(gt)
    m0[2, 1] = 1
    target = np.zeros_like(gt)
    target[2, 2] = 1
    m1 = m0 | target
    m1[2, 3] = 1  # Same target lesion, outside the authorized LOCAL scope.
    m1[6, 5] = 1  # A truly different GT lesion.
    m1[0, 0] = 1  # Background false addition.
    metrics = correction_metrics(
        gt=gt,
        m0=m0,
        m1=m1,
        operation="ADD",
        authorized_target=target,
        scribble=target,
        spacing_xyz=(2.0, 3.0),
        distal_radius_mm=0.0,
    )
    assert metrics["unauthorized_addition_voxels"] == 3.0
    assert metrics["target_lesion_scope_overreach_voxels"] == 1.0
    assert metrics["other_lesion_capture_voxels"] == 1.0
    assert metrics["background_false_addition_voxels"] == 1.0
    assert metrics["unauthorized_addition_physical_measure"] == 18.0
    assert metrics["target_lesion_scope_overreach_physical_measure"] == 6.0
    assert metrics["other_lesion_capture_physical_measure"] == 6.0


def test_no_addition_precision_and_empty_distal_recall_are_json_null() -> None:
    gt = np.zeros((5, 5), dtype=np.uint8)
    gt[2, 1:4] = 1
    m0 = np.zeros_like(gt)
    m0[2, 1] = 1
    target = np.zeros_like(gt)
    target[2, 2] = 1
    metrics = correction_metrics(
        gt=gt,
        m0=m0,
        m1=m0,
        operation="ADD",
        authorized_target=target,
        scribble=target,
        spacing_xyz=(1.0, 1.0),
        distal_radius_mm=1.0,
    )
    assert metrics["residual_precision"] is None
    assert metrics["residual_precision_defined"] is False
    assert metrics["residual_precision_denominator_voxels"] == 0.0
    assert metrics["prompt_distal_recall"] is None
    assert metrics["prompt_distal_recall_defined"] is False
    assert metrics["prompt_distal_recall_denominator_voxels"] == 0.0
    json.dumps(metrics, allow_nan=False)


def test_bridge_between_different_gt_lesions_is_harm() -> None:
    gt = np.zeros((7, 9), dtype=np.uint8)
    gt[3, 1:3] = 1
    gt[3, 5:7] = 1
    m0 = np.zeros_like(gt)
    m0[3, 1] = 1
    m0[3, 6] = 1
    target = np.zeros_like(gt)
    target[3, 2] = 1
    m1 = m0 | target
    m1[3, 3:6] = 1
    metrics = correction_metrics(
        gt=gt,
        m0=m0,
        m1=m1,
        operation="ADD",
        authorized_target=target,
        scribble=target,
        spacing_xyz=(1.0, 1.0),
        distal_radius_mm=0.0,
    )
    assert metrics["merged_existing_component_count"] == 1.0
    assert metrics["legal_same_lesion_reconnection_count"] == 0.0
    assert metrics["target_added_bridged_to_other_lesion_or_background"] == 1.0
    assert metrics["unintended_bridge_or_merge"] == 1.0
