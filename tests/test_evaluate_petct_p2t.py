from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "evaluation"))

from evaluate_petct_p2t import (  # noqa: E402
    P2T_RELEASE_PREDICTION_FIELDS,
    _per_goal_diagnostics,
    _per_strategy_diagnostics,
    _validate_prediction_rows_for_release,
    decode_legal_joint,
)


def test_decoder_emits_only_six_joint_defined_slot_tuples() -> None:
    joint_logits = torch.eye(6) * 10.0
    joint, decoded_operation, decoded_target, decoded_scope = decode_legal_joint(joint_logits)
    assert joint.tolist() == list(range(6))
    assert all(
        (r, s) in {(0, 0), (0, 1), (1, 1)}
        for r, s in zip(decoded_target.tolist(), decoded_scope.tolist())
    )
    assert decoded_operation.tolist() == [0, 1, 0, 1, 0, 1]


def test_evaluator_emits_per_strategy_and_per_goal_diagnostics() -> None:
    diagnostics = _per_strategy_diagnostics(
        ["centerline", "centerline", "boundary", "boundary"],
        ["p1", "p2", "p3", "p4"],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 4, 5],
        [0, 3, 4, 5],
    )
    assert set(diagnostics) == {"boundary", "centerline"}
    assert diagnostics["centerline"]["episode_count"] == 2
    assert diagnostics["boundary"]["joint_confusion_matrix"]["episode_count"] == 2

    per_goal = _per_goal_diagnostics([0, 1, 4, 5], [0, 3, 4, 5])
    assert per_goal["ADD_SAME_LOCAL"]["recall"] == 1.0
    assert per_goal["REMOVE_SAME_LOCAL"]["recall"] == 0.0
    assert per_goal["REMOVE_NEW_COMPLETE"]["gold_episode_count"] == 1


@pytest.mark.parametrize(
    "forbidden_field",
    ["gold_joint_id", "joint_true", "gt", "evaluation_npz_path"],
)
def test_release_prediction_allowlist_rejects_truth_and_evaluation_paths(
    forbidden_field: str,
) -> None:
    row = {field: None for field in P2T_RELEASE_PREDICTION_FIELDS}
    row[forbidden_field] = "must-not-leak"
    with pytest.raises(RuntimeError, match="allowlist mismatch"):
        _validate_prediction_rows_for_release([row])


def test_release_prediction_allowlist_accepts_only_exact_public_schema() -> None:
    row = {field: None for field in P2T_RELEASE_PREDICTION_FIELDS}
    _validate_prediction_rows_for_release([row])
