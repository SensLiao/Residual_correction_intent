from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for directory in (SCRIPTS, SCRIPTS / "evaluation", SCRIPTS / "common"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common.petct_program_contract import (  # noqa: E402
    NEW_CUE_SENTINEL,
    protected_refs_policy,
)
from evaluation import evaluate_petct_program_v3 as evaluator  # noqa: E402


class _StubValidator:
    def is_valid(self, payload) -> bool:
        return True


def _natural_label(episode_id: str, patient_id: str, goal: str, operation: str):
    return {
        "episode_id": episode_id,
        "partition": "val",
        "patient_id": patient_id,
        "goal": goal,
        "operation": operation,
    }


def _prediction(episode_id: str, operation: str, family: str, operand: str, goal: str):
    return {
        "episode_id": episode_id,
        "decision": "PREDICT",
        "operation": operation,
        "family": family,
        "operand": operand,
        "goal": goal,
        "protected_refs": dict(protected_refs_policy(operation, operand)),
    }


def test_compiler_metrics_accepts_natural_labels_without_matched_groups() -> None:
    labels = [
        _natural_label("ep-a", "patient-a", "ADD_NEW_COMPLETE", "ADD"),
        _natural_label("ep-b", "patient-b", "REMOVE_NEW_COMPLETE", "REMOVE"),
    ]
    predictions = [
        _prediction("ep-a", "ADD", "CREATE_NEW", NEW_CUE_SENTINEL, "ADD_NEW_COMPLETE"),
        _prediction("ep-b", "REMOVE", "DELETE_COMPONENT", "comp-1", "REMOVE_NEW_COMPLETE"),
    ]
    candidates = {
        "ep-a": {"components": []},
        "ep-b": {
            "components": [{"component_key": "comp-1"}],
            "cue_hit_component_position": 0,
        },
    }
    metrics = evaluator._compiler_metrics(
        predictions, labels, candidates, {}, _StubValidator(), partition="val"
    )
    assert metrics["expected_episodes"] == 2
    assert metrics["legal_call_rate"] == 1.0
    assert metrics["matched_pair_count"] == 0
    assert metrics["matched_pair_joint_accuracy"] is None
    assert metrics["matched_triplet_count"] == 0
    assert metrics["matched_triplet_exact_accuracy"] is None
