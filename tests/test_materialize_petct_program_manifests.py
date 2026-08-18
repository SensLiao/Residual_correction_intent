from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for directory in (SCRIPTS, SCRIPTS / "data"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import materialize_petct_program_manifests as materializer  # noqa: E402
from common.petct_program_learning import LearningContractError  # noqa: E402


def _rich_row():
    return {
        "episode_id": "opaque-state-cue-token",
        "case_id": "case-1",
        "patient_id": "patient-1",
        "partition": "train",
        "goal": "ADD_SAME_LOCAL",
        "operation": "ADD",
        "matched_state_group_id": "audit-group",
        "visible_npz": "/visible/opaque-state-cue-token.npz",
        "visible_sha256": "a" * 64,
        "evaluation_npz": "/evaluation/opaque-state-cue-token.npz",
        "evaluation_sha256": "b" * 64,
        "learning_split_sha256": "c" * 64,
        "source_evaluation": {"gt_path": "/private/gt.nii.gz"},
    }


def test_physical_inference_manifest_has_only_allowlisted_visible_fields():
    inference, labels, audit = materializer.split_rows([_rich_row()])
    assert set(inference[0]) == {
        "schema_version", "episode_id", "partition", "operation",
        "visible_npz", "visible_sha256",
    }
    serialized = str(inference[0]).casefold()
    assert "patient" not in serialized
    assert "goal" not in serialized
    assert "evaluation" not in serialized
    assert labels[0]["goal"] == "ADD_SAME_LOCAL"
    assert audit[0]["source_record"]["source_evaluation"]["gt_path"]


def test_natural_rich_row_without_matched_group_splits_cleanly():
    row = _rich_row()
    row.pop("matched_state_group_id")
    inference, labels, audit = materializer.split_rows([row])
    assert "matched_state_group_id" not in labels[0]
    assert labels[0]["episode_id"] == row["episode_id"]
    assert "patient" not in str(inference[0]).casefold()


def test_controlled_rich_row_still_carries_matched_group_id():
    inference, labels, audit = materializer.split_rows([_rich_row()])
    assert labels[0]["matched_state_group_id"] == "audit-group"


def test_label_derived_visible_path_fails_closed():
    row = _rich_row()
    row["visible_npz"] = "/visible/ADD_SAME_LOCAL/m0.npz"
    with pytest.raises(LearningContractError, match="label-derived"):
        materializer.split_rows([row])


def test_locked_test_row_is_never_published_by_generic_materializer():
    row = _rich_row()
    row["partition"] = "test"
    with pytest.raises(LearningContractError, match="locked test"):
        materializer.split_rows([row])
