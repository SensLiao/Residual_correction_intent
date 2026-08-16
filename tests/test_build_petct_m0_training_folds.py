from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "baseline"
sys.path.insert(0, str(SCRIPT_DIR))

from build_petct_m0_training_folds import (  # noqa: E402
    SplitContractError,
    build_folds,
)


def _split() -> dict:
    patients = [
        {
            "patient_id": f"P{index:02d}",
            "partition": "train" if index < 7 else "val",
            "case_ids": [f"CASE_{index:02d}"],
        }
        for index in range(10)
    ]
    patients.append(
        {
            "patient_id": "LOCKED",
            "partition": "test",
            "case_ids": ["TEST_A", "TEST_B"],
        }
    )
    return {"patients": patients}


def test_builds_nnunet_list_with_patient_disjoint_test_free_folds() -> None:
    folds = build_folds(_split(), expected_learning_cases=10)

    assert isinstance(folds, list)
    assert len(folds) == 5
    learning = {f"CASE_{index:02d}" for index in range(10)}
    validation = [case_id for fold in folds for case_id in fold["val"]]
    assert set(validation) == learning
    assert len(validation) == len(set(validation))
    for fold in folds:
        train = set(fold["train"])
        val = set(fold["val"])
        assert not train & val
        assert train | val == learning
        assert not {"TEST_A", "TEST_B"} & (train | val)


def test_fold_assignment_is_deterministic_across_input_order() -> None:
    original = _split()
    reversed_rows = deepcopy(original)
    reversed_rows["patients"].reverse()

    assert build_folds(original, expected_learning_cases=10) == build_folds(
        reversed_rows, expected_learning_cases=10
    )


def test_rejects_duplicate_case_and_wrong_learning_count() -> None:
    duplicate = _split()
    duplicate["patients"][1]["case_ids"] = ["CASE_00"]
    with pytest.raises(SplitContractError, match="globally unique"):
        build_folds(duplicate, expected_learning_cases=10)

    with pytest.raises(SplitContractError, match="expected 11"):
        build_folds(_split(), expected_learning_cases=11)
