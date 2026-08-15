from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for directory in (SCRIPTS, SCRIPTS / "common", SCRIPTS / "data"):
    sys.path.insert(0, str(directory))

from build_petct_learning_split import build_learning_split  # noqa: E402
from validate_petct_learning_split import validate_learning_split  # noqa: E402


def _fixture():
    rows = []
    for case_index in range(597):
        patient_index = case_index if case_index < 378 else case_index - 378
        rows.append(
            {"case_id": f"case-{case_index:03d}", "patient_id": f"patient-{patient_index:03d}"}
        )
    config = {
        "dataset": {
            "name": "PSMA-PET-CT-Lesions-v3",
            "cases": 597,
            "patients": 378,
            "split_unit": "patient",
            "learning_split": {
                "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
                "algorithm": "stable-patient-hash-v1",
                "seed": 20260717,
                "target_patient_counts": {"train": 264, "val": 57, "test": 57},
                "case_counts": "FROZEN_IN_GENERATED_SPLIT_RECEIPT",
            },
        }
    }
    return rows, config


def test_builds_exact_patient_level_264_57_57_split() -> None:
    rows, config = _fixture()
    document = build_learning_split(rows, config)
    result = validate_learning_split(document, rows, config)

    assert result["patient_counts"] == {"train": 264, "val": 57, "test": 57}
    assert sum(result["case_counts"].values()) == 597
    assert len({row["patient_id"] for row in document["patients"]}) == 378
    assert all(row["case_ids"] for row in document["patients"])


def test_split_is_invariant_to_case_manifest_order() -> None:
    rows, config = _fixture()
    shuffled = list(rows)
    random.Random(91).shuffle(shuffled)

    first = build_learning_split(rows, config)
    second = build_learning_split(shuffled, config)

    assert first == second


def test_duplicate_case_fails_closed() -> None:
    rows, config = _fixture()
    rows[-1] = dict(rows[0])

    with pytest.raises(RuntimeError, match="duplicate case_id"):
        build_learning_split(rows, config)
