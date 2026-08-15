from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "data"))

from validate_petct_learning_split import main, validate_learning_split  # noqa: E402


def _partition(patient_index: int) -> str:
    if patient_index < 264:
        return "train"
    if patient_index < 321:
        return "val"
    return "test"


def _fixture():
    source = []
    for case_index in range(597):
        patient_index = case_index if case_index < 378 else case_index - 378
        source.append(
            {"case_id": "c%d" % case_index, "patient_id": "p%d" % patient_index}
        )
    patients = []
    case_counts = {"train": 0, "val": 0, "test": 0}
    for patient_index in range(378):
        partition = _partition(patient_index)
        case_ids = [
            row["case_id"]
            for row in source
            if row["patient_id"] == "p%d" % patient_index
        ]
        case_counts[partition] += len(case_ids)
        patients.append(
            {
                "patient_id": "p%d" % patient_index,
                "partition": partition,
                "case_ids": case_ids,
            }
        )
    target_patient_counts = {"train": 264, "val": 57, "test": 57}
    document = {
        "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
        "status": "FROZEN_BEFORE_MODEL_SELECTION",
        "dataset": "PSMA-PET-CT-Lesions-v3",
        "split_unit": "patient",
        "case_count": 597,
        "patient_count": 378,
        "algorithm": "stable-patient-hash-v1",
        "seed": 20260717,
        "target_patient_counts": target_patient_counts,
        "case_counts": case_counts,
        "patients": patients,
    }
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
                "target_patient_counts": target_patient_counts,
                "case_counts": "FROZEN_IN_GENERATED_SPLIT_RECEIPT",
            },
        }
    }
    return document, source, config


def test_frozen_split_exactly_joins_all_cases_and_records_receipt() -> None:
    document, source, config = _fixture()
    result = validate_learning_split(document, source, config)
    assert result["case_count"] == 597
    assert result["patient_counts"] == {"train": 264, "val": 57, "test": 57}
    assert result["case_counts"] == document["case_counts"]
    assert result["algorithm"] == "stable-patient-hash-v1"
    assert result["seed"] == 20260717


def test_frozen_split_rejects_case_patient_remap() -> None:
    document, source, config = _fixture()
    source[0]["patient_id"] = "wrong"
    with pytest.raises(RuntimeError, match="patient count|case-to-patient"):
        validate_learning_split(document, source, config)


def test_frozen_split_rejects_duplicate_source_case_before_set_join() -> None:
    document, source, config = _fixture()
    source.append(dict(source[0]))
    with pytest.raises(RuntimeError, match="duplicate case_id"):
        validate_learning_split(document, source, config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("algorithm", "unreceipted", "invalid algorithm"),
        ("seed", 7, "invalid seed"),
        (
            "target_patient_counts",
            {"train": 263, "val": 58, "test": 57},
            "invalid target_patient_counts",
        ),
        ("case_counts", {"train": 482, "val": 58, "test": 57}, "case_counts"),
    ],
)
def test_frozen_split_rejects_changed_receipt(key, value, message) -> None:
    document, source, config = _fixture()
    document[key] = value
    with pytest.raises(RuntimeError, match=message):
        validate_learning_split(document, source, config)


def test_cli_main_consumes_experiment_config_and_emits_exact_counts(
    tmp_path: Path, capsys
) -> None:
    document, source, config = _fixture()
    split_path = tmp_path / "split.json"
    cases_path = tmp_path / "cases.jsonl"
    config_path = tmp_path / "experiment.json"
    split_path.write_text(json.dumps(document), encoding="utf-8")
    cases_path.write_text(
        "".join(json.dumps(row) + "\n" for row in source), encoding="utf-8"
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert (
        main(
            [
                "--split",
                str(split_path),
                "--case-manifest",
                str(cases_path),
                "--experiment-config",
                str(config_path),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PASS"
    assert receipt["case_counts"] == document["case_counts"]
