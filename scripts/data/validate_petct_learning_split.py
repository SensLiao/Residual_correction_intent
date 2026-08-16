#!/usr/bin/env python3
"""Validate an immutable patient-level train/val/test split against all PSMA cases."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import sha256_file  # noqa: E402


SCHEMA_VERSION = "PETCT-LEARNING-SPLIT-v1.0"
PARTITIONS = ("train", "val", "test")
PARTITION_SET = set(PARTITIONS)
FROZEN_CASE_COUNT_RECEIPT = "FROZEN_IN_GENERATED_SPLIT_RECEIPT"


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("%s must be an object" % label)
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError("%s must be a positive integer" % label)
    return int(value)


def _partition_counts(value: Any, *, label: str) -> dict[str, int]:
    value = _mapping(value, label=label)
    if set(value) != PARTITION_SET:
        raise RuntimeError("%s must contain exactly train/val/test" % label)
    return {
        partition: _positive_integer(value[partition], label="%s.%s" % (label, partition))
        for partition in PARTITIONS
    }


def learning_split_contract_from_config(
    experiment_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the frozen split contract without accepting implicit defaults."""

    config = _mapping(experiment_config, label="experiment config")
    dataset = _mapping(config.get("dataset"), label="experiment config dataset")
    learning = _mapping(
        dataset.get("learning_split"), label="experiment config dataset.learning_split"
    )
    schema_version = learning.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise RuntimeError("experiment config learning split schema mismatch")
    algorithm = learning.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise RuntimeError("experiment config learning split algorithm is required")
    seed = learning.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RuntimeError("experiment config learning split seed must be an integer")
    target_patient_counts = _partition_counts(
        learning.get("target_patient_counts"),
        label="experiment config target_patient_counts",
    )
    case_count = _positive_integer(dataset.get("cases"), label="dataset.cases")
    patient_count = _positive_integer(dataset.get("patients"), label="dataset.patients")
    if sum(target_patient_counts.values()) != patient_count:
        raise RuntimeError("configured target patient counts do not sum to dataset.patients")
    if learning.get("case_counts") != FROZEN_CASE_COUNT_RECEIPT:
        raise RuntimeError(
            "experiment config case_counts must require the generated split receipt"
        )
    dataset_name = dataset.get("name")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise RuntimeError("dataset.name is required")
    split_unit = dataset.get("split_unit")
    if split_unit != "patient":
        raise RuntimeError("dataset.split_unit must be patient")
    return {
        "schema_version": schema_version,
        "dataset": dataset_name,
        "split_unit": split_unit,
        "case_count": case_count,
        "patient_count": patient_count,
        "algorithm": algorithm,
        "seed": int(seed),
        "target_patient_counts": target_patient_counts,
    }


def validate_learning_split(
    document: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    experiment_config: Mapping[str, Any],
) -> dict[str, Any]:
    contract = learning_split_contract_from_config(experiment_config)
    document = _mapping(document, label="learning split")
    expected_header = {
        "schema_version": contract["schema_version"],
        "status": "FROZEN_BEFORE_MODEL_SELECTION",
        "dataset": contract["dataset"],
        "split_unit": contract["split_unit"],
        "case_count": contract["case_count"],
        "patient_count": contract["patient_count"],
        "algorithm": contract["algorithm"],
        "seed": contract["seed"],
        "target_patient_counts": contract["target_patient_counts"],
    }
    for key, expected in expected_header.items():
        if document.get(key) != expected:
            raise RuntimeError("learning split has invalid %s" % key)

    source_case_ids: list[str] = []
    source: dict[str, Mapping[str, Any]] = {}
    source_patients: set[str] = set()
    for row_number, row in enumerate(source_rows, start=1):
        row = _mapping(row, label="source row %d" % row_number)
        raw_case_id, raw_patient_id = row.get("case_id"), row.get("patient_id")
        if raw_case_id is None or raw_patient_id is None:
            raise RuntimeError("source rows require case_id and patient_id")
        case_id = str(raw_case_id)
        patient_id = str(raw_patient_id).casefold()
        if not case_id or not patient_id:
            raise RuntimeError("source rows require non-empty case_id and patient_id")
        source_case_ids.append(case_id)
        if case_id in source:
            raise RuntimeError("source cohort contains duplicate case_id: %s" % case_id)
        source[case_id] = row
        source_patients.add(patient_id)
    if len(source) != contract["case_count"]:
        raise RuntimeError("source cohort case count differs from experiment config")
    if len(source_patients) != contract["patient_count"]:
        raise RuntimeError("source cohort patient count differs from experiment config")

    patient_rows = document.get("patients")
    if not isinstance(patient_rows, list) or len(patient_rows) != contract["patient_count"]:
        raise RuntimeError(
            "learning split must list exactly %d patients" % contract["patient_count"]
        )
    split_case_to_patient: dict[str, str] = {}
    split_case_to_partition: dict[str, str] = {}
    patients: set[str] = set()
    patient_counts: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()
    for row_number, raw_row in enumerate(patient_rows, start=1):
        row = _mapping(raw_row, label="split patient row %d" % row_number)
        patient = str(row.get("patient_id") or "").casefold()
        partition = str(row.get("partition") or "")
        raw_case_ids = row.get("case_ids")
        if not patient or patient in patients or partition not in PARTITION_SET:
            raise RuntimeError("invalid/duplicate patient or partition in learning split")
        if not isinstance(raw_case_ids, list) or not raw_case_ids:
            raise RuntimeError("split patient must list unique non-empty cases")
        case_ids = [str(case_id) for case_id in raw_case_ids]
        if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
            raise RuntimeError("split patient must list unique non-empty cases")
        patients.add(patient)
        patient_counts[partition] += 1
        case_counts[partition] += len(case_ids)
        for case_id in case_ids:
            if case_id in split_case_to_patient:
                raise RuntimeError("case appears more than once in learning split")
            split_case_to_patient[case_id] = patient
            split_case_to_partition[case_id] = partition

    computed_patient_counts = {
        partition: int(patient_counts[partition]) for partition in PARTITIONS
    }
    if computed_patient_counts != contract["target_patient_counts"]:
        raise RuntimeError("computed patient counts differ from frozen targets")
    computed_case_counts = {
        partition: int(case_counts[partition]) for partition in PARTITIONS
    }
    if sum(computed_case_counts.values()) != contract["case_count"]:
        raise RuntimeError("computed split case count differs from experiment config")
    if document.get("case_counts") != computed_case_counts:
        raise RuntimeError("learning split case_counts receipt differs from computed counts")
    if set(source) != set(split_case_to_patient):
        raise RuntimeError("learning split case inventory differs from source cohort")
    for case_id, row in source.items():
        if str(row["patient_id"]).casefold() != split_case_to_patient[case_id]:
            raise RuntimeError("learning split case-to-patient mapping differs")
    return {
        "case_to_partition": split_case_to_partition,
        "patient_count": len(patients),
        "case_count": len(source_case_ids),
        "algorithm": contract["algorithm"],
        "seed": contract["seed"],
        "target_patient_counts": dict(contract["target_patient_counts"]),
        "patient_counts": computed_patient_counts,
        "case_counts": computed_case_counts,
    }


def load_and_validate_learning_split(
    path: Path,
    source_rows: Sequence[Mapping[str, Any]],
    experiment_config: Mapping[str, Any],
):
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    result = validate_learning_split(document, source_rows, experiment_config)
    return document, {**result, "split_sha256": sha256_file(path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    args = parser.parse_args(argv)
    from common.petct_learning import load_jsonl

    with args.experiment_config.open("r", encoding="utf-8") as stream:
        experiment_config = json.load(stream)
    _, result = load_and_validate_learning_split(
        args.split,
        load_jsonl(args.case_manifest),
        experiment_config,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                **result,
                "experiment_config_sha256": sha256_file(args.experiment_config),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
