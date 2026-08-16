#!/usr/bin/env python3
"""Build the frozen patient-level 264/57/57 learning split from the case manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPTS = Path(__file__).resolve().parents[1]
for directory in (SCRIPTS, SCRIPTS / "common", SCRIPTS / "data"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common.petct_learning import load_jsonl, sha256_file  # noqa: E402
from data.validate_petct_learning_split import (  # noqa: E402
    PARTITIONS,
    learning_split_contract_from_config,
    validate_learning_split,
)


def _patient_rank(patient_id: str, seed: int) -> tuple[str, str]:
    normalized = patient_id.casefold()
    digest = hashlib.sha256(
        f"PETCT-STABLE-PATIENT-HASH-v1|{seed}|{normalized}".encode("utf-8")
    ).hexdigest()
    return digest, normalized


def build_learning_split(
    source_rows: Sequence[Mapping[str, Any]],
    experiment_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Assign whole patients by a deterministic hash ranking and validate the result."""

    contract = learning_split_contract_from_config(experiment_config)
    patient_cases: dict[str, list[str]] = defaultdict(list)
    seen_cases: set[str] = set()
    for index, row in enumerate(source_rows, start=1):
        if not isinstance(row, Mapping):
            raise RuntimeError(f"source row {index} must be an object")
        case_id = str(row.get("case_id") or "")
        patient_id = str(row.get("patient_id") or "").casefold()
        if not case_id or not patient_id:
            raise RuntimeError("source rows require non-empty case_id and patient_id")
        if case_id in seen_cases:
            raise RuntimeError(f"source cohort contains duplicate case_id: {case_id}")
        seen_cases.add(case_id)
        patient_cases[patient_id].append(case_id)

    if len(seen_cases) != contract["case_count"]:
        raise RuntimeError("source cohort case count differs from experiment config")
    if len(patient_cases) != contract["patient_count"]:
        raise RuntimeError("source cohort patient count differs from experiment config")

    ordered_patients = sorted(
        patient_cases,
        key=lambda patient_id: _patient_rank(patient_id, contract["seed"]),
    )
    boundaries: dict[str, tuple[int, int]] = {}
    start = 0
    for partition in PARTITIONS:
        end = start + contract["target_patient_counts"][partition]
        boundaries[partition] = (start, end)
        start = end
    if start != len(ordered_patients):
        raise RuntimeError("frozen partition counts do not cover all patients")

    assigned: dict[str, str] = {}
    for partition, (begin, end) in boundaries.items():
        for patient_id in ordered_patients[begin:end]:
            assigned[patient_id] = partition

    case_counts = {partition: 0 for partition in PARTITIONS}
    patients = []
    for patient_id in ordered_patients:
        partition = assigned[patient_id]
        case_ids = sorted(patient_cases[patient_id])
        case_counts[partition] += len(case_ids)
        patients.append(
            {
                "patient_id": patient_id,
                "partition": partition,
                "case_ids": case_ids,
                "rank_sha256": _patient_rank(patient_id, contract["seed"])[0],
            }
        )

    document = {
        "schema_version": contract["schema_version"],
        "status": "FROZEN_BEFORE_MODEL_SELECTION",
        "dataset": contract["dataset"],
        "split_unit": "patient",
        "case_count": contract["case_count"],
        "patient_count": contract["patient_count"],
        "algorithm": contract["algorithm"],
        "algorithm_definition": (
            "ascending SHA256('PETCT-STABLE-PATIENT-HASH-v1|seed|casefold(patient_id)'); "
            "patient_id tie-break; contiguous train/val/test quotas"
        ),
        "seed": contract["seed"],
        "target_patient_counts": dict(contract["target_patient_counts"]),
        "case_counts": case_counts,
        "patients": patients,
    }
    validate_learning_split(document, source_rows, experiment_config)
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    with args.experiment_config.open("r", encoding="utf-8") as stream:
        experiment_config = json.load(stream)
    source_rows = load_jsonl(args.case_manifest)
    document = build_learning_split(source_rows, experiment_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_MODEL_SELECTION",
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "case_manifest_sha256": sha256_file(args.case_manifest),
                "experiment_config_sha256": sha256_file(args.experiment_config),
                "patient_counts": document["target_patient_counts"],
                "case_counts": document["case_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
