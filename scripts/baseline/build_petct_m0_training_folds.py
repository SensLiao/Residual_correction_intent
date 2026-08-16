#!/usr/bin/env python3
"""Build the clean M0 five-fold nnU-Net split over train+validation cases.

The locked test partition is physically excluded. Patients are assigned to one
validation fold by a deterministic salted rank and greedy case-count balance.
The output is nnU-Net's canonical ``list[fold]`` JSON structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FOLD_COUNT = 5
FOLD_SALT = "PETCT-M0-EXCLTEST-FOLD-v1|"


class SplitContractError(ValueError):
    """Raised when the learning split cannot produce an isolated five-fold set."""


def _rank(patient_id: str) -> str:
    return hashlib.sha256(f"{FOLD_SALT}{patient_id}".encode()).hexdigest()


def build_folds(
    split: dict[str, Any], *, expected_learning_cases: int = 506
) -> list[dict[str, list[str]]]:
    """Return deterministic patient-disjoint folds with locked test excluded."""
    rows = split.get("patients")
    if not isinstance(rows, list) or not rows:
        raise SplitContractError("learning split must contain a non-empty patients list")

    learning_patients: list[tuple[str, list[str]]] = []
    test_cases: set[str] = set()
    all_cases: set[str] = set()
    patient_ids: set[str] = set()

    for row in rows:
        patient_id = row.get("patient_id")
        partition = row.get("partition")
        case_ids = row.get("case_ids")
        if not isinstance(patient_id, str) or not patient_id:
            raise SplitContractError("every patient requires a non-empty patient_id")
        if patient_id in patient_ids:
            raise SplitContractError(f"duplicate patient_id: {patient_id}")
        patient_ids.add(patient_id)
        if partition not in {"train", "val", "test"}:
            raise SplitContractError(f"invalid partition for {patient_id}: {partition}")
        if not isinstance(case_ids, list) or not case_ids:
            raise SplitContractError(f"patient {patient_id} has no case_ids")
        if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
            raise SplitContractError(f"patient {patient_id} has an invalid case_id")
        duplicates = all_cases.intersection(case_ids)
        if duplicates or len(case_ids) != len(set(case_ids)):
            duplicate = sorted(duplicates or set(case_ids))[0]
            raise SplitContractError(f"case_id is not globally unique: {duplicate}")
        all_cases.update(case_ids)

        if partition == "test":
            test_cases.update(case_ids)
        else:
            learning_patients.append((patient_id, list(case_ids)))

    learning_case_count = sum(len(case_ids) for _, case_ids in learning_patients)
    if learning_case_count != expected_learning_cases:
        raise SplitContractError(
            "expected "
            f"{expected_learning_cases} train+val cases, got {learning_case_count}"
        )

    learning_patients.sort(key=lambda item: _rank(item[0]))
    fold_patients: list[list[tuple[str, list[str]]]] = [
        [] for _ in range(FOLD_COUNT)
    ]
    fold_case_counts = [0] * FOLD_COUNT
    for patient_id, case_ids in learning_patients:
        fold = min(range(FOLD_COUNT), key=lambda index: fold_case_counts[index])
        fold_patients[fold].append((patient_id, case_ids))
        fold_case_counts[fold] += len(case_ids)

    learning_cases = {
        case_id for _, case_ids in learning_patients for case_id in case_ids
    }
    folds: list[dict[str, list[str]]] = []
    validation_occurrences: list[str] = []
    for fold in range(FOLD_COUNT):
        val_cases = {
            case_id
            for _, case_ids in fold_patients[fold]
            for case_id in case_ids
        }
        train_cases = learning_cases - val_cases
        if train_cases & val_cases:
            raise SplitContractError(f"fold {fold} train/validation overlap")
        if train_cases | val_cases != learning_cases:
            raise SplitContractError(f"fold {fold} does not cover the learning cases")
        if (train_cases | val_cases) & test_cases:
            raise SplitContractError(f"fold {fold} contains a locked test case")
        validation_occurrences.extend(val_cases)
        folds.append({"train": sorted(train_cases), "val": sorted(val_cases)})

    if set(validation_occurrences) != learning_cases:
        raise SplitContractError("validation folds do not cover every learning case")
    if len(validation_occurrences) != len(set(validation_occurrences)):
        raise SplitContractError("a learning case appears in multiple validation folds")
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-learning-cases", type=int, default=506)
    args = parser.parse_args()

    split = json.loads(args.learning_split.read_text(encoding="utf-8"))
    folds = build_folds(
        split, expected_learning_cases=args.expected_learning_cases
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(folds, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "FOLDS_OK "
        f"cases={args.expected_learning_cases} "
        f"train_per_fold={[len(fold['train']) for fold in folds]} "
        f"val_per_fold={[len(fold['val']) for fold in folds]} "
        "test_never_touched=1",
        flush=True,
    )


if __name__ == "__main__":
    main()
