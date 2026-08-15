from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for directory in (SCRIPTS, SCRIPTS / "common", SCRIPTS / "data"):
    sys.path.insert(0, str(directory))

from audit_psma_v3_dataset import AUDIT_VERSION  # noqa: E402
import build_petct_source_case_manifest as source_builder  # noqa: E402
from build_petct_source_case_manifest import (  # noqa: E402
    IDENTITY_STATE,
    LOCKED_STATE,
    MATERIALIZED_STATE,
    build_identity_case_rows,
    materialize_source_case_rows,
    publish_jsonl,
    validate_materialized_source_case_rows,
)


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fixture(tmp_path: Path):
    root = (tmp_path / "PSMA_v3").resolve()
    (root / "imagesTr").mkdir(parents=True)
    (root / "labelsTr").mkdir()
    cases = {
        "patient_a_exam1": "patient_a",
        "patient_a_exam2": "patient_a",
        "patient_b_exam1": "patient_b",
        "patient_c_exam1": "patient_c",
    }
    shape = (3, 4, 5)
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    for index, case_id in enumerate(cases):
        ct = np.full(shape, index, dtype=np.float32)
        pet = np.full(shape, index + 1, dtype=np.float32)
        gt = np.zeros(shape, dtype=np.uint8)
        gt[1, 1, 1] = index % 2
        nib.save(nib.Nifti1Image(ct, affine), root / "imagesTr" / f"{case_id}_0000.nii.gz")
        nib.save(nib.Nifti1Image(pet, affine), root / "imagesTr" / f"{case_id}_0001.nii.gz")
        nib.save(nib.Nifti1Image(gt, affine), root / "labelsTr" / f"{case_id}.nii.gz")
    fold0_val = ["patient_a_exam1", "patient_a_exam2"]
    fold1_val = ["patient_b_exam1", "patient_c_exam1"]
    all_cases = set(cases)
    splits = [
        {"train": sorted(all_cases - set(fold0_val)), "val": fold0_val},
        {"train": sorted(all_cases - set(fold1_val)), "val": fold1_val},
    ]
    splits_sha = _digest(splits)
    audit = {
        "status": "PASS",
        "audit_version": AUDIT_VERSION,
        "dataset_root": str(root),
        "errors": [],
        "source_hashes": {"splits_final_sha256": splits_sha},
        "summary": {
            "case_count": 4,
            "patient_count": 3,
            "failed_case_count": 0,
            "unreadable_label_count": 0,
            "invalid_label_count": 0,
        },
        "per_case": [
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "status": "PASS",
                "errors": [],
                "shape": list(shape),
            }
            for case_id, patient_id in cases.items()
        ],
    }
    return root, splits, splits_sha, audit


def _learning_split() -> dict:
    return {
        "status": "FROZEN_BEFORE_MODEL_SELECTION",
        "patients": [
            {
                "patient_id": "patient_a",
                "partition": "train",
                "case_ids": ["patient_a_exam1", "patient_a_exam2"],
            },
            {
                "patient_id": "patient_b",
                "partition": "val",
                "case_ids": ["patient_b_exam1"],
            },
            {
                "patient_id": "patient_c",
                "partition": "test",
                "case_ids": ["patient_c_exam1"],
            },
        ],
    }


def test_builds_identity_only_rows_without_opening_or_hashing_nifti(
    tmp_path: Path, monkeypatch
) -> None:
    root, splits, splits_sha, audit = _fixture(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("identity stage opened or hashed a NIfTI leaf")

    monkeypatch.setattr(source_builder.nib, "load", forbidden)
    monkeypatch.setattr(source_builder, "sha256_file", forbidden)

    rows, inventory_sha = build_identity_case_rows(
        root,
        splits,
        audit,
        splits_sha256=splits_sha,
        expected_cases=4,
        expected_patients=3,
        expected_folds=2,
    )

    assert [row["case_id"] for row in rows] == sorted(row["case_id"] for row in rows)
    assert {row["held_out_fold"] for row in rows} == {0, 1}
    assert {row["patient_id"] for row in rows} == {"patient_a", "patient_b", "patient_c"}
    assert all(
        set(row)
        == {
            "case_id",
            "patient_id",
            "held_out_fold",
            "ct_path",
            "pet_path",
            "gt_path",
            "truth_materialization",
        }
        for row in rows
    )
    assert {row["truth_materialization"] for row in rows} == {IDENTITY_STATE}
    assert len(inventory_sha) == 64
    assert all(Path(row["ct_path"]).name.endswith("_0000.nii.gz") for row in rows)
    assert all(Path(row["pet_path"]).name.endswith("_0001.nii.gz") for row in rows)
    assert all(Path(row["gt_path"]).parent.name == "labelsTr" for row in rows)


def test_val_scoped_materialization_never_opens_unselected_test_gt(
    tmp_path: Path, monkeypatch
) -> None:
    root, splits, splits_sha, audit = _fixture(tmp_path)
    identity, _ = build_identity_case_rows(
        root,
        splits,
        audit,
        splits_sha256=splits_sha,
        expected_cases=4,
        expected_patients=3,
        expected_folds=2,
    )
    test_gt = (root / "labelsTr" / "patient_c_exam1.nii.gz").resolve()
    original_load = source_builder.nib.load

    def guarded_load(path, *args, **kwargs):
        if Path(path).resolve() == test_gt:
            raise AssertionError("locked test GT was opened")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(source_builder.nib, "load", guarded_load)
    rows, inventory_sha = materialize_source_case_rows(
        identity,
        _learning_split(),
        authorized_partitions=("val",),
        expected_cases=4,
        expected_patients=3,
        expected_folds=2,
    )

    assert len(inventory_sha) == 64
    assert len(rows) == 4
    selected = next(row for row in rows if row["case_id"] == "patient_b_exam1")
    locked_test = next(row for row in rows if row["case_id"] == "patient_c_exam1")
    assert selected["partition"] == "val"
    assert selected["truth_materialization"] == MATERIALIZED_STATE
    assert selected["gt_sha256"] == hashlib.sha256(
        Path(selected["gt_path"]).read_bytes()
    ).hexdigest()
    assert locked_test["partition"] == "test"
    assert locked_test["truth_materialization"] == LOCKED_STATE
    assert not any(key.endswith(("_sha256", "_bytes")) for key in locked_test)


def test_selected_gt_content_drift_fails_closed(tmp_path: Path) -> None:
    root, splits, splits_sha, audit = _fixture(tmp_path)
    identity, _ = build_identity_case_rows(
        root,
        splits,
        audit,
        splits_sha256=splits_sha,
        expected_cases=4,
        expected_patients=3,
        expected_folds=2,
    )
    rows, _ = materialize_source_case_rows(
        identity,
        _learning_split(),
        authorized_partitions=("val",),
        expected_cases=4,
        expected_patients=3,
        expected_folds=2,
    )
    gt_path = root / "labelsTr" / "patient_b_exam1.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.uint8), np.diag([2.0, 2.0, 3.0, 1.0])),
        gt_path,
    )

    with pytest.raises(RuntimeError, match=r"selected GT .* drift"):
        validate_materialized_source_case_rows(
            rows,
            authorized_partitions=("val",),
            expected_cases=4,
            expected_patients=3,
            expected_folds=2,
        )


def test_rejects_case_missing_or_repeated_in_validation_folds(tmp_path: Path) -> None:
    root, splits, splits_sha, audit = _fixture(tmp_path)
    broken = copy.deepcopy(splits)
    broken[1]["val"].remove("patient_c_exam1")
    broken[1]["train"].append("patient_c_exam1")

    with pytest.raises(RuntimeError, match="exactly one validation fold"):
        build_identity_case_rows(
            root,
            broken,
            audit,
            splits_sha256=splits_sha,
            expected_cases=4,
            expected_patients=3,
            expected_folds=2,
        )


def test_rejects_patient_split_across_validation_folds(tmp_path: Path) -> None:
    root, splits, splits_sha, audit = _fixture(tmp_path)
    broken = copy.deepcopy(splits)
    broken[0]["val"] = ["patient_a_exam1"]
    broken[0]["train"] = sorted(set(row for row in broken[0]["train"]) | {"patient_a_exam2"})
    broken[1]["val"] = ["patient_a_exam2", "patient_b_exam1", "patient_c_exam1"]
    broken[1]["train"] = ["patient_a_exam1"]

    with pytest.raises(RuntimeError, match="patient leakage"):
        build_identity_case_rows(
            root,
            broken,
            audit,
            splits_sha256=splits_sha,
            expected_cases=4,
            expected_patients=3,
            expected_folds=2,
        )


def test_identity_stage_does_not_materialize_historical_audit_shape(tmp_path: Path) -> None:
    root, splits, splits_sha, audit = _fixture(tmp_path)
    audit["per_case"][0]["shape"] = [9, 9, 9]

    rows, _ = build_identity_case_rows(
        root,
        splits,
        audit,
        splits_sha256=splits_sha,
        expected_cases=4,
        expected_patients=3,
        expected_folds=2,
    )
    assert len(rows) == 4
    assert all("nifti_shape" not in row for row in rows)


def test_manifest_publication_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "source_cases.jsonl"
    rows = [
        {
            "case_id": "case",
            "patient_id": "patient",
            "held_out_fold": 0,
            "ct_path": "/ct",
            "pet_path": "/pet",
            "gt_path": "/gt",
        }
    ]

    first_hash = publish_jsonl(rows, output)

    assert len(first_hash) == 64
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_jsonl(rows, output)
