from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "data"))

from audit_psma_v3_dataset import audit_dataset, inspect_case, write_outputs  # noqa: E402


def _save(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), path)


def _make_case(root: Path, case: str, *, positive: bool, affine=None) -> None:
    shape = (8, 7, 6)
    ct = np.linspace(-1000, 500, np.prod(shape), dtype=np.float32).reshape(shape)
    pet = np.linspace(0, 12, np.prod(shape), dtype=np.float32).reshape(shape)
    label = np.zeros(shape, dtype=np.uint8)
    if positive:
        label[2:4, 2:5, 1:3] = 1
    _save(root / "imagesTr" / f"{case}_0000.nii.gz", ct, affine)
    _save(root / "imagesTr" / f"{case}_0001.nii.gz", pet, affine)
    _save(root / "labelsTr" / f"{case}.nii.gz", label, affine)


def _make_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "PSMA-PET-CT-Lesions_v3"
    cases = [
        "psma_patienta_2020-01-01",
        "psma_patienta_2021-01-01",
        "psma_patientb_2020-06-01",
    ]
    _make_case(root, cases[0], positive=True)
    _make_case(root, cases[1], positive=False)
    _make_case(root, cases[2], positive=True)
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": {"0": "CT", "1": "CT"},
                "labels": {"background": 0, "tumor": 1},
                "numTraining": 3,
                "file_ending": ".nii.gz",
            }
        ),
        encoding="utf-8",
    )
    (root / "splits_final.json").write_text(
        json.dumps(
            [
                {"train": [cases[2]], "val": cases[:2]},
                {"train": cases[:2], "val": [cases[2]]},
            ]
        ),
        encoding="utf-8",
    )
    with (root / "psma_metadata.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Subject ID",
                "Study Date",
                "age",
                "manufacturer_model_name",
                "pet_radionuclide",
                "ct_contrast_agent",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "Subject ID": "PSMA_patienta",
                "Study Date": "2020-01-01",
                "age": "70",
                "manufacturer_model_name": "scanner-a",
                "pet_radionuclide": "18F",
                "ct_contrast_agent": "yes",
            }
        )
        w.writerow(
            {
                "Subject ID": "PSMA_patienta",
                "Study Date": "2021-01-01",
                "age": "71",
                "manufacturer_model_name": "scanner-a",
                "pet_radionuclide": "18F",
                "ct_contrast_agent": "yes",
            }
        )
        w.writerow(
            {
                "Subject ID": "PSMA_patientb",
                "Study Date": "2020-06-01",
                "age": "66",
                "manufacturer_model_name": "scanner-b",
                "pet_radionuclide": "68Ga",
                "ct_contrast_agent": "no",
            }
        )
    return root


def test_full_audit_counts_cases_patients_empty_masks_and_split(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    report = audit_dataset(
        root,
        workers=1,
        expected_cases=3,
        expected_patients=2,
        expected_empty=1,
        expected_folds=2,
    )

    assert report["status"] == "PASS"
    assert report["summary"]["case_count"] == 3
    assert report["summary"]["patient_count"] == 2
    assert report["summary"]["empty_label_count"] == 1
    assert report["summary"]["positive_label_count"] == 2
    assert report["summary"]["component_count"] == 2
    assert report["split_audit"]["patient_overlap_total"] == 0
    assert report["split_audit"]["patients_in_multiple_val_folds"] == 0
    assert report["split_audit"]["val_cases_not_exactly_once"] == 0
    assert report["dataset_json"]["channel_names"] == {"0": "CT", "1": "CT"}


def test_case_audit_flags_geometry_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    case = "psma_patientc_2020-01-01"
    _make_case(root, case, positive=True)
    bad_affine = np.eye(4)
    bad_affine[0, 3] = 3.0
    pet = np.ones((8, 7, 6), dtype=np.float32)
    _save(root / "imagesTr" / f"{case}_0001.nii.gz", pet, bad_affine)

    result = inspect_case(
        case,
        root / "imagesTr" / f"{case}_0000.nii.gz",
        root / "imagesTr" / f"{case}_0001.nii.gz",
        root / "labelsTr" / f"{case}.nii.gz",
    )

    assert result["status"] == "FAIL"
    assert "affine_mismatch" in result["errors"]


def test_micro_spacing_delta_within_official_tolerance_passes(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    case = "psma_patientc_2020-01-01"
    affine = np.diag([1.0, 1.0, 3.27, 1.0])
    _make_case(root, case, positive=True, affine=affine)
    pet_affine = affine.copy()
    pet_affine[2, 2] = 3.27002
    pet = np.ones((8, 7, 6), dtype=np.float32)
    _save(root / "imagesTr" / f"{case}_0001.nii.gz", pet, pet_affine)

    result = inspect_case(
        case,
        root / "imagesTr" / f"{case}_0000.nii.gz",
        root / "imagesTr" / f"{case}_0001.nii.gz",
        root / "labelsTr" / f"{case}.nii.gz",
    )

    assert result["status"] == "PASS"
    assert "spacing_mismatch" not in result["errors"]


def test_material_spacing_delta_outside_official_tolerance_fails(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    case = "psma_patientc_2020-01-01"
    affine = np.diag([1.0, 1.0, 3.27, 1.0])
    _make_case(root, case, positive=True, affine=affine)
    pet_affine = affine.copy()
    pet_affine[2, 2] = 3.272
    pet = np.ones((8, 7, 6), dtype=np.float32)
    _save(root / "imagesTr" / f"{case}_0001.nii.gz", pet, pet_affine)

    result = inspect_case(
        case,
        root / "imagesTr" / f"{case}_0000.nii.gz",
        root / "imagesTr" / f"{case}_0001.nii.gz",
        root / "labelsTr" / f"{case}.nii.gz",
    )

    assert result["status"] == "FAIL"
    assert "spacing_mismatch" in result["errors"]


def test_unreadable_label_returns_fixed_failure_schema(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    case = "psma_patientc_2020-01-01"
    _make_case(root, case, positive=True)
    label_path = root / "labelsTr" / f"{case}.nii.gz"
    label_path.write_bytes(b"not-a-nifti")

    result = inspect_case(
        case,
        root / "imagesTr" / f"{case}_0000.nii.gz",
        root / "imagesTr" / f"{case}_0001.nii.gz",
        label_path,
    )

    assert result["status"] == "FAIL"
    assert result["label_state"] == "unreadable"
    assert result["label_voxels"] is None
    assert result["component_count"] is None


def test_unreadable_label_is_not_counted_as_empty(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    bad_case = "psma_patientb_2020-06-01"
    (root / "labelsTr" / f"{bad_case}.nii.gz").write_bytes(b"not-a-nifti")

    report = audit_dataset(
        root,
        workers=1,
        expected_cases=3,
        expected_patients=2,
        expected_empty=1,
        expected_folds=2,
    )

    assert report["status"] == "FAIL"
    assert report["summary"]["empty_label_count"] == 1
    assert report["summary"]["unreadable_label_count"] == 1
    assert report["summary"]["positive_label_count"] == 1


def test_source_dataset_channel_contract_rejects_swapped_roles(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    dataset_json = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    dataset_json["channel_names"] = {"0": "PET", "1": "CT"}
    (root / "dataset.json").write_text(json.dumps(dataset_json), encoding="utf-8")

    report = audit_dataset(
        root,
        workers=1,
        expected_cases=3,
        expected_patients=2,
        expected_empty=1,
        expected_folds=2,
    )

    assert report["status"] == "FAIL"
    assert "dataset_json_channel_names" in report["errors"]


def test_split_train_must_be_validation_complement(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    splits = json.loads((root / "splits_final.json").read_text(encoding="utf-8"))
    splits[0]["train"] = []
    (root / "splits_final.json").write_text(json.dumps(splits), encoding="utf-8")

    report = audit_dataset(
        root,
        workers=1,
        expected_cases=3,
        expected_patients=2,
        expected_empty=1,
        expected_folds=2,
    )

    assert report["status"] == "FAIL"
    assert report["split_audit"]["folds"][0]["train_is_val_complement"] is False
    assert "split_train_not_val_complement" in report["errors"]


def test_corner_diagonal_label_voxels_use_18_connectivity(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    case = "psma_patientc_2020-01-01"
    _make_case(root, case, positive=False)
    label = np.zeros((8, 7, 6), dtype=np.uint8)
    label[2, 2, 2] = 1
    label[3, 3, 3] = 1
    label_path = root / "labelsTr" / f"{case}.nii.gz"
    _save(label_path, label)

    result = inspect_case(
        case,
        root / "imagesTr" / f"{case}_0000.nii.gz",
        root / "imagesTr" / f"{case}_0001.nii.gz",
        label_path,
    )

    assert result["status"] == "PASS"
    assert result["component_count"] == 2
    assert result["connectivity"] == 18


def test_edge_diagonal_label_voxels_remain_connected_with_18_connectivity(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    case = "psma_patientc_2020-01-01"
    _make_case(root, case, positive=False)
    label = np.zeros((8, 7, 6), dtype=np.uint8)
    label[2, 2, 2] = 1
    label[3, 3, 2] = 1
    label_path = root / "labelsTr" / f"{case}.nii.gz"
    _save(label_path, label)

    result = inspect_case(
        case,
        root / "imagesTr" / f"{case}_0000.nii.gz",
        root / "imagesTr" / f"{case}_0001.nii.gz",
        label_path,
    )

    assert result["status"] == "PASS"
    assert result["component_count"] == 1
    assert result["connectivity"] == 18


def test_write_outputs_commits_json_and_csv_with_hash_receipt(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path)
    report = audit_dataset(
        root,
        workers=1,
        expected_cases=3,
        expected_patients=2,
        expected_empty=1,
        expected_folds=2,
    )
    output_dir = tmp_path / "audit-output"

    write_outputs(report, output_dir)

    completion = json.loads((output_dir / "AUDIT_COMPLETE.json").read_text(encoding="utf-8"))
    assert completion["status"] == "COMMITTED"
    for filename in ("psma_v3_nifti_audit.json", "psma_v3_case_audit.csv"):
        path = output_dir / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert completion["outputs"][filename]["sha256"] == digest
