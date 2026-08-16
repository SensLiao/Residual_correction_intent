#!/usr/bin/env python3
"""Patient/split/NIfTI readiness audit for PSMA-PET-CT-Lesions v3.

This is intentionally separate from archive extraction. It verifies the
scientific contracts required before nnU-Net planning: one CT/PET/label triplet
per examination, metadata/patient mapping, patient-disjoint reference folds,
and image geometry/value/label validity. It does not modify source data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage


NIFTI_SUFFIX = ".nii.gz"
DEFAULT_EXPECTED_CASES = 597
DEFAULT_EXPECTED_PATIENTS = 378
DEFAULT_EXPECTED_EMPTY = 58
DEFAULT_EXPECTED_FOLDS = 5
AFFINE_ATOL = 1e-4
ZOOM_ATOL = 1e-3
LESION_CONNECTIVITY = 18
LESION_STRUCTURE = ndimage.generate_binary_structure(3, 2)
AUDIT_VERSION = "1.2.0"
SOURCE_CHANNEL_NAMES = {"0": "CT", "1": "CT"}
SOURCE_LABELS = {"background": 0, "tumor": 1}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_from_case(case_id: str) -> str:
    """Remove the final shifted study-date token from a PSMA case id."""
    if "_" not in case_id:
        raise ValueError(f"case id has no study token: {case_id}")
    return case_id.rsplit("_", 1)[0].casefold()


def _nifti_case_from_image(path: Path, channel: str) -> str:
    suffix = f"_{channel}{NIFTI_SUFFIX}"
    if not path.name.casefold().endswith(suffix.casefold()):
        raise ValueError(f"unexpected image filename: {path.name}")
    return path.name[: -len(suffix)]


def _nifti_case_from_label(path: Path) -> str:
    if not path.name.casefold().endswith(NIFTI_SUFFIX):
        raise ValueError(f"unexpected label filename: {path.name}")
    return path.name[: -len(NIFTI_SUFFIX)]


def discover_triplets(dataset_root: Path) -> tuple[dict[str, tuple[Path, Path, Path]], list[str]]:
    images = dataset_root / "imagesTr"
    labels = dataset_root / "labelsTr"
    ct = {_nifti_case_from_image(p, "0000"): p for p in sorted(images.glob("*_0000.nii.gz"))}
    pet = {_nifti_case_from_image(p, "0001"): p for p in sorted(images.glob("*_0001.nii.gz"))}
    seg = {_nifti_case_from_label(p): p for p in sorted(labels.glob("*.nii.gz"))}
    all_cases = set(ct) | set(pet) | set(seg)
    errors: list[str] = []
    triplets: dict[str, tuple[Path, Path, Path]] = {}
    for case in sorted(all_cases):
        missing = [role for role, mapping in (("ct", ct), ("pet", pet), ("label", seg)) if case not in mapping]
        if missing:
            errors.append(f"incomplete_triplet:{case}:{','.join(missing)}")
        else:
            triplets[case] = (ct[case], pet[case], seg[case])
    extra_images = [p.name for p in images.glob("*.nii.gz") if not (p.name.endswith("_0000.nii.gz") or p.name.endswith("_0001.nii.gz"))]
    if extra_images:
        errors.append(f"unclassified_images:{len(extra_images)}")
    return triplets, errors


def _metadata_case(row: dict[str, str]) -> str:
    return f"{row['Subject ID'].strip().casefold()}_{row['Study Date'].strip()}"


def load_metadata(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    errors: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "Subject ID",
        "Study Date",
        "age",
        "manufacturer_model_name",
        "pet_radionuclide",
        "ct_contrast_agent",
    }
    if not rows:
        return {}, ["metadata_empty"]
    missing_columns = sorted(required - set(rows[0]))
    if missing_columns:
        errors.append(f"metadata_missing_columns:{','.join(missing_columns)}")
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        if not required.issubset(row):
            continue
        empty_required = sorted(key for key in required if not row.get(key, "").strip())
        if empty_required:
            errors.append(f"metadata_empty_required:{','.join(empty_required)}")
            continue
        case = _metadata_case(row)
        if case in mapping:
            errors.append(f"metadata_duplicate_case:{case}")
        mapping[case] = row
    return mapping, errors


def audit_splits(
    splits: list[dict],
    cases: set[str],
    *,
    case_to_patient: dict[str, str] | None = None,
    expected_folds: int = DEFAULT_EXPECTED_FOLDS,
) -> dict:
    folds: list[dict] = []
    val_counts: Counter[str] = Counter()
    val_patient_folds: dict[str, set[int]] = defaultdict(set)
    patient_overlap_total = 0
    case_overlap_total = 0
    incomplete_coverage_folds = 0
    train_not_val_complement_folds = 0
    unknown_cases: set[str] = set()

    def patient(case: str) -> str:
        if case_to_patient and case in case_to_patient:
            return case_to_patient[case]
        try:
            return patient_from_case(case)
        except ValueError:
            return f"__invalid_case__:{case}"

    for fold_idx, fold in enumerate(splits):
        train = [str(x) for x in fold.get("train", [])]
        val = [str(x) for x in fold.get("val", [])]
        train_set = set(train)
        val_set = set(val)
        unknown_cases.update((train_set | val_set) - cases)
        case_overlap = train_set & val_set
        case_overlap_total += len(case_overlap)
        covers_all_cases = train_set | val_set == cases
        train_is_val_complement = train_set == cases - val_set
        incomplete_coverage_folds += int(not covers_all_cases)
        train_not_val_complement_folds += int(not train_is_val_complement)
        train_patients = {patient(x) for x in train}
        val_patients = {patient(x) for x in val}
        overlap = sorted(train_patients & val_patients)
        patient_overlap_total += len(overlap)
        val_counts.update(val)
        for case in val:
            val_patient_folds[patient_from_case(case)].add(fold_idx)
        folds.append(
            {
                "fold": fold_idx,
                "train_cases": len(train),
                "val_cases": len(val),
                "train_patients": len(train_patients),
                "val_patients": len(val_patients),
                "patient_overlap": len(overlap),
                "overlap_examples": overlap[:10],
                "case_overlap": len(case_overlap),
                "case_overlap_examples": sorted(case_overlap)[:10],
                "covers_all_cases": covers_all_cases,
                "train_is_val_complement": train_is_val_complement,
            }
        )
    val_not_once = {case: val_counts.get(case, 0) for case in sorted(cases) if val_counts.get(case, 0) != 1}
    patient_multi = {p: sorted(fs) for p, fs in val_patient_folds.items() if len(fs) > 1}
    return {
        "fold_count": len(splits),
        "expected_fold_count": expected_folds,
        "fold_count_matches": len(splits) == expected_folds,
        "folds": folds,
        "patient_overlap_total": patient_overlap_total,
        "case_overlap_total": case_overlap_total,
        "incomplete_coverage_folds": incomplete_coverage_folds,
        "train_not_val_complement_folds": train_not_val_complement_folds,
        "val_cases_not_exactly_once": len(val_not_once),
        "val_cases_not_exactly_once_examples": list(val_not_once.items())[:10],
        "patients_in_multiple_val_folds": len(patient_multi),
        "patients_in_multiple_val_folds_examples": list(patient_multi.items())[:10],
        "unknown_case_count": len(unknown_cases),
        "unknown_case_examples": sorted(unknown_cases)[:10],
    }


def _volume_bin(volume_ml: float) -> str:
    if volume_ml < 1:
        return "lt_1ml"
    if volume_ml < 5:
        return "1_to_5ml"
    if volume_ml < 10:
        return "5_to_10ml"
    if volume_ml < 50:
        return "10_to_50ml"
    return "ge_50ml"


def _failed_case_result(case_id: str, error: str, *, label_state: str = "unreadable") -> dict:
    try:
        patient_id = patient_from_case(case_id)
    except ValueError:
        patient_id = ""
    return {
        "case_id": case_id,
        "patient_id": patient_id,
        "status": "FAIL",
        "errors": [error],
        "shape": None,
        "spacing": None,
        "orientation": None,
        "affine_max_abs_ct_pet": None,
        "affine_max_abs_ct_label": None,
        "label_state": label_state,
        "label_unique": [],
        "label_voxels": None,
        "voxel_volume_ml": None,
        "total_lesion_ml": None,
        "component_count": None,
        "connectivity": LESION_CONNECTIVITY,
        "component_volume_ml_min": None,
        "component_volume_ml_median": None,
        "component_volume_ml_max": None,
        "component_volume_bins": {},
        "ct_nonfinite_voxels": None,
        "pet_nonfinite_voxels": None,
        "pet_negative_voxels": None,
        "ct_min": None,
        "ct_max": None,
        "pet_min": None,
        "pet_max": None,
        "lesion_pet_median": None,
        "lesion_pet_mean": None,
        "lesion_pet_max": None,
    }


def _inspect_case_impl(case_id: str, ct_path: Path, pet_path: Path, label_path: Path) -> dict:
    errors: list[str] = []
    try:
        ct_img = nib.load(str(ct_path))
        pet_img = nib.load(str(pet_path))
        label_img = nib.load(str(label_path))
    except Exception as exc:  # malformed NIfTI must be recorded, not crash the batch
        return _failed_case_result(case_id, f"nifti_load:{type(exc).__name__}:{exc}")

    shapes = [tuple(int(x) for x in image.shape[:3]) for image in (ct_img, pet_img, label_img)]
    if len(set(shapes)) != 1:
        errors.append("shape_mismatch")
    affines = [image.affine for image in (ct_img, pet_img, label_img)]
    if not (np.allclose(affines[0], affines[1], atol=AFFINE_ATOL, rtol=0) and np.allclose(affines[0], affines[2], atol=AFFINE_ATOL, rtol=0)):
        errors.append("affine_mismatch")
    zooms = [tuple(float(x) for x in image.header.get_zooms()[:3]) for image in (ct_img, pet_img, label_img)]
    if not (np.allclose(zooms[0], zooms[1], atol=ZOOM_ATOL, rtol=0) and np.allclose(zooms[0], zooms[2], atol=ZOOM_ATOL, rtol=0)):
        errors.append("spacing_mismatch")
    orientations = ["".join(nib.aff2axcodes(image.affine)) for image in (ct_img, pet_img, label_img)]
    if len(set(orientations)) != 1:
        errors.append("orientation_mismatch")

    try:
        label = np.asarray(label_img.dataobj)
        label_unique = np.unique(label)
        if not np.all(np.isin(label_unique, [0, 1])):
            errors.append("invalid_label_values")
            label_state = "invalid_values"
            label_bool = None
            label_voxels = None
            voxel_volume_ml = total_lesion_ml = None
            component_count = None
            component_volumes_ml = np.asarray([], dtype=np.float64)
        else:
            label_bool = label == 1
            label_voxels = int(label_bool.sum())
            label_state = "valid_positive" if label_voxels else "valid_empty"
            voxel_volume_ml = float(abs(np.linalg.det(label_img.affine[:3, :3])) / 1000.0)
            total_lesion_ml = label_voxels * voxel_volume_ml
            if label_voxels:
                labeled, component_count = ndimage.label(label_bool, structure=LESION_STRUCTURE)
                component_voxels = np.bincount(labeled.ravel())[1:]
                component_volumes_ml = component_voxels.astype(np.float64) * voxel_volume_ml
            else:
                component_count = 0
                component_volumes_ml = np.asarray([], dtype=np.float64)

        ct = np.asarray(ct_img.dataobj, dtype=np.float32)
        pet = np.asarray(pet_img.dataobj, dtype=np.float32)
        ct_nonfinite = int((~np.isfinite(ct)).sum())
        pet_nonfinite = int((~np.isfinite(pet)).sum())
        if ct_nonfinite:
            errors.append("ct_nonfinite")
        if pet_nonfinite:
            errors.append("pet_nonfinite")
        pet_negative_voxels = int((pet < 0).sum())
        if label_voxels and label_bool is not None and not pet_nonfinite:
            lesion_pet = pet[label_bool]
            lesion_pet_median = float(np.median(lesion_pet))
            lesion_pet_mean = float(np.mean(lesion_pet))
            lesion_pet_max = float(np.max(lesion_pet))
        else:
            lesion_pet_median = lesion_pet_mean = lesion_pet_max = None
    except Exception as exc:
        errors.append(f"array_audit:{type(exc).__name__}:{exc}")
        label_unique = np.asarray([])
        label_state = "unreadable"
        label_voxels = component_count = None
        voxel_volume_ml = total_lesion_ml = None
        component_volumes_ml = np.asarray([], dtype=np.float64)
        ct_nonfinite = pet_nonfinite = pet_negative_voxels = -1
        lesion_pet_median = lesion_pet_mean = lesion_pet_max = None
        ct = pet = None

    bins = Counter(_volume_bin(float(v)) for v in component_volumes_ml)
    result = {
        "case_id": case_id,
        "patient_id": patient_from_case(case_id),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "shape": list(shapes[0]),
        "spacing": list(zooms[0]),
        "orientation": orientations[0],
        "affine_max_abs_ct_pet": float(np.max(np.abs(affines[0] - affines[1]))),
        "affine_max_abs_ct_label": float(np.max(np.abs(affines[0] - affines[2]))),
        "label_state": label_state,
        "label_unique": [float(x) for x in label_unique.tolist()],
        "label_voxels": label_voxels,
        "voxel_volume_ml": voxel_volume_ml,
        "total_lesion_ml": float(total_lesion_ml) if total_lesion_ml is not None else None,
        "component_count": int(component_count) if component_count is not None else None,
        "connectivity": LESION_CONNECTIVITY,
        "component_volume_ml_min": float(component_volumes_ml.min()) if component_volumes_ml.size else None,
        "component_volume_ml_median": float(np.median(component_volumes_ml)) if component_volumes_ml.size else None,
        "component_volume_ml_max": float(component_volumes_ml.max()) if component_volumes_ml.size else None,
        "component_volume_bins": dict(bins),
        "ct_nonfinite_voxels": ct_nonfinite,
        "pet_nonfinite_voxels": pet_nonfinite,
        "pet_negative_voxels": pet_negative_voxels,
        "ct_min": float(np.min(ct)) if ct is not None and not ct_nonfinite else None,
        "ct_max": float(np.max(ct)) if ct is not None and not ct_nonfinite else None,
        "pet_min": float(np.min(pet)) if pet is not None and not pet_nonfinite else None,
        "pet_max": float(np.max(pet)) if pet is not None and not pet_nonfinite else None,
        "lesion_pet_median": lesion_pet_median,
        "lesion_pet_mean": lesion_pet_mean,
        "lesion_pet_max": lesion_pet_max,
    }
    return result


def inspect_case(case_id: str, ct_path: Path, pet_path: Path, label_path: Path) -> dict:
    try:
        return _inspect_case_impl(case_id, ct_path, pet_path, label_path)
    except Exception as exc:
        return _failed_case_result(case_id, f"worker_exception:{type(exc).__name__}:{exc}")


def _inspect_case_args(args: tuple[str, Path, Path, Path]) -> dict:
    try:
        return inspect_case(*args)
    except Exception as exc:
        return _failed_case_result(args[0], f"worker_exception:{type(exc).__name__}:{exc}")


def _distribution(rows: Iterable[dict[str, str]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(key, "") for row in rows).items()))


def audit_dataset(
    dataset_root: Path | str,
    *,
    workers: int = 2,
    expected_cases: int = DEFAULT_EXPECTED_CASES,
    expected_patients: int = DEFAULT_EXPECTED_PATIENTS,
    expected_empty: int = DEFAULT_EXPECTED_EMPTY,
    expected_folds: int = DEFAULT_EXPECTED_FOLDS,
) -> dict:
    dataset_root = Path(dataset_root).resolve()
    triplets, errors = discover_triplets(dataset_root)
    metadata_path = dataset_root / "psma_metadata.csv"
    dataset_json_path = dataset_root / "dataset.json"
    splits_path = dataset_root / "splits_final.json"
    for path in (metadata_path, dataset_json_path, splits_path):
        if not path.is_file():
            errors.append(f"missing_required_file:{path.name}")
    if errors and not all(path.is_file() for path in (metadata_path, dataset_json_path, splits_path)):
        return {"status": "FAIL", "errors": errors, "dataset_root": str(dataset_root)}

    metadata, metadata_errors = load_metadata(metadata_path)
    errors.extend(metadata_errors)
    dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if dataset_json.get("channel_names") != SOURCE_CHANNEL_NAMES:
        errors.append("dataset_json_channel_names")
    if dataset_json.get("labels") != SOURCE_LABELS:
        errors.append("dataset_json_labels")
    if dataset_json.get("file_ending") != NIFTI_SUFFIX:
        errors.append("dataset_json_file_ending")
    case_ids = set(triplets)
    metadata_ids = set(metadata)
    missing_metadata = sorted(case_ids - metadata_ids)
    extra_metadata = sorted(metadata_ids - case_ids)
    if missing_metadata:
        errors.append(f"missing_metadata_cases:{len(missing_metadata)}")
    if extra_metadata:
        errors.append(f"extra_metadata_cases:{len(extra_metadata)}")

    case_to_patient = {
        case: row["Subject ID"].strip().casefold()
        for case, row in metadata.items()
        if row.get("Subject ID", "").strip()
    }
    split_audit = audit_splits(
        splits,
        case_ids,
        case_to_patient=case_to_patient,
        expected_folds=expected_folds,
    )
    if not split_audit["fold_count_matches"]:
        errors.append("split_fold_count")
    if split_audit["patient_overlap_total"]:
        errors.append("split_patient_overlap")
    if split_audit["case_overlap_total"]:
        errors.append("split_case_overlap")
    if split_audit["incomplete_coverage_folds"]:
        errors.append("split_incomplete_coverage")
    if split_audit["train_not_val_complement_folds"]:
        errors.append("split_train_not_val_complement")
    if split_audit["val_cases_not_exactly_once"]:
        errors.append("split_val_case_coverage")
    if split_audit["patients_in_multiple_val_folds"]:
        errors.append("split_patient_multiple_val_folds")
    if split_audit["unknown_case_count"]:
        errors.append("split_unknown_cases")

    args = [(case, *triplets[case]) for case in sorted(triplets)]
    if workers <= 1:
        per_case = [_inspect_case_args(item) for item in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            per_case = list(pool.map(_inspect_case_args, args, chunksize=1))
    for row in per_case:
        if row["case_id"] in case_to_patient:
            row["patient_id"] = case_to_patient[row["case_id"]]
    failed_cases = [row for row in per_case if row["status"] != "PASS"]
    if failed_cases:
        errors.append(f"nifti_failed_cases:{len(failed_cases)}")

    patients = {row["patient_id"] for row in per_case}
    empty_count = sum(row["label_state"] == "valid_empty" for row in per_case)
    positive_count = sum(row["label_state"] == "valid_positive" for row in per_case)
    unreadable_count = sum(row["label_state"] == "unreadable" for row in per_case)
    invalid_label_count = sum(row["label_state"] == "invalid_values" for row in per_case)
    component_bins: Counter[str] = Counter()
    for row in per_case:
        component_bins.update(row["component_volume_bins"])
    if len(per_case) != expected_cases:
        errors.append(f"case_count:{len(per_case)}!={expected_cases}")
    if len(patients) != expected_patients:
        errors.append(f"patient_count:{len(patients)}!={expected_patients}")
    if empty_count != expected_empty:
        errors.append(f"empty_label_count:{empty_count}!={expected_empty}")
    if dataset_json.get("numTraining") != expected_cases:
        errors.append(f"dataset_json_numTraining:{dataset_json.get('numTraining')}!={expected_cases}")

    metadata_rows = [metadata[c] for c in sorted(case_ids & metadata_ids)]
    report = {
        "status": "PASS" if not errors else "FAIL",
        "audit_version": AUDIT_VERSION,
        "dataset_root": str(dataset_root),
        "errors": errors,
        "source_hashes": {
            "dataset_json_sha256": sha256_file(dataset_json_path),
            "splits_final_sha256": sha256_file(splits_path),
            "psma_metadata_sha256": sha256_file(metadata_path),
        },
        "dataset_json": dataset_json,
        "metadata_audit": {
            "row_count": len(metadata),
            "missing_case_count": len(missing_metadata),
            "missing_case_examples": missing_metadata[:10],
            "extra_case_count": len(extra_metadata),
            "extra_case_examples": extra_metadata[:10],
            "radionuclide_distribution": _distribution(metadata_rows, "pet_radionuclide"),
            "scanner_distribution": _distribution(metadata_rows, "manufacturer_model_name"),
            "contrast_distribution": _distribution(metadata_rows, "ct_contrast_agent"),
        },
        "split_audit": split_audit,
        "summary": {
            "case_count": len(per_case),
            "patient_count": len(patients),
            "positive_label_count": positive_count,
            "empty_label_count": empty_count,
            "unreadable_label_count": unreadable_count,
            "invalid_label_count": invalid_label_count,
            "component_count": int(sum((row["component_count"] or 0) for row in per_case)),
            "connectivity": LESION_CONNECTIVITY,
            "component_volume_bins": dict(sorted(component_bins.items())),
            "failed_case_count": len(failed_cases),
            "pet_negative_case_count": int(sum((row["pet_negative_voxels"] or 0) > 0 for row in per_case)),
            "pet_negative_voxel_count": int(sum(max(row["pet_negative_voxels"] or 0, 0) for row in per_case)),
        },
        "failed_case_examples": failed_cases[:20],
        "per_case": per_case,
    }
    return report


def _write_text_fsync(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def write_outputs(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "psma_v3_nifti_audit.json"
    csv_path = output_dir / "psma_v3_case_audit.csv"
    completion_path = output_dir / "AUDIT_COMPLETE.json"
    for path in (report_path, csv_path, completion_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite audit output: {path}")
    report_temp = output_dir / ".psma_v3_nifti_audit.json.partial"
    csv_temp = output_dir / ".psma_v3_case_audit.csv.partial"
    completion_temp = output_dir / ".AUDIT_COMPLETE.json.partial"
    for path in (report_temp, csv_temp, completion_temp):
        if path.exists():
            raise FileExistsError(f"partial audit output already exists: {path}")
    _write_text_fsync(
        report_temp,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    rows = report.get("per_case", [])
    if rows:
        fieldnames = [key for key in rows[0] if key not in {"errors", "shape", "spacing", "label_unique", "component_volume_bins"}]
        with csv_temp.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        _write_text_fsync(csv_temp, "")
    os.replace(report_temp, report_path)
    os.replace(csv_temp, csv_path)
    completion = {
        "status": "COMMITTED",
        "audit_status": report.get("status"),
        "audit_version": AUDIT_VERSION,
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": {
            report_path.name: {"bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)},
            csv_path.name: {"bytes": csv_path.stat().st_size, "sha256": sha256_file(csv_path)},
        },
    }
    _write_text_fsync(
        completion_temp,
        json.dumps(completion, ensure_ascii=False, indent=2) + "\n",
    )
    os.replace(completion_temp, completion_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--expected-cases", type=int, default=DEFAULT_EXPECTED_CASES)
    parser.add_argument("--expected-patients", type=int, default=DEFAULT_EXPECTED_PATIENTS)
    parser.add_argument("--expected-empty", type=int, default=DEFAULT_EXPECTED_EMPTY)
    parser.add_argument("--expected-folds", type=int, default=DEFAULT_EXPECTED_FOLDS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit_dataset(
        args.dataset_root,
        workers=args.workers,
        expected_cases=args.expected_cases,
        expected_patients=args.expected_patients,
        expected_empty=args.expected_empty,
        expected_folds=args.expected_folds,
    )
    write_outputs(report, args.output_dir)
    print(json.dumps({"status": report["status"], "summary": report.get("summary"), "errors": report.get("errors")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
