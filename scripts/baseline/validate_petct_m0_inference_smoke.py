#!/usr/bin/env python3
"""Validate and publish the Dataset901 one-case inference smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

import nibabel as nib
import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
for support_dir in (SCRIPTS_ROOT, SCRIPTS_ROOT / "data"):
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))

from audit_psma_v3_dataset import patient_from_case  # noqa: E402
from prepare_nnunet_m0_dataset import commit_run_directory  # noqa: E402
from validate_petct_m0_full_training import (  # noqa: E402
    validate_training_prerequisites,
)
from validate_petct_m0_preprocess import (  # noqa: E402
    ContractError,
    _load_json,
    _sha256,
    _verify_record,
    _write_json_exclusive,
)


DATASET_FOLDER = "Dataset901_PSMA_M0_AutoPETVNorm"
TRAINER_FOLDER = "nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres"
EXPECTED_TRAINER = "nnUNetTrainer_1epoch"
FOLD = 0
CHECKPOINT_NAME = "checkpoint_final.pth"
CASE_SELECTION_VERSION = "PETCT-M0-INFERENCE-SMOKE-CASE-v1"
CONTRACT_VERSION = "1.0.0"
PHASE = "FOLD0_ACTUAL_CASE_INFERENCE_SMOKE"
INFERENCE_RUNTIME_CONTRACT = {
    "predictor": "nnUNetPredictor",
    "pinned_nnunet_version": "2.8.1",
    "compile_mode": "disabled",
    "nnunet_compile": "false",
    "scope": "INFERENCE_ONLY",
    "official_compile_trigger": "nnUNet_compile=true|1|t",
    "future_oof_compile_mode": "disabled",
}


def _require_fields(
    payload: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ContractError(
                f"{label} requires {key}={value!r}; observed {payload.get(key)!r}"
            )


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ContractError(f"required regular non-empty file is missing: {path}")
    display = str(path)
    if relative_to is not None:
        display = path.relative_to(relative_to.resolve()).as_posix()
    return {"path": display, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read_json_any(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ContractError(f"{label} is missing or empty")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid UTF-8 JSON") from exc


def _selection_hash(case_id: str) -> str:
    return hashlib.sha256(
        f"{CASE_SELECTION_VERSION}|fold0|{case_id}".encode("utf-8")
    ).hexdigest()


def select_fold0_validation_case(split_contract: dict[str, Any]) -> dict[str, Any]:
    """Select the immutable minimum-hash fold-0 validation case."""

    folds = split_contract.get("folds")
    fold = folds.get("0") if isinstance(folds, dict) else None
    if not isinstance(fold, dict):
        raise ContractError("split contract has no fold0 manifest")
    train = fold.get("train")
    val = fold.get("val")
    if not isinstance(train, list) or not isinstance(val, list) or not train or not val:
        raise ContractError("fold0 train/val manifests must be non-empty lists")
    if not all(isinstance(case, str) and case for case in train + val):
        raise ContractError("fold0 contains an invalid case identifier")
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ContractError("fold0 manifests contain duplicate cases")
    if set(train) & set(val):
        raise ContractError("fold0 train and validation cases overlap")

    selected = min(val, key=lambda case: (_selection_hash(case), case))
    try:
        selected_patient = patient_from_case(selected)
        train_patients = {patient_from_case(case) for case in train}
    except ValueError as exc:
        raise ContractError("fold0 contains an invalid patient/case identity") from exc
    if selected_patient in train_patients:
        raise ContractError(
            "selected fold0 validation case is not patient-excluded from fold0 train"
        )
    canonical_val = "\n".join(sorted(val))
    return {
        "case_id": selected,
        "patient_id": selected_patient,
        "held_out_fold": FOLD,
        "selection_version": CASE_SELECTION_VERSION,
        "selection_sha256": _selection_hash(selected),
        "fold0_val_manifest_sha256": hashlib.sha256(
            canonical_val.encode("utf-8")
        ).hexdigest(),
        "patient_excluded_from_train": True,
    }


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def validate_inference_prerequisites(
    preprocess_ready_path: Path,
    smoke_ready_path: Path,
    source_root: Path,
    *,
    expected_case_count: int = 597,
    base_prerequisite_validator: Callable[..., dict[str, Any]] = (
        validate_training_prerequisites
    ),
) -> dict[str, Any]:
    """Revalidate both upstream gates and bind one fold-0 case/checkpoint."""

    base = base_prerequisite_validator(
        preprocess_ready_path,
        smoke_ready_path,
        source_root,
        expected_case_count=expected_case_count,
    )
    if base.get("status") != "PASS":
        raise ContractError("base training prerequisites are not PASS")
    selected = select_fold0_validation_case(base.get("split_contract", {}))

    preprocess = base.get("preprocess")
    smoke = base.get("smoke")
    paths = base.get("paths")
    bound = base.get("bound_hashes")
    if not all(isinstance(item, dict) for item in (preprocess, smoke, paths, bound)):
        raise ContractError("base prerequisite payload is incomplete")
    assert isinstance(preprocess, dict)
    assert isinstance(smoke, dict)
    assert isinstance(paths, dict)
    assert isinstance(bound, dict)

    raw_root_raw = preprocess.get("nnunet_raw")
    smoke_run_raw = smoke.get("run_dir")
    split_raw = paths.get("splits_final")
    if not all(
        isinstance(item, str) and item
        for item in (raw_root_raw, smoke_run_raw, split_raw)
    ):
        raise ContractError("base prerequisite paths are incomplete")
    raw_root = Path(str(raw_root_raw)).resolve()
    smoke_run = Path(str(smoke_run_raw)).resolve()
    split_path = Path(str(split_raw)).resolve()
    case_id = selected["case_id"]
    images = raw_root / DATASET_FOLDER / "imagesTr"
    source_ct = images / f"{case_id}_0000.nii.gz"
    source_pet = images / f"{case_id}_0001.nii.gz"
    model_dir = smoke_run / "nnUNet_results" / DATASET_FOLDER / TRAINER_FOLDER
    checkpoint = model_dir / f"fold_{FOLD}" / CHECKPOINT_NAME
    plans_path = model_dir / "plans.json"
    dataset_path = model_dir / "dataset.json"

    records = {
        "preprocess_ready": _record(preprocess_ready_path),
        "smoke_ready": _record(smoke_ready_path),
        "splits_final": _record(split_path),
        "checkpoint_final": _record(checkpoint),
        "plans": _record(plans_path),
        "dataset_json": _record(dataset_path),
        "source_ct": _record(source_ct),
        "source_pet": _record(source_pet),
    }
    plans = _read_json_any(plans_path, "smoke model plans.json")
    dataset = _read_json_any(dataset_path, "smoke model dataset.json")
    if not isinstance(plans, dict) or plans.get("dataset_name") != DATASET_FOLDER:
        raise ContractError("smoke model plans do not identify Dataset901")
    if not isinstance(dataset, dict):
        raise ContractError("smoke model dataset.json must be an object")
    if dataset.get("channel_names") != {"0": "CT", "1": "PET"}:
        raise ContractError("smoke model dataset channels must be exactly CT/PET")
    if dataset.get("labels") != {"background": 0, "tumor": 1}:
        raise ContractError("smoke model labels must be exactly background/tumor")
    if dataset.get("file_ending") != ".nii.gz":
        raise ContractError("smoke model file ending must be .nii.gz")

    source_tree_hash = _require_hash(bound.get("source_tree"), "source tree hash")
    bound_hashes = {key: value["sha256"] for key, value in records.items()}
    bound_hashes["source_tree"] = source_tree_hash
    return {
        "status": "PASS",
        "selected_case": selected,
        "paths": {
            "preprocess_ready": str(Path(preprocess_ready_path).resolve()),
            "smoke_ready": str(Path(smoke_ready_path).resolve()),
            "source_root": str(Path(source_root).resolve()),
            "splits_final": str(split_path),
            "model_training_output_dir": str(model_dir.resolve()),
            "checkpoint_final": str(checkpoint.resolve()),
            "plans": str(plans_path.resolve()),
            "dataset_json": str(dataset_path.resolve()),
            "source_ct": str(source_ct.resolve()),
            "source_pet": str(source_pet.resolve()),
        },
        "input_records": records,
        "bound_hashes": bound_hashes,
        "runtime": base.get("runtime"),
        "split_contract": base.get("split_contract"),
        "preprocess": preprocess,
        "smoke": smoke,
    }


def stage_inference_smoke_run(
    preprocess_ready_path: Path,
    smoke_ready_path: Path,
    source_root: Path,
    staging_root: Path,
    final_root: Path,
    run_id: str,
    *,
    gpu_id: str,
    expected_case_count: int = 597,
    prerequisite_validator: Callable[..., dict[str, Any]] = (
        validate_inference_prerequisites
    ),
) -> dict[str, Any]:
    """Create a fresh run-owned staging directory for the one-case gate."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ContractError("unsafe inference-smoke run_id")
    if not re.fullmatch(r"[0-9]+", gpu_id):
        raise ContractError("visible GPU id must be a non-negative integer")
    staging_root = staging_root.resolve()
    final_root = final_root.resolve()
    if staging_root.is_symlink() or not staging_root.is_dir():
        raise ContractError("staging root must be a fresh regular directory")
    if staging_root.name != f".partial-{run_id}":
        raise ContractError("staging directory does not match run identity")
    if final_root.name != run_id or final_root.parent != staging_root.parent:
        raise ContractError("staging and final paths must be sibling run paths")
    if any(staging_root.iterdir()):
        raise ContractError("inference-smoke staging root must be empty")
    if os.path.lexists(final_root):
        raise FileExistsError(
            f"refusing existing inference-smoke destination: {final_root}"
        )

    prerequisites = prerequisite_validator(
        preprocess_ready_path,
        smoke_ready_path,
        source_root,
        expected_case_count=expected_case_count,
    )
    if prerequisites.get("status") != "PASS":
        raise ContractError("inference-smoke prerequisites are not PASS")
    owner = {
        "status": "OWNED",
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "staging_dir_name": staging_root.name,
        "owner_token": uuid4().hex,
    }
    _write_json_exclusive(staging_root / "RUN_OWNER.json", owner)
    predictions = staging_root / "predictions"
    predictions.mkdir()
    spec = {
        "status": "STAGED",
        "contract_version": CONTRACT_VERSION,
        "phase": PHASE,
        "run_id": run_id,
        "committed_run_dir": str(final_root),
        "visible_gpu_id": gpu_id,
        "selected_case": prerequisites["selected_case"],
        "paths": prerequisites["paths"],
        "input_records": prerequisites["input_records"],
        "prerequisite_bound_hashes": prerequisites["bound_hashes"],
        "inference_runtime_contract": INFERENCE_RUNTIME_CONTRACT,
        "inference_smoke_status": "NOT_STARTED",
        "full_training_status": "NOT_STARTED",
        "scientific_metrics_computed": False,
        "oof_prediction_count": 0,
        "result_count": 0,
        "scribble_count": 0,
        "intent_count": 0,
    }
    _write_json_exclusive(staging_root / "INFERENCE_SMOKE_SPEC.json", spec)
    return spec


def _resolve_run_record(record: Any, run_root: Path, label: str) -> Path:
    raw = record.get("path") if isinstance(record, dict) else None
    if not isinstance(raw, str) or not raw:
        raise ContractError(f"{label} record path is missing")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate.resolve()
    if ".." in candidate.parts:
        raise ContractError(f"unsafe {label} relative path")
    resolved = (run_root / candidate).resolve()
    if not resolved.is_relative_to(run_root):
        raise ContractError(f"{label} escapes the inference-smoke run")
    return resolved


def _count_publication_files(roots: Iterable[Path], label: str) -> int:
    total = 0
    for root in roots:
        root = root.resolve()
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ContractError(f"{label} root is not a regular directory: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ContractError(f"{label} root contains a symlink: {path}")
            if path.is_file():
                total += 1
    return total


def _grid_signature(image: nib.spatialimages.SpatialImage) -> dict[str, Any]:
    return {
        "shape": list(image.shape),
        "affine": np.asarray(image.affine, dtype=float).tolist(),
        "zooms": [float(value) for value in image.header.get_zooms()[:3]],
        "orientation": list(nib.aff2axcodes(image.affine)),
    }


def _same_grid(
    left: nib.spatialimages.SpatialImage,
    right: nib.spatialimages.SpatialImage,
) -> bool:
    return (
        left.shape == right.shape
        and np.allclose(left.affine, right.affine, rtol=0.0, atol=1e-4)
        and np.allclose(
            left.header.get_zooms()[:3],
            right.header.get_zooms()[:3],
            rtol=0.0,
            atol=1e-5,
        )
        and nib.aff2axcodes(left.affine) == nib.aff2axcodes(right.affine)
    )


def validate_inference_smoke_output(
    run_root: Path,
    *,
    oof_roots: Iterable[Path] = (),
    result_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Validate the exact single-case mask/probability output contract."""

    run_root = run_root.resolve()
    if run_root.is_symlink() or not run_root.is_dir():
        raise ContractError("inference-smoke run root must be a regular directory")
    allowed_root = {
        "RUN_OWNER.json",
        "INFERENCE_SMOKE_SPEC.json",
        "INFERENCE_SMOKE_BUNDLE.json",
        "console.log",
        "predictions",
    }
    unexpected = {item.name for item in run_root.iterdir()} - allowed_root
    if unexpected:
        raise ContractError(
            f"inference-smoke run contains unexpected outputs: {sorted(unexpected)}"
        )

    owner = _load_json(run_root / "RUN_OWNER.json", label="RUN_OWNER")
    run_id = owner.get("run_id")
    if (
        owner.get("status") != "OWNED"
        or not isinstance(run_id, str)
        or run_root.name not in {run_id, f".partial-{run_id}"}
    ):
        raise ContractError("inference-smoke owner identity mismatch")
    spec = _load_json(
        run_root / "INFERENCE_SMOKE_SPEC.json", label="INFERENCE_SMOKE_SPEC"
    )
    _require_fields(
        spec,
        {
            "status": "STAGED",
            "contract_version": CONTRACT_VERSION,
            "phase": PHASE,
            "run_id": run_id,
            "full_training_status": "NOT_STARTED",
            "scientific_metrics_computed": False,
            "oof_prediction_count": 0,
            "result_count": 0,
            "scribble_count": 0,
            "intent_count": 0,
        },
        "INFERENCE_SMOKE_SPEC",
    )
    if spec.get("inference_runtime_contract") != INFERENCE_RUNTIME_CONTRACT:
        raise ContractError("inference runtime/compile contract drifted")
    committed_raw = spec.get("committed_run_dir")
    if not isinstance(committed_raw, str):
        raise ContractError("INFERENCE_SMOKE_SPEC committed run is missing")
    committed = Path(committed_raw).resolve()
    if committed.name != run_id or committed.parent != run_root.parent:
        raise ContractError("inference-smoke committed run identity mismatch")
    if run_root.name == run_id and run_root != committed:
        raise ContractError("committed inference-smoke path differs from its spec")

    selected = spec.get("selected_case")
    paths = spec.get("paths")
    records = spec.get("input_records")
    if not all(isinstance(value, dict) for value in (selected, paths, records)):
        raise ContractError("inference-smoke spec bindings are incomplete")
    assert isinstance(selected, dict)
    assert isinstance(paths, dict)
    assert isinstance(records, dict)
    case_id = selected.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ContractError("selected inference-smoke case is missing")
    if selected.get("held_out_fold") != 0 or not selected.get(
        "patient_excluded_from_train"
    ):
        raise ContractError("selected case is not a patient-excluded fold0 val case")
    for label, record in records.items():
        path = _resolve_run_record(record, run_root, str(label))
        _verify_record(path, record, label=f"bound input {label}")

    prediction_root = run_root / "predictions"
    if prediction_root.is_symlink() or not prediction_root.is_dir():
        raise ContractError("prediction directory is missing")
    expected_names = {
        f"{case_id}.nii.gz",
        f"{case_id}.npz",
        f"{case_id}.pkl",
        "dataset.json",
        "plans.json",
        "predict_from_raw_data_args.json",
    }
    observed_names = {item.name for item in prediction_root.iterdir()}
    if observed_names != expected_names:
        raise ContractError(
            "prediction directory must contain exactly one mask/probability case "
            f"and official metadata; observed {sorted(observed_names)}"
        )
    artifacts = {
        name: _record(prediction_root / name, relative_to=run_root)
        for name in sorted(expected_names)
    }

    dataset_output = _read_json_any(prediction_root / "dataset.json", "output dataset")
    plans_output = _read_json_any(prediction_root / "plans.json", "output plans")
    source_dataset_path = Path(str(paths.get("dataset_json", ""))).resolve()
    source_plans_path = Path(str(paths.get("plans", ""))).resolve()
    if dataset_output != _read_json_any(source_dataset_path, "source model dataset"):
        raise ContractError("output dataset metadata differs from the bound model")
    if plans_output != _read_json_any(source_plans_path, "source model plans"):
        raise ContractError("output plans metadata differs from the bound model")

    args_payload = _read_json_any(
        prediction_root / "predict_from_raw_data_args.json", "predictor arguments"
    )
    expected_arg_keys = {
        "list_of_lists_or_source_folder",
        "output_folder_or_list_of_truncated_output_files",
        "save_probabilities",
        "overwrite",
        "num_processes_preprocessing",
        "num_processes_segmentation_export",
        "folder_with_segs_from_prev_stage",
        "num_parts",
        "part_id",
    }
    if not isinstance(args_payload, dict) or set(args_payload) != expected_arg_keys:
        raise ContractError("official predictor argument receipt is incomplete")
    expected_inputs = [
        str(Path(str(paths.get("source_ct", ""))).resolve()),
        str(Path(str(paths.get("source_pet", ""))).resolve()),
    ]
    observed_inputs = args_payload.get("list_of_lists_or_source_folder")
    if not (
        isinstance(observed_inputs, list)
        and len(observed_inputs) == 1
        and isinstance(observed_inputs[0], list)
        and [str(Path(item).resolve()) for item in observed_inputs[0]]
        == expected_inputs
    ):
        raise ContractError(
            "predictor arguments do not contain the one bound CT/PET case"
        )
    observed_outputs = args_payload.get(
        "output_folder_or_list_of_truncated_output_files"
    )
    observed_prefix = (
        Path(observed_outputs[0]).resolve()
        if isinstance(observed_outputs, list)
        and len(observed_outputs) == 1
        and isinstance(observed_outputs[0], str)
        else None
    )
    if not (
        observed_prefix is not None
        and observed_prefix.name == case_id
        and observed_prefix.parent.name == "predictions"
        and observed_prefix.parent.parent.parent == run_root.parent
        and observed_prefix.parent.parent.name in {run_id, f".partial-{run_id}"}
    ):
        raise ContractError(
            "predictor arguments do not bind the run-scoped case output"
        )
    _require_fields(
        args_payload,
        {
            "save_probabilities": True,
            "overwrite": False,
            "num_processes_preprocessing": 2,
            "num_processes_segmentation_export": 2,
            "folder_with_segs_from_prev_stage": None,
            "num_parts": 1,
            "part_id": 0,
        },
        "predictor arguments",
    )

    try:
        ct_image = nib.load(expected_inputs[0])
        pet_image = nib.load(expected_inputs[1])
        mask_path = prediction_root / f"{case_id}.nii.gz"
        mask_image = nib.load(str(mask_path))
    except (OSError, ValueError) as exc:
        raise ContractError("source or prediction NIfTI is unreadable") from exc
    if not _same_grid(ct_image, pet_image) or not _same_grid(ct_image, mask_image):
        raise ContractError("prediction does not preserve original-grid geometry")
    mask = np.asarray(mask_image.dataobj)
    if not np.all(np.isfinite(mask)) or not np.all(np.isin(mask, (0, 1))):
        raise ContractError("prediction mask must be finite and binary")
    mask = mask.astype(np.uint8, copy=False)

    probability_path = prediction_root / f"{case_id}.npz"
    try:
        with np.load(probability_path, allow_pickle=False) as archive:
            if set(archive.files) != {"probabilities"}:
                raise ContractError(
                    "probability archive must contain only probabilities"
                )
            probabilities = np.asarray(archive["probabilities"])
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ContractError("probability archive is unreadable") from exc
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise ContractError("probabilities must use a floating dtype")
    # nnU-Net's official SimpleITK reader/writer contract stores probability
    # arrays as [C, z, y, x], while nibabel exposes the written NIfTI mask as
    # [x, y, z]. export_prediction.py has already restored the physical grid;
    # this final axis reversal is only the reader/writer memory convention.
    expected_official_shape = (2, *tuple(reversed(ct_image.shape)))
    if probabilities.shape != expected_official_shape:
        raise ContractError(
            "probability tensor does not match the official CZYX original-grid shape"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ContractError("probabilities must be finite")
    probability_min = float(probabilities.min())
    probability_max = float(probabilities.max())
    if probability_min < 0.0 or probability_max > 1.0:
        raise ContractError("probabilities must remain in [0, 1]")
    if not np.allclose(probabilities.sum(axis=0), 1.0, rtol=0.0, atol=1e-4):
        raise ContractError("foreground/background probabilities must sum to one")
    probabilities_xyz = probabilities.transpose(0, 3, 2, 1)
    if probabilities_xyz.shape != (2, *ct_image.shape):
        raise ContractError("probability tensor cannot be mapped to the NIfTI XYZ grid")
    probability_mask = np.argmax(probabilities_xyz, axis=0).astype(np.uint8)
    if not np.array_equal(mask, probability_mask):
        raise ContractError("binary mask is not consistent with exported probabilities")

    console = run_root / "console.log"
    console_record = _record(console, relative_to=run_root)
    console_text = console.read_text(encoding="utf-8", errors="strict")
    for marker in (
        f"Predicting {case_id}",
        "GPU prediction completed",
        "Segmentation export complete",
    ):
        if marker not in console_text:
            raise ContractError(f"inference console is missing marker: {marker}")

    oof_count = _count_publication_files(oof_roots, "OOF publication")
    if oof_count:
        raise ContractError("inference smoke must not publish OOF predictions")
    result_count = _count_publication_files(result_roots, "scientific result")
    if result_count:
        raise ContractError("inference smoke must not publish scientific results")
    return {
        "status": "PASS",
        "case_id": case_id,
        "selected_case": selected,
        "geometry": {
            "original_grid_match": True,
            "source": _grid_signature(ct_image),
            "prediction": _grid_signature(mask_image),
        },
        "probability": {
            "finite": True,
            "range": [0.0, 1.0],
            "observed_min": probability_min,
            "observed_max": probability_max,
            "foreground_channel": 1,
            "channel_sum_to_one": True,
            "official_axis_order": "CZYX",
            "official_shape": list(probabilities.shape),
            "nifti_axis_order": "CXYZ",
            "nifti_grid_shape": list(probabilities_xyz.shape),
        },
        "mask_probability_consistent": True,
        "foreground_voxel_count": int(mask.sum()),
        "prediction_count": 1,
        "inference_runtime_contract": INFERENCE_RUNTIME_CONTRACT,
        "scientific_metrics_computed": False,
        "oof_prediction_count": 0,
        "result_count": 0,
        "scribble_count": 0,
        "intent_count": 0,
        "artifacts": {**artifacts, "console.log": console_record},
    }


def build_inference_smoke_bundle(
    preprocess_ready_path: Path,
    smoke_ready_path: Path,
    source_root: Path,
    *,
    run_id: str,
    committed_run_dir: Path,
    inventory: dict[str, Any],
    expected_case_count: int = 597,
    prerequisite_validator: Callable[..., dict[str, Any]] = (
        validate_inference_prerequisites
    ),
) -> dict[str, Any]:
    prerequisites = prerequisite_validator(
        preprocess_ready_path,
        smoke_ready_path,
        source_root,
        expected_case_count=expected_case_count,
    )
    committed_run_dir = committed_run_dir.resolve()
    if committed_run_dir.name != run_id:
        raise ContractError("inference-smoke bundle run identity mismatch")
    if inventory.get("status") != "PASS":
        raise ContractError("inference-smoke output inventory is not PASS")
    if inventory.get("case_id") != prerequisites.get("selected_case", {}).get(
        "case_id"
    ):
        raise ContractError("inference-smoke output is not the frozen selected case")
    _require_fields(
        inventory,
        {
            "prediction_count": 1,
            "inference_runtime_contract": INFERENCE_RUNTIME_CONTRACT,
            "scientific_metrics_computed": False,
            "oof_prediction_count": 0,
            "result_count": 0,
            "scribble_count": 0,
            "intent_count": 0,
        },
        "inference-smoke inventory",
    )
    return {
        "status": "VALIDATED",
        "inference_smoke_status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "phase": PHASE,
        "run_id": run_id,
        "committed_run_dir": str(committed_run_dir),
        "preprocess_ready": _record(preprocess_ready_path),
        "smoke_ready": _record(smoke_ready_path),
        "source_root": str(Path(source_root).resolve()),
        "selected_case": prerequisites["selected_case"],
        "prerequisite_bound_hashes": prerequisites["bound_hashes"],
        "inference_runtime_contract": INFERENCE_RUNTIME_CONTRACT,
        "output_contract": inventory,
        "full_training_status": "NOT_STARTED",
        "scientific_metrics_computed": False,
        "thesis_citable": False,
        "prediction_disposable": True,
        "receipt_retention_required": True,
        "oof_prediction_count": 0,
        "result_count": 0,
        "scribble_count": 0,
        "intent_count": 0,
    }


def publish_inference_smoke_ready(
    run_dir: Path,
    bundle_path: Path,
    ready_path: Path,
    *,
    expected_case_count: int = 597,
    prerequisite_validator: Callable[..., dict[str, Any]] = (
        validate_inference_prerequisites
    ),
    output_validator: Callable[[Path], dict[str, Any]] = (
        validate_inference_smoke_output
    ),
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    bundle_path = bundle_path.resolve()
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ContractError("committed inference-smoke run is missing")
    if bundle_path.parent != run_dir:
        raise ContractError("INFERENCE_SMOKE_BUNDLE must be inside its committed run")
    bundle_bytes = bundle_path.read_bytes()
    bundle = _load_json(bundle_path, label="INFERENCE_SMOKE_BUNDLE")
    _require_fields(
        bundle,
        {
            "status": "VALIDATED",
            "inference_smoke_status": "PASS",
            "contract_version": CONTRACT_VERSION,
            "phase": PHASE,
            "run_id": run_dir.name,
            "committed_run_dir": str(run_dir),
            "full_training_status": "NOT_STARTED",
            "scientific_metrics_computed": False,
            "thesis_citable": False,
            "prediction_disposable": True,
            "receipt_retention_required": True,
            "inference_runtime_contract": INFERENCE_RUNTIME_CONTRACT,
            "oof_prediction_count": 0,
            "result_count": 0,
            "scribble_count": 0,
            "intent_count": 0,
        },
        "INFERENCE_SMOKE_BUNDLE",
    )
    preprocess_record = bundle.get("preprocess_ready")
    smoke_record = bundle.get("smoke_ready")
    preprocess_path = _resolve_run_record(
        preprocess_record, run_dir, "PREPROCESS_READY"
    )
    smoke_path = _resolve_run_record(smoke_record, run_dir, "SMOKE_READY")
    _verify_record(preprocess_path, preprocess_record, label="PREPROCESS_READY")
    _verify_record(smoke_path, smoke_record, label="SMOKE_READY")
    source_root_raw = bundle.get("source_root")
    if not isinstance(source_root_raw, str):
        raise ContractError("INFERENCE_SMOKE_BUNDLE source root is missing")
    prerequisites = prerequisite_validator(
        preprocess_path,
        smoke_path,
        Path(source_root_raw),
        expected_case_count=expected_case_count,
    )
    if prerequisites.get("bound_hashes") != bundle.get("prerequisite_bound_hashes"):
        raise ContractError("inference-smoke prerequisite hashes changed")
    if prerequisites.get("selected_case") != bundle.get("selected_case"):
        raise ContractError("frozen inference-smoke case selection changed")
    fresh_output = output_validator(run_dir)
    if fresh_output != bundle.get("output_contract"):
        raise ContractError("inference-smoke output changed before publication")
    bundle_hash = hashlib.sha256(bundle_bytes).hexdigest()
    if _sha256(bundle_path) != bundle_hash:
        raise ContractError("INFERENCE_SMOKE_BUNDLE changed during publication")
    published = {
        "status": "COMMITTED",
        "inference_smoke_status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "phase": PHASE,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "run_receipt": {
            "path": str(bundle_path),
            "bytes": len(bundle_bytes),
            "sha256": bundle_hash,
        },
        "selected_case": bundle["selected_case"],
        "full_training_status": "NOT_STARTED",
        "scientific_metrics_computed": False,
        "thesis_citable": False,
        "prediction_disposable": True,
        "receipt_retention_required": True,
        "inference_runtime_contract": INFERENCE_RUNTIME_CONTRACT,
        "oof_prediction_count": 0,
        "result_count": 0,
        "scribble_count": 0,
        "intent_count": 0,
        "validated_bundle": bundle,
    }
    _write_json_exclusive(ready_path, published)
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prereq = commands.add_parser("validate-prerequisites")
    prereq.add_argument("preprocess_ready", type=Path)
    prereq.add_argument("smoke_ready", type=Path)
    prereq.add_argument("source_root", type=Path)
    prereq.add_argument("--expected-case-count", type=int, default=597)
    stage = commands.add_parser("stage")
    stage.add_argument("preprocess_ready", type=Path)
    stage.add_argument("smoke_ready", type=Path)
    stage.add_argument("source_root", type=Path)
    stage.add_argument("staging_root", type=Path)
    stage.add_argument("final_root", type=Path)
    stage.add_argument("run_id")
    stage.add_argument("gpu_id")
    stage.add_argument("--expected-case-count", type=int, default=597)
    validate = commands.add_parser("validate-output")
    validate.add_argument("preprocess_ready", type=Path)
    validate.add_argument("smoke_ready", type=Path)
    validate.add_argument("source_root", type=Path)
    validate.add_argument("run_root", type=Path)
    validate.add_argument("committed_run_dir", type=Path)
    validate.add_argument("run_id")
    validate.add_argument("receipt", type=Path)
    validate.add_argument("--expected-case-count", type=int, default=597)
    validate.add_argument("--oof-root", type=Path, action="append", default=[])
    validate.add_argument("--result-root", type=Path, action="append", default=[])
    commit = commands.add_parser("commit-run")
    commit.add_argument("staging_dir", type=Path)
    commit.add_argument("final_dir", type=Path)
    commit.add_argument("receipt", type=Path)
    publish = commands.add_parser("publish-ready")
    publish.add_argument("run_dir", type=Path)
    publish.add_argument("bundle", type=Path)
    publish.add_argument("ready", type=Path)
    publish.add_argument("--expected-case-count", type=int, default=597)
    publish.add_argument("--oof-root", type=Path, action="append", default=[])
    publish.add_argument("--result-root", type=Path, action="append", default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate-prerequisites":
        payload = validate_inference_prerequisites(
            args.preprocess_ready,
            args.smoke_ready,
            args.source_root,
            expected_case_count=args.expected_case_count,
        )
    elif args.command == "stage":
        payload = stage_inference_smoke_run(
            args.preprocess_ready,
            args.smoke_ready,
            args.source_root,
            args.staging_root,
            args.final_root,
            args.run_id,
            gpu_id=args.gpu_id,
            expected_case_count=args.expected_case_count,
        )
    elif args.command == "validate-output":
        inventory = validate_inference_smoke_output(
            args.run_root,
            oof_roots=args.oof_root,
            result_roots=args.result_root,
        )
        payload = build_inference_smoke_bundle(
            args.preprocess_ready,
            args.smoke_ready,
            args.source_root,
            run_id=args.run_id,
            committed_run_dir=args.committed_run_dir,
            inventory=inventory,
            expected_case_count=args.expected_case_count,
        )
        _write_json_exclusive(args.receipt, payload)
    elif args.command == "commit-run":
        payload = commit_run_directory(args.staging_dir, args.final_dir, args.receipt)
    else:
        payload = publish_inference_smoke_ready(
            args.run_dir,
            args.bundle,
            args.ready,
            expected_case_count=args.expected_case_count,
            output_validator=lambda run: validate_inference_smoke_output(
                run,
                oof_roots=args.oof_root,
                result_roots=args.result_root,
            ),
        )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
