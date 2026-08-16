#!/usr/bin/env python3
"""Execute exactly one val-only Dataset901 OOF fold with nnU-Net v2.8.1."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np

# nnU-Net creates Python ``spawn`` workers.  In that interpreter the project
# ``scripts/`` directory can precede this file's directory and contains an
# older same-named validator module.  Pin the sibling baseline modules first
# so child workers import the exact validator used by the parent OOF process.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [_SCRIPT_DIR] + [entry for entry in sys.path if entry != _SCRIPT_DIR]

from validate_petct_m0_oof import (  # noqa: E402
    EXPECTED_NNUNET_SOURCE_TREE_SHA256,
    INFERENCE_COMPILE_CONTRACT,
    OOF_CONTRACT_VERSION,
    commit_fold_done,
    validate_fold_plan_binding,
)
from validate_petct_m0_preprocess import (  # noqa: E402
    ContractError,
    _load_json,
    _sha256,
    _verify_record,
    _write_json_exclusive,
    validate_live_nnunet_runtime,
)


OFFICIAL_METADATA = {
    "dataset.json",
    "plans.json",
    "predict_from_raw_data_args.json",
}
ACTUAL_VALIDATION_HANDOFF_SCHEMA = (
    "PETCT-M0-OOF-ACTUAL-VALIDATION-HANDOFF-v1.0"
)
DEDICATED_PREDICTION_SOURCE = "DEDICATED_OOF_INFERENCE"
HANDOFF_PREDICTION_SOURCE = "TRAINING_ACTUAL_VALIDATION_HANDOFF"


def _validate_external_record(record: dict[str, Any], label: str) -> Path:
    raw = record.get("path") if isinstance(record, dict) else None
    if not isinstance(raw, str) or not raw:
        raise ContractError(f"{label} path is missing")
    candidate = Path(raw)
    if candidate.is_symlink() or not candidate.is_file():
        raise ContractError(f"{label} is missing or is not a regular file")
    path = candidate.resolve()
    _verify_record(path, record, label=label)
    return path


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"required regular file is missing: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _resolve_campaign_record(
    record: dict[str, Any], campaign_root: Path, *, label: str
) -> Path:
    raw = record.get("path") if isinstance(record, dict) else None
    if not isinstance(raw, str) or not raw:
        raise ContractError(f"{label} path is missing")
    candidate = Path(raw)
    if not candidate.is_absolute() and ".." in candidate.parts:
        raise ContractError(f"unsafe relative {label} path")
    unresolved = candidate if candidate.is_absolute() else campaign_root / candidate
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ContractError(f"{label} is missing or is not a regular file")
    path = unresolved.resolve()
    if not path.is_relative_to(campaign_root):
        raise ContractError(f"{label} escapes the frozen training campaign")
    _verify_record(path, record, label=label)
    return path


def _artifact_index(
    records: Any,
    campaign_root: Path,
    *,
    expected_names: set[str],
    label: str,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    if not isinstance(records, list):
        raise ContractError(f"{label} inventory is missing")
    indexed: dict[str, tuple[Path, dict[str, Any]]] = {}
    for record in records:
        path = _resolve_campaign_record(record, campaign_root, label=label)
        if path.name in indexed:
            raise ContractError(f"{label} inventory contains duplicate names")
        indexed[path.name] = (path, record)
    if set(indexed) != expected_names:
        raise ContractError(f"{label} inventory does not exactly match held-out cases")
    return indexed


def _fresh_output_directories(run_root: Path, fold: int) -> tuple[Path, Path]:
    masks = run_root / "outputs" / f"fold_{fold}" / "masks"
    probabilities = run_root / "outputs" / f"fold_{fold}" / "probabilities"
    for label, root in (("mask", masks), ("probability", probabilities)):
        if root.is_symlink() or not root.is_dir():
            raise ContractError(f"fold {fold} {label} output directory is missing")
        if any(root.iterdir()):
            raise ContractError(
                f"fold {fold} {label} output directory must be empty before inference"
            )
    return masks, probabilities


def _official_predictor_factory(device: str) -> Any:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    return nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )


def _extract_foreground_probability(
    source: Path,
    destination: Path,
    *,
    expected_mask_shape: tuple[int, int, int],
) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite OOF probability: {destination}")
    with np.load(source, allow_pickle=False) as archive:
        if archive.files != ["probabilities"]:
            raise ContractError("official nnU-Net NPZ must contain only probabilities")
        probabilities = archive["probabilities"]
    if probabilities.ndim != 4 or probabilities.shape[0] != 2:
        raise ContractError("official nnU-Net probability tensor must be [2, x, y, z]")
    if probabilities.shape[1:] != tuple(reversed(expected_mask_shape)):
        raise ContractError(
            "official nnU-Net probability tensor is not CZYX on the mask grid"
        )
    foreground = np.ascontiguousarray(
        probabilities[1].transpose(2, 1, 0), dtype=np.float32
    )
    if (
        not np.isfinite(foreground).all()
        or float(foreground.min()) < 0.0
        or float(foreground.max()) > 1.0
    ):
        raise ContractError("official foreground probability is not finite in [0,1]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        np.savez_compressed(stream, foreground_probability=foreground)


def handoff_fold_actual_validation(
    run_root: Path,
    fold: int,
    *,
    plan_validator: Callable[[Path, int], tuple[Path, dict[str, Any]]] = (
        validate_fold_plan_binding
    ),
    fold_committer: Callable[..., dict[str, Any]] = commit_fold_done,
) -> dict[str, Any]:
    """Materialize frozen nnU-Net actual-validation outputs as the OOF fold.

    This is a provenance-preserving CPU/I/O handoff, not a second inference.
    The mask bytes are copied unchanged and the official two-class CZYX softmax
    is deterministically reduced to the existing OOF foreground XYZ contract.
    """

    run_root = run_root.resolve()
    _, plan = plan_validator(run_root, fold)
    if plan.get("schema_version") != OOF_CONTRACT_VERSION:
        raise ContractError("OOF handoff plan schema mismatch")
    val_case_ids = list(plan.get("val_case_ids", []))
    if not val_case_ids or len(val_case_ids) != len(set(val_case_ids)):
        raise ContractError("OOF handoff plan contains missing/duplicate val cases")
    if set(val_case_ids) & set(plan.get("train_case_ids", [])):
        raise ContractError("OOF handoff plan has train/val overlap")
    by_case = {item.get("case_id"): item for item in plan.get("cases", [])}
    if set(by_case) != set(val_case_ids):
        raise ContractError("OOF handoff cases are not exactly the held-out fold")

    model = plan.get("model", {})
    full_ready_path = _validate_external_record(
        model.get("full_train_ready"), "FULL_TRAIN_READY"
    )
    fold_receipt_path = _validate_external_record(
        model.get("fold_receipt"), f"fold {fold} training receipt"
    )
    full_ready = _load_json(full_ready_path, label="FULL_TRAIN_READY")
    training_contract = full_ready.get("training_contract")
    if (
        full_ready.get("status") != "COMMITTED"
        or full_ready.get("full_training_status") != "PASS"
        or full_ready.get("oof_handoff_inputs_present") is not True
        or full_ready.get("actual_inference_gate_required") is not False
        or not isinstance(training_contract, dict)
        or training_contract.get("actual_validation") is not True
        or training_contract.get("export_probabilities") is not True
    ):
        raise ContractError(
            "FULL_TRAIN_READY does not authorize actual-validation OOF handoff"
        )
    campaign_raw = full_ready.get("campaign_root")
    if not isinstance(campaign_raw, str) or not campaign_raw:
        raise ContractError("FULL_TRAIN_READY campaign_root is missing")
    campaign_candidate = Path(campaign_raw)
    if campaign_candidate.is_symlink() or not campaign_candidate.is_dir():
        raise ContractError("frozen training campaign root is unavailable")
    campaign_root = campaign_candidate.resolve()

    fold_receipt = _load_json(
        fold_receipt_path, label=f"fold {fold} training receipt"
    )
    output_contract = fold_receipt.get("output_contract")
    if (
        fold_receipt.get("status") != "COMMITTED"
        or fold_receipt.get("fold") != fold
        or not isinstance(output_contract, dict)
        or output_contract.get("status") != "PASS"
        or output_contract.get("fold") != fold
        or output_contract.get("actual_validation") is not True
        or output_contract.get("export_probabilities") is not True
        or output_contract.get("oof_handoff_inputs_present") is not True
        or output_contract.get("validation_case_count") != len(val_case_ids)
        or output_contract.get("validation_probability_count") != len(val_case_ids)
    ):
        raise ContractError(
            f"fold {fold} receipt does not contain a complete actual-validation handoff"
        )
    artifacts = output_contract.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContractError(f"fold {fold} output artifacts are missing")
    summary_path = _resolve_campaign_record(
        artifacts.get("validation_summary"),
        campaign_root,
        label=f"fold {fold} validation summary",
    )
    expected_masks = {f"{case_id}.nii.gz" for case_id in val_case_ids}
    expected_probabilities = {f"{case_id}.npz" for case_id in val_case_ids}
    expected_properties = {f"{case_id}.pkl" for case_id in val_case_ids}
    source_masks = _artifact_index(
        artifacts.get("validation_masks"),
        campaign_root,
        expected_names=expected_masks,
        label=f"fold {fold} validation mask",
    )
    source_probabilities = _artifact_index(
        artifacts.get("validation_probabilities"),
        campaign_root,
        expected_names=expected_probabilities,
        label=f"fold {fold} validation probability",
    )
    source_properties = _artifact_index(
        artifacts.get("validation_properties"),
        campaign_root,
        expected_names=expected_properties,
        label=f"fold {fold} validation properties",
    )

    masks, probability_root = _fresh_output_directories(run_root, fold)
    case_bindings: list[dict[str, Any]] = []
    for case_id in val_case_ids:
        source_mask, _ = source_masks[f"{case_id}.nii.gz"]
        source_probability, _ = source_probabilities[f"{case_id}.npz"]
        source_property, _ = source_properties[f"{case_id}.pkl"]
        destination_mask = masks / f"{case_id}.nii.gz"
        with source_mask.open("rb") as source, destination_mask.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        if _sha256(destination_mask) != _sha256(source_mask):
            raise ContractError(f"fold {fold} {case_id} mask handoff changed bytes")
        try:
            mask_shape = tuple(int(value) for value in nib.load(str(destination_mask)).shape)
        except (OSError, ValueError) as exc:
            raise ContractError(
                f"fold {fold} {case_id} actual-validation mask is unreadable"
            ) from exc
        if len(mask_shape) != 3:
            raise ContractError(
                f"fold {fold} {case_id} actual-validation mask is not 3D"
            )
        destination_probability = probability_root / f"{case_id}.npz"
        _extract_foreground_probability(
            source_probability,
            destination_probability,
            expected_mask_shape=mask_shape,
        )
        case_bindings.append(
            {
                "case_id": case_id,
                "source_mask": _record(source_mask),
                "source_probability": _record(source_probability),
                "source_properties": _record(source_property),
                "oof_mask": _record(destination_mask),
                "oof_foreground_probability": _record(destination_probability),
                "mask_transform": "BYTE_IDENTICAL_COPY",
                "probability_transform": (
                    "NNUNET_SOFTMAX_CZYX_CLASS1_TO_FLOAT32_XYZ_FOREGROUND"
                ),
            }
        )

    handoff_path = run_root / "outputs" / f"fold_{fold}" / "HANDOFF_SOURCE.json"
    handoff = {
        "schema_version": ACTUAL_VALIDATION_HANDOFF_SCHEMA,
        "status": "PASS",
        "prediction_source": HANDOFF_PREDICTION_SOURCE,
        "fold": fold,
        "case_count": len(val_case_ids),
        "val_case_ids": val_case_ids,
        "full_train_ready": _record(full_ready_path),
        "training_fold_receipt": _record(fold_receipt_path),
        "validation_summary": _record(summary_path),
        "case_bindings": case_bindings,
        "new_inference_executed": False,
    }
    _write_json_exclusive(handoff_path, handoff)
    handoff_record = _record(handoff_path)
    receipt = fold_committer(
        run_root,
        fold,
        prediction_source=HANDOFF_PREDICTION_SOURCE,
        source_artifacts={"handoff_source": handoff_record},
    )
    return {
        "status": "COMMITTED",
        "fold": fold,
        "prediction_count": receipt["prediction_count"],
        "prediction_source": HANDOFF_PREDICTION_SOURCE,
        "new_inference_executed": False,
        "handoff_source_sha256": handoff_record["sha256"],
    }


def execute_fold(
    run_root: Path,
    fold: int,
    *,
    device: str = "cuda",
    predictor_factory: Callable[[str], Any] = _official_predictor_factory,
    runtime_validator: Callable[[Path], dict[str, Any]] | None = None,
    plan_validator: Callable[[Path, int], tuple[Path, dict[str, Any]]] = (
        validate_fold_plan_binding
    ),
    fold_committer: Callable[[Path, int], dict[str, Any]] = commit_fold_done,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Run one held-out fold and commit its exact val-only completion receipt."""

    run_root = run_root.resolve()
    _, plan = plan_validator(run_root, fold)
    if plan.get("schema_version") != OOF_CONTRACT_VERSION:
        raise ContractError("OOF execution plan schema mismatch")
    if len(plan.get("val_case_ids", [])) != len(set(plan.get("val_case_ids", []))):
        raise ContractError("OOF execution plan contains duplicate val cases")
    if set(plan.get("val_case_ids", [])) & set(plan.get("train_case_ids", [])):
        raise ContractError("OOF execution plan has train/val overlap")
    if os.environ.get("nnUNet_compile") != INFERENCE_COMPILE_CONTRACT["value"]:
        raise ContractError("OOF inference requires nnUNet_compile=false")
    by_case = {item.get("case_id"): item for item in plan.get("cases", [])}
    if set(by_case) != set(plan.get("val_case_ids", [])):
        raise ContractError("OOF execution case records are not exactly val cases")

    model = plan.get("model", {})
    _validate_external_record(model.get("full_train_ready"), "FULL_TRAIN_READY")
    _validate_external_record(model.get("fold_receipt"), f"fold {fold} receipt")
    checkpoint = _validate_external_record(model.get("checkpoint"), "checkpoint")
    plans = _validate_external_record(model.get("plans"), "plans")
    dataset_json = _validate_external_record(model.get("dataset_json"), "dataset_json")
    trainer_root = Path(model.get("model_training_output_dir", "")).resolve()
    if checkpoint.parent != trainer_root / f"fold_{fold}":
        raise ContractError("OOF checkpoint does not belong to its held-out fold")
    if plans.parent != trainer_root or dataset_json.parent != trainer_root:
        raise ContractError("OOF model metadata root mismatch")
    if model.get("source_tree_sha256") != EXPECTED_NNUNET_SOURCE_TREE_SHA256:
        raise ContractError("OOF execution source tree hash mismatch")
    if runtime_validator is not None:
        if source_root is None:
            raise ContractError("source_root is required for live runtime validation")
        runtime = runtime_validator(source_root.resolve())
        if runtime.get("source_tree_sha256") != EXPECTED_NNUNET_SOURCE_TREE_SHA256:
            raise ContractError("live nnUNet runtime source tree mismatch")

    masks, probability_root = _fresh_output_directories(run_root, fold)
    inputs: list[list[str]] = []
    outputs: list[str] = []
    for case_id in plan["val_case_ids"]:
        case = by_case[case_id]
        ct = _validate_external_record(case.get("input_ct"), f"{case_id} CT")
        pet = _validate_external_record(case.get("input_pet"), f"{case_id} PET")
        _validate_external_record(case.get("input_gt"), f"{case_id} GT")
        inputs.append([str(ct), str(pet)])
        outputs.append(str(masks / case_id))

    predictor = predictor_factory(device)
    predictor.initialize_from_trained_model_folder(
        str(trainer_root), use_folds=(fold,), checkpoint_name="checkpoint_final.pth"
    )
    predictor.predict_from_files(
        inputs,
        outputs,
        save_probabilities=True,
        overwrite=False,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=4,
    )

    for case_id in plan["val_case_ids"]:
        raw_probability = masks / f"{case_id}.npz"
        properties = masks / f"{case_id}.pkl"
        mask = masks / f"{case_id}.nii.gz"
        if mask.is_symlink() or not mask.is_file():
            raise ContractError(
                f"official prediction did not produce mask for {case_id}"
            )
        try:
            mask_shape = tuple(int(value) for value in nib.load(str(mask)).shape)
        except (OSError, ValueError) as exc:
            raise ContractError(
                f"official prediction mask is unreadable for {case_id}"
            ) from exc
        if len(mask_shape) != 3:
            raise ContractError(f"official prediction mask is not 3D for {case_id}")
        _extract_foreground_probability(
            raw_probability,
            probability_root / f"{case_id}.npz",
            expected_mask_shape=mask_shape,
        )
        # These are exact run-owned intermediate files emitted by the official
        # exporter; the final OOF contract retains only mask + foreground NPZ.
        raw_probability.unlink()
        if properties.is_file() and not properties.is_symlink():
            properties.unlink()
    for name in OFFICIAL_METADATA:
        metadata = masks / name
        if metadata.is_file() and not metadata.is_symlink():
            metadata.unlink()
    leftovers = {item.name for item in masks.iterdir()}
    expected_masks = {f"{case}.nii.gz" for case in plan["val_case_ids"]}
    if leftovers != expected_masks:
        raise ContractError("official prediction left unexpected artifacts")
    receipt = fold_committer(run_root, fold)
    return {
        "status": "COMMITTED",
        "fold": fold,
        "prediction_count": receipt["prediction_count"],
        "prediction_source": DEDICATED_PREDICTION_SOURCE,
        "new_inference_executed": True,
        "compile_contract": INFERENCE_COMPILE_CONTRACT,
        "checkpoint_sha256": _sha256(checkpoint),
        "plans_sha256": _sha256(plans),
        "dataset_json_sha256": _sha256(dataset_json),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("fold", type=int, choices=range(5))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--from-actual-validation",
        action="store_true",
        help="Reuse the frozen fold actual-validation outputs; do not run inference.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.from_actual_validation:
        if args.source_root is not None:
            raise ContractError(
                "actual-validation handoff must not accept an inference source root"
            )
        receipt = handoff_fold_actual_validation(args.run_root, args.fold)
    else:
        if args.source_root is None:
            raise ContractError("dedicated OOF inference requires --source-root")
        receipt = execute_fold(
            args.run_root,
            args.fold,
            device=args.device,
            source_root=args.source_root,
            runtime_validator=validate_live_nnunet_runtime,
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
