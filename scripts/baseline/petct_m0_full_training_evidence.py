#!/usr/bin/env python3
"""Evidence validation for one completed standard PET/CT nnU-Net fold."""

from __future__ import annotations

import math
import numbers
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from validate_petct_m0_preprocess import ContractError, _load_json, _sha256


DATASET_FOLDER = "Dataset901_PSMA_M0_AutoPETVNorm"
CONFIGURATION = "3d_fullres"
FOLDS = (0, 1, 2, 3, 4)
TRAINER = "nnUNetTrainer"
PLANS_IDENTIFIER = "nnUNetPlans"
NUM_EPOCHS = 1000
TRAINER_FOLDER = f"{TRAINER}__{PLANS_IDENTIFIER}__{CONFIGURATION}"


def _require_fields(
    payload: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ContractError(
                f"{label} requires {key}={value!r}; observed {payload.get(key)!r}"
            )


def _record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"required regular file is missing: {path}")
    return {
        "path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _fold_root(campaign_root: Path, fold: int) -> Path:
    return (
        campaign_root
        / "nnUNet_results"
        / DATASET_FOLDER
        / TRAINER_FOLDER
        / f"fold_{fold}"
    )


def _finite_series(logging: dict[str, Any], key: str) -> list[float]:
    values = logging.get(key)
    if not isinstance(values, list) or len(values) != NUM_EPOCHS:
        raise ContractError(f"checkpoint {key} must contain exactly 1000 values")
    converted = []
    for value in values:
        if (
            not isinstance(value, numbers.Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ContractError(f"checkpoint {key} contains a non-finite value")
        converted.append(float(value))
    return converted


def _count_files(roots: Iterable[Path], label: str) -> int:
    count = 0
    for root in roots:
        root = root.resolve()
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ContractError(f"{label} root is not a regular directory")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ContractError(f"{label} root contains a symlink")
            if path.is_file():
                count += 1
    return count


def _training_log_timestamp(path: Path) -> datetime:
    match = re.fullmatch(
        r"training_log_(\d{4})_(\d{1,2})_(\d{1,2})_(\d{2})_(\d{2})_(\d{2})\.txt",
        path.name,
    )
    if match is None:
        raise ContractError(f"official training log name is invalid: {path.name}")
    try:
        return datetime(*(int(value) for value in match.groups()))
    except ValueError as exc:
        raise ContractError(
            f"official training log timestamp is invalid: {path.name}"
        ) from exc


def load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ContractError("checkpoint_final is not a dictionary")
    return payload


def validate_fold_completion(
    campaign_root: Path,
    fold: int,
    split_contract: dict[str, Any],
    *,
    actual_validation: bool,
    export_probabilities: bool,
    checkpoint_loader: Callable[[Path], dict[str, Any]] = load_checkpoint,
    oof_roots: Iterable[Path] = (),
    result_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Validate checkpoints, finite logs and optional actual-validation inputs."""

    if fold not in FOLDS:
        raise ContractError("invalid fold")
    if export_probabilities and not actual_validation:
        raise ContractError("probability export requires actual validation")
    campaign_root = campaign_root.resolve()
    campaign_spec = _load_json(
        campaign_root / "CAMPAIGN_SPEC.json", label="CAMPAIGN_SPEC"
    )
    campaign_contract = campaign_spec.get("training_contract", {})
    if campaign_contract.get("actual_validation") != actual_validation:
        raise ContractError("fold actual-validation mode differs from campaign")
    if campaign_contract.get("export_probabilities") != export_probabilities:
        raise ContractError("fold probability mode differs from campaign")
    fold_root = _fold_root(campaign_root, fold)
    if fold_root.is_symlink() or not fold_root.is_dir():
        raise ContractError(f"fold {fold} output is missing")
    trainer_root = fold_root.parent
    for name in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
        path = trainer_root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ContractError(f"trainer metadata is missing: {name}")

    final_checkpoint = fold_root / "checkpoint_final.pth"
    best_checkpoint = fold_root / "checkpoint_best.pth"
    for path in (final_checkpoint, best_checkpoint):
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ContractError(f"required checkpoint is missing: {path.name}")
    checkpoints = {path.name for path in fold_root.glob("checkpoint_*.pth")}
    if checkpoints != {"checkpoint_final.pth", "checkpoint_best.pth"}:
        raise ContractError(
            "completed fold checkpoint set must be exactly final + best"
        )
    for name in ("progress.png", "debug.json"):
        path = fold_root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ContractError(f"required fold output is missing: {name}")

    logs = sorted(fold_root.glob("training_log_*.txt"), key=_training_log_timestamp)
    if not logs or any(path.is_symlink() or path.stat().st_size == 0 for path in logs):
        raise ContractError("fold requires non-empty official training logs")
    completed_logs: list[Path] = []
    interrupted_logs: list[Path] = []
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="strict")
        if re.search(r"(?i)(?<![A-Za-z])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z])", text):
            raise ContractError("training log contains non-finite values")
        if "Training done." in text:
            completed_logs.append(path)
        else:
            interrupted_logs.append(path)
    if not completed_logs:
        raise ContractError("no official training log records completion")
    completion_log = completed_logs[-1]
    if completion_log != logs[-1]:
        raise ContractError("latest official training log does not record completion")
    log_root = campaign_root / "logs" / f"fold_{fold}"
    console_logs = sorted(log_root.glob("attempt_*.log"))
    if not console_logs or any(
        path.is_symlink() or path.stat().st_size == 0 for path in console_logs
    ):
        raise ContractError("fold console attempt log is missing")
    runtime_receipts = sorted(log_root.glob("attempt_*.runtime.json"))
    if not runtime_receipts:
        raise ContractError("fold compile/runtime receipt is missing")
    runtime_receipt = runtime_receipts[-1]
    paired_console = runtime_receipt.with_name(
        runtime_receipt.name.removesuffix(".runtime.json") + ".log"
    )
    if paired_console not in console_logs:
        raise ContractError("fold runtime receipt has no paired console log")
    runtime_payload = _load_json(runtime_receipt, label="fold runtime receipt")
    _require_fields(
        runtime_payload,
        {
            "status": "FOLD_PROCESS_COMPLETED",
            "dataset_id": 901,
            "configuration": CONFIGURATION,
            "fold": fold,
            "trainer": TRAINER,
            "plans_identifier": PLANS_IDENTIFIER,
            "num_epochs": NUM_EPOCHS,
            "actual_validation": actual_validation,
            "export_probabilities": export_probabilities,
            "compile_contract": campaign_contract.get("compile_contract"),
        },
        "fold runtime receipt",
    )
    runtime_output = runtime_payload.get("output_folder")
    if (
        not isinstance(runtime_output, str)
        or Path(runtime_output).resolve() != fold_root
    ):
        raise ContractError("fold runtime receipt output folder differs from campaign")

    checkpoint = checkpoint_loader(final_checkpoint)
    if checkpoint.get("trainer_name") != TRAINER:
        raise ContractError("checkpoint trainer is not standard nnUNetTrainer")
    if checkpoint.get("current_epoch") != NUM_EPOCHS:
        raise ContractError("checkpoint does not contain 1000 completed epochs")
    weights = checkpoint.get("network_weights")
    if not isinstance(weights, dict) or not weights:
        raise ContractError("checkpoint network weights are missing")
    logging = checkpoint.get("logging")
    if not isinstance(logging, dict):
        raise ContractError("checkpoint logging is missing")
    train_losses = _finite_series(logging, "train_losses")
    val_losses = _finite_series(logging, "val_losses")
    learning_rates = _finite_series(logging, "lrs")
    starts = _finite_series(logging, "epoch_start_timestamps")
    ends = _finite_series(logging, "epoch_end_timestamps")
    if any(end < start for start, end in zip(starts, ends)):
        raise ContractError("checkpoint epoch timestamps are reversed")

    fold_split = split_contract.get("folds", {}).get(str(fold))
    if not isinstance(fold_split, dict) or not isinstance(fold_split.get("val"), list):
        raise ContractError("split contract is missing requested fold")
    val_ids = fold_split["val"]
    validation_case_count = 0
    probability_count = 0
    summary_path: Path | None = None
    validation_masks: list[Path] = []
    validation_probabilities: list[Path] = []
    validation_properties: list[Path] = []
    validation_root = fold_root / "validation"
    if actual_validation:
        if validation_root.is_symlink() or not validation_root.is_dir():
            raise ContractError("actual validation output is missing")
        expected_masks = {f"{identifier}.nii.gz" for identifier in val_ids}
        observed_masks = {path.name for path in validation_root.glob("*.nii.gz")}
        if observed_masks != expected_masks:
            raise ContractError("actual validation masks do not match the fold split")
        validation_masks = sorted(validation_root / name for name in expected_masks)
        validation_case_count = len(observed_masks)
        expected_npz = {f"{identifier}.npz" for identifier in val_ids}
        expected_pkl = {f"{identifier}.pkl" for identifier in val_ids}
        observed_npz = {path.name for path in validation_root.glob("*.npz")}
        observed_pkl = {path.name for path in validation_root.glob("*.pkl")}
        if export_probabilities:
            if observed_npz != expected_npz or observed_pkl != expected_pkl:
                raise ContractError("validation probability artifacts are incomplete")
            validation_probabilities = sorted(
                validation_root / name for name in expected_npz
            )
            validation_properties = sorted(
                validation_root / name for name in expected_pkl
            )
            probability_count = len(observed_npz)
        elif observed_npz or observed_pkl:
            raise ContractError("unexpected validation probability artifacts")
        summary_path = validation_root / "summary.json"
        summary = _load_json(summary_path, label="validation summary")
        metrics = summary.get("metric_per_case")
        if not isinstance(metrics, list) or len(metrics) != len(val_ids):
            raise ContractError("validation summary case count differs from split")
    elif validation_root.exists() or validation_root.is_symlink():
        raise ContractError(
            "training-only fold unexpectedly contains actual validation"
        )

    if _count_files(oof_roots, "OOF publication"):
        raise ContractError("dedicated OOF publication must remain empty")
    if _count_files(result_roots, "result publication"):
        raise ContractError("dedicated result publication must remain empty")
    artifacts: dict[str, Any] = {
        "trainer_metadata": [
            _record(trainer_root / name, relative_to=campaign_root)
            for name in ("plans.json", "dataset.json", "dataset_fingerprint.json")
        ],
        "checkpoint_final": _record(final_checkpoint, relative_to=campaign_root),
        "checkpoint_best": _record(best_checkpoint, relative_to=campaign_root),
        "progress": _record(fold_root / "progress.png", relative_to=campaign_root),
        "debug": _record(fold_root / "debug.json", relative_to=campaign_root),
        "training_logs": [_record(path, relative_to=campaign_root) for path in logs],
        "completion_training_log": _record(
            completion_log, relative_to=campaign_root
        ),
        "historical_interrupted_training_logs": [
            _record(path, relative_to=campaign_root) for path in interrupted_logs
        ],
        "console_logs": [
            _record(path, relative_to=campaign_root) for path in console_logs
        ],
        "runtime_receipts": [
            _record(path, relative_to=campaign_root) for path in runtime_receipts
        ],
    }
    if summary_path is not None:
        artifacts["validation_summary"] = _record(
            summary_path, relative_to=campaign_root
        )
        artifacts["validation_masks"] = [
            _record(path, relative_to=campaign_root) for path in validation_masks
        ]
        artifacts["validation_probabilities"] = [
            _record(path, relative_to=campaign_root)
            for path in validation_probabilities
        ]
        artifacts["validation_properties"] = [
            _record(path, relative_to=campaign_root) for path in validation_properties
        ]
    return {
        "status": "PASS",
        "fold": fold,
        "epoch_count": NUM_EPOCHS,
        "checkpoint_count": 2,
        "training_log_count": len(logs),
        "historical_interrupted_training_log_count": len(interrupted_logs),
        "actual_validation": actual_validation,
        "export_probabilities": export_probabilities,
        "validation_case_count": validation_case_count,
        "validation_probability_count": probability_count,
        "finite_summary": {
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "final_learning_rate": learning_rates[-1],
        },
        "split_sha256": split_contract["sha256"],
        "oof_handoff_inputs_present": actual_validation and export_probabilities,
        "actual_inference_gate_required": not (
            actual_validation and export_probabilities
        ),
        "oof_publication_count": 0,
        "result_publication_count": 0,
        "artifacts": artifacts,
    }
