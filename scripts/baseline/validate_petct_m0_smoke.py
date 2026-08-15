#!/usr/bin/env python3
"""Validate and atomically publish the PET/CT M0 one-epoch smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from prepare_nnunet_m0_dataset import commit_run_directory
from validate_petct_m0_preprocess import (
    CONTRACT_VERSION as PREPROCESS_CONTRACT_VERSION,
    ContractError,
    _load_json,
    _sha256,
    _verify_record,
    _write_json_exclusive,
    validate_planning_ready,
    validate_preprocessed_output,
)


DATASET_ID = 901
DATASET_FOLDER = "Dataset901_PSMA_M0_AutoPETVNorm"
CONFIGURATION = "3d_fullres"
FOLD = 0
TRAINER = "nnUNetTrainer_1epoch"
PLANS_IDENTIFIER = "nnUNetPlans"
CONTRACT_VERSION = "1.0.0"
PHASE = "FOLD0_1EPOCH_SMOKE"
TRAINER_FOLDER = f"{TRAINER}__{PLANS_IDENTIFIER}__{CONFIGURATION}"
CUDA_DRIVER_LINK_STUB = (
    "/usr/local/cuda-11.6/targets/x86_64-linux/lib/stubs/libcuda.so"
)
CUDA_DRIVER_LINK_STUB_SHA256 = (
    "81dcabbb572826da2e9e5edcffb7ca98a1d4728f38a3892a4999dea74716f198"
)
CUDA_DRIVER_LINK_STUB_BYTES = 58080
TRAINING_CONTRACT = {
    "dataset_id": DATASET_ID,
    "configuration": CONFIGURATION,
    "fold": FOLD,
    "trainer": TRAINER,
    "plans_identifier": PLANS_IDENTIFIER,
    "device": "cuda",
    "actual_validation": False,
    "export_probabilities": False,
    "nnunet_compile": True,
    "cuda_driver_link_mode": "LIBRARY_PATH_COMPILE_ONLY",
    "cuda_driver_stub_in_ld_library_path": False,
    "cuda_driver_link_stub": {
        "path": CUDA_DRIVER_LINK_STUB,
        "bytes": CUDA_DRIVER_LINK_STUB_BYTES,
        "sha256": CUDA_DRIVER_LINK_STUB_SHA256,
    },
}


def _require_fields(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ContractError(
                f"{label} requires {key}={value!r}; observed {payload.get(key)!r}"
            )


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"required regular file is missing: {path}")
    display = str(path.resolve())
    if relative_to is not None:
        display = path.resolve().relative_to(relative_to.resolve()).as_posix()
    return {"path": display, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _resolve_run_record(record: dict[str, Any], run_dir: Path, label: str) -> Path:
    raw = record.get("path") if isinstance(record, dict) else None
    if not isinstance(raw, str) or not raw:
        raise ContractError(f"{label} record path is missing")
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        if ".." in candidate.parts:
            raise ContractError(f"unsafe {label} relative path")
        resolved = (run_dir / candidate).resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        raise ContractError(f"{label} escapes its committed run")
    return resolved


def _validate_raw_source_bindings(
    run_dir: Path, planning: dict[str, Any]
) -> dict[str, dict[str, str]]:
    """Prove that the committed raw views remain the planning-owned sources.

    nnU-Net preprocessing deliberately keeps ``imagesTr`` and ``labelsTr`` as
    directory symlinks.  They are not arbitrary output-tree symlinks: their
    exact targets are frozen by the validated planning receipt.  Downstream
    OOF staging consumes this explicit binding instead of applying a blanket
    symlink rejection that contradicts the native preprocessing layout.
    """

    raw_sources = planning.get("raw_source_paths")
    if not isinstance(raw_sources, dict):
        raise ContractError("validated planning receipt has no raw_source_paths")
    raw_dataset = run_dir / "nnUNet_raw" / DATASET_FOLDER
    bindings: dict[str, dict[str, str]] = {}
    for name in ("imagesTr", "labelsTr"):
        raw_target = raw_sources.get(name)
        if not isinstance(raw_target, str) or not raw_target:
            raise ContractError(f"planning raw_source_paths is missing {name}")
        target = Path(raw_target)
        if target.is_symlink() or not target.is_dir():
            raise ContractError(f"planning {name} source is not a regular directory")
        target = target.resolve()
        link = raw_dataset / name
        if not link.is_symlink():
            raise ContractError(f"committed raw {name} must remain a directory symlink")
        if not link.resolve().is_dir() or link.resolve() != target:
            raise ContractError(
                f"committed raw {name} symlink no longer targets planning raw_source_paths"
            )
        bindings[name] = {
            "link_path": str(link),
            "target_path": str(target),
            "policy": "PLANNING_RECEIPT_BOUND_DIRECTORY_SYMLINK",
        }
    return bindings


def validate_preprocess_ready(
    ready_path: Path,
    *,
    output_validator: Callable[[Path], dict[str, Any]] = validate_preprocessed_output,
    planning_validator: Callable[[Path], dict[str, Any]] = validate_planning_ready,
    raw_source_validator: Callable[
        [Path, dict[str, Any]], dict[str, dict[str, str]]
    ] = _validate_raw_source_bindings,
) -> dict[str, Any]:
    """Rehash PREPROCESS_READY and revalidate its bound planning/output state."""

    ready_path = ready_path.resolve()
    ready = _load_json(ready_path, label="PREPROCESS_READY")
    _require_fields(
        ready,
        {
            "status": "COMMITTED",
            "preprocessing_status": "PASS",
            "contract_version": PREPROCESS_CONTRACT_VERSION,
            "phase": "PREPROCESSING_ONLY",
            "training_status": "NOT_STARTED",
            "training_performed": False,
            "checkpoint_count": 0,
            "oof_prediction_count": 0,
            "result_count": 0,
        },
        "PREPROCESS_READY",
    )
    run_dir_raw = ready.get("run_dir")
    if not isinstance(run_dir_raw, str):
        raise ContractError("PREPROCESS_READY run_dir is missing")
    run_dir = Path(run_dir_raw).resolve()
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise ContractError("PREPROCESS_READY committed run directory is missing")
    if ready.get("run_id") != run_dir.name:
        raise ContractError("PREPROCESS_READY run identity mismatch")

    bundle_record = ready.get("run_receipt")
    bundle_path = _resolve_run_record(bundle_record, run_dir, "preprocessing bundle")
    _verify_record(bundle_path, bundle_record, label="preprocessing bundle")
    bundle = _load_json(bundle_path, label="PREPROCESSING_BUNDLE")
    if ready.get("validated_bundle") != bundle:
        raise ContractError("PREPROCESS_READY embedded bundle differs from its hashed receipt")
    _require_fields(
        bundle,
        {
            "status": "VALIDATED",
            "preprocessing_status": "PASS",
            "contract_version": PREPROCESS_CONTRACT_VERSION,
            "phase": "PREPROCESSING_ONLY",
            "run_id": run_dir.name,
            "committed_run_dir": str(run_dir),
            "training_status": "NOT_STARTED",
            "training_performed": False,
            "checkpoint_count": 0,
            "oof_prediction_count": 0,
            "result_count": 0,
        },
        "PREPROCESSING_BUNDLE",
    )

    planning_record = bundle.get("planning_ready")
    planning_raw = planning_record.get("path") if isinstance(planning_record, dict) else None
    if not isinstance(planning_raw, str):
        raise ContractError("PREPROCESSING_BUNDLE planning receipt is missing")
    planning_path = Path(planning_raw).resolve()
    _verify_record(planning_path, planning_record, label="PLANNING_READY")
    planning = planning_validator(planning_path)
    if bundle.get("planning_bound_hashes") != planning.get("bound_hashes"):
        raise ContractError("planning hashes changed after preprocessing")

    current_output = output_validator(run_dir)
    if current_output != bundle.get("output_contract"):
        raise ContractError("preprocessed output changed after PREPROCESS_READY publication")
    raw_root = run_dir / "nnUNet_raw"
    preprocessed_root = run_dir / "nnUNet_preprocessed"
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise ContractError("committed preprocessing run has no regular nnUNet_raw root")
    if not preprocessed_root.is_dir() or preprocessed_root.is_symlink():
        raise ContractError("committed preprocessing run has no regular nnUNet_preprocessed root")
    raw_source_bindings = raw_source_validator(run_dir, planning)

    return {
        "status": "PASS",
        "preprocess_run_dir": str(run_dir),
        "nnunet_raw": str(raw_root.resolve()),
        "nnunet_preprocessed": str(preprocessed_root.resolve()),
        "raw_source_bindings": raw_source_bindings,
        "output_contract": current_output,
        "bound_hashes": {
            "preprocess_ready": _sha256(ready_path),
            "preprocessing_bundle": _sha256(bundle_path),
            **{
                f"planning_{key}": value
                for key, value in planning.get("bound_hashes", {}).items()
            },
        },
    }


def stage_smoke_run(
    preprocess_ready_path: Path,
    staging_root: Path,
    final_root: Path,
    run_id: str,
    *,
    gpu_id: str,
    preprocess_validator: Callable[[Path], dict[str, Any]] = validate_preprocess_ready,
) -> dict[str, Any]:
    """Create a fresh run-owned smoke directory without touching training outputs."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ContractError("unsafe smoke run_id")
    if not re.fullmatch(r"[0-9]+", gpu_id):
        raise ContractError("visible GPU id must be a non-negative integer")
    staging_root = staging_root.resolve()
    final_root = final_root.resolve()
    if not staging_root.is_dir() or staging_root.is_symlink():
        raise ContractError("staging root must be a fresh regular directory")
    if staging_root.name != f".partial-{run_id}":
        raise ContractError("staging directory does not match run identity")
    if staging_root.parent != final_root.parent or final_root.name != run_id:
        raise ContractError("staging and committed destination must be sibling run paths")
    if any(staging_root.iterdir()):
        raise ContractError("smoke staging root must be empty")
    if os.path.lexists(final_root):
        raise FileExistsError(f"refusing existing smoke destination: {final_root}")

    preprocess = preprocess_validator(preprocess_ready_path)
    results_root = staging_root / "nnUNet_results"
    results_root.mkdir()
    owner = {
        "status": "OWNED",
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "staging_dir_name": staging_root.name,
        "owner_token": uuid4().hex,
    }
    _write_json_exclusive(staging_root / "RUN_OWNER.json", owner)
    contract = {**TRAINING_CONTRACT, "visible_gpu_id": gpu_id}
    spec = {
        "status": "STAGED",
        "contract_version": CONTRACT_VERSION,
        "phase": PHASE,
        "run_id": run_id,
        "committed_run_dir": str(final_root),
        "preprocess_ready": _record(preprocess_ready_path),
        "preprocess_bound_hashes": preprocess["bound_hashes"],
        "preprocess_run_dir": preprocess["preprocess_run_dir"],
        "training_contract": contract,
        "smoke_training_status": "NOT_STARTED",
        "full_training_status": "NOT_STARTED",
        "oof_prediction_count": 0,
        "result_count": 0,
    }
    _write_json_exclusive(staging_root / "SMOKE_SPEC.json", spec)
    return spec


def _official_checkpoint_loader(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ContractError("checkpoint_final is not a dictionary")
    return payload


def _finite_single(logging: dict[str, Any], key: str, label: str) -> float:
    values = logging.get(key)
    if not isinstance(values, list) or len(values) != 1:
        raise ContractError(f"checkpoint must contain exactly one {label}")
    value = values[0]
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or not math.isfinite(value):
        raise ContractError(f"checkpoint {label} is not finite")
    return float(value)


def _count_publication_files(roots: Iterable[Path], *, label: str) -> int:
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


def validate_smoke_output(
    run_root: Path,
    *,
    checkpoint_loader: Callable[[Path], dict[str, Any]] = _official_checkpoint_loader,
    oof_roots: Iterable[Path] = (),
    result_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Validate the exact one-epoch output and reject any OOF/result publication."""

    run_root = run_root.resolve()
    if run_root.is_symlink() or not run_root.is_dir():
        raise ContractError("smoke run root must be a regular directory")
    allowed_root = {
        "RUN_OWNER.json",
        "SMOKE_SPEC.json",
        "SMOKE_BUNDLE.json",
        "console.log",
        "nnUNet_results",
    }
    unexpected = {item.name for item in run_root.iterdir()} - allowed_root
    if unexpected:
        raise ContractError(f"smoke run root contains unexpected outputs: {sorted(unexpected)}")

    owner = _load_json(run_root / "RUN_OWNER.json", label="RUN_OWNER")
    run_id = owner.get("run_id")
    if (
        owner.get("status") != "OWNED"
        or not isinstance(run_id, str)
        or run_root.name not in {run_id, f".partial-{run_id}"}
    ):
        raise ContractError("smoke run owner identity mismatch")
    spec = _load_json(run_root / "SMOKE_SPEC.json", label="SMOKE_SPEC")
    _require_fields(
        spec,
        {
            "status": "STAGED",
            "contract_version": CONTRACT_VERSION,
            "phase": PHASE,
            "run_id": run_id,
            "full_training_status": "NOT_STARTED",
            "oof_prediction_count": 0,
            "result_count": 0,
        },
        "SMOKE_SPEC",
    )
    committed_raw = spec.get("committed_run_dir")
    if not isinstance(committed_raw, str):
        raise ContractError("SMOKE_SPEC committed run directory is missing")
    committed_run = Path(committed_raw).resolve()
    if committed_run.name != run_id or committed_run.parent != run_root.parent:
        raise ContractError("SMOKE_SPEC committed run identity mismatch")
    if run_root.name == run_id and committed_run != run_root:
        raise ContractError("committed smoke run path differs from SMOKE_SPEC")
    contract = spec.get("training_contract")
    if not isinstance(contract, dict):
        raise ContractError("SMOKE_SPEC training contract is missing")
    for key, value in TRAINING_CONTRACT.items():
        if contract.get(key) != value:
            raise ContractError(f"SMOKE_SPEC training contract drift: {key}")

    results_root = run_root / "nnUNet_results"
    dataset_root = results_root / DATASET_FOLDER
    trainer_root = dataset_root / TRAINER_FOLDER
    fold_root = trainer_root / f"fold_{FOLD}"
    if not fold_root.is_dir() or fold_root.is_symlink():
        raise ContractError("exact Dataset901/3d_fullres/fold_0 output is missing")
    if {p.name for p in results_root.iterdir()} != {DATASET_FOLDER}:
        raise ContractError("nnUNet_results contains a non-Dataset901 output")
    if {p.name for p in dataset_root.iterdir()} != {TRAINER_FOLDER}:
        raise ContractError("Dataset901 contains a non-smoke trainer output")
    trainer_allowed = {
        f"fold_{FOLD}",
        "plans.json",
        "dataset.json",
        "dataset_fingerprint.json",
    }
    if not {p.name for p in trainer_root.iterdir()}.issubset(trainer_allowed):
        raise ContractError("trainer output contains an unexpected configuration artifact")
    if (fold_root / "validation").exists() or (fold_root / "validation").is_symlink():
        raise ContractError("smoke must not run actual validation or create predictions")

    console_log = run_root / "console.log"
    if console_log.is_symlink() or not console_log.is_file() or console_log.stat().st_size == 0:
        raise ContractError("smoke console log is missing or empty")
    logs = list(fold_root.glob("training_log_*.txt"))
    if len(logs) != 1 or logs[0].is_symlink() or logs[0].stat().st_size == 0:
        raise ContractError("smoke requires exactly one non-empty training log")
    training_log = logs[0]
    log_text = training_log.read_text(encoding="utf-8", errors="strict")
    if re.search(r"(?i)(?<![A-Za-z])(?:nan|[+-]?inf(?:inity)?)(?![A-Za-z])", log_text):
        raise ContractError("training log contains non-finite values")
    for marker in ("Epoch 0", "train_loss", "val_loss", "Epoch time"):
        if marker not in log_text:
            raise ContractError(f"training log is missing marker: {marker}")

    final_checkpoint = fold_root / "checkpoint_final.pth"
    best_checkpoint = fold_root / "checkpoint_best.pth"
    for checkpoint in (final_checkpoint, best_checkpoint):
        if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise ContractError(f"required {checkpoint.name} is missing or empty")
    checkpoints = list(fold_root.glob("checkpoint_*.pth"))
    if {path.name for path in checkpoints} != {
        "checkpoint_final.pth",
        "checkpoint_best.pth",
    }:
        raise ContractError("smoke checkpoint set must be exactly final + best")
    for required in (fold_root / "progress.png", fold_root / "debug.json"):
        if required.is_symlink() or not required.is_file() or required.stat().st_size == 0:
            raise ContractError(f"required smoke output is missing or empty: {required.name}")

    checkpoint = checkpoint_loader(final_checkpoint)
    if checkpoint.get("trainer_name") != TRAINER:
        raise ContractError("checkpoint trainer_name is not nnUNetTrainer_1epoch")
    if checkpoint.get("current_epoch") != 1:
        raise ContractError("checkpoint does not record exactly one completed epoch")
    weights = checkpoint.get("network_weights")
    if not isinstance(weights, dict) or not weights:
        raise ContractError("checkpoint network weights are missing")
    logging = checkpoint.get("logging")
    if not isinstance(logging, dict):
        raise ContractError("checkpoint logging payload is missing")
    train_loss = _finite_single(logging, "train_losses", "finite train loss")
    val_loss = _finite_single(logging, "val_losses", "finite validation loss")
    learning_rate = _finite_single(logging, "lrs", "finite learning rate")
    start = _finite_single(logging, "epoch_start_timestamps", "finite epoch start")
    end = _finite_single(logging, "epoch_end_timestamps", "finite epoch end")
    if end < start:
        raise ContractError("checkpoint epoch timestamps are reversed")

    oof_count = _count_publication_files(oof_roots, label="OOF publication")
    if oof_count != 0:
        raise ContractError("OOF publication must remain empty after smoke")
    result_count = _count_publication_files(result_roots, label="result publication")
    if result_count != 0:
        raise ContractError("result publication must remain empty after smoke")

    required_files = {
        "checkpoint_final": final_checkpoint,
        "checkpoint_best": best_checkpoint,
        "training_log": training_log,
        "console_log": console_log,
        "progress": fold_root / "progress.png",
        "debug": fold_root / "debug.json",
    }
    return {
        "status": "PASS",
        "training_contract": contract,
        "fold_output_dir": fold_root.relative_to(run_root).as_posix(),
        "epoch_count": 1,
        "checkpoint_count": 2,
        "finite": {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate,
        },
        "epoch_duration_seconds": end - start,
        "actual_validation_output_count": 0,
        "oof_prediction_count": 0,
        "result_count": 0,
        "artifacts": {
            label: _record(path, relative_to=run_root)
            for label, path in required_files.items()
        },
    }


def build_smoke_bundle(
    preprocess_ready_path: Path,
    *,
    run_id: str,
    committed_run_dir: Path,
    inventory: dict[str, Any],
    preprocess_validator: Callable[[Path], dict[str, Any]] = validate_preprocess_ready,
) -> dict[str, Any]:
    preprocess = preprocess_validator(preprocess_ready_path)
    committed_run_dir = committed_run_dir.resolve()
    if committed_run_dir.name != run_id:
        raise ContractError("smoke bundle run identity mismatch")
    if inventory.get("status") != "PASS":
        raise ContractError("smoke output inventory is not PASS")
    if inventory.get("oof_prediction_count") != 0 or inventory.get("result_count") != 0:
        raise ContractError("smoke bundle cannot contain OOF or result publication")
    return {
        "status": "VALIDATED",
        "smoke_status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "phase": PHASE,
        "run_id": run_id,
        "committed_run_dir": str(committed_run_dir),
        "preprocess_ready": _record(preprocess_ready_path),
        "preprocess_bound_hashes": preprocess["bound_hashes"],
        "training_contract": inventory["training_contract"],
        "output_contract": inventory,
        "smoke_training_performed": True,
        "full_training_status": "NOT_STARTED",
        "full_training_performed": False,
        "checkpoint_count": inventory["checkpoint_count"],
        "oof_prediction_count": 0,
        "result_count": 0,
        "thesis_citable": False,
    }


def publish_smoke_ready(
    run_dir: Path,
    bundle_path: Path,
    ready_path: Path,
    *,
    output_validator: Callable[[Path], dict[str, Any]] = validate_smoke_output,
    preprocess_validator: Callable[[Path], dict[str, Any]] = validate_preprocess_ready,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    bundle_path = bundle_path.resolve()
    if bundle_path.parent != run_dir:
        raise ContractError("SMOKE_BUNDLE must be inside its committed run")
    bundle_bytes = bundle_path.read_bytes()
    bundle = _load_json(bundle_path, label="SMOKE_BUNDLE")
    _require_fields(
        bundle,
        {
            "status": "VALIDATED",
            "smoke_status": "PASS",
            "contract_version": CONTRACT_VERSION,
            "phase": PHASE,
            "run_id": run_dir.name,
            "committed_run_dir": str(run_dir),
            "smoke_training_performed": True,
            "full_training_status": "NOT_STARTED",
            "full_training_performed": False,
            "oof_prediction_count": 0,
            "result_count": 0,
            "thesis_citable": False,
        },
        "SMOKE_BUNDLE",
    )
    preprocess_record = bundle.get("preprocess_ready")
    preprocess_raw = preprocess_record.get("path") if isinstance(preprocess_record, dict) else None
    if not isinstance(preprocess_raw, str):
        raise ContractError("SMOKE_BUNDLE preprocess receipt is missing")
    preprocess_path = Path(preprocess_raw).resolve()
    _verify_record(preprocess_path, preprocess_record, label="PREPROCESS_READY")
    preprocess = preprocess_validator(preprocess_path)
    if preprocess.get("bound_hashes") != bundle.get("preprocess_bound_hashes"):
        raise ContractError("preprocessing hashes changed before smoke publication")
    fresh_output = output_validator(run_dir)
    if fresh_output != bundle.get("output_contract"):
        raise ContractError("smoke output changed before fixed receipt publication")
    bundle_hash = hashlib.sha256(bundle_bytes).hexdigest()
    if _sha256(bundle_path) != bundle_hash:
        raise ContractError("SMOKE_BUNDLE changed during publication")
    published = {
        "status": "COMMITTED",
        "smoke_status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "phase": PHASE,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "run_receipt": {
            "path": str(bundle_path),
            "bytes": len(bundle_bytes),
            "sha256": bundle_hash,
        },
        "smoke_training_performed": True,
        "full_training_status": "NOT_STARTED",
        "full_training_performed": False,
        "checkpoint_count": fresh_output["checkpoint_count"],
        "oof_prediction_count": 0,
        "result_count": 0,
        "thesis_citable": False,
        "validated_bundle": bundle,
    }
    _write_json_exclusive(ready_path, published)
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate_pre = commands.add_parser("validate-preprocess-ready")
    validate_pre.add_argument("receipt", type=Path)
    stage = commands.add_parser("stage")
    stage.add_argument("preprocess_ready", type=Path)
    stage.add_argument("staging_root", type=Path)
    stage.add_argument("final_root", type=Path)
    stage.add_argument("run_id")
    stage.add_argument("gpu_id")
    validate = commands.add_parser("validate-smoke")
    validate.add_argument("--preprocess-ready", type=Path, required=True)
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--committed-run-dir", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--oof-root", type=Path, action="append", default=[])
    validate.add_argument("--result-root", type=Path, action="append", default=[])
    commit = commands.add_parser("commit-run")
    commit.add_argument("staging_dir", type=Path)
    commit.add_argument("final_dir", type=Path)
    commit.add_argument("receipt", type=Path)
    publish = commands.add_parser("publish-smoke-ready")
    publish.add_argument("run_dir", type=Path)
    publish.add_argument("bundle", type=Path)
    publish.add_argument("ready", type=Path)
    publish.add_argument("--oof-root", type=Path, action="append", default=[])
    publish.add_argument("--result-root", type=Path, action="append", default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate-preprocess-ready":
        payload = validate_preprocess_ready(args.receipt)
    elif args.command == "stage":
        payload = stage_smoke_run(
            args.preprocess_ready,
            args.staging_root,
            args.final_root,
            args.run_id,
            gpu_id=args.gpu_id,
        )
    elif args.command == "validate-smoke":
        inventory = validate_smoke_output(
            args.run_root,
            oof_roots=args.oof_root,
            result_roots=args.result_root,
        )
        payload = build_smoke_bundle(
            args.preprocess_ready,
            run_id=args.run_id,
            committed_run_dir=args.committed_run_dir,
            inventory=inventory,
        )
        _write_json_exclusive(args.receipt, payload)
    elif args.command == "commit-run":
        payload = commit_run_directory(args.staging_dir, args.final_dir, args.receipt)
    else:
        payload = publish_smoke_ready(
            args.run_dir,
            args.bundle,
            args.ready,
            output_validator=lambda run: validate_smoke_output(
                run, oof_roots=args.oof_root, result_roots=args.result_root
            ),
        )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
