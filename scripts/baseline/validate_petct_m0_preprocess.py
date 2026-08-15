#!/usr/bin/env python3
"""Validate and publish the PSMA M0 nnU-Net preprocessing-only gate.

This gate consumes the fixed planning-only receipt, stages a new isolated run,
and accepts only the pinned Dataset901/nnUNetPlans/3d_fullres preprocessing
contract.  It never starts training, creates checkpoints, produces OOF masks,
or records experimental results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from prepare_nnunet_m0_dataset import (
    EXPECTED_NNUNET_SOURCE_TREE_SHA256,
    EXPECTED_NNUNET_VERSION,
    commit_run_directory,
    source_tree_sha256,
    validate_fingerprint,
    validate_plan_contract,
)


DATASET_ID = 901
DATASET_FOLDER = "Dataset901_PSMA_M0_AutoPETVNorm"
DATASET_METADATA_NAME = "PSMA_M0_AutoPETVNorm"
EXPECTED_CASES = 597
EXPECTED_CHANNELS = {"0": "CT", "1": "PET"}
PLANS_IDENTIFIER = "nnUNetPlans"
CONFIGURATION = "3d_fullres"
DATA_IDENTIFIER = "nnUNetPlans_3d_fullres"
NUM_PROCESSES = 4
EXPECTED_NORMALIZATION = ["CTNormalization", "ZScoreNormalization"]
EXPECTED_MASKS = [False, False]
CONTRACT_VERSION = "1.0.0"
PLANNING_CONTRACT_VERSION = "2.0.0"
PREPROCESS_API = {
    "dataset_ids": [DATASET_ID],
    "plans_identifier": PLANS_IDENTIFIER,
    "configurations": [CONFIGURATION],
    "num_processes": [NUM_PROCESSES],
}
METADATA_FILES = {
    "dataset_json": "dataset.json",
    "dataset_fingerprint": "dataset_fingerprint.json",
    "nnunet_plans": "nnUNetPlans.json",
    "splits_final": "splits_final.json",
}
PLANNING_ARTIFACT_LABELS = {
    "dataset_json": "preprocessed_dataset_json",
    "dataset_fingerprint": "dataset_fingerprint",
    "nnunet_plans": "nnunet_plans",
    "splits_final": "splits_final",
}


class ContractError(RuntimeError):
    """A planning or preprocessing artifact violates the frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} is missing or is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label} must contain a JSON object")
    return payload


def _file_record(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"required regular file is missing: {path}")
    return {
        "path": display_path if display_path is not None else str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _relative_record(root: Path, path: Path) -> dict[str, Any]:
    return _file_record(path, display_path=path.relative_to(root).as_posix())


def _verify_record(path: Path, record: dict[str, Any], *, label: str) -> None:
    if not isinstance(record, dict):
        raise ContractError(f"{label} record is missing")
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
        raise ContractError(f"{label} hash mismatch")
    expected_bytes = record.get("bytes")
    if not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes:
        raise ContractError(f"{label} byte-size mismatch")


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_record_path(
    record: dict[str, Any], *, root: Path, label: str
) -> Path:
    raw = record.get("path") if isinstance(record, dict) else None
    if not isinstance(raw, str) or not raw:
        raise ContractError(f"{label} record path is missing")
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        if ".." in candidate.parts:
            raise ContractError(f"unsafe {label} relative path")
        resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ContractError(f"{label} escapes its committed run")
    return resolved


def _require_fields(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ContractError(
                f"{label} requires {key}={value!r}; observed {payload.get(key)!r}"
            )


def _validate_dataset_json(payload: dict[str, Any]) -> None:
    if payload.get("channel_names") != EXPECTED_CHANNELS:
        raise ContractError("dataset.json does not preserve the frozen CT/PET channels")
    if payload.get("numTraining") != EXPECTED_CASES:
        raise ContractError("dataset.json does not contain exactly 597 training cases")
    if payload.get("name") != DATASET_METADATA_NAME:
        raise ContractError("dataset.json metadata name mismatch")
    if payload.get("file_ending") != ".nii.gz":
        raise ContractError("dataset.json file ending mismatch")


def validate_planning_ready(
    ready_path: Path, *, source_link_policy: str = "REQUIRE_SYMLINK"
) -> dict[str, Any]:
    """Verify the fixed planning receipt and every preprocessing input hash."""

    ready_path = ready_path.resolve()
    ready = _load_json(ready_path, label="PLANNING_READY")
    _require_fields(
        ready,
        {
            "status": "COMMITTED",
            "planning_status": "PASS",
            "contract_version": PLANNING_CONTRACT_VERSION,
            "phase": "PLANNING_ONLY",
            "preprocessing_status": "NOT_STARTED",
            "preprocessing_performed": False,
            "training_status": "NOT_STARTED",
            "training_performed": False,
        },
        "PLANNING_READY",
    )
    run_dir_raw = ready.get("run_dir")
    if not isinstance(run_dir_raw, str):
        raise ContractError("PLANNING_READY run_dir is missing")
    planning_run = Path(run_dir_raw).resolve()
    if not planning_run.is_dir():
        raise ContractError("PLANNING_READY committed run directory is missing")
    if ready.get("run_id") != planning_run.name:
        raise ContractError("PLANNING_READY run identity mismatch")

    run_receipt_record = ready.get("run_receipt")
    run_receipt = _resolve_record_path(
        run_receipt_record, root=planning_run, label="planning bundle"
    )
    _verify_record(run_receipt, run_receipt_record, label="planning bundle")
    bundle = _load_json(run_receipt, label="PLANNING_BUNDLE")
    if ready.get("validated_bundle") != bundle:
        raise ContractError("PLANNING_READY embedded bundle differs from its hashed receipt")
    _require_fields(
        bundle,
        {
            "status": "VALIDATED",
            "planning_status": "PASS",
            "contract_version": PLANNING_CONTRACT_VERSION,
            "phase": "PLANNING_ONLY",
            "preprocessing_status": "NOT_STARTED",
            "preprocessing_performed": False,
            "training_status": "NOT_STARTED",
            "training_performed": False,
            "run_id": planning_run.name,
            "committed_run_dir": str(planning_run),
        },
        "PLANNING_BUNDLE",
    )
    expected_dataset = {
        "id": DATASET_ID,
        "folder": DATASET_FOLDER,
        "metadata_name": DATASET_METADATA_NAME,
        "source_release": "PSMA-PET-CT-Lesions_v3",
        "scope": "PSMA v3 only",
    }
    if bundle.get("dataset") != expected_dataset:
        raise ContractError("PLANNING_BUNDLE dataset identity mismatch")
    dataset_contract = bundle.get("dataset_contract", {})
    if dataset_contract.get("derived_channel_names") != EXPECTED_CHANNELS:
        raise ContractError("PLANNING_BUNDLE does not bind the frozen CT/PET contract")
    if dataset_contract.get("expected_3d_fullres_normalization") != EXPECTED_NORMALIZATION:
        raise ContractError("PLANNING_BUNDLE dataset normalization contract mismatch")

    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContractError("PLANNING_BUNDLE artifact records are missing")
    artifact_paths: dict[str, Path] = {}
    artifact_records: dict[str, dict[str, Any]] = {}
    for output_label, planning_label in PLANNING_ARTIFACT_LABELS.items():
        record = artifacts.get(planning_label)
        path = _resolve_record_path(
            record, root=planning_run, label=f"planning artifact {planning_label}"
        )
        _verify_record(path, record, label=f"planning artifact {planning_label}")
        artifact_paths[output_label] = path
        artifact_records[output_label] = record

    derived_record = artifacts.get("derived_dataset_json")
    derived_path = _resolve_record_path(
        derived_record, root=planning_run, label="derived dataset.json"
    )
    _verify_record(derived_path, derived_record, label="derived dataset.json")
    if _sha256(derived_path) != _sha256(artifact_paths["dataset_json"]):
        raise ContractError("derived and planned dataset.json hashes differ")
    dataset = _load_json(artifact_paths["dataset_json"], label="planned dataset.json")
    _validate_dataset_json(dataset)
    fingerprint = _load_json(
        artifact_paths["dataset_fingerprint"], label="dataset fingerprint"
    )
    fingerprint_contract = validate_fingerprint(fingerprint)
    if fingerprint_contract != {"case_count": EXPECTED_CASES, "channel_keys": ["0", "1"]}:
        raise ContractError("fingerprint must bind exactly 597 CT/PET cases")
    if bundle.get("fingerprint_contract") != fingerprint_contract:
        raise ContractError("PLANNING_BUNDLE fingerprint contract drift from 597 cases")
    plans = _load_json(artifact_paths["nnunet_plans"], label="nnUNet plans")
    plan_contract = validate_plan_contract(plans)
    expected_plan_fields = {
        "dataset_name": DATASET_FOLDER,
        "plans_name": PLANS_IDENTIFIER,
        "channel_keys": ["0", "1"],
        "data_identifier": DATA_IDENTIFIER,
        "configuration": CONFIGURATION,
        "normalization_schemes": EXPECTED_NORMALIZATION,
        "use_mask_for_norm": EXPECTED_MASKS,
    }
    for key, value in expected_plan_fields.items():
        if plan_contract.get(key) != value:
            name = "mask" if key == "use_mask_for_norm" else key
            raise ContractError(f"frozen plan {name} contract mismatch")
    if bundle.get("plan_contract") != plan_contract:
        embedded_plan = bundle.get("plan_contract", {})
        if embedded_plan.get("normalization_schemes") != EXPECTED_NORMALIZATION:
            raise ContractError("PLANNING_BUNDLE normalization contract drift")
        if embedded_plan.get("use_mask_for_norm") != EXPECTED_MASKS:
            raise ContractError("PLANNING_BUNDLE mask contract drift")
        raise ContractError("PLANNING_BUNDLE plan contract drift")

    raw_dataset = planning_run / "nnUNet_raw" / DATASET_FOLDER
    if source_link_policy not in {"REQUIRE_SYMLINK", "TEST_DIRECTORY_FIXTURE"}:
        raise ContractError("unknown raw source-link validation policy")
    for name in ("imagesTr", "labelsTr"):
        link = raw_dataset / name
        if source_link_policy == "REQUIRE_SYMLINK":
            if not link.is_symlink() or not link.resolve().is_dir():
                raise ContractError(f"planned raw {name} source link is missing")
        elif link.is_symlink() or not link.is_dir():
            raise ContractError(f"test raw {name} fixture is not a regular directory")
    return {
        "status": "PASS",
        "planning_run_dir": str(planning_run),
        "planning_bundle_path": str(run_receipt),
        "dataset": {
            "id": DATASET_ID,
            "folder": DATASET_FOLDER,
            "case_count": EXPECTED_CASES,
            "channel_names": EXPECTED_CHANNELS,
        },
        "preprocess_api": PREPROCESS_API,
        "plan_contract": plan_contract,
        "artifact_paths": {key: str(value) for key, value in artifact_paths.items()},
        "raw_source_paths": {
            name: str((raw_dataset / name).resolve()) for name in ("imagesTr", "labelsTr")
        },
        "bound_hashes": {
            "planning_ready": _sha256(ready_path),
            "planning_bundle": _sha256(run_receipt),
            "dataset_json": _sha256(artifact_paths["dataset_json"]),
            "dataset_fingerprint": _sha256(artifact_paths["dataset_fingerprint"]),
            "nnunet_plans": _sha256(artifact_paths["nnunet_plans"]),
            "splits_final": _sha256(artifact_paths["splits_final"]),
        },
    }


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _stage_source_symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(source, target_is_directory=True)


def stage_preprocessing_run(
    planning_ready_path: Path,
    staging_root: Path,
    final_root: Path,
    run_id: str,
    *,
    planning_validator: Callable[[Path], dict[str, Any]] = validate_planning_ready,
    raw_source_stager: Callable[[Path, Path], None] = _stage_source_symlink,
) -> dict[str, Any]:
    """Stage immutable planning metadata into one fresh preprocessing run."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ContractError("unsafe preprocessing run_id")
    staging_root = staging_root.resolve()
    final_root = final_root.resolve()
    if not staging_root.is_dir() or staging_root.is_symlink():
        raise ContractError("staging root must be a fresh regular directory")
    if staging_root.name != f".partial-{run_id}":
        raise ContractError("staging directory does not match run identity")
    if staging_root.parent != final_root.parent or final_root.name != run_id:
        raise ContractError("staging and committed destination must be sibling run paths")
    if any(staging_root.iterdir()):
        raise ContractError("preprocessing staging root must be empty")
    if os.path.lexists(final_root):
        raise FileExistsError(f"refusing existing preprocessing destination: {final_root}")

    planning = planning_validator(planning_ready_path)
    raw_dataset = staging_root / "nnUNet_raw" / DATASET_FOLDER
    preprocessed_dataset = staging_root / "nnUNet_preprocessed" / DATASET_FOLDER
    results_root = staging_root / "nnUNet_results"
    raw_dataset.mkdir(parents=True)
    preprocessed_dataset.mkdir(parents=True)
    results_root.mkdir()
    _copy_exclusive(Path(planning["artifact_paths"]["dataset_json"]), raw_dataset / "dataset.json")
    for name in ("imagesTr", "labelsTr"):
        raw_source_stager(
            Path(planning["raw_source_paths"][name]), raw_dataset / name
        )
    for label, filename in METADATA_FILES.items():
        _copy_exclusive(
            Path(planning["artifact_paths"][label]), preprocessed_dataset / filename
        )

    owner = {
        "status": "OWNED",
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "staging_dir_name": staging_root.name,
        "owner_token": uuid4().hex,
    }
    _write_json_exclusive(staging_root / "RUN_OWNER.json", owner)
    staged = {
        "status": "STAGED",
        "contract_version": CONTRACT_VERSION,
        "phase": "PREPROCESSING_ONLY",
        "run_id": run_id,
        "staging_run_dir": str(staging_root),
        "committed_run_dir": str(final_root),
        "planning_ready": _file_record(planning_ready_path.resolve()),
        "planning_bound_hashes": planning["bound_hashes"],
        "preprocess_api": PREPROCESS_API,
        "metadata_artifacts": {
            label: _relative_record(staging_root, preprocessed_dataset / filename)
            for label, filename in METADATA_FILES.items()
        },
        "training_status": "NOT_STARTED",
        "checkpoint_count": 0,
        "oof_prediction_count": 0,
        "result_count": 0,
    }
    _write_json_exclusive(staging_root / "STAGING_RECEIPT.json", staged)
    return staged


def validate_preprocessed_inventory(
    configuration_names: Iterable[str],
    gt_names: Iterable[str],
    *,
    load_case_hook: Callable[[str], dict[str, Any]],
    hook_kind: str,
) -> dict[str, Any]:
    """Validate the exact 597-case file inventory and one loadable case."""

    config = list(configuration_names)
    ground_truth = list(gt_names)
    if len(config) != len(set(config)):
        raise ContractError("preprocessed configuration contains duplicate filenames")
    if len(ground_truth) != len(set(ground_truth)):
        raise ContractError("ground-truth folder contains duplicate filenames")
    data_ids: set[str] = set()
    seg_ids: set[str] = set()
    pkl_ids: set[str] = set()
    for name in config:
        if "/" in name or "\\" in name:
            raise ContractError("unexpected nested preprocessing filename")
        if name.endswith("_seg.b2nd"):
            seg_ids.add(name[: -len("_seg.b2nd")])
        elif name.endswith(".b2nd"):
            data_ids.add(name[: -len(".b2nd")])
        elif name.endswith(".pkl"):
            pkl_ids.add(name[: -len(".pkl")])
        else:
            raise ContractError(f"unexpected preprocessing artifact: {name}")
    if len(data_ids) != EXPECTED_CASES:
        raise ContractError("preprocessing must contain exactly 597 data .b2nd files")
    if data_ids != seg_ids or data_ids != pkl_ids:
        raise ContractError("preprocessing case triplet sets do not match")
    gt_ids: set[str] = set()
    for name in ground_truth:
        if "/" in name or "\\" in name or not name.endswith(".nii.gz"):
            raise ContractError(f"unexpected ground-truth artifact: {name}")
        gt_ids.add(name[: -len(".nii.gz")])
    if len(gt_ids) != EXPECTED_CASES:
        raise ContractError("preprocessing must contain exactly 597 ground-truth labels")
    if gt_ids != data_ids:
        raise ContractError("ground-truth identifiers do not match preprocessing cases")

    identifier = sorted(data_ids)[0]
    load_receipt = load_case_hook(identifier)
    if not isinstance(load_receipt, dict):
        raise ContractError("one-case load hook did not return a validation receipt")
    data_shape = load_receipt.get("data_shape")
    seg_shape = load_receipt.get("seg_shape")
    if (
        not isinstance(data_shape, list)
        or len(data_shape) != 4
        or data_shape[0] != 2
        or not isinstance(seg_shape, list)
        or len(seg_shape) != 4
        or seg_shape[0] != 1
        or data_shape[1:] != seg_shape[1:]
    ):
        raise ContractError("one-case load does not preserve 2-channel data/1-channel label shapes")
    if load_receipt.get("properties_type") != "dict":
        raise ContractError("one-case load properties are not a dictionary")
    official = hook_kind == "OFFICIAL_NNUNET_V2_8_1"
    return {
        "case_count": EXPECTED_CASES,
        "identifier_manifest_sha256": hashlib.sha256(
            "\n".join(sorted(data_ids)).encode("utf-8")
        ).hexdigest(),
        "artifact_counts": {
            ".b2nd": EXPECTED_CASES,
            "_seg.b2nd": EXPECTED_CASES,
            ".pkl": EXPECTED_CASES,
            "gt_segmentations": EXPECTED_CASES,
        },
        "one_case_load": {
            "status": "PASS",
            "identifier": identifier,
            "hook_kind": hook_kind,
            "official_nnunet_load_claimed": official,
            "data_shape": data_shape,
            "seg_shape": seg_shape,
            "properties_type": "dict",
        },
    }


def _official_load_hook(configuration_dir: Path) -> Callable[[str], dict[str, Any]]:
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    def load(identifier: str) -> dict[str, Any]:
        dataset = nnUNetDatasetBlosc2(str(configuration_dir), identifiers=[identifier])
        data, segmentation, previous_stage, properties = dataset.load_case(identifier)
        if previous_stage is not None:
            raise ContractError("3d_fullres M0 unexpectedly contains previous-stage segmentation")
        return {
            "data_shape": [int(value) for value in data.shape],
            "seg_shape": [int(value) for value in segmentation.shape],
            "properties_type": type(properties).__name__,
        }

    return load


def validate_preprocessed_output(run_root: Path) -> dict[str, Any]:
    """Perform filesystem and official nnU-Net one-case validation."""

    run_root = run_root.resolve()
    if not run_root.is_dir() or run_root.is_symlink():
        raise ContractError("preprocessing run root is missing")
    allowed_root = {
        "nnUNet_raw",
        "nnUNet_preprocessed",
        "nnUNet_results",
        "RUN_OWNER.json",
        "STAGING_RECEIPT.json",
        "PREPROCESSING_BUNDLE.json",
    }
    if not {item.name for item in run_root.iterdir()}.issubset(allowed_root):
        raise ContractError("preprocessing run root contains unexpected outputs")
    results_root = run_root / "nnUNet_results"
    if not results_root.is_dir() or any(results_root.iterdir()):
        raise ContractError("preprocessing-only run contains training/checkpoint outputs")
    pre_root = run_root / "nnUNet_preprocessed" / DATASET_FOLDER
    expected_pre_entries = set(METADATA_FILES.values()) | {
        DATA_IDENTIFIER,
        "gt_segmentations",
    }
    if not pre_root.is_dir() or {item.name for item in pre_root.iterdir()} != expected_pre_entries:
        raise ContractError("preprocessed dataset root whitelist mismatch")
    configuration = pre_root / DATA_IDENTIFIER
    ground_truth = pre_root / "gt_segmentations"
    if any(item.is_symlink() or not item.is_file() for item in configuration.iterdir()):
        raise ContractError("preprocessed case artifacts must be regular files")
    if any(item.is_symlink() or not item.is_file() for item in ground_truth.iterdir()):
        raise ContractError("preprocessed ground-truth labels must be regular files")

    staging_receipt = _load_json(run_root / "STAGING_RECEIPT.json", label="staging receipt")
    if staging_receipt.get("run_id") != run_root.name:
        # Before commit the directory name is .partial-<run_id>.
        if run_root.name != f".partial-{staging_receipt.get('run_id')}":
            raise ContractError("staging receipt run identity mismatch")
    metadata_records = staging_receipt.get("metadata_artifacts", {})
    metadata_hashes: dict[str, str] = {}
    for label, filename in METADATA_FILES.items():
        path = pre_root / filename
        _verify_record(path, metadata_records.get(label), label=f"metadata {label}")
        metadata_hashes[label] = _sha256(path)
    _validate_dataset_json(_load_json(pre_root / "dataset.json", label="dataset.json"))
    fingerprint_contract = validate_fingerprint(
        _load_json(pre_root / "dataset_fingerprint.json", label="dataset fingerprint")
    )
    if fingerprint_contract.get("case_count") != EXPECTED_CASES:
        raise ContractError("preprocessed fingerprint no longer binds 597 cases")
    plan_contract = validate_plan_contract(
        _load_json(pre_root / "nnUNetPlans.json", label="nnUNet plans")
    )
    if (
        plan_contract.get("normalization_schemes") != EXPECTED_NORMALIZATION
        or plan_contract.get("use_mask_for_norm") != EXPECTED_MASKS
    ):
        raise ContractError("preprocessed plan normalization/mask contract drift")

    inventory = validate_preprocessed_inventory(
        [item.name for item in configuration.iterdir()],
        [item.name for item in ground_truth.iterdir()],
        load_case_hook=_official_load_hook(configuration),
        hook_kind="OFFICIAL_NNUNET_V2_8_1",
    )
    inventory["metadata_hashes"] = metadata_hashes
    inventory["configuration"] = CONFIGURATION
    inventory["data_identifier"] = DATA_IDENTIFIER
    return inventory


def _validate_inventory_for_publication(inventory: dict[str, Any]) -> None:
    if inventory.get("case_count") != EXPECTED_CASES:
        raise ContractError("preprocessing inventory does not contain 597 cases")
    if inventory.get("artifact_counts") != {
        ".b2nd": EXPECTED_CASES,
        "_seg.b2nd": EXPECTED_CASES,
        ".pkl": EXPECTED_CASES,
        "gt_segmentations": EXPECTED_CASES,
    }:
        raise ContractError("preprocessing artifact count contract mismatch")
    load = inventory.get("one_case_load", {})
    if (
        load.get("status") != "PASS"
        or load.get("hook_kind") != "OFFICIAL_NNUNET_V2_8_1"
        or load.get("official_nnunet_load_claimed") is not True
    ):
        raise ContractError("official nnU-Net v2.8.1 one-case load gate did not pass")


def build_preprocessing_bundle(
    planning_ready_path: Path,
    *,
    run_id: str,
    committed_run_dir: Path,
    inventory: dict[str, Any],
    planning_validator: Callable[[Path], dict[str, Any]] = validate_planning_ready,
) -> dict[str, Any]:
    planning = planning_validator(planning_ready_path)
    committed_run_dir = committed_run_dir.resolve()
    if committed_run_dir.name != run_id:
        raise ContractError("preprocessing bundle run identity mismatch")
    _validate_inventory_for_publication(inventory)
    return {
        "status": "VALIDATED",
        "preprocessing_status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "phase": "PREPROCESSING_ONLY",
        "run_id": run_id,
        "committed_run_dir": str(committed_run_dir),
        "dataset": {
            "id": DATASET_ID,
            "folder": DATASET_FOLDER,
            "case_count": EXPECTED_CASES,
            "channel_names": EXPECTED_CHANNELS,
        },
        "planning_ready": _file_record(planning_ready_path.resolve()),
        "planning_bound_hashes": planning["bound_hashes"],
        "official_preprocess_api": {
            "module": "nnunetv2.experiment_planning.plan_and_preprocess_api",
            "function": "preprocess",
            **PREPROCESS_API,
        },
        "plan_contract": planning["plan_contract"],
        "output_contract": inventory,
        "training_status": "NOT_STARTED",
        "training_performed": False,
        "checkpoint_count": 0,
        "oof_prediction_count": 0,
        "result_count": 0,
    }


def publish_preprocess_ready(
    run_dir: Path,
    bundle_path: Path,
    ready_path: Path,
    *,
    output_validator: Callable[[Path], dict[str, Any]] = validate_preprocessed_output,
    planning_validator: Callable[[Path], dict[str, Any]] = validate_planning_ready,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    bundle_path = bundle_path.resolve()
    if not bundle_path.is_relative_to(run_dir) or bundle_path.parent != run_dir:
        raise ContractError("PREPROCESSING_BUNDLE must be inside its committed run")
    bundle_bytes = bundle_path.read_bytes()
    bundle = _load_json(bundle_path, label="PREPROCESSING_BUNDLE")
    _require_fields(
        bundle,
        {
            "status": "VALIDATED",
            "preprocessing_status": "PASS",
            "contract_version": CONTRACT_VERSION,
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
    planning_ready_record = bundle.get("planning_ready")
    planning_ready_path = Path(planning_ready_record.get("path", "")).resolve()
    _verify_record(planning_ready_path, planning_ready_record, label="PLANNING_READY")
    planning = planning_validator(planning_ready_path)
    if bundle.get("planning_bound_hashes") != planning.get("bound_hashes"):
        raise ContractError("planning hashes changed before preprocessing publication")
    fresh_inventory = output_validator(run_dir)
    _validate_inventory_for_publication(fresh_inventory)
    if fresh_inventory != bundle.get("output_contract"):
        raise ContractError("preprocessing output changed before fixed receipt publication")
    if hashlib.sha256(bundle_path.read_bytes()).hexdigest() != hashlib.sha256(bundle_bytes).hexdigest():
        raise ContractError("PREPROCESSING_BUNDLE changed during publication")
    published = {
        "status": "COMMITTED",
        "preprocessing_status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "phase": "PREPROCESSING_ONLY",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "run_receipt": {
            "path": str(bundle_path),
            "bytes": len(bundle_bytes),
            "sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        },
        "training_status": "NOT_STARTED",
        "training_performed": False,
        "checkpoint_count": 0,
        "oof_prediction_count": 0,
        "result_count": 0,
        "validated_bundle": bundle,
    }
    _write_json_exclusive(ready_path, published)
    return published


def validate_live_nnunet_runtime(source_root: Path) -> dict[str, Any]:
    """Fail closed unless the executing interpreter imports the pinned source."""

    import nnunetv2

    source_root = source_root.resolve()
    package_file = Path(nnunetv2.__file__).resolve()
    if not package_file.is_relative_to(source_root):
        raise ContractError("live nnunetv2 import is not from the pinned source tree")
    version = importlib_metadata.version("nnunetv2")
    if version != EXPECTED_NNUNET_VERSION:
        raise ContractError("live nnunetv2 version is not 2.8.1")
    tree_hash = source_tree_sha256(source_root)
    if tree_hash != EXPECTED_NNUNET_SOURCE_TREE_SHA256:
        raise ContractError("live nnUNet source tree hash changed")
    return {
        "status": "PASS",
        "version": version,
        "package_file": str(package_file),
        "source_tree_sha256": tree_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_planning = subparsers.add_parser("validate-planning-ready")
    validate_planning.add_argument("receipt", type=Path)
    runtime = subparsers.add_parser("validate-runtime")
    runtime.add_argument("source_root", type=Path)
    stage = subparsers.add_parser("stage")
    stage.add_argument("planning_ready", type=Path)
    stage.add_argument("staging_root", type=Path)
    stage.add_argument("final_root", type=Path)
    stage.add_argument("run_id")
    validate = subparsers.add_parser("validate-preprocessing")
    validate.add_argument("--planning-ready", type=Path, required=True)
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--committed-run-dir", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    commit = subparsers.add_parser("commit-run")
    commit.add_argument("staging_dir", type=Path)
    commit.add_argument("final_dir", type=Path)
    commit.add_argument("receipt", type=Path)
    publish = subparsers.add_parser("publish-preprocess-ready")
    publish.add_argument("run_dir", type=Path)
    publish.add_argument("bundle", type=Path)
    publish.add_argument("ready", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate-planning-ready":
        payload = validate_planning_ready(args.receipt)
    elif args.command == "validate-runtime":
        payload = validate_live_nnunet_runtime(args.source_root)
    elif args.command == "stage":
        payload = stage_preprocessing_run(
            args.planning_ready, args.staging_root, args.final_root, args.run_id
        )
    elif args.command == "validate-preprocessing":
        inventory = validate_preprocessed_output(args.run_root)
        payload = build_preprocessing_bundle(
            args.planning_ready,
            run_id=args.run_id,
            committed_run_dir=args.committed_run_dir,
            inventory=inventory,
        )
        _write_json_exclusive(args.receipt, payload)
    elif args.command == "commit-run":
        payload = commit_run_directory(args.staging_dir, args.final_dir, args.receipt)
    else:
        payload = publish_preprocess_ready(args.run_dir, args.bundle, args.ready)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
