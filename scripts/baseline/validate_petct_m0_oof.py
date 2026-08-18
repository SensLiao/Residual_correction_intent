#!/usr/bin/env python3
"""Validate the Dataset901 patient-excluded five-fold OOF M0 gate.

This module is deliberately upstream-only.  It stages val-only inference
plans, binds every output to the held-out fold and immutable model/source
assets, and publishes ``OOF_READY`` only after exact cohort coverage passes.
It never generates scribbles, intent, or experimental results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
for support_dir in (SCRIPTS_ROOT, SCRIPTS_ROOT / "data"):
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))

from audit_psma_v3_dataset import patient_from_case  # noqa: E402
from common.petct_mainline_lineage import M0_V6_OOF_SCHEMA  # noqa: E402
from prepare_nnunet_m0_dataset import (  # noqa: E402
    EXPECTED_NNUNET_COMMIT,
    EXPECTED_NNUNET_SOURCE_TREE_SHA256,
    commit_run_directory,
)  # noqa: E402
from validate_petct_m0_preprocess import (  # noqa: E402
    ContractError,
    _load_json,
    _sha256,
    _verify_record,
    _write_json_exclusive,
)  # noqa: E402
from validate_petct_m0_smoke import validate_preprocess_ready  # noqa: E402


DATASET_ID = 901
DATASET_FOLDER = "Dataset901_PSMA_M0_AutoPETVNorm"
EXPECTED_CASES = 597
EXPECTED_PATIENTS = 378
EXPECTED_FOLDS = 5
CONFIGURATION = "3d_fullres"
TRAINER = "nnUNetTrainer"
PLANS_IDENTIFIER = "nnUNetPlans"
CHECKPOINT_NAME = "checkpoint_final.pth"

FULL_TRAIN_READY_VERSION = "1.0.0"
FULL_TRAIN_PHASE = "STANDARD_5FOLD_FULL_TRAINING"
OOF_CONTRACT_VERSION = "PETCT-M0-OOF-v1.0"
OOF_READY_VERSION = "PETCT-M0-OOF-READY-v1.0"
OOF_PHASE = "PATIENT_EXCLUDED_5FOLD_OOF"
NATURAL_PROVENANCE_VERSION = "PETCT-M0-NATURAL-PROVENANCE-v1.0"
TRUTH_BINDING_VERSION = "PETCT-M0-OOF-TRUTH-BINDING-v1.0"
PREDICTION_SOURCES = {
    "DEDICATED_OOF_INFERENCE",
    "TRAINING_ACTUAL_VALIDATION_HANDOFF",
}
INFERENCE_COMPILE_CONTRACT = {
    "environment_variable": "nnUNet_compile",
    "value": "false",
    "mode": "PINNED_OFFICIAL_PREDICTOR_NO_COMPILE",
}
PROBABILITY_VERIFICATION_BOUNDARY = {
    "artifact_format": "NPZ_FOREGROUND_ARRAY_WITHOUT_SPATIAL_HEADER",
    "independent_affine_available": False,
    "verified_binding": "exact_shape_match_to_mask_on_reference_CT_GT_grid",
}


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"required regular file is missing: {path}")
    display = str(path)
    if root is not None:
        display = path.relative_to(root.resolve()).as_posix()
    return {"path": display, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _resolve_record(record: dict[str, Any], root: Path, label: str) -> Path:
    raw = record.get("path") if isinstance(record, dict) else None
    if not isinstance(raw, str) or not raw:
        raise ContractError(f"{label} record path is missing")
    candidate = Path(raw)
    if not candidate.is_absolute() and ".." in candidate.parts:
        raise ContractError(f"unsafe {label} relative path")
    unresolved = candidate if candidate.is_absolute() else root / candidate
    if unresolved.is_symlink():
        raise ContractError(f"{label} must not be a symlink")
    path = unresolved.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ContractError(f"{label} escapes its OOF run")
    return path


def _require(payload: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ContractError(
                f"{label} requires {key}={value!r}; observed {payload.get(key)!r}"
            )


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_split_document(splits: Any) -> dict[str, Any]:
    """Prove exact case coverage and patient exclusion for five folds."""

    if not isinstance(splits, list) or len(splits) != EXPECTED_FOLDS:
        raise ValueError("splits_final must contain exactly five folds")
    normalized: list[tuple[list[str], list[str]]] = []
    for fold, item in enumerate(splits):
        if not isinstance(item, dict) or set(item) != {"train", "val"}:
            raise ValueError(f"fold {fold} must contain only train and val")
        train, val = item["train"], item["val"]
        if not isinstance(train, list) or not isinstance(val, list):
            raise ValueError(f"fold {fold} train and val must be lists")
        if any(not isinstance(case, str) or not case for case in train + val):
            raise ValueError(f"fold {fold} contains an invalid case id")
        if len(train) != len(set(train)) or len(val) != len(set(val)):
            raise ValueError(f"fold {fold} contains duplicate case ids")
        overlap = set(train) & set(val)
        if overlap:
            raise ValueError(f"fold {fold} train/val case overlap is not zero")
        normalized.append((sorted(train), sorted(val)))

    val_counts = Counter(case for _, val in normalized for case in val)
    if any(count != 1 for count in val_counts.values()):
        raise ValueError("every case must appear in validation exactly once")
    universe = set(val_counts)
    if len(universe) != EXPECTED_CASES:
        raise ValueError(
            f"validation must cover exactly {EXPECTED_CASES} cases exactly once"
        )
    for fold, (train, val) in enumerate(normalized):
        if set(train) != universe - set(val):
            raise ValueError(f"fold {fold} train is not the val complement")

    case_to_patient = {case: patient_from_case(case) for case in universe}
    patients = set(case_to_patient.values())
    if len(patients) != EXPECTED_PATIENTS:
        raise ValueError(f"splits must contain exactly {EXPECTED_PATIENTS} patients")
    patient_folds: defaultdict[str, set[int]] = defaultdict(set)
    folds_out: list[dict[str, Any]] = []
    case_to_fold: dict[str, int] = {}
    for fold, (train, val) in enumerate(normalized):
        train_patients = {case_to_patient[case] for case in train}
        val_patients = {case_to_patient[case] for case in val}
        overlap = train_patients & val_patients
        if overlap:
            raise ValueError(
                f"patient does not have a single held-out fold: fold {fold} "
                "train/val patient overlap is not zero"
            )
        for case in val:
            patient_folds[case_to_patient[case]].add(fold)
            case_to_fold[case] = fold
        folds_out.append(
            {
                "fold": fold,
                "train_case_count": len(train),
                "val_case_count": len(val),
                "train_patient_count": len(train_patients),
                "val_patient_count": len(val_patients),
                "train_val_case_overlap": 0,
                "train_val_patient_overlap": 0,
            }
        )
    if any(len(held_out) != 1 for held_out in patient_folds.values()):
        raise ValueError("every patient must have a single held-out fold")
    return {
        "fold_count": EXPECTED_FOLDS,
        "case_count": EXPECTED_CASES,
        "patient_count": EXPECTED_PATIENTS,
        "val_exact_once": True,
        "patient_single_held_out_fold": True,
        "folds": folds_out,
        "case_to_fold": dict(sorted(case_to_fold.items())),
        "case_to_patient": dict(sorted(case_to_patient.items())),
    }


def validate_authoritative_splits(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"authoritative splits_final is missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError("authoritative splits_final is invalid JSON") from exc
    contract = validate_split_document(document)
    contract["splits_final"] = _record(path)
    contract["document"] = document
    return contract


def _validate_full_train_ready(
    ready_paths: Iterable[Path], preprocess_ready: Path, splits_path: Path
) -> dict[int, dict[str, Any]]:
    """Validate the one canonical aggregate FULL_TRAIN_READY and five folds."""

    expected_preprocess_hash = _sha256(preprocess_ready.resolve())
    expected_splits_hash = _sha256(splits_path.resolve())
    supplied = [Path(path).resolve() for path in ready_paths]
    if len(supplied) != 1:
        raise ContractError("OOF requires exactly one aggregate FULL_TRAIN_READY.json")
    ready_path = supplied[0]
    ready = _load_json(ready_path, label="FULL_TRAIN_READY")
    _require(
        ready,
        {
            "status": "COMMITTED",
            "contract_version": FULL_TRAIN_READY_VERSION,
            "phase": FULL_TRAIN_PHASE,
            "full_training_status": "PASS",
            "full_training_performed": True,
            "folds_completed": list(range(EXPECTED_FOLDS)),
            "checkpoint_count": EXPECTED_FOLDS * 2,
            "oof_status": "NOT_STARTED",
            "oof_prediction_count": 0,
            "result_count": 0,
            "thesis_citable": False,
        },
        "FULL_TRAIN_READY",
    )
    campaign_root = Path(ready.get("campaign_root", "")).resolve()
    if (
        campaign_root.is_symlink()
        or not campaign_root.is_dir()
        or ready.get("campaign_id") != campaign_root.name
    ):
        raise ContractError("FULL_TRAIN_READY campaign identity is invalid")
    aggregate_record = _record(ready_path)
    bound = ready.get("prerequisite_bound_hashes")
    if not isinstance(bound, dict):
        raise ContractError("FULL_TRAIN_READY prerequisite hashes are missing")
    if bound.get("preprocess_ready") != expected_preprocess_hash:
        raise ContractError("FULL_TRAIN_READY PREPROCESS_READY hash mismatch")
    if bound.get("splits_final") != expected_splits_hash:
        raise ContractError("FULL_TRAIN_READY splits_final hash mismatch")
    if bound.get("source_tree") != EXPECTED_NNUNET_SOURCE_TREE_SHA256:
        raise ContractError("FULL_TRAIN_READY nnUNet source tree mismatch")
    contract = ready.get("training_contract")
    if not isinstance(contract, dict):
        raise ContractError("FULL_TRAIN_READY training contract is missing")
    for key, value in {
        "dataset_id": DATASET_ID,
        "configuration": CONFIGURATION,
        "folds": list(range(EXPECTED_FOLDS)),
        "trainer": TRAINER,
        "plans_identifier": PLANS_IDENTIFIER,
        "num_epochs": 1000,
        "device": "cuda",
    }.items():
        if contract.get(key) != value:
            raise ContractError(f"FULL_TRAIN_READY training contract {key} mismatch")
    spec_record = ready.get("campaign_spec")
    spec_raw = spec_record.get("path") if isinstance(spec_record, dict) else None
    if not isinstance(spec_raw, str):
        raise ContractError("FULL_TRAIN_READY campaign spec record is missing")
    spec_path = Path(spec_raw).resolve()
    if not spec_path.is_relative_to(campaign_root):
        raise ContractError("FULL_TRAIN_READY campaign spec escapes campaign")
    _verify_record(spec_path, spec_record, label="CAMPAIGN_SPEC")
    spec = _load_json(spec_path, label="CAMPAIGN_SPEC")
    _require(
        spec,
        {
            "status": "STAGED",
            "contract_version": FULL_TRAIN_READY_VERSION,
            "phase": FULL_TRAIN_PHASE,
            "campaign_id": campaign_root.name,
            "campaign_root": str(campaign_root),
            "prerequisite_bound_hashes": bound,
            "training_contract": contract,
            "full_training_status": "NOT_STARTED",
            "oof_status": "NOT_STARTED",
            "oof_prediction_count": 0,
            "result_count": 0,
            "thesis_citable": False,
        },
        "CAMPAIGN_SPEC",
    )
    trainer_root = (
        campaign_root
        / "nnUNet_results"
        / DATASET_FOLDER
        / f"{TRAINER}__{PLANS_IDENTIFIER}__{CONFIGURATION}"
    )
    plans = trainer_root / "plans.json"
    dataset_json = trainer_root / "dataset.json"
    plans_record, dataset_record = _record(plans), _record(dataset_json)
    receipt_records = ready.get("fold_receipts")
    if not isinstance(receipt_records, list) or len(receipt_records) != EXPECTED_FOLDS:
        raise ContractError("FULL_TRAIN_READY must bind exactly five fold receipts")
    by_fold: dict[int, dict[str, Any]] = {}
    for receipt_record in receipt_records:
        raw = receipt_record.get("path") if isinstance(receipt_record, dict) else None
        if not isinstance(raw, str):
            raise ContractError("FULL_TRAIN_READY fold receipt record is missing")
        path = Path(raw).resolve()
        if not path.is_relative_to(campaign_root):
            raise ContractError("FULL_TRAIN_READY fold receipt escapes campaign")
        _verify_record(path, receipt_record, label="fold receipt")
        fold_ready = _load_json(path, label="fold receipt")
        fold = fold_ready.get("fold")
        if not isinstance(fold, int) or fold not in range(EXPECTED_FOLDS):
            raise ContractError("fold receipt fold must be 0..4")
        if fold in by_fold:
            raise ContractError(f"duplicate fold receipt {fold}")
        _require(
            fold_ready,
            {
                "status": "COMMITTED",
                "contract_version": FULL_TRAIN_READY_VERSION,
                "phase": FULL_TRAIN_PHASE,
                "campaign_id": campaign_root.name,
                "fold": fold,
                "full_fold_training_status": "PASS",
                "oof_status": "NOT_STARTED",
                "oof_prediction_count": 0,
                "result_count": 0,
                "thesis_citable": False,
            },
            f"fold {fold} receipt",
        )
        if fold_ready.get("prerequisite_bound_hashes") != bound:
            raise ContractError(f"fold {fold} prerequisite hashes changed")
        output = fold_ready.get("output_contract")
        if not isinstance(output, dict):
            raise ContractError(f"fold {fold} output contract is missing")
        for key, value in {
            "status": "PASS",
            "fold": fold,
            "epoch_count": 1000,
            "checkpoint_count": 2,
            "split_sha256": expected_splits_hash,
            "oof_publication_count": 0,
            "result_publication_count": 0,
        }.items():
            if output.get(key) != value:
                raise ContractError(f"fold {fold} output contract {key} mismatch")
        checkpoint_record = output.get("artifacts", {}).get("checkpoint_final")
        checkpoint_raw = (
            checkpoint_record.get("path")
            if isinstance(checkpoint_record, dict)
            else None
        )
        if not isinstance(checkpoint_raw, str):
            raise ContractError(f"fold {fold} checkpoint_final record is missing")
        checkpoint = Path(checkpoint_raw)
        checkpoint = (
            checkpoint.resolve()
            if checkpoint.is_absolute()
            else (campaign_root / checkpoint).resolve()
        )
        expected_checkpoint = trainer_root / f"fold_{fold}" / CHECKPOINT_NAME
        if checkpoint != expected_checkpoint:
            raise ContractError(f"fold {fold} checkpoint path mismatch")
        _verify_record(checkpoint, checkpoint_record, label=f"fold {fold} checkpoint")
        by_fold[fold] = {
            "full_train_ready": aggregate_record,
            "fold_receipt": _record(path),
            "receipt": _record(path),
            "receipt_path": str(path),
            "checkpoint": _record(checkpoint),
            "plans": plans_record,
            "dataset_json": dataset_record,
            "model_training_output_dir": str(trainer_root),
            "source_tree_sha256": EXPECTED_NNUNET_SOURCE_TREE_SHA256,
            "source_commit": EXPECTED_NNUNET_COMMIT,
        }
    if set(by_fold) != set(range(EXPECTED_FOLDS)):
        raise ContractError(
            "FULL_TRAIN_READY does not bind every fold 0..4 exactly once"
        )
    return by_fold


def stage_oof_run(
    preprocess_ready_path: Path,
    full_train_ready_paths: Iterable[Path],
    splits_path: Path,
    staging_root: Path,
    final_root: Path,
    run_id: str,
    *,
    preprocess_validator: Callable[[Path], dict[str, Any]] = validate_preprocess_ready,
) -> dict[str, Any]:
    """Stage immutable, fold-locked, val-only OOF inference plans."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ContractError("unsafe OOF run_id")
    staging_root, final_root = staging_root.resolve(), final_root.resolve()
    if (
        staging_root.is_symlink()
        or not staging_root.is_dir()
        or any(staging_root.iterdir())
    ):
        raise ContractError("OOF staging root must be a fresh empty regular directory")
    if staging_root.name != f".partial-{run_id}":
        raise ContractError("OOF staging directory does not match run identity")
    if staging_root.parent != final_root.parent or final_root.name != run_id:
        raise ContractError("OOF staging and destination must be sibling run paths")
    if os.path.lexists(final_root):
        raise FileExistsError(f"refusing existing OOF destination: {final_root}")

    preprocess = preprocess_validator(preprocess_ready_path)
    if preprocess.get("status") != "PASS":
        raise ContractError("PREPROCESS_READY did not pass validation")
    splits = validate_authoritative_splits(splits_path)
    bound = preprocess.get("bound_hashes", {})
    if bound.get("planning_splits_final") != splits["splits_final"]["sha256"]:
        raise ContractError(
            "authoritative splits_final is not bound by PREPROCESS_READY"
        )
    models = _validate_full_train_ready(
        list(full_train_ready_paths), preprocess_ready_path, splits_path
    )
    raw_dataset = Path(preprocess["nnunet_raw"]) / DATASET_FOLDER
    bindings = preprocess.get("raw_source_bindings")
    raw_dirs: dict[str, Path] = {}
    for name in ("imagesTr", "labelsTr"):
        raw_dir = raw_dataset / name
        if bindings is None:
            # Early immutable PREPROCESS_READY receipts predate
            # raw_source_bindings. They are admissible only when OOF can use
            # a materialized directory and cannot follow an unrecorded link.
            if raw_dir.is_symlink() or not raw_dir.is_dir():
                raise ContractError(
                    f"legacy PREPROCESS_READY {name} must be a materialized directory"
                )
        else:
            if not isinstance(bindings, dict):
                raise ContractError("PREPROCESS_READY raw-source bindings are invalid")
            binding = bindings.get(name)
            if not isinstance(binding, dict):
                raise ContractError(
                    f"PREPROCESS_READY raw-source binding is missing {name}"
                )
            policy = binding.get("policy")
            if policy == "PLANNING_RECEIPT_BOUND_DIRECTORY_SYMLINK":
                if (
                    not raw_dir.is_symlink()
                    or str(raw_dir) != binding.get("link_path")
                    or str(raw_dir.resolve()) != binding.get("target_path")
                    or not raw_dir.resolve().is_dir()
                ):
                    raise ContractError(f"PREPROCESS_READY {name} symlink binding changed")
            elif policy == "TEST_DIRECTORY_FIXTURE":
                if raw_dir.is_symlink() or not raw_dir.is_dir():
                    raise ContractError(
                        f"PREPROCESS_READY test {name} fixture is missing"
                    )
            else:
                raise ContractError(
                    f"PREPROCESS_READY {name} binding policy is invalid"
                )
        raw_dirs[name] = raw_dir
    raw_images = raw_dirs["imagesTr"]

    (staging_root / "fold_plans").mkdir()
    (staging_root / "outputs").mkdir()
    document = splits["document"]
    for fold in range(EXPECTED_FOLDS):
        train_case_ids = sorted(document[fold]["train"])
        val_case_ids = sorted(document[fold]["val"])
        train_patient_ids = sorted(
            {splits["case_to_patient"][case_id] for case_id in train_case_ids}
        )
        val_patient_ids = sorted(
            {splits["case_to_patient"][case_id] for case_id in val_case_ids}
        )
        masks = staging_root / "outputs" / f"fold_{fold}" / "masks"
        probabilities = staging_root / "outputs" / f"fold_{fold}" / "probabilities"
        masks.mkdir(parents=True)
        probabilities.mkdir()
        cases: list[dict[str, Any]] = []
        for case_id in val_case_ids:
            ct = raw_images / f"{case_id}_0000.nii.gz"
            pet = raw_images / f"{case_id}_0001.nii.gz"
            gt = raw_dirs["labelsTr"] / f"{case_id}.nii.gz"
            cases.append(
                {
                    "case_id": case_id,
                    "patient_id": splits["case_to_patient"][case_id],
                    "held_out_fold": fold,
                    "input_ct": _record(ct),
                    "input_pet": _record(pet),
                    "input_gt": _record(gt),
                    "mask_output": f"outputs/fold_{fold}/masks/{case_id}.nii.gz",
                    "probability_output": f"outputs/fold_{fold}/probabilities/{case_id}.npz",
                }
            )
        plan = {
            "schema_version": OOF_CONTRACT_VERSION,
            "phase": OOF_PHASE,
            "run_id": run_id,
            "fold": fold,
            "use_folds": [fold],
            "configuration": CONFIGURATION,
            "trainer": TRAINER,
            "plans_identifier": PLANS_IDENTIFIER,
            "checkpoint_name": CHECKPOINT_NAME,
            "compile_contract": INFERENCE_COMPILE_CONTRACT,
            "save_probabilities": True,
            "overwrite": False,
            "train_case_ids": train_case_ids,
            "val_case_ids": val_case_ids,
            "train_patient_ids": train_patient_ids,
            "val_patient_ids": val_patient_ids,
            "authoritative_split_binding": {
                "splits_final_sha256": splits["splits_final"]["sha256"],
                "train_case_ids_sha256": _canonical_hash(train_case_ids),
                "val_case_ids_sha256": _canonical_hash(val_case_ids),
                "train_patient_ids_sha256": _canonical_hash(train_patient_ids),
                "val_patient_ids_sha256": _canonical_hash(val_patient_ids),
            },
            "cases": cases,
            "model": models[fold],
        }
        _write_json_exclusive(staging_root / "fold_plans" / f"fold_{fold}.json", plan)

    _write_json_exclusive(
        staging_root / "RUN_OWNER.json",
        {
            "status": "OWNED",
            "schema_version": OOF_CONTRACT_VERSION,
            "run_id": run_id,
            "owner_token": uuid4().hex,
        },
    )
    spec = {
        "status": "STAGED",
        "schema_version": OOF_CONTRACT_VERSION,
        "contract_version": OOF_CONTRACT_VERSION,
        "phase": OOF_PHASE,
        "run_id": run_id,
        "committed_run_dir": str(final_root),
        "preprocess_ready": _record(preprocess_ready_path.resolve()),
        "preprocess_bound_hashes": bound,
        "splits_final": splits["splits_final"],
        "full_train_ready": models[0]["full_train_ready"],
        "fold_receipts": [
            models[fold]["fold_receipt"] for fold in range(EXPECTED_FOLDS)
        ],
        "fold_count": EXPECTED_FOLDS,
        "case_count": EXPECTED_CASES,
        "patient_count": EXPECTED_PATIENTS,
        "val_exact_once": True,
        "patient_single_held_out_fold": True,
        "compile_contract": INFERENCE_COMPILE_CONTRACT,
        "probability_verification_boundary": PROBABILITY_VERIFICATION_BOUNDARY,
        "scribble_generation_count": 0,
        "intent_generation_count": 0,
        "experiment_result_count": 0,
        "thesis_citable": False,
    }
    _write_json_exclusive(staging_root / "OOF_SPEC.json", spec)
    return spec


def _read_plan(run_root: Path, fold: int) -> tuple[Path, dict[str, Any]]:
    path = run_root / "fold_plans" / f"fold_{fold}.json"
    plan = _load_json(path, label=f"fold {fold} plan")
    _require(
        plan,
        {
            "schema_version": OOF_CONTRACT_VERSION,
            "phase": OOF_PHASE,
            "fold": fold,
            "use_folds": [fold],
            "save_probabilities": True,
            "overwrite": False,
            "compile_contract": INFERENCE_COMPILE_CONTRACT,
        },
        f"fold {fold} plan",
    )
    return path, plan


def _validate_fold_plan_semantics(
    plan: dict[str, Any],
    fold: int,
    *,
    run_id: str,
    splits: dict[str, Any],
    expected_model: dict[str, Any] | None = None,
) -> None:
    document = splits["document"]
    train_case_ids = sorted(document[fold]["train"])
    val_case_ids = sorted(document[fold]["val"])
    train_patient_ids = sorted(
        {splits["case_to_patient"][case_id] for case_id in train_case_ids}
    )
    val_patient_ids = sorted(
        {splits["case_to_patient"][case_id] for case_id in val_case_ids}
    )
    expected_fields = {
        "run_id": run_id,
        "configuration": CONFIGURATION,
        "trainer": TRAINER,
        "plans_identifier": PLANS_IDENTIFIER,
        "checkpoint_name": CHECKPOINT_NAME,
        "train_case_ids": train_case_ids,
        "val_case_ids": val_case_ids,
        "train_patient_ids": train_patient_ids,
        "val_patient_ids": val_patient_ids,
        "authoritative_split_binding": {
            "splits_final_sha256": splits["splits_final"]["sha256"],
            "train_case_ids_sha256": _canonical_hash(train_case_ids),
            "val_case_ids_sha256": _canonical_hash(val_case_ids),
            "train_patient_ids_sha256": _canonical_hash(train_patient_ids),
            "val_patient_ids_sha256": _canonical_hash(val_patient_ids),
        },
    }
    for key, expected in expected_fields.items():
        if plan.get(key) != expected:
            raise ContractError(
                f"fold {fold} plan {key} differs from authoritative splits_final"
            )
    if expected_model is not None and plan.get("model") != expected_model:
        raise ContractError(f"fold {fold} plan model differs from its fold receipt")

    case_rows = plan.get("cases")
    if not isinstance(case_rows, list) or len(case_rows) != len(val_case_ids):
        raise ContractError(f"fold {fold} plan case records do not match val cases")
    by_case = {row.get("case_id"): row for row in case_rows if isinstance(row, dict)}
    if set(by_case) != set(val_case_ids) or len(by_case) != len(case_rows):
        raise ContractError(f"fold {fold} plan case records are not exactly val cases")
    for case_id in val_case_ids:
        row = by_case[case_id]
        patient_id = splits["case_to_patient"][case_id]
        if row.get("patient_id") != patient_id or row.get("held_out_fold") != fold:
            raise ContractError(
                f"fold {fold} plan patient set differs from authoritative splits_final"
            )
        expected_outputs = {
            "mask_output": f"outputs/fold_{fold}/masks/{case_id}.nii.gz",
            "probability_output": (
                f"outputs/fold_{fold}/probabilities/{case_id}.npz"
            ),
        }
        for key, expected in expected_outputs.items():
            if row.get(key) != expected:
                raise ContractError(f"fold {fold} {case_id} {key} changed")
        paths: dict[str, Path] = {}
        for label in ("input_ct", "input_pet", "input_gt"):
            record = row.get(label)
            raw = record.get("path") if isinstance(record, dict) else None
            if not isinstance(raw, str):
                raise ContractError(f"fold {fold} {case_id} {label} is missing")
            paths[label] = Path(raw).resolve()
        if (
            paths["input_ct"].name != f"{case_id}_0000.nii.gz"
            or paths["input_pet"] != paths["input_ct"].with_name(
                f"{case_id}_0001.nii.gz"
            )
            or paths["input_gt"]
            != paths["input_ct"].parent.parent / "labelsTr" / f"{case_id}.nii.gz"
        ):
            raise ContractError(f"fold {fold} {case_id} input grid sources changed")


def validate_fold_plan_binding(
    run_root: Path, fold: int
) -> tuple[Path, dict[str, Any]]:
    """Rebind one execution plan to the hashed split and its exact fold model."""

    run_root = run_root.resolve()
    spec = _load_json(run_root / "OOF_SPEC.json", label="OOF_SPEC")
    split_record = spec.get("splits_final")
    split_path = Path(
        split_record.get("path", "") if isinstance(split_record, dict) else ""
    ).resolve()
    _verify_record(split_path, split_record, label="OOF_SPEC splits_final")
    splits = validate_authoritative_splits(split_path)
    if splits["splits_final"] != split_record:
        raise ContractError("OOF_SPEC splits_final record is not authoritative")
    receipts = spec.get("fold_receipts")
    if not isinstance(receipts, list) or len(receipts) != EXPECTED_FOLDS:
        raise ContractError("OOF_SPEC fold receipt set is incomplete")
    plan_path, plan = _read_plan(run_root, fold)
    model = plan.get("model")
    if not isinstance(model, dict):
        raise ContractError(f"fold {fold} plan model is missing")
    if model.get("full_train_ready") != spec.get("full_train_ready"):
        raise ContractError(f"fold {fold} plan uses a different training aggregate")
    if model.get("fold_receipt") != receipts[fold]:
        raise ContractError(f"fold {fold} plan uses a swapped fold receipt")
    receipt_record = receipts[fold]
    receipt_path = Path(receipt_record.get("path", "")).resolve()
    _verify_record(receipt_path, receipt_record, label=f"fold {fold} receipt")
    receipt = _load_json(receipt_path, label=f"fold {fold} receipt")
    if receipt.get("fold") != fold:
        raise ContractError(f"fold {fold} plan uses a swapped fold receipt")
    checkpoint_record = model.get("checkpoint")
    output_checkpoint = (
        receipt.get("output_contract", {}).get("artifacts", {}).get("checkpoint_final")
    )
    # Training receipts may store checkpoint paths relative to their campaign
    # root, whereas OOF plans deliberately use absolute paths so the detached
    # inference worker can validate them without depending on its CWD.  Compare
    # the resolved regular file and its immutable content record, rather than
    # requiring the two serialisations of the same path to be byte-identical.
    checkpoint_raw = (
        checkpoint_record.get("path") if isinstance(checkpoint_record, dict) else None
    )
    receipt_checkpoint_raw = (
        output_checkpoint.get("path") if isinstance(output_checkpoint, dict) else None
    )
    if not isinstance(checkpoint_raw, str) or not isinstance(receipt_checkpoint_raw, str):
        raise ContractError(f"fold {fold} checkpoint record is missing")
    checkpoint_path = Path(checkpoint_raw).resolve()
    receipt_checkpoint = Path(receipt_checkpoint_raw)
    campaign_root = receipt_path.parent.parent.resolve()
    if not receipt_checkpoint.is_absolute():
        if ".." in receipt_checkpoint.parts:
            raise ContractError(f"fold {fold} receipt checkpoint path is unsafe")
        receipt_checkpoint = campaign_root / receipt_checkpoint
    receipt_checkpoint = receipt_checkpoint.resolve()
    if (
        checkpoint_path != receipt_checkpoint
        or checkpoint_record.get("bytes") != output_checkpoint.get("bytes")
        or checkpoint_record.get("sha256") != output_checkpoint.get("sha256")
    ):
        raise ContractError(f"fold {fold} plan checkpoint differs from fold receipt")
    _verify_record(checkpoint_path, checkpoint_record, label=f"fold {fold} plan checkpoint")
    _verify_record(
        receipt_checkpoint,
        output_checkpoint,
        label=f"fold {fold} receipt checkpoint",
    )
    _validate_fold_plan_semantics(
        plan,
        fold,
        run_id=str(spec.get("run_id") or ""),
        splits=splits,
    )
    return plan_path, plan


def _inventory_names(root: Path, suffix: str) -> set[str]:
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"missing OOF output directory: {root}")
    names: set[str] = set()
    for item in root.iterdir():
        if item.is_symlink() or not item.is_file():
            raise ContractError(f"OOF output must be a regular file: {item}")
        if not item.name.endswith(suffix):
            raise ContractError(f"unexpected OOF output artifact: {item.name}")
        names.add(item.name[: -len(suffix)])
    return names


def _validate_prediction_source_artifacts(
    prediction_source: str,
    source_artifacts: dict[str, Any] | None,
    *,
    fold: int,
) -> dict[str, Any] | None:
    if prediction_source not in PREDICTION_SOURCES:
        raise ContractError("OOF prediction source is not an allowed native path")
    if prediction_source == "DEDICATED_OOF_INFERENCE":
        if source_artifacts not in (None, {}):
            raise ContractError("dedicated OOF inference must not claim handoff artifacts")
        return None
    if not isinstance(source_artifacts, dict) or not source_artifacts:
        raise ContractError("actual-validation handoff requires hash-bound source artifacts")
    normalized: dict[str, Any] = {}
    for label, record in sorted(source_artifacts.items()):
        if not isinstance(label, str) or not label or not isinstance(record, dict):
            raise ContractError("handoff source_artifacts must be named file records")
        raw = record.get("path")
        if not isinstance(raw, str) or not raw:
            raise ContractError(f"handoff source artifact path is missing: {label}")
        path = Path(raw)
        if path.is_symlink():
            raise ContractError(f"handoff source artifact is a symlink: {label}")
        path = path.resolve()
        _verify_record(path, record, label=f"handoff source artifact {label}")
        normalized[label] = dict(record)
    handoff_record = normalized.get("handoff_source")
    if isinstance(handoff_record, dict):
        document = _load_json(
            Path(handoff_record["path"]).resolve(), label="HANDOFF_SOURCE"
        )
        if (
            document.get("fold") != fold
            or document.get("prediction_source")
            != "TRAINING_ACTUAL_VALIDATION_HANDOFF"
            or document.get("status") not in {"PASS", "COMMITTED", "VALIDATED"}
        ):
            raise ContractError("HANDOFF_SOURCE semantic binding is invalid")
    return normalized


def commit_fold_done(
    run_root: Path,
    fold: int,
    *,
    prediction_source: str = "DEDICATED_OOF_INFERENCE",
    source_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a no-clobber per-fold completion receipt after exact val output."""

    run_root = run_root.resolve()
    plan_path, plan = validate_fold_plan_binding(run_root, fold)
    expected = set(plan["val_case_ids"])
    fold_root = run_root / "outputs" / f"fold_{fold}"
    if _inventory_names(fold_root / "masks", ".nii.gz") != expected:
        raise ContractError(f"fold {fold} mask inventory is not exactly its val cases")
    if _inventory_names(fold_root / "probabilities", ".npz") != expected:
        raise ContractError(
            f"fold {fold} probability inventory is not exactly its val cases"
        )
    validated_source_artifacts = _validate_prediction_source_artifacts(
        prediction_source, source_artifacts, fold=fold
    )
    receipt = {
        "status": "COMMITTED",
        "schema_version": OOF_CONTRACT_VERSION,
        "phase": OOF_PHASE,
        "fold": fold,
        "plan": _record(plan_path, root=run_root),
        "prediction_count": len(expected),
        "foreground_probability_count": len(expected),
        "compile_contract": INFERENCE_COMPILE_CONTRACT,
        "prediction_source": prediction_source,
        "source_artifacts": validated_source_artifacts,
        "scribble_generation_count": 0,
        "intent_generation_count": 0,
    }
    _write_json_exclusive(fold_root / "FOLD_DONE.json", receipt)
    return receipt


def inspect_oof_pair(
    mask: Path, probability: Path, reference_ct: Path, reference_gt: Path
) -> dict[str, Any]:
    """Validate OOF artifacts on the immutable CT/GT grid.

    The probability NPZ has no spatial header. Its spatial claim is therefore
    limited to an exact array-shape binding to the affine-verified mask.
    """

    import nibabel as nib
    import numpy as np

    mask_image = nib.load(str(mask))
    ct_image = nib.load(str(reference_ct))
    gt_image = nib.load(str(reference_gt))
    for label, image in (("CT", ct_image), ("GT", gt_image)):
        if image.shape != mask_image.shape or not np.allclose(
            image.affine, mask_image.affine, atol=1e-4, rtol=0.0
        ):
            raise ContractError(f"OOF mask grid differs from reference {label}")
    if len(mask_image.shape) != 3 or not np.isfinite(mask_image.affine).all():
        raise ContractError("OOF mask affine/grid must be finite 3D")
    mask_data = np.asanyarray(mask_image.dataobj)
    values = np.unique(mask_data)
    if (
        mask_data.ndim != 3
        or not np.isfinite(mask_data).all()
        or not set(values.tolist()) <= {0, 1}
    ):
        raise ContractError("OOF mask must be a finite 3D binary NIfTI")
    with np.load(probability, allow_pickle=False) as archive:
        if archive.files != ["foreground_probability"]:
            raise ContractError(
                "OOF probability NPZ must contain only foreground_probability"
            )
        foreground = archive["foreground_probability"]
    if foreground.dtype != np.float32:
        raise ContractError("OOF foreground probability must be float32")
    if foreground.shape != mask_data.shape:
        raise ContractError("OOF mask/probability shapes differ")
    finite = bool(np.isfinite(foreground).all())
    minimum, maximum = float(foreground.min()), float(foreground.max())
    if not finite or minimum < 0.0 or maximum > 1.0:
        raise ContractError("OOF foreground probability is not finite in [0,1]")
    return {
        "mask_shape": list(mask_data.shape),
        "probability_shape": list(foreground.shape),
        "mask_values": [int(value) for value in values],
        "probability_key": "foreground_probability",
        "probability_dtype": "float32",
        "probability_finite": True,
        "probability_min": minimum,
        "probability_max": maximum,
        "reference_grid": {
            "shape": list(mask_image.shape),
            "affine": np.asarray(mask_image.affine, dtype=float).tolist(),
            "ct_sha256": _sha256(reference_ct),
            "gt_sha256": _sha256(reference_gt),
        },
        "probability_verification_boundary": PROBABILITY_VERIFICATION_BOUNDARY,
    }


def validate_oof_output(
    run_root: Path,
    *,
    output_inspector: Callable[[Path, Path, Path, Path], dict[str, Any]] = inspect_oof_pair,
) -> dict[str, Any]:
    """Rehash every input/model/output and prove exact val-only prediction."""

    run_root = run_root.resolve()
    if run_root.is_symlink() or not run_root.is_dir():
        raise ContractError("OOF run root is missing")
    spec = _load_json(run_root / "OOF_SPEC.json", label="OOF_SPEC")
    _require(
        spec,
        {
            "status": "STAGED",
            "schema_version": OOF_CONTRACT_VERSION,
            "phase": OOF_PHASE,
            "fold_count": EXPECTED_FOLDS,
            "case_count": EXPECTED_CASES,
            "patient_count": EXPECTED_PATIENTS,
            "scribble_generation_count": 0,
            "intent_generation_count": 0,
            "experiment_result_count": 0,
            "compile_contract": INFERENCE_COMPILE_CONTRACT,
            "probability_verification_boundary": PROBABILITY_VERIFICATION_BOUNDARY,
        },
        "OOF_SPEC",
    )
    for label in ("preprocess_ready", "splits_final", "full_train_ready"):
        record = spec.get(label)
        raw = record.get("path") if isinstance(record, dict) else None
        if not isinstance(raw, str):
            raise ContractError(f"OOF_SPEC {label} record is missing")
        _verify_record(Path(raw).resolve(), record, label=f"OOF_SPEC {label}")
    preprocess_path = Path(spec["preprocess_ready"]["path"]).resolve()
    split_path = Path(spec["splits_final"]["path"]).resolve()
    full_train_path = Path(spec["full_train_ready"]["path"]).resolve()
    splits = validate_authoritative_splits(split_path)
    if splits["splits_final"] != spec["splits_final"]:
        raise ContractError("OOF_SPEC splits_final record is not authoritative")
    models = _validate_full_train_ready(
        [full_train_path], preprocess_path, split_path
    )
    if spec.get("full_train_ready") != models[0]["full_train_ready"]:
        raise ContractError("OOF_SPEC training aggregate changed")
    expected_receipts = [
        models[fold]["fold_receipt"] for fold in range(EXPECTED_FOLDS)
    ]
    if spec.get("fold_receipts") != expected_receipts:
        raise ContractError("OOF_SPEC fold receipt/model order changed")
    cases_out: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    patient_folds: defaultdict[str, set[int]] = defaultdict(set)
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(EXPECTED_FOLDS):
        plan_path, plan = _read_plan(run_root, fold)
        _validate_fold_plan_semantics(
            plan,
            fold,
            run_id=str(spec.get("run_id") or ""),
            splits=splits,
            expected_model=models[fold],
        )
        expected = set(plan.get("val_case_ids", []))
        if expected & set(plan.get("train_case_ids", [])):
            raise ContractError(f"fold {fold} plan has train/val case overlap")
        if len(plan.get("cases", [])) != len(expected):
            raise ContractError(f"fold {fold} plan case records do not match val cases")
        fold_root = run_root / "outputs" / f"fold_{fold}"
        mask_names = _inventory_names(fold_root / "masks", ".nii.gz")
        probability_names = _inventory_names(fold_root / "probabilities", ".npz")
        if mask_names != expected:
            raise ValueError(f"fold {fold} mask inventory differs from val-only plan")
        if probability_names != expected:
            raise ValueError(
                f"fold {fold} probability inventory differs from val-only plan"
            )
        done_path = fold_root / "FOLD_DONE.json"
        done = _load_json(done_path, label=f"fold {fold} completion receipt")
        _require(
            done,
            {
                "status": "COMMITTED",
                "schema_version": OOF_CONTRACT_VERSION,
                "phase": OOF_PHASE,
                "fold": fold,
                "prediction_count": len(expected),
                "foreground_probability_count": len(expected),
                "compile_contract": INFERENCE_COMPILE_CONTRACT,
            },
            f"fold {fold} completion receipt",
        )
        _verify_record(plan_path, done.get("plan"), label=f"fold {fold} plan")
        prediction_source = done.get("prediction_source")
        source_artifacts = _validate_prediction_source_artifacts(
            prediction_source, done.get("source_artifacts"), fold=fold
        )
        model = plan.get("model", {})
        for label in (
            "full_train_ready",
            "fold_receipt",
            "checkpoint",
            "plans",
            "dataset_json",
        ):
            record = model.get(label)
            raw = record.get("path") if isinstance(record, dict) else None
            if not isinstance(raw, str):
                raise ContractError(f"fold {fold} model {label} record is missing")
            _verify_record(
                Path(raw).resolve(), record, label=f"fold {fold} model {label}"
            )
        if model.get("source_tree_sha256") != EXPECTED_NNUNET_SOURCE_TREE_SHA256:
            raise ContractError(f"fold {fold} source tree identity changed")
        by_case = {item.get("case_id"): item for item in plan["cases"]}
        if set(by_case) != expected:
            raise ContractError(f"fold {fold} case records are not exactly val cases")
        for case_id in sorted(expected):
            if case_id in seen_cases:
                raise ContractError("OOF case was predicted more than once")
            seen_cases.add(case_id)
            item = by_case[case_id]
            patient_id = patient_from_case(case_id)
            if (
                item.get("patient_id") != patient_id
                or item.get("held_out_fold") != fold
            ):
                raise ContractError("OOF case patient/fold binding changed")
            input_paths: dict[str, Path] = {}
            for label in ("input_ct", "input_pet", "input_gt"):
                input_record = item.get(label)
                raw = (
                    input_record.get("path") if isinstance(input_record, dict) else None
                )
                if not isinstance(raw, str):
                    raise ContractError(f"OOF {label} record is missing")
                _verify_record(
                    Path(raw).resolve(), input_record, label=f"{case_id} {label}"
                )
                input_paths[label] = Path(raw).resolve()
            mask = run_root / item["mask_output"]
            probability = run_root / item["probability_output"]
            inspection = output_inspector(
                mask,
                probability,
                input_paths["input_ct"],
                input_paths["input_gt"],
            )
            if not isinstance(inspection, dict):
                raise ContractError("OOF output inspector did not return a receipt")
            patient_folds[patient_id].add(fold)
            cases_out.append(
                {
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "held_out_fold": fold,
                    "mask": _record(mask, root=run_root),
                    "foreground_probability": _record(probability, root=run_root),
                    "checkpoint_sha256": model["checkpoint"]["sha256"],
                    "plans_sha256": model["plans"]["sha256"],
                    "dataset_json_sha256": model["dataset_json"]["sha256"],
                    "source_tree_sha256": model["source_tree_sha256"],
                    "source_commit": model["source_commit"],
                    "splits_final_sha256": spec["splits_final"]["sha256"],
                    "preprocess_ready_sha256": spec["preprocess_ready"]["sha256"],
                    "full_train_ready_sha256": model["full_train_ready"]["sha256"],
                    "fold_receipt_sha256": model["fold_receipt"]["sha256"],
                    "input_ct_sha256": item["input_ct"]["sha256"],
                    "input_ct_bytes": item["input_ct"]["bytes"],
                    "input_pet_sha256": item["input_pet"]["sha256"],
                    "input_pet_bytes": item["input_pet"]["bytes"],
                    "input_gt_sha256": item["input_gt"]["sha256"],
                    "input_gt_bytes": item["input_gt"]["bytes"],
                    "inspection": inspection,
                    "prediction_source": prediction_source,
                    "source_artifacts_sha256": (
                        _canonical_hash(source_artifacts)
                        if source_artifacts is not None
                        else None
                    ),
                }
            )
        fold_summaries.append(
            {
                "fold": fold,
                "prediction_count": len(expected),
                "patient_count": len({patient_from_case(c) for c in expected}),
                "prediction_source": prediction_source,
                "source_artifacts": source_artifacts,
            }
        )
    if len(seen_cases) != EXPECTED_CASES:
        raise ContractError("OOF output does not contain exactly 597 cases")
    if len(patient_folds) != EXPECTED_PATIENTS or any(
        len(value) != 1 for value in patient_folds.values()
    ):
        raise ContractError("OOF patients do not map to a single held-out fold")
    cases_out.sort(key=lambda item: item["case_id"])
    return {
        "status": "PASS",
        "schema_version": OOF_CONTRACT_VERSION,
        "contract_version": OOF_CONTRACT_VERSION,
        "phase": OOF_PHASE,
        "case_count": EXPECTED_CASES,
        "patient_count": EXPECTED_PATIENTS,
        "prediction_count": EXPECTED_CASES,
        "foreground_probability_count": EXPECTED_CASES,
        "val_exact_once": True,
        "patient_single_held_out_fold": True,
        "folds": fold_summaries,
        "prediction_sources": sorted(
            {summary["prediction_source"] for summary in fold_summaries}
        ),
        "cases": cases_out,
        "case_manifest_sha256": _canonical_hash(cases_out),
        "scribble_generation_count": 0,
        "intent_generation_count": 0,
        "experiment_result_count": 0,
        "compile_contract": INFERENCE_COMPILE_CONTRACT,
        "probability_verification_boundary": PROBABILITY_VERIFICATION_BOUNDARY,
    }


def _validate_inventory(inventory: dict[str, Any]) -> None:
    _require(
        inventory,
        {
            "status": "PASS",
            "schema_version": OOF_CONTRACT_VERSION,
            "phase": OOF_PHASE,
            "case_count": EXPECTED_CASES,
            "patient_count": EXPECTED_PATIENTS,
            "prediction_count": EXPECTED_CASES,
            "foreground_probability_count": EXPECTED_CASES,
            "val_exact_once": True,
            "patient_single_held_out_fold": True,
            "scribble_generation_count": 0,
            "intent_generation_count": 0,
            "experiment_result_count": 0,
            "compile_contract": INFERENCE_COMPILE_CONTRACT,
            "probability_verification_boundary": PROBABILITY_VERIFICATION_BOUNDARY,
        },
        "OOF inventory",
    )
    if (
        not isinstance(inventory.get("cases"), list)
        or len(inventory["cases"]) != EXPECTED_CASES
    ):
        raise ContractError("OOF inventory case manifest is incomplete")
    prediction_sources = inventory.get("prediction_sources")
    if (
        not isinstance(prediction_sources, list)
        or not prediction_sources
        or prediction_sources != sorted(set(prediction_sources))
        or not set(prediction_sources) <= PREDICTION_SOURCES
    ):
        raise ContractError("OOF inventory prediction_sources is invalid")


def build_oof_bundle(
    preprocess_ready_path: Path,
    full_train_ready_paths: Iterable[Path],
    *,
    run_id: str,
    committed_run_dir: Path,
    inventory: dict[str, Any],
    preprocess_validator: Callable[[Path], dict[str, Any]] = validate_preprocess_ready,
) -> dict[str, Any]:
    preprocess = preprocess_validator(preprocess_ready_path)
    committed_run_dir = committed_run_dir.resolve()
    if committed_run_dir.name != run_id:
        raise ContractError("OOF bundle run identity mismatch")
    _validate_inventory(inventory)
    spec_root = committed_run_dir
    if not spec_root.exists():
        partial = committed_run_dir.with_name(f".partial-{run_id}")
        spec_root = partial if partial.is_dir() else committed_run_dir
    spec = _load_json(spec_root / "OOF_SPEC.json", label="OOF_SPEC")
    split_record = spec.get("splits_final")
    split_path = Path(split_record.get("path", "")).resolve()
    _verify_record(split_path, split_record, label="splits_final")
    models = _validate_full_train_ready(
        list(full_train_ready_paths), preprocess_ready_path, split_path
    )
    if preprocess.get("bound_hashes") != spec.get("preprocess_bound_hashes"):
        raise ContractError("PREPROCESS_READY hashes changed during OOF generation")
    if inventory["case_manifest_sha256"] != _canonical_hash(inventory["cases"]):
        raise ContractError("OOF case manifest hash mismatch")
    return {
        "status": "VALIDATED",
        "oof_status": "PASS",
        "schema_version": OOF_CONTRACT_VERSION,
        "contract_version": OOF_CONTRACT_VERSION,
        "phase": OOF_PHASE,
        "run_id": run_id,
        "committed_run_dir": str(committed_run_dir),
        "preprocess_ready": _record(preprocess_ready_path.resolve()),
        "preprocess_bound_hashes": preprocess["bound_hashes"],
        "splits_final": split_record,
        "full_train_ready": models[0]["full_train_ready"],
        "fold_receipts": [
            models[fold]["fold_receipt"] for fold in range(EXPECTED_FOLDS)
        ],
        "patient_excluded": True,
        "compile_contract": INFERENCE_COMPILE_CONTRACT,
        "probability_verification_boundary": PROBABILITY_VERIFICATION_BOUNDARY,
        "prediction_sources": inventory["prediction_sources"],
        "output_contract": inventory,
        "scribble_generation_count": 0,
        "intent_generation_count": 0,
        "experiment_result_count": 0,
        "thesis_citable": False,
    }


def publish_oof_ready(
    run_dir: Path,
    bundle_path: Path,
    ready_path: Path,
    *,
    output_validator: Callable[[Path], dict[str, Any]] = validate_oof_output,
    preprocess_validator: Callable[[Path], dict[str, Any]] = validate_preprocess_ready,
) -> dict[str, Any]:
    """Atomically publish OOF_READY; any validation failure leaves no receipt."""

    run_dir, bundle_path = run_dir.resolve(), bundle_path.resolve()
    if run_dir.is_symlink() or not run_dir.is_dir() or bundle_path.parent != run_dir:
        raise ContractError("OOF_BUNDLE must be inside its committed run")
    bundle_bytes = bundle_path.read_bytes()
    bundle = _load_json(bundle_path, label="OOF_BUNDLE")
    _require(
        bundle,
        {
            "status": "VALIDATED",
            "oof_status": "PASS",
            "schema_version": OOF_CONTRACT_VERSION,
            "phase": OOF_PHASE,
            "run_id": run_dir.name,
            "committed_run_dir": str(run_dir),
            "patient_excluded": True,
            "compile_contract": INFERENCE_COMPILE_CONTRACT,
            "probability_verification_boundary": PROBABILITY_VERIFICATION_BOUNDARY,
            "scribble_generation_count": 0,
            "intent_generation_count": 0,
            "experiment_result_count": 0,
            "thesis_citable": False,
        },
        "OOF_BUNDLE",
    )
    preprocess_record = bundle.get("preprocess_ready")
    preprocess_path = Path(preprocess_record.get("path", "")).resolve()
    _verify_record(preprocess_path, preprocess_record, label="PREPROCESS_READY")
    preprocess = preprocess_validator(preprocess_path)
    if preprocess.get("bound_hashes") != bundle.get("preprocess_bound_hashes"):
        raise ContractError("PREPROCESS_READY hashes changed before OOF publication")
    fresh = output_validator(run_dir)
    _validate_inventory(fresh)
    if fresh != bundle.get("output_contract"):
        raise ContractError("OOF output changed before publication")
    if bundle.get("prediction_sources") != fresh.get("prediction_sources"):
        raise ContractError("OOF_BUNDLE prediction source summary changed")
    if _sha256(bundle_path) != hashlib.sha256(bundle_bytes).hexdigest():
        raise ContractError("OOF_BUNDLE changed during publication")
    ready = {
        "schema_version": OOF_READY_VERSION,
        "contract_version": OOF_CONTRACT_VERSION,
        "status": "COMMITTED",
        "oof_status": "PASS",
        "phase": OOF_PHASE,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "run_receipt": {
            "path": str(bundle_path),
            "bytes": len(bundle_bytes),
            "sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        },
        "patient_excluded": True,
        "compile_contract": INFERENCE_COMPILE_CONTRACT,
        "probability_verification_boundary": PROBABILITY_VERIFICATION_BOUNDARY,
        "prediction_sources": fresh["prediction_sources"],
        "case_count": EXPECTED_CASES,
        "patient_count": EXPECTED_PATIENTS,
        "prediction_count": EXPECTED_CASES,
        "foreground_probability_count": EXPECTED_CASES,
        "scribble_generation_count": 0,
        "intent_generation_count": 0,
        "experiment_result_count": 0,
        "thesis_citable": False,
        "validated_bundle": bundle,
    }
    _write_json_exclusive(ready_path.resolve(), ready)
    return ready


def _load_oof_ready_envelope(
    ready_path: Path,
) -> tuple[Path, dict[str, Any], Path, Path, dict[str, Any], dict[str, Any]]:
    """Validate only the receipt envelope and its embedded immutable bundle."""

    if ready_path.is_symlink():
        raise ContractError("OOF_READY must be a regular non-symlink file")
    ready_path = ready_path.resolve()
    ready = _load_json(ready_path, label="OOF_READY")
    _require(
        ready,
        {
            "schema_version": OOF_READY_VERSION,
            "contract_version": OOF_CONTRACT_VERSION,
            "status": "COMMITTED",
            "oof_status": "PASS",
            "phase": OOF_PHASE,
            "patient_excluded": True,
            "compile_contract": INFERENCE_COMPILE_CONTRACT,
            "probability_verification_boundary": PROBABILITY_VERIFICATION_BOUNDARY,
            "case_count": EXPECTED_CASES,
            "patient_count": EXPECTED_PATIENTS,
            "prediction_count": EXPECTED_CASES,
            "foreground_probability_count": EXPECTED_CASES,
            "scribble_generation_count": 0,
            "intent_generation_count": 0,
            "experiment_result_count": 0,
            "thesis_citable": False,
        },
        "OOF_READY",
    )
    run_dir = Path(ready.get("run_dir", "")).resolve()
    if (
        run_dir.is_symlink()
        or not run_dir.is_dir()
        or ready.get("run_id") != run_dir.name
    ):
        raise ContractError("OOF_READY run identity is invalid")
    receipt = ready.get("run_receipt")
    bundle = _resolve_record(receipt, run_dir, "OOF_BUNDLE")
    _verify_record(bundle, receipt, label="OOF_BUNDLE")
    document = _load_json(bundle, label="OOF_BUNDLE")
    if ready.get("validated_bundle") != document:
        raise ContractError("OOF_READY embedded bundle differs from hashed receipt")
    _require(
        document,
        {
            "status": "VALIDATED",
            "oof_status": "PASS",
            "schema_version": OOF_CONTRACT_VERSION,
            "contract_version": OOF_CONTRACT_VERSION,
            "phase": OOF_PHASE,
            "run_id": run_dir.name,
            "committed_run_dir": str(run_dir),
            "patient_excluded": True,
            "compile_contract": INFERENCE_COMPILE_CONTRACT,
            "probability_verification_boundary": PROBABILITY_VERIFICATION_BOUNDARY,
            "scribble_generation_count": 0,
            "intent_generation_count": 0,
            "experiment_result_count": 0,
            "thesis_citable": False,
        },
        "OOF_BUNDLE",
    )
    inventory = document.get("output_contract")
    if not isinstance(inventory, dict):
        raise ContractError("OOF_BUNDLE output contract is missing")
    _validate_inventory(inventory)
    if (
        ready.get("prediction_sources") != inventory.get("prediction_sources")
        or document.get("prediction_sources") != inventory.get("prediction_sources")
    ):
        raise ContractError("OOF prediction source summary differs across receipts")
    cases = inventory.get("cases")
    if inventory.get("case_manifest_sha256") != _canonical_hash(cases):
        raise ContractError("OOF embedded case manifest hash mismatch")
    seen: set[str] = set()
    patients: defaultdict[str, set[int]] = defaultdict(set)
    for case in cases:
        if not isinstance(case, dict):
            raise ContractError("OOF embedded case record must be an object")
        case_id = case.get("case_id")
        patient_id = case.get("patient_id")
        fold = case.get("held_out_fold")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen
            or not isinstance(patient_id, str)
            or not patient_id
            or isinstance(fold, bool)
            or not isinstance(fold, int)
            or fold not in range(EXPECTED_FOLDS)
        ):
            raise ContractError("OOF embedded case identity/fold is invalid")
        seen.add(case_id)
        patients[patient_id].add(fold)
        for key in ("mask", "foreground_probability"):
            record = case.get(key)
            raw = record.get("path") if isinstance(record, dict) else None
            size = record.get("bytes") if isinstance(record, dict) else None
            digest = record.get("sha256") if isinstance(record, dict) else None
            if (
                not isinstance(raw, str)
                or not raw
                or Path(raw).is_absolute()
                or ".." in Path(raw).parts
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ContractError(f"OOF embedded {key} record is invalid")
        for modality in ("ct", "pet", "gt"):
            digest = case.get(f"input_{modality}_sha256")
            size = case.get(f"input_{modality}_bytes")
            if (
                not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
            ):
                raise ContractError(
                    f"OOF embedded input_{modality} content binding is invalid"
                )
    if len(seen) != EXPECTED_CASES or len(patients) != EXPECTED_PATIENTS:
        raise ContractError("OOF embedded case/patient inventory is incomplete")
    if any(len(folds) != 1 for folds in patients.values()):
        raise ContractError("OOF embedded patient is assigned to multiple held-out folds")
    return ready_path, ready, run_dir, bundle, document, inventory


def validate_oof_ready_receipt_only(ready_path: Path) -> dict[str, Any]:
    """Validate OOF_READY without reading or hashing CT, PET, GT, or leaf outputs.

    This is the safe entry point before a downstream stage selects an authorized
    learning partition.  Leaf truth and prediction files must subsequently be
    closed with :func:`validate_oof_case_leaf` for each selected case.
    """

    ready_path, _, run_dir, _, _, inventory = _load_oof_ready_envelope(ready_path)
    return {
        "status": "PASS",
        "schema_version": OOF_READY_VERSION,
        "contract_version": OOF_CONTRACT_VERSION,
        "phase": OOF_PHASE,
        "ready_path": str(ready_path),
        "ready_sha256": _sha256(ready_path),
        "run_dir": str(run_dir),
        "patient_excluded": True,
        "validation_scope": "RECEIPT_ONLY_NO_LEAF_IO",
        "probability_verification_boundary": PROBABILITY_VERIFICATION_BOUNDARY,
        "prediction_sources": inventory["prediction_sources"],
        "cases": {item["case_id"]: item for item in inventory["cases"]},
    }


def validate_oof_ready(ready_path: Path) -> dict[str, Any]:
    """Rehash OOF_READY, its bundle, and the complete OOF output contract."""

    receipt_only = validate_oof_ready_receipt_only(ready_path)
    ready_path = Path(receipt_only["ready_path"])
    _, _, run_dir, _, document, _ = _load_oof_ready_envelope(ready_path)
    fresh = validate_oof_output(run_dir)
    if fresh != document.get("output_contract"):
        raise ContractError("OOF output changed after OOF_READY publication")
    return {
        **{key: value for key, value in receipt_only.items() if key != "cases"},
        "validation_scope": "FULL_REHASH_ALL_INPUTS_AND_OUTPUTS",
        "cases": {item["case_id"]: item for item in fresh["cases"]},
    }


def validate_oof_case_leaf(
    validated: dict[str, Any],
    *,
    ready_path: Path,
    case_id: str,
    source_record: dict[str, Any],
) -> dict[str, Any]:
    """Close one authorized case against source, OOF input, and leaf bytes.

    Callers must select the learning partition before invoking this function.
    It intentionally performs all CT/PET/GT and prediction leaf I/O, making the
    access boundary explicit and testable.
    """

    ready_path = ready_path.resolve()
    if (
        validated.get("status") != "PASS"
        or validated.get("patient_excluded") is not True
        or Path(str(validated.get("ready_path") or "")).resolve() != ready_path
        or validated.get("ready_sha256") != _sha256(ready_path)
    ):
        raise ContractError("validated OOF_READY receipt binding is invalid")
    cases = validated.get("cases")
    oof = cases.get(case_id) if isinstance(cases, dict) else None
    if not isinstance(oof, dict):
        raise ContractError("selected case is absent from OOF_READY")
    patient_id = str(source_record.get("patient_id") or "").casefold()
    held_out_fold = source_record.get("held_out_fold")
    if (
        source_record.get("case_id") != case_id
        or not patient_id
        or patient_id != oof.get("patient_id")
        or isinstance(held_out_fold, bool)
        or not isinstance(held_out_fold, int)
        or held_out_fold != oof.get("held_out_fold")
    ):
        raise ContractError("source case patient/fold differs from OOF_READY")

    inputs: dict[str, dict[str, Any]] = {}
    for modality in ("ct", "pet", "gt"):
        raw = Path(str(source_record.get(f"{modality}_path") or ""))
        if raw.is_symlink():
            raise ContractError(f"source {modality.upper()} must not be a symlink")
        path = raw.resolve()
        if not path.is_file():
            raise ContractError(f"source {modality.upper()} is missing")
        observed_bytes = path.stat().st_size
        observed_sha256 = _sha256(path)
        source_bytes = source_record.get(f"{modality}_bytes")
        source_sha256 = source_record.get(f"{modality}_sha256")
        oof_bytes = oof.get(f"input_{modality}_bytes")
        oof_sha256 = oof.get(f"input_{modality}_sha256")
        if (
            observed_bytes != source_bytes
            or observed_bytes != oof_bytes
            or observed_sha256 != source_sha256
            or observed_sha256 != oof_sha256
        ):
            raise ContractError(
                f"{case_id} {modality.upper()} observed/source/OOF content binding differs"
            )
        inputs[modality] = {
            "path": str(path),
            "bytes": observed_bytes,
            "sha256": observed_sha256,
            "source_manifest_sha256": source_sha256,
            "oof_input_sha256": oof_sha256,
        }

    run_dir = Path(str(validated.get("run_dir") or "")).resolve()
    leaf_records: dict[str, dict[str, Any]] = {}
    for label, key in (
        ("m0", "mask"),
        ("foreground_probability", "foreground_probability"),
    ):
        record = oof.get(key)
        path = _resolve_record(record, run_dir, label)
        _verify_record(path, record, label=f"{case_id} {label}")
        leaf_records[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    binding = {
        "schema_version": TRUTH_BINDING_VERSION,
        "case_id": case_id,
        "patient_id": patient_id,
        "held_out_fold": held_out_fold,
        "oof_ready_sha256": validated["ready_sha256"],
        "inputs": inputs,
        "m0": leaf_records["m0"],
        "foreground_probability": leaf_records["foreground_probability"],
    }
    binding["binding_sha256"] = _canonical_hash(binding)
    return binding


def build_natural_oof_binding_from_validated(
    validated: dict[str, Any],
    *,
    ready_path: Path,
    case_id: str,
    patient_id: str,
    m0_path: Path,
    leaf_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind one M0 after a caller has validated OOF_READY exactly once.

    Batch residual and scribble stages use this helper so the complete 597-case
    OOF inventory is not rehashed once per case.  The selected case artifacts
    are still rehashed here before their provenance binding is returned.
    """

    ready_path = ready_path.resolve()
    if (
        validated.get("status") != "PASS"
        or validated.get("patient_excluded") is not True
    ):
        raise ContractError("OOF_READY did not pass patient-exclusion validation")
    if Path(validated.get("ready_path", "")).resolve() != ready_path.resolve():
        raise ContractError("validated OOF_READY path differs from requested receipt")
    if validated.get("ready_sha256") != _sha256(ready_path.resolve()):
        raise ContractError("validated OOF_READY hash differs from requested receipt")
    cases = validated.get("cases", {})
    record = cases.get(case_id) if isinstance(cases, dict) else None
    if not isinstance(record, dict):
        raise ValueError("case_id is absent from OOF_READY")
    if record.get("patient_id") != patient_id.casefold():
        raise ValueError("patient identity does not match the OOF case record")
    run_dir = Path(validated["run_dir"]).resolve()
    expected_m0 = _resolve_record(record.get("mask"), run_dir, "M0")
    supplied_m0 = m0_path.resolve()
    if supplied_m0 != expected_m0:
        raise ValueError("M0 path does not match the OOF case record")
    if leaf_binding is None:
        if (
            expected_m0.is_symlink()
            or not expected_m0.is_file()
            or expected_m0.stat().st_size != record["mask"].get("bytes")
            or _sha256(expected_m0) != record["mask"].get("sha256")
        ):
            raise ValueError("M0 hash or byte-size does not match the OOF case record")
    elif (
        leaf_binding.get("case_id") != case_id
        or leaf_binding.get("patient_id") != patient_id.casefold()
        or leaf_binding.get("oof_ready_sha256") != validated["ready_sha256"]
        or leaf_binding.get("m0", {}).get("path") != str(expected_m0)
        or leaf_binding.get("m0", {}).get("sha256") != record["mask"].get("sha256")
    ):
        raise ValueError("truth binding does not match the selected natural M0")
    probability = _resolve_record(
        record.get("foreground_probability"), run_dir, "foreground probability"
    )
    if leaf_binding is None:
        if (
            probability.is_symlink()
            or not probability.is_file()
            or probability.stat().st_size != record["foreground_probability"].get("bytes")
            or _sha256(probability) != record["foreground_probability"].get("sha256")
        ):
            raise ValueError("foreground probability hash or byte-size mismatch")
    elif (
        leaf_binding.get("foreground_probability", {}).get("path") != str(probability)
        or leaf_binding.get("foreground_probability", {}).get("sha256")
        != record["foreground_probability"].get("sha256")
    ):
        raise ValueError("truth binding does not match the foreground probability")
    if validated.get("schema_version") == M0_V6_OOF_SCHEMA:
        # v6 OOF per-case records carry no training sha fields; resolve them
        # from the validated envelope (fold checkpoints, splits_final) and the
        # live trainer-root files.  The three legacy receipt hashes have no v6
        # equivalent canonical file and are bound as null (auditable absence).
        fold = int(record["held_out_fold"])
        checkpoints = validated.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 5:
            raise ContractError("M0 v6 validated checkpoint inventory is missing")
        checkpoint_record = checkpoints[fold].get("checkpoint")
        if not isinstance(checkpoint_record, dict):
            raise ContractError(f"fold {fold} checkpoint record is missing")
        checkpoint_path = Path(str(checkpoint_record["path"])).resolve()
        if (
            checkpoint_path.is_symlink()
            or not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != checkpoint_record.get("sha256")
        ):
            raise ContractError(f"fold {fold} checkpoint hash mismatch")
        trainer_root = checkpoint_path.parent.parent
        plans_path = trainer_root / "plans.json"
        dataset_path = trainer_root / "dataset.json"
        for path in (plans_path, dataset_path):
            if path.is_symlink() or not path.is_file():
                raise ContractError(f"trainer-root {path.name} is missing")
        splits_record = validated.get("splits_final")
        if not isinstance(splits_record, dict) or not isinstance(
            splits_record.get("path"), str
        ):
            raise ContractError("M0 v6 validated splits_final record is missing")
        splits_path = Path(str(splits_record["path"])).resolve()
        if (
            splits_path.is_symlink()
            or not splits_path.is_file()
            or _sha256(splits_path) != splits_record.get("sha256")
        ):
            raise ContractError("M0 v6 splits_final hash mismatch")
        sha_fields = {
            "checkpoint_sha256": checkpoint_record["sha256"],
            "plans_sha256": _sha256(plans_path),
            "dataset_json_sha256": _sha256(dataset_path),
            "source_tree_sha256": EXPECTED_NNUNET_SOURCE_TREE_SHA256,
            "splits_final_sha256": splits_record["sha256"],
            "preprocess_ready_sha256": None,
            "full_train_ready_sha256": None,
            "fold_receipt_sha256": None,
        }
    else:
        sha_fields = {
            "checkpoint_sha256": record["checkpoint_sha256"],
            "plans_sha256": record["plans_sha256"],
            "dataset_json_sha256": record["dataset_json_sha256"],
            "source_tree_sha256": record["source_tree_sha256"],
            "splits_final_sha256": record["splits_final_sha256"],
            "preprocess_ready_sha256": record["preprocess_ready_sha256"],
            "full_train_ready_sha256": record["full_train_ready_sha256"],
            "fold_receipt_sha256": record["fold_receipt_sha256"],
        }
    binding = {
        "kind": "patient_excluded_oof",
        "schema_version": NATURAL_PROVENANCE_VERSION,
        "contract_version": OOF_CONTRACT_VERSION,
        "held_out_fold": record["held_out_fold"],
        "oof_ready_sha256": validated["ready_sha256"],
        "m0_sha256": record["mask"]["sha256"],
        "foreground_probability_sha256": record["foreground_probability"]["sha256"],
        **sha_fields,
        "input_ct_sha256": record["input_ct_sha256"],
        "input_pet_sha256": record["input_pet_sha256"],
        "input_gt_sha256": record["input_gt_sha256"],
        "truth_binding_sha256": (
            leaf_binding.get("binding_sha256") if leaf_binding is not None else None
        ),
    }
    binding["binding_sha256"] = _canonical_hash(binding)
    return binding


def validate_natural_oof_binding(
    ready_path: Path,
    *,
    case_id: str,
    patient_id: str,
    m0_path: Path,
    ready_validator: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate OOF_READY and bind one natural episode M0 to its receipt."""

    ready_path = ready_path.resolve()
    if ready_validator is None:
        try:
            schema_version = json.loads(
                ready_path.read_text(encoding="utf-8")
            ).get("schema_version")
        except (OSError, UnicodeError, json.JSONDecodeError):
            schema_version = None
        if schema_version == M0_V6_OOF_SCHEMA:
            # The v6 OOF envelope carries no per-case training receipt fields;
            # validate it with the v6 lineage validator and let the binding
            # function's v6 branch resolve checkpoint/plan/split hashes.
            from common.petct_mainline_lineage import validate_m0_v6_oof_ready

            ready_validator = validate_m0_v6_oof_ready
        else:
            ready_validator = validate_oof_ready
    validated = ready_validator(ready_path)
    return build_natural_oof_binding_from_validated(
        validated,
        ready_path=ready_path,
        case_id=case_id,
        patient_id=patient_id,
        m0_path=m0_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    split = subs.add_parser("validate-splits")
    split.add_argument("splits", type=Path)
    stage = subs.add_parser("stage")
    stage.add_argument("preprocess_ready", type=Path)
    stage.add_argument("splits", type=Path)
    stage.add_argument("staging", type=Path)
    stage.add_argument("final", type=Path)
    stage.add_argument("run_id")
    stage.add_argument("full_train_ready", type=Path)
    done = subs.add_parser("commit-fold")
    done.add_argument("run_root", type=Path)
    done.add_argument("fold", type=int)
    validate = subs.add_parser("validate-oof")
    validate.add_argument("run_root", type=Path)
    validate.add_argument("preprocess_ready", type=Path)
    validate.add_argument("committed_run", type=Path)
    validate.add_argument("run_id")
    validate.add_argument("bundle", type=Path)
    validate.add_argument("full_train_ready", type=Path)
    commit = subs.add_parser("commit-run")
    commit.add_argument("staging", type=Path)
    commit.add_argument("final", type=Path)
    commit.add_argument("receipt", type=Path)
    publish = subs.add_parser("publish")
    publish.add_argument("run_root", type=Path)
    publish.add_argument("bundle", type=Path)
    publish.add_argument("ready", type=Path)
    check = subs.add_parser("validate-ready")
    check.add_argument("ready", type=Path)
    receipt_check = subs.add_parser("validate-ready-receipt")
    receipt_check.add_argument("ready", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate-splits":
        payload = validate_authoritative_splits(args.splits)
        payload.pop("document", None)
    elif args.command == "stage":
        payload = stage_oof_run(
            args.preprocess_ready,
            [args.full_train_ready],
            args.splits,
            args.staging,
            args.final,
            args.run_id,
        )
    elif args.command == "commit-fold":
        payload = commit_fold_done(args.run_root, args.fold)
    elif args.command == "validate-oof":
        inventory = validate_oof_output(args.run_root)
        payload = build_oof_bundle(
            args.preprocess_ready,
            [args.full_train_ready],
            run_id=args.run_id,
            committed_run_dir=args.committed_run,
            inventory=inventory,
        )
        _write_json_exclusive(args.bundle, payload)
    elif args.command == "commit-run":
        payload = commit_run_directory(args.staging, args.final, args.receipt)
    elif args.command == "publish":
        payload = publish_oof_ready(args.run_root, args.bundle, args.ready)
    elif args.command == "validate-ready-receipt":
        payload = validate_oof_ready_receipt_only(args.ready)
        payload["cases"] = len(payload["cases"])
    else:
        payload = validate_oof_ready(args.ready)
        payload["cases"] = len(payload["cases"])
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
