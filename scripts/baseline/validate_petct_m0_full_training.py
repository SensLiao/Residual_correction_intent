#!/usr/bin/env python3
"""Validate standard five-fold PET/CT M0 training without publishing OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from petct_m0_full_training_evidence import (
    load_checkpoint as _load_checkpoint,
    validate_fold_completion,
)

from validate_petct_m0_preprocess import (
    ContractError,
    _load_json,
    _sha256,
    _verify_record,
    _write_json_exclusive,
    validate_live_nnunet_runtime,
)
from validate_petct_m0_smoke import (
    CONTRACT_VERSION as SMOKE_CONTRACT_VERSION,
    PHASE as SMOKE_PHASE,
    validate_preprocess_ready,
    validate_smoke_output,
)


DATASET_ID = 901
DATASET_FOLDER = "Dataset901_PSMA_M0_AutoPETVNorm"
CONFIGURATION = "3d_fullres"
FOLDS = (0, 1, 2, 3, 4)
TRAINER = "nnUNetTrainer"
PLANS_IDENTIFIER = "nnUNetPlans"
NUM_EPOCHS = 1000
TRAINER_FOLDER = f"{TRAINER}__{PLANS_IDENTIFIER}__{CONFIGURATION}"
CONTRACT_VERSION = "1.0.0"
PHASE = "STANDARD_5FOLD_FULL_TRAINING"
COMPILE_MODES = {"triton-stub-link", "disabled"}


def _require_fields(
    payload: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
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


def _resolve_record(record: dict[str, Any], root: Path, label: str) -> Path:
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
        raise ContractError(f"{label} escapes its run root")
    return resolved


def _read_json_any(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} is missing or is not a regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid UTF-8 JSON") from exc


def validate_split_contract(
    splits_path: Path, *, expected_case_count: int = 597
) -> dict[str, Any]:
    """Require five fixed folds with disjoint train/val and exact-once val."""

    splits_path = splits_path.resolve()
    payload = _read_json_any(splits_path, "splits_final")
    if not isinstance(payload, list) or len(payload) != 5:
        raise ContractError("splits_final must contain exactly five folds")
    universe: set[str] | None = None
    fold_contract: dict[str, Any] = {}
    validation_counter: Counter[str] = Counter()
    for fold, entry in enumerate(payload):
        if not isinstance(entry, dict) or set(entry) != {"train", "val"}:
            raise ContractError(f"fold {fold} must contain only train and val")
        train = entry["train"]
        val = entry["val"]
        if not isinstance(train, list) or not isinstance(val, list):
            raise ContractError(f"fold {fold} train/val must be lists")
        if not all(isinstance(item, str) and item for item in train + val):
            raise ContractError(f"fold {fold} contains an invalid case identifier")
        if len(set(train)) != len(train) or len(set(val)) != len(val):
            raise ContractError(f"fold {fold} contains duplicate identifiers")
        train_set, val_set = set(train), set(val)
        if not train_set or not val_set or train_set & val_set:
            raise ContractError(f"fold {fold} train/val are not disjoint and non-empty")
        current_universe = train_set | val_set
        if universe is None:
            universe = current_universe
        elif current_universe != universe:
            raise ContractError("fold case universe differs across splits")
        validation_counter.update(val)
        fold_contract[str(fold)] = {
            "train": train,
            "val": val,
            "train_count": len(train),
            "val_count": len(val),
            "train_sha256": hashlib.sha256(
                "\n".join(train).encode("utf-8")
            ).hexdigest(),
            "val_sha256": hashlib.sha256("\n".join(val).encode("utf-8")).hexdigest(),
        }
    assert universe is not None
    if len(universe) != expected_case_count:
        raise ContractError(
            f"splits_final requires {expected_case_count} cases; observed {len(universe)}"
        )
    if set(validation_counter) != universe or any(
        count != 1 for count in validation_counter.values()
    ):
        raise ContractError("each case must occur in validation exactly once")
    return {
        "status": "PASS",
        "sha256": _sha256(splits_path),
        "case_count": len(universe),
        "fold_count": 5,
        "validation_exact_once": True,
        "folds": fold_contract,
    }


def validate_smoke_ready(
    ready_path: Path,
    *,
    preprocess_validator: Callable[[Path], dict[str, Any]] = validate_preprocess_ready,
    output_validator: Callable[[Path], dict[str, Any]] = validate_smoke_output,
) -> dict[str, Any]:
    """Rehash SMOKE_READY and revalidate its one-epoch output contract."""

    ready_path = ready_path.resolve()
    ready = _load_json(ready_path, label="SMOKE_READY")
    _require_fields(
        ready,
        {
            "status": "COMMITTED",
            "smoke_status": "PASS",
            "contract_version": SMOKE_CONTRACT_VERSION,
            "phase": SMOKE_PHASE,
            "smoke_training_performed": True,
            "full_training_status": "NOT_STARTED",
            "full_training_performed": False,
            "oof_prediction_count": 0,
            "result_count": 0,
            "thesis_citable": False,
        },
        "SMOKE_READY",
    )
    run_raw = ready.get("run_dir")
    if not isinstance(run_raw, str):
        raise ContractError("SMOKE_READY run_dir is missing")
    run_dir = Path(run_raw).resolve()
    if (
        run_dir.is_symlink()
        or not run_dir.is_dir()
        or ready.get("run_id") != run_dir.name
    ):
        raise ContractError("SMOKE_READY run identity is invalid")
    bundle_record = ready.get("run_receipt")
    bundle_path = _resolve_record(bundle_record, run_dir, "SMOKE_BUNDLE")
    _verify_record(bundle_path, bundle_record, label="SMOKE_BUNDLE")
    bundle = _load_json(bundle_path, label="SMOKE_BUNDLE")
    if bundle != ready.get("validated_bundle"):
        raise ContractError("SMOKE_READY embedded bundle differs from its receipt")
    preprocess_record = bundle.get("preprocess_ready")
    preprocess_raw = (
        preprocess_record.get("path") if isinstance(preprocess_record, dict) else None
    )
    if not isinstance(preprocess_raw, str):
        raise ContractError("SMOKE_BUNDLE preprocess receipt is missing")
    preprocess_path = Path(preprocess_raw).resolve()
    _verify_record(preprocess_path, preprocess_record, label="PREPROCESS_READY")
    preprocess = preprocess_validator(preprocess_path)
    if preprocess.get("bound_hashes") != bundle.get("preprocess_bound_hashes"):
        raise ContractError("smoke preprocessing hashes changed")
    output = output_validator(run_dir)
    if output != bundle.get("output_contract"):
        raise ContractError("smoke output changed after publication")
    return {
        "status": "PASS",
        "run_dir": str(run_dir),
        "bound_hashes": {
            "smoke_ready": _sha256(ready_path),
            "smoke_bundle": _sha256(bundle_path),
            **{
                f"preprocess_{key}": value
                for key, value in preprocess.get("bound_hashes", {}).items()
            },
        },
    }


def validate_inference_smoke_ready(
    ready_path: Path,
    base_prerequisites: dict[str, Any],
    *,
    output_validator: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rehash the actual-case mask/probability smoke gate without recursion."""

    ready_path = ready_path.resolve()
    ready = _load_json(ready_path, label="INFERENCE_SMOKE_READY")
    _require_fields(
        ready,
        {
            "status": "COMMITTED",
            "inference_smoke_status": "PASS",
            "contract_version": "1.0.0",
            "phase": "FOLD0_ACTUAL_CASE_INFERENCE_SMOKE",
            "full_training_status": "NOT_STARTED",
            "scientific_metrics_computed": False,
            "thesis_citable": False,
            "prediction_disposable": True,
            "receipt_retention_required": True,
            "oof_prediction_count": 0,
            "result_count": 0,
            "scribble_count": 0,
            "intent_count": 0,
        },
        "INFERENCE_SMOKE_READY",
    )
    run_raw = ready.get("run_dir")
    if not isinstance(run_raw, str):
        raise ContractError("INFERENCE_SMOKE_READY run_dir is missing")
    run_dir = Path(run_raw).resolve()
    if (
        run_dir.is_symlink()
        or not run_dir.is_dir()
        or ready.get("run_id") != run_dir.name
    ):
        raise ContractError("INFERENCE_SMOKE_READY run identity is invalid")
    bundle_record = ready.get("run_receipt")
    bundle_path = _resolve_record(bundle_record, run_dir, "INFERENCE_SMOKE_BUNDLE")
    _verify_record(bundle_path, bundle_record, label="INFERENCE_SMOKE_BUNDLE")
    bundle = _load_json(bundle_path, label="INFERENCE_SMOKE_BUNDLE")
    if bundle != ready.get("validated_bundle"):
        raise ContractError("INFERENCE_SMOKE_READY embedded bundle changed")
    _require_fields(
        bundle,
        {
            "status": "VALIDATED",
            "inference_smoke_status": "PASS",
            "contract_version": "1.0.0",
            "phase": "FOLD0_ACTUAL_CASE_INFERENCE_SMOKE",
            "run_id": run_dir.name,
            "committed_run_dir": str(run_dir),
            "full_training_status": "NOT_STARTED",
            "scientific_metrics_computed": False,
            "thesis_citable": False,
            "oof_prediction_count": 0,
            "result_count": 0,
            "scribble_count": 0,
            "intent_count": 0,
        },
        "INFERENCE_SMOKE_BUNDLE",
    )
    base_bound = base_prerequisites.get("bound_hashes")
    inference_bound = bundle.get("prerequisite_bound_hashes")
    if not isinstance(base_bound, dict) or not isinstance(inference_bound, dict):
        raise ContractError("inference-smoke prerequisite hashes are missing")
    for key in ("preprocess_ready", "smoke_ready", "splits_final", "source_tree"):
        if inference_bound.get(key) != base_bound.get(key):
            raise ContractError(
                f"inference-smoke {key} differs from full-training base"
            )
    base_paths = base_prerequisites.get("paths")
    if (
        not isinstance(base_paths, dict)
        or Path(str(bundle.get("source_root"))).resolve()
        != Path(str(base_paths.get("source_root"))).resolve()
    ):
        raise ContractError("inference-smoke nnU-Net source root changed")
    if output_validator is None:
        from validate_petct_m0_inference_smoke import validate_inference_smoke_output

        output_validator = validate_inference_smoke_output
    output = output_validator(run_dir)
    if output != bundle.get("output_contract"):
        raise ContractError("inference-smoke output changed after publication")
    probability = output.get("probability")
    _require_fields(
        output,
        {
            "status": "PASS",
            "prediction_count": 1,
            "mask_probability_consistent": True,
            "scientific_metrics_computed": False,
            "oof_prediction_count": 0,
            "result_count": 0,
            "scribble_count": 0,
            "intent_count": 0,
        },
        "inference-smoke output",
    )
    if not isinstance(probability, dict) or not (
        probability.get("finite") is True
        and probability.get("channel_sum_to_one") is True
        and probability.get("foreground_channel") == 1
    ):
        raise ContractError("inference-smoke probability evidence is incomplete")
    return {
        "status": "PASS",
        "run_dir": str(run_dir),
        "selected_case": ready.get("selected_case"),
        "ready_sha256": _sha256(ready_path),
        "bundle_sha256": _sha256(bundle_path),
        "base_bound_hashes": base_bound,
    }


def validate_training_prerequisites(
    preprocess_ready_path: Path,
    smoke_ready_path: Path,
    source_root: Path,
    *,
    expected_case_count: int = 597,
    preprocess_validator: Callable[[Path], dict[str, Any]] = validate_preprocess_ready,
    smoke_validator: Callable[[Path], dict[str, Any]] = validate_smoke_ready,
    runtime_validator: Callable[[Path], dict[str, Any]] = validate_live_nnunet_runtime,
    splits_path: Path | None = None,
) -> dict[str, Any]:
    preprocess = preprocess_validator(preprocess_ready_path)
    smoke = smoke_validator(smoke_ready_path)
    runtime = runtime_validator(source_root)
    if runtime.get("status") != "PASS" or runtime.get("version") != "2.8.1":
        raise ContractError("live nnU-Net runtime is not pinned v2.8.1")
    if splits_path is None:
        splits_path = (
            Path(preprocess["nnunet_preprocessed"])
            / DATASET_FOLDER
            / "splits_final.json"
        )
    split_contract = validate_split_contract(
        splits_path, expected_case_count=expected_case_count
    )
    planning_split_hash = preprocess.get("bound_hashes", {}).get(
        "planning_splits_final"
    )
    if (
        planning_split_hash is not None
        and planning_split_hash != split_contract["sha256"]
    ):
        raise ContractError(
            "live splits_final differs from PREPROCESS_READY planning hash"
        )
    preprocess_ready_path = preprocess_ready_path.resolve()
    smoke_ready_path = smoke_ready_path.resolve()
    source_root = source_root.resolve()
    return {
        "status": "PASS",
        "paths": {
            "preprocess_ready": str(preprocess_ready_path),
            "smoke_ready": str(smoke_ready_path),
            "source_root": str(source_root),
            "splits_final": str(splits_path.resolve()),
        },
        "bound_hashes": {
            "preprocess_ready": _sha256(preprocess_ready_path),
            "smoke_ready": _sha256(smoke_ready_path),
            "source_tree": runtime["source_tree_sha256"],
            "splits_final": split_contract["sha256"],
        },
        "runtime": runtime,
        "split_contract": split_contract,
        "preprocess": preprocess,
        "smoke": smoke,
    }


def validate_full_training_prerequisites(
    preprocess_ready_path: Path,
    smoke_ready_path: Path,
    inference_smoke_ready_path: Path,
    source_root: Path,
    *,
    expected_case_count: int = 597,
    splits_path: Path | None = None,
    base_prerequisite_validator: Callable[..., dict[str, Any]] = (
        validate_training_prerequisites
    ),
    inference_ready_validator: Callable[
        [Path, dict[str, Any]], dict[str, Any]
    ] = validate_inference_smoke_ready,
) -> dict[str, Any]:
    """Bind PREPROCESS + one-epoch + actual-case inference before full training."""

    base = base_prerequisite_validator(
        preprocess_ready_path,
        smoke_ready_path,
        source_root,
        expected_case_count=expected_case_count,
        splits_path=splits_path,
    )
    if base.get("status") != "PASS":
        raise ContractError("base full-training prerequisites are not PASS")
    inference = inference_ready_validator(inference_smoke_ready_path, base)
    if inference.get("status") != "PASS":
        raise ContractError("actual-case inference smoke is not PASS")
    paths = dict(base["paths"])
    paths["inference_smoke_ready"] = str(inference_smoke_ready_path.resolve())
    bound = dict(base["bound_hashes"])
    bound["inference_smoke_ready"] = inference["ready_sha256"]
    bound["inference_smoke_bundle"] = inference["bundle_sha256"]
    return {**base, "paths": paths, "bound_hashes": bound, "inference_smoke": inference}


def _compile_contract(mode: str, cuda_stub_dir: Path) -> dict[str, Any]:
    if mode not in COMPILE_MODES:
        raise ContractError(f"unsupported compile mode: {mode}")
    stub_dir = cuda_stub_dir.resolve()
    if mode == "disabled":
        return {
            "mode": mode,
            "nnunet_compile": "false",
            "library_path_injection": False,
            "ld_library_path_stub_forbidden": True,
            "cuda_stub_dir": str(stub_dir),
            "cuda_stub_libcuda_path": None,
            "cuda_stub_libcuda_sha256": None,
        }
    stub = stub_dir / "libcuda.so"
    if not stub.is_file():
        raise ContractError("compile-time libcuda stub is missing")
    return {
        "mode": mode,
        "nnunet_compile": "true",
        "library_path_injection": True,
        "ld_library_path_stub_forbidden": True,
        "cuda_stub_dir": str(stub_dir),
        "cuda_stub_libcuda_path": str(stub.resolve()),
        "cuda_stub_libcuda_sha256": _sha256(stub),
    }


def _training_contract(
    actual_validation: bool,
    export_probabilities: bool,
    compile_mode: str,
    cuda_stub_dir: Path,
) -> dict[str, Any]:
    if export_probabilities and not actual_validation:
        raise ContractError("probability export requires actual validation")
    return {
        "dataset_id": DATASET_ID,
        "configuration": CONFIGURATION,
        "folds": list(FOLDS),
        "trainer": TRAINER,
        "plans_identifier": PLANS_IDENTIFIER,
        "num_epochs": NUM_EPOCHS,
        "device": "cuda",
        "actual_validation": actual_validation,
        "export_probabilities": export_probabilities,
        "compile_contract": _compile_contract(compile_mode, cuda_stub_dir),
    }


def initialize_campaign(
    campaign_root: Path,
    campaign_id: str,
    prerequisites: dict[str, Any],
    *,
    actual_validation: bool = True,
    export_probabilities: bool = True,
    compile_mode: str = "triton-stub-link",
    cuda_stub_dir: Path = Path("/usr/local/cuda-11.6/targets/x86_64-linux/lib/stubs"),
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", campaign_id):
        raise ContractError("unsafe campaign id")
    campaign_root = campaign_root.resolve()
    if campaign_root.is_symlink() or not campaign_root.is_dir():
        raise ContractError("campaign root must be a fresh regular directory")
    if campaign_root.name != campaign_id:
        raise ContractError("campaign root does not match campaign id")
    if any(campaign_root.iterdir()):
        raise ContractError("campaign root must be empty")
    if prerequisites.get("status") != "PASS":
        raise ContractError("training prerequisites are not PASS")
    if prerequisites.get("inference_smoke", {}).get("status") != "PASS":
        raise ContractError("actual-case inference smoke prerequisite is not PASS")
    if "inference_smoke_ready" not in prerequisites.get("paths", {}):
        raise ContractError("INFERENCE_SMOKE_READY path is not bound")
    contract = _training_contract(
        actual_validation, export_probabilities, compile_mode, cuda_stub_dir
    )
    for name in ("nnUNet_results", "fold_receipts", "logs", "locks"):
        (campaign_root / name).mkdir()
    owner = {
        "status": "OWNED",
        "contract_version": CONTRACT_VERSION,
        "campaign_id": campaign_id,
        "owner_token": uuid4().hex,
    }
    _write_json_exclusive(campaign_root / "RUN_OWNER.json", owner)
    spec = {
        "status": "STAGED",
        "contract_version": CONTRACT_VERSION,
        "phase": PHASE,
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
        "prerequisite_paths": prerequisites["paths"],
        "prerequisite_bound_hashes": prerequisites["bound_hashes"],
        "expected_case_count": prerequisites["split_contract"]["case_count"],
        "training_contract": contract,
        "full_training_status": "NOT_STARTED",
        "oof_status": "NOT_STARTED",
        "oof_prediction_count": 0,
        "result_count": 0,
        "thesis_citable": False,
    }
    _write_json_exclusive(campaign_root / "CAMPAIGN_SPEC.json", spec)
    return spec


def _prerequisites_from_paths(
    paths: dict[str, Any], expected_case_count: int
) -> dict[str, Any]:
    return validate_full_training_prerequisites(
        Path(paths["preprocess_ready"]),
        Path(paths["smoke_ready"]),
        Path(paths["inference_smoke_ready"]),
        Path(paths["source_root"]),
        expected_case_count=expected_case_count,
        splits_path=Path(paths["splits_final"]),
    )


def validate_campaign(
    campaign_root: Path,
    *,
    prerequisite_validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    campaign_root = campaign_root.resolve()
    if campaign_root.is_symlink() or not campaign_root.is_dir():
        raise ContractError("campaign root is missing")
    allowed = {
        "RUN_OWNER.json",
        "CAMPAIGN_SPEC.json",
        "nnUNet_results",
        "fold_receipts",
        "logs",
        "locks",
    }
    if not {item.name for item in campaign_root.iterdir()}.issubset(allowed):
        raise ContractError("campaign root contains unexpected entries")
    owner = _load_json(campaign_root / "RUN_OWNER.json", label="RUN_OWNER")
    spec = _load_json(campaign_root / "CAMPAIGN_SPEC.json", label="CAMPAIGN_SPEC")
    campaign_id = campaign_root.name
    _require_fields(
        owner,
        {
            "status": "OWNED",
            "contract_version": CONTRACT_VERSION,
            "campaign_id": campaign_id,
        },
        "RUN_OWNER",
    )
    _require_fields(
        spec,
        {
            "status": "STAGED",
            "contract_version": CONTRACT_VERSION,
            "phase": PHASE,
            "campaign_id": campaign_id,
            "campaign_root": str(campaign_root),
            "full_training_status": "NOT_STARTED",
            "oof_status": "NOT_STARTED",
            "oof_prediction_count": 0,
            "result_count": 0,
            "thesis_citable": False,
        },
        "CAMPAIGN_SPEC",
    )
    contract = spec.get("training_contract")
    if not isinstance(contract, dict):
        raise ContractError("CAMPAIGN_SPEC training contract is missing")
    expected_contract = _training_contract(
        bool(contract.get("actual_validation")),
        bool(contract.get("export_probabilities")),
        str(contract.get("compile_contract", {}).get("mode")),
        Path(str(contract.get("compile_contract", {}).get("cuda_stub_dir", ""))),
    )
    if contract != expected_contract:
        raise ContractError("CAMPAIGN_SPEC standard training contract drift")
    paths = spec.get("prerequisite_paths")
    if not isinstance(paths, dict):
        raise ContractError("CAMPAIGN_SPEC prerequisite paths are missing")
    validator = prerequisite_validator or (
        lambda value: _prerequisites_from_paths(value, int(spec["expected_case_count"]))
    )
    prerequisites = validator(paths)
    if prerequisites.get("bound_hashes") != spec.get("prerequisite_bound_hashes"):
        raise ContractError(
            "training prerequisites changed after campaign initialization"
        )
    for name in ("nnUNet_results", "fold_receipts", "logs", "locks"):
        path = campaign_root / name
        if path.is_symlink() or not path.is_dir():
            raise ContractError(f"campaign directory is missing or unsafe: {name}")
    return {**spec, "prerequisites": prerequisites}


def _fold_root(campaign_root: Path, fold: int) -> Path:
    return (
        campaign_root
        / "nnUNet_results"
        / DATASET_FOLDER
        / TRAINER_FOLDER
        / f"fold_{fold}"
    )


def determine_fold_action(
    campaign_root: Path,
    fold: int,
    *,
    completed_receipt_validator: Callable[[Path, int], dict[str, Any]] | None = None,
) -> str:
    if fold not in FOLDS:
        raise ContractError("fold must be one of 0,1,2,3,4")
    campaign_root = campaign_root.resolve()
    receipt_path = campaign_root / "fold_receipts" / f"fold_{fold}.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        if completed_receipt_validator is None:
            receipt = _load_json(receipt_path, label=f"fold {fold} receipt")
            _require_fields(
                receipt, {"status": "COMMITTED", "fold": fold}, "fold receipt"
            )
        else:
            validated = completed_receipt_validator(campaign_root, fold)
            if validated.get("status") != "PASS":
                raise ContractError("completed fold receipt did not revalidate")
        return "SKIP_VERIFIED"
    fold_root = _fold_root(campaign_root, fold)
    if not fold_root.exists() and not fold_root.is_symlink():
        return "FRESH"
    if fold_root.is_symlink() or not fold_root.is_dir():
        raise ContractError("fold output is not a regular directory")
    checkpoints = [
        fold_root / name
        for name in (
            "checkpoint_latest.pth",
            "checkpoint_best.pth",
            "checkpoint_final.pth",
        )
    ]
    if any(path.is_file() and not path.is_symlink() for path in checkpoints):
        return "RESUME"
    raise ContractError("partial fold has no checkpoint and cannot resume safely")


def build_fold_receipt(
    campaign_root: Path,
    fold: int,
    prerequisites: dict[str, Any],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    if inventory.get("status") != "PASS" or inventory.get("fold") != fold:
        raise ContractError("fold inventory is not validated for requested fold")
    if inventory.get("oof_publication_count") or inventory.get(
        "result_publication_count"
    ):
        raise ContractError("fold receipt cannot publish OOF or results")
    campaign_root = campaign_root.resolve()
    return {
        "status": "COMMITTED",
        "contract_version": CONTRACT_VERSION,
        "phase": PHASE,
        "campaign_id": campaign_root.name,
        "fold": fold,
        "campaign_spec": _record(campaign_root / "CAMPAIGN_SPEC.json"),
        "prerequisite_bound_hashes": prerequisites["bound_hashes"],
        "output_contract": inventory,
        "full_fold_training_status": "PASS",
        "oof_status": "NOT_STARTED",
        "oof_prediction_count": 0,
        "result_count": 0,
        "thesis_citable": False,
    }


def publish_fold_receipt(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    _write_json_exclusive(path, payload)
    return payload


def validate_fold_receipt(
    campaign_root: Path,
    fold: int,
    prerequisites: dict[str, Any],
    *,
    checkpoint_loader: Callable[[Path], dict[str, Any]] = _load_checkpoint,
    oof_roots: Iterable[Path] = (),
    result_roots: Iterable[Path] = (),
) -> dict[str, Any]:
    receipt_path = campaign_root / "fold_receipts" / f"fold_{fold}.json"
    receipt = _load_json(receipt_path, label=f"fold {fold} receipt")
    _require_fields(
        receipt,
        {
            "status": "COMMITTED",
            "contract_version": CONTRACT_VERSION,
            "phase": PHASE,
            "campaign_id": campaign_root.name,
            "fold": fold,
            "full_fold_training_status": "PASS",
            "oof_status": "NOT_STARTED",
            "oof_prediction_count": 0,
            "result_count": 0,
            "thesis_citable": False,
        },
        "fold receipt",
    )
    if receipt.get("prerequisite_bound_hashes") != prerequisites.get("bound_hashes"):
        raise ContractError("fold receipt prerequisite hashes changed")
    spec_record = receipt.get("campaign_spec")
    spec_path = Path(spec_record.get("path", "")).resolve()
    _verify_record(spec_path, spec_record, label="CAMPAIGN_SPEC")
    campaign = _load_json(spec_path, label="CAMPAIGN_SPEC")
    contract = campaign["training_contract"]
    inventory = validate_fold_completion(
        campaign_root,
        fold,
        prerequisites["split_contract"],
        actual_validation=contract["actual_validation"],
        export_probabilities=contract["export_probabilities"],
        checkpoint_loader=checkpoint_loader,
        oof_roots=oof_roots,
        result_roots=result_roots,
    )
    if inventory != receipt.get("output_contract"):
        raise ContractError("fold output changed after receipt publication")
    return inventory


def publish_full_train_ready(
    campaign_root: Path,
    ready_path: Path,
    prerequisites: dict[str, Any],
    *,
    fold_validator: Callable[[Path, int, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    campaign_root = campaign_root.resolve()
    spec = _load_json(campaign_root / "CAMPAIGN_SPEC.json", label="CAMPAIGN_SPEC")
    inventories = []
    receipt_records = []
    for fold in FOLDS:
        receipt_path = campaign_root / "fold_receipts" / f"fold_{fold}.json"
        receipt = _load_json(receipt_path, label=f"fold {fold} receipt")
        inventory = fold_validator(campaign_root, fold, prerequisites)
        if receipt.get("output_contract") != inventory:
            raise ContractError(f"fold {fold} receipt differs from current output")
        inventories.append(inventory)
        receipt_records.append(_record(receipt_path))
    handoff_present = all(item["oof_handoff_inputs_present"] for item in inventories)
    published = {
        "status": "COMMITTED",
        "contract_version": CONTRACT_VERSION,
        "phase": PHASE,
        "campaign_id": campaign_root.name,
        "campaign_root": str(campaign_root),
        "campaign_spec": _record(campaign_root / "CAMPAIGN_SPEC.json"),
        "fold_receipts": receipt_records,
        "prerequisite_bound_hashes": prerequisites["bound_hashes"],
        "training_contract": spec["training_contract"],
        "full_training_status": "PASS",
        "full_training_performed": True,
        "folds_completed": list(FOLDS),
        "checkpoint_count": sum(item["checkpoint_count"] for item in inventories),
        "oof_handoff_inputs_present": handoff_present,
        "actual_inference_gate_required": not handoff_present,
        "oof_status": "NOT_STARTED",
        "oof_prediction_count": 0,
        "result_count": 0,
        "thesis_citable": False,
    }
    _write_json_exclusive(ready_path, published)
    return published


def _bool(value: str) -> bool:
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def _prerequisites_for_campaign(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = validate_campaign(root)
    return campaign, campaign["prerequisites"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-campaign")
    init.add_argument("--preprocess-ready", type=Path, required=True)
    init.add_argument("--smoke-ready", type=Path, required=True)
    init.add_argument("--inference-smoke-ready", type=Path, required=True)
    init.add_argument("--source-root", type=Path, required=True)
    init.add_argument("--campaign-root", type=Path, required=True)
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--actual-validation", type=_bool, required=True)
    init.add_argument("--export-probabilities", type=_bool, required=True)
    init.add_argument("--compile-mode", choices=sorted(COMPILE_MODES), required=True)
    init.add_argument("--cuda-stub-dir", type=Path, required=True)
    validate = commands.add_parser("validate-campaign")
    validate.add_argument("campaign_root", type=Path)
    action = commands.add_parser("fold-action")
    action.add_argument("campaign_root", type=Path)
    action.add_argument("fold", type=int, choices=FOLDS)
    fold = commands.add_parser("validate-fold")
    fold.add_argument("campaign_root", type=Path)
    fold.add_argument("fold", type=int, choices=FOLDS)
    fold.add_argument("--oof-root", type=Path, action="append", default=[])
    fold.add_argument("--result-root", type=Path, action="append", default=[])
    full = commands.add_parser("publish-full-ready")
    full.add_argument("campaign_root", type=Path)
    full.add_argument("ready", type=Path)
    full.add_argument("--oof-root", type=Path, action="append", default=[])
    full.add_argument("--result-root", type=Path, action="append", default=[])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "init-campaign":
        prerequisites = validate_full_training_prerequisites(
            args.preprocess_ready,
            args.smoke_ready,
            args.inference_smoke_ready,
            args.source_root,
        )
        payload = initialize_campaign(
            args.campaign_root,
            args.campaign_id,
            prerequisites,
            actual_validation=args.actual_validation,
            export_probabilities=args.export_probabilities,
            compile_mode=args.compile_mode,
            cuda_stub_dir=args.cuda_stub_dir,
        )
    elif args.command == "validate-campaign":
        payload = validate_campaign(args.campaign_root)
    elif args.command == "fold-action":
        campaign, prerequisites = _prerequisites_for_campaign(args.campaign_root)
        del campaign
        action = determine_fold_action(
            args.campaign_root,
            args.fold,
            completed_receipt_validator=lambda root, fold: validate_fold_receipt(
                root, fold, prerequisites
            ),
        )
        payload = {"status": "PASS", "fold": args.fold, "action": action}
    elif args.command == "validate-fold":
        campaign, prerequisites = _prerequisites_for_campaign(args.campaign_root)
        contract = campaign["training_contract"]
        inventory = validate_fold_completion(
            args.campaign_root,
            args.fold,
            prerequisites["split_contract"],
            actual_validation=contract["actual_validation"],
            export_probabilities=contract["export_probabilities"],
            oof_roots=args.oof_root,
            result_roots=args.result_root,
        )
        receipt = build_fold_receipt(
            args.campaign_root, args.fold, prerequisites, inventory
        )
        payload = publish_fold_receipt(
            args.campaign_root / "fold_receipts" / f"fold_{args.fold}.json",
            receipt,
        )
    else:
        campaign, prerequisites = _prerequisites_for_campaign(args.campaign_root)
        contract = campaign["training_contract"]
        payload = publish_full_train_ready(
            args.campaign_root,
            args.ready,
            prerequisites,
            fold_validator=lambda root, fold, pre: validate_fold_receipt(
                root,
                fold,
                pre,
                oof_roots=args.oof_root,
                result_roots=args.result_root,
            ),
        )
        if payload["training_contract"] != contract:
            raise ContractError("FULL_TRAIN_READY training contract changed")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
