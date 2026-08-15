#!/usr/bin/env python3
"""Run the pinned nnInteractive v1 checkpoint on frozen PET/M0/scribble records.

This adapter is intentionally narrow: PET is the only image channel, M0 is supplied
through nnInteractive's native initial-label/prev-seg interaction, and exactly one
foreground 3D scribble is then submitted. It never synthesizes points, boxes, lasso
prompts, background prompts, crops, or additional interaction rounds.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import nibabel as nib
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "scripts"
for directory in (SCRIPTS, SCRIPTS / "common"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common.petct_learning import (  # noqa: E402
    LearningContractError,
    validate_manifest_rows_against_frozen_learning_split,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)

DEFAULT_CONFIG = PROJECT / "configs" / "petct_external_comparators.json"
DEFAULT_EXPERIMENT_CONFIG = PROJECT / "configs" / "petct_route_a_experiment.json"
DEFAULT_MODEL_FOLDER = PROJECT / "models" / "nnInteractive" / "nnInteractive_v1.0"

ADAPTER_VERSION = "PETCT-NNINTERACTIVE-ADAPTER-v1.0"
METHOD_ID = "nninteractive"
SOURCE_COMMIT = "bbe12fdccc876cb2d4e0a47133811e362608e000"
INPUT_SCHEMA = "PETCT-EXTERNAL-COMPARATOR-INPUT-v1.0"
OUTPUT_SCHEMA = "PETCT-EXTERNAL-COMPARATOR-OUTPUT-v1.0"
OUTPUT_POLICIES = ("native_full_mask", "union_with_m0")
PUBLIC_TO_INTERNAL_PARTITION = {"train": "train", "validation": "val", "test": "test"}
PRETRAINING_EXPOSURE = "KNOWN_PUBLIC_COHORT_EXPOSURE"
EXACT_PSMA_V3_EXPOSURE = "UNKNOWN"
CHECKPOINT_RELATIVE_PATH = Path("fold_0") / "checkpoint_final.pth"
CHECKPOINT_SHA256 = "b3ac4421f85457bbd1aa0d87f5e67bcb7bc8e2ce6b824b6ac45077cc5d630ea9"
LICENSE_RELATIVE_PATH = Path("LICENSE")
LICENSE_SHA256 = "4f60f5747c5506020923866690c2a41a3c74ffa85b7371eac2b02e23185f91d5"
LICENSE_ID = "CC BY-NC-SA 4.0"
REQUIRED_MODEL_FILES = (
    Path("dataset.json"),
    Path("plans.json"),
    Path("inference_session_class.json"),
    LICENSE_RELATIVE_PATH,
    CHECKPOINT_RELATIVE_PATH,
)
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.-]+$")
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "offline",
    "PIP_NO_INDEX": "1",
}


class AdapterError(RuntimeError):
    """Raised when the comparator cannot honor its scientific contract."""


@dataclass(frozen=True)
class AdapterOptions:
    input_manifest: Path
    output_manifest: Path
    output_dir: Path
    model_folder: Path
    config: Path
    output_policy: str
    device: str
    torch_threads: int
    partition: str
    learning_split: Path
    experiment_config: Path
    test_access_receipt: Path | None
    run_root: Path | None
    derived_union_output_manifest: Path | None = None
    derived_union_output_dir: Path | None = None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AdapterError(f"{label} must contain a JSON object: {path}")
    return payload


def validate_model_folder(model_folder: Path, config_path: Path) -> dict[str, Any]:
    """Validate the fixed local model, license, and matching comparator declaration."""

    model_folder = model_folder.resolve()
    missing = [str(path) for path in REQUIRED_MODEL_FILES if not (model_folder / path).is_file()]
    if missing:
        raise AdapterError(f"nnInteractive model folder is incomplete; missing {missing}")

    checkpoint = model_folder / CHECKPOINT_RELATIVE_PATH
    license_file = model_folder / LICENSE_RELATIVE_PATH
    checkpoint_hash = sha256_file(checkpoint)
    license_hash = sha256_file(license_file)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise AdapterError(
            f"checkpoint hash mismatch: expected {CHECKPOINT_SHA256}, observed {checkpoint_hash}"
        )
    if license_hash != LICENSE_SHA256:
        raise AdapterError(f"license hash mismatch: expected {LICENSE_SHA256}, observed {license_hash}")
    first_nonempty = next(
        (line.strip() for line in license_file.read_text(encoding="utf-8").splitlines() if line.strip()),
        "",
    )
    if first_nonempty != LICENSE_ID:
        raise AdapterError(f"unexpected model license identifier: {first_nonempty!r}")

    config = _load_json(config_path.resolve(), "comparator config")
    methods = config.get("methods")
    if not isinstance(methods, list):
        raise AdapterError("comparator config methods must be a list")
    method = next((item for item in methods if isinstance(item, dict) and item.get("id") == METHOD_ID), None)
    if method is None:
        raise AdapterError(f"comparator config does not declare {METHOD_ID}")
    availability = method.get("pretraining", {}).get("local_checkpoint_availability", {})
    declared_checkpoint_hash = availability.get("sha256")
    declared_license_hash = availability.get("license_sha256")
    declared_license = availability.get("license")
    declared_exposure = method.get("pretraining", {}).get("current_psma_v3_exposure")
    declared_source_commit = method.get("source", {}).get("pinned_commit")
    if declared_checkpoint_hash != checkpoint_hash:
        raise AdapterError("comparator config checkpoint hash disagrees with the pinned local checkpoint")
    if declared_license_hash != license_hash or declared_license != LICENSE_ID:
        raise AdapterError("comparator config license declaration disagrees with the pinned model license")
    if declared_exposure != PRETRAINING_EXPOSURE:
        raise AdapterError(
            f"comparator config must retain exposure label {PRETRAINING_EXPOSURE}, got {declared_exposure!r}"
        )
    if declared_source_commit != SOURCE_COMMIT:
        raise AdapterError(
            f"comparator config must retain pinned source commit {SOURCE_COMMIT}, got {declared_source_commit!r}"
        )
    return {
        "model_folder": str(model_folder),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "license_path": str(license_file),
        "license_sha256": license_hash,
        "license_id": LICENSE_ID,
        "source_commit": SOURCE_COMMIT,
    }


def _validate_input_manifest(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != INPUT_SCHEMA:
        raise AdapterError(f"input schema must be {INPUT_SCHEMA}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise AdapterError("input manifest records must be a non-empty list")
    required = {
        "case_id": str,
        "patient_id": str,
        "split": str,
        "fold": int,
        "step": int,
        "pet_path": str,
        "ct_path": str,
        "m0_path": str,
        "fg_scribble_path": str,
        "original_grid_reference": str,
        "scribble_strategy": str,
        "scribble_polarity": str,
    }
    seen: set[tuple[str, int, int]] = set()
    clean: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise AdapterError(f"input record {index} must be an object")
        for field, expected in required.items():
            value = raw.get(field)
            if not isinstance(value, expected) or isinstance(value, bool):
                raise AdapterError(f"input record {index}.{field} must be {expected.__name__}")
            if expected is str and not value:
                raise AdapterError(f"input record {index}.{field} must not be empty")
        if raw["split"] not in {"train", "validation", "test"}:
            raise AdapterError(f"input record {index}.split is invalid")
        if raw["fold"] not in {0, 1, 2, 3, 4}:
            raise AdapterError(f"input record {index}.fold must be in [0, 4]")
        if raw["step"] < 1:
            raise AdapterError(f"input record {index}.step must be positive")
        if raw["scribble_strategy"] not in {"centerline", "random", "boundary"}:
            raise AdapterError(f"input record {index}.scribble_strategy is invalid")
        if raw["scribble_polarity"] != "foreground":
            raise AdapterError("nnInteractive adapter accepts only the frozen foreground scribble")
        if raw.get("bg_scribble_path") is not None:
            raise AdapterError("background scribbles are forbidden in this one-foreground-scribble comparator")
        if not SAFE_IDENTIFIER.fullmatch(raw["case_id"]):
            raise AdapterError(f"unsafe case_id in input record {index}: {raw['case_id']!r}")
        key = (raw["case_id"], raw["fold"], raw["step"])
        if key in seen:
            raise AdapterError(f"duplicate case/fold/step input record: {key}")
        seen.add(key)
        clean.append(dict(raw))
    return clean


def _validate_records_against_frozen_split(
    records: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    learning_split: Path,
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        internal = PUBLIC_TO_INTERNAL_PARTITION.get(str(record.get("split") or ""))
        if internal != partition:
            raise AdapterError(
                f"input record {index}.split does not match --partition {partition}"
            )
        split_receipt = record.get("patient_split_receipt")
        if not isinstance(split_receipt, Mapping):
            raise AdapterError(f"input record {index} omits patient_split_receipt")
        normalized.append(
            {
                "case_id": record["case_id"],
                "patient_id": record["patient_id"],
                "partition": internal,
                "learning_split_sha256": split_receipt.get(
                    "learning_split_sha256"
                ),
            }
        )
    try:
        return validate_manifest_rows_against_frozen_learning_split(
            normalized,
            learning_split,
            require_episode_id=False,
            allowed_partitions={partition},
        )
    except LearningContractError as exc:
        raise AdapterError(f"frozen learning-split validation failed: {exc}") from exc


def _resolve_record_path(raw: str, manifest_parent: Path) -> Path:
    path = Path(raw)
    return (manifest_parent / path).resolve() if not path.is_absolute() else path.resolve()


def _load_nifti(path: Path, label: str) -> nib.spatialimages.SpatialImage:
    if not path.is_file():
        raise AdapterError(f"missing {label}: {path}")
    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise AdapterError(f"cannot read {label} {path}: {exc}") from exc
    if len(image.shape) != 3:
        raise AdapterError(f"{label} must be a 3D NIfTI, got shape {image.shape}")
    return image


def _assert_same_grid(
    image: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
    label: str,
) -> None:
    if image.shape != reference.shape:
        raise AdapterError(f"{label} shape {image.shape} differs from original grid {reference.shape}")
    if not np.allclose(image.affine, reference.affine, rtol=0.0, atol=1e-4):
        raise AdapterError(f"{label} affine differs from original grid; implicit resampling is forbidden")


def _binary_array(image: nib.spatialimages.SpatialImage, label: str, *, nonempty: bool) -> np.ndarray:
    values = np.asanyarray(image.dataobj)
    if not np.all(np.isfinite(values)):
        raise AdapterError(f"{label} contains non-finite values")
    unique = np.unique(values)
    if not set(unique.tolist()) <= {0, 1}:
        raise AdapterError(f"{label} must be binary 0/1, observed values {unique[:10].tolist()}")
    output = np.asarray(values, dtype=np.uint8, order="C")
    if nonempty and not np.any(output):
        raise AdapterError(f"{label} is empty")
    return output


def _pet_array(image: nib.spatialimages.SpatialImage) -> np.ndarray:
    values = np.asarray(image.dataobj, dtype=np.float32, order="C")
    if not np.all(np.isfinite(values)):
        raise AdapterError("PET image contains non-finite values")
    return values


def validate_session_api(session: Any) -> None:
    required_parameters = {
        "set_image": {"image"},
        "set_target_buffer": {"target_buffer"},
        "add_initial_seg_interaction": {"initial_seg", "run_prediction"},
        "add_scribble_interaction": {"scribble_image", "include_interaction", "run_prediction"},
    }
    for method_name, parameter_names in required_parameters.items():
        method = getattr(session, method_name, None)
        if not callable(method):
            raise AdapterError(f"installed nnInteractive session lacks {method_name}")
        observed = set(inspect.signature(method).parameters)
        missing = parameter_names - observed
        if missing:
            raise AdapterError(f"{method_name} API is incompatible; missing parameters {sorted(missing)}")
    if getattr(session, "supports_initial_label", None) is not True:
        raise AdapterError("loaded checkpoint does not advertise native initial-label/prev-seg support")
    supported = getattr(session, "supported_interactions", None)
    if not isinstance(supported, Mapping) or supported.get("scribble") is not True:
        raise AdapterError("loaded checkpoint does not advertise native scribble support")
    if getattr(session, "license", None) != LICENSE_ID:
        raise AdapterError(f"runtime model license is not {LICENSE_ID}")


def _default_session_factory(device: str, torch_threads: int, model_folder: Path) -> Any:
    os.environ.update(OFFLINE_ENVIRONMENT)
    try:
        import torch
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    except ImportError as exc:
        raise AdapterError(
            "nnInteractive runtime is unavailable; run scripts/setup/setup_nninteractive_env.sh"
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise AdapterError(f"CUDA device requested but torch.cuda.is_available() is false: {device}")
    session = nnInteractiveInferenceSession(
        device=torch.device(device),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=torch_threads,
        do_autozoom=True,
        enable_undo=False,
    )
    session.initialize_from_trained_model_folder(
        str(model_folder), use_fold=0, checkpoint_name="checkpoint_final.pth"
    )
    validate_session_api(session)
    return session


def _start_memory_probe(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _finish_memory_probe(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    import torch

    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def _atomic_save_nifti(mask: np.ndarray, reference: nib.spatialimages.SpatialImage, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp.nii.gz"
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qcode))
    if sform is not None:
        output.set_sform(sform, int(scode))
    try:
        nib.save(output, str(temp))
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def _prediction_path(output_dir: Path, record: Mapping[str, Any], output_policy: str) -> Path:
    name = f"{record['case_id']}__fold{record['fold']}__step{record['step']}__{output_policy}.nii.gz"
    return (output_dir / name).resolve()


def _failure_record(
    record: Mapping[str, Any],
    target: Path,
    runtime: float,
    peak_memory: float | None,
    error: BaseException,
    model: Mapping[str, Any],
    output_policy: str,
) -> dict[str, Any]:
    return {
        "case_id": record["case_id"],
        "patient_id": record["patient_id"],
        "method_id": METHOD_ID,
        "prediction_path": str(target),
        "original_grid_reference": record["original_grid_reference"],
        "prediction_semantics": "full_mask",
        "runtime_seconds": runtime,
        "peak_gpu_memory_mib": peak_memory,
        "source_checkpoint_id": f"nnInteractive_v1.0:{model['checkpoint_sha256']}",
        "status": "failed",
        "failure_reason": f"{type(error).__name__}: {error}",
        "prediction_sha256": None,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "model_license": model["license_id"],
        "model_license_sha256": model["license_sha256"],
        "pretraining_exposure": PRETRAINING_EXPOSURE,
        "exact_psma_v3_exposure": EXACT_PSMA_V3_EXPOSURE,
        "headline_eligible": False,
        "output_policy": output_policy,
        "m0_preservation_guaranteed": output_policy == "union_with_m0",
        "adapter_version": ADAPTER_VERSION,
    }


def _run_case(
    session: Any,
    record: Mapping[str, Any],
    manifest_parent: Path,
    target: Path,
    model: Mapping[str, Any],
    output_policy: str,
    device: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    peak_memory: float | None = None
    pet_path = _resolve_record_path(record["pet_path"], manifest_parent)
    m0_path = _resolve_record_path(record["m0_path"], manifest_parent)
    scribble_path = _resolve_record_path(record["fg_scribble_path"], manifest_parent)
    grid_path = _resolve_record_path(record["original_grid_reference"], manifest_parent)
    try:
        reference = _load_nifti(grid_path, "original grid reference")
        pet_image = _load_nifti(pet_path, "PET image")
        m0_image = _load_nifti(m0_path, "patient-excluded M0")
        scribble_image = _load_nifti(scribble_path, "frozen foreground scribble")
        _assert_same_grid(pet_image, reference, "PET image")
        _assert_same_grid(m0_image, reference, "M0")
        _assert_same_grid(scribble_image, reference, "foreground scribble")
        pet = _pet_array(pet_image)
        m0 = _binary_array(m0_image, "M0", nonempty=False)
        scribble = _binary_array(scribble_image, "foreground scribble", nonempty=True)

        _start_memory_probe(device)
        target_buffer = np.zeros(reference.shape, dtype=np.uint8)
        session.set_image(pet[None])
        session.set_target_buffer(target_buffer)
        # Native state interaction: patient-excluded M0 becomes prev_seg. No prediction yet.
        session.add_initial_seg_interaction(m0, run_prediction=False)
        # Exactly one native full-volume 3D foreground scribble; this triggers the only prediction.
        session.add_scribble_interaction(scribble, include_interaction=True, run_prediction=True)
        peak_memory = _finish_memory_probe(device)

        native = np.asarray(target_buffer)
        if native.shape != reference.shape or not set(np.unique(native).tolist()) <= {0, 1}:
            raise AdapterError("nnInteractive returned a non-binary mask or changed the original grid shape")
        prediction = (
            native.astype(np.uint8, copy=False)
            if output_policy == "native_full_mask"
            else np.logical_or(native, m0).astype(np.uint8)
        )
        _atomic_save_nifti(prediction, reference, target)
        runtime = time.perf_counter() - started
        return {
            "case_id": record["case_id"],
            "patient_id": record["patient_id"],
            "method_id": METHOD_ID,
            "prediction_path": str(target),
            "original_grid_reference": record["original_grid_reference"],
            "prediction_semantics": "full_mask",
            "runtime_seconds": runtime,
            "peak_gpu_memory_mib": peak_memory,
            "source_checkpoint_id": f"nnInteractive_v1.0:{model['checkpoint_sha256']}",
            "status": "complete",
            "failure_reason": None,
            "prediction_sha256": sha256_file(target),
            "checkpoint_sha256": model["checkpoint_sha256"],
            "model_license": model["license_id"],
            "model_license_sha256": model["license_sha256"],
            "pretraining_exposure": PRETRAINING_EXPOSURE,
            "exact_psma_v3_exposure": EXACT_PSMA_V3_EXPOSURE,
            "headline_eligible": False,
            "output_policy": output_policy,
            "m0_preservation_guaranteed": output_policy == "union_with_m0",
            "adapter_version": ADAPTER_VERSION,
            "pet_only_image_channel": True,
            "ct_consumed_by_model": False,
            "interaction_sequence": [
                "add_initial_seg_interaction(run_prediction=False)",
                "add_scribble_interaction(include_interaction=True,run_prediction=True)",
            ],
            "fg_scribble_sha256": sha256_file(scribble_path),
        }
    except Exception as exc:
        try:
            peak_memory = _finish_memory_probe(device)
        except Exception:
            peak_memory = None
        return _failure_record(
            record,
            target,
            time.perf_counter() - started,
            peak_memory,
            exc,
            model,
            output_policy,
        )


def _atomic_write_json(payload: Mapping[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def _derive_union_bundle(
    *,
    native_output: Mapping[str, Any],
    native_manifest_path: Path,
    input_records: Sequence[Mapping[str, Any]],
    input_manifest_parent: Path,
    output_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create the legacy positive-only diagnostic from one native model call."""

    output_manifest = output_manifest.resolve()
    output_dir = output_dir.resolve()
    if output_manifest.exists() or output_manifest.is_symlink():
        raise AdapterError(f"refusing to overwrite derived output manifest: {output_manifest}")
    if output_dir.exists() or output_dir.is_symlink():
        raise AdapterError(f"refusing to overwrite derived output directory: {output_dir}")
    if output_manifest == output_dir or output_manifest in output_dir.parents or output_dir in output_manifest.parents:
        raise AdapterError("derived output manifest and directory must not contain one another")
    native_records = native_output.get("records")
    if not isinstance(native_records, list) or len(native_records) != len(input_records):
        raise AdapterError("native output/input record count differs during union derivation")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    derived_records: list[dict[str, Any]] = []
    try:
        for source, input_record in zip(native_records, input_records):
            if not isinstance(source, dict) or source.get("status") != "complete":
                raise AdapterError("cannot derive union policy from a failed native record")
            if source.get("case_id") != input_record.get("case_id"):
                raise AdapterError("native output/input identity differs during union derivation")
            native_path = Path(str(source.get("prediction_path", ""))).resolve()
            declared_sha = source.get("prediction_sha256")
            if not native_path.is_file() or native_path.is_symlink() or sha256_file(native_path) != declared_sha:
                raise AdapterError("native prediction changed before union derivation")
            m0_path = _resolve_record_path(input_record["m0_path"], input_manifest_parent)
            grid_path = _resolve_record_path(
                input_record["original_grid_reference"], input_manifest_parent
            )
            reference = _load_nifti(grid_path, "original grid reference")
            native_image = _load_nifti(native_path, "native prediction")
            m0_image = _load_nifti(m0_path, "patient-excluded M0")
            _assert_same_grid(native_image, reference, "native prediction")
            _assert_same_grid(m0_image, reference, "M0")
            native = _binary_array(native_image, "native prediction", nonempty=False)
            m0 = _binary_array(m0_image, "M0", nonempty=False)
            final_target = _prediction_path(output_dir, input_record, "union_with_m0")
            staged_target = staging_dir / final_target.name
            _atomic_save_nifti(np.logical_or(native, m0).astype(np.uint8), reference, staged_target)
            row = dict(source)
            row.update(
                {
                    "prediction_path": str(final_target),
                    "prediction_sha256": sha256_file(staged_target),
                    "output_policy": "union_with_m0",
                    "m0_preservation_guaranteed": True,
                    "derived_without_model_inference": True,
                    "derived_from_output_policy": "native_full_mask",
                    "derived_from_prediction_sha256": declared_sha,
                    "native_inference_reused": True,
                }
            )
            derived_records.append(row)
        derived = dict(native_output)
        derived.update(
            {
                "output_policy": "union_with_m0",
                "derivation": {
                    "operation": "voxelwise_or(native_full_mask,m0)",
                    "model_inference_calls": 0,
                    "native_manifest_sha256": sha256_file(native_manifest_path),
                },
                "records": derived_records,
            }
        )
        os.rename(staging_dir, output_dir)
        try:
            _atomic_write_json(derived, output_manifest)
        except Exception:
            shutil.rmtree(output_dir)
            raise
        return derived
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def run_adapter(
    options: AdapterOptions,
    *,
    session_factory: Callable[[str, int, Path], Any] | None = None,
) -> dict[str, Any]:
    os.environ.update(OFFLINE_ENVIRONMENT)
    if options.output_policy not in OUTPUT_POLICIES:
        raise AdapterError(f"output policy must be one of {OUTPUT_POLICIES}")
    if (options.derived_union_output_manifest is None) != (options.derived_union_output_dir is None):
        raise AdapterError("derived union output manifest and directory must be supplied together")
    if options.derived_union_output_manifest is not None and options.output_policy != "native_full_mask":
        raise AdapterError("paired union derivation requires native_full_mask inference")
    if options.torch_threads < 1:
        raise AdapterError("torch_threads must be positive")
    input_manifest = options.input_manifest.resolve()
    output_manifest = options.output_manifest.resolve()
    output_dir = options.output_dir.resolve()
    try:
        access_receipt = enforce_partition_access(
            options.partition,
            receipt_path=options.test_access_receipt,
            experiment_config=options.experiment_config,
            learning_split=options.learning_split,
            run_root=options.run_root,
            output_paths=tuple(
                [output_manifest, output_dir]
                + (
                    [
                        options.derived_union_output_manifest.resolve(),
                        options.derived_union_output_dir.resolve(),
                    ]
                    if options.derived_union_output_manifest is not None
                    and options.derived_union_output_dir is not None
                    else []
                )
            ),
        )
    except TestAccessError as exc:
        raise AdapterError(str(exc)) from exc
    if output_manifest.exists():
        raise AdapterError(f"refusing to overwrite output manifest: {output_manifest}")

    payload = _load_json(input_manifest, "input manifest")
    records = _validate_input_manifest(payload)
    split_validation = _validate_records_against_frozen_split(
        records,
        partition=options.partition,
        learning_split=options.learning_split,
    )
    contains_test = options.partition == "test"
    model = validate_model_folder(options.model_folder, options.config)

    targets = [_prediction_path(output_dir, record, options.output_policy) for record in records]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise AdapterError(f"refusing to overwrite existing prediction(s): {existing[:5]}")

    factory = session_factory or _default_session_factory
    try:
        session = factory(options.device, options.torch_threads, options.model_folder.resolve())
        validate_session_api(session)
        init_error: BaseException | None = None
    except Exception as exc:
        session = None
        init_error = exc

    output_records: list[dict[str, Any]] = []
    for record, target in zip(records, targets):
        if init_error is not None:
            output_records.append(
                _failure_record(record, target, 0.0, None, init_error, model, options.output_policy)
            )
        else:
            output_records.append(
                _run_case(
                    session,
                    record,
                    input_manifest.parent,
                    target,
                    model,
                    options.output_policy,
                    options.device,
                )
            )

    output = {
        "schema_version": OUTPUT_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "method_id": METHOD_ID,
        "network_policy": "NO_DOWNLOADS",
        "offline_environment": OFFLINE_ENVIRONMENT,
        "output_policy": options.output_policy,
        "test_accessed": contains_test,
        "learning_split_sha256": split_validation["learning_split_sha256"],
        "test_access_receipt_sha256": (
            sha256_file(options.test_access_receipt)
            if access_receipt is not None
            else None
        ),
        "receipt_bound_run_root": (
            str(options.run_root.resolve()) if access_receipt is not None else None
        ),
        "model": model,
        "pretraining_exposure": PRETRAINING_EXPOSURE,
        "exact_psma_v3_exposure": EXACT_PSMA_V3_EXPOSURE,
        "headline_eligible": False,
        "records": output_records,
        "counts": {
            "total": len(output_records),
            "complete": sum(record["status"] == "complete" for record in output_records),
            "failed": sum(record["status"] == "failed" for record in output_records),
        },
    }
    _atomic_write_json(output, output_manifest)
    if options.derived_union_output_manifest is not None and options.derived_union_output_dir is not None:
        _derive_union_bundle(
            native_output=output,
            native_manifest_path=output_manifest,
            input_records=records,
            input_manifest_parent=input_manifest.parent,
            output_manifest=options.derived_union_output_manifest,
            output_dir=options.derived_union_output_dir,
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path, default=DEFAULT_MODEL_FOLDER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-policy", choices=OUTPUT_POLICIES, required=True)
    parser.add_argument("--derived-union-output-manifest", type=Path)
    parser.add_argument("--derived-union-output-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--partition", choices=("val", "test"), required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument(
        "--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG
    )
    add_leaf_test_access_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = AdapterOptions(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        output_dir=args.output_dir,
        model_folder=args.model_folder,
        config=args.config,
        output_policy=args.output_policy,
        device=args.device,
        torch_threads=args.torch_threads,
        partition=args.partition,
        learning_split=args.learning_split,
        experiment_config=args.experiment_config,
        test_access_receipt=args.test_access_receipt,
        run_root=args.run_root,
        derived_union_output_manifest=args.derived_union_output_manifest,
        derived_union_output_dir=args.derived_union_output_dir,
    )
    try:
        output = run_adapter(options)
    except AdapterError as exc:
        raise SystemExit(f"nnInteractive adapter contract failure: {exc}") from exc
    print(json.dumps(output["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
