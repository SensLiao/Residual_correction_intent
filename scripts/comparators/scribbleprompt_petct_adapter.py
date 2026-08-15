#!/usr/bin/env python3
"""Run the pinned ScribblePrompt-UNet on one frozen PET scribble slice.

This adapter intentionally implements a narrow comparator contract.  It uses PET
only, one foreground AutoPET-V raster, no click/box/background prompt, and M0 as
the previous-mask channel.  It never selects a slice from GT or an error mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


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


INPUT_SCHEMA = "PETCT-EXTERNAL-COMPARATOR-INPUT-v1.0"
OUTPUT_SCHEMA = "PETCT-EXTERNAL-COMPARATOR-OUTPUT-v1.0"
ADAPTER_SCHEMA = "PETCT-SCRIBBLEPROMPT-ADAPTER-v1.0"
METHOD_ID = "scribbleprompt"
SOURCE_COMMIT = "182c44975f77749b559974ce8db558c8bde57788"
CHECKPOINT_SHA256 = "43f57ee8fa8ec529c31be281e06749f9e629b30157bbbcc9baf200cddec1acbe"
CHECKPOINT_BYTES = 15_977_486
PUBLIC_TO_INTERNAL_PARTITION = {"train": "train", "validation": "val", "test": "test"}
MODEL_SIZE = (128, 128)
THRESHOLD = 0.5
OUTPUT_POLICIES = ("native_slice_replace", "union_with_m0")
PET_NORMALIZATION = "log1p-positive-iqr-z-clip[-5,5]-linear[0,1]-v1"
SOURCE_FILE_SHA256 = {
    "scribbleprompt/models/unet.py": (
        "155dd82338728a1761658536be602eb4bb438193e00fc55b47c5476b73c0d66a"
    ),
    "scribbleprompt/models/network.py": (
        "067030e73bc42d5290384c474ea2001876f1801c6df5a715d83d1f98bc9a4cda"
    ),
    "scribbleprompt/__init__.py": (
        "ecacb2233599c8de019587b0b0abe8540855f7b7b2ada6c6be93a748c408608e"
    ),
    "LICENSE": "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6",
}
REQUIRED_FIELDS = {
    "case_id",
    "patient_id",
    "split",
    "fold",
    "step",
    "pet_path",
    "ct_path",
    "m0_path",
    "fg_scribble_path",
    "bg_scribble_path",
    "original_grid_reference",
    "scribble_strategy",
    "scribble_polarity",
}
FORBIDDEN_MODEL_INPUT_FIELDS = {
    "gt_path",
    "label_path",
    "target_path",
    "fn_path",
    "fp_path",
    "fn_residual_path",
    "gold_intent",
    "intent_target",
}


class AdapterError(RuntimeError):
    """Raised when the fixed comparator contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise AdapterError("checkpoint does not exist: %s" % path)
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != CHECKPOINT_BYTES:
        raise AdapterError(
            "checkpoint byte count mismatch: %d != %d" % (size, CHECKPOINT_BYTES)
        )
    if digest != CHECKPOINT_SHA256:
        raise AdapterError("checkpoint SHA256 mismatch")
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest}


def verify_source(source_root: Path) -> Dict[str, Any]:
    if not source_root.is_dir():
        raise AdapterError(
            "ScribblePrompt source root does not exist: %s" % source_root
        )
    actual: Dict[str, str] = {}
    for relative, expected in SOURCE_FILE_SHA256.items():
        source_file = source_root / relative
        if not source_file.is_file():
            raise AdapterError("required source file is missing: %s" % source_file)
        digest = sha256_file(source_file)
        if digest != expected:
            raise AdapterError("pinned source file SHA256 mismatch: %s" % relative)
        actual[relative] = digest
    return {
        "path": str(source_root.resolve()),
        "pinned_commit": SOURCE_COMMIT,
        "required_file_sha256": actual,
        "required_files_bundle_sha256": _canonical_json_sha256(actual),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(
            "cannot read input manifest %s: %s" % (path, error)
        ) from error
    if not isinstance(value, dict):
        raise AdapterError("input manifest must be a JSON object")
    return value


def validate_input_manifest(value: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    if value.get("schema_version") != INPUT_SCHEMA:
        raise AdapterError("unsupported input manifest schema_version")
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise AdapterError("input manifest records must be a non-empty list")
    seen = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise AdapterError("input record %d must be an object" % index)
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            raise AdapterError(
                "input record %d is missing %s" % (index, sorted(missing))
            )
        forbidden = FORBIDDEN_MODEL_INPUT_FIELDS & set(record)
        if forbidden:
            raise AdapterError(
                "input record %d contains forbidden model input fields %s"
                % (index, sorted(forbidden))
            )
        for key in (
            "case_id",
            "patient_id",
            "pet_path",
            "ct_path",
            "m0_path",
            "fg_scribble_path",
            "original_grid_reference",
        ):
            if not isinstance(record[key], str) or not record[key]:
                raise AdapterError(
                    "input record %d.%s must be a non-empty string" % (index, key)
                )
        if record["bg_scribble_path"] is not None and (
            not isinstance(record["bg_scribble_path"], str)
            or not record["bg_scribble_path"]
        ):
            raise AdapterError(
                "input record %d.bg_scribble_path must be null or a non-empty string"
                % index
            )
        if record["split"] not in ("train", "validation", "test"):
            raise AdapterError("input record %d has unsupported split" % index)
        if isinstance(record["fold"], bool) or not isinstance(record["fold"], int):
            raise AdapterError("input record %d.fold must be an integer" % index)
        if (
            isinstance(record["step"], bool)
            or not isinstance(record["step"], int)
            or record["step"] < 1
        ):
            raise AdapterError(
                "input record %d.step must be a positive integer" % index
            )
        if record["scribble_strategy"] not in ("centerline", "random", "boundary"):
            raise AdapterError(
                "input record %d has unsupported scribble strategy" % index
            )
        if record["scribble_polarity"] != "foreground":
            raise AdapterError("only one foreground scribble is permitted")
        key = (record["case_id"], record["patient_id"], record["fold"], record["step"])
        if key in seen:
            raise AdapterError("duplicate comparator input key: %s" % (key,))
        seen.add(key)
    return records


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
                "input record %d split does not match --partition %s"
                % (index, partition)
            )
        split_receipt = record.get("patient_split_receipt")
        if not isinstance(split_receipt, Mapping):
            raise AdapterError(
                "input record %d omits patient_split_receipt" % index
            )
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
        raise AdapterError(
            "frozen learning-split validation failed: %s" % exc
        ) from exc


def _resolve_input_path(raw: str, manifest_dir: Path) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = manifest_dir / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise AdapterError("input file does not exist: %s" % candidate)
    return candidate


def _load_image(path: Path, label: str) -> Tuple[nib.Nifti1Image, np.ndarray]:
    try:
        image = nib.load(str(path))
        data = np.asarray(image.dataobj, dtype=np.float32)
    except Exception as error:
        raise AdapterError(
            "cannot load %s NIfTI %s: %s" % (label, path, error)
        ) from error
    if data.ndim != 3:
        raise AdapterError("%s must be a 3D NIfTI, got shape %s" % (label, data.shape))
    if not np.isfinite(data).all():
        raise AdapterError("%s contains non-finite voxels" % label)
    return image, data


def _assert_same_grid(
    reference: nib.Nifti1Image,
    candidate: nib.Nifti1Image,
    reference_name: str,
    candidate_name: str,
) -> None:
    if reference.shape != candidate.shape:
        raise AdapterError(
            "%s and %s shapes differ: %s != %s"
            % (reference_name, candidate_name, reference.shape, candidate.shape)
        )
    if not np.allclose(reference.affine, candidate.affine, rtol=0.0, atol=1e-4):
        raise AdapterError(
            "%s and %s affines differ" % (reference_name, candidate_name)
        )


def prompted_axial_slice(fg_scribble: np.ndarray, affine: np.ndarray) -> int:
    orientation = nib.aff2axcodes(affine)
    if len(orientation) != 3 or orientation[2] not in ("S", "I"):
        raise AdapterError(
            "grid axis 2 is not superior/inferior; cannot call it an axial slice: %s"
            % (orientation,)
        )
    occupied = np.flatnonzero(np.any(fg_scribble > 0, axis=(0, 1)))
    if occupied.size != 1:
        raise AdapterError(
            "foreground scribble must occupy exactly one prompted axial slice; got %s"
            % occupied.tolist()
        )
    return int(occupied[0])


def robust_pet_to_unit_interval(pet: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the fixed input-only PET mapping required by the 2D checkpoint."""

    clipped_suv = np.maximum(pet.astype(np.float32, copy=False), 0.0)
    logged = np.log1p(clipped_suv)
    positive = logged[clipped_suv > 0]
    if positive.size < 2:
        raise AdapterError("PET volume has fewer than two positive voxels")
    q25, median, q75 = np.percentile(positive, [25.0, 50.0, 75.0])
    iqr = max(float(q75 - q25), 1e-6)
    robust_z = (logged - float(median)) / iqr
    mapped = (np.clip(robust_z, -5.0, 5.0) + 5.0) / 10.0
    mapped[clipped_suv <= 0] = 0.0
    mapped = mapped.astype(np.float32, copy=False)
    if float(mapped.min()) < 0.0 or float(mapped.max()) > 1.0:
        raise AdapterError("PET normalization failed to produce [0,1]")
    return mapped, {
        "id": PET_NORMALIZATION,
        "statistics_scope": "whole input PET volume; positive voxels only",
        "q25_log1p_positive": float(q25),
        "median_log1p_positive": float(median),
        "q75_log1p_positive": float(q75),
        "iqr_epsilon": 1e-6,
        "iqr_used": iqr,
        "robust_z_clip": [-5.0, 5.0],
        "nonpositive_suv_output": 0.0,
    }


def _resize_2d(array: np.ndarray, *, mode: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32)).reshape(
        1, 1, *array.shape
    )
    kwargs: Dict[str, Any] = {"size": MODEL_SIZE, "mode": mode}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs)


def _restore_probability(
    probability: torch.Tensor, shape: Tuple[int, int]
) -> np.ndarray:
    restored = F.interpolate(
        probability.detach().float().cpu(),
        size=shape,
        mode="bilinear",
        align_corners=False,
    )
    return np.asarray(restored.squeeze().numpy(), dtype=np.float32)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise AdapterError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def load_official_model(source_root: Path, checkpoint: Path, device: str) -> Any:
    source_text = str(source_root.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        from scribbleprompt import ScribblePromptUNet
    except Exception as error:
        raise AdapterError(
            "cannot import pinned ScribblePrompt source: %s" % error
        ) from error
    ScribblePromptUNet.weights = {"v1": checkpoint.resolve()}
    try:
        model = ScribblePromptUNet(version="v1", device=device)
        model.model.eval()
    except Exception as error:
        raise AdapterError(
            "cannot load official ScribblePrompt-UNet: %s" % error
        ) from error
    return model


def _safe_stem(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    if not safe or safe in (".", ".."):
        raise AdapterError("case_id cannot be converted to a safe output name")
    return safe


def _save_mask(mask: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header=header)
    output.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    output.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(output, str(path))


def _input_hashes(paths: Mapping[str, Optional[Path]]) -> Dict[str, Optional[str]]:
    return {
        key: None if path is None else sha256_file(path)
        for key, path in sorted(paths.items())
    }


def _run_record(
    record: Mapping[str, Any],
    *,
    record_index: int,
    manifest_dir: Path,
    staging_dir: Path,
    output_dir: Path,
    output_policy: str,
    model: Any,
    device: str,
    checkpoint_sha256: str,
) -> Dict[str, Any]:
    started = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    resolved: Dict[str, Optional[Path]] = {
        "pet_path": _resolve_input_path(record["pet_path"], manifest_dir),
        "m0_path": _resolve_input_path(record["m0_path"], manifest_dir),
        "fg_scribble_path": _resolve_input_path(
            record["fg_scribble_path"], manifest_dir
        ),
        "bg_scribble_path": (
            None
            if record["bg_scribble_path"] is None
            else _resolve_input_path(record["bg_scribble_path"], manifest_dir)
        ),
        "original_grid_reference": _resolve_input_path(
            record["original_grid_reference"], manifest_dir
        ),
    }
    reference_image, _ = _load_image(
        resolved["original_grid_reference"],
        "original_grid_reference",  # type: ignore[arg-type]
    )
    pet_image, pet = _load_image(resolved["pet_path"], "PET")  # type: ignore[arg-type]
    m0_image, m0 = _load_image(resolved["m0_path"], "M0")  # type: ignore[arg-type]
    fg_image, fg = _load_image(
        resolved["fg_scribble_path"],
        "foreground scribble",  # type: ignore[arg-type]
    )
    for name, image in (
        ("PET", pet_image),
        ("M0", m0_image),
        ("foreground scribble", fg_image),
    ):
        _assert_same_grid(reference_image, image, "original_grid_reference", name)
    bg_path = resolved["bg_scribble_path"]
    if bg_path is not None:
        bg_image, bg = _load_image(bg_path, "background scribble")
        _assert_same_grid(
            reference_image, bg_image, "original_grid_reference", "background scribble"
        )
        if np.any(bg != 0):
            raise AdapterError("negative/background prompt must be empty")
    if not np.all(np.isclose(m0, 0.0) | np.isclose(m0, 1.0)):
        raise AdapterError("M0 must be binary 0/1")
    if not np.any(fg > 0):
        raise AdapterError("foreground scribble is empty")
    slice_index = prompted_axial_slice(fg, reference_image.affine)
    pet_unit, normalization = robust_pet_to_unit_interval(pet)

    image_tensor = _resize_2d(pet_unit[:, :, slice_index], mode="bilinear").to(device)
    m0_tensor = _resize_2d(
        (m0[:, :, slice_index] > 0).astype(np.float32), mode="nearest"
    ).to(device)
    fg_tensor = _resize_2d(
        (fg[:, :, slice_index] > 0).astype(np.float32), mode="bilinear"
    ).to(device)
    scribble_tensor = torch.cat((fg_tensor, torch.zeros_like(fg_tensor)), dim=1)
    if image_tensor.shape != (1, 1, 128, 128):
        raise AdapterError("explicit 128x128 image resize failed")
    with torch.no_grad():
        probability = model.predict(
            img=image_tensor,
            point_coords=None,
            point_labels=None,
            scribbles=scribble_tensor,
            box=None,
            mask_input=m0_tensor,
            return_logits=False,
        )
    if not isinstance(probability, torch.Tensor) or probability.shape != (
        1,
        1,
        128,
        128,
    ):
        raise AdapterError(
            "official model returned unexpected shape: %s"
            % (getattr(probability, "shape", None),)
        )
    raw_slice = _restore_probability(probability, m0.shape[:2]) >= THRESHOLD
    m0_binary = m0 > 0
    prediction = m0_binary.copy()
    if output_policy == "native_slice_replace":
        prediction[:, :, slice_index] = raw_slice
    elif output_policy == "union_with_m0":
        prediction[:, :, slice_index] = m0_binary[:, :, slice_index] | raw_slice
    else:  # pragma: no cover - argparse and run() both guard this
        raise AdapterError("unsupported output policy: %s" % output_policy)

    filename = "%05d_%s_step%d.nii.gz" % (
        record_index,
        _safe_stem(str(record["case_id"])),
        int(record["step"]),
    )
    staged_prediction = staging_dir / filename
    _save_mask(prediction, reference_image, staged_prediction)
    if device == "cuda":
        torch.cuda.synchronize()
        peak_gpu_memory_mib: Optional[float] = float(
            torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        )
    else:
        peak_gpu_memory_mib = None
    runtime = float(time.perf_counter() - started)
    final_prediction = output_dir / filename
    hashes = _input_hashes(resolved)
    # CT is part of the shared manifest but is deliberately not a model input.
    hashes["ct_path"] = None
    return {
        "case_id": record["case_id"],
        "patient_id": record["patient_id"],
        "method_id": METHOD_ID,
        "prediction_path": str(final_prediction.resolve()),
        "original_grid_reference": str(resolved["original_grid_reference"]),
        "prediction_semantics": "full_mask",
        "runtime_seconds": runtime,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "source_checkpoint_id": checkpoint_sha256,
        "status": "complete",
        "fold": record["fold"],
        "split": record["split"],
        "step": record["step"],
        "prompted_axial_slice_index": slice_index,
        "output_policy": output_policy,
        "m0_preservation_guaranteed": output_policy == "union_with_m0",
        "outside_prompted_slice_policy": "retain_m0",
        "prompted_slice_policy": (
            "replace_m0_with_native_thresholded_prediction"
            if output_policy == "native_slice_replace"
            else "union_native_thresholded_prediction_with_m0"
        ),
        "model_input": {
            "image": "PET only",
            "ct_consumed_by_model": False,
            "image_size": [128, 128],
            "previous_mask": "binary M0 direct in official previous-mask channel",
            "positive_scribble": "frozen AutoPET V raster, bilinear resize",
            "negative_scribble": "empty",
            "clicks": "empty",
            "box": "empty",
        },
        "prompt_budget": {
            "rounds": 1,
            "foreground_scribbles": 1,
            "foreground_scribble_voxels_original_grid": int(np.count_nonzero(fg > 0)),
            "background_scribbles": 0,
            "clicks": 0,
            "boxes": 0,
        },
        "pet_normalization": normalization,
        "input_sha256": hashes,
        "prediction_sha256": sha256_file(staged_prediction),
    }


def _failure_record(
    record: Mapping[str, Any],
    error: Exception,
    runtime: float,
    checkpoint_sha256: str,
    manifest_dir: Path,
) -> Dict[str, Any]:
    input_sha256: Dict[str, Optional[str]] = {}
    for key in (
        "pet_path",
        "ct_path",
        "m0_path",
        "fg_scribble_path",
        "bg_scribble_path",
        "original_grid_reference",
    ):
        raw = record.get(key)
        if raw is None:
            input_sha256[key] = None
            continue
        try:
            path = Path(str(raw))
            if not path.is_absolute():
                path = manifest_dir / path
            path = path.resolve()
            input_sha256[key] = sha256_file(path) if path.is_file() else None
        except OSError:
            input_sha256[key] = None
    return {
        "case_id": str(record.get("case_id", "")),
        "patient_id": str(record.get("patient_id", "")),
        "method_id": METHOD_ID,
        "prediction_path": "",
        "original_grid_reference": str(record.get("original_grid_reference", "")),
        "prediction_semantics": "full_mask",
        "runtime_seconds": float(runtime),
        "peak_gpu_memory_mib": None,
        "source_checkpoint_id": checkpoint_sha256,
        "status": "failed",
        "fold": record.get("fold"),
        "split": record.get("split"),
        "step": record.get("step"),
        "input_sha256": input_sha256,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _prepare_output_paths(output_manifest: Path, output_dir: Path) -> Tuple[Path, Path]:
    output_manifest = output_manifest.resolve()
    output_dir = output_dir.resolve()
    if output_manifest.exists() or os.path.lexists(str(output_manifest)):
        raise AdapterError("refusing existing output manifest: %s" % output_manifest)
    if output_dir.exists() or os.path.lexists(str(output_dir)):
        raise AdapterError("refusing existing output directory: %s" % output_dir)
    if (
        output_manifest == output_dir
        or output_manifest in output_dir.parents
        or output_dir in output_manifest.parents
    ):
        raise AdapterError(
            "output manifest and prediction directory must not contain one another"
        )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return output_manifest, output_dir


def _publish_bundle(
    *, staging_dir: Path, output_dir: Path, manifest_stage: Path, output_manifest: Path
) -> None:
    moved_dir = False
    try:
        os.rename(staging_dir, output_dir)
        moved_dir = True
        os.rename(manifest_stage, output_manifest)
    except Exception:
        if manifest_stage.exists():
            manifest_stage.unlink()
        if moved_dir and output_dir.exists():
            shutil.rmtree(output_dir)
        elif staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def _derive_union_bundle(
    *,
    native_manifest: Mapping[str, Any],
    native_manifest_path: Path,
    input_records: Sequence[Mapping[str, Any]],
    manifest_dir: Path,
    output_manifest: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Derive the legacy positive-only diagnostic from one native prediction set."""

    output_manifest, output_dir = _prepare_output_paths(output_manifest, output_dir)
    native_records = native_manifest.get("records")
    if not isinstance(native_records, list) or len(native_records) != len(input_records):
        raise AdapterError("native output/input record count differs during union derivation")
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".%s." % output_dir.name, dir=str(output_dir.parent))
    )
    manifest_fd, manifest_stage_raw = tempfile.mkstemp(
        prefix=".%s." % output_manifest.name,
        suffix=".partial",
        dir=str(output_manifest.parent),
    )
    os.close(manifest_fd)
    manifest_stage = Path(manifest_stage_raw)
    derived_records: list[Dict[str, Any]] = []
    try:
        for index, (source, input_record) in enumerate(zip(native_records, input_records)):
            if not isinstance(source, dict) or source.get("status") != "complete":
                raise AdapterError("cannot derive union policy from a failed native record")
            if source.get("case_id") != input_record.get("case_id") or source.get("step") != input_record.get("step"):
                raise AdapterError("native output/input identity differs during union derivation")
            native_path = Path(str(source.get("prediction_path", ""))).resolve()
            declared_sha = source.get("prediction_sha256")
            if not native_path.is_file() or native_path.is_symlink() or sha256_file(native_path) != declared_sha:
                raise AdapterError("native prediction changed before union derivation")
            m0_path = _resolve_input_path(input_record["m0_path"], manifest_dir)
            reference_path = _resolve_input_path(
                input_record["original_grid_reference"], manifest_dir
            )
            native_image, native = _load_image(native_path, "native prediction")
            m0_image, m0 = _load_image(m0_path, "M0")
            reference_image, _ = _load_image(reference_path, "original_grid_reference")
            _assert_same_grid(reference_image, native_image, "original_grid_reference", "native prediction")
            _assert_same_grid(reference_image, m0_image, "original_grid_reference", "M0")
            if not np.all(np.isclose(native, 0.0) | np.isclose(native, 1.0)):
                raise AdapterError("native prediction must be binary during union derivation")
            if not np.all(np.isclose(m0, 0.0) | np.isclose(m0, 1.0)):
                raise AdapterError("M0 must be binary during union derivation")
            final_name = native_path.name
            staged_path = staging_dir / final_name
            final_path = output_dir / final_name
            _save_mask(np.logical_or(native > 0, m0 > 0), reference_image, staged_path)
            row = dict(source)
            row.update(
                {
                    "prediction_path": str(final_path.resolve()),
                    "prediction_sha256": sha256_file(staged_path),
                    "output_policy": "union_with_m0",
                    "m0_preservation_guaranteed": True,
                    "prompted_slice_policy": "union_native_thresholded_prediction_with_m0",
                    "derived_without_model_inference": True,
                    "derived_from_output_policy": "native_slice_replace",
                    "derived_from_prediction_sha256": declared_sha,
                    "native_inference_reused": True,
                }
            )
            derived_records.append(row)
        derived = dict(native_manifest)
        derived.update(
            {
                "output_policy": "union_with_m0",
                "output_policy_was_explicit": True,
                "derivation": {
                    "operation": "voxelwise_or(native_slice_replace,m0)",
                    "model_inference_calls": 0,
                    "native_manifest_sha256": sha256_file(native_manifest_path),
                },
                "records": derived_records,
            }
        )
        manifest_stage.write_text(
            json.dumps(derived, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _publish_bundle(
            staging_dir=staging_dir,
            output_dir=output_dir,
            manifest_stage=manifest_stage,
            output_manifest=output_manifest,
        )
        return derived
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if manifest_stage.exists():
            manifest_stage.unlink()
        raise


def run(
    *,
    input_manifest: Path,
    output_manifest: Path,
    output_dir: Path,
    checkpoint: Path,
    source_root: Path,
    output_policy: str,
    device_request: str,
    partition: str,
    learning_split: Path,
    experiment_config: Path,
    test_access_receipt: Path | None,
    run_root: Path | None,
    derived_union_output_manifest: Path | None = None,
    derived_union_output_dir: Path | None = None,
) -> int:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PIP_NO_INDEX"] = "1"
    if output_policy not in OUTPUT_POLICIES:
        raise AdapterError("--output-policy must be explicit and supported")
    if (derived_union_output_manifest is None) != (derived_union_output_dir is None):
        raise AdapterError("derived union output manifest and directory must be supplied together")
    if derived_union_output_manifest is not None and output_policy != "native_slice_replace":
        raise AdapterError("paired union derivation requires native_slice_replace inference")
    input_manifest = input_manifest.resolve()
    output_manifest = output_manifest.resolve()
    output_dir = output_dir.resolve()
    access_outputs = [output_manifest, output_dir]
    if derived_union_output_manifest is not None and derived_union_output_dir is not None:
        access_outputs.extend(
            [derived_union_output_manifest.resolve(), derived_union_output_dir.resolve()]
        )
    try:
        access_receipt = enforce_partition_access(
            partition,
            receipt_path=test_access_receipt,
            experiment_config=experiment_config,
            learning_split=learning_split,
            run_root=run_root,
            output_paths=tuple(access_outputs),
        )
    except TestAccessError as exc:
        raise AdapterError(str(exc)) from exc
    output_manifest, output_dir = _prepare_output_paths(output_manifest, output_dir)
    input_value = _read_json(input_manifest)
    records = validate_input_manifest(input_value)
    split_validation = _validate_records_against_frozen_split(
        records,
        partition=partition,
        learning_split=learning_split,
    )
    checkpoint_info = verify_checkpoint(checkpoint.resolve())
    source_info = verify_source(source_root.resolve())
    device = _resolve_device(device_request)
    model = load_official_model(source_root.resolve(), checkpoint.resolve(), device)

    staging_dir = Path(
        tempfile.mkdtemp(prefix=".%s." % output_dir.name, dir=str(output_dir.parent))
    )
    manifest_fd, manifest_stage_raw = tempfile.mkstemp(
        prefix=".%s." % output_manifest.name,
        suffix=".partial",
        dir=str(output_manifest.parent),
    )
    os.close(manifest_fd)
    manifest_stage = Path(manifest_stage_raw)
    output_records = []
    failed = 0
    try:
        for index, record in enumerate(records):
            record_started = time.perf_counter()
            try:
                output_records.append(
                    _run_record(
                        record,
                        record_index=index,
                        manifest_dir=input_manifest.parent,
                        staging_dir=staging_dir,
                        output_dir=output_dir,
                        output_policy=output_policy,
                        model=model,
                        device=device,
                        checkpoint_sha256=checkpoint_info["sha256"],
                    )
                )
            except Exception as error:
                failed += 1
                output_records.append(
                    _failure_record(
                        record,
                        error,
                        time.perf_counter() - record_started,
                        checkpoint_info["sha256"],
                        input_manifest.parent,
                    )
                )
        manifest = {
            "schema_version": OUTPUT_SCHEMA,
            "adapter_schema_version": ADAPTER_SCHEMA,
            "status": "complete" if failed == 0 else "completed_with_failures",
            "method_id": METHOD_ID,
            "input_manifest": str(input_manifest),
            "input_manifest_sha256": sha256_file(input_manifest),
            "output_policy": output_policy,
            "output_policy_was_explicit": True,
            "threshold": THRESHOLD,
            "device": device,
            "offline_execution": True,
            "network_policy": "NO_DOWNLOADS",
            "test_accessed": any(record["split"] == "test" for record in records),
            "learning_split_sha256": split_validation["learning_split_sha256"],
            "test_access_receipt_sha256": (
                sha256_file(test_access_receipt)
                if access_receipt is not None
                else None
            ),
            "receipt_bound_run_root": (
                str(run_root.resolve()) if access_receipt is not None else None
            ),
            "source": source_info,
            "checkpoint": checkpoint_info,
            "checkpoint_license_status": "UNVERIFIED_NO_INDEPENDENT_WEIGHT_LICENSE_TEXT",
            "normalization_contract": PET_NORMALIZATION,
            "prompt_contract": {
                "slice_selection": "only non-empty axial slice in frozen foreground scribble",
                "oracle_slice_selection": False,
                "rounds": 1,
                "foreground_scribbles": 1,
                "negative_scribbles": 0,
                "clicks": 0,
                "boxes": 0,
                "previous_mask": "M0",
            },
            "record_count": len(output_records),
            "complete_count": len(output_records) - failed,
            "failed_count": failed,
            "records": output_records,
        }
        manifest_stage.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _publish_bundle(
            staging_dir=staging_dir,
            output_dir=output_dir,
            manifest_stage=manifest_stage,
            output_manifest=output_manifest,
        )
        if derived_union_output_manifest is not None and derived_union_output_dir is not None:
            _derive_union_bundle(
                native_manifest=manifest,
                native_manifest_path=output_manifest,
                input_records=records,
                manifest_dir=input_manifest.parent,
                output_manifest=derived_union_output_manifest.resolve(),
                output_dir=derived_union_output_dir.resolve(),
            )
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if manifest_stage.exists():
            manifest_stage.unlink()
        raise
    print(json.dumps(manifest, ensure_ascii=False))
    return 0 if failed == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project
        / "models/ScribblePrompt/ScribblePrompt_unet_v1_nf192_res128.pt",
    )
    parser.add_argument(
        "--source-root", type=Path, default=project / "upstream/ScribblePrompt"
    )
    parser.add_argument("--output-policy", choices=OUTPUT_POLICIES, required=True)
    parser.add_argument("--derived-union-output-manifest", type=Path)
    parser.add_argument("--derived-union-output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--partition", choices=("val", "test"), required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=project / "configs/petct_route_a_experiment.json",
    )
    add_leaf_test_access_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(
            input_manifest=args.input_manifest,
            output_manifest=args.output_manifest,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            source_root=args.source_root,
            output_policy=args.output_policy,
            derived_union_output_manifest=args.derived_union_output_manifest,
            derived_union_output_dir=args.derived_union_output_dir,
            device_request=args.device,
            partition=args.partition,
            learning_split=args.learning_split,
            experiment_config=args.experiment_config,
            test_access_receipt=args.test_access_receipt,
            run_root=args.run_root,
        )
    except AdapterError as error:
        print("ERROR: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
