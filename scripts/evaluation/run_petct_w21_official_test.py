#!/usr/bin/env python3
"""Run W2.1 Binary/EDT with five accumulated official correction rounds.

Formal test execution is possible only through a consumed W2.1 receipt.  The
runner produces six original-grid states per case (initial plus five
corrections), computes per-case AUC-Dice/AUC-DMM, and can resume only from
sealed, hash-validated case receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_w21_test_access import (  # noqa: E402
    EXPECTED_CHECKPOINT_NAME,
    EXPECTED_MODEL_FOLD,
    OFFICIAL_METRICS_SHA256,
    OFFICIAL_SIMULATOR_SHA256,
    PROTOCOL,
    W21AccessError,
    sha256_file,
    validate_receipt,
)


CASE_SCHEMA = "PETCT-W21-OFFICIAL-TEST-CASE-v2.0"
ARM_SCHEMA = "PETCT-W21-OFFICIAL-TEST-ARM-v2.0"
SUMMARY_SCHEMA = "PETCT-W21-OFFICIAL-TEST-SUMMARY-v2.0"
SMOKE_CONFIRMATION = "I_CONFIRM_THIS_IS_NOT_LOCKED_TEST"
EDT_RADIUS = 2
SYNTHETIC_SMOKE_SHAPE = (32, 32, 32)
SYNTHETIC_SMOKE_CASE_ID = "synthetic-no-subject-smoke"


class W21RunError(RuntimeError):
    """Raised when inference or evaluation violates the frozen protocol."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _seal(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(payload)
    value[field] = _canonical_sha256(value)
    return value


def _verify_seal(value: Mapping[str, Any], field: str, *, label: str) -> None:
    observed = value.get(field)
    core = {key: item for key, item in value.items() if key != field}
    if not isinstance(observed, str) or observed != _canonical_sha256(core):
        raise W21RunError(f"{label} seal is invalid")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ) + "\n"
    descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise


def _regular(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise W21RunError(f"{label} must be a non-symlink regular file: {raw}")
    return raw.resolve()


def _file_record(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _regular(path, label=label)
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _load_module(path: Path, name: str):
    source = _regular(path, label=name)
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise W21RunError(f"cannot load {name}: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_pinned_official_smoke_code(
    simulator_path: Path, metrics_path: Path
) -> tuple[Path, Path]:
    simulator = _regular(simulator_path, label="smoke official simulator")
    metrics = _regular(metrics_path, label="smoke official metrics")
    if sha256_file(simulator) != OFFICIAL_SIMULATOR_SHA256:
        raise W21RunError("smoke simulator is not the pinned official file")
    if sha256_file(metrics) != OFFICIAL_METRICS_SHA256:
        raise W21RunError("smoke metrics is not the pinned official file")
    return simulator, metrics


def encode_edt(scribble: np.ndarray, radius: int = EDT_RADIUS) -> np.ndarray:
    """Apply the W2.1 curve-generalized nnInteractive EDT encoding."""
    source = np.asarray(scribble) > 0
    output = np.zeros(source.shape, dtype=np.float32)
    indices = np.argwhere(source)
    if not len(indices):
        return output
    try:
        from skimage.morphology import ball
    except ImportError as exc:
        raise W21RunError("scikit-image is required for the frozen EDT encoding") from exc
    structure = ball(radius)
    peak = float(distance_transform_edt(structure).max())
    padding = radius + 1
    low = np.maximum(indices.min(axis=0) - padding, 0)
    high = np.minimum(indices.max(axis=0) + padding + 1, np.array(source.shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    region = binary_dilation(source[slices], structure=structure)
    output[slices] = np.clip(
        distance_transform_edt(region) / peak, 0.0, 1.0
    ).astype(np.float32)
    return output


def _candidate(simulator: Any, residual: np.ndarray, strategy: str) -> tuple[list[list[int]], int]:
    """Normalize the official empty-residual two-value return to three values."""
    if not np.any(residual):
        return [], 0
    result = simulator.simulate_scribble_from_label(
        residual.astype(np.uint8), strategy, int(PROTOCOL["simulator_seed"])
    )
    if not isinstance(result, tuple) or len(result) not in {2, 3}:
        raise W21RunError("official simulator returned an unknown contract")
    coordinates = result[0]
    size = 0 if len(result) == 2 else int(result[2])
    normalized = [[int(value) for value in coordinate] for coordinate in coordinates]
    if any(len(coordinate) != 3 for coordinate in normalized):
        raise W21RunError("official simulator returned a non-3D coordinate")
    if size != len(normalized):
        raise W21RunError("official simulator coordinate count differs from its size")
    return normalized, size


def choose_correction(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    strategy: str,
    simulator: Any,
) -> dict[str, Any]:
    """Reproduce the official candidate-size polarity decision."""
    pred = np.asarray(prediction) > 0
    gt = np.asarray(ground_truth) > 0
    if pred.shape != gt.shape:
        raise W21RunError("prediction and ground truth shapes differ")
    background, fp_size = _candidate(simulator, pred & ~gt, strategy)
    foreground, fn_size = _candidate(simulator, ~pred & gt, strategy)
    if fp_size <= fn_size:
        polarity, selected = "foreground", foreground
    else:
        polarity, selected = "background", background
    return {
        "polarity": polarity,
        "coordinates": selected,
        "selected_size": len(selected),
        "background_candidate_size": fp_size,
        "foreground_candidate_size": fn_size,
    }


def _volume_from_coordinates(
    coordinates: set[tuple[int, int, int]], shape: Sequence[int]
) -> np.ndarray:
    volume = np.zeros(tuple(int(value) for value in shape), dtype=np.uint8)
    for coordinate in coordinates:
        if len(coordinate) != 3 or any(
            value < 0 or value >= volume.shape[axis]
            for axis, value in enumerate(coordinate)
        ):
            raise W21RunError("accumulated scribble coordinate is out of bounds")
        volume[coordinate] = 1
    return volume


def _dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float | None:
    pred = np.asarray(prediction) > 0
    gt = np.asarray(ground_truth) > 0
    if not np.any(gt):
        return None
    denominator = int(pred.sum()) + int(gt.sum())
    return 0.0 if denominator == 0 else float(2 * np.logical_and(pred, gt).sum() / denominator)


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def compute_state_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    *,
    spacing: Sequence[float],
    metric_evaluator_class: Any,
    case_name: str,
) -> dict[str, Any]:
    pred = (np.asarray(prediction) > 0).astype(np.uint8)
    gt = (np.asarray(ground_truth) > 0).astype(np.uint8)
    voxel_ml = float(np.prod(np.asarray(spacing, dtype=float)) / 1000.0)
    raw_voxel_fp = float(np.logical_and(pred > 0, gt == 0).sum() * voxel_ml)
    raw_voxel_fn = float(np.logical_and(pred == 0, gt > 0).sum() * voxel_ml)
    evaluator = metric_evaluator_class(
        overlap_threshold=float(PROTOCOL["dmm_iou_threshold"]),
        connectivity=int(PROTOCOL["dmm_connectivity"]),
    )
    official = evaluator(
        prediction=pred,
        ground_truth=gt,
        case_name=case_name,
        spacing=spacing,
    )
    if not isinstance(official, Mapping) or not {"f1", "fpv", "fnv"} <= set(official):
        raise W21RunError("official component metric omitted f1/fpv/fnv")
    positive_gt = bool(np.any(gt))
    return {
        "dice": _dice(pred, gt) if positive_gt else None,
        "dmm_f1": _finite_or_none(official["f1"]) if positive_gt else None,
        "fpv_ml": _finite_or_none(official["fpv"]),
        "fnv_ml": _finite_or_none(official["fnv"]),
        "raw_voxel_fp_volume_ml": raw_voxel_fp,
        "raw_voxel_fn_volume_ml": raw_voxel_fn,
        "metric_eligibility": (
            "ELIGIBLE_POSITIVE_GT"
            if positive_gt
            else "EMPTY_GT_EXCLUDED_FROM_DICE_DMM_AUC"
        ),
    }


def per_case_auc(states: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    if len(states) != int(PROTOCOL["evaluation_states"]):
        raise W21RunError("formal per-case AUC requires exactly six states")
    x = np.arange(len(states), dtype=float)
    dice = [state.get("dice") for state in states]
    dmm = [state.get("dmm_f1") for state in states]
    if any(value is None for value in dice + dmm):
        return {"auc_dice": None, "auc_dmm": None, "combined_score": None}
    auc_dice = float(np.trapz(np.asarray(dice, dtype=float), x))
    auc_dmm = float(np.trapz(np.asarray(dmm, dtype=float), x))
    return {
        "auc_dice": auc_dice,
        "auc_dmm": auc_dmm,
        "combined_score": 0.5 * auc_dice + 0.5 * auc_dmm,
    }


def _load_source_rows(identity_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in _regular(identity_path, label="source identity").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["case_id"])] = row
    return rows


def _load_predictor(
    model_dir: Path,
    device: str,
    *,
    fold: int = EXPECTED_MODEL_FOLD,
    checkpoint_name: str = EXPECTED_CHECKPOINT_NAME,
):
    try:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    except ImportError as exc:
        raise W21RunError("the pinned nnU-Net inference environment is unavailable") from exc
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=(int(fold),), checkpoint_name=checkpoint_name
    )
    return predictor


def _predict_state(
    predictor: Any,
    *,
    ct_path: Path,
    pet_path: Path,
    reference: nib.Nifti1Image,
    foreground: np.ndarray,
    background: np.ndarray,
    arm: str,
    state: int,
    case_dir: Path,
) -> Path:
    inputs = case_dir / ".inputs"
    inputs.mkdir(exist_ok=True)
    fg_path = inputs / f"state_{state}_0002.nii.gz"
    bg_path = inputs / f"state_{state}_0003.nii.gz"
    if arm == "edt":
        foreground = encode_edt(foreground)
        background = encode_edt(background)
    nib.save(
        nib.Nifti1Image(foreground.astype(np.float32), reference.affine, reference.header),
        fg_path,
    )
    nib.save(
        nib.Nifti1Image(background.astype(np.float32), reference.affine, reference.header),
        bg_path,
    )
    output_truncated = case_dir / f"state_{state}"
    predictor.predict_from_files_sequential(
        [[str(ct_path), str(pet_path), str(fg_path), str(bg_path)]],
        [str(output_truncated)],
        save_probabilities=False,
        overwrite=False,
    )
    output = case_dir / f"state_{state}.nii.gz"
    if not output.is_file() or output.is_symlink():
        raise W21RunError(f"nnU-Net did not publish state {state}")
    fg_path.unlink()
    bg_path.unlink()
    return output


def _validate_case_receipt(path: Path, *, arm: str, case_id: str) -> dict[str, Any]:
    raw = _regular(path, label="case receipt")
    value = json.loads(raw.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise W21RunError("case receipt must be a JSON object")
    _verify_seal(value, "case_sha256", label="case receipt")
    if (
        value.get("schema_version") != CASE_SCHEMA
        or value.get("arm") != arm
        or value.get("case_id") != case_id
        or len(value.get("states", [])) != int(PROTOCOL["evaluation_states"])
    ):
        raise W21RunError("case receipt identity/state contract is invalid")
    for state in value["states"]:
        record = state.get("prediction")
        if not isinstance(record, dict) or _file_record(
            Path(str(record.get("path") or "")), label="case prediction"
        ) != record:
            raise W21RunError("case prediction changed after publication")
    return value


def run_one_case(
    *,
    predictor: Any,
    arm: str,
    source: Mapping[str, Any],
    strategy: str,
    simulator: Any,
    metric_evaluator_class: Any,
    output_parent: Path,
    state_count: int,
) -> dict[str, Any]:
    case_id = str(source["case_id"])
    final_dir = output_parent / "cases" / case_id
    final_receipt = final_dir / "case.json"
    if final_dir.exists() or final_dir.is_symlink():
        return _validate_case_receipt(final_receipt, arm=arm, case_id=case_id)
    work_parent = output_parent / ".work"
    work_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{case_id}.", dir=work_parent))
    try:
        ct_path = _regular(Path(str(source["ct_path"])), label=f"{case_id} CT")
        pet_path = _regular(Path(str(source["pet_path"])), label=f"{case_id} PET")
        gt_path = _regular(Path(str(source["gt_path"])), label=f"{case_id} GT")
        gt_image = nib.load(str(gt_path))
        gt = (np.asanyarray(gt_image.dataobj) > 0).astype(np.uint8)
        for image_path, label in ((ct_path, "CT"), (pet_path, "PET")):
            image = nib.load(str(image_path))
            if image.shape != gt_image.shape or not np.allclose(image.affine, gt_image.affine):
                raise W21RunError(f"{case_id} {label}/GT geometry differs")
        spacing = tuple(float(value) for value in gt_image.header.get_zooms()[:3])
        foreground: set[tuple[int, int, int]] = set()
        background: set[tuple[int, int, int]] = set()
        previous: np.ndarray | None = None
        states: list[dict[str, Any]] = []
        for state_index in range(state_count):
            correction = None
            if state_index > 0:
                if previous is None:
                    raise W21RunError("previous prediction is missing")
                if np.any(gt):
                    correction = choose_correction(
                        previous, gt, strategy=strategy, simulator=simulator
                    )
                    selected = {
                        tuple(int(value) for value in coordinate)
                        for coordinate in correction["coordinates"]
                    }
                    if correction["polarity"] == "foreground":
                        foreground.update(selected)
                    else:
                        background.update(selected)
                else:
                    correction = {
                        "polarity": "none-empty-gt",
                        "coordinates": [],
                        "selected_size": 0,
                        "background_candidate_size": None,
                        "foreground_candidate_size": None,
                    }
            fg_volume = _volume_from_coordinates(foreground, gt.shape)
            bg_volume = _volume_from_coordinates(background, gt.shape)
            prediction_path = _predict_state(
                predictor,
                ct_path=ct_path,
                pet_path=pet_path,
                reference=gt_image,
                foreground=fg_volume,
                background=bg_volume,
                arm=arm,
                state=state_index,
                case_dir=temporary,
            )
            prediction_image = nib.load(str(prediction_path))
            if prediction_image.shape != gt_image.shape or not np.allclose(
                prediction_image.affine, gt_image.affine
            ):
                raise W21RunError("prediction was not restored to the original grid")
            previous = (np.asanyarray(prediction_image.dataobj) > 0).astype(np.uint8)
            metrics = compute_state_metrics(
                previous,
                gt,
                spacing=spacing,
                metric_evaluator_class=metric_evaluator_class,
                case_name=f"{arm}-{case_id}-state-{state_index}",
            )
            prediction_record = _file_record(
                prediction_path, label=f"{case_id} state {state_index}"
            )
            # The whole completed case directory is atomically renamed below.
            # Bind the durable post-rename path, while retaining the hash and
            # size measured from the staged file.
            prediction_record["path"] = str(final_dir / prediction_path.name)
            states.append(
                {
                    "state": state_index,
                    "correction": correction,
                    "foreground_scribble_voxels": len(foreground),
                    "background_scribble_voxels": len(background),
                    "prediction": prediction_record,
                    **metrics,
                }
            )
        inputs = temporary / ".inputs"
        if inputs.exists() and not any(inputs.iterdir()):
            inputs.rmdir()
        payload = _seal(
            {
                "schema_version": CASE_SCHEMA,
                "arm": arm,
                "case_id": case_id,
                "patient_id": str(source["patient_id"]),
                "data_scope": str(source.get("data_scope") or "AUTHORIZED_LOCKED_TEST"),
                "strategy": strategy,
                "protocol": {
                    **PROTOCOL,
                    "evaluation_states": state_count,
                    "correction_rounds": state_count - 1,
                },
                "source": {
                    "ct": _file_record(ct_path, label=f"{case_id} CT"),
                    "pet": _file_record(pet_path, label=f"{case_id} PET"),
                    "gt": _file_record(gt_path, label=f"{case_id} GT"),
                },
                "states": states,
                "auc": per_case_auc(states)
                if state_count == int(PROTOCOL["evaluation_states"])
                else None,
            },
            "case_sha256",
        )
        _write_json_exclusive(temporary / "case.json", payload)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, final_dir)
        return payload
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not finite else float(np.mean(finite))


def _publish_arm_summary(
    *, arm: str, output: Path, cases: Sequence[Mapping[str, Any]], receipt_sha256: str
) -> dict[str, Any]:
    eligible = [case for case in cases if case["auc"]["auc_dice"] is not None]
    empty = [case for case in cases if case["auc"]["auc_dice"] is None]
    payload = _seal(
        {
            "schema_version": ARM_SCHEMA,
            "arm": arm,
            "protocol": PROTOCOL,
            "test_access_receipt_sha256": receipt_sha256,
            "case_count": len(cases),
            "eligible_positive_gt_case_count": len(eligible),
            "empty_gt_case_count": len(empty),
            "mean_auc_dice": _mean([case["auc"]["auc_dice"] for case in eligible]),
            "mean_auc_dmm": _mean([case["auc"]["auc_dmm"] for case in eligible]),
            "mean_combined_score": _mean(
                [case["auc"]["combined_score"] for case in eligible]
            ),
            "mean_final_dice": _mean(
                [case["states"][-1]["dice"] for case in eligible]
            ),
            "mean_final_dmm": _mean(
                [case["states"][-1]["dmm_f1"] for case in eligible]
            ),
            "case_receipts": [
                {
                    "case_id": case["case_id"],
                    "case_sha256": case["case_sha256"],
                }
                for case in cases
            ],
        },
        "arm_sha256",
    )
    _write_json_exclusive(output / "ARM_SUMMARY.json", payload)
    return payload


def formal_run(receipt_path: Path, *, arm: str, device: str) -> dict[str, Any]:
    receipt = validate_receipt(receipt_path)
    binding = receipt["consumption"]["binding"]
    if binding["protocol"] != PROTOCOL:
        raise W21RunError("receipt protocol differs from the runner")
    if Path(binding["code"]["runner"]["path"]) != Path(__file__).resolve():
        raise W21RunError("formal runner path differs from the receipt")
    output = Path(binding["outputs"][arm])
    summary_path = output / "ARM_SUMMARY.json"
    if summary_path.exists():
        summary = json.loads(_regular(summary_path, label="arm summary").read_text(encoding="utf-8"))
        _verify_seal(summary, "arm_sha256", label="arm summary")
        return summary
    if output.is_symlink():
        raise W21RunError("arm output must not be a symlink")
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise W21RunError("arm output exists but is not a real directory")
    else:
        output.mkdir(parents=True, exist_ok=False)
    source_rows = _load_source_rows(Path(binding["source_identity"]["path"]))
    inventory = binding["test_inventory"]["cases"]
    official_code = binding["code"]["official_autopetv"]
    simulator = _load_module(
        Path(official_code["simulator"]["path"]), "w21_official_simulator"
    )
    metrics = _load_module(
        Path(official_code["metrics"]["path"]), "w21_official_metrics"
    )
    model_binding = binding["models"][arm]
    predictor = _load_predictor(
        Path(model_binding["model_dir"]),
        device,
        fold=int(model_binding["fold"]),
        checkpoint_name=str(model_binding["checkpoint_name"]),
    )
    cases = []
    for item in inventory:
        case_id = item["case_id"]
        source = dict(source_rows[case_id])
        source["case_id"] = case_id
        cases.append(
            run_one_case(
                predictor=predictor,
                arm=arm,
                source=source,
                strategy=item["strategy"],
                simulator=simulator,
                metric_evaluator_class=metrics.MetricEvaluator,
                output_parent=output,
                state_count=int(PROTOCOL["evaluation_states"]),
            )
        )
    return _publish_arm_summary(
        arm=arm,
        output=output,
        cases=cases,
        receipt_sha256=receipt["receipt_sha256"],
    )


def aggregate(receipt_path: Path) -> dict[str, Any]:
    receipt = validate_receipt(receipt_path)
    binding = receipt["consumption"]["binding"]
    summaries = {}
    for arm in ("binary", "edt"):
        path = Path(binding["outputs"][arm]) / "ARM_SUMMARY.json"
        summary = json.loads(_regular(path, label=f"{arm} summary").read_text(encoding="utf-8"))
        _verify_seal(summary, "arm_sha256", label=f"{arm} summary")
        if summary.get("arm") != arm or summary.get("case_count") != 91:
            raise W21RunError(f"{arm} summary identity/count is invalid")
        summaries[arm] = summary
    summary_path = Path(binding["outputs"]["summary"])
    if summary_path.exists() or summary_path.is_symlink():
        existing = json.loads(
            _regular(summary_path, label="W2.1 combined summary").read_text(
                encoding="utf-8"
            )
        )
        _verify_seal(existing, "summary_sha256", label="W2.1 combined summary")
        return existing
    payload = _seal(
        {
            "schema_version": SUMMARY_SCHEMA,
            "protocol": PROTOCOL,
            "test_access_receipt_sha256": receipt["receipt_sha256"],
            "arms": {
                arm: {
                    key: summaries[arm][key]
                    for key in (
                        "arm_sha256",
                        "case_count",
                        "eligible_positive_gt_case_count",
                        "empty_gt_case_count",
                        "mean_auc_dice",
                        "mean_auc_dmm",
                        "mean_combined_score",
                        "mean_final_dice",
                        "mean_final_dmm",
                    )
                }
                for arm in ("binary", "edt")
            },
        },
        "summary_sha256",
    )
    _write_json_exclusive(summary_path, payload)
    return payload


def _build_synthetic_smoke_source(output: Path) -> dict[str, Any]:
    """Materialize fixed geometry-only smoke inputs with no real subject data."""

    inputs = output / "synthetic_inputs"
    inputs.mkdir(parents=False, exist_ok=False)
    shape = SYNTHETIC_SMOKE_SHAPE
    grid = np.indices(shape, dtype=np.float32)
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0
    squared_distance = sum(
        (grid[axis] - center[axis]) ** 2 for axis in range(len(shape))
    )
    ct = ((grid[0] + grid[1] + grid[2]) / float(sum(shape))).astype(np.float32)
    pet = np.exp(-squared_distance / 64.0).astype(np.float32)
    gt = (squared_distance <= 25.0).astype(np.uint8)
    affine = np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float64)
    paths = {
        "ct_path": inputs / f"{SYNTHETIC_SMOKE_CASE_ID}_0000.nii.gz",
        "pet_path": inputs / f"{SYNTHETIC_SMOKE_CASE_ID}_0001.nii.gz",
        "gt_path": inputs / f"{SYNTHETIC_SMOKE_CASE_ID}.nii.gz",
    }
    nib.save(nib.Nifti1Image(ct, affine), paths["ct_path"])
    nib.save(nib.Nifti1Image(pet, affine), paths["pet_path"])
    nib.save(nib.Nifti1Image(gt, affine), paths["gt_path"])
    return {
        "case_id": SYNTHETIC_SMOKE_CASE_ID,
        "patient_id": "synthetic-no-subject",
        "data_scope": "SYNTHETIC_NO_REAL_SUBJECT",
        **{key: str(path) for key, path in paths.items()},
    }


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    if args.confirm_non_test != SMOKE_CONFIRMATION:
        raise W21RunError("exact non-test smoke confirmation is required")
    simulator_path, metrics_path = _validate_pinned_official_smoke_code(
        args.simulator, args.metrics
    )
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        raise W21RunError("smoke output already exists")
    output.mkdir(parents=True)
    simulator = _load_module(simulator_path, "w21_smoke_simulator")
    metrics = _load_module(metrics_path, "w21_smoke_metrics")
    predictor = _load_predictor(args.model_dir, args.device)
    source = _build_synthetic_smoke_source(output)
    case = run_one_case(
        predictor=predictor,
        arm=args.arm,
        source=source,
        strategy="centerline",
        simulator=simulator,
        metric_evaluator_class=metrics.MetricEvaluator,
        output_parent=output,
        state_count=args.states,
    )
    return {
        "status": "PASS",
        "mode": "NON_TEST_SMOKE",
        "data_scope": "SYNTHETIC_NO_REAL_SUBJECT",
        "arm": args.arm,
        "states": len(case["states"]),
        "case_sha256": case["case_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--receipt", type=Path, required=True)
    run.add_argument("--arm", choices=("binary", "edt"), required=True)
    run.add_argument("--device", default="cuda:0")
    combine = commands.add_parser("aggregate")
    combine.add_argument("--receipt", type=Path, required=True)
    probe = commands.add_parser("smoke")
    probe.add_argument("--arm", choices=("binary", "edt"), required=True)
    probe.add_argument("--model-dir", type=Path, required=True)
    probe.add_argument("--simulator", type=Path, required=True)
    probe.add_argument("--metrics", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--states", type=int, choices=range(1, 7), default=2)
    probe.add_argument("--device", default="cuda:0")
    probe.add_argument("--confirm-non-test", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            payload = formal_run(args.receipt, arm=args.arm, device=args.device)
        elif args.command == "aggregate":
            payload = aggregate(args.receipt)
        else:
            payload = smoke(args)
    except (W21RunError, W21AccessError, FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
