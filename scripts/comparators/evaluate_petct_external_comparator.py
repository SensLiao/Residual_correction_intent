#!/usr/bin/env python3
"""Evaluate one external comparator/policy on the frozen original-grid episodes.

Each invocation produces exactly one method-dimensionality-policy table.  The
2D ScribblePrompt and 3D nnInteractive results are intentionally never pooled.
The legacy ``union_with_m0`` projection is a positive-only diagnostic, never a
current v2 primary or bidirectional baseline. Each native output is reported in
its own explicitly labelled diagnostic table.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1]
for directory in (SCRIPTS, SCRIPTS / "common", SCRIPTS / "comparators"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common.petct_learning import (  # noqa: E402
    LearningContractError,
    encode_json,
    encode_jsonl,
    load_jsonl,
    sha256_file,
    validate_manifest_rows_against_frozen_learning_split,
    write_bytes_bundle_exclusive,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)
from common.petct_route_a_core import (  # noqa: E402
    correction_metrics,
    dice,
    patient_cluster_summary,
    prompt_distal_mask,
    safe_recall,
)
from comparators.run_petct_external_comparator import (  # noqa: E402
    load_and_validate_contract,
    validate_manifest,
)


SCHEMA_VERSION = "PETCT-EXTERNAL-COMPARATOR-METRICS-v1.0"
PUBLIC_TO_INTERNAL_PARTITION = {"train": "train", "validation": "val", "test": "test"}
METHODS = {
    "scribbleprompt": {
        "spatial_dimensionality": "2D",
        "fairness_table_id": "EXTERNAL-SPATIAL-2D-SCRIBBLEPROMPT",
        "native_policy": "native_slice_replace",
    },
    "nninteractive": {
        "spatial_dimensionality": "3D",
        "fairness_table_id": "EXTERNAL-SPATIAL-3D-NNINTERACTIVE-EXPOSED",
        "native_policy": "native_full_mask",
    },
}
REQUIRED_METRICS = (
    "dice",
    "dice_delta_vs_m0",
    "dmm",
    "false_positive_volume_ml",
    "false_negative_volume_ml",
    "authorized_residual_recall",
    "prompt_distal_recall",
    "unauthorized_addition_volume_ml",
    "m0_preservation_rate",
    "other_lesion_harm",
    "unintended_bridge_or_merge_rate",
    "runtime_seconds",
    "peak_gpu_memory_mib",
)


class EvaluationError(RuntimeError):
    """Raised when an external result cannot support a valid comparison row."""


def _validate_external_records_against_frozen_split(
    records: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    learning_split: Path,
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        internal = PUBLIC_TO_INTERNAL_PARTITION.get(str(record.get("split") or ""))
        if internal != partition:
            raise EvaluationError(
                f"input record {index}.split does not match --partition {partition}"
            )
        split_receipt = record.get("patient_split_receipt")
        if not isinstance(split_receipt, Mapping):
            raise EvaluationError(
                f"input record {index} omits patient_split_receipt"
            )
        normalized.append(
            {
                "case_id": record.get("case_id"),
                "patient_id": record.get("patient_id"),
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
        raise EvaluationError(
            f"frozen learning-split validation failed: {exc}"
        ) from exc


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot load {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"{label} must be a JSON object")
    return payload


def _regular(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise EvaluationError(f"missing regular {label}: {resolved}")
    return resolved


def _resolve(raw: Any, base: Path, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise EvaluationError(f"{label} path is empty")
    candidate = Path(raw)
    path = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    return _regular(path, label=label)


def _load_mask(
    path: Path, *, label: str
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    try:
        image = nib.load(str(path))
        values = np.asanyarray(image.dataobj)
    except Exception as exc:
        raise EvaluationError(f"cannot load {label} {path}: {exc}") from exc
    if len(image.shape) != 3:
        raise EvaluationError(f"{label} must be a 3D NIfTI")
    if not np.all(np.isfinite(values)) or not np.all(np.isin(values, [0, 1])):
        raise EvaluationError(f"{label} must be finite binary 0/1")
    return image, np.asarray(values, dtype=np.uint8) > 0


def _load_pet(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    try:
        image = nib.load(str(path))
        values = np.asarray(image.dataobj, dtype=np.float32)
    except Exception as exc:
        raise EvaluationError(f"cannot load PET {path}: {exc}") from exc
    if len(image.shape) != 3 or not np.all(np.isfinite(values)):
        raise EvaluationError("PET must be a finite 3D NIfTI")
    return image, values


def _same_grid(
    reference: nib.spatialimages.SpatialImage,
    candidate: nib.spatialimages.SpatialImage,
    *,
    label: str,
) -> None:
    if reference.shape != candidate.shape or not np.allclose(
        reference.affine, candidate.affine, rtol=0.0, atol=1e-4
    ):
        raise EvaluationError(f"{label} is not on the frozen original grid")


def _finite_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_official_metric_evaluator(metrics_file: Path):
    spec = importlib.util.spec_from_file_location(
        "pinned_autopetv_external_metrics", str(metrics_file.resolve())
    )
    if spec is None or spec.loader is None:
        raise EvaluationError("cannot load pinned AutoPET-V metric module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator = getattr(module, "MetricEvaluator", None)
    if evaluator is None:
        raise EvaluationError("pinned AutoPET-V metric module has no MetricEvaluator")
    return evaluator


def _verified_episode_path(
    row: Mapping[str, Any], path_key: str, hash_key: str
) -> Path:
    path = _regular(Path(str(row.get(path_key) or "")), label=path_key)
    if sha256_file(path) != row.get(hash_key):
        raise EvaluationError(f"natural episode artifact hash changed: {path_key}")
    return path


def _failure_metrics() -> dict[str, None]:
    return {metric: None for metric in REQUIRED_METRICS}


def _evaluate_complete_record(
    *,
    input_record: Mapping[str, Any],
    output_record: Mapping[str, Any],
    episode: Mapping[str, Any],
    input_parent: Path,
    output_parent: Path,
    metric_evaluator_class: Any,
    distal_radius_mm: float,
    output_policy: str,
) -> dict[str, Any]:
    gt_path = _verified_episode_path(episode, "gt_path", "gt_sha256")
    m0_path = _verified_episode_path(episode, "m0_path", "m0_sha256")
    authorized_path = _verified_episode_path(
        episode, "authorized_path", "authorized_sha256"
    )
    pet_path = _resolve(input_record["pet_path"], input_parent, label="PET")
    input_m0_path = _resolve(input_record["m0_path"], input_parent, label="input M0")
    scribble_path = _resolve(
        input_record["fg_scribble_path"], input_parent, label="foreground scribble"
    )
    prediction_path = _resolve(
        output_record["prediction_path"], output_parent, label="prediction"
    )
    reference_path = _resolve(
        input_record["original_grid_reference"], input_parent, label="original grid reference"
    )
    if input_m0_path != m0_path:
        raise EvaluationError("comparator input M0 differs from the frozen natural episode")
    input_hashes = input_record.get("input_sha256")
    if not isinstance(input_hashes, Mapping):
        raise EvaluationError("comparator input lacks immutable input hashes")
    for path, key in (
        (pet_path, "pet"),
        (reference_path, "ct"),
        (m0_path, "m0"),
        (scribble_path, "fg_scribble"),
    ):
        if sha256_file(path) != input_hashes.get(key):
            raise EvaluationError(f"comparator input hash changed: {key}")
    prediction_sha = sha256_file(prediction_path)
    declared_prediction_sha = output_record.get("prediction_sha256")
    if declared_prediction_sha is not None and declared_prediction_sha != prediction_sha:
        raise EvaluationError("adapter prediction hash changed")

    reference = nib.load(str(reference_path))
    gt_image, gt = _load_mask(gt_path, label="GT")
    m0_image, m0 = _load_mask(m0_path, label="M0")
    authorized_image, authorized = _load_mask(authorized_path, label="authorized residual")
    scribble_image, scribble = _load_mask(scribble_path, label="foreground scribble")
    prediction_image, prediction = _load_mask(prediction_path, label="prediction")
    pet_image, pet = _load_pet(pet_path)
    for label, image in (
        ("GT", gt_image),
        ("M0", m0_image),
        ("authorized residual", authorized_image),
        ("foreground scribble", scribble_image),
        ("prediction", prediction_image),
        ("PET", pet_image),
    ):
        _same_grid(reference, image, label=label)
    if not authorized.any() or np.any(authorized & (~gt | m0)):
        raise EvaluationError("authorized target is not a non-empty subset of GT\\M0")
    if not scribble.any() or np.any(scribble & ~authorized):
        raise EvaluationError("frozen foreground scribble is not inside its authorized target")

    added = prediction & ~m0
    projected_add_only = m0 | added
    is_positive_only_diagnostic = output_policy == "union_with_m0"
    if is_positive_only_diagnostic and not np.array_equal(
        prediction, projected_add_only
    ):
        raise EvaluationError("union_with_m0 output removed M0 voxels")
    spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
    if len(spacing) != 3 or any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise EvaluationError("original-grid spacing is invalid")
    voxel_ml = float(np.prod(np.asarray(spacing, dtype=float)) / 1000.0)

    # Addition safety is evaluated against an explicit M0-union projection for
    # both policies. Native-mask removal harm is recorded separately below.
    addition_metrics = correction_metrics(
        operation="ADD",
        gt=gt,
        m0=m0,
        m1=projected_add_only,
        authorized_target=authorized,
        scribble=scribble,
        spacing_xyz=spacing,
        distal_radius_mm=distal_radius_mm,
    )
    distal = prompt_distal_mask(authorized, scribble, spacing, distal_radius_mm)
    false_positive = prediction & ~gt
    false_negative = gt & ~prediction
    m0_count = int(m0.sum())
    m0_preserved = int((prediction & m0).sum())
    protected_existing_truth = m0 & gt
    removed_existing_truth = protected_existing_truth & ~prediction
    protected_count = int(protected_existing_truth.sum())
    other_lesion_harm = (
        None if protected_count == 0 else float(removed_existing_truth.sum() / protected_count)
    )
    metric_evaluator = metric_evaluator_class(overlap_threshold=0.1, connectivity=18)
    official = metric_evaluator(
        prediction.astype(np.uint8),
        gt.astype(np.uint8),
        str(input_record["case_id"]),
        spacing=spacing,
        suv=pet,
    )
    if not isinstance(official, Mapping):
        raise EvaluationError("official AutoPET-V metric evaluator returned a non-object")
    return {
        "dice": dice(prediction, gt),
        "dice_m0": dice(m0, gt),
        "dice_delta_vs_m0": dice(prediction, gt) - dice(m0, gt),
        "dmm": _finite_or_none(official.get("f1")),
        "dmm_tp": _finite_or_none(official.get("tp")),
        "dmm_fp": _finite_or_none(official.get("fp")),
        "dmm_fn": _finite_or_none(official.get("fn")),
        "false_positive_volume_ml": float(false_positive.sum() * voxel_ml),
        "false_negative_volume_ml": float(false_negative.sum() * voxel_ml),
        "authorized_residual_recall": safe_recall(prediction, authorized),
        "prompt_distal_recall": safe_recall(prediction, distal),
        "unauthorized_addition_volume_ml": float(
            (added & ~authorized).sum() * voxel_ml
        ),
        "m0_preservation_rate": (
            None if m0_count == 0 else float(m0_preserved / m0_count)
        ),
        "m0_preservation_rate_defined": m0_count > 0,
        "m0_preservation_rate_denominator_voxels": float(m0_count),
        "other_lesion_harm": other_lesion_harm,
        "other_lesion_harm_defined": protected_count > 0,
        "other_lesion_harm_denominator_voxels": float(protected_count),
        "unintended_bridge_or_merge_rate": float(
            addition_metrics["unintended_bridge_or_merge"]
        ),
        "runtime_seconds": _finite_or_none(output_record.get("runtime_seconds")),
        "peak_gpu_memory_mib": _finite_or_none(
            output_record.get("peak_gpu_memory_mib")
        ),
        "prediction_sha256": prediction_sha,
        "authorized_addition_voxels": addition_metrics["authorized_addition_voxels"],
        "unauthorized_addition_voxels": addition_metrics[
            "unauthorized_addition_voxels"
        ],
        "m0_removed_voxels": float((m0 & ~prediction).sum()),
        "positive_only_diagnostic_contract_pass": bool(
            not is_positive_only_diagnostic
            or np.array_equal(prediction, projected_add_only)
        ),
        "metric_grid": "full_original_3d_grid",
        "voxel_volume_ml": voxel_ml,
    }


def evaluate_external_output(
    *,
    input_manifest: Path,
    output_manifest: Path,
    natural_episode_manifest: Path,
    comparator_config: Path,
    experiment_config: Path,
    learning_split: Path,
    official_metrics: Path,
    partition: str,
    test_access_receipt: Path | None,
    run_root: Path | None,
    method_id: str,
    output_policy: str,
    rows_path: Path,
    summary_path: Path,
    metric_evaluator_class: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        access_receipt = enforce_partition_access(
            partition,
            receipt_path=test_access_receipt,
            experiment_config=experiment_config,
            learning_split=learning_split,
            run_root=run_root,
            output_paths=(rows_path, summary_path),
        )
    except TestAccessError as exc:
        raise EvaluationError(str(exc)) from exc
    if method_id not in METHODS:
        raise EvaluationError(f"unsupported external method: {method_id}")
    method_contract = METHODS[method_id]
    allowed_policies = {"union_with_m0", method_contract["native_policy"]}
    if output_policy not in allowed_policies:
        raise EvaluationError(
            f"{method_id} output policy must be one of {sorted(allowed_policies)}"
        )
    experiment_document = _load_json(
        experiment_config, label="experiment config"
    )
    if experiment_document.get("schema_version") == "PETCT-ROUTE-A-EXPERIMENT-v2.0":
        raise EvaluationError(
            "REMOVE_UNSUPPORTED: legacy positive-only external adapters and "
            "union_with_m0 outputs are not current v2 bidirectional results"
        )
    contract = load_and_validate_contract(comparator_config)
    declared_method = next(
        (method for method in contract["methods"] if method["id"] == method_id), None
    )
    if declared_method is None:
        raise EvaluationError("comparator config does not declare the selected method")
    if declared_method["spatial_dimensionality"] != method_contract["spatial_dimensionality"]:
        raise EvaluationError("method dimensionality differs from the frozen fairness family")
    if tuple(contract["metrics_contract"]["required_metrics"]) != REQUIRED_METRICS:
        raise EvaluationError("external metrics contract changed")

    input_payload = _load_json(input_manifest, label="comparator input manifest")
    output_payload = _load_json(output_manifest, label="comparator output manifest")
    validate_manifest(input_payload, contract["input_manifest_contract"], "input")
    validate_manifest(output_payload, contract["output_manifest_contract"], "output")
    if output_payload.get("method_id", method_id) != method_id:
        raise EvaluationError("output manifest method_id differs from requested method")
    if output_payload.get("output_policy", output_policy) != output_policy:
        raise EvaluationError("output manifest policy differs from requested policy")
    input_records = input_payload["records"]
    output_records = output_payload["records"]
    split_validation = _validate_external_records_against_frozen_split(
        input_records,
        partition=partition,
        learning_split=learning_split,
    )
    inputs_by_case = {str(record["case_id"]): record for record in input_records}
    outputs_by_case = {str(record["case_id"]): record for record in output_records}
    if len(inputs_by_case) != len(input_records) or len(outputs_by_case) != len(output_records):
        raise EvaluationError("input/output manifest has duplicate case_id")
    if set(inputs_by_case) != set(outputs_by_case):
        raise EvaluationError("output manifest does not exactly cover the frozen input cases")

    natural_path = _regular(natural_episode_manifest, label="natural episode manifest")
    provenance = input_payload.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get(
        "natural_episode_manifest_sha256"
    ) != sha256_file(natural_path):
        raise EvaluationError("input manifest is not bound to this natural episode manifest")
    natural_rows = load_jsonl(natural_path)
    selected_natural_rows = [
        row for row in natural_rows if row.get("partition") == partition
    ]
    try:
        validate_manifest_rows_against_frozen_learning_split(
            selected_natural_rows,
            learning_split,
            require_episode_id=True,
            allowed_partitions={partition},
        )
    except LearningContractError as exc:
        raise EvaluationError(
            f"natural episode frozen learning-split validation failed: {exc}"
        ) from exc
    episodes = {
        str(row.get("episode_id") or ""): row for row in selected_natural_rows
    }
    if "" in episodes:
        raise EvaluationError("natural episode manifest has an empty episode_id")
    with _regular(experiment_config, label="experiment config").open(
        "r", encoding="utf-8"
    ) as stream:
        experiment = json.load(stream)
    distal_radius_mm = float(experiment["editor"]["local_radius_mm"])
    if not math.isfinite(distal_radius_mm) or distal_radius_mm <= 0:
        raise EvaluationError("frozen distal radius is invalid")
    metric_class = metric_evaluator_class or load_official_metric_evaluator(
        _regular(official_metrics, label="official AutoPET-V metrics")
    )

    rows: list[dict[str, Any]] = []
    for case_id in sorted(inputs_by_case):
        input_record = inputs_by_case[case_id]
        output_record = outputs_by_case[case_id]
        episode_id = str(input_record.get("episode_id") or "")
        episode = episodes.get(episode_id)
        if (
            episode is None
            or episode.get("case_id") != case_id
            or str(episode.get("patient_id") or "").casefold()
            != str(input_record["patient_id"]).casefold()
            or episode.get("partition") != partition
        ):
            raise EvaluationError("comparator input does not resolve to its frozen episode")
        if str(output_record.get("patient_id") or "").casefold() != str(
            input_record["patient_id"]
        ).casefold():
            raise EvaluationError("output case-to-patient mapping changed")
        if output_record.get("method_id") != method_id:
            raise EvaluationError("output contains a different method_id")
        output_grid = Path(str(output_record["original_grid_reference"]))
        input_grid = Path(str(input_record["original_grid_reference"]))
        if output_grid.resolve() != input_grid.resolve():
            raise EvaluationError("output original-grid reference changed")
        status = str(output_record.get("status") or "")
        row: dict[str, Any] = {
            "case_id": case_id,
            "episode_id": episode_id,
            "patient_id": str(input_record["patient_id"]),
            "split": input_record["split"],
            "fold": int(input_record["fold"]),
            "step": int(input_record["step"]),
            "method_id": method_id,
            "spatial_dimensionality": method_contract["spatial_dimensionality"],
            "fairness_table_id": method_contract["fairness_table_id"],
            "output_policy": output_policy,
            "comparison_role": (
                "POSITIVE_ONLY_DIAGNOSTIC"
                if output_policy == "union_with_m0"
                else "NATIVE_DIAGNOSTIC"
            ),
            "status": status,
            "source_checkpoint_id": output_record["source_checkpoint_id"],
            "pretraining_exposure": declared_method["pretraining"][
                "current_psma_v3_exposure"
            ],
            "headline_eligible": bool(declared_method["headline"]["eligible"]),
        }
        if status == "complete":
            row.update(
                _evaluate_complete_record(
                    input_record=input_record,
                    output_record=output_record,
                    episode=episode,
                    input_parent=input_manifest.resolve().parent,
                    output_parent=output_manifest.resolve().parent,
                    metric_evaluator_class=metric_class,
                    distal_radius_mm=distal_radius_mm,
                    output_policy=output_policy,
                )
            )
            row["failure_reason"] = None
        elif status == "failed":
            row.update(_failure_metrics())
            row["failure_reason"] = output_record.get(
                "failure_reason", output_record.get("error")
            )
        else:
            raise EvaluationError(f"unsupported output status: {status!r}")
        rows.append(row)

    failed = sum(row["status"] == "failed" for row in rows)
    rows_bytes = encode_jsonl(rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE" if failed == 0 else "INCOMPLETE_WITH_EXPLICIT_FAILURES",
        "method_id": method_id,
        "spatial_dimensionality": method_contract["spatial_dimensionality"],
        "fairness_table_id": method_contract["fairness_table_id"],
        "cross_dimensional_pooling": "FORBIDDEN",
        "separate_fairness_table": True,
        "output_policy": output_policy,
        "comparison_role": (
            "POSITIVE_ONLY_DIAGNOSTIC"
            if output_policy == "union_with_m0"
            else "NATIVE_DIAGNOSTIC"
        ),
        "positive_only_diagnostic": output_policy == "union_with_m0",
        "record_count": len(rows),
        "patient_count": len({row["patient_id"] for row in rows}),
        "complete_count": len(rows) - failed,
        "failed_count": failed,
        "headline_eligible": bool(declared_method["headline"]["eligible"]),
        "pretraining_exposure": declared_method["pretraining"][
            "current_psma_v3_exposure"
        ],
        "input_manifest_sha256": sha256_file(input_manifest),
        "output_manifest_sha256": sha256_file(output_manifest),
        "natural_episode_manifest_sha256": sha256_file(natural_path),
        "experiment_config_sha256": sha256_file(experiment_config),
        "learning_split_sha256": split_validation["learning_split_sha256"],
        "test_access_receipt_sha256": (
            sha256_file(test_access_receipt) if access_receipt is not None else None
        ),
        "official_metrics_sha256": sha256_file(official_metrics),
        "metric_rows_sha256": hashlib.sha256(rows_bytes).hexdigest(),
        "distal_radius_mm": distal_radius_mm,
        "official_autoPETV_metric_contract": {
            "overlap_threshold": 0.1,
            "connectivity": 18,
            "metric_grid": "full_original_3d_grid",
            "dmm_field": "MetricEvaluator per-case f1",
        },
        "metric_definitions": {
            "authorized_residual_recall": "prediction overlap with the frozen episode-authorized GT\\M0 residual divided by that residual",
            "unauthorized_addition_volume_ml": "new prediction voxels outside the frozen authorized residual, measured against M0",
            "m0_preservation_rate": "M0 voxels retained by the native/final prediction divided by M0 voxels",
            "other_lesion_harm": "correctly segmented pre-existing GT-and-M0 voxels removed by the native/final prediction divided by those protected voxels",
            "unintended_bridge_or_merge_rate": "per-case 0/1 addition-induced harmful bridge indicator from the shared correction evaluator",
        },
        "patient_clustered": {
            metric: patient_cluster_summary(rows, metric) for metric in REQUIRED_METRICS
        },
        "failure_denominator_policy": (
            "failed records remain in record_count and patient/case inventory; metric values are null"
        ),
        "claim_boundary": (
            "descriptive external spatial comparator only; not a same-weight intent causal arm"
        ),
    }
    write_bytes_bundle_exclusive(
        {rows_path.resolve(): rows_bytes, summary_path.resolve(): encode_json(summary)}
    )
    return rows, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--natural-episode-manifest", type=Path, required=True)
    parser.add_argument("--comparator-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--official-metrics", type=Path, required=True)
    parser.add_argument("--partition", choices=("val", "test"), required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--output-policy", required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    add_leaf_test_access_arguments(parser)
    args = parser.parse_args(argv)
    _, summary = evaluate_external_output(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        natural_episode_manifest=args.natural_episode_manifest,
        comparator_config=args.comparator_config,
        experiment_config=args.experiment_config,
        learning_split=args.learning_split,
        official_metrics=args.official_metrics,
        partition=args.partition,
        test_access_receipt=args.test_access_receipt,
        run_root=args.run_root,
        method_id=args.method,
        output_policy=args.output_policy,
        rows_path=args.rows,
        summary_path=args.summary,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["failed_count"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
