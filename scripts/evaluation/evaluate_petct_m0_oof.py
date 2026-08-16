#!/usr/bin/env python3
"""Evaluate patient-excluded OOF M0 with the pinned autoPET V metric code."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import numbers
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import nibabel as nib
import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
for support_dir in (SCRIPTS_ROOT / "baseline", SCRIPTS_ROOT / "data"):
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))

from baseline.validate_petct_m0_oof import (  # noqa: E402
    validate_oof_case_leaf,
    validate_oof_ready_receipt_only,
)
from common.petct_learning import (  # noqa: E402
    load_jsonl,
    sha256_file,
    write_bytes_bundle_exclusive,
)  # noqa: E402
from common.petct_route_a_core import (  # noqa: E402
    patient_cluster_summary,
    validate_patient_folds,
)
from common.petct_test_access import (  # noqa: E402
    add_leaf_test_access_arguments,
    enforce_partition_access,
)
from data.validate_petct_learning_split import (  # noqa: E402
    load_and_validate_learning_split,
)


DEFAULT_PARTITIONS = ("train", "val")
PARTITIONS = ("train", "val", "test")
SCHEMA_VERSION = "PETCT-M0-OOF-EVALUATION-v1.1"


def load_official_metric_evaluator(metrics_file: Path):
    spec = importlib.util.spec_from_file_location(
        "pinned_autopetv_metrics", str(metrics_file.resolve())
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official autoPET V metrics module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluator = getattr(module, "MetricEvaluator", None)
    if evaluator is None:
        raise RuntimeError("official autoPET V metrics module has no MetricEvaluator")
    return evaluator


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _required_finite_metric(metrics: Mapping[str, Any], key: str) -> float:
    value = _finite_or_none(metrics.get(key))
    if value is None:
        raise RuntimeError(f"official metric {key} is undefined for positive GT")
    return value


def _required_count(metrics: Mapping[str, Any], key: str) -> int:
    value = metrics.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
        or float(value) < 0
        or not float(value).is_integer()
    ):
        raise RuntimeError(f"official metric {key} is not a non-negative count")
    return int(value)


def _load_experiment_config(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("frozen experiment config is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("frozen experiment config is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("frozen experiment config must be an object")
    return payload


def _selected_partitions(
    partitions: Sequence[str],
) -> tuple[str, ...]:
    selected = tuple(str(partition) for partition in partitions)
    if not selected or len(selected) != len(set(selected)):
        raise RuntimeError("evaluation partitions must be unique and non-empty")
    if any(partition not in PARTITIONS for partition in selected):
        raise RuntimeError("evaluation partition must be train, val, or test")
    return selected


def _load_image(path: Path, *, label: str) -> nib.spatialimages.SpatialImage:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or not a regular file")
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} is missing or not a regular file")
    try:
        image = nib.load(str(path))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{label} is not a readable NIfTI") from exc
    if len(image.shape) != 3 or not np.isfinite(image.affine).all():
        raise RuntimeError(f"{label} must have a finite 3D grid")
    return image


def _same_grid(
    reference: nib.spatialimages.SpatialImage,
    candidate: nib.spatialimages.SpatialImage,
    *,
    label: str,
) -> None:
    if candidate.shape != reference.shape or not np.allclose(
        candidate.affine, reference.affine, atol=1e-3, rtol=0.0
    ):
        raise RuntimeError(f"{label} geometry mismatch")


def _resolve_oof_mask(run_dir: Path, record: Mapping[str, Any]) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("OOF mask record path is missing")
    candidate = Path(raw)
    if not candidate.is_absolute() and ".." in candidate.parts:
        raise RuntimeError("OOF mask relative path is unsafe")
    unresolved = candidate if candidate.is_absolute() else run_dir / candidate
    if unresolved.is_symlink():
        raise RuntimeError("OOF mask must not be a symlink")
    resolved = unresolved.resolve()
    if not resolved.is_relative_to(run_dir):
        raise RuntimeError("OOF mask escapes its committed run")
    return resolved


def _json_bytes(payload: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(dict(row)) for row in rows)


def evaluate_m0_oof(
    *,
    oof_ready: Path,
    case_manifest: Path,
    learning_split: Path,
    experiment_config: Path,
    official_metrics: Path,
    rows_path: Path,
    summary_path: Path,
    partitions: Sequence[str] = DEFAULT_PARTITIONS,
    test_access_receipt: Path | None = None,
    run_root: Path | None = None,
    overlap_threshold: float = 0.1,
    connectivity: int = 18,
    ready_validator: Callable[[Path], dict[str, Any]] = validate_oof_ready_receipt_only,
    leaf_validator: Callable[..., dict[str, Any]] = validate_oof_case_leaf,
    split_loader: Callable[
        [Path, Sequence[Mapping[str, Any]], Mapping[str, Any]], tuple[Any, dict[str, Any]]
    ] = load_and_validate_learning_split,
    test_access_validator: Callable[..., dict[str, Any] | None] = (
        enforce_partition_access
    ),
    metric_evaluator_class: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate selected frozen partitions and atomically publish finite JSON."""

    selected_partitions = _selected_partitions(partitions)
    if rows_path.resolve() == summary_path.resolve():
        raise RuntimeError("metric rows and summary outputs must be distinct")
    test_access = test_access_validator(
        selected_partitions,
        receipt_path=test_access_receipt,
        experiment_config=experiment_config,
        learning_split=learning_split,
        run_root=run_root,
        output_paths=(rows_path, summary_path),
    )
    if not math.isfinite(overlap_threshold) or overlap_threshold < 0:
        raise RuntimeError("overlap threshold must be finite and non-negative")
    if connectivity != 18:
        raise RuntimeError("M0 OOF evaluation connectivity must remain 18")

    validated = ready_validator(oof_ready.resolve())
    if validated.get("status") != "PASS" or validated.get("patient_excluded") is not True:
        raise RuntimeError("OOF_READY did not pass patient-exclusion validation")
    source_rows = load_jsonl(case_manifest.resolve())
    validate_patient_folds(source_rows)
    source = {str(row.get("case_id") or ""): row for row in source_rows}
    if "" in source or len(source) != len(source_rows):
        raise RuntimeError("case manifest contains an empty or duplicate case_id")
    oof_cases = validated.get("cases")
    if not isinstance(oof_cases, dict) or set(source) != set(oof_cases):
        raise RuntimeError("case manifest and OOF_READY case sets differ")
    for case_id, row in source.items():
        oof = oof_cases[case_id]
        if (
            str(row.get("patient_id") or "").casefold() != oof.get("patient_id")
            or int(row.get("held_out_fold", -1)) != int(oof.get("held_out_fold", -2))
        ):
            raise RuntimeError("case patient/fold differs from OOF_READY")

    experiment = _load_experiment_config(experiment_config.resolve())
    _, split_receipt = split_loader(
        learning_split.resolve(), source_rows, experiment
    )
    learning_split_sha256 = sha256_file(learning_split.resolve())
    if split_receipt.get("split_sha256") not in (None, learning_split_sha256):
        raise RuntimeError("validated learning split hash differs from selected file")
    case_to_partition = split_receipt.get("case_to_partition")
    if not isinstance(case_to_partition, dict) or set(case_to_partition) != set(source):
        raise RuntimeError("learning split case inventory differs from case manifest")
    if any(partition not in PARTITIONS for partition in case_to_partition.values()):
        raise RuntimeError("learning split contains an unknown partition")

    metric_class = metric_evaluator_class or load_official_metric_evaluator(
        official_metrics.resolve()
    )
    positive_evaluator = metric_class(
        overlap_threshold=overlap_threshold, connectivity=connectivity
    )
    empty_evaluator = metric_class(
        overlap_threshold=overlap_threshold, connectivity=connectivity
    )
    output_rows: list[dict[str, Any]] = []
    for case_id in sorted(source):
        partition = str(case_to_partition[case_id])
        if partition not in selected_partitions:
            continue
        row = source[case_id]
        oof = oof_cases[case_id]
        truth_binding = leaf_validator(
            validated,
            ready_path=oof_ready,
            case_id=case_id,
            source_record=row,
        )
        prediction_image = _load_image(
            Path(truth_binding["m0"]["path"]), label=f"{case_id} prediction"
        )
        gt_image = _load_image(
            Path(truth_binding["inputs"]["gt"]["path"]), label=f"{case_id} GT"
        )
        _same_grid(gt_image, prediction_image, label="prediction/GT")
        prediction = (np.asarray(prediction_image.dataobj) > 0).astype(np.uint8)
        gt = (np.asarray(gt_image.dataobj) > 0).astype(np.uint8)
        if not np.isfinite(prediction).all() or not np.isfinite(gt).all():
            raise RuntimeError("prediction/GT contains non-finite values")
        pet = None
        if truth_binding["inputs"].get("pet"):
            pet_image = _load_image(
                Path(truth_binding["inputs"]["pet"]["path"]),
                label=f"{case_id} PET",
            )
            _same_grid(gt_image, pet_image, label="PET/GT")
            pet = np.asarray(pet_image.dataobj)
            if not np.isfinite(pet).all():
                raise RuntimeError("PET contains non-finite values")

        gt_voxels = int(gt.sum())
        prediction_voxels = int(prediction.sum())
        gt_positive = gt_voxels > 0
        evaluator = positive_evaluator if gt_positive else empty_evaluator
        metrics = evaluator(
            prediction,
            gt,
            case_id,
            spacing=gt_image.header.get_zooms()[:3],
            suv=pet,
        )
        if not isinstance(metrics, Mapping):
            raise RuntimeError("official metric evaluator returned a non-object")
        tp = _required_count(metrics, "tp")
        fp = _required_count(metrics, "fp")
        fn = _required_count(metrics, "fn")
        fpv_ml = _required_finite_metric(metrics, "fpv")
        if gt_positive:
            dice = _required_finite_metric(metrics, "dsc")
            dmm_f1 = _required_finite_metric(metrics, "f1")
            fnv_ml = _required_finite_metric(metrics, "fnv")
            if tp + fn <= 0:
                raise RuntimeError("positive GT has zero official lesion denominator")
        else:
            if tp != 0 or fn != 0:
                raise RuntimeError("empty GT produced non-zero TP/FN counts")
            dice = None
            dmm_f1 = None
            fnv_ml = None
        voxel_volume_ml = float(np.prod(gt_image.header.get_zooms()[:3]) / 1000.0)
        if not math.isfinite(voxel_volume_ml) or voxel_volume_ml <= 0:
            raise RuntimeError("GT voxel volume is invalid")
        output_rows.append(
            {
                "case_id": case_id,
                "patient_id": str(row["patient_id"]).casefold(),
                "held_out_fold": int(row["held_out_fold"]),
                "partition": partition,
                "gt_positive": gt_positive,
                "official_metric_eligible": gt_positive,
                "official_metric_ineligibility_reason": (
                    None if gt_positive else "EMPTY_GT"
                ),
                "official_metric_denominators": {
                    "gt_voxels": gt_voxels,
                    "gt_lesions": tp + fn,
                },
                "gt_voxel_count": gt_voxels,
                "prediction_voxel_count": prediction_voxels,
                "dice": dice,
                "dmm_f1": dmm_f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "fpv_ml": fpv_ml,
                "fnv_ml": fnv_ml,
                "empty_gt_false_positive": (
                    prediction_voxels > 0 if not gt_positive else None
                ),
                "empty_gt_prediction_volume_ml": (
                    prediction_voxels * voxel_volume_ml
                    if not gt_positive
                    else None
                ),
                "truth_binding": truth_binding,
            }
        )
    if not output_rows:
        raise RuntimeError("selected learning partitions contain no M0 cases")

    positive_rows = [row for row in output_rows if row["gt_positive"]]
    empty_rows = [row for row in output_rows if not row["gt_positive"]]
    total_tp = sum(int(row["tp"]) for row in positive_rows)
    total_fp = sum(int(row["fp"]) for row in positive_rows)
    total_fn = sum(int(row["fn"]) for row in positive_rows)
    if positive_rows:
        official = positive_evaluator.aggregate(weighted=False)
        official_dsc = _required_finite_metric(official, "dsc")
        official_f1 = _required_finite_metric(official, "f1_aggregated")
    else:
        official_dsc = None
        official_f1 = None

    partition_counts = Counter(str(row["partition"]) for row in output_rows)
    false_positive_empty_rows = [
        row for row in empty_rows if bool(row["empty_gt_false_positive"])
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE_WITH_EXPLICIT_METRIC_ELIGIBILITY",
        "selected_partitions": list(selected_partitions),
        "test_access": {
            "required": "test" in selected_partitions,
            "consumed_receipt_sha256": (
                str(test_access["receipt_sha256"])
                if test_access is not None and test_access_receipt is not None
                else None
            ),
            "bound_run_root": (
                str(run_root.resolve())
                if test_access is not None and run_root is not None
                else None
            ),
        },
        "source_case_count": len(source_rows),
        "case_count": len(output_rows),
        "patient_count": len({row["patient_id"] for row in output_rows}),
        "partition_case_counts": {
            partition: int(partition_counts[partition])
            for partition in selected_partitions
        },
        "oof_ready_sha256": sha256_file(oof_ready.resolve()),
        "case_manifest_sha256": sha256_file(case_manifest.resolve()),
        "truth_binding_sha256": {
            row["case_id"]: row["truth_binding"]["binding_sha256"]
            for row in output_rows
        },
        "learning_split_sha256": learning_split_sha256,
        "experiment_config_sha256": sha256_file(experiment_config.resolve()),
        "official_metrics_sha256": sha256_file(official_metrics.resolve()),
        "official_autoPETV": {
            "dsc": official_dsc,
            "dmm_f1_aggregated": official_f1,
            "overlap_threshold": overlap_threshold,
            "connectivity": connectivity,
            "aggregation_population": "positive_gt_eligible_cases_only",
            "eligibility_rule": "GT contains at least one positive voxel",
            "eligible_case_count": len(positive_rows),
            "eligible_patient_count": len(
                {row["patient_id"] for row in positive_rows}
            ),
            "ineligible_empty_gt_case_count": len(empty_rows),
            "denominators": {
                "dsc_cases": len(positive_rows),
                "dmm_tp": total_tp,
                "dmm_fp": total_fp,
                "dmm_fn": total_fn,
                "dmm_gt_lesions": total_tp + total_fn,
            },
        },
        "positive_gt_patient_clustered": {
            metric: patient_cluster_summary(positive_rows, metric)
            for metric in ("dice", "dmm_f1", "fpv_ml", "fnv_ml")
        },
        "empty_gt_false_positive_diagnostics": {
            "case_count": len(empty_rows),
            "patient_count": len({row["patient_id"] for row in empty_rows}),
            "false_positive_case_count": len(false_positive_empty_rows),
            "false_positive_patient_count": len(
                {row["patient_id"] for row in false_positive_empty_rows}
            ),
            "false_positive_lesion_count": sum(int(row["fp"]) for row in empty_rows),
            "prediction_volume_ml_total": sum(
                float(row["empty_gt_prediction_volume_ml"]) for row in empty_rows
            ),
            "patient_clustered_fpv_ml": patient_cluster_summary(
                empty_rows, "fpv_ml"
            ),
            "patient_clustered_prediction_volume_ml": patient_cluster_summary(
                empty_rows, "empty_gt_prediction_volume_ml"
            ),
            "official_dice_and_dmm_policy": (
                "undefined for empty GT and serialized as JSON null; false positives "
                "remain explicit diagnostics"
            ),
        },
        "claim_boundary": (
            "OOF M0 quality on explicitly selected frozen learning partitions only; "
            "not evidence that intent or correction works"
        ),
    }
    rows_bytes = _jsonl_bytes(output_rows)
    summary_bytes = _json_bytes(summary, indent=2)
    write_bytes_bundle_exclusive(
        {
            rows_path.resolve(): rows_bytes,
            summary_path.resolve(): summary_bytes,
        }
    )
    return output_rows, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-ready", type=Path, required=True)
    parser.add_argument(
        "--case-manifest",
        type=Path,
        required=True,
        help=(
            "JSONL: case_id, patient_id, held_out_fold, gt_path, optional pet_path"
        ),
    )
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--official-metrics", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=list(PARTITIONS),
        default=list(DEFAULT_PARTITIONS),
    )
    add_leaf_test_access_arguments(parser)
    parser.add_argument("--overlap-threshold", type=float, default=0.1)
    parser.add_argument("--connectivity", type=int, choices=[18], default=18)
    args = parser.parse_args(argv)
    try:
        _, summary = evaluate_m0_oof(
            oof_ready=args.oof_ready,
            case_manifest=args.case_manifest,
            learning_split=args.learning_split,
            experiment_config=args.experiment_config,
            official_metrics=args.official_metrics,
            rows_path=args.rows,
            summary_path=args.summary,
            partitions=args.partitions,
            test_access_receipt=args.test_access_receipt,
            run_root=args.run_root,
            overlap_threshold=args.overlap_threshold,
            connectivity=args.connectivity,
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
