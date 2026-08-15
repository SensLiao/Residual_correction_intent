#!/usr/bin/env python3
"""Audit completed nnU-Net fold validation summaries without publishing OOF results.

This is an early, fold-local QA lane.  It reads only already committed validation
artifacts and writes a non-headline receipt outside the frozen training campaign.
The formal M0 evaluation remains gated by the complete five-fold ``OOF_READY``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "PETCT-M0-FOLD-VALIDATION-QA-v1.0"
FOLDS = (0, 1, 2, 3, 4)
CASE_PATTERN = re.compile(r"^(psma_[0-9a-f]+)_\d{4}-\d{2}-\d{2}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing regular {label}: {path}")
    return path.resolve()


def _load_json(path: Path, *, label: str) -> Any:
    path = _regular(path, label=label)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label} JSON") from exc


def _record(path: Path) -> dict[str, Any]:
    path = _regular(path, label="receipt input")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _verify_record(
    record: Mapping[str, Any], *, label: str, relative_to: Path | None = None
) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"{label} record path is missing")
    candidate = Path(raw)
    if not candidate.is_absolute():
        if relative_to is None or ".." in candidate.parts:
            raise RuntimeError(f"{label} record has an unsafe relative path")
        candidate = relative_to.resolve() / candidate
    path = _regular(candidate, label=label)
    if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
        raise RuntimeError(f"{label} changed after receipt publication")
    return path


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise RuntimeError(f"{label} must be finite")
    return converted


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{label} must be a non-negative integer")
    return value


def _metric_case_id(row: Mapping[str, Any]) -> str:
    prediction = row.get("prediction_file")
    if not isinstance(prediction, str) or not prediction.endswith(".nii.gz"):
        raise RuntimeError("validation metric row has no NIfTI prediction_file")
    return Path(prediction).name.removesuffix(".nii.gz")


def _patient_id(case_id: str) -> str:
    match = CASE_PATTERN.fullmatch(case_id)
    if match is None:
        raise RuntimeError(f"case id does not encode the frozen PSMA patient identity: {case_id}")
    return match.group(1)


def _inventory_names(records: Any, *, label: str) -> set[str]:
    if not isinstance(records, list):
        raise RuntimeError(f"fold receipt is missing {label}")
    names: set[str] = set()
    for record in records:
        raw = record.get("path") if isinstance(record, Mapping) else None
        if not isinstance(raw, str) or not raw or Path(raw).name in names:
            raise RuntimeError(f"fold receipt contains invalid/duplicate {label}")
        names.add(Path(raw).name)
    return names


def _validate_metric_row(row: Mapping[str, Any], *, case_id: str) -> dict[str, Any]:
    reference = row.get("reference_file")
    prediction = row.get("prediction_file")
    if not isinstance(reference, str) or not isinstance(prediction, str):
        raise RuntimeError(f"{case_id} metric paths are missing")
    _regular(Path(reference), label=f"{case_id} reference mask")
    _regular(Path(prediction), label=f"{case_id} prediction mask")
    metrics = row.get("metrics")
    foreground = metrics.get("1") if isinstance(metrics, Mapping) else None
    if not isinstance(foreground, Mapping):
        raise RuntimeError(f"{case_id} foreground metrics are missing")
    counts = {
        key: _nonnegative_int(foreground.get(key), label=f"{case_id} {key}")
        for key in ("TP", "FP", "FN", "TN", "n_pred", "n_ref")
    }
    if counts["TP"] + counts["FN"] != counts["n_ref"]:
        raise RuntimeError(f"{case_id} TP+FN differs from n_ref")
    if counts["TP"] + counts["FP"] != counts["n_pred"]:
        raise RuntimeError(f"{case_id} TP+FP differs from n_pred")

    dice_raw = foreground.get("Dice")
    iou_raw = foreground.get("IoU")
    dice_denominator = counts["n_ref"] + counts["n_pred"]
    iou_denominator = counts["TP"] + counts["FP"] + counts["FN"]
    if dice_denominator:
        dice = _finite_number(dice_raw, label=f"{case_id} Dice")
        expected_dice = 2.0 * counts["TP"] / dice_denominator
        if not 0.0 <= dice <= 1.0 or not math.isclose(dice, expected_dice, abs_tol=1e-12):
            raise RuntimeError(f"{case_id} Dice is inconsistent with TP/FP/FN")
    else:
        dice = None
        if isinstance(dice_raw, (int, float)) and math.isfinite(float(dice_raw)):
            raise RuntimeError(f"{case_id} empty/empty Dice must remain undefined")
    if iou_denominator:
        iou = _finite_number(iou_raw, label=f"{case_id} IoU")
        expected_iou = counts["TP"] / iou_denominator
        if not 0.0 <= iou <= 1.0 or not math.isclose(iou, expected_iou, abs_tol=1e-12):
            raise RuntimeError(f"{case_id} IoU is inconsistent with TP/FP/FN")
    else:
        iou = None
        if isinstance(iou_raw, (int, float)) and math.isfinite(float(iou_raw)):
            raise RuntimeError(f"{case_id} empty/empty IoU must remain undefined")
    return {"dice": dice, "iou": iou, **counts}


def evaluate_fold(campaign_root: Path, fold: int, spec: Mapping[str, Any], splits: Any) -> dict[str, Any]:
    if fold not in FOLDS:
        raise RuntimeError("fold must be one of 0,1,2,3,4")
    receipt_path = _regular(campaign_root / "fold_receipts" / f"fold_{fold}.json", label=f"fold {fold} receipt")
    receipt = _load_json(receipt_path, label=f"fold {fold} receipt")
    expected_receipt = {
        "status": "COMMITTED",
        "phase": "STANDARD_5FOLD_FULL_TRAINING",
        "campaign_id": campaign_root.name,
        "fold": fold,
        "full_fold_training_status": "PASS",
        "oof_status": "NOT_STARTED",
        "oof_prediction_count": 0,
        "result_count": 0,
        "thesis_citable": False,
    }
    if any(receipt.get(key) != value for key, value in expected_receipt.items()):
        raise RuntimeError(f"fold {fold} receipt is not a committed non-result fold receipt")
    _verify_record(receipt.get("campaign_spec", {}), label="CAMPAIGN_SPEC")
    if receipt.get("prerequisite_bound_hashes") != spec.get("prerequisite_bound_hashes"):
        raise RuntimeError(f"fold {fold} prerequisite hashes differ from campaign")
    output = receipt.get("output_contract")
    if not isinstance(output, Mapping) or any(
        output.get(key) != value
        for key, value in {
            "status": "PASS",
            "fold": fold,
            "actual_validation": True,
            "export_probabilities": True,
            "oof_handoff_inputs_present": True,
            "oof_publication_count": 0,
            "result_publication_count": 0,
        }.items()
    ):
        raise RuntimeError(f"fold {fold} output contract is not eligible for validation QA")
    if not isinstance(splits, list) or len(splits) != len(FOLDS):
        raise RuntimeError("frozen splits_final must contain five folds")
    split = splits[fold]
    if not isinstance(split, Mapping) or not isinstance(split.get("train"), list) or not isinstance(split.get("val"), list):
        raise RuntimeError(f"frozen split {fold} is invalid")
    train_cases = set(split["train"])
    val_cases = set(split["val"])
    if not train_cases or not val_cases or train_cases & val_cases:
        raise RuntimeError(f"fold {fold} train/validation cases overlap")
    train_patients = {_patient_id(case_id) for case_id in train_cases}
    val_patients = {_patient_id(case_id) for case_id in val_cases}
    if train_patients & val_patients:
        raise RuntimeError(f"fold {fold} train/validation patients overlap")

    artifacts = output.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError(f"fold {fold} receipt artifacts are missing")
    summary_path = _verify_record(
        artifacts.get("validation_summary", {}),
        label=f"fold {fold} validation summary",
        relative_to=campaign_root,
    )
    summary = _load_json(summary_path, label=f"fold {fold} validation summary")
    rows = summary.get("metric_per_case") if isinstance(summary, Mapping) else None
    if not isinstance(rows, list) or len(rows) != len(val_cases):
        raise RuntimeError(f"fold {fold} validation summary case count differs from split")
    observed_cases = {_metric_case_id(row) for row in rows if isinstance(row, Mapping)}
    if observed_cases != val_cases or len(observed_cases) != len(rows):
        raise RuntimeError(f"fold {fold} validation summary inventory differs from split")
    expected_masks = {f"{case_id}.nii.gz" for case_id in val_cases}
    expected_probabilities = {f"{case_id}.npz" for case_id in val_cases}
    expected_properties = {f"{case_id}.pkl" for case_id in val_cases}
    if _inventory_names(artifacts.get("validation_masks"), label="validation masks") != expected_masks:
        raise RuntimeError(f"fold {fold} mask receipt inventory differs from split")
    if _inventory_names(artifacts.get("validation_probabilities"), label="validation probabilities") != expected_probabilities:
        raise RuntimeError(f"fold {fold} probability receipt inventory differs from split")
    if _inventory_names(artifacts.get("validation_properties"), label="validation properties") != expected_properties:
        raise RuntimeError(f"fold {fold} properties receipt inventory differs from split")

    validated = {
        case_id: _validate_metric_row(row, case_id=case_id)
        for case_id, row in sorted((_metric_case_id(row), row) for row in rows)
    }
    finite_dice = [item["dice"] for item in validated.values() if item["dice"] is not None]
    foreground_mean = summary.get("foreground_mean")
    reported_dice = _finite_number(
        foreground_mean.get("Dice") if isinstance(foreground_mean, Mapping) else None,
        label=f"fold {fold} foreground mean Dice",
    )
    computed_dice = mean(finite_dice)
    if not math.isclose(reported_dice, computed_dice, abs_tol=1e-12):
        raise RuntimeError(f"fold {fold} reported foreground mean Dice does not match case rows")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "campaign_id": campaign_root.name,
        "fold": fold,
        "fold_receipt": _record(receipt_path),
        "validation_summary": _record(summary_path),
        "patient_exclusion": "PASS",
        "case_count": len(validated),
        "patient_count": len(val_patients),
        "positive_or_nonempty_case_count": len(finite_dice),
        "empty_empty_case_count": len(validated) - len(finite_dice),
        "nnunet_validation_diagnostics": {
            "macro_dice": computed_dice,
            "median_dice": median(finite_dice),
            "min_dice": min(finite_dice),
            "max_dice": max(finite_dice),
            "total_tp_voxels": sum(item["TP"] for item in validated.values()),
            "total_fp_voxels": sum(item["FP"] for item in validated.values()),
            "total_fn_voxels": sum(item["FN"] for item in validated.values()),
        },
        "artifact_integrity_scope": (
            "COMMITTED fold receipt inventory plus rehashed validation summary and regular prediction/reference paths; "
            "large probability arrays remain bound by the earlier full fold receipt validation"
        ),
        "official_autopetv_oof_evaluation": "NOT_RUN_REQUIRES_COMPLETE_FIVE_FOLD_OOF_READY",
        "scientific_role": "FOLD_LOCAL_VALIDATION_DIAGNOSTIC_ONLY",
        "thesis_citable": False,
        "claim_boundary": (
            "Partial training-fold validation QA only; not the five-fold OOF M0 estimate, not a held-out test result, "
            "and not evidence that P2T or residual correction works."
        ),
    }


def evaluate_folds(campaign_root: Path, folds: Sequence[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    campaign_root = campaign_root.resolve()
    if campaign_root.is_symlink() or not campaign_root.is_dir():
        raise RuntimeError("campaign root is missing or unsafe")
    if not folds or len(set(folds)) != len(folds) or any(fold not in FOLDS for fold in folds):
        raise RuntimeError("fold selection must be unique members of 0,1,2,3,4")
    spec_path = campaign_root / "CAMPAIGN_SPEC.json"
    spec = _load_json(spec_path, label="CAMPAIGN_SPEC")
    expected_spec = {
        "status": "STAGED",
        "phase": "STANDARD_5FOLD_FULL_TRAINING",
        "campaign_id": campaign_root.name,
        "full_training_status": "NOT_STARTED",
        "oof_status": "NOT_STARTED",
        "oof_prediction_count": 0,
        "result_count": 0,
        "thesis_citable": False,
    }
    if any(spec.get(key) != value for key, value in expected_spec.items()):
        raise RuntimeError("campaign is not the frozen pre-OOF full-training campaign")
    paths = spec.get("prerequisite_paths")
    split_path = _regular(Path(paths.get("splits_final", "")), label="splits_final") if isinstance(paths, Mapping) else None
    if split_path is None:
        raise RuntimeError("campaign does not bind splits_final")
    split_sha = sha256_file(split_path)
    if spec.get("prerequisite_bound_hashes", {}).get("splits_final") != split_sha:
        raise RuntimeError("splits_final changed after campaign initialization")
    splits = _load_json(split_path, label="splits_final")
    records = [evaluate_fold(campaign_root, fold, spec, splits) for fold in folds]
    combined_dice = [record["nnunet_validation_diagnostics"]["macro_dice"] for record in records]
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "campaign_id": campaign_root.name,
        "campaign_spec": _record(spec_path),
        "splits_final": _record(split_path),
        "folds_evaluated": list(folds),
        "case_count": sum(record["case_count"] for record in records),
        "patient_count_sum_by_fold": sum(record["patient_count"] for record in records),
        "fold_macro_dice_diagnostic_mean": mean(combined_dice),
        "official_autopetv_oof_evaluation": "NOT_RUN_REQUIRES_COMPLETE_FIVE_FOLD_OOF_READY",
        "scientific_role": "FOLD_LOCAL_VALIDATION_DIAGNOSTIC_ONLY",
        "thesis_citable": False,
        "claim_boundary": (
            "This receipt proves fold-local artifact and metric consistency for the selected completed folds only; "
            "the canonical five-fold OOF evaluator remains blocked until FULL_TRAIN_READY and OOF_READY."
        ),
    }
    return records, completion


def publish_evaluation(output_dir: Path, records: Sequence[Mapping[str, Any]], completion: Mapping[str, Any]) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite fold validation QA output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent))
    try:
        for record in records:
            path = staging / f"fold_{record['fold']}_validation_qa.json"
            path.write_text(json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bound_completion = dict(completion)
        bound_completion["fold_receipts"] = [
            _record(staging / f"fold_{record['fold']}_validation_qa.json") for record in records
        ]
        (staging / "FOLD_VALIDATION_QA_COMPLETE.json").write_text(
            json.dumps(bound_completion, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir / "FOLD_VALIDATION_QA_COMPLETE.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    records, completion = evaluate_folds(args.campaign_root, args.folds)
    receipt = publish_evaluation(args.output_dir, records, completion)
    print(json.dumps({"status": "COMPLETE", "receipt": str(receipt), "folds": args.folds}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
