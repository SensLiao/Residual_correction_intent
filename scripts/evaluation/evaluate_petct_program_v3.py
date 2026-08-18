#!/usr/bin/env python3
"""Offline v3 compiler/editor evaluation with complete-denominator accounting.

Compiler inference must finish first and publish an immutable prediction
receipt. This evaluator then opens the physically separate label/audit
lanes. Missing, abstained, malformed, and illegal calls remain in the full
expected denominator and count as classification errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_program_contract import (  # noqa: E402
    NEW_CUE_SENTINEL,
    family_to_id,
    goal_to_family_id,
    protected_refs_policy,
    render_goal,
    validate_legal_call,
)
from common.petct_program_learning import (  # noqa: E402
    LearningContractError,
    _sha256_file,
    load_jsonl,
    load_label_manifest,
)
from common.petct_mainline_lineage import (  # noqa: E402
    MAINLINE_SOURCE,
    validate_r13_lineage_receipt,
)

EVAL_SCHEMA = "PETCT-PROGRAM-EVAL-v1.1"
ERROR_SENTINEL = -1


def _unique_by_episode(rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if not episode_id or episode_id in result:
            raise LearningContractError("%s has missing/duplicate episode_id" % role)
        result[episode_id] = dict(row)
    return result


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise LearningContractError("missing JSON artifact: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def _tree_records(directory: Path) -> dict[str, dict]:
    if not directory.is_dir():
        raise LearningContractError("missing sidecar directory: %s" % directory)
    return _unique_by_episode(
        [_load_json(path) for path in sorted(directory.glob("*.json"))], str(directory)
    )


def _load_nifti_binary(path: Path, expected_sha256: str | None = None) -> np.ndarray:
    if expected_sha256 is not None and _sha256_file(path) != expected_sha256:
        raise LearningContractError("3D evaluation mask hash mismatch: %s" % path)
    try:
        import nibabel as nib
    except ImportError:
        raise RuntimeError("nibabel is required for 3D editor evaluation") from None
    array = np.asarray(nib.load(str(path)).dataobj)
    if array.ndim != 3:
        raise LearningContractError("mask must be 3D: %s" % path)
    return (array > 0).astype(np.uint8)


def _dice(binary_a: np.ndarray, binary_b: np.ndarray) -> float | None:
    intersection = float(np.logical_and(binary_a, binary_b).sum())
    total = float(binary_a.sum()) + float(binary_b.sum())
    return 2.0 * intersection / total if total else None


def _patient_balanced_macro_f1(true_ids, pred_ids, patients, classes):
    """D14 estimand allowing ERROR_SENTINEL predictions in the denominator."""

    if not (len(true_ids) == len(pred_ids) == len(patients)) or not true_ids:
        raise LearningContractError("metric vectors must be non-empty and aligned")
    grouped: dict[str, list[int]] = {}
    for index, patient in enumerate(patients):
        grouped.setdefault(str(patient), []).append(index)
    per_class_f1: dict[str, float | None] = {}
    per_class_support: dict[str, int] = {}
    class_scores = []
    for class_id in classes:
        scores = []
        for indices in grouped.values():
            tp = sum(true_ids[i] == class_id and pred_ids[i] == class_id for i in indices)
            fp = sum(true_ids[i] != class_id and pred_ids[i] == class_id for i in indices)
            fn = sum(true_ids[i] == class_id and pred_ids[i] != class_id for i in indices)
            denominator = 2 * tp + fp + fn
            if denominator:
                scores.append(2.0 * tp / denominator)
        key = str(class_id)
        per_class_support[key] = len(scores)
        per_class_f1[key] = float(np.mean(scores)) if scores else None
        if scores:
            class_scores.append(float(np.mean(scores)))
    if not class_scores:
        raise LearningContractError("no defined patient/class F1 cells")
    return {
        "estimate": float(np.mean(class_scores)),
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
        "patient_count": len(grouped),
        "defined_class_count": len(class_scores),
    }


def _schema_validator(schema: Mapping[str, Any]):
    try:
        import jsonschema
    except ImportError:
        raise RuntimeError("jsonschema is required for program evaluation") from None
    return jsonschema.Draft202012Validator(schema)


def _prediction_family_id(prediction, label, candidate, validator) -> tuple[int, str]:
    if prediction is None:
        return ERROR_SENTINEL, "missing"
    if not validator.is_valid(prediction):
        return ERROR_SENTINEL, "schema_invalid"
    if prediction.get("decision") == "ABSTAIN":
        return ERROR_SENTINEL, "abstain"
    operation = str(label["operation"])
    if str(prediction.get("operation")) != operation:
        return ERROR_SENTINEL, "operation_changed"
    family = str(prediction.get("family") or "")
    operand = str(prediction.get("operand") or "")
    try:
        validate_legal_call(operation, family, operand)
    except Exception:
        return ERROR_SENTINEL, "illegal_call"
    if str(prediction.get("goal")) != render_goal(operation, family):
        return ERROR_SENTINEL, "inconsistent_goal"
    if dict(prediction.get("protected_refs") or {}) != dict(
        protected_refs_policy(operation, operand)
    ):
        return ERROR_SENTINEL, "inconsistent_protection"
    components = candidate.get("components", [])
    keys = [str(component.get("component_key") or "") for component in components]
    if operand != NEW_CUE_SENTINEL and operand not in keys:
        return ERROR_SENTINEL, "unknown_operand"
    if operation == "REMOVE":
        hit = candidate.get("cue_hit_component_position")
        if hit is None or not 0 <= int(hit) < len(keys) or operand != keys[int(hit)]:
            return ERROR_SENTINEL, "remove_operand_not_cue_hit"
    return family_to_id(family), "legal"


def _compiler_metrics(
    predictions,
    rows,
    candidates: Mapping[str, Mapping[str, Any]],
    pointer_targets: Mapping[str, Mapping[str, Any]],
    validator,
    *,
    partition: str = "val",
):
    pred_by_episode = _unique_by_episode(predictions, "predictions")
    expected = [row for row in rows if row.get("partition") == partition]
    if not expected:
        raise LearningContractError("evaluation partition is empty")
    expected_ids = {str(row["episode_id"]) for row in expected}
    if set(pred_by_episode) - expected_ids:
        raise LearningContractError("predictions contain unexpected episodes")
    true_ids, pred_ids, patients = [], [], []
    statuses: dict[str, int] = {}
    pointer_scores = []
    matched_mode = any("matched_state_group_id" in row for row in expected)
    group_rows: dict[str, list[tuple[int, int]]] = {}
    for row in expected:
        episode_id = str(row["episode_id"])
        if episode_id not in candidates:
            raise LearningContractError("missing candidates for expected episode")
        gold_id = goal_to_family_id(str(row["goal"]))
        predicted_id, status = _prediction_family_id(
            pred_by_episode.get(episode_id), row, candidates[episode_id], validator
        )
        statuses[status] = statuses.get(status, 0) + 1
        true_ids.append(gold_id)
        pred_ids.append(predicted_id)
        patients.append(str(row["patient_id"]))
        if matched_mode:
            group_key = "%s|%s" % (row["matched_state_group_id"], row["operation"])
            group_rows.setdefault(group_key, []).append((gold_id, predicted_id))
        if row["operation"] == "ADD" and row["goal"] != "ADD_NEW_COMPLETE":
            target = pointer_targets.get(episode_id)
            if target is None:
                raise LearningContractError("missing pointer target for ADD existing episode")
            prediction = pred_by_episode.get(episode_id)
            operand = str(prediction.get("operand") or "") if prediction else ""
            target_keys = {str(value) for value in target["pointer_target_component_keys"]}
            pointer_scores.append(
                1.0 if status == "legal" and operand in target_keys else 0.0
            )
    metric = _patient_balanced_macro_f1(true_ids, pred_ids, patients, list(range(6)))
    pair_total = pair_correct = triplet_total = triplet_correct = 0
    # Matched pair/triplet accuracy only exists in the controlled matched-state
    # lane; natural single-round labels carry no matched_state_group_id and
    # report None with zero counts instead.
    for values in group_rows.values():
        if len(values) != 3 or len({truth for truth, _ in values}) != 3:
            raise LearningContractError("matched same-operation group is not a three-family set")
        triplet_total += 1
        triplet_correct += int(all(truth == prediction for truth, prediction in values))
        for left in range(3):
            for right in range(left + 1, 3):
                pair_total += 1
                pair_correct += int(
                    values[left][0] == values[left][1]
                    and values[right][0] == values[right][1]
                )
    total = len(expected)
    return {
        "patient_balanced_family_macro_f1": metric["estimate"],
        "per_family_f1": metric["per_class_f1"],
        "per_family_patient_support": metric["per_class_support"],
        "patient_count": metric["patient_count"],
        "legal_call_rate": statuses.get("legal", 0) / total,
        "prediction_coverage_rate": (total - statuses.get("missing", 0)) / total,
        "status_counts": statuses,
        "expected_episodes": total,
        "matched_pair_joint_accuracy": pair_correct / pair_total if pair_total else None,
        "matched_pair_count": pair_total,
        "matched_triplet_exact_accuracy": triplet_correct / triplet_total if triplet_total else None,
        "matched_triplet_count": triplet_total,
        "pointer_recall_at_1": float(np.mean(pointer_scores)) if pointer_scores else None,
        "pointer_episodes": len(pointer_scores),
    }


def _editor_metrics_2d(label, visible, delta):
    with np.load(label["evaluation_npz"], allow_pickle=False) as bundle:
        if set(bundle.files) != {"target", "authorized", "gt"}:
            raise LearningContractError("evaluation tensor schema mismatch")
        gt = np.asarray(bundle["gt"], dtype=np.float32) > 0
        authorized = np.asarray(bundle["authorized"], dtype=np.float32) > 0
    with np.load(visible["visible_npz"], allow_pickle=False) as bundle:
        m0 = np.asarray(bundle["m0"], dtype=np.float32) > 0
    delta = np.asarray(delta) > 0
    if delta.shape != m0.shape or gt.shape != m0.shape or authorized.shape != m0.shape:
        raise LearningContractError("2D editor arrays have inconsistent geometry")
    corrected = (m0 | delta) if label["operation"] == "ADD" else (m0 & ~delta)
    before, after = _dice(m0, gt), _dice(corrected, gt)
    unauthorized = np.logical_and(delta, ~authorized)
    return {
        "denominator_domain": "2d_prompted_plane",
        "dice_before": before,
        "dice_after": after,
        "delta_dice": None if before is None or after is None else after - before,
        "target_recovery": (
            float(np.logical_and(delta, authorized).sum()) / float(authorized.sum())
            if authorized.sum() else None
        ),
        "unauthorized_edit_voxels": int(unauthorized.sum()),
        "unauthorized_edit_fraction_of_delta": (
            float(unauthorized.sum()) / float(delta.sum()) if delta.sum() else 0.0
        ),
    }


def _editor_metrics_3d(label, audit, volume_delta_path: Path, volume_delta_sha256: str):
    source = audit.get("source_record", {}).get("source_evaluation", {})
    required = (
        "m0_path", "m0_sha256", "authorized_path", "authorized_sha256",
        "gt_path", "gt_sha256",
    )
    if not all(source.get(key) for key in required):
        raise LearningContractError("audit lane lacks hash-bound 3D evaluation sources")
    m0 = _load_nifti_binary(Path(source["m0_path"]), str(source["m0_sha256"]))
    authorized = _load_nifti_binary(
        Path(source["authorized_path"]), str(source["authorized_sha256"])
    )
    gt = _load_nifti_binary(Path(source["gt_path"]), str(source["gt_sha256"]))
    delta = _load_nifti_binary(volume_delta_path, volume_delta_sha256)
    if not (m0.shape == authorized.shape == gt.shape == delta.shape):
        raise LearningContractError("3D editor arrays have inconsistent geometry")
    corrected = (m0 | delta) if label["operation"] == "ADD" else (m0 & ~delta)
    before, after = _dice(m0, gt), _dice(corrected, gt)
    unauthorized = np.logical_and(delta, ~authorized)
    return {
        "denominator_domain": "3d_full_volume",
        "dice_before": before,
        "dice_after": after,
        "delta_dice": None if before is None or after is None else after - before,
        "target_recovery_3d": (
            float(np.logical_and(delta, authorized).sum()) / float(authorized.sum())
            if authorized.sum() else None
        ),
        "unauthorized_edit_voxels_3d": int(unauthorized.sum()),
    }


def _verify_prediction_receipt(receipt_path: Path, predictions_path: Path) -> dict:
    receipt = _load_json(receipt_path)
    if (
        receipt.get("schema_version") != "PETCT-PROGRAM-PREDICTIONS-v1.0"
        or receipt.get("status") != "PASS"
        or receipt.get("label_lane_opened") is not False
        or Path(str(receipt.get("predictions_path") or "")).resolve() != predictions_path.resolve()
        or receipt.get("predictions_sha256") != _sha256_file(predictions_path)
    ):
        raise LearningContractError("prediction receipt is invalid or stale")
    return receipt


def _verify_editor_receipt(receipt_path: Path, manifest_path: Path) -> dict:
    receipt = _load_json(receipt_path)
    if (
        receipt.get("schema_version") != "PETCT-PROGRAM-EDITOR-PREDICTIONS-v1.0"
        or receipt.get("status") != "PASS"
        or Path(str(receipt.get("output_manifest") or "")).resolve()
        != manifest_path.resolve()
        or receipt.get("output_manifest_sha256") != _sha256_file(manifest_path)
        or receipt.get("program_source")
        not in ("predicted_compiler", "gold_oracle_ceiling")
    ):
        raise LearningContractError("editor prediction receipt is invalid or stale")
    expected_label_access = receipt["program_source"] == "gold_oracle_ceiling"
    if receipt.get("label_lane_opened") is not expected_label_access:
        raise LearningContractError("editor receipt misdeclares label-lane access")
    return receipt


def _evaluate_editor_manifest(manifest_path, expected, visible, audit):
    editor_map = _unique_by_episode(load_jsonl(manifest_path), "editor predictions")
    if set(editor_map) != set(expected):
        raise LearningContractError("editor predictions require exact expected coverage")
    rows_2d, rows_3d = [], []
    for episode_id, label in expected.items():
        if episode_id not in visible:
            raise LearningContractError("visible manifest missing editor episode")
        entry = editor_map[episode_id]
        delta_path = Path(str(entry.get("delta_path") or ""))
        if not delta_path.is_file() or entry.get("delta_sha256") != _sha256_file(delta_path):
            raise LearningContractError("2D editor delta is missing or hash-stale")
        with np.load(delta_path, allow_pickle=False) as delta_bundle:
            if set(delta_bundle.files) != {"delta"}:
                raise LearningContractError("editor delta NPZ schema mismatch")
            delta = np.asarray(delta_bundle["delta"])
        rows_2d.append(
            {"episode_id": episode_id, **_editor_metrics_2d(label, visible[episode_id], delta)}
        )
        if entry.get("volume_delta_path") is not None:
            if episode_id not in audit:
                raise LearningContractError("3D delta requires the audit evaluation lane")
            rows_3d.append({
                "episode_id": episode_id,
                **_editor_metrics_3d(
                    label, audit[episode_id], Path(str(entry["volume_delta_path"])),
                    str(entry.get("volume_delta_sha256") or ""),
                ),
            })

    def mean_defined(rows, key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "2d_prompted_plane": rows_2d,
        "3d_full_volume": rows_3d,
        "2d_coverage": len(rows_2d) / len(expected) if expected else None,
        "3d_coverage": len(rows_3d) / len(expected) if expected else None,
        "mean_delta_dice_2d": mean_defined(rows_2d, "delta_dice"),
        "mean_delta_dice_3d": mean_defined(rows_3d, "delta_dice"),
    }


def _paired_patient_gap(predicted, oracle, labels, *, seed: int, samples: int):
    predicted_rows = {row["episode_id"]: row for row in predicted["2d_prompted_plane"]}
    oracle_rows = {row["episode_id"]: row for row in oracle["2d_prompted_plane"]}
    if set(predicted_rows) != set(oracle_rows):
        raise LearningContractError("predicted/oracle editor coverage differs")
    per_patient: dict[str, list[float]] = {}
    for episode_id in predicted_rows:
        left = predicted_rows[episode_id]["delta_dice"]
        right = oracle_rows[episode_id]["delta_dice"]
        if left is None or right is None:
            continue
        patient = str(labels[episode_id]["patient_id"])
        per_patient.setdefault(patient, []).append(float(left) - float(right))
    if not per_patient:
        return None
    patient_means = np.asarray(
        [np.mean(per_patient[patient]) for patient in sorted(per_patient)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        np.mean(rng.choice(patient_means, size=len(patient_means), replace=True))
        for _ in range(samples)
    ])
    return {
        "estimand": "patient-balanced paired mean(predicted-call delta_Dice - gold-call delta_Dice)",
        "estimate": float(patient_means.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "patient_count": len(patient_means),
        "bootstrap_samples": samples,
        "seed": seed,
        "same_editor_checkpoint_required": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-receipt", type=Path, required=True)
    parser.add_argument("--lineage-receipt", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pointer-targets", type=Path, required=True)
    parser.add_argument("--editor-predictions", type=Path, default=None)
    parser.add_argument("--editor-receipt", type=Path, default=None)
    parser.add_argument("--oracle-editor-predictions", type=Path, default=None)
    parser.add_argument("--oracle-editor-receipt", type=Path, default=None)
    parser.add_argument("--audit-manifest", type=Path, default=None)
    parser.add_argument("--partition", choices=("train", "val"), default="val")
    parser.add_argument(
        "--schema", type=Path,
        default=SCRIPTS_ROOT.parent / "schemas" / "petct_program_v1.schema.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    lineage = validate_r13_lineage_receipt(args.lineage_receipt)
    prediction_receipt = _verify_prediction_receipt(args.prediction_receipt, args.predictions)
    if (
        prediction_receipt.get("source_m0_lineage") != MAINLINE_SOURCE
        or prediction_receipt.get("lineage_receipt_sha256")
        != lineage["receipt_sha256"]
    ):
        raise LearningContractError("compiler predictions are not R13-lineage bound")
    labels = load_label_manifest(args.labels, require_matched_groups=False)
    label_rows = list(labels.values())
    visible = _unique_by_episode(load_jsonl(args.inference_manifest), "inference manifest")
    candidates = _tree_records(args.candidates)
    pointer_targets = _tree_records(args.pointer_targets)
    compiler = _compiler_metrics(
        load_jsonl(args.predictions), label_rows, candidates, pointer_targets,
        _schema_validator(_load_json(args.schema)), partition=args.partition,
    )
    expected = {
        episode_id: label for episode_id, label in labels.items()
        if label["partition"] == args.partition
    }
    if args.bootstrap_samples < 1:
        parser.error("bootstrap samples must be positive")
    audit = (
        _unique_by_episode(load_jsonl(args.audit_manifest), "audit manifest")
        if args.audit_manifest is not None else {}
    )
    editor = None
    editor_receipt = None
    if args.editor_predictions is not None:
        if args.editor_receipt is None:
            parser.error("--editor-predictions requires --editor-receipt")
        editor_receipt = _verify_editor_receipt(args.editor_receipt, args.editor_predictions)
        if (
            editor_receipt.get("source_m0_lineage") != MAINLINE_SOURCE
            or editor_receipt.get("lineage_receipt_sha256")
            != lineage["receipt_sha256"]
        ):
            raise LearningContractError("editor predictions are not R13-lineage bound")
        if editor_receipt["program_source"] != "predicted_compiler":
            raise LearningContractError("primary editor artifact must use predicted calls")
        editor = _evaluate_editor_manifest(args.editor_predictions, expected, visible, audit)
    oracle_editor = None
    oracle_receipt = None
    if args.oracle_editor_predictions is not None:
        if args.oracle_editor_receipt is None or editor_receipt is None:
            parser.error("oracle editor comparison requires both editor receipts")
        oracle_receipt = _verify_editor_receipt(
            args.oracle_editor_receipt, args.oracle_editor_predictions
        )
        if oracle_receipt.get("lineage_receipt_sha256") != lineage["receipt_sha256"]:
            raise LearningContractError("oracle editor receipt binds another lineage")
        if oracle_receipt["program_source"] != "gold_oracle_ceiling":
            raise LearningContractError("oracle editor artifact is not label-declared")
        if oracle_receipt["editor_checkpoint_sha256"] != editor_receipt["editor_checkpoint_sha256"]:
            raise LearningContractError("predicted/oracle comparison changed editor weights")
        oracle_editor = _evaluate_editor_manifest(
            args.oracle_editor_predictions, expected, visible, audit
        )
    causal_gap = (
        _paired_patient_gap(
            editor, oracle_editor, labels,
            seed=args.bootstrap_seed, samples=args.bootstrap_samples,
        )
        if editor is not None and oracle_editor is not None else None
    )
    report = {
        "schema_version": EVAL_SCHEMA,
        "partition": args.partition,
        "source_m0_lineage": MAINLINE_SOURCE,
        "lineage_receipt_sha256": lineage["receipt_sha256"],
        "compiler": compiler,
        "editor_predicted_calls": editor,
        "editor_gold_call_ceiling": oracle_editor,
        "predicted_vs_gold_same_editor_gap": causal_gap,
        "denominator_domains": {
            "2d_prompted_plane": "prompted axial plane only",
            "3d_full_volume": "full original volume; never pooled with 2D",
        },
        "artifacts": {
            "predictions_sha256": _sha256_file(args.predictions),
            "prediction_receipt_sha256": _sha256_file(args.prediction_receipt),
            "labels_sha256": _sha256_file(args.labels),
            "inference_manifest_sha256": _sha256_file(args.inference_manifest),
            "schema_sha256": _sha256_file(args.schema),
            "compiler_checkpoint_sha256": prediction_receipt["checkpoint_sha256"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "PASS", "output": str(args.output), "compiler": compiler}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
