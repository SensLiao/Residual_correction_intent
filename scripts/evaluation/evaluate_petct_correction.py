#!/usr/bin/env python3
"""Compute correction metrics and patient-clustered summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import nibabel as nib
from scipy import ndimage

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import (  # noqa: E402
    LearningContractError,
    load_jsonl,
    sha256_file,
    validate_manifest_rows_against_frozen_learning_split,
)
from common.petct_route_a_core import patient_cluster_summary  # noqa: E402
from common.petct_models import EDITOR_PRIMARY_ARCHITECTURE_ID  # noqa: E402
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)
from evaluate_petct_m0_oof import load_official_metric_evaluator  # noqa: E402
from petct_bidirectional_metrics import (  # noqa: E402
    BidirectionalMetricError,
    bidirectional_correction_metrics,
)


METRIC_GRID = "full_original_3d_with_single_central_slice_operation"
GT_LESION_IDENTITY_BASIS = "18_connected_full_3d_gt_projected_to_central_slice"
WRONG_SCOPE_ELIGIBILITY = "SAME_ONLY_SCOPE_FLIP_DESCRIPTIVE"

# The editor writes one axial slice; a *_COMPLETE authorised target spans the
# whole 3-D residual.  Scoring the 2-D editor with the 3-D ruler alone conflates
# "the model missed it" with "the protocol never let the model reach it", so
# every authorised-class metric is reported over both domains and the ceiling
# that connects them travels with them.  Exactly: full = operable * ceiling.
OPERABLE_DOMAIN = "central_axial_slice_reachable_by_the_2d_editor"
OPERABLE_PAIRED_METRICS = (
    "target_residual_recall",
    "authorized_delta_recall",
    "authorized_add_recall",
    "authorized_remove_recall",
    "prompt_distal_recall",
    "complete_fp_removal",
)


def require_paired_authorized_metrics(row: Mapping[str, Any]) -> None:
    """Fail closed unless both halves and the ceiling are present.

    Reporting only ``_operable`` hides the protocol ceiling and flatters the
    model; reporting only ``_full`` hides that the ceiling -- not the model --
    caused the loss.  Neither is acceptable on its own.
    """

    if "single_slice_ceiling" not in row:
        raise RuntimeError(
            "authorised metrics require single_slice_ceiling alongside them"
        )
    if row.get("operable_domain") != OPERABLE_DOMAIN:
        raise RuntimeError(f"operable_domain must be {OPERABLE_DOMAIN}")
    missing = [
        f"{name}_{half}"
        for name in OPERABLE_PAIRED_METRICS
        for half in ("operable", "full")
        if f"{name}_{half}" not in row
    ]
    if missing:
        raise RuntimeError(
            "authorised metrics must be reported paired over both the operable "
            f"and the full domain; missing: {', '.join(sorted(missing))}"
        )


def _episode_id_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    identifiers = sorted(str(row.get("episode_id") or "") for row in rows)
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise RuntimeError("condition eligibility has empty/duplicate episode_id")
    payload = json.dumps(identifiers, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


METRICS = (
    "dice_gain",
    "delta_dice",
    "dmm_f1",
    "fpv_ml",
    "fnv_ml",
    "target_residual_recall",
    "authorized_add_recall",
    "authorized_remove_recall",
    "prompt_distal_recall",
    "unauthorized_addition_voxels",
    "unauthorized_addition_physical_measure",
    "unauthorized_removal_voxels",
    "unauthorized_removal_physical_measure",
    "residual_precision",
    "true_positive_preservation_rate",
    "complete_fp_removal",
    "operation_algebra_safety_pass",
    "target_operation_safety_pass",
    "single_slice_ceiling",
) + tuple(
    f"{name}_{half}"
    for name in OPERABLE_PAIRED_METRICS
    for half in ("operable", "full")
)


def _strict_json_bytes(payload: Any, *, jsonl: bool = False) -> bytes:
    if jsonl:
        text = "".join(
            json.dumps(row, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n"
            for row in payload
        )
    else:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    return text.encode("utf-8")


def publish_evaluation_outputs_atomic(
    *,
    rows_path: Path,
    summary_path: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Stage both files, then publish no-clobber links with rollback on failure."""

    targets = [Path(rows_path), Path(summary_path)]
    if targets[0].resolve() == targets[1].resolve():
        raise RuntimeError("rows and summary outputs must be different paths")
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite evaluation output: %s" % existing[0])
    payloads = [_strict_json_bytes(rows, jsonl=True), _strict_json_bytes(summary)]
    staged = []
    published = []
    try:
        for target, payload in zip(targets, payloads):
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, raw_path = tempfile.mkstemp(
                prefix=".%s." % target.name,
                suffix=".tmp",
                dir=str(target.parent),
            )
            staged_path = Path(raw_path)
            staged.append(staged_path)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for staged_path, target in zip(staged, targets):
            os.link(str(staged_path), str(target))
            published.append(target)
    except Exception:
        for target in reversed(published):
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for staged_path in staged:
            try:
                staged_path.unlink()
            except FileNotFoundError:
                pass


def inverse_crop_binary(
    crop: np.ndarray,
    *,
    output_shape,
    center_xy,
    original_spacing_xy,
    field_mm: float,
) -> np.ndarray:
    value = np.asarray(crop)
    if value.ndim != 2:
        raise RuntimeError("prediction crop must be 2D")
    x, y = np.meshgrid(
        np.arange(output_shape[0], dtype=float),
        np.arange(output_shape[1], dtype=float),
        indexing="ij",
    )
    spacing = np.asarray(original_spacing_xy, dtype=float)
    center = np.asarray(center_xy, dtype=float)
    crop_x = ((x - center[0]) * spacing[0] / field_mm + 0.5) * value.shape[0] - 0.5
    crop_y = ((y - center[1]) * spacing[1] / field_mm + 0.5) * value.shape[1] - 0.5
    return (
        ndimage.map_coordinates(
            value.astype(np.float32),
            [crop_x, crop_y],
            order=0,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        > 0.5
    )


def physical_crop_labels(
    labels: np.ndarray,
    *,
    output_shape,
    center_xy,
    original_spacing_xy,
    field_mm: float,
) -> np.ndarray:
    value = np.asarray(labels)
    if value.ndim != 2 or len(output_shape) != 2:
        raise RuntimeError("GT lesion identity crop must be 2D")
    spacing = np.asarray(original_spacing_xy, dtype=float)
    center = np.asarray(center_xy, dtype=float)
    offsets_x = (
        (np.arange(int(output_shape[0]), dtype=float) + 0.5) / int(output_shape[0])
        - 0.5
    ) * float(field_mm)
    offsets_y = (
        (np.arange(int(output_shape[1]), dtype=float) + 0.5) / int(output_shape[1])
        - 0.5
    ) * float(field_mm)
    x = center[0] + offsets_x / spacing[0]
    y = center[1] + offsets_y / spacing[1]
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    return ndimage.map_coordinates(
        value,
        [grid_x, grid_y],
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(np.int32)


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument(
        "--official-metrics",
        type=Path,
        required=True,
        help="pinned AutoPET V MetricEvaluator source used for DMM",
    )
    parser.add_argument("--distal-radius-mm", type=float)
    add_leaf_test_access_arguments(parser)
    args = parser.parse_args()
    for input_path, label in (
        (args.prediction_manifest, "prediction manifest"),
        (args.experiment_config, "experiment config"),
        (args.learning_split, "learning split"),
        (args.official_metrics, "official metrics"),
    ):
        if input_path.is_symlink() or not input_path.is_file():
            parser.error(f"{label} must be a non-symlink regular file")
    with args.experiment_config.open("r", encoding="utf-8") as stream:
        experiment_config = json.load(stream)
    frozen_radius = float(experiment_config["editor"]["local_radius_mm"])
    if not np.isfinite(frozen_radius) or frozen_radius <= 0:
        raise RuntimeError("frozen local radius must be positive and finite")
    if args.distal_radius_mm is not None and not np.isclose(
        args.distal_radius_mm, frozen_radius, atol=0, rtol=0
    ):
        parser.error("--distal-radius-mm differs from the frozen experiment config")
    config_sha256 = sha256_file(args.experiment_config)
    prediction_rows = load_jsonl(args.prediction_manifest)
    metric_evaluator_class = load_official_metric_evaluator(args.official_metrics)
    prediction_manifest_sha256 = sha256_file(args.prediction_manifest)
    episodes = [str(row.get("episode_id") or "") for row in prediction_rows]
    if any(not episode for episode in episodes) or len(episodes) != len(set(episodes)):
        raise RuntimeError("prediction manifest has empty or duplicate episode_id")

    def singleton(key):
        values = {json.dumps(row.get(key), sort_keys=True) for row in prediction_rows}
        if len(values) != 1:
            raise RuntimeError("prediction manifest mixes %s" % key)
        return prediction_rows[0].get(key)

    singleton_values = {}
    for key in (
        "condition",
        "checkpoint_sha256",
        "threshold",
        "learning_manifest",
        "learning_manifest_sha256",
        "partition",
        "experiment_config_sha256",
        "architecture_id",
        "parameter_count",
    ):
        singleton_values[key] = singleton(key)
    if (
        singleton_values["architecture_id"] != EDITOR_PRIMARY_ARCHITECTURE_ID
        or experiment_config["editor"].get("primary_architecture_id")
        != EDITOR_PRIMARY_ARCHITECTURE_ID
    ):
        raise RuntimeError("prediction architecture differs from the frozen v2 editor")
    condition = str(singleton_values["condition"])
    for row in prediction_rows:
        gold_operation = str(row.get("gold_operation") or "")
        execution_operation = str(row.get("execution_operation") or "")
        if execution_operation not in {"ADD", "REMOVE"}:
            raise RuntimeError("prediction omits a legal execution operation")
        expected_match = execution_operation == gold_operation
        if row.get("execution_operation_matches_gold") is not expected_match:
            raise RuntimeError("prediction execution-operation receipt is inconsistent")
        is_ood = condition == "wrong_operation_OOD"
        if bool(row.get("ood_stress_only")) != is_ood:
            raise RuntimeError("prediction OOD stress role is inconsistent with condition")
        if is_ood:
            expected_conditioning = "REMOVE" if gold_operation == "ADD" else "ADD"
            if (
                row.get("conditioning_operation") != expected_conditioning
                or row.get("conditioning_operation_matches_gold") is not False
            ):
                raise RuntimeError("wrong_operation_OOD did not flip only the conditioning token")
            if execution_operation != gold_operation:
                raise RuntimeError("wrong_operation_OOD must not change physical set algebra")
        if condition != "predicted_slots" and execution_operation != gold_operation:
            raise RuntimeError("same-weight/control arm changed physical set algebra")
        if condition == "predicted_slots" and row.get("conditioning_operation") != execution_operation:
            raise RuntimeError("predicted_slots conditioning and execution operations diverged")
        if condition == "same_weight_wrong_scope":
            gold_target = str(row.get("gold_target") or "")
            gold_scope = str(row.get("gold_scope") or "")
            expected_scope = "COMPLETE" if gold_scope == "LOCAL" else "LOCAL"
            if (
                gold_target != "SAME"
                or gold_scope not in {"LOCAL", "COMPLETE"}
                or row.get("conditioning_target") != gold_target
                or row.get("conditioning_scope") != expected_scope
            ):
                raise RuntimeError(
                    "same_weight_wrong_scope must flip only scope on SAME episodes"
                )
    partition = str(singleton_values["partition"])
    try:
        test_access = enforce_partition_access(
            partition,
            receipt_path=args.test_access_receipt,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=args.run_root,
            output_paths=(args.rows, args.summary),
        )
    except TestAccessError as error:
        parser.error(str(error))
    test_access_sha256 = (
        None if test_access is None else str(test_access["receipt_sha256"])
    )
    if prediction_rows[0]["experiment_config_sha256"] != config_sha256:
        raise RuntimeError("predictions were produced with a different experiment config")
    raw_learning_manifest = Path(str(prediction_rows[0]["learning_manifest"]))
    if raw_learning_manifest.is_symlink() or not raw_learning_manifest.is_file():
        raise RuntimeError("learning manifest must be a non-symlink regular file")
    learning_manifest = raw_learning_manifest.resolve()
    if sha256_file(learning_manifest) != prediction_rows[0]["learning_manifest_sha256"]:
        raise RuntimeError("learning manifest hash mismatch")
    learning_rows = load_jsonl(learning_manifest)
    try:
        validate_manifest_rows_against_frozen_learning_split(
            learning_rows,
            args.learning_split,
            require_episode_id=True,
            allowed_partitions={"train", "val", "test"},
        )
    except LearningContractError as error:
        parser.error(str(error))
    all_expected_rows = [
        row for row in learning_rows if row["partition"] == prediction_rows[0]["partition"]
    ]
    ineligible_rows: list[Mapping[str, Any]] = []
    if condition == "same_weight_wrong_scope":
        expected_rows = [
            row
            for row in all_expected_rows
            if row.get("target") == "SAME"
            and row.get("scope") in {"LOCAL", "COMPLETE"}
        ]
        ineligible_rows = [
            row
            for row in all_expected_rows
            if row.get("target") == "NEW"
            and row.get("scope") == "COMPLETE"
        ]
        if len(expected_rows) + len(ineligible_rows) != len(all_expected_rows):
            raise RuntimeError(
                "same_weight_wrong_scope eligibility found an illegal target/scope"
            )
        if not expected_rows:
            raise RuntimeError(
                "same_weight_wrong_scope has no eligible SAME episodes"
            )
        ineligible_hash = _episode_id_set_sha256(ineligible_rows)
        if any(
            row.get("condition_eligibility") != WRONG_SCOPE_ELIGIBILITY
            or row.get("ineligible_episode_count") != len(ineligible_rows)
            or row.get("ineligible_episode_set_sha256") != ineligible_hash
            for row in prediction_rows
        ):
            raise RuntimeError(
                "same_weight_wrong_scope eligibility receipt is incomplete or stale"
            )
    else:
        expected_rows = all_expected_rows
    learning_split_sha256 = sha256_file(args.learning_split)
    if {row.get("learning_split_sha256") for row in expected_rows} != {
        learning_split_sha256
    }:
        raise RuntimeError(
            "learning manifest differs from the receipt-bound learning split"
        )
    if any(row.get("experiment_config_sha256") != config_sha256 for row in expected_rows):
        raise RuntimeError("learning tensors were built with a different experiment config")
    expected_episodes = {str(row["episode_id"]) for row in expected_rows}
    if set(episodes) != expected_episodes:
        raise RuntimeError("prediction manifest is not the complete frozen partition")
    expected_patients = {
        str(row["episode_id"]): str(row["patient_id"]) for row in expected_rows
    }
    expected_by_episode = {
        str(row["episode_id"]): row for row in expected_rows
    }
    if any(
        str(row.get("patient_id")) != expected_patients[str(row["episode_id"])]
        for row in prediction_rows
    ):
        raise RuntimeError("episode-to-patient mapping changed")
    output_rows = []
    for row in prediction_rows:
        episode = str(row["episode_id"])
        source = expected_by_episode[episode]
        if Path(row["evaluation_npz"]).resolve() != Path(source["evaluation_npz"]).resolve():
            raise RuntimeError("prediction row points at the wrong evaluation artifact")
        if row.get("evaluation_npz_sha256") != source.get("evaluation_sha256"):
            raise RuntimeError("prediction row evaluation hash differs from manifest")
        if row.get("visible_npz_sha256") != source.get("visible_sha256"):
            raise RuntimeError("prediction row visible hash differs from manifest")
        if sha256_file(Path(row["prediction_npz"])) != row.get("prediction_npz_sha256"):
            raise RuntimeError("prediction artifact hash mismatch")
        visible_path = Path(source["visible_npz"])
        if sha256_file(visible_path) != source["visible_sha256"]:
            raise RuntimeError("frozen visible tensor hash mismatch")

        operation = str(source.get("operation") or "")
        execution_operation = str(row.get("execution_operation") or "")
        scope = str(source.get("scope") or "")
        if row.get("gold_operation") != operation:
            raise RuntimeError("prediction row changed the gold operation")
        with np.load(row["prediction_npz"], allow_pickle=False) as prediction, np.load(
            row["evaluation_npz"], allow_pickle=False
        ) as evaluation, np.load(visible_path, allow_pickle=False) as visible:
            expected_prediction_fields = {
                "delta", "m1", "m0", "cue", "scribble", "spacing_xy"
            }
            if set(prediction.files) != expected_prediction_fields:
                raise RuntimeError("prediction NPZ schema mismatch")
            if not np.array_equal(prediction["m0"], visible["m0"]):
                raise RuntimeError("prediction redundantly stored a different M0")
            if not np.array_equal(prediction["scribble"], visible["scribble"]):
                raise RuntimeError("prediction redundantly stored a different cue support")
            expected_cue = np.asarray(visible["cue_fg"], dtype=np.int8) - np.asarray(
                visible["cue_bg"], dtype=np.int8
            )
            if not np.array_equal(prediction["cue"], expected_cue):
                raise RuntimeError("prediction redundantly stored a different signed cue")
            if not np.array_equal(prediction["spacing_xy"], visible["spacing_xy"]):
                raise RuntimeError("prediction redundantly stored different spacing")
            delta = np.asarray(prediction["delta"])
            crop_m0 = np.asarray(visible["m0"])
            crop_m1 = np.asarray(prediction["m1"])
            crop_gt = np.asarray(evaluation["gt"])
            crop_authorized = np.asarray(evaluation["authorized"])
            crop_cue = np.asarray(visible["scribble"])
            crop_spacing = np.asarray(visible["spacing_xy"])
        if not np.all(np.isin(delta, [0, 1])) or not np.all(np.isin(crop_m1, [0, 1])):
            raise RuntimeError("prediction masks must be binary")
        expected_crop_m1 = (
            (crop_m0 > 0) | (delta > 0)
            if execution_operation == "ADD"
            else (crop_m0 > 0) & ~(delta > 0)
        )
        if not np.array_equal(crop_m1 > 0, expected_crop_m1):
            raise RuntimeError(f"prediction violates {execution_operation} set algebra")
        try:
            roi_metrics = bidirectional_correction_metrics(
                gt=crop_gt,
                m0=crop_m0,
                m1=crop_m1,
                authorized_target=crop_authorized,
                cue_support=crop_cue,
                operation=operation,
                execution_operation=execution_operation,
                scope=scope,
                spacing_xyz=crop_spacing.tolist(),
                distal_radius_mm=frozen_radius,
            )
        except BidirectionalMetricError as error:
            raise RuntimeError(f"ROI operation metric contract failed: {error}") from error

        source_eval = source["source_evaluation"]
        source_images = {}
        for name in ("gt", "m0", "authorized"):
            path = Path(source_eval[name + "_path"])
            if sha256_file(path) != source_eval[name + "_sha256"]:
                raise RuntimeError("full-volume %s source hash mismatch" % name)
            source_images[name] = nib.load(str(path))
        center_z = int(source_eval["center_z"])
        full_gt = np.asarray(source_images["gt"].dataobj) > 0
        full_m0 = np.asarray(source_images["m0"].dataobj) > 0
        full_authorized = np.asarray(source_images["authorized"].dataobj) > 0
        full_delta_slice = inverse_crop_binary(
            delta,
            output_shape=full_m0.shape[:2],
            center_xy=source["geometry"]["crop_center_xy_voxel"],
            original_spacing_xy=source["geometry"]["original_spacing_xy"],
            field_mm=float(source["geometry"]["crop_field_mm"]),
        )
        full_delta = np.zeros_like(full_m0, dtype=bool)
        full_delta[:, :, center_z] = full_delta_slice
        full_m1 = (
            full_m0 | full_delta
            if execution_operation == "ADD"
            else full_m0 & ~full_delta
        )
        full_cue = np.zeros_like(full_m0, dtype=bool)
        coordinates = source_eval.get("cue_coordinates_xyz") or source_eval.get(
            "scribble_coordinates_xyz"
        )
        if not isinstance(coordinates, list) or not coordinates:
            raise RuntimeError("source evaluation omits cue coordinates")
        for x, y, z in coordinates:
            if int(z) != center_z:
                raise RuntimeError("cue coordinate changed central slice")
            full_cue[int(x), int(y), int(z)] = True
        spacing_xyz = tuple(float(value) for value in source_images["gt"].header.get_zooms()[:3])
        try:
            metrics = bidirectional_correction_metrics(
                gt=full_gt,
                m0=full_m0,
                m1=full_m1,
                authorized_target=full_authorized,
                cue_support=full_cue,
                operation=operation,
                execution_operation=execution_operation,
                scope=scope,
                spacing_xyz=spacing_xyz,
                distal_radius_mm=frozen_radius,
            )
        except BidirectionalMetricError as error:
            raise RuntimeError(f"full-volume operation metric contract failed: {error}") from error

        # Same ruler, restricted to the slice the 2-D editor is allowed to write.
        # cue lies on center_z and cue subset authorized, so this is never empty.
        operable_authorized = np.zeros_like(full_authorized)
        operable_authorized[:, :, center_z] = full_authorized[:, :, center_z]
        authorized_full_voxels = float(full_authorized.sum())
        authorized_operable_voxels = float(operable_authorized.sum())
        try:
            operable_metrics = bidirectional_correction_metrics(
                gt=full_gt,
                m0=full_m0,
                m1=full_m1,
                authorized_target=operable_authorized,
                cue_support=full_cue,
                operation=operation,
                execution_operation=execution_operation,
                scope=scope,
                spacing_xyz=spacing_xyz,
                distal_radius_mm=frozen_radius,
            )
        except BidirectionalMetricError as error:
            raise RuntimeError(
                f"operable-domain operation metric contract failed: {error}"
            ) from error
        metrics["operable_domain"] = OPERABLE_DOMAIN
        metrics["single_slice_ceiling"] = (
            authorized_operable_voxels / authorized_full_voxels
        )
        metrics["authorized_voxels_operable"] = authorized_operable_voxels
        metrics["authorized_voxels_full"] = authorized_full_voxels
        for name in OPERABLE_PAIRED_METRICS:
            metrics[f"{name}_operable"] = operable_metrics[name]
            metrics[f"{name}_full"] = metrics[name]
        require_paired_authorized_metrics(metrics)

        official = metric_evaluator_class(overlap_threshold=0.1, connectivity=18)(
            full_m1.astype(np.uint8),
            full_gt.astype(np.uint8),
            episode,
            spacing=spacing_xyz,
            suv=None,
        )
        if not isinstance(official, Mapping):
            raise RuntimeError("official AutoPET V evaluator returned a non-object")
        metrics.update(
            {
                "dmm_f1": _finite_or_none(official.get("f1")),
                "dmm_tp": _finite_or_none(official.get("tp")),
                "dmm_fp": _finite_or_none(official.get("fp")),
                "dmm_fn": _finite_or_none(official.get("fn")),
            }
        )
        output_rows.append(
            {
                **row,
                **metrics,
                "experiment_config_sha256": config_sha256,
                "distal_radius_mm": frozen_radius,
                "metric_grid": METRIC_GRID,
                "gt_lesion_identity_basis": GT_LESION_IDENTITY_BASIS,
                "learning_split_sha256": learning_split_sha256,
                "test_access_receipt_sha256": test_access_sha256,
                "roi_diagnostic": roi_metrics,
            }
        )
    physical_measure_units = {str(row["physical_measure_unit"]) for row in output_rows}
    if physical_measure_units != {"mm^3"}:
        raise RuntimeError("canonical correction metrics must use full-volume mm^3")
    metric_rows_bytes = _strict_json_bytes(output_rows, jsonl=True)
    safety_summary = {
        "physical_measure_unit": "mm^3",
        "unauthorized_addition_voxels_episode_sum": float(
            sum(row["unauthorized_addition_voxels"] for row in output_rows)
        ),
        "unauthorized_removal_voxels_episode_sum": float(
            sum(row["unauthorized_removal_voxels"] for row in output_rows)
        ),
        "operation_algebra_safety_pass": all(
            row["operation_algebra_safety_pass"] == 1.0 for row in output_rows
        ),
        "target_operation_safety_pass": all(
            row["target_operation_safety_pass"] == 1.0 for row in output_rows
        ),
        "execution_operation_match_rate": float(
            np.mean(
                [
                    float(row["execution_operation_matches_target"])
                    for row in output_rows
                ]
            )
        ),
        "true_positive_preservation": patient_cluster_summary(
            output_rows, "true_positive_preservation_rate"
        ),
    }
    summary = {
        "schema_version": "PETCT-BIDIRECTIONAL-CORRECTION-METRICS-v2.0",
        "condition": output_rows[0]["condition"],
        "analysis_role": (
            "OOD_STRESS_ONLY"
            if output_rows[0]["condition"] == "wrong_operation_OOD"
            else "DESCRIPTIVE_SAME_ONLY_SCOPE_DIAGNOSTIC"
            if output_rows[0]["condition"] == "same_weight_wrong_scope"
            else "PRIMARY_OR_ABLATION"
        ),
        "partition": partition,
        "checkpoint_sha256": output_rows[0]["checkpoint_sha256"],
        "architecture_id": output_rows[0]["architecture_id"],
        "parameter_count": output_rows[0]["parameter_count"],
        "threshold": output_rows[0]["threshold"],
        "learning_manifest_sha256": output_rows[0]["learning_manifest_sha256"],
        "experiment_config_sha256": config_sha256,
        "learning_split_sha256": learning_split_sha256,
        "test_access_receipt_sha256": test_access_sha256,
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "metric_rows_sha256": hashlib.sha256(metric_rows_bytes).hexdigest(),
        "distal_radius_mm": frozen_radius,
        "metric_grid": METRIC_GRID,
        "gt_lesion_identity_basis": GT_LESION_IDENTITY_BASIS,
        "operable_domain": OPERABLE_DOMAIN,
        "physical_measure_unit": "mm^3",
        "episode_count": len(output_rows),
        "condition_eligibility": output_rows[0].get(
            "condition_eligibility", "ALL_LEGAL_EPISODES"
        ),
        "ineligible_episode_count": len(ineligible_rows),
        "ineligible_episode_set_sha256": _episode_id_set_sha256(ineligible_rows),
        "patient_clustered": {
            metric: patient_cluster_summary(output_rows, metric) for metric in METRICS
        },
        "patient_clustered_by_operation": {
            operation: {
                metric: patient_cluster_summary(
                    [row for row in output_rows if row["operation"] == operation],
                    metric,
                )
                for metric in METRICS
            }
            for operation in ("ADD", "REMOVE")
            if any(row["operation"] == operation for row in output_rows)
        },
        "safety_summary": safety_summary,
        "inferential_test": "DEFERRED_TO_LOCKED_MULTI_CONDITION_ANALYSIS",
    }
    publish_evaluation_outputs_atomic(
        rows_path=args.rows,
        summary_path=args.summary,
        rows=output_rows,
        summary=summary,
    )
    print(json.dumps(summary, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
