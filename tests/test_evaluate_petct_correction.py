from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "evaluation"))

import evaluate_petct_correction as evaluator  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _save_nifti(path: Path, value: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(value.astype(np.uint8), np.eye(4)), str(path))


def test_main_emits_strict_null_metrics_and_atomic_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "editor": {
                    "local_radius_mm": 1.0,
                    "primary_architecture_id": (
                        "simple_operation_conditioned_residual_unet_v2"
                    ),
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config_sha = _sha256(config_path)
    learning_split = tmp_path / "learning-split.json"
    learning_split.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
                "status": "FROZEN_BEFORE_MODEL_SELECTION",
                "patient_count": 1,
                "case_count": 1,
                "case_counts": {"train": 0, "val": 1, "test": 0},
                "patients": [
                    {
                        "patient_id": "patient-1",
                        "partition": "val",
                        "case_ids": ["case-1"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    learning_split_sha = _sha256(learning_split)

    gt_volume = np.zeros((5, 5, 1), dtype=np.uint8)
    gt_volume[1:4, 2, 0] = 1
    m0_volume = np.zeros_like(gt_volume)
    m0_volume[1, 2, 0] = 1
    authorized_volume = np.zeros_like(gt_volume)
    authorized_volume[2, 2, 0] = 1
    gt_path = tmp_path / "gt.nii.gz"
    m0_path = tmp_path / "m0.nii.gz"
    authorized_path = tmp_path / "authorized.nii.gz"
    _save_nifti(gt_path, gt_volume)
    _save_nifti(m0_path, m0_volume)
    _save_nifti(authorized_path, authorized_volume)

    gt = gt_volume[:, :, 0]
    m0 = m0_volume[:, :, 0]
    authorized = authorized_volume[:, :, 0]
    scribble = authorized.copy()
    spacing_xy = np.asarray([1.0, 1.0], dtype=np.float32)
    visible_path = tmp_path / "visible.npz"
    evaluation_path = tmp_path / "evaluation.npz"
    prediction_path = tmp_path / "prediction.npz"
    np.savez_compressed(
        visible_path,
        m0=m0,
        scribble=scribble,
        cue_fg=scribble,
        cue_bg=np.zeros_like(scribble),
        spacing_xy=spacing_xy,
    )
    np.savez_compressed(evaluation_path, gt=gt, authorized=authorized)
    delta = np.zeros_like(m0)
    np.savez_compressed(
        prediction_path,
        delta=delta,
        m1=m0,
        m0=m0,
        cue=scribble.astype(np.int8),
        scribble=scribble,
        spacing_xy=spacing_xy,
    )

    learning_path = tmp_path / "learning.jsonl"
    learning_row = {
        "case_id": "case-1",
        "episode_id": "episode-1",
        "patient_id": "patient-1",
        "partition": "val",
        "learning_split_sha256": learning_split_sha,
        "experiment_config_sha256": config_sha,
        "operation": "ADD",
        "target": "SAME",
        "scope": "LOCAL",
        "visible_npz": str(visible_path.resolve()),
        "visible_sha256": _sha256(visible_path),
        "evaluation_npz": str(evaluation_path.resolve()),
        "evaluation_sha256": _sha256(evaluation_path),
        "geometry": {
            "crop_center_xy_voxel": [2.0, 2.0],
            "crop_field_mm": 5.0,
            "original_spacing_xy": [1.0, 1.0],
        },
        "source_evaluation": {
            "gt_path": str(gt_path.resolve()),
            "gt_sha256": _sha256(gt_path),
            "m0_path": str(m0_path.resolve()),
            "m0_sha256": _sha256(m0_path),
            "authorized_path": str(authorized_path.resolve()),
            "authorized_sha256": _sha256(authorized_path),
            "center_z": 0,
            "scribble_coordinates_xyz": [[2, 2, 0]],
        },
    }
    _write_jsonl(learning_path, [learning_row])

    prediction_manifest = tmp_path / "predictions.jsonl"
    prediction_row = {
        "episode_id": "episode-1",
        "patient_id": "patient-1",
        "condition": "same_weight_NULL",
        "checkpoint_sha256": "checkpoint",
        "threshold": 0.5,
        "learning_manifest": str(learning_path.resolve()),
        "learning_manifest_sha256": _sha256(learning_path),
        "partition": "val",
        "experiment_config_sha256": config_sha,
        "prediction_npz": str(prediction_path.resolve()),
        "architecture_id": "simple_operation_conditioned_residual_unet_v2",
        "parameter_count": 123,
        "gold_operation": "ADD",
        "gold_target": "SAME",
        "gold_scope": "LOCAL",
        "execution_operation": "ADD",
        "conditioning_operation": "NULL",
        "conditioning_target": "NULL",
        "conditioning_scope": "NULL",
        "execution_operation_matches_gold": True,
        "conditioning_operation_matches_gold": False,
        "ood_stress_only": False,
        "prediction_npz_sha256": _sha256(prediction_path),
        "evaluation_npz": str(evaluation_path.resolve()),
        "evaluation_npz_sha256": _sha256(evaluation_path),
        "visible_npz_sha256": _sha256(visible_path),
    }
    _write_jsonl(prediction_manifest, [prediction_row])

    rows_path = tmp_path / "metrics.jsonl"
    summary_path = tmp_path / "summary.json"
    official_metrics = tmp_path / "official_metrics.py"
    official_metrics.write_text(
        "class MetricEvaluator:\n"
        "    def __init__(self, overlap_threshold, connectivity):\n"
        "        pass\n"
        "    def __call__(self, prediction, target, case_id, spacing, suv):\n"
        "        return {'f1': 0.5, 'tp': 1, 'fp': 0, 'fn': 1}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_petct_correction.py",
            "--prediction-manifest",
            str(prediction_manifest),
            "--rows",
            str(rows_path),
            "--summary",
            str(summary_path),
            "--experiment-config",
            str(config_path),
            "--learning-split",
            str(learning_split),
            "--official-metrics",
            str(official_metrics),
        ],
    )
    assert evaluator.main() == 0
    raw_rows = rows_path.read_text(encoding="utf-8")
    assert "NaN" not in raw_rows
    row = json.loads(raw_rows)
    assert row["residual_precision"] is None
    assert row["residual_precision_defined"] is False
    assert row["residual_precision_denominator_voxels"] == 0.0
    assert row["prompt_distal_recall"] is None
    assert row["metric_grid"] == "full_original_3d_with_single_central_slice_operation"
    assert row["operation"] == "ADD"
    assert row["dmm_f1"] == 0.5
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["partition"] == "val"
    assert summary["metric_rows_sha256"] == _sha256(rows_path)
    assert summary["prediction_manifest_sha256"] == _sha256(prediction_manifest)
    assert summary["patient_clustered"]["residual_precision"]["defined"] is False
    assert summary["schema_version"] == "PETCT-BIDIRECTIONAL-CORRECTION-METRICS-v2.0"
    assert "unauthorized_addition_physical_measure" in summary["patient_clustered"]
    assert "unauthorized_removal_physical_measure" in summary["patient_clustered"]
    assert summary["safety_summary"]["unauthorized_addition_voxels_episode_sum"] == 0.0
    assert summary["safety_summary"]["unauthorized_removal_voxels_episode_sum"] == 0.0
    assert summary["safety_summary"]["operation_algebra_safety_pass"] is True


def test_atomic_multi_file_publish_rolls_back_first_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows_path = tmp_path / "rows.jsonl"
    summary_path = tmp_path / "summary.json"
    real_link = os.link
    call_count = 0

    def fail_second_link(source, target):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("synthetic second-output failure")
        return real_link(source, target)

    monkeypatch.setattr(evaluator.os, "link", fail_second_link)
    with pytest.raises(OSError, match="second-output failure"):
        evaluator.publish_evaluation_outputs_atomic(
            rows_path=rows_path,
            summary_path=summary_path,
            rows=[{"metric": None, "defined": False, "denominator": 0}],
            summary={"status": "test"},
        )
    assert not rows_path.exists()
    assert not summary_path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def _paired_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a 3-slice authorised column whose editor may only write z == 1."""

    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "editor": {
                    "local_radius_mm": 1.0,
                    "primary_architecture_id": (
                        "simple_operation_conditioned_residual_unet_v2"
                    ),
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config_sha = _sha256(config_path)
    learning_split = tmp_path / "learning-split.json"
    learning_split.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
                "status": "FROZEN_BEFORE_MODEL_SELECTION",
                "patient_count": 1,
                "case_count": 1,
                "case_counts": {"train": 0, "val": 1, "test": 0},
                "patients": [
                    {
                        "patient_id": "patient-1",
                        "partition": "val",
                        "case_ids": ["case-1"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    learning_split_sha = _sha256(learning_split)

    center_z = 1
    gt_volume = np.zeros((5, 5, 3), dtype=np.uint8)
    gt_volume[2, 2, :] = 1
    m0_volume = np.zeros_like(gt_volume)
    authorized_volume = np.zeros_like(gt_volume)
    authorized_volume[2, 2, :] = 1
    gt_path = tmp_path / "gt.nii.gz"
    m0_path = tmp_path / "m0.nii.gz"
    authorized_path = tmp_path / "authorized.nii.gz"
    _save_nifti(gt_path, gt_volume)
    _save_nifti(m0_path, m0_volume)
    _save_nifti(authorized_path, authorized_volume)

    gt = gt_volume[:, :, center_z]
    m0 = m0_volume[:, :, center_z]
    authorized = authorized_volume[:, :, center_z]
    scribble = authorized.copy()
    spacing_xy = np.asarray([1.0, 1.0], dtype=np.float32)
    visible_path = tmp_path / "visible.npz"
    evaluation_path = tmp_path / "evaluation.npz"
    prediction_path = tmp_path / "prediction.npz"
    np.savez_compressed(
        visible_path,
        m0=m0,
        scribble=scribble,
        cue_fg=scribble,
        cue_bg=np.zeros_like(scribble),
        spacing_xy=spacing_xy,
    )
    np.savez_compressed(evaluation_path, gt=gt, authorized=authorized)
    delta = authorized.copy()
    np.savez_compressed(
        prediction_path,
        delta=delta,
        m1=((m0 > 0) | (delta > 0)).astype(m0.dtype),
        m0=m0,
        cue=scribble.astype(np.int8),
        scribble=scribble,
        spacing_xy=spacing_xy,
    )

    learning_path = tmp_path / "learning.jsonl"
    _write_jsonl(
        learning_path,
        [
            {
                "case_id": "case-1",
                "episode_id": "episode-1",
                "patient_id": "patient-1",
                "partition": "val",
                "learning_split_sha256": learning_split_sha,
                "experiment_config_sha256": config_sha,
                "operation": "ADD",
                "target": "SAME",
                "scope": "COMPLETE",
                "visible_npz": str(visible_path.resolve()),
                "visible_sha256": _sha256(visible_path),
                "evaluation_npz": str(evaluation_path.resolve()),
                "evaluation_sha256": _sha256(evaluation_path),
                "geometry": {
                    "crop_center_xy_voxel": [2.0, 2.0],
                    "crop_field_mm": 5.0,
                    "original_spacing_xy": [1.0, 1.0],
                },
                "source_evaluation": {
                    "gt_path": str(gt_path.resolve()),
                    "gt_sha256": _sha256(gt_path),
                    "m0_path": str(m0_path.resolve()),
                    "m0_sha256": _sha256(m0_path),
                    "authorized_path": str(authorized_path.resolve()),
                    "authorized_sha256": _sha256(authorized_path),
                    "center_z": center_z,
                    "scribble_coordinates_xyz": [[2, 2, center_z]],
                },
            }
        ],
    )

    prediction_manifest = tmp_path / "predictions.jsonl"
    _write_jsonl(
        prediction_manifest,
        [
            {
                "episode_id": "episode-1",
                "patient_id": "patient-1",
                "condition": "scribble_plus_intent",
                "checkpoint_sha256": "checkpoint",
                "threshold": 0.5,
                "learning_manifest": str(learning_path.resolve()),
                "learning_manifest_sha256": _sha256(learning_path),
                "partition": "val",
                "experiment_config_sha256": config_sha,
                "prediction_npz": str(prediction_path.resolve()),
                "architecture_id": "simple_operation_conditioned_residual_unet_v2",
                "parameter_count": 123,
                "gold_operation": "ADD",
                "gold_target": "SAME",
                "gold_scope": "COMPLETE",
                "execution_operation": "ADD",
                "conditioning_operation": "ADD",
                "conditioning_target": "SAME",
                "conditioning_scope": "COMPLETE",
                "execution_operation_matches_gold": True,
                "conditioning_operation_matches_gold": True,
                "ood_stress_only": False,
                "prediction_npz_sha256": _sha256(prediction_path),
                "evaluation_npz": str(evaluation_path.resolve()),
                "evaluation_npz_sha256": _sha256(evaluation_path),
                "visible_npz_sha256": _sha256(visible_path),
            }
        ],
    )

    rows_path = tmp_path / "metrics.jsonl"
    summary_path = tmp_path / "summary.json"
    official_metrics = tmp_path / "official_metrics.py"
    official_metrics.write_text(
        "class MetricEvaluator:\n"
        "    def __init__(self, overlap_threshold, connectivity):\n"
        "        pass\n"
        "    def __call__(self, prediction, target, case_id, spacing, suv):\n"
        "        return {'f1': 0.5, 'tp': 1, 'fp': 0, 'fn': 1}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_petct_correction.py",
            "--prediction-manifest",
            str(prediction_manifest),
            "--rows",
            str(rows_path),
            "--summary",
            str(summary_path),
            "--experiment-config",
            str(config_path),
            "--learning-split",
            str(learning_split),
            "--official-metrics",
            str(official_metrics),
        ],
    )
    return rows_path, summary_path


def test_authorized_metrics_are_paired_operable_and_full_with_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2-D editor must not be scored by the 3-D ruler alone.

    The authorised target of a COMPLETE goal spans the whole 3-D residual, but
    the editor can only write the central axial slice.  Every authorised-class
    metric therefore has to be reported twice -- once over the domain the editor
    can actually reach (_operable) and once over the full 3-D authorised region
    (_full) -- alongside the single_slice_ceiling that connects them, so a low
    number can be attributed to the model or to the protocol.
    """

    rows_path, summary_path = _paired_fixture(tmp_path, monkeypatch)
    assert evaluator.main() == 0
    row = json.loads(rows_path.read_text(encoding="utf-8"))

    assert row["operable_domain"] == evaluator.OPERABLE_DOMAIN
    assert row["single_slice_ceiling"] == pytest.approx(1.0 / 3.0)
    assert row["authorized_voxels_operable"] == 1.0
    assert row["authorized_voxels_full"] == 3.0

    for name in evaluator.OPERABLE_PAIRED_METRICS:
        assert f"{name}_operable" in row, name
        assert f"{name}_full" in row, name
        assert row[f"{name}_full"] == row[name], name

    assert row["target_residual_recall_operable"] == pytest.approx(1.0)
    assert row["target_residual_recall_full"] == pytest.approx(1.0 / 3.0)
    assert row["target_residual_recall_full"] == pytest.approx(
        row["target_residual_recall_operable"] * row["single_slice_ceiling"]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["operable_domain"] == evaluator.OPERABLE_DOMAIN
    assert "single_slice_ceiling" in summary["patient_clustered"]
    assert "target_residual_recall_operable" in summary["patient_clustered"]
    assert "target_residual_recall_full" in summary["patient_clustered"]


def test_half_reported_authorized_metric_is_rejected() -> None:
    """Emitting only the flattering half must fail closed, not pass silently."""

    complete = {
        "single_slice_ceiling": 0.5,
        "operable_domain": evaluator.OPERABLE_DOMAIN,
    }
    for name in evaluator.OPERABLE_PAIRED_METRICS:
        complete[name] = 0.25
        complete[f"{name}_operable"] = 0.5
        complete[f"{name}_full"] = 0.25
    evaluator.require_paired_authorized_metrics(complete)

    dropped_full = dict(complete)
    victim = evaluator.OPERABLE_PAIRED_METRICS[0]
    del dropped_full[f"{victim}_full"]
    with pytest.raises(RuntimeError, match="paired"):
        evaluator.require_paired_authorized_metrics(dropped_full)

    dropped_ceiling = dict(complete)
    del dropped_ceiling["single_slice_ceiling"]
    with pytest.raises(RuntimeError, match="single_slice_ceiling"):
        evaluator.require_paired_authorized_metrics(dropped_ceiling)
