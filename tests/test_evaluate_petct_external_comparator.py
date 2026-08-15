from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
for directory in (SCRIPTS, SCRIPTS / "comparators"):
    sys.path.insert(0, str(directory))

from common.petct_learning import sha256_file  # noqa: E402
from evaluate_petct_external_comparator import (  # noqa: E402
    EvaluationError,
    evaluate_external_output,
)


# The module-level blanket skip took the partition-leakage and M0-removal
# safety guards offline together with the genuinely v1-only table semantics --
# seven tests, four of which have nothing to do with the v1 campaign.  The skip
# now applies only to the three that really are v1 table semantics.
legacy_v1_table = pytest.mark.skip(
    reason=(
        "LEGACY_V1_PROVENANCE_ONLY: positive-only/union-with-M0 comparator "
        "evaluation is REMOVE_UNSUPPORTED in the current v2 campaign"
    )
)


CONFIG = PROJECT / "configs" / "petct_external_comparators.json"


class FakeMetricEvaluator:
    def __init__(self, *, overlap_threshold: float, connectivity: int) -> None:
        assert overlap_threshold == 0.1
        assert connectivity == 18

    def __call__(self, prediction, gt, case_id, *, spacing, suv):
        assert prediction.shape == gt.shape == suv.shape
        assert case_id == "case-001"
        assert len(spacing) == 3
        return {"f1": 0.75, "tp": 1, "fp": 0, "fn": 0}


def _save(path: Path, values: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(values, np.eye(4)), str(path))


def _fixture(tmp_path: Path, *, method: str, policy: str, prediction: np.ndarray):
    shape = (5, 5, 1)
    gt = np.zeros(shape, dtype=np.uint8)
    gt[1:4, 2, 0] = 1
    m0 = np.zeros(shape, dtype=np.uint8)
    m0[1, 2, 0] = 1
    authorized = np.zeros(shape, dtype=np.uint8)
    authorized[2:4, 2, 0] = 1
    scribble = np.zeros(shape, dtype=np.uint8)
    scribble[2, 2, 0] = 1
    paths = {
        "gt": tmp_path / "gt.nii.gz",
        "m0": tmp_path / "m0.nii.gz",
        "authorized": tmp_path / "authorized.nii.gz",
        "scribble": tmp_path / "scribble.nii.gz",
        "pet": tmp_path / "pet.nii.gz",
        "ct": tmp_path / "ct.nii.gz",
        "prediction": tmp_path / "prediction.nii.gz",
    }
    for key, values in (
        ("gt", gt),
        ("m0", m0),
        ("authorized", authorized),
        ("scribble", scribble),
        ("pet", np.ones(shape, dtype=np.float32)),
        ("ct", np.zeros(shape, dtype=np.float32)),
        ("prediction", prediction.astype(np.uint8)),
    ):
        _save(paths[key], values)

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
                        "patient_id": "patient-001",
                        "partition": "val",
                        "case_ids": ["case-001"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    split_sha = sha256_file(learning_split)
    natural = tmp_path / "natural.jsonl"
    episode = {
        "case_id": "case-001",
        "patient_id": "patient-001",
        "episode_id": "episode-001",
        "partition": "val",
        "learning_split_sha256": split_sha,
        "gt_path": str(paths["gt"].resolve()),
        "gt_sha256": sha256_file(paths["gt"]),
        "m0_path": str(paths["m0"].resolve()),
        "m0_sha256": sha256_file(paths["m0"]),
        "authorized_path": str(paths["authorized"].resolve()),
        "authorized_sha256": sha256_file(paths["authorized"]),
    }
    natural.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    input_manifest = tmp_path / "input.json"
    input_payload = {
        "schema_version": "PETCT-EXTERNAL-COMPARATOR-INPUT-v1.0",
        "provenance": {"natural_episode_manifest_sha256": sha256_file(natural)},
        "records": [
            {
                "case_id": "case-001",
                "patient_id": "patient-001",
                "split": "validation",
                "fold": 0,
                "step": 1,
                "pet_path": str(paths["pet"].resolve()),
                "ct_path": str(paths["ct"].resolve()),
                "m0_path": str(paths["m0"].resolve()),
                "fg_scribble_path": str(paths["scribble"].resolve()),
                "bg_scribble_path": None,
                "original_grid_reference": str(paths["ct"].resolve()),
                "scribble_strategy": "centerline",
                "scribble_polarity": "foreground",
                "episode_id": "episode-001",
                "input_sha256": {
                    "pet": sha256_file(paths["pet"]),
                    "ct": sha256_file(paths["ct"]),
                    "m0": sha256_file(paths["m0"]),
                    "fg_scribble": sha256_file(paths["scribble"]),
                },
                "patient_split_receipt": {
                    "internal_partition": "val",
                    "learning_split_sha256": split_sha,
                },
            }
        ],
    }
    input_manifest.write_text(json.dumps(input_payload), encoding="utf-8")
    output_manifest = tmp_path / "output.json"
    output_payload = {
        "schema_version": "PETCT-EXTERNAL-COMPARATOR-OUTPUT-v1.0",
        "method_id": method,
        "output_policy": policy,
        "records": [
            {
                "case_id": "case-001",
                "patient_id": "patient-001",
                "method_id": method,
                "prediction_path": str(paths["prediction"].resolve()),
                "original_grid_reference": str(paths["ct"].resolve()),
                "prediction_semantics": "full_mask",
                "runtime_seconds": 2.5,
                "peak_gpu_memory_mib": 512.0,
                "source_checkpoint_id": "checkpoint-sha",
                "status": "complete",
                "prediction_sha256": sha256_file(paths["prediction"]),
                "output_policy": policy,
            }
        ],
    }
    output_manifest.write_text(json.dumps(output_payload), encoding="utf-8")
    experiment = tmp_path / "experiment.json"
    experiment.write_text(json.dumps({"editor": {"local_radius_mm": 0.5}}), encoding="utf-8")
    official = tmp_path / "metrics.py"
    official.write_text("# pinned unit-test metric module\n", encoding="utf-8")
    return {
        "gt": gt,
        "m0": m0,
        "authorized": authorized,
        "input": input_manifest,
        "output": output_manifest,
        "natural": natural,
        "experiment": experiment,
        "official": official,
        "learning_split": learning_split,
    }


def _evaluate(
    tmp_path: Path,
    fixture,
    *,
    method: str,
    policy: str,
    partition: str = "val",
    test_access_receipt: Path | None = None,
    run_root: Path | None = None,
):
    return evaluate_external_output(
        input_manifest=fixture["input"],
        output_manifest=fixture["output"],
        natural_episode_manifest=fixture["natural"],
        comparator_config=CONFIG,
        experiment_config=fixture["experiment"],
        learning_split=fixture["learning_split"],
        official_metrics=fixture["official"],
        partition=partition,
        test_access_receipt=test_access_receipt,
        run_root=run_root,
        method_id=method,
        output_policy=policy,
        rows_path=tmp_path / "rows.jsonl",
        summary_path=tmp_path / "summary.json",
        metric_evaluator_class=FakeMetricEvaluator,
    )


@legacy_v1_table
def test_scribbleprompt_union_is_separate_2d_add_only_table(tmp_path: Path) -> None:
    prediction = np.zeros((5, 5, 1), dtype=np.uint8)
    prediction[1:4, 2, 0] = 1
    fixture = _fixture(
        tmp_path,
        method="scribbleprompt",
        policy="union_with_m0",
        prediction=prediction,
    )
    rows, summary = _evaluate(
        tmp_path, fixture, method="scribbleprompt", policy="union_with_m0"
    )

    assert summary["status"] == "COMPLETE"
    assert summary["spatial_dimensionality"] == "2D"
    assert summary["comparison_role"] == "POSITIVE_ONLY_DIAGNOSTIC"
    assert summary["positive_only_diagnostic"] is True
    assert summary["cross_dimensional_pooling"] == "FORBIDDEN"
    assert summary["fairness_table_id"] == "EXTERNAL-SPATIAL-2D-SCRIBBLEPROMPT"
    assert rows[0]["dice"] == 1.0
    assert rows[0]["authorized_residual_recall"] == 1.0
    assert rows[0]["m0_preservation_rate"] == 1.0
    assert rows[0]["dmm"] == 0.75


@legacy_v1_table
def test_nninteractive_native_removal_is_diagnostic_not_add_only(tmp_path: Path) -> None:
    prediction = np.zeros((5, 5, 1), dtype=np.uint8)
    prediction[2:4, 2, 0] = 1  # recovers residual but removes the existing M0 voxel
    fixture = _fixture(
        tmp_path,
        method="nninteractive",
        policy="native_full_mask",
        prediction=prediction,
    )
    rows, summary = _evaluate(
        tmp_path, fixture, method="nninteractive", policy="native_full_mask"
    )

    assert summary["spatial_dimensionality"] == "3D"
    assert summary["comparison_role"] == "NATIVE_DIAGNOSTIC"
    assert summary["positive_only_diagnostic"] is False
    assert summary["fairness_table_id"] == "EXTERNAL-SPATIAL-3D-NNINTERACTIVE-EXPOSED"
    assert rows[0]["m0_preservation_rate"] == 0.0
    assert rows[0]["other_lesion_harm"] == 1.0
    assert rows[0]["m0_removed_voxels"] == 1.0


def test_union_policy_rejects_any_m0_removal(tmp_path: Path) -> None:
    prediction = np.zeros((5, 5, 1), dtype=np.uint8)
    prediction[2:4, 2, 0] = 1
    fixture = _fixture(
        tmp_path,
        method="scribbleprompt",
        policy="union_with_m0",
        prediction=prediction,
    )
    with pytest.raises(EvaluationError, match="removed M0"):
        _evaluate(tmp_path, fixture, method="scribbleprompt", policy="union_with_m0")
    assert not (tmp_path / "rows.jsonl").exists()
    assert not (tmp_path / "summary.json").exists()


@legacy_v1_table
def test_failed_outputs_stay_in_the_denominator_with_null_metrics(tmp_path: Path) -> None:
    prediction = np.zeros((5, 5, 1), dtype=np.uint8)
    fixture = _fixture(
        tmp_path,
        method="nninteractive",
        policy="union_with_m0",
        prediction=prediction,
    )
    payload = json.loads(fixture["output"].read_text(encoding="utf-8"))
    payload["records"][0].update(
        status="failed", failure_reason="synthetic failure", prediction_path="not-created.nii.gz"
    )
    fixture["output"].write_text(json.dumps(payload), encoding="utf-8")
    rows, summary = _evaluate(
        tmp_path, fixture, method="nninteractive", policy="union_with_m0"
    )

    assert summary["status"] == "INCOMPLETE_WITH_EXPLICIT_FAILURES"
    assert summary["record_count"] == 1
    assert summary["failed_count"] == 1
    assert rows[0]["dice"] is None
    assert rows[0]["failure_reason"] == "synthetic failure"
    assert summary["patient_clustered"]["dice"]["defined"] is False


def test_test_case_relabelled_as_validation_is_rejected_before_image_or_gt_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        method="nninteractive",
        policy="union_with_m0",
        prediction=np.zeros((5, 5, 1), dtype=np.uint8),
    )
    split = json.loads(fixture["learning_split"].read_text(encoding="utf-8"))
    split["case_counts"] = {"train": 0, "val": 0, "test": 1}
    split["patients"][0]["partition"] = "test"
    fixture["learning_split"].write_text(json.dumps(split), encoding="utf-8")
    split_sha = sha256_file(fixture["learning_split"])
    natural_row = json.loads(fixture["natural"].read_text(encoding="utf-8"))
    natural_row["partition"] = "test"
    natural_row["learning_split_sha256"] = split_sha
    fixture["natural"].write_text(json.dumps(natural_row) + "\n", encoding="utf-8")
    input_payload = json.loads(fixture["input"].read_text(encoding="utf-8"))
    input_payload["provenance"]["natural_episode_manifest_sha256"] = sha256_file(
        fixture["natural"]
    )
    input_payload["records"][0]["patient_split_receipt"][
        "learning_split_sha256"
    ] = split_sha
    fixture["input"].write_text(json.dumps(input_payload), encoding="utf-8")
    monkeypatch.setattr(
        "evaluate_petct_external_comparator._load_mask",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("image/GT opened before frozen split validation")
        ),
    )

    with pytest.raises(EvaluationError, match="partition differs from frozen split"):
        _evaluate(
            tmp_path,
            fixture,
            method="nninteractive",
            policy="union_with_m0",
        )


def test_test_evaluation_requires_consumed_receipt_before_manifest_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        method="nninteractive",
        policy="union_with_m0",
        prediction=np.zeros((5, 5, 1), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "evaluate_petct_external_comparator._load_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("manifest read before receipt enforcement")
        ),
    )
    with pytest.raises(EvaluationError, match="consumed test-access receipt"):
        _evaluate(
            tmp_path,
            fixture,
            method="nninteractive",
            policy="union_with_m0",
            partition="test",
            run_root=tmp_path,
        )


def test_validation_evaluation_rejects_test_receipt_before_manifest_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(
        tmp_path,
        method="nninteractive",
        policy="union_with_m0",
        prediction=np.zeros((5, 5, 1), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "evaluate_petct_external_comparator._load_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("manifest read before receipt rejection")
        ),
    )
    with pytest.raises(EvaluationError, match="rejects a test receipt"):
        _evaluate(
            tmp_path,
            fixture,
            method="nninteractive",
            policy="union_with_m0",
            test_access_receipt=tmp_path / "not-read.json",
        )
