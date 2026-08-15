from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common.petct_learning import write_bytes_bundle_exclusive  # noqa: E402
from data.build_petct_source_case_manifest import (  # noqa: E402
    LOCKED_STATE,
    MATERIALIZED_STATE,
)
from evaluation.run_petct_m0_test_baseline import (  # noqa: E402
    CLAIM_BOUNDARY,
    OUTPUT_METRIC_ROWS,
    OUTPUT_SOURCE_CASES,
    OUTPUT_SUMMARY,
    _assert_aggregate_only,
    _copy_safe_aggregate_sections,
    run_m0_test_baseline,
)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    inputs: dict[str, Path] = {}
    for name in (
        "oof_ready",
        "identity_manifest",
        "learning_split",
        "experiment_config",
        "official_metrics",
        "test_access_receipt",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        inputs[name] = path
    inputs["run_root"] = tmp_path / "formal-run"
    inputs["run_root"].mkdir()
    inputs["ledger_root"] = tmp_path / "global-ledger"
    inputs["ledger_root"].mkdir()
    return inputs


def _identity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition, count in (("train", 415), ("val", 91)):
        for index in range(count):
            rows.append(
                {
                    "case_id": f"{partition}-case-{index:03d}",
                    "patient_id": f"{partition}-patient-{index:03d}",
                    "held_out_fold": index % 5,
                }
            )
    fold_inventory = [0] * 20 + [1] * 14 + [2] * 23 + [3] * 19 + [4] * 15
    for index, fold in enumerate(fold_inventory):
        rows.append(
            {
                "case_id": f"test-case-{index:03d}",
                "patient_id": f"test-patient-{index:03d}",
                "held_out_fold": fold,
            }
        )
    assert len(rows) == 597
    return rows


def _source_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identity in _identity_rows():
        partition = str(identity["case_id"]).split("-", 1)[0]
        row = {
            **identity,
            "partition": partition,
            "truth_materialization": (
                MATERIALIZED_STATE if partition == "test" else LOCKED_STATE
            ),
        }
        if partition == "test":
            row.update(
                {
                    "nifti_shape": [2, 2, 2],
                    "ct_bytes": 1,
                    "ct_sha256": "a" * 64,
                    "pet_bytes": 1,
                    "pet_sha256": "b" * 64,
                    "gt_bytes": 1,
                    "gt_sha256": "c" * 64,
                }
            )
        rows.append(row)
    return rows


def _cluster_summary() -> dict[str, Any]:
    return {
        "defined": True,
        "episode_count": 80.0,
        "defined_episode_count": 80.0,
        "patient_count": 50.0,
        "mean": 0.6,
        "median": 0.6,
        "std": 0.1,
        "std_defined": True,
    }


def test_receipt_is_enforced_before_test_materialization_and_evaluation(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    events: list[str] = []
    access_calls: list[dict[str, Any]] = []
    ephemeral_paths: list[Path] = []

    def enforce_access(**kwargs: Any) -> dict[str, Any]:
        events.append("access")
        access_calls.append(kwargs)
        assert kwargs["runner_script"].name == "run_petct_m0_test_baseline.py"
        assert kwargs["evaluator_script"].name == "evaluate_petct_m0_oof.py"
        assert kwargs["run_root"] == inputs["run_root"].resolve()
        assert kwargs["ledger_root"] == inputs["ledger_root"]
        assert tuple(path.name for path in kwargs["output_paths"]) == (OUTPUT_SUMMARY,)
        return {"status": "AUTHORIZED", "receipt_sha256": "d" * 64}

    def load_identity(path: Path) -> list[dict[str, Any]]:
        events.append("identity")
        assert path == inputs["identity_manifest"].resolve()
        return _identity_rows()

    def load_experiment(path: Path) -> dict[str, Any]:
        events.append("experiment")
        assert path == inputs["experiment_config"].resolve()
        return {"schema_version": "fixture"}

    def load_split(
        path: Path, identity: list[dict[str, Any]], experiment: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        events.append("split")
        assert path == inputs["learning_split"].resolve()
        assert identity == _identity_rows()
        assert experiment == {"schema_version": "fixture"}
        return {"status": "FROZEN_BEFORE_MODEL_SELECTION"}, {
            "case_counts": {"train": 415, "val": 91, "test": 91}
        }

    def materialize(
        identity: list[dict[str, Any]],
        split: dict[str, Any],
        *,
        authorized_partitions: tuple[str, ...],
    ) -> tuple[list[dict[str, Any]], str]:
        events.append("materialize")
        assert events[0] == "access"
        assert identity == _identity_rows()
        assert split["status"] == "FROZEN_BEFORE_MODEL_SELECTION"
        assert authorized_partitions == ("test",)
        return _source_rows(), "e" * 64

    def evaluate(**kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        events.append("evaluate")
        assert (
            events.index("access")
            < events.index("materialize")
            < events.index("evaluate")
        )
        assert kwargs["partitions"] == ("test",)
        assert kwargs["run_root"] == inputs["run_root"].resolve()
        materialized_rows = [
            json.loads(line)
            for line in kwargs["case_manifest"].read_text(encoding="utf-8").splitlines()
        ]
        assert materialized_rows == _source_rows()
        for row in materialized_rows[:506]:
            assert row["truth_materialization"] == LOCKED_STATE
            assert not any(key.endswith(("_bytes", "_sha256")) for key in row)
        ephemeral_paths.extend(
            (kwargs["case_manifest"], kwargs["rows_path"], kwargs["summary_path"])
        )
        validated = kwargs["test_access_validator"](
            kwargs["partitions"],
            receipt_path=kwargs["test_access_receipt"],
            experiment_config=kwargs["experiment_config"],
            learning_split=kwargs["learning_split"],
            run_root=kwargs["run_root"],
            output_paths=(kwargs["rows_path"], kwargs["summary_path"]),
        )
        assert validated["receipt_sha256"] == "d" * 64
        metric_rows = [
            {
                "case_id": row["case_id"],
                "partition": "test",
                "held_out_fold": row["held_out_fold"],
            }
            for row in _identity_rows()
            if str(row["case_id"]).startswith("test-")
        ]
        return metric_rows, {
            "schema_version": "PETCT-M0-OOF-EVALUATION-v1.1",
            "status": "COMPLETE_WITH_EXPLICIT_METRIC_ELIGIBILITY",
            "selected_partitions": ["test"],
            "source_case_count": 597,
            "case_count": 91,
            "patient_count": 57,
            "partition_case_counts": {"test": 91},
            "oof_ready_sha256": "1" * 64,
            "learning_split_sha256": "2" * 64,
            "experiment_config_sha256": "3" * 64,
            "official_metrics_sha256": "4" * 64,
            "official_autoPETV": {
                "dsc": 0.6,
                "dmm_f1_aggregated": 0.7,
                "overlap_threshold": 0.1,
                "connectivity": 18,
                "aggregation_population": "positive_gt_eligible_cases_only",
                "eligibility_rule": "GT contains at least one positive voxel",
                "eligible_case_count": 80,
                "eligible_patient_count": 50,
                "ineligible_empty_gt_case_count": 11,
                "denominators": {
                    "dsc_cases": 80,
                    "dmm_tp": 90,
                    "dmm_fp": 10,
                    "dmm_fn": 20,
                    "dmm_gt_lesions": 110,
                },
            },
            "positive_gt_patient_clustered": {
                metric: _cluster_summary()
                for metric in ("dice", "dmm_f1", "fpv_ml", "fnv_ml")
            },
            "empty_gt_false_positive_diagnostics": {
                "case_count": 11,
                "patient_count": 7,
                "false_positive_case_count": 3,
                "false_positive_patient_count": 2,
                "false_positive_lesion_count": 4,
                "prediction_volume_ml_total": 12.0,
                "patient_clustered_fpv_ml": _cluster_summary(),
                "patient_clustered_prediction_volume_ml": _cluster_summary(),
                "official_dice_and_dmm_policy": (
                    "undefined for empty GT and serialized as JSON null; false "
                    "positives remain explicit diagnostics"
                ),
            },
            "case_manifest_sha256": "e" * 64,
            "truth_binding_sha256": {
                row["case_id"]: "f" * 64 for row in metric_rows
            },
            "test_access": {
                "required": True,
                "consumed_receipt_sha256": "d" * 64,
                "bound_run_root": str(inputs["run_root"].resolve()),
            },
            "claim_boundary": (
                "OOF M0 quality on explicitly selected frozen learning partitions "
                "only; not evidence that intent or correction works"
            ),
        }

    def publish(payloads: dict[Path, bytes]) -> None:
        events.append("bundle")
        write_bytes_bundle_exclusive(payloads)

    summary = run_m0_test_baseline(
        partition="test",
        **inputs,
        access_enforcer=enforce_access,
        identity_loader=load_identity,
        experiment_loader=load_experiment,
        split_loader=load_split,
        materializer=materialize,
        evaluator=evaluate,
        bundle_writer=publish,
    )

    assert events == [
        "access",
        "identity",
        "experiment",
        "split",
        "materialize",
        "evaluate",
        "bundle",
    ]
    assert len(access_calls) == 1
    assert summary["claim_boundary"] == CLAIM_BOUNDARY
    assert summary["baseline_definition"] == {
        "prediction_source": "existing committed OOF_READY",
        "checkpoint_policy": "exactly one patient-excluded held-out-fold checkpoint per case",
        "five_fold_ensemble": False,
        "p2t_evidence": False,
        "editor_evidence": False,
    }
    assert summary["source_materialization"] == {
        "authorized_partitions": ["test"],
        "materialized_test_case_count": 91,
        "locked_unread_train_val_case_count": 506,
    }
    assert summary["held_out_fold_case_counts"] == {
        "0": 20,
        "1": 14,
        "2": 23,
        "3": 19,
        "4": 15,
    }
    assert summary["test_access"]["stage"] == "M0_BASELINE_ONLY"
    assert "truth_binding_sha256" not in summary
    assert "case_manifest_sha256" not in summary
    assert not (inputs["run_root"] / OUTPUT_SOURCE_CASES).exists()
    assert not (inputs["run_root"] / OUTPUT_METRIC_ROWS).exists()
    assert ephemeral_paths and all(not path.exists() for path in ephemeral_paths)
    saved_summary = json.loads(
        (inputs["run_root"] / OUTPUT_SUMMARY).read_text(encoding="utf-8")
    )
    assert saved_summary == summary
    saved_text = json.dumps(saved_summary, ensure_ascii=False, sort_keys=True)
    assert "test-case-" not in saved_text
    assert "test-patient-" not in saved_text


def test_aggregate_publication_rejects_nested_identity_path_and_extra_metric_keys() -> None:
    with pytest.raises(RuntimeError, match="forbidden key: case_id"):
        _assert_aggregate_only({"nested": {"case_id": "case-001"}})
    with pytest.raises(RuntimeError, match="absolute path"):
        _assert_aggregate_only({"nested": {"note": "/secret/test/path"}})
    with pytest.raises(RuntimeError, match="forbidden sequence"):
        _assert_aggregate_only(
            {
                "official_autoPETV": {
                    "aggregation_population": [0.1] * 91,
                }
            }
        )

    official = {
        "dsc": 0.6,
        "dmm_f1_aggregated": 0.7,
        "overlap_threshold": 0.1,
        "connectivity": 18,
        "aggregation_population": "positive_gt_eligible_cases_only",
        "eligibility_rule": "GT contains at least one positive voxel",
        "eligible_case_count": 80,
        "eligible_patient_count": 50,
        "ineligible_empty_gt_case_count": 11,
        "denominators": {
            "dsc_cases": 80,
            "dmm_tp": 90,
            "dmm_fp": 10,
            "dmm_fn": 20,
            "dmm_gt_lesions": 110,
        },
        "case_id": "leak",
    }
    raw = {
        key: None
        for key in (
            "schema_version",
            "status",
            "selected_partitions",
            "test_access",
            "source_case_count",
            "case_count",
            "patient_count",
            "partition_case_counts",
            "oof_ready_sha256",
            "case_manifest_sha256",
            "truth_binding_sha256",
            "learning_split_sha256",
            "experiment_config_sha256",
            "official_metrics_sha256",
            "official_autoPETV",
            "positive_gt_patient_clustered",
            "empty_gt_false_positive_diagnostics",
            "claim_boundary",
        )
    }
    raw["official_autoPETV"] = official
    with pytest.raises(RuntimeError, match="aggregate shape is invalid"):
        _copy_safe_aggregate_sections(raw)


def test_denied_receipt_prevents_every_loader_materializer_and_evaluator(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    events: list[str] = []

    def deny(**_kwargs: Any) -> dict[str, Any]:
        events.append("access")
        raise RuntimeError("receipt denied")

    def forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("test truth work ran before receipt validation")

    with pytest.raises(RuntimeError, match="receipt denied"):
        run_m0_test_baseline(
            partition="test",
            **inputs,
            access_enforcer=deny,
            identity_loader=forbidden,
            experiment_loader=forbidden,
            split_loader=forbidden,
            materializer=forbidden,
            evaluator=forbidden,
        )
    assert events == ["access"]
    assert not any(
        (inputs["run_root"] / name).exists()
        for name in (OUTPUT_SOURCE_CASES, OUTPUT_METRIC_ROWS, OUTPUT_SUMMARY)
    )


def test_rejects_non_test_partition_before_receipt_access(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)

    def forbidden(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("receipt must not be touched for an invalid partition")

    with pytest.raises(RuntimeError, match="exact partition 'test'"):
        run_m0_test_baseline(
            partition="val",
            **inputs,
            access_enforcer=forbidden,
        )


def test_no_clobber_fails_before_receipt_consumption(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    existing = inputs["run_root"] / OUTPUT_SUMMARY
    existing.write_text("keep\n", encoding="utf-8")

    def forbidden(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("receipt must not be consumed for a clobbering run")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_m0_test_baseline(
            partition="test",
            **inputs,
            access_enforcer=forbidden,
        )
    assert existing.read_text(encoding="utf-8") == "keep\n"
