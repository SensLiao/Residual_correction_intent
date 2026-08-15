from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
for directory in (
    SCRIPTS,
    SCRIPTS / "common",
    SCRIPTS / "data",
    SCRIPTS / "baseline",
    SCRIPTS / "p2t",
    SCRIPTS / "orchestration",
):
    sys.path.insert(0, str(directory))

from common.petct_learning import (  # noqa: E402
    EDITOR_CHECKPOINT_SCHEMA,
    P2T_CHECKPOINT_SCHEMA,
    P2T_METRICS_SCHEMA,
)
from orchestration.validate_petct_route_a_receipt_pipeline import (  # noqa: E402
    CONTROLLED_STAGE_ORDER,
    GENERATION_STAGE_ORDER,
    MATCHED_STATE_SCHEMA,
    PIPELINE_RECEIPT_SCHEMA,
    _canonical_json_hash,
    _record,
    _regular,
    _require_metric_hash_inventory,
    _validate_confirmatory,
    _validate_confirmatory_input_inventory,
    _validate_embedded_file_records,
    _validate_editor_metric_leaf_artifacts,
    _validate_episode_data_ready,
    _validate_frozen_checkpoint_subset,
    _validate_p2t_metric_artifact_pairs,
    _validate_upstream_receipt,
    validate_controlled_episode_rows,
    validate_editor_metric_receipts,
    validate_m0_evaluation,
    validate_pipeline,
    validate_p2t_metric_receipts,
    validate_robustness_all_rows,
)
import orchestration.validate_petct_route_a_receipt_pipeline as pipeline_module  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _bucket(cases: list[str], patients: list[str]) -> dict[str, object]:
    return {
        "case_count": len(cases),
        "patient_count": len(patients),
        "case_ids": cases,
        "patient_ids": patients,
        "case_ids_sha256": _canonical_json_hash(cases),
        "patient_ids_sha256": _canonical_json_hash(patients),
    }


def _controlled_rows():
    base = {
        "schema_version": MATCHED_STATE_SCHEMA,
        "lane": "controlled_p2t",
        "matched_state_group_id": "matched-1",
        "case_id": "case-1",
        "patient_id": "patient-1",
        "partition": "train",
        "strategy": "centerline",
        "shared_physical_scribble_sha256": "a" * 64,
        "coordinates_xyz": [[4, 5, 6]],
        "experiment_config_sha256": "b" * 64,
        "learning_split_sha256": "c" * 64,
        "scribble_generation": {"stage_order": list(CONTROLLED_STAGE_ORDER)},
    }
    return [
        {
            **base,
            "episode_id": f"episode-{index}",
            "goal": goal,
            "m0_sha256": str((index + 1) // 2) * 64,
        }
        for index, goal in enumerate(
            [
                "ADD_SAME_LOCAL",
                "REMOVE_SAME_LOCAL",
                "ADD_SAME_COMPLETE",
                "REMOVE_SAME_COMPLETE",
                "ADD_NEW_COMPLETE",
                "REMOVE_NEW_COMPLETE",
            ],
            start=1,
        )
    ]


def test_controlled_receipt_requires_one_shared_scribble_and_three_distinct_m0() -> None:
    result = validate_controlled_episode_rows(
        _controlled_rows(),
        config_sha256="b" * 64,
        split_sha256="c" * 64,
        case_to_partition={"case-1": "train"},
    )

    assert result["groups"] == 1
    assert result["episodes"] == 6


def test_controlled_receipt_rejects_changed_scribble_within_triplet() -> None:
    rows = _controlled_rows()
    rows[-1]["coordinates_xyz"] = [[9, 9, 9]]

    with pytest.raises(RuntimeError, match="shared field coordinates_xyz"):
        validate_controlled_episode_rows(
            rows,
            config_sha256="b" * 64,
            split_sha256="c" * 64,
            case_to_partition={"case-1": "train"},
        )


def _config():
    return {
        "p2t": {
            "training": {"seeds": [11, 22]},
            "primary_architecture_id": "simple_signed_scribble_state_pool_v2",
            "simple_first_input_arms": ["full", "no_M0"],
        },
        "editor": {
            "conditions": ["scribble_plus_intent", "same_weight_NULL"],
            "training_conditions": ["scribble_plus_intent"],
            "training": {"seeds": [11, 22]},
            "primary_architecture_id": "simple_operation_conditioned_residual_unet_v2",
            "fusion_plan": {
                "primary": "simple structured conditioning",
                "deferred_ablations": ["FiLM", "concat", "gated", "intent-image cross_attention"],
            },
        },
    }


def test_canonical_editor_freeze_requires_four_contracts_by_three_seeds() -> None:
    config = json.loads(
        (SCRIPTS.parent / "configs" / "petct_route_a_experiment.json").read_text(
            encoding="utf-8"
        )
    )
    roles = pipeline_module._expected_frozen_checkpoint_roles(config, "editor")
    assert len(roles) == 12
    assert all(":oracle_slots:" not in role for role in roles)
    assert all(":predicted_slots:" not in role for role in roles)
    assert sum(":scribble_plus_intent:" in role for role in roles) == 3


def test_episode_data_ready_revalidates_attempt_denominator_and_output_tree(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    inputs = {}
    for name in ("residual_manifest", "residual_ready", "oof_ready", "config", "split"):
        path = run_root / f"{name}.json"
        path.write_text(f"{name}\n", encoding="utf-8")
        inputs[name] = path
    manifest = run_root / "episodes.jsonl"
    exclusions = run_root / "exclusions.jsonl"
    manifest.write_text(
        json.dumps({"attempt_id": "attempt-generated"}) + "\n", encoding="utf-8"
    )
    exclusions.write_text(
        json.dumps({"attempt_id": "attempt-excluded", "reason": "NO_VALID_PROMPT"})
        + "\n",
        encoding="utf-8",
    )
    visible = run_root / "visible"
    visible.mkdir()
    leaf = visible / "episode.json"
    leaf.write_text("{}\n", encoding="utf-8")
    entries = [
        {
            "path": "episode.json",
            "sha256": _sha(leaf),
            "bytes": leaf.stat().st_size,
        }
    ]
    tree = {
        "path": str(visible.resolve()),
        "file_count": 1,
        "bytes": leaf.stat().st_size,
        "tree_sha256": _canonical_json_hash(entries),
    }
    ready = {
        "schema_version": "PETCT-SCRIBBLE-DATA-READY-v1.0",
        "status": "PASS",
        "phase": "OFFICIAL_FN_SCRIBBLE_EPISODE_MATERIALIZATION",
        "lane": "natural",
        "strategy_mode": "primary",
        "selected_partitions": ["train", "val"],
        "inputs": {
            "residual_manifest": _full_record(inputs["residual_manifest"]),
            "residual_ready": _full_record(inputs["residual_ready"]),
            "oof_ready": _full_record(inputs["oof_ready"]),
            "experiment_config": _full_record(inputs["config"]),
            "learning_split": _full_record(inputs["split"]),
        },
        "outputs": {
            "manifest": _full_record(manifest),
            "exclusions": _full_record(exclusions),
            "visible": tree,
        },
        "cohort": {
            "source": _bucket(["case-a", "case-b"], ["patient-a", "patient-b"]),
            "selected_source": _bucket(
                ["case-a", "case-b"], ["patient-a", "patient-b"]
            ),
            "eligible": _bucket(["case-a"], ["patient-a"]),
            "excluded": _bucket(["case-b"], ["patient-b"]),
        },
        "attempts": {
            "requested_count": 2,
            "requested_ids": ["attempt-excluded", "attempt-generated"],
            "requested_ids_sha256": _canonical_json_hash(
                ["attempt-excluded", "attempt-generated"]
            ),
            "generated_count": 1,
            "generated_ids": ["attempt-generated"],
            "generated_ids_sha256": _canonical_json_hash(["attempt-generated"]),
            "excluded_count": 1,
            "excluded_ids": ["attempt-excluded"],
            "excluded_ids_sha256": _canonical_json_hash(["attempt-excluded"]),
            "episodes": 1,
        },
        "exclusions_by_reason": {
            "NO_VALID_PROMPT": {
                "count": 1,
                "attempt_ids": ["attempt-excluded"],
                "attempt_ids_sha256": _canonical_json_hash(["attempt-excluded"]),
            }
        },
    }
    ready["binding_sha256"] = _canonical_json_hash(ready)
    ready_path = run_root / "NATURAL_PRIMARY_DATA_READY.json"
    ready_path.write_text(json.dumps(ready) + "\n", encoding="utf-8")

    validated = _validate_episode_data_ready(
        ready_path,
        schema_version="PETCT-SCRIBBLE-DATA-READY-v1.0",
        phase="OFFICIAL_FN_SCRIBBLE_EPISODE_MATERIALIZATION",
        lane="natural",
        strategy_mode="primary",
        selected_partitions={"train", "val"},
        manifest_path=manifest,
        run_root=run_root,
        expected_input_files={
            "residual_manifest": inputs["residual_manifest"],
            "residual_ready": inputs["residual_ready"],
            "oof_ready": inputs["oof_ready"],
            "experiment_config": inputs["config"],
            "learning_split": inputs["split"],
        },
        goals_per_generated_attempt=1,
    )
    assert validated["validated_attempts"] == {
        "requested": 2,
        "generated": 1,
        "excluded": 1,
    }

    exclusions.write_text(
        json.dumps({"attempt_id": "attempt-excluded", "reason": "CHANGED"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="content binding mismatch"):
        _validate_episode_data_ready(
            ready_path,
            schema_version="PETCT-SCRIBBLE-DATA-READY-v1.0",
            phase="OFFICIAL_FN_SCRIBBLE_EPISODE_MATERIALIZATION",
            lane="natural",
            strategy_mode="primary",
            selected_partitions={"train", "val"},
            manifest_path=manifest,
            run_root=run_root,
            expected_input_files={
                "residual_manifest": inputs["residual_manifest"],
                "residual_ready": inputs["residual_ready"],
                "oof_ready": inputs["oof_ready"],
                "experiment_config": inputs["config"],
                "learning_split": inputs["split"],
            },
            goals_per_generated_attempt=1,
        )


def test_p2t_result_gate_requires_every_seed_by_ablation_cell() -> None:
    training_manifest_sha256_by_run_cell = {
        (seed, arm): "e" * 64
        for seed in (11, 22)
        for arm in ("full", "no_M0")
    }
    documents = [
        {
            "schema_version": P2T_METRICS_SCHEMA,
            "experiment_config_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "training_manifest_sha256": "e" * 64,
            "inference_manifest_sha256": "b" * 64,
            "partition": "test",
            "learning_split_sha256": "c" * 64,
            "test_access_receipt_sha256": "d" * 64,
            "checkpoint_seed": seed,
            "input_ablation": arm,
        }
        for seed in (11, 22)
        for arm in ("full", "no_M0")
    ]
    assert validate_p2t_metric_receipts(
        documents,
        config=_config(),
        config_sha256="a" * 64,
        inference_manifest_sha256="b" * 64,
        training_manifest_sha256_by_run_cell=(
            training_manifest_sha256_by_run_cell
        ),
        learning_split_sha256="c" * 64,
        partition="test",
        test_access_receipt_sha256="d" * 64,
    )["metric_receipts"] == 4

    with pytest.raises(RuntimeError, match="every frozen seed x ablation"):
        validate_p2t_metric_receipts(
            documents[:-1],
            config=_config(),
            config_sha256="a" * 64,
            inference_manifest_sha256="b" * 64,
            training_manifest_sha256_by_run_cell=(
                training_manifest_sha256_by_run_cell
            ),
            learning_split_sha256="c" * 64,
            partition="test",
            test_access_receipt_sha256="d" * 64,
        )

    drifted = [dict(document) for document in documents]
    drifted[0]["training_manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="checkpoint training manifest"):
        validate_p2t_metric_receipts(
            drifted,
            config=_config(),
            config_sha256="a" * 64,
            inference_manifest_sha256="b" * 64,
            training_manifest_sha256_by_run_cell=(
                training_manifest_sha256_by_run_cell
            ),
            learning_split_sha256="c" * 64,
            partition="test",
            test_access_receipt_sha256="d" * 64,
        )

    wrong_inference = [dict(document) for document in documents]
    wrong_inference[0]["inference_manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="current inference manifest"):
        validate_p2t_metric_receipts(
            wrong_inference,
            config=_config(),
            config_sha256="a" * 64,
            inference_manifest_sha256="b" * 64,
            training_manifest_sha256_by_run_cell=(
                training_manifest_sha256_by_run_cell
            ),
            learning_split_sha256="c" * 64,
            partition="test",
            test_access_receipt_sha256="d" * 64,
        )

    incomplete_training_bindings = dict(training_manifest_sha256_by_run_cell)
    incomplete_training_bindings.pop((22, "no_M0"))
    with pytest.raises(RuntimeError, match="checkpoint training manifests"):
        validate_p2t_metric_receipts(
            documents,
            config=_config(),
            config_sha256="a" * 64,
            inference_manifest_sha256="b" * 64,
            training_manifest_sha256_by_run_cell=incomplete_training_bindings,
            learning_split_sha256="c" * 64,
            partition="test",
            test_access_receipt_sha256="d" * 64,
        )


def test_editor_result_gate_requires_every_condition_by_seed_cell() -> None:
    documents = []
    for condition_index, condition in enumerate(
        ("scribble_plus_intent", "same_weight_NULL"), start=1
    ):
        for seed_index in (1, 2):
            documents.append(
                {
                    "schema_version": "PETCT-BIDIRECTIONAL-CORRECTION-METRICS-v2.0",
                    "condition": condition,
                    "learning_manifest_sha256": "d" * 64,
                    "checkpoint_sha256": f"{condition_index}{seed_index}" * 32,
                    "experiment_config_sha256": "e" * 64,
                    "learning_split_sha256": "f" * 64,
                    "partition": "val",
                    "test_access_receipt_sha256": None,
                    "architecture_id": "simple_operation_conditioned_residual_unet_v2",
                    "analysis_role": "PRIMARY_OR_ABLATION",
                    "parameter_count": 100,
                }
            )
    assert validate_editor_metric_receipts(
        documents,
        config=_config(),
        config_sha256="e" * 64,
        manifest_sha256="d" * 64,
        learning_split_sha256="f" * 64,
        partition="val",
        test_access_receipt_sha256=None,
    )["metric_receipts"] == 4


def _scribble_config():
    return {
        "scribble": {
            "primary_strategy_mode": "primary",
            "primary_assignment": "stable-patient-hash",
            "primary_strategy_salt": "PETCT-PRIMARY-v1",
            "strategies": ["centerline", "random", "boundary"],
        }
    }


def test_primary_strategy_coverage_rejects_wrong_stable_hash_assignment() -> None:
    rows = _controlled_rows()
    from data.build_petct_scribble_episode import assign_scribble_strategy

    expected = assign_scribble_strategy("patient-1", salt="PETCT-PRIMARY-v1")
    for row in rows:
        row.update(
            strategy=expected,
            strategy_mode="primary",
            strategy_assignment="stable-patient-hash",
            strategy_salt="PETCT-PRIMARY-v1",
        )
        row["scribble_generation"].update(
            strategy_mode="primary",
            strategy_assignment="stable-patient-hash",
            strategy_salt="PETCT-PRIMARY-v1",
            selected_strategy=expected,
        )
    result = validate_controlled_episode_rows(
        rows,
        config_sha256="b" * 64,
        split_sha256="c" * 64,
        case_to_partition={"case-1": "train"},
        config=_scribble_config(),
    )
    assert result["strategy_coverage"]["all_rows_match_expected_assignment"] is True
    rows[0]["strategy_salt"] = "wrong"
    with pytest.raises(RuntimeError, match="stable-patient-hash"):
        validate_controlled_episode_rows(
            rows,
            config_sha256="b" * 64,
            split_sha256="c" * 64,
            case_to_partition={"case-1": "train"},
            config=_scribble_config(),
        )


def test_m0_evaluation_validates_null_definedness_and_exact_scope() -> None:
    source = [
        {"case_id": "c1", "patient_id": "P1", "held_out_fold": 0},
        {"case_id": "c2", "patient_id": "P2", "held_out_fold": 1},
    ]
    split = {"case_to_partition": {"c1": "train", "c2": "val"}}
    oof = {
        "ready_sha256": "a" * 64,
        "cases": {
            "c1": {"patient_id": "P1", "held_out_fold": 0},
            "c2": {"patient_id": "P2", "held_out_fold": 1},
        },
    }
    rows = [
        {
            "case_id": "c1",
            "patient_id": "p1",
            "held_out_fold": 0,
            "partition": "train",
            "gt_positive": True,
            "official_metric_eligible": True,
            "official_metric_ineligibility_reason": None,
            "tp": 1,
            "fp": 0,
            "fn": 0,
            "gt_voxel_count": 4,
            "prediction_voxel_count": 4,
            "fpv_ml": 0.0,
            "dice": 1.0,
            "dmm_f1": 1.0,
            "fnv_ml": 0.0,
            "empty_gt_false_positive": None,
            "empty_gt_prediction_volume_ml": None,
        },
        {
            "case_id": "c2",
            "patient_id": "p2",
            "held_out_fold": 1,
            "partition": "val",
            "gt_positive": False,
            "official_metric_eligible": False,
            "official_metric_ineligibility_reason": "EMPTY_GT",
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "gt_voxel_count": 0,
            "prediction_voxel_count": 0,
            "fpv_ml": 0.0,
            "dice": None,
            "dmm_f1": None,
            "fnv_ml": None,
            "empty_gt_false_positive": False,
            "empty_gt_prediction_volume_ml": 0.0,
        },
    ]
    summary = {
        "schema_version": "PETCT-M0-OOF-EVALUATION-v1.1",
        "status": "COMPLETE_WITH_EXPLICIT_METRIC_ELIGIBILITY",
        "selected_partitions": ["train", "val"],
        "test_access": {
            "required": False,
            "consumed_receipt_sha256": None,
            "bound_run_root": None,
        },
        "source_case_count": 2,
        "case_count": 2,
        "patient_count": 2,
        "partition_case_counts": {"train": 1, "val": 1},
        "oof_ready_sha256": "a" * 64,
        "case_manifest_sha256": "b" * 64,
        "learning_split_sha256": "c" * 64,
        "experiment_config_sha256": "d" * 64,
        "official_metrics_sha256": "e" * 64,
        "official_autoPETV": {
            "dsc": 1.0,
            "dmm_f1_aggregated": 1.0,
            "eligible_case_count": 1,
            "ineligible_empty_gt_case_count": 1,
            "overlap_threshold": 0.1,
            "connectivity": 18,
        },
        "empty_gt_false_positive_diagnostics": {"case_count": 1},
    }
    result = validate_m0_evaluation(
        rows=rows,
        summary=summary,
        source_rows=source,
        split=split,
        oof=oof,
        selected_partitions=["train", "val"],
        config_sha256="d" * 64,
        case_manifest_sha256="b" * 64,
        learning_split_sha256="c" * 64,
        official_metrics_sha256="e" * 64,
        evaluation_partition="val",
        test_access_receipt_sha256=None,
        run_root=Path("C:/formal-run"),
    )
    assert result["positive_gt_defined_case_count"] == 1
    assert result["empty_gt_null_case_count"] == 1
    rows[0]["dice"] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_m0_evaluation(
            rows=rows,
            summary=summary,
            source_rows=source,
            split=split,
            oof=oof,
            selected_partitions=["train", "val"],
            config_sha256="d" * 64,
            case_manifest_sha256="b" * 64,
            learning_split_sha256="c" * 64,
            official_metrics_sha256="e" * 64,
            evaluation_partition="val",
            test_access_receipt_sha256=None,
            run_root=Path("C:/formal-run"),
        )


def test_robustness_all_requires_three_strategies_and_exact_primary_correspondence(
    tmp_path: Path,
) -> None:
    visible = tmp_path / "visible.json"
    evaluation = tmp_path / "evaluation.json"
    authorized = tmp_path / "authorized.nii.gz"
    visible.write_text("visible", encoding="utf-8")
    evaluation.write_text("evaluation", encoding="utf-8")
    authorized.write_bytes(b"authorized")
    base = {
        "case_id": "c1",
        "patient_id": "p1",
        "goal": "ADD_SAME_LOCAL",
        "partition": "val",
        "held_out_fold": 0,
        "m0_sha256": "a" * 64,
        "gt_sha256": "b" * 64,
        "fn_sha256": "c" * 64,
        "fn_mask_sha256": "d" * 64,
        "authorized_path": str(authorized),
        "authorized_sha256": _sha(authorized),
        "visible_document": str(visible),
        "visible_document_sha256": _sha(visible),
        "evaluation_document": str(evaluation),
        "evaluation_document_sha256": _sha(evaluation),
        "experiment_config_sha256": "f" * 64,
        "learning_split_sha256": "1" * 64,
        "m0_provenance": {"kind": "patient_excluded_oof"},
        "coordinates_xyz": [[1, 2, 3]],
    }
    from data.build_petct_scribble_episode import assign_scribble_strategy

    primary_strategy = assign_scribble_strategy("p1", salt="PETCT-PRIMARY-v1")
    primary = [
        {
            **base,
            "episode_id": f"ep-{primary_strategy}",
            "strategy": primary_strategy,
            "scribble_generation": {
                "stage_order": list(GENERATION_STAGE_ORDER),
            },
        }
    ]
    robustness = [
        {
            **base,
            "episode_id": f"ep-{strategy}",
            "strategy": strategy,
            "strategy_mode": "all",
            "strategy_assignment": "stable-patient-hash",
            "strategy_salt": "PETCT-PRIMARY-v1",
            "scribble_generation": {
                "strategy_mode": "all",
                "strategy_assignment": "stable-patient-hash",
                "strategy_salt": "PETCT-PRIMARY-v1",
                "selected_strategy": strategy,
                "stage_order": list(GENERATION_STAGE_ORDER),
            },
        }
        for strategy in ("centerline", "random", "boundary")
    ]
    result = validate_robustness_all_rows(
        primary, robustness, config=_scribble_config()
    )
    assert result["row_count"] == 3
    with pytest.raises(RuntimeError, match="exactly three strategies"):
        validate_robustness_all_rows(
            primary, robustness[:-1], config=_scribble_config()
        )

    robustness[0]["visible_document_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="visible_document hash mismatch"):
        validate_robustness_all_rows(primary, robustness, config=_scribble_config())


def test_not_supported_editor_analysis_is_still_valid_for_receipt_gate(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "editor-confirmatory.json"
    analysis.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-BIDIRECTIONAL-EDITOR-CONFIRMATORY-v2.0",
                "analysis_status": "VALID",
                "experiment_config_sha256": "a" * 64,
                "family_verdicts": {"f": "NOT_SUPPORTED"},
                "input_runs": [],
            }
        ),
        encoding="utf-8",
    )
    document, _ = _validate_confirmatory(
        analysis, kind="editor", config_sha256="a" * 64
    )
    assert document["family_verdicts"] == {"f": "NOT_SUPPORTED"}


def test_metric_hash_inventory_rejects_a_for_statistics_b_for_inventory(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    replacement.write_text("replacement\n", encoding="utf-8")
    documents = [
        {"predictions_sha256": _sha(first)},
        {"predictions_sha256": _sha(replacement)},
    ]
    with pytest.raises(RuntimeError, match="exactly close"):
        _require_metric_hash_inventory(
            documents,
            field="predictions_sha256",
            paths=[first, second],
            label="P2T counterexample",
        )


def test_p2t_artifact_pairs_close_training_inference_and_learning_hashes(
    tmp_path: Path,
) -> None:
    training_sha = "a" * 64
    inference_sha = "b" * 64
    shared = {
        "episode_id": "episode-1",
        "checkpoint_sha256": "c" * 64,
        "input_ablation": "full",
        "partition": "test",
        "experiment_config_sha256": "d" * 64,
        "learning_split_sha256": "e" * 64,
        "test_access_receipt_sha256": "f" * 64,
        "learning_manifest_sha256": inference_sha,
        "training_manifest_sha256": training_sha,
    }
    predictions = tmp_path / "predictions.jsonl"
    paired = tmp_path / "paired.jsonl"
    predictions.write_text(
        json.dumps({**shared, "inference_manifest_sha256": inference_sha}) + "\n",
        encoding="utf-8",
    )
    paired.write_text(
        json.dumps({**shared, "checkpoint_seed": 11}) + "\n",
        encoding="utf-8",
    )
    metric = {
        **{key: value for key, value in shared.items() if key != "episode_id"},
        "checkpoint_seed": 11,
        "manifest_sha256": inference_sha,
        "training_manifest_sha256": training_sha,
        "inference_manifest_sha256": inference_sha,
        "predictions_sha256": _sha(predictions),
        "paired_evaluation_rows_sha256": _sha(paired),
        "episode_count": 1,
        "paired_evaluation_row_count": 1,
    }
    _validate_p2t_metric_artifact_pairs(
        [metric], prediction_paths=[predictions], paired_paths=[paired]
    )

    paired.write_text(
        json.dumps({**shared, "checkpoint_seed": 11, "training_manifest_sha256": "0" * 64})
        + "\n",
        encoding="utf-8",
    )
    metric["paired_evaluation_rows_sha256"] = _sha(paired)
    with pytest.raises(RuntimeError, match="paired-row manifest provenance"):
        _validate_p2t_metric_artifact_pairs(
            [metric], prediction_paths=[predictions], paired_paths=[paired]
        )


def test_confirmatory_input_runs_must_match_formal_records_by_run_cell(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    metrics = tmp_path / "metrics.json"
    paired = tmp_path / "paired.jsonl"
    other = tmp_path / "other.json"
    for path, body in (
        (checkpoint, b"checkpoint"),
        (metrics, b"metrics"),
        (paired, b"paired"),
        (other, b"other"),
    ):
        path.write_bytes(body)
    expected = {
        (11, "full"): {
            "checkpoint": _record(checkpoint),
            "metrics": _record(metrics),
            "paired_rows": _record(paired),
        }
    }
    document = {
        "input_runs": [
            {
                "seed": 11,
                "arm": "full",
                "checkpoint": _record(checkpoint),
                "metrics": _record(other),
                "paired_rows": _record(paired),
            }
        ]
    }
    with pytest.raises(RuntimeError, match="different metrics artifact"):
        _validate_confirmatory_input_inventory(
            document, kind="P2T", expected=expected
        )


def test_upstream_receipt_revalidates_embedded_artifact_records(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("stable", encoding="utf-8")
    common = {"evaluation_partition": "val"}
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": PIPELINE_RECEIPT_SCHEMA,
                "status": "PASS",
                "target": "p2t_data",
                "common": common,
                "artifact_bindings": {"artifact": _record(artifact)},
                "upstream_receipts": {},
            }
        ),
        encoding="utf-8",
    )
    _validate_upstream_receipt(
        receipt, expected_target="p2t_data", common=common
    )
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="embedded file record changed"):
        _validate_upstream_receipt(
            receipt, expected_target="p2t_data", common=common
        )


def test_embedded_file_record_rejects_extra_or_partial_keys(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    valid = {"path": str(artifact.resolve()), "sha256": _sha(artifact)}
    _validate_embedded_file_records(valid, label="valid")

    with pytest.raises(RuntimeError, match="exactly path/sha256"):
        _validate_embedded_file_records(
            {**valid, "ignored": True}, label="extra-key"
        )
    with pytest.raises(RuntimeError, match="exactly path/sha256"):
        _validate_embedded_file_records(
            {"path": valid["path"]}, label="partial-path"
        )
    with pytest.raises(RuntimeError, match="exactly path/sha256"):
        _validate_embedded_file_records(
            {"sha256": valid["sha256"]}, label="partial-sha"
        )


def test_regular_rejects_symlink_before_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "synthetic-link.json"
    link.write_text("{}", encoding="utf-8")
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve

    def fake_is_symlink(path: Path) -> bool:
        return path == link or original_is_symlink(path)

    def fail_if_resolved_first(path: Path, *args, **kwargs):
        if path == link:
            raise AssertionError("resolve was called before the symlink check")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", fake_is_symlink)
    monkeypatch.setattr(Path, "resolve", fail_if_resolved_first)
    with pytest.raises(RuntimeError, match="non-symlink"):
        _regular(link, label="counterexample")


def test_frozen_checkpoint_subset_requires_exact_role_path_and_sha(
    tmp_path: Path,
) -> None:
    training_manifest = tmp_path / "training.jsonl"
    training_manifest.write_text("{}\n", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint = {
        "schema_version": P2T_CHECKPOINT_SCHEMA,
        "status": "TRAINED_WHEN_THIS_SCRIPT_IS_EXECUTED",
        "seed": 11,
        "seed_registry": [11],
        "architecture_id": "primary_arch",
        "input_ablation": "full",
        "arm_role": "primary",
        "checkpoint_criterion": "criterion",
        "manifest": str(training_manifest.resolve()),
        "manifest_sha256": _sha(training_manifest),
    }
    torch.save(checkpoint, checkpoint_path)
    role = "selected_checkpoint:p2t:primary_arch:full:seed11"
    frozen = {
        "role": role,
        "path": str(checkpoint_path.resolve()),
        "sha256": _sha(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "checkpoint_metadata": {
            key: checkpoint[key]
            for key in (
                "schema_version",
                "status",
                "seed",
                "seed_registry",
                "architecture_id",
                "input_ablation",
                "arm_role",
                "checkpoint_criterion",
            )
        },
        "training_manifest": {
            "role": "training_manifest",
            "path": str(training_manifest.resolve()),
            "sha256": _sha(training_manifest),
            "bytes": training_manifest.stat().st_size,
        },
    }
    bindings = {
        "selected_checkpoint_roles": [role],
        "checkpoints": [frozen],
    }
    observed = _validate_frozen_checkpoint_subset(
        bindings=bindings,
        checkpoint_paths=[checkpoint_path],
        kind="p2t",
        expected_roles={role},
    )
    assert observed == {
        role: {"path": str(checkpoint_path.resolve()), "sha256": _sha(checkpoint_path)}
    }

    moved = tmp_path / "moved.pth"
    moved.write_bytes(checkpoint_path.read_bytes())
    with pytest.raises(RuntimeError, match="path differs from final freeze"):
        _validate_frozen_checkpoint_subset(
            bindings=bindings,
            checkpoint_paths=[moved],
            kind="p2t",
            expected_roles={role},
        )


def test_test_pipeline_rejects_receipt_freeze_inventory_mismatch_before_data_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    paths = {}
    for name in (
        "experiment_config",
        "case_manifest",
        "learning_split",
        "test_access_receipt",
        "frozen_checkpoint_bindings",
    ):
        path = run_root / f"{name}.json"
        path.write_text("not-readable-scientific-data\n", encoding="utf-8")
        paths[name] = path
    freeze = tmp_path / "final-freeze.json"
    freeze.write_text("{}\n", encoding="utf-8")
    freeze_record = {
        "role": "final_development_freeze",
        "path": str(freeze.resolve()),
        "sha256": _sha(freeze),
        "bytes": freeze.stat().st_size,
    }
    inputs_path = run_root / "inputs.json"
    inputs_path.write_text(
        json.dumps(
            {
                "experiment_config": str(paths["experiment_config"]),
                "case_manifest": str(paths["case_manifest"]),
                "learning_split": str(paths["learning_split"]),
                "evaluation_partition": "test",
                "run_root": str(run_root),
                "test_access_receipt": str(paths["test_access_receipt"]),
                "frozen_checkpoint_bindings": str(
                    paths["frozen_checkpoint_bindings"]
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_consumed_receipt",
        lambda *_args, **_kwargs: {
            "receipt_sha256": "a" * 64,
            "consumption": {
                "final_development_freeze": {
                    key: freeze_record[key]
                    for key in ("path", "sha256", "bytes")
                },
                "checkpoint_inventory_sha256": "b" * 64,
            },
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "validate_frozen_checkpoint_bindings",
        lambda _path: {
            "final_development_freeze": freeze_record,
            "checkpoint_inventory_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        pipeline_module,
        "_validate_f0_binding",
        lambda _value: {
            "receipt": {"path": str(tmp_path / "F0_READY.json"), "sha256": "f" * 64},
            "source_bundle_sha256": "e" * 64,
            "environment_bundle_sha256": "d" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="different checkpoint inventories"):
        validate_pipeline(inputs_path, "p2t_data")


def test_final_editor_leaf_closure_rehashes_prediction_evaluation_and_visible(
    tmp_path: Path,
) -> None:
    visible = tmp_path / "visible.npz"
    evaluation = tmp_path / "evaluation.npz"
    prediction = tmp_path / "prediction.npz"
    visible.write_bytes(b"visible")
    evaluation.write_bytes(b"evaluation")
    prediction.write_bytes(b"prediction")
    split = tmp_path / "split.json"
    split.write_text("{}\n", encoding="utf-8")
    learning_manifest = tmp_path / "learning.jsonl"
    learning_row = {
        "episode_id": "episode-1",
        "patient_id": "patient-1",
        "partition": "val",
        "visible_npz": str(visible.resolve()),
        "visible_sha256": _sha(visible),
        "evaluation_npz": str(evaluation.resolve()),
        "evaluation_sha256": _sha(evaluation),
    }
    learning_manifest.write_text(json.dumps(learning_row) + "\n", encoding="utf-8")
    prediction_manifest = tmp_path / "predictions.jsonl"
    row = {
        "episode_id": "episode-1",
        "patient_id": "patient-1",
        "partition": "val",
        "learning_manifest": str(learning_manifest.resolve()),
        "learning_manifest_sha256": _sha(learning_manifest),
        "training_manifest": str(learning_manifest.resolve()),
        "training_manifest_sha256": _sha(learning_manifest),
        "inference_manifest": str(learning_manifest.resolve()),
        "inference_manifest_sha256": _sha(learning_manifest),
        "prediction_manifest": str(prediction_manifest.resolve()),
        "prediction_npz": str(prediction.resolve()),
        "prediction_npz_sha256": _sha(prediction),
        "visible_npz": str(visible.resolve()),
        "visible_npz_sha256": _sha(visible),
        "evaluation_npz": str(evaluation.resolve()),
        "evaluation_npz_sha256": _sha(evaluation),
    }
    prediction_manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows_path = tmp_path / "metrics.jsonl"
    rows_path.write_text(json.dumps({**row, "metric": 0.5}) + "\n", encoding="utf-8")
    checkpoint_path = tmp_path / "editor.pth"
    torch.save(
        {
            "schema_version": EDITOR_CHECKPOINT_SCHEMA,
            "status": "TRAINED_WHEN_THIS_SCRIPT_IS_EXECUTED",
            "manifest": str(learning_manifest.resolve()),
            "manifest_sha256": _sha(learning_manifest),
            "training_manifest": str(learning_manifest.resolve()),
            "training_manifest_sha256": _sha(learning_manifest),
            "learning_split": str(split.resolve()),
            "learning_split_sha256": _sha(split),
        },
        checkpoint_path,
    )
    summary = {
        "partition": "val",
        "prediction_manifest_sha256": _sha(prediction_manifest),
        "learning_split_sha256": _sha(split),
    }
    leaf = _validate_editor_metric_leaf_artifacts(
        rows_path=rows_path,
        summary=summary,
        checkpoint_path=checkpoint_path,
    )
    assert set(leaf["episodes"]["episode-1"]) == {
        "prediction_npz",
        "evaluation_npz",
        "visible_npz",
    }

    prediction.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="prediction_npz changed"):
        _validate_editor_metric_leaf_artifacts(
            rows_path=rows_path,
            summary=summary,
            checkpoint_path=checkpoint_path,
        )
