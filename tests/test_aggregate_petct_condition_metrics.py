from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "evaluation"))

from aggregate_petct_condition_metrics import (  # noqa: E402
    _decision,
    holm_adjust,
    main,
    paired_cluster_bootstrap,
    patient_cluster_permutation,
    patient_means,
)
from common.petct_learning import EDITOR_CHECKPOINT_SCHEMA  # noqa: E402


GATE_ACTIVE = "ACTIVE_AFTER_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"
GATE_BLOCKED = "BLOCKED_UNTIL_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"
# Sentinel for "omit the whole config section", which is a different failure
# path from "section present but the gate key is null".
OMIT_SECTION = object()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _shared_leaf_inputs(tmp_path: Path) -> dict[str, object]:
    leaf_root = tmp_path / "leaf-inputs"
    leaf_root.mkdir(exist_ok=True)
    learning_rows = []
    by_episode = {}
    for index in (1, 2):
        episode = f"episode-{index}"
        visible = leaf_root / f"{episode}-visible.npz"
        evaluation = leaf_root / f"{episode}-evaluation.npz"
        visible.write_bytes(f"visible-{index}".encode())
        evaluation.write_bytes(f"evaluation-{index}".encode())
        row = {
            "episode_id": episode,
            "patient_id": f"patient-{index}",
            "partition": "val",
            "visible_npz": str(visible.resolve()),
            "visible_sha256": _sha(visible),
            "evaluation_npz": str(evaluation.resolve()),
            "evaluation_sha256": _sha(evaluation),
        }
        learning_rows.append(row)
        by_episode[episode] = row
    manifest = leaf_root / "learning.jsonl"
    _write_jsonl(manifest, learning_rows)
    learning_split = leaf_root / "learning-split.json"
    learning_split.write_text('{"fixture":true}\n', encoding="utf-8")
    return {
        "manifest": manifest,
        "manifest_sha256": _sha(manifest),
        "learning_split": learning_split,
        "learning_split_sha256": _sha(learning_split),
        "by_episode": by_episode,
    }


def _write_run_rows(
    *,
    tmp_path: Path,
    run_name: str,
    rows: list[dict[str, object]],
    leaf: dict[str, object],
) -> tuple[Path, Path]:
    prediction_manifest = tmp_path / f"{run_name}-predictions.jsonl"
    by_episode = leaf["by_episode"]
    assert isinstance(by_episode, dict)
    for row in rows:
        episode = str(row["episode_id"])
        source = by_episode[episode]
        prediction = tmp_path / f"{run_name}-{episode}-prediction.npz"
        prediction.write_bytes(f"prediction-{run_name}-{episode}".encode())
        row.update(
            {
                "checkpoint_status": "TRAINED_WHEN_THIS_SCRIPT_IS_EXECUTED",
                "learning_manifest": str(Path(leaf["manifest"]).resolve()),
                "learning_manifest_sha256": leaf["manifest_sha256"],
                "training_manifest": str(Path(leaf["manifest"]).resolve()),
                "training_manifest_sha256": leaf["manifest_sha256"],
                "inference_manifest": str(Path(leaf["manifest"]).resolve()),
                "inference_manifest_sha256": leaf["manifest_sha256"],
                "prediction_manifest": str(prediction_manifest.resolve()),
                "prediction_npz": str(prediction.resolve()),
                "prediction_npz_sha256": _sha(prediction),
                "visible_npz": source["visible_npz"],
                "visible_npz_sha256": source["visible_sha256"],
                "evaluation_npz": source["evaluation_npz"],
                "evaluation_npz_sha256": source["evaluation_sha256"],
            }
        )
    _write_jsonl(prediction_manifest, rows)
    rows_path = tmp_path / f"{run_name}.jsonl"
    _write_jsonl(rows_path, rows)
    return rows_path, prediction_manifest


def test_patient_means_prevent_episode_pseudoreplication() -> None:
    rows = [
        {"patient_id": "p1", "metric": 0.0},
        {"patient_id": "p1", "metric": 1.0},
        {"patient_id": "p2", "metric": 1.0},
    ]
    assert patient_means(rows, "metric") == {"p1": 0.5, "p2": 1.0}


def test_patient_cluster_bootstrap_and_swap_are_deterministic() -> None:
    treatment = {"p1": 0.7, "p2": 0.8, "p3": 0.9}
    baseline = {"p1": 0.5, "p2": 0.7, "p3": 0.6}
    first = paired_cluster_bootstrap(
        treatment, baseline, seed=42, samples=500, alpha=0.05
    )
    second = paired_cluster_bootstrap(
        treatment, baseline, seed=42, samples=500, alpha=0.05
    )
    assert first == second
    delta = {patient: treatment[patient] - baseline[patient] for patient in treatment}
    permutation = patient_cluster_permutation(
        delta,
        seed=19,
        samples=500,
        alternative="greater",
        null_margin=0.0,
    )
    assert permutation["permutation_unit"].startswith("whole patient cluster")
    assert permutation["null_transform"] == (
        "sign_flip_of_patient_delta_minus_null_margin"
    )
    assert permutation["exchangeability_unit"] == "patient_cluster"
    assert permutation["same_swap_across_registered_seeds"] is True
    assert 0 < permutation["p_value"] <= 1


@pytest.mark.parametrize(
    ("alternative", "margin"),
    [("noninferiority", -0.05), ("noninferiority-upper", 25.0)],
)
def test_noninferiority_margin_shift_is_algebraically_equivalent_to_zero_null(
    alternative: str, margin: float
) -> None:
    patient_delta = {"p1": margin + 0.1, "p2": margin - 0.2, "p3": margin + 0.3}
    shifted = {
        patient: value - margin for patient, value in patient_delta.items()
    }
    with_margin = patient_cluster_permutation(
        patient_delta,
        seed=91,
        samples=500,
        alternative=alternative,
        null_margin=margin,
    )
    at_zero = patient_cluster_permutation(
        shifted,
        seed=91,
        samples=500,
        alternative=alternative,
        null_margin=0.0,
    )
    assert with_margin["observed_minus_margin"] == pytest.approx(
        at_zero["observed_minus_margin"]
    )
    assert with_margin["p_value"] == at_zero["p_value"]
    assert with_margin["margin_adjustment"] == {
        "formula": "patient_delta - null_margin",
        "null_margin": margin,
    }


def test_holm_adjust_is_applied_monotonically() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_explicit_decision_rules_cover_effect_and_both_ni_directions() -> None:
    assert _decision(
        rule="holm_p_lt_alpha_and_point_estimate_ge_threshold",
        p_value_holm=0.01,
        alpha=0.05,
        estimate=0.02,
        threshold=0.02,
    )
    assert _decision(
        rule="holm_p_lt_alpha_and_point_estimate_gt_lower_margin",
        p_value_holm=0.01,
        alpha=0.05,
        estimate=-0.049,
        threshold=-0.05,
    )
    assert _decision(
        rule="holm_p_lt_alpha_and_point_estimate_lt_upper_margin",
        p_value_holm=0.01,
        alpha=0.05,
        estimate=24.9,
        threshold=25.0,
    )


def _fixture(
    tmp_path: Path,
    *,
    gate_p2t=GATE_ACTIVE,
    gate_statistics=GATE_ACTIVE,
):
    seeds = [11, 22, 33]
    config = {
        "p2t": {"confirmatory_execution_gate": gate_p2t},
        "editor": {"training": {"seeds": seeds}},
        "statistics": {
            "multiplicity": "Holm within each frozen confirmatory family",
            "alpha": 0.1,
            "bootstrap_samples": 100,
            "permutation_samples": 200,
            "bootstrap_seed": 11,
            "permutation_seed": 101,
            "confirmatory_execution_gate": gate_statistics,
            "confirmatory_same_weight_families": {"family": ["a", "b"]},
            "effect_thresholds": {"gain": 0.50},
            "confirmatory_contrasts": [
                {
                    "family": "family",
                    "treatment": "a",
                    "comparator": "b",
                    "metric": "prompt_distal_recall",
                    "alternative": "greater",
                    "threshold_ref": "gain",
                    "decision_rule": "holm_p_lt_alpha_and_point_estimate_ge_threshold",
                }
            ],
        },
    }
    if gate_p2t is OMIT_SECTION:
        del config["p2t"]
    if gate_statistics is OMIT_SECTION:
        del config["statistics"]["confirmatory_execution_gate"]
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    config_sha = _sha(config_path)
    leaf = _shared_leaf_inputs(tmp_path)
    runs: list[list[str]] = []
    # Each seed has the same two patients.  Treatment improves by only 0.1, so
    # the statistical analysis remains valid while the frozen 0.5 effect gate
    # returns NOT_SUPPORTED.
    for seed_index, seed in enumerate(seeds):
        for condition, values in {"a": [0.7, 0.8], "b": [0.6, 0.7]}.items():
            checkpoint = tmp_path / f"{condition}-{seed}.pth"
            torch.save(
                {
                    "schema_version": EDITOR_CHECKPOINT_SCHEMA,
                    "seed": seed,
                    "seed_registry": seeds,
                    "condition": condition,
                    "experiment_config_sha256": config_sha,
                    "status": "TRAINED_WHEN_THIS_SCRIPT_IS_EXECUTED",
                    "manifest": str(Path(leaf["manifest"]).resolve()),
                    "manifest_sha256": leaf["manifest_sha256"],
                    "training_manifest": str(Path(leaf["manifest"]).resolve()),
                    "training_manifest_sha256": leaf["manifest_sha256"],
                    "learning_split": str(
                        Path(leaf["learning_split"]).resolve()
                    ),
                    "learning_split_sha256": leaf["learning_split_sha256"],
                },
                checkpoint,
            )
            checkpoint_sha = _sha(checkpoint)
            rows = []
            for index, value in enumerate(values, start=1):
                rows.append(
                    {
                        "episode_id": f"episode-{index}",
                        "patient_id": f"patient-{index}",
                        "condition": condition,
                        "checkpoint_sha256": checkpoint_sha,
                        "experiment_config_sha256": config_sha,
                        "learning_split_sha256": leaf[
                            "learning_split_sha256"
                        ],
                        "partition": "val",
                        "test_access_receipt_sha256": None,
                        "threshold": 0.5,
                        "distal_radius_mm": 15.0,
                        "metric_grid": "full-grid",
                        "prompt_distal_recall": value + seed_index * 0.01,
                    }
                )
            rows_path, prediction_manifest = _write_run_rows(
                tmp_path=tmp_path,
                run_name=f"{condition}-{seed}",
                rows=rows,
                leaf=leaf,
            )
            summary = {
                "schema_version": "PETCT-BIDIRECTIONAL-CORRECTION-METRICS-v2.0",
                "condition": condition,
                "checkpoint_sha256": checkpoint_sha,
                "experiment_config_sha256": config_sha,
                "learning_manifest_sha256": leaf["manifest_sha256"],
                "learning_split_sha256": leaf["learning_split_sha256"],
                "partition": "val",
                "test_access_receipt_sha256": None,
                "threshold": 0.5,
                "distal_radius_mm": 15.0,
                "metric_grid": "full-grid",
                "metric_rows_sha256": _sha(rows_path),
                "prediction_manifest_sha256": _sha(prediction_manifest),
                "episode_count": len(rows),
            }
            summary_path = tmp_path / f"{condition}-{seed}.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            runs.append(
                [
                    str(seed),
                    condition,
                    str(checkpoint),
                    str(summary_path),
                    str(rows_path),
                ]
            )
    return config_path, runs


def _argv(config: Path, runs: list[list[str]], output: Path) -> list[str]:
    argv = ["aggregate_petct_condition_metrics.py"]
    for run in runs:
        argv.extend(["--run", *run])
    argv.extend(["--experiment-config", str(config), "--output", str(output)])
    return argv


def test_main_aggregates_all_three_seeds_once_and_negative_result_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _fixture(tmp_path)
    output = tmp_path / "aggregate.json"
    monkeypatch.setattr(sys, "argv", _argv(config, runs, output))
    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert (
        payload["schema_version"]
        == "PETCT-BIDIRECTIONAL-EDITOR-CONFIRMATORY-v2.0"
    )
    assert payload["analysis_status"] == "VALID"
    assert payload["seed_estimand"]["registered_seeds"] == [11, 22, 33]
    assert payload["seed_estimand"]["population_inference_over_seeds"] is False
    comparison = payload["comparisons"][0]
    assert len(comparison["per_seed_descriptive"]) == 3
    assert comparison["estimate"] == pytest.approx(0.1)
    assert comparison["hypothesis_verdict"] == "NOT_SUPPORTED"
    assert payload["holm_applications_per_family"] == {"family": 1}
    first_rows = [
        json.loads(line)
        for line in Path(runs[0][4]).read_text(encoding="utf-8").splitlines()
    ]
    manifest_sha256 = first_rows[0]["learning_manifest_sha256"]
    learning_split_sha256 = first_rows[0]["learning_split_sha256"]
    assert payload["provenance_lock"] == {
        "training_manifest_sha256": manifest_sha256,
        "inference_manifest_sha256": manifest_sha256,
        "learning_manifest_sha256": manifest_sha256,
        "learning_split_sha256": learning_split_sha256,
        "partition": "val",
        "test_access_receipt_sha256": None,
    }


@pytest.mark.parametrize(
    "gate_p2t,gate_statistics",
    [
        (GATE_BLOCKED, GATE_ACTIVE),
        (GATE_ACTIVE, GATE_BLOCKED),
        (GATE_BLOCKED, GATE_BLOCKED),
        (None, GATE_ACTIVE),
        (OMIT_SECTION, GATE_ACTIVE),
        (GATE_ACTIVE, OMIT_SECTION),
    ],
)
def test_main_refuses_to_run_while_the_execution_gate_is_not_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_p2t,
    gate_statistics,
) -> None:
    # The BLOCKED_UNTIL gate was only enforced by the orchestration shell, so a
    # direct CLI call emitted an editor verdict anyway.  Both config sections
    # must be active, exactly as run_petct_route_a_after_baseline.sh requires
    # before it runs this aggregator, and exactly as the P2T side enforces.
    config, runs = _fixture(
        tmp_path, gate_p2t=gate_p2t, gate_statistics=gate_statistics
    )
    output = tmp_path / "aggregate.json"
    monkeypatch.setattr(sys, "argv", _argv(config, runs, output))
    with pytest.raises(RuntimeError, match="confirmatory_execution_gate"):
        main()
    assert not output.exists()


def test_main_runs_once_both_execution_gates_are_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The gate is a fail-closed precondition, not a permanent lockout: the exact
    # frozen ACTIVE string must let the confirmatory aggregation through.
    config, runs = _fixture(
        tmp_path, gate_p2t=GATE_ACTIVE, gate_statistics=GATE_ACTIVE
    )
    output = tmp_path / "aggregate.json"
    monkeypatch.setattr(sys, "argv", _argv(config, runs, output))
    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["analysis_status"] == "VALID"


def test_main_rejects_missing_seed_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _fixture(tmp_path)
    monkeypatch.setattr(
        sys, "argv", _argv(config, runs[:-1], tmp_path / "aggregate.json")
    )
    with pytest.raises(RuntimeError, match="exactly cover all 3"):
        main()


def test_main_rejects_wrong_operation_ood_from_confirmatory_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _fixture(tmp_path)
    runs[0][1] = "wrong_operation_OOD"
    monkeypatch.setattr(
        sys, "argv", _argv(config, runs, tmp_path / "aggregate.json")
    )
    with pytest.raises(RuntimeError, match="stress-only"):
        main()


def test_main_rejects_same_only_wrong_scope_from_confirmatory_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _fixture(tmp_path)
    runs[0][1] = "same_weight_wrong_scope"
    monkeypatch.setattr(
        sys, "argv", _argv(config, runs, tmp_path / "aggregate.json")
    )
    with pytest.raises(RuntimeError, match="SAME-only descriptive"):
        main()


def test_main_rejects_summary_rows_hash_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _fixture(tmp_path)
    rows_path = Path(runs[0][-1])
    rows_path.write_text(rows_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _argv(config, runs, tmp_path / "aggregate.json"))
    with pytest.raises(RuntimeError, match="does not bind"):
        main()


@pytest.mark.parametrize(
    "field",
    [
        "learning_manifest_sha256",
        "learning_split_sha256",
        "partition",
    ],
)
def test_main_rejects_cross_run_provenance_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    config, runs = _fixture(tmp_path)
    checkpoint_path = Path(runs[0][2])
    summary_path = Path(runs[0][3])
    rows_path = Path(runs[0][4])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    replacement = {
        "learning_manifest_sha256": "e" * 64,
        "learning_split_sha256": "f" * 64,
        "partition": "test",
    }[field]
    if field == "partition":
        # A test partition also requires one internally consistent receipt hash.
        summary["test_access_receipt_sha256"] = "1" * 64
        for row in rows:
            row["test_access_receipt_sha256"] = "1" * 64
    summary[field] = replacement
    for row in rows:
        row[field] = replacement
    if field == "learning_manifest_sha256":
        checkpoint["manifest_sha256"] = replacement
    if field == "learning_split_sha256":
        checkpoint["learning_split_sha256"] = replacement
    torch.save(checkpoint, checkpoint_path)
    new_checkpoint_sha = _sha(checkpoint_path)
    summary["checkpoint_sha256"] = new_checkpoint_sha
    for row in rows:
        row["checkpoint_sha256"] = new_checkpoint_sha
    _write_jsonl(rows_path, rows)
    summary["metric_rows_sha256"] = _sha(rows_path)
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _argv(config, runs, tmp_path / "aggregate.json"))
    with pytest.raises(RuntimeError):
        main()


def test_main_rejects_cross_run_test_access_receipt_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _fixture(tmp_path)
    for run_index, run in enumerate(runs):
        summary_path = Path(run[3])
        rows_path = Path(run[4])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        receipt_hash = ("1" if run_index else "2") * 64
        summary["partition"] = "test"
        summary["test_access_receipt_sha256"] = receipt_hash
        for row in rows:
            row["partition"] = "test"
            row["test_access_receipt_sha256"] = receipt_hash
        _write_jsonl(rows_path, rows)
        summary["metric_rows_sha256"] = _sha(rows_path)
        summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _argv(config, runs, tmp_path / "aggregate.json"))
    with pytest.raises(RuntimeError):
        main()


@pytest.mark.parametrize("field", ["visible_npz_sha256", "evaluation_npz_sha256"])
def test_main_rejects_episode_artifact_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    config, runs = _fixture(tmp_path)
    summary_path = Path(runs[0][3])
    rows_path = Path(runs[0][4])
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    rows[0][field] = "e" * 64
    _write_jsonl(rows_path, rows)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["metric_rows_sha256"] = _sha(rows_path)
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _argv(config, runs, tmp_path / "aggregate.json"))
    with pytest.raises(RuntimeError, match="prediction provenance|leaf artifact"):
        main()


@pytest.mark.parametrize(
    "path_field", ["prediction_npz", "evaluation_npz", "visible_npz"]
)
def test_main_rehashes_every_editor_leaf_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_field: str,
) -> None:
    config, runs = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in Path(runs[0][4]).read_text(encoding="utf-8").splitlines()
    ]
    Path(rows[0][path_field]).write_bytes(b"tampered-after-evaluation")
    monkeypatch.setattr(sys, "argv", _argv(config, runs, tmp_path / "aggregate.json"))
    with pytest.raises(RuntimeError, match="leaf artifact hash changed"):
        main()


def _same_weight_fixture(tmp_path: Path) -> tuple[Path, list[list[str]]]:
    seeds = [11, 22, 33]
    conditions = {
        "scribble_plus_intent": ("scribble_plus_intent", [0.8, 0.9]),
        "same_weight_NULL": ("scribble_plus_intent", [0.6, 0.7]),
        "oracle_slots": ("scribble_plus_intent", [0.82, 0.92]),
        "predicted_slots": ("scribble_plus_intent", [0.80, 0.90]),
    }
    config = {
        "p2t": {"confirmatory_execution_gate": GATE_ACTIVE},
        "editor": {"training": {"seeds": seeds}},
        "statistics": {
            "multiplicity": "Holm within each frozen confirmatory family",
            "alpha": 0.1,
            "bootstrap_samples": 100,
            "permutation_samples": 200,
            "bootstrap_seed": 11,
            "permutation_seed": 101,
            "confirmatory_execution_gate": GATE_ACTIVE,
            "confirmatory_same_weight_families": {
                "rich_intent_causal": [
                    "scribble_plus_intent",
                    "same_weight_NULL",
                ],
                "slot_retention": ["oracle_slots", "predicted_slots"],
            },
            "effect_thresholds": {"gain": 0.02, "lower_margin": -0.05},
            "confirmatory_contrasts": [
                {
                    "family": "rich_intent_causal",
                    "treatment": "scribble_plus_intent",
                    "comparator": "same_weight_NULL",
                    "metric": "prompt_distal_recall",
                    "alternative": "greater",
                    "threshold_ref": "gain",
                    "decision_rule": (
                        "holm_p_lt_alpha_and_point_estimate_ge_threshold"
                    ),
                },
                {
                    "family": "slot_retention",
                    "treatment": "predicted_slots",
                    "comparator": "oracle_slots",
                    "metric": "prompt_distal_recall",
                    "alternative": "noninferiority",
                    "threshold_ref": "lower_margin",
                    "decision_rule": (
                        "holm_p_lt_alpha_and_point_estimate_gt_lower_margin"
                    ),
                },
            ],
        },
    }
    config_path = tmp_path / "same-weight-experiment.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    config_sha = _sha(config_path)
    leaf = _shared_leaf_inputs(tmp_path)
    runs: list[list[str]] = []
    for seed in seeds:
        checkpoints: dict[str, Path] = {}
        for checkpoint_condition in {value[0] for value in conditions.values()}:
            checkpoint = tmp_path / f"{checkpoint_condition}-{seed}.pth"
            torch.save(
                {
                    "schema_version": EDITOR_CHECKPOINT_SCHEMA,
                    "seed": seed,
                    "seed_registry": seeds,
                    "condition": checkpoint_condition,
                    "experiment_config_sha256": config_sha,
                    "status": "TRAINED_WHEN_THIS_SCRIPT_IS_EXECUTED",
                    "manifest": str(Path(leaf["manifest"]).resolve()),
                    "manifest_sha256": leaf["manifest_sha256"],
                    "training_manifest": str(Path(leaf["manifest"]).resolve()),
                    "training_manifest_sha256": leaf["manifest_sha256"],
                    "learning_split": str(
                        Path(leaf["learning_split"]).resolve()
                    ),
                    "learning_split_sha256": leaf["learning_split_sha256"],
                },
                checkpoint,
            )
            checkpoints[checkpoint_condition] = checkpoint
        for condition, (checkpoint_condition, values) in conditions.items():
            checkpoint = checkpoints[checkpoint_condition]
            checkpoint_sha = _sha(checkpoint)
            rows = [
                {
                    "episode_id": f"episode-{index}",
                    "patient_id": f"patient-{index}",
                    "condition": condition,
                    "checkpoint_sha256": checkpoint_sha,
                    "experiment_config_sha256": config_sha,
                    "learning_split_sha256": leaf["learning_split_sha256"],
                    "partition": "val",
                    "test_access_receipt_sha256": None,
                    "threshold": 0.5,
                    "distal_radius_mm": 15.0,
                    "metric_grid": "full-grid",
                    "prompt_distal_recall": value,
                }
                for index, value in enumerate(values, start=1)
            ]
            rows_path, prediction_manifest = _write_run_rows(
                tmp_path=tmp_path,
                run_name=f"{condition}-{seed}",
                rows=rows,
                leaf=leaf,
            )
            summary = {
                "schema_version": "PETCT-BIDIRECTIONAL-CORRECTION-METRICS-v2.0",
                "condition": condition,
                "checkpoint_sha256": checkpoint_sha,
                "experiment_config_sha256": config_sha,
                "learning_manifest_sha256": leaf["manifest_sha256"],
                "learning_split_sha256": leaf["learning_split_sha256"],
                "partition": "val",
                "test_access_receipt_sha256": None,
                "threshold": 0.5,
                "distal_radius_mm": 15.0,
                "metric_grid": "full-grid",
                "metric_rows_sha256": _sha(rows_path),
                "prediction_manifest_sha256": _sha(prediction_manifest),
                "episode_count": len(rows),
            }
            summary_path = tmp_path / f"{condition}-{seed}.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            runs.append(
                [
                    str(seed),
                    condition,
                    str(checkpoint),
                    str(summary_path),
                    str(rows_path),
                ]
            )
    return config_path, runs


def test_main_records_same_weight_checkpoint_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, runs = _same_weight_fixture(tmp_path)
    output = tmp_path / "aggregate.json"
    monkeypatch.setattr(sys, "argv", _argv(config, runs, output))
    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    groups = payload["same_weight_checkpoint_groups"]
    assert len(groups) == 3
    assert {
        (group["checkpoint_condition"], tuple(group["conditions"]))
        for group in groups
    } == {
        (
            "scribble_plus_intent",
            (
                "oracle_slots",
                "predicted_slots",
                "same_weight_NULL",
                "scribble_plus_intent",
            ),
        ),
    }


@pytest.mark.parametrize(
    "alias", ["same_weight_NULL", "oracle_slots", "predicted_slots"]
)
def test_main_rejects_different_checkpoint_for_same_weight_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    config, runs = _same_weight_fixture(tmp_path)
    run = next(row for row in runs if row[0] == "11" and row[1] == alias)
    original = torch.load(Path(run[2]), map_location="cpu", weights_only=True)
    original["tamper_nonce"] = alias
    replacement = tmp_path / f"replacement-{alias}.pth"
    torch.save(original, replacement)
    replacement_sha = _sha(replacement)
    run[2] = str(replacement)
    summary_path = Path(run[3])
    rows_path = Path(run[4])
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    for row in rows:
        row["checkpoint_sha256"] = replacement_sha
    _write_jsonl(rows_path, rows)
    prediction_manifest = Path(rows[0]["prediction_manifest"])
    prediction_rows = [
        json.loads(line)
        for line in prediction_manifest.read_text(encoding="utf-8").splitlines()
    ]
    for row in prediction_rows:
        row["checkpoint_sha256"] = replacement_sha
    _write_jsonl(prediction_manifest, prediction_rows)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["checkpoint_sha256"] = replacement_sha
    summary["metric_rows_sha256"] = _sha(rows_path)
    summary["prediction_manifest_sha256"] = _sha(prediction_manifest)
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", _argv(config, runs, tmp_path / "aggregate.json"))
    with pytest.raises(RuntimeError, match="same-weight editor aliases"):
        main()
