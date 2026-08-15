from __future__ import annotations

import copy
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

import aggregate_petct_p2t_confirmatory as aggregator  # noqa: E402
from aggregate_petct_p2t_confirmatory import (  # noqa: E402
    P2T_PAIRED_EVALUATION_ROW_SCHEMA,
    _metric,
    _validate_pairing,
    main,
    paired_cross_seed_inference,
)
from common.petct_learning import (  # noqa: E402
    P2T_CHECKPOINT_SCHEMA,
    P2T_METRICS_SCHEMA,
)


# Frozen six-class bidirectional joint order (D-2026-07-31-01).  Pinned as a
# literal on purpose: the aggregator must not be able to silently reorder or
# resize the ontology it scores.
FROZEN_SIX_CLASS_JOINT_ORDER = (
    "ADD_SAME_LOCAL",
    "REMOVE_SAME_LOCAL",
    "ADD_SAME_COMPLETE",
    "REMOVE_SAME_COMPLETE",
    "ADD_NEW_COMPLETE",
    "REMOVE_NEW_COMPLETE",
)
STRUCTURALLY_INVALID_JOINTS = ("ADD_NEW_LOCAL", "REMOVE_NEW_LOCAL")
GATE_ACTIVE = "ACTIVE_AFTER_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"
GATE_BLOCKED = "BLOCKED_UNTIL_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"


def _runs():
    seeds = [11, 22, 33]
    truth = [
        ("e1", "p1", 0),
        ("e2", "p1", 1),
        ("e3", "p2", 2),
        ("e4", "p3", 0),
    ]
    runs = {}
    for seed in seeds:
        for arm in ("full", "no_M0"):
            predictions = (
                [0, 1, 2, 0]
                if arm == "full"
                else [1, 1, 0, 0]
            )
            runs[(seed, arm)] = [
                {
                    "episode_id": episode,
                    "patient_id": patient,
                    "gold_joint_id": gold,
                    "predicted_joint_id": prediction,
                    "matched_state_group_id": f"group-{episode}",
                    "strategy": "centerline",
                    "visible_npz_sha256": "a" * 64,
                    "evaluation_npz_sha256": "b" * 64,
                }
                for (episode, patient, gold), prediction in zip(truth, predictions)
            ]
    return seeds, runs


def test_p2t_inference_recomputes_nonlinear_f1_with_synchronous_patient_resampling() -> None:
    seeds, runs = _runs()
    first = paired_cross_seed_inference(
        runs,
        seeds=seeds,
        treatment="full",
        comparator="no_M0",
        bootstrap_seed=7,
        bootstrap_samples=200,
        permutation_seed=17,
        permutation_samples=300,
        alpha=0.05,
    )
    second = paired_cross_seed_inference(
        runs,
        seeds=seeds,
        treatment="full",
        comparator="no_M0",
        bootstrap_seed=7,
        bootstrap_samples=200,
        permutation_seed=17,
        permutation_samples=300,
        alpha=0.05,
    )
    assert first == second
    assert first["permutation_unit"].startswith("whole patient cluster")
    assert first["cluster_resampling"].startswith("patients synchronously")
    assert len(first["per_seed_descriptive"]) == 3

    # A deliberately invalid shortcut would average separate per-patient F1
    # scores.  Macro-F1 is nonlinear, and the registered estimand recomputes it
    # over the patient-balanced population instead.
    per_patient_shortcut = []
    for patient in ("p1", "p2", "p3"):
        full = [row for row in runs[(11, "full")] if row["patient_id"] == patient]
        no_m0 = [row for row in runs[(11, "no_M0")] if row["patient_id"] == patient]
        per_patient_shortcut.append(_metric(full) - _metric(no_m0))
    assert first["estimate"] != pytest.approx(
        sum(per_patient_shortcut) / len(per_patient_shortcut)
    )


def test_p2t_pairing_rejects_gold_or_patient_remap() -> None:
    seeds, runs = _runs()
    _validate_pairing(runs, seeds=seeds, treatment="full", comparator="no_M0")
    tampered = copy.deepcopy(runs)
    tampered[(22, "no_M0")][0]["gold_joint_id"] = 2
    with pytest.raises(RuntimeError, match="exact episode/patient/gold pairing"):
        _validate_pairing(
            tampered, seeds=seeds, treatment="full", comparator="no_M0"
        )


def test_p2t_grid_is_fixed_seed_replicates_not_n_equals_three_inference() -> None:
    seeds, runs = _runs()
    result = paired_cross_seed_inference(
        runs,
        seeds=seeds,
        treatment="full",
        comparator="no_M0",
        bootstrap_seed=3,
        bootstrap_samples=20,
        permutation_seed=4,
        permutation_samples=20,
        alpha=0.05,
    )
    assert result["patient_count"] == 3
    assert "seed_standard_error" not in result
    assert "t_test" not in result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prepare_main_argv(
    tmp_path: Path,
    seeds,
    episodes,
    predict,
    *,
    partition="val",
    test_access_receipt=None,
    gate_p2t=GATE_ACTIVE,
    gate_statistics=GATE_ACTIVE,
    row_overrides=None,
):
    """Materialise the seed x arm run grid main() binds against.

    ``episodes`` is a sequence of ``(patient_id, gold_joint_id)`` and
    ``predict(arm, gold)`` returns the predicted joint id for that arm.
    ``partition`` and ``test_access_receipt`` accept either a scalar or a
    ``(seed, arm)`` callable so a single arm can be desynchronised.
    ``row_overrides`` is merged into every paired row last, so a test can
    desynchronise the rows from their own metrics document.
    """

    def _per_run(value, seed, arm):
        return value(seed, arm) if callable(value) else value

    config = {
        "p2t": {
            "training": {"seeds": list(seeds)},
            "confirmatory_execution_gate": gate_p2t,
            "confirmatory_contrast": {
                "treatment": "full",
                "comparator": "no_M0",
                "metric": "joint_goal_macro_f1",
                "alternative": "greater",
                "minimum_absolute_effect": 0.02,
                "threshold_ref": "p2t_gain",
                "decision_rule": "p_value_lt_alpha_and_point_estimate_ge_threshold",
            },
        },
        "statistics": {
            "alpha": 0.05,
            "bootstrap_samples": 20,
            "permutation_samples": 20,
            "bootstrap_seed": 7,
            "permutation_seed": 17,
            "effect_thresholds": {"p2t_gain": 0.02},
            "confirmatory_execution_gate": gate_statistics,
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_sha = _sha(config_path)
    argv = ["aggregate_petct_p2t_confirmatory.py"]
    for seed in seeds:
        for arm in ("full", "no_M0"):
            checkpoint = tmp_path / f"{seed}-{arm}.pth"
            torch.save(
                {
                    "schema_version": P2T_CHECKPOINT_SCHEMA,
                    "seed": seed,
                    "input_ablation": arm,
                    "experiment_config_sha256": config_sha,
                },
                checkpoint,
            )
            checkpoint_sha = _sha(checkpoint)
            run_partition = _per_run(partition, seed, arm)
            run_receipt = _per_run(test_access_receipt, seed, arm)
            rows = []
            for index, (patient, gold) in enumerate(episodes, start=1):
                rows.append(
                    {
                        "schema_version": P2T_PAIRED_EVALUATION_ROW_SCHEMA,
                        "episode_id": f"e{index}",
                        "patient_id": patient,
                        "matched_state_group_id": f"g{index}",
                        "strategy": "centerline",
                        "gold_joint_id": gold,
                        "predicted_joint_id": predict(arm, gold),
                        "checkpoint_seed": seed,
                        "input_ablation": arm,
                        "checkpoint_sha256": checkpoint_sha,
                        "experiment_config_sha256": config_sha,
                        "learning_manifest_sha256": "a" * 64,
                        "visible_npz_sha256": "b" * 64,
                        "evaluation_npz_sha256": "c" * 64,
                        "partition": run_partition,
                        "test_access_receipt_sha256": run_receipt,
                        **(row_overrides or {}),
                    }
                )
            rows_path = tmp_path / f"{seed}-{arm}.jsonl"
            _write_jsonl(rows_path, rows)
            metrics = {
                "schema_version": P2T_METRICS_SCHEMA,
                "checkpoint_seed": seed,
                "input_ablation": arm,
                "checkpoint_sha256": checkpoint_sha,
                "experiment_config_sha256": config_sha,
                "manifest_sha256": "a" * 64,
                "paired_evaluation_rows_schema": P2T_PAIRED_EVALUATION_ROW_SCHEMA,
                "paired_evaluation_rows_sha256": _sha(rows_path),
                "paired_evaluation_row_count": len(rows),
                "partition": run_partition,
                "test_access_receipt_sha256": run_receipt,
            }
            metrics_path = tmp_path / f"{seed}-{arm}.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            argv.extend(
                [
                    "--run",
                    str(seed),
                    arm,
                    str(checkpoint),
                    str(metrics_path),
                    str(rows_path),
                ]
            )
    output = tmp_path / "confirmatory.json"
    argv.extend(["--experiment-config", str(config_path), "--output", str(output)])
    return argv, output


def test_main_requires_exact_three_seed_by_two_arm_grid_and_binds_real_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds = [11, 22, 33]
    argv, output = _prepare_main_argv(
        tmp_path,
        seeds,
        (("p1", 0), ("p2", 1), ("p3", 2)),
        lambda arm, gold: gold if arm == "full" else 0,
    )
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["analysis_status"] == "VALID"
    assert payload["registered_seeds"] == seeds
    assert payload["seed_population_inference"] is False
    assert len(payload["input_runs"]) == 6


def test_confirmatory_labels_match_the_frozen_six_class_joint_order() -> None:
    assert aggregator.LABEL_NAMES == FROZEN_SIX_CLASS_JOINT_ORDER
    assert aggregator.LABELS == (0, 1, 2, 3, 4, 5)
    assert aggregator.SCHEMA_VERSION == "PETCT-P2T-CONFIRMATORY-v2.0"
    # Only the class count moves; the registered estimand stays put.
    assert aggregator.ESTIMAND == (
        "mean_of_registered_seed_specific_patient_balanced_macro_f1_deltas"
    )


def test_confirmatory_scores_all_six_joint_classes_and_stamps_v2_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeds = [11, 22, 33]
    episodes = (
        ("p1", 0),
        ("p1", 1),
        ("p2", 2),
        ("p2", 3),
        ("p3", 4),
        ("p3", 5),
    )
    argv, output = _prepare_main_argv(
        tmp_path, seeds, episodes, lambda arm, gold: gold if arm == "full" else 0
    )
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "PETCT-P2T-CONFIRMATORY-v2.0"
    assert payload["analysis_status"] == "VALID"
    assert payload["joint_label_names"] == list(FROZEN_SIX_CLASS_JOINT_ORDER)
    assert payload["estimand"] == (
        "mean_of_registered_seed_specific_patient_balanced_macro_f1_deltas"
    )
    # An arm that only ever answers ADD_SAME_LOCAL must not tie a perfect arm
    # once all six classes are actually scored.
    assert payload["estimate"] > 0.0


@pytest.mark.parametrize("field", ["gold_joint_id", "predicted_joint_id"])
@pytest.mark.parametrize("illegal_joint_id", [6, 7])
def test_confirmatory_fails_closed_on_structurally_invalid_new_local_joints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    illegal_joint_id: int,
) -> None:
    # ADD_NEW_LOCAL / REMOVE_NEW_LOCAL have no place in the frozen enumeration,
    # so ids 6 and 7 (what they would occupy if someone appended them) must be
    # rejected rather than silently scored.
    assert set(aggregator.LABEL_NAMES).isdisjoint(STRUCTURALLY_INVALID_JOINTS)
    seeds = [11, 22, 33]
    if field == "gold_joint_id":
        episodes = (("p1", illegal_joint_id), ("p2", 1), ("p3", 2))
        predict = lambda arm, gold: 0  # noqa: E731
    else:
        episodes = (("p1", 0), ("p2", 1), ("p3", 2))
        predict = lambda arm, gold: illegal_joint_id  # noqa: E731
    argv, _ = _prepare_main_argv(tmp_path, seeds, episodes, predict)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match=f"P2T paired row has invalid {field}"):
        main()


# --- locked-test / execution-gate firewall -----------------------------------

_SEEDS = [11, 22, 33]
_EPISODES = (("p1", 0), ("p2", 1), ("p3", 2))
_PREDICT = lambda arm, gold: gold if arm == "full" else 0  # noqa: E731


@pytest.mark.parametrize(
    "gate_p2t,gate_statistics",
    [
        (GATE_BLOCKED, GATE_ACTIVE),
        (GATE_ACTIVE, GATE_BLOCKED),
        (GATE_BLOCKED, GATE_BLOCKED),
        (None, GATE_ACTIVE),
    ],
)
def test_confirmatory_refuses_to_run_while_the_execution_gate_is_not_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate_p2t,
    gate_statistics,
) -> None:
    # The BLOCKED_UNTIL gate was only enforced by the orchestration shell, so a
    # direct CLI call emitted a verdict anyway.  Both config sections must be
    # active, exactly as run_petct_route_a_after_baseline.sh requires.
    argv, _ = _prepare_main_argv(
        tmp_path,
        _SEEDS,
        _EPISODES,
        _PREDICT,
        gate_p2t=gate_p2t,
        gate_statistics=gate_statistics,
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="confirmatory_execution_gate"):
        main()


@pytest.mark.parametrize("partition", ["train", "TEST", "", None])
def test_confirmatory_rejects_a_partition_outside_val_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, partition
) -> None:
    argv, _ = _prepare_main_argv(
        tmp_path, _SEEDS, _EPISODES, _PREDICT, partition=partition
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="partition must be val or test"):
        main()


@pytest.mark.parametrize("receipt", [None, "", "not-a-sha256", "d" * 63])
def test_confirmatory_refuses_a_test_partition_without_a_valid_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, receipt
) -> None:
    # This is the only script that emits a P2T locked-test verdict, so it must
    # be able to tell on its own that it is reading test-partition rows.
    argv, _ = _prepare_main_argv(
        tmp_path,
        _SEEDS,
        _EPISODES,
        _PREDICT,
        partition="test",
        test_access_receipt=receipt,
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="test-access receipt"):
        main()


def test_confirmatory_refuses_a_validation_run_that_carries_a_test_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, _ = _prepare_main_argv(
        tmp_path,
        _SEEDS,
        _EPISODES,
        _PREDICT,
        partition="val",
        test_access_receipt="d" * 64,
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="must not carry a test-access receipt"):
        main()


def test_confirmatory_refuses_arms_that_disagree_on_the_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, _ = _prepare_main_argv(
        tmp_path,
        _SEEDS,
        _EPISODES,
        _PREDICT,
        partition=lambda seed, arm: "test" if arm == "full" else "val",
        test_access_receipt=lambda seed, arm: "d" * 64 if arm == "full" else None,
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="different partitions"):
        main()


def test_confirmatory_refuses_rows_that_disagree_with_their_own_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, _ = _prepare_main_argv(
        tmp_path,
        _SEEDS,
        _EPISODES,
        _PREDICT,
        partition="val",
        row_overrides={"partition": "test"},
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        main()


def test_confirmatory_accepts_an_authorised_test_run_and_records_its_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = "d" * 64
    argv, output = _prepare_main_argv(
        tmp_path,
        _SEEDS,
        _EPISODES,
        _PREDICT,
        partition="test",
        test_access_receipt=receipt,
    )
    monkeypatch.setattr(sys, "argv", argv)
    assert main() == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["analysis_status"] == "VALID"
    # A verdict document must state on its face which partition produced it.
    assert payload["partition"] == "test"
    assert payload["test_access_receipt_sha256"] == receipt
