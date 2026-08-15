#!/usr/bin/env python3
"""Run the frozen cross-seed paired P2T confirmatory analysis."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import (  # noqa: E402
    P2T_CHECKPOINT_SCHEMA,
    P2T_METRICS_SCHEMA,
    load_jsonl,
    patient_balanced_macro_f1,
    sha256_file,
    write_json_exclusive,
)
from common.petct_models import LEGAL_GOALS  # noqa: E402
from evaluation.evaluate_petct_p2t import (  # noqa: E402
    P2T_PAIRED_EVALUATION_ROW_SCHEMA,
)


SCHEMA_VERSION = "PETCT-P2T-CONFIRMATORY-v2.0"
ESTIMAND = "mean_of_registered_seed_specific_patient_balanced_macro_f1_deltas"
DECISION_RULE = "p_value_lt_alpha_and_point_estimate_ge_threshold"
# Six-class bidirectional joint ontology.  The order is inherited from the
# frozen LEGAL_GOALS tuple so this aggregator can never score a different
# enumeration than the joint head that produced the paired rows, and the two
# structurally invalid NEW_LOCAL joints stay out by construction.
CONFIRMATORY_GATE_ACTIVE = "ACTIVE_AFTER_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"
GATED_CONFIG_SECTIONS = ("p2t", "statistics")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STRUCTURALLY_INVALID_GOALS = ("ADD_NEW_LOCAL", "REMOVE_NEW_LOCAL")
LABEL_NAMES = tuple(LEGAL_GOALS)
LABELS = tuple(range(len(LABEL_NAMES)))
if len(LABEL_NAMES) != 6 or not set(LABEL_NAMES).isdisjoint(
    STRUCTURALLY_INVALID_GOALS
):
    raise RuntimeError(
        "P2T confirmatory analysis requires the frozen six-class joint ontology "
        "without the structurally invalid NEW_LOCAL joints"
    )


def _finite_number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
    ):
        raise RuntimeError(f"{label} must be a finite number")
    return float(value)


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"{label} must be an integer >= 1")
    return int(value)


def _require_active_confirmatory_gate(config: Mapping[str, Any]) -> None:
    """Fail closed unless the frozen execution gate is open in every section.

    The gate used to be enforced only by run_petct_route_a_after_baseline.sh,
    so calling this CLI directly emitted a verdict while the gate was still
    BLOCKED.  Mirror the shell's requirement here.
    """

    for section in GATED_CONFIG_SECTIONS:
        block = config.get(section)
        if (
            not isinstance(block, Mapping)
            or block.get("confirmatory_execution_gate") != CONFIRMATORY_GATE_ACTIVE
        ):
            raise RuntimeError(
                f"{section}.confirmatory_execution_gate must be "
                f"{CONFIRMATORY_GATE_ACTIVE} before a P2T confirmatory verdict"
            )


def _run_partition_and_receipt(metrics: Mapping[str, Any]) -> tuple[str, str | None]:
    """Return the run's partition plus its locked-test receipt hash.

    Same contract as aggregate_petct_condition_metrics.py: a test run must
    carry a valid receipt hash, and a validation run must carry none.
    """

    partition = metrics.get("partition")
    if partition not in {"val", "test"}:
        raise RuntimeError("P2T metrics partition must be val or test")
    test_access_sha256 = metrics.get("test_access_receipt_sha256")
    if partition == "test":
        if not SHA256_RE.fullmatch(str(test_access_sha256 or "")):
            raise RuntimeError("test P2T run requires a valid test-access receipt hash")
    elif test_access_sha256 is not None:
        raise RuntimeError("validation P2T run must not carry a test-access receipt")
    return str(partition), test_access_sha256


def _patient_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        patient = str(row.get("patient_id") or "")
        if not patient:
            raise RuntimeError("paired P2T row omits patient_id")
        grouped[patient].append(index)
    return dict(grouped)


def _metric_for_patient_draw(
    rows: Sequence[Mapping[str, Any]], selected_patients: Sequence[str]
) -> float:
    grouped = _patient_index(rows)
    truth: list[int] = []
    prediction: list[int] = []
    synthetic_patients: list[str] = []
    for draw_index, patient in enumerate(selected_patients):
        for row_index in grouped[str(patient)]:
            row = rows[row_index]
            truth.append(int(row["gold_joint_id"]))
            prediction.append(int(row["predicted_joint_id"]))
            synthetic_patients.append(f"{draw_index}:{patient}")
    return patient_balanced_macro_f1(
        truth, prediction, synthetic_patients, LABELS
    )


def _metric(
    rows: Sequence[Mapping[str, Any]], *, prediction_by_episode: Mapping[str, int] | None = None
) -> float:
    truth = [int(row["gold_joint_id"]) for row in rows]
    prediction = [
        int(
            row["predicted_joint_id"]
            if prediction_by_episode is None
            else prediction_by_episode[str(row["episode_id"])]
        )
        for row in rows
    ]
    patients = [str(row["patient_id"]) for row in rows]
    return patient_balanced_macro_f1(truth, prediction, patients, LABELS)


def paired_cross_seed_inference(
    runs: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    *,
    seeds: Sequence[int],
    treatment: str,
    comparator: str,
    bootstrap_seed: int,
    bootstrap_samples: int,
    permutation_seed: int,
    permutation_samples: int,
    alpha: float,
) -> dict[str, Any]:
    """Recompute the nonlinear F1 estimand under patient-cluster resampling."""

    reference = list(runs[(int(seeds[0]), treatment)])
    patients = sorted(_patient_index(reference))
    if len(patients) < 2:
        raise RuntimeError("P2T confirmatory analysis needs at least two patients")
    per_seed = []
    for seed in seeds:
        treatment_estimate = _metric(runs[(int(seed), treatment)])
        comparator_estimate = _metric(runs[(int(seed), comparator)])
        per_seed.append(
            {
                "seed": int(seed),
                "treatment_patient_balanced_macro_f1": treatment_estimate,
                "comparator_patient_balanced_macro_f1": comparator_estimate,
                "delta": treatment_estimate - comparator_estimate,
            }
        )
    observed = float(np.mean([row["delta"] for row in per_seed]))

    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    bootstrap_draws = []
    for _ in range(bootstrap_samples):
        selected = bootstrap_rng.choice(patients, size=len(patients), replace=True).tolist()
        deltas = []
        for seed in seeds:
            deltas.append(
                _metric_for_patient_draw(runs[(int(seed), treatment)], selected)
                - _metric_for_patient_draw(runs[(int(seed), comparator)], selected)
            )
        bootstrap_draws.append(float(np.mean(deltas)))

    by_key = {
        key: {str(row["episode_id"]): row for row in rows}
        for key, rows in runs.items()
    }
    patient_episodes = {
        patient: [
            str(row["episode_id"])
            for row in reference
            if str(row["patient_id"]) == patient
        ]
        for patient in patients
    }
    permutation_rng = np.random.default_rng(permutation_seed)
    null_draws = []
    for _ in range(permutation_samples):
        swap = dict(zip(patients, permutation_rng.integers(0, 2, size=len(patients))))
        deltas = []
        for seed in seeds:
            treatment_rows = runs[(int(seed), treatment)]
            comparator_rows = runs[(int(seed), comparator)]
            treatment_index = by_key[(int(seed), treatment)]
            comparator_index = by_key[(int(seed), comparator)]
            permuted_treatment: dict[str, int] = {}
            permuted_comparator: dict[str, int] = {}
            for patient, episodes in patient_episodes.items():
                for episode in episodes:
                    treatment_prediction = int(
                        treatment_index[episode]["predicted_joint_id"]
                    )
                    comparator_prediction = int(
                        comparator_index[episode]["predicted_joint_id"]
                    )
                    if swap[patient]:
                        treatment_prediction, comparator_prediction = (
                            comparator_prediction,
                            treatment_prediction,
                        )
                    permuted_treatment[episode] = treatment_prediction
                    permuted_comparator[episode] = comparator_prediction
            deltas.append(
                _metric(
                    treatment_rows, prediction_by_episode=permuted_treatment
                )
                - _metric(
                    comparator_rows, prediction_by_episode=permuted_comparator
                )
            )
        null_draws.append(float(np.mean(deltas)))
    p_value = float(
        (1 + sum(draw >= observed for draw in null_draws))
        / (permutation_samples + 1)
    )
    return {
        "patient_count": len(patients),
        "estimate": observed,
        "ci_low": float(np.quantile(bootstrap_draws, alpha / 2)),
        "ci_high": float(np.quantile(bootstrap_draws, 1 - alpha / 2)),
        "p_value": p_value,
        "p_value_holm": p_value,
        "per_seed_descriptive": per_seed,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "permutation_samples": permutation_samples,
        "permutation_seed": permutation_seed,
        "cluster_resampling": "patients synchronously across arms and registered seeds",
        "permutation_unit": "whole patient cluster synchronously across registered seeds",
    }


def _validate_pairing(
    runs: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    *,
    seeds: Sequence[int],
    treatment: str,
    comparator: str,
) -> None:
    reference_key = (int(seeds[0]), treatment)
    reference_rows = runs[reference_key]
    reference = {
        str(row["episode_id"]): (
            str(row["patient_id"]),
            int(row["gold_joint_id"]),
            row.get("matched_state_group_id"),
            str(row["strategy"]),
            str(row["visible_npz_sha256"]),
            str(row["evaluation_npz_sha256"]),
        )
        for row in reference_rows
    }
    if len(reference) != len(reference_rows):
        raise RuntimeError("paired P2T rows contain duplicate episode_id values")
    for key, rows in runs.items():
        current = {
            str(row["episode_id"]): (
                str(row["patient_id"]),
                int(row["gold_joint_id"]),
                row.get("matched_state_group_id"),
                str(row["strategy"]),
                str(row["visible_npz_sha256"]),
                str(row["evaluation_npz_sha256"]),
            )
            for row in rows
        }
        if len(current) != len(rows) or current != reference:
            raise RuntimeError(
                "P2T seed/arm rows differ in exact episode/patient/gold pairing: "
                f"{key}"
            )
    for seed in seeds:
        if (int(seed), treatment) not in runs or (int(seed), comparator) not in runs:
            raise RuntimeError("P2T paired run grid is incomplete")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=5,
        required=True,
        metavar=("SEED", "ARM", "CHECKPOINT", "METRICS", "PAIRED_ROWS"),
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")

    if args.experiment_config.is_symlink() or not args.experiment_config.is_file():
        parser.error("experiment config must be a non-symlink regular file")
    config_path = args.experiment_config.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    config_sha256 = sha256_file(config_path)
    _require_active_confirmatory_gate(config)
    seeds = [int(seed) for seed in config["p2t"]["training"]["seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise RuntimeError("P2T confirmatory analysis requires exactly 3 registered seeds")
    contrast = config["p2t"]["confirmatory_contrast"]
    treatment = str(contrast["treatment"])
    comparator = str(contrast["comparator"])
    if {treatment, comparator} != {"full", "no_M0"}:
        raise RuntimeError("frozen P2T contrast must be exactly full versus no_M0")
    if contrast.get("alternative") != "greater":
        raise RuntimeError("P2T confirmatory contrast must use the greater alternative")
    statistics = config["statistics"]
    threshold_ref = contrast.get("threshold_ref")
    if (
        contrast.get("decision_rule") != DECISION_RULE
        or not isinstance(threshold_ref, str)
        or threshold_ref not in statistics.get("effect_thresholds", {})
    ):
        raise RuntimeError(
            "P2T contrast requires the explicit single-contrast decision_rule/threshold_ref"
        )
    minimum_effect = _finite_number(
        statistics["effect_thresholds"][threshold_ref],
        label=f"statistics.effect_thresholds.{threshold_ref}",
    )
    if contrast.get("minimum_absolute_effect") not in (None, minimum_effect):
        raise RuntimeError("P2T inline minimum effect differs from threshold_ref")
    alpha = _finite_number(statistics.get("alpha"), label="statistics.alpha")
    if not 0 < alpha < 1:
        raise RuntimeError("statistics.alpha must be between zero and one")
    bootstrap_samples = _positive_int(
        statistics.get("bootstrap_samples"), label="statistics.bootstrap_samples"
    )
    permutation_samples = _positive_int(
        statistics.get("permutation_samples"), label="statistics.permutation_samples"
    )
    bootstrap_seed = int(statistics["bootstrap_seed"])
    permutation_seed = int(statistics["permutation_seed"])

    expected = {(seed, arm) for seed in seeds for arm in (treatment, comparator)}
    observed: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    input_runs = []
    controlled_manifest_sha256: str | None = None
    controlled_partition: str | None = None
    controlled_test_access_sha256: str | None = None
    for raw_seed, raw_arm, checkpoint_raw, metrics_raw, rows_raw in args.run:
        seed, arm = int(raw_seed), str(raw_arm)
        key = (seed, arm)
        if key not in expected or key in observed:
            raise RuntimeError("P2T runs must exactly cover the registered seed x paired-arm grid")
        checkpoint_path = Path(checkpoint_raw)
        metrics_path = Path(metrics_raw)
        rows_path = Path(rows_raw)
        for path, label in (
            (checkpoint_path, "checkpoint"),
            (metrics_path, "metrics"),
            (rows_path, "paired rows"),
        ):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"missing regular P2T {label}: {path}")
        checkpoint_path = checkpoint_path.resolve()
        metrics_path = metrics_path.resolve()
        rows_path = rows_path.resolve()
        checkpoint_sha256 = sha256_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("schema_version") != P2T_CHECKPOINT_SCHEMA
            or int(checkpoint.get("seed", -1)) != seed
            or checkpoint.get("input_ablation") != arm
            or checkpoint.get("experiment_config_sha256") != config_sha256
        ):
            raise RuntimeError("P2T checkpoint does not match its registered seed/arm/config")
        with metrics_path.open("r", encoding="utf-8") as stream:
            metrics = json.load(stream)
        rows = load_jsonl(rows_path)
        if not rows:
            raise RuntimeError("P2T paired rows are empty")
        if (
            metrics.get("schema_version") != P2T_METRICS_SCHEMA
            or int(metrics.get("checkpoint_seed", -1)) != seed
            or metrics.get("input_ablation") != arm
            or metrics.get("checkpoint_sha256") != checkpoint_sha256
            or metrics.get("experiment_config_sha256") != config_sha256
            or metrics.get("paired_evaluation_rows_schema")
            != P2T_PAIRED_EVALUATION_ROW_SCHEMA
            or metrics.get("paired_evaluation_rows_sha256") != sha256_file(rows_path)
            or int(metrics.get("paired_evaluation_row_count", -1)) != len(rows)
        ):
            raise RuntimeError("P2T metrics do not bind checkpoint/config/paired rows")
        manifest_sha = str(metrics.get("manifest_sha256") or "")
        if controlled_manifest_sha256 is None:
            controlled_manifest_sha256 = manifest_sha
        elif manifest_sha != controlled_manifest_sha256:
            raise RuntimeError("P2T paired arms use different controlled manifests")
        partition, test_access_sha256 = _run_partition_and_receipt(metrics)
        if controlled_partition is None:
            controlled_partition = partition
            controlled_test_access_sha256 = test_access_sha256
        elif (
            partition != controlled_partition
            or test_access_sha256 != controlled_test_access_sha256
        ):
            raise RuntimeError(
                "P2T paired arms use different partitions or test-access receipts"
            )
        required_row_values = {
            "schema_version": P2T_PAIRED_EVALUATION_ROW_SCHEMA,
            "checkpoint_seed": seed,
            "input_ablation": arm,
            "checkpoint_sha256": checkpoint_sha256,
            "experiment_config_sha256": config_sha256,
            "learning_manifest_sha256": manifest_sha,
            "partition": partition,
            "test_access_receipt_sha256": test_access_sha256,
        }
        for row in rows:
            if any(row.get(name) != value for name, value in required_row_values.items()):
                raise RuntimeError("P2T paired row provenance mismatch")
            for name in ("gold_joint_id", "predicted_joint_id"):
                value = row.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value not in LABELS:
                    raise RuntimeError(f"P2T paired row has invalid {name}")
        observed[key] = rows
        input_runs.append(
            {
                "seed": seed,
                "arm": arm,
                "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256},
                "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
                "paired_rows": {"path": str(rows_path), "sha256": sha256_file(rows_path)},
            }
        )
    if set(observed) != expected:
        raise RuntimeError("P2T runs must exactly cover 3 registered seeds x full/no_M0")
    _validate_pairing(
        observed, seeds=seeds, treatment=treatment, comparator=comparator
    )
    inference = paired_cross_seed_inference(
        observed,
        seeds=seeds,
        treatment=treatment,
        comparator=comparator,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
        permutation_seed=permutation_seed,
        permutation_samples=permutation_samples,
        alpha=alpha,
    )
    supported = (
        inference["p_value"] < alpha
        and inference["estimate"] >= minimum_effect
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "VALID",
        "hypothesis_verdict": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "estimand": ESTIMAND,
        "treatment": treatment,
        "comparator": comparator,
        "metric": str(contrast["metric"]),
        "joint_label_names": list(LABEL_NAMES),
        "registered_seeds": seeds,
        "seed_population_inference": False,
        "minimum_absolute_effect": minimum_effect,
        "threshold_ref": threshold_ref,
        "decision_rule": DECISION_RULE,
        "alpha": alpha,
        "controlled_tensor_manifest_sha256": controlled_manifest_sha256,
        "partition": controlled_partition,
        "test_access_receipt_sha256": controlled_test_access_sha256,
        "experiment_config_sha256": config_sha256,
        "input_runs": sorted(input_runs, key=lambda item: (item["seed"], item["arm"])),
        **inference,
    }
    json.dumps(payload, allow_nan=False, sort_keys=True)
    write_json_exclusive(args.output, payload)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
