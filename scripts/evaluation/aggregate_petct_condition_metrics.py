#!/usr/bin/env python3
"""Run one frozen cross-seed confirmatory analysis for editor conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import (  # noqa: E402
    EDITOR_CHECKPOINT_SCHEMA,
    load_jsonl,
    sha256_file,
    write_json_exclusive,
)
from common.petct_route_a_core import (  # noqa: E402
    EDITOR_CHECKPOINT_CONDITION_ALIASES as CHECKPOINT_CONDITION_ALIASES,
)


SCHEMA_VERSION = "PETCT-BIDIRECTIONAL-EDITOR-CONFIRMATORY-v2.0"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EDITOR_TRAINING_STATUS = "TRAINED_WHEN_THIS_SCRIPT_IS_EXECUTED"
CONFIRMATORY_GATE_ACTIVE = "ACTIVE_AFTER_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"
# The config carries no editor-specific gate: the freeze is joint, and
# run_petct_route_a_after_baseline.sh derives one CONFIRMATORY_FROZEN flag from
# these two sections before it runs either aggregator.  Same tuple as
# aggregate_petct_p2t_confirmatory.py so both sides fail closed identically.
GATED_CONFIG_SECTIONS = ("p2t", "statistics")
CONFIRMATORY_EXCLUDED_CONDITIONS = {
    "wrong_operation_OOD": "stress-only operation-token perturbation",
    "same_weight_wrong_scope": (
        "SAME-only descriptive scope diagnostic with a condition-specific "
        "episode set"
    ),
}


def _require_active_confirmatory_gate(config: Mapping[str, Any]) -> None:
    """Fail closed unless the frozen execution gate is open in every section.

    The gate used to be enforced only by run_petct_route_a_after_baseline.sh,
    so calling this CLI directly emitted an editor verdict while the gate was
    still BLOCKED.  Mirror the shell's requirement here.
    """

    for section in GATED_CONFIG_SECTIONS:
        block = config.get(section)
        if (
            not isinstance(block, Mapping)
            or block.get("confirmatory_execution_gate") != CONFIRMATORY_GATE_ACTIVE
        ):
            raise RuntimeError(
                f"{section}.confirmatory_execution_gate must be "
                f"{CONFIRMATORY_GATE_ACTIVE} before an editor confirmatory verdict"
            )


def _reject_confirmatory_excluded(condition: str) -> None:
    reason = CONFIRMATORY_EXCLUDED_CONDITIONS.get(condition)
    if reason is not None:
        raise RuntimeError(
            f"{condition} is {reason} and forbidden in confirmatory analysis"
        )


def _regular_record(path: Path, *, label: str) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing non-symlink regular editor {label}: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"missing regular editor {label}: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _row_singleton(rows: Sequence[Mapping[str, Any]], field: str) -> Any:
    values = {json.dumps(row.get(field), sort_keys=True) for row in rows}
    if len(values) != 1:
        raise RuntimeError(f"editor metric rows mix {field}")
    return rows[0].get(field)


def patient_means(
    rows: Sequence[Mapping[str, object]], metric: str
) -> Dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(float(value))
        ):
            raise ValueError(f"metric {metric} must contain only finite numbers or null")
        patient = str(row.get("patient_id") or "")
        if not patient:
            raise ValueError("metric row omits patient_id")
        grouped[patient].append(float(value))
    return {patient: float(np.mean(values)) for patient, values in grouped.items()}


def paired_cluster_bootstrap(
    treatment: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    seed: int,
    samples: int,
    alpha: float,
) -> Dict[str, object]:
    """Compatibility helper for one-seed additive metrics (not used by main)."""

    patients = sorted(set(treatment) & set(baseline))
    if len(patients) < 2 or samples < 1 or not 0 < alpha < 1:
        raise ValueError("paired analysis needs at least two shared patients")
    delta = np.asarray([treatment[p] - baseline[p] for p in patients], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(samples, len(delta)))
    boot = delta[draws].mean(axis=1)
    return {
        "paired_patient_count": len(patients),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "ci_low": float(np.quantile(boot, alpha / 2)),
        "ci_high": float(np.quantile(boot, 1 - alpha / 2)),
        "bootstrap_samples": samples,
        "alpha": alpha,
        "bootstrap_seed": seed,
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_signflip_pvalue(
    treatment: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    seed: int,
    samples: int,
    alternative: str,
    null_margin: Optional[float] = None,
) -> Dict[str, object]:
    """Compatibility wrapper using whole-patient cluster assignment swaps."""

    patients = sorted(set(treatment) & set(baseline))
    delta = {patient: treatment[patient] - baseline[patient] for patient in patients}
    result = patient_cluster_permutation(
        delta,
        seed=seed,
        samples=samples,
        alternative=alternative,
        null_margin=0.0 if null_margin is None else float(null_margin),
    )
    return {
        "paired_patient_count": len(patients),
        "observed_mean_delta": result["estimate"],
        "null_margin": result["null_margin"],
        "observed_mean_delta_minus_margin": result["observed_minus_margin"],
        "alternative": alternative,
        "p_value": result["p_value"],
        "permutation_samples": samples,
        "permutation_seed": seed,
    }


def patient_cluster_inference(
    patient_delta: Mapping[str, float],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
    alpha: float,
) -> dict[str, Any]:
    patients = sorted(patient_delta)
    if len(patients) < 2:
        raise RuntimeError("confirmatory contrast needs at least two paired patients")
    delta = np.asarray([patient_delta[patient] for patient in patients], dtype=float)
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
    bootstrap = delta[draws].mean(axis=1)
    return {
        "paired_patient_count": len(patients),
        "estimate": float(delta.mean()),
        "median_patient_delta": float(np.median(delta)),
        "ci_low": float(np.quantile(bootstrap, alpha / 2)),
        "ci_high": float(np.quantile(bootstrap, 1 - alpha / 2)),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "alpha": alpha,
        "bootstrap_unit": "patient cluster synchronously across conditions and seeds",
    }


def patient_cluster_permutation(
    patient_delta: Mapping[str, float],
    *,
    seed: int,
    samples: int,
    alternative: str,
    null_margin: float,
) -> dict[str, Any]:
    """Swap the paired condition assignment for each whole patient cluster."""

    if alternative not in {
        "greater",
        "less",
        "two-sided",
        "noninferiority",
        "noninferiority-upper",
    }:
        raise ValueError("invalid paired patient-cluster permutation alternative")
    patients = sorted(patient_delta)
    if len(patients) < 2 or samples < 1:
        raise ValueError("paired permutation needs at least two patients and one sample")
    raw = np.asarray([patient_delta[patient] for patient in patients], dtype=float)
    centered = raw - float(null_margin)
    observed = float(raw.mean())
    observed_centered = float(centered.mean())
    rng = np.random.default_rng(seed)
    # One assignment swap per patient.  The patient value already averages the
    # three registered seeds, so the same swap necessarily applies to all seeds.
    swaps = rng.integers(0, 2, size=(samples, len(patients)))
    null = (centered * np.where(swaps == 1, -1.0, 1.0)).mean(axis=1)
    if alternative in {"greater", "noninferiority"}:
        extreme = null >= observed_centered
    elif alternative in {"less", "noninferiority-upper"}:
        extreme = null <= observed_centered
    else:
        extreme = np.abs(null) >= abs(observed_centered)
    return {
        "estimate": observed,
        "null_margin": float(null_margin),
        "observed_minus_margin": observed_centered,
        "alternative": alternative,
        "p_value": float((1 + int(extreme.sum())) / (samples + 1)),
        "permutation_samples": samples,
        "permutation_seed": seed,
        "permutation_unit": "whole patient cluster synchronously across registered seeds",
        "null_transform": "sign_flip_of_patient_delta_minus_null_margin",
        "margin_adjustment": {
            "formula": "patient_delta - null_margin",
            "null_margin": float(null_margin),
        },
        "exchangeability_unit": "patient_cluster",
        "same_swap_across_registered_seeds": True,
    }


def _frozen_integer(statistics: Mapping[str, Any], key: str) -> int:
    value = statistics.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"statistics.{key} must be an integer >= 1")
    return int(value)


def _condition_seed_patient_means(
    rows_by_run: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
    *,
    seeds: Sequence[int],
    condition: str,
    metric: str,
) -> dict[int, dict[str, float]]:
    result = {
        int(seed): patient_means(rows_by_run[(int(seed), condition)], metric)
        for seed in seeds
    }
    patient_sets = {frozenset(values) for values in result.values()}
    if len(patient_sets) != 1:
        raise RuntimeError(
            f"condition {condition} has different defined-patient sets across seeds for {metric}"
        )
    return result


def _decision(
    *,
    rule: str,
    p_value_holm: float,
    alpha: float,
    estimate: float,
    threshold: float,
) -> bool:
    if rule == "holm_p_lt_alpha_and_point_estimate_ge_threshold":
        return p_value_holm < alpha and estimate >= threshold
    if rule == "holm_p_lt_alpha_and_point_estimate_gt_lower_margin":
        return p_value_holm < alpha and estimate > threshold
    if rule == "holm_p_lt_alpha_and_point_estimate_lt_upper_margin":
        return p_value_holm < alpha and estimate < threshold
    raise RuntimeError(f"unsupported explicit decision_rule: {rule}")


def _validate_run(
    *,
    seed: int,
    condition: str,
    checkpoint_path: Path,
    summary_path: Path,
    rows_path: Path,
    config_sha256: str,
    registered_seeds: Sequence[int],
) -> tuple[list[Mapping[str, Any]], dict[str, Any], dict[str, Any]]:
    for path, label in (
        (checkpoint_path, "checkpoint"),
        (summary_path, "summary"),
        (rows_path, "metric rows"),
    ):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"missing regular editor {label}: {path}")
    checkpoint_path = checkpoint_path.resolve()
    summary_path = summary_path.resolve()
    rows_path = rows_path.resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_checkpoint_condition = CHECKPOINT_CONDITION_ALIASES.get(condition, condition)
    if (
        checkpoint.get("schema_version") != EDITOR_CHECKPOINT_SCHEMA
        or int(checkpoint.get("seed", -1)) != seed
        or [int(value) for value in checkpoint.get("seed_registry", [])]
        != [int(value) for value in registered_seeds]
        or checkpoint.get("condition") != expected_checkpoint_condition
        or checkpoint.get("experiment_config_sha256") != config_sha256
        or checkpoint.get("status") != EDITOR_TRAINING_STATUS
    ):
        raise RuntimeError("editor checkpoint seed/condition/config contract mismatch")
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    rows = load_jsonl(rows_path)
    if not rows:
        raise RuntimeError("editor metric rows are empty")
    if (
        summary.get("schema_version")
        != "PETCT-BIDIRECTIONAL-CORRECTION-METRICS-v2.0"
        or summary.get("condition") != condition
        or summary.get("checkpoint_sha256") != checkpoint_sha256
        or summary.get("experiment_config_sha256") != config_sha256
        or summary.get("metric_rows_sha256") != sha256_file(rows_path)
        or int(summary.get("episode_count", -1)) != len(rows)
    ):
        raise RuntimeError("editor summary does not bind its rows/checkpoint/config")
    episode_ids = [str(row.get("episode_id") or "") for row in rows]
    if any(not episode for episode in episode_ids) or len(episode_ids) != len(
        set(episode_ids)
    ):
        raise RuntimeError("editor rows contain empty/duplicate episode_id values")
    singleton_fields = (
        "condition",
        "checkpoint_sha256",
        "experiment_config_sha256",
        "learning_manifest_sha256",
        "learning_split_sha256",
        "partition",
        "test_access_receipt_sha256",
        "threshold",
        "distal_radius_mm",
        "metric_grid",
    )
    for field in singleton_fields:
        values = {json.dumps(row.get(field), sort_keys=True) for row in rows}
        if len(values) != 1 or rows[0].get(field) != summary.get(field):
            raise RuntimeError(f"editor rows/summary disagree on {field}")
    for field in ("learning_manifest_sha256", "learning_split_sha256"):
        if not SHA256_RE.fullmatch(str(summary.get(field) or "")):
            raise RuntimeError(f"editor summary has invalid {field}")
    partition = summary.get("partition")
    if partition not in {"val", "test"}:
        raise RuntimeError("editor summary partition must be val or test")
    test_access_sha256 = summary.get("test_access_receipt_sha256")
    if partition == "test":
        if not SHA256_RE.fullmatch(str(test_access_sha256 or "")):
            raise RuntimeError("test editor run requires a valid test-access receipt hash")
    elif test_access_sha256 is not None:
        raise RuntimeError("validation editor run must not carry a test-access receipt")
    training_manifest_sha256 = _row_singleton(
        rows, "training_manifest_sha256"
    )
    inference_manifest_sha256 = _row_singleton(
        rows, "inference_manifest_sha256"
    )
    learning_manifest_sha256 = _row_singleton(rows, "learning_manifest_sha256")
    for field, value in (
        ("training_manifest_sha256", training_manifest_sha256),
        ("inference_manifest_sha256", inference_manifest_sha256),
        ("learning_manifest_sha256", learning_manifest_sha256),
    ):
        if not SHA256_RE.fullmatch(str(value or "")):
            raise RuntimeError(f"editor rows have invalid {field}")
    if learning_manifest_sha256 != inference_manifest_sha256:
        raise RuntimeError("editor learning manifest must alias the inference manifest")
    if (
        checkpoint.get("manifest_sha256") != training_manifest_sha256
        or checkpoint.get("training_manifest_sha256")
        != training_manifest_sha256
    ):
        raise RuntimeError("editor checkpoint does not bind the training manifest")
    if checkpoint.get("learning_split_sha256") != summary["learning_split_sha256"]:
        raise RuntimeError("editor checkpoint and summary use different learning splits")

    training_manifest = _regular_record(
        Path(str(_row_singleton(rows, "training_manifest") or "")),
        label="training manifest",
    )
    inference_manifest = _regular_record(
        Path(str(_row_singleton(rows, "inference_manifest") or "")),
        label="inference manifest",
    )
    learning_manifest = _regular_record(
        Path(str(_row_singleton(rows, "learning_manifest") or "")),
        label="learning manifest",
    )
    prediction_manifest = _regular_record(
        Path(str(_row_singleton(rows, "prediction_manifest") or "")),
        label="prediction manifest",
    )
    learning_split = _regular_record(
        Path(str(checkpoint.get("learning_split") or "")),
        label="learning split",
    )
    if training_manifest["sha256"] != training_manifest_sha256:
        raise RuntimeError("current editor training manifest hash changed")
    if inference_manifest["sha256"] != inference_manifest_sha256:
        raise RuntimeError("current editor inference manifest hash changed")
    if learning_manifest != inference_manifest:
        raise RuntimeError("editor learning/inference manifest paths differ")
    if prediction_manifest["sha256"] != summary.get("prediction_manifest_sha256"):
        raise RuntimeError("editor summary does not bind the current prediction manifest")
    if learning_split["sha256"] != summary["learning_split_sha256"]:
        raise RuntimeError("current editor learning split hash changed")
    for field in ("manifest", "training_manifest"):
        checkpoint_manifest = _regular_record(
            Path(str(checkpoint.get(field) or "")),
            label=f"checkpoint {field}",
        )
        if checkpoint_manifest != training_manifest:
            raise RuntimeError(f"editor checkpoint {field} path/hash mismatch")

    prediction_rows = load_jsonl(Path(prediction_manifest["path"]))
    prediction_by_episode = {
        str(row.get("episode_id") or ""): row for row in prediction_rows
    }
    metric_by_episode = {str(row["episode_id"]): row for row in rows}
    if (
        "" in prediction_by_episode
        or len(prediction_by_episode) != len(prediction_rows)
        or set(prediction_by_episode) != set(metric_by_episode)
    ):
        raise RuntimeError("editor prediction/metric episode inventories differ")
    learning_rows = load_jsonl(Path(inference_manifest["path"]))
    learning_by_episode = {
        str(row.get("episode_id") or ""): row
        for row in learning_rows
        if row.get("partition") == partition
    }
    if "" in learning_by_episode or set(learning_by_episode) != set(metric_by_episode):
        raise RuntimeError("editor learning manifest does not exactly cover metric rows")

    episode_leaf_artifacts: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if not str(row.get("patient_id") or ""):
            raise RuntimeError("editor metric row omits patient_id")
        for field in ("visible_npz_sha256", "evaluation_npz_sha256"):
            if not SHA256_RE.fullmatch(str(row.get(field) or "")):
                raise RuntimeError(f"editor metric row has invalid {field}")
        episode = str(row["episode_id"])
        prediction_row = prediction_by_episode[episode]
        if any(row.get(field) != value for field, value in prediction_row.items()):
            raise RuntimeError("editor metric row changed prediction provenance")
        learning_row = learning_by_episode[episode]
        if str(learning_row.get("patient_id") or "") != str(row["patient_id"]):
            raise RuntimeError("editor episode-to-patient mapping changed")
        expected_learning = {
            "visible_npz": row.get("visible_npz"),
            "visible_sha256": row.get("visible_npz_sha256"),
            "evaluation_npz": row.get("evaluation_npz"),
            "evaluation_sha256": row.get("evaluation_npz_sha256"),
        }
        if any(learning_row.get(field) != value for field, value in expected_learning.items()):
            raise RuntimeError("editor metric row differs from its learning manifest")
        leaf_records = {
            "prediction_npz": _regular_record(
                Path(str(row.get("prediction_npz") or "")),
                label=f"prediction NPZ for {episode}",
            ),
            "evaluation_npz": _regular_record(
                Path(str(row.get("evaluation_npz") or "")),
                label=f"evaluation NPZ for {episode}",
            ),
            "visible_npz": _regular_record(
                Path(str(row.get("visible_npz") or "")),
                label=f"visible NPZ for {episode}",
            ),
        }
        expected_hashes = {
            "prediction_npz": row.get("prediction_npz_sha256"),
            "evaluation_npz": row.get("evaluation_npz_sha256"),
            "visible_npz": row.get("visible_npz_sha256"),
        }
        if any(
            leaf_records[role]["sha256"] != expected_hash
            for role, expected_hash in expected_hashes.items()
        ):
            raise RuntimeError("editor leaf artifact hash changed")
        episode_leaf_artifacts[episode] = leaf_records
    summary = dict(summary)
    summary["training_manifest_sha256"] = training_manifest_sha256
    summary["inference_manifest_sha256"] = inference_manifest_sha256
    return rows, summary, {
        "seed": seed,
        "condition": condition,
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256},
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "rows": {"path": str(rows_path), "sha256": sha256_file(rows_path)},
        "leaf_artifacts": {
            "training_manifest": training_manifest,
            "inference_manifest": inference_manifest,
            "learning_split": learning_split,
            "prediction_manifest": prediction_manifest,
            "episodes": episode_leaf_artifacts,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=5,
        required=True,
        metavar=("SEED", "CONDITION", "CHECKPOINT", "SUMMARY", "ROWS"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
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
    statistics = config["statistics"]
    seeds = [int(seed) for seed in config["editor"]["training"]["seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise RuntimeError("editor confirmatory analysis requires exactly 3 registered seeds")
    alpha = statistics.get("alpha")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not np.isfinite(float(alpha))
        or not 0 < float(alpha) < 1
    ):
        raise RuntimeError("statistics.alpha must be finite and between zero and one")
    alpha = float(alpha)
    bootstrap_samples = _frozen_integer(statistics, "bootstrap_samples")
    permutation_samples = _frozen_integer(statistics, "permutation_samples")
    bootstrap_seed = int(statistics["bootstrap_seed"])
    permutation_seed = int(statistics["permutation_seed"])
    contrasts = statistics.get("confirmatory_contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        raise RuntimeError("statistics.confirmatory_contrasts must be non-empty")
    required_conditions = {
        str(contrast[key]) for contrast in contrasts for key in ("treatment", "comparator")
    }
    for condition in sorted(required_conditions):
        _reject_confirmatory_excluded(condition)
    same_weight_families = {
        str(name): {str(member) for member in members}
        for name, members in statistics["confirmatory_same_weight_families"].items()
    }
    expected = {
        (int(seed), condition) for seed in seeds for condition in required_conditions
    }
    observed: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    summaries: dict[tuple[int, str], Mapping[str, Any]] = {}
    input_runs = []
    for raw_seed, condition, checkpoint_raw, summary_raw, rows_raw in args.run:
        seed = int(raw_seed)
        _reject_confirmatory_excluded(str(condition))
        key = (seed, str(condition))
        if key not in expected or key in observed:
            raise RuntimeError(
                "editor runs must exactly cover registered seeds x confirmatory conditions"
            )
        rows, summary, record = _validate_run(
            seed=seed,
            condition=str(condition),
            checkpoint_path=Path(checkpoint_raw),
            summary_path=Path(summary_raw),
            rows_path=Path(rows_raw),
            config_sha256=config_sha256,
            registered_seeds=seeds,
        )
        observed[key] = rows
        summaries[key] = summary
        input_runs.append(record)
    if set(observed) != expected:
        raise RuntimeError(
            "editor runs must exactly cover all 3 registered seeds x confirmatory conditions"
        )

    common_provenance_fields = (
        "training_manifest_sha256",
        "inference_manifest_sha256",
        "learning_manifest_sha256",
        "learning_split_sha256",
        "partition",
        "test_access_receipt_sha256",
    )
    common_provenance: dict[str, Any] = {}
    for field in common_provenance_fields:
        values = {
            json.dumps(summary.get(field), sort_keys=True)
            for summary in summaries.values()
        }
        if len(values) != 1:
            raise RuntimeError(
                f"editor conditions/seeds do not share one frozen {field}"
            )
        common_provenance[field] = next(iter(summaries.values())).get(field)

    reference_key = (seeds[0], sorted(required_conditions)[0])
    reference_episode_provenance = {
        str(row["episode_id"]): {
            "patient_id": str(row["patient_id"]),
            "visible_npz_sha256": str(row["visible_npz_sha256"]),
            "evaluation_npz_sha256": str(row["evaluation_npz_sha256"]),
        }
        for row in observed[reference_key]
    }
    reference_map = {
        episode: record["patient_id"]
        for episode, record in reference_episode_provenance.items()
    }
    for key, rows in observed.items():
        mapping = {
            str(row["episode_id"]): {
                "patient_id": str(row["patient_id"]),
                "visible_npz_sha256": str(row["visible_npz_sha256"]),
                "evaluation_npz_sha256": str(row["evaluation_npz_sha256"]),
            }
            for row in rows
        }
        if len(mapping) != len(rows) or mapping != reference_episode_provenance:
            raise RuntimeError(
                f"editor episode patient/visible/evaluation provenance differs for {key}"
            )
    common_grids = {
        str(summary["metric_grid"]) for summary in summaries.values()
    }
    if len(common_grids) != 1:
        raise RuntimeError("editor conditions use different metric_grid values")

    checkpoint_sha_by_run = {
        (int(record["seed"]), str(record["condition"])): str(
            record["checkpoint"]["sha256"]
        )
        for record in input_runs
    }
    checkpoint_alias_groups: dict[str, list[str]] = defaultdict(list)
    for condition in sorted(required_conditions):
        checkpoint_condition = CHECKPOINT_CONDITION_ALIASES.get(condition, condition)
        checkpoint_alias_groups[checkpoint_condition].append(condition)
    same_weight_checkpoint_groups = []
    for checkpoint_condition, conditions in sorted(checkpoint_alias_groups.items()):
        if len(conditions) < 2:
            continue
        for seed in seeds:
            hashes = {
                checkpoint_sha_by_run[(int(seed), condition)]
                for condition in conditions
            }
            if len(hashes) != 1:
                raise RuntimeError(
                    "same-weight editor aliases must use one identical checkpoint "
                    f"SHA per seed: seed={seed}, conditions={conditions}"
                )
            same_weight_checkpoint_groups.append(
                {
                    "seed": int(seed),
                    "checkpoint_condition": checkpoint_condition,
                    "conditions": conditions,
                    "checkpoint_sha256": next(iter(hashes)),
                }
            )

    comparisons = []
    thresholds = statistics.get("effect_thresholds")
    if not isinstance(thresholds, Mapping):
        raise RuntimeError("statistics.effect_thresholds must be an object")
    for contrast_index, contrast in enumerate(contrasts):
        treatment = str(contrast["treatment"])
        comparator = str(contrast["comparator"])
        metric = str(contrast["metric"])
        family = str(contrast["family"])
        if family not in same_weight_families or {
            treatment,
            comparator,
        } - same_weight_families[family]:
            raise RuntimeError("contrast members are outside their declared family")
        threshold_ref = contrast.get("threshold_ref")
        decision_rule = contrast.get("decision_rule")
        if not isinstance(threshold_ref, str) or threshold_ref not in thresholds:
            raise RuntimeError("every contrast requires an explicit valid threshold_ref")
        if not isinstance(decision_rule, str) or not decision_rule:
            raise RuntimeError("every contrast requires an explicit decision_rule")
        threshold = thresholds[threshold_ref]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not np.isfinite(float(threshold))
        ):
            raise RuntimeError("contrast threshold_ref resolves to a non-finite value")
        threshold = float(threshold)
        alternative = str(contrast["alternative"])
        expected_rule = {
            "greater": "holm_p_lt_alpha_and_point_estimate_ge_threshold",
            "noninferiority": "holm_p_lt_alpha_and_point_estimate_gt_lower_margin",
            "noninferiority-upper": "holm_p_lt_alpha_and_point_estimate_lt_upper_margin",
        }.get(alternative)
        if decision_rule != expected_rule:
            raise RuntimeError("contrast alternative and decision_rule are inconsistent")
        inline_threshold = (
            contrast.get("minimum_absolute_effect")
            if alternative == "greater"
            else contrast.get("noninferiority_margin")
            if alternative == "noninferiority"
            else contrast.get("noninferiority_margin_mm2")
        )
        if inline_threshold not in (None, threshold):
            raise RuntimeError("inline contrast threshold differs from threshold_ref")
        treatment_means = _condition_seed_patient_means(
            observed, seeds=seeds, condition=treatment, metric=metric
        )
        comparator_means = _condition_seed_patient_means(
            observed, seeds=seeds, condition=comparator, metric=metric
        )
        patient_sets = {
            frozenset(values)
            for values in (*treatment_means.values(), *comparator_means.values())
        }
        if len(patient_sets) != 1:
            raise RuntimeError(
                f"contrast {treatment} vs {comparator} has different defined-patient sets"
            )
        patients = sorted(next(iter(patient_sets)))
        patient_delta = {
            patient: float(
                np.mean(
                    [
                        treatment_means[seed][patient]
                        - comparator_means[seed][patient]
                        for seed in seeds
                    ]
                )
            )
            for patient in patients
        }
        inference = patient_cluster_inference(
            patient_delta,
            bootstrap_seed=bootstrap_seed + contrast_index,
            bootstrap_samples=bootstrap_samples,
            alpha=alpha,
        )
        null_margin = threshold if alternative.startswith("noninferiority") else 0.0
        permutation = patient_cluster_permutation(
            patient_delta,
            seed=permutation_seed + contrast_index,
            samples=permutation_samples,
            alternative=alternative,
            null_margin=null_margin,
        )
        comparisons.append(
            {
                "family": family,
                "treatment": treatment,
                "comparator": comparator,
                "metric": metric,
                "alternative": alternative,
                "threshold_ref": threshold_ref,
                "threshold": threshold,
                "decision_rule": decision_rule,
                "per_seed_descriptive": [
                    {
                        "seed": seed,
                        "mean_paired_delta": float(
                            np.mean(
                                [
                                    treatment_means[seed][patient]
                                    - comparator_means[seed][patient]
                                    for patient in patients
                                ]
                            )
                        ),
                    }
                    for seed in seeds
                ],
                **inference,
                **{
                    key: value
                    for key, value in permutation.items()
                    if key not in {"estimate"}
                },
            }
        )
    family_holm_applications: dict[str, int] = {}
    for family in sorted({row["family"] for row in comparisons}):
        indices = [i for i, row in enumerate(comparisons) if row["family"] == family]
        adjusted = holm_adjust([comparisons[index]["p_value"] for index in indices])
        family_holm_applications[family] = 1
        for index, value in zip(indices, adjusted):
            comparison = comparisons[index]
            comparison["p_value_holm"] = value
            supported = _decision(
                rule=str(comparison["decision_rule"]),
                p_value_holm=value,
                alpha=alpha,
                estimate=float(comparison["estimate"]),
                threshold=float(comparison["threshold"]),
            )
            comparison["hypothesis_verdict"] = (
                "SUPPORTED" if supported else "NOT_SUPPORTED"
            )
    family_verdicts = {
        family: (
            "SUPPORTED"
            if all(
                row["hypothesis_verdict"] == "SUPPORTED"
                for row in comparisons
                if row["family"] == family
            )
            else "NOT_SUPPORTED"
        )
        for family in sorted({row["family"] for row in comparisons})
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "analysis_status": "VALID",
        "seed_estimand": {
            "registered_seeds": seeds,
            "role": "fixed_algorithm_replicates",
            "population_inference_over_seeds": False,
            "aggregation": "patient condition means, paired delta per seed, then mean over registered seeds",
        },
        "comparisons": comparisons,
        "family_verdicts": family_verdicts,
        "multiplicity": statistics["multiplicity"],
        "holm_applications_per_family": family_holm_applications,
        "metric_grid": next(iter(common_grids)),
        "provenance_lock": common_provenance,
        "same_weight_checkpoint_groups": same_weight_checkpoint_groups,
        "episode_to_patient_mapping_sha256": hashlib.sha256(
            json.dumps(
                reference_map, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "episode_provenance_mapping_sha256": hashlib.sha256(
            json.dumps(
                reference_episode_provenance,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "experiment_config_sha256": config_sha256,
        "input_runs": sorted(
            input_runs, key=lambda item: (item["seed"], item["condition"])
        ),
        "statistical_parameters": {
            "alpha": alpha,
            "bootstrap_samples": bootstrap_samples,
            "permutation_samples": permutation_samples,
            "bootstrap_seed": bootstrap_seed,
            "permutation_seed": permutation_seed,
            "contrast_seed_policy": "frozen_base_seed_plus_zero_based_contrast_index",
        },
    }
    json.dumps(payload, allow_nan=False, sort_keys=True)
    write_json_exclusive(args.output, payload)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
