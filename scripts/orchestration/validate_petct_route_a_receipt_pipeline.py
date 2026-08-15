#!/usr/bin/env python3
"""Validate Route A's two experiment lanes from immutable artifact receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


SCRIPTS = Path(__file__).resolve().parents[1]
for directory in (
    SCRIPTS,
    SCRIPTS / "common",
    SCRIPTS / "data",
    SCRIPTS / "baseline",
    SCRIPTS / "p2t",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from baseline.validate_petct_m0_oof import (  # noqa: E402
    validate_oof_ready_receipt_only,
)
from common.petct_learning import (  # noqa: E402
    EDITOR_CHECKPOINT_SCHEMA,
    P2T_CHECKPOINT_SCHEMA,
    P2T_METRICS_SCHEMA,
    load_editor_architecture_contract,
    load_jsonl,
    sha256_file,
)
from common.petct_models import EDITOR_PRIMARY_ARCHITECTURE_ID  # noqa: E402
from common.petct_development_freeze import (  # noqa: E402
    DevelopmentFreezeError,
    SELECTED_CHECKPOINT_PREFIX,
    TRAINED_CHECKPOINT_STATUS,
    validate_frozen_checkpoint_bindings,
)
from common.petct_route_a_core import (  # noqa: E402
    EDITOR_CHECKPOINT_CONDITION_ALIASES,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    validate_consumed_receipt,
)
from data.build_petct_scribble_dataset import (  # noqa: E402
    GENERATION_STAGE_ORDER,
    validate_residual_ready,
)
from data.build_petct_scribble_episode import (  # noqa: E402
    assign_scribble_strategy,
)
from data.validate_petct_learning_split import (  # noqa: E402
    load_and_validate_learning_split,
)
from p2t.build_petct_matched_state_dataset import (  # noqa: E402
    CONTROLLED_STAGE_ORDER,
    MATCHED_STATE_SCHEMA,
)
from validate_petct_route_a_f0 import (  # noqa: E402
    BLOCKER_IDS as F0_BLOCKER_IDS,
    F0Error,
    validate_receipt as validate_f0_receipt,
)


PIPELINE_RECEIPT_SCHEMA = "PETCT-ROUTE-A-PIPELINE-RECEIPT-v2.0"
M0_EVALUATION_READY_SCHEMA = "PETCT-M0-EVALUATION-READY-v1.0"
ROBUSTNESS_ALL_READY_SCHEMA = "PETCT-SCRIBBLE-ROBUSTNESS-ALL-READY-v1.0"
P2T_CONFIRMATORY_SCHEMA = "PETCT-P2T-CONFIRMATORY-v1.0"
EDITOR_CONFIRMATORY_SCHEMA = "PETCT-BIDIRECTIONAL-EDITOR-CONFIRMATORY-v2.0"
P2T_DESCRIPTIVE_SCHEMA = "PETCT-P2T-DESCRIPTIVE-v2.0"
EDITOR_DESCRIPTIVE_SCHEMA = "PETCT-EDITOR-DESCRIPTIVE-v2.0"
M0_EVALUATION_SCHEMA = "PETCT-M0-OOF-EVALUATION-v1.1"
GOALS = {
    "ADD_SAME_LOCAL",
    "ADD_SAME_COMPLETE",
    "ADD_NEW_COMPLETE",
    "REMOVE_SAME_LOCAL",
    "REMOVE_SAME_COMPLETE",
    "REMOVE_NEW_COMPLETE",
}
TARGETS = (
    "m0_evaluation",
    "p2t_data",
    "p2t_results",
    "editor_data",
    "editor_results",
    "complete",
    "robustness_all",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _regular(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing non-symlink regular {label}: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"missing regular {label}: {resolved}")
    return resolved


def _directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"missing non-symlink directory {label}: {path}")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise RuntimeError(f"missing directory {label}: {resolved}")
    return resolved


def _resolve(base: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"pipeline inputs require {label}")
    path = Path(value)
    return _regular(path if path.is_absolute() else base / path, label=label)


def _resolve_many(base: Path, value: Any, *, label: str) -> list[Path]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"pipeline inputs require non-empty {label}")
    return [_resolve(base, item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def _record(path: Path) -> dict[str, str]:
    path = _regular(path, label="bound artifact")
    return {"path": str(path), "sha256": sha256_file(path)}


def _canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_live_file_record(
    value: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    required_root: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a file record")
    path = _regular(Path(str(value.get("path") or "")), label=label)
    if expected_path is not None and path != expected_path.resolve():
        raise RuntimeError(f"{label} path mismatch")
    if required_root is not None and not path.is_relative_to(required_root.resolve()):
        raise RuntimeError(f"{label} escapes the receipt-bound run root")
    current = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if any(value.get(key) != current[key] for key in current):
        raise RuntimeError(f"{label} content binding mismatch")
    return current


def _validate_live_tree_record(
    value: Any, *, label: str, required_root: Path
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a tree record")
    raw = Path(str(value.get("path") or ""))
    if raw.is_symlink() or not raw.is_dir():
        raise RuntimeError(f"{label} tree is missing or is a symlink")
    root = raw.resolve()
    if not root.is_relative_to(required_root.resolve()):
        raise RuntimeError(f"{label} tree escapes the receipt-bound run root")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"{label} tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"{label} tree contains a non-regular entry")
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    current = {
        "path": str(root),
        "file_count": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "tree_sha256": _canonical_json_hash(entries),
    }
    if any(value.get(key) != current[key] for key in current):
        raise RuntimeError(f"{label} tree binding mismatch")
    return current


def _validate_cohort_bucket(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a cohort bucket")
    case_ids = value.get("case_ids")
    patient_ids = value.get("patient_ids")
    if (
        not isinstance(case_ids, list)
        or case_ids != sorted(set(case_ids))
        or not all(isinstance(item, str) and item for item in case_ids)
        or not isinstance(patient_ids, list)
        or patient_ids != sorted(set(patient_ids))
        or not all(isinstance(item, str) and item for item in patient_ids)
    ):
        raise RuntimeError(f"{label} IDs must be sorted unique strings")
    expected = {
        "case_count": len(case_ids),
        "patient_count": len(patient_ids),
        "case_ids_sha256": _canonical_json_hash(case_ids),
        "patient_ids_sha256": _canonical_json_hash(patient_ids),
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError(f"{label} count/hash closure mismatch")
    return dict(value)


def _validate_episode_data_ready(
    ready_path: Path,
    *,
    schema_version: str,
    phase: str,
    lane: str,
    strategy_mode: str,
    selected_partitions: set[str],
    manifest_path: Path,
    run_root: Path,
    expected_input_files: Mapping[str, Path],
    goals_per_generated_attempt: int,
) -> dict[str, Any]:
    """Revalidate a native episode-materialization receipt and its denominator."""

    ready_path = _regular(ready_path, label=f"{lane} data-ready receipt")
    document = _load_json(ready_path)
    if not isinstance(document, Mapping):
        raise RuntimeError(f"{lane} data-ready receipt must be an object")
    expected_header = {
        "schema_version": schema_version,
        "status": "PASS",
        "phase": phase,
        "lane": lane,
        "strategy_mode": strategy_mode,
    }
    if any(document.get(key) != value for key, value in expected_header.items()):
        raise RuntimeError(f"{lane} data-ready header mismatch")
    if set(document.get("selected_partitions", [])) != selected_partitions:
        raise RuntimeError(f"{lane} data-ready selected partitions mismatch")
    unbound = dict(document)
    binding_sha256 = unbound.pop("binding_sha256", None)
    if binding_sha256 != _canonical_json_hash(unbound):
        raise RuntimeError(f"{lane} data-ready binding_sha256 mismatch")

    inputs = document.get("inputs")
    outputs = document.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise RuntimeError(f"{lane} data-ready inputs/outputs are missing")
    for key, expected_path in expected_input_files.items():
        _validate_live_file_record(
            inputs.get(key), label=f"{lane}.inputs.{key}", expected_path=expected_path
        )
    _validate_live_file_record(
        outputs.get("manifest"),
        label=f"{lane}.outputs.manifest",
        expected_path=manifest_path,
        required_root=run_root,
    )
    exclusions_record = _validate_live_file_record(
        outputs.get("exclusions"),
        label=f"{lane}.outputs.exclusions",
        required_root=run_root,
    )
    for key, value in outputs.items():
        if key not in {"manifest", "exclusions"}:
            _validate_live_tree_record(
                value, label=f"{lane}.outputs.{key}", required_root=run_root
            )

    manifest_rows = load_jsonl(manifest_path)
    exclusion_rows = load_jsonl(Path(exclusions_record["path"]))
    manifest_attempt_ids = [str(row.get("attempt_id") or "") for row in manifest_rows]
    excluded_attempt_ids = [str(row.get("attempt_id") or "") for row in exclusion_rows]
    if (
        "" in manifest_attempt_ids
        or "" in excluded_attempt_ids
        or len(excluded_attempt_ids) != len(set(excluded_attempt_ids))
    ):
        raise RuntimeError(f"{lane} has missing/duplicate exclusion attempt IDs")
    generated_ids = set(manifest_attempt_ids)
    excluded_ids = set(excluded_attempt_ids)
    if generated_ids & excluded_ids:
        raise RuntimeError(f"{lane} generated/excluded attempt sets overlap")
    attempts = document.get("attempts")
    if not isinstance(attempts, Mapping):
        raise RuntimeError(f"{lane} data-ready attempts are missing")
    requested_list = attempts.get("requested_ids")
    generated_list = attempts.get("generated_ids")
    excluded_list = attempts.get("excluded_ids")
    for label, values in (
        ("requested", requested_list),
        ("generated", generated_list),
        ("excluded", excluded_list),
    ):
        if not isinstance(values, list) or values != sorted(set(values)):
            raise RuntimeError(f"{lane} {label} attempt inventory is not exact")
        if attempts.get(f"{label}_count") != len(values) or attempts.get(
            f"{label}_ids_sha256"
        ) != _canonical_json_hash(values):
            raise RuntimeError(f"{lane} {label} attempt count/hash mismatch")
    if (
        set(generated_list) != generated_ids
        or set(excluded_list) != excluded_ids
        or generated_ids | excluded_ids != set(requested_list)
        or attempts.get("episodes") != len(manifest_rows)
    ):
        raise RuntimeError(f"{lane} requested/generated/excluded closure mismatch")
    per_attempt = Counter(manifest_attempt_ids)
    if any(count != goals_per_generated_attempt for count in per_attempt.values()):
        raise RuntimeError(f"{lane} generated attempt episode multiplicity mismatch")
    if goals_per_generated_attempt == len(GOALS):
        goals_by_attempt: dict[str, set[str]] = defaultdict(set)
        for row in manifest_rows:
            goals_by_attempt[str(row["attempt_id"])].add(str(row.get("goal") or ""))
        if any(goals != GOALS for goals in goals_by_attempt.values()):
            raise RuntimeError(f"{lane} generated attempt goal inventory mismatch")
        if attempts.get("goals_per_generated_attempt") != len(GOALS):
            raise RuntimeError(f"{lane} goals-per-attempt receipt mismatch")

    expected_reason_summary: dict[str, Any] = {}
    reasons: dict[str, set[str]] = defaultdict(set)
    for row in exclusion_rows:
        reason = str(row.get("reason") or "")
        if not reason:
            raise RuntimeError(f"{lane} exclusion reason is missing")
        reasons[reason].add(str(row["attempt_id"]))
    for reason, ids in sorted(reasons.items()):
        ordered = sorted(ids)
        expected_reason_summary[reason] = {
            "count": len(ordered),
            "attempt_ids": ordered,
            "attempt_ids_sha256": _canonical_json_hash(ordered),
        }
    if document.get("exclusions_by_reason") != expected_reason_summary:
        raise RuntimeError(f"{lane} exclusion reason summary mismatch")

    cohort = document.get("cohort")
    if not isinstance(cohort, Mapping):
        raise RuntimeError(f"{lane} cohort register is missing")
    validated_cohort = {
        key: _validate_cohort_bucket(value, label=f"{lane}.cohort.{key}")
        for key, value in cohort.items()
    }
    return {
        **dict(document),
        "ready_path": str(ready_path),
        "ready_sha256": sha256_file(ready_path),
        "validated_cohort": validated_cohort,
        "validated_attempts": {
            "requested": len(requested_list),
            "generated": len(generated_list),
            "excluded": len(excluded_list),
        },
    }


def _validate_f0_binding(value: Any) -> dict[str, Any]:
    """Revalidate the launcher-gated F0 receipt and return its live binding."""

    record = _validate_live_file_record(value, label="f0_readiness")
    path = Path(record["path"])
    document = _load_json(path)
    if not isinstance(document, Mapping):
        raise RuntimeError("F0 readiness receipt must be an object")
    canonical = document.get("canonical_inputs")
    environment = document.get("environment")
    if not isinstance(canonical, Mapping) or not isinstance(environment, Mapping):
        raise RuntimeError("F0 readiness receipt omits canonical/environment bindings")
    runtime = canonical.get("official_autopetv_runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("F0 readiness receipt omits AutoPET V runtime binding")

    def record_path(container: Mapping[str, Any], key: str) -> Path:
        item = container.get(key)
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise RuntimeError(f"F0 readiness receipt omits {key} path")
        return Path(str(item["path"]))

    try:
        validated = validate_f0_receipt(
            receipt_path=path,
            project_root=Path(str(document.get("project_root") or "")),
            experiment_config=record_path(canonical, "experiment_config"),
            environment_marker=record_path(environment, "marker"),
            official_simulator=record_path(canonical, "official_simulator"),
            official_metrics=record_path(canonical, "official_metrics"),
            official_runtime_manifest=record_path(runtime, "manifest"),
        )
    except F0Error as error:
        raise RuntimeError(f"F0 readiness revalidation failed: {error}") from error
    if validated.get("closed_blocker_ids") != list(F0_BLOCKER_IDS):
        raise RuntimeError("F0 readiness no longer closes the exact seven blockers")
    if validated.get("receipt") != record:
        raise RuntimeError("F0 readiness input binding differs from live receipt")
    return validated


def _current_freeze_file_record(
    value: Any, *, label: str, expected_role: str | None = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a frozen file record")
    required = {"path", "sha256", "bytes"}
    if not required.issubset(value):
        raise RuntimeError(f"{label} omits path/sha256/bytes")
    if expected_role is not None and value.get("role") != expected_role:
        raise RuntimeError(f"{label} role mismatch")
    path = _regular(Path(str(value.get("path") or "")), label=label)
    current = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if any(value.get(key) != current[key] for key in required):
        raise RuntimeError(f"{label} changed after final freeze")
    return current


def _frozen_binding_index(bindings: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = bindings.get("checkpoints")
    if not isinstance(records, list) or not records:
        raise RuntimeError("frozen checkpoint bindings omit selected checkpoints")
    index: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("frozen checkpoint binding row must be an object")
        role = str(record.get("role") or "")
        if not role.startswith(SELECTED_CHECKPOINT_PREFIX) or role in index:
            raise RuntimeError("frozen checkpoint roles are missing or duplicated")
        index[role] = record
    if list(index) != bindings.get("selected_checkpoint_roles"):
        raise RuntimeError("frozen checkpoint role ordering/inventory mismatch")
    return index


def _checkpoint_role(kind: str, checkpoint: Mapping[str, Any]) -> str:
    seed = int(checkpoint.get("seed", -1))
    if kind in {"p2t", "p2t_secondary"}:
        return (
            f"{SELECTED_CHECKPOINT_PREFIX}p2t:"
            f"{checkpoint.get('architecture_id')}:{checkpoint.get('input_ablation')}:"
            f"seed{seed}"
        )
    if kind == "editor":
        return (
            f"{SELECTED_CHECKPOINT_PREFIX}editor:{checkpoint.get('condition')}:"
            f"{checkpoint.get('architecture_id')}:seed{seed}"
        )
    raise RuntimeError(f"unknown frozen checkpoint kind: {kind}")


def _validate_frozen_checkpoint_subset(
    *,
    bindings: Mapping[str, Any],
    checkpoint_paths: Sequence[Path],
    kind: str,
    expected_roles: set[str],
) -> dict[str, dict[str, str]]:
    binding_by_role = _frozen_binding_index(bindings)
    observed: dict[str, dict[str, str]] = {}
    for path in checkpoint_paths:
        path = _regular(path, label=f"{kind} frozen checkpoint")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(f"{kind} checkpoint must be a mapping")
        role = _checkpoint_role(kind, checkpoint)
        if role in observed or role not in expected_roles:
            raise RuntimeError(f"{kind} checkpoint role inventory is not exact")
        frozen = binding_by_role.get(role)
        if frozen is None:
            raise RuntimeError(f"{kind} checkpoint role is absent from final freeze")
        current = _current_freeze_file_record(
            frozen, label=role, expected_role=role
        )
        if current["path"] != str(path):
            raise RuntimeError(f"{kind} checkpoint path differs from final freeze")
        metadata = frozen.get("checkpoint_metadata")
        if not isinstance(metadata, Mapping) or any(
            checkpoint.get(field) != expected
            for field, expected in metadata.items()
        ):
            raise RuntimeError(f"{kind} checkpoint metadata differs from final freeze")
        training_manifest = frozen.get("training_manifest")
        current_training = _current_freeze_file_record(
            training_manifest,
            label=f"{role} training manifest",
            expected_role="training_manifest",
        )
        if (
            checkpoint.get("manifest_sha256") != current_training["sha256"]
            or Path(str(checkpoint.get("manifest") or "")).is_symlink()
            or Path(str(checkpoint.get("manifest") or "")).resolve()
            != Path(current_training["path"])
        ):
            raise RuntimeError(
                f"{kind} checkpoint training manifest differs from final freeze"
            )
        if kind == "editor" and (
            checkpoint.get("training_manifest_sha256")
            != current_training["sha256"]
            or Path(str(checkpoint.get("training_manifest") or "")).is_symlink()
            or Path(str(checkpoint.get("training_manifest") or "")).resolve()
            != Path(current_training["path"])
        ):
            raise RuntimeError(
                "editor checkpoint explicit training manifest differs from final freeze"
            )
        observed[role] = {"path": current["path"], "sha256": current["sha256"]}
    if set(observed) != expected_roles:
        raise RuntimeError(f"{kind} checkpoints do not exactly cover frozen roles")
    return observed


def _expected_frozen_checkpoint_roles(
    config: Mapping[str, Any], kind: str
) -> set[str]:
    if kind == "p2t":
        architecture = str(config["p2t"]["primary_architecture_id"])
        return {
            f"{SELECTED_CHECKPOINT_PREFIX}p2t:{architecture}:{arm}:seed{seed}"
            for seed in config["p2t"]["training"]["seeds"]
            for arm in config["p2t"]["simple_first_input_arms"]
        }
    if kind == "p2t_secondary":
        return set()
    if kind == "editor":
        architecture = load_editor_architecture_contract(config)[
            "primary_architecture_id"
        ]
        trained_conditions = set(config["editor"]["training_conditions"])
        seeds = config["editor"]["training"]["seeds"]
        return {
            f"{SELECTED_CHECKPOINT_PREFIX}editor:{condition}:{architecture}:seed{seed}"
            for seed in seeds
            for condition in trained_conditions
        }
    raise RuntimeError(f"unknown frozen checkpoint kind: {kind}")


def _record_counter(records: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    return Counter(
        (str(record.get("path") or ""), str(record.get("sha256") or ""))
        for record in records
    )


def _path_record_counter(paths: Sequence[Path]) -> Counter[tuple[str, str]]:
    return _record_counter([_record(path) for path in paths])


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, label=f"{label}[{index}]")


def _strategy_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    scribble = config.get("scribble")
    if not isinstance(scribble, Mapping):
        raise RuntimeError("experiment config omits scribble contract")
    expected = {
        "mode": scribble.get("primary_strategy_mode"),
        "assignment": scribble.get("primary_assignment"),
        "salt": scribble.get("primary_strategy_salt"),
        "strategies": list(scribble.get("strategies") or []),
    }
    if (
        expected["mode"] != "primary"
        or expected["assignment"] != "stable-patient-hash"
        or not isinstance(expected["salt"], str)
        or not expected["salt"]
        or set(expected["strategies"]) != {"centerline", "random", "boundary"}
    ):
        raise RuntimeError("frozen primary scribble assignment contract is invalid")
    return expected


def _validate_primary_strategy_rows(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    contract = _strategy_contract(config)
    patient_assignment: dict[str, str] = {}
    for row in rows:
        patient = str(row.get("patient_id") or "")
        strategy = str(row.get("strategy") or "")
        expected_strategy = assign_scribble_strategy(patient, salt=contract["salt"])
        generation = row.get("scribble_generation")
        if (
            not patient
            or row.get("strategy_mode") != contract["mode"]
            or row.get("strategy_assignment") != contract["assignment"]
            or row.get("strategy_salt") != contract["salt"]
            or strategy != expected_strategy
            or not isinstance(generation, Mapping)
            or generation.get("strategy_mode") != contract["mode"]
            or generation.get("strategy_assignment") != contract["assignment"]
            or generation.get("strategy_salt") != contract["salt"]
            or generation.get("selected_strategy") != strategy
        ):
            raise RuntimeError("primary stable-patient-hash strategy assignment mismatch")
        if patient in patient_assignment and patient_assignment[patient] != strategy:
            raise RuntimeError("one patient changes primary scribble strategy across cases")
        patient_assignment[patient] = strategy
    counts = {
        strategy: sum(value == strategy for value in patient_assignment.values())
        for strategy in contract["strategies"]
    }
    return {
        "schema_version": "PETCT-SCRIBBLE-PRIMARY-COVERAGE-v1.0",
        "assignment": contract["assignment"],
        "salt": contract["salt"],
        "patient_assignment_map_sha256": _canonical_sha256(patient_assignment),
        "all_rows_match_expected_assignment": True,
        "patient_count": len(patient_assignment),
        "counts_by_strategy": counts,
    }


def validate_controlled_episode_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config_sha256: str,
    split_sha256: str,
    case_to_partition: Mapping[str, str],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    episode_ids: set[str] = set()
    for row in rows:
        if row.get("schema_version") != MATCHED_STATE_SCHEMA or row.get("lane") != "controlled_p2t":
            raise RuntimeError("controlled manifest has wrong schema/lane")
        episode_id = str(row.get("episode_id") or "")
        group_id = str(row.get("matched_state_group_id") or "")
        if not episode_id or episode_id in episode_ids or not group_id:
            raise RuntimeError("controlled manifest has duplicate/missing episode or group")
        episode_ids.add(episode_id)
        case_id = str(row.get("case_id") or "")
        if row.get("partition") != case_to_partition.get(case_id):
            raise RuntimeError("controlled episode partition differs from frozen split")
        if row.get("experiment_config_sha256") != config_sha256:
            raise RuntimeError("controlled episode config hash mismatch")
        if row.get("learning_split_sha256") != split_sha256:
            raise RuntimeError("controlled episode split hash mismatch")
        generation = row.get("scribble_generation")
        if not isinstance(generation, Mapping) or generation.get("stage_order") != list(CONTROLLED_STAGE_ORDER):
            raise RuntimeError("controlled episode stage order is invalid")
        groups[group_id].append(row)
    if not groups:
        raise RuntimeError("controlled P2T manifest is empty")
    for group_id, group in groups.items():
        if len(group) != len(GOALS) or {str(row.get("goal")) for row in group} != GOALS:
            raise RuntimeError(f"matched group {group_id} is not one complete legal triplet")
        invariant_keys = (
            "case_id",
            "patient_id",
            "partition",
            "strategy",
            "shared_physical_scribble_sha256",
            "coordinates_xyz",
        )
        for key in invariant_keys:
            values = {json.dumps(row.get(key), sort_keys=True) for row in group}
            if len(values) != 1:
                raise RuntimeError(f"matched group {group_id} changes shared field {key}")
        if len({str(row.get("m0_sha256")) for row in group}) != 3:
            raise RuntimeError(f"matched group {group_id} does not have three M0 states")
    result = {
        "groups": len(groups),
        "episodes": len(episode_ids),
        "episode_ids": episode_ids,
    }
    if config is not None:
        result["strategy_coverage"] = _validate_primary_strategy_rows(rows, config)
    return result


def validate_natural_episode_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    config_sha256: str,
    split_sha256: str,
    oof_ready_sha256: str,
    case_to_partition: Mapping[str, str],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    episode_ids: set[str] = set()
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if not episode_id or episode_id in episode_ids:
            raise RuntimeError("natural manifest has duplicate/missing episode_id")
        episode_ids.add(episode_id)
        if str(row.get("goal")) not in GOALS:
            raise RuntimeError("natural episode has illegal goal")
        case_id = str(row.get("case_id") or "")
        if row.get("partition") != case_to_partition.get(case_id):
            raise RuntimeError("natural episode partition differs from frozen split")
        if row.get("experiment_config_sha256") != config_sha256:
            raise RuntimeError("natural episode config hash mismatch")
        if row.get("learning_split_sha256") != split_sha256:
            raise RuntimeError("natural episode split hash mismatch")
        provenance = row.get("m0_provenance")
        if not isinstance(provenance, Mapping) or provenance.get("kind") != "patient_excluded_oof":
            raise RuntimeError("natural editor episode lacks patient-excluded OOF provenance")
        if provenance.get("oof_ready_sha256") != oof_ready_sha256:
            raise RuntimeError("natural editor episode binds a different OOF_READY")
        generation = row.get("scribble_generation")
        if not isinstance(generation, Mapping) or generation.get("stage_order") != list(GENERATION_STAGE_ORDER):
            raise RuntimeError("natural episode did not preserve OOF->FN->scribble->intent order")
    if not episode_ids:
        raise RuntimeError("natural editor manifest is empty")
    result = {"episodes": len(episode_ids), "episode_ids": episode_ids}
    if config is not None:
        result["strategy_coverage"] = _validate_primary_strategy_rows(rows, config)
    return result


def validate_tensor_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_episode_ids: set[str],
    config_sha256: str,
    split_sha256: str,
) -> dict[str, Any]:
    observed: set[str] = set()
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if not episode_id or episode_id in observed:
            raise RuntimeError("tensor manifest has duplicate/missing episode_id")
        observed.add(episode_id)
        if row.get("experiment_config_sha256") != config_sha256:
            raise RuntimeError("tensor manifest config hash mismatch")
        if row.get("learning_split_sha256") != split_sha256:
            raise RuntimeError("tensor manifest split hash mismatch")
        for key in ("visible_npz", "evaluation_npz"):
            _regular(Path(str(row.get(key) or "")), label=key)
        for path_key, hash_key in (
            ("visible_npz", "visible_sha256"),
            ("evaluation_npz", "evaluation_sha256"),
        ):
            if sha256_file(Path(str(row[path_key]))) != row.get(hash_key):
                raise RuntimeError(f"tensor artifact changed: {path_key}")
    if observed != expected_episode_ids:
        raise RuntimeError("tensor manifest does not exactly cover episode manifest")
    return {"episodes": len(observed)}


def validate_p2t_metric_receipts(
    documents: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    inference_manifest_sha256: str,
    training_manifest_sha256_by_run_cell: Mapping[tuple[int, str], str],
    learning_split_sha256: str,
    partition: str,
    test_access_receipt_sha256: str | None,
) -> dict[str, Any]:
    p2t = config["p2t"]
    expected = {
        (int(seed), str(arm))
        for seed in p2t["training"]["seeds"]
        for arm in p2t["simple_first_input_arms"]
    }
    if set(training_manifest_sha256_by_run_cell) != expected or any(
        not isinstance(value, str) or len(value) != 64
        for value in training_manifest_sha256_by_run_cell.values()
    ):
        raise RuntimeError(
            "P2T checkpoint training manifests do not cover every frozen seed x ablation"
        )
    observed: set[tuple[int, str]] = set()
    for document in documents:
        if document.get("schema_version") != P2T_METRICS_SCHEMA:
            raise RuntimeError("P2T metrics schema mismatch")
        if document.get("experiment_config_sha256") != config_sha256:
            raise RuntimeError("P2T metrics config hash mismatch")
        key = (int(document.get("checkpoint_seed")), str(document.get("input_ablation")))
        if key not in expected:
            raise RuntimeError("P2T metrics contain an unknown seed/ablation run cell")
        if document.get("manifest_sha256") != inference_manifest_sha256:
            raise RuntimeError("P2T metrics use a different controlled tensor manifest")
        if (
            document.get("training_manifest_sha256")
            != training_manifest_sha256_by_run_cell[key]
            or document.get("inference_manifest_sha256")
            != inference_manifest_sha256
        ):
            raise RuntimeError(
                "P2T metrics do not bind the checkpoint training manifest and current inference manifest"
            )
        if document.get("partition") != partition:
            raise RuntimeError("P2T metrics partition mismatch")
        if document.get("learning_split_sha256") != learning_split_sha256:
            raise RuntimeError("P2T metrics learning-split hash mismatch")
        if (
            document.get("test_access_receipt_sha256")
            != test_access_receipt_sha256
        ):
            raise RuntimeError("P2T metrics test-access receipt mismatch")
        if key in observed:
            raise RuntimeError("duplicate P2T seed/ablation receipt")
        observed.add(key)
    if observed != expected:
        raise RuntimeError("P2T metrics do not cover every frozen seed x ablation")
    return {"metric_receipts": len(observed)}


def validate_editor_metric_receipts(
    documents: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    manifest_sha256: str,
    learning_split_sha256: str,
    partition: str,
    test_access_receipt_sha256: str | None,
) -> dict[str, Any]:
    expected_conditions = set(config["editor"]["conditions"])
    seed_count = len(config["editor"]["training"]["seeds"])
    architecture = load_editor_architecture_contract(config)[
        "primary_architecture_id"
    ]
    by_condition: dict[str, set[str]] = defaultdict(set)
    for document in documents:
        if (
            document.get("schema_version")
            != "PETCT-BIDIRECTIONAL-CORRECTION-METRICS-v2.0"
        ):
            raise RuntimeError("editor metrics schema mismatch")
        condition = str(document.get("condition") or "")
        if condition not in expected_conditions:
            raise RuntimeError("editor metrics contain an unfrozen condition")
        if document.get("learning_manifest_sha256") != manifest_sha256:
            raise RuntimeError("editor metrics use a different natural tensor manifest")
        if document.get("experiment_config_sha256") != config_sha256:
            raise RuntimeError("editor metrics config hash mismatch")
        if document.get("learning_split_sha256") != learning_split_sha256:
            raise RuntimeError("editor metrics learning-split hash mismatch")
        if document.get("partition") != partition:
            raise RuntimeError("editor metrics partition mismatch")
        if (
            document.get("test_access_receipt_sha256")
            != test_access_receipt_sha256
        ):
            raise RuntimeError("editor metrics test-access receipt mismatch")
        checkpoint_sha = str(document.get("checkpoint_sha256") or "")
        if len(checkpoint_sha) != 64:
            raise RuntimeError("editor metrics lack checkpoint hash")
        if document.get("architecture_id") != architecture:
            raise RuntimeError("editor metrics architecture mismatch")
        expected_analysis_role = (
            "OOD_STRESS_ONLY"
            if condition == "wrong_operation_OOD"
            else "PRIMARY_OR_ABLATION"
        )
        if document.get("analysis_role") != expected_analysis_role:
            raise RuntimeError("editor metrics analysis role mismatch")
        parameter_count = document.get("parameter_count")
        if (
            isinstance(parameter_count, bool)
            or not isinstance(parameter_count, int)
            or parameter_count <= 0
        ):
            raise RuntimeError("editor metrics omit positive parameter_count")
        if checkpoint_sha in by_condition[condition]:
            raise RuntimeError("duplicate editor checkpoint receipt within condition")
        by_condition[condition].add(checkpoint_sha)
    if set(by_condition) != expected_conditions or any(
        len(hashes) != seed_count for hashes in by_condition.values()
    ):
        raise RuntimeError("editor metrics do not cover every frozen condition x seed")
    return {
        "metric_receipts": sum(len(hashes) for hashes in by_condition.values()),
        "conditions": len(by_condition),
        "architecture_id": architecture,
    }


def validate_m0_evaluation(
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    oof: Mapping[str, Any],
    selected_partitions: Sequence[str],
    config_sha256: str,
    case_manifest_sha256: str,
    learning_split_sha256: str,
    official_metrics_sha256: str,
    evaluation_partition: str,
    test_access_receipt_sha256: str | None,
    run_root: Path,
) -> dict[str, Any]:
    selected = tuple(str(value) for value in selected_partitions)
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(value not in {"train", "val", "test"} for value in selected)
    ):
        raise RuntimeError("M0 evaluation partitions are invalid")
    expected = {
        str(row["case_id"]): row
        for row in source_rows
        if split["case_to_partition"][str(row["case_id"])] in selected
    }
    if not expected:
        raise RuntimeError("M0 evaluation scope contains no cases")
    observed: dict[str, Mapping[str, Any]] = {}
    positive_count = 0
    empty_count = 0
    for row in rows:
        _reject_nonfinite(row, label="M0 metric row")
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in observed or case_id not in expected:
            raise RuntimeError("M0 rows contain duplicate/out-of-scope case_id")
        source = expected[case_id]
        partition = split["case_to_partition"][case_id]
        oof_case = oof["cases"].get(case_id)
        if (
            str(row.get("patient_id") or "") != str(source["patient_id"]).casefold()
            or int(row.get("held_out_fold", -1)) != int(source["held_out_fold"])
            or row.get("partition") != partition
            or not isinstance(oof_case, Mapping)
            or str(oof_case.get("patient_id") or "").casefold()
            != str(source["patient_id"]).casefold()
            or int(oof_case.get("held_out_fold", -1))
            != int(source["held_out_fold"])
        ):
            raise RuntimeError("M0 row patient/fold/partition differs from frozen sources")
        gt_positive = row.get("gt_positive")
        if not isinstance(gt_positive, bool):
            raise RuntimeError("M0 row gt_positive must be boolean")
        if row.get("official_metric_eligible") is not gt_positive:
            raise RuntimeError("M0 official metric eligibility contradicts GT definedness")
        for name in ("tp", "fp", "fn", "gt_voxel_count", "prediction_voxel_count"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"M0 row {name} must be a non-negative integer")
        fpv = row.get("fpv_ml")
        if (
            isinstance(fpv, bool)
            or not isinstance(fpv, (int, float))
            or float(fpv) < 0
        ):
            raise RuntimeError("M0 row fpv_ml must be finite and non-negative")
        if gt_positive:
            positive_count += 1
            if row.get("official_metric_ineligibility_reason") is not None:
                raise RuntimeError("eligible M0 row has an ineligibility reason")
            for name in ("dice", "dmm_f1", "fnv_ml"):
                value = row.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise RuntimeError(f"positive-GT M0 row has undefined {name}")
            if (
                row.get("empty_gt_false_positive") is not None
                or row.get("empty_gt_prediction_volume_ml") is not None
            ):
                raise RuntimeError("positive-GT M0 row defines an empty-GT diagnostic")
        else:
            empty_count += 1
            if (
                row.get("official_metric_ineligibility_reason") != "EMPTY_GT"
                or any(row.get(name) is not None for name in ("dice", "dmm_f1", "fnv_ml"))
                or not isinstance(row.get("empty_gt_false_positive"), bool)
                or not isinstance(row.get("empty_gt_prediction_volume_ml"), (int, float))
            ):
                raise RuntimeError("empty-GT M0 row has invalid null/diagnostic definedness")
        observed[case_id] = row
    if set(observed) != set(expected):
        raise RuntimeError("M0 rows do not exactly cover the scoped frozen cases")

    _reject_nonfinite(summary, label="M0 summary")
    partition_counts = {
        partition: sum(row["partition"] == partition for row in rows)
        for partition in selected
    }
    official = summary.get("official_autoPETV")
    empty = summary.get("empty_gt_false_positive_diagnostics")
    test_access = summary.get("test_access")
    expected_selected = (
        ["train", "val", "test"]
        if evaluation_partition == "test"
        else ["train", "val"]
    )
    if (
        summary.get("schema_version") != M0_EVALUATION_SCHEMA
        or summary.get("status") != "COMPLETE_WITH_EXPLICIT_METRIC_ELIGIBILITY"
        or summary.get("selected_partitions") != list(selected)
        or list(selected) != expected_selected
        or summary.get("source_case_count") != len(source_rows)
        or summary.get("case_count") != len(rows)
        or summary.get("patient_count")
        != len({str(row["patient_id"]) for row in rows})
        or summary.get("partition_case_counts") != partition_counts
        or summary.get("oof_ready_sha256") != oof["ready_sha256"]
        or summary.get("case_manifest_sha256") != case_manifest_sha256
        or summary.get("learning_split_sha256") != learning_split_sha256
        or summary.get("experiment_config_sha256") != config_sha256
        or summary.get("official_metrics_sha256") != official_metrics_sha256
        or not isinstance(official, Mapping)
        or official.get("eligible_case_count") != positive_count
        or official.get("ineligible_empty_gt_case_count") != empty_count
        or official.get("connectivity") != 18
        or not isinstance(official.get("overlap_threshold"), (int, float))
        or not math.isfinite(float(official["overlap_threshold"]))
        or not isinstance(empty, Mapping)
        or empty.get("case_count") != empty_count
        or not isinstance(test_access, Mapping)
        or test_access.get("required") is not (evaluation_partition == "test")
        or test_access.get("consumed_receipt_sha256")
        != test_access_receipt_sha256
        or test_access.get("bound_run_root")
        != (str(run_root) if evaluation_partition == "test" else None)
    ):
        raise RuntimeError("M0 summary counts/provenance/definedness mismatch")
    for name in ("dsc", "dmm_f1_aggregated"):
        value = official.get(name)
        if positive_count and not isinstance(value, (int, float)):
            raise RuntimeError(f"M0 official summary omits defined {name}")
        if not positive_count and value is not None:
            raise RuntimeError(f"M0 official summary must null undefined {name}")
    return {
        "status": "EVALUATION_READY",
        "selected_partitions": list(selected),
        "case_count": len(rows),
        "patient_count": len({str(row["patient_id"]) for row in rows}),
        "positive_gt_defined_case_count": positive_count,
        "empty_gt_null_case_count": empty_count,
        "all_numeric_values_finite_or_explicitly_null": True,
        "overlap_threshold": float(official["overlap_threshold"]),
        "connectivity": 18,
    }


def _validate_upstream_receipt(
    path: Path,
    *,
    expected_target: str,
    common: Mapping[str, Any],
) -> dict[str, str]:
    document = _load_json(_regular(path, label=f"{expected_target} receipt"))
    if (
        not isinstance(document, Mapping)
        or document.get("status") != "PASS"
        or document.get("target") != expected_target
        or document.get("schema_version")
        not in {PIPELINE_RECEIPT_SCHEMA, M0_EVALUATION_READY_SCHEMA}
        or document.get("common") != common
        or not isinstance(document.get("artifact_bindings"), Mapping)
        or not isinstance(document.get("upstream_receipts"), Mapping)
    ):
        raise RuntimeError(f"upstream {expected_target} receipt contract mismatch")
    _reject_nonfinite(document, label=f"upstream {expected_target} receipt")
    _validate_embedded_file_records(
        document.get("artifact_bindings"),
        label=f"upstream {expected_target}.artifact_bindings",
    )
    _validate_embedded_file_records(
        document.get("upstream_receipts"),
        label=f"upstream {expected_target}.upstream_receipts",
    )
    return _record(path)


def _validate_embedded_file_records(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        record_fields = {"path", "sha256"} & set(value)
        if record_fields:
            if set(value) != {"path", "sha256"}:
                raise RuntimeError(
                    f"{label} embedded file record must contain exactly path/sha256"
                )
            if not isinstance(value["path"], str) or not isinstance(
                value["sha256"], str
            ):
                raise RuntimeError(f"{label} embedded file record types are invalid")
            path = _regular(Path(str(value["path"])), label=label)
            if sha256_file(path) != value["sha256"]:
                raise RuntimeError(f"{label} embedded file record changed")
            return
        for key, child in value.items():
            _validate_embedded_file_records(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_embedded_file_records(child, label=f"{label}[{index}]")


def _contains_file_record(value: Any, record: Mapping[str, str]) -> bool:
    if isinstance(value, Mapping):
        return value == record or any(
            _contains_file_record(child, record) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_file_record(child, record) for child in value)
    return False


def _validate_confirmatory(
    path: Path,
    *,
    kind: str,
    config_sha256: str,
    controlled_tensor_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    document = _load_json(_regular(path, label=f"{kind} confirmatory analysis"))
    expected_schema = (
        P2T_CONFIRMATORY_SCHEMA if kind == "p2t" else EDITOR_CONFIRMATORY_SCHEMA
    )
    descriptive_schema = (
        P2T_DESCRIPTIVE_SCHEMA if kind == "p2t" else EDITOR_DESCRIPTIVE_SCHEMA
    )
    if (
        isinstance(document, Mapping)
        and document.get("schema_version") == descriptive_schema
        and document.get("analysis_status")
        == "DESCRIPTIVE_ONLY_PENDING_EFFECT_FREEZE"
        and document.get("experiment_config_sha256") == config_sha256
    ):
        _reject_nonfinite(document, label=f"{kind} descriptive analysis")
        _validate_embedded_file_records(
            document.get("input_runs"), label=f"{kind}.input_runs"
        )
        return dict(document), _record(path)
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != expected_schema
        or document.get("analysis_status") != "VALID"
        or document.get("experiment_config_sha256") != config_sha256
        or document.get("hypothesis_verdict")
        not in {None, "SUPPORTED", "NOT_SUPPORTED"}
    ):
        raise RuntimeError(f"{kind} confirmatory analysis is not VALID")
    if (
        kind == "p2t"
        and document.get("controlled_tensor_manifest_sha256")
        != controlled_tensor_sha256
    ):
        raise RuntimeError("P2T confirmatory analysis binds a different tensor manifest")
    _reject_nonfinite(document, label=f"{kind} confirmatory analysis")
    _validate_embedded_file_records(document.get("input_runs"), label=f"{kind}.input_runs")
    return dict(document), _record(path)


def _require_metric_hash_inventory(
    documents: Sequence[Mapping[str, Any]],
    *,
    field: str,
    paths: Sequence[Path],
    label: str,
) -> None:
    observed = Counter(str(document.get(field) or "") for document in documents)
    expected = Counter(sha256_file(path) for path in paths)
    if observed != expected:
        raise RuntimeError(f"{label} does not exactly close over the formal inventory")


def _validate_p2t_metric_artifact_pairs(
    documents: Sequence[Mapping[str, Any]],
    *,
    prediction_paths: Sequence[Path],
    paired_paths: Sequence[Path],
) -> None:
    prediction_by_hash = {sha256_file(path): path for path in prediction_paths}
    paired_by_hash = {sha256_file(path): path for path in paired_paths}
    if (
        len(prediction_by_hash) != len(prediction_paths)
        or len(paired_by_hash) != len(paired_paths)
    ):
        raise RuntimeError("P2T prediction/paired inventories contain duplicate bytes")
    for document in documents:
        prediction_path = prediction_by_hash.get(
            str(document.get("predictions_sha256") or "")
        )
        paired_path = paired_by_hash.get(
            str(document.get("paired_evaluation_rows_sha256") or "")
        )
        if prediction_path is None or paired_path is None:
            raise RuntimeError("P2T metric points outside formal prediction inventories")
        predictions = load_jsonl(prediction_path)
        paired = load_jsonl(paired_path)
        if (
            not predictions
            or len(predictions) != len(paired)
            or int(document.get("paired_evaluation_row_count", -1)) != len(paired)
            or int(document.get("episode_count", -1)) != len(predictions)
        ):
            raise RuntimeError("P2T metric/prediction/paired row counts differ")
        prediction_by_episode = {
            str(row.get("episode_id") or ""): row for row in predictions
        }
        paired_by_episode = {
            str(row.get("episode_id") or ""): row for row in paired
        }
        if (
            "" in prediction_by_episode
            or "" in paired_by_episode
            or len(prediction_by_episode) != len(predictions)
            or len(paired_by_episode) != len(paired)
            or set(prediction_by_episode) != set(paired_by_episode)
        ):
            raise RuntimeError("P2T prediction/paired episode inventory differs")
        common_expected = {
            "checkpoint_sha256": document.get("checkpoint_sha256"),
            "input_ablation": document.get("input_ablation"),
            "partition": document.get("partition"),
            "experiment_config_sha256": document.get("experiment_config_sha256"),
            "learning_split_sha256": document.get("learning_split_sha256"),
            "test_access_receipt_sha256": document.get(
                "test_access_receipt_sha256"
            ),
        }
        training_manifest_sha256 = str(
            document.get("training_manifest_sha256") or ""
        )
        inference_manifest_sha256 = str(
            document.get("inference_manifest_sha256") or ""
        )
        if (
            document.get("manifest_sha256") != inference_manifest_sha256
            or len(training_manifest_sha256) != 64
            or len(inference_manifest_sha256) != 64
        ):
            raise RuntimeError("P2T metric manifest provenance is incomplete")
        for episode, prediction in prediction_by_episode.items():
            paired_row = paired_by_episode[episode]
            if any(
                prediction.get(field) != expected
                or paired_row.get(field) != expected
                for field, expected in common_expected.items()
            ):
                raise RuntimeError("P2T metric and row provenance do not match")
            expected_prediction_manifests = {
                "learning_manifest_sha256": inference_manifest_sha256,
                "training_manifest_sha256": training_manifest_sha256,
                "inference_manifest_sha256": inference_manifest_sha256,
            }
            if any(
                prediction.get(field) != expected
                for field, expected in expected_prediction_manifests.items()
            ):
                raise RuntimeError("P2T prediction manifest provenance does not match")
            if (
                paired_row.get("learning_manifest_sha256")
                != inference_manifest_sha256
                or paired_row.get("training_manifest_sha256")
                != training_manifest_sha256
                or (
                    "inference_manifest_sha256" in paired_row
                    and paired_row.get("inference_manifest_sha256")
                    != inference_manifest_sha256
                )
            ):
                raise RuntimeError("P2T paired-row manifest provenance does not match")
            if int(paired_row.get("checkpoint_seed", -1)) != int(
                document.get("checkpoint_seed", -1)
            ):
                raise RuntimeError("P2T paired row uses another checkpoint seed")


def _singleton_row_value(
    rows: Sequence[Mapping[str, Any]], field: str, *, label: str
) -> Any:
    values = {json.dumps(row.get(field), sort_keys=True) for row in rows}
    if len(values) != 1:
        raise RuntimeError(f"{label} mixes {field}")
    return rows[0].get(field)


def _validate_editor_metric_leaf_artifacts(
    *,
    rows_path: Path,
    summary: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    rows = load_jsonl(rows_path)
    if not rows:
        raise RuntimeError("editor metric rows are empty")
    label = "editor metric rows"
    training_sha = str(
        _singleton_row_value(rows, "training_manifest_sha256", label=label) or ""
    )
    inference_sha = str(
        _singleton_row_value(rows, "inference_manifest_sha256", label=label) or ""
    )
    learning_sha = str(
        _singleton_row_value(rows, "learning_manifest_sha256", label=label) or ""
    )
    if (
        len(training_sha) != 64
        or len(inference_sha) != 64
        or learning_sha != inference_sha
    ):
        raise RuntimeError("editor manifest hashes are incomplete or inconsistent")
    path_fields = {
        field: _regular(
            Path(
                str(_singleton_row_value(rows, field, label=label) or "")
            ),
            label=f"editor {field}",
        )
        for field in (
            "training_manifest",
            "inference_manifest",
            "learning_manifest",
            "prediction_manifest",
        )
    }
    if (
        sha256_file(path_fields["training_manifest"]) != training_sha
        or sha256_file(path_fields["inference_manifest"]) != inference_sha
        or path_fields["learning_manifest"] != path_fields["inference_manifest"]
        or sha256_file(path_fields["learning_manifest"]) != learning_sha
        or sha256_file(path_fields["prediction_manifest"])
        != summary.get("prediction_manifest_sha256")
    ):
        raise RuntimeError("editor manifest artifact changed after evaluation")
    checkpoint = torch.load(
        _regular(checkpoint_path, label="editor checkpoint"),
        map_location="cpu",
        weights_only=True,
    )
    if (
        checkpoint.get("schema_version") != EDITOR_CHECKPOINT_SCHEMA
        or checkpoint.get("status") != TRAINED_CHECKPOINT_STATUS
        or checkpoint.get("manifest_sha256") != training_sha
        or checkpoint.get("training_manifest_sha256") != training_sha
    ):
        raise RuntimeError("editor checkpoint training provenance mismatch")
    for field in ("manifest", "training_manifest"):
        raw = Path(str(checkpoint.get(field) or ""))
        if (
            raw.is_symlink()
            or raw.resolve() != path_fields["training_manifest"]
        ):
            raise RuntimeError(f"editor checkpoint {field} path mismatch")
    learning_split_path = _regular(
        Path(str(checkpoint.get("learning_split") or "")),
        label="editor checkpoint learning split",
    )
    if (
        checkpoint.get("learning_split_sha256")
        != summary.get("learning_split_sha256")
        or sha256_file(learning_split_path)
        != summary.get("learning_split_sha256")
    ):
        raise RuntimeError("editor learning split changed after training")

    prediction_rows = load_jsonl(path_fields["prediction_manifest"])
    prediction_by_episode = {
        str(row.get("episode_id") or ""): row for row in prediction_rows
    }
    metric_by_episode = {str(row.get("episode_id") or ""): row for row in rows}
    partition = summary.get("partition")
    learning_by_episode = {
        str(row.get("episode_id") or ""): row
        for row in load_jsonl(path_fields["inference_manifest"])
        if row.get("partition") == partition
    }
    if (
        "" in prediction_by_episode
        or "" in metric_by_episode
        or "" in learning_by_episode
        or len(prediction_by_episode) != len(prediction_rows)
        or set(prediction_by_episode) != set(metric_by_episode)
        or set(learning_by_episode) != set(metric_by_episode)
    ):
        raise RuntimeError("editor leaf episode inventories are not exact")
    episode_records: dict[str, dict[str, dict[str, str]]] = {}
    for episode in sorted(metric_by_episode):
        row = metric_by_episode[episode]
        prediction_row = prediction_by_episode[episode]
        source = learning_by_episode[episode]
        if any(row.get(field) != value for field, value in prediction_row.items()):
            raise RuntimeError("editor metric row changed prediction provenance")
        expected_source = {
            "patient_id": row.get("patient_id"),
            "visible_npz": row.get("visible_npz"),
            "visible_sha256": row.get("visible_npz_sha256"),
            "evaluation_npz": row.get("evaluation_npz"),
            "evaluation_sha256": row.get("evaluation_npz_sha256"),
        }
        if any(source.get(field) != value for field, value in expected_source.items()):
            raise RuntimeError("editor metric row differs from inference manifest")
        episode_records[episode] = {}
        for role, path_field, sha_field in (
            ("prediction_npz", "prediction_npz", "prediction_npz_sha256"),
            ("evaluation_npz", "evaluation_npz", "evaluation_npz_sha256"),
            ("visible_npz", "visible_npz", "visible_npz_sha256"),
        ):
            path = _regular(
                Path(str(row.get(path_field) or "")),
                label=f"editor {role} for {episode}",
            )
            if sha256_file(path) != row.get(sha_field):
                raise RuntimeError(f"editor {role} changed after evaluation")
            episode_records[episode][role] = _record(path)
    return {
        "training_manifest": _record(path_fields["training_manifest"]),
        "inference_manifest": _record(path_fields["inference_manifest"]),
        "learning_split": _record(learning_split_path),
        "prediction_manifest": _record(path_fields["prediction_manifest"]),
        "episodes": episode_records,
    }


def _validate_confirmatory_input_inventory(
    document: Mapping[str, Any],
    *,
    kind: str,
    expected: Mapping[tuple[int, str], Mapping[str, Mapping[str, str]]],
) -> None:
    runs = document.get("input_runs")
    if not isinstance(runs, list):
        raise RuntimeError(f"{kind} confirmatory input_runs must be a list")
    arm_field = "arm" if kind == "P2T" else "condition"
    observed_keys = Counter(
        (int(run.get("seed", -1)), str(run.get(arm_field) or ""))
        for run in runs
        if isinstance(run, Mapping)
    )
    if len(observed_keys) != len(runs) or observed_keys != Counter(expected.keys()):
        raise RuntimeError(
            f"{kind} confirmatory input_runs do not exactly cover the formal run inventory"
        )
    for run in runs:
        key = (int(run["seed"]), str(run[arm_field]))
        expected_records = expected[key]
        for record_name, expected_record in expected_records.items():
            if run.get(record_name) != expected_record:
                raise RuntimeError(
                    f"{kind} confirmatory run {key} uses a different {record_name} artifact"
                )
    for record_name in sorted(
        {name for records in expected.values() for name in records}
    ):
        observed_records = [run[record_name] for run in runs]
        expected_records = [records[record_name] for records in expected.values()]
        if not all(
            isinstance(record, Mapping)
            and set(record) == {"path", "sha256"}
            for record in expected_records
        ):
            continue
        if _record_counter(observed_records) != _record_counter(expected_records):
            raise RuntimeError(
                f"{kind} confirmatory {record_name} Counter does not close exactly"
            )


def validate_robustness_all_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    robustness_rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _strategy_contract(config)
    configured = set(contract["strategies"])

    def validate_artifacts(row: Mapping[str, Any], *, label: str) -> None:
        for path_field, hash_field in (
            ("visible_document", "visible_document_sha256"),
            ("evaluation_document", "evaluation_document_sha256"),
            ("authorized_path", "authorized_sha256"),
        ):
            path = _regular(Path(str(row.get(path_field) or "")), label=f"{label}.{path_field}")
            if sha256_file(path) != row.get(hash_field):
                raise RuntimeError(f"{label} {path_field} hash mismatch")

    primary_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in primary_rows:
        key = (str(row.get("case_id") or ""), str(row.get("goal") or ""))
        if not all(key) or key in primary_by_key:
            raise RuntimeError("primary robustness reference has duplicate/missing case-goal")
        generation = row.get("scribble_generation")
        if (
            not isinstance(generation, Mapping)
            or generation.get("stage_order") != list(GENERATION_STAGE_ORDER)
        ):
            raise RuntimeError("primary robustness reference has incomplete stage order")
        validate_artifacts(row, label="primary robustness row")
        primary_by_key[key] = row
    robustness_by_key: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in robustness_rows:
        key = (str(row.get("case_id") or ""), str(row.get("goal") or ""))
        strategy = str(row.get("strategy") or "")
        generation = row.get("scribble_generation")
        if (
            not all(key)
            or strategy not in configured
            or strategy in robustness_by_key[key]
            or row.get("strategy_mode") != "all"
            or row.get("strategy_assignment") != contract["assignment"]
            or row.get("strategy_salt") != contract["salt"]
            or not isinstance(generation, Mapping)
            or generation.get("strategy_mode") != "all"
            or generation.get("strategy_assignment") != contract["assignment"]
            or generation.get("strategy_salt") != contract["salt"]
            or generation.get("selected_strategy") != strategy
            or generation.get("stage_order") != list(GENERATION_STAGE_ORDER)
        ):
            raise RuntimeError("robustness-all row has invalid mode/strategy/coverage")
        validate_artifacts(row, label="robustness-all row")
        robustness_by_key[key][strategy] = row
    if not set(primary_by_key) <= set(robustness_by_key) or any(
        set(robustness_by_key[key]) != configured for key in primary_by_key
    ):
        raise RuntimeError(
            "robustness-all does not provide exactly three strategies per primary source"
        )
    invariant = (
        "case_id",
        "patient_id",
        "partition",
        "held_out_fold",
        "m0_sha256",
        "gt_sha256",
        "fn_sha256",
        "experiment_config_sha256",
        "learning_split_sha256",
        "m0_provenance",
    )
    exact_primary = (
        "episode_id",
        "coordinates_xyz",
        "fn_mask_sha256",
        "authorized_sha256",
        "target_stats",
    )
    for key, primary in primary_by_key.items():
        rows = robustness_by_key[key]
        for field in invariant:
            values = {json.dumps(row.get(field), sort_keys=True) for row in rows.values()}
            values.add(json.dumps(primary.get(field), sort_keys=True))
            if len(values) != 1:
                raise RuntimeError(f"robustness-all changes shared scientific field {field}")
        expected_strategy = assign_scribble_strategy(
            str(primary["patient_id"]), salt=contract["salt"]
        )
        if primary.get("strategy") != expected_strategy:
            raise RuntimeError("primary reference strategy is not stable-hash assigned")
        matching = rows[expected_strategy]
        if any(matching.get(field) != primary.get(field) for field in exact_primary):
            raise RuntimeError("robustness-all primary-strategy row differs from primary corpus")
    return {
        "source_count": len(primary_by_key),
        "row_count": len(robustness_rows),
        "generated_source_count": len(robustness_by_key),
        "additional_nonprimary_source_count": len(
            set(robustness_by_key) - set(primary_by_key)
        ),
        "complete_three_strategy_source_count": sum(
            set(rows) == configured for rows in robustness_by_key.values()
        ),
        "strategies": sorted(configured),
        "primary_correspondence_exact": True,
    }


def validate_pipeline(inputs_path: Path, target: str) -> dict[str, Any]:
    inputs_path = _regular(inputs_path, label="pipeline inputs")
    inputs = _load_json(inputs_path)
    if not isinstance(inputs, Mapping):
        raise RuntimeError("pipeline inputs must be an object")
    _reject_nonfinite(inputs, label="pipeline inputs")
    base = inputs_path.parent
    config_path = _resolve(base, inputs.get("experiment_config"), label="experiment_config")
    case_manifest = _resolve(base, inputs.get("case_manifest"), label="case_manifest")
    split_path = _resolve(base, inputs.get("learning_split"), label="learning_split")
    evaluation_partition = inputs.get("evaluation_partition")
    if evaluation_partition not in {"val", "test"}:
        raise RuntimeError("pipeline inputs require evaluation_partition=val|test")
    run_root_raw = inputs.get("run_root")
    if not isinstance(run_root_raw, str) or not run_root_raw:
        raise RuntimeError("pipeline inputs require run_root")
    raw_run_root = Path(run_root_raw)
    run_root = _directory(
        raw_run_root if raw_run_root.is_absolute() else base / raw_run_root,
        label="run_root",
    )
    if not inputs_path.is_relative_to(run_root):
        raise RuntimeError("pipeline inputs manifest is outside the declared run_root")
    f0 = _validate_f0_binding(inputs.get("f0_readiness"))
    test_access_receipt_sha256: str | None = None
    test_access_record: dict[str, str] | None = None
    frozen_bindings: Mapping[str, Any] | None = None
    frozen_bindings_path: Path | None = None
    frozen_bindings_record: dict[str, str] | None = None
    final_freeze_record: dict[str, str] | None = None
    checkpoint_inventory_sha256: str | None = None
    frozen_singleton_records: dict[str, dict[str, str]] = {}
    raw_test_access = inputs.get("test_access_receipt")
    raw_frozen_bindings = inputs.get("frozen_checkpoint_bindings")
    if evaluation_partition == "test":
        receipt_path = _resolve(
            base, raw_test_access, label="test_access_receipt"
        )
        try:
            receipt = validate_consumed_receipt(
                receipt_path,
                experiment_config=config_path,
                learning_split=split_path,
                run_root=run_root,
            )
        except TestAccessError as error:
            raise RuntimeError(f"test-access receipt validation failed: {error}") from error
        test_access_receipt_sha256 = str(receipt["receipt_sha256"])
        test_access_record = _record(receipt_path)
        consumption = receipt.get("consumption")
        if not isinstance(consumption, Mapping):
            raise RuntimeError("test-access receipt omits consumption contract")
        consumed_freeze = consumption.get("final_development_freeze")
        if not isinstance(consumed_freeze, Mapping):
            raise RuntimeError("test-access receipt omits final development freeze")
        binding_freeze: Mapping[str, Any] = consumed_freeze
        checkpoint_inventory_sha256 = str(
            consumption.get("checkpoint_inventory_sha256") or ""
        )
        if raw_frozen_bindings not in (None, ""):
            frozen_bindings_path = _resolve(
                base,
                raw_frozen_bindings,
                label="frozen_checkpoint_bindings",
            )
            try:
                frozen_bindings = validate_frozen_checkpoint_bindings(
                    frozen_bindings_path
                )
            except DevelopmentFreezeError as error:
                raise RuntimeError(
                    f"frozen checkpoint binding validation failed: {error}"
                ) from error
            candidate_freeze = frozen_bindings.get("final_development_freeze")
            if not isinstance(candidate_freeze, Mapping) or any(
                consumed_freeze.get(field) != candidate_freeze.get(field)
                for field in ("path", "sha256", "bytes")
            ):
                raise RuntimeError(
                    "test-access receipt and checkpoint bindings use different final freezes"
                )
            if (
                frozen_bindings.get("checkpoint_inventory_sha256")
                != checkpoint_inventory_sha256
            ):
                raise RuntimeError(
                    "test-access receipt and final freeze use different checkpoint inventories"
                )
            binding_freeze = candidate_freeze
            frozen_bindings_record = _record(frozen_bindings_path)
        elif target != "m0_evaluation":
            raise RuntimeError(
                "formal Route A stage lacks frozen checkpoint bindings"
            )
        current_freeze = _current_freeze_file_record(
            binding_freeze,
            label="final development freeze",
            expected_role=(
                "final_development_freeze"
                if frozen_bindings is not None
                else None
            ),
        )
        final_freeze_record = {
            "path": current_freeze["path"],
            "sha256": current_freeze["sha256"],
        }
        freeze_document = _load_json(Path(current_freeze["path"]))
        required_artifacts = freeze_document.get("required_artifacts")
        if not isinstance(required_artifacts, list):
            raise RuntimeError("final development freeze omits required artifacts")
        required_by_role = {
            str(record.get("role") or ""): record
            for record in required_artifacts
            if isinstance(record, Mapping)
        }
        singleton_inputs = {
            "frozen_m0_validation_receipt": "m0_validation_receipt",
            "frozen_environment_receipt": "environment_receipt",
            "frozen_oof_receipt": "m0_oof_receipt",
        }
        for input_key, role in singleton_inputs.items():
            path = _resolve(base, inputs.get(input_key), label=input_key)
            frozen_record = required_by_role.get(role)
            current = _current_freeze_file_record(
                frozen_record,
                label=f"final freeze {role}",
                expected_role=role,
            )
            if str(path) != current["path"]:
                raise RuntimeError(
                    f"{input_key} differs from the consumed final freeze"
                )
            frozen_singleton_records[role] = {
                "path": current["path"],
                "sha256": current["sha256"],
            }
    elif raw_test_access not in (None, ""):
        raise RuntimeError("validation pipeline must not carry a test-access receipt")
    elif raw_frozen_bindings not in (None, ""):
        raise RuntimeError("validation pipeline must not carry frozen checkpoints")
    elif any(
        inputs.get(key) not in (None, "")
        for key in (
            "frozen_m0_validation_receipt",
            "frozen_environment_receipt",
            "frozen_oof_receipt",
        )
    ):
        raise RuntimeError("validation pipeline must not carry frozen singleton receipts")
    config = _load_json(config_path)
    source_rows = load_jsonl(case_manifest)
    _, split = load_and_validate_learning_split(split_path, source_rows, config)
    config_sha = sha256_file(config_path)
    common = {
        "experiment_config_sha256": config_sha,
        "case_manifest_sha256": sha256_file(case_manifest),
        "learning_split_sha256": split["split_sha256"],
        "evaluation_partition": evaluation_partition,
        "run_root": str(run_root),
        "test_access_receipt_sha256": test_access_receipt_sha256,
        "final_development_freeze_sha256": (
            None if final_freeze_record is None else final_freeze_record["sha256"]
        ),
        "checkpoint_inventory_sha256": checkpoint_inventory_sha256,
        "f0_readiness_sha256": f0["receipt"]["sha256"],
        "f0_source_bundle_sha256": f0["source_bundle_sha256"],
        "f0_environment_bundle_sha256": f0["environment_bundle_sha256"],
        "patient_counts": split["patient_counts"],
        "case_counts": split["case_counts"],
    }
    result: dict[str, Any] = {
        "schema_version": (
            M0_EVALUATION_READY_SCHEMA
            if target == "m0_evaluation"
            else ROBUSTNESS_ALL_READY_SCHEMA
            if target == "robustness_all"
            else PIPELINE_RECEIPT_SCHEMA
        ),
        "status": "PASS",
        "target": target,
        "inputs_manifest_sha256": sha256_file(inputs_path),
        "truth_boundary": "artifact receipts validated; hypothesis support is not required for completion",
        "common": common,
        "upstream_receipts": {},
        "artifact_bindings": {
            "pipeline_inputs": _record(inputs_path),
            "experiment_config": _record(config_path),
            "case_manifest": _record(case_manifest),
            "learning_split": _record(split_path),
            "f0_readiness": {
                "path": f0["receipt"]["path"],
                "sha256": f0["receipt"]["sha256"],
            },
        },
    }
    if test_access_record is not None:
        result["artifact_bindings"]["test_access_receipt"] = test_access_record
    if final_freeze_record is not None:
        result["artifact_bindings"].update(
            {
                "final_development_freeze": final_freeze_record,
                "frozen_singleton_receipts": frozen_singleton_records,
            }
        )
    if frozen_bindings_record is not None:
        result["artifact_bindings"][
            "frozen_checkpoint_bindings"
        ] = frozen_bindings_record

    if target == "m0_evaluation":
        oof_path = _resolve(base, inputs.get("oof_ready"), label="oof_ready")
        official_path = _resolve(
            base, inputs.get("official_metrics"), label="official_metrics"
        )
        rows_path = _resolve(base, inputs.get("m0_rows"), label="m0_rows")
        summary_path = _resolve(base, inputs.get("m0_summary"), label="m0_summary")
        partitions = inputs.get("evaluation_partitions")
        if not isinstance(partitions, list):
            raise RuntimeError("M0 inputs require evaluation_partitions")
        oof = validate_oof_ready_receipt_only(oof_path)
        result["m0_evaluation"] = validate_m0_evaluation(
            rows=load_jsonl(rows_path),
            summary=_load_json(summary_path),
            source_rows=source_rows,
            split=split,
            oof=oof,
            selected_partitions=partitions,
            config_sha256=config_sha,
            case_manifest_sha256=sha256_file(case_manifest),
            learning_split_sha256=sha256_file(split_path),
            official_metrics_sha256=sha256_file(official_path),
            evaluation_partition=str(evaluation_partition),
            test_access_receipt_sha256=test_access_receipt_sha256,
            run_root=run_root,
        )
        result["artifact_bindings"].update(
            {
                "oof_ready": _record(oof_path),
                "official_metrics": _record(official_path),
                "m0_rows": _record(rows_path),
                "m0_summary": _record(summary_path),
            }
        )
        return result

    upstream_by_target = {
        "p2t_data": (("m0_evaluation_ready", "m0_evaluation"),),
        "editor_data": (("m0_evaluation_ready", "m0_evaluation"),),
        "p2t_results": (
            ("m0_evaluation_ready", "m0_evaluation"),
            ("p2t_data_ready", "p2t_data"),
        ),
        "editor_results": (
            ("m0_evaluation_ready", "m0_evaluation"),
            ("editor_data_ready", "editor_data"),
            ("p2t_results_ready", "p2t_results"),
        ),
        "complete": (
            ("m0_evaluation_ready", "m0_evaluation"),
            ("p2t_data_ready", "p2t_data"),
            ("editor_data_ready", "editor_data"),
            ("p2t_results_ready", "p2t_results"),
            ("editor_results_ready", "editor_results"),
        ),
        "robustness_all": (("primary_complete", "complete"),),
    }
    for input_key, expected_target in upstream_by_target[target]:
        receipt_path = _resolve(base, inputs.get(input_key), label=input_key)
        result["upstream_receipts"][expected_target] = _validate_upstream_receipt(
            receipt_path, expected_target=expected_target, common=common
        )

    if target == "robustness_all":
        selected_partitions = {"train", "val"}
        if evaluation_partition == "test":
            selected_partitions.add("test")
        primary_path = _resolve(
            base,
            inputs.get("primary_natural_episode_manifest"),
            label="primary_natural_episode_manifest",
        )
        robustness_path = _resolve(
            base,
            inputs.get("robustness_natural_episode_manifest"),
            label="robustness_natural_episode_manifest",
        )
        primary_record = _record(primary_path)
        primary_complete_document = _load_json(
            _resolve(base, inputs.get("primary_complete"), label="primary_complete")
        )
        if not _contains_file_record(
            primary_complete_document.get("artifact_bindings"), primary_record
        ):
            raise RuntimeError(
                "ROUTE_A_COMPLETE did not bind the primary natural episode manifest"
            )
        robustness_ready_path = _resolve(
            base,
            inputs.get("natural_robustness_data_ready"),
            label="natural_robustness_data_ready",
        )
        robustness_ready_document = _load_json(robustness_ready_path)
        robustness_inputs = (
            robustness_ready_document.get("inputs")
            if isinstance(robustness_ready_document, Mapping)
            else None
        )
        if not isinstance(robustness_inputs, Mapping):
            raise RuntimeError("natural robustness readiness omits inputs")

        def robustness_input_path(key: str) -> Path:
            value = robustness_inputs.get(key)
            if not isinstance(value, Mapping):
                raise RuntimeError(f"natural robustness readiness omits {key}")
            return Path(str(value.get("path") or ""))

        robustness_expected_inputs = {
            "residual_manifest": robustness_input_path("residual_manifest"),
            "residual_ready": robustness_input_path("residual_ready"),
            "oof_ready": robustness_input_path("oof_ready"),
            "experiment_config": config_path,
            "learning_split": split_path,
        }
        for key in ("residual_manifest", "residual_ready", "oof_ready"):
            record = _record(_regular(robustness_expected_inputs[key], label=key))
            if not _contains_file_record(
                primary_complete_document.get("artifact_bindings"), record
            ):
                raise RuntimeError(
                    f"natural robustness {key} differs from ROUTE_A_COMPLETE"
                )
        robustness_ready = _validate_episode_data_ready(
            robustness_ready_path,
            schema_version="PETCT-SCRIBBLE-DATA-READY-v1.0",
            phase="OFFICIAL_FN_SCRIBBLE_EPISODE_MATERIALIZATION",
            lane="natural",
            strategy_mode="all",
            selected_partitions=selected_partitions,
            manifest_path=robustness_path,
            run_root=run_root,
            expected_input_files=robustness_expected_inputs,
            goals_per_generated_attempt=1,
        )
        result["robustness_all"] = {
            **validate_robustness_all_rows(
                load_jsonl(primary_path), load_jsonl(robustness_path), config=config
            ),
            "secondary": True,
            "not_in_primary_confirmatory": True,
            "does_not_invalidate_primary_route_a": True,
            "raw_denominator": robustness_ready["validated_cohort"],
            "attempts": robustness_ready["validated_attempts"],
            "exclusions_by_reason": robustness_ready["exclusions_by_reason"],
            "survivor_coverage": robustness_ready["survivor_coverage"],
        }
        result["artifact_bindings"].update(
            {
                "primary_natural_episode_manifest": _record(primary_path),
                "robustness_natural_episode_manifest": _record(robustness_path),
                "natural_robustness_data_ready": _record(robustness_ready_path),
            }
        )
        return result

    controlled_needed = target in {"p2t_data", "p2t_results", "complete"}
    p2t_results_needed = target in {"p2t_results", "complete"}
    natural_needed = target in {"editor_data", "editor_results", "complete"}
    editor_results_needed = target in {"editor_results", "complete"}
    selected_partitions = {"train", "val"}
    if evaluation_partition == "test":
        selected_partitions.add("test")
    result_artifact_paths: list[Path] = []
    confirmatory_artifact_paths: list[Path] = []
    controlled_tensor_sha: str | None = None
    residual_ready: Mapping[str, Any] | None = None
    residual_manifest_path: Path | None = None
    shared_oof_path: Path | None = None
    if controlled_needed or natural_needed:
        shared_oof_path = _resolve(base, inputs.get("oof_ready"), label="oof_ready")
        residual_ready_path = _resolve(
            base, inputs.get("residual_ready"), label="residual_ready"
        )
        residual_ready_document = _load_json(residual_ready_path)
        if not isinstance(residual_ready_document, Mapping):
            raise RuntimeError("RESIDUAL_READY must be an object")
        residual_manifest_record = residual_ready_document.get("residual_manifest")
        if not isinstance(residual_manifest_record, Mapping):
            raise RuntimeError("RESIDUAL_READY omits residual_manifest")
        residual_manifest_path = _regular(
            Path(str(residual_manifest_record.get("path") or "")),
            label="residual_manifest",
        )
        residual_ready = validate_residual_ready(
            residual_ready_path,
            residual_manifest=residual_manifest_path,
            oof_ready=shared_oof_path,
            selected_partitions=selected_partitions,
        )
        result["artifact_bindings"].update(
            {
                "residual_ready": _record(residual_ready_path),
                "residual_manifest": _record(residual_manifest_path),
            }
        )
    if controlled_needed:
        controlled_path = _resolve(
            base,
            inputs.get("controlled_episode_manifest"),
            label="controlled_episode_manifest",
        )
        controlled = validate_controlled_episode_rows(
            load_jsonl(controlled_path),
            config_sha256=config_sha,
            split_sha256=split["split_sha256"],
            case_to_partition=split["case_to_partition"],
            config=config,
        )
        tensor_path = _resolve(
            base,
            inputs.get("controlled_tensor_manifest"),
            label="controlled_tensor_manifest",
        )
        tensor = validate_tensor_rows(
            load_jsonl(tensor_path),
            expected_episode_ids=controlled.pop("episode_ids"),
            config_sha256=config_sha,
            split_sha256=split["split_sha256"],
        )
        controlled_tensor_sha = sha256_file(tensor_path)
        controlled_ready_path = _resolve(
            base,
            inputs.get("controlled_data_ready"),
            label="controlled_data_ready",
        )
        controlled_ready = _validate_episode_data_ready(
            controlled_ready_path,
            schema_version="PETCT-CONTROLLED-P2T-DATA-READY-v1.0",
            phase="CONTROLLED_MATCHED_STATE_MATERIALIZATION",
            lane="controlled_p2t",
            strategy_mode="primary",
            selected_partitions=selected_partitions,
            manifest_path=controlled_path,
            run_root=run_root,
            expected_input_files={
                "case_manifest": case_manifest,
                "learning_split": split_path,
                "experiment_config": config_path,
            },
            goals_per_generated_attempt=len(GOALS),
        )
        result["artifact_bindings"].update(
            {
                "controlled_episode_manifest": _record(controlled_path),
                "controlled_tensor_manifest": _record(tensor_path),
                "controlled_data_ready": _record(controlled_ready_path),
            }
        )
        result["p2t_lane"] = {
            "status": "DATA_READY",
            **controlled,
            "tensor_episodes": tensor["episodes"],
            "episode_manifest_sha256": sha256_file(controlled_path),
            "tensor_manifest_sha256": controlled_tensor_sha,
            "materialization_ready_sha256": controlled_ready["ready_sha256"],
            "cohort": controlled_ready["validated_cohort"],
            "attempts": controlled_ready["validated_attempts"],
        }
        if p2t_results_needed:
            metric_paths = _resolve_many(
                base, inputs.get("p2t_metrics"), label="p2t_metrics"
            )
            checkpoint_paths = _resolve_many(
                base, inputs.get("p2t_checkpoints"), label="p2t_checkpoints"
            )
            prediction_paths = _resolve_many(
                base, inputs.get("p2t_predictions"), label="p2t_predictions"
            )
            paired_paths = _resolve_many(
                base, inputs.get("p2t_paired_rows"), label="p2t_paired_rows"
            )
            documents = [_load_json(path) for path in metric_paths]
            if not (
                len(metric_paths)
                == len(checkpoint_paths)
                == len(prediction_paths)
                == len(paired_paths)
            ):
                raise RuntimeError("P2T result artifact inventories have different lengths")
            metric_path_by_key = {
                (int(document["checkpoint_seed"]), str(document["input_ablation"])): path
                for document, path in zip(documents, metric_paths)
            }
            if len(metric_path_by_key) != len(metric_paths):
                raise RuntimeError("P2T metric inventory contains duplicate run cells")
            checkpoint_path_by_key: dict[tuple[int, str], Path] = {}
            checkpoint_document_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
            for path in checkpoint_paths:
                document = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(document, Mapping):
                    raise RuntimeError("P2T checkpoint must be a mapping")
                key = (
                    int(document.get("seed", -1)),
                    str(document.get("input_ablation") or ""),
                )
                if (
                    document.get("schema_version") != P2T_CHECKPOINT_SCHEMA
                    or document.get("status") != TRAINED_CHECKPOINT_STATUS
                    or document.get("experiment_config_sha256") != config_sha
                    or document.get("learning_split_sha256")
                    != split["split_sha256"]
                    or key in checkpoint_path_by_key
                ):
                    raise RuntimeError("P2T result inventory contains an invalid checkpoint")
                checkpoint_manifest_path = _regular(
                    Path(str(document.get("manifest") or "")),
                    label="P2T checkpoint training manifest",
                )
                checkpoint_manifest_sha = str(document.get("manifest_sha256") or "")
                if (
                    len(checkpoint_manifest_sha) != 64
                    or sha256_file(checkpoint_manifest_path) != checkpoint_manifest_sha
                ):
                    raise RuntimeError("P2T checkpoint training manifest changed")
                if frozen_bindings is None and (
                    checkpoint_manifest_sha != controlled_tensor_sha
                    or checkpoint_manifest_path != tensor_path
                ):
                    raise RuntimeError(
                        "validation P2T checkpoint was not trained on the current controlled tensor manifest"
                    )
                checkpoint_path_by_key[key] = path
                checkpoint_document_by_key[key] = document
            if set(checkpoint_path_by_key) != set(metric_path_by_key):
                raise RuntimeError("P2T checkpoint inventory does not match metric run cells")
            if frozen_bindings is not None:
                result["artifact_bindings"]["frozen_p2t_checkpoint_roles"] = (
                    _validate_frozen_checkpoint_subset(
                        bindings=frozen_bindings,
                        checkpoint_paths=checkpoint_paths,
                        kind="p2t",
                        expected_roles=_expected_frozen_checkpoint_roles(
                            config, "p2t"
                        ),
                    )
                )
            training_manifest_sha256_by_run_cell = {
                key: str(document["manifest_sha256"])
                for key, document in checkpoint_document_by_key.items()
            }
            metrics = validate_p2t_metric_receipts(
                documents,
                config=config,
                config_sha256=config_sha,
                inference_manifest_sha256=controlled_tensor_sha,
                training_manifest_sha256_by_run_cell=(
                    training_manifest_sha256_by_run_cell
                ),
                learning_split_sha256=split["split_sha256"],
                partition=str(evaluation_partition),
                test_access_receipt_sha256=test_access_receipt_sha256,
            )
            if any(
                document.get("checkpoint_sha256")
                != sha256_file(
                    checkpoint_path_by_key[
                        (
                            int(document["checkpoint_seed"]),
                            str(document["input_ablation"]),
                        )
                    ]
                )
                for document in documents
            ):
                raise RuntimeError("P2T metric uses another run cell's checkpoint")
            _require_metric_hash_inventory(
                documents,
                field="checkpoint_sha256",
                paths=checkpoint_paths,
                label="P2T metrics/checkpoints",
            )
            _require_metric_hash_inventory(
                documents,
                field="predictions_sha256",
                paths=prediction_paths,
                label="P2T metrics/predictions",
            )
            _require_metric_hash_inventory(
                documents,
                field="paired_evaluation_rows_sha256",
                paths=paired_paths,
                label="P2T metrics/paired rows",
            )
            _validate_p2t_metric_artifact_pairs(
                documents,
                prediction_paths=prediction_paths,
                paired_paths=paired_paths,
            )
            prediction_path_by_hash = {
                sha256_file(path): path for path in prediction_paths
            }
            paired_path_by_hash = {sha256_file(path): path for path in paired_paths}
            if (
                len(prediction_path_by_hash) != len(prediction_paths)
                or len(paired_path_by_hash) != len(paired_paths)
            ):
                raise RuntimeError("P2T formal result inventory contains duplicate bytes")
            p2t_confirmatory_path = _resolve(
                base, inputs.get("p2t_confirmatory"), label="p2t_confirmatory"
            )
            confirmatory, confirmatory_record = _validate_confirmatory(
                p2t_confirmatory_path,
                kind="p2t",
                config_sha256=config_sha,
                controlled_tensor_sha256=controlled_tensor_sha,
            )
            if confirmatory["analysis_status"] == "VALID":
                contrast = config["p2t"].get("confirmatory_contrast")
                if not isinstance(contrast, Mapping):
                    raise RuntimeError("active P2T confirmatory analysis lacks frozen contrast")
                confirmatory_arms = {
                    str(contrast["treatment"]),
                    str(contrast["comparator"]),
                }
                metrics_by_key = {
                    (int(document["checkpoint_seed"]), str(document["input_ablation"])): document
                    for document in documents
                }
                confirmatory_expected = {
                    key: {
                        "checkpoint": _record(checkpoint_path_by_key[key]),
                        "metrics": _record(metric_path_by_key[key]),
                        "paired_rows": _record(
                            paired_path_by_hash[
                                str(metrics_by_key[key]["paired_evaluation_rows_sha256"])
                            ]
                        ),
                    }
                    for key in metric_path_by_key
                    if key[1] in confirmatory_arms
                }
                _validate_confirmatory_input_inventory(
                    confirmatory,
                    kind="P2T",
                    expected=confirmatory_expected,
                )
            result["artifact_bindings"].update(
                {
                    "p2t_metrics": [_record(path) for path in metric_paths],
                    "p2t_checkpoints": [_record(path) for path in checkpoint_paths],
                    "p2t_predictions": [_record(path) for path in prediction_paths],
                    "p2t_paired_rows": [_record(path) for path in paired_paths],
                    "p2t_confirmatory": confirmatory_record,
                }
            )
            result_artifact_paths.extend(
                metric_paths
                + checkpoint_paths
                + prediction_paths
                + paired_paths
                + [p2t_confirmatory_path]
            )
            confirmatory_artifact_paths.append(p2t_confirmatory_path)
            result["p2t_lane"].update(
                status="RESULTS_READY",
                **metrics,
                confirmatory_analysis_status=confirmatory["analysis_status"],
                hypothesis_verdict=confirmatory.get("hypothesis_verdict"),
            )

    if natural_needed:
        if shared_oof_path is None or residual_manifest_path is None or residual_ready is None:
            raise RuntimeError("natural lane lacks shared OOF/residual readiness")
        oof_path = shared_oof_path
        oof = validate_oof_ready_receipt_only(oof_path)
        natural_path = _resolve(
            base,
            inputs.get("natural_episode_manifest"),
            label="natural_episode_manifest",
        )
        natural = validate_natural_episode_rows(
            load_jsonl(natural_path),
            config_sha256=config_sha,
            split_sha256=split["split_sha256"],
            oof_ready_sha256=oof["ready_sha256"],
            case_to_partition=split["case_to_partition"],
            config=config,
        )
        tensor_path = _resolve(
            base,
            inputs.get("natural_tensor_manifest"),
            label="natural_tensor_manifest",
        )
        tensor = validate_tensor_rows(
            load_jsonl(tensor_path),
            expected_episode_ids=natural.pop("episode_ids"),
            config_sha256=config_sha,
            split_sha256=split["split_sha256"],
        )
        natural_ready_path = _resolve(
            base,
            inputs.get("natural_primary_data_ready"),
            label="natural_primary_data_ready",
        )
        natural_ready = _validate_episode_data_ready(
            natural_ready_path,
            schema_version="PETCT-SCRIBBLE-DATA-READY-v1.0",
            phase="OFFICIAL_FN_SCRIBBLE_EPISODE_MATERIALIZATION",
            lane="natural",
            strategy_mode="primary",
            selected_partitions=selected_partitions,
            manifest_path=natural_path,
            run_root=run_root,
            expected_input_files={
                "residual_manifest": residual_manifest_path,
                "residual_ready": Path(str(residual_ready["ready_path"])),
                "oof_ready": oof_path,
                "experiment_config": config_path,
                "learning_split": split_path,
            },
            goals_per_generated_attempt=1,
        )
        result["artifact_bindings"].update(
            {
                "oof_ready": _record(oof_path),
                "natural_episode_manifest": _record(natural_path),
                "natural_tensor_manifest": _record(tensor_path),
                "natural_primary_data_ready": _record(natural_ready_path),
            }
        )
        result["editor_lane"] = {
            "status": "DATA_READY",
            **natural,
            "tensor_episodes": tensor["episodes"],
            "oof_ready_sha256": oof["ready_sha256"],
            "episode_manifest_sha256": sha256_file(natural_path),
            "tensor_manifest_sha256": sha256_file(tensor_path),
            "materialization_ready_sha256": natural_ready["ready_sha256"],
            "cohort": natural_ready["validated_cohort"],
            "attempts": natural_ready["validated_attempts"],
        }
        if editor_results_needed:
            natural_metric_paths = _resolve_many(
                base,
                inputs.get("p2t_natural_metrics"),
                label="p2t_natural_metrics",
            )
            natural_prediction_paths = _resolve_many(
                base,
                inputs.get("p2t_natural_predictions"),
                label="p2t_natural_predictions",
            )
            natural_paired_paths = _resolve_many(
                base,
                inputs.get("p2t_natural_paired_rows"),
                label="p2t_natural_paired_rows",
            )
            if not (
                len(natural_metric_paths)
                == len(natural_prediction_paths)
                == len(natural_paired_paths)
                == len(config["p2t"]["training"]["seeds"])
            ):
                raise RuntimeError("natural P2T inventory must cover every registered seed")
            natural_documents = [_load_json(path) for path in natural_metric_paths]
            controlled_dependency = _resolve(
                base,
                inputs.get("controlled_tensor_manifest"),
                label="controlled_tensor_manifest dependency",
            )
            controlled_dependency_sha = sha256_file(controlled_dependency)
            expected_seeds = {int(seed) for seed in config["p2t"]["training"]["seeds"]}
            if (
                {int(document.get("checkpoint_seed", -1)) for document in natural_documents}
                != expected_seeds
                or any(
                    document.get("schema_version") != P2T_METRICS_SCHEMA
                    or document.get("input_ablation") != "full"
                    or document.get("manifest_sha256") != sha256_file(tensor_path)
                    or document.get("inference_manifest_sha256")
                    != sha256_file(tensor_path)
                    or document.get("training_manifest_sha256")
                    != controlled_dependency_sha
                    or document.get("experiment_config_sha256") != config_sha
                    or document.get("learning_split_sha256")
                    != split["split_sha256"]
                    or document.get("partition") != evaluation_partition
                    or document.get("test_access_receipt_sha256")
                    != test_access_receipt_sha256
                    for document in natural_documents
                )
            ):
                raise RuntimeError("natural P2T metrics do not bind the exact seed artifacts")
            _require_metric_hash_inventory(
                natural_documents,
                field="predictions_sha256",
                paths=natural_prediction_paths,
                label="natural P2T metrics/predictions",
            )
            _require_metric_hash_inventory(
                natural_documents,
                field="paired_evaluation_rows_sha256",
                paths=natural_paired_paths,
                label="natural P2T metrics/paired rows",
            )
            _validate_p2t_metric_artifact_pairs(
                natural_documents,
                prediction_paths=natural_prediction_paths,
                paired_paths=natural_paired_paths,
            )
            p2t_dependency_paths = _resolve_many(
                base,
                inputs.get("p2t_checkpoints"),
                label="natural P2T checkpoint dependencies",
            )
            full_checkpoint_paths: dict[int, Path] = {}
            for dependency_path in p2t_dependency_paths:
                checkpoint = torch.load(
                    dependency_path, map_location="cpu", weights_only=True
                )
                if checkpoint.get("input_ablation") != "full":
                    continue
                seed = int(checkpoint.get("seed", -1))
                if (
                    checkpoint.get("schema_version") != P2T_CHECKPOINT_SCHEMA
                    or checkpoint.get("status") != TRAINED_CHECKPOINT_STATUS
                    or checkpoint.get("experiment_config_sha256") != config_sha
                    or checkpoint.get("manifest_sha256") != controlled_dependency_sha
                    or checkpoint.get("learning_split_sha256")
                    != split["split_sha256"]
                    or seed in full_checkpoint_paths
                ):
                    raise RuntimeError("natural P2T checkpoint dependency is invalid")
                full_checkpoint_paths[seed] = dependency_path
            if set(full_checkpoint_paths) != expected_seeds:
                raise RuntimeError("natural P2T lacks one full-arm checkpoint per seed")
            if any(
                document.get("checkpoint_sha256")
                != sha256_file(
                    full_checkpoint_paths[int(document["checkpoint_seed"])]
                )
                for document in natural_documents
            ):
                raise RuntimeError("natural P2T metric uses another seed's checkpoint")
            _require_metric_hash_inventory(
                natural_documents,
                field="checkpoint_sha256",
                paths=list(full_checkpoint_paths.values()),
                label="natural P2T metrics/checkpoints",
            )
            result["artifact_bindings"].update(
                {
                    "p2t_natural_metrics": [
                        _record(path) for path in natural_metric_paths
                    ],
                    "p2t_natural_predictions": [
                        _record(path) for path in natural_prediction_paths
                    ],
                    "p2t_natural_paired_rows": [
                        _record(path) for path in natural_paired_paths
                    ],
                    "controlled_tensor_manifest_dependency": _record(
                        controlled_dependency
                    ),
                    "p2t_natural_checkpoint_dependencies": [
                        _record(full_checkpoint_paths[seed])
                        for seed in sorted(full_checkpoint_paths)
                    ],
                }
            )
            result_artifact_paths.extend(
                natural_metric_paths + natural_prediction_paths + natural_paired_paths
            )
            summary_paths = _resolve_many(
                base, inputs.get("editor_summaries"), label="editor_summaries"
            )
            row_paths = _resolve_many(
                base, inputs.get("editor_rows"), label="editor_rows"
            )
            checkpoint_paths = _resolve_many(
                base, inputs.get("editor_checkpoints"), label="editor_checkpoints"
            )
            documents = [_load_json(path) for path in summary_paths]
            metrics = validate_editor_metric_receipts(
                documents,
                config=config,
                config_sha256=config_sha,
                manifest_sha256=sha256_file(tensor_path),
                learning_split_sha256=split["split_sha256"],
                partition=str(evaluation_partition),
                test_access_receipt_sha256=test_access_receipt_sha256,
            )
            if len(summary_paths) != len(row_paths):
                raise RuntimeError("editor summary/row inventories have different lengths")
            row_hashes = [sha256_file(path) for path in row_paths]
            architecture = load_editor_architecture_contract(config)[
                "primary_architecture_id"
            ]
            editor_seeds = [int(seed) for seed in config["editor"]["training"]["seeds"]]
            trainable_conditions = set(config["editor"]["training_conditions"])
            expected_checkpoint_descriptors = {
                (seed, condition, architecture)
                for seed in editor_seeds
                for condition in trainable_conditions
            }
            checkpoint_path_by_descriptor: dict[tuple[int, str, str], Path] = {}
            for path in checkpoint_paths:
                document = torch.load(path, map_location="cpu", weights_only=True)
                descriptor = (
                    int(document.get("seed", -1)),
                    str(document.get("condition") or ""),
                    str(document.get("architecture_id") or ""),
                )
                if (
                    document.get("schema_version") != EDITOR_CHECKPOINT_SCHEMA
                    or document.get("status") != TRAINED_CHECKPOINT_STATUS
                    or document.get("experiment_config_sha256") != config_sha
                    or (
                        frozen_bindings is None
                        and document.get("manifest_sha256")
                        != sha256_file(tensor_path)
                    )
                    or document.get("learning_split_sha256")
                    != split["split_sha256"]
                    or descriptor in checkpoint_path_by_descriptor
                ):
                    raise RuntimeError("editor inventory contains an invalid checkpoint")
                checkpoint_path_by_descriptor[descriptor] = path
            if set(checkpoint_path_by_descriptor) != expected_checkpoint_descriptors:
                raise RuntimeError("editor checkpoint inventory is not the frozen exact grid")
            if frozen_bindings is not None:
                result["artifact_bindings"]["frozen_editor_checkpoint_roles"] = (
                    _validate_frozen_checkpoint_subset(
                        bindings=frozen_bindings,
                        checkpoint_paths=checkpoint_paths,
                        kind="editor",
                        expected_roles=_expected_frozen_checkpoint_roles(
                            config, "editor"
                        ),
                    )
                )
            summary_path_by_key: dict[tuple[int, str, str], Path] = {}
            row_path_by_key: dict[tuple[int, str, str], Path] = {}
            for document, summary_path in zip(documents, summary_paths):
                condition = str(document.get("condition") or "")
                architecture_id = str(document.get("architecture_id") or "")
                checkpoint_condition = EDITOR_CHECKPOINT_CONDITION_ALIASES.get(
                    condition, condition
                )
                matching_seeds = [
                    seed
                    for seed in editor_seeds
                    if sha256_file(
                        checkpoint_path_by_descriptor[
                            (seed, checkpoint_condition, architecture_id)
                        ]
                    )
                    == document.get("checkpoint_sha256")
                ] if (checkpoint_condition, architecture_id) in {
                    (descriptor[1], descriptor[2])
                    for descriptor in checkpoint_path_by_descriptor
                } else []
                if len(matching_seeds) != 1:
                    raise RuntimeError(
                        "editor summary checkpoint does not identify one frozen seed cell"
                    )
                key = (matching_seeds[0], condition, architecture_id)
                if key in summary_path_by_key:
                    raise RuntimeError("duplicate editor summary run cell")
                summary_path_by_key[key] = summary_path
            expected_summary_keys = {
                (seed, str(condition), architecture)
                for seed in editor_seeds
                for condition in config["editor"]["conditions"]
            }
            if set(summary_path_by_key) != expected_summary_keys:
                raise RuntimeError("editor summary inventory is not the frozen exact grid")
            row_path_by_hash = {sha256_file(path): path for path in row_paths}
            if (
                len(row_path_by_hash) != len(row_paths)
                or Counter(
                    document.get("metric_rows_sha256") for document in documents
                )
                != Counter(row_hashes)
            ):
                raise RuntimeError("editor summaries do not bind the exact row inventory")
            for key, summary_path in summary_path_by_key.items():
                summary_document = documents[summary_paths.index(summary_path)]
                row_path_by_key[key] = row_path_by_hash[
                    str(summary_document["metric_rows_sha256"])
                ]
            leaf_artifacts_by_key = {}
            for key, summary_path in summary_path_by_key.items():
                summary_document = documents[summary_paths.index(summary_path)]
                checkpoint_descriptor = (
                    key[0],
                    EDITOR_CHECKPOINT_CONDITION_ALIASES.get(key[1], key[1]),
                    key[2],
                )
                leaf_artifacts_by_key[key] = (
                    _validate_editor_metric_leaf_artifacts(
                        rows_path=row_path_by_key[key],
                        summary=summary_document,
                        checkpoint_path=checkpoint_path_by_descriptor[
                            checkpoint_descriptor
                        ],
                    )
                )
            editor_confirmatory_path = _resolve(
                base, inputs.get("editor_confirmatory"), label="editor_confirmatory"
            )
            confirmatory, confirmatory_record = _validate_confirmatory(
                editor_confirmatory_path,
                kind="editor",
                config_sha256=config_sha,
            )
            if confirmatory["analysis_status"] == "VALID":
                contrasts = config["statistics"].get("confirmatory_contrasts")
                if not isinstance(contrasts, list) or not contrasts:
                    raise RuntimeError("active editor analysis lacks frozen contrasts")
                confirmatory_conditions = {
                    str(contrast[key])
                    for contrast in contrasts
                    for key in ("treatment", "comparator")
                }
                confirmatory_expected = {}
                for seed in editor_seeds:
                    for condition in confirmatory_conditions:
                        summary_key = (seed, condition, architecture)
                        checkpoint_descriptor = (
                            seed,
                            EDITOR_CHECKPOINT_CONDITION_ALIASES.get(
                                condition, condition
                            ),
                            architecture,
                        )
                        confirmatory_expected[(seed, condition)] = {
                            "checkpoint": _record(
                                checkpoint_path_by_descriptor[checkpoint_descriptor]
                            ),
                            "summary": _record(summary_path_by_key[summary_key]),
                            "rows": _record(row_path_by_key[summary_key]),
                            "leaf_artifacts": leaf_artifacts_by_key[summary_key],
                        }
                _validate_confirmatory_input_inventory(
                    confirmatory,
                    kind="editor",
                    expected=confirmatory_expected,
                )
            result["artifact_bindings"].update(
                {
                    "editor_summaries": [_record(path) for path in summary_paths],
                    "editor_rows": [_record(path) for path in row_paths],
                    "editor_checkpoints": [_record(path) for path in checkpoint_paths],
                    "editor_confirmatory": confirmatory_record,
                }
            )
            result_artifact_paths.extend(
                summary_paths
                + row_paths
                + checkpoint_paths
                + [editor_confirmatory_path]
            )
            confirmatory_artifact_paths.append(editor_confirmatory_path)
            result["editor_lane"].update(
                status="RESULTS_READY",
                **metrics,
                confirmatory_analysis_status=confirmatory["analysis_status"],
                family_verdicts=confirmatory.get("family_verdicts"),
            )
    if target == "complete":
        if inputs.get("p2t_secondary_checkpoints") not in (None, []):
            raise RuntimeError(
                "six-class simple-first completion forbids deferred P2T architectures"
            )
        expected_paths = _resolve_many(
            base,
            inputs.get("expected_result_artifacts"),
            label="expected_result_artifacts",
        )
        if len(expected_paths) != len({str(path) for path in expected_paths}):
            raise RuntimeError("final expected result artifact inventory contains duplicates")
        if Counter(str(path) for path in expected_paths) != Counter(
            str(path) for path in result_artifact_paths
        ):
            raise RuntimeError("final expected result artifact inventory is not exact")
        if Counter(str(path) for path in confirmatory_artifact_paths) != Counter(
            {
                str(_resolve(base, inputs.get("p2t_confirmatory"), label="p2t_confirmatory")),
                str(
                    _resolve(
                        base,
                        inputs.get("editor_confirmatory"),
                        label="editor_confirmatory",
                    )
                ),
            }
        ) or any(path not in expected_paths for path in confirmatory_artifact_paths):
            raise RuntimeError("every confirmatory analysis must be an expected artifact")
        result["artifact_bindings"]["expected_result_artifacts"] = [
            _record(path) for path in expected_paths
        ]
        result["completion"] = {
            "status": "ROUTE_A_COMPLETE",
            "analysis_state": (
                "CONFIRMATORY_VALID"
                if all(
                    _load_json(path).get("analysis_status") == "VALID"
                    for path in confirmatory_artifact_paths
                )
                else "DESCRIPTIVE_ONLY_PENDING_EFFECT_FREEZE"
            ),
            "hypothesis_support_required": False,
            "all_expected_result_hashes_bound": True,
            "deferred_architecture_results_in_current_campaign": False,
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    receipt = validate_pipeline(args.inputs, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({**receipt, "receipt_sha256": sha256_file(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
