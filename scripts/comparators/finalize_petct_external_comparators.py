#!/usr/bin/env python3
"""Build or validate the immutable external-comparator completion receipt.

The receipt is an inventory, never a pooled comparison table.  It binds every
input/output/metric/prediction artifact actually used by each selected method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "PETCT-EXTERNAL-COMPARATORS-COMPLETE-v1.2"
STATUS = "COMPLETE"
METRICS_SCHEMA = "PETCT-EXTERNAL-COMPARATOR-METRICS-v1.0"
OUTPUT_SCHEMA = "PETCT-EXTERNAL-COMPARATOR-OUTPUT-v1.0"
INPUT_SCHEMA = "PETCT-EXTERNAL-COMPARATOR-INPUT-v1.0"
REQUIRED_METRICS = (
    "dice",
    "dice_delta_vs_m0",
    "dmm",
    "false_positive_volume_ml",
    "false_negative_volume_ml",
    "authorized_residual_recall",
    "prompt_distal_recall",
    "unauthorized_addition_volume_ml",
    "m0_preservation_rate",
    "other_lesion_harm",
    "unintended_bridge_or_merge_rate",
    "runtime_seconds",
    "peak_gpu_memory_mib",
)
PUBLIC_TO_INTERNAL_PARTITION = {
    "train": "train",
    "validation": "val",
    "test": "test",
}
METRIC_RANGES: dict[str, tuple[float | None, float | None]] = {
    "dice": (0.0, 1.0),
    "dice_delta_vs_m0": (-1.0, 1.0),
    "dmm": (0.0, 1.0),
    "false_positive_volume_ml": (0.0, None),
    "false_negative_volume_ml": (0.0, None),
    "authorized_residual_recall": (0.0, 1.0),
    "prompt_distal_recall": (0.0, 1.0),
    "unauthorized_addition_volume_ml": (0.0, None),
    "m0_preservation_rate": (0.0, 1.0),
    "other_lesion_harm": (0.0, 1.0),
    "unintended_bridge_or_merge_rate": (0.0, 1.0),
    "runtime_seconds": (0.0, None),
    "peak_gpu_memory_mib": (0.0, None),
}
METHOD_POLICIES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "scribbleprompt": (
        ("union_with_m0", "2D", "POSITIVE_ONLY_DIAGNOSTIC"),
        ("native_slice_replace", "2D", "NATIVE_DIAGNOSTIC"),
    ),
    "nninteractive": (
        ("union_with_m0", "3D", "POSITIVE_ONLY_DIAGNOSTIC"),
        ("native_full_mask", "3D", "NATIVE_DIAGNOSTIC"),
    ),
}


class ExternalCompleteError(RuntimeError):
    """Raised when an external completion inventory is incomplete or stale."""


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ExternalCompleteError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise ExternalCompleteError(f"{label} is missing: {resolved}")
    return resolved


def _record(path: Path, *, role: str) -> dict[str, Any]:
    resolved = _regular(path, label=role)
    return {
        "role": role,
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _executable_record(path: Path, *, role: str) -> dict[str, Any]:
    record = _record(path, role=role)
    if not os.access(record["path"], os.X_OK):
        raise ExternalCompleteError(f"{role} is not executable: {record['path']}")
    return record


def _same_file_record(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("path", "sha256", "bytes"))


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ExternalCompleteError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_nonfinite(child, label=label)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite(child, label=label)


def _required_metric_value(row: Mapping[str, Any], metric: str) -> float:
    value = row.get(metric)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ExternalCompleteError(
            f"metric row {metric} must be a finite real numeric value"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ExternalCompleteError(f"metric row {metric} is non-finite")
    lower, upper = METRIC_RANGES[metric]
    if (lower is not None and numeric < lower) or (
        upper is not None and numeric > upper
    ):
        raise ExternalCompleteError(f"metric row {metric} is outside its valid range")
    return numeric


def _load_object(path: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular(path, label=label)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalCompleteError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ExternalCompleteError(f"{label} must be a JSON object")
    return resolved, payload


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parse_runtime_receipts(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        method, separator, raw_path = value.partition("=")
        if not separator or method not in METHOD_POLICIES or not raw_path:
            raise ExternalCompleteError(
                "runtime receipt must be METHOD=PATH for a supported method"
            )
        if method in result:
            raise ExternalCompleteError(f"duplicate runtime receipt: {method}")
        result[method] = Path(raw_path)
    return result


def _selected_methods(values: Sequence[str]) -> list[str]:
    selected = list(values)
    if not selected or any(value not in METHOD_POLICIES for value in selected):
        raise ExternalCompleteError("selected methods must be a non-empty supported subset")
    if len(selected) != len(set(selected)):
        raise ExternalCompleteError("selected methods must not contain duplicates")
    return sorted(selected)


def _load_jsonl(path: Path, *, label: str) -> tuple[Path, list[dict[str, Any]]]:
    resolved = _regular(path, label=label)
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            resolved.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ExternalCompleteError(
                    f"{label} line {line_number} must be a JSON object"
                )
            rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalCompleteError(f"{label} must be valid UTF-8 JSONL") from exc
    if not rows:
        raise ExternalCompleteError(f"{label} must not be empty")
    return resolved, rows


def _validate_input_inventory(
    *,
    input_payload: Mapping[str, Any],
    input_rows: Sequence[Mapping[str, Any]],
    learning_split: Path,
    natural_episode_manifest: Path,
    partition: str,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Bind every input episode to the frozen split and episode inventory."""

    common_dir = Path(__file__).resolve().parents[1] / "common"
    if str(common_dir) not in sys.path:
        sys.path.insert(0, str(common_dir))
    from petct_learning import (  # noqa: PLC0415
        LearningContractError,
        validate_manifest_rows_against_frozen_learning_split,
    )

    public_partition = "validation" if partition == "val" else "test"
    normalized_inputs: list[dict[str, Any]] = []
    for index, row in enumerate(input_rows, start=1):
        split_receipt = row.get("patient_split_receipt")
        if not isinstance(split_receipt, Mapping):
            raise ExternalCompleteError(
                f"external input row {index} omits patient_split_receipt"
            )
        normalized_inputs.append(
            {
                "case_id": row.get("case_id"),
                "patient_id": row.get("patient_id"),
                "partition": PUBLIC_TO_INTERNAL_PARTITION.get(
                    str(row.get("split") or "")
                ),
                "episode_id": row.get("episode_id"),
                "learning_split_sha256": split_receipt.get(
                    "learning_split_sha256"
                ),
            }
        )
    natural_path, natural_rows = _load_jsonl(
        natural_episode_manifest, label="frozen natural episode manifest"
    )
    selected_natural = [
        row for row in natural_rows if row.get("partition") == partition
    ]
    if not selected_natural:
        raise ExternalCompleteError(
            "frozen natural episode manifest has no selected partition"
        )
    try:
        input_validation = validate_manifest_rows_against_frozen_learning_split(
            normalized_inputs,
            learning_split,
            require_episode_id=True,
            allowed_partitions={partition},
        )
        natural_validation = validate_manifest_rows_against_frozen_learning_split(
            selected_natural,
            learning_split,
            require_episode_id=True,
            allowed_partitions={partition},
        )
    except LearningContractError as exc:
        raise ExternalCompleteError(
            f"external input frozen learning-split validation failed: {exc}"
        ) from exc
    provenance = input_payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ExternalCompleteError("external input manifest omits provenance")
    if (
        provenance.get("learning_split_sha256")
        != input_validation["learning_split_sha256"]
        or provenance.get("natural_episode_manifest_sha256")
        != _sha256_file(natural_path)
    ):
        raise ExternalCompleteError(
            "external input provenance differs from frozen split/episode inventory"
        )

    def identity(row: Mapping[str, Any], *, public: bool) -> tuple[str, str, str, str]:
        raw_partition = str(row.get("split" if public else "partition") or "")
        normalized_partition = (
            PUBLIC_TO_INTERNAL_PARTITION.get(raw_partition)
            if public
            else raw_partition
        )
        return (
            str(row.get("episode_id") or ""),
            str(row.get("case_id") or ""),
            str(row.get("patient_id") or "").casefold(),
            str(normalized_partition or ""),
        )

    input_inventory = {identity(row, public=True) for row in input_rows}
    natural_inventory = {identity(row, public=False) for row in selected_natural}
    if len(input_inventory) != len(input_rows) or len(natural_inventory) != len(
        selected_natural
    ):
        raise ExternalCompleteError("frozen episode inventory contains duplicates")
    if input_inventory != natural_inventory:
        raise ExternalCompleteError(
            "external input does not exactly match the frozen episode-subset inventory"
        )
    by_case: dict[str, dict[str, str]] = {}
    for episode_id, case_id, patient_id, internal_partition in input_inventory:
        if case_id in by_case:
            raise ExternalCompleteError(
                "external completion requires one frozen episode per case"
            )
        by_case[case_id] = {
            "episode_id": episode_id,
            "patient_id": patient_id,
            "partition": internal_partition,
            "public_partition": public_partition,
        }
    return by_case, {
        "manifest": _record(natural_path, role="natural_episode_manifest"),
        "partition": partition,
        "episode_inventory_sha256": _canonical_sha256(
            sorted(input_inventory)
        ),
        "episode_count": len(input_inventory),
        "learning_split_sha256": natural_validation["learning_split_sha256"],
    }


def _validate_runtime_admissions(
    *,
    comparator_config: Path,
    selected: Sequence[str],
    runtime_receipts: Mapping[str, Path],
) -> tuple[list[dict[str, Any]], Mapping[str, Any] | None]:
    """Revalidate each current config-bound runtime before artifact closure."""

    _, config = _load_object(comparator_config, label="comparator config")
    methods = config.get("methods")
    if not isinstance(methods, list):
        raise ExternalCompleteError("comparator config omits methods")
    by_method = {
        str(method.get("id") or ""): method
        for method in methods
        if isinstance(method, Mapping)
    }
    comparator_dir = Path(__file__).resolve().parent
    if str(comparator_dir) not in sys.path:
        sys.path.insert(0, str(comparator_dir))
    from run_petct_external_comparator import (  # noqa: PLC0415
        ContractError,
        validate_execution_admission,
    )

    admissions: list[dict[str, Any]] = []
    nninteractive_checkpoint: Mapping[str, Any] | None = None
    for method_id in selected:
        method = by_method.get(method_id)
        if method is None:
            raise ExternalCompleteError(
                f"comparator config does not declare selected method {method_id}"
            )
        try:
            admission = validate_execution_admission(
                method,
                comparator_config,
                variables={
                    "project_root": str(comparator_config.parent.parent.resolve())
                },
            )
        except ContractError as exc:
            raise ExternalCompleteError(
                f"{method_id} runtime admission is invalid: {exc}"
            ) from exc
        supplied_receipt = _record(
            runtime_receipts[method_id], role=f"runtime_receipt:{method_id}"
        )
        if any(
            admission["receipt"].get(key) != supplied_receipt.get(key)
            for key in ("path", "sha256")
        ):
            raise ExternalCompleteError(
                f"{method_id} runtime receipt differs from current comparator config"
            )
        admissions.append(dict(admission))
        if method_id == "nninteractive":
            checkpoint = admission.get("bound_files", {}).get("checkpoint_sha256")
            if not isinstance(checkpoint, Mapping):
                raise ExternalCompleteError(
                    "nninteractive runtime admission omits checkpoint binding"
                )
            nninteractive_checkpoint = checkpoint
    return admissions, nninteractive_checkpoint


def _prediction_records(
    *,
    output_manifest: Path,
    prediction_dir: Path,
    method: str,
    policy: str,
    input_by_case: Mapping[str, Mapping[str, str]],
    learning_split_sha256: str,
    test_access_receipt_sha256: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, payload = _load_object(output_manifest, label=f"{method}/{policy} output manifest")
    if (
        payload.get("schema_version") != OUTPUT_SCHEMA
        or payload.get("method_id") != method
        or payload.get("output_policy") != policy
        or payload.get("learning_split_sha256") != learning_split_sha256
        or payload.get("test_access_receipt_sha256")
        != test_access_receipt_sha256
    ):
        raise ExternalCompleteError("output manifest method/policy differs from its table")
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ExternalCompleteError("output manifest records must be non-empty")
    expected_root = prediction_dir.resolve()
    if not expected_root.is_dir() or prediction_dir.is_symlink():
        raise ExternalCompleteError(f"prediction directory is missing: {expected_root}")
    records: list[dict[str, Any]] = []
    observed_cases: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping) or raw.get("status") != "complete":
            raise ExternalCompleteError("completion receipt rejects failed prediction rows")
        case_id = str(raw.get("case_id") or "")
        input_identity = input_by_case.get(case_id)
        if (
            not case_id
            or case_id in observed_cases
            or input_identity is None
            or raw.get("method_id") != method
            or raw.get("output_policy") != policy
            or str(raw.get("patient_id") or "").casefold()
            != input_identity["patient_id"]
        ):
            raise ExternalCompleteError(
                "output manifest has duplicate, missing, or out-of-scope case_id"
            )
        observed_cases.add(case_id)
        raw_path = raw.get("prediction_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ExternalCompleteError("complete output row omits prediction_path")
        path = Path(raw_path)
        if not path.is_absolute():
            path = output_manifest.resolve().parent / path
        resolved = _regular(path, label=f"{method}/{policy} prediction {index}")
        if not resolved.is_relative_to(expected_root):
            raise ExternalCompleteError("prediction leaf escapes its method/policy directory")
        record = _record(
            resolved,
            role=f"prediction:{method}:{policy}:{index:06d}",
        )
        declared_sha = raw.get("prediction_sha256")
        if declared_sha != record["sha256"]:
            raise ExternalCompleteError("output manifest prediction SHA changed")
        records.append(record)
    if observed_cases != set(input_by_case):
        raise ExternalCompleteError(
            "output manifest does not exactly cover the frozen input cases"
        )
    if len({record["path"] for record in records}) != len(records):
        raise ExternalCompleteError("one prediction leaf cannot satisfy multiple output rows")
    inventory: set[str] = set()
    for leaf in expected_root.rglob("*"):
        if leaf.is_symlink():
            raise ExternalCompleteError("prediction inventory contains a symlink")
        if leaf.is_file():
            inventory.add(str(leaf.resolve()))
    if inventory != {record["path"] for record in records}:
        raise ExternalCompleteError(
            "prediction directory contains missing or unreferenced leaf files"
        )
    return records, [dict(row) for row in rows]


def _table_record(
    *,
    run_root: Path,
    input_manifest: Path,
    comparator_config: Path,
    experiment_config: Path,
    learning_split: Path,
    method: str,
    policy: str,
    dimensionality: str,
    comparison_role: str,
    frozen_external_admission: Path | None,
    input_by_case: Mapping[str, Mapping[str, str]],
    official_metrics: Path,
    test_access_receipt_sha256: str | None,
    natural_episode_manifest_sha256: str,
    nninteractive_checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    artifacts = run_root / "artifacts"
    metrics = run_root / "metrics"
    output_manifest = artifacts / f"{method}_{policy}_output.json"
    rows_path = metrics / f"{method}_{policy}_rows.jsonl"
    summary_path = metrics / f"{method}_{policy}_summary.json"
    prediction_dir = run_root / "predictions" / f"{method}_{policy}"
    predictions, output_rows = _prediction_records(
        output_manifest=output_manifest,
        prediction_dir=prediction_dir,
        method=method,
        policy=policy,
        input_by_case=input_by_case,
        learning_split_sha256=_sha256_file(learning_split),
        test_access_receipt_sha256=test_access_receipt_sha256,
    )
    _, summary = _load_object(summary_path, label=f"{method}/{policy} summary")
    expected_summary = {
        "schema_version": METRICS_SCHEMA,
        "status": "COMPLETE",
        "method_id": method,
        "output_policy": policy,
        "spatial_dimensionality": dimensionality,
        "comparison_role": comparison_role,
        "cross_dimensional_pooling": "FORBIDDEN",
        "separate_fairness_table": True,
        "input_manifest_sha256": _sha256_file(input_manifest),
        "output_manifest_sha256": _sha256_file(output_manifest),
        "experiment_config_sha256": _sha256_file(experiment_config),
        "learning_split_sha256": _sha256_file(learning_split),
        "test_access_receipt_sha256": test_access_receipt_sha256,
        "official_metrics_sha256": _sha256_file(official_metrics),
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ExternalCompleteError(f"{method}/{policy} summary binding is incomplete")
    rows_path = _regular(rows_path, label="metric rows")
    row_lines = [
        line
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(row_lines) != len(output_rows)
        or summary.get("record_count") != len(output_rows)
        or summary.get("complete_count") != len(output_rows)
        or summary.get("failed_count") != 0
        or summary.get("metric_rows_sha256") != _sha256_file(rows_path)
    ):
        raise ExternalCompleteError("metric row/output record counts differ")
    natural_sha = summary.get("natural_episode_manifest_sha256")
    if natural_sha != natural_episode_manifest_sha256:
        raise ExternalCompleteError("metric summary omits natural-episode provenance")
    patient_clustered = summary.get("patient_clustered")
    if not isinstance(patient_clustered, Mapping) or set(patient_clustered) != set(
        REQUIRED_METRICS
    ):
        raise ExternalCompleteError("metric summary patient-clustered metrics are incomplete")
    observed_cases: set[str] = set()
    metric_rows: list[dict[str, Any]] = []
    for line in row_lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExternalCompleteError("metric rows are not valid JSONL") from exc
        if not isinstance(row, Mapping):
            raise ExternalCompleteError("metric row must be a JSON object")
        _reject_nonfinite(row, label="metric row")
        case_id = str(row.get("case_id") or "")
        input_identity = input_by_case.get(case_id)
        if (
            row.get("method_id") != method
            or row.get("output_policy") != policy
            or row.get("status") != "complete"
            or not case_id
            or case_id in observed_cases
            or input_identity is None
            or str(row.get("patient_id") or "").casefold()
            != input_identity["patient_id"]
            or row.get("split") != input_identity["public_partition"]
            or str(row.get("episode_id") or "")
            != input_identity["episode_id"]
            or any(name not in row for name in REQUIRED_METRICS)
        ):
            raise ExternalCompleteError("metric row method/policy differs from its table")
        for metric in REQUIRED_METRICS:
            _required_metric_value(row, metric)
        observed_cases.add(case_id)
        metric_rows.append(dict(row))
    if observed_cases != set(input_by_case):
        raise ExternalCompleteError("metric rows do not exactly cover frozen input cases")
    common_dir = Path(__file__).resolve().parents[1] / "common"
    if str(common_dir) not in sys.path:
        sys.path.insert(0, str(common_dir))
    from petct_route_a_core import (  # noqa: PLC0415
        ContractError,
        patient_cluster_summary,
    )

    try:
        recomputed_clustered = {
            metric: patient_cluster_summary(metric_rows, metric)
            for metric in REQUIRED_METRICS
        }
    except ContractError as exc:
        raise ExternalCompleteError(
            f"patient-clustered metric recomputation failed: {exc}"
        ) from exc
    if patient_clustered != recomputed_clustered:
        raise ExternalCompleteError(
            "metric summary patient_clustered differs from recomputed rows"
        )
    if summary.get("patient_count") != len(
        {identity["patient_id"] for identity in input_by_case.values()}
    ):
        raise ExternalCompleteError("metric summary patient_count differs from input")
    if method == "nninteractive":
        if not isinstance(nninteractive_checkpoint, Mapping):
            raise ExternalCompleteError(
                "nninteractive table omits current runtime checkpoint"
            )
        checkpoint_sha = nninteractive_checkpoint.get("sha256")
        if not isinstance(checkpoint_sha, str) or not checkpoint_sha:
            raise ExternalCompleteError(
                "nninteractive runtime admission omits checkpoint SHA"
            )
        expected_source_id = f"nnInteractive_v1.0:{checkpoint_sha}"
        if any(
            row.get("checkpoint_sha256") != checkpoint_sha
            or row.get("source_checkpoint_id") != expected_source_id
            for row in output_rows
        ):
            raise ExternalCompleteError(
                "nnInteractive output did not use the current admitted checkpoint"
            )
    if method == "nninteractive" and frozen_external_admission is not None:
        _, admission = _load_object(
            frozen_external_admission, label="frozen external admission"
        )
        checkpoint = admission.get("model", {}).get("checkpoint", {})
        checkpoint_sha = checkpoint.get("sha256")
        if not isinstance(checkpoint_sha, str) or not checkpoint_sha:
            raise ExternalCompleteError("frozen external admission omits checkpoint SHA")
        if checkpoint_sha != nninteractive_checkpoint.get("sha256"):
            raise ExternalCompleteError(
                "current nninteractive runtime checkpoint differs from final freeze"
            )
    return {
        "method_id": method,
        "spatial_dimensionality": dimensionality,
        "output_policy": policy,
        "comparison_role": comparison_role,
        "pretraining_exposure": summary.get("pretraining_exposure"),
        "headline_eligible": summary.get("headline_eligible"),
        "output_manifest": _record(output_manifest, role="output_manifest"),
        "metric_rows": _record(rows_path, role="metric_rows"),
        "summary": _record(summary_path, role="summary"),
        "prediction_artifacts": predictions,
        "prediction_inventory_sha256": _canonical_sha256(predictions),
        "runtime_checkpoint": (
            dict(nninteractive_checkpoint)
            if method == "nninteractive" and nninteractive_checkpoint is not None
            else None
        ),
    }


def _derive_payload(
    *,
    run_root: Path,
    partition: str,
    selected_methods: Sequence[str],
    comparator_config: Path,
    experiment_config: Path,
    learning_split: Path,
    input_manifest: Path,
    natural_episode_manifest: Path,
    runtime_receipts: Mapping[str, Path],
    core_python: Path,
    official_metrics: Path,
    nninteractive_python: Path | None,
    test_access_receipt: Path | None,
    frozen_external_admission: Path | None,
    route_a_run_root: Path | None,
    output_path: Path | None = None,
    ledger_root: Path | None = None,
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    if partition not in {"val", "test"}:
        raise ExternalCompleteError("partition must be val or test")
    selected = _selected_methods(selected_methods)
    if set(runtime_receipts) != set(selected):
        raise ExternalCompleteError("runtime receipt roles must exactly match selected methods")
    if "nninteractive" in selected and nninteractive_python is None:
        raise ExternalCompleteError("nninteractive completion requires its Python executable")
    resolved_test: dict[str, Any] | None = None
    if partition == "val":
        if (
            test_access_receipt is not None
            or frozen_external_admission is not None
            or route_a_run_root is not None
        ):
            raise ExternalCompleteError("validation rejects test/frozen-admission inputs")
    else:
        if selected != ["nninteractive"]:
            raise ExternalCompleteError(
                "formal test external execution is admitted only for nninteractive"
            )
        if (
            test_access_receipt is None
            or frozen_external_admission is None
            or route_a_run_root is None
        ):
            raise ExternalCompleteError(
                "formal test completion requires access receipt, Route-A root, and external admission"
            )
        common_dir = Path(__file__).resolve().parents[1] / "common"
        if str(common_dir) not in sys.path:
            sys.path.insert(0, str(common_dir))
        from petct_development_freeze import (  # noqa: PLC0415
            DevelopmentFreezeError,
            resolve_test_external_binding,
        )
        from petct_test_access import TestAccessError, enforce_partition_access  # noqa: PLC0415

        try:
            resolved_test = resolve_test_external_binding(
                test_access_receipt=test_access_receipt,
                experiment_config=experiment_config,
                run_root=route_a_run_root,
                method_id="nninteractive",
                ledger_root=ledger_root,
            )
            access_kwargs: dict[str, Any] = {}
            if ledger_root is not None:
                access_kwargs["ledger_root"] = ledger_root
            enforce_partition_access(
                "test",
                receipt_path=test_access_receipt,
                experiment_config=experiment_config,
                learning_split=learning_split,
                run_root=route_a_run_root,
                output_paths=(() if output_path is None else (output_path,)),
                **access_kwargs,
            )
        except (DevelopmentFreezeError, TestAccessError) as exc:
            raise ExternalCompleteError(
                f"formal test access/freeze binding is invalid: {exc}"
            ) from exc
        route_root = Path(route_a_run_root).resolve()
        if not run_root.is_relative_to(route_root) or run_root == route_root:
            raise ExternalCompleteError(
                "external formal-test run root must be a child of the receipt-bound Route-A root"
            )

    comparator_config = _regular(comparator_config, label="comparator config")
    experiment_config = _regular(experiment_config, label="experiment config")
    learning_split = _regular(learning_split, label="learning split")
    core_python_record = _executable_record(
        core_python, role="core_python_executable"
    )
    official_metrics_record = _record(
        official_metrics, role="official_autopetv_metrics"
    )
    nninteractive_python_record = (
        _executable_record(
            nninteractive_python, role="nninteractive_python_executable"
        )
        if nninteractive_python is not None
        else None
    )
    if resolved_test is not None:
        expected_records = (
            (core_python_record, resolved_test["core_python"], "core Python"),
            (
                official_metrics_record,
                resolved_test["official_metrics"],
                "official AutoPET V metrics",
            ),
            (
                nninteractive_python_record or {},
                resolved_test["nninteractive_python"],
                "nninteractive Python",
            ),
        )
        if any(not _same_file_record(actual, expected) for actual, expected, _ in expected_records):
            raise ExternalCompleteError(
                "formal test executable/metrics differ from the receipt-bound final freeze"
            )
        for actual_path, expected_record, label in (
            (comparator_config, resolved_test["comparator_config"], "comparator config"),
            (experiment_config, resolved_test["experiment_config"], "experiment config"),
            (learning_split, resolved_test["learning_split"], "learning split"),
            (
                frozen_external_admission,
                resolved_test["admission"],
                "external admission",
            ),
        ):
            if Path(actual_path).resolve() != Path(expected_record["path"]).resolve():
                raise ExternalCompleteError(
                    f"formal test {label} differs from the receipt-bound final freeze"
                )

    runtime_records = [
        {
            "method_id": method,
            "receipt": _record(
                runtime_receipts[method], role=f"runtime_receipt:{method}"
            ),
        }
        for method in selected
    ]
    runtime_admissions, nninteractive_checkpoint = _validate_runtime_admissions(
        comparator_config=comparator_config,
        selected=selected,
        runtime_receipts=runtime_receipts,
    )
    if resolved_test is not None:
        runtime_record = next(
            row["receipt"]
            for row in runtime_records
            if row["method_id"] == "nninteractive"
        )
        if not _same_file_record(runtime_record, resolved_test["runtime_receipt"]):
            raise ExternalCompleteError(
                "formal test runtime receipt differs from the frozen external admission"
            )
        if (
            not isinstance(nninteractive_checkpoint, Mapping)
            or any(
                nninteractive_checkpoint.get(key)
                != resolved_test["model_checkpoint"].get(key)
                for key in ("path", "sha256")
            )
        ):
            raise ExternalCompleteError(
                "formal test runtime checkpoint differs from the receipt-bound final freeze"
            )

    input_manifest = _regular(input_manifest, label="external input manifest")
    _, input_payload = _load_object(input_manifest, label="external input manifest")
    input_rows = input_payload.get("records")
    public_partition = "validation" if partition == "val" else "test"
    if (
        input_payload.get("schema_version") != INPUT_SCHEMA
        or input_payload.get("status") != "FROZEN_INPUT_READY"
        or input_payload.get("partition") != public_partition
        or not isinstance(input_rows, list)
        or not input_rows
        or input_payload.get("record_count") != len(input_rows)
    ):
        raise ExternalCompleteError("external input manifest records must be non-empty")
    if any(
        not isinstance(row, Mapping) or row.get("split") != public_partition
        for row in input_rows
    ):
        raise ExternalCompleteError("external input manifest contains another partition")
    input_by_case, episode_inventory = _validate_input_inventory(
        input_payload=input_payload,
        input_rows=input_rows,
        learning_split=learning_split,
        natural_episode_manifest=natural_episode_manifest,
        partition=partition,
    )
    natural_episode_sha256 = episode_inventory["manifest"]["sha256"]

    test_access_receipt_record = (
        _record(test_access_receipt, role="test_access_receipt")
        if test_access_receipt is not None
        else None
    )
    tables = [
        _table_record(
            run_root=run_root,
            input_manifest=input_manifest,
            comparator_config=comparator_config,
            experiment_config=experiment_config,
            learning_split=learning_split,
            method=method,
            policy=policy,
            dimensionality=dimensionality,
            comparison_role=comparison_role,
            frozen_external_admission=frozen_external_admission,
            input_by_case=input_by_case,
            official_metrics=Path(official_metrics_record["path"]),
            test_access_receipt_sha256=(
                None
                if test_access_receipt_record is None
                else test_access_receipt_record["sha256"]
            ),
            natural_episode_manifest_sha256=natural_episode_sha256,
            nninteractive_checkpoint=nninteractive_checkpoint,
        )
        for method in selected
        for policy, dimensionality, comparison_role in METHOD_POLICIES[method]
    ]
    unsigned: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "partition": partition,
        "selected_methods": selected,
        "comparator_config": _record(comparator_config, role="comparator_config"),
        "experiment_config": _record(experiment_config, role="experiment_config"),
        "learning_split": _record(learning_split, role="learning_split"),
        "input_manifest": _record(input_manifest, role="input_manifest"),
        "natural_episode_inventory": episode_inventory,
        "runtime_receipts": runtime_records,
        "runtime_admissions": runtime_admissions,
        "execution_artifacts": {
            "core_python": core_python_record,
            "official_metrics": official_metrics_record,
            "nninteractive_python": nninteractive_python_record,
        },
        "tables": tables,
        "no_combined_fairness_table": True,
        "fairness_boundaries": {
            "cross_dimensional_pooling": "FORBIDDEN",
            "native_vs_add_only_pooling": "FORBIDDEN",
            "exposed_pretraining_separate": True,
        },
        "claim_boundary": (
            "external spatial comparators only; no pooled 2D-vs-3D, native-vs-add-only, "
            "or exposed-vs-target-unseen inferential table"
        ),
    }
    if test_access_receipt is not None:
        unsigned["test_access_receipt"] = test_access_receipt_record
        unsigned["route_a_run_root"] = str(Path(route_a_run_root).resolve())
    if frozen_external_admission is not None:
        unsigned["frozen_external_admission"] = _record(
            frozen_external_admission, role="frozen_external_admission"
        )
    return {**unsigned, "receipt_sha256": _canonical_sha256(unsigned)}


def build_external_complete(
    *,
    run_root: Path,
    partition: str,
    selected_methods: Sequence[str],
    comparator_config: Path,
    experiment_config: Path,
    learning_split: Path,
    input_manifest: Path,
    natural_episode_manifest: Path,
    runtime_receipts: Mapping[str, Path],
    core_python: Path,
    official_metrics: Path,
    output: Path,
    nninteractive_python: Path | None = None,
    test_access_receipt: Path | None = None,
    frozen_external_admission: Path | None = None,
    route_a_run_root: Path | None = None,
    ledger_root: Path | None = None,
) -> dict[str, Any]:
    payload = _derive_payload(
        run_root=run_root,
        partition=partition,
        selected_methods=selected_methods,
        comparator_config=comparator_config,
        experiment_config=experiment_config,
        learning_split=learning_split,
        input_manifest=input_manifest,
        natural_episode_manifest=natural_episode_manifest,
        runtime_receipts=runtime_receipts,
        core_python=core_python,
        official_metrics=official_metrics,
        nninteractive_python=nninteractive_python,
        test_access_receipt=test_access_receipt,
        frozen_external_admission=frozen_external_admission,
        route_a_run_root=route_a_run_root,
        output_path=output,
        ledger_root=ledger_root,
    )
    try:
        _write_exclusive(output, payload)
    except FileExistsError as exc:
        raise ExternalCompleteError(f"external completion receipt exists: {output}") from exc
    return payload


def validate_external_complete(path: Path) -> dict[str, Any]:
    _, payload = _load_object(path, label="external completion receipt")
    seal = payload.get("receipt_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if seal != _canonical_sha256(unsigned):
        raise ExternalCompleteError("external completion receipt self-hash mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != STATUS:
        raise ExternalCompleteError("external completion receipt contract mismatch")
    try:
        run_root = Path(str(payload["input_manifest"]["path"])).resolve().parents[1]
        selected = payload["selected_methods"]
        runtime_receipts = {
            row["method_id"]: Path(row["receipt"]["path"])
            for row in payload["runtime_receipts"]
        }
        execution = payload["execution_artifacts"]
        expected = _derive_payload(
            run_root=run_root,
            partition=str(payload["partition"]),
            selected_methods=selected,
            comparator_config=Path(payload["comparator_config"]["path"]),
            experiment_config=Path(payload["experiment_config"]["path"]),
            learning_split=Path(payload["learning_split"]["path"]),
            input_manifest=Path(payload["input_manifest"]["path"]),
            natural_episode_manifest=Path(
                payload["natural_episode_inventory"]["manifest"]["path"]
            ),
            runtime_receipts=runtime_receipts,
            core_python=Path(execution["core_python"]["path"]),
            official_metrics=Path(execution["official_metrics"]["path"]),
            nninteractive_python=(
                Path(execution["nninteractive_python"]["path"])
                if execution.get("nninteractive_python") is not None
                else None
            ),
            test_access_receipt=(
                Path(payload["test_access_receipt"]["path"])
                if "test_access_receipt" in payload
                else None
            ),
            frozen_external_admission=(
                Path(payload["frozen_external_admission"]["path"])
                if "frozen_external_admission" in payload
                else None
            ),
            route_a_run_root=(
                Path(payload["route_a_run_root"])
                if "route_a_run_root" in payload
                else None
            ),
        )
    except (KeyError, TypeError, IndexError) as exc:
        raise ExternalCompleteError("external completion receipt structure is invalid") from exc
    if payload != expected:
        raise ExternalCompleteError(
            "external completion receipt differs from recursively rehashed artifacts"
        )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--run-root", type=Path, required=True)
    build.add_argument("--partition", choices=("val", "test"), required=True)
    build.add_argument("--method", action="append", required=True)
    build.add_argument("--comparator-config", type=Path, required=True)
    build.add_argument("--experiment-config", type=Path, required=True)
    build.add_argument("--learning-split", type=Path, required=True)
    build.add_argument("--input-manifest", type=Path, required=True)
    build.add_argument("--natural-episode-manifest", type=Path, required=True)
    build.add_argument("--runtime-receipt", action="append", default=[])
    build.add_argument("--core-python", type=Path, required=True)
    build.add_argument("--official-metrics", type=Path, required=True)
    build.add_argument("--nninteractive-python", type=Path)
    build.add_argument("--test-access-receipt", type=Path)
    build.add_argument("--frozen-external-admission", type=Path)
    build.add_argument("--route-a-run-root", type=Path)
    build.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            payload = build_external_complete(
                run_root=args.run_root,
                partition=args.partition,
                selected_methods=args.method,
                comparator_config=args.comparator_config,
                experiment_config=args.experiment_config,
                learning_split=args.learning_split,
                input_manifest=args.input_manifest,
                natural_episode_manifest=args.natural_episode_manifest,
                runtime_receipts=_parse_runtime_receipts(args.runtime_receipt),
                core_python=args.core_python,
                official_metrics=args.official_metrics,
                nninteractive_python=args.nninteractive_python,
                output=args.output,
                test_access_receipt=args.test_access_receipt,
                frozen_external_admission=args.frozen_external_admission,
                route_a_run_root=args.route_a_run_root,
            )
        else:
            payload = validate_external_complete(args.receipt)
    except ExternalCompleteError as exc:
        parser.error(str(exc))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
