#!/usr/bin/env python3
"""Build and validate the content-addressed final development freeze.

This artifact is deliberately separate from the human test-access grant.  A
grant is not evidence that development really stopped: the freeze first binds
the final config, patient split, code/statistics inventories, validation/OOF
receipts, selected checkpoints, and environment receipt.  Only then may a
director/Codex grant authorize the one test evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


FREEZE_SCHEMA = "PETCT-FINAL-DEVELOPMENT-FREEZE-v2.0"
INPUT_SCHEMA = "PETCT-FINAL-DEVELOPMENT-FREEZE-INPUT-v1.0"
FREEZE_STATUS = "ALL_DEVELOPMENT_FROZEN"
FROZEN_EXPERIMENT_CONFIG_STATUS = (
    "SIX_CLASS_CANONICAL_FROZEN_LOCAL_IMPLEMENTATION_PRESENT_"
    "CONFIRMATORY_BLOCKED_NOT_EXECUTED"
)
CONFIRMATORY_ACTIVE_CONFIG_STATUS = (
    "SIX_CLASS_CANONICAL_FROZEN_CONFIRMATORY_ACTIVE_NOT_EXECUTED"
)
SPLIT_STATUS = "FROZEN_BEFORE_MODEL_SELECTION"
CHECKPOINT_BINDINGS_SCHEMA = "PETCT-FROZEN-CHECKPOINT-BINDINGS-v1.0"
EXTERNAL_ADMISSION_SCHEMA = "PETCT-EXTERNAL-METHOD-ADMISSION-FREEZE-v1.1"
EXTERNAL_ADMISSION_STATUS = "ADMITTED_FOR_OPTIONAL_FORMAL_TEST"
EXTERNAL_COMPLETE_SCHEMA = "PETCT-EXTERNAL-COMPARATORS-COMPLETE-v1.2"
EXTERNAL_METHOD_ROLE_PREFIX = "selected_external_method:"
NNINTERACTIVE_METHOD_ID = "nninteractive"
NNINTERACTIVE_ROLE = f"{EXTERNAL_METHOD_ROLE_PREFIX}{NNINTERACTIVE_METHOD_ID}"
P2T_CHECKPOINT_SCHEMA = "PETCT-P2T-CHECKPOINT-v2.0"
EDITOR_CHECKPOINT_SCHEMA = "PETCT-EDITOR-CHECKPOINT-v1.3"
TRAINED_CHECKPOINT_STATUS = "TRAINED_WHEN_THIS_SCRIPT_IS_EXECUTED"

# Each role is a scientifically distinct decision surface.  Combining these
# into one vague "misc" file would make a later audit unable to determine what
# was actually frozen.
REQUIRED_SINGLETON_ROLES = frozenset(
    {
        "code_inventory",
        "statistics_plan",
        "m0_oof_receipt",
        "m0_validation_receipt",
        "p2t_validation_receipt",
        "editor_validation_receipt",
        "environment_receipt",
    }
)
SELECTED_CHECKPOINT_PREFIX = "selected_checkpoint:"
ROLE_CONTRACTS: dict[str, dict[str, str]] = {
    "code_inventory": {
        "schema_version": "PETCT-FROZEN-CODE-INVENTORY-v1.0",
        "status": "FROZEN",
    },
    "statistics_plan": {
        "schema_version": "PETCT-FROZEN-STATISTICS-PLAN-v1.0",
        "status": "FROZEN",
    },
    "m0_oof_receipt": {
        "schema_version": "PETCT-M0-OOF-READY-v1.0",
        "status": "COMMITTED",
    },
    "m0_validation_receipt": {
        "schema_version": "PETCT-M0-EVALUATION-READY-v1.0",
        "status": "PASS",
        "target": "m0_evaluation",
    },
    "p2t_validation_receipt": {
        "schema_version": "PETCT-ROUTE-A-PIPELINE-RECEIPT-v2.0",
        "status": "PASS",
        "target": "p2t_results",
    },
    "editor_validation_receipt": {
        "schema_version": "PETCT-ROUTE-A-PIPELINE-RECEIPT-v2.0",
        "status": "PASS",
        "target": "editor_results",
    },
    "environment_receipt": {
        "schema_version": "PETCT-NNUNET-ENV-MARKER-v1.0",
        "status": "ENVIRONMENT_EVIDENCE_COMPLETE",
    },
}

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class DevelopmentFreezeError(RuntimeError):
    """Raised when a final-development freeze is incomplete or stale."""


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


def _regular_file(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise DevelopmentFreezeError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise DevelopmentFreezeError(f"{label} is missing: {resolved}")
    return resolved


def _file_record(path: Path, *, label: str, role: str) -> dict[str, Any]:
    resolved = _regular_file(path, label=label)
    return {
        "role": role,
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _executable_record(path: Path, *, label: str, role: str) -> dict[str, Any]:
    resolved = _regular_file(path, label=label)
    if not os.access(resolved, os.X_OK):
        raise DevelopmentFreezeError(f"{label} is not executable: {resolved}")
    return _file_record(resolved, label=label, role=role)


def _load_object(path: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = _regular_file(path, label=label)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DevelopmentFreezeError(f"{label} must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise DevelopmentFreezeError(f"{label} must be a JSON object")
    return resolved, payload


def _derive_core_runtime(environment_marker: Path) -> dict[str, Any]:
    """Resolve core Python and official metrics through the frozen env marker."""

    marker_path, marker = _load_object(
        environment_marker, label="core environment marker"
    )
    if (
        marker.get("schema_version") != "PETCT-NNUNET-ENV-MARKER-v1.0"
        or marker.get("status") != "ENVIRONMENT_EVIDENCE_COMPLETE"
    ):
        raise DevelopmentFreezeError("core environment marker contract changed")
    receipt_path = _regular_file(
        Path(str(marker.get("receipt_path") or "")),
        label="core environment receipt",
    )
    if marker.get("receipt_sha256") != _sha256_file(receipt_path):
        raise DevelopmentFreezeError("core environment receipt hash changed")
    _, receipt = _load_object(receipt_path, label="core environment receipt")
    if (
        receipt.get("schema_version") != "PETCT-NNUNET-ENV-v1.1"
        or receipt.get("status")
        != "PREFLIGHT_PASS_PENDING_ATOMIC_EVIDENCE_PUBLICATION"
    ):
        raise DevelopmentFreezeError("core environment receipt contract changed")

    raw_prefix = Path(str(receipt.get("conda_prefix") or ""))
    if raw_prefix.is_symlink():
        raise DevelopmentFreezeError("core Conda prefix must not be a symlink")
    prefix = raw_prefix.resolve()
    if not prefix.is_dir():
        raise DevelopmentFreezeError("core Conda prefix is missing")
    python_record = _executable_record(
        prefix / "bin" / "python",
        label="core Python executable",
        role="core_python_executable",
    )

    preflight = receipt.get("official_autopetv_preflight")
    if not isinstance(preflight, Mapping):
        raise DevelopmentFreezeError("core receipt omits official AutoPET V preflight")
    metrics = preflight.get("metrics")
    if not isinstance(metrics, Mapping):
        raise DevelopmentFreezeError("core receipt omits official AutoPET V metrics")
    metrics_record = _file_record(
        Path(str(metrics.get("path") or "")),
        label="official AutoPET V metrics",
        role="official_autopetv_metrics",
    )
    if (
        metrics.get("import_status") != "PASS"
        or metrics.get("required_callable") != "MetricEvaluator"
        or metrics.get("sha256") != metrics_record["sha256"]
    ):
        raise DevelopmentFreezeError("official AutoPET V metrics preflight changed")
    return {
        "environment_marker": _file_record(
            marker_path,
            label="core environment marker",
            role="environment_receipt",
        ),
        "environment_receipt": _file_record(
            receipt_path,
            label="core environment receipt",
            role="core_environment_receipt",
        ),
        "conda_prefix": str(prefix),
        "python_executable": python_record,
        "official_metrics": metrics_record,
    }


def _external_template_path(value: Any, *, project_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DevelopmentFreezeError(f"{label} path is missing from comparator config")
    rendered = value.replace("{project_root}", str(project_root))
    if "{" in rendered or "}" in rendered:
        raise DevelopmentFreezeError(f"{label} contains an unresolved template variable")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return _regular_file(candidate, label=label)


def _source_bundle_records(root: Path) -> tuple[list[dict[str, Any]], str]:
    raw_root = Path(root)
    if raw_root.is_symlink():
        raise DevelopmentFreezeError("nnInteractive source root must not be a symlink")
    root = raw_root.resolve()
    if not root.is_dir():
        raise DevelopmentFreezeError("nnInteractive source root is missing")
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(
            part == "__pycache__" or part.endswith(".egg-info")
            for part in relative.parts
        ) or path.suffix in {".pyc", ".pyo"}:
            continue
        record = _file_record(
            path,
            label=f"nnInteractive source {relative.as_posix()}",
            role=f"source:{relative.as_posix()}",
        )
        record["relative_path"] = relative.as_posix()
        records.append(record)
    if not records:
        raise DevelopmentFreezeError("nnInteractive source bundle is empty")
    source_lines = "".join(
        f"{record['sha256']}  {record['relative_path']}\n" for record in records
    ).encode("utf-8")
    return records, hashlib.sha256(source_lines).hexdigest()


def _validate_external_complete_receipt(path: Path) -> dict[str, Any]:
    comparator_dir = Path(__file__).resolve().parents[1] / "comparators"
    import sys

    if str(comparator_dir) not in sys.path:
        sys.path.insert(0, str(comparator_dir))
    from finalize_petct_external_comparators import (  # noqa: PLC0415
        ExternalCompleteError,
        validate_external_complete,
    )

    try:
        return validate_external_complete(path)
    except ExternalCompleteError as exc:
        raise DevelopmentFreezeError(
            f"external validation completion receipt is invalid: {exc}"
        ) from exc


def _nninteractive_method(config: Mapping[str, Any]) -> Mapping[str, Any]:
    methods = config.get("methods")
    if not isinstance(methods, list):
        raise DevelopmentFreezeError("comparator config omits methods")
    matches = [
        method
        for method in methods
        if isinstance(method, Mapping) and method.get("id") == NNINTERACTIVE_METHOD_ID
    ]
    if len(matches) != 1:
        raise DevelopmentFreezeError(
            "comparator config must declare nninteractive exactly once"
        )
    method = matches[0]
    if (
        method.get("role") != "RUN"
        or method.get("selection") != "RUN_SECONDARY_EXPOSED_PRETRAINING"
        or method.get("spatial_dimensionality") != "3D"
        or method.get("headline", {}).get("eligible") is not False
        or method.get("pretraining", {}).get("current_psma_v3_exposure")
        != "KNOWN_PUBLIC_COHORT_EXPOSURE"
    ):
        raise DevelopmentFreezeError(
            "only the exposed-pretraining 3D nninteractive RUN role is admissible"
        )
    execution = method.get("execution")
    if (
        not isinstance(execution, Mapping)
        or execution.get("state") != "ARGV_WIRED"
        or execution.get("network_policy") != "NO_DOWNLOADS"
    ):
        raise DevelopmentFreezeError("nninteractive execution is not offline ARGV_WIRED")
    return method


def _derive_nninteractive_external_admission(
    *,
    comparator_config: Path,
    experiment_config: Path,
    learning_split: Path,
    validation_complete: Path,
) -> dict[str, Any]:
    comparator_path, comparator = _load_object(
        comparator_config, label="external comparator config"
    )
    if (
        comparator.get("schema_version")
        != "PETCT-EXTERNAL-COMPARATOR-CONTRACT-v1.0"
        or comparator.get("status") != "CONTRACT_ONLY_NOT_EXECUTED"
    ):
        raise DevelopmentFreezeError("external comparator config contract is invalid")
    _, _, experiment_record, split_record = _validate_final_config_and_split(
        experiment_config, learning_split
    )
    comparator_record = _file_record(
        comparator_path,
        label="external comparator config",
        role="comparator_config",
    )
    method = _nninteractive_method(comparator)
    project_root = comparator_path.parent.parent.resolve()
    execution = method["execution"]
    argv = execution.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(value, str) and value for value in argv)
    ):
        raise DevelopmentFreezeError("nninteractive execution argv is invalid")
    python_path = _external_template_path(
        argv[0],
        project_root=project_root,
        label="nninteractive Python executable",
    )
    python_record = _executable_record(
        python_path,
        label="nninteractive Python executable",
        role="nninteractive_python_executable",
    )
    admission_contract = execution.get("admission")
    if not isinstance(admission_contract, Mapping):
        raise DevelopmentFreezeError("nninteractive config omits runtime admission")
    if (
        admission_contract.get("schema_version") != "PETCT-NNINTERACTIVE-ENV-v1.1"
        or admission_contract.get("status") != "PASS"
    ):
        raise DevelopmentFreezeError("nninteractive runtime admission schema/status changed")
    runtime_path = _external_template_path(
        admission_contract.get("receipt"),
        project_root=project_root,
        label="nninteractive runtime receipt",
    )
    _, runtime = _load_object(runtime_path, label="nninteractive runtime receipt")
    if (
        runtime.get("schema_version") != admission_contract["schema_version"]
        or runtime.get("status") != admission_contract["status"]
        or runtime.get("config_sha256") != comparator_record["sha256"]
        or runtime.get("cuda_available") is not True
        or not str(runtime.get("smoke_device") or "").startswith("cuda")
    ):
        raise DevelopmentFreezeError(
            "nninteractive runtime receipt is not rebound to the current comparator config"
        )
    runtime_prefix = Path(str(runtime.get("conda_prefix") or "")).resolve()
    if runtime_prefix != python_path.parent.parent.resolve():
        raise DevelopmentFreezeError(
            "nninteractive Python executable differs from the admitted Conda prefix"
        )
    required_pass_fields = admission_contract.get("required_pass_fields")
    if not isinstance(required_pass_fields, list) or any(
        runtime.get(field) != "PASS" for field in required_pass_fields
    ):
        raise DevelopmentFreezeError("nninteractive runtime smoke gates are incomplete")
    exact_fields = admission_contract.get("exact_fields")
    if not isinstance(exact_fields, Mapping) or any(
        runtime.get(field) != expected for field, expected in exact_fields.items()
    ):
        raise DevelopmentFreezeError("nninteractive runtime exact fields changed")
    file_fields = admission_contract.get("file_sha256_fields")
    expected_file_fields = {
        "adapter_sha256",
        "checkpoint_sha256",
        "license_sha256",
        "environment_freeze_sha256",
        "source_license_sha256",
    }
    if not isinstance(file_fields, Mapping) or set(file_fields) != expected_file_fields:
        raise DevelopmentFreezeError("nninteractive runtime file hash roles are not exact")
    bound_files = {
        field: _external_template_path(
            raw_path,
            project_root=project_root,
            label=f"nninteractive {field}",
        )
        for field, raw_path in file_fields.items()
    }
    if any(runtime.get(field) != _sha256_file(path) for field, path in bound_files.items()):
        raise DevelopmentFreezeError("nninteractive runtime file binding changed")

    source_root = bound_files["source_license_sha256"].parent
    source_records, source_bundle_sha = _source_bundle_records(source_root)
    source_contract = method.get("source")
    if not isinstance(source_contract, Mapping):
        raise DevelopmentFreezeError("nninteractive source contract is missing")
    if (
        runtime.get("source_commit") != source_contract.get("pinned_commit")
        or runtime.get("source_bundle_sha256") != source_bundle_sha
        or runtime.get("source_bundle_file_count") != len(source_records)
        or source_contract.get("license") != "Apache-2.0"
        or source_contract.get("license_sha256")
        != _sha256_file(bound_files["source_license_sha256"])
    ):
        raise DevelopmentFreezeError("nninteractive source revision/bundle/license changed")

    checkpoint = bound_files["checkpoint_sha256"]
    model_root = checkpoint.parent.parent
    expected_checkpoint = model_root / "fold_0" / "checkpoint_final.pth"
    if checkpoint != expected_checkpoint.resolve():
        raise DevelopmentFreezeError("nninteractive checkpoint layout changed")
    availability = method.get("pretraining", {}).get("local_checkpoint_availability")
    if not isinstance(availability, Mapping):
        raise DevelopmentFreezeError("nninteractive checkpoint contract is missing")
    configured_checkpoint = _external_template_path(
        availability.get("path"),
        project_root=project_root,
        label="nninteractive configured checkpoint",
    )
    configured_model_license = _external_template_path(
        availability.get("license_file"),
        project_root=project_root,
        label="nninteractive configured model license",
    )
    if (
        availability.get("status") != "PRESENT_HASHED"
        or configured_checkpoint != checkpoint
        or configured_model_license != bound_files["license_sha256"]
        or availability.get("sha256") != _sha256_file(checkpoint)
        or availability.get("license") != "CC BY-NC-SA 4.0"
        or availability.get("license_sha256")
        != _sha256_file(bound_files["license_sha256"])
        or runtime.get("license") != availability.get("license")
    ):
        raise DevelopmentFreezeError("nninteractive checkpoint/license contract changed")
    metadata_names = ("dataset.json", "plans.json", "inference_session_class.json")
    metadata_records = [
        _file_record(
            model_root / name,
            label=f"nninteractive model metadata {name}",
            role=f"model_metadata:{name}",
        )
        for name in metadata_names
    ]
    if runtime.get("model_metadata_sha256") != {
        Path(record["path"]).name: record["sha256"] for record in metadata_records
    }:
        raise DevelopmentFreezeError("nninteractive model metadata changed")

    complete_path = _regular_file(
        validation_complete, label="validation external completion receipt"
    )
    complete = _validate_external_complete_receipt(complete_path)
    if (
        complete.get("schema_version") != EXTERNAL_COMPLETE_SCHEMA
        or complete.get("partition") != "val"
        or NNINTERACTIVE_METHOD_ID not in complete.get("selected_methods", [])
        or complete.get("comparator_config") != comparator_record
        or complete.get("experiment_config") != experiment_record
        or complete.get("learning_split") != split_record
    ):
        raise DevelopmentFreezeError(
            "nninteractive admission requires the matching validation completion receipt"
        )
    runtime_records = [
        row
        for row in complete.get("runtime_receipts", [])
        if isinstance(row, Mapping) and row.get("method_id") == NNINTERACTIVE_METHOD_ID
    ]
    runtime_record = _file_record(
        runtime_path, label="nninteractive runtime receipt", role="runtime_receipt:nninteractive"
    )
    if len(runtime_records) != 1 or runtime_records[0].get("receipt") != runtime_record:
        raise DevelopmentFreezeError(
            "validation completion used a different nninteractive runtime receipt"
        )
    runtime_admissions = [
        row
        for row in complete.get("runtime_admissions", [])
        if isinstance(row, Mapping)
        and row.get("method_id") == NNINTERACTIVE_METHOD_ID
    ]
    if len(runtime_admissions) != 1:
        raise DevelopmentFreezeError(
            "validation completion omits current nninteractive runtime admission"
        )
    validation_runtime = runtime_admissions[0]
    validation_checkpoint = validation_runtime.get("bound_files", {}).get(
        "checkpoint_sha256"
    )
    if (
        not isinstance(validation_checkpoint, Mapping)
        or any(
            validation_runtime.get("receipt", {}).get(key)
            != runtime_record.get(key)
            for key in ("path", "sha256")
        )
        or any(
            validation_runtime.get("config", {}).get(key)
            != comparator_record.get(key)
            for key in ("path", "sha256")
        )
        or validation_checkpoint.get("path") != str(checkpoint.resolve())
        or validation_checkpoint.get("sha256") != _sha256_file(checkpoint)
    ):
        raise DevelopmentFreezeError(
            "validation completion runtime/config/checkpoint binding changed"
        )
    validation_execution = complete.get("execution_artifacts")
    if not isinstance(validation_execution, Mapping):
        raise DevelopmentFreezeError(
            "validation completion omits execution-artifact bindings"
        )
    if validation_execution.get("nninteractive_python") != python_record:
        raise DevelopmentFreezeError(
            "validation completion used a different nninteractive Python executable"
        )
    for role in ("core_python", "official_metrics"):
        if not isinstance(validation_execution.get(role), Mapping):
            raise DevelopmentFreezeError(
                f"validation completion omits {role} execution binding"
            )
    expected_tables = {
        ("union_with_m0", "3D", "POSITIVE_ONLY_DIAGNOSTIC"),
        ("native_full_mask", "3D", "NATIVE_DIAGNOSTIC"),
    }
    observed_tables = {
        (
            row.get("output_policy"),
            row.get("spatial_dimensionality"),
            row.get("comparison_role"),
        )
        for row in complete.get("tables", [])
        if isinstance(row, Mapping) and row.get("method_id") == NNINTERACTIVE_METHOD_ID
    }
    if observed_tables != expected_tables:
        raise DevelopmentFreezeError(
            "validation completion omits an nninteractive fairness table"
        )
    nninteractive_tables = [
        row
        for row in complete.get("tables", [])
        if isinstance(row, Mapping)
        and row.get("method_id") == NNINTERACTIVE_METHOD_ID
    ]
    if any(
        row.get("runtime_checkpoint") != validation_checkpoint
        for row in nninteractive_tables
    ):
        raise DevelopmentFreezeError(
            "validation completion table checkpoint differs from runtime admission"
        )

    unsigned = {
        "schema_version": EXTERNAL_ADMISSION_SCHEMA,
        "status": EXTERNAL_ADMISSION_STATUS,
        "role": NNINTERACTIVE_ROLE,
        "method_id": NNINTERACTIVE_METHOD_ID,
        "selection": "RUN_SECONDARY_EXPOSED_PRETRAINING",
        "spatial_dimensionality": "3D",
        "headline_eligible": False,
        "pretraining_exposure": "KNOWN_PUBLIC_COHORT_EXPOSURE",
        "comparator_config": comparator_record,
        "experiment_config": experiment_record,
        "learning_split": split_record,
        "validation_complete": _file_record(
            complete_path,
            label="validation external completion receipt",
            role="validation_external_complete",
        ),
        "validation_complete_receipt_sha256": complete["receipt_sha256"],
        "runtime_receipt": runtime_record,
        "validation_execution_artifacts": dict(validation_execution),
        "execution": {
            "conda_prefix": str(runtime_prefix),
            "python_executable": python_record,
            "argv0": str(python_path),
        },
        "source": {
            "root": str(source_root.resolve()),
            "repository": source_contract.get("repository"),
            "pinned_commit": source_contract.get("pinned_commit"),
            "bundle_sha256": source_bundle_sha,
            "bundle_file_count": len(source_records),
            "files": source_records,
            "license": "Apache-2.0",
            "license_file": _file_record(
                bound_files["source_license_sha256"],
                label="nninteractive source license",
                role="source_license",
            ),
        },
        "adapter": _file_record(
            bound_files["adapter_sha256"],
            label="nninteractive adapter",
            role="adapter",
        ),
        "environment_freeze": _file_record(
            bound_files["environment_freeze_sha256"],
            label="nninteractive environment freeze",
            role="environment_freeze",
        ),
        "model": {
            "folder": str(model_root.resolve()),
            "checkpoint": _file_record(
                checkpoint,
                label="nninteractive checkpoint",
                role="model_checkpoint",
            ),
            "metadata": metadata_records,
            "metadata_inventory_sha256": _canonical_sha256(metadata_records),
            "license": "CC BY-NC-SA 4.0",
            "license_file": _file_record(
                bound_files["license_sha256"],
                label="nninteractive model license",
                role="model_license",
            ),
        },
        "execution_policy": {
            "network": "NO_DOWNLOADS",
            "evaluation_only": True,
            "primary_route_a_gate": False,
            "failure_revokes_primary": False,
        },
    }
    return {**unsigned, "admission_sha256": _canonical_sha256(unsigned)}


def build_nninteractive_external_admission(
    *,
    comparator_config: Path,
    experiment_config: Path,
    learning_split: Path,
    validation_complete: Path,
    output: Path,
) -> dict[str, Any]:
    payload = _derive_nninteractive_external_admission(
        comparator_config=comparator_config,
        experiment_config=experiment_config,
        learning_split=learning_split,
        validation_complete=validation_complete,
    )
    try:
        _write_exclusive(output, payload)
    except FileExistsError as exc:
        raise DevelopmentFreezeError(f"external admission exists: {output}") from exc
    return payload


def validate_nninteractive_external_admission(path: Path) -> dict[str, Any]:
    _, payload = _load_object(path, label="nninteractive external admission")
    seal = payload.get("admission_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "admission_sha256"}
    if seal != _canonical_sha256(unsigned):
        raise DevelopmentFreezeError("nninteractive external admission self-hash mismatch")
    if (
        payload.get("schema_version") != EXTERNAL_ADMISSION_SCHEMA
        or payload.get("status") != EXTERNAL_ADMISSION_STATUS
        or payload.get("role") != NNINTERACTIVE_ROLE
        or payload.get("method_id") != NNINTERACTIVE_METHOD_ID
    ):
        raise DevelopmentFreezeError("nninteractive external admission contract mismatch")
    try:
        expected = _derive_nninteractive_external_admission(
            comparator_config=Path(payload["comparator_config"]["path"]),
            experiment_config=Path(payload["experiment_config"]["path"]),
            learning_split=Path(payload["learning_split"]["path"]),
            validation_complete=Path(payload["validation_complete"]["path"]),
        )
    except (KeyError, TypeError) as exc:
        raise DevelopmentFreezeError(
            "nninteractive external admission structure is invalid"
        ) from exc
    if payload != expected:
        raise DevelopmentFreezeError(
            "nninteractive external admission differs from recursively rehashed evidence"
        )
    return payload


def _derive_external_method_binding(admission_path: Path) -> dict[str, Any]:
    admission = validate_nninteractive_external_admission(admission_path)
    return {
        "role": NNINTERACTIVE_ROLE,
        "method_id": NNINTERACTIVE_METHOD_ID,
        "admission": _file_record(
            admission_path,
            label="nninteractive external admission",
            role="external_method_admission:nninteractive",
        ),
        "admission_sha256": admission["admission_sha256"],
        "comparator_config": admission["comparator_config"],
        "experiment_config": admission["experiment_config"],
        "runtime_receipt": admission["runtime_receipt"],
        "python_executable": admission["execution"]["python_executable"],
        "validation_execution_artifacts": admission[
            "validation_execution_artifacts"
        ],
        "source_bundle_sha256": admission["source"]["bundle_sha256"],
        "adapter": admission["adapter"],
        "environment_freeze": admission["environment_freeze"],
        "model_checkpoint": admission["model"]["checkpoint"],
        "model_license": admission["model"]["license_file"],
        "validation_complete": admission["validation_complete"],
        "pretraining_exposure": admission["pretraining_exposure"],
        "headline_eligible": admission["headline_eligible"],
    }


def _validate_final_config_and_split(
    experiment_config: Path, learning_split: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path, config = _load_object(experiment_config, label="experiment config")
    split_path, split = _load_object(learning_split, label="learning split")
    if config.get("status") not in {
        FROZEN_EXPERIMENT_CONFIG_STATUS,
        CONFIRMATORY_ACTIVE_CONFIG_STATUS,
    }:
        raise DevelopmentFreezeError(
            "experiment config must be the immutable six-class, "
            "confirmatory-active pre-execution contract; "
            f"observed {config.get('status')!r}"
        )
    if (
        config.get("p2t", {}).get("confirmatory_execution_gate")
        != "ACTIVE_AFTER_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"
        or config.get("statistics", {}).get("confirmatory_execution_gate")
        != "ACTIVE_AFTER_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"
    ):
        raise DevelopmentFreezeError(
            "final development freeze is blocked until the error-atlas feasibility "
            "and effect-threshold freeze activates both confirmatory gates"
        )
    statistics = config.get("statistics", {})
    thresholds = statistics.get("effect_thresholds")
    required_fields = {
        "family",
        "treatment",
        "comparator",
        "metric",
        "threshold_ref",
        "null_margin",
        "alternative",
    }
    contrasts = [config.get("p2t", {}).get("confirmatory_contrast")]
    editor_contrasts = statistics.get("confirmatory_contrasts")
    if (
        set(statistics.get("required_frozen_contrast_fields") or [])
        != required_fields
        or not isinstance(thresholds, Mapping)
        or not thresholds
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in thresholds.values()
        )
        or not isinstance(editor_contrasts, list)
        or not editor_contrasts
    ):
        raise DevelopmentFreezeError(
            "active confirmatory state lacks finite frozen thresholds/contrasts"
        )
    contrasts.extend(editor_contrasts)
    for contrast in contrasts:
        if (
            not isinstance(contrast, Mapping)
            or not required_fields.issubset(contrast)
            or contrast.get("threshold_ref") not in thresholds
            or isinstance(contrast.get("null_margin"), bool)
            or not isinstance(contrast.get("null_margin"), (int, float))
            or not math.isfinite(float(contrast["null_margin"]))
        ):
            raise DevelopmentFreezeError(
                "active confirmatory state contains an incomplete contrast"
            )
    if split.get("status") != SPLIT_STATUS:
        raise DevelopmentFreezeError("learning split is not frozen before model selection")
    try:
        expected_split_schema = config["dataset"]["learning_split"]["schema_version"]
    except (KeyError, TypeError) as exc:
        raise DevelopmentFreezeError("experiment config omits learning split schema") from exc
    if split.get("schema_version") != expected_split_schema:
        raise DevelopmentFreezeError("learning split schema differs from final config")
    return (
        config,
        split,
        _file_record(config_path, label="experiment config", role="experiment_config"),
        _file_record(split_path, label="learning split", role="learning_split"),
    )


def _validate_role_inventory(
    records: Sequence[Mapping[str, Any]],
) -> None:
    roles = [str(record.get("role") or "") for record in records]
    singleton_roles = {role for role in roles if role in REQUIRED_SINGLETON_ROLES}
    missing = sorted(REQUIRED_SINGLETON_ROLES - singleton_roles)
    if missing:
        raise DevelopmentFreezeError(
            "required artifact roles are missing: " + ", ".join(missing)
        )
    for role in REQUIRED_SINGLETON_ROLES:
        if roles.count(role) != 1:
            raise DevelopmentFreezeError(f"artifact role must occur exactly once: {role}")
    # Checkpoint roles and paths are deliberately NOT accepted from this input
    # manifest.  They are derived below from the already-validated result
    # receipts plus the deterministic config grid.  Allowing the same manifest
    # to declare both the expected roles and the satisfying files would make an
    # arbitrary non-empty file a circular proof of model selection.
    unknown = sorted(set(roles) - REQUIRED_SINGLETON_ROLES)
    if unknown:
        raise DevelopmentFreezeError("unknown required artifact roles: " + ", ".join(unknown))
    if len(roles) != len(set(roles)):
        raise DevelopmentFreezeError("required artifact roles must be unique")


def _materialize_contract_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    role = str(entry["role"])
    record = _file_record(Path(str(entry["path"])), label=role, role=role)
    expected_sha256 = entry.get("expected_sha256")
    if expected_sha256 != record["sha256"]:
        raise DevelopmentFreezeError(f"{role} differs from its predeclared SHA-256")
    record["expected_sha256"] = expected_sha256
    contract = ROLE_CONTRACTS[role]
    expected_schema = entry.get("expected_schema_version")
    expected_status = entry.get("expected_status")
    if (
        expected_schema != contract["schema_version"]
        or expected_status != contract["status"]
    ):
        raise DevelopmentFreezeError(f"{role} expected schema/status is not the frozen contract")
    _, document = _load_object(Path(str(entry["path"])), label=role)
    for key, expected in contract.items():
        if document.get(key) != expected:
            raise DevelopmentFreezeError(f"{role} has invalid {key}")
    record["expected_schema_version"] = expected_schema
    record["expected_status"] = expected_status
    if "target" in contract:
        if entry.get("expected_target") != contract["target"]:
            raise DevelopmentFreezeError(f"{role} expected_target is invalid")
        record["expected_target"] = contract["target"]
    return record


def _materialize_required_artifacts(manifest: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path, payload = _load_object(manifest, label="freeze input manifest")
    if payload.get("schema_version") != INPUT_SCHEMA:
        raise DevelopmentFreezeError("freeze input manifest schema mismatch")
    raw_records = payload.get("artifacts")
    if not isinstance(raw_records, list) or not raw_records:
        raise DevelopmentFreezeError("freeze input manifest artifacts must be non-empty")
    normalized_inputs: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_records, start=1):
        if not isinstance(raw, Mapping):
            raise DevelopmentFreezeError(f"artifact entry {index} must be an object")
        role = str(raw.get("role") or "")
        path = raw.get("path")
        if not role or not isinstance(path, str) or not path:
            raise DevelopmentFreezeError(f"artifact entry {index} requires role and path")
        normalized_inputs.append({"role": role, "path": path})
    if payload.get("selected_checkpoint_roles") not in (None, []):
        raise DevelopmentFreezeError(
            "selected checkpoint roles must be derived from validation receipts, not self-declared"
        )
    _validate_role_inventory(normalized_inputs)
    # Preserve all contract assertions rather than only role/path.
    by_role = {str(raw["role"]): raw for raw in raw_records}
    records = [_materialize_contract_entry(by_role[entry["role"]]) for entry in normalized_inputs]
    if len({record["path"] for record in records}) != len(records):
        raise DevelopmentFreezeError("one physical file cannot satisfy multiple freeze roles")
    records.sort(key=lambda record: record["role"])
    manifest_record = _file_record(
        manifest_path, label="freeze input manifest", role="freeze_input_manifest"
    )
    return manifest_record, records


def _identifier(value: Any, *, label: str) -> str:
    result = str(value or "")
    if not result or _SAFE_ID.fullmatch(result) is None:
        raise DevelopmentFreezeError(f"{label} must be a stable identifier")
    return result


def _integer_list(value: Any, *, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise DevelopmentFreezeError(f"{label} must be a non-empty integer list")
    result = [int(item) for item in value]
    if len(result) != len(set(result)):
        raise DevelopmentFreezeError(f"{label} must not contain duplicates")
    return result


def _identifier_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise DevelopmentFreezeError(f"{label} must be a non-empty list")
    result = [_identifier(item, label=label) for item in value]
    if len(result) != len(set(result)):
        raise DevelopmentFreezeError(f"{label} must not contain duplicates")
    return result


def _expected_checkpoint_contracts(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Derive the exact model grid from the frozen config and code registry."""

    try:
        p2t = config["p2t"]
        editor = config["editor"]
        p2t_seeds = _integer_list(p2t["training"]["seeds"], label="p2t seeds")
        p2t_criterion = str(p2t["training"]["checkpoint_criterion"])
        p2t_arms = _identifier_list(
            p2t["simple_first_input_arms"], label="p2t simple-first arms"
        )
        primary_architecture = _identifier(
            p2t["primary_architecture_id"], label="p2t primary architecture"
        )
        editor_seeds = _integer_list(
            editor["training"]["seeds"], label="editor seeds"
        )
        editor_criterion = str(editor["training"]["checkpoint_criterion"])
        trained_conditions = _identifier_list(
            editor["training_conditions"], label="editor training conditions"
        )
        editor_architecture = _identifier(
            editor["primary_architecture_id"], label="editor primary architecture"
        )
    except (KeyError, TypeError) as exc:
        raise DevelopmentFreezeError(
            "final experiment config omits the deterministic checkpoint grid"
        ) from exc

    expected: dict[str, dict[str, Any]] = {}
    for seed in p2t_seeds:
        for arm in p2t_arms:
            role = (
                f"{SELECTED_CHECKPOINT_PREFIX}p2t:{primary_architecture}:"
                f"{arm}:seed{seed}"
            )
            expected[role] = {
                "kind": "p2t_primary",
                "schema_version": P2T_CHECKPOINT_SCHEMA,
                "status": TRAINED_CHECKPOINT_STATUS,
                "seed": seed,
                "seed_registry": p2t_seeds,
                "architecture_id": primary_architecture,
                "input_ablation": arm,
                "arm_role": "primary" if arm == "full" else "ablation",
                "checkpoint_criterion": p2t_criterion,
                "receipt_role": "p2t_validation_receipt",
                "training_manifest_binding": "controlled_tensor_manifest",
            }

    for seed in editor_seeds:
        for condition in trained_conditions:
            role = (
                f"{SELECTED_CHECKPOINT_PREFIX}editor:{condition}:"
                f"{editor_architecture}:seed{seed}"
            )
            expected[role] = {
                "kind": "editor",
                "schema_version": EDITOR_CHECKPOINT_SCHEMA,
                "status": TRAINED_CHECKPOINT_STATUS,
                "seed": seed,
                "seed_registry": editor_seeds,
                "condition": condition,
                "architecture_id": editor_architecture,
                "input_ablation": "full",
                "checkpoint_criterion": editor_criterion,
                "receipt_role": "editor_validation_receipt",
                "training_manifest_binding": "natural_tensor_manifest",
            }
    if not expected:
        raise DevelopmentFreezeError("deterministic checkpoint grid is empty")
    return expected


def _embedded_record(value: Any, *, label: str, role: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DevelopmentFreezeError(f"{label} must be a file record")
    current = _file_record(Path(str(value.get("path") or "")), label=label, role=role)
    if value.get("sha256") != current["sha256"]:
        raise DevelopmentFreezeError(f"{label} embedded SHA-256 mismatch")
    return current


def _load_checkpoint(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise DevelopmentFreezeError(
            f"{label} failed safe weights-only checkpoint load"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise DevelopmentFreezeError(f"{label} must contain a checkpoint mapping")
    return checkpoint


def _validation_receipt_context(
    receipt_path: Path,
    *,
    role: str,
    config_record: Mapping[str, Any],
    split_record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Mapping[str, Any]], dict[str, Any]]:
    _, receipt = _load_object(receipt_path, label=role)
    common = receipt.get("common")
    bindings = receipt.get("artifact_bindings")
    expected_target = (
        "p2t_results" if role == "p2t_validation_receipt" else "editor_results"
    )
    if (
        receipt.get("target") != expected_target
        or not isinstance(common, Mapping)
        or not isinstance(bindings, Mapping)
        or common.get("evaluation_partition") != "val"
        or common.get("experiment_config_sha256") != config_record["sha256"]
        or common.get("learning_split_sha256") != split_record["sha256"]
    ):
        raise DevelopmentFreezeError(f"{role} is not final val evidence for config/split")
    checkpoint_key = (
        "p2t_checkpoints" if role == "p2t_validation_receipt" else "editor_checkpoints"
    )
    manifest_key = (
        "controlled_tensor_manifest"
        if role == "p2t_validation_receipt"
        else "natural_tensor_manifest"
    )
    checkpoints = bindings.get(checkpoint_key)
    if not isinstance(checkpoints, list):
        raise DevelopmentFreezeError(f"{role} omits {checkpoint_key}")
    training = _embedded_record(
        bindings.get(manifest_key), label=f"{role} {manifest_key}", role="training_manifest"
    )
    return receipt, checkpoints, training


def _checkpoint_descriptor(checkpoint: Mapping[str, Any]) -> tuple[str, int, str, str]:
    schema = checkpoint.get("schema_version")
    seed = checkpoint.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DevelopmentFreezeError("checkpoint seed must be an integer")
    if schema == P2T_CHECKPOINT_SCHEMA:
        return (
            "p2t",
            int(seed),
            str(checkpoint.get("architecture_id") or ""),
            str(checkpoint.get("input_ablation") or ""),
        )
    if schema == EDITOR_CHECKPOINT_SCHEMA:
        return (
            "editor",
            int(seed),
            str(checkpoint.get("condition") or ""),
            str(checkpoint.get("architecture_id") or ""),
        )
    raise DevelopmentFreezeError("checkpoint schema is unsupported")


def _expected_descriptor(contract: Mapping[str, Any]) -> tuple[str, int, str, str]:
    if str(contract["kind"]).startswith("p2t"):
        return (
            "p2t",
            int(contract["seed"]),
            str(contract["architecture_id"]),
            str(contract["input_ablation"]),
        )
    return (
        "editor",
        int(contract["seed"]),
        str(contract["condition"]),
        str(contract["architecture_id"]),
    )


def _derive_checkpoint_records(
    *,
    config: Mapping[str, Any],
    config_record: Mapping[str, Any],
    split_record: Mapping[str, Any],
    singleton_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = _expected_checkpoint_contracts(config)
    expected_by_descriptor = {
        _expected_descriptor(contract): (role, contract)
        for role, contract in expected.items()
    }
    if len(expected_by_descriptor) != len(expected):
        raise DevelopmentFreezeError("deterministic checkpoint descriptors are not unique")
    singleton_by_role = {str(record["role"]): record for record in singleton_records}
    observed: dict[str, dict[str, Any]] = {}
    training_by_receipt: dict[str, dict[str, Any]] = {}
    for receipt_role in (
        "p2t_validation_receipt",
        "editor_validation_receipt",
    ):
        receipt_record = singleton_by_role[receipt_role]
        _, embedded, training_manifest = _validation_receipt_context(
            Path(str(receipt_record["path"])),
            role=receipt_role,
            config_record=config_record,
            split_record=split_record,
        )
        training_by_receipt[receipt_role] = training_manifest
        for index, raw_checkpoint_record in enumerate(embedded, start=1):
            file_record = _embedded_record(
                raw_checkpoint_record,
                label=f"{receipt_role} checkpoint {index}",
                role="candidate_checkpoint",
            )
            checkpoint_path = Path(file_record["path"])
            checkpoint = _load_checkpoint(
                checkpoint_path, label=f"{receipt_role} checkpoint {index}"
            )
            descriptor = _checkpoint_descriptor(checkpoint)
            selected = expected_by_descriptor.get(descriptor)
            if selected is None:
                raise DevelopmentFreezeError(
                    f"{receipt_role} contains an extra or swapped checkpoint descriptor: {descriptor}"
                )
            role, contract = selected
            if contract["receipt_role"] != receipt_role:
                raise DevelopmentFreezeError(
                    f"checkpoint descriptor is bound by the wrong validation receipt: {role}"
                )
            if role in observed:
                raise DevelopmentFreezeError(f"duplicate selected checkpoint role: {role}")
            required_metadata = {
                key: contract[key]
                for key in (
                    "schema_version",
                    "status",
                    "seed",
                    "seed_registry",
                    "input_ablation",
                    "checkpoint_criterion",
                )
            }
            if str(contract["kind"]).startswith("p2t"):
                required_metadata.update(
                    architecture_id=contract["architecture_id"],
                    arm_role=contract["arm_role"],
                )
            else:
                required_metadata.update(
                    condition=contract["condition"],
                    architecture_id=contract["architecture_id"],
                )
            if any(checkpoint.get(key) != value for key, value in required_metadata.items()):
                raise DevelopmentFreezeError(
                    f"selected checkpoint metadata differs from deterministic role: {role}"
                )
            state_dict = checkpoint.get("state_dict")
            if not isinstance(state_dict, Mapping) or not state_dict:
                raise DevelopmentFreezeError(
                    f"selected checkpoint omits a non-empty state_dict: {role}"
                )
            if (
                checkpoint.get("experiment_config_sha256") != config_record["sha256"]
                or checkpoint.get("learning_split_sha256") != split_record["sha256"]
                or checkpoint.get("manifest_sha256") != training_manifest["sha256"]
            ):
                raise DevelopmentFreezeError(
                    f"selected checkpoint config/split/training manifest mismatch: {role}"
                )
            try:
                raw_checkpoint_config = Path(str(checkpoint["experiment_config"]))
                raw_checkpoint_split = Path(str(checkpoint["learning_split"]))
                raw_checkpoint_manifest = Path(str(checkpoint["manifest"]))
                if any(
                    path.is_symlink()
                    for path in (
                        raw_checkpoint_config,
                        raw_checkpoint_split,
                        raw_checkpoint_manifest,
                    )
                ):
                    raise DevelopmentFreezeError(
                        f"selected checkpoint source path is a symlink: {role}"
                    )
                checkpoint_config = raw_checkpoint_config.resolve()
                checkpoint_split = raw_checkpoint_split.resolve()
                checkpoint_manifest = raw_checkpoint_manifest.resolve()
            except (KeyError, OSError) as exc:
                raise DevelopmentFreezeError(
                    f"selected checkpoint omits bound source paths: {role}"
                ) from exc
            if (
                checkpoint_config != Path(str(config_record["path"]))
                or checkpoint_split != Path(str(split_record["path"]))
                or checkpoint_manifest != Path(str(training_manifest["path"]))
            ):
                raise DevelopmentFreezeError(
                    f"selected checkpoint source paths differ from frozen sources: {role}"
                )
            if contract["kind"] == "editor":
                raw_training_manifest = Path(
                    str(checkpoint.get("training_manifest") or "")
                )
                if (
                    raw_training_manifest.is_symlink()
                    or raw_training_manifest.resolve() != checkpoint_manifest
                    or checkpoint.get("training_manifest_sha256")
                    != training_manifest["sha256"]
                ):
                    raise DevelopmentFreezeError(
                        f"editor checkpoint training manifest alias mismatch: {role}"
                    )
            observed[role] = {
                **_file_record(checkpoint_path, label=role, role=role),
                "checkpoint_metadata": required_metadata,
                "training_manifest": training_manifest,
                "validation_receipt": {
                    "role": receipt_role,
                    "path": receipt_record["path"],
                    "sha256": receipt_record["sha256"],
                },
            }
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise DevelopmentFreezeError(
            "selected checkpoint grid is not exact; missing="
            + repr(missing)
            + ", extra="
            + repr(extra)
        )
    paths = [record["path"] for record in observed.values()]
    if len(paths) != len(set(paths)):
        raise DevelopmentFreezeError("one checkpoint file cannot satisfy multiple roles")
    return [observed[role] for role in sorted(observed)]


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
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


def build_final_development_freeze(
    *,
    experiment_config: Path,
    learning_split: Path,
    required_artifacts_manifest: Path,
    output: Path,
    external_method_admission: Path | None = None,
) -> dict[str, Any]:
    """Publish the immutable freeze; refuse design-only or incomplete state."""

    config, _, config_record, split_record = _validate_final_config_and_split(
        experiment_config, learning_split
    )
    input_manifest_record, singleton_records = _materialize_required_artifacts(
        required_artifacts_manifest
    )
    checkpoint_records = _derive_checkpoint_records(
        config=config,
        config_record=config_record,
        split_record=split_record,
        singleton_records=singleton_records,
    )
    environment_record = next(
        record for record in singleton_records if record["role"] == "environment_receipt"
    )
    core_runtime = _derive_core_runtime(Path(environment_record["path"]))
    if any(
        core_runtime["environment_marker"].get(key) != environment_record.get(key)
        for key in ("path", "sha256", "bytes")
    ):
        raise DevelopmentFreezeError(
            "core runtime marker differs from the required environment receipt"
        )
    required_artifacts = [
        config_record,
        split_record,
        *singleton_records,
        *checkpoint_records,
    ]
    artifact_set_sha256 = _canonical_sha256(required_artifacts)
    checkpoint_inventory_sha256 = _canonical_sha256(checkpoint_records)
    external_method_bindings = (
        [_derive_external_method_binding(external_method_admission)]
        if external_method_admission is not None
        else []
    )
    if external_method_bindings:
        validation_execution = external_method_bindings[0][
            "validation_execution_artifacts"
        ]
        for validation_key, core_key in (
            ("core_python", "python_executable"),
            ("official_metrics", "official_metrics"),
        ):
            validation_record = validation_execution[validation_key]
            core_record = core_runtime[core_key]
            if any(
                validation_record.get(key) != core_record.get(key)
                for key in ("path", "sha256", "bytes")
            ):
                raise DevelopmentFreezeError(
                    f"external validation {validation_key} differs from final core runtime"
                )
    unsigned = {
        "schema_version": FREEZE_SCHEMA,
        "status": FREEZE_STATUS,
        "all_development_frozen": True,
        "required_roles": sorted(
            ["experiment_config", "learning_split", *REQUIRED_SINGLETON_ROLES]
        ),
        "selected_checkpoint_roles": [record["role"] for record in checkpoint_records],
        "selected_checkpoint_count": len(checkpoint_records),
        "checkpoint_inventory_sha256": checkpoint_inventory_sha256,
        "core_runtime": core_runtime,
        "external_method_bindings": external_method_bindings,
        "external_method_inventory_sha256": _canonical_sha256(
            external_method_bindings
        ),
        "freeze_input_manifest": input_manifest_record,
        "required_artifacts": required_artifacts,
        "artifact_set_sha256": artifact_set_sha256,
    }
    payload = {**unsigned, "freeze_sha256": _canonical_sha256(unsigned)}
    try:
        _write_exclusive(output, payload)
    except FileExistsError as exc:
        raise DevelopmentFreezeError(f"final development freeze exists: {output}") from exc
    return payload


def validate_final_development_freeze(
    freeze_path: Path,
    *,
    experiment_config: Path,
    learning_split: Path,
) -> dict[str, Any]:
    """Recompute every bound artifact hash and reject stale/tampered freezes."""

    _, freeze = _load_object(freeze_path, label="final development freeze")
    observed_seal = freeze.get("freeze_sha256")
    unsigned = {key: value for key, value in freeze.items() if key != "freeze_sha256"}
    if observed_seal != _canonical_sha256(unsigned):
        raise DevelopmentFreezeError("final development freeze self-hash mismatch")
    if (
        freeze.get("schema_version") != FREEZE_SCHEMA
        or freeze.get("status") != FREEZE_STATUS
        or freeze.get("all_development_frozen") is not True
    ):
        raise DevelopmentFreezeError("final development freeze status contract is invalid")
    config, _, config_record, split_record = _validate_final_config_and_split(
        experiment_config, learning_split
    )
    raw_records = freeze.get("required_artifacts")
    if not isinstance(raw_records, list):
        raise DevelopmentFreezeError("final freeze omits required_artifacts")
    records = [dict(raw) for raw in raw_records if isinstance(raw, Mapping)]
    if len(records) != len(raw_records):
        raise DevelopmentFreezeError("final freeze contains a non-object artifact record")
    if records[:2] != [config_record, split_record]:
        raise DevelopmentFreezeError("final freeze does not bind the requested config/split")
    singleton_count = len(REQUIRED_SINGLETON_ROLES)
    raw_singletons = records[2 : 2 + singleton_count]
    _validate_role_inventory(raw_singletons)
    current_singletons: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_singletons, start=1):
        if not isinstance(raw, Mapping):
            raise DevelopmentFreezeError(f"frozen artifact {index} is not an object")
        role = str(raw.get("role") or "")
        try:
            current = _materialize_contract_entry(raw)
        except DevelopmentFreezeError as exc:
            raise DevelopmentFreezeError(
                f"frozen artifact changed after freeze: {role}: {exc}"
            ) from exc
        if current != dict(raw):
            raise DevelopmentFreezeError(f"frozen artifact changed after freeze: {role}")
        current_singletons.append(current)
    current_checkpoints = _derive_checkpoint_records(
        config=config,
        config_record=config_record,
        split_record=split_record,
        singleton_records=current_singletons,
    )
    expected_records = [
        config_record,
        split_record,
        *current_singletons,
        *current_checkpoints,
    ]
    if records != expected_records:
        raise DevelopmentFreezeError("frozen artifact inventory changed after freeze")
    if len({record["path"] for record in current_checkpoints}) != len(current_checkpoints):
        raise DevelopmentFreezeError("frozen artifact paths are not unique")
    if freeze.get("artifact_set_sha256") != _canonical_sha256(expected_records):
        raise DevelopmentFreezeError("final freeze artifact-set hash mismatch")
    selected_roles = [record["role"] for record in current_checkpoints]
    if freeze.get("selected_checkpoint_roles") != selected_roles:
        raise DevelopmentFreezeError("final freeze selected-checkpoint roles mismatch")
    if freeze.get("selected_checkpoint_count") != len(current_checkpoints):
        raise DevelopmentFreezeError("final freeze selected-checkpoint count mismatch")
    if freeze.get("checkpoint_inventory_sha256") != _canonical_sha256(
        current_checkpoints
    ):
        raise DevelopmentFreezeError("final freeze checkpoint-inventory hash mismatch")
    environment_record = next(
        record for record in current_singletons if record["role"] == "environment_receipt"
    )
    current_core_runtime = _derive_core_runtime(Path(environment_record["path"]))
    if freeze.get("core_runtime") != current_core_runtime:
        raise DevelopmentFreezeError("final freeze core-runtime binding changed")
    raw_external_bindings = freeze.get("external_method_bindings")
    if not isinstance(raw_external_bindings, list) or len(raw_external_bindings) > 1:
        raise DevelopmentFreezeError("final freeze external-method inventory is invalid")
    current_external_bindings: list[dict[str, Any]] = []
    if raw_external_bindings:
        raw_binding = raw_external_bindings[0]
        if not isinstance(raw_binding, Mapping):
            raise DevelopmentFreezeError("final freeze external-method binding is not an object")
        admission_record = raw_binding.get("admission")
        if not isinstance(admission_record, Mapping):
            raise DevelopmentFreezeError("external-method binding omits its admission record")
        current_binding = _derive_external_method_binding(
            Path(str(admission_record.get("path") or ""))
        )
        if current_binding != dict(raw_binding):
            raise DevelopmentFreezeError(
                "external-method binding changed after final freeze"
            )
        validation_execution = current_binding["validation_execution_artifacts"]
        for validation_key, core_key in (
            ("core_python", "python_executable"),
            ("official_metrics", "official_metrics"),
        ):
            if any(
                validation_execution[validation_key].get(key)
                != current_core_runtime[core_key].get(key)
                for key in ("path", "sha256", "bytes")
            ):
                raise DevelopmentFreezeError(
                    f"external validation {validation_key} differs from final core runtime"
                )
        current_external_bindings.append(current_binding)
    if freeze.get("external_method_inventory_sha256") != _canonical_sha256(
        current_external_bindings
    ):
        raise DevelopmentFreezeError("final freeze external-method inventory hash mismatch")
    input_record = freeze.get("freeze_input_manifest")
    if not isinstance(input_record, Mapping):
        raise DevelopmentFreezeError("final freeze omits freeze input manifest")
    current_input = _file_record(
        Path(str(input_record.get("path") or "")),
        label="freeze input manifest",
        role="freeze_input_manifest",
    )
    if current_input != dict(input_record):
        raise DevelopmentFreezeError("freeze input manifest changed after freeze")
    return freeze


def export_frozen_checkpoint_bindings(
    freeze_path: Path,
    *,
    experiment_config: Path,
    learning_split: Path,
    output: Path,
) -> dict[str, Any]:
    """Export the only checkpoint paths a formal test run may open."""

    freeze = validate_final_development_freeze(
        freeze_path,
        experiment_config=experiment_config,
        learning_split=learning_split,
    )
    checkpoints = [
        dict(record)
        for record in freeze["required_artifacts"]
        if str(record.get("role") or "").startswith(SELECTED_CHECKPOINT_PREFIX)
    ]
    unsigned = {
        "schema_version": CHECKPOINT_BINDINGS_SCHEMA,
        "status": "FROZEN_CHECKPOINTS_ONLY",
        "final_development_freeze": _file_record(
            freeze_path,
            label="final development freeze",
            role="final_development_freeze",
        ),
        "freeze_sha256": freeze["freeze_sha256"],
        "checkpoint_inventory_sha256": freeze["checkpoint_inventory_sha256"],
        "selected_checkpoint_roles": freeze["selected_checkpoint_roles"],
        "checkpoints": checkpoints,
    }
    payload = {**unsigned, "bindings_sha256": _canonical_sha256(unsigned)}
    try:
        _write_exclusive(output, payload)
    except FileExistsError as exc:
        raise DevelopmentFreezeError(f"checkpoint bindings output exists: {output}") from exc
    return payload


def validate_frozen_checkpoint_bindings(bindings_path: Path) -> dict[str, Any]:
    """Lightweight immutable-link check used before every launcher lookup."""

    _, bindings = _load_object(bindings_path, label="frozen checkpoint bindings")
    observed = bindings.get("bindings_sha256")
    unsigned = {
        key: value for key, value in bindings.items() if key != "bindings_sha256"
    }
    if observed != _canonical_sha256(unsigned):
        raise DevelopmentFreezeError("frozen checkpoint bindings self-hash mismatch")
    if (
        bindings.get("schema_version") != CHECKPOINT_BINDINGS_SCHEMA
        or bindings.get("status") != "FROZEN_CHECKPOINTS_ONLY"
    ):
        raise DevelopmentFreezeError("frozen checkpoint bindings contract mismatch")
    freeze_record = bindings.get("final_development_freeze")
    if not isinstance(freeze_record, Mapping):
        raise DevelopmentFreezeError("frozen checkpoint bindings omit freeze record")
    current_freeze_record = _file_record(
        Path(str(freeze_record.get("path") or "")),
        label="final development freeze",
        role="final_development_freeze",
    )
    if current_freeze_record != dict(freeze_record):
        raise DevelopmentFreezeError("final development freeze changed after binding export")
    _, freeze = _load_object(
        Path(current_freeze_record["path"]), label="final development freeze"
    )
    freeze_unsigned = {
        key: value for key, value in freeze.items() if key != "freeze_sha256"
    }
    if freeze.get("freeze_sha256") != _canonical_sha256(freeze_unsigned):
        raise DevelopmentFreezeError("final development freeze self-hash mismatch")
    checkpoints = [
        dict(record)
        for record in freeze.get("required_artifacts", [])
        if isinstance(record, Mapping)
        and str(record.get("role") or "").startswith(SELECTED_CHECKPOINT_PREFIX)
    ]
    if (
        bindings.get("freeze_sha256") != freeze.get("freeze_sha256")
        or bindings.get("checkpoint_inventory_sha256")
        != _canonical_sha256(checkpoints)
        or bindings.get("checkpoint_inventory_sha256")
        != freeze.get("checkpoint_inventory_sha256")
        or bindings.get("selected_checkpoint_roles")
        != [record["role"] for record in checkpoints]
        or bindings.get("checkpoints") != checkpoints
    ):
        raise DevelopmentFreezeError("checkpoint bindings differ from final freeze")
    return bindings


def resolve_test_external_binding(
    *,
    test_access_receipt: Path,
    experiment_config: Path,
    run_root: Path,
    method_id: str,
    ledger_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve the optional external role from the receipt-bound final freeze.

    This is intentionally a single all-fields lookup.  A caller cannot resolve
    a checkpoint from one freeze and a runtime/config/adapter from another.
    """

    if method_id != NNINTERACTIVE_METHOD_ID:
        raise DevelopmentFreezeError(
            "formal test external admission exists only for nninteractive"
        )
    _, untrusted_receipt = _load_object(
        test_access_receipt, label="consumed test receipt"
    )
    untrusted_core = untrusted_receipt.get("consumption")
    if not isinstance(untrusted_core, Mapping):
        raise DevelopmentFreezeError("consumed test receipt omits its consumption core")
    split_record = untrusted_core.get("learning_split")
    if not isinstance(split_record, Mapping):
        raise DevelopmentFreezeError("consumed test receipt omits its learning split")
    learning_split = Path(str(split_record.get("path") or ""))
    import sys

    common_dir = Path(__file__).resolve().parent
    if str(common_dir) not in sys.path:
        sys.path.insert(0, str(common_dir))
    from petct_test_access import (  # noqa: PLC0415
        TestAccessError,
        validate_consumed_receipt,
    )

    try:
        validate_kwargs: dict[str, Any] = {
            "experiment_config": experiment_config,
            "learning_split": learning_split,
            "run_root": run_root,
        }
        if ledger_root is not None:
            validate_kwargs["ledger_root"] = ledger_root
        receipt = validate_consumed_receipt(test_access_receipt, **validate_kwargs)
    except TestAccessError as exc:
        raise DevelopmentFreezeError(f"consumed test receipt is invalid: {exc}") from exc
    core = receipt["consumption"]
    freeze_record = core.get("final_development_freeze")
    if not isinstance(freeze_record, Mapping):
        raise DevelopmentFreezeError("consumed test receipt omits final freeze")
    freeze = validate_final_development_freeze(
        Path(str(freeze_record.get("path") or "")),
        experiment_config=experiment_config,
        learning_split=learning_split,
    )
    bindings = [
        row
        for row in freeze.get("external_method_bindings", [])
        if isinstance(row, Mapping) and row.get("role") == NNINTERACTIVE_ROLE
    ]
    if len(bindings) != 1:
        raise DevelopmentFreezeError(
            "receipt-bound final freeze does not contain nninteractive external admission"
        )
    binding = dict(bindings[0])
    admission = validate_nninteractive_external_admission(
        Path(binding["admission"]["path"])
    )
    if _derive_external_method_binding(Path(binding["admission"]["path"])) != binding:
        raise DevelopmentFreezeError("receipt-bound external method binding changed")
    return {
        "schema_version": "PETCT-RESOLVED-TEST-EXTERNAL-BINDING-v1.1",
        "status": "FROZEN_EVALUATION_ONLY",
        "method_id": NNINTERACTIVE_METHOD_ID,
        "final_development_freeze": dict(freeze_record),
        "external_method_inventory_sha256": freeze[
            "external_method_inventory_sha256"
        ],
        "admission": binding["admission"],
        "admission_sha256": admission["admission_sha256"],
        "comparator_config": admission["comparator_config"],
        "experiment_config": admission["experiment_config"],
        "runtime_receipt": admission["runtime_receipt"],
        "nninteractive_python": admission["execution"]["python_executable"],
        "core_python": freeze["core_runtime"]["python_executable"],
        "official_metrics": freeze["core_runtime"]["official_metrics"],
        "source_root": admission["source"]["root"],
        "source_bundle_sha256": admission["source"]["bundle_sha256"],
        "adapter": admission["adapter"],
        "environment_freeze": admission["environment_freeze"],
        "model_folder": admission["model"]["folder"],
        "model_checkpoint": admission["model"]["checkpoint"],
        "model_license": admission["model"]["license_file"],
        "learning_split": admission["learning_split"],
        "validation_complete": admission["validation_complete"],
        "evaluation_only": True,
        "primary_route_a_gate": False,
    }


def resolve_frozen_checkpoint_field(
    bindings_path: Path, *, role: str, field: str
) -> str:
    """Resolve one allowed path while rehashing the selected physical file."""

    bindings = validate_frozen_checkpoint_bindings(bindings_path)
    rows = [row for row in bindings["checkpoints"] if row.get("role") == role]
    if len(rows) != 1:
        raise DevelopmentFreezeError(
            f"freeze binding does not contain exactly one role: {role}"
        )
    if field == "path":
        expected = rows[0]
        label = role
    elif field == "training_manifest.path":
        expected = rows[0].get("training_manifest")
        label = f"{role} training manifest"
    else:
        raise DevelopmentFreezeError("only checkpoint/training-manifest paths may be resolved")
    if not isinstance(expected, Mapping):
        raise DevelopmentFreezeError(f"{label} omits its file record")
    current = _file_record(
        Path(str(expected.get("path") or "")), label=label, role=str(expected.get("role") or "")
    )
    if any(current.get(key) != expected.get(key) for key in ("path", "sha256", "bytes")):
        raise DevelopmentFreezeError(f"{label} changed after final freeze")
    return str(current["path"])


def resolve_frozen_artifact_path(bindings_path: Path, *, role: str) -> str:
    """Resolve one freeze-bound non-checkpoint prerequisite by exact hash."""

    if role not in REQUIRED_SINGLETON_ROLES:
        raise DevelopmentFreezeError(f"role is not a frozen singleton artifact: {role}")
    bindings = validate_frozen_checkpoint_bindings(bindings_path)
    freeze_path = Path(bindings["final_development_freeze"]["path"])
    _, freeze = _load_object(freeze_path, label="final development freeze")
    rows = [
        row
        for row in freeze.get("required_artifacts", [])
        if isinstance(row, Mapping) and row.get("role") == role
    ]
    if len(rows) != 1:
        raise DevelopmentFreezeError(
            f"final freeze does not contain exactly one singleton role: {role}"
        )
    expected = rows[0]
    current = _file_record(
        Path(str(expected.get("path") or "")), label=role, role=role
    )
    if any(current.get(key) != expected.get(key) for key in ("path", "sha256", "bytes")):
        raise DevelopmentFreezeError(f"{role} changed after final freeze")
    return str(current["path"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--experiment-config", type=Path, required=True)
    build.add_argument("--learning-split", type=Path, required=True)
    build.add_argument("--required-artifacts-manifest", type=Path, required=True)
    build.add_argument(
        "--external-method-admission",
        type=Path,
        help="optional validated nnInteractive admission; never a primary completion gate",
    )
    build.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--freeze", type=Path, required=True)
    validate.add_argument("--experiment-config", type=Path, required=True)
    validate.add_argument("--learning-split", type=Path, required=True)
    export = subparsers.add_parser(
        "export-checkpoints",
        help="export the exact freeze-bound checkpoint inventory for formal test",
    )
    export.add_argument("--freeze", type=Path, required=True)
    export.add_argument("--experiment-config", type=Path, required=True)
    export.add_argument("--learning-split", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    resolve = subparsers.add_parser(
        "resolve-checkpoint",
        help="resolve one freeze-bound checkpoint or training-manifest path",
    )
    resolve.add_argument("--bindings", type=Path, required=True)
    resolve.add_argument("--role", required=True)
    resolve.add_argument(
        "--field", choices=("path", "training_manifest.path"), required=True
    )
    artifact = subparsers.add_parser(
        "resolve-artifact",
        help="resolve one freeze-bound non-checkpoint prerequisite",
    )
    artifact.add_argument("--bindings", type=Path, required=True)
    artifact.add_argument("--role", choices=sorted(REQUIRED_SINGLETON_ROLES), required=True)
    external_build = subparsers.add_parser(
        "build-external-admission",
        help="freeze the admitted nnInteractive validation/runtime/source/model closure",
    )
    external_build.add_argument("--comparator-config", type=Path, required=True)
    external_build.add_argument("--experiment-config", type=Path, required=True)
    external_build.add_argument("--learning-split", type=Path, required=True)
    external_build.add_argument("--validation-complete", type=Path, required=True)
    external_build.add_argument("--output", type=Path, required=True)
    external_validate = subparsers.add_parser("validate-external-admission")
    external_validate.add_argument("--admission", type=Path, required=True)
    external_resolve = subparsers.add_parser(
        "resolve-test-external",
        help="resolve all nnInteractive paths from one consumed receipt/final freeze",
    )
    external_resolve.add_argument("--test-access-receipt", type=Path, required=True)
    external_resolve.add_argument("--experiment-config", type=Path, required=True)
    external_resolve.add_argument("--run-root", type=Path, required=True)
    external_resolve.add_argument("--method", choices=(NNINTERACTIVE_METHOD_ID,), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            payload = build_final_development_freeze(
                experiment_config=args.experiment_config,
                learning_split=args.learning_split,
                required_artifacts_manifest=args.required_artifacts_manifest,
                output=args.output,
                external_method_admission=args.external_method_admission,
            )
        elif args.command == "validate":
            payload = validate_final_development_freeze(
                args.freeze,
                experiment_config=args.experiment_config,
                learning_split=args.learning_split,
            )
        elif args.command == "export-checkpoints":
            payload = export_frozen_checkpoint_bindings(
                args.freeze,
                experiment_config=args.experiment_config,
                learning_split=args.learning_split,
                output=args.output,
            )
        elif args.command == "resolve-checkpoint":
            print(
                resolve_frozen_checkpoint_field(
                    args.bindings, role=args.role, field=args.field
                )
            )
            return 0
        elif args.command == "resolve-artifact":
            print(resolve_frozen_artifact_path(args.bindings, role=args.role))
            return 0
        elif args.command == "build-external-admission":
            payload = build_nninteractive_external_admission(
                comparator_config=args.comparator_config,
                experiment_config=args.experiment_config,
                learning_split=args.learning_split,
                validation_complete=args.validation_complete,
                output=args.output,
            )
        elif args.command == "validate-external-admission":
            payload = validate_nninteractive_external_admission(args.admission)
        else:
            payload = resolve_test_external_binding(
                test_access_receipt=args.test_access_receipt,
                experiment_config=args.experiment_config,
                run_root=args.run_root,
                method_id=args.method,
            )
    except DevelopmentFreezeError as exc:
        parser.error(str(exc))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
