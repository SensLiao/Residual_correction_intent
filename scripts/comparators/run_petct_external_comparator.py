#!/usr/bin/env python3
"""Validate and plan external PET/CT comparators; execution is explicit and fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import string
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "scripts"
for directory in (SCRIPTS, SCRIPTS / "common"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common.petct_learning import (  # noqa: E402
    LearningContractError,
    validate_manifest_rows_against_frozen_learning_split,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    enforce_partition_access,
)

DEFAULT_CONFIG = PROJECT / "configs" / "petct_external_comparators.json"
DEFAULT_EXPERIMENT_CONFIG = PROJECT / "configs" / "petct_route_a_experiment.json"
PUBLIC_TO_INTERNAL_PARTITION = {"train": "train", "validation": "val", "test": "test"}
SAFE_EXECUTION_STATES = {"NOT_WIRED", "ARGV_WIRED"}
VALID_DIMENSIONS = {"2D", "2.5D", "3D"}
VALID_EXPOSURE_STATES = {
    "KNOWN_PUBLIC_COHORT_EXPOSURE",
    "UNVERIFIED",
    "NOT_APPLICABLE_FROM_SCRATCH",
}
VALID_ADMISSION_ROLES = {"RUN", "ADAPT", "REFERENCE_ONLY"}
FORBIDDEN_EXECUTABLES = {
    "bash",
    "cmd",
    "cmd.exe",
    "curl",
    "curl.exe",
    "fish",
    "git",
    "git.exe",
    "hf",
    "huggingface-cli",
    "pip",
    "pip.exe",
    "pip3",
    "pip3.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "sh",
    "wget",
    "wget.exe",
    "zsh",
}
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "WANDB_MODE": "offline",
    "PIP_NO_INDEX": "1",
}


class ContractError(ValueError):
    """Raised when a comparator contract or manifest is unsafe or inconsistent."""


def _expect(value: Any, expected: type, location: str) -> Any:
    if not isinstance(value, expected):
        raise ContractError(f"{location} must be {expected.__name__}")
    return value


def _required(mapping: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in mapping:
        raise ContractError(f"{location} is missing {key}")
    return mapping[key]


def _validate_argv(argv: Any, location: str, *, allow_empty: bool) -> list[str]:
    if not isinstance(argv, list) or not all(isinstance(item, str) and item for item in argv):
        raise ContractError(f"{location} must be a list of non-empty strings")
    if not argv:
        if allow_empty:
            return argv
        raise ContractError(
            f"{location} must contain a non-empty argv when state is ARGV_WIRED"
        )
    executable = Path(argv[0]).name.lower()
    if executable in FORBIDDEN_EXECUTABLES:
        raise ContractError(f"{location} uses a forbidden shell or downloader executable: {executable}")
    if executable.startswith("python") and "-c" in argv[1:]:
        raise ContractError(f"{location} may not execute inline Python with -c")
    return argv


def _validate_admission_spec(value: Any, location: str) -> dict[str, Any]:
    spec = _expect(value, dict, location)
    for key in ("receipt", "schema_version", "status", "config_sha256_field"):
        item = _expect(_required(spec, key, location), str, f"{location}.{key}")
        if not item:
            raise ContractError(f"{location}.{key} must not be empty")
    required_pass_fields = _expect(
        _required(spec, "required_pass_fields", location),
        list,
        f"{location}.required_pass_fields",
    )
    if not required_pass_fields or not all(
        isinstance(item, str) and item for item in required_pass_fields
    ):
        raise ContractError(f"{location}.required_pass_fields must be non-empty strings")
    if len(required_pass_fields) != len(set(required_pass_fields)):
        raise ContractError(f"{location}.required_pass_fields contains duplicates")
    exact_fields = _expect(
        _required(spec, "exact_fields", location), dict, f"{location}.exact_fields"
    )
    file_sha256_fields = _expect(
        _required(spec, "file_sha256_fields", location),
        dict,
        f"{location}.file_sha256_fields",
    )
    if not exact_fields:
        raise ContractError(f"{location}.exact_fields must not be empty")
    if not file_sha256_fields:
        raise ContractError(f"{location}.file_sha256_fields must not be empty")
    for field, path in file_sha256_fields.items():
        if not isinstance(field, str) or not field or not isinstance(path, str) or not path:
            raise ContractError(
                f"{location}.file_sha256_fields must map non-empty fields to paths"
            )
    return spec


def _validate_manifest_contract(spec: Any, location: str) -> dict[str, Any]:
    spec = _expect(spec, dict, location)
    _expect(_required(spec, "schema_version", location), str, f"{location}.schema_version")
    _expect(_required(spec, "record_key", location), str, f"{location}.record_key")
    min_records = _required(spec, "min_records", location)
    if not isinstance(min_records, int) or isinstance(min_records, bool) or min_records < 0:
        raise ContractError(f"{location}.min_records must be a non-negative integer")
    fields = _expect(_required(spec, "required_fields", location), list, f"{location}.required_fields")
    seen: set[str] = set()
    valid_types = {"string", "integer", "number", "boolean", "array", "object"}
    for index, raw_field in enumerate(fields):
        field = _expect(raw_field, dict, f"{location}.required_fields[{index}]")
        name = _expect(_required(field, "name", location), str, f"{location}.field.name")
        field_type = _expect(_required(field, "type", location), str, f"{location}.{name}.type")
        if name in seen:
            raise ContractError(f"{location} repeats field {name}")
        if field_type not in valid_types:
            raise ContractError(f"{location}.{name} has unsupported type {field_type}")
        if "nullable" in field and not isinstance(field["nullable"], bool):
            raise ContractError(f"{location}.{name}.nullable must be boolean")
        if "enum" in field:
            enum = _expect(field["enum"], list, f"{location}.{name}.enum")
            if not enum:
                raise ContractError(f"{location}.{name}.enum must not be empty")
        seen.add(name)
    if not seen:
        raise ContractError(f"{location}.required_fields must not be empty")
    return spec


def _validate_machine_admission_register(
    contract: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    review = _expect(
        _required(contract, "method_selection_review", "contract"),
        dict,
        "method_selection_review",
    )
    register = _expect(
        _required(
            review, "machine_readable_admission_register", "method_selection_review"
        ),
        dict,
        "method_selection_review.machine_readable_admission_register",
    )
    location = "method_selection_review.machine_readable_admission_register"
    if register.get("schema_version") != "PETCT-METHOD-ADMISSION-REGISTER-v1.0":
        raise ContractError(f"{location}.schema_version is unsupported")
    required_evidence = _expect(
        _required(register, "required_evidence", location),
        list,
        f"{location}.required_evidence",
    )
    expected_evidence = {
        "paper_identity",
        "source_revision",
        "source_license",
        "checkpoint_identity_or_explicit_none",
        "checkpoint_license_or_blocker",
        "training_exposure",
        "role",
        "local_contract_smoke",
        "minimal_runtime_package",
        "current_config_runtime_receipt",
        "admission_state",
    }
    if set(required_evidence) != expected_evidence or len(required_evidence) != len(
        expected_evidence
    ):
        raise ContractError(f"{location}.required_evidence is incomplete or duplicated")
    records = _expect(_required(register, "records", location), list, f"{location}.records")
    if not records:
        raise ContractError(f"{location}.records must not be empty")
    by_id: dict[str, dict[str, Any]] = {}
    required_text = (
        "id",
        "role",
        "selection",
        "local_evidence_status",
        "evidence_ref",
        "server_package_status",
        "runtime_receipt_status",
        "execution_state",
        "admission_state",
    )
    for index, raw in enumerate(records):
        record_location = f"{location}.records[{index}]"
        record = _expect(raw, dict, record_location)
        for key in required_text:
            value = _expect(
                _required(record, key, record_location),
                str,
                f"{record_location}.{key}",
            )
            if not value:
                raise ContractError(f"{record_location}.{key} must not be empty")
        method_id = record["id"]
        if method_id in by_id:
            raise ContractError(f"duplicate admission register id {method_id}")
        if record["role"] not in VALID_ADMISSION_ROLES:
            raise ContractError(f"{record_location}.role is invalid")
        blockers = _expect(
            _required(record, "blockers", record_location),
            list,
            f"{record_location}.blockers",
        )
        seen_blockers: set[str] = set()
        for blocker_index, raw_blocker in enumerate(blockers):
            blocker_location = f"{record_location}.blockers[{blocker_index}]"
            blocker = _expect(raw_blocker, dict, blocker_location)
            for key in ("id", "status", "evidence_ref"):
                value = _expect(
                    _required(blocker, key, blocker_location),
                    str,
                    f"{blocker_location}.{key}",
                )
                if not value:
                    raise ContractError(f"{blocker_location}.{key} must not be empty")
            if blocker["id"] in seen_blockers:
                raise ContractError(f"{record_location} repeats blocker {blocker['id']}")
            seen_blockers.add(blocker["id"])
        by_id[method_id] = record

    category_roles = {
        "adapt_protocol_only": "ADAPT",
        "adapt_clean_room_inspired_only": "ADAPT",
        "descriptive_secondary_not_admitted": "REFERENCE_ONLY",
        "run_candidates_not_admitted": "RUN",
        "license_blocked_candidates": "RUN",
        "run_secondary_exposed_candidate": "RUN",
        "gated_native_protocol_candidate": "RUN",
        "reference_only": "REFERENCE_ONLY",
    }
    for category, expected_role in category_roles.items():
        identifiers = _expect(
            _required(review, category, "method_selection_review"),
            list,
            f"method_selection_review.{category}",
        )
        if not all(isinstance(item, str) and item for item in identifiers):
            raise ContractError(f"method_selection_review.{category} has invalid ids")
        for method_id in identifiers:
            record = by_id.get(method_id)
            if record is None:
                raise ContractError(f"admission register omits classified id {method_id}")
            if record["role"] != expected_role:
                raise ContractError(
                    f"admission register role mismatch for {method_id}: {record['role']}"
                )
    return by_id


def validate_contract(contract: Any) -> dict[str, Any]:
    """Validate the static adapter contract and return it unchanged."""

    contract = _expect(contract, dict, "contract")
    if _required(contract, "schema_version", "contract") != "PETCT-EXTERNAL-COMPARATOR-CONTRACT-v2.0":
        raise ContractError("unsupported contract schema_version")
    policy = _expect(_required(contract, "execution_policy", "contract"), dict, "execution_policy")
    if policy.get("default_mode") != "DRY_RUN":
        raise ContractError("execution_policy.default_mode must be DRY_RUN")
    token = policy.get("confirmation_token")
    if not isinstance(token, str) or len(token) < 12:
        raise ContractError("execution_policy.confirmation_token must be an explicit non-secret phrase")
    if policy.get("network_policy") != "NO_DOWNLOADS":
        raise ContractError("execution_policy.network_policy must be NO_DOWNLOADS")

    admission_records = _validate_machine_admission_register(contract)
    input_contract = _validate_manifest_contract(
        _required(contract, "input_manifest_contract", "contract"), "input_manifest_contract"
    )
    output_contract = _validate_manifest_contract(
        _required(contract, "output_manifest_contract", "contract"), "output_manifest_contract"
    )
    metrics_contract = _expect(
        _required(contract, "metrics_contract", "contract"), dict, "metrics_contract"
    )
    metrics_version = _expect(
        _required(metrics_contract, "schema_version", "metrics_contract"),
        str,
        "metrics_contract.schema_version",
    )
    required_metrics = _expect(
        _required(metrics_contract, "required_metrics", "metrics_contract"),
        list,
        "metrics_contract.required_metrics",
    )
    if not required_metrics or not all(isinstance(item, str) and item for item in required_metrics):
        raise ContractError("metrics_contract.required_metrics must be non-empty strings")
    if len(required_metrics) != len(set(required_metrics)):
        raise ContractError("metrics_contract.required_metrics contains duplicates")

    methods = _expect(_required(contract, "methods", "contract"), list, "methods")
    if not methods:
        raise ContractError("methods must not be empty")
    seen_ids: set[str] = set()
    expected_refs = {
        "input_manifest": input_contract["schema_version"],
        "output_manifest": output_contract["schema_version"],
        "metrics": metrics_version,
    }
    for index, raw_method in enumerate(methods):
        location = f"methods[{index}]"
        method = _expect(raw_method, dict, location)
        method_id = _expect(_required(method, "id", location), str, f"{location}.id")
        if method_id in seen_ids:
            raise ContractError(f"duplicate method id {method_id}")
        seen_ids.add(method_id)
        _expect(_required(method, "display_name", location), str, f"{location}.display_name")
        modalities = _expect(
            _required(method, "prompt_modalities", location), list, f"{location}.prompt_modalities"
        )
        if not modalities or not all(isinstance(item, str) and item for item in modalities):
            raise ContractError(f"{location}.prompt_modalities must be non-empty strings")
        if method.get("spatial_dimensionality") not in VALID_DIMENSIONS:
            raise ContractError(f"{location}.spatial_dimensionality must be one of {sorted(VALID_DIMENSIONS)}")
        pretraining = _expect(_required(method, "pretraining", location), dict, f"{location}.pretraining")
        if pretraining.get("current_psma_v3_exposure") not in VALID_EXPOSURE_STATES:
            raise ContractError(f"{location}.pretraining.current_psma_v3_exposure is invalid")
        headline = _expect(_required(method, "headline", location), dict, f"{location}.headline")
        if not isinstance(headline.get("eligible"), bool) or not isinstance(headline.get("reason"), str):
            raise ContractError(f"{location}.headline requires boolean eligible and string reason")
        if not headline["reason"]:
            raise ContractError(f"{location}.headline.reason must not be empty")
        if method.get("contracts") != expected_refs:
            raise ContractError(f"{location}.contracts must reference the unified contracts")
        if method.get("metric_set") != required_metrics:
            raise ContractError(f"{location}.metric_set must equal metrics_contract.required_metrics")
        _expect(_required(method, "input_adapter", location), dict, f"{location}.input_adapter")
        _expect(_required(method, "output_adapter", location), dict, f"{location}.output_adapter")
        role = _expect(_required(method, "role", location), str, f"{location}.role")
        selection = _expect(
            _required(method, "selection", location), str, f"{location}.selection"
        )
        admission_state = _expect(
            _required(method, "admission_state", location),
            str,
            f"{location}.admission_state",
        )
        execution = _expect(_required(method, "execution", location), dict, f"{location}.execution")
        state = execution.get("state")
        if state not in SAFE_EXECUTION_STATES:
            raise ContractError(f"{location}.execution.state must be one of {sorted(SAFE_EXECUTION_STATES)}")
        if execution.get("network_policy") != "NO_DOWNLOADS":
            raise ContractError(f"{location}.execution.network_policy must be NO_DOWNLOADS")
        argv = _validate_argv(
            execution.get("argv"),
            f"{location}.execution.argv",
            allow_empty=state != "ARGV_WIRED",
        )
        if state == "NOT_WIRED" and argv:
            raise ContractError(f"{location}.execution.argv must remain empty while state is NOT_WIRED")
        if role == "REFERENCE_ONLY" and state != "NOT_WIRED":
            raise ContractError(
                f"{location} is REFERENCE_ONLY and must remain NOT_WIRED"
            )
        checkpoint_license = pretraining.get("checkpoint_license")
        if (
            method_id == "scribbleprompt"
            and checkpoint_license == "UNRESOLVED_NO_INDEPENDENT_WEIGHT_LICENSE_FOUND"
            and state != "NOT_WIRED"
        ):
            raise ContractError(
                "scribbleprompt cannot be wired while its independent checkpoint license is unresolved"
            )
        admission = execution.get("admission")
        if state == "ARGV_WIRED":
            _validate_admission_spec(admission, f"{location}.execution.admission")
        elif admission is not None:
            raise ContractError(
                f"{location}.execution.admission is allowed only when state is ARGV_WIRED"
            )
        cwd = execution.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ContractError(f"{location}.execution.cwd must be a string or null")
        register_record = admission_records.get(method_id)
        if register_record is None:
            raise ContractError(f"admission register omits method {method_id}")
        expected_register_fields = {
            "role": role,
            "selection": selection,
            "admission_state": admission_state,
            "execution_state": state,
        }
        for key, expected in expected_register_fields.items():
            if register_record.get(key) != expected:
                raise ContractError(
                    f"admission register {method_id}.{key} differs from methods[]"
                )

    protocol_adapters = _expect(
        _required(contract, "protocol_adapters", "contract"),
        list,
        "protocol_adapters",
    )
    for index, raw_adapter in enumerate(protocol_adapters):
        location = f"protocol_adapters[{index}]"
        adapter = _expect(raw_adapter, dict, location)
        adapter_id = _expect(_required(adapter, "id", location), str, f"{location}.id")
        selection = _expect(
            _required(adapter, "selection", location), str, f"{location}.selection"
        )
        record = admission_records.get(adapter_id)
        if record is None or record.get("role") != "ADAPT":
            raise ContractError(f"admission register omits ADAPT protocol {adapter_id}")
        if record.get("selection") != selection:
            raise ContractError(
                f"admission register {adapter_id}.selection differs from protocol_adapters[]"
            )

    references = _expect(
        _required(contract, "conditioning_module_references", "contract"),
        list,
        "conditioning_module_references",
    )
    for index, raw_reference in enumerate(references):
        location = f"conditioning_module_references[{index}]"
        reference = _expect(raw_reference, dict, location)
        reference_id = _expect(
            _required(reference, "id", location), str, f"{location}.id"
        )
        decision = _expect(
            _required(reference, "decision", location), str, f"{location}.decision"
        )
        record = admission_records.get(reference_id)
        if record is None or record.get("selection") != decision:
            raise ContractError(
                f"admission register {reference_id} differs from conditioning reference"
            )
    return contract


def load_and_validate_contract(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load comparator contract {path}: {exc}") from exc
    return validate_contract(payload)


def _matches_type(value: Any, declared: str) -> bool:
    if declared == "string":
        return isinstance(value, str)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "array":
        return isinstance(value, list)
    if declared == "object":
        return isinstance(value, dict)
    return False


def validate_manifest(document: Any, spec: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """Validate a concrete unified input or output manifest."""

    document = _expect(document, dict, f"{kind} manifest")
    if document.get("schema_version") != spec["schema_version"]:
        raise ContractError(f"{kind} manifest has wrong schema_version")
    records = document.get(spec["record_key"])
    if not isinstance(records, list):
        raise ContractError(f"{kind} manifest {spec['record_key']} must be a list")
    if len(records) < spec["min_records"]:
        raise ContractError(f"{kind} manifest requires at least {spec['min_records']} record(s)")
    fields = spec["required_fields"]
    for record_index, raw_record in enumerate(records):
        record = _expect(raw_record, dict, f"{kind} manifest record {record_index}")
        for field in fields:
            name = field["name"]
            if name not in record:
                raise ContractError(f"{kind} manifest record {record_index} is missing {name}")
            value = record[name]
            if value is None and field.get("nullable", False):
                continue
            if not _matches_type(value, field["type"]):
                raise ContractError(
                    f"{kind} manifest record {record_index}.{name} must be {field['type']}"
                )
            if "enum" in field and value not in field["enum"]:
                raise ContractError(
                    f"{kind} manifest record {record_index}.{name} must be one of {field['enum']}"
                )
    return document


def _load_manifest(path: Path, spec: Mapping[str, Any], kind: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load {kind} manifest {path}: {exc}") from exc
    return validate_manifest(payload, spec, kind)


def validate_input_against_frozen_learning_split(
    document: Mapping[str, Any],
    *,
    record_key: str,
    partition: str,
    learning_split: Path,
) -> dict[str, Any]:
    records = document.get(record_key)
    if not isinstance(records, list) or not records:
        raise ContractError("input manifest records must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ContractError(f"input manifest record {index} must be an object")
        internal = PUBLIC_TO_INTERNAL_PARTITION.get(str(record.get("split") or ""))
        if internal != partition:
            raise ContractError(
                f"input record {index}.split does not match --partition {partition}"
            )
        split_receipt = record.get("patient_split_receipt")
        if not isinstance(split_receipt, Mapping):
            raise ContractError(
                f"input record {index} omits patient_split_receipt"
            )
        normalized.append(
            {
                "case_id": record.get("case_id"),
                "patient_id": record.get("patient_id"),
                "partition": internal,
                "learning_split_sha256": split_receipt.get(
                    "learning_split_sha256"
                ),
            }
        )
    try:
        return validate_manifest_rows_against_frozen_learning_split(
            normalized,
            learning_split,
            require_episode_id=False,
            allowed_partitions={partition},
        )
    except LearningContractError as exc:
        raise ContractError(f"frozen learning-split validation failed: {exc}") from exc


def _render(value: str, variables: Mapping[str, str], location: str) -> str:
    missing = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(value)
        if field_name and field_name not in variables
    }
    if missing:
        raise ContractError(f"{location} has unresolved placeholder(s): {sorted(missing)}")
    try:
        return value.format_map(dict(variables))
    except (KeyError, ValueError) as exc:
        raise ContractError(f"cannot render {location}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, location: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{location} must be an existing regular non-symlink file: {path}")
    return path.resolve()


def validate_execution_admission(
    method: Mapping[str, Any],
    config_path: Path,
    *,
    variables: Mapping[str, str],
) -> dict[str, Any]:
    """Revalidate a current-config-bound runtime receipt before execution.

    Static ``execution.state=ARGV_WIRED`` means only that an argv has been wired. It
    never authorizes execution by itself.  The receipt must bind the exact
    config and every method-specific runtime file declared by the contract.
    """

    execution = method["execution"]
    spec = _validate_admission_spec(
        execution.get("admission"), f"{method['id']}.execution.admission"
    )
    config_path = _require_regular_file(config_path, "comparator config")
    render_variables = dict(variables)
    render_variables.setdefault("project_root", str(PROJECT.resolve()))
    receipt_path = _require_regular_file(
        Path(
            _render(
                spec["receipt"],
                render_variables,
                f"{method['id']}.execution.admission.receipt",
            )
        ),
        "runtime admission receipt",
    )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load runtime admission receipt {receipt_path}: {exc}") from exc
    receipt = _expect(receipt, dict, "runtime admission receipt")
    if receipt.get("schema_version") != spec["schema_version"]:
        raise ContractError("runtime admission receipt schema_version mismatch")
    if receipt.get("status") != spec["status"]:
        raise ContractError("runtime admission receipt status is not execution-admitted")
    config_sha256 = _sha256_file(config_path)
    if receipt.get(spec["config_sha256_field"]) != config_sha256:
        raise ContractError("runtime admission receipt is stale for the current comparator config")
    for field in spec["required_pass_fields"]:
        if receipt.get(field) != "PASS":
            raise ContractError(f"runtime admission receipt field {field} is not PASS")
    for field, expected in spec["exact_fields"].items():
        if receipt.get(field) != expected:
            raise ContractError(f"runtime admission receipt field {field} changed")
    bound_files: dict[str, dict[str, str]] = {}
    for field, raw_path in spec["file_sha256_fields"].items():
        path = _require_regular_file(
            Path(
                _render(
                    raw_path,
                    render_variables,
                    f"{method['id']}.execution.admission.file_sha256_fields.{field}",
                )
            ),
            f"runtime admission file for {field}",
        )
        observed = _sha256_file(path)
        if receipt.get(field) != observed:
            raise ContractError(f"runtime admission receipt hash mismatch for {field}")
        bound_files[field] = {"path": str(path), "sha256": observed}
    return {
        "schema_version": "PETCT-EXTERNAL-COMPARATOR-ADMISSION-CHECK-v1.0",
        "status": "PASS",
        "method_id": method["id"],
        "receipt": {"path": str(receipt_path), "sha256": _sha256_file(receipt_path)},
        "config": {"path": str(config_path), "sha256": config_sha256},
        "bound_files": bound_files,
    }


def build_execution_plan(
    contract: Mapping[str, Any],
    method_id: str,
    input_manifest: Path,
    output_manifest: Path,
    *,
    variables: Mapping[str, str],
    execute: bool,
) -> dict[str, Any]:
    """Build a serializable dry-run or execution plan without launching anything."""

    methods = {method["id"]: method for method in contract["methods"]}
    if method_id not in methods:
        raise ContractError(f"unknown method {method_id}; choose one of {sorted(methods)}")
    method = methods[method_id]
    execution = method["execution"]
    render_variables = dict(variables)
    render_variables.update(
        {
            "input_manifest": str(input_manifest.resolve()),
            "output_manifest": str(output_manifest.resolve()),
            "project_root": str(PROJECT.resolve()),
        }
    )
    if execute and execution["state"] != "ARGV_WIRED":
        raise ContractError(
            f"method {method_id} is {execution['state']}; audit and wire argv before execution"
        )
    rendered_argv = [
        _render(item, render_variables, f"{method_id}.execution.argv")
        for item in execution["argv"]
    ]
    rendered_cwd = (
        _render(execution["cwd"], render_variables, f"{method_id}.execution.cwd")
        if execution["cwd"] is not None
        else None
    )
    return {
        "schema_version": "PETCT-EXTERNAL-COMPARATOR-PLAN-v1.0",
        "mode": "EXECUTE" if execute else "DRY_RUN",
        "would_execute": bool(execute),
        "method_id": method_id,
        "display_name": method["display_name"],
        "execution_state": execution["state"],
        "runtime_admission_required": execution["state"] == "ARGV_WIRED",
        "headline_eligible": method["headline"]["eligible"],
        "current_psma_v3_exposure": method["pretraining"]["current_psma_v3_exposure"],
        "input_manifest": str(input_manifest.resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "argv": rendered_argv,
        "cwd": rendered_cwd,
        "network_policy": "NO_DOWNLOADS",
        "contract_refs": method["contracts"],
        "metric_set": method["metric_set"],
    }


def _parse_variables(raw_items: Sequence[str]) -> dict[str, str]:
    variables: dict[str, str] = {}
    for raw in raw_items:
        if "=" not in raw:
            raise ContractError("--var must use KEY=VALUE")
        key, value = raw.split("=", 1)
        if not key or key in {
            "input_manifest",
            "output_manifest",
            "project_root",
            "learning_split",
            "experiment_config",
            "partition",
        }:
            raise ContractError(f"invalid or reserved --var key: {key!r}")
        variables[key] = value
    return variables


def _offline_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(OFFLINE_ENVIRONMENT)
    return environment


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--method", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--partition", choices=("val", "test"))
    parser.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--execute", action="store_true", help="opt in to a wired adapter")
    parser.add_argument("--confirm", help="exact non-secret execution confirmation phrase")
    args = parser.parse_args(argv)

    contract = load_and_validate_contract(args.config)
    if args.execute and args.confirm != contract["execution_policy"]["confirmation_token"]:
        raise ContractError(
            "--execute requires the exact confirmation token from execution_policy.confirmation_token"
        )
    variables = _parse_variables(args.var)
    if args.learning_split is not None:
        variables["learning_split"] = str(args.learning_split.resolve())
    variables["experiment_config"] = str(args.experiment_config.resolve())
    if args.partition is not None:
        variables["partition"] = args.partition
    plan = build_execution_plan(
        contract,
        args.method,
        args.input_manifest,
        args.output_manifest,
        variables=variables,
        execute=args.execute,
    )
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.partition == "test":
        raise ContractError(
            "generic runner cannot execute the test partition; use the formal "
            "external comparator launcher with a consumed test-access receipt"
        )
    if args.partition != "val" or args.learning_split is None:
        raise ContractError(
            "generic execution requires --partition val and --learning-split; "
            "formal test execution is launcher-only"
        )
    try:
        enforce_partition_access(
            "val",
            receipt_path=None,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=None,
            output_paths=(args.output_manifest,),
        )
    except TestAccessError as exc:
        raise ContractError(str(exc)) from exc

    method = next(item for item in contract["methods"] if item["id"] == args.method)
    admission = validate_execution_admission(
        method,
        args.config,
        variables={**variables, "project_root": str(PROJECT.resolve())},
    )
    plan["runtime_admission"] = admission
    input_document = _load_manifest(
        args.input_manifest, contract["input_manifest_contract"], "input"
    )
    validate_input_against_frozen_learning_split(
        input_document,
        record_key=contract["input_manifest_contract"]["record_key"],
        partition="val",
        learning_split=args.learning_split,
    )
    subprocess.run(
        plan["argv"],
        cwd=plan["cwd"],
        check=True,
        shell=False,
        env=_offline_environment(),
    )
    output = _load_manifest(args.output_manifest, contract["output_manifest_contract"], "output")
    wrong_method = [
        record["case_id"]
        for record in output[contract["output_manifest_contract"]["record_key"]]
        if record["method_id"] != args.method
    ]
    if wrong_method:
        raise ContractError(f"output manifest contains records for a different method: {wrong_method}")
    result = dict(plan)
    result["executed"] = True
    result["validated_output_records"] = len(
        output[contract["output_manifest_contract"]["record_key"]]
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
