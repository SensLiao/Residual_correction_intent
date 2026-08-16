#!/usr/bin/env python3
"""Build and validate the PSMA-only patient-level development manifest.

The input is an enriched, server-side case-audit export.  This tool reads the
JSON receipt only; it never opens PET, CT, or GT assets.  Ten patients are
selected by a frozen SHA-256 rule, repeated examinations remain clustered, and
the anonymous public manifest is kept physically separate from the private
source mapping used by the evaluation plane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from build_petct_scribble_episode import (
    STRATEGIES,
    assign_scribble_strategy,
)


DATASET_ID = "PSMA-PET-CT-Lesions-v3"
SOURCE_EXPORT_VERSION = "PETCT-PSMA-CASE-AUDIT-EXPORT-v1.0"
PATIENT_SPLIT_AUTHORITY_VERSION = "PETCT-PATIENT-SPLIT-AUTHORITY-v1.0"
RULE_VERSION = "PETCT-DEVELOPMENT-PATIENT-SELECTION-v1.0"
PUBLIC_SCHEMA_VERSION = "PETCT-DEVELOPMENT-MANIFEST-PUBLIC-v1.0"
PRIVATE_SCHEMA_VERSION = "PETCT-DEVELOPMENT-MANIFEST-PRIVATE-v1.0"
RECEIPT_SCHEMA_VERSION = "PETCT-DEVELOPMENT-MANIFEST-RECEIPT-v1.0"
SELECTED_PATIENT_COUNT = 10
STRATEGY_SALT = "PETCT-PILOT-v1"
STRATEGY_QUOTAS = {"centerline": 4, "random": 3, "boundary": 3}
SOURCE_ASSET_ROLES = ("pet", "ct", "gt")
DEVELOPMENT_POOL = "development_pool"
ALLOWED_SPLIT_PARTITIONS = {DEVELOPMENT_POOL, "locked"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_PROMPT_TOKENS = {"click", "clicks", "point", "points"}
FORBIDDEN_DATASET_TOKENS = {"fdg"}
PROHIBITED_ACTION_KEYS = (
    "scribble_generation",
    "state_materialization",
    "codex_inference",
    "training",
    "metric_computation",
    "result_creation",
)


class DevelopmentManifestError(ValueError):
    """Raised when a development-manifest contract is violated."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(document: Any) -> bytes:
    return (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _rule_digest(purpose: str, patient_id: str) -> str:
    return _sha256_bytes(f"{RULE_VERSION}|{purpose}|{patient_id}".encode("utf-8"))


def _case_priority_digest(patient_id: str, case_id: str) -> str:
    return _sha256_bytes(
        f"{RULE_VERSION}|case-priority|{patient_id}|{case_id}".encode("utf-8")
    )


def _tokens(value: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return set(re.findall(r"[a-z0-9]+", spaced.casefold()))


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_strings(child)
    elif isinstance(value, str):
        yield value


def _reject_forbidden_semantics(value: Any) -> None:
    for item in _walk_strings(value):
        tokens = _tokens(item)
        if tokens & FORBIDDEN_PROMPT_TOKENS:
            raise DevelopmentManifestError(
                "scribble-only contract rejects click or point prompt semantics"
            )
        if tokens & FORBIDDEN_DATASET_TOKENS:
            raise DevelopmentManifestError("FDG is outside the PSMA-only contract")


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DevelopmentManifestError(f"{name} must be an object")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise DevelopmentManifestError(f"{name} must be a list")
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DevelopmentManifestError(f"missing or invalid {name}")
    if value != value.strip():
        raise DevelopmentManifestError(f"{name} must not contain outer whitespace")
    return value


def _canonical_identifier(value: Any, name: str) -> str:
    return _require_text(value, name).casefold()


def _require_sha256(value: Any, name: str) -> str:
    digest = _require_text(value, name).casefold()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise DevelopmentManifestError(f"missing or invalid {name}")
    return digest


def _require_exact_int(value: Any, expected: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise DevelopmentManifestError(f"{name} must equal {expected}")


def _validate_source_asset(value: Any, *, case_id: str, role: str) -> dict[str, str]:
    asset = _require_mapping(value, f"source asset {role} for {case_id}")
    path = _require_text(asset.get("path"), f"source asset path {role} for {case_id}")
    if not path.startswith("/"):
        raise DevelopmentManifestError(
            f"source asset path must be an absolute server path: {case_id}:{role}"
        )
    digest = _require_sha256(
        asset.get("sha256"),
        f"source asset hash {role} for {case_id}",
    )
    return {"path": path, "sha256": digest}


def _validate_source_document(source: Any) -> tuple[list[dict[str, Any]], int]:
    document = _require_mapping(source, "source audit export")
    _reject_forbidden_semantics(document)
    if document.get("schema_version") != SOURCE_EXPORT_VERSION:
        raise DevelopmentManifestError(
            f"source schema_version must be {SOURCE_EXPORT_VERSION}"
        )
    if document.get("audit_status") != "PASS":
        raise DevelopmentManifestError("source audit_status must be PASS")
    if document.get("dataset_id") != DATASET_ID:
        raise DevelopmentManifestError("PSMA-only dataset_id mismatch")
    if document.get("dataset_scope") != "PSMA":
        raise DevelopmentManifestError("PSMA-only dataset_scope mismatch")
    if document.get("split_name") != DEVELOPMENT_POOL:
        raise DevelopmentManifestError("source pool must be development-only")
    if document.get("split_unit") != "patient":
        raise DevelopmentManifestError("split must be patient-level, never case-level")
    if document.get("patient_disjoint") is not True:
        raise DevelopmentManifestError("patient-disjoint source split is required")
    _require_sha256(
        document.get("source_audit_receipt_sha256"),
        "source audit receipt hash",
    )
    _require_sha256(
        document.get("source_nifti_audit_sha256"),
        "source NIfTI audit hash",
    )
    _require_sha256(
        document.get("source_split_manifest_sha256"),
        "source split manifest hash",
    )

    raw_records = _require_list(document.get("case_records"), "case_records")
    if not raw_records:
        raise DevelopmentManifestError("case_records must not be empty")
    normalized: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    patient_splits: dict[str, set[str]] = defaultdict(set)
    for index, raw_record in enumerate(raw_records):
        record = _require_mapping(raw_record, f"case_records[{index}]")
        case_id = _canonical_identifier(record.get("case_id"), "case_id")
        patient_id = _canonical_identifier(record.get("patient_id"), "patient_id")
        if case_id in case_ids:
            raise DevelopmentManifestError(f"duplicate case_id: {case_id}")
        case_ids.add(case_id)
        if record.get("status") != "PASS":
            raise DevelopmentManifestError(f"case audit status is not PASS: {case_id}")
        if record.get("dataset_id") != DATASET_ID:
            raise DevelopmentManifestError(
                f"PSMA-only case dataset mismatch: {case_id}"
            )
        if record.get("dataset_scope") != "PSMA":
            raise DevelopmentManifestError(f"PSMA-only case scope mismatch: {case_id}")
        if record.get("tracer") != "PSMA":
            if str(record.get("tracer", "")).casefold() == "fdg":
                raise DevelopmentManifestError("FDG is outside the PSMA-only contract")
            raise DevelopmentManifestError(f"PSMA-only tracer mismatch: {case_id}")
        split = _require_text(record.get("split"), f"split for {case_id}")
        if split != DEVELOPMENT_POOL:
            raise DevelopmentManifestError(
                f"case records must be development-only: {case_id}"
            )
        patient_splits[patient_id].add(split)
        source_assets = _require_mapping(
            record.get("source_assets"),
            f"source assets for {case_id}",
        )
        if set(source_assets) != set(SOURCE_ASSET_ROLES):
            raise DevelopmentManifestError(
                f"source asset roles for {case_id} must be pet/ct/gt"
            )
        normalized_assets = {
            role: _validate_source_asset(
                source_assets[role],
                case_id=case_id,
                role=role,
            )
            for role in SOURCE_ASSET_ROLES
        }
        normalized.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "source_assets": normalized_assets,
            }
        )

    if any(splits != {DEVELOPMENT_POOL} for splits in patient_splits.values()):
        raise DevelopmentManifestError("a patient appears in multiple source splits")
    unique_patients = len(patient_splits)
    if unique_patients < SELECTED_PATIENT_COUNT:
        raise DevelopmentManifestError(
            f"at least {SELECTED_PATIENT_COUNT} unique development patients are required"
        )
    declared_cases = document.get("case_count")
    _require_exact_int(declared_cases, len(normalized), "declared case_count")
    declared_patients = document.get("patient_count")
    _require_exact_int(
        declared_patients,
        unique_patients,
        "declared patient_count",
    )
    return sorted(normalized, key=lambda row: row["case_id"]), unique_patients


def _validate_authority_documents(
    source: dict[str, Any],
    *,
    audit_receipt_document: Any,
    audit_receipt_sha256: str,
    split_document: Any,
    split_input_sha256: str,
) -> dict[str, list[str]]:
    audit_receipt = _require_mapping(
        audit_receipt_document,
        "authoritative audit receipt",
    )
    observed_audit_receipt_sha = _require_sha256(
        audit_receipt_sha256,
        "authoritative audit receipt hash",
    )
    if source.get("source_audit_receipt_sha256") != observed_audit_receipt_sha:
        raise DevelopmentManifestError("authoritative audit receipt hash mismatch")
    if (
        audit_receipt.get("status") != "COMMITTED"
        or audit_receipt.get("audit_status") != "PASS"
    ):
        raise DevelopmentManifestError("authoritative audit receipt is not PASS")
    outputs = _require_mapping(
        audit_receipt.get("outputs"),
        "authoritative audit outputs",
    )
    nifti_audit = _require_mapping(
        outputs.get("psma_v3_nifti_audit.json"),
        "authoritative NIfTI audit output",
    )
    nifti_audit_sha = _require_sha256(
        nifti_audit.get("sha256"),
        "authoritative NIfTI audit hash",
    )
    if source.get("source_nifti_audit_sha256") != nifti_audit_sha:
        raise DevelopmentManifestError("authoritative NIfTI audit hash mismatch")

    split = _require_mapping(split_document, "frozen patient split")
    _reject_forbidden_semantics(split)
    observed_split_sha = _require_sha256(
        split_input_sha256,
        "frozen patient split hash",
    )
    if source.get("source_split_manifest_sha256") != observed_split_sha:
        raise DevelopmentManifestError("frozen patient split hash mismatch")
    if split.get("schema_version") != PATIENT_SPLIT_AUTHORITY_VERSION:
        raise DevelopmentManifestError("frozen patient split schema mismatch")
    if split.get("status") != "FROZEN_CONTRACT_ONLY":
        raise DevelopmentManifestError("patient split must be frozen before selection")
    if split.get("dataset_id") != DATASET_ID or split.get("dataset_scope") != "PSMA":
        raise DevelopmentManifestError("frozen patient split is not PSMA-only")
    if split.get("split_unit") != "patient":
        raise DevelopmentManifestError("frozen split must be patient-level")
    if split.get("patient_disjoint") is not True:
        raise DevelopmentManifestError("frozen split must be patient-disjoint")
    if split.get("source_audit_receipt_sha256") != observed_audit_receipt_sha:
        raise DevelopmentManifestError("split/audit receipt binding mismatch")
    if split.get("source_nifti_audit_sha256") != nifti_audit_sha:
        raise DevelopmentManifestError("split/NIfTI audit binding mismatch")

    patient_rows = _require_list(split.get("patients"), "frozen split patients")
    patient_map: dict[str, list[str]] = {}
    case_ids: set[str] = set()
    development_map: dict[str, list[str]] = {}
    for index, raw_row in enumerate(patient_rows):
        row = _require_mapping(raw_row, f"frozen split patient[{index}]")
        _require_exact_keys(
            row,
            {"patient_id", "partition", "case_ids"},
            "frozen split patient",
        )
        patient_id = _canonical_identifier(row.get("patient_id"), "patient_id")
        if patient_id in patient_map:
            raise DevelopmentManifestError(
                f"duplicate patient in frozen split: {patient_id}"
            )
        partition = _require_text(row.get("partition"), f"partition for {patient_id}")
        if partition not in ALLOWED_SPLIT_PARTITIONS:
            raise DevelopmentManifestError(f"invalid split partition: {partition}")
        raw_case_ids = _require_list(row.get("case_ids"), f"case_ids for {patient_id}")
        normalized_case_ids = sorted(
            _canonical_identifier(case_id, "case_id") for case_id in raw_case_ids
        )
        if not normalized_case_ids:
            raise DevelopmentManifestError(
                f"patient has no cases in split: {patient_id}"
            )
        if len(set(normalized_case_ids)) != len(normalized_case_ids):
            raise DevelopmentManifestError(
                f"duplicate case within patient split: {patient_id}"
            )
        overlap = case_ids.intersection(normalized_case_ids)
        if overlap:
            raise DevelopmentManifestError(
                f"case assigned to multiple patients in split: {sorted(overlap)[0]}"
            )
        case_ids.update(normalized_case_ids)
        patient_map[patient_id] = normalized_case_ids
        if partition == DEVELOPMENT_POOL:
            development_map[patient_id] = normalized_case_ids
    _require_exact_int(
        split.get("full_cohort_patient_count"),
        len(patient_map),
        "full_cohort_patient_count",
    )
    _require_exact_int(
        split.get("full_cohort_case_count"),
        len(case_ids),
        "full_cohort_case_count",
    )
    if len(development_map) < SELECTED_PATIENT_COUNT:
        raise DevelopmentManifestError(
            f"development pool requires at least {SELECTED_PATIENT_COUNT} patients"
        )
    return development_map


def _select_patient_ids(patient_ids: Iterable[str]) -> list[str]:
    candidates = sorted(set(patient_ids))
    selected: list[str] = []
    for strategy in STRATEGIES:
        quota = STRATEGY_QUOTAS[strategy]
        stratum = [
            patient_id
            for patient_id in candidates
            if assign_scribble_strategy(patient_id, salt=STRATEGY_SALT) == strategy
        ]
        stratum.sort(
            key=lambda patient_id: (
                _rule_digest("select", patient_id),
                patient_id,
            )
        )
        if len(stratum) < quota:
            raise DevelopmentManifestError(
                f"development pool lacks {quota} patients in {strategy} stratum"
            )
        selected.extend(stratum[:quota])
    selected.sort(
        key=lambda patient_id: (
            _rule_digest("select", patient_id),
            patient_id,
        )
    )
    return selected


def compile_development_manifests(
    source_document: Any,
    source_input_sha256: str,
    *,
    audit_receipt_document: Any,
    audit_receipt_sha256: str,
    split_document: Any,
    split_input_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Compile deterministic public/private documents without writing files."""

    input_digest = _require_sha256(source_input_sha256, "source input hash")
    source = _require_mapping(source_document, "source audit export")
    records, candidate_patient_count = _validate_source_document(source)
    development_map = _validate_authority_documents(
        source,
        audit_receipt_document=audit_receipt_document,
        audit_receipt_sha256=audit_receipt_sha256,
        split_document=split_document,
        split_input_sha256=split_input_sha256,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["patient_id"]].append(record)
    observed_development_map = {
        patient_id: sorted(record["case_id"] for record in patient_records)
        for patient_id, patient_records in grouped.items()
    }
    if set(observed_development_map) != set(development_map):
        raise DevelopmentManifestError(
            "source export does not match the frozen development pool patient set"
        )
    if observed_development_map != development_map:
        raise DevelopmentManifestError(
            "source export does not match the frozen development pool case set"
        )
    if candidate_patient_count != len(development_map):
        raise DevelopmentManifestError("candidate patient count authority mismatch")
    selected_patient_ids = _select_patient_ids(grouped)

    private_patients: list[dict[str, Any]] = []
    public_patients: list[dict[str, Any]] = []
    for ordinal, patient_id in enumerate(selected_patient_ids, start=1):
        public_patient_id = f"DEV-P{ordinal:03d}"
        ordered_records = sorted(
            grouped[patient_id],
            key=lambda row: (
                _case_priority_digest(patient_id, row["case_id"]),
                row["case_id"],
            ),
        )
        cases = [
            {
                "case_id": record["case_id"],
                "case_priority": priority,
                "case_priority_sha256": _case_priority_digest(
                    patient_id,
                    record["case_id"],
                ),
                "source_assets": record["source_assets"],
            }
            for priority, record in enumerate(ordered_records, start=1)
        ]
        strategy = assign_scribble_strategy(patient_id, salt=STRATEGY_SALT)
        private_patients.append(
            {
                "public_patient_id": public_patient_id,
                "patient_id": patient_id,
                "selection_sha256": _rule_digest("select", patient_id),
                "scribble_strategy": strategy,
                "primary_case_id": cases[0]["case_id"],
                "case_count": len(cases),
                "cases": cases,
            }
        )
        public_patients.append(
            {
                "public_patient_id": public_patient_id,
                "scribble_strategy": strategy,
                "case_count": len(cases),
            }
        )

    split_contract = {
        "name": "development",
        "unit": "patient",
        "patient_disjoint": True,
    }
    prompt_contract = {
        "modality": "scribble",
        "polarity": "foreground",
        "strategies": list(STRATEGIES),
    }
    prohibited_actions = dict.fromkeys(PROHIBITED_ACTION_KEYS, 0)
    private = {
        "schema_version": PRIVATE_SCHEMA_VERSION,
        "status": "SELECTED_NOT_MATERIALIZED",
        "dataset_id": DATASET_ID,
        "dataset_scope": "PSMA",
        "split_contract": split_contract,
        "selection_rule_version": RULE_VERSION,
        "provenance": {
            "source_schema_version": source["schema_version"],
            "source_input_sha256": input_digest,
            "source_audit_receipt_sha256": _require_sha256(
                source["source_audit_receipt_sha256"],
                "source audit receipt hash",
            ),
            "source_nifti_audit_sha256": _require_sha256(
                source["source_nifti_audit_sha256"],
                "source NIfTI audit hash",
            ),
            "source_split_manifest_sha256": _require_sha256(
                source["source_split_manifest_sha256"],
                "source split manifest hash",
            ),
            "selection_algorithm": (
                "shared strategy strata then ascending "
                "sha256(rule_version|select|canonical_patient_id), patient_id tie-break"
            ),
            "strategy_algorithm": (
                "build_petct_scribble_episode.assign_scribble_strategy with "
                "PETCT-PILOT-v1, fixed selected quotas 4/3/3"
            ),
            "strategy_salt": STRATEGY_SALT,
            "case_priority_algorithm": (
                "ascending sha256(rule_version|case-priority|patient_id|case_id), "
                "case_id tie-break"
            ),
            "source_asset_hash_contract": "DECLARED_SHA256_NOT_REHASHED",
        },
        "candidate_patient_count": candidate_patient_count,
        "selected_patient_count": len(private_patients),
        "selected_case_count": sum(row["case_count"] for row in private_patients),
        "strategy_counts": dict(
            sorted(
                Counter(row["scribble_strategy"] for row in private_patients).items()
            )
        ),
        "prompt_contract": prompt_contract,
        "patients": private_patients,
        "prohibited_actions_completed": prohibited_actions,
    }
    private_sha256 = _sha256_bytes(_json_bytes(private))
    public = {
        "schema_version": PUBLIC_SCHEMA_VERSION,
        "status": "SELECTED_NOT_MATERIALIZED",
        "dataset_id": DATASET_ID,
        "dataset_scope": "PSMA",
        "split_contract": split_contract,
        "selection_rule_version": RULE_VERSION,
        "source_input_sha256": input_digest,
        "source_audit_receipt_sha256": private["provenance"][
            "source_audit_receipt_sha256"
        ],
        "source_nifti_audit_sha256": private["provenance"]["source_nifti_audit_sha256"],
        "source_split_manifest_sha256": private["provenance"][
            "source_split_manifest_sha256"
        ],
        "candidate_patient_count": candidate_patient_count,
        "selected_patient_count": len(public_patients),
        "selected_case_count": sum(row["case_count"] for row in public_patients),
        "strategy_counts": private["strategy_counts"],
        "prompt_contract": prompt_contract,
        "patients": public_patients,
        "private_manifest_sha256": private_sha256,
        "prohibited_actions_completed": prohibited_actions,
    }
    validate_development_manifest_documents(public, private)
    return {"public": public, "private": private}


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise DevelopmentManifestError(
            f"{name} fields mismatch; missing={missing}, extra={extra}"
        )


def validate_development_manifest_documents(
    public_document: Any,
    private_document: Any,
) -> dict[str, Any]:
    """Validate privacy, patient clustering, quotas, and cross-plane binding."""

    public = _require_mapping(public_document, "public manifest")
    private = _require_mapping(private_document, "private manifest")
    _reject_forbidden_semantics(public)
    _reject_forbidden_semantics(private)
    _require_exact_keys(
        public,
        {
            "schema_version",
            "status",
            "dataset_id",
            "dataset_scope",
            "split_contract",
            "selection_rule_version",
            "source_input_sha256",
            "source_audit_receipt_sha256",
            "source_nifti_audit_sha256",
            "source_split_manifest_sha256",
            "candidate_patient_count",
            "selected_patient_count",
            "selected_case_count",
            "strategy_counts",
            "prompt_contract",
            "patients",
            "private_manifest_sha256",
            "prohibited_actions_completed",
        },
        "public manifest",
    )
    _require_exact_keys(
        private,
        {
            "schema_version",
            "status",
            "dataset_id",
            "dataset_scope",
            "split_contract",
            "selection_rule_version",
            "provenance",
            "candidate_patient_count",
            "selected_patient_count",
            "selected_case_count",
            "strategy_counts",
            "prompt_contract",
            "patients",
            "prohibited_actions_completed",
        },
        "private manifest",
    )
    if public["schema_version"] != PUBLIC_SCHEMA_VERSION:
        raise DevelopmentManifestError("public manifest schema_version mismatch")
    if private["schema_version"] != PRIVATE_SCHEMA_VERSION:
        raise DevelopmentManifestError("private manifest schema_version mismatch")
    for name, document in (("public", public), ("private", private)):
        if document["status"] != "SELECTED_NOT_MATERIALIZED":
            raise DevelopmentManifestError(f"{name} status mismatch")
        if document["dataset_id"] != DATASET_ID or document["dataset_scope"] != "PSMA":
            raise DevelopmentManifestError(f"{name} manifest is not PSMA-only")
        if document["selection_rule_version"] != RULE_VERSION:
            raise DevelopmentManifestError(f"{name} selection rule mismatch")
        if document["split_contract"] != {
            "name": "development",
            "unit": "patient",
            "patient_disjoint": True,
        }:
            raise DevelopmentManifestError(
                f"{name} split contract is not development-only patient-level"
            )
        if document["prompt_contract"] != {
            "modality": "scribble",
            "polarity": "foreground",
            "strategies": list(STRATEGIES),
        }:
            raise DevelopmentManifestError(
                f"{name} prompt contract is not scribble-only"
            )
        _require_exact_int(
            document["selected_patient_count"],
            SELECTED_PATIENT_COUNT,
            f"{name} selected_patient_count",
        )
        if document["strategy_counts"] != dict(sorted(STRATEGY_QUOTAS.items())):
            raise DevelopmentManifestError(f"{name} strategy quota mismatch")
        if document["prohibited_actions_completed"] != dict.fromkeys(
            PROHIBITED_ACTION_KEYS,
            0,
        ):
            raise DevelopmentManifestError(
                f"{name} prohibited-action ledger must contain exact zero counts"
            )

    public_patients = _require_list(public["patients"], "public patients")
    private_patients = _require_list(private["patients"], "private patients")
    if (
        len(public_patients) != SELECTED_PATIENT_COUNT
        or len(private_patients) != SELECTED_PATIENT_COUNT
    ):
        raise DevelopmentManifestError("manifest patient list length mismatch")
    public_by_id: dict[str, dict[str, Any]] = {}
    for row in public_patients:
        patient = _require_mapping(row, "public patient")
        _require_exact_keys(
            patient,
            {"public_patient_id", "scribble_strategy", "case_count"},
            "public patient",
        )
        public_id = _require_text(patient["public_patient_id"], "public_patient_id")
        if public_id in public_by_id:
            raise DevelopmentManifestError(f"duplicate public patient: {public_id}")
        public_by_id[public_id] = patient

    private_ids: set[str] = set()
    public_ids_private: set[str] = set()
    all_case_ids: set[str] = set()
    for row in private_patients:
        patient = _require_mapping(row, "private patient")
        _require_exact_keys(
            patient,
            {
                "public_patient_id",
                "patient_id",
                "selection_sha256",
                "scribble_strategy",
                "primary_case_id",
                "case_count",
                "cases",
            },
            "private patient",
        )
        patient_id = _canonical_identifier(patient["patient_id"], "patient_id")
        if patient_id in private_ids:
            raise DevelopmentManifestError(f"duplicate patient: {patient_id}")
        private_ids.add(patient_id)
        public_id = _require_text(patient["public_patient_id"], "public_patient_id")
        if public_id in public_ids_private:
            raise DevelopmentManifestError(f"duplicate public patient: {public_id}")
        public_ids_private.add(public_id)
        if patient["selection_sha256"] != _rule_digest("select", patient_id):
            raise DevelopmentManifestError(f"selection hash mismatch: {patient_id}")
        expected_strategy = assign_scribble_strategy(
            patient_id,
            salt=STRATEGY_SALT,
        )
        if patient["scribble_strategy"] != expected_strategy:
            raise DevelopmentManifestError(
                f"shared strategy authority mismatch: {patient_id}"
            )
        cases = _require_list(patient["cases"], f"cases for {patient_id}")
        if not cases:
            raise DevelopmentManifestError(f"patient has no cases: {patient_id}")
        try:
            _require_exact_int(
                patient["case_count"],
                len(cases),
                f"case_count for {patient_id}",
            )
        except DevelopmentManifestError:
            raise DevelopmentManifestError(f"case_count mismatch: {patient_id}")
        observed_priorities: list[int] = []
        observed_priority_hashes: list[str] = []
        for raw_case in cases:
            case = _require_mapping(raw_case, f"case for {patient_id}")
            _require_exact_keys(
                case,
                {
                    "case_id",
                    "case_priority",
                    "case_priority_sha256",
                    "source_assets",
                },
                "private case",
            )
            case_id = _canonical_identifier(case["case_id"], "case_id")
            if case_id in all_case_ids:
                raise DevelopmentManifestError(f"duplicate case_id: {case_id}")
            all_case_ids.add(case_id)
            expected_priority_hash = _case_priority_digest(patient_id, case_id)
            if case["case_priority_sha256"] != expected_priority_hash:
                raise DevelopmentManifestError(
                    f"case priority hash mismatch: {case_id}"
                )
            observed_priorities.append(case["case_priority"])
            observed_priority_hashes.append(case["case_priority_sha256"])
            assets = _require_mapping(
                case["source_assets"], f"source assets for {case_id}"
            )
            if set(assets) != set(SOURCE_ASSET_ROLES):
                raise DevelopmentManifestError(
                    f"source asset roles mismatch: {case_id}"
                )
            for role in SOURCE_ASSET_ROLES:
                _validate_source_asset(assets[role], case_id=case_id, role=role)
        if observed_priorities != list(range(1, len(cases) + 1)):
            raise DevelopmentManifestError(
                f"case priorities are not contiguous: {patient_id}"
            )
        if observed_priority_hashes != sorted(observed_priority_hashes):
            raise DevelopmentManifestError(
                f"case priority order mismatch: {patient_id}"
            )
        if patient["primary_case_id"] != cases[0]["case_id"]:
            raise DevelopmentManifestError(f"primary case mismatch: {patient_id}")
        public_patient = public_by_id.get(public_id)
        if public_patient is None:
            raise DevelopmentManifestError(
                f"private patient missing in public plane: {public_id}"
            )
        if public_patient != {
            "public_patient_id": public_id,
            "scribble_strategy": patient["scribble_strategy"],
            "case_count": patient["case_count"],
        }:
            raise DevelopmentManifestError(
                f"public/private patient mismatch: {public_id}"
            )

    if set(public_by_id) != public_ids_private:
        raise DevelopmentManifestError("public/private patient identifiers differ")
    if Counter(row["scribble_strategy"] for row in private_patients) != Counter(
        STRATEGY_QUOTAS
    ):
        raise DevelopmentManifestError("private strategy quota mismatch")
    selected_case_count = sum(row["case_count"] for row in private_patients)
    if (
        public["selected_case_count"] != selected_case_count
        or private["selected_case_count"] != selected_case_count
    ):
        raise DevelopmentManifestError("selected_case_count mismatch")
    if public["candidate_patient_count"] != private["candidate_patient_count"]:
        raise DevelopmentManifestError("candidate_patient_count mismatch")

    provenance = _require_mapping(private["provenance"], "private provenance")
    _require_exact_keys(
        provenance,
        {
            "source_schema_version",
            "source_input_sha256",
            "source_audit_receipt_sha256",
            "source_nifti_audit_sha256",
            "source_split_manifest_sha256",
            "selection_algorithm",
            "strategy_algorithm",
            "strategy_salt",
            "case_priority_algorithm",
            "source_asset_hash_contract",
        },
        "private provenance",
    )
    if provenance["source_schema_version"] != SOURCE_EXPORT_VERSION:
        raise DevelopmentManifestError("source schema provenance mismatch")
    if provenance["source_asset_hash_contract"] != "DECLARED_SHA256_NOT_REHASHED":
        raise DevelopmentManifestError("source asset hash contract mismatch")
    if provenance["strategy_salt"] != STRATEGY_SALT:
        raise DevelopmentManifestError("shared strategy salt mismatch")
    if public["source_input_sha256"] != provenance.get("source_input_sha256"):
        raise DevelopmentManifestError("source input hash mismatch")
    if public["source_audit_receipt_sha256"] != provenance.get(
        "source_audit_receipt_sha256"
    ):
        raise DevelopmentManifestError("source audit receipt hash mismatch")
    if public["source_nifti_audit_sha256"] != provenance.get(
        "source_nifti_audit_sha256"
    ):
        raise DevelopmentManifestError("source NIfTI audit hash mismatch")
    if public["source_split_manifest_sha256"] != provenance.get(
        "source_split_manifest_sha256"
    ):
        raise DevelopmentManifestError("source split manifest hash mismatch")
    _require_sha256(public["source_input_sha256"], "source input hash")
    _require_sha256(
        public["source_audit_receipt_sha256"],
        "source audit receipt hash",
    )
    expected_private_sha = _sha256_bytes(_json_bytes(private))
    if public["private_manifest_sha256"] != expected_private_sha:
        raise DevelopmentManifestError("private manifest hash mismatch")
    return {
        "status": "PASS",
        "selected_patient_count": SELECTED_PATIENT_COUNT,
        "selected_case_count": selected_case_count,
        "strategy_counts": dict(sorted(STRATEGY_QUOTAS.items())),
    }


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return first == second or first in second.parents or second in first.parents


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _load_json_bytes(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise DevelopmentManifestError(f"missing {name}: {path}") from exc
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DevelopmentManifestError(f"invalid JSON {name}: {path}") from exc
    return _require_mapping(document, name), payload


def build_development_manifests(
    source_path: Path | str,
    *,
    audit_receipt_path: Path | str,
    patient_split_path: Path | str,
    visible_root: Path | str,
    eval_root: Path | str,
) -> dict[str, Any]:
    """Build no-clobber public/private manifests and a private receipt."""

    source_path = Path(source_path).resolve()
    audit_receipt_path = Path(audit_receipt_path).resolve()
    patient_split_path = Path(patient_split_path).resolve()
    visible_root = Path(visible_root).resolve()
    eval_root = Path(eval_root).resolve()
    if _paths_overlap(visible_root, eval_root):
        raise DevelopmentManifestError(
            "visible and evaluation roots must be physically disjoint"
        )
    source, source_payload = _load_json_bytes(source_path, "source audit export")
    audit_receipt, audit_receipt_payload = _load_json_bytes(
        audit_receipt_path,
        "authoritative audit receipt",
    )
    patient_split, patient_split_payload = _load_json_bytes(
        patient_split_path,
        "frozen patient split",
    )
    source_input_sha256 = _sha256_bytes(source_payload)
    artifacts = compile_development_manifests(
        source,
        source_input_sha256,
        audit_receipt_document=audit_receipt,
        audit_receipt_sha256=_sha256_bytes(audit_receipt_payload),
        split_document=patient_split,
        split_input_sha256=_sha256_bytes(patient_split_payload),
    )
    public_path = visible_root / "petct_development_manifest.public.json"
    private_path = eval_root / "petct_development_manifest.private.json"
    receipt_path = eval_root / "petct_development_manifest.receipt.json"
    targets = (public_path, private_path, receipt_path)
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite development manifest output: {existing[0]}"
        )
    visible_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)
    public_payload = _json_bytes(artifacts["public"])
    private_payload = _json_bytes(artifacts["private"])
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "SELECTED_NOT_MATERIALIZED",
        "source_path": source_path.as_posix(),
        "source_input_sha256": source_input_sha256,
        "audit_receipt_path": audit_receipt_path.as_posix(),
        "source_audit_receipt_sha256": artifacts["public"][
            "source_audit_receipt_sha256"
        ],
        "patient_split_path": patient_split_path.as_posix(),
        "source_split_manifest_sha256": artifacts["public"][
            "source_split_manifest_sha256"
        ],
        "source_nifti_audit_sha256": artifacts["public"]["source_nifti_audit_sha256"],
        "selection_rule_version": RULE_VERSION,
        "public_manifest_path": public_path.as_posix(),
        "public_manifest_sha256": _sha256_bytes(public_payload),
        "private_manifest_path": private_path.as_posix(),
        "private_manifest_sha256": _sha256_bytes(private_payload),
        "receipt_path": receipt_path.as_posix(),
        "selected_patient_count": SELECTED_PATIENT_COUNT,
        "selected_case_count": artifacts["public"]["selected_case_count"],
        "strategy_counts": dict(sorted(STRATEGY_QUOTAS.items())),
        "real_pilot_executed": False,
        "scribble_generated": False,
        "codex_call_count": 0,
        "training_run_count": 0,
        "experiment_result_count": 0,
    }
    receipt_payload = _json_bytes(receipt)
    created: list[Path] = []
    try:
        _write_exclusive(private_path, private_payload)
        created.append(private_path)
        _write_exclusive(public_path, public_payload)
        created.append(public_path)
        _write_exclusive(receipt_path, receipt_payload)
        created.append(receipt_path)
    except Exception:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return receipt


def validate_published_development_manifests(
    *,
    source_path: Path | str,
    audit_receipt_path: Path | str,
    patient_split_path: Path | str,
    public_manifest_path: Path | str,
    private_manifest_path: Path | str,
    receipt_path: Path | str,
) -> dict[str, Any]:
    """Validate persisted hashes and recompute the frozen selection from source."""

    source_path = Path(source_path).resolve()
    audit_receipt_path = Path(audit_receipt_path).resolve()
    patient_split_path = Path(patient_split_path).resolve()
    public_path = Path(public_manifest_path).resolve()
    private_path = Path(private_manifest_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    if _paths_overlap(public_path.parent, private_path.parent):
        raise DevelopmentManifestError(
            "visible and evaluation roots must be physically disjoint"
        )
    if receipt_path == public_path or public_path.parent in receipt_path.parents:
        raise DevelopmentManifestError(
            "receipt must not be stored in the visible plane"
        )
    if receipt_path.parent != private_path.parent:
        raise DevelopmentManifestError("receipt must remain in the evaluation plane")
    source, source_payload = _load_json_bytes(source_path, "source audit export")
    audit_receipt, audit_receipt_payload = _load_json_bytes(
        audit_receipt_path,
        "authoritative audit receipt",
    )
    patient_split, patient_split_payload = _load_json_bytes(
        patient_split_path,
        "frozen patient split",
    )
    public, public_payload = _load_json_bytes(public_path, "public manifest")
    private, private_payload = _load_json_bytes(private_path, "private manifest")
    receipt, _ = _load_json_bytes(receipt_path, "manifest receipt")
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise DevelopmentManifestError("receipt schema_version mismatch")
    digest_checks = {
        "source input": (
            receipt.get("source_input_sha256"),
            _sha256_bytes(source_payload),
        ),
        "audit receipt": (
            receipt.get("source_audit_receipt_sha256"),
            _sha256_bytes(audit_receipt_payload),
        ),
        "patient split": (
            receipt.get("source_split_manifest_sha256"),
            _sha256_bytes(patient_split_payload),
        ),
        "public manifest": (
            receipt.get("public_manifest_sha256"),
            _sha256_bytes(public_payload),
        ),
        "private manifest": (
            receipt.get("private_manifest_sha256"),
            _sha256_bytes(private_payload),
        ),
    }
    for name, (declared, observed) in digest_checks.items():
        if declared != observed:
            raise DevelopmentManifestError(f"{name} hash mismatch")
    expected_paths = {
        "source_path": source_path.as_posix(),
        "audit_receipt_path": audit_receipt_path.as_posix(),
        "patient_split_path": patient_split_path.as_posix(),
        "public_manifest_path": public_path.as_posix(),
        "private_manifest_path": private_path.as_posix(),
        "receipt_path": receipt_path.as_posix(),
    }
    for key, expected_path in expected_paths.items():
        if receipt.get(key) != expected_path:
            raise DevelopmentManifestError(f"receipt {key} mismatch")
    if receipt.get("status") != "SELECTED_NOT_MATERIALIZED":
        raise DevelopmentManifestError("receipt status mismatch")
    if receipt.get("selection_rule_version") != RULE_VERSION:
        raise DevelopmentManifestError("receipt selection rule mismatch")
    receipt_bindings = {
        "source_nifti_audit_sha256": public.get("source_nifti_audit_sha256"),
        "selected_patient_count": public.get("selected_patient_count"),
        "selected_case_count": public.get("selected_case_count"),
        "strategy_counts": public.get("strategy_counts"),
    }
    for key, expected_value in receipt_bindings.items():
        if receipt.get(key) != expected_value:
            raise DevelopmentManifestError(f"receipt binding mismatch: {key}")
    expected_inactive_state = {
        "real_pilot_executed": False,
        "scribble_generated": False,
        "codex_call_count": 0,
        "training_run_count": 0,
        "experiment_result_count": 0,
    }
    for key, expected_value in expected_inactive_state.items():
        if receipt.get(key) != expected_value:
            raise DevelopmentManifestError(f"receipt inactive-state mismatch: {key}")
    validation = validate_development_manifest_documents(public, private)
    expected = compile_development_manifests(
        source,
        _sha256_bytes(source_payload),
        audit_receipt_document=audit_receipt,
        audit_receipt_sha256=_sha256_bytes(audit_receipt_payload),
        split_document=patient_split,
        split_input_sha256=_sha256_bytes(patient_split_payload),
    )
    if public != expected["public"]:
        raise DevelopmentManifestError("published public manifest is not reproducible")
    if private != expected["private"]:
        raise DevelopmentManifestError("published private manifest is not reproducible")
    return {
        "status": "PASS",
        "source_input_sha256": _sha256_bytes(source_payload),
        "public_manifest_sha256": _sha256_bytes(public_payload),
        "private_manifest_sha256": _sha256_bytes(private_payload),
        **{key: value for key, value in validation.items() if key != "status"},
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build no-clobber manifests")
    build.add_argument("--source-audit", type=Path, required=True)
    build.add_argument("--audit-receipt", type=Path, required=True)
    build.add_argument("--patient-split", type=Path, required=True)
    build.add_argument("--visible-root", type=Path, required=True)
    build.add_argument("--eval-root", type=Path, required=True)
    validate = subparsers.add_parser("validate", help="validate published manifests")
    validate.add_argument("--source-audit", type=Path, required=True)
    validate.add_argument("--audit-receipt", type=Path, required=True)
    validate.add_argument("--patient-split", type=Path, required=True)
    validate.add_argument("--public-manifest", type=Path, required=True)
    validate.add_argument("--private-manifest", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "build":
        result = build_development_manifests(
            args.source_audit,
            audit_receipt_path=args.audit_receipt,
            patient_split_path=args.patient_split,
            visible_root=args.visible_root,
            eval_root=args.eval_root,
        )
    else:
        result = validate_published_development_manifests(
            source_path=args.source_audit,
            audit_receipt_path=args.audit_receipt,
            patient_split_path=args.patient_split,
            public_manifest_path=args.public_manifest,
            private_manifest_path=args.private_manifest,
            receipt_path=args.receipt,
        )
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
