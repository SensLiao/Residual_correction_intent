#!/usr/bin/env python3
"""Validate and archive a protocol-locked six-class PETCT-INTENT-v2.0 response."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "PETCT-INTENT-v2.0"
GOAL_MAPPING = {
    "ADD_SAME_LOCAL": ("ADD", "SAME", "LOCAL"),
    "REMOVE_SAME_LOCAL": ("REMOVE", "SAME", "LOCAL"),
    "ADD_SAME_COMPLETE": ("ADD", "SAME", "COMPLETE"),
    "REMOVE_SAME_COMPLETE": ("REMOVE", "SAME", "COMPLETE"),
    "ADD_NEW_COMPLETE": ("ADD", "NEW", "COMPLETE"),
    "REMOVE_NEW_COMPLETE": ("REMOVE", "NEW", "COMPLETE"),
}
REQUIRED_FIELDS = {
    "schema_version",
    "decision",
    "goal",
    "target",
    "scope",
    "operation",
    "preserve",
    "intent_text",
    "alternatives",
    "confidence",
}
PRESERVE_CONTRACT = [
    "PRESERVE_UNAUTHORIZED_M0",
    "DO_NOT_CHANGE_OUTSIDE_AUTHORIZED_TARGET",
]
OPAQUE_EPISODE_PATTERN = re.compile(r"ep-[0-9a-f]{6,64}")
DIAGNOSIS_ASSERTION_TERMS = (
    "metastasis",
    "metastatic",
    "malignant",
    "prostate cancer",
    "lymphoma",
)


class IntentResponseError(RuntimeError):
    """Raised when a raw model response cannot be mechanically parsed."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_raw_response(raw_response: str) -> dict[str, Any]:
    """Require one bare JSON object; markdown/prose repair is intentionally absent."""
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise IntentResponseError("response must be one bare JSON object")
    stripped = raw_response.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        raise IntentResponseError("response must be one bare JSON object")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise IntentResponseError("response must be one bare JSON object") from exc
    if not isinstance(parsed, dict):
        raise IntentResponseError("response must be one bare JSON object")
    return parsed


def validate_intent_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate semantic consistency without treating self-confidence as probability."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return {
            "contract_version": "PETCT-INTENT-VALIDATOR-v2.0",
            "valid": False,
            "errors": ["parsed response is not an object"],
            "normalized_decision": "INVALID_OUTPUT",
            "normalized_goal": None,
            "confidence_is_probability": False,
            "unsupported_diagnosis_terms": [],
        }

    fields = set(payload)
    missing = sorted(REQUIRED_FIELDS - fields)
    unexpected = sorted(fields - REQUIRED_FIELDS)
    if missing:
        errors.append(f"missing fields: {missing}")
    if unexpected:
        errors.append(f"unexpected fields: {unexpected}")

    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    decision = payload.get("decision")
    if decision not in {"PREDICT", "ABSTAIN"}:
        errors.append("decision must be PREDICT or ABSTAIN")
    if decision == "PREDICT" and payload.get("operation") not in {"ADD", "REMOVE"}:
        errors.append("PREDICT operation must be ADD or REMOVE")
    if payload.get("preserve") != PRESERVE_CONTRACT:
        errors.append(f"preserve must equal {PRESERVE_CONTRACT}")

    text = payload.get("intent_text")
    if not isinstance(text, str) or not text.strip():
        errors.append("intent_text must be a non-empty string")
        text_for_scan = ""
    elif len(text) > 1000:
        errors.append("intent_text exceeds 1000 characters")
        text_for_scan = text.casefold()
    else:
        text_for_scan = text.casefold()
    diagnosis_terms = [
        term for term in DIAGNOSIS_ASSERTION_TERMS if term in text_for_scan
    ]
    if diagnosis_terms:
        errors.append(
            f"intent_text contains unsupported diagnosis assertion terms: {diagnosis_terms}"
        )

    confidence = payload.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            errors.append("confidence must be null or a numeric self-report in [0,1]")
        elif not 0.0 <= float(confidence) <= 1.0:
            errors.append("confidence must be in [0,1]")

    alternatives = payload.get("alternatives")
    if not isinstance(alternatives, list):
        errors.append("alternatives must be a list")
        alternatives = []
    else:
        invalid_alternatives = [
            value for value in alternatives if value not in GOAL_MAPPING
        ]
        if invalid_alternatives:
            errors.append(f"alternatives contain invalid goals: {invalid_alternatives}")
        if len(set(alternatives)) != len(alternatives):
            errors.append("alternatives must not contain duplicates")

    goal = payload.get("goal")
    target = payload.get("target")
    scope = payload.get("scope")
    if decision == "PREDICT":
        if goal not in GOAL_MAPPING:
            errors.append(f"goal must be one of {sorted(GOAL_MAPPING)}")
        elif (payload.get("operation"), target, scope) != GOAL_MAPPING[goal]:
            errors.append(
                "goal/operation/target/scope are inconsistent with PETCT-INTENT-v2.0"
            )
        if goal in alternatives:
            errors.append("alternatives must not repeat the primary goal")
    elif decision == "ABSTAIN":
        if goal is not None:
            errors.append("ABSTAIN must use goal=null, not a fourth semantic class")
        if target != "UNCERTAIN" or scope != "UNCERTAIN":
            errors.append("ABSTAIN must set target/scope to UNCERTAIN")
        if payload.get("operation") != "UNCERTAIN":
            errors.append("ABSTAIN must set operation to UNCERTAIN")

    valid = not errors
    return {
        "contract_version": "PETCT-INTENT-VALIDATOR-v2.0",
        "valid": valid,
        "errors": errors,
        "normalized_decision": decision if valid else "INVALID_OUTPUT",
        "normalized_goal": goal if valid and decision == "PREDICT" else None,
        "confidence_is_probability": False,
        "unsupported_diagnosis_terms": diagnosis_terms,
    }


def _is_same_or_nested(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


def _write_records_exclusive(records: list[tuple[Path, bytes]]) -> None:
    paths = [path for path, _ in records]
    if any(path.exists() for path in paths):
        existing = next(path for path in paths if path.exists())
        raise FileExistsError(f"refusing to overwrite existing response record: {existing}")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for path, payload in records:
            with path.open("xb") as stream:
                created.append(path)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
    except Exception:
        for path in created:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def publish_response_records(
    *,
    episode_id: str,
    attempt: int,
    raw_response: str,
    parsed_response: dict[str, Any],
    validator_record: dict[str, Any],
    raw_root: Path,
    parsed_root: Path,
    validator_root: Path,
) -> dict[str, Any]:
    """Persist raw, parsed, and validator records in separate no-clobber roots."""
    if not OPAQUE_EPISODE_PATTERN.fullmatch(episode_id):
        raise IntentResponseError("episode_id must be an opaque ep-<hex> token")
    if attempt not in {1, 2, 3}:
        raise IntentResponseError("attempt must be 1..3 (initial + at most two retries)")
    roots = [Path(raw_root).resolve(), Path(parsed_root).resolve(), Path(validator_root).resolve()]
    if any(
        _is_same_or_nested(roots[left], roots[right])
        for left in range(len(roots))
        for right in range(left + 1, len(roots))
    ):
        raise IntentResponseError("raw/parsed/validator roots must be physically disjoint")
    stem = f"{episode_id}_attempt-{attempt:02d}"
    raw_path = roots[0] / f"{stem}.txt"
    parsed_path = roots[1] / f"{stem}.json"
    validator_path = roots[2] / f"{stem}.json"
    raw_bytes = raw_response.encode("utf-8")
    parsed_bytes = (
        json.dumps(parsed_response, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    validator_bytes = (
        json.dumps(validator_record, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _write_records_exclusive(
        [
            (raw_path, raw_bytes),
            (parsed_path, parsed_bytes),
            (validator_path, validator_bytes),
        ]
    )
    return {
        "status": "COMMITTED",
        "episode_id": episode_id,
        "attempt": attempt,
        "raw_path": str(raw_path),
        "raw_sha256": _sha256_bytes(raw_bytes),
        "parsed_path": str(parsed_path),
        "parsed_sha256": _sha256_bytes(parsed_bytes),
        "validator_path": str(validator_path),
        "validator_sha256": _sha256_bytes(validator_bytes),
    }


def build_inference_receipt(
    *,
    episode_id: str,
    attempt: int,
    model_environment: str,
    run_date: str,
    prompt_version: str,
    prompt_sha256: str,
    montage_sha256: str,
    raw_response_sha256: str,
    retry_reason: str | None,
) -> dict[str, Any]:
    """Bind legal inputs and raw output without importing evaluation truth."""
    if not OPAQUE_EPISODE_PATTERN.fullmatch(episode_id):
        raise IntentResponseError("episode_id must be opaque")
    for label, digest in (
        ("prompt", prompt_sha256),
        ("montage", montage_sha256),
        ("raw response", raw_response_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise IntentResponseError(f"{label} SHA-256 is invalid")
    return {
        "contract_version": "PETCT-CODEX-INFERENCE-RECEIPT-v2.0",
        "episode_id": episode_id,
        "attempt": int(attempt),
        "run_date": run_date,
        "model_environment": model_environment,
        "checkpoint_id": None,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "montage_sha256": montage_sha256,
        "raw_response_sha256": raw_response_sha256,
        "retry_reason": retry_reason,
        "confidence_interpretation": "exploratory self-report; not a calibrated probability",
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--raw-response", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--parsed-root", type=Path, required=True)
    parser.add_argument("--validator-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    raw = args.raw_response.read_text(encoding="utf-8")
    try:
        parsed = parse_raw_response(raw)
        verdict = validate_intent_response(parsed)
    except IntentResponseError as exc:
        parsed = {}
        verdict = {
            "contract_version": "PETCT-INTENT-VALIDATOR-v2.0",
            "valid": False,
            "errors": [str(exc)],
            "normalized_decision": "INVALID_OUTPUT",
            "normalized_goal": None,
            "confidence_is_probability": False,
            "unsupported_diagnosis_terms": [],
        }
    receipt = publish_response_records(
        episode_id=args.episode_id,
        attempt=args.attempt,
        raw_response=raw,
        parsed_response=parsed,
        validator_record=verdict,
        raw_root=args.raw_root,
        parsed_root=args.parsed_root,
        validator_root=args.validator_root,
    )
    sys.stdout.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0 if verdict["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
