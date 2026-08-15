from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from common.petct_route_a_core import LEGAL_JOINT_GOALS, intent_slots_from_goal  # noqa: E402
from p2t.validate_petct_intent_response import (  # noqa: E402
    PRESERVE_CONTRACT,
    REQUIRED_FIELDS,
    IntentResponseError,
    build_inference_receipt,
    parse_raw_response,
    publish_response_records,
    validate_intent_response,
)


def _predict(goal: str) -> dict:
    operation, target, scope = intent_slots_from_goal(goal)
    return {
        "schema_version": "PETCT-INTENT-v2.0",
        "decision": "PREDICT",
        "goal": goal,
        "operation": operation,
        "target": target,
        "scope": scope,
        "preserve": list(PRESERVE_CONTRACT),
        "intent_text": f"Execute {goal} within the authorized residual.",
        "alternatives": [],
        "confidence": None,
    }


@pytest.mark.parametrize("goal", LEGAL_JOINT_GOALS)
def test_all_six_predict_frames_validate_with_exact_ten_fields(goal: str) -> None:
    payload = _predict(goal)
    assert set(payload) == REQUIRED_FIELDS
    assert validate_intent_response(payload)["valid"] is True


def test_abstain_is_uncertain_not_a_seventh_class() -> None:
    payload = {
        "schema_version": "PETCT-INTENT-v2.0",
        "decision": "ABSTAIN",
        "goal": None,
        "operation": "UNCERTAIN",
        "target": "UNCERTAIN",
        "scope": "UNCERTAIN",
        "preserve": list(PRESERVE_CONTRACT),
        "intent_text": "Insufficient evidence to choose one legal intent.",
        "alternatives": ["ADD_SAME_LOCAL", "REMOVE_SAME_LOCAL"],
        "confidence": None,
    }
    verdict = validate_intent_response(payload)
    assert verdict["valid"] is True
    assert verdict["normalized_goal"] is None


@pytest.mark.parametrize("goal", ["SAME_LOCAL", "ADD_NEW_LOCAL"])
def test_legacy_and_illegal_goals_fail_closed(goal: str) -> None:
    payload = _predict("ADD_SAME_LOCAL")
    payload["goal"] = goal
    assert validate_intent_response(payload)["valid"] is False


def test_protocol_is_v2_utf8_json() -> None:
    path = PROJECT / "protocols" / "petct_codex_prompt_v2.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    assert protocol["intent_schema_version"] == "PETCT-INTENT-v2.0"
    assert list(protocol["goal_contract"]) == list(LEGAL_JOINT_GOALS)


def test_parser_refuses_markdown_repair() -> None:
    with pytest.raises(IntentResponseError):
        parse_raw_response("```json\n{}\n```")


@pytest.mark.parametrize("mutation", ["missing", "extra", "retired_relation", "schema"])
def test_exact_field_and_schema_contract_fail_closed(mutation: str) -> None:
    payload = _predict("ADD_SAME_LOCAL")
    if mutation == "missing":
        payload.pop("scope")
    elif mutation == "extra":
        payload["explanation"] = "not allowed"
    elif mutation == "retired_relation":
        payload["relation"] = "ATTACHED"
    else:
        payload["schema_version"] = "PETCT-INTENT-v1.0"
    assert validate_intent_response(payload)["valid"] is False


def test_goal_slot_mismatch_and_alternative_failures_are_detected() -> None:
    mismatch = _predict("ADD_SAME_LOCAL")
    mismatch["operation"] = "REMOVE"
    assert validate_intent_response(mismatch)["valid"] is False
    duplicate = _predict("ADD_SAME_LOCAL")
    duplicate["alternatives"] = [
        "REMOVE_SAME_LOCAL",
        "REMOVE_SAME_LOCAL",
    ]
    assert validate_intent_response(duplicate)["valid"] is False
    self_alt = _predict("ADD_SAME_LOCAL")
    self_alt["alternatives"] = ["ADD_SAME_LOCAL"]
    assert validate_intent_response(self_alt)["valid"] is False


@pytest.mark.parametrize("confidence", [-0.1, 1.1, True, "high"])
def test_confidence_range_and_type_are_strict(confidence) -> None:
    payload = _predict("REMOVE_SAME_COMPLETE")
    payload["confidence"] = confidence
    assert validate_intent_response(payload)["valid"] is False


def test_abstain_requires_all_three_uncertain_slots() -> None:
    payload = {
        **_predict("ADD_SAME_LOCAL"),
        "decision": "ABSTAIN",
        "goal": None,
        "operation": "ADD",
        "target": "UNCERTAIN",
        "scope": "UNCERTAIN",
    }
    assert validate_intent_response(payload)["valid"] is False


def test_response_records_are_hash_bound_disjoint_and_no_clobber(tmp_path: Path) -> None:
    parsed = _predict("ADD_SAME_LOCAL")
    raw = json.dumps(parsed, ensure_ascii=False)
    verdict = validate_intent_response(parsed)
    receipt = publish_response_records(
        episode_id="ep-abcdef",
        attempt=1,
        raw_response=raw,
        parsed_response=parsed,
        validator_record=verdict,
        raw_root=tmp_path / "raw",
        parsed_root=tmp_path / "parsed",
        validator_root=tmp_path / "validator",
    )
    assert all(len(receipt[key]) == 64 for key in ("raw_sha256", "parsed_sha256", "validator_sha256"))
    with pytest.raises(FileExistsError):
        publish_response_records(
            episode_id="ep-abcdef",
            attempt=1,
            raw_response=raw,
            parsed_response=parsed,
            validator_record=verdict,
            raw_root=tmp_path / "raw",
            parsed_root=tmp_path / "parsed",
            validator_root=tmp_path / "validator",
        )
    with pytest.raises(IntentResponseError, match="disjoint"):
        publish_response_records(
            episode_id="ep-fedcba",
            attempt=1,
            raw_response=raw,
            parsed_response=parsed,
            validator_record=verdict,
            raw_root=tmp_path / "same",
            parsed_root=tmp_path / "same" / "parsed",
            validator_root=tmp_path / "other",
        )


def test_inference_receipt_rejects_unbound_hashes() -> None:
    with pytest.raises(IntentResponseError, match="SHA-256"):
        build_inference_receipt(
            episode_id="ep-abcdef",
            attempt=1,
            model_environment="local",
            run_date="2026-07-31",
            prompt_version="PETCT-CODEX-PROMPT-v2.0",
            prompt_sha256="bad",
            montage_sha256="a" * 64,
            raw_response_sha256="b" * 64,
            retry_reason=None,
        )
