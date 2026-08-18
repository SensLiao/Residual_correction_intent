#!/usr/bin/env python3
"""Pure five-round trajectory primitives shared by the corpus builders.

Identity hashing, teacher-forced state algebra, state provenance, and the
round 1-4 visible/evaluation document construction.  This module holds no
pipeline I/O; ``build_petct_r13_trajectory_5r`` re-exports everything here so
the frozen test surface keeps importing from one place.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
for support_dir in (SCRIPTS_ROOT / "baseline", SCRIPTS_ROOT / "data"):
    if str(support_dir) not in sys.path:
        sys.path.insert(0, str(support_dir))

from data.build_petct_scribble_dataset import (  # noqa: E402
    opaque_episode_id,
    scribble_attempt_id,
)
from data.build_petct_scribble_episode import (  # noqa: E402
    INTENT_SCHEMA_VERSION,
    EpisodeContractError,
    _assert_visible_safe,
    _binary_3d,
    _visible_m0_provenance,
    compute_fn_residual,
    compute_fp_residual,
)

TRAJECTORY_VISIBLE_SCHEMA = "PETCT-TRAJECTORY-VISIBLE-v1.0"
TRAJECTORY_EVAL_SCHEMA = "PETCT-TRAJECTORY-EVAL-v1.0"
TRAJECTORY_STATE_SCHEMA = "PETCT-TRAJECTORY-STATE-v1.0"
TRAJECTORY_SUMMARY_SCHEMA = "PETCT-TRAJECTORY-SUMMARY-v1.0"
TRAJECTORY_READY_SCHEMA = "PETCT-TRAJECTORY-EPISODES-READY-v1.0"
TRAJECTORY_READY_PHASE = "FIVE_ROUND_TEACHER_FORCED_TRAJECTORY_MATERIALIZATION"
MAX_ROUNDS = 5
STATUS_COMPLETE = "COMPLETE_5_ROUNDS"
STATUS_EXHAUSTED = "RESIDUAL_EXHAUSTED"
STATUS_TRUNCATED = "TRUNCATED"
ROUND_RESIDUAL_CONTRACT = "TEACHER_FORCED_STATE_RESIDUAL_v1"
PARITY_SINGLE_ROUND_DATASET = "R13-main-single-round"

# The five-round row carries the single-round field set plus this explicit
# identity block; round-0 rows are otherwise field-identical to R13-main.
TRAJECTORY_ROW_KEYS = (
    "trajectory_id",
    "round_index",
    "round_count",
    "trajectory_status",
    "termination_reason",
)

# The visible scribble projection mirrors the frozen single-round document.
TRAJECTORY_SCRIBBLE_VISIBLE_KEYS = (
    "polarity",
    "strategy",
    "requested_strategy",
    "effective_strategy",
    "strategy_fallback",
    "fallback_reason",
    "strategy_audit",
    "seed",
    "coordinates_xyz",
    "coordinate_count",
    "coordinate_sha256",
    "scribble_density_mode",
    "fallback_mode",
)


def trajectory_id(case_id: str, operation: str, strategy: str) -> str:
    digest = hashlib.sha256(
        ("PETCT-TRAJECTORY-v1|%s|%s|%s" % (case_id, operation, strategy)).encode(
            "utf-8"
        )
    ).hexdigest()
    return "petct-traj-" + digest[:24]


def trajectory_attempt_id(
    case_id: str, operation: str, strategy: str, round_index: int
) -> str:
    """Round-0 attempts keep the frozen single-round attempt identity."""
    if int(round_index) == 0:
        return scribble_attempt_id("natural", case_id, operation, strategy)
    digest = hashlib.sha256(
        (
            "PETCT-TRAJECTORY-ATTEMPT-v1|%s|%s|%s|%d"
            % (case_id, operation, strategy, int(round_index))
        ).encode("utf-8")
    ).hexdigest()
    return "traj-attempt-" + digest[:24]


def trajectory_episode_id(trajectory_id_: str, round_index: int) -> str:
    digest = hashlib.sha256(
        ("PETCT-TRAJECTORY-EPISODE-v1|%s|%d" % (trajectory_id_, int(round_index)))
        .encode("utf-8")
    ).hexdigest()
    return "petct-traj-ep-" + digest[:24]


def round0_episode_id(case_id: str, goal: str, strategy: str) -> str:
    """Round-0 episode ids reuse the frozen single-round formula (parity)."""
    return opaque_episode_id(case_id, goal, strategy)


def teacher_forced_state(
    current: np.ndarray, authorized: np.ndarray, *, operation: str
) -> np.ndarray:
    """Apply one oracle correction: ADD unions, REMOVE subtracts, fail closed."""
    current_bin = _binary_3d(current, name="current state")
    authorized_bin = _binary_3d(authorized, name="authorized target")
    if current_bin.shape != authorized_bin.shape:
        raise EpisodeContractError(
            "current state and authorized target shape mismatch"
        )
    if operation == "ADD":
        if np.any(current_bin & authorized_bin):
            raise EpisodeContractError(
                "ADD authorized target must be disjoint from the current state"
            )
        return current_bin | authorized_bin
    if operation == "REMOVE":
        if np.any(authorized_bin & ~current_bin):
            raise EpisodeContractError(
                "REMOVE authorized target must be a subset of the current state"
            )
        return current_bin & ~authorized_bin
    raise EpisodeContractError("operation must be ADD or REMOVE")


def advance_trajectory_state(
    gt: np.ndarray,
    state: np.ndarray,
    authorized: np.ndarray,
    *,
    operation: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Advance one round and return (next state, next residual, exhausted)."""
    gt_bin = _binary_3d(gt, name="GT")
    state_bin = _binary_3d(state, name="state")
    authorized_bin = _binary_3d(authorized, name="authorized target")
    if not (gt_bin.shape == state_bin.shape == authorized_bin.shape):
        raise EpisodeContractError("GT/state/authorized target shape mismatch")
    if not authorized_bin.any():
        raise EpisodeContractError(
            "authorized target must be non-empty to advance a trajectory"
        )
    residual_fn = (
        compute_fn_residual if operation == "ADD" else compute_fp_residual
    )
    if operation not in {"ADD", "REMOVE"}:
        raise EpisodeContractError("operation must be ADD or REMOVE")
    current_residual = residual_fn(gt_bin, state_bin)
    if not current_residual.any():
        raise EpisodeContractError(
            "cannot advance an already-exhausted trajectory"
        )
    if np.any(authorized_bin & ~current_residual):
        raise EpisodeContractError(
            "authorized target must be a subset of the current operation residual"
        )
    next_state = teacher_forced_state(
        state_bin, authorized_bin, operation=operation
    )
    next_residual = residual_fn(gt_bin, next_state)
    if int(next_residual.sum()) >= int(current_residual.sum()):
        raise EpisodeContractError(
            "teacher-forced advance did not shrink the operation residual"
        )
    return next_state, next_residual, not next_residual.any()


def build_state_provenance(
    *,
    trajectory_id: str,
    round_index: int,
    operation: str,
    state_path: Path,
    state_sha256: str,
    base_m0_sha256: str,
    parent_state_sha256: str,
    corrections: Sequence[Mapping[str, Any]],
    input_ct_sha256: str,
    input_pet_sha256: str,
    held_out_fold: int,
) -> dict[str, Any]:
    """Audit-lane provenance of a teacher-forced intermediate state."""
    return {
        "kind": "teacher_forced_oracle_state",
        "schema_version": TRAJECTORY_STATE_SCHEMA,
        "contract_version": "PETCT-TRAJECTORY-STATE-v1.0",
        "teacher_forcing": "ORACLE_AUTHORIZED_TARGET",
        "operation": operation,
        "trajectory_id": trajectory_id,
        "round_index": int(round_index),
        "state_path": str(state_path),
        "m0_sha256": state_sha256,
        "base_m0_sha256": base_m0_sha256,
        "parent_state_sha256": parent_state_sha256,
        "corrections": [dict(correction) for correction in corrections],
        "input_ct_sha256": input_ct_sha256,
        "input_pet_sha256": input_pet_sha256,
        "held_out_fold": int(held_out_fold),
    }


def build_trajectory_round_documents(
    *,
    episode_id: str,
    trajectory_id: str,
    round_index: int,
    lane: str,
    patient_group_hash: str,
    montage_reference: str,
    state_provenance: Mapping[str, Any],
    scribble_record: Mapping[str, Any],
    source_case_id: str,
    source_patient_id: str,
    residual_sha256: str,
    residual_voxels: int,
    gold_intent: Mapping[str, Any],
    state_relative_derivation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create separable visible/evaluation documents for rounds 1-4."""
    if not episode_id or Path(episode_id).name != episode_id:
        raise EpisodeContractError("episode_id must be an opaque filename-safe token")
    if not trajectory_id or Path(trajectory_id).name != trajectory_id:
        raise EpisodeContractError("trajectory_id must be an opaque token")
    if lane != "natural":
        raise EpisodeContractError("trajectory rounds require the natural lane")
    if (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or not 1 <= round_index < MAX_ROUNDS
    ):
        raise EpisodeContractError(
            "round_index must be 1..4: round 0 belongs to the single-round builder"
        )
    if len(patient_group_hash) != 64:
        raise EpisodeContractError("patient_group_hash must be a SHA-256 digest")
    operation = gold_intent.get("operation")
    expected_polarity = {"ADD": "foreground", "REMOVE": "background"}.get(operation)
    if (
        expected_polarity is None
        or scribble_record.get("polarity") != expected_polarity
    ):
        raise EpisodeContractError("cue polarity must agree with the gold operation")
    if gold_intent.get("schema_version") != INTENT_SCHEMA_VERSION:
        raise EpisodeContractError("gold intent does not match PETCT-INTENT-v2.0")
    visible = {
        "schema_version": TRAJECTORY_VISIBLE_SCHEMA,
        "episode_id": episode_id,
        "trajectory_id": trajectory_id,
        "round_index": int(round_index),
        "lane": lane,
        "patient_group_hash": patient_group_hash,
        "montage_reference": montage_reference,
        "m0_provenance": _visible_m0_provenance(lane, dict(state_provenance)),
        "scribble": {
            key: scribble_record[key]
            for key in TRAJECTORY_SCRIBBLE_VISIBLE_KEYS
            if key in scribble_record
        },
        "expected_model_output_schema": INTENT_SCHEMA_VERSION,
    }
    _assert_visible_safe(visible)
    evaluation = {
        "schema_version": TRAJECTORY_EVAL_SCHEMA,
        "episode_id": episode_id,
        "trajectory_id": trajectory_id,
        "round_index": int(round_index),
        "lane": lane,
        "source_case_id": source_case_id,
        "source_patient_id": source_patient_id,
        "residual_sha256": residual_sha256,
        "residual_voxels": int(residual_voxels),
        "gold_intent": dict(gold_intent),
        "scribble_provenance": dict(scribble_record),
        "m0_provenance": dict(state_provenance),
        "state_relative_derivation": dict(state_relative_derivation),
    }
    return visible, evaluation


def _single_round_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): row[key] for key in row if str(key) not in TRAJECTORY_ROW_KEYS
    }

