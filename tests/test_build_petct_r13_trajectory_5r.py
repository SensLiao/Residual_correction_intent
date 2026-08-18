"""Contract tests for the R13 five-round trajectory corpus builder and lineage.

The five-round corpus (R13-trajectory-5r) differs from the frozen R13-main
single-round corpus only in round count: strategy geometry, sibling structure,
exclusion rules and the three-lane split stay identical, while every episode
carries an explicit trajectory_id + round_index and the current state advances
by teacher-forced oracle correction (ADD = union of the gold authorized
target, REMOVE = set difference).  This file pins the pure state-algebra,
identity, document, and lineage-validation surfaces before any pipeline I/O.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from common.petct_trajectory_lineage import (  # noqa: E402
    TRAJECTORY_DATASET_ID,
    TRAJECTORY_EPISODE_SCHEMA,
    TRAJECTORY_LINEAGE_SCHEMA,
    TrajectoryContractError,
    issue_trajectory_lineage_receipt,
    validate_r13_trajectory_rows,
    validate_trajectory_lineage_receipt,
)
from data.build_petct_r13_trajectory_5r import (  # noqa: E402
    TRAJECTORY_EVAL_SCHEMA,
    TRAJECTORY_STATE_SCHEMA,
    TRAJECTORY_VISIBLE_SCHEMA,
    EpisodeContractError,
    advance_trajectory_state,
    build_state_provenance,
    build_trajectory_round_documents,
    teacher_forced_state,
    trajectory_attempt_id,
    trajectory_episode_id,
    trajectory_id,
)
from data.build_petct_scribble_dataset import scribble_attempt_id  # noqa: E402
from data.build_petct_scribble_episode import (  # noqa: E402
    _assert_visible_safe,
    canonical_intent_frame,
    generate_residual_scribble,
)


# --- shared fixture helpers -------------------------------------------------


def _sparse_simulator(mask, *, strategy, seed):
    """Deterministic stand-in for the pinned simulator.

    Picks the slice with the most residual pixels, takes the largest in-plane
    component there, and returns a sparse top-left 3x3 cluster (y-major
    order) of that component.
    """

    del strategy, seed
    z = max(range(mask.shape[2]), key=lambda index: int(mask[:, :, index].sum()))
    plane = np.asarray(mask[:, :, z]) > 0
    labels, count = _ndimage_label_2d(plane)
    areas = np.bincount(labels.reshape(-1))
    areas = np.pad(areas, (0, count + 1 - len(areas)))
    areas[0] = 0
    label_id = int(np.argmax(areas))
    coords = np.argwhere(labels == label_id)
    coords = coords[np.lexsort((coords[:, 0], coords[:, 1]))][:9]
    result = [[int(x), int(y), int(z)] for x, y in coords]
    return result, True, len(result)


def _ndimage_label_2d(plane):
    from scipy import ndimage

    return ndimage.label(plane, structure=np.ones((3, 3), dtype=np.uint8))


def _scribble_record(operation):
    residual = np.zeros((12, 12, 3), dtype=np.uint8)
    residual[2:8, 3:9, 1] = 1
    return generate_residual_scribble(
        residual,
        operation=operation,
        strategy="random",
        simulator=_sparse_simulator,
        upstream_commit="abc",
        seed=42,
    )


def _state_provenance(overrides=None):
    provenance = {
        "kind": "teacher_forced_oracle_state",
        "schema_version": TRAJECTORY_STATE_SCHEMA,
        "contract_version": "PETCT-TRAJECTORY-STATE-v1.0",
        "teacher_forcing": "ORACLE_AUTHORIZED_TARGET",
        "operation": "ADD",
        "trajectory_id": "petct-traj-0123456789abcdef01234567",
        "round_index": 1,
        "m0_sha256": "a" * 64,
        "base_m0_sha256": "b" * 64,
        "parent_state_sha256": "b" * 64,
        "corrections": [
            {"round_index": 0, "operation": "ADD", "authorized_sha256": "c" * 64}
        ],
        "input_ct_sha256": "d" * 64,
        "input_pet_sha256": "e" * 64,
        "held_out_fold": 2,
    }
    if overrides:
        provenance.update(overrides)
    return provenance


# --- trajectory identities -------------------------------------------------


def test_trajectory_id_is_deterministic_opaque_and_operation_sensitive() -> None:
    first = trajectory_id("case-a", "ADD", "centerline")
    assert first == trajectory_id("case-a", "ADD", "centerline")
    assert first.startswith("petct-traj-") and Path(first).name == first
    assert first != trajectory_id("case-a", "REMOVE", "centerline")
    assert first != trajectory_id("case-a", "ADD", "boundary")
    assert first != trajectory_id("case-b", "ADD", "centerline")


def test_round_zero_attempt_id_matches_the_single_round_attempt_id() -> None:
    # Round 0 of a five-round trajectory IS the frozen single-round attempt;
    # its attempt identity must be bit-identical for the parity contract.
    for operation in ("ADD", "REMOVE"):
        assert trajectory_attempt_id("case-a", operation, "random", 0) == (
            scribble_attempt_id("natural", "case-a", operation, "random")
        )
        assert trajectory_attempt_id("case-a", operation, "random", 1) != (
            scribble_attempt_id("natural", "case-a", operation, "random")
        )


def test_trajectory_round_episode_ids_are_unique_per_round() -> None:
    tid = trajectory_id("case-a", "ADD", "centerline")
    assert trajectory_episode_id(tid, 1) != trajectory_episode_id(tid, 2)
    assert trajectory_episode_id(tid, 1).startswith("petct-traj-ep-")
    assert (
        Path(trajectory_episode_id(tid, 1)).name
        == trajectory_episode_id(tid, 1)
    )


# --- teacher-forced state algebra -------------------------------------------


def test_teacher_forced_add_is_a_disjoint_union() -> None:
    current = np.zeros((6, 6, 2), dtype=np.uint8)
    current[1:4, 1:4, 0] = 1
    authorized = np.zeros_like(current)
    authorized[4:6, 1:4, 0] = 1
    state = teacher_forced_state(current, authorized, operation="ADD")
    assert np.array_equal(state, current | authorized)
    with pytest.raises(EpisodeContractError, match="disjoint"):
        teacher_forced_state(current, current, operation="ADD")


def test_teacher_forced_remove_is_a_set_difference() -> None:
    current = np.zeros((6, 6, 2), dtype=np.uint8)
    current[1:5, 1:5, 0] = 1
    authorized = np.zeros_like(current)
    authorized[2:4, 1:5, 0] = 1
    state = teacher_forced_state(current, authorized, operation="REMOVE")
    assert np.array_equal(state, current & ~authorized)
    escaping = np.zeros_like(current)
    escaping[0, 0, 0] = 1
    with pytest.raises(EpisodeContractError, match="subset"):
        teacher_forced_state(current, escaping, operation="REMOVE")


def test_teacher_forced_state_rejects_shape_and_binary_violations() -> None:
    with pytest.raises(EpisodeContractError, match="shape"):
        teacher_forced_state(
            np.zeros((4, 4, 1), dtype=np.uint8),
            np.zeros((4, 4, 2), dtype=np.uint8),
            operation="ADD",
        )
    with pytest.raises(EpisodeContractError, match="binary"):
        teacher_forced_state(
            np.ones((4, 4, 1), dtype=np.uint8) * 2,
            np.zeros((4, 4, 1), dtype=np.uint8),
            operation="ADD",
        )
    with pytest.raises(EpisodeContractError, match="operation"):
        teacher_forced_state(
            np.zeros((4, 4, 1), dtype=np.uint8),
            np.zeros((4, 4, 1), dtype=np.uint8),
            operation="UPDATE",
        )


def test_advance_state_residual_is_strictly_monotone_until_exhausted() -> None:
    gt = np.zeros((10, 10, 3), dtype=np.uint8)
    gt[2:8, 2:8, 1] = 1
    state = np.zeros_like(gt)
    state[2:4, 2:8, 1] = 1  # FN: columns 4..7
    authorized = np.zeros_like(gt)
    authorized[4:6, 2:8, 1] = 1  # first correction
    next_state, next_residual, exhausted = advance_trajectory_state(
        gt, state, authorized, operation="ADD"
    )
    assert not exhausted
    assert int(next_residual.sum()) < int(((gt > 0) & ~(state > 0)).sum())
    assert np.array_equal(next_residual, (gt > 0) & ~(next_state > 0))
    # Second correction exhausts the FN residual.
    final_state, final_residual, exhausted = advance_trajectory_state(
        gt, next_state, next_residual, operation="ADD"
    )
    assert exhausted
    assert np.array_equal(final_state, gt)
    assert np.array_equal(final_residual, np.zeros_like(gt))


def test_advance_state_remove_contract_and_exhaustion() -> None:
    gt = np.zeros((10, 10, 2), dtype=np.uint8)
    gt[2:8, 2:8, 0] = 1
    state = np.zeros_like(gt)
    state[2:8, 2:8, 0] = 1
    state[0:3, 0:3, 1] = 1  # FP blob
    authorized = np.zeros_like(gt)
    authorized[0:3, 0:3, 1] = 1
    next_state, next_residual, exhausted = advance_trajectory_state(
        gt, state, authorized, operation="REMOVE"
    )
    assert exhausted
    assert np.array_equal(next_state, gt)
    assert np.array_equal(next_residual, np.zeros_like(gt))
    with pytest.raises(EpisodeContractError, match="subset"):
        advance_trajectory_state(
            gt, state, (gt > 0).astype(np.uint8), operation="REMOVE"
        )
    with pytest.raises(EpisodeContractError, match="non-empty"):
        advance_trajectory_state(
            gt, state, np.zeros_like(gt), operation="ADD"
        )


# --- teacher-forced state provenance ----------------------------------------


def test_state_provenance_records_the_oracle_correction_chain() -> None:
    provenance = build_state_provenance(
        trajectory_id="petct-traj-0123456789abcdef01234567",
        round_index=2,
        operation="ADD",
        state_path=Path("/audit/states/round_2_state.nii.gz"),
        state_sha256="a" * 64,
        base_m0_sha256="b" * 64,
        parent_state_sha256="c" * 64,
        corrections=[
            {"round_index": 0, "operation": "ADD", "authorized_sha256": "d" * 64},
            {"round_index": 1, "operation": "ADD", "authorized_sha256": "e" * 64},
        ],
        input_ct_sha256="f" * 64,
        input_pet_sha256="1" * 64,
        held_out_fold=4,
    )
    assert provenance["kind"] == "teacher_forced_oracle_state"
    assert provenance["teacher_forcing"] == "ORACLE_AUTHORIZED_TARGET"
    assert provenance["schema_version"] == TRAJECTORY_STATE_SCHEMA
    assert provenance["round_index"] == 2
    assert len(provenance["corrections"]) == 2
    assert provenance["corrections"][1]["authorized_sha256"] == "e" * 64
    assert provenance["m0_sha256"] == "a" * 64
    assert provenance["parent_state_sha256"] == "c" * 64
    assert provenance["base_m0_sha256"] == "b" * 64


# --- five-round episode documents and the visible firewall ------------------


def test_round_documents_carry_five_round_schema_and_split_lanes() -> None:
    scribble = _scribble_record("ADD")
    visible, evaluation = build_trajectory_round_documents(
        episode_id="petct-traj-ep-0123456789abcdef01234567",
        trajectory_id="petct-traj-0123456789abcdef01234567",
        round_index=1,
        lane="natural",
        patient_group_hash="a" * 64,
        montage_reference="learning-visible/petct-traj-ep-0123456789abcdef01234567.npz",
        state_provenance=_state_provenance(),
        scribble_record=scribble,
        source_case_id="hidden-case",
        source_patient_id="hidden-patient",
        residual_sha256="b" * 64,
        residual_voxels=12,
        gold_intent=canonical_intent_frame("ADD_NEW_COMPLETE"),
        state_relative_derivation={"authorized_target_sha256": "c" * 64},
    )
    assert visible["schema_version"] == TRAJECTORY_VISIBLE_SCHEMA
    assert visible["trajectory_id"] == "petct-traj-0123456789abcdef01234567"
    assert visible["round_index"] == 1
    assert "gold_intent" not in visible
    assert "corrections" not in json.dumps(visible)
    assert "authorized" not in json.dumps(visible).casefold()
    _assert_visible_safe(visible)
    assert evaluation["schema_version"] == TRAJECTORY_EVAL_SCHEMA
    assert evaluation["gold_intent"]["goal"] == "ADD_NEW_COMPLETE"
    assert evaluation["m0_provenance"]["corrections"][0]["round_index"] == 0
    assert (
        evaluation["state_relative_derivation"]["authorized_target_sha256"]
        == "c" * 64
    )


def test_round_visible_document_refuses_label_and_target_material() -> None:
    scribble = _scribble_record("ADD")
    with pytest.raises(EpisodeContractError, match="forbidden evaluation"):
        build_trajectory_round_documents(
            episode_id="petct-traj-ep-0123456789abcdef01234567",
            trajectory_id="petct-traj-0123456789abcdef01234567",
            round_index=1,
            lane="natural",
            patient_group_hash="a" * 64,
            montage_reference="learning-visible/ADD_NEW_COMPLETE.npz",
            state_provenance=_state_provenance(),
            scribble_record=scribble,
            source_case_id="hidden-case",
            source_patient_id="hidden-patient",
            residual_sha256="b" * 64,
            residual_voxels=12,
            gold_intent=canonical_intent_frame("ADD_NEW_COMPLETE"),
            state_relative_derivation={},
        )
    # The visible state provenance is an allowlist projection: the audit
    # corrections chain and trajectory identity must stay eval-lane only.
    visible, evaluation = build_trajectory_round_documents(
        episode_id="petct-traj-ep-0123456789abcdef01234567",
        trajectory_id="petct-traj-0123456789abcdef01234567",
        round_index=1,
        lane="natural",
        patient_group_hash="a" * 64,
        montage_reference="learning-visible/petct-traj-ep-0123456789abcdef01234567.npz",
        state_provenance=_state_provenance(),
        scribble_record=scribble,
        source_case_id="hidden-case",
        source_patient_id="hidden-patient",
        residual_sha256="b" * 64,
        residual_voxels=12,
        gold_intent=canonical_intent_frame("ADD_NEW_COMPLETE"),
        state_relative_derivation={},
    )
    assert visible["m0_provenance"] == {
        "kind": "teacher_forced_oracle_state",
        "schema_version": TRAJECTORY_STATE_SCHEMA,
        "contract_version": "PETCT-TRAJECTORY-STATE-v1.0",
        "m0_sha256": "a" * 64,
        "input_ct_sha256": "d" * 64,
        "input_pet_sha256": "e" * 64,
    }
    assert "corrections" not in visible["m0_provenance"]
    assert evaluation["m0_provenance"]["corrections"][0]["authorized_sha256"]


def test_round_documents_require_consistent_episode_identity() -> None:
    scribble = _scribble_record("ADD")
    with pytest.raises(EpisodeContractError, match="round_index"):
        build_trajectory_round_documents(
            episode_id="petct-traj-ep-0123456789abcdef01234567",
            trajectory_id="petct-traj-0123456789abcdef01234567",
            round_index=0,  # round 0 documents belong to the single-round builder
            lane="natural",
            patient_group_hash="a" * 64,
            montage_reference="learning-visible/petct-traj-ep-0123456789abcdef01234567.npz",
            state_provenance=_state_provenance(overrides={"round_index": 0}),
            scribble_record=scribble,
            source_case_id="hidden-case",
            source_patient_id="hidden-patient",
            residual_sha256="b" * 64,
            residual_voxels=12,
            gold_intent=canonical_intent_frame("ADD_NEW_COMPLETE"),
            state_relative_derivation={},
        )


# --- trajectory row validation -----------------------------------------------


def _row(
    episode_id,
    *,
    trajectory,
    case_id="case-a",
    patient_id="patient-a",
    partition="train",
    operation="ADD",
    strategy="centerline",
    round_index=0,
    round_count=5,
    status="COMPLETE_5_ROUNDS",
):
    return {
        "episode_id": episode_id,
        "episode_family_id": trajectory,
        "trajectory_id": trajectory,
        "case_id": case_id,
        "patient_id": patient_id,
        "partition": partition,
        "operation": operation,
        "strategy": strategy,
        "round_index": round_index,
        "round_count": round_count,
        "trajectory_status": status,
        "scribble_count": 1,
        "source_m0_lineage": "M0_V6_FIVEFOLD_OOF",
    }


def test_trajectory_row_validation_accepts_the_legal_sibling_structure() -> None:
    rows = []
    for strategy in ("centerline", "random", "boundary"):
        for round_index in range(3):
            rows.append(
                _row(
                    "ep-%s-%d" % (strategy, round_index),
                    trajectory="traj-" + strategy,
                    strategy=strategy,
                    round_index=round_index,
                    round_count=3,
                )
            )
    validate_r13_trajectory_rows(rows)  # must not raise


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        (lambda rows: rows[0].update({"episode_id": rows[1]["episode_id"]}), "unique"),
        (lambda rows: rows[0].update({"trajectory_id": ""}), "trajectory"),
        (lambda rows: rows[0].update({"episode_family_id": "other"}), "family"),
        (lambda rows: rows[0].update({"round_index": 5}), "round"),
        (lambda rows: rows[0].update({"round_index": 2}), "contiguous"),
        (lambda rows: rows[0].update({"round_count": 9}), "round_count"),
        (lambda rows: rows[0].update({"scribble_count": 2}), "scribble"),
        (lambda rows: rows[0].update({"strategy": "freehand"}), "strategy"),
        (lambda rows: rows[0].update({"operation": "REMOVE"}), "operation"),
        (lambda rows: rows[0].update({"partition": "test"}), "partition"),
        (
            lambda rows: rows.append(
                _row(
                    "ep-extra",
                    trajectory="traj-extra",
                    strategy="random",
                    round_index=0,
                    round_count=1,
                )
            ),
            "at most three",
        ),
        (lambda rows: rows[0].update({"source_m0_lineage": "OLD"}), "lineage"),
    ),
)
def test_trajectory_row_validation_rejects_contract_violations(
    mutate, match: str
) -> None:
    rows = []
    for strategy in ("centerline", "random", "boundary"):
        for round_index in range(3):
            rows.append(
                _row(
                    "ep-%s-%d" % (strategy, round_index),
                    trajectory="traj-" + strategy,
                    strategy=strategy,
                    round_index=round_index,
                    round_count=3,
                )
            )
    mutate(rows)
    with pytest.raises(TrajectoryContractError, match=match):
        validate_r13_trajectory_rows(rows)


def test_trajectory_row_validation_rejects_empty_and_gapped_inputs() -> None:
    with pytest.raises(TrajectoryContractError, match="empty"):
        validate_r13_trajectory_rows([])
    rows = [
        _row("ep-0", trajectory="traj-a", round_index=0, round_count=3),
        _row("ep-2", trajectory="traj-a", round_index=2, round_count=3),
    ]
    with pytest.raises(TrajectoryContractError, match="contiguous"):
        validate_r13_trajectory_rows(rows)


# --- trajectory lineage receipt ----------------------------------------------


def _file_record(path: Path) -> dict:
    import hashlib

    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_lineage_receipt_pins_the_five_round_identity(tmp_path: Path) -> None:
    oof = tmp_path / "oof-ready.json"
    oof.write_text('{"oof":true}\n', encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text('{"split":true}\n', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text('{"config":true}\n', encoding="utf-8")
    receipt = tmp_path / "lineage.json"
    payload = {
        "schema_version": TRAJECTORY_LINEAGE_SCHEMA,
        "status": "PASS",
        "dataset_id": TRAJECTORY_DATASET_ID,
        "source_m0_lineage": "M0_V6_FIVEFOLD_OOF",
        "mainline_eligible": False,
        "lifecycle": "active",
        "episode_schema": TRAJECTORY_EPISODE_SCHEMA,
        "round_count": 5,
        "scribbles_per_episode": 1,
        "strategy_is_label": False,
        "partitions": ["train", "val"],
        "locked_test_present": False,
        "teacher_forcing": "ORACLE_AUTHORIZED_TARGET",
        "oof_ready": _file_record(oof),
        "learning_split": _file_record(split),
        "experiment_config": _file_record(config),
    }
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    validated = validate_trajectory_lineage_receipt(receipt)
    assert validated["dataset_id"] == TRAJECTORY_DATASET_ID
    assert validated["round_count"] == 5
    assert validated["receipt_sha256"]


def test_lineage_receipt_rejects_single_round_and_mainline_fields(
    tmp_path: Path,
) -> None:
    oof = tmp_path / "oof-ready.json"
    oof.write_text('{"oof":true}\n', encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text('{"split":true}\n', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text('{"config":true}\n', encoding="utf-8")
    base = {
        "schema_version": TRAJECTORY_LINEAGE_SCHEMA,
        "status": "PASS",
        "dataset_id": TRAJECTORY_DATASET_ID,
        "source_m0_lineage": "M0_V6_FIVEFOLD_OOF",
        "mainline_eligible": False,
        "lifecycle": "active",
        "episode_schema": TRAJECTORY_EPISODE_SCHEMA,
        "round_count": 5,
        "scribbles_per_episode": 1,
        "strategy_is_label": False,
        "partitions": ["train", "val"],
        "locked_test_present": False,
        "oof_ready": _file_record(oof),
        "learning_split": _file_record(split),
        "experiment_config": _file_record(config),
    }
    single_round = dict(base)
    single_round["round_count"] = 1
    single_round["episode_schema"] = "single_round_one_scribble_one_strategy_v1"
    single_round["dataset_id"] = "R13-main-single-round"
    path = tmp_path / "single-round.json"
    path.write_text(
        json.dumps(single_round, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(
        TrajectoryContractError, match="round_count|schema_version|dataset_id"
    ):
        validate_trajectory_lineage_receipt(path)
    mainline = dict(base)
    mainline["mainline_eligible"] = True
    path = tmp_path / "mainline.json"
    path.write_text(json.dumps(mainline, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(TrajectoryContractError, match="mainline"):
        validate_trajectory_lineage_receipt(path)


def test_lineage_issue_fails_closed_on_invalid_oof(tmp_path: Path) -> None:
    oof = tmp_path / "oof-ready.json"
    oof.write_text('{"oof":true}\n', encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text('{"split":true}\n', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text('{"config":true}\n', encoding="utf-8")
    output = tmp_path / "lineage.json"
    with pytest.raises(TrajectoryContractError):
        issue_trajectory_lineage_receipt(
            oof_ready=oof,
            learning_split=split,
            experiment_config=config,
            output=output,
        )
    assert not output.exists()


def test_lineage_issue_refuses_to_overwrite(tmp_path: Path, monkeypatch) -> None:
    import common.petct_trajectory_lineage as lineage

    monkeypatch.setattr(lineage, "validate_m0_v6_oof_ready", lambda path: {})
    oof = tmp_path / "oof-ready.json"
    oof.write_text('{"oof":true}\n', encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text('{"split":true}\n', encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text('{"config":true}\n', encoding="utf-8")
    output = tmp_path / "lineage.json"
    issue_trajectory_lineage_receipt(
        oof_ready=oof, learning_split=split, experiment_config=config, output=output
    )
    assert output.exists()
    with pytest.raises(TrajectoryContractError, match="overwrite"):
        issue_trajectory_lineage_receipt(
            oof_ready=oof, learning_split=split, experiment_config=config, output=output
        )


# --- parity anchor ------------------------------------------------------------


def test_round_zero_identity_reuses_the_single_round_opaque_episode_formula() -> None:
    # The builder is required to derive round-0 episode ids from the frozen
    # opaque_episode_id(case, goal, strategy) formula so the parity test can
    # match rows by episode_id directly.
    from data.build_petct_r13_trajectory_5r import round0_episode_id
    from data.build_petct_scribble_dataset import opaque_episode_id

    for goal in ("ADD_SAME_LOCAL", "REMOVE_NEW_COMPLETE"):
        assert round0_episode_id("case-a", goal, "random") == opaque_episode_id(
            "case-a", goal, "random"
        )
