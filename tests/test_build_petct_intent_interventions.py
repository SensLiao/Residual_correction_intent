from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "editor"))

from build_petct_intent_interventions import build_nonidentity_shuffle  # noqa: E402
from common.petct_learning import SHUFFLE_ALGORITHM  # noqa: E402


JOINTS = (
    ("ADD", "SAME", "LOCAL"),
    ("ADD", "SAME", "COMPLETE"),
    ("ADD", "NEW", "COMPLETE"),
    ("REMOVE", "SAME", "LOCAL"),
    ("REMOVE", "SAME", "COMPLETE"),
    ("REMOVE", "NEW", "COMPLETE"),
)


def _row(episode, joint):
    operation, target, scope = joint
    return {
        "episode_id": episode,
        "operation": operation,
        "target": target,
        "scope": scope,
    }


def test_shuffle_is_deterministic_and_never_identity() -> None:
    rows = [_row(str(index), joint) for index, joint in enumerate(JOINTS)]
    first = build_nonidentity_shuffle(rows, seed=42)
    assert first == build_nonidentity_shuffle(rows, seed=42)
    assert all(
        (row["operation"], row["target"], row["scope"])
        != (
            row["original_operation"],
            row["original_target"],
            row["original_scope"],
        )
        for row in first
    )


def test_shuffle_is_order_invariant_without_replacement_and_preserves_marginals() -> (
    None
):
    rows = [
        _row(f"{index}-{replicate}", joint)
        for index, joint in enumerate(JOINTS)
        for replicate in (0, 1)
    ]
    shuffled = build_nonidentity_shuffle(rows, seed=20260717)
    assert shuffled == build_nonidentity_shuffle(list(reversed(rows)), seed=20260717)
    assert {row["episode_id"] for row in shuffled} == {
        row["source_episode_id"] for row in shuffled
    }
    assert len({row["source_episode_id"] for row in shuffled}) == len(rows)
    assert all(row["episode_id"] != row["source_episode_id"] for row in shuffled)
    assert Counter(
        (row["operation"], row["target"], row["scope"])
        for row in shuffled
    ) == Counter(
        (row["operation"], row["target"], row["scope"]) for row in rows
    )
    assert {row["algorithm"] for row in shuffled} == {SHUFFLE_ALGORITHM}
    assert {row["seed"] for row in shuffled} == {20260717}


def test_shuffle_fails_closed_when_changed_label_permutation_is_infeasible() -> None:
    rows = [
        _row("a1", JOINTS[0]),
        _row("a2", JOINTS[0]),
        _row("a3", JOINTS[0]),
        _row("b1", JOINTS[-1]),
    ]
    with pytest.raises(RuntimeError, match="infeasible"):
        build_nonidentity_shuffle(rows, seed=42)
