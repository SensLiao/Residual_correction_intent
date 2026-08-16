from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from data.build_petct_scribble_dataset import (  # noqa: E402
    opaque_episode_id,
    scribble_attempt_id,
)
from data.materialize_petct_pilot3_states import (  # noqa: E402
    Pilot3MaterializationError,
    construct_pilot3_states,
)
from data.materialize_petct_pilot6_states import (  # noqa: E402
    DATASET_ID,
    GOALS,
    construct_pilot6_states,
    label_components_18,
)
from p2t.build_petct_matched_state_dataset import (  # noqa: E402
    _episode_id,
    _group_id,
    build_matched_state_six,
)


def _case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (64, 64, 11)
    gt = np.zeros(shape, dtype=np.uint8)
    gt[20:44, 20:44, 4:7] = 1
    return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32), gt


def _dense_largest_slice_simulator(mask, *, strategy, seed):
    del strategy, seed
    z = max(range(mask.shape[2]), key=lambda index: int(mask[:, :, index].sum()))
    coordinates = np.argwhere(mask[:, :, z] > 0)
    # v2 (D-2026-08-16-01): the scribble is a central patch, not the whole
    # slice, so Euclidean far-voxels (beyond 15 mm) exist and the COMPLETE
    # states remain constructible.
    if len(coordinates):
        center = coordinates.mean(axis=0)
        radius = 5.0
        coordinates = coordinates[
            np.linalg.norm(coordinates - center, axis=1) <= radius
        ]
    result = [[int(x), int(y), int(z)] for x, y in coordinates]
    return result, True, len(result)


def test_pilot6_uses_binary_topology_and_closes_all_six_classes() -> None:
    pet, ct, gt = _case()
    result = construct_pilot6_states(
        pet, ct, gt, split="development", dataset_id=DATASET_ID
    )
    assert result["eligible"] is True
    assert tuple(result["states"]) == GOALS
    assert result["receipt"]["class_coverage_complete"] is True
    assert result["receipt"]["instance_identity_source"] == (
        "BINARY_MASK_18_CONNECTIVITY_NOT_INSTANCE_IDS"
    )
    for goal, state in result["states"].items():
        operation = goal.split("_", 1)[0]
        m0 = state["m0"] > 0
        expected = (gt > 0) & ~m0 if operation == "ADD" else m0 & ~(gt > 0)
        assert np.array_equal(state["operation_residual"] > 0, expected)
    _, gt_components = label_components_18(gt)
    assert gt_components == 1


def test_official_cues_precede_intent_and_rederive_exact_six_topologies() -> None:
    pet, ct, gt = _case()
    result = build_matched_state_six(
        pet,
        ct,
        gt,
        spacing_xy=(2.0, 2.0),
        strategy="random",
        simulator=_dense_largest_slice_simulator,
        generation={"official_commit": "abc", "seed": 42},
        local_radius_mm=15.0,
        minimum_local_area_mm2=50.0,
    )
    assert result["eligible"] is True
    assert tuple(result["states"]) == GOALS
    assert result["receipt"]["stage_order"].index(
        "official_autopetv_operation_specific_scribble_on_shared_residual_support"
    ) < result["receipt"]["stage_order"].index("canonical_intent_rendering")
    assert result["scribbles"]["ADD"]["polarity"] == "foreground"
    assert result["scribbles"]["REMOVE"]["polarity"] == "background"
    for goal, state in result["states"].items():
        operation, target, scope = goal.split("_")
        assert state["operation"] == operation
        assert state["target_stats"]["operation"] == operation
        assert state["target_stats"]["target"] == target
        assert state["gold_intent"]["scope"] == scope


def test_legacy_pilot3_is_provenance_only_and_fails_closed() -> None:
    pet, ct, gt = _case()
    with pytest.raises(Pilot3MaterializationError, match="legacy provenance"):
        construct_pilot3_states(
            pet, ct, gt, split="development", dataset_id=DATASET_ID
        )


def test_v2_ids_are_deterministic_operation_distinct_and_not_v1_collisions() -> None:
    add = opaque_episode_id("case-a", "ADD_SAME_LOCAL", "random")
    remove = opaque_episode_id("case-a", "REMOVE_SAME_LOCAL", "random")
    assert add == opaque_episode_id("case-a", "ADD_SAME_LOCAL", "random")
    assert add != remove
    legacy = "petct-" + hashlib.sha256(
        b"PETCT-EPISODE-v1|case-a|ADD_SAME_LOCAL|random"
    ).hexdigest()[:24]
    assert add != legacy
    assert scribble_attempt_id("controlled_p2t", "case-a", "ADD", "random") != (
        scribble_attempt_id("controlled_p2t", "case-a", "REMOVE", "random")
    )
    group = _group_id("case-a", "random")
    assert group == _group_id("case-a", "random")
    assert len({_episode_id(group, goal) for goal in GOALS}) == len(GOALS)
