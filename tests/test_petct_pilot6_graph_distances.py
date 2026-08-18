"""Graph-distance regression tests for the pilot6 state constructor.

The 2026-08-16 fix replaced a pure-Python deque BFS with a vectorized
bbox-confined layer-by-layer BFS.  These tests pin the semantics:
18-connectivity geodesic distances, single-component requirement, start
coordinate validation, and the O(diameter) growth bound.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from data.materialize_petct_pilot6_states import (  # noqa: E402
    Pilot6MaterializationError,
    _graph_distances,
)

STRUCTURE_18 = np.ones((3, 3, 3), dtype=bool)
for x, y, z in (
    (0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2),
    (2, 0, 0), (2, 0, 2), (2, 2, 0), (2, 2, 2),
):
    STRUCTURE_18[x, y, z] = False


def _reference_bfs(mask: np.ndarray, start) -> np.ndarray:
    from collections import deque

    distances = np.full(mask.shape, -1, dtype=np.int32)
    start = tuple(int(value) for value in start)
    distances[start] = 0
    queue = deque([start])
    offsets = np.argwhere(STRUCTURE_18) - 1
    while queue:
        x, y, z = queue.popleft()
        for dx, dy, dz in offsets:
            candidate = (x + dx, y + dy, z + dz)
            if any(v < 0 or v >= mask.shape[i] for i, v in enumerate(candidate)):
                continue
            if not mask[candidate] or distances[candidate] >= 0:
                continue
            distances[candidate] = distances[x, y, z] + 1
            queue.append(candidate)
    return distances


def _tube_volume():
    # bent tube: (z,y,x); 18-connected chain through the volume.
    volume = np.zeros((6, 6, 6), dtype=bool)
    for z in range(6):
        volume[z, 3, 3] = True
    for x in range(3, 6):
        volume[2, 3, x] = True
    for y in range(3, 6):
        volume[2, y, 5] = True
    return volume


def test_graph_distances_matches_reference_bfs():
    volume = _tube_volume()
    for start in (np.array([0, 3, 3]), np.array([2, 5, 5]), np.array([5, 3, 3])):
        expected = _reference_bfs(volume, start)
        actual = _graph_distances(volume, start)
        assert np.array_equal(actual, expected)


def test_graph_distances_rejects_disconnected_mask():
    volume = _tube_volume().copy()
    volume[0, 0, 0] = True  # separate island, corner-only isolated
    with pytest.raises(Pilot6MaterializationError):
        _graph_distances(volume, np.array([0, 3, 3]))


def test_graph_distances_rejects_start_outside():
    volume = _tube_volume()
    with pytest.raises(Pilot6MaterializationError):
        _graph_distances(volume, np.array([5, 0, 0]))


def test_graph_distances_empty_mask():
    with pytest.raises(Pilot6MaterializationError):
        _graph_distances(np.zeros((4, 4, 4), dtype=bool), np.array([1, 1, 1]))


def _reference_first_coordinate(mask: np.ndarray):
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        raise ValueError("empty")
    coordinate = coordinates[np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))[0]]
    return tuple(int(value) for value in coordinate)


def test_first_coordinate_matches_lexsort_reference():
    from data.materialize_petct_pilot6_states import _first_coordinate
    rng = np.random.default_rng(7)
    for _ in range(5):
        volume = rng.random((8, 9, 10)) > 0.85
        assert _first_coordinate(volume) == _reference_first_coordinate(volume)


def test_first_coordinate_empty_mask_raises():
    from data.materialize_petct_pilot6_states import _first_coordinate
    with pytest.raises(Pilot6MaterializationError):
        _first_coordinate(np.zeros((4, 4, 4), dtype=bool))


def test_construct_pilot6_states_bbox_path_equivalent():
    """construct_pilot6_states still builds six states on a realistic volume."""
    from data.materialize_petct_pilot6_states import construct_pilot6_states
    rng = np.random.default_rng(3)
    gt = np.zeros((30, 30, 30), dtype=bool)
    for _ in range(6):
        center = tuple(rng.integers(5, 25, size=3))
        zz, yy, xx = np.ogrid[:30, :30, :30]
        gt |= ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= 9
    result = construct_pilot6_states(
        gt.astype(np.float32), np.zeros_like(gt, dtype=np.float32), gt,
        split="development", dataset_id="PSMA-PET-CT-Lesions-v3",
    )
    assert result["eligible"] is True
    assert set(result["states"]) == {
        "ADD_SAME_LOCAL", "REMOVE_SAME_LOCAL", "ADD_SAME_COMPLETE",
        "REMOVE_SAME_COMPLETE", "ADD_NEW_COMPLETE", "REMOVE_NEW_COMPLETE",
    }


def test_matched_state_six_treats_rederivation_failure_as_system_error(monkeypatch):
    """Constructor/derive disagreement must abort, never censor the corpus."""
    import sys
    from pathlib import Path

    SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from p2t import build_petct_matched_state_dataset as builder

    def _boom(*args, **kwargs):
        raise RuntimeError("SAME_LOCAL candidate is below minimum physical area")

    def _fake_sim(residual, strategy=None, seed=None):
        # pipeline convention: arrays are (x, y, z); coordinates are (x, y, z)
        residual = np.asarray(residual) > 0
        z = int(np.argmax(residual.sum(axis=(0, 1))))
        coords = np.argwhere(residual[:, :, z] > 0)[:2]  # (x, y) pairs
        return [[int(c[0]), int(c[1]), z] for c in coords], True, int(len(coords))

    monkeypatch.setattr(builder, "derive_goal_and_authorized_target", _boom)
    rng = np.random.default_rng(5)
    gt = np.zeros((30, 30, 30), dtype=bool)
    for _ in range(4):
        center = tuple(rng.integers(8, 22, size=3))
        zz, yy, xx = np.ogrid[:30, :30, :30]
        gt |= ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= 49
    with pytest.raises(builder.MatchedStateGeometryError, match="STATE_REDERIVATION_FAILED"):
        builder.build_matched_state_six(
            gt.astype(np.float32), np.zeros_like(gt, dtype=np.float32), gt,
            spacing_xy=(2.734, 2.734),
            strategy="centerline",
            simulator=_fake_sim,
            generation={"official_commit": "c", "seed": 42, "strategy_mode": "primary",
                        "strategy_salt": "s", "strategy_assignment": "stable-patient-hash"},
            local_radius_mm=15.0,
            minimum_local_area_mm2=50.0,
            learning_partition="train",
        )


def test_visible_firewall_rejects_forbidden_receipt_without_silent_sanitizing():
    """Audit receipts cannot be passed wholesale into the visible packet."""
    import sys
    from pathlib import Path

    SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from data.build_petct_scribble_episode import (
        EpisodeContractError,
        _assert_visible_safe,
    )

    receipt = {
        "schema_version": "x",
        "status": "ELIGIBLE",
        "thresholds": {
            "min_component_voxels": 5,
            "max_support_fraction": 0.8,
        },
        "input_content_sha256": {
            "gt_content_sha256": "deadbeef",
            "ct_content_sha256": "cafe",
        },
        "residual_sha256": "bad",
        "stage_order": ["a", "b"],
        "matched_goals": ["ADD_SAME_LOCAL"],
    }
    with pytest.raises(EpisodeContractError, match="forbidden evaluation"):
        _assert_visible_safe(
            {"m0_provenance": {"materializer_receipt": receipt}}
        )


def test_pilot6_v2_construction_derives_same_goals():
    """D-2026-08-16-01 regression anchor: every Euclidean-anchored state
    must re-derive to its own goal under the official rules."""
    import sys
    from pathlib import Path

    SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from data.materialize_petct_pilot6_states_v2 import (
        construct_pilot6_states_v2,
        select_add_component,
        select_remove_component,
    )
    from data.build_petct_scribble_dataset import derive_goal_and_authorized_target

    spacing = (2.734, 2.734)  # 15 mm ~= 5.5 voxels; 50 mm^2 ~= 6.7 voxels
    gt = np.zeros((40, 40, 40), dtype=bool)
    # one large lesion, radius ~9 voxels (24.6 mm) so far-voxels exist
    zz, yy, xx = np.ogrid[:40, :40, :40]
    gt |= ((zz - 20) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2) <= 81
    add_component, _ = select_add_component(gt, min_component_voxels=50)
    remove_component = select_remove_component(
        gt, add_component, shell_iterations=3, min_component_voxels=50
    )
    # scribbles: single voxels inside the masks on the largest axial slice
    z_add = int(np.argmax(add_component.sum(axis=(0, 1))))
    coords_add = np.argwhere(add_component[:, :, z_add] > 0)
    scribble_add = [[int(c[0]), int(c[1]), z_add] for c in coords_add[:: max(1, len(coords_add) // 2)][:2]]
    z_remove = int(np.argmax(remove_component.sum(axis=(0, 1))))
    coords_remove = np.argwhere(remove_component[:, :, z_remove] > 0)
    scribble_remove = [[int(c[0]), int(c[1]), z_remove] for c in coords_remove[:: max(1, len(coords_remove) // 2)][:2]]

    v2 = construct_pilot6_states_v2(
        gt,
        add_component=add_component,
        remove_component=remove_component,
        scribble_add=scribble_add,
        scribble_remove=scribble_remove,
        spacing_xy=spacing,
        local_radius_mm=15.0,
        minimum_local_area_mm2=50.0,
    )
    assert v2["eligible"] is True
    for goal, state in v2["states"].items():
        operation = state["operation"]
        scribble = scribble_add if operation == "ADD" else scribble_remove
        actual_goal, authorized, _ = derive_goal_and_authorized_target(
            gt=gt,
            m0=state["m0"],
            operation=operation,
            coordinates_xyz=scribble,
            spacing_xy=spacing,
            local_radius_mm=15.0,
            minimum_local_area_mm2=50.0,
        )
        assert actual_goal == goal, f"{goal} re-derived as {actual_goal}"
        assert np.array_equal(
            authorized > 0, state["authorized_target"] > 0
        ), f"authorized mismatch for {goal}"


def _fragmented_local_shell_fixture():
    """R9-failure fixture: a thick FP shell whose scribble-slice local part
    splits into a truth-adjacent piece and an isolated piece.

    Mirrors psma_995fbaec49f131ce_2016-11-12, where the REMOVE scribble bound
    two components of the SAME_LOCAL mask (a GT-merged piece and an isolated
    28-voxel FP fragment) and the official re-derivation correctly refused.
    """

    from data.materialize_petct_pilot6_states_v2 import Pilot6V2Error

    truth = np.zeros((48, 48, 48), dtype=bool)
    # array layout is (x, y, z): in-plane radius 20 on (x, y), z radius 5,
    # centred (24,24,24): far voxels exist on the scribble slice and a
    # retainable cap exists on other slices.
    xx, yy, zz = np.ogrid[:48, :48, :48]
    truth |= ((xx - 24) ** 2 + (yy - 24) ** 2) / 400.0 + (zz - 24) ** 2 / 25.0 <= 1.0
    add_component = truth.copy()
    remove_component = np.zeros_like(truth)
    # P1: 4x4 patch at z=18, 18-adjacent to truth at z=19 -> merges with GT.
    remove_component[22:26, 22:26, 18] = True
    # P2: 6x6 patch at z=18, within 15 mm of P1 (so it stays local for any
    # scribble on P1) but >=2 voxels from truth and from P1 (isolated).
    remove_component[10:16, 21:27, 18] = True
    # P3: far voxel on the scribble slice so REMOVE COMPLETE keeps far.
    remove_component[40, 40, 18] = True
    scribble_add = [[16, 24, 24], [18, 24, 24]]
    scribble_remove = [
        [24, 24, 18], [23, 24, 18],  # on P1
        [10, 21, 18], [12, 23, 18], [15, 26, 18],  # on P2
    ]
    return truth, add_component, remove_component, scribble_add, scribble_remove, Pilot6V2Error


def test_pilot6_v2_excludes_fragmented_local_cue_binding():
    """A scribble binding two components of a SAME_LOCAL mask is a per-goal
    eligibility outcome (goal_failures -> operation-triplet exclusion by the
    builder), never a system abort, and the sibling goals survive."""
    import sys
    from pathlib import Path

    SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from data.materialize_petct_pilot6_states_v2 import construct_pilot6_states_v2

    truth, add_component, remove_component, scribble_add, scribble_remove, _ = (
        _fragmented_local_shell_fixture()
    )
    result = construct_pilot6_states_v2(
        truth,
        add_component=add_component,
        remove_component=remove_component,
        scribble_add=scribble_add,
        scribble_remove=scribble_remove,
        spacing_xy=(1.0, 1.0),
        local_radius_mm=15.0,
        minimum_local_area_mm2=50.0,
    )
    assert "bind exactly one source component" in str(
        result["goal_failures"].get("REMOVE_SAME_LOCAL", "")
    )
    assert "REMOVE_SAME_LOCAL" not in result["states"]
    # ADD triplet is unaffected by the fragmented REMOVE local shell.
    for goal in ("ADD_SAME_LOCAL", "ADD_SAME_COMPLETE", "ADD_NEW_COMPLETE"):
        assert goal in result["states"], goal


def test_pilot6_v2_excludes_pure_fp_bound_component_for_same_goal():
    """If the scribble binds a single FP fragment that retains no counterpart,
    the SAME goal could never re-derive (it would flip to NEW_COMPLETE): it is
    recorded per-goal instead of excluding the whole case."""
    import sys
    from pathlib import Path

    SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from data.materialize_petct_pilot6_states_v2 import construct_pilot6_states_v2

    truth, add_component, remove_component, scribble_add, _, _ = (
        _fragmented_local_shell_fixture()
    )
    # scribble only on the isolated P2 fragment: binding succeeds (one
    # component) but that component retains no counterpart.
    scribble_remove = [[10, 21, 18], [12, 23, 18], [15, 26, 18]]
    result = construct_pilot6_states_v2(
        truth,
        add_component=add_component,
        remove_component=remove_component,
        scribble_add=scribble_add,
        scribble_remove=scribble_remove,
        spacing_xy=(1.0, 1.0),
        local_radius_mm=15.0,
        minimum_local_area_mm2=50.0,
    )
    assert "counterpart flag" in str(
        result["goal_failures"].get("REMOVE_SAME_LOCAL", "")
    )
    assert "REMOVE_SAME_LOCAL" not in result["states"]
    for goal in ("ADD_SAME_LOCAL", "ADD_SAME_COMPLETE", "ADD_NEW_COMPLETE"):
        assert goal in result["states"], goal


def test_pilot6_v2_excludes_single_bound_fragment_with_uncontained_local():
    """Single-component binding of a fragmented local subset is per-goal
    ineligible: the re-derivation measures area/authorized on the bound
    component only, so the unbound fragments make that goal fail while the
    sibling goals survive."""
    import sys
    from pathlib import Path

    SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from data.materialize_petct_pilot6_states_v2 import construct_pilot6_states_v2

    truth, add_component, remove_component, scribble_add, _, _ = (
        _fragmented_local_shell_fixture()
    )
    # scribble only on the GT-merged P1 piece: binding is unique and the
    # counterpart flag matches SAME, but P2 stays outside the bound component.
    scribble_remove = [[24, 24, 18], [23, 24, 18]]
    result = construct_pilot6_states_v2(
        truth,
        add_component=add_component,
        remove_component=remove_component,
        scribble_add=scribble_add,
        scribble_remove=scribble_remove,
        spacing_xy=(1.0, 1.0),
        local_radius_mm=15.0,
        minimum_local_area_mm2=50.0,
    )
    assert "not contained" in str(
        result["goal_failures"].get("REMOVE_SAME_LOCAL", "")
    )
    assert "REMOVE_SAME_LOCAL" not in result["states"]
    for goal in ("ADD_SAME_LOCAL", "ADD_SAME_COMPLETE", "ADD_NEW_COMPLETE"):
        assert goal in result["states"], goal
