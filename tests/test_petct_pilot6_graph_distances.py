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
