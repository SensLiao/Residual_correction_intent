from __future__ import annotations

import numpy as np
import pytest

from scripts.data.build_petct_legacy_m0_d3_corpus import (
    STRATEGY_ORDER,
    annotate_geometry_similarity,
    attach_geometry_audit,
    d3_episode_id,
    fixed_authorized_slice,
)


def _record(strategy: str, coordinates: list[list[int]]) -> dict:
    return {"strategy": strategy, "coordinates_xyz": coordinates}


def test_d3_episode_id_is_deterministic_and_strategy_specific() -> None:
    first = d3_episode_id("case::r2", "centerline")
    assert first == d3_episode_id("case::r2", "centerline")
    assert first != d3_episode_id("case::r2", "random")
    assert first.startswith("petct-legacy-d3-")


def test_fixed_authorized_slice_preserves_only_frozen_plane() -> None:
    authorized = np.zeros((5, 6, 4), dtype=np.uint8)
    authorized[1:3, 2:5, 1] = 1
    authorized[0:2, 0:2, 2] = 1

    frozen = fixed_authorized_slice(authorized, center_z=1)

    assert frozen[:, :, 1].sum() == 6
    assert frozen[:, :, 2].sum() == 0
    assert frozen.sum() == 6


def test_fixed_authorized_slice_rejects_empty_frozen_plane() -> None:
    authorized = np.zeros((5, 6, 4), dtype=np.uint8)
    authorized[1, 1, 2] = 1
    with pytest.raises(RuntimeError, match="empty on frozen center_z"):
        fixed_authorized_slice(authorized, center_z=1)


def test_similarity_flags_exact_and_near_duplicates_without_dropping_rows() -> None:
    records = [
        _record("centerline", [[1, 1, 3], [1, 2, 3], [1, 3, 3]]),
        _record("random", [[1, 1, 3], [1, 2, 3], [1, 3, 3]]),
        _record("boundary", [[1, 1, 3], [1, 2, 3], [2, 3, 3]]),
    ]

    audit = annotate_geometry_similarity(records, near_duplicate_jaccard=0.49)

    assert len(records) == len(STRATEGY_ORDER)
    assert audit["exact_pairs"] == [["centerline", "random"]]
    assert len(audit["near_pairs"]) == 2
    assert audit["distinct_geometry_count"] == 2
    assert records[0]["validity_flags"]["geometry_exact_duplicate_of"] == [
        "random"
    ]
    assert records[2]["validity_flags"]["geometry_near_duplicate_of"] == [
        "centerline",
        "random",
    ]


def test_similarity_rejects_duplicate_strategy_labels() -> None:
    records = [
        _record("centerline", [[1, 1, 3]]),
        _record("centerline", [[2, 2, 3]]),
    ]
    with pytest.raises(RuntimeError, match="duplicate strategy labels"):
        annotate_geometry_similarity(records, near_duplicate_jaccard=0.8)


def test_geometry_audit_is_preserved_in_scribble_generation_metadata() -> None:
    records = [
        {
            **_record("centerline", [[1, 1, 3]]),
            "scribble_generation": {},
        },
        {
            **_record("random", [[2, 2, 3]]),
            "scribble_generation": {},
        },
    ]
    similarity = annotate_geometry_similarity(
        records, near_duplicate_jaccard=0.8
    )

    attach_geometry_audit(records, similarity)

    assert records[0]["scribble_generation"]["validity_flags"] == records[0][
        "validity_flags"
    ]
    assert (
        records[1]["scribble_generation"]["triplet_geometry_similarity"]
        == similarity
    )
