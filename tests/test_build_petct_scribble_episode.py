from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from common.petct_route_a_core import LEGAL_JOINT_GOALS  # noqa: E402
from data.build_petct_scribble_episode import (  # noqa: E402
    CUE_ELIGIBILITY_RULE,
    EpisodeContractError,
    ResidualCueIneligibleError,
    build_episode_documents,
    canonical_intent_frame,
    compute_fn_residual,
    compute_fp_residual,
    generate_residual_scribble,
    official_simulator_provenance,
    residual_component_census,
    resolve_m0_provenance,
    publish_episode_documents,
)


def _simulator(mask, *, strategy, seed):
    del strategy, seed
    z = max(range(mask.shape[2]), key=lambda index: int(mask[:, :, index].sum()))
    coords = np.argwhere(mask[:, :, z] > 0)
    result = [[int(x), int(y), int(z)] for x, y in coords]
    return result, True, len(result)


def _recording_simulator(seen: list[np.ndarray]):
    """Capture the exact mask the pinned simulator would have been handed."""

    def simulator(mask, *, strategy, seed):
        seen.append(np.asarray(mask).copy())
        return _simulator(mask, strategy=strategy, seed=seed)

    return simulator


def test_fn_and_fp_residuals_are_exact_binary_set_differences() -> None:
    gt = np.zeros((8, 8, 2), dtype=np.uint8)
    m0 = np.zeros_like(gt)
    gt[1:4, 1:4, 0] = 1
    m0[2:6, 2:6, 0] = 1
    assert np.array_equal(compute_fn_residual(gt, m0), (gt > 0) & ~(m0 > 0))
    assert np.array_equal(compute_fp_residual(gt, m0), (m0 > 0) & ~(gt > 0))


@pytest.mark.parametrize(
    ("operation", "polarity", "residual_kind"),
    (("ADD", "foreground", "FN"), ("REMOVE", "background", "FP")),
)
def test_official_residual_cue_preserves_operation_polarity(
    operation: str, polarity: str, residual_kind: str
) -> None:
    residual = np.zeros((12, 12, 3), dtype=np.uint8)
    residual[2:8, 3:9, 1] = 1
    record = generate_residual_scribble(
        residual,
        operation=operation,
        strategy="random",
        simulator=_simulator,
        upstream_commit="abc",
        seed=42,
    )
    assert record["contract_version"] == "PETCT-RESIDUAL-CUE-v2.0"
    assert record["operation"] == operation
    assert record["polarity"] == polarity
    assert record["residual_kind"] == residual_kind


def test_canonical_renderer_has_exact_six_goals_and_rejects_legacy() -> None:
    frames = [canonical_intent_frame(goal) for goal in LEGAL_JOINT_GOALS]
    assert [frame["goal"] for frame in frames] == list(LEGAL_JOINT_GOALS)
    assert all(frame["schema_version"] == "PETCT-INTENT-v2.0" for frame in frames)
    with pytest.raises(EpisodeContractError):
        canonical_intent_frame("SAME_LOCAL")
    with pytest.raises(EpisodeContractError):
        canonical_intent_frame("ADD_NEW_LOCAL")


def test_episode_document_rejects_cue_operation_mismatch() -> None:
    scribble = generate_residual_scribble(
        np.pad(np.ones((4, 4, 1), dtype=np.uint8), ((1, 1), (1, 1), (0, 0))),
        operation="ADD",
        strategy="random",
        simulator=_simulator,
        upstream_commit="abc",
    )
    with pytest.raises(EpisodeContractError, match="polarity"):
        build_episode_documents(
            episode_id="ep-abcdef",
            lane="controlled",
            patient_group_hash="a" * 64,
            montage_reference="visible.npz",
            m0_provenance={"kind": "controlled"},
            scribble_record=scribble,
            source_case_id="case",
            source_patient_id="patient",
            residual_sha256="b" * 64,
            residual_voxels=16,
            gold_intent=canonical_intent_frame("REMOVE_SAME_LOCAL"),
        )


@pytest.mark.parametrize("operation", ["UPDATE", "", "ADD_ONLY"])
def test_invalid_operation_fails_before_simulator(operation: str) -> None:
    with pytest.raises(EpisodeContractError, match="operation"):
        generate_residual_scribble(
            np.ones((3, 3, 1), dtype=np.uint8),
            operation=operation,
            strategy="random",
            simulator=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("simulator must not run")
            ),
            upstream_commit="abc",
        )


def test_empty_residual_and_invalid_strategy_fail_closed() -> None:
    with pytest.raises(EpisodeContractError, match="non-empty"):
        generate_residual_scribble(
            np.zeros((3, 3, 1), dtype=np.uint8),
            operation="ADD",
            strategy="random",
            simulator=_simulator,
            upstream_commit="abc",
        )
    with pytest.raises(EpisodeContractError, match="strategy"):
        generate_residual_scribble(
            np.ones((3, 3, 1), dtype=np.uint8),
            operation="REMOVE",
            strategy="freehand",
            simulator=_simulator,
            upstream_commit="abc",
        )


@pytest.mark.parametrize(
    "simulator",
    [
        lambda mask, **kwargs: ([], True, 0),
        lambda mask, **kwargs: ([[99, 99, 0]], True, 1),
        lambda mask, **kwargs: ([[0, 0, 0]], False, 1),
        lambda mask, **kwargs: ([[0, 0, 0]], True, 2),
        lambda mask, **kwargs: ([],),
    ],
)
def test_malformed_official_outputs_are_never_repaired(simulator) -> None:
    residual = np.ones((3, 3, 1), dtype=np.uint8)
    with pytest.raises(EpisodeContractError):
        generate_residual_scribble(
            residual,
            operation="ADD",
            strategy="random",
            simulator=simulator,
            upstream_commit="abc",
        )


def test_visible_and_evaluation_outputs_are_disjoint_and_no_clobber(tmp_path: Path) -> None:
    scribble = generate_residual_scribble(
        np.ones((4, 4, 1), dtype=np.uint8),
        operation="ADD",
        strategy="random",
        simulator=_simulator,
        upstream_commit="abc",
    )
    visible, evaluation = build_episode_documents(
        episode_id="ep-abcdef",
        lane="controlled",
        patient_group_hash="a" * 64,
        montage_reference="visible.npz",
        m0_provenance={"kind": "controlled"},
        scribble_record=scribble,
        source_case_id="hidden-case",
        source_patient_id="hidden-patient",
        residual_sha256="b" * 64,
        residual_voxels=16,
        gold_intent=canonical_intent_frame("ADD_SAME_LOCAL"),
    )
    assert "gold_intent" not in visible
    assert evaluation["gold_intent"]["operation"] == "ADD"
    with pytest.raises(EpisodeContractError, match="disjoint"):
        publish_episode_documents(
            visible,
            evaluation,
            visible_root=tmp_path / "same",
            eval_root=tmp_path / "same" / "nested",
        )
    receipt = publish_episode_documents(
        visible,
        evaluation,
        visible_root=tmp_path / "visible",
        eval_root=tmp_path / "evaluation",
    )
    assert len(receipt["visible_sha256"]) == 64
    assert len(receipt["eval_sha256"]) == 64
    with pytest.raises(FileExistsError):
        publish_episode_documents(
            visible,
            evaluation,
            visible_root=tmp_path / "visible",
            eval_root=tmp_path / "evaluation",
        )


def test_official_source_rejects_symlink_before_hash_read(tmp_path: Path) -> None:
    target = tmp_path / "simulate_scribbles.py"
    target.write_text("pass\n", encoding="utf-8")
    link = tmp_path / "link.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(EpisodeContractError, match="symlink"):
        official_simulator_provenance(link)


def _residual(shape=(20, 20, 5)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def test_component_below_minimum_best_slice_pixels_is_ineligible() -> None:
    residual = _residual()
    residual[2:4, 2:4, 1] = 1  # 2x2 == 4 pixels on its only slice
    eligible, census = residual_component_census(
        residual, minimum_best_slice_pixels=5
    )
    assert not eligible.any()
    assert census["rule"] == CUE_ELIGIBILITY_RULE
    assert census["component_total"] == 1
    assert census["component_eligible"] == 0
    assert census["component_excluded"] == 1
    assert census["max_excluded_best_slice_pixels"] == 4
    with pytest.raises(ResidualCueIneligibleError) as error:
        generate_residual_scribble(
            residual,
            operation="ADD",
            strategy="random",
            simulator=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("simulator must not run on an ineligible residual")
            ),
            upstream_commit="abc",
            minimum_best_slice_pixels=5,
        )
    assert error.value.census["component_excluded"] == 1


def test_component_at_exactly_minimum_best_slice_pixels_is_admitted() -> None:
    residual = _residual()
    residual[2:7, 3, 1] = 1  # a 5-pixel line: exactly at the threshold
    eligible, census = residual_component_census(
        residual, minimum_best_slice_pixels=5
    )
    assert census["component_eligible"] == 1
    assert census["component_excluded"] == 0
    assert int(eligible.sum()) == 5
    record = generate_residual_scribble(
        residual,
        operation="ADD",
        strategy="random",
        simulator=_simulator,
        upstream_commit="abc",
        minimum_best_slice_pixels=5,
    )
    assert record["source_component_area"] == 5
    assert record["cue_eligibility"]["minimum_best_slice_pixels"] == 5
    assert record["coordinate_count"] == 5


def test_ineligible_fragments_never_reach_the_official_simulator() -> None:
    residual = _residual()
    residual[2:5, 2:5, 1] = 1  # 9-pixel eligible component
    residual[10, 10, 0] = 1  # pixel-scale fragments on three slices
    residual[13, 13, 1] = 1
    residual[16, 16, 4] = 1
    seen: list[np.ndarray] = []
    record = generate_residual_scribble(
        residual,
        operation="REMOVE",
        strategy="random",
        simulator=_recording_simulator(seen),
        upstream_commit="abc",
        minimum_best_slice_pixels=5,
    )
    expected = np.zeros_like(residual)
    expected[2:5, 2:5, 1] = 1
    assert len(seen) == 1
    assert np.array_equal(seen[0] > 0, expected > 0)
    assert record["cue_eligibility"]["component_total"] == 4
    assert record["cue_eligibility"]["component_excluded"] == 3
    assert record["cue_eligibility"]["max_excluded_best_slice_pixels"] == 1
    # The mining residual stays the full FP set; only the cue candidate shrinks.
    assert record["residual_voxels"] == 12
    assert record["cue_eligibility"]["eligible_voxels"] == 9


@pytest.mark.parametrize("stray_slice", [0, 2])
def test_scribble_outside_an_eligible_component_still_fails_closed(
    stray_slice: int,
) -> None:
    residual = _residual()
    residual[1:4, 1:4, 0] = 1  # 9 pixels: the component that will be selected
    residual[10:15, 10, stray_slice] = 1  # a second ELIGIBLE 5-pixel component
    stray = [[10, 10, stray_slice]]
    with pytest.raises(EpisodeContractError) as error:
        generate_residual_scribble(
            residual,
            operation="ADD",
            strategy="random",
            simulator=lambda mask, **kwargs: (stray, True, 1),
            upstream_commit="abc",
            minimum_best_slice_pixels=5,
        )
    assert "outside selected source component" in str(error.value)
    # It must stay a hard contract breach, never a counted eligibility exclusion.
    assert not isinstance(error.value, ResidualCueIneligibleError)


def test_default_threshold_leaves_the_residual_untouched() -> None:
    residual = _residual()
    residual[2:4, 2:4, 1] = 1
    residual[10, 10, 0] = 1
    eligible, census = residual_component_census(
        residual, minimum_best_slice_pixels=1
    )
    assert census["enforced"] is False
    assert np.array_equal(eligible, residual > 0)


@pytest.mark.parametrize("threshold", [0, -1, 2.5, True, "5", None])
def test_non_positive_integer_thresholds_fail_closed(threshold) -> None:
    residual = _residual()
    residual[2:7, 3, 1] = 1
    with pytest.raises(EpisodeContractError, match="minimum_best_slice_pixels"):
        residual_component_census(
            residual, minimum_best_slice_pixels=threshold
        )


def test_natural_m0_provenance_cannot_be_caller_supplied() -> None:
    with pytest.raises(EpisodeContractError, match="refuses"):
        resolve_m0_provenance(
            lane="natural",
            provenance_json='{"kind":"forged"}',
            oof_ready=None,
            case_id="case",
            patient_id="patient",
            m0_path=Path("m0.nii.gz"),
        )
