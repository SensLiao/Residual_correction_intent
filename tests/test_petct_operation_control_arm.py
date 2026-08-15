"""Regression guard for the `operation_control` P2T input arm.

The four frozen arms each leave at least one ADD/REMOVE leak path open, so the
`operation` axis had no control group at all:

* `no_M0` zeroes the state but keeps the FG/BG cue identity (channels 15/16).
* `polarity_blind` merges the cue identity but keeps M0 (channels 10..14), and the
  gold construction forces ``scribble subset (GT \\ M0)`` for ADD and
  ``scribble subset (M0 \\ GT)`` for REMOVE, so containment still names the operation.
* `geometry_only` keeps both.

`operation_control` closes both paths at once.  It is an *added* arm; the four frozen
arms keep their exact semantics so already-trained checkpoints stay meaningful.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "common"))

from petct_models import build_p2t_model  # noqa: E402
from petct_learning import (  # noqa: E402
    LearningContractError,
    apply_input_ablation,
    load_p2t_evaluation_contract,
)


SIZE = 48
CENTER_M0_CHANNEL = 12  # axial_stack(radius=2) puts the scribble's own slice at 10 + 2
CUE_FG_CHANNEL = 15
CUE_BG_CHANNEL = 16


def _disk(center: tuple[float, float], radius: float) -> np.ndarray:
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    return ((xs - center[0]) ** 2 + (ys - center[1]) ** 2 <= radius**2).astype(np.float32)


def _contract_legal_episode(operation: str, rng: np.random.Generator) -> np.ndarray:
    """Build a [17,H,W] visible tensor that obeys the frozen gold construction.

    Mirrors materialize_petct_learning_tensors.py:277-284 — the authorized residual is
    ``GT \\ M0`` for ADD and ``M0 \\ GT`` for REMOVE, and the scribble lies inside it.
    """

    m0_center = (rng.uniform(18, 30), rng.uniform(18, 30))
    gt_center = (m0_center[0] + rng.uniform(3, 7), m0_center[1] + rng.uniform(-7, 7))
    m0 = _disk(m0_center, 8.0)
    gt = _disk(gt_center, 8.0)
    if operation == "ADD":
        authorized = ((gt > 0) & ~(m0 > 0)).astype(np.float32)
    else:
        authorized = ((m0 > 0) & ~(gt > 0)).astype(np.float32)
    if not authorized.any():
        raise RuntimeError("degenerate synthetic episode")

    candidates = np.argwhere(authorized > 0)
    chosen = candidates[
        rng.choice(len(candidates), size=min(10, len(candidates)), replace=False)
    ]
    scribble = np.zeros_like(authorized)
    scribble[chosen[:, 0], chosen[:, 1]] = 1.0

    # The frozen loader re-asserts exactly these predicates (petct_learning.py:727-754).
    assert not np.any(scribble > authorized)
    if operation == "ADD":
        assert not np.any(authorized > gt) and not np.any(authorized * m0)
    else:
        assert not np.any(authorized > m0) and not np.any(authorized * gt)

    cue_fg = scribble if operation == "ADD" else np.zeros_like(scribble)
    cue_bg = scribble if operation == "REMOVE" else np.zeros_like(scribble)
    m0_stack = np.stack(
        [_disk(m0_center, 8.0 + offset) for offset in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    )
    m0_stack[2] = m0
    return np.concatenate(
        [
            rng.normal(size=(5, SIZE, SIZE)).astype(np.float32),  # PET5
            rng.normal(size=(5, SIZE, SIZE)).astype(np.float32),  # CT5
            m0_stack,
            cue_fg[None],
            cue_bg[None],
        ],
        axis=0,
    ).astype(np.float32)


def _read_operation_from_m0_containment(visual: np.ndarray) -> str:
    """Label-free attacker: cue support inside M0 means REMOVE, outside means ADD."""

    support = (visual[CUE_FG_CHANNEL] > 0) | (visual[CUE_BG_CHANNEL] > 0)
    if not support.any():
        return "UNDECIDABLE"
    return "REMOVE" if np.any(support & (visual[CENTER_M0_CHANNEL] > 0)) else "ADD"


def _read_operation_from_channel_identity(visual: np.ndarray) -> str:
    """Label-free attacker: which of the two cue channels carries the scribble."""

    fg = bool((visual[CUE_FG_CHANNEL] > 0).any())
    bg = bool((visual[CUE_BG_CHANNEL] > 0).any())
    if fg == bg:
        return "UNDECIDABLE"
    return "ADD" if fg else "REMOVE"


def _balanced_episodes(count: int = 40) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(20260806)
    return [
        (
            "ADD" if index % 2 == 0 else "REMOVE",
            _contract_legal_episode("ADD" if index % 2 == 0 else "REMOVE", rng),
        )
        for index in range(count)
    ]


def _leaks(arm: str, attacker, episodes: list[tuple[str, np.ndarray]]) -> bool:
    """Does the arm let this label-free attacker's output depend on the gold operation?

    Accuracy is the wrong probe: once M0 is zeroed the containment attacker answers a
    constant "ADD", which still scores 0.5 on a balanced set without carrying any
    information.  Leakage is exactly "prediction distribution differs between the ADD
    and the REMOVE episodes", so that is what we test.
    """

    grouped: dict[str, list[str]] = {"ADD": [], "REMOVE": []}
    for operation, visual in episodes:
        grouped[operation].append(attacker(apply_input_ablation(visual, arm)))
    return sorted(grouped["ADD"]) != sorted(grouped["REMOVE"])


def test_operation_control_zeroes_M0_and_merges_cue_polarity() -> None:
    rng = np.random.default_rng(7)
    visual = _contract_legal_episode("REMOVE", rng)

    ablated = apply_input_ablation(visual, "operation_control")

    assert np.all(ablated[10:15] == 0), "M0 leak path is still open"
    assert np.array_equal(
        ablated[CUE_FG_CHANNEL], ablated[CUE_BG_CHANNEL]
    ), "cue polarity leak path is still open"
    # The merged support must still be the union, so the prompt is not destroyed.
    assert np.array_equal(
        ablated[CUE_FG_CHANNEL],
        np.maximum(visual[CUE_FG_CHANNEL], visual[CUE_BG_CHANNEL]),
    )
    # PET/CT are untouched — this arm removes operation evidence, not imaging.
    assert np.array_equal(ablated[0:10], visual[0:10])


def test_operation_control_blinds_both_label_free_attackers() -> None:
    episodes = _balanced_episodes()

    assert not _leaks(
        "operation_control", _read_operation_from_m0_containment, episodes
    ), "M0 containment still names the operation"
    assert not _leaks(
        "operation_control", _read_operation_from_channel_identity, episodes
    ), "cue channel identity still names the operation"


def test_each_frozen_arm_leaves_at_least_one_operation_leak_open() -> None:
    """The defect this control arm fixes: no frozen arm closes both paths."""

    episodes = _balanced_episodes()
    leaks = {
        arm: (
            _leaks(arm, _read_operation_from_m0_containment, episodes),
            _leaks(arm, _read_operation_from_channel_identity, episodes),
        )
        for arm in ("full", "no_M0", "polarity_blind", "geometry_only")
    }

    # `polarity_blind` shut the channel-identity path but M0 containment is intact.
    assert leaks["polarity_blind"] == (True, False)
    # `no_M0` shut the containment path but the channel identity is intact.
    assert leaks["no_M0"] == (False, True)
    # `full` and `geometry_only` leak through both.
    assert leaks["full"] == (True, True)
    assert leaks["geometry_only"] == (True, True)
    assert all(any(paths) for paths in leaks.values()), (
        "a frozen arm already closed both paths — the control arm would be redundant"
    )


def test_frozen_arms_keep_their_trained_semantics() -> None:
    """Already-trained checkpoints stay meaningful — the four arms are add-only frozen."""

    rng = np.random.default_rng(11)
    visual = _contract_legal_episode("ADD", rng)

    assert np.array_equal(apply_input_ablation(visual, "full"), visual)

    no_m0 = apply_input_ablation(visual, "no_M0")
    assert np.all(no_m0[10:15] == 0)
    assert np.array_equal(no_m0[15], visual[15])
    assert np.array_equal(no_m0[16], visual[16])

    polarity_blind = apply_input_ablation(visual, "polarity_blind")
    assert np.array_equal(polarity_blind[10:15], visual[10:15])
    assert np.array_equal(polarity_blind[15], polarity_blind[16])

    geometry_only = apply_input_ablation(visual, "geometry_only")
    assert np.all(geometry_only[0:10] == 0)
    assert np.array_equal(geometry_only[10:15], visual[10:15])
    assert np.array_equal(geometry_only[15], visual[15])
    assert np.array_equal(geometry_only[16], visual[16])


def test_input_ablation_never_mutates_the_caller_tensor() -> None:
    rng = np.random.default_rng(13)
    visual = _contract_legal_episode("REMOVE", rng)
    before = visual.copy()

    apply_input_ablation(visual, "operation_control")

    assert np.array_equal(visual, before)


def test_unknown_input_ablation_still_raises() -> None:
    rng = np.random.default_rng(17)
    visual = _contract_legal_episode("ADD", rng)

    with pytest.raises(LearningContractError, match="unknown input ablation"):
        apply_input_ablation(visual, "operation_contro1")
    with pytest.raises(LearningContractError, match="unknown input ablation"):
        apply_input_ablation(visual, "no_relpos")


def test_operation_control_tensor_is_accepted_by_the_frozen_p2t_model() -> None:
    """The merged cue satisfies the model's polarity-blind branch — no model change."""

    rng = np.random.default_rng(19)
    ablated = apply_input_ablation(
        _contract_legal_episode("REMOVE", rng), "operation_control"
    )

    model = build_p2t_model(width=32)
    output = model(
        torch.from_numpy(ablated)[None], torch.tensor([[2.0, 2.0]])
    )

    assert output["joint_logits"].shape == (1, 6)


# --- contract-layer activation of the fifth arm -----------------------------
#
# The frozen v2 config's sha256 is recorded episode-by-episode across the 739
# corpus rows, so the fifth arm cannot be switched on by editing it in place --
# that would break the provenance chain of every number already reported.  The
# arm is activated by a SEPARATE, separately-versioned config that carries its
# parent's digest, so the two live side by side without contaminating each
# other.

import hashlib  # noqa: E402
import json  # noqa: E402

FROZEN_CONFIG = PROJECT / "configs" / "petct_route_a_experiment.json"
OPERATION_CONTROL_CONFIG = (
    PROJECT / "configs" / "petct_route_a_experiment_operation_control_v2_1.json"
)
FROZEN_ARMS = ["full", "no_M0", "polarity_blind", "geometry_only"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_frozen_config_still_carries_exactly_the_four_original_arms() -> None:
    """The v2 config must not gain the fifth arm; its digest is load-bearing."""

    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    assert frozen["schema_version"] == "PETCT-ROUTE-A-EXPERIMENT-v2.0"
    assert frozen["p2t"]["simple_first_input_arms"] == FROZEN_ARMS
    assert "operation_control" not in frozen["p2t"]["input_arms"]


def test_operation_control_config_is_a_separate_version_naming_its_parent() -> None:
    """The fifth arm ships as its own version, with the parent digest recorded."""

    child = json.loads(OPERATION_CONTROL_CONFIG.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))

    assert child["schema_version"] == "PETCT-ROUTE-A-EXPERIMENT-v2.1"
    provenance = child["derived_from"]
    assert provenance["parent_schema_version"] == frozen["schema_version"]
    assert provenance["parent_sha256"] == _sha256_file(FROZEN_CONFIG)
    assert provenance["only_change"] == "adds the operation_control input arm"

    # The four frozen arms survive byte-identically, in order, ahead of the new one.
    assert child["p2t"]["simple_first_input_arms"] == FROZEN_ARMS + [
        "operation_control"
    ]
    for arm in FROZEN_ARMS:
        assert child["p2t"]["input_arms"][arm] == frozen["p2t"]["input_arms"][arm]
    assert "operation_control" in child["p2t"]["input_arms"]

    # Nothing else moved: strip the two intended deltas and the rest must match.
    stripped = json.loads(json.dumps(child))
    del stripped["derived_from"]
    stripped["schema_version"] = frozen["schema_version"]
    stripped["p2t"]["simple_first_input_arms"] = FROZEN_ARMS
    del stripped["p2t"]["input_arms"]["operation_control"]
    assert stripped == frozen


def test_trainer_accepts_the_fifth_arm_only_against_the_new_config() -> None:
    """Running the new arm against the frozen config must fail, not mislabel."""

    frozen = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    child = json.loads(OPERATION_CONTROL_CONFIG.read_text(encoding="utf-8"))
    assert "operation_control" not in load_p2t_evaluation_contract(frozen)[
        "ablation_inputs"
    ]
    assert "operation_control" in load_p2t_evaluation_contract(child)[
        "ablation_inputs"
    ]

    # The CLI must actually offer the arm, otherwise the new config is
    # unreachable.  Ask argparse itself rather than grepping the source -- a
    # source-text assertion would pass against a file that merely mentions the
    # string.
    import subprocess

    helped = subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts" / "p2t" / "train_petct_p2t.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert helped.returncode == 0, helped.stderr
    assert "operation_control" in helped.stdout
