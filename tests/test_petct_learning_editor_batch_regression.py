"""Regression guard for the editor batch path (`load_evaluation=True`).

Why this file exists
--------------------
`EpisodeDataset.__getitem__` loaded the evaluation delta mask into a local named
`target`, then a few lines later rebound the same name to the gold TARGET slot label
("SAME"/"NEW").  Every editor batch therefore died at `torch.from_numpy(target[None])`
with `TypeError: string indices must be integers`.

The pre-existing suite never caught it because nothing exercised
`load_evaluation=True` against a real npz pair -- it is the editor trainer's only
code path, and the P2T path (`load_evaluation=False`) skips the shadowed block
entirely.  These tests build a genuine visible/evaluation npz pair, one ADD and one
REMOVE, and assert the tensors that come out.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from common.petct_learning import EpisodeDataset  # noqa: E402

SIZE = 16
SPACING = 2.0
STATE_CHANNELS = 15
CUE_FG_CH, CUE_BG_CH = 15, 16


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode(root: Path, episode_id: str, operation: str) -> dict:
    """One structurally faithful episode: 17-channel visual + disjoint cue pair."""
    rng = np.random.default_rng(abs(hash(episode_id)) % (2**32))
    m0 = np.zeros((SIZE, SIZE), np.float32)
    m0[4:9, 4:9] = 1.0
    gt = np.zeros((SIZE, SIZE), np.float32)
    gt[6:12, 6:12] = 1.0
    # ADD repairs GT\M0; REMOVE repairs M0\GT.  authorized == target by contract.
    residual = (gt > 0) & ~(m0 > 0) if operation == "ADD" else (m0 > 0) & ~(gt > 0)
    authorized = residual.astype(np.float32)
    assert authorized.any(), "fixture must leave a non-empty residual"

    cue = np.zeros((SIZE, SIZE), np.float32)
    first = np.argwhere(authorized > 0)[0]
    cue[first[0], first[1]] = 1.0
    cue_fg = cue if operation == "ADD" else np.zeros_like(cue)
    cue_bg = cue if operation == "REMOVE" else np.zeros_like(cue)

    visual = np.zeros((STATE_CHANNELS + 2, SIZE, SIZE), np.float32)
    visual[:10] = rng.normal(size=(10, SIZE, SIZE))
    visual[10:15] = m0
    visual[CUE_FG_CH] = cue_fg
    visual[CUE_BG_CH] = cue_bg

    visible = root / f"{episode_id}-visible.npz"
    evaluation = root / f"{episode_id}-evaluation.npz"
    np.savez_compressed(
        visible,
        visual=visual,
        m0=m0,
        scribble=cue,
        cue_fg=cue_fg,
        cue_bg=cue_bg,
        spacing_xy=np.asarray([SPACING, SPACING], np.float32),
    )
    np.savez_compressed(evaluation, target=authorized, gt=gt, authorized=authorized)

    return {
        "episode_id": episode_id,
        "case_id": f"case-{episode_id}",
        "patient_id": f"patient-{episode_id}",
        "partition": "train",
        "strategy": "centerline",
        "goal": f"{operation}_SAME_COMPLETE",
        "operation": operation,
        "target": "SAME",
        "scope": "COMPLETE",
        "visible_npz": str(visible),
        "visible_sha256": _sha(visible),
        "evaluation_npz": str(evaluation),
        "evaluation_sha256": _sha(evaluation),
        "geometry": {"output_spacing_xy": [SPACING, SPACING]},
    }


@pytest.fixture()
def manifest(tmp_path: Path) -> Path:
    rows = [_episode(tmp_path, "ep-add", "ADD"), _episode(tmp_path, "ep-remove", "REMOVE")]
    path = tmp_path / "learning_tensors.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_editor_batch_returns_the_evaluation_mask_not_the_gold_label(manifest: Path) -> None:
    """The regression itself: `target` must come back as a mask, never as "SAME"."""
    dataset = EpisodeDataset(manifest, "train", editor_condition="scribble_plus_intent")
    assert dataset.load_evaluation is True

    for index in range(len(dataset)):
        sample = dataset[index]
        for key in ("target", "gt", "authorized"):
            tensor = sample[key]
            assert tensor.shape == (1, SIZE, SIZE), f"{key} has shape {tensor.shape}"
            assert tensor.dtype.is_floating_point
        # target IS the authorized delta -- that equality is the loaded contract
        assert np.array_equal(sample["target"].numpy(), sample["authorized"].numpy())
        assert sample["target"].sum() > 0
        # the gold slot label survives independently, in its own key
        assert int(sample["target_gold"]) == 0  # SAME


def test_editor_batch_keeps_the_two_polarities_distinguishable(manifest: Path) -> None:
    """An ADD and a REMOVE episode must not collapse to the same editor input."""
    dataset = EpisodeDataset(manifest, "train", editor_condition="scribble_plus_intent")
    samples = {dataset[i]["episode_id"]: dataset[i] for i in range(len(dataset))}
    add, remove = samples["ep-add"], samples["ep-remove"]

    assert int(add["operation_gold"]) != int(remove["operation_gold"])
    assert add["cue_fg"].sum() > 0 and add["cue_bg"].sum() == 0
    assert remove["cue_bg"].sum() > 0 and remove["cue_fg"].sum() == 0
    # signed cue the editor derives: cue_fg - cue_bg
    add_signed = (add["cue_fg"] - add["cue_bg"]).numpy()
    remove_signed = (remove["cue_fg"] - remove["cue_bg"]).numpy()
    assert add_signed.max() == 1.0 and remove_signed.min() == -1.0
    assert not np.array_equal(add_signed, remove_signed)


def test_p2t_batch_path_is_unaffected(manifest: Path) -> None:
    """`load_evaluation=False` must keep skipping the evaluation bundle entirely."""
    dataset = EpisodeDataset(manifest, "train", load_evaluation=False)
    sample = dataset[0]
    assert "target" not in sample and "gt" not in sample and "authorized" not in sample
    assert sample["visual"].shape == (STATE_CHANNELS + 2, SIZE, SIZE)
