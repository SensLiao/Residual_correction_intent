"""The P2T trainer's entry point had zero execution coverage.

External audit F-008/F-007 and a local `grep -rn "import.*train_petct_p2t"` both
found the same thing: nothing in the repository ever imported or ran this module.
Every test that mentioned it asserted on *source text* -- so a file that merely
contained the right strings, with no working logic at all, would have passed.
The model that produced the reported six-class score therefore came out of code
no test had ever executed.

This test executes it: a synthetic three-episode corpus, one real training run
through `main()`, then assertions on the artefact it actually wrote.  It needs
no GPU and no real data.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "common"))
sys.path.insert(0, str(SCRIPTS / "p2t"))

import train_petct_p2t as trainer  # noqa: E402


SIZE = 16
CONFIG = PROJECT / "configs" / "petct_route_a_experiment.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode(tmp_path: Path, name: str, operation: str, rng) -> tuple[Path, Path]:
    """Write a contract-legal visible/evaluation NPZ pair."""

    m0 = np.zeros((SIZE, SIZE), dtype=np.float32)
    m0[4:10, 4:10] = 1.0
    gt = np.zeros_like(m0)
    gt[6:12, 6:12] = 1.0
    if operation == "ADD":
        authorized = ((gt > 0) & ~(m0 > 0)).astype(np.float32)
    else:
        authorized = ((m0 > 0) & ~(gt > 0)).astype(np.float32)
    coordinates = np.argwhere(authorized > 0)
    scribble = np.zeros_like(m0)
    scribble[coordinates[0][0], coordinates[0][1]] = 1.0
    cue_fg = scribble.copy() if operation == "ADD" else np.zeros_like(scribble)
    cue_bg = scribble.copy() if operation == "REMOVE" else np.zeros_like(scribble)

    visual = np.concatenate(
        [
            rng.normal(size=(5, SIZE, SIZE)).astype(np.float32),
            rng.normal(size=(5, SIZE, SIZE)).astype(np.float32),
            np.repeat(m0[None], 5, axis=0),
            cue_fg[None],
            cue_bg[None],
        ],
        axis=0,
    ).astype(np.float32)
    spacing = np.asarray([1.0, 1.0], dtype=np.float32)

    visible_path = tmp_path / f"{name}-visible.npz"
    evaluation_path = tmp_path / f"{name}-evaluation.npz"
    np.savez_compressed(
        visible_path,
        visual=visual,
        m0=m0,
        scribble=scribble,
        cue_fg=cue_fg,
        cue_bg=cue_bg,
        spacing_xy=spacing,
    )
    np.savez_compressed(
        evaluation_path, target=authorized, gt=gt, authorized=authorized
    )
    return visible_path, evaluation_path


def _corpus(tmp_path: Path) -> tuple[Path, Path, str]:
    rng = np.random.default_rng(20260807)
    split_path = tmp_path / "learning-split.json"
    patients = [
        {"patient_id": "p-1", "partition": "train", "case_ids": ["c-1"]},
        {"patient_id": "p-2", "partition": "train", "case_ids": ["c-2"]},
        {"patient_id": "p-3", "partition": "val", "case_ids": ["c-3"]},
    ]
    split_path.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
                "status": "FROZEN_BEFORE_MODEL_SELECTION",
                "patient_count": len(patients),
                "case_count": len(patients),
                "case_counts": {"train": 2, "val": 1, "test": 0},
                "patients": patients,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    split_sha = _sha256(split_path)
    config_sha = _sha256(CONFIG)

    rows = []
    plan = [
        ("c-1", "p-1", "train", "ADD", "SAME", "LOCAL"),
        ("c-2", "p-2", "train", "REMOVE", "SAME", "COMPLETE"),
        ("c-3", "p-3", "val", "ADD", "NEW", "COMPLETE"),
    ]
    for case_id, patient_id, partition, operation, target, scope in plan:
        visible_path, evaluation_path = _episode(tmp_path, case_id, operation, rng)
        rows.append(
            {
                "case_id": case_id,
                "episode_id": f"episode-{case_id}",
                "patient_id": patient_id,
                "partition": partition,
                "learning_split_sha256": split_sha,
                "experiment_config_sha256": config_sha,
                "operation": operation,
                "target": target,
                "scope": scope,
                "visible_npz": str(visible_path.resolve()),
                "visible_sha256": _sha256(visible_path),
                "evaluation_npz": str(evaluation_path.resolve()),
                "evaluation_sha256": _sha256(evaluation_path),
                "geometry": {"output_spacing_xy": [1.0, 1.0]},
            }
        )
    manifest_path = tmp_path / "learning.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest_path, split_path, config_sha


def test_training_entrypoint_runs_and_writes_a_loadable_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute the trainer for real and check the artefact it produced."""

    manifest_path, split_path, config_sha = _corpus(tmp_path)
    output = tmp_path / "p2t-smoke.pth"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_petct_p2t.py",
            "--manifest",
            str(manifest_path),
            "--learning-split",
            str(split_path),
            "--experiment-config",
            str(CONFIG),
            "--seed",
            "3407",
            "--epochs",
            "2",
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    monkeypatch.setattr(trainer, "load_training_contract", _two_epoch_contract)

    assert trainer.main() == 0
    assert output.is_file()

    # Fail-closed load: the trainer must never store a non-primitive that would
    # make the evaluator's weights_only=True read blow up.
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert checkpoint["experiment_config_sha256"] == config_sha
    assert checkpoint["seed"] == 3407
    assert checkpoint["input_ablation"] == "full"
    assert isinstance(checkpoint["runtime"]["torch"], str)

    state = checkpoint["state_dict"]
    assert state, "trainer wrote an empty state dict"
    assert all(torch.isfinite(tensor).all() for tensor in state.values())

    history = checkpoint["history"]
    assert len(history) == 2
    for epoch in history:
        assert np.isfinite(epoch["train_loss"])
        assert np.isfinite(epoch["val_loss"])

    # Checkpoint selection actually ran and picked one of the epochs it saw.
    assert checkpoint["best_epoch"] in range(len(history))


def _two_epoch_contract(config, section):
    """Shrink only the epoch budget; every other frozen value is untouched.

    A real 100-epoch run would be pure wall-clock in a unit test, and the thing
    under test is that the entry point executes end-to-end, not that it can
    count to a hundred.
    """

    from petct_learning import load_training_contract as real

    contract = dict(real(config, section))
    contract["epochs"] = 2
    return contract
