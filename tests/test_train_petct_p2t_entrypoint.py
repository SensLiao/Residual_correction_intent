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
sys.path.insert(0, str(SCRIPTS / "evaluation"))

import train_petct_p2t as trainer  # noqa: E402
import evaluate_petct_p2t as evaluator  # noqa: E402


SIZE = 16
CONFIG = PROJECT / "configs" / "petct_route_a_experiment_v3.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _r13_data_ready(tmp_path: Path, manifest: Path, split: Path) -> Path:
    oof = tmp_path / "oof-ready.json"
    oof.write_text("{}\n", encoding="utf-8")
    lineage = tmp_path / "lineage.json"
    lineage.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-R13-LINEAGE-v1.0",
                "status": "PASS",
                "dataset_id": "R13-main-single-round",
                "source_m0_lineage": "M0_V6_FIVEFOLD_OOF",
                "mainline_eligible": True,
                "lifecycle": "active",
                "episode_schema": "single_round_one_scribble_one_strategy_v1",
                "round_count": 1,
                "scribbles_per_episode": 1,
                "strategy_is_label": False,
                "partitions": ["train", "val"],
                "locked_test_present": False,
                "oof_ready": _record(oof),
                "learning_split": _record(split),
                "experiment_config": _record(CONFIG),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    program = tmp_path / "program-receipt.json"
    program.write_text("{}\n", encoding="utf-8")
    outputs = {}
    for name in ("inference_visible", "label_only", "audit_only", "candidate_summary", "pointer_summary"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        outputs[name] = _record(path)
    outputs["rich_tensors"] = _record(manifest)
    ready = tmp_path / "data-ready.json"
    ready.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-R13-DATA-READY-v1.0",
                "status": "PASS",
                "dataset_id": "R13-main-single-round",
                "source_m0_lineage": "M0_V6_FIVEFOLD_OOF",
                "mainline_eligible": True,
                "lifecycle": "active",
                "locked_test_present": False,
                "lineage_receipt": _record(lineage),
                "program_manifest_receipt": _record(program),
                "outputs": outputs,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return ready


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
        {"patient_id": "p-4", "partition": "val", "case_ids": ["c-4"]},
    ]
    split_path.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
                "status": "FROZEN_BEFORE_MODEL_SELECTION",
                "patient_count": len(patients),
                "case_count": len(patients),
                "case_counts": {"train": 2, "val": 2, "test": 0},
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
        ("c-4", "p-4", "val", "REMOVE", "NEW", "COMPLETE"),
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
                "strategy": "centerline",
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
    data_ready = _r13_data_ready(tmp_path, manifest_path, split_path)
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
                "--r13-data-ready",
                str(data_ready),
                "--baseline-arm",
                "J1",
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

    predictions = tmp_path / "predictions.jsonl"
    paired = tmp_path / "paired.jsonl"
    metrics = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_petct_p2t.py",
            "--manifest", str(manifest_path),
            "--training-manifest", str(manifest_path),
            "--experiment-config", str(CONFIG),
            "--learning-split", str(split_path),
            "--checkpoint", str(output),
            "--r13-data-ready", str(data_ready),
            "--partition", "val",
            "--predictions", str(predictions),
            "--paired-evaluation-rows", str(paired),
            "--metrics", str(metrics),
            "--bootstrap-samples", "20",
            "--device", "cpu",
        ],
    )
    monkeypatch.setattr(
        evaluator, "load_p2t_evaluation_contract", _twenty_bootstrap_contract
    )
    monkeypatch.setattr(evaluator, "load_training_contract", _two_epoch_contract)
    assert evaluator.main() == 0
    metric_payload = json.loads(metrics.read_text(encoding="utf-8"))
    assert metric_payload["source_m0_lineage"] == "M0_V6_FIVEFOLD_OOF"
    assert metric_payload["r13_data_ready_sha256"] == _sha256(data_ready)


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


def _twenty_bootstrap_contract(config):
    from petct_learning import load_p2t_evaluation_contract as real

    contract = dict(real(config))
    contract["bootstrap_samples"] = 20
    return contract


def test_j1_joint_and_j2_independent_slot_decoders_are_distinct() -> None:
    assert hasattr(trainer, "decode_flat_baseline")
    output = {
        "joint_logits": torch.tensor([[0.0, 0.0, 0.0, 0.0, 8.0, 0.0]]),
        "operation_logits": torch.tensor([[8.0, 0.0]]),
        "target_logits": torch.tensor([[0.0, 8.0]]),
        "scope_logits": torch.tensor([[8.0, 0.0]]),
    }
    j1 = trainer.decode_flat_baseline(output, "J1")
    j2 = trainer.decode_flat_baseline(output, "J2")
    assert j1["joint_id"].tolist() == [4]
    assert j1["raw_illegal"].tolist() == [False]
    # J2 raw slots predict ADD_NEW_LOCAL (illegal), then the declared legal
    # projection changes only scope to COMPLETE for executable output.
    assert j2["raw_illegal"].tolist() == [True]
    assert j2["joint_id"].tolist() == [4]
    assert j2["scope_id"].tolist() == [1]
