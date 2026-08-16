"""CPU one-epoch smoke for the split-bound v3 compiler and editor trainers."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
for directory in (SCRIPTS, SCRIPTS / "p2t", SCRIPTS / "editor"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import train_petct_program_editor_v3 as editor_train  # noqa: E402
import train_petct_program_v3 as compiler_train  # noqa: E402


GOALS = (
    ("ADD", "ADD_SAME_LOCAL"),
    ("ADD", "ADD_SAME_COMPLETE"),
    ("ADD", "ADD_NEW_COMPLETE"),
    ("REMOVE", "REMOVE_SAME_LOCAL"),
    ("REMOVE", "REMOVE_SAME_COMPLETE"),
    ("REMOVE", "REMOVE_NEW_COMPLETE"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _tree_sha(directory: Path) -> str:
    rows = [
        (path.relative_to(directory).as_posix(), _sha(path))
        for path in sorted(value for value in directory.rglob("*") if value.is_file())
    ]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def test_j3_placebo_changes_membership_not_only_group_names() -> None:
    labels = {}
    goals_by_operation = {
        "ADD": ["ADD_SAME_LOCAL", "ADD_SAME_COMPLETE", "ADD_NEW_COMPLETE"],
        "REMOVE": [
            "REMOVE_SAME_LOCAL", "REMOVE_NEW_COMPLETE", "REMOVE_SAME_COMPLETE"
        ],
    }
    source_group = {}
    for operation, goals in goals_by_operation.items():
        for group_index in range(2):
            for family_index, goal in enumerate(goals):
                episode = f"{operation}-{group_index}-{family_index}"
                group = f"{operation}-source-{group_index}"
                labels[episode] = {
                    "partition": "train",
                    "operation": operation,
                    "goal": goal,
                    "matched_state_group_id": group,
                }
                source_group[episode] = group
    assignment, digest = compiler_train._build_placebo_groups(
        labels, partition="train", seed=20260816
    )
    assert len(digest) == 64
    members = {}
    for episode, placebo in assignment.items():
        members.setdefault(placebo, set()).add(source_group[episode])
    assert members
    assert all(len(groups) >= 2 for groups in members.values())


def test_v3_trainers_run_one_epoch_with_frozen_split_receipt(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    targets = tmp_path / "targets"
    candidates.mkdir()
    targets.mkdir()
    visible_rows, label_rows, split_patients = [], [], []
    index = 0
    for partition in ("train", "val"):
        for operation, goal in GOALS:
            episode = f"{partition}-episode-{index}"
            case_id = f"case-{index}"
            patient_id = f"patient-{index}"
            visual = np.zeros((17, 8, 8), dtype=np.float32)
            cue_fg = np.zeros((8, 8), dtype=np.float32)
            cue_bg = np.zeros((8, 8), dtype=np.float32)
            (cue_fg if operation == "ADD" else cue_bg)[4, 4] = 1
            visual[15], visual[16] = cue_fg, cue_bg
            m0 = np.zeros((8, 8), dtype=np.float32)
            if operation == "REMOVE":
                m0[2:6, 2:6] = 1
            visible_path = tmp_path / f"{episode}-visible.npz"
            np.savez_compressed(
                visible_path, visual=visual, m0=m0, scribble=cue_fg + cue_bg,
                cue_fg=cue_fg, cue_bg=cue_bg,
                spacing_xy=np.asarray([1.0, 1.0], dtype=np.float32),
            )
            target = np.zeros((8, 8), dtype=np.float32)
            target[4, 4] = 1
            gt = m0.copy()
            evaluation_path = tmp_path / f"{episode}-evaluation.npz"
            np.savez_compressed(evaluation_path, target=target, authorized=target, gt=gt)
            visible_rows.append(
                {
                    "schema_version": "PETCT-PROGRAM-INFERENCE-MANIFEST-v1.0",
                    "episode_id": episode,
                    "partition": partition,
                    "operation": operation,
                    "visible_npz": str(visible_path),
                    "visible_sha256": _sha(visible_path),
                }
            )
            split_patients.append(
                {
                    "patient_id": patient_id,
                    "partition": partition,
                    "case_ids": [case_id],
                }
            )
            key = f"{episode}|component|0"
            mask = (m0 > 0).astype(np.uint8)
            if operation == "ADD":
                mask[3:6, 3:6] = 1
            candidate = {
                "episode_id": episode,
                "m_sha256": "c" * 64,
                "enumeration_version": "smoke",
                "component_count": 1,
                "central_masks_available": True,
                "cue_hit_component_position": 0 if operation == "REMOVE" else None,
                "components": [
                    {
                        "candidate_position": 0,
                        "component_key": key,
                        "log_volume": 1.0,
                        "z_span": 1.0,
                        "prompted_slice_overlap": 1.0,
                        "centroid_dx_mm": -1.0,
                        "centroid_dy_mm": 1.0,
                        "cue_overlap_voxels": 1.0,
                        "distance_from_cue_mm": 0.0,
                        "prompted_slice_mask": mask.tolist(),
                    }
                ],
            }
            (candidates / f"{episode}.json").write_text(
                json.dumps(candidate, sort_keys=True), encoding="utf-8"
            )
            if operation == "ADD" and goal != "ADD_NEW_COMPLETE":
                (targets / f"{episode}.json").write_text(
                    json.dumps(
                        {
                            "episode_id": episode,
                            "pointer_targets": [0],
                            "pointer_target_positions": [0],
                            "pointer_target_component_keys": [key],
                        },
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            label_rows.append(
                {
                    "schema_version": "PETCT-PROGRAM-LABEL-MANIFEST-v1.0",
                    "episode_id": episode,
                    "case_id": case_id,
                    "patient_id": patient_id,
                    "partition": partition,
                    "goal": goal,
                    "operation": operation,
                    "matched_state_group_id": f"{partition}-{operation.lower()}",
                    "evaluation_npz": str(evaluation_path),
                    "evaluation_sha256": _sha(evaluation_path),
                    "learning_split_sha256": "PENDING",
                }
            )
            index += 1
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
                "status": "FROZEN_BEFORE_MODEL_SELECTION",
                "split_unit": "patient",
                "patients": split_patients,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    split_sha = _sha(split)
    for row in label_rows:
        row["learning_split_sha256"] = split_sha
    visible_manifest = tmp_path / "visible.jsonl"
    label_manifest = tmp_path / "labels.jsonl"
    _write_jsonl(visible_manifest, visible_rows)
    _write_jsonl(label_manifest, label_rows)
    receipt = tmp_path / "manifest-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-PROGRAM-MANIFEST-READY-v1.0",
                "status": "PASS",
                "locked_test_present": False,
                "learning_split": {"path": str(split.resolve()), "sha256": split_sha},
                "outputs": {
                    "inference": {
                        "path": str(visible_manifest.resolve()), "sha256": _sha(visible_manifest)
                    },
                    "labels": {
                        "path": str(label_manifest.resolve()), "sha256": _sha(label_manifest)
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = PROJECT / "configs" / "petct_route_a_experiment_v3.json"
    compiler_checkpoint = tmp_path / "compiler.pt"
    assert compiler_train.main.__name__ == "main"
    old_argv = sys.argv
    try:
        sys.argv = [
            "train_petct_program_v3.py",
            "--episodes", str(visible_manifest),
            "--labels", str(label_manifest),
            "--learning-split", str(split),
            "--manifest-receipt", str(receipt),
            "--candidates", str(candidates),
            "--experiment-config", str(config),
            "--output", str(compiler_checkpoint),
            "--arm", "J0",
            "--epochs", "1",
            "--batch-size", "6",
            "--seed", "3407",
            "--device", "cpu",
        ]
        assert compiler_train.main() == 0
    finally:
        sys.argv = old_argv
    compiler_payload = torch.load(compiler_checkpoint, map_location="cpu", weights_only=False)
    assert compiler_payload["best"]["val_loss"] >= 0
    assert compiler_payload["learning_split_sha256"] == split_sha
    assert compiler_payload["candidates_tree_sha256"] == _tree_sha(candidates)

    editor_checkpoint = tmp_path / "editor.pt"
    try:
        sys.argv = [
            "train_petct_program_editor_v3.py",
            "--episodes", str(visible_manifest),
            "--labels", str(label_manifest),
            "--learning-split", str(split),
            "--manifest-receipt", str(receipt),
            "--candidates", str(candidates),
            "--pointer-targets", str(targets),
            "--experiment-config", str(config),
            "--output", str(editor_checkpoint),
            "--arm", "J9",
            "--epochs", "1",
            "--batch-size", "6",
            "--seed", "3407",
            "--device", "cpu",
        ]
        assert editor_train.main() == 0
    finally:
        sys.argv = old_argv
    editor_payload = torch.load(editor_checkpoint, map_location="cpu", weights_only=False)
    assert editor_payload["best"]["val_loss"] >= 0
    assert editor_payload["learning_split_sha256"] == split_sha

    continuous_checkpoint = tmp_path / "editor-j8.pt"
    try:
        sys.argv = [
            "train_petct_program_editor_v3.py",
            "--episodes", str(visible_manifest),
            "--labels", str(label_manifest),
            "--learning-split", str(split),
            "--manifest-receipt", str(receipt),
            "--candidates", str(candidates),
            "--pointer-targets", str(targets),
            "--compiler-checkpoint", str(compiler_checkpoint),
            "--experiment-config", str(config),
            "--output", str(continuous_checkpoint),
            "--arm", "J8",
            "--epochs", "1",
            "--batch-size", "6",
            "--seed", "3407",
            "--device", "cpu",
        ]
        assert editor_train.main() == 0
    finally:
        sys.argv = old_argv
    continuous = torch.load(
        continuous_checkpoint, map_location="cpu", weights_only=False
    )
    assert continuous["best"]["val_loss"] >= 0
    assert continuous["compiler_checkpoint_sha256"] == _sha(compiler_checkpoint)
