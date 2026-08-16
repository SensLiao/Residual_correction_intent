"""One-process synthetic dry run of the prediction-first v3 chain."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
for directory in (SCRIPTS, SCRIPTS / "p2t", SCRIPTS / "editor", SCRIPTS / "evaluation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import evaluate_petct_program_v3 as evaluator  # noqa: E402
import infer_petct_program_editor_v3 as editor_infer  # noqa: E402
import infer_petct_program_v3 as compiler_infer  # noqa: E402
import render_petct_gold_program_calls_v3 as gold_calls  # noqa: E402
from common.petct_program_models import (  # noqa: E402
    ProgramCompilerNet,
    ProgramEditorUNet2D,
    ProgramEmbedding,
)


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


def _candidate_tree_sha(directory: Path) -> str:
    records = []
    for path in sorted(directory.glob("*.json")):
        records.append((path.name, _sha(path)))
    return hashlib.sha256(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def test_global_family_zero_is_not_padding() -> None:
    embedding = ProgramEmbedding(dim=8)
    output = embedding(torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
    assert output.shape == (1, 8)
    output.sum().backward()
    assert embedding.family.weight.grad[1].abs().sum() > 0
    assert embedding.family.weight.grad[0].abs().sum() == 0


def test_prediction_first_compiler_editor_evaluator_dry_run(tmp_path: Path) -> None:
    visible_rows, label_rows = [], []
    candidates_dir = tmp_path / "candidates"
    targets_dir = tmp_path / "targets"
    candidates_dir.mkdir()
    targets_dir.mkdir()
    for index, (operation, goal) in enumerate(GOALS):
        episode = f"episode-{index}"
        visual = np.zeros((17, 16, 16), dtype=np.float32)
        cue_fg = np.zeros((16, 16), dtype=np.float32)
        cue_bg = np.zeros((16, 16), dtype=np.float32)
        (cue_fg if operation == "ADD" else cue_bg)[8, 8] = 1
        visual[15] = cue_fg
        visual[16] = cue_bg
        m0 = np.zeros((16, 16), dtype=np.float32)
        if operation == "REMOVE":
            m0[6:10, 6:10] = 1
        visible_path = tmp_path / f"{episode}-visible.npz"
        np.savez_compressed(
            visible_path,
            visual=visual,
            m0=m0,
            scribble=cue_fg + cue_bg,
            cue_fg=cue_fg,
            cue_bg=cue_bg,
            spacing_xy=np.asarray([1.0, 1.0], dtype=np.float32),
        )
        target = np.zeros((16, 16), dtype=np.float32)
        target[8, 8] = 1
        gt = (m0 > 0).astype(np.float32)
        if operation == "ADD":
            gt[8, 8] = 1
        else:
            gt[8, 8] = 0
        evaluation_path = tmp_path / f"{episode}-evaluation.npz"
        np.savez_compressed(evaluation_path, target=target, authorized=target, gt=gt)
        visible_rows.append(
            {
                "schema_version": "PETCT-PROGRAM-INFERENCE-MANIFEST-v1.0",
                "episode_id": episode,
                "partition": "val",
                "operation": operation,
                "visible_npz": str(visible_path),
                "visible_sha256": _sha(visible_path),
            }
        )
        label_rows.append(
            {
                "schema_version": "PETCT-PROGRAM-LABEL-MANIFEST-v1.0",
                "episode_id": episode,
                "case_id": f"case-{index}",
                "patient_id": f"patient-{index // 2}",
                "partition": "val",
                "goal": goal,
                "operation": operation,
                "matched_state_group_id": "g-add" if operation == "ADD" else "g-remove",
                "evaluation_npz": str(evaluation_path),
                "evaluation_sha256": _sha(evaluation_path),
                "learning_split_sha256": "a" * 64,
            }
        )
        key = f"{episode}|component|0"
        mask = (m0 > 0).astype(np.uint8)
        if operation == "ADD":
            mask[7:10, 7:10] = 1
        candidate = {
            "episode_id": episode,
            "m_sha256": "b" * 64,
            "enumeration_version": "test-enumeration",
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
                    "centroid_dx_mm": -2.0,
                    "centroid_dy_mm": 3.0,
                    "cue_overlap_voxels": 1.0,
                    "distance_from_cue_mm": 0.0,
                    "prompted_slice_mask": mask.tolist(),
                }
            ],
        }
        (candidates_dir / f"{episode}.json").write_text(
            json.dumps(candidate, sort_keys=True), encoding="utf-8"
        )
        if operation == "ADD" and goal != "ADD_NEW_COMPLETE":
            (targets_dir / f"{episode}.json").write_text(
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
    visible_manifest = tmp_path / "visible.jsonl"
    labels_manifest = tmp_path / "labels.jsonl"
    _write_jsonl(visible_manifest, visible_rows)
    _write_jsonl(labels_manifest, label_rows)

    compiler = ProgramCompilerNet(include_repair=True)
    compiler_checkpoint = tmp_path / "compiler.pt"
    torch.save(
        {
            "schema_version": "PETCT-PROGRAM-COMPILER-CHECKPOINT-v1.0",
            "architecture_id": "matched_legal_component_program_v1",
            "episodes_sha256": _sha(visible_manifest),
            "candidates_tree_sha256": _candidate_tree_sha(candidates_dir),
            "hyperparameters": {"include_repair": True},
            "state_dict": compiler.state_dict(),
        },
        compiler_checkpoint,
    )
    predictions = tmp_path / "predictions.jsonl"
    prediction_receipt = tmp_path / "predictions.receipt.json"
    assert compiler_infer.main(
        [
            "--episodes", str(visible_manifest),
            "--candidates", str(candidates_dir),
            "--checkpoint", str(compiler_checkpoint),
            "--partition", "val",
            "--output", str(predictions),
            "--receipt", str(prediction_receipt),
            "--batch-size", "3",
            "--device", "cpu",
        ]
    ) == 0
    predicted_rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert len(predicted_rows) == 6
    assert all(row["decision"] == "PREDICT" for row in predicted_rows)

    editor = ProgramEditorUNet2D(visual_channels=13, conditioner="program")
    editor_checkpoint = tmp_path / "editor.pt"
    torch.save(
        {
            "schema_version": "PETCT-PROGRAM-EDITOR-CHECKPOINT-v1.0",
            "episodes_sha256": _sha(visible_manifest),
            "candidates_tree_sha256": _candidate_tree_sha(candidates_dir),
            "arm": "J9",
            "state_dict": editor.state_dict(),
        },
        editor_checkpoint,
    )
    editor_dir = tmp_path / "editor-deltas"
    editor_manifest = tmp_path / "editor.jsonl"
    editor_receipt = tmp_path / "editor.receipt.json"
    assert editor_infer.main(
        [
            "--episodes", str(visible_manifest),
            "--candidates", str(candidates_dir),
            "--program-predictions", str(predictions),
            "--program-receipt", str(prediction_receipt),
            "--editor-checkpoint", str(editor_checkpoint),
            "--partition", "val",
            "--output-dir", str(editor_dir),
            "--output-manifest", str(editor_manifest),
            "--receipt", str(editor_receipt),
            "--batch-size", "3",
            "--device", "cpu",
        ]
    ) == 0

    oracle_calls = tmp_path / "oracle-calls.jsonl"
    oracle_call_receipt = tmp_path / "oracle-calls.receipt.json"
    assert gold_calls.main(
        [
            "--labels", str(labels_manifest),
            "--candidates", str(candidates_dir),
            "--pointer-targets", str(targets_dir),
            "--partition", "val",
            "--output", str(oracle_calls),
            "--receipt", str(oracle_call_receipt),
        ]
    ) == 0
    oracle_dir = tmp_path / "oracle-deltas"
    oracle_editor_manifest = tmp_path / "oracle-editor.jsonl"
    oracle_editor_receipt = tmp_path / "oracle-editor.receipt.json"
    assert editor_infer.main(
        [
            "--episodes", str(visible_manifest),
            "--candidates", str(candidates_dir),
            "--program-predictions", str(oracle_calls),
            "--program-receipt", str(oracle_call_receipt),
            "--editor-checkpoint", str(editor_checkpoint),
            "--partition", "val",
            "--output-dir", str(oracle_dir),
            "--output-manifest", str(oracle_editor_manifest),
            "--receipt", str(oracle_editor_receipt),
            "--batch-size", "3",
            "--device", "cpu",
        ]
    ) == 0

    report_path = tmp_path / "report.json"
    assert evaluator.main(
        [
            "--predictions", str(predictions),
            "--prediction-receipt", str(prediction_receipt),
            "--labels", str(labels_manifest),
            "--inference-manifest", str(visible_manifest),
            "--candidates", str(candidates_dir),
            "--pointer-targets", str(targets_dir),
            "--editor-predictions", str(editor_manifest),
            "--editor-receipt", str(editor_receipt),
            "--oracle-editor-predictions", str(oracle_editor_manifest),
            "--oracle-editor-receipt", str(oracle_editor_receipt),
            "--bootstrap-samples", "50",
            "--output", str(report_path),
        ]
    ) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["compiler"]["expected_episodes"] == 6
    assert report["compiler"]["legal_call_rate"] == 1.0
    assert report["editor_predicted_calls"]["2d_coverage"] == 1.0
    assert report["editor_gold_call_ceiling"]["2d_coverage"] == 1.0
    assert report["predicted_vs_gold_same_editor_gap"]["same_editor_checkpoint_required"]


def test_missing_prediction_remains_in_denominator() -> None:
    summary = evaluator._patient_balanced_macro_f1(
        [0, 0, 1], [0, evaluator.ERROR_SENTINEL, 1], ["p1", "p1", "p2"], [0, 1]
    )
    assert summary["estimate"] < 1.0
    assert summary["per_class_support"] == {"0": 1, "1": 1}
