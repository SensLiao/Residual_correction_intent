"""Contract tests for the B1 2D gold-call ceiling pass.

The gold-call ceiling quantifies the same-editor performance bound: the
frozen effect-val editor checkpoint is re-run with evaluator-lane oracle
calls (the label-derived legal program) instead of predicted compiler calls.
This file covers the construction of those oracle calls for all six frozen
goals, the label-lane receipt semantics that keep the visible and label lanes
separate, the J6 program-blind arm behavior, and the orchestration launcher's
dry-run plan and argument guards.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
for directory in (
    SCRIPTS,
    SCRIPTS / "editor",
    SCRIPTS / "evaluation",
    SCRIPTS / "common",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import infer_petct_program_editor_v3 as editor_infer  # noqa: E402
import render_petct_gold_program_calls_v3 as gold_calls  # noqa: E402
from common.petct_program_contract import (  # noqa: E402
    GOAL_TO_FAMILY,
    NEW_CUE_SENTINEL,
    family_to_id,
    protected_refs_policy,
    render_goal,
)
from common.petct_program_learning import (  # noqa: E402
    LearningContractError,
    _sha256_file,
)
from evaluation import evaluate_petct_program_v3 as evaluator  # noqa: E402

GOALS = (
    ("ADD", "ADD_SAME_LOCAL"),
    ("ADD", "ADD_SAME_COMPLETE"),
    ("ADD", "ADD_NEW_COMPLETE"),
    ("REMOVE", "REMOVE_SAME_LOCAL"),
    ("REMOVE", "REMOVE_SAME_COMPLETE"),
    ("REMOVE", "REMOVE_NEW_COMPLETE"),
)

# goal -> (operand suffix or NEW_CUE sentinel, oracle selection policy)
EXPECTED_CALLS = {
    "ADD_SAME_LOCAL": ("1", "nearest_gold_positive_then_position"),
    "ADD_SAME_COMPLETE": ("1", "nearest_gold_positive_then_position"),
    "ADD_NEW_COMPLETE": (NEW_CUE_SENTINEL, "NEW_CUE"),
    "REMOVE_SAME_LOCAL": ("0", "deterministic_cue_hit"),
    "REMOVE_SAME_COMPLETE": ("0", "deterministic_cue_hit"),
    "REMOVE_NEW_COMPLETE": ("0", "deterministic_cue_hit"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_gold_lane_fixture(tmp_path: Path) -> dict[str, Path]:
    """Minimal VAL lane with all six frozen goals.

    Same shapes and contracts as the end-to-end dry run: a visible bundle,
    a label row, a candidate record and (for ADD existing-object episodes)
    pointer targets per episode.  ADD episodes carry two pointer-positive
    components with different cue distances so the nearest-positive selection
    policy is exercised deterministically.
    """
    candidates_dir = tmp_path / "candidates"
    targets_dir = tmp_path / "targets"
    candidates_dir.mkdir()
    targets_dir.mkdir()
    visible_rows, label_rows = [], []
    for index, (operation, goal) in enumerate(GOALS):
        episode = f"episode-{index}"
        visual = np.zeros((17, 16, 16), dtype=np.float32)
        cue_fg = np.zeros((16, 16), dtype=np.float32)
        cue_bg = np.zeros((16, 16), dtype=np.float32)
        (cue_fg if operation == "ADD" else cue_bg)[8, 8] = 1
        visual[15], visual[16] = cue_fg, cue_bg
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
        gt[8, 8] = 1.0 if operation == "ADD" else 0.0
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
        components = []
        cue_hit = None
        if operation == "REMOVE":
            components.append(
                {
                    "candidate_position": 0,
                    "component_key": f"{episode}|component|0",
                    "log_volume": 1.0,
                    "z_span": 1.0,
                    "prompted_slice_overlap": 1.0,
                    "centroid_dx_mm": -2.0,
                    "centroid_dy_mm": 3.0,
                    "cue_overlap_voxels": 1.0,
                    "distance_from_cue_mm": 0.0,
                    "prompted_slice_mask": (m0 > 0).astype(np.uint8).tolist(),
                }
            )
            cue_hit = 0
        elif goal != "ADD_NEW_COMPLETE":
            for position, distance in ((0, 5.0), (1, 1.0)):
                mask = np.zeros((16, 16), dtype=np.uint8)
                mask[position * 5 + 1 : position * 5 + 5, 2:6] = 1
                components.append(
                    {
                        "candidate_position": position,
                        "component_key": f"{episode}|component|{position}",
                        "log_volume": 1.0,
                        "z_span": 1.0,
                        "prompted_slice_overlap": 1.0,
                        "centroid_dx_mm": -2.0,
                        "centroid_dy_mm": 3.0,
                        "cue_overlap_voxels": 1.0,
                        "distance_from_cue_mm": float(distance),
                        "prompted_slice_mask": mask.tolist(),
                    }
                )
        candidate = {
            "episode_id": episode,
            "m_sha256": "b" * 64,
            "enumeration_version": "test-enumeration",
            "component_count": len(components),
            "central_masks_available": bool(components),
            "cue_hit_component_position": cue_hit,
            "components": components,
        }
        (candidates_dir / f"{episode}.json").write_text(
            json.dumps(candidate, sort_keys=True), encoding="utf-8"
        )
        if operation == "ADD" and goal != "ADD_NEW_COMPLETE":
            (targets_dir / f"{episode}.json").write_text(
                json.dumps(
                    {
                        "episode_id": episode,
                        "pointer_targets": [0, 1],
                        "pointer_target_positions": [0, 1],
                        "pointer_target_component_keys": [
                            f"{episode}|component|0",
                            f"{episode}|component|1",
                        ],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
    visible_manifest = tmp_path / "visible.jsonl"
    labels_manifest = tmp_path / "labels.jsonl"
    _write_jsonl(visible_manifest, visible_rows)
    _write_jsonl(labels_manifest, label_rows)
    return {
        "visible": visible_manifest,
        "labels": labels_manifest,
        "candidates": candidates_dir,
        "targets": targets_dir,
    }


def _oracle_argv(lane: dict[str, Path], output: Path, receipt: Path) -> list[str]:
    return [
        "--labels",
        str(lane["labels"]),
        "--candidates",
        str(lane["candidates"]),
        "--pointer-targets",
        str(lane["targets"]),
        "--partition",
        "val",
        "--output",
        str(output),
        "--receipt",
        str(receipt),
    ]


@pytest.fixture
def lane(tmp_path):
    return _build_gold_lane_fixture(tmp_path)


@pytest.fixture
def built_oracle(tmp_path, lane):
    calls = tmp_path / "oracle-calls.jsonl"
    receipt = tmp_path / "oracle-calls.receipt.json"
    assert gold_calls.main(_oracle_argv(lane, calls, receipt)) == 0
    rows = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    return {
        "calls": calls,
        "receipt": receipt,
        "rows": {str(row["episode_id"]): row for row in rows},
    }


# ---------------------------------------------------------------------------
# Six-class derivation from labels (ADD/REMOVE x SAME/NEW x LOCAL/COMPLETE)
# ---------------------------------------------------------------------------


def test_oracle_calls_render_all_six_goals_from_labels(built_oracle):
    rows = built_oracle["rows"]
    assert len(rows) == 6
    for index, (operation, goal) in enumerate(GOALS):
        row = rows[f"episode-{index}"]
        family = GOAL_TO_FAMILY[goal]
        operand_suffix, selection = EXPECTED_CALLS[goal]
        operand = (
            NEW_CUE_SENTINEL
            if operand_suffix == NEW_CUE_SENTINEL
            else f"episode-{index}|component|{operand_suffix}"
        )
        assert row["schema_version"] == "PETCT-PROGRAM-ORACLE-CALLS-v1.0"
        assert row["decision"] == "PREDICT"
        assert row["operation"] == operation
        assert row["family"] == family
        assert row["operand"] == operand
        assert row["goal"] == render_goal(operation, family) == goal
        assert row["protected_refs"] == dict(protected_refs_policy(operation, operand))
        assert row["oracle_selection"] == selection
        assert row["source_lane"] == "evaluator_label_only"


def test_oracle_receipt_declares_label_lane_and_binds_artifacts(built_oracle, lane):
    receipt = json.loads(built_oracle["receipt"].read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "PETCT-PROGRAM-ORACLE-CALLS-READY-v1.0"
    assert receipt["status"] == "PASS"
    assert receipt["partition"] == "val"
    assert receipt["call_count"] == 6
    assert receipt["label_lane_opened"] is True
    assert receipt["oracle_calls"] is True
    assert receipt["thesis_result"] is False
    assert Path(receipt["calls_path"]).resolve() == built_oracle["calls"].resolve()
    assert receipt["calls_sha256"] == _sha256_file(built_oracle["calls"])
    assert receipt["labels_sha256"] == _sha256_file(lane["labels"])
    assert receipt["candidates_tree_sha256"] == gold_calls._tree_sha(lane["candidates"])
    assert receipt["pointer_targets_tree_sha256"] == gold_calls._tree_sha(
        lane["targets"]
    )


def test_oracle_builder_fail_closed_on_missing_candidate(tmp_path, lane):
    (lane["candidates"] / "episode-0.json").unlink()
    with pytest.raises(LearningContractError):
        gold_calls.main(
            _oracle_argv(
                lane,
                tmp_path / "oracle-calls.jsonl",
                tmp_path / "oracle-calls.receipt.json",
            )
        )


def test_oracle_builder_fail_closed_on_empty_partition(tmp_path, lane):
    with pytest.raises(LearningContractError):
        gold_calls.main(
            [
                "--labels",
                str(lane["labels"]),
                "--candidates",
                str(lane["candidates"]),
                "--pointer-targets",
                str(lane["targets"]),
                "--partition",
                "train",
                "--output",
                str(tmp_path / "oracle-calls.jsonl"),
                "--receipt",
                str(tmp_path / "oracle-calls.receipt.json"),
            ]
        )


# ---------------------------------------------------------------------------
# label_lane_opened semantics across the two receipt verifiers
# ---------------------------------------------------------------------------


def _receipt_variant(tmp_path: Path, source: Path, name: str, **overrides) -> Path:
    document = json.loads(source.read_text(encoding="utf-8"))
    document.update(overrides)
    variant = tmp_path / ("receipt-%s.json" % name)
    variant.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    return variant


def test_oracle_call_receipt_opens_label_lane_and_tampers_fail_closed(
    tmp_path, built_oracle
):
    verified = editor_infer._verify_call_receipt(
        built_oracle["receipt"], built_oracle["calls"]
    )
    assert verified["program_source"] == "gold_oracle_ceiling"
    assert verified["label_lane_opened"] is True
    assert verified["oracle_calls"] is True
    with pytest.raises(LearningContractError):
        editor_infer._verify_call_receipt(
            _receipt_variant(
                tmp_path,
                built_oracle["receipt"],
                "lane-closed",
                label_lane_opened=False,
            ),
            built_oracle["calls"],
        )
    with pytest.raises(LearningContractError):
        editor_infer._verify_call_receipt(
            _receipt_variant(
                tmp_path,
                built_oracle["receipt"],
                "not-oracle",
                oracle_calls=False,
            ),
            built_oracle["calls"],
        )


def test_editor_receipt_label_lane_must_match_program_source(tmp_path):
    manifest = tmp_path / "editor-val.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")

    def receipt(name: str, **overrides) -> Path:
        document = {
            "schema_version": "PETCT-PROGRAM-EDITOR-PREDICTIONS-v1.0",
            "status": "PASS",
            "output_manifest": str(manifest.resolve()),
            "output_manifest_sha256": _sha256_file(manifest),
            "program_source": "gold_oracle_ceiling",
            "label_lane_opened": True,
        }
        document.update(overrides)
        path = tmp_path / ("%s.json" % name)
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
        )
        return path

    verified = evaluator._verify_editor_receipt(receipt("gold"), manifest)
    assert verified["program_source"] == "gold_oracle_ceiling"
    # gold source must declare the label lane open
    with pytest.raises(LearningContractError):
        evaluator._verify_editor_receipt(
            receipt("gold-closed", label_lane_opened=False), manifest
        )
    # predicted source must declare the label lane closed
    with pytest.raises(LearningContractError):
        evaluator._verify_editor_receipt(
            receipt(
                "predicted-open",
                program_source="predicted_compiler",
                label_lane_opened=True,
            ),
            manifest,
        )
    predicted = evaluator._verify_editor_receipt(
        receipt(
            "predicted",
            program_source="predicted_compiler",
            label_lane_opened=False,
        ),
        manifest,
    )
    assert predicted["program_source"] == "predicted_compiler"
    # unknown program source never passes
    with pytest.raises(LearningContractError):
        evaluator._verify_editor_receipt(
            receipt("unknown", program_source="hand_annotated"), manifest
        )


# ---------------------------------------------------------------------------
# J6 program-blind arm: gold conditioning must never leak into its 12-channel
# visual; the gold run only removes the abstention zeroing.
# ---------------------------------------------------------------------------


def _candidates_map(directory: Path) -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    }


@pytest.fixture
def dataset_inputs(lane, built_oracle):
    return {
        "manifest": lane["visible"],
        "candidates": _candidates_map(lane["candidates"]),
        "calls": dict(built_oracle["rows"]),
    }


def test_j6_is_program_blind_while_j9_conditions_on_gold(dataset_inputs):
    j6 = editor_infer._EditorInferenceDataset(
        dataset_inputs["manifest"],
        "val",
        dataset_inputs["candidates"],
        dataset_inputs["calls"],
        "J6",
    )
    j9 = editor_infer._EditorInferenceDataset(
        dataset_inputs["manifest"],
        "val",
        dataset_inputs["candidates"],
        dataset_inputs["calls"],
        "J9",
    )
    for index, (operation, goal) in enumerate(GOALS):
        item6, item9 = j6[index], j9[index]
        assert item6["visual"].shape[0] == 12
        assert item9["visual"].shape[0] == 13
        assert int(item6["family_id"]) == editor_infer.NULL_FAMILY_ID
        assert int(item6["operand_mode"]) == 2
        assert bool(item6["active"])
        expected_family = family_to_id(GOAL_TO_FAMILY[goal])
        assert int(item9["family_id"]) == expected_family
        if goal == "ADD_NEW_COMPLETE":
            assert int(item9["operand_mode"]) == 1
            assert not np.asarray(item9["selected_component"]).any()
        else:
            assert int(item9["operand_mode"]) == 0
            assert np.asarray(item9["selected_component"]).any()
        if operation == "ADD" and goal != "ADD_NEW_COMPLETE":
            # gold operand mask enters the J9 visual only; J6 never sees it
            assert np.array_equal(
                np.asarray(item9["visual"][12]),
                np.asarray(item9["selected_component"][0]),
            )


def test_j6_gold_run_only_removes_abstention_zeroing(dataset_inputs):
    calls = dict(dataset_inputs["calls"])
    abstained = dict(calls["episode-0"])
    abstained["decision"] = "ABSTAIN"
    calls["episode-0"] = abstained
    dataset = editor_infer._EditorInferenceDataset(
        dataset_inputs["manifest"], "val", dataset_inputs["candidates"], calls, "J6"
    )
    item = dataset[0]
    assert not bool(item["active"])
    assert int(item["family_id"]) == editor_infer.NULL_FAMILY_ID
    assert int(item["operand_mode"]) == 2
    assert int(item["support_mode"]) == 1


# ---------------------------------------------------------------------------
# Orchestration launcher dry run and argument guards
# ---------------------------------------------------------------------------


def _usable_bash() -> str | None:
    # PATH may resolve to a WSL relay shim that fails without a distro;
    # prefer the real Git Bash binary when it is installed.
    candidates = [r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            probe = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except OSError:
            continue
        if probe.returncode == 0 and "bash" in probe.stdout.lower():
            return candidate
    return None


BASH = _usable_bash()
requires_bash = pytest.mark.skipif(BASH is None, reason="bash is unavailable")


def _launcher() -> Path:
    return PROJECT / "scripts" / "orchestration" / "launch_petct_gold_call_ceiling.sh"


def _posix(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if len(text) >= 2 and text[1] == ":":
        text = "/" + text[0].lower() + text[2:]
    return text


def _run_roots() -> tuple[str, str, str]:
    base = _posix(PROJECT / "route_a" / "runs")
    return (
        base + "/PETCT-R13-MAIN-DRYTEST",
        base + "/PETCT-R13-EFFECT-VAL-DRYTEST",
        base + "/PETCT-R13-GOLD-CEILING-DRYTEST",
    )


def _run_launcher(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, str(_launcher()), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_launcher_wires_the_full_gold_chain():
    text = _launcher().read_text(encoding="utf-8")
    assert "render_petct_gold_program_calls_natural.py" in text
    assert "infer_petct_program_editor_v3.py" in text
    assert "evaluate_petct_program_v3.py" in text
    assert "--oracle-editor-predictions" in text
    assert "--oracle-editor-receipt" in text
    assert "PETCT-R13-GOLD-CEILING-" in text
    assert "--dry-run" in text


@requires_bash
def test_launcher_dry_run_emits_expected_plan():
    main, effect, gold = _run_roots()
    result = _run_launcher(main, effect, gold, "0", "1", "--dry-run")
    assert result.returncode == 0, result.stderr
    steps = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    names = [step["step"] for step in steps]
    assert names == [
        "oracle-calls",
        "J9-gold-infer",
        "J9-gold-evaluate",
        "J8-gold-infer",
        "J8-gold-evaluate",
        "J6-gold-infer",
        "J6-gold-evaluate",
        "J7-gold-infer",
        "J7-gold-evaluate",
        "arm-semantics",
    ]
    oracle = steps[0]
    assert all(isinstance(command, str) for command in oracle["commands"])
    oracle_command = " ".join(oracle["commands"])
    assert "render_petct_gold_program_calls_natural.py" in oracle_command
    assert "oracle-calls-val.jsonl" in oracle_command
    j8 = next(step for step in steps if step["step"] == "J8-gold-infer")
    assert "--compiler-checkpoint" in j8["commands"]
    assert any("models/J9C.pt" in command for command in j8["commands"])
    j9 = next(step for step in steps if step["step"] == "J9-gold-infer")
    assert "--compiler-checkpoint" not in j9["commands"]
    j6 = next(step for step in steps if step["step"] == "J6-gold-infer")
    assert any("models/J6.pt" in command for command in j6["commands"])
    assert any("oracle-calls-val.receipt.json" in command for command in j6["commands"])
    for step in steps:
        if not step["step"].endswith("-gold-evaluate"):
            continue
        commands = " ".join(step["commands"])
        assert "--editor-predictions" in commands
        assert "--editor-receipt" in commands
        assert "--oracle-editor-predictions" in commands
        assert "--oracle-editor-receipt" in commands
        assert "--audit-manifest" in commands
        assert "-gold-val.receipt.json" in commands
    semantics = next(step for step in steps if step["step"] == "arm-semantics")
    assert "program-blind" in semantics["J6"]
    assert semantics["partition"] == "val"


@requires_bash
def test_launcher_rejects_bad_arguments_fail_closed():
    main, effect, gold = _run_roots()
    # real mode with missing effect-val artifacts must exit 2, not run
    assert _run_launcher(main, effect, gold, "0", "1").returncode == 2
    # unknown trailing flag
    assert _run_launcher(main, effect, gold, "0", "1", "--bogus").returncode == 2
    # identical GPUs
    assert _run_launcher(main, effect, gold, "0", "0", "--dry-run").returncode == 2
    # non-numeric GPU
    assert _run_launcher(main, effect, gold, "a", "1", "--dry-run").returncode == 2
    # gold root outside the run base naming pattern
    assert (
        _run_launcher(
            main, effect, "/tmp/not-a-run-root/gold", "0", "1", "--dry-run"
        ).returncode
        == 2
    )
    # wrong argument count
    assert _run_launcher(main).returncode == 2
