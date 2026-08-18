"""Contract tests for the B2 2D 1-5 round trajectory ceiling runner.

B2 measures the five-round trajectory ceiling on R13 VAL: round 1 consumes
the natural-lane oracle calls (label-derived gold program), rounds 2-5
regenerate an official-simulator scribble on the CURRENT state residual and
derive a residual-driven gold call; every round advances the 3D state with
the actual frozen editor output restricted to the prompted plane.  This file
covers the residual-driven gold-call rules, the per-arm conditioning table,
the fail-closed guards, a synthetic end-to-end dry run on J9/J8, and the
launcher dry-run plan.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
for directory in (
    SCRIPTS,
    SCRIPTS / "data",
    SCRIPTS / "editor",
    SCRIPTS / "evaluation",
    SCRIPTS / "common",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common.petct_program_contract import (  # noqa: E402
    NEW_CUE_SENTINEL,
    family_to_id,
    protected_refs_policy,
    render_goal,
)
from common.petct_program_learning import (  # noqa: E402
    LearningContractError,
    _sha256_file,
)
from common.petct_program_models import (  # noqa: E402
    NULL_FAMILY_ID,
    ProgramCompilerNet,
    ProgramEditorUNet2D,
)
from run_petct_w21_official_test import _verify_seal  # noqa: E402

try:
    import run_petct_b2_trajectory_ceiling as runner  # noqa: E402
except ImportError:  # RED phase: the runner does not exist yet
    runner = None


# ---------------------------------------------------------------------------
# Pure-function units: residual-driven gold call + per-arm conditioning
# ---------------------------------------------------------------------------


def _component(position, key, overlap, distance):
    return {
        "candidate_position": position,
        "component_key": key,
        "cue_overlap_voxels": float(overlap),
        "distance_from_cue_mm": float(distance),
    }


def test_residual_driven_gold_call_add_picks_max_overlap_component():
    components = [
        _component(0, "k0", overlap=0.0, distance=1.0),
        _component(1, "k1", overlap=3.0, distance=9.0),
        _component(2, "k2", overlap=3.0, distance=2.0),
    ]
    call = runner._gold_call_residual_driven("ADD", components, None)
    # equal overlap -> nearest cue distance wins
    assert call["family"] == "COMPLETE_EXISTING"
    assert call["operand"] == "k2"
    assert call["goal"] == render_goal("ADD", "COMPLETE_EXISTING")
    assert call["protected_refs"] == dict(protected_refs_policy("ADD", "k2"))
    assert call["policy"] == "residual_driven_2d"


def test_residual_driven_gold_call_add_without_overlap_creates_new():
    components = [_component(0, "k0", overlap=0.0, distance=1.0)]
    call = runner._gold_call_residual_driven("ADD", components, None)
    assert call["family"] == "CREATE_NEW"
    assert call["operand"] == NEW_CUE_SENTINEL
    assert call["goal"] == render_goal("ADD", "CREATE_NEW")


def test_residual_driven_gold_call_remove_uses_cue_hit_and_fails_closed():
    components = [_component(0, "k0", overlap=2.0, distance=0.0)]
    call = runner._gold_call_residual_driven("REMOVE", components, 0)
    assert call["family"] == "DELETE_COMPONENT"
    assert call["operand"] == "k0"
    assert call["goal"] == render_goal("REMOVE", "DELETE_COMPONENT")
    with pytest.raises(LearningContractError):
        runner._gold_call_residual_driven("REMOVE", components, None)


def test_residual_driven_gold_call_tie_break_is_position():
    components = [
        _component(1, "k1", overlap=4.0, distance=5.0),
        _component(0, "k0", overlap=4.0, distance=5.0),
    ]
    call = runner._gold_call_residual_driven("ADD", components, None)
    assert call["operand"] == "k0"


def test_arm_conditioning_table():
    family_id = family_to_id("GROW_LOCAL")
    assert runner._arm_conditioning("J6", family_id, 0) == (
        NULL_FAMILY_ID,
        2,
        12,
        False,
    )
    assert runner._arm_conditioning("J7", family_id, 0) == (family_id, 0, 13, False)
    assert runner._arm_conditioning("J8", family_id, 0) == (NULL_FAMILY_ID, 2, 13, True)
    assert runner._arm_conditioning("J9", family_id, 1) == (family_id, 1, 13, False)
    with pytest.raises(LearningContractError):
        runner._arm_conditioning("J1", family_id, 0)


def test_case_row_partition_guard_fails_closed():
    assert (
        runner._require_val_case(
            {
                "case_id": "c1",
                "partition": "val",
                "truth_materialization": "MATERIALIZED_AUTHORIZED",
            }
        )
        == "c1"
    )
    with pytest.raises(LearningContractError):
        runner._require_val_case(
            {
                "case_id": "c1",
                "partition": "test",
                "truth_materialization": "LOCKED_UNREAD",
            }
        )
    with pytest.raises(LearningContractError):
        runner._require_val_case(
            {
                "case_id": "c1",
                "partition": "val",
                "truth_materialization": "LOCKED_UNREAD",
            }
        )


# ---------------------------------------------------------------------------
# Synthetic VAL fixture: two episodes (ADD + REMOVE), small 3D volumes,
# fake pinned simulator, random-init editor/compiler checkpoints.
# ---------------------------------------------------------------------------

SHAPE = (24, 24, 20)
Z_CENTER = 10
CROP_SIZE = 32
FIELD_MM = 32.0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_nifti(path: Path, array: np.ndarray) -> None:
    image = nib.Nifti1Image(array.astype(np.float32), np.eye(4), header=None)
    image.header.set_zooms((1.0, 1.0, 1.0))
    nib.save(image, str(path))


def _tree_sha(directory: Path) -> str:
    rows = [
        (path.relative_to(directory).as_posix(), _sha256_file(path))
        for path in sorted(value for value in directory.rglob("*") if value.is_file())
    ]
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _build_b2_fixture(tmp_path: Path) -> dict[str, Path]:
    """Tiny VAL lane with one ADD and one REMOVE episode on a 24x24x20 grid.

    GT is a 4x4x4 blob; the ADD episode's M0 misses its corner voxel and the
    REMOVE episode's M0 carries one extra FP voxel, so both round-1 and the
    residual-driven rounds have non-empty residuals.
    """
    rng = np.random.default_rng(3407)
    cases_dir = tmp_path / "cases"
    candidates_dir = tmp_path / "candidates"
    targets_dir = tmp_path / "targets"
    for directory in (cases_dir, candidates_dir, targets_dir):
        directory.mkdir()
    gt = np.zeros(SHAPE, dtype=np.uint8)
    gt[9:13, 9:13, 8:12] = 1
    gt_voxel = (12, 12, Z_CENTER)
    fp_voxel = (15, 15, Z_CENTER)
    m0_add = gt.copy()
    m0_add[gt_voxel] = 0
    m0_remove = gt.copy()
    m0_remove[fp_voxel] = 1
    case_rows, episode_rows, label_rows, rich_rows = [], [], [], []
    candidate_map = {}
    target_map = {}
    for index, (operation, goal, m0, scribble_xyz) in enumerate(
        (
            ("ADD", "ADD_SAME_LOCAL", m0_add, [gt_voxel]),
            ("REMOVE", "REMOVE_SAME_LOCAL", m0_remove, [fp_voxel]),
        )
    ):
        case_id = f"case-{index}"
        episode = f"episode-{index}"
        pet = np.full(SHAPE, 3.0, dtype=np.float32) + rng.normal(
            0.0, 0.01, SHAPE
        ).astype(np.float32)
        ct = np.full(SHAPE, 5.0, dtype=np.float32)
        pet_path, ct_path = (
            cases_dir / f"{case_id}-pet.nii.gz",
            cases_dir / f"{case_id}-ct.nii.gz",
        )
        gt_path, m0_path = (
            cases_dir / f"{case_id}-gt.nii.gz",
            cases_dir / f"{case_id}-m0.nii.gz",
        )
        _write_nifti(pet_path, pet)
        _write_nifti(ct_path, ct)
        _write_nifti(gt_path, gt)
        _write_nifti(m0_path, m0)
        case_rows.append(
            {
                "case_id": case_id,
                "patient_id": f"patient-{index}",
                "held_out_fold": index,
                "partition": "val",
                "ct_path": str(ct_path),
                "pet_path": str(pet_path),
                "gt_path": str(gt_path),
                "ct_bytes": ct_path.stat().st_size,
                "pet_bytes": pet_path.stat().st_size,
                "gt_bytes": gt_path.stat().st_size,
                "ct_sha256": _sha(ct_path),
                "pet_sha256": _sha(pet_path),
                "gt_sha256": _sha(gt_path),
                "nifti_shape": list(SHAPE),
                "truth_materialization": "MATERIALIZED_AUTHORIZED",
            }
        )
        # Identity crop: field == N * spacing with the center at the half-grid
        # offset makes crop index i sample original voxel i exactly, so the
        # fake crop-space tensors align with the 3D volumes voxel-for-voxel.
        blob = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        blob[9:13, 9:13] = 1
        m0_slice = blob.copy()
        gt_slice = blob.copy()
        if operation == "ADD":
            m0_slice[12, 12] = 0
            target_slice = np.zeros_like(blob)
            target_slice[12, 12] = 1
        else:
            m0_slice[15, 15] = 1
            target_slice = np.zeros_like(blob)
            target_slice[15, 15] = 1
        visual = np.zeros((17, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        visual[10:15] = 0.5
        cue_fg = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        cue_bg = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
        (cue_fg if operation == "ADD" else cue_bg)[12, 12] = 1
        visual[15], visual[16] = cue_fg, cue_bg
        visible_path = tmp_path / f"{episode}-visible.npz"
        np.savez_compressed(
            visible_path,
            visual=visual,
            m0=m0_slice,
            scribble=cue_fg + cue_bg,
            cue_fg=cue_fg,
            cue_bg=cue_bg,
            spacing_xy=np.asarray([1.0, 1.0], dtype=np.float32),
        )
        evaluation_path = tmp_path / f"{episode}-evaluation.npz"
        np.savez_compressed(
            evaluation_path,
            target=target_slice,
            authorized=target_slice,
            gt=gt_slice,
        )
        episode_rows.append(
            {
                "schema_version": "PETCT-PROGRAM-INFERENCE-MANIFEST-v1.0",
                "episode_id": episode,
                "partition": "val",
                "operation": operation,
                "visible_npz": str(visible_path),
                "visible_sha256": _sha(visible_path),
            }
        )
        rich_rows.append(
            {
                "episode_id": episode,
                "case_id": case_id,
                "patient_id": f"patient-{index}",
                "partition": "val",
                "strategy": "centerline",
                "goal": goal,
                "operation": operation,
                "visible_npz": str(visible_path),
                "visible_sha256": _sha(visible_path),
                "evaluation_npz": str(evaluation_path),
                "evaluation_sha256": _sha(evaluation_path),
                "center_z": Z_CENTER,
                "experiment_config_sha256": "c" * 64,
                "learning_split_sha256": "a" * 64,
                "geometry": {
                    "crop_center_xy_voxel": [15.5, 15.5],
                    "crop_field_mm": FIELD_MM,
                    "output_size_px": CROP_SIZE,
                    "original_spacing_xy": [1.0, 1.0],
                    "output_spacing_xy": [1.0, 1.0],
                    "image_interpolation": "linear",
                    "mask_interpolation": "nearest",
                },
                "source_evaluation": {
                    "gt_path": str(gt_path),
                    "gt_sha256": _sha(gt_path),
                    "m0_path": str(m0_path),
                    "m0_sha256": _sha(m0_path),
                    "authorized_path": str(m0_path),
                    "authorized_sha256": _sha(m0_path),
                    "center_z": Z_CENTER,
                    "scribble_coordinates_xyz": [list(value) for value in scribble_xyz],
                },
            }
        )
        label_rows.append(
            {
                "schema_version": "PETCT-PROGRAM-LABEL-MANIFEST-v1.0",
                "episode_id": episode,
                "case_id": case_id,
                "patient_id": f"patient-{index}",
                "partition": "val",
                "goal": goal,
                "operation": operation,
                "evaluation_npz": str(evaluation_path),
                "evaluation_sha256": _sha(evaluation_path),
                "learning_split_sha256": "a" * 64,
            }
        )
        key = f"{episode}|component|0"
        candidate_map[episode] = {
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
                    "prompted_slice_mask": (m0_slice > 0).astype(np.uint8).tolist(),
                }
            ],
        }
        (candidates_dir / f"{episode}.json").write_text(
            json.dumps(candidate_map[episode], sort_keys=True), encoding="utf-8"
        )
        if operation == "ADD":
            target_map[episode] = {
                "episode_id": episode,
                "pointer_targets": [0],
                "pointer_target_positions": [0],
                "pointer_target_component_keys": [key],
            }
            (targets_dir / f"{episode}.json").write_text(
                json.dumps(target_map[episode], sort_keys=True), encoding="utf-8"
            )
    case_manifest = tmp_path / "case-manifest.jsonl"
    episodes_manifest = tmp_path / "episodes.jsonl"
    rich_manifest = tmp_path / "rich.jsonl"
    labels_manifest = tmp_path / "labels.jsonl"
    _write_jsonl(case_manifest, case_rows)
    _write_jsonl(episodes_manifest, episode_rows)
    _write_jsonl(rich_manifest, rich_rows)
    _write_jsonl(labels_manifest, label_rows)

    # Natural-lane oracle calls through the real frozen builder.
    import render_petct_gold_program_calls_natural as natural_calls  # noqa: E402

    oracle_calls = tmp_path / "oracle-calls.jsonl"
    oracle_receipt = tmp_path / "oracle-calls.receipt.json"
    assert (
        natural_calls.main(
            [
                "--labels",
                str(labels_manifest),
                "--candidates",
                str(candidates_dir),
                "--pointer-targets",
                str(targets_dir),
                "--partition",
                "val",
                "--output",
                str(oracle_calls),
                "--receipt",
                str(oracle_receipt),
            ]
        )
        == 0
    )

    # Lineage receipt (same shape as the R13 mainline fixture).
    oof_ready = tmp_path / "M0_V6_FIVEFOLD_OOF_READY.json"
    learning_split = tmp_path / "learning_split.json"
    experiment_config = tmp_path / "experiment_v3.json"
    for path in (oof_ready, learning_split, experiment_config):
        path.write_text("{}\n", encoding="utf-8")
    lineage = tmp_path / "lineage-receipt.json"
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
                "oof_ready": {
                    "path": str(oof_ready.resolve()),
                    "bytes": oof_ready.stat().st_size,
                    "sha256": _sha(oof_ready),
                },
                "learning_split": {
                    "path": str(learning_split.resolve()),
                    "bytes": learning_split.stat().st_size,
                    "sha256": _sha(learning_split),
                },
                "experiment_config": {
                    "path": str(experiment_config.resolve()),
                    "bytes": experiment_config.stat().st_size,
                    "sha256": _sha(experiment_config),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # Fake pinned official simulator (sha pinned by monkeypatch in the tests).
    simulator_path = tmp_path / "fake_simulate_scribbles.py"
    simulator_path.write_text(
        "import numpy as np\n"
        "def simulate_scribble_from_label(residual, strategy, seed):\n"
        "    coordinates = np.argwhere(np.asarray(residual) > 0)[:6].tolist()\n"
        "    return (coordinates, None, len(coordinates))\n",
        encoding="utf-8",
    )

    # Editor checkpoints for the four arms + compiler for J8.
    lineage_sha = _sha(lineage)
    episodes_sha = _sha(episodes_manifest)
    candidates_sha = _tree_sha(candidates_dir)
    checkpoints: dict[str, Path] = {}
    torch.manual_seed(3407)
    compiler_model = ProgramCompilerNet(include_repair=True)
    compiler_path = tmp_path / "compiler.pt"
    torch.save(
        {
            "schema_version": "PETCT-PROGRAM-COMPILER-CHECKPOINT-v1.0",
            "architecture_id": "matched_legal_component_program_v1",
            "episodes_sha256": episodes_sha,
            "candidates_tree_sha256": candidates_sha,
            "source_m0_lineage": "M0_V6_FIVEFOLD_OOF",
            "lineage_receipt_sha256": lineage_sha,
            "hyperparameters": {"include_repair": True},
            "state_dict": compiler_model.state_dict(),
        },
        compiler_path,
    )
    compiler_sha = _sha(compiler_path)
    for arm, channels, conditioner in (
        ("J6", 12, "program"),
        ("J7", 13, "program"),
        ("J8", 13, "continuous"),
        ("J9", 13, "program"),
    ):
        model = ProgramEditorUNet2D(visual_channels=channels, conditioner=conditioner)
        # Deep-negative output bias pins every smoke editor to an identity
        # edit (empty delta): trajectories stay on the labeled residuals, so
        # later-round scribbles never wander into degenerate crop boundaries.
        model.output.bias.data.fill_(-6.0)
        payload = {
            "schema_version": "PETCT-PROGRAM-EDITOR-CHECKPOINT-v1.0",
            "episodes_sha256": episodes_sha,
            "candidates_tree_sha256": candidates_sha,
            "source_m0_lineage": "M0_V6_FIVEFOLD_OOF",
            "lineage_receipt_sha256": lineage_sha,
            "arm": arm,
            "state_dict": model.state_dict(),
        }
        if arm == "J8":
            payload["compiler_checkpoint_sha256"] = compiler_sha
        path = tmp_path / f"editor-{arm}.pt"
        torch.save(payload, path)
        checkpoints[arm] = path
    return {
        "case_manifest": case_manifest,
        "episodes": episodes_manifest,
        "rich": rich_manifest,
        "candidates": candidates_dir,
        "oracle_calls": oracle_calls,
        "oracle_receipt": oracle_receipt,
        "lineage": lineage,
        "simulator": simulator_path,
        "editor_checkpoints": checkpoints,
        "compiler_checkpoint": compiler_path,
    }


def _runner_argv(fixture, arm: str, output: Path) -> list[str]:
    argv = [
        "--case-manifest",
        str(fixture["case_manifest"]),
        "--episodes",
        str(fixture["episodes"]),
        "--rich-manifest",
        str(fixture["rich"]),
        "--candidates",
        str(fixture["candidates"]),
        "--oracle-calls",
        str(fixture["oracle_calls"]),
        "--oracle-receipt",
        str(fixture["oracle_receipt"]),
        "--editor-checkpoint",
        str(fixture["editor_checkpoints"][arm]),
        "--lineage-receipt",
        str(fixture["lineage"]),
        "--partition",
        "val",
        "--official-simulator",
        str(fixture["simulator"]),
        "--field-mm",
        str(FIELD_MM),
        "--output-size",
        str(CROP_SIZE),
        "--device",
        "cpu",
        "--output",
        str(output),
    ]
    if arm == "J8":
        argv += ["--compiler-checkpoint", str(fixture["compiler_checkpoint"])]
    return argv


# ---------------------------------------------------------------------------
# Synthetic end-to-end dry runs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def b2_fixture(tmp_path_factory):
    return _build_b2_fixture(tmp_path_factory.mktemp("b2-fixture"))


def _run_arm(fixture, arm: str, tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setattr(runner, "OFFICIAL_SIMULATOR_SHA256", _sha(fixture["simulator"]))
    output = tmp_path / f"trajectory-{arm}.json"
    assert runner.main(_runner_argv(fixture, arm, output)) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    _verify_seal(report, "arm_sha256", label="trajectory arm report")
    return report


@pytest.mark.parametrize("arm", ["J9", "J8"])
def test_synthetic_trajectory_dry_run(tmp_path, b2_fixture, monkeypatch, arm):
    report = _run_arm(b2_fixture, arm, tmp_path, monkeypatch)
    assert report["schema_version"] == runner.ARM_SCHEMA
    assert report["arm"] == arm
    assert report["partition"] == "val"
    assert report["aggregates"]["episode_count"] == 2
    episodes = report["episodes"]
    assert len(episodes) == 2
    oracle_rows = {
        json.loads(line)["episode_id"]: json.loads(line)
        for line in b2_fixture["oracle_calls"].read_text(encoding="utf-8").splitlines()
    }
    for episode in episodes:
        rounds = episode["rounds"]
        assert len(rounds) == 5
        first = rounds[0]
        assert first["round"] == 1
        assert first["gold_call"]["policy"] == "natural_label_oracle"
        oracle = oracle_rows[episode["episode_id"]]
        assert first["gold_call"]["family"] == oracle["family"]
        assert first["gold_call"]["operand"] == oracle["operand"]
        assert first["crop_plane"]["dice_before"] is not None
        assert first["crop_plane"]["delta_dice"] is not None
        for round_record in rounds[1:]:
            if round_record["correction"] is None:
                assert round_record["crop_plane"]["delta_dice"] is None
                continue
            if round_record["correction"]["coordinates"]:
                assert round_record["gold_call"]["policy"] == "residual_driven_2d"
                assert round_record["crop_plane"]["delta_dice"] is not None
                operation = round_record["operation"]
                assert round_record["gold_call"]["goal"] == render_goal(
                    operation, round_record["gold_call"]["family"]
                )
    aggregates = report["aggregates"]
    assert set(aggregates["per_round_mean_crop_delta_dice"]) == {
        "1",
        "2",
        "3",
        "4",
        "5",
    }
    assert aggregates["patient_count"] == 2


def test_runner_fail_closed_guards(tmp_path, b2_fixture):
    output = tmp_path / "trajectory.json"
    # output already exists
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        runner.main(_runner_argv(b2_fixture, "J9", output))
    output.unlink()
    # tampered oracle receipt: label lane closed
    receipt = json.loads(b2_fixture["oracle_receipt"].read_text(encoding="utf-8"))
    receipt["label_lane_opened"] = False
    tampered = tmp_path / "tampered.receipt.json"
    tampered.write_text(json.dumps(receipt), encoding="utf-8")
    argv = _runner_argv(b2_fixture, "J9", output)
    argv[argv.index("--oracle-receipt") + 1] = str(tampered)
    with pytest.raises(SystemExit):
        runner.main(argv)
    # unknown arm label fails closed at the checkpoint guard
    with pytest.raises(LearningContractError):
        runner._require_arm_from_checkpoint("J5")


# ---------------------------------------------------------------------------
# Launcher dry run and argument guards
# ---------------------------------------------------------------------------


def _usable_bash() -> str | None:
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
    return (
        PROJECT / "scripts" / "orchestration" / "launch_petct_b2_trajectory_ceiling.sh"
    )


def _posix(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if len(text) >= 2 and text[1] == ":":
        text = "/" + text[0].lower() + text[2:]
    return text


def _run_roots() -> tuple[str, str, str, str, str]:
    base = _posix(PROJECT / "route_a" / "runs")
    return (
        base + "/PETCT-R13-MAIN-DRYTEST",
        base + "/PETCT-R13-EFFECT-VAL-DRYTEST",
        base + "/PETCT-R13-GOLD-CEILING-DRYTEST",
        base + "/PETCT-R13-B2-TRAJECTORY-DRYTEST",
        base + "/PETCT-R13-MAIN-DRYTEST/cases.jsonl",
    )


def _run_launcher(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [BASH, str(_launcher()), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_launcher_wires_the_trajectory_chain():
    text = _launcher().read_text(encoding="utf-8")
    assert "run_petct_b2_trajectory_ceiling.py" in text
    assert "oracle-calls-val.jsonl" in text
    assert "tensors-rich.jsonl" in text
    assert "--compiler-checkpoint" in text
    assert "PETCT-R13-B2-TRAJECTORY-" in text
    assert "--dry-run" in text


@requires_bash
def test_launcher_dry_run_emits_expected_plan():
    main, effect, gold, b2, cases = _run_roots()
    result = _run_launcher(main, effect, gold, b2, cases, "0", "1", "--dry-run")
    assert result.returncode == 0, result.stderr
    steps = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    names = [step["step"] for step in steps]
    assert names == [
        "J9-trajectory",
        "J8-trajectory",
        "J6-trajectory",
        "J7-trajectory",
        "arm-semantics",
    ]
    j9 = steps[0]
    commands = " ".join(j9["commands"])
    assert "run_petct_b2_trajectory_ceiling.py" in commands
    assert "models/J9.pt" in commands
    assert "--compiler-checkpoint" not in commands
    j8 = next(step for step in steps if step["step"] == "J8-trajectory")
    j8_commands = " ".join(j8["commands"])
    assert "models/J8.pt" in j8_commands
    assert "--compiler-checkpoint" in j8_commands
    assert "models/J9C.pt" in j8_commands
    for step in steps:
        if not step["step"].endswith("-trajectory"):
            continue
        joined = " ".join(step["commands"])
        assert "oracle-calls-val.jsonl" in joined
        assert "tensors-rich.jsonl" in joined
        assert "--partition val" in joined
    semantics = next(step for step in steps if step["step"] == "arm-semantics")
    assert "residual_driven" in semantics["rounds_2_to_5"]
    assert semantics["domain"] == "2d_prompted_plane"


@requires_bash
def test_launcher_rejects_bad_arguments_fail_closed():
    main, effect, gold, b2, cases = _run_roots()
    # real mode with missing artifacts must exit 2, not run
    assert _run_launcher(main, effect, gold, b2, cases, "0", "1").returncode == 2
    assert (
        _run_launcher(main, effect, gold, b2, cases, "0", "1", "--bogus").returncode
        == 2
    )
    assert (
        _run_launcher(main, effect, gold, b2, cases, "0", "0", "--dry-run").returncode
        == 2
    )
    assert (
        _run_launcher(
            main, effect, gold, "/tmp/not-a-run-root", cases, "0", "1", "--dry-run"
        ).returncode
        == 2
    )
    assert _run_launcher(main).returncode == 2
