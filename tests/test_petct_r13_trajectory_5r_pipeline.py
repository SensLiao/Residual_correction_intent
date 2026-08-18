"""End-to-end and parity tests for the R13 five-round trajectory corpus chain.

Runs the real five-round builder, the five-round tensor and program-manifest
materializers, and the unchanged candidate/pointer-target materializers on a
synthetic two-case fixture, then diffs round-0 rows and round-0 exclusions
against the frozen single-round builder field by field.  The official
simulator and the natural OOF binding are replaced with deterministic doubles;
every frozen input file (config, split, residual manifest, ready receipts) is
validated by the same code the server will run.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "data"))

import build_petct_r13_trajectory_5r as trajectory_builder  # noqa: E402
import build_petct_scribble_dataset as single_round_builder  # noqa: E402
import materialize_petct_r13_trajectory_5r_programs as trajectory_programs  # noqa: E402
import materialize_petct_r13_trajectory_5r_tensors as trajectory_tensors  # noqa: E402
from common.petct_program_learning import (  # noqa: E402
    INFERENCE_MANIFEST_ALLOWED_FIELDS,
)
from common.petct_trajectory_lineage import (  # noqa: E402
    TRAJECTORY_DATASET_ID,
    issue_trajectory_lineage_receipt,
    seal_trajectory_data_ready,
    validate_r13_trajectory_rows,
    validate_trajectory_data_ready,
    validate_trajectory_lineage_receipt,
)
from data.build_petct_scribble_dataset import _canonical_hash  # noqa: E402
from data.build_petct_scribble_episode import _sha256_json  # noqa: E402
from common.petct_learning import sha256_file  # noqa: E402


# --- deterministic doubles ---------------------------------------------------


def _fake_simulator(mask, *, strategy, seed):
    """Sparse top-left 3x3 cue on the widest residual slice (y-major order)."""
    del strategy, seed
    z = max(range(mask.shape[2]), key=lambda index: int(mask[:, :, index].sum()))
    plane = np.asarray(mask[:, :, z]) > 0
    from scipy import ndimage

    labels, count = ndimage.label(plane, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.reshape(-1))
    areas = np.pad(areas, (0, count + 1 - len(areas)))
    areas[0] = 0
    label_id = int(np.argmax(areas))
    coords = np.argwhere(labels == label_id)
    coords = coords[np.lexsort((coords[:, 0], coords[:, 1]))][:9]
    result = [[int(x), int(y), int(z)] for x, y in coords]
    return result, True, len(result)


_NATURAL_KEYS = (
    "oof_ready_sha256",
    "m0_sha256",
    "foreground_probability_sha256",
    "checkpoint_sha256",
    "plans_sha256",
    "dataset_json_sha256",
    "source_tree_sha256",
    "splits_final_sha256",
    "preprocess_ready_sha256",
    "full_train_ready_sha256",
    "fold_receipt_sha256",
    "input_ct_sha256",
    "input_pet_sha256",
    "input_gt_sha256",
)


def _natural_provenance(input_gt_sha256: str) -> dict:
    unsigned = {
        "kind": "patient_excluded_oof",
        "schema_version": "PETCT-M0-V6-OOF-READY-v1.0",
        "contract_version": "PETCT-M0-V6-OOF-READY-v1.0",
        "operation": "ADD",
        "held_out_fold": 0,
        **{key: "f" * 64 for key in _NATURAL_KEYS},
        "input_gt_sha256": input_gt_sha256,
    }
    return {**unsigned, "binding_sha256": _sha256_json(unsigned)}


def _fake_oof_validation(path):
    del path
    return {}


def _fake_provenance_loader(path, *, expected_commit, expected_sha256, runtime_manifest=None):
    del path, expected_commit, expected_sha256, runtime_manifest
    fake = _fake_simulator
    fake._petct_official_provenance = {  # type: ignore[attr-defined]
        "repository": "lab-midas/autoPETV",
        "commit": single_round_builder.resolve_scribble_generation_contract(
            _experiment_config_document()
        )["official_commit"],
        "file_sha256": single_round_builder.resolve_scribble_generation_contract(
            _experiment_config_document()
        )["simulator_file_sha256"],
        "relative_path": "interactive/simulate_scribbles.py",
        "provenance_mode": "CLEAN_GIT_WORKTREE",
        "git_worktree": "CLEAN_FOR_SIMULATOR_FILE",
    }
    return fake


_EXPERIMENT_CONFIG_CACHE = None


def _experiment_config_document() -> dict:
    global _EXPERIMENT_CONFIG_CACHE
    if _EXPERIMENT_CONFIG_CACHE is None:
        _EXPERIMENT_CONFIG_CACHE = json.loads(
            (PROJECT / "configs" / "petct_route_a_experiment.json").read_text(
                encoding="utf-8"
            )
        )
    return _EXPERIMENT_CONFIG_CACHE


# --- fixture -----------------------------------------------------------------


def _write_nifti(path: Path, array: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(array.astype(np.float32), np.eye(4)), str(path))


def _bucket(case_ids, patient_ids):
    cases = sorted(case_ids)
    patients = sorted(patient_ids)
    return {
        "case_count": len(cases),
        "patient_count": len(patients),
        "case_ids": cases,
        "patient_ids": patients,
        "case_ids_sha256": _canonical_hash(cases),
        "patient_ids_sha256": _canonical_hash(patients),
    }


def _file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _volumes(tmp_path: Path):
    shape = (24, 24, 6)
    # case-a: lesion A (slabs on z=1..2, M0 covers the left half), lesion B
    # (fully missed NEW component on z=4), FP blob C on z=3.
    gt_a = np.zeros(shape, dtype=np.uint8)
    gt_a[3:21, 3:21, 1:3] = 1
    gt_a[4:14, 4:14, 4] = 1
    m0_a = np.zeros(shape, dtype=np.uint8)
    m0_a[3:12, 3:21, 1:3] = 1
    m0_a[14:20, 4:10, 3] = 1
    # case-b: perfect M0 (both residuals empty -> excluded identically).
    gt_b = np.zeros(shape, dtype=np.uint8)
    gt_b[3:12, 3:18, 1:2] = 1
    m0_b = gt_b.copy()
    volumes = {"case-a": (gt_a, m0_a), "case-b": (gt_b, m0_b)}
    paths = {}
    for case_id, (gt, m0) in volumes.items():
        ct = np.zeros(shape, dtype=np.float32)
        pet = np.zeros(shape, dtype=np.float32)
        fn = (gt > 0) & ~(m0 > 0)
        fp = (m0 > 0) & ~(gt > 0)
        written = {}
        for name, array in (
            ("ct", ct),
            ("pet", pet),
            ("gt", gt),
            ("m0", m0),
            ("fn", fn),
            ("fp", fp),
        ):
            path = tmp_path / ("%s-%s.nii.gz" % (case_id, name))
            _write_nifti(path, array)
            written[name] = path
        paths[case_id] = written
    return paths


def _learning_split(tmp_path: Path) -> Path:
    document = {
        "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
        "status": "FROZEN_BEFORE_MODEL_SELECTION",
        "split_unit": "patient",
        "patient_count": 2,
        "case_count": 2,
        "case_counts": {"train": 2, "val": 0, "test": 0},
        "patients": [
            {"patient_id": "patient-a", "partition": "train", "case_ids": ["case-a"]},
            {"patient_id": "patient-b", "partition": "train", "case_ids": ["case-b"]},
        ],
    }
    path = tmp_path / "learning-split.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _residual_ready(tmp_path: Path, residual_manifest: Path, oof_ready: Path) -> Path:
    document = {
        "schema_version": "PETCT-FN-FP-RESIDUAL-READY-v2.0",
        "status": "PASS",
        "phase": "FN_FP_RESIDUAL_DERIVATION",
        "selected_partitions": ["train"],
        "residual_manifest": _file_record(residual_manifest),
        "oof_ready": _file_record(oof_ready),
        "cohort": {
            "source": _bucket(["case-a", "case-b"], ["patient-a", "patient-b"]),
            "selected_source": _bucket(
                ["case-a", "case-b"], ["patient-a", "patient-b"]
            ),
            "generated": _bucket(["case-a", "case-b"], ["patient-a", "patient-b"]),
            "fn_positive": _bucket(["case-a"], ["patient-a"]),
            "zero_fn": _bucket(["case-b"], ["patient-b"]),
            "fp_positive": _bucket(["case-a"], ["patient-a"]),
            "zero_fp": _bucket(["case-b"], ["patient-b"]),
            "excluded": _bucket([], []),
        },
    }
    path = tmp_path / "RESIDUAL_READY.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, monkeypatch):
    """Synthetic two-case natural-lane fixture with deterministic doubles."""

    volume_paths = _volumes(tmp_path)
    case_provenance = {
        case_id: _natural_provenance(sha256_file(written["gt"]))
        for case_id, written in volume_paths.items()
    }

    def fake_binding(*args, **kwargs):
        del args
        case_id = kwargs.get("case_id")
        if case_id not in case_provenance:
            raise AssertionError("unexpected case in fake binding: %r" % case_id)
        return copy.deepcopy(case_provenance[case_id])

    monkeypatch.setattr(
        trajectory_builder, "load_official_simulator", _fake_provenance_loader
    )
    monkeypatch.setattr(
        single_round_builder, "load_official_simulator", _fake_provenance_loader
    )
    monkeypatch.setattr(
        trajectory_builder, "validate_m0_v6_oof_ready", _fake_oof_validation
    )
    monkeypatch.setattr(
        trajectory_builder,
        "build_natural_oof_binding_from_validated",
        fake_binding,
    )
    import baseline.validate_petct_m0_oof as oof_module

    monkeypatch.setattr(
        oof_module, "build_natural_oof_binding_from_validated", fake_binding
    )
    import common.petct_mainline_lineage as mainline

    monkeypatch.setattr(mainline, "validate_m0_v6_oof_ready", _fake_oof_validation)
    import common.petct_trajectory_lineage as trajectory_lineage

    monkeypatch.setattr(
        trajectory_lineage, "validate_m0_v6_oof_ready", _fake_oof_validation
    )

    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(_experiment_config_document()), encoding="utf-8"
    )
    split_path = _learning_split(tmp_path)
    oof_ready = tmp_path / "oof-ready.json"
    oof_ready.write_text(
        json.dumps({"schema_version": "PETCT-M0-V6-OOF-READY-v1.0"}), encoding="utf-8"
    )
    residual_rows = []
    for case_id, patient_id in (("case-a", "patient-a"), ("case-b", "patient-b")):
        written = volume_paths[case_id]
        fn = nib.load(str(written["fn"])).get_fdata()
        fp = nib.load(str(written["fp"])).get_fdata()
        row = {
            "case_id": case_id,
            "patient_id": patient_id,
            "partition": "train",
            "held_out_fold": 0,
            "fn_voxels": int((fn > 0).sum()),
            "fp_voxels": int((fp > 0).sum()),
            "m0_provenance": copy.deepcopy(case_provenance[case_id]),
            "learning_split_sha256": sha256_file(split_path),
        }
        for name, path in written.items():
            row[name + "_path"] = str(path)
            row[name + "_sha256"] = sha256_file(path)
        residual_rows.append(row)
    residual_manifest = tmp_path / "residuals.jsonl"
    residual_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in residual_rows), encoding="utf-8"
    )
    residual_ready = _residual_ready(tmp_path, residual_manifest, oof_ready)
    runtime_manifest = PROJECT / "protocols" / "autopetv_protocol_runtime.json"
    return {
        "config": config_path,
        "split": split_path,
        "oof_ready": oof_ready,
        "residual_manifest": residual_manifest,
        "residual_ready": residual_ready,
        "runtime_manifest": runtime_manifest,
    }


def _common_builder_args(fixture: dict, root: Path) -> list[str]:
    return [
        "--residual-manifest",
        str(fixture["residual_manifest"]),
        "--residual-ready",
        str(fixture["residual_ready"]),
        "--official-simulator",
        str(root / "simulator.py"),
        "--official-runtime-manifest",
        str(fixture["runtime_manifest"]),
        "--experiment-config",
        str(fixture["config"]),
        "--learning-split",
        str(fixture["split"]),
        "--partitions",
        "train",
        "--strategy-mode",
        "primary",
        "--seed",
        "42",
        "--lane",
        "natural",
        "--oof-ready",
        str(fixture["oof_ready"]),
    ]


def _run_trajectory_builder(fixture: dict, tmp_path: Path) -> dict:
    root = tmp_path / "trajectory"
    argv = [
        *_common_builder_args(fixture, root),
        "--visible-root",
        str(root / "visible"),
        "--evaluation-root",
        str(root / "evaluation"),
        "--authorized-root",
        str(root / "authorized"),
        "--state-root",
        str(root / "states"),
        "--output-manifest",
        str(root / "episodes.jsonl"),
        "--trajectories",
        str(root / "trajectories.jsonl"),
        "--exclusions",
        str(root / "exclusions.jsonl"),
        "--ready-receipt",
        str(root / "EPISODES_READY.json"),
    ]
    assert trajectory_builder.main(argv) == 0
    rows = [
        json.loads(line)
        for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    trajectories = [
        json.loads(line)
        for line in (root / "trajectories.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    exclusions = [
        json.loads(line)
        for line in (root / "exclusions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    return {
        "root": root,
        "rows": rows,
        "trajectories": trajectories,
        "exclusions": exclusions,
    }


def _run_single_round_builder(fixture: dict, tmp_path: Path) -> dict:
    root = tmp_path / "single-round"
    argv = [
        *_common_builder_args(fixture, root),
        "--visible-root",
        str(root / "visible"),
        "--evaluation-root",
        str(root / "evaluation"),
        "--authorized-root",
        str(root / "authorized"),
        "--output-manifest",
        str(root / "episodes.jsonl"),
        "--exclusions",
        str(root / "exclusions.jsonl"),
        "--ready-receipt",
        str(root / "EPISODES_READY.json"),
    ]
    assert single_round_builder.main(argv) == 0
    return {
        "root": root,
        "rows": [
            json.loads(line)
            for line in (root / "episodes.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ],
        "exclusions": [
            json.loads(line)
            for line in (root / "exclusions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ],
    }


# --- trajectory build --------------------------------------------------------


def test_teacher_forced_trajectories_advance_and_terminate_explicitly(
    tmp_path: Path, monkeypatch
) -> None:
    result = _run_trajectory_builder(_fixture(tmp_path, monkeypatch), tmp_path)
    rows, trajectories, exclusions = (
        result["rows"],
        result["trajectories"],
        result["exclusions"],
    )
    # case-b is perfectly corrected already: both residuals empty, both
    # operations excluded at round 0 under the frozen single-round reasons.
    assert {row["reason"] for row in exclusions} == {
        "EMPTY_FN_RESIDUAL",
        "EMPTY_FP_RESIDUAL",
    }
    assert {row["case_id"] for row in exclusions} == {"case-b"}
    assert all(row["round_index"] == 0 for row in exclusions)
    assert len(trajectories) == 2
    by_id = {row["trajectory_id"]: row for row in trajectories}
    add_trajectory = next(
        row for row in trajectories if row["operation"] == "ADD"
    )
    remove_trajectory = next(
        row for row in trajectories if row["operation"] == "REMOVE"
    )
    # ADD: round 0 fixes the SAME lesion, round 1 the NEW lesion, then the FN
    # residual is empty -> RESIDUAL_EXHAUSTED after round 1.
    assert add_trajectory["round_count"] == 2
    assert add_trajectory["trajectory_status"] == "RESIDUAL_EXHAUSTED"
    assert add_trajectory["termination_reason"] == "EMPTY_FN_RESIDUAL"
    # REMOVE: one round removes the only FP blob -> exhausted after round 0.
    assert remove_trajectory["round_count"] == 1
    assert remove_trajectory["trajectory_status"] == "RESIDUAL_EXHAUSTED"
    assert remove_trajectory["termination_reason"] == "EMPTY_FP_RESIDUAL"
    assert len(rows) == 3
    add_rows = sorted(
        [row for row in rows if row["operation"] == "ADD"],
        key=lambda row: row["round_index"],
    )
    assert [row["round_index"] for row in add_rows] == [0, 1]
    assert add_rows[0]["goal"] == "ADD_SAME_COMPLETE"
    assert add_rows[1]["goal"] == "ADD_NEW_COMPLETE"
    assert add_rows[1]["m0_path"] != add_rows[0]["m0_path"]
    assert Path(add_rows[1]["m0_path"]).is_file()
    assert Path(add_rows[1]["residual_path"]).is_file()
    assert Path(add_rows[1]["authorized_path"]).is_file()
    # The round-1 row's current mask is the teacher-forced state: it must
    # equal M0 union the round-0 authorized target.
    m0 = nib.load(str(add_rows[0]["m0_path"])).get_fdata() > 0
    authorized_0 = nib.load(str(add_rows[0]["authorized_path"])).get_fdata() > 0
    state_1 = nib.load(str(add_rows[1]["m0_path"])).get_fdata() > 0
    assert np.array_equal(state_1, m0 | authorized_0)
    # Round rows carry the explicit five-round identity.
    for row in rows:
        assert row["trajectory_id"].startswith("petct-traj-")
        assert 0 <= row["round_index"] <= 4
        assert row["round_count"] in (1, 2)
        assert row["trajectory_status"] == "RESIDUAL_EXHAUSTED"
        assert row["strategy"] in (
            "random",
            "centerline",
            "boundary",
        )
    # The trajectory summary pins the round-0 episode id for parity.
    assert add_trajectory["round0_episode_id"] == add_rows[0]["episode_id"]
    assert by_id[add_rows[1]["trajectory_id"]]["episode_ids"] == [
        add_rows[0]["episode_id"],
        add_rows[1]["episode_id"],
    ]
    ready = json.loads(
        (result["root"] / "EPISODES_READY.json").read_text(encoding="utf-8")
    )
    assert ready["schema_version"] == "PETCT-TRAJECTORY-EPISODES-READY-v1.0"
    assert ready["trajectory_stats"]["trajectories"] == 2
    assert ready["trajectory_stats"]["episodes"] == 3
    assert ready["trajectory_stats"]["residual_exhausted"] == 2
    assert ready["locked_test_present"] is False


def test_round_zero_rows_and_exclusions_match_the_single_round_builder(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    trajectory = _run_trajectory_builder(fixture, tmp_path)
    single = _run_single_round_builder(fixture, tmp_path)
    assert len(single["rows"]) == 2  # case-a ADD + REMOVE; case-b fully excluded
    round0_rows = [
        row for row in trajectory["rows"] if row["round_index"] == 0
    ]
    by_episode = {row["episode_id"]: row for row in round0_rows}
    assert set(by_episode) == {row["episode_id"] for row in single["rows"]}
    root_bound_paths = {"authorized_path", "visible_document", "evaluation_document"}
    for single_row in single["rows"]:
        trajectory_row = by_episode[single_row["episode_id"]]
        assert trajectory_row["round_index"] == 0
        for key, value in single_row.items():
            if key in root_bound_paths:
                expected = Path(value).relative_to(single["root"])
                observed = Path(trajectory_row[key]).relative_to(trajectory["root"])
                assert observed == expected, "round-0 field %s drifted" % key
            else:
                assert trajectory_row[key] == value, "round-0 field %s drifted" % key
        # The physical documents are bit-identical too.
        visible = single["root"] / "visible" / (single_row["episode_id"] + ".json")
        trajectory_visible = (
            trajectory["root"] / "visible" / (single_row["episode_id"] + ".json")
        )
        assert sha256_file(visible) == sha256_file(trajectory_visible)
        evaluation = single["root"] / "evaluation" / (
            single_row["episode_id"] + ".json"
        )
        trajectory_evaluation = (
            trajectory["root"] / "evaluation" / (single_row["episode_id"] + ".json")
        )
        assert sha256_file(evaluation) == sha256_file(trajectory_evaluation)
    # case-b exclusions: identical reason codes and fields at round 0.
    single_exclusions = sorted(
        single["exclusions"], key=lambda row: (row["attempt_id"],)
    )
    trajectory_exclusions = [
        row for row in trajectory["exclusions"] if row["round_index"] == 0
    ]
    trajectory_exclusions = sorted(
        trajectory_exclusions, key=lambda row: (row["attempt_id"],)
    )
    assert {row["reason"] for row in single_exclusions} == {
        "EMPTY_FN_RESIDUAL",
        "EMPTY_FP_RESIDUAL",
    }
    assert len(single_exclusions) == len(trajectory_exclusions)
    for single_exclusion, trajectory_exclusion in zip(
        single_exclusions, trajectory_exclusions
    ):
        for key, value in single_exclusion.items():
            assert trajectory_exclusion[key] == value, (
                "round-0 exclusion field %s drifted" % key
            )
    # The parity anchor in the receipt hashes the round-0 projection.
    ready = json.loads(
        (trajectory["root"] / "EPISODES_READY.json").read_text(encoding="utf-8")
    )
    assert ready["parity_contract"]["single_round_corpus"] == "R13-main-single-round"
    assert ready["parity_contract"]["only_difference"] == "round_count"
    assert ready["trajectory_stats"]["round0_rows_sha256"]


# --- tensor materialization ---------------------------------------------------


def test_tensor_materializer_attaches_explicit_trajectory_identity(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    built = _run_trajectory_builder(fixture, tmp_path)
    root = tmp_path / "tensors"
    argv = [
        "--episode-manifest",
        str(built["root"] / "episodes.jsonl"),
        "--visible-root",
        str(root / "visible"),
        "--evaluation-root",
        str(root / "evaluation"),
        "--output-manifest",
        str(root / "tensors.jsonl"),
        "--experiment-config",
        str(fixture["config"]),
        "--learning-split",
        str(fixture["split"]),
        "--partitions",
        "train",
    ]
    assert trajectory_tensors.main(argv) == 0
    rows = [
        json.loads(line)
        for line in (root / "tensors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 3
    pairs = [(row["trajectory_id"], row["round_index"]) for row in rows]
    assert len(pairs) == len(set(pairs))
    for row in rows:
        assert row["trajectory_id"].startswith("petct-traj-")
        assert 0 <= row["round_index"] <= 4
        assert row["round_count"] in (1, 2)
        assert row["trajectory_status"] == "RESIDUAL_EXHAUSTED"
        assert row["termination_reason"] in ("EMPTY_FN_RESIDUAL", "EMPTY_FP_RESIDUAL")
        assert Path(row["visible_npz"]).is_file()
        assert Path(row["evaluation_npz"]).is_file()
    validate_r13_trajectory_rows(
        [
            {
                "episode_id": row["episode_id"],
                "episode_family_id": row["trajectory_id"],
                "trajectory_id": row["trajectory_id"],
                "case_id": row["case_id"],
                "patient_id": row["patient_id"],
                "partition": row["partition"],
                "operation": row["operation"],
                "strategy": row["strategy"],
                "round_index": row["round_index"],
                "round_count": row["round_count"],
                "trajectory_status": row["trajectory_status"],
                "scribble_count": 1,
                "source_m0_lineage": "M0_V6_FIVEFOLD_OOF",
            }
            for row in rows
        ]
    )
    # Round-1 tensors encode the teacher-forced state as their current M0.
    round1 = next(row for row in rows if row["round_index"] == 1)
    round1_source_m0 = Path(round1["source_evaluation"]["m0_path"])
    assert round1_source_m0.is_file()
    assert round1_source_m0 != Path(
        next(row for row in rows if row["round_index"] == 0)["source_evaluation"][
            "m0_path"
        ]
    )


# --- program manifests and the three-lane firewall ----------------------------


def _run_program_chain(tmp_path: Path, built: dict, fixture: dict) -> dict:
    tensors_root = tmp_path / "tensors"
    argv = [
        "--episode-manifest",
        str(built["root"] / "episodes.jsonl"),
        "--visible-root",
        str(tensors_root / "visible"),
        "--evaluation-root",
        str(tensors_root / "evaluation"),
        "--output-manifest",
        str(tensors_root / "tensors.jsonl"),
        "--experiment-config",
        str(fixture["config"]),
        "--learning-split",
        str(fixture["split"]),
        "--partitions",
        "train",
    ]
    assert trajectory_tensors.main(argv) == 0
    import materialize_petct_component_candidates as candidates_module

    candidates_root = tmp_path / "candidates"
    assert (
        candidates_module.main(
            [
                "--learning-manifest",
                str(tensors_root / "tensors.jsonl"),
                "--output",
                str(candidates_root / "candidates"),
                "--summary",
                str(candidates_root / "candidates.jsonl"),
            ]
        )
        == 0
    )
    import materialize_petct_component_targets as targets_module

    targets_root = tmp_path / "targets"
    assert (
        targets_module.main(
            [
                "--learning-manifest",
                str(tensors_root / "tensors.jsonl"),
                "--candidate-summary",
                str(candidates_root / "candidates.jsonl"),
                "--output",
                str(targets_root / "pointer-targets"),
                "--summary",
                str(targets_root / "pointer-targets.jsonl"),
            ]
        )
        == 0
    )
    lineage_receipt = tmp_path / "lineage-receipt.json"
    issue_trajectory_lineage_receipt(
        oof_ready=fixture["oof_ready"],
        learning_split=fixture["split"],
        experiment_config=fixture["config"],
        output=lineage_receipt,
    )
    programs_root = tmp_path / "programs"
    programs_argv = [
        "--source",
        str(tensors_root / "tensors.jsonl"),
        "--learning-split",
        str(fixture["split"]),
        "--lineage-receipt",
        str(lineage_receipt),
        "--candidate-summary",
        str(candidates_root / "candidates.jsonl"),
        "--inference",
        str(programs_root / "inference.jsonl"),
        "--labels",
        str(programs_root / "labels.jsonl"),
        "--audit",
        str(programs_root / "audit.jsonl"),
        "--receipt",
        str(programs_root / "program-manifest-receipt.json"),
    ]
    assert trajectory_programs.main(programs_argv) == 0
    data_ready = tmp_path / "data-ready.json"
    sealed = seal_trajectory_data_ready(
        lineage_receipt=lineage_receipt,
        manifest_receipt=programs_root / "program-manifest-receipt.json",
        inference_manifest=programs_root / "inference.jsonl",
        label_manifest=programs_root / "labels.jsonl",
        audit_manifest=programs_root / "audit.jsonl",
        rich_tensor_manifest=tensors_root / "tensors.jsonl",
        candidate_summary=candidates_root / "candidates.jsonl",
        pointer_summary=targets_root / "pointer-targets.jsonl",
        trajectories_summary=built["root"] / "trajectories.jsonl",
        output=data_ready,
    )
    return {
        "programs": programs_root,
        "lineage_receipt": lineage_receipt,
        "data_ready": data_ready,
        "sealed": sealed,
    }


def test_three_lane_firewall_and_trajectory_data_ready_seal(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    built = _run_trajectory_builder(fixture, tmp_path)
    chain = _run_program_chain(tmp_path, built, fixture)
    programs = chain["programs"]
    inference_rows = [
        json.loads(line)
        for line in (programs / "inference.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    label_rows = [
        json.loads(line)
        for line in (programs / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    audit_rows = [
        json.loads(line)
        for line in (programs / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(inference_rows) == len(label_rows) == len(audit_rows) == 3
    allowed = set(INFERENCE_MANIFEST_ALLOWED_FIELDS) | {"candidate_json", "candidate_sha256"}
    for row in inference_rows:
        assert set(row) <= allowed, "inference row leaks %s" % (set(row) - allowed)
        assert "trajectory_id" not in row
        assert "goal" not in row
        assert "case_id" not in row
        assert row["dataset_id"] == TRAJECTORY_DATASET_ID
        assert row["round_index"] in (0, 1)
        assert row["scribble_count"] == 1
    for row in label_rows:
        assert row["trajectory_id"] == row["episode_family_id"]
        assert row["dataset_id"] == TRAJECTORY_DATASET_ID
        assert row["goal"] in {
            "ADD_SAME_COMPLETE",
            "ADD_NEW_COMPLETE",
            "REMOVE_NEW_COMPLETE",
        }
        assert row["source_m0_lineage"] == "M0_V6_FIVEFOLD_OOF"
        assert row["round_count"] in (1, 2)
        assert row["trajectory_status"] == "RESIDUAL_EXHAUSTED"
    for row in audit_rows:
        assert "source_record" in row
        assert row["source_record"]["trajectory_id"] == row["trajectory_id"]
        assert row["episode_family_id"] == row["trajectory_id"]
    receipt = json.loads(
        (programs / "program-manifest-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["dataset_id"] == TRAJECTORY_DATASET_ID
    assert receipt["mainline_eligible"] is False
    assert receipt["locked_test_present"] is False
    assert receipt["lifecycle"] == "active"
    validated = validate_trajectory_data_ready(chain["data_ready"])
    assert validated["dataset_id"] == TRAJECTORY_DATASET_ID
    assert validated["mainline_eligible"] is False
    assert validated["row_count"] == 3
    assert validated["outputs"]["trajectories_summary"]
    assert chain["sealed"]["status"] == "PASS"


def test_trajectory_lineage_rejects_a_single_round_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    # The two schemas must never cross: a valid R13-main single-round lineage
    # receipt is rejected by the trajectory validator (and vice versa).
    fixture = _fixture(tmp_path, monkeypatch)
    single_round_lineage = tmp_path / "single-round-lineage.json"
    single_round_lineage.write_text(
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
                "oof_ready": _file_record(fixture["oof_ready"]),
                "learning_split": _file_record(fixture["split"]),
                "experiment_config": _file_record(fixture["config"]),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(
        Exception, match="round_count|mainline|dataset_id|episode_schema|schema_version"
    ):
        validate_trajectory_lineage_receipt(single_round_lineage)
