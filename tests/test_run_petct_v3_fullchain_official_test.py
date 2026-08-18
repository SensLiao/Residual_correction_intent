"""Contract tests for the v3 full-chain official TEST runner.

All inputs are synthetic geometry-only fixtures; no locked test truth is
opened, no receipt is consumed, and no scientific result is produced.
"""

import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluation.run_petct_v3_fullchain_official_test import (  # noqa: E402
    CASE_SCHEMA,
    V3FullChainError,
    _load_editor_bundle,
    _verified_case_row,
    build_synthetic_smoke_models,
    run_one_case,
)
from common.petct_w21_test_access import PROTOCOL  # noqa: E402

STATE_COUNT = int(PROTOCOL["evaluation_states"])


class StubSimulator:
    """Deterministic official-interface stub: scribble = full component voxels."""

    @staticmethod
    def simulate_scribble_from_label(residual, strategy, seed):
        residual = np.asarray(residual) > 0
        if not np.any(residual):
            return [], 0, 0
        coordinates = np.argwhere(residual).astype(np.int64).tolist()
        coordinates = [[int(c) for c in coordinate] for coordinate in coordinates]
        return coordinates, len(coordinates), len(coordinates)


class StubMetricEvaluator:
    """Official-interface stub returning voxel dice as f1 (not citable)."""

    def __init__(self, overlap_threshold, connectivity):
        self.overlap_threshold = overlap_threshold
        self.connectivity = connectivity

    def __call__(self, prediction, ground_truth, case_name, spacing):
        pred = np.asarray(prediction) > 0
        gt = np.asarray(ground_truth) > 0
        denominator = int(pred.sum()) + int(gt.sum())
        dice = 0.0 if denominator == 0 else 2 * float(np.logical_and(pred, gt).sum()) / denominator
        fpv = float(np.logical_and(pred, ~gt).sum())
        fnv = float(np.logical_and(~pred, gt).sum())
        return {"f1": dice, "fpv": fpv, "fnv": fnv}


def _write_nifti(path: Path, volume: np.ndarray) -> str:
    image = nib.Nifti1Image(volume.astype(np.uint8), np.eye(4))
    nib.save(image, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def synthetic_case(tmp_path: Path) -> dict:
    shape = (32, 32, 16)
    pet = np.zeros(shape, dtype=np.float32)
    pet[8:24, 8:24, 4:12] = 5.0
    ct = np.zeros(shape, dtype=np.float32)
    ct[8:24, 8:24, 4:12] = 40.0
    gt = np.zeros(shape, dtype=np.uint8)
    gt[8:20, 8:20, 4:10] = 1
    m0 = np.zeros(shape, dtype=np.uint8)
    m0[8:18, 8:18, 4:9] = 1  # FN: 18:20 band; plus small FP below
    m0[20:22, 20:22, 10:12] = 1  # FP component (REMOVE candidate)
    paths = {
        "pet_path": tmp_path / "pet.nii.gz",
        "ct_path": tmp_path / "ct.nii.gz",
        "gt_path": tmp_path / "gt.nii.gz",
        "m0_path": tmp_path / "m0.nii.gz",
    }
    row = {"case_id": "synthetic-case-1", "patient_id": "synthetic-patient"}
    for key, path in paths.items():
        volume = {"pet_path": pet, "ct_path": ct, "gt_path": gt, "m0_path": m0}[key]
        row[key.replace("_path", "") + "_sha256"] = _write_nifti(path, volume)
        row[key] = str(path)
    return row


@pytest.fixture()
def smoke_models(tmp_path: Path) -> tuple[Path, Path]:
    return build_synthetic_smoke_models(tmp_path / "smoke-models")


def test_verified_case_row_accepts_matching_hashes(synthetic_case: dict) -> None:
    row = _verified_case_row(synthetic_case)
    assert row["case_id"] == "synthetic-case-1"


def test_verified_case_row_rejects_hash_mismatch(synthetic_case: dict) -> None:
    broken = dict(synthetic_case)
    broken["m0_sha256"] = "0" * 64
    with pytest.raises(V3FullChainError):
        _verified_case_row(broken)


def test_editor_bundle_loading(smoke_models: tuple[Path, Path]) -> None:
    _, editor_path = smoke_models
    model, checkpoint = _load_editor_bundle(editor_path, torch.device("cpu"))
    assert model.architecture_id == "matched_legal_component_program_editor_v1"
    assert checkpoint["arm"] == "J9"


def test_run_one_case_six_states_and_seals(
    synthetic_case: dict, smoke_models: tuple[Path, Path], tmp_path: Path
) -> None:
    compiler_path, editor_path = smoke_models
    from common.petct_program_models import LegalCallCompiler, ProgramCompilerNet

    compiler_ckpt = torch.load(str(compiler_path), map_location="cpu", weights_only=False)
    compiler_model = ProgramCompilerNet(include_repair=True)
    compiler_model.load_state_dict(compiler_ckpt["state_dict"], strict=True)
    compiler_model.eval()
    compiler = LegalCallCompiler(include_repair=True)
    editor_model, _ = _load_editor_bundle(editor_path, torch.device("cpu"))
    del compiler_ckpt

    output_parent = tmp_path / "arm-output"
    payload = run_one_case(
        source=synthetic_case,
        strategy="centerline",
        simulator=StubSimulator(),
        metric_evaluator_class=StubMetricEvaluator,
        compiler_model=compiler_model,
        compiler=compiler,
        editor_model=editor_model,
        model_config={"field_mm": 64.0, "output_size": 128, "expected_spacing": 1.0},
        output_parent=output_parent,
        device=torch.device("cpu"),
    )
    assert payload["schema_version"] == CASE_SCHEMA
    assert payload["case_id"] == "synthetic-case-1"
    assert len(payload["states"]) == STATE_COUNT
    assert payload["states"][0]["trace_recorded"] is False
    for state in payload["states"][1:]:
        assert state["trace_recorded"] is True
        assert state["dice"] is None or np.isfinite(state["dice"])
    assert payload["auc"]["auc_dice_3d"] is None or np.isfinite(payload["auc"]["auc_dice_3d"])
    assert payload["bridge_protocol"]["pending_formal_amendment"] is True
    # Sealed case receipt written; a second call must resume identically.
    case_dir = output_parent / "cases" / "synthetic-case-1"
    assert (case_dir / "case.json").is_file()
    resumed = run_one_case(
        source=synthetic_case,
        strategy="centerline",
        simulator=StubSimulator(),
        metric_evaluator_class=StubMetricEvaluator,
        compiler_model=compiler_model,
        compiler=compiler,
        editor_model=editor_model,
        model_config={"field_mm": 64.0, "output_size": 128, "expected_spacing": 1.0},
        output_parent=output_parent,
        device=torch.device("cpu"),
    )
    assert resumed["case_sha256"] == payload["case_sha256"]
    # Typed traces are JSON-finite on disk (M-15 boundary).
    for state_index in range(1, STATE_COUNT):
        trace_path = case_dir / f"typed_trace_state_{state_index}.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        assert trace["grammar_version"]
        assert trace["typed_trace"]


def test_run_one_case_resume_rejects_strategy_mismatch(
    synthetic_case: dict, smoke_models: tuple[Path, Path], tmp_path: Path
) -> None:
    compiler_path, editor_path = smoke_models
    from common.petct_program_models import LegalCallCompiler, ProgramCompilerNet
    from evaluation.run_petct_v3_fullchain_official_test import _load_editor_bundle

    compiler_ckpt = torch.load(str(compiler_path), map_location="cpu", weights_only=False)
    compiler_model = ProgramCompilerNet(include_repair=True)
    compiler_model.load_state_dict(compiler_ckpt["state_dict"], strict=True)
    compiler_model.eval()
    editor_model, _ = _load_editor_bundle(editor_path, torch.device("cpu"))
    output_parent = tmp_path / "arm-output"
    kwargs = dict(
        source=synthetic_case,
        simulator=StubSimulator(),
        metric_evaluator_class=StubMetricEvaluator,
        compiler_model=compiler_model,
        compiler=LegalCallCompiler(include_repair=True),
        editor_model=editor_model,
        model_config={"field_mm": 64.0, "output_size": 128, "expected_spacing": 1.0},
        output_parent=output_parent,
        device=torch.device("cpu"),
    )
    run_one_case(strategy="centerline", **kwargs)
    with pytest.raises(V3FullChainError):
        run_one_case(strategy="random", **kwargs)


def test_editor_bundle_rejects_unknown_arm(smoke_models: tuple[Path, Path], tmp_path: Path) -> None:
    _, editor_path = smoke_models
    checkpoint = torch.load(str(editor_path), map_location="cpu", weights_only=False)
    checkpoint["arm"] = "J99"
    broken = tmp_path / "broken_editor.pt"
    torch.save(checkpoint, broken)
    with pytest.raises(V3FullChainError):
        _load_editor_bundle(broken, torch.device("cpu"))
