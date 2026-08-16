from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "baseline"))

from run_petct_m0_oof_fold import (  # noqa: E402
    ACTUAL_VALIDATION_HANDOFF_SCHEMA,
    HANDOFF_PREDICTION_SOURCE,
    execute_fold,
    handoff_fold_actual_validation,
)
from validate_petct_m0_oof import (  # noqa: E402
    EXPECTED_NNUNET_SOURCE_TREE_SHA256,
    INFERENCE_COMPILE_CONTRACT,
    OOF_CONTRACT_VERSION,
    OOF_PHASE,
)


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class _FakePredictor:
    def __init__(self) -> None:
        self.initialized: tuple[str, tuple[int, ...], str] | None = None

    def initialize_from_trained_model_folder(
        self, root: str, *, use_folds: tuple[int, ...], checkpoint_name: str
    ) -> None:
        self.initialized = (root, use_folds, checkpoint_name)

    def predict_from_files(
        self,
        inputs: list[list[str]],
        outputs: list[str],
        *,
        save_probabilities: bool,
        overwrite: bool,
        num_processes_preprocessing: int,
        num_processes_segmentation_export: int,
    ) -> None:
        assert len(inputs) == len(outputs) == 1
        assert save_probabilities is True
        assert overwrite is False
        output = Path(outputs[0])
        foreground_xyz = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
        mask_xyz = (foreground_xyz >= 0.5).astype(np.uint8)
        nib.save(
            nib.Nifti1Image(mask_xyz, np.eye(4)),
            str(output.with_suffix(".nii.gz")),
        )
        probabilities_xyz = np.stack([1.0 - foreground_xyz, foreground_xyz])
        np.savez_compressed(
            str(output) + ".npz",
            probabilities=probabilities_xyz.transpose(0, 3, 2, 1),
        )
        output.with_suffix(".pkl").write_bytes(b"properties")
        for name in ("dataset.json", "plans.json", "predict_from_raw_data_args.json"):
            (output.parent / name).write_text("{}\n", encoding="utf-8")


def test_fold_runner_uses_only_held_out_fold_and_exports_foreground_npz(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("nnUNet_compile", INFERENCE_COMPILE_CONTRACT["value"])
    run = tmp_path / "oof-run"
    masks = run / "outputs" / "fold_2" / "masks"
    probabilities = run / "outputs" / "fold_2" / "probabilities"
    masks.mkdir(parents=True)
    probabilities.mkdir()
    model = tmp_path / "model" / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    checkpoint = model / "fold_2" / "checkpoint_final.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    plans = model / "plans.json"
    dataset = model / "dataset.json"
    plans.write_text("{}\n", encoding="utf-8")
    dataset.write_text("{}\n", encoding="utf-8")
    full_ready = tmp_path / "FULL_TRAIN_READY.json"
    fold_receipt = tmp_path / "fold_2.json"
    full_ready.write_text("{}\n", encoding="utf-8")
    fold_receipt.write_text("{}\n", encoding="utf-8")
    ct, pet = tmp_path / "case_0000.nii.gz", tmp_path / "case_0001.nii.gz"
    gt = tmp_path / "case.nii.gz"
    ct.write_bytes(b"ct")
    pet.write_bytes(b"pet")
    gt.write_bytes(b"gt")
    case_id = "psma_patient_20200101"
    plan = {
        "schema_version": OOF_CONTRACT_VERSION,
        "phase": OOF_PHASE,
        "fold": 2,
        "use_folds": [2],
        "save_probabilities": True,
        "overwrite": False,
        "compile_contract": INFERENCE_COMPILE_CONTRACT,
        "train_case_ids": ["another_case_20200101"],
        "val_case_ids": [case_id],
        "cases": [
            {
                "case_id": case_id,
                "input_ct": _record(ct),
                "input_pet": _record(pet),
                "input_gt": _record(gt),
            }
        ],
        "model": {
            "full_train_ready": _record(full_ready),
            "fold_receipt": _record(fold_receipt),
            "checkpoint": _record(checkpoint),
            "plans": _record(plans),
            "dataset_json": _record(dataset),
            "model_training_output_dir": str(model.resolve()),
            "source_tree_sha256": EXPECTED_NNUNET_SOURCE_TREE_SHA256,
        },
    }
    _write(run / "fold_plans" / "fold_2.json", plan)
    fake = _FakePredictor()

    def _commit(_run_root: Path, committed_fold: int) -> dict[str, Any]:
        payload = {"prediction_count": 1, "fold": committed_fold}
        _write(
            run / "outputs" / f"fold_{committed_fold}" / "FOLD_DONE.json",
            payload,
        )
        return payload

    receipt = execute_fold(
        run,
        2,
        predictor_factory=lambda _device: fake,
        source_root=tmp_path / "source",
        runtime_validator=lambda _source: {
            "source_tree_sha256": EXPECTED_NNUNET_SOURCE_TREE_SHA256
        },
        plan_validator=lambda _root, _fold: (
            run / "fold_plans" / "fold_2.json",
            plan,
        ),
        fold_committer=_commit,
    )

    assert receipt["status"] == "COMMITTED"
    assert fake.initialized == (str(model.resolve()), (2,), "checkpoint_final.pth")
    assert {item.name for item in masks.iterdir()} == {f"{case_id}.nii.gz"}
    assert {item.name for item in probabilities.iterdir()} == {f"{case_id}.npz"}
    with np.load(probabilities / f"{case_id}.npz", allow_pickle=False) as archive:
        assert archive.files == ["foreground_probability"]
        assert archive["foreground_probability"].dtype == np.float32
        expected = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
        assert archive["foreground_probability"].shape == (2, 3, 4)
        assert np.allclose(archive["foreground_probability"], expected)
    assert (run / "outputs" / "fold_2" / "FOLD_DONE.json").is_file()


def test_actual_validation_handoff_reuses_fold_outputs_without_predictor(
    tmp_path: Path,
) -> None:
    run = tmp_path / "oof-run"
    masks = run / "outputs" / "fold_2" / "masks"
    probabilities = run / "outputs" / "fold_2" / "probabilities"
    masks.mkdir(parents=True)
    probabilities.mkdir()
    campaign = tmp_path / "campaign"
    validation = campaign / "model" / "fold_2" / "validation"
    validation.mkdir(parents=True)
    case_id = "psma_patient_20200101"
    source_mask = validation / f"{case_id}.nii.gz"
    mask_xyz = np.zeros((2, 3, 4), dtype=np.uint8)
    mask_xyz[1, 1, 2] = 1
    nib.save(nib.Nifti1Image(mask_xyz, np.eye(4)), str(source_mask))
    foreground_xyz = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(2, 3, 4)
    source_probability = validation / f"{case_id}.npz"
    np.savez_compressed(
        source_probability,
        probabilities=np.stack(
            [1.0 - foreground_xyz, foreground_xyz]
        ).transpose(0, 3, 2, 1),
    )
    source_properties = validation / f"{case_id}.pkl"
    source_properties.write_bytes(b"properties")
    summary = validation / "summary.json"
    summary.write_text('{"metric_per_case": []}\n', encoding="utf-8")

    full_ready = tmp_path / "FULL_TRAIN_READY.json"
    fold_receipt = campaign / "fold_receipts" / "fold_2.json"
    full_ready_payload = {
        "status": "COMMITTED",
        "full_training_status": "PASS",
        "campaign_root": str(campaign.resolve()),
        "oof_handoff_inputs_present": True,
        "actual_inference_gate_required": False,
        "training_contract": {
            "actual_validation": True,
            "export_probabilities": True,
        },
    }
    _write(full_ready, full_ready_payload)
    def relative(path: Path) -> dict:
        return {
            **_record(path),
            "path": path.resolve().relative_to(campaign.resolve()).as_posix(),
        }
    _write(
        fold_receipt,
        {
            "status": "COMMITTED",
            "fold": 2,
            "output_contract": {
                "status": "PASS",
                "fold": 2,
                "actual_validation": True,
                "export_probabilities": True,
                "oof_handoff_inputs_present": True,
                "validation_case_count": 1,
                "validation_probability_count": 1,
                "artifacts": {
                    "validation_summary": relative(summary),
                    "validation_masks": [relative(source_mask)],
                    "validation_probabilities": [relative(source_probability)],
                    "validation_properties": [relative(source_properties)],
                },
            },
        },
    )
    plan = {
        "schema_version": OOF_CONTRACT_VERSION,
        "train_case_ids": ["another_case_20200101"],
        "val_case_ids": [case_id],
        "cases": [{"case_id": case_id}],
        "model": {
            "full_train_ready": _record(full_ready),
            "fold_receipt": _record(fold_receipt),
        },
    }

    committed: dict[str, Any] = {}

    def _commit(
        _run_root: Path,
        committed_fold: int,
        *,
        prediction_source: str,
        source_artifacts: dict[str, Any],
    ) -> dict[str, Any]:
        committed.update(
            fold=committed_fold,
            prediction_source=prediction_source,
            source_artifacts=source_artifacts,
        )
        return {"prediction_count": 1}

    result = handoff_fold_actual_validation(
        run,
        2,
        plan_validator=lambda _root, _fold: (tmp_path / "plan.json", plan),
        fold_committer=_commit,
    )

    assert result["prediction_source"] == HANDOFF_PREDICTION_SOURCE
    assert result["new_inference_executed"] is False
    destination_mask = masks / f"{case_id}.nii.gz"
    assert destination_mask.read_bytes() == source_mask.read_bytes()
    with np.load(probabilities / f"{case_id}.npz", allow_pickle=False) as archive:
        assert archive.files == ["foreground_probability"]
        assert np.allclose(archive["foreground_probability"], foreground_xyz)
    handoff = json.loads(
        (run / "outputs" / "fold_2" / "HANDOFF_SOURCE.json").read_text(
            encoding="utf-8"
        )
    )
    assert handoff["schema_version"] == ACTUAL_VALIDATION_HANDOFF_SCHEMA
    assert handoff["new_inference_executed"] is False
    assert committed["prediction_source"] == HANDOFF_PREDICTION_SOURCE
    assert committed["source_artifacts"]["handoff_source"]["sha256"] == result[
        "handoff_source_sha256"
    ]
