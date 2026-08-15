from __future__ import annotations

import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "baseline"))

from run_petct_m0_inference_smoke import run_inference_smoke  # noqa: E402
from validate_petct_m0_inference_smoke import (  # noqa: E402
    CASE_SELECTION_VERSION,
    DATASET_FOLDER,
    TRAINER_FOLDER,
    ContractError,
    build_inference_smoke_bundle,
    publish_inference_smoke_ready,
    select_fold0_validation_case,
    stage_inference_smoke_run,
    validate_inference_prerequisites,
    validate_inference_smoke_output,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_hash(case_id: str) -> str:
    return hashlib.sha256(
        f"{CASE_SELECTION_VERSION}|fold0|{case_id}".encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _split_contract() -> dict[str, Any]:
    val = [
        "psma_patienta_20200101",
        "psma_patientb_20200202",
        "psma_patientc_20200303",
    ]
    train = ["psma_patientd_20200404", "psma_patiente_20200505"]
    return {
        "status": "PASS",
        "sha256": "1" * 64,
        "case_count": 5,
        "fold_count": 5,
        "validation_exact_once": True,
        "folds": {
            "0": {
                "train": train,
                "val": val,
                "train_count": len(train),
                "val_count": len(val),
                "train_sha256": hashlib.sha256("\n".join(train).encode()).hexdigest(),
                "val_sha256": hashlib.sha256("\n".join(val).encode()).hexdigest(),
            }
        },
    }


def _make_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _fixture_tree(tmp_path: Path) -> tuple[dict[str, Any], Any]:
    preprocess_ready = tmp_path / "manifests" / "PREPROCESS_READY.json"
    smoke_ready = tmp_path / "manifests" / "SMOKE_READY.json"
    split_path = tmp_path / "preprocessed" / DATASET_FOLDER / "splits_final.json"
    source_root = tmp_path / "nnunet-source"
    raw_root = tmp_path / "raw"
    smoke_run = tmp_path / "smoke-run"
    for path in (preprocess_ready, smoke_ready):
        _write_json(path, {"synthetic": True})
    _write_json(split_path, [{"train": [], "val": []}])
    source_root.mkdir()

    split_contract = _split_contract()
    selected = min(
        split_contract["folds"]["0"]["val"], key=lambda case: (_case_hash(case), case)
    )
    images = raw_root / DATASET_FOLDER / "imagesTr"
    affine = np.diag([2.0, 2.5, 3.0, 1.0])
    shape = (4, 5, 6)
    _make_nifti(
        images / f"{selected}_0000.nii.gz", np.zeros(shape, dtype=np.float32), affine
    )
    _make_nifti(
        images / f"{selected}_0001.nii.gz", np.ones(shape, dtype=np.float32), affine
    )

    model_dir = smoke_run / "nnUNet_results" / DATASET_FOLDER / TRAINER_FOLDER
    fold_dir = model_dir / "fold_0"
    fold_dir.mkdir(parents=True)
    (fold_dir / "checkpoint_final.pth").write_bytes(b"synthetic-checkpoint")
    plans = {"dataset_name": DATASET_FOLDER, "plans_name": "nnUNetPlans"}
    dataset = {
        "channel_names": {"0": "CT", "1": "PET"},
        "labels": {"background": 0, "tumor": 1},
        "file_ending": ".nii.gz",
    }
    _write_json(model_dir / "plans.json", plans)
    _write_json(model_dir / "dataset.json", dataset)

    base = {
        "status": "PASS",
        "paths": {
            "preprocess_ready": str(preprocess_ready.resolve()),
            "smoke_ready": str(smoke_ready.resolve()),
            "source_root": str(source_root.resolve()),
            "splits_final": str(split_path.resolve()),
        },
        "bound_hashes": {
            "preprocess_ready": _sha256(preprocess_ready),
            "smoke_ready": _sha256(smoke_ready),
            "source_tree": "2" * 64,
            "splits_final": "1" * 64,
        },
        "runtime": {
            "status": "PASS",
            "version": "2.8.1",
            "source_tree_sha256": "2" * 64,
        },
        "split_contract": split_contract,
        "preprocess": {
            "status": "PASS",
            "nnunet_raw": str(raw_root.resolve()),
            "nnunet_preprocessed": str((tmp_path / "preprocessed").resolve()),
            "bound_hashes": {"planning_splits_final": "1" * 64},
        },
        "smoke": {
            "status": "PASS",
            "run_dir": str(smoke_run.resolve()),
            "bound_hashes": {"smoke_ready": _sha256(smoke_ready)},
        },
    }

    def base_validator(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return copy.deepcopy(base)

    return {
        "preprocess_ready": preprocess_ready,
        "smoke_ready": smoke_ready,
        "split_path": split_path,
        "source_root": source_root,
        "raw_root": raw_root,
        "smoke_run": smoke_run,
        "model_dir": model_dir,
        "selected": selected,
        "shape": shape,
        "affine": affine,
        "dataset": dataset,
        "plans": plans,
    }, base_validator


def _prerequisites(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], Any]:
    tree, base_validator = _fixture_tree(tmp_path)
    prerequisites = validate_inference_prerequisites(
        tree["preprocess_ready"],
        tree["smoke_ready"],
        tree["source_root"],
        expected_case_count=5,
        base_prerequisite_validator=base_validator,
    )
    return tree, prerequisites, base_validator


def _stage(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    tree, prerequisites, base_validator = _prerequisites(tmp_path)
    runs = tmp_path / "inference-runs"
    runs.mkdir()
    staging = runs / ".partial-inference-smoke-001"
    final = runs / "inference-smoke-001"
    staging.mkdir()
    spec = stage_inference_smoke_run(
        tree["preprocess_ready"],
        tree["smoke_ready"],
        tree["source_root"],
        staging,
        final,
        "inference-smoke-001",
        gpu_id="0",
        expected_case_count=5,
        prerequisite_validator=lambda *args, **kwargs: copy.deepcopy(prerequisites),
    )
    assert spec["selected_case"]["case_id"] == tree["selected"]
    return tree, prerequisites, staging, final


def _populate_valid_prediction(tree: dict[str, Any], run_root: Path) -> None:
    case_id = tree["selected"]
    output = run_root / "predictions"
    mask = np.zeros(tree["shape"], dtype=np.uint8)
    mask[1:3, 2:4, 3:5] = 1
    foreground = np.where(mask == 1, 0.75, 0.25).astype(np.float32)
    probabilities_xyz = np.stack([1.0 - foreground, foreground])
    probabilities = probabilities_xyz.transpose(0, 3, 2, 1)
    _make_nifti(output / f"{case_id}.nii.gz", mask, tree["affine"])
    np.savez_compressed(output / f"{case_id}.npz", probabilities=probabilities)
    with (output / f"{case_id}.pkl").open("wb") as stream:
        pickle.dump({"shape_before_cropping": tuple(reversed(tree["shape"]))}, stream)
    _write_json(output / "dataset.json", tree["dataset"])
    _write_json(output / "plans.json", tree["plans"])
    _write_json(
        output / "predict_from_raw_data_args.json",
        {
            "list_of_lists_or_source_folder": [
                [
                    str(
                        tree["raw_root"]
                        / DATASET_FOLDER
                        / "imagesTr"
                        / f"{case_id}_0000.nii.gz"
                    ),
                    str(
                        tree["raw_root"]
                        / DATASET_FOLDER
                        / "imagesTr"
                        / f"{case_id}_0001.nii.gz"
                    ),
                ]
            ],
            "output_folder_or_list_of_truncated_output_files": [str(output / case_id)],
            "save_probabilities": True,
            "overwrite": False,
            "num_processes_preprocessing": 2,
            "num_processes_segmentation_export": 2,
            "folder_with_segs_from_prev_stage": None,
            "num_parts": 1,
            "part_id": 0,
        },
    )
    (run_root / "console.log").write_text(
        f"Predicting {case_id}\nGPU prediction completed.\nSegmentation export complete.\n",
        encoding="utf-8",
    )


def test_fold0_case_selection_is_hash_frozen_and_patient_excluded() -> None:
    split = _split_contract()
    selected = select_fold0_validation_case(split)
    reversed_split = copy.deepcopy(split)
    reversed_split["folds"]["0"]["val"].reverse()

    assert selected == select_fold0_validation_case(reversed_split)
    assert selected["case_id"] == min(
        split["folds"]["0"]["val"],
        key=lambda case: (_case_hash(case), case),
    )
    assert selected["selection_sha256"] == _case_hash(selected["case_id"])
    assert selected["held_out_fold"] == 0
    assert selected["patient_excluded_from_train"] is True


def test_case_selection_rejects_patient_overlap() -> None:
    split = _split_contract()
    chosen = min(
        split["folds"]["0"]["val"],
        key=lambda case: (_case_hash(case), case),
    )
    patient = chosen.rsplit("_", 1)[0]
    split["folds"]["0"]["train"].append(f"{patient}_19990101")

    with pytest.raises(ContractError, match="patient-excluded"):
        select_fold0_validation_case(split)


def test_prerequisites_bind_checkpoint_plans_split_source_and_receipts(
    tmp_path: Path,
) -> None:
    tree, prerequisites, _ = _prerequisites(tmp_path)

    assert prerequisites["status"] == "PASS"
    assert prerequisites["selected_case"]["case_id"] == tree["selected"]
    assert prerequisites["selected_case"]["patient_excluded_from_train"] is True
    for key in (
        "checkpoint_final",
        "plans",
        "dataset_json",
        "splits_final",
        "source_ct",
        "source_pet",
        "preprocess_ready",
        "smoke_ready",
    ):
        assert len(prerequisites["bound_hashes"][key]) == 64
    assert prerequisites["bound_hashes"]["source_tree"] == "2" * 64


def test_prerequisites_reject_missing_checkpoint_or_changed_source(
    tmp_path: Path,
) -> None:
    tree, base_validator = _fixture_tree(tmp_path)
    checkpoint = tree["model_dir"] / "fold_0" / "checkpoint_final.pth"
    checkpoint.unlink()

    with pytest.raises(ContractError, match="checkpoint_final"):
        validate_inference_prerequisites(
            tree["preprocess_ready"],
            tree["smoke_ready"],
            tree["source_root"],
            expected_case_count=5,
            base_prerequisite_validator=base_validator,
        )


def test_stage_is_run_scoped_empty_and_no_clobber(tmp_path: Path) -> None:
    tree, prerequisites, staging, final = _stage(tmp_path)
    del tree

    assert (staging / "INFERENCE_SMOKE_SPEC.json").is_file()
    assert (staging / "RUN_OWNER.json").is_file()
    assert (staging / "predictions").is_dir()
    assert prerequisites["selected_case"]["case_id"]
    with pytest.raises(ContractError, match="empty"):
        stage_inference_smoke_run(
            Path("preprocess"),
            Path("smoke"),
            Path("source"),
            staging,
            final,
            "inference-smoke-001",
            gpu_id="0",
            prerequisite_validator=lambda *args, **kwargs: prerequisites,
        )


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def device(name: str) -> str:
        return name


class _FakePredictor:
    instances: list["_FakePredictor"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.initialized: dict[str, Any] | None = None
        self.predicted: dict[str, Any] | None = None
        self.trainer_name = "nnUNetTrainer_1epoch"
        self.instances.append(self)

    def initialize_from_trained_model_folder(self, folder: str, **kwargs: Any) -> None:
        self.initialized = {"folder": folder, **kwargs}

    def predict_from_files(self, inputs: Any, outputs: Any, **kwargs: Any) -> None:
        self.predicted = {"inputs": inputs, "outputs": outputs, **kwargs}
        truncated = Path(outputs[0])
        truncated.with_suffix(".nii.gz").write_bytes(b"mask")
        truncated.with_suffix(".npz").write_bytes(b"probability")
        truncated.with_suffix(".pkl").write_bytes(b"properties")


def test_runner_uses_official_actual_case_predictor_with_probability_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("nnUNet_compile", "false")
    _FakePredictor.instances.clear()
    model = tmp_path / "model"
    model.mkdir()
    images = [tmp_path / "case_0000.nii.gz", tmp_path / "case_0001.nii.gz"]
    for path in images:
        path.write_bytes(b"image")
    output = tmp_path / "predictions"
    output.mkdir()

    receipt = run_inference_smoke(
        model_training_output_dir=model,
        case_id="psma_patienta_20200101",
        image_files=images,
        output_dir=output,
        predictor_factory=_FakePredictor,
        torch_module=_FakeTorch,
    )

    predictor = _FakePredictor.instances[-1]
    assert predictor.kwargs == {
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "use_mirroring": True,
        "perform_everything_on_device": True,
        "device": "cuda",
        "verbose": False,
        "verbose_preprocessing": False,
        "allow_tqdm": False,
    }
    assert predictor.initialized == {
        "folder": str(model.resolve()),
        "use_folds": (0,),
        "checkpoint_name": "checkpoint_final.pth",
    }
    assert predictor.predicted is not None
    assert predictor.predicted["save_probabilities"] is True
    assert predictor.predicted["overwrite"] is False
    assert predictor.predicted["inputs"] == [[str(path.resolve()) for path in images]]
    assert receipt["scientific_metrics_computed"] is False
    assert receipt["oof_prediction_count"] == 0
    assert receipt["compile_mode"] == "disabled"
    assert receipt["nnunet_compile"] == "false"


def test_runner_rejects_unfrozen_compile_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("nnUNet_compile", "true")
    model = tmp_path / "model"
    model.mkdir()
    images = [tmp_path / "case_0000.nii.gz", tmp_path / "case_0001.nii.gz"]
    for path in images:
        path.write_bytes(b"image")
    output = tmp_path / "predictions"
    output.mkdir()

    with pytest.raises(RuntimeError, match="nnUNet_compile=false"):
        run_inference_smoke(
            model_training_output_dir=model,
            case_id="psma_patienta_20200101",
            image_files=images,
            output_dir=output,
            predictor_factory=_FakePredictor,
            torch_module=_FakeTorch,
        )


def test_output_validates_original_grid_probability_and_mask_consistency(
    tmp_path: Path,
) -> None:
    tree, prerequisites, staging, _ = _stage(tmp_path)
    _populate_valid_prediction(tree, staging)

    inventory = validate_inference_smoke_output(staging)

    assert inventory["status"] == "PASS"
    assert inventory["case_id"] == tree["selected"]
    assert inventory["geometry"]["original_grid_match"] is True
    assert inventory["probability"]["finite"] is True
    assert inventory["probability"]["range"] == [0.0, 1.0]
    assert inventory["probability"]["foreground_channel"] == 1
    assert inventory["probability"]["official_axis_order"] == "CZYX"
    assert inventory["probability"]["official_shape"] == [2, 6, 5, 4]
    assert inventory["probability"]["nifti_axis_order"] == "CXYZ"
    assert inventory["probability"]["nifti_grid_shape"] == [2, 4, 5, 6]
    assert inventory["mask_probability_consistent"] is True
    assert inventory["scientific_metrics_computed"] is False
    assert inventory["oof_prediction_count"] == 0
    assert inventory["result_count"] == 0
    assert prerequisites["selected_case"]["case_id"] == tree["selected"]


@pytest.mark.parametrize(
    "mutation,error",
    [("nan", "finite"), ("range", r"\[0, 1\]"), ("mask", "consistent")],
)
def test_output_rejects_invalid_probability_or_mask(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    tree, _, staging, _ = _stage(tmp_path)
    _populate_valid_prediction(tree, staging)
    case_id = tree["selected"]
    probability_path = staging / "predictions" / f"{case_id}.npz"
    probabilities = np.load(probability_path)["probabilities"]
    if mutation == "nan":
        probabilities[1, 0, 0, 0] = np.nan
        np.savez_compressed(probability_path, probabilities=probabilities)
    elif mutation == "range":
        probabilities[1, 0, 0, 0] = 1.1
        np.savez_compressed(probability_path, probabilities=probabilities)
    else:
        mask_path = staging / "predictions" / f"{case_id}.nii.gz"
        mask = np.asarray(nib.load(str(mask_path)).dataobj, dtype=np.uint8)
        mask[0, 0, 0] = 1 - mask[0, 0, 0]
        _make_nifti(mask_path, mask, tree["affine"])

    with pytest.raises(ContractError, match=error):
        validate_inference_smoke_output(staging)


def test_output_rejects_prediction_geometry_drift(tmp_path: Path) -> None:
    tree, _, staging, _ = _stage(tmp_path)
    _populate_valid_prediction(tree, staging)
    mask_path = staging / "predictions" / f"{tree['selected']}.nii.gz"
    mask = np.asarray(nib.load(str(mask_path)).dataobj, dtype=np.uint8)
    changed_affine = tree["affine"].copy()
    changed_affine[0, 3] = 9.0
    _make_nifti(mask_path, mask, changed_affine)

    with pytest.raises(ContractError, match="original-grid geometry"):
        validate_inference_smoke_output(staging)


def test_bundle_and_ready_are_non_scientific_no_clobber_receipts(
    tmp_path: Path,
) -> None:
    tree, prerequisites, staging, final = _stage(tmp_path)
    _populate_valid_prediction(tree, staging)
    inventory = validate_inference_smoke_output(staging)
    bundle = build_inference_smoke_bundle(
        tree["preprocess_ready"],
        tree["smoke_ready"],
        tree["source_root"],
        run_id=final.name,
        committed_run_dir=final,
        inventory=inventory,
        expected_case_count=5,
        prerequisite_validator=lambda *args, **kwargs: copy.deepcopy(prerequisites),
    )
    _write_json(staging / "INFERENCE_SMOKE_BUNDLE.json", bundle)
    staging.rename(final)
    ready = tmp_path / "manifests" / "INFERENCE_SMOKE_READY.json"

    published = publish_inference_smoke_ready(
        final,
        final / "INFERENCE_SMOKE_BUNDLE.json",
        ready,
        expected_case_count=5,
        prerequisite_validator=lambda *args, **kwargs: copy.deepcopy(prerequisites),
    )

    assert published["status"] == "COMMITTED"
    assert published["inference_smoke_status"] == "PASS"
    assert published["full_training_status"] == "NOT_STARTED"
    assert published["oof_prediction_count"] == 0
    assert published["result_count"] == 0
    assert published["scientific_metrics_computed"] is False
    assert published["thesis_citable"] is False
    assert published["prediction_disposable"] is True
    assert published["receipt_retention_required"] is True
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_inference_smoke_ready(
            final,
            final / "INFERENCE_SMOKE_BUNDLE.json",
            ready,
            expected_case_count=5,
            prerequisite_validator=lambda *args, **kwargs: prerequisites,
        )


def test_shell_places_inference_gate_between_training_smoke_and_full_training() -> None:
    shell = (SCRIPTS / "baseline" / "run_petct_m0_inference_smoke.sh").read_text(encoding="utf-8")

    assert "PREPROCESS_READY.json" in shell
    assert "SMOKE_READY.json" in shell
    assert "INFERENCE_SMOKE_READY.json" in shell
    assert "run_petct_m0_inference_smoke.py" in shell
    assert "save_probabilities" not in shell
    assert "full training" not in shell.casefold()
    assert "OOF_READY" not in shell
    assert "export nnUNet_compile=false" in shell
