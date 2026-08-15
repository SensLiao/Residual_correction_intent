from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "baseline"))

from run_petct_m0_one_epoch import run_one_epoch_smoke  # noqa: E402
from validate_petct_m0_smoke import (  # noqa: E402
    CONFIGURATION,
    DATASET_FOLDER,
    DATASET_ID,
    FOLD,
    PLANS_IDENTIFIER,
    TRAINER,
    TRAINING_CONTRACT,
    ContractError,
    build_smoke_bundle,
    publish_smoke_ready,
    stage_smoke_run,
    validate_preprocess_ready,
    validate_smoke_output,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, relative_to: Path | None = None) -> dict:
    display = path if relative_to is None else path.relative_to(relative_to)
    return {
        "path": display.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _preprocess_fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    run = tmp_path / "preprocess_runs" / "pre-1"
    raw = run / "nnUNet_raw"
    raw_dataset = raw / DATASET_FOLDER
    preprocessed = run / "nnUNet_preprocessed"
    results = run / "nnUNet_results"
    raw_source_root = tmp_path / "planning_raw"
    raw_sources = {
        name: raw_source_root / name for name in ("imagesTr", "labelsTr")
    }
    for directory in (
        raw_dataset,
        preprocessed,
        results,
        *raw_sources.values(),
        *(raw_dataset / name for name in raw_sources),
    ):
        directory.mkdir(parents=True, exist_ok=True)

    planning_ready = tmp_path / "manifests" / "PLANNING_READY.json"
    _write_json(planning_ready, {"status": "COMMITTED", "run_id": "plan-1"})
    output_contract = {
        "case_count": 597,
        "artifact_counts": {
            ".b2nd": 597,
            "_seg.b2nd": 597,
            ".pkl": 597,
            "gt_segmentations": 597,
        },
        "one_case_load": {
            "status": "PASS",
            "hook_kind": "OFFICIAL_NNUNET_V2_8_1",
            "official_nnunet_load_claimed": True,
        },
    }
    planning_hashes = {"planning_ready": "a" * 64, "nnunet_plans": "b" * 64}
    bundle = {
        "status": "VALIDATED",
        "preprocessing_status": "PASS",
        "contract_version": "1.0.0",
        "phase": "PREPROCESSING_ONLY",
        "run_id": "pre-1",
        "committed_run_dir": str(run.resolve()),
        "planning_ready": _record(planning_ready),
        "planning_bound_hashes": planning_hashes,
        "output_contract": output_contract,
        "training_status": "NOT_STARTED",
        "training_performed": False,
        "checkpoint_count": 0,
        "oof_prediction_count": 0,
        "result_count": 0,
    }
    bundle_path = run / "PREPROCESSING_BUNDLE.json"
    _write_json(bundle_path, bundle)
    ready = tmp_path / "manifests" / "PREPROCESS_READY.json"
    published = {
        "status": "COMMITTED",
        "preprocessing_status": "PASS",
        "contract_version": "1.0.0",
        "phase": "PREPROCESSING_ONLY",
        "run_id": "pre-1",
        "run_dir": str(run.resolve()),
        "run_receipt": _record(bundle_path),
        "training_status": "NOT_STARTED",
        "training_performed": False,
        "checkpoint_count": 0,
        "oof_prediction_count": 0,
        "result_count": 0,
        "validated_bundle": bundle,
    }
    _write_json(ready, published)
    return ready, run, output_contract, planning_hashes


def _validate_fixture_ready(
    ready: Path, output_contract: dict, planning_hashes: dict
) -> dict:
    return validate_preprocess_ready(
        ready,
        output_validator=lambda _: output_contract,
        planning_validator=lambda _: {
            "bound_hashes": planning_hashes,
            "raw_source_paths": {
                name: str((ready.parents[1] / "planning_raw" / name).resolve())
                for name in ("imagesTr", "labelsTr")
            },
        },
        raw_source_validator=lambda run_dir, planning: {
            name: {
                "link_path": str(run_dir / "nnUNet_raw" / DATASET_FOLDER / name),
                "target_path": planning["raw_source_paths"][name],
                "policy": "PLANNING_RECEIPT_BOUND_DIRECTORY_SYMLINK",
            }
            for name in ("imagesTr", "labelsTr")
        },
    )


def _smoke_output_fixture(run: Path) -> tuple[Path, dict]:
    if not (run / "RUN_OWNER.json").exists():
        _write_json(
            run / "RUN_OWNER.json",
            {"status": "OWNED", "run_id": run.name, "owner_token": "fixture"},
        )
    if not (run / "SMOKE_SPEC.json").exists():
        _write_json(
            run / "SMOKE_SPEC.json",
            {
                "status": "STAGED",
                "contract_version": "1.0.0",
                "phase": "FOLD0_1EPOCH_SMOKE",
                "run_id": run.name,
                "committed_run_dir": str(run.resolve()),
                "full_training_status": "NOT_STARTED",
                "oof_prediction_count": 0,
                "result_count": 0,
                "training_contract": {**TRAINING_CONTRACT, "visible_gpu_id": "0"},
            },
        )
    fold = (
        run
        / "nnUNet_results"
        / "Dataset901_PSMA_M0_AutoPETVNorm"
        / "nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres"
        / "fold_0"
    )
    fold.mkdir(parents=True)
    (fold / "checkpoint_final.pth").write_bytes(b"checkpoint-final")
    (fold / "checkpoint_best.pth").write_bytes(b"checkpoint-best")
    (fold / "progress.png").write_bytes(b"png")
    (fold / "debug.json").write_text("{}\n", encoding="utf-8")
    (fold / "training_log_2026_7_17_00_00_00.txt").write_text(
        "Epoch 0\ntrain_loss 0.8123\nval_loss 0.7345\nEpoch time: 1.0 s\n",
        encoding="utf-8",
    )
    (run / "console.log").write_text("one epoch complete\n", encoding="utf-8")
    checkpoint = {
        "network_weights": {"encoder.weight": object()},
        "optimizer_state": {"state": {}},
        "logging": {
            "train_losses": [0.8123],
            "val_losses": [0.7345],
            "lrs": [0.01],
            "epoch_start_timestamps": [1.0],
            "epoch_end_timestamps": [2.0],
        },
        "current_epoch": 1,
        "trainer_name": "nnUNetTrainer_1epoch",
    }
    return fold, checkpoint


def test_frozen_smoke_identity_is_dataset901_fullres_fold0_one_epoch() -> None:
    assert DATASET_ID == 901
    assert CONFIGURATION == "3d_fullres"
    assert FOLD == 0
    assert TRAINER == "nnUNetTrainer_1epoch"
    assert PLANS_IDENTIFIER == "nnUNetPlans"


def test_preprocess_ready_is_rehashed_and_binds_current_output(
    tmp_path: Path,
) -> None:
    ready, run, inventory, planning_hashes = _preprocess_fixture(tmp_path)

    validated = _validate_fixture_ready(ready, inventory, planning_hashes)

    assert validated["preprocess_run_dir"] == str(run.resolve())
    assert validated["nnunet_raw"] == str((run / "nnUNet_raw").resolve())
    assert validated["nnunet_preprocessed"] == str(
        (run / "nnUNet_preprocessed").resolve()
    )
    assert validated["bound_hashes"]["preprocess_ready"] == _sha256(ready)
    assert validated["bound_hashes"]["preprocessing_bundle"] == _sha256(
        run / "PREPROCESSING_BUNDLE.json"
    )

    (run / "PREPROCESSING_BUNDLE.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="hash"):
        _validate_fixture_ready(ready, inventory, planning_hashes)


def test_preprocess_ready_rejects_drift_or_nonzero_prior_outputs(
    tmp_path: Path,
) -> None:
    ready, _, inventory, planning_hashes = _preprocess_fixture(tmp_path)
    payload = json.loads(ready.read_text(encoding="utf-8"))
    payload["checkpoint_count"] = 1
    _write_json(ready, payload)

    with pytest.raises(ContractError, match="checkpoint_count"):
        _validate_fixture_ready(ready, inventory, planning_hashes)


def test_stage_is_run_scoped_and_refuses_foreign_or_existing_destination(
    tmp_path: Path,
) -> None:
    ready, _, inventory, planning_hashes = _preprocess_fixture(tmp_path)
    runs = tmp_path / "smoke_runs"
    runs.mkdir()
    staging = runs / ".partial-smoke-1"
    final = runs / "smoke-1"
    staging.mkdir()

    staged = stage_smoke_run(
        ready,
        staging,
        final,
        "smoke-1",
        gpu_id="0",
        preprocess_validator=lambda path: _validate_fixture_ready(
            path, inventory, planning_hashes
        ),
    )

    assert staged["status"] == "STAGED"
    assert staged["training_contract"] == {
        **TRAINING_CONTRACT,
        "visible_gpu_id": "0",
    }
    assert (staging / "nnUNet_results").is_dir()
    assert not final.exists()

    foreign = runs / ".partial-smoke-2"
    foreign.mkdir()
    (foreign / "foreign.txt").write_text("owned", encoding="utf-8")
    with pytest.raises(ContractError, match="empty"):
        stage_smoke_run(
            ready,
            foreign,
            runs / "smoke-2",
            "smoke-2",
            gpu_id="0",
            preprocess_validator=lambda _: {},
        )

    existing = runs / ".partial-smoke-3"
    existing.mkdir()
    (runs / "smoke-3").mkdir()
    with pytest.raises(FileExistsError, match="destination"):
        stage_smoke_run(
            ready,
            existing,
            runs / "smoke-3",
            "smoke-3",
            gpu_id="0",
            preprocess_validator=lambda _: {},
        )


def test_staged_output_validates_before_atomic_rename(tmp_path: Path) -> None:
    ready, _, inventory, planning_hashes = _preprocess_fixture(tmp_path)
    runs = tmp_path / "smoke_runs"
    runs.mkdir()
    staging = runs / ".partial-smoke-1"
    final = runs / "smoke-1"
    staging.mkdir()
    stage_smoke_run(
        ready,
        staging,
        final,
        "smoke-1",
        gpu_id="0",
        preprocess_validator=lambda path: _validate_fixture_ready(
            path, inventory, planning_hashes
        ),
    )
    _, checkpoint = _smoke_output_fixture(staging)

    validated = validate_smoke_output(
        staging, checkpoint_loader=lambda _: checkpoint
    )

    assert validated["status"] == "PASS"
    assert validated["fold_output_dir"].endswith("fold_0")


def test_smoke_output_requires_exact_checkpoint_log_and_finite_epoch(
    tmp_path: Path,
) -> None:
    run = tmp_path / "smoke-1"
    run.mkdir()
    fold, checkpoint = _smoke_output_fixture(run)

    inventory = validate_smoke_output(
        run, checkpoint_loader=lambda _: checkpoint
    )

    assert inventory["checkpoint_count"] == 2
    assert inventory["epoch_count"] == 1
    assert inventory["finite"] == {
        "train_loss": 0.8123,
        "val_loss": 0.7345,
        "learning_rate": 0.01,
    }
    assert inventory["actual_validation_output_count"] == 0
    assert inventory["fold_output_dir"].endswith("fold_0")
    assert all(math.isfinite(value) for value in inventory["finite"].values())

    bad = dict(checkpoint)
    bad["logging"] = {**checkpoint["logging"], "train_losses": [float("nan")]}
    with pytest.raises(ContractError, match="finite train loss"):
        validate_smoke_output(run, checkpoint_loader=lambda _: bad)

    (fold / "checkpoint_final.pth").unlink()
    with pytest.raises(ContractError, match="checkpoint_final"):
        validate_smoke_output(run, checkpoint_loader=lambda _: checkpoint)


def test_smoke_output_rejects_validation_oof_or_result_publication(
    tmp_path: Path,
) -> None:
    run = tmp_path / "smoke-1"
    run.mkdir()
    fold, checkpoint = _smoke_output_fixture(run)
    (fold / "validation").mkdir()
    (fold / "validation" / "case.nii.gz").write_bytes(b"prediction")

    with pytest.raises(ContractError, match="actual validation"):
        validate_smoke_output(run, checkpoint_loader=lambda _: checkpoint)

    (fold / "validation" / "case.nii.gz").unlink()
    (fold / "validation").rmdir()
    oof = tmp_path / "oof_predictions"
    oof.mkdir()
    (oof / "case.nii.gz").write_bytes(b"prediction")
    with pytest.raises(ContractError, match="OOF publication"):
        validate_smoke_output(
            run,
            checkpoint_loader=lambda _: checkpoint,
            oof_roots=[oof],
        )

    results = tmp_path / "evaluation"
    results.mkdir()
    (results / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ContractError, match="result publication"):
        validate_smoke_output(
            run,
            checkpoint_loader=lambda _: checkpoint,
            result_roots=[results],
        )


def test_bundle_and_ready_publish_smoke_only_not_full_training(
    tmp_path: Path,
) -> None:
    ready, _, preprocess_inventory, planning_hashes = _preprocess_fixture(tmp_path)
    run = tmp_path / "smoke_runs" / "smoke-1"
    run.mkdir(parents=True)
    _, checkpoint = _smoke_output_fixture(run)
    smoke_inventory = validate_smoke_output(
        run, checkpoint_loader=lambda _: checkpoint
    )
    bundle = build_smoke_bundle(
        ready,
        run_id="smoke-1",
        committed_run_dir=run,
        inventory=smoke_inventory,
        preprocess_validator=lambda path: _validate_fixture_ready(
            path, preprocess_inventory, planning_hashes
        ),
    )
    bundle_path = run / "SMOKE_BUNDLE.json"
    _write_json(bundle_path, bundle)
    fixed = tmp_path / "manifests" / "SMOKE_READY.json"

    published = publish_smoke_ready(
        run,
        bundle_path,
        fixed,
        output_validator=lambda _: smoke_inventory,
        preprocess_validator=lambda path: _validate_fixture_ready(
            path, preprocess_inventory, planning_hashes
        ),
    )

    assert published["status"] == "COMMITTED"
    assert published["smoke_status"] == "PASS"
    assert published["smoke_training_performed"] is True
    assert published["full_training_status"] == "NOT_STARTED"
    assert published["full_training_performed"] is False
    assert published["oof_prediction_count"] == 0
    assert published["result_count"] == 0
    assert published["thesis_citable"] is False
    before = fixed.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_smoke_ready(
            run,
            bundle_path,
            fixed,
            output_validator=lambda _: smoke_inventory,
            preprocess_validator=lambda path: _validate_fixture_ready(
                path, preprocess_inventory, planning_hashes
            ),
        )
    assert fixed.read_bytes() == before


def test_one_epoch_runner_uses_official_trainer_without_actual_validation() -> None:
    calls: list[dict] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def device(name: str) -> str:
            return name

        @staticmethod
        def set_num_threads(value: int) -> None:
            assert value == 1

        @staticmethod
        def set_num_interop_threads(value: int) -> None:
            assert value == 1

    class FakeTrainer:
        num_epochs = 1
        disable_checkpointing = False
        output_folder = "/run/nnUNet_results/fold_0"

        def run_training(self) -> None:
            calls.append({"run_training": True})

        def perform_actual_validation(self, *_args) -> None:
            raise AssertionError("actual validation must not run in the smoke gate")

    def factory(**kwargs):
        calls.append(kwargs)
        return FakeTrainer()

    result = run_one_epoch_smoke(trainer_factory=factory, torch_module=FakeTorch())

    assert calls[0] == {
        "dataset_name_or_id": "901",
        "configuration": "3d_fullres",
        "fold": 0,
        "trainer_name": "nnUNetTrainer_1epoch",
        "plans_identifier": "nnUNetPlans",
        "continue_training": False,
        "device": "cuda",
    }
    assert calls[1] == {"run_training": True}
    assert result["actual_validation"] is False
    assert result["export_probabilities"] is False


def test_shell_wrapper_is_locked_no_clobber_and_publishes_only_after_validation() -> None:
    shell = (SCRIPTS / "baseline" / "run_petct_m0_smoke.sh").read_text(encoding="utf-8")

    assert "training is disabled" not in shell
    assert "flock" in shell
    assert "PREPROCESS_READY.json" in shell
    assert "SMOKE_BUNDLE.json" in shell
    assert "SMOKE_READY.json" in shell
    assert "mktemp -d" in shell
    assert "run_petct_m0_one_epoch.py" in shell
    assert "nnUNetTrainer_1epoch" in shell
    assert "export nnUNet_compile=true" in shell
    assert "LIBRARY_PATH" in shell
    assert "Refusing compile-time CUDA stub directory in LD_LIBRARY_PATH" in shell
    assert "export LD_LIBRARY_PATH" not in shell
    assert (
        "81dcabbb572826da2e9e5edcffb7ca98a1d4728f38a3892a4999dea74716f198"
        in shell
    )
    assert "validate-smoke" in shell
    assert "commit-run" in shell
    assert "publish-smoke-ready" in shell
    assert shell.index("validate-smoke") < shell.index("publish-smoke-ready")
    assert re.search(r"(^|\s)--npz(?:\s|$)", shell) is None
    assert re.search(r"(^|\s)--c(?:\s|$)", shell) is None
    assert "run_petct_m0_fold.sh" not in shell
