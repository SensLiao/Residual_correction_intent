from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "baseline"))

from run_petct_m0_full_fold import run_standard_fold  # noqa: E402
from validate_petct_m0_full_training import (  # noqa: E402
    COMPILE_MODES,
    CONFIGURATION,
    DATASET_ID,
    FOLDS,
    NUM_EPOCHS,
    PLANS_IDENTIFIER,
    TRAINER,
    ContractError,
    build_fold_receipt,
    determine_fold_action,
    initialize_campaign,
    publish_full_train_ready,
    publish_fold_receipt,
    validate_campaign,
    validate_fold_completion,
    validate_fold_receipt,
    validate_full_training_prerequisites,
    validate_split_contract,
    validate_training_prerequisites,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_fixture(tmp_path: Path, case_count: int = 10) -> Path:
    identifiers = [f"psma_{index:04d}" for index in range(case_count)]
    folds = []
    for fold in range(5):
        val = identifiers[fold::5]
        folds.append(
            {"train": [item for item in identifiers if item not in val], "val": val}
        )
    path = tmp_path / "splits_final.json"
    _write_json(path, folds)
    return path


def _prerequisites(tmp_path: Path, *, case_count: int = 10) -> dict:
    preprocess = tmp_path / "manifests" / "PREPROCESS_READY.json"
    smoke = tmp_path / "manifests" / "SMOKE_READY.json"
    source = tmp_path / "upstream" / "nnUNet"
    source.mkdir(parents=True)
    _write_json(preprocess, {"status": "COMMITTED"})
    _write_json(smoke, {"status": "COMMITTED"})
    splits = _split_fixture(tmp_path, case_count)
    split_contract = validate_split_contract(splits, expected_case_count=case_count)
    return {
        "status": "PASS",
        "paths": {
            "preprocess_ready": str(preprocess.resolve()),
            "smoke_ready": str(smoke.resolve()),
            "source_root": str(source.resolve()),
            "splits_final": str(splits.resolve()),
        },
        "bound_hashes": {
            "preprocess_ready": _sha256(preprocess),
            "smoke_ready": _sha256(smoke),
            "source_tree": "a" * 64,
            "splits_final": _sha256(splits),
        },
        "runtime": {
            "status": "PASS",
            "version": "2.8.1",
            "source_tree_sha256": "a" * 64,
        },
        "split_contract": split_contract,
        "preprocess": {"status": "PASS"},
        "smoke": {"status": "PASS"},
    }


def _campaign(
    tmp_path: Path,
    *,
    actual_validation: bool = True,
    export_probabilities: bool = True,
) -> tuple[Path, dict]:
    prerequisites = _prerequisites(tmp_path)
    inference_ready = tmp_path / "manifests" / "INFERENCE_SMOKE_READY.json"
    _write_json(inference_ready, {"status": "COMMITTED"})
    prerequisites["paths"]["inference_smoke_ready"] = str(inference_ready.resolve())
    prerequisites["bound_hashes"]["inference_smoke_ready"] = _sha256(inference_ready)
    prerequisites["bound_hashes"]["inference_smoke_bundle"] = "c" * 64
    prerequisites["inference_smoke"] = {"status": "PASS"}
    stub_dir = tmp_path / "cuda-stubs"
    stub_dir.mkdir()
    (stub_dir / "libcuda.so").write_bytes(b"compile-time-only-stub")
    root = tmp_path / "full_training_runs" / "full-1"
    root.mkdir(parents=True)
    initialize_campaign(
        root,
        "full-1",
        prerequisites,
        actual_validation=actual_validation,
        export_probabilities=export_probabilities,
        compile_mode="triton-stub-link",
        cuda_stub_dir=stub_dir,
    )
    return root, prerequisites


def _fold_output(
    campaign: Path,
    fold: int,
    val_ids: list[str],
    *,
    actual_validation: bool = True,
    export_probabilities: bool = True,
) -> tuple[Path, dict]:
    trainer_root = (
        campaign
        / "nnUNet_results"
        / "Dataset901_PSMA_M0_AutoPETVNorm"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )
    fold_root = trainer_root / f"fold_{fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    for name in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
        _write_json(trainer_root / name, {"name": name})
    (fold_root / "checkpoint_final.pth").write_bytes(b"final")
    (fold_root / "checkpoint_best.pth").write_bytes(b"best")
    (fold_root / "progress.png").write_bytes(b"png")
    _write_json(fold_root / "debug.json", {"num_epochs": 1000})
    (fold_root / "training_log_2026_7_17_00_00_00.txt").write_text(
        "Epoch 0\ntrain_loss 1.0\nval_loss 1.0\n"
        "Epoch 999\ntrain_loss 0.1\nval_loss 0.2\nTraining done.\n",
        encoding="utf-8",
    )
    console = campaign / "logs" / f"fold_{fold}" / "attempt_1.log"
    console.parent.mkdir(parents=True, exist_ok=True)
    console.write_text("standard fold completed\n", encoding="utf-8")
    campaign_spec = json.loads(
        (campaign / "CAMPAIGN_SPEC.json").read_text(encoding="utf-8")
    )
    _write_json(
        console.with_suffix(".runtime.json"),
        {
            "status": "FOLD_PROCESS_COMPLETED",
            "dataset_id": 901,
            "configuration": "3d_fullres",
            "fold": fold,
            "trainer": "nnUNetTrainer",
            "plans_identifier": "nnUNetPlans",
            "num_epochs": 1000,
            "actual_validation": actual_validation,
            "export_probabilities": export_probabilities,
            "compile_contract": campaign_spec["training_contract"]["compile_contract"],
            "output_folder": str(fold_root),
        },
    )
    if actual_validation:
        validation = fold_root / "validation"
        validation.mkdir()
        for identifier in val_ids:
            (validation / f"{identifier}.nii.gz").write_bytes(b"mask")
            if export_probabilities:
                (validation / f"{identifier}.npz").write_bytes(b"probability")
                (validation / f"{identifier}.pkl").write_bytes(b"properties")
        _write_json(
            validation / "summary.json",
            {
                "metric_per_case": [{} for _ in val_ids],
                "foreground_mean": {"Dice": 0.1},
            },
        )
    logging = {
        "train_losses": [1.0] * 1000,
        "val_losses": [1.0] * 1000,
        "lrs": [0.01] * 1000,
        "epoch_start_timestamps": [float(index) for index in range(1000)],
        "epoch_end_timestamps": [float(index) + 0.5 for index in range(1000)],
    }
    checkpoint = {
        "network_weights": {"encoder.weight": object()},
        "optimizer_state": {"state": {}},
        "logging": logging,
        "current_epoch": 1000,
        "trainer_name": "nnUNetTrainer",
    }
    return fold_root, checkpoint


def test_full_training_identity_preserves_standard_nnunet_defaults() -> None:
    assert DATASET_ID == 901
    assert CONFIGURATION == "3d_fullres"
    assert FOLDS == (0, 1, 2, 3, 4)
    assert TRAINER == "nnUNetTrainer"
    assert PLANS_IDENTIFIER == "nnUNetPlans"
    assert NUM_EPOCHS == 1000
    assert COMPILE_MODES == {"triton-stub-link", "disabled"}


def test_split_contract_requires_five_exact_disjoint_folds(tmp_path: Path) -> None:
    split_path = _split_fixture(tmp_path)

    contract = validate_split_contract(split_path, expected_case_count=10)

    assert contract["fold_count"] == 5
    assert contract["case_count"] == 10
    assert contract["validation_exact_once"] is True
    assert contract["sha256"] == _sha256(split_path)
    assert contract["folds"]["0"]["val"] == ["psma_0000", "psma_0005"]

    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload[1]["val"].append(payload[0]["val"][0])
    _write_json(split_path, payload)
    with pytest.raises(ContractError, match="exactly once|disjoint|universe"):
        validate_split_contract(split_path, expected_case_count=10)


def test_prerequisites_bind_preprocess_smoke_source_and_split_hashes(
    tmp_path: Path,
) -> None:
    prerequisites = _prerequisites(tmp_path)
    paths = prerequisites["paths"]

    validated = validate_training_prerequisites(
        Path(paths["preprocess_ready"]),
        Path(paths["smoke_ready"]),
        Path(paths["source_root"]),
        expected_case_count=10,
        preprocess_validator=lambda _: {
            "status": "PASS",
            "preprocess_run_dir": str(tmp_path),
            "nnunet_preprocessed": str(tmp_path),
            "bound_hashes": {
                "preprocess_ready": _sha256(Path(paths["preprocess_ready"]))
            },
        },
        smoke_validator=lambda _: {
            "status": "PASS",
            "bound_hashes": {"smoke_ready": _sha256(Path(paths["smoke_ready"]))},
        },
        runtime_validator=lambda _: {
            "status": "PASS",
            "version": "2.8.1",
            "source_tree_sha256": "a" * 64,
        },
        splits_path=Path(paths["splits_final"]),
    )

    assert validated["bound_hashes"]["source_tree"] == "a" * 64
    assert validated["bound_hashes"]["splits_final"] == _sha256(
        Path(paths["splits_final"])
    )
    assert validated["split_contract"]["case_count"] == 10


def test_full_training_prerequisites_require_current_inference_smoke_ready(
    tmp_path: Path,
) -> None:
    base = _prerequisites(tmp_path)
    inference_ready = tmp_path / "manifests" / "INFERENCE_SMOKE_READY.json"
    _write_json(inference_ready, {"status": "COMMITTED"})
    observed: dict[str, object] = {}

    def capture_base(*_: object, **kwargs: object) -> dict:
        observed.update(kwargs)
        return base

    validated = validate_full_training_prerequisites(
        Path(base["paths"]["preprocess_ready"]),
        Path(base["paths"]["smoke_ready"]),
        inference_ready,
        Path(base["paths"]["source_root"]),
        expected_case_count=10,
        splits_path=Path(base["paths"]["splits_final"]),
        base_prerequisite_validator=capture_base,
        inference_ready_validator=lambda path, current: {
            "status": "PASS",
            "ready_sha256": _sha256(path),
            "bundle_sha256": "c" * 64,
            "base_bound_hashes": current["bound_hashes"],
        },
    )

    assert validated["paths"]["inference_smoke_ready"] == str(inference_ready.resolve())
    assert validated["bound_hashes"]["inference_smoke_ready"] == _sha256(
        inference_ready
    )
    assert validated["bound_hashes"]["inference_smoke_bundle"] == "c" * 64
    assert observed["expected_case_count"] == 10
    assert observed["splits_path"] == Path(base["paths"]["splits_final"])


def test_campaign_is_receipt_first_and_no_clobber(tmp_path: Path) -> None:
    root, prerequisites = _campaign(tmp_path)

    validated = validate_campaign(root, prerequisite_validator=lambda _: prerequisites)

    assert validated["training_contract"]["trainer"] == "nnUNetTrainer"
    assert validated["training_contract"]["num_epochs"] == 1000
    assert validated["training_contract"]["actual_validation"] is True
    assert validated["training_contract"]["export_probabilities"] is True
    assert validated["training_contract"]["compile_contract"]["mode"] == (
        "triton-stub-link"
    )
    assert (
        validated["training_contract"]["compile_contract"][
            "ld_library_path_stub_forbidden"
        ]
        is True
    )
    assert validated["full_training_status"] == "NOT_STARTED"
    assert (root / "fold_receipts").is_dir()
    assert (root / "nnUNet_results").is_dir()
    with pytest.raises(ContractError, match="empty"):
        initialize_campaign(root, "full-1", prerequisites)


def test_fold_action_is_fresh_resume_or_verified_skip(tmp_path: Path) -> None:
    root, _ = _campaign(tmp_path)

    assert determine_fold_action(root, 0) == "FRESH"
    fold_root = (
        root
        / "nnUNet_results"
        / "Dataset901_PSMA_M0_AutoPETVNorm"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        / "fold_0"
    )
    fold_root.mkdir(parents=True)
    (fold_root / "checkpoint_latest.pth").write_bytes(b"partial")
    assert determine_fold_action(root, 0) == "RESUME"

    receipt = root / "fold_receipts" / "fold_0.json"
    _write_json(receipt, {"status": "COMMITTED", "fold": 0})
    assert (
        determine_fold_action(
            root,
            0,
            completed_receipt_validator=lambda *_: {"status": "PASS"},
        )
        == "SKIP_VERIFIED"
    )


def test_standard_runner_preserves_1000_epochs_and_controls_actual_validation() -> None:
    calls: list[object] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    class FakeCudnn:
        deterministic = True
        benchmark = False

    class FakeBackends:
        cudnn = FakeCudnn()

    class FakeTorch:
        cuda = FakeCuda()
        backends = FakeBackends()

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
        num_epochs = 1000
        disable_checkpointing = False
        output_folder = "/run/fold_2"

        def run_training(self) -> None:
            calls.append("train")

        def perform_actual_validation(self, save_probabilities: bool) -> None:
            calls.append(("validate", save_probabilities))

    def factory(**kwargs):
        calls.append(kwargs)
        return FakeTrainer()

    def load_checkpoint(trainer, resume, validation_only, pretrained) -> None:
        calls.append((trainer, resume, validation_only, pretrained))

    stub_dir = Path(__file__).parent / "fake-stub"
    stub_hash = "b" * 64
    environment: dict[str, str] = {}
    result = run_standard_fold(
        fold=2,
        resume=True,
        actual_validation=True,
        export_probabilities=True,
        compile_mode="triton-stub-link",
        cuda_stub_dir=stub_dir,
        expected_stub_sha256=stub_hash,
        environment=environment,
        stub_validator=lambda path, expected: (
            path == stub_dir / "libcuda.so" and expected == stub_hash
        ),
        trainer_factory=factory,
        checkpoint_loader=load_checkpoint,
        torch_module=FakeTorch(),
    )

    assert calls[0]["dataset_name_or_id"] == "901"
    assert calls[0]["trainer_name"] == "nnUNetTrainer"
    assert calls[0]["continue_training"] is True
    assert calls[2:] == ["train", ("validate", True)]
    assert result["num_epochs"] == 1000
    assert result["actual_validation"] is True
    assert result["export_probabilities"] is True
    assert result["compile_contract"]["mode"] == "triton-stub-link"
    assert environment["nnUNet_compile"] == "true"
    assert environment["LIBRARY_PATH"].split(";")[0] == str(stub_dir)
    assert "LD_LIBRARY_PATH" not in environment


def test_runner_rejects_cuda_stub_in_runtime_loader_path(tmp_path: Path) -> None:
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    (stub_dir / "libcuda.so").write_bytes(b"stub")

    with pytest.raises(RuntimeError, match="LD_LIBRARY_PATH"):
        run_standard_fold(
            fold=0,
            resume=False,
            actual_validation=False,
            export_probabilities=False,
            compile_mode="triton-stub-link",
            cuda_stub_dir=stub_dir,
            expected_stub_sha256=_sha256(stub_dir / "libcuda.so"),
            environment={"LD_LIBRARY_PATH": str(stub_dir)},
            trainer_factory=lambda **_: None,
            checkpoint_loader=lambda *_: None,
            torch_module=object(),
        )


def test_fold_completion_requires_finite_1000_epoch_checkpoint_and_oof_inputs(
    tmp_path: Path,
) -> None:
    root, prerequisites = _campaign(tmp_path)
    split_contract = prerequisites["split_contract"]
    val_ids = split_contract["folds"]["0"]["val"]
    fold_root, checkpoint = _fold_output(root, 0, val_ids)

    inventory = validate_fold_completion(
        root,
        0,
        split_contract,
        actual_validation=True,
        export_probabilities=True,
        checkpoint_loader=lambda _: checkpoint,
    )

    assert inventory["epoch_count"] == 1000
    assert inventory["checkpoint_count"] == 2
    assert inventory["validation_case_count"] == len(val_ids)
    assert inventory["validation_probability_count"] == len(val_ids)
    assert inventory["oof_publication_count"] == 0
    assert inventory["result_publication_count"] == 0
    assert inventory["oof_handoff_inputs_present"] is True
    assert len(inventory["artifacts"]["validation_masks"]) == len(val_ids)
    assert len(inventory["artifacts"]["validation_probabilities"]) == len(val_ids)
    assert len(inventory["artifacts"]["validation_properties"]) == len(val_ids)

    receipt = build_fold_receipt(root, 0, prerequisites, inventory)
    publish_fold_receipt(root / "fold_receipts" / "fold_0.json", receipt)
    (fold_root / "validation" / f"{val_ids[0]}.npz").write_bytes(b"changed")
    with pytest.raises(ContractError, match="changed after receipt"):
        validate_fold_receipt(
            root,
            0,
            prerequisites,
            checkpoint_loader=lambda _: checkpoint,
        )

    checkpoint["logging"]["val_losses"][999] = float("nan")
    with pytest.raises(ContractError, match="finite"):
        validate_fold_completion(
            root,
            0,
            split_contract,
            actual_validation=True,
            export_probabilities=True,
            checkpoint_loader=lambda _: checkpoint,
        )


def test_interrupted_training_log_then_resume_can_publish_and_revalidate_receipt(
    tmp_path: Path,
) -> None:
    root, prerequisites = _campaign(
        tmp_path, actual_validation=False, export_probabilities=False
    )
    split_contract = prerequisites["split_contract"]
    fold_root, checkpoint = _fold_output(
        root,
        0,
        split_contract["folds"]["0"]["val"],
        actual_validation=False,
        export_probabilities=False,
    )
    interrupted = fold_root / "training_log_2026_7_16_23_59_59.txt"
    interrupted.write_text(
        "Epoch 0\ntrain_loss 1.0\nval_loss 1.0\n",
        encoding="utf-8",
    )

    inventory = validate_fold_completion(
        root,
        0,
        split_contract,
        actual_validation=False,
        export_probabilities=False,
        checkpoint_loader=lambda _: checkpoint,
    )

    assert inventory["training_log_count"] == 2
    assert inventory["historical_interrupted_training_log_count"] == 1
    assert inventory["artifacts"]["completion_training_log"]["path"].endswith(
        "training_log_2026_7_17_00_00_00.txt"
    )
    assert [
        record["path"]
        for record in inventory["artifacts"][
            "historical_interrupted_training_logs"
        ]
    ] == [
        interrupted.resolve().relative_to(root.resolve()).as_posix()
    ]

    receipt = build_fold_receipt(root, 0, prerequisites, inventory)
    publish_fold_receipt(root / "fold_receipts" / "fold_0.json", receipt)
    revalidated = validate_fold_receipt(
        root,
        0,
        prerequisites,
        checkpoint_loader=lambda _: checkpoint,
    )
    assert revalidated == inventory


def test_training_only_mode_remains_explicitly_blocked_from_oof_handoff(
    tmp_path: Path,
) -> None:
    root, prerequisites = _campaign(
        tmp_path, actual_validation=False, export_probabilities=False
    )
    split_contract = prerequisites["split_contract"]
    _, checkpoint = _fold_output(
        root,
        0,
        split_contract["folds"]["0"]["val"],
        actual_validation=False,
        export_probabilities=False,
    )

    inventory = validate_fold_completion(
        root,
        0,
        split_contract,
        actual_validation=False,
        export_probabilities=False,
        checkpoint_loader=lambda _: checkpoint,
    )

    assert inventory["validation_case_count"] == 0
    assert inventory["oof_handoff_inputs_present"] is False
    assert inventory["actual_inference_gate_required"] is True


def test_fold_receipt_and_full_ready_never_claim_oof_or_results(
    tmp_path: Path,
) -> None:
    root, prerequisites = _campaign(tmp_path)
    inventories = {}
    for fold in range(5):
        inventories[fold] = {
            "status": "PASS",
            "fold": fold,
            "epoch_count": 1000,
            "checkpoint_count": 2,
            "actual_validation": True,
            "export_probabilities": True,
            "oof_handoff_inputs_present": True,
            "oof_publication_count": 0,
            "result_publication_count": 0,
        }
        payload = build_fold_receipt(root, fold, prerequisites, inventories[fold])
        publish_fold_receipt(root / "fold_receipts" / f"fold_{fold}.json", payload)

    ready = tmp_path / "manifests" / "FULL_TRAIN_READY.json"
    published = publish_full_train_ready(
        root,
        ready,
        prerequisites,
        fold_validator=lambda _, fold, __: inventories[fold],
    )

    assert published["full_training_status"] == "PASS"
    assert published["folds_completed"] == [0, 1, 2, 3, 4]
    assert published["oof_status"] == "NOT_STARTED"
    assert published["oof_prediction_count"] == 0
    assert published["result_count"] == 0
    assert published["thesis_citable"] is False
    assert published["oof_handoff_inputs_present"] is True
    before = ready.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_full_train_ready(
            root,
            ready,
            prerequisites,
            fold_validator=lambda _, fold, __: inventories[fold],
        )
    assert ready.read_bytes() == before


def test_shells_enforce_receipts_resume_and_per_gpu_serial_schedule() -> None:
    fold_shell = (SCRIPTS / "baseline" / "run_petct_m0_fold.sh").read_text(encoding="utf-8")
    launcher = (SCRIPTS / "baseline" / "launch_petct_m0_full_training.sh").read_text(
        encoding="utf-8"
    )

    assert "training is disabled" not in fold_shell
    assert "PREPROCESS_READY.json" in fold_shell
    assert "SMOKE_READY.json" in fold_shell
    assert "INFERENCE_SMOKE_READY.json" in fold_shell
    assert "FULL_TRAIN_READY.json" in fold_shell
    assert "flock" in fold_shell
    assert "m0_gpu_" in fold_shell
    assert "fold-action" in fold_shell
    assert "run_petct_m0_full_fold.py" in fold_shell
    assert 'export nnUNet_n_proc_DA="${PETCT_NNUNET_N_PROC_DA:-4}"' in fold_shell
    assert "validate-fold" in fold_shell
    assert "compile_contract" in fold_shell
    assert "--runtime-receipt" in fold_shell
    assert "nnUNetTrainer_1epoch" not in fold_shell

    assert "init-campaign" in launcher
    assert "INFERENCE_SMOKE_READY.json" in launcher
    assert "--inference-smoke-ready" in launcher
    assert "run_petct_m0_fold.sh" in launcher
    assert "publish-full-ready" in launcher
    assert launcher.index('"${FOLD_RUNNER}" "${CAMPAIGN_ID}" 0') < launcher.index(
        "run_parallel_workers"
    )
    assert "run_gpu_sequence" in launcher
    assert "wait" in launcher
    assert "FULL_TRAIN_READY.json" in launcher
    assert "--compile-mode" in launcher
    assert "triton-stub-link" in launcher
    assert "scribble" not in (fold_shell + launcher).lower()
    assert "intent" not in (fold_shell + launcher).lower()
