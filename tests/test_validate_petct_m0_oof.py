from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "baseline"))

from validate_petct_m0_oof import (  # noqa: E402
    ContractError,
    DATASET_FOLDER,
    EXPECTED_NNUNET_SOURCE_TREE_SHA256,
    FULL_TRAIN_READY_VERSION,
    OOF_CONTRACT_VERSION,
    OOF_PHASE,
    OOF_READY_VERSION,
    PROBABILITY_VERIFICATION_BOUNDARY,
    _validate_full_train_ready,
    build_oof_bundle,
    commit_fold_done,
    inspect_oof_pair,
    publish_oof_ready,
    stage_oof_run,
    validate_authoritative_splits,
    validate_fold_plan_binding,
    validate_natural_oof_binding,
    validate_oof_case_leaf,
    validate_oof_output,
    validate_oof_ready_receipt_only,
    validate_split_document,
)
import validate_petct_m0_oof as oof_contract  # noqa: E402
from common.petct_mainline_lineage import LineageContractError  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _case(patient: int, exam: int) -> str:
    return f"psma_p{patient:03d}_{20200101 + exam}"


def _cohort() -> tuple[list[str], dict[str, str]]:
    cases: list[str] = []
    case_to_patient: dict[str, str] = {}
    # 219 patients have two examinations and 159 have one: 597 / 378.
    for patient in range(378):
        examination_count = 2 if patient < 219 else 1
        for exam in range(examination_count):
            case_id = _case(patient, exam)
            cases.append(case_id)
            case_to_patient[case_id] = f"psma_p{patient:03d}"
    assert len(cases) == 597
    return sorted(cases), case_to_patient


def _splits() -> tuple[list[dict[str, list[str]]], dict[str, str]]:
    cases, case_to_patient = _cohort()
    patient_fold = {f"psma_p{patient:03d}": patient % 5 for patient in range(378)}
    folds: list[dict[str, list[str]]] = []
    for fold in range(5):
        val = [case for case in cases if patient_fold[case_to_patient[case]] == fold]
        train = [case for case in cases if patient_fold[case_to_patient[case]] != fold]
        folds.append({"train": train, "val": val})
    return folds, case_to_patient


def test_split_contract_proves_597_exact_once_and_378_patient_exclusion(
    tmp_path: Path,
) -> None:
    splits, _ = _splits()
    path = tmp_path / "splits_final.json"
    _write_json(path, splits)

    contract = validate_authoritative_splits(path)

    assert contract["fold_count"] == 5
    assert contract["case_count"] == 597
    assert contract["patient_count"] == 378
    assert contract["val_exact_once"] is True
    assert contract["patient_single_held_out_fold"] is True
    assert all(fold["train_val_patient_overlap"] == 0 for fold in contract["folds"])
    assert contract["splits_final"]["sha256"] == _sha(path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("duplicate_val", "exactly once"),
        ("patient_cross_fold", "single held-out fold"),
        ("train_val_overlap", "train/val case overlap"),
        ("missing_case", "train is not the val complement"),
    ],
)
def test_split_contract_rejects_leakage_and_incomplete_coverage(
    mutation: str, match: str
) -> None:
    splits, _ = _splits()
    if mutation == "duplicate_val":
        case_id = splits[0]["val"][0]
        splits[1]["val"].append(case_id)
        splits[1]["train"].remove(case_id)
    elif mutation == "patient_cross_fold":
        patient_cases = [_case(0, 0), _case(0, 1)]
        splits[0]["val"].remove(patient_cases[1])
        splits[0]["train"].append(patient_cases[1])
        splits[1]["train"].remove(patient_cases[1])
        splits[1]["val"].append(patient_cases[1])
    elif mutation == "train_val_overlap":
        splits[0]["train"].append(splits[0]["val"][0])
    else:
        splits[0]["train"].remove(splits[1]["val"][0])

    with pytest.raises(ValueError, match=match):
        validate_split_document(splits)


def _full_train_ready(
    root: Path,
    *,
    preprocess_ready: Path,
    splits_path: Path,
) -> Path:
    campaign = root / "campaign-oof-source"
    campaign.mkdir(parents=True)
    trainer_root = (
        campaign
        / "nnUNet_results"
        / DATASET_FOLDER
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )
    trainer_root.mkdir(parents=True, exist_ok=True)
    plans = trainer_root / "plans.json"
    dataset_json = trainer_root / "dataset.json"
    plans.write_text('{"plans_name":"nnUNetPlans"}\n', encoding="utf-8")
    dataset_json.write_text('{"name":"PSMA_M0_AutoPETVNorm"}\n', encoding="utf-8")
    bound = {
        "preprocess_ready": _sha(preprocess_ready),
        "smoke_ready": "f" * 64,
        "source_tree": EXPECTED_NNUNET_SOURCE_TREE_SHA256,
        "splits_final": _sha(splits_path),
    }
    training_contract = {
        "dataset_id": 901,
        "configuration": "3d_fullres",
        "folds": list(range(5)),
        "trainer": "nnUNetTrainer",
        "plans_identifier": "nnUNetPlans",
        "num_epochs": 1000,
        "device": "cuda",
        "actual_validation": False,
        "export_probabilities": False,
        "compile_contract": {"mode": "disabled"},
    }
    spec = {
        "status": "STAGED",
        "contract_version": FULL_TRAIN_READY_VERSION,
        "phase": "STANDARD_5FOLD_FULL_TRAINING",
        "campaign_id": campaign.name,
        "campaign_root": str(campaign.resolve()),
        "prerequisite_bound_hashes": bound,
        "training_contract": training_contract,
        "full_training_status": "NOT_STARTED",
        "oof_status": "NOT_STARTED",
        "oof_prediction_count": 0,
        "result_count": 0,
        "thesis_citable": False,
    }
    spec_path = campaign / "CAMPAIGN_SPEC.json"
    _write_json(spec_path, spec)
    fold_records = []
    for fold in range(5):
        checkpoint = trainer_root / f"fold_{fold}" / "checkpoint_final.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-fold-{fold}".encode())
        fold_receipt = {
            "status": "COMMITTED",
            "contract_version": FULL_TRAIN_READY_VERSION,
            "phase": "STANDARD_5FOLD_FULL_TRAINING",
            "campaign_id": campaign.name,
            "fold": fold,
            "prerequisite_bound_hashes": bound,
            "output_contract": {
                "status": "PASS",
                "fold": fold,
                "epoch_count": 1000,
                "checkpoint_count": 2,
                "split_sha256": _sha(splits_path),
                "oof_publication_count": 0,
                "result_publication_count": 0,
                "artifacts": {"checkpoint_final": _record(checkpoint)},
            },
            "full_fold_training_status": "PASS",
            "oof_status": "NOT_STARTED",
            "oof_prediction_count": 0,
            "result_count": 0,
            "thesis_citable": False,
        }
        fold_path = campaign / "fold_receipts" / f"fold_{fold}.json"
        _write_json(fold_path, fold_receipt)
        fold_records.append(_record(fold_path))
    receipt = {
        "status": "COMMITTED",
        "contract_version": FULL_TRAIN_READY_VERSION,
        "phase": "STANDARD_5FOLD_FULL_TRAINING",
        "campaign_id": campaign.name,
        "campaign_root": str(campaign.resolve()),
        "campaign_spec": _record(spec_path),
        "fold_receipts": fold_records,
        "prerequisite_bound_hashes": bound,
        "training_contract": training_contract,
        "full_training_status": "PASS",
        "full_training_performed": True,
        "folds_completed": list(range(5)),
        "checkpoint_count": 10,
        "oof_handoff_inputs_present": False,
        "actual_inference_gate_required": True,
        "oof_status": "NOT_STARTED",
        "oof_prediction_count": 0,
        "result_count": 0,
        "thesis_citable": False,
    }
    ready = root / "FULL_TRAIN_READY.json"
    _write_json(ready, receipt)
    return ready


def _stage_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[Path], dict[str, Any]]:
    splits, _ = _splits()
    splits_path = tmp_path / "authoritative" / "splits_final.json"
    _write_json(splits_path, splits)
    preprocess_ready = tmp_path / "PREPROCESS_READY.json"
    preprocess_ready.write_text('{"status":"fixture"}\n', encoding="utf-8")
    raw_root = tmp_path / "preprocess" / "nnUNet_raw"
    images = raw_root / DATASET_FOLDER / "imagesTr"
    labels = raw_root / DATASET_FOLDER / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir()
    for case_id in sorted({case for fold in splits for case in fold["val"]}):
        (images / f"{case_id}_0000.nii.gz").write_bytes(f"ct:{case_id}".encode())
        (images / f"{case_id}_0001.nii.gz").write_bytes(f"pet:{case_id}".encode())
        (labels / f"{case_id}.nii.gz").write_bytes(f"gt:{case_id}".encode())
    full_train = [
        _full_train_ready(
            tmp_path / "full-training",
            preprocess_ready=preprocess_ready,
            splits_path=splits_path,
        )
    ]
    preprocess_contract = {
        "status": "PASS",
        "nnunet_raw": str(raw_root.resolve()),
        "preprocess_run_dir": str((tmp_path / "preprocess").resolve()),
        "bound_hashes": {
            "preprocess_ready": _sha(preprocess_ready),
            "planning_splits_final": _sha(splits_path),
            "planning_nnunet_plans": "a" * 64,
            "planning_dataset_json": "b" * 64,
        },
        "raw_source_bindings": {
            name: {
                "link_path": str((raw_root / DATASET_FOLDER / name)),
                "target_path": str((raw_root / DATASET_FOLDER / name).resolve()),
                "policy": "TEST_DIRECTORY_FIXTURE",
            }
            for name in ("imagesTr", "labelsTr")
        },
    }
    staging = tmp_path / "oof_runs" / ".partial-oof-test"
    final = tmp_path / "oof_runs" / "oof-test"
    staging.mkdir(parents=True)
    stage_oof_run(
        preprocess_ready,
        full_train,
        splits_path,
        staging,
        final,
        "oof-test",
        preprocess_validator=lambda _path: preprocess_contract,
    )
    return staging, final, preprocess_ready, full_train, preprocess_contract


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("aggregate_status", "full_training_status"),
        ("fold_receipt", "fold receipt.*hash"),
        ("checkpoint", "checkpoint.*hash"),
    ],
)
def test_aggregate_full_train_ready_is_rehashed_before_oof(
    tmp_path: Path, mutation: str, match: str
) -> None:
    splits, _ = _splits()
    splits_path = tmp_path / "splits_final.json"
    _write_json(splits_path, splits)
    preprocess_ready = tmp_path / "PREPROCESS_READY.json"
    preprocess_ready.write_text('{"status":"fixture"}\n', encoding="utf-8")
    ready = _full_train_ready(
        tmp_path / "full-training",
        preprocess_ready=preprocess_ready,
        splits_path=splits_path,
    )
    assert set(
        _validate_full_train_ready([ready], preprocess_ready, splits_path)
    ) == set(range(5))

    aggregate = json.loads(ready.read_text(encoding="utf-8"))
    if mutation == "aggregate_status":
        aggregate["full_training_status"] = "NOT_STARTED"
        _write_json(ready, aggregate)
    elif mutation == "fold_receipt":
        fold_path = Path(aggregate["fold_receipts"][0]["path"])
        fold_path.write_text('{"status":"tampered"}\n', encoding="utf-8")
    else:
        campaign = Path(aggregate["campaign_root"])
        checkpoint = (
            campaign
            / "nnUNet_results"
            / DATASET_FOLDER
            / "nnUNetTrainer__nnUNetPlans__3d_fullres"
            / "fold_0"
            / "checkpoint_final.pth"
        )
        checkpoint.write_bytes(b"tampered")

    with pytest.raises((RuntimeError, ValueError), match=match):
        _validate_full_train_ready([ready], preprocess_ready, splits_path)


def test_stage_builds_fold_locked_val_only_plans(tmp_path: Path) -> None:
    staging, final, _, _, _ = _stage_fixture(tmp_path)
    spec = json.loads((staging / "OOF_SPEC.json").read_text(encoding="utf-8"))
    assert spec["status"] == "STAGED"
    assert spec["committed_run_dir"] == str(final.resolve())
    assert spec["case_count"] == 597
    assert spec["patient_count"] == 378
    assert spec["scribble_generation_count"] == 0
    assert spec["intent_generation_count"] == 0
    all_cases: list[str] = []
    for fold in range(5):
        plan = json.loads((staging / "fold_plans" / f"fold_{fold}.json").read_text())
        assert plan["fold"] == fold
        assert plan["use_folds"] == [fold]
        assert plan["save_probabilities"] is True
        assert plan["overwrite"] is False
        assert not set(plan["val_case_ids"]) & set(plan["train_case_ids"])
        assert set(plan["val_patient_ids"]) == {
            case["patient_id"] for case in plan["cases"]
        }
        assert (
            plan["authoritative_split_binding"]["splits_final_sha256"]
            == spec["splits_final"]["sha256"]
        )
        assert len(plan["cases"]) == len(plan["val_case_ids"])
        assert all("input_gt" in case for case in plan["cases"])
        all_cases.extend(plan["val_case_ids"])
    assert len(all_cases) == len(set(all_cases)) == 597


def test_stage_accepts_legacy_materialized_preprocess_without_raw_bindings(
    tmp_path: Path,
) -> None:
    _, _, preprocess_ready, full_train, preprocess_contract = _stage_fixture(tmp_path)
    legacy_contract = dict(preprocess_contract)
    legacy_contract.pop("raw_source_bindings")
    splits_path = tmp_path / "authoritative" / "splits_final.json"
    staging = tmp_path / "oof_runs" / ".partial-legacy-oof"
    final = tmp_path / "oof_runs" / "legacy-oof"
    staging.mkdir()

    stage_oof_run(
        preprocess_ready,
        full_train,
        splits_path,
        staging,
        final,
        "legacy-oof",
        preprocess_validator=lambda _path: legacy_contract,
    )

    assert (staging / "OOF_SPEC.json").is_file()
    assert validate_fold_plan_binding(staging, 0)[0].is_file()


def test_fold_plan_rejects_val_set_reassigned_to_another_authoritative_fold(
    tmp_path: Path,
) -> None:
    staging, _, _, _, _ = _stage_fixture(tmp_path)
    fold_zero_path = staging / "fold_plans" / "fold_0.json"
    fold_one = json.loads(
        (staging / "fold_plans" / "fold_1.json").read_text(encoding="utf-8")
    )
    fold_zero = json.loads(fold_zero_path.read_text(encoding="utf-8"))
    fold_zero["val_case_ids"] = fold_one["val_case_ids"]
    fold_zero["val_patient_ids"] = fold_one["val_patient_ids"]
    fold_zero["cases"] = fold_one["cases"]
    _write_json(fold_zero_path, fold_zero)

    with pytest.raises((RuntimeError, ValueError), match="authoritative splits_final"):
        validate_fold_plan_binding(staging, 0)


def test_oof_output_rejects_rehashed_plan_with_swapped_fold_model(
    tmp_path: Path,
) -> None:
    staging, _, _, _, _ = _stage_fixture(tmp_path)
    _materialize_fake_outputs(staging)
    fold_zero_path = staging / "fold_plans" / "fold_0.json"
    fold_zero = json.loads(fold_zero_path.read_text(encoding="utf-8"))
    fold_one = json.loads(
        (staging / "fold_plans" / "fold_1.json").read_text(encoding="utf-8")
    )
    fold_zero["model"] = fold_one["model"]
    _write_json(fold_zero_path, fold_zero)
    done_path = staging / "outputs" / "fold_0" / "FOLD_DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8"))
    done["plan"] = _record(fold_zero_path)
    _write_json(done_path, done)

    with pytest.raises(
        (RuntimeError, ValueError), match="model differs|receipt/model order"
    ):
        validate_oof_output(staging, output_inspector=_valid_inspector)


def _materialize_fake_outputs(run_root: Path) -> None:
    for plan_path in sorted((run_root / "fold_plans").glob("fold_*.json")):
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        fold_root = run_root / "outputs" / f"fold_{plan['fold']}"
        masks = fold_root / "masks"
        probabilities = fold_root / "probabilities"
        for case in plan["cases"]:
            (masks / f"{case['case_id']}.nii.gz").write_bytes(
                f"mask:{case['case_id']}".encode()
            )
            (probabilities / f"{case['case_id']}.npz").write_bytes(
                f"prob:{case['case_id']}".encode()
            )
        commit_fold_done(run_root, int(plan["fold"]))


def _valid_inspector(
    _mask: Path, _probability: Path, _reference_ct: Path, _reference_gt: Path
) -> dict[str, Any]:
    return {
        "mask_shape": [2, 2, 2],
        "probability_shape": [2, 2, 2],
        "mask_values": [0, 1],
        "probability_key": "foreground_probability",
        "probability_dtype": "float32",
        "probability_finite": True,
        "probability_min": 0.0,
        "probability_max": 1.0,
        "probability_verification_boundary": PROBABILITY_VERIFICATION_BOUNDARY,
    }


def test_oof_pair_requires_mask_on_ct_and_gt_grid_and_declares_npz_boundary(
    tmp_path: Path,
) -> None:
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    mask = tmp_path / "mask.nii.gz"
    ct = tmp_path / "ct.nii.gz"
    gt = tmp_path / "gt.nii.gz"
    probability = tmp_path / "probability.npz"
    nib.save(
        nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.uint8), affine), str(mask)
    )
    nib.save(
        nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.float32), affine), str(ct)
    )
    nib.save(
        nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.uint8), affine), str(gt)
    )
    np.savez_compressed(
        probability,
        foreground_probability=np.zeros((2, 3, 4), dtype=np.float32),
    )

    inspection = inspect_oof_pair(mask, probability, ct, gt)
    assert (
        inspection["probability_verification_boundary"]
        == PROBABILITY_VERIFICATION_BOUNDARY
    )
    assert inspection["reference_grid"]["shape"] == [2, 3, 4]

    shifted = affine.copy()
    shifted[0, 3] = 5.0
    nib.save(
        nib.Nifti1Image(np.zeros((2, 3, 4), dtype=np.uint8), shifted), str(gt)
    )
    with pytest.raises(RuntimeError, match="reference GT"):
        inspect_oof_pair(mask, probability, ct, gt)


def test_oof_output_and_ready_bind_every_case_to_fold_and_model_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging, final, preprocess_ready, full_train, preprocess_contract = _stage_fixture(
        tmp_path
    )
    _materialize_fake_outputs(staging)
    inventory = validate_oof_output(staging, output_inspector=_valid_inspector)
    assert inventory["case_count"] == 597
    assert inventory["prediction_count"] == 597
    assert inventory["foreground_probability_count"] == 597
    assert inventory["val_exact_once"] is True
    assert inventory["patient_single_held_out_fold"] is True
    assert len(inventory["cases"]) == 597
    assert all(record["checkpoint_sha256"] for record in inventory["cases"])
    assert all(record["plans_sha256"] for record in inventory["cases"])
    assert all(record["source_tree_sha256"] for record in inventory["cases"])

    bundle = build_oof_bundle(
        preprocess_ready,
        full_train,
        run_id="oof-test",
        committed_run_dir=final,
        inventory=inventory,
        preprocess_validator=lambda _path: preprocess_contract,
    )
    _write_json(staging / "OOF_BUNDLE.json", bundle)
    staging.rename(final)
    ready_path = tmp_path / "OOF_READY.json"
    ready = publish_oof_ready(
        final,
        final / "OOF_BUNDLE.json",
        ready_path,
        output_validator=lambda run: validate_oof_output(
            run, output_inspector=_valid_inspector
        ),
        preprocess_validator=lambda _path: preprocess_contract,
    )
    assert ready["schema_version"] == OOF_READY_VERSION
    assert ready["status"] == "COMMITTED"
    assert ready["oof_status"] == "PASS"
    assert ready["patient_excluded"] is True
    assert ready["prediction_count"] == 597
    assert ready["foreground_probability_count"] == 597
    assert (
        ready["probability_verification_boundary"]
        == PROBABILITY_VERIFICATION_BOUNDARY
    )
    assert ready["scribble_generation_count"] == 0
    assert ready["intent_generation_count"] == 0
    assert ready["experiment_result_count"] == 0

    allowed_hashes = {ready_path.resolve(), (final / "OOF_BUNDLE.json").resolve()}
    original_sha256 = oof_contract._sha256

    def receipt_only_sha256(path: Path) -> str:
        resolved = Path(path).resolve()
        if resolved not in allowed_hashes:
            raise AssertionError(f"receipt-only validator touched a leaf: {resolved}")
        return original_sha256(resolved)

    monkeypatch.setattr(oof_contract, "_sha256", receipt_only_sha256)
    receipt_only = validate_oof_ready_receipt_only(ready_path)
    assert receipt_only["validation_scope"] == "RECEIPT_ONLY_NO_LEAF_IO"
    assert len(receipt_only["cases"]) == 597
    assert receipt_only["prediction_sources"] == ["DEDICATED_OOF_INFERENCE"]
    monkeypatch.setattr(oof_contract, "_sha256", original_sha256)

    with pytest.raises(FileExistsError, match="overwrite"):
        publish_oof_ready(
            final,
            final / "OOF_BUNDLE.json",
            ready_path,
            output_validator=lambda run: inventory,
            preprocess_validator=lambda _path: preprocess_contract,
        )


def test_leaf_truth_binding_requires_observed_source_and_oof_hash_equality(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "oof"
    run_dir.mkdir()
    source: dict[str, Any] = {
        "case_id": "case-a",
        "patient_id": "patient-a",
        "held_out_fold": 2,
    }
    for modality, content in (("ct", b"ct"), ("pet", b"pet"), ("gt", b"gt")):
        path = tmp_path / f"{modality}.nii.gz"
        path.write_bytes(content)
        source[f"{modality}_path"] = str(path)
        source[f"{modality}_bytes"] = path.stat().st_size
        source[f"{modality}_sha256"] = _sha(path)
    mask = run_dir / "mask.nii.gz"
    probability = run_dir / "probability.npz"
    mask.write_bytes(b"mask")
    probability.write_bytes(b"probability")
    ready_path = tmp_path / "OOF_READY.json"
    ready_path.write_text('{"status":"fixture"}\n', encoding="utf-8")
    case = {
        "case_id": "case-a",
        "patient_id": "patient-a",
        "held_out_fold": 2,
        "mask": {**_record(mask), "path": mask.relative_to(run_dir).as_posix()},
        "foreground_probability": {
            **_record(probability),
            "path": probability.relative_to(run_dir).as_posix(),
        },
        **{
            f"input_{modality}_bytes": source[f"{modality}_bytes"]
            for modality in ("ct", "pet", "gt")
        },
        **{
            f"input_{modality}_sha256": source[f"{modality}_sha256"]
            for modality in ("ct", "pet", "gt")
        },
    }
    validated = {
        "status": "PASS",
        "patient_excluded": True,
        "ready_path": str(ready_path.resolve()),
        "ready_sha256": _sha(ready_path),
        "run_dir": str(run_dir.resolve()),
        "cases": {"case-a": case},
    }
    binding = validate_oof_case_leaf(
        validated,
        ready_path=ready_path,
        case_id="case-a",
        source_record=source,
    )
    assert binding["inputs"]["gt"]["sha256"] == source["gt_sha256"]
    assert binding["m0"]["sha256"] == _sha(mask)
    assert len(binding["binding_sha256"]) == 64

    Path(source["gt_path"]).write_bytes(b"tampered")
    with pytest.raises(ContractError, match="observed/source/OOF"):
        validate_oof_case_leaf(
            validated,
            ready_path=ready_path,
            case_id="case-a",
            source_record=source,
        )


def test_actual_validation_handoff_source_receipt_is_semantically_hash_bound(
    tmp_path: Path,
) -> None:
    handoff = tmp_path / "HANDOFF_SOURCE.json"
    _write_json(
        handoff,
        {
            "status": "PASS",
            "fold": 3,
            "prediction_source": "TRAINING_ACTUAL_VALIDATION_HANDOFF",
        },
    )
    artifacts = {"handoff_source": _record(handoff)}
    assert oof_contract._validate_prediction_source_artifacts(
        "TRAINING_ACTUAL_VALIDATION_HANDOFF", artifacts, fold=3
    ) == artifacts
    with pytest.raises(ContractError, match="semantic binding"):
        oof_contract._validate_prediction_source_artifacts(
            "TRAINING_ACTUAL_VALIDATION_HANDOFF", artifacts, fold=2
        )
    with pytest.raises(ContractError, match="must not claim"):
        oof_contract._validate_prediction_source_artifacts(
            "DEDICATED_OOF_INFERENCE", artifacts, fold=3
        )


def test_oof_output_rejects_missing_probability_or_train_case(tmp_path: Path) -> None:
    staging, _, _, _, _ = _stage_fixture(tmp_path)
    _materialize_fake_outputs(staging)
    first_plan = json.loads(
        (staging / "fold_plans" / "fold_0.json").read_text(encoding="utf-8")
    )
    missing_case = first_plan["val_case_ids"][0]
    (staging / "outputs" / "fold_0" / "probabilities" / f"{missing_case}.npz").unlink()
    with pytest.raises(ValueError, match="probability inventory"):
        validate_oof_output(staging, output_inspector=_valid_inspector)

    (
        staging / "outputs" / "fold_0" / "probabilities" / f"{missing_case}.npz"
    ).write_bytes(b"restored")
    train_case = first_plan["train_case_ids"][0]
    (staging / "outputs" / "fold_0" / "masks" / f"{train_case}.nii.gz").write_bytes(
        b"leak"
    )
    with pytest.raises(ValueError, match="mask inventory"):
        validate_oof_output(staging, output_inspector=_valid_inspector)


def test_natural_binding_requires_matching_case_patient_fold_ready_and_m0_hash(
    tmp_path: Path,
) -> None:
    m0 = tmp_path / "m0" / "case.nii.gz"
    probability = tmp_path / "prob" / "case.npz"
    m0.parent.mkdir(parents=True)
    probability.parent.mkdir(parents=True)
    m0.write_bytes(b"binary-mask")
    probability.write_bytes(b"foreground-probability")
    ready_path = tmp_path / "OOF_READY.json"
    ready_path.write_text('{"status":"fixture"}\n', encoding="utf-8")
    case_record = {
        "case_id": "psma_patienta_20200101",
        "patient_id": "psma_patienta",
        "held_out_fold": 3,
        "mask": _record(m0),
        "foreground_probability": _record(probability),
        "checkpoint_sha256": "1" * 64,
        "plans_sha256": "2" * 64,
        "dataset_json_sha256": "3" * 64,
        "source_tree_sha256": EXPECTED_NNUNET_SOURCE_TREE_SHA256,
        "splits_final_sha256": "4" * 64,
        "preprocess_ready_sha256": "5" * 64,
        "full_train_ready_sha256": "6" * 64,
        "fold_receipt_sha256": "7" * 64,
        "input_ct_sha256": "8" * 64,
        "input_pet_sha256": "9" * 64,
        "input_gt_sha256": "a" * 64,
    }
    validated_ready = {
        "status": "PASS",
        "schema_version": OOF_READY_VERSION,
        "contract_version": OOF_CONTRACT_VERSION,
        "phase": OOF_PHASE,
        "ready_path": str(ready_path.resolve()),
        "ready_sha256": _sha(ready_path),
        "run_dir": str(tmp_path.resolve()),
        "patient_excluded": True,
        "cases": {case_record["case_id"]: case_record},
    }

    binding = validate_natural_oof_binding(
        ready_path,
        case_id=case_record["case_id"],
        patient_id=case_record["patient_id"],
        m0_path=m0,
        ready_validator=lambda _path: validated_ready,
    )
    assert binding["kind"] == "patient_excluded_oof"
    assert binding["held_out_fold"] == 3
    assert binding["m0_sha256"] == _sha(m0)
    assert case_record["patient_id"] not in json.dumps(binding).casefold()
    assert "patient_id" not in binding
    assert "case_id" not in binding

    with pytest.raises(ValueError, match="patient"):
        validate_natural_oof_binding(
            ready_path,
            case_id=case_record["case_id"],
            patient_id="wrong-patient",
            m0_path=m0,
            ready_validator=lambda _path: validated_ready,
        )
    with pytest.raises(ValueError, match="M0 path"):
        validate_natural_oof_binding(
            ready_path,
            case_id=case_record["case_id"],
            patient_id=case_record["patient_id"],
            m0_path=tmp_path / "wrong.nii.gz",
            ready_validator=lambda _path: validated_ready,
        )
    m0.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        validate_natural_oof_binding(
            ready_path,
            case_id=case_record["case_id"],
            patient_id=case_record["patient_id"],
            m0_path=m0,
            ready_validator=lambda _path: validated_ready,
        )


def test_natural_binding_routes_v6_oof_ready_to_v6_validator(tmp_path: Path) -> None:
    ready_path = tmp_path / "OOF_READY_v6.json"
    ready_path.write_text(
        json.dumps({"schema_version": "PETCT-M0-V6-OOF-READY-v1.0"}),
        encoding="utf-8",
    )
    with pytest.raises(LineageContractError, match="M0 v6 OOF"):
        validate_natural_oof_binding(
            ready_path,
            case_id="case",
            patient_id="patient",
            m0_path=tmp_path / "m0.nii.gz",
        )


def test_contract_constants_do_not_claim_results_or_downstream_actions() -> None:
    assert OOF_CONTRACT_VERSION == "PETCT-M0-OOF-v1.0"
    assert OOF_PHASE == "PATIENT_EXCLUDED_5FOLD_OOF"


def test_launcher_orders_full_ready_then_five_folds_then_oof_ready() -> None:
    shell = (SCRIPTS / "baseline" / "run_petct_m0_oof.sh").read_text(encoding="utf-8")
    assert 'FULL_TRAIN_READY="${EXP_ROOT}/manifests/FULL_TRAIN_READY.json"' in shell
    assert 'OOF_READY="${EXP_ROOT}/manifests/OOF_READY.json"' in shell
    assert "for FOLD in 0 1 2 3 4" in shell
    assert '"${VALIDATOR}" publish' in shell
    assert shell.index('"${VALIDATOR}" stage') < shell.index("for FOLD in 0 1 2 3 4")
    assert shell.index("for FOLD in 0 1 2 3 4") < shell.index(
        '"${VALIDATOR}" validate-oof'
    )
    assert shell.index('"${VALIDATOR}" validate-oof') < shell.index(
        '"${VALIDATOR}" publish'
    )
    assert 'if [[ "${OOF_HANDOFF_AVAILABLE}" == "true" ]]' in shell
    assert "--from-actual-validation" in shell
    assert 'ready.get("oof_handoff_inputs_present") is True' in shell
    assert shell.index("--from-actual-validation") < shell.index(
        '--device cuda:0 --source-root "${NNUNET_SOURCE}"'
    )
    assert "scribble" not in shell.casefold()
    assert "intent" not in shell.casefold()
