from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "baseline"))

from prepare_nnunet_m0_dataset import (  # noqa: E402
    ContractError,
    _validate_env_evidence,
    _verify_record,
    _write_json_exclusive,
    build_derived_dataset_json,
    publish_planning_ready,
    source_tree_sha256,
    validate_autopetv_reference_dataset,
    validate_derived_dataset_json,
    validate_derived_dataset_file,
    validate_fingerprint,
    validate_plan_contract,
    validate_plan_normalization,
    write_derived_dataset,
)


def _source_dataset_json() -> dict:
    return {
        "channel_names": {"0": "CT", "1": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": 597,
        "file_ending": ".nii.gz",
        "name": "PSMA-PET-CT-Lesions_v2",
    }


def _valid_plans() -> dict:
    return {
        "dataset_name": "Dataset901_PSMA_M0_AutoPETVNorm",
        "plans_name": "nnUNetPlans",
        "experiment_planner_used": "ExperimentPlanner",
        "image_reader_writer": "SimpleITKIO",
        "foreground_intensity_properties_per_channel": {
            "0": {"mean": 1.0},
            "1": {"mean": 2.0},
        },
        "configurations": {
            "3d_fullres": {
                "data_identifier": "nnUNetPlans_3d_fullres",
                "preprocessor_name": "DefaultPreprocessor",
                "normalization_schemes": [
                    "CTNormalization",
                    "ZScoreNormalization",
                ],
                "use_mask_for_norm": [False, False],
            }
        },
    }


def _valid_fingerprint() -> dict:
    return {
        "spacings": [[2.0, 2.0, 2.0]] * 597,
        "shapes_after_crop": [[10, 10, 10]] * 597,
        "foreground_intensity_properties_per_channel": {
            "0": {"mean": 1.0},
            "1": {"mean": 2.0},
        },
        "median_relative_size_after_cropping": 1.0,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic_env_evidence(root: Path) -> Path:
    receipt = root / "petct_nnunet_v281.json"
    conda = root / "petct_nnunet_v281.conda-explicit.txt"
    pip = root / "petct_nnunet_v281.pip-freeze.txt"
    forbidden = root / "petct_nnunet_v281.forbidden-symlinks.txt"
    hashes = root / "petct_nnunet_v281.receipt-sha256.txt"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-NNUNET-ENV-v1.1",
                "status": "PREFLIGHT_PASS_PENDING_ATOMIC_EVIDENCE_PUBLICATION",
            }
        ),
        encoding="utf-8",
    )
    conda.write_text("@EXPLICIT\n", encoding="utf-8")
    pip.write_text("torch==2.6.0\n", encoding="utf-8")
    forbidden.write_text("", encoding="utf-8")
    hashes.write_text(
        "".join(
            f"{_sha256(path)}  {path}\n" for path in (receipt, conda, pip)
        ),
        encoding="utf-8",
    )
    evidence = (receipt, conda, pip, hashes, forbidden)
    records = [
        {"name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in evidence
    ]
    core = {
        "schema_version": "PETCT-NNUNET-ENV-EVIDENCE-BUNDLE-v1.0",
        "status": "ENVIRONMENT_EVIDENCE_COMPLETE",
        "setup_mode": "VERIFY_EXISTING_NO_INSTALL",
        "files": records,
    }
    bundle_sha = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    bundle_root = root / "evidence-bundles" / bundle_sha
    bundle_root.mkdir(parents=True)
    for source in evidence:
        (bundle_root / source.name).write_bytes(source.read_bytes())
    bundle = {**core, "bundle_sha256": bundle_sha}
    bundle_manifest = bundle_root / "bundle.json"
    bundle_manifest.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    marker = {
        "schema_version": "PETCT-NNUNET-ENV-MARKER-v1.0",
        "status": "ENVIRONMENT_EVIDENCE_COMPLETE",
        "setup_mode": "VERIFY_EXISTING_NO_INSTALL",
        "bundle_path": str(bundle_root.resolve()),
        "bundle_sha256": bundle_sha,
        "bundle_manifest_sha256": _sha256(bundle_manifest),
        "receipt_path": str((bundle_root / receipt.name).resolve()),
        "receipt_sha256": _sha256(bundle_root / receipt.name),
    }
    (root / "ENV_READY.done").write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def test_environment_evidence_requires_atomic_bundle_and_hash_bound_marker(
    tmp_path: Path,
) -> None:
    receipt = _write_atomic_env_evidence(tmp_path)
    environment, files = _validate_env_evidence(receipt)
    assert environment["schema_version"] == "PETCT-NNUNET-ENV-v1.1"
    assert files["completion_marker"].name == "ENV_READY.done"
    marker = json.loads(files["completion_marker"].read_text(encoding="utf-8"))
    bundle_receipt = Path(marker["bundle_path"]) / receipt.name
    bundle_receipt.write_text("tampered", encoding="utf-8")
    with pytest.raises(ContractError, match="bundle file changed"):
        _validate_env_evidence(receipt)


def test_environment_evidence_rejects_in_progress_marker(tmp_path: Path) -> None:
    receipt = _write_atomic_env_evidence(tmp_path)
    marker_path = tmp_path / "ENV_READY.done"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["status"] = "ENVIRONMENT_MUTATION_IN_PROGRESS_NOT_READY"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ContractError, match="not complete"):
        _validate_env_evidence(receipt)


def test_derived_dataset_uses_autopet_v_ct_pet_semantics_without_mutating_source() -> None:
    source = _source_dataset_json()
    original = copy.deepcopy(source)

    derived = build_derived_dataset_json(source)

    assert source == original
    assert derived["channel_names"] == {"0": "CT", "1": "PET"}
    assert derived["name"] == "PSMA_M0_AutoPETVNorm"
    assert derived["labels"] == source["labels"]
    assert derived["numTraining"] == 597
    assert derived["file_ending"] == ".nii.gz"


def test_derived_dataset_rejects_unexpected_source_channel_contract() -> None:
    source = _source_dataset_json()
    source["channel_names"] = {"0": "CT", "1": "PET"}

    with pytest.raises(ContractError, match="source channel_names"):
        build_derived_dataset_json(source)


def test_derived_dataset_validation_rejects_metadata_drift() -> None:
    source = _source_dataset_json()
    derived = build_derived_dataset_json(source)
    derived["channel_names"]["1"] = "CT"

    with pytest.raises(ContractError, match="derived dataset.json"):
        validate_derived_dataset_json(source, derived)


def test_write_derived_dataset_can_defer_receipt_until_final_path(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    target_path = tmp_path / "staging" / "dataset.json"
    target_path.parent.mkdir()
    source_path.write_text(json.dumps(_source_dataset_json()), encoding="utf-8")

    validation = write_derived_dataset(source_path, target_path)

    assert validation["status"] == "PASS"
    assert json.loads(target_path.read_text(encoding="utf-8"))["channel_names"] == {
        "0": "CT",
        "1": "PET",
    }


def test_dataset_receipt_does_not_claim_filesystem_immutability(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.json"
    target_path = tmp_path / "derived.json"
    receipt_path = tmp_path / "receipt.json"
    source_path.write_text(json.dumps(_source_dataset_json()), encoding="utf-8")
    target_path.write_text(
        json.dumps(build_derived_dataset_json(_source_dataset_json())),
        encoding="utf-8",
    )

    receipt = validate_derived_dataset_file(source_path, target_path, receipt_path)

    assert receipt.get("source_immutable") is not True
    assert receipt["source_files_modified_by_run"] is False
    assert receipt["filesystem_immutability_claimed"] is False


def test_plan_gate_accepts_ct_normalization_plus_pet_zscore() -> None:
    plans = _valid_plans()

    receipt = validate_plan_normalization(plans)

    assert receipt == {
        "configuration": "3d_fullres",
        "normalization_schemes": [
            "CTNormalization",
            "ZScoreNormalization",
        ],
        "use_mask_for_norm": [False, False],
    }


def test_plan_gate_binds_dataset_identity_and_exactly_two_channels() -> None:
    receipt = validate_plan_contract(_valid_plans())

    assert receipt["dataset_name"] == "Dataset901_PSMA_M0_AutoPETVNorm"
    assert receipt["plans_name"] == "nnUNetPlans"
    assert receipt["channel_keys"] == ["0", "1"]
    assert receipt["data_identifier"] == "nnUNetPlans_3d_fullres"


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda p: p.update(dataset_name="Dataset901_Wrong"), "dataset_name"),
        (lambda p: p.update(plans_name="OtherPlans"), "plans_name"),
        (
            lambda p: p["foreground_intensity_properties_per_channel"].pop("1"),
            "channel",
        ),
        (
            lambda p: p["foreground_intensity_properties_per_channel"].update(
                {"2": {"mean": 3.0}}
            ),
            "channel",
        ),
        (
            lambda p: p["configurations"]["3d_fullres"].update(
                data_identifier="Other_3d_fullres"
            ),
            "data_identifier",
        ),
        (
            lambda p: p.update(experiment_planner_used="OtherPlanner"),
            "experiment_planner_used",
        ),
        (
            lambda p: p["configurations"]["3d_fullres"].update(
                preprocessor_name="OtherPreprocessor"
            ),
            "preprocessor_name",
        ),
    ],
)
def test_plan_gate_rejects_wrong_identity_even_when_normalization_is_correct(
    mutation, error: str
) -> None:
    plans = _valid_plans()
    mutation(plans)

    with pytest.raises(ContractError, match=error):
        validate_plan_contract(plans)


def test_fingerprint_requires_597_cases_and_exactly_two_channels() -> None:
    receipt = validate_fingerprint(_valid_fingerprint())

    assert receipt["case_count"] == 597
    assert receipt["channel_keys"] == ["0", "1"]


@pytest.mark.parametrize("field", ["spacings", "shapes_after_crop"])
def test_fingerprint_rejects_wrong_case_count(field: str) -> None:
    fingerprint = _valid_fingerprint()
    fingerprint[field] = fingerprint[field][:-1]

    with pytest.raises(ContractError, match="597"):
        validate_fingerprint(fingerprint)


def test_fingerprint_rejects_extra_channel() -> None:
    fingerprint = _valid_fingerprint()
    fingerprint["foreground_intensity_properties_per_channel"]["2"] = {
        "mean": 3.0
    }

    with pytest.raises(ContractError, match="channel"):
        validate_fingerprint(fingerprint)


@pytest.mark.parametrize(
    "normalization_schemes,use_mask_for_norm",
    [
        (["CTNormalization", "CTNormalization"], [False, False]),
        (["CTNormalization", "ZScoreNormalization"], [False, True]),
        (["CTNormalization"], [False]),
    ],
)
def test_plan_gate_rejects_non_autopet_v_normalization_contract(
    normalization_schemes: list[str], use_mask_for_norm: list[bool]
) -> None:
    plans = {
        "configurations": {
            "3d_fullres": {
                "normalization_schemes": normalization_schemes,
                "use_mask_for_norm": use_mask_for_norm,
            }
        }
    }

    with pytest.raises(ContractError, match="normalization contract"):
        validate_plan_normalization(plans)


def test_source_tree_hash_is_deterministic_and_ignores_runtime_cache(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nnUNet"
    package = source / "nnunetv2"
    cache = package / "__pycache__"
    cache.mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (source / "setup.py").write_text("pass\n", encoding="utf-8")
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (cache / "module.pyc").write_bytes(b"runtime cache")

    first = source_tree_sha256(source)
    (cache / "module.pyc").write_bytes(b"changed runtime cache")
    second = source_tree_sha256(source)

    assert first == second
    (package / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert source_tree_sha256(source) != first


def test_publish_planning_ready_rejects_incomplete_evidence_chain(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "planning_runs" / "run-1"
    artifact = run_dir / "nnUNet_preprocessed" / "Dataset901_PSMA_M0_AutoPETVNorm" / "nnUNetPlans.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    evidence = tmp_path / "audit.json"
    evidence.write_text('{"status":"PASS"}\n', encoding="utf-8")
    run_receipt = run_dir / "PLANNING_BUNDLE.json"
    bundle = {
        "status": "VALIDATED",
        "contract_version": "2.0.0",
        "phase": "PLANNING_ONLY",
        "preprocessing_status": "NOT_STARTED",
        "run_id": "run-1",
        "committed_run_dir": str(run_dir.resolve()),
        "dataset": {
            "id": 901,
            "folder": "Dataset901_PSMA_M0_AutoPETVNorm",
        },
        "source_files_modified_by_run": False,
        "filesystem_immutability_claimed": False,
        "evidence": {
            "audit": {"path": str(evidence.resolve()), "sha256": _sha256(evidence)}
        },
        "artifacts": {
            "plans": {
                "path": str(artifact.relative_to(run_dir)).replace("\\", "/"),
                "sha256": _sha256(artifact),
            }
        },
        "plan_contract": validate_plan_contract(_valid_plans()),
    }
    run_receipt.write_text(json.dumps(bundle), encoding="utf-8")
    ready = tmp_path / "manifests" / "PLANNING_READY.json"
    ready.parent.mkdir()

    with pytest.raises(ContractError, match="critical field|allowlist|dataset identity"):
        publish_planning_ready(run_dir, run_receipt, ready)

    assert not ready.exists()


def test_publish_planning_ready_rejects_tampered_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifact = run_dir / "plan.json"
    artifact.write_text("before", encoding="utf-8")
    receipt = run_dir / "bundle.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "VALIDATED",
                "phase": "PLANNING_ONLY",
                "preprocessing_status": "NOT_STARTED",
                "run_id": "run",
                "committed_run_dir": str(run_dir.resolve()),
                "dataset": {
                    "id": 901,
                    "folder": "Dataset901_PSMA_M0_AutoPETVNorm",
                },
                "evidence": {},
                "artifacts": {
                    "plans": {"path": "plan.json", "sha256": _sha256(artifact)}
                },
            }
        ),
        encoding="utf-8",
    )
    artifact.write_text("after", encoding="utf-8")

    with pytest.raises(ContractError, match="hash"):
        _verify_record(artifact, {"sha256": _sha256(receipt)}, label="plan")


def test_publish_planning_ready_never_overwrites_existing_receipt(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    receipt = run_dir / "bundle.json"
    bundle = {
        "status": "VALIDATED",
        "phase": "PLANNING_ONLY",
        "preprocessing_status": "NOT_STARTED",
        "run_id": "run",
        "committed_run_dir": str(run_dir.resolve()),
        "dataset": {"id": 901, "folder": "Dataset901_PSMA_M0_AutoPETVNorm"},
        "evidence": {},
        "artifacts": {},
    }
    receipt.write_text(json.dumps(bundle), encoding="utf-8")
    ready = tmp_path / "ready.json"
    ready.write_text('{"owner":"first"}\n', encoding="utf-8")
    before = ready.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json_exclusive(ready, {"owner": "second"})

    assert ready.read_bytes() == before


def test_shell_has_locked_planning_smoke_and_full_training_gates() -> None:
    project = Path(__file__).resolve().parents[1]
    prepare = (project / "scripts" / "baseline" / "prepare_petct_m0_nnunet.sh").read_text(
        encoding="utf-8"
    )
    smoke = (project / "scripts" / "baseline" / "run_petct_m0_smoke.sh").read_text(
        encoding="utf-8"
    )
    fold = (project / "scripts" / "baseline" / "run_petct_m0_fold.sh").read_text(
        encoding="utf-8"
    )
    launcher = (
        project / "scripts" / "baseline" / "launch_petct_m0_full_training.sh"
    ).read_text(encoding="utf-8")

    assert "flock" in prepare
    assert "--no_pp" in prepare
    assert "--clean" in prepare
    assert "nnUNetv2_preprocess" not in prepare
    assert 'touch "${EXP_ROOT}/manifests/PREPROCESS_READY.done"' not in prepare
    assert "PLANNING_READY.json" in prepare
    assert "PREPROCESS_READY.done" not in smoke
    assert "nnUNetv2_train" not in smoke
    assert "training is disabled" not in smoke
    assert "PREPROCESS_READY.json" in smoke
    assert "SMOKE_READY.json" in smoke
    assert "run_petct_m0_one_epoch.py" in smoke
    assert "nnUNetTrainer_1epoch" in smoke
    assert "validate-smoke" in smoke
    assert "publish-smoke-ready" in smoke
    assert "nnUNetv2_train" not in fold
    assert "training is disabled" not in fold
    assert "PREPROCESS_READY.json" in fold
    assert "SMOKE_READY.json" in fold
    assert "INFERENCE_SMOKE_READY.json" in fold
    assert "FULL_TRAIN_READY.json" in fold
    assert "flock" in fold
    assert "fold-action" in fold
    assert "run_petct_m0_full_fold.py" in fold
    assert "validate-fold" in fold
    assert "FULL_TRAIN_READY.json" in launcher
    assert "INFERENCE_SMOKE_READY.json" in launcher
    assert "publish-full-ready" in launcher
    assert '"${FOLD_RUNNER}" "${CAMPAIGN_ID}" 0 "${GPU0_ID}"' in launcher
    assert 'run_gpu_sequence "${GPU0_ID}" 2 4 &' in launcher
    assert 'run_gpu_sequence "${GPU1_ID}" 1 3 &' in launcher
    assert prepare.index(" preflight ") < prepare.index(" capture-runtime ")
    assert prepare.index(" capture-runtime ") < prepare.index(
        " -m nnunetv2.experiment_planning.plan_and_preprocess_entrypoints"
    )
    assert "nnUNetv2_plan_and_preprocess" not in prepare
    assert "commit-run" in prepare


def test_official_autopet_v_dataset_binds_channel_roles() -> None:
    dataset = {
        "channel_names": {"0": "CT", "1": "PET", "2": "FG", "3": "BG"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": 1611,
        "file_ending": ".nii.gz",
        "name": "AutoPETV",
    }

    contract = validate_autopetv_reference_dataset(dataset)

    assert contract["channel_names"] == {
        "0": "CT",
        "1": "PET",
        "2": "FG",
        "3": "BG",
    }
    changed = copy.deepcopy(dataset)
    changed["channel_names"]["1"] = "CT"
    with pytest.raises(ContractError, match="channel-role"):
        validate_autopetv_reference_dataset(changed)
