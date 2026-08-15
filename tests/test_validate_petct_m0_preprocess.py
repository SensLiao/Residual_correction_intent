from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "baseline"))

from validate_petct_m0_preprocess import (  # noqa: E402
    ContractError,
    build_preprocessing_bundle,
    publish_preprocess_ready,
    stage_preprocessing_run,
    validate_planning_ready,
    validate_preprocessed_inventory,
)
from validate_petct_m0_smoke import validate_preprocess_ready  # noqa: E402


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


def _planning_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    run = tmp_path / "planning_runs" / "plan-1"
    raw = run / "nnUNet_raw" / "Dataset901_PSMA_M0_AutoPETVNorm"
    pre = run / "nnUNet_preprocessed" / "Dataset901_PSMA_M0_AutoPETVNorm"
    results = run / "nnUNet_results"
    source_images = raw / "imagesTr"
    source_labels = raw / "labelsTr"
    for directory in (raw, pre, results, source_images, source_labels):
        directory.mkdir(parents=True, exist_ok=True)

    dataset = {
        "channel_names": {"0": "CT", "1": "PET"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": 597,
        "file_ending": ".nii.gz",
        "name": "PSMA_M0_AutoPETVNorm",
    }
    fingerprint = {
        "spacings": [[2.0, 2.0, 2.0]] * 597,
        "shapes_after_crop": [[10, 10, 10]] * 597,
        "foreground_intensity_properties_per_channel": {
            "0": {"mean": 1.0},
            "1": {"mean": 2.0},
        },
        "median_relative_size_after_cropping": 1.0,
    }
    plans = {
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
    splits = [{"train": ["case_000"], "val": ["case_001"]}]
    for root in (raw, pre):
        _write_json(root / "dataset.json", dataset)
    _write_json(pre / "dataset_fingerprint.json", fingerprint)
    _write_json(pre / "nnUNetPlans.json", plans)
    _write_json(pre / "splits_final.json", splits)

    artifacts = {
        "derived_dataset_json": _record(raw / "dataset.json", relative_to=run),
        "preprocessed_dataset_json": _record(pre / "dataset.json", relative_to=run),
        "dataset_fingerprint": _record(
            pre / "dataset_fingerprint.json", relative_to=run
        ),
        "nnunet_plans": _record(pre / "nnUNetPlans.json", relative_to=run),
        "splits_final": _record(pre / "splits_final.json", relative_to=run),
    }
    bundle = {
        "status": "VALIDATED",
        "planning_status": "PASS",
        "contract_version": "2.0.0",
        "phase": "PLANNING_ONLY",
        "preprocessing_status": "NOT_STARTED",
        "preprocessing_performed": False,
        "training_status": "NOT_STARTED",
        "training_performed": False,
        "run_id": "plan-1",
        "committed_run_dir": str(run.resolve()),
        "dataset": {
            "id": 901,
            "folder": "Dataset901_PSMA_M0_AutoPETVNorm",
            "metadata_name": "PSMA_M0_AutoPETVNorm",
            "source_release": "PSMA-PET-CT-Lesions_v3",
            "scope": "PSMA v3 only",
        },
        "dataset_contract": {
            "source_channel_names": {"0": "CT", "1": "CT"},
            "derived_channel_names": {"0": "CT", "1": "PET"},
            "expected_3d_fullres_normalization": [
                "CTNormalization",
                "ZScoreNormalization",
            ],
        },
        "fingerprint_contract": {"case_count": 597, "channel_keys": ["0", "1"]},
        "plan_contract": {
            "dataset_name": "Dataset901_PSMA_M0_AutoPETVNorm",
            "plans_name": "nnUNetPlans",
            "channel_keys": ["0", "1"],
            "data_identifier": "nnUNetPlans_3d_fullres",
            "experiment_planner_used": "ExperimentPlanner",
            "preprocessor_name": "DefaultPreprocessor",
            "image_reader_writer": "SimpleITKIO",
            "configuration": "3d_fullres",
            "normalization_schemes": [
                "CTNormalization",
                "ZScoreNormalization",
            ],
            "use_mask_for_norm": [False, False],
        },
        "artifacts": artifacts,
        "evidence": {},
    }
    planning_bundle = run / "PLANNING_BUNDLE.json"
    _write_json(planning_bundle, bundle)
    ready = tmp_path / "manifests" / "PLANNING_READY.json"
    published = {
        "status": "COMMITTED",
        "planning_status": "PASS",
        "contract_version": "2.0.0",
        "phase": "PLANNING_ONLY",
        "preprocessing_status": "NOT_STARTED",
        "preprocessing_performed": False,
        "training_status": "NOT_STARTED",
        "training_performed": False,
        "run_id": "plan-1",
        "run_dir": str(run.resolve()),
        "run_receipt": _record(planning_bundle),
        "validated_bundle": bundle,
    }
    _write_json(ready, published)
    return ready, run, published


def _valid_inventory() -> tuple[list[str], list[str]]:
    identifiers = [f"psma_{index:04d}" for index in range(597)]
    config_names = []
    for identifier in identifiers:
        config_names.extend(
            [f"{identifier}.b2nd", f"{identifier}_seg.b2nd", f"{identifier}.pkl"]
        )
    gt_names = [f"{identifier}.nii.gz" for identifier in identifiers]
    return config_names, gt_names


def _test_planning_validator(path: Path) -> dict:
    return validate_planning_ready(path, source_link_policy="TEST_DIRECTORY_FIXTURE")


def test_preprocess_ready_proves_raw_symlinks_target_planning_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "preprocess-runs" / "pre-1"
    raw_dataset = run / "nnUNet_raw" / "Dataset901_PSMA_M0_AutoPETVNorm"
    (run / "nnUNet_preprocessed").mkdir(parents=True)
    raw_dataset.mkdir(parents=True)
    source_root = tmp_path / "planning-source"
    sources = {name: source_root / name for name in ("imagesTr", "labelsTr")}
    for source in sources.values():
        source.mkdir(parents=True)
    simulated_links: dict[Path, Path] | None = None
    try:
        for name, source in sources.items():
            (raw_dataset / name).symlink_to(source, target_is_directory=True)
    except OSError:  # Windows hosts may deny symlink creation without Developer Mode.
        simulated_links = {
            raw_dataset / name: source.resolve() for name, source in sources.items()
        }
        for link in simulated_links:
            if link.is_symlink():
                link.unlink()
            link.mkdir()
        original_is_symlink = Path.is_symlink
        original_resolve = Path.resolve
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda self: self in simulated_links or original_is_symlink(self),
        )

        def simulated_resolve(self: Path, *args, **kwargs) -> Path:
            if self in simulated_links:
                return simulated_links[self]
            return original_resolve(self, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", simulated_resolve)

    planning_ready = tmp_path / "PLANNING_READY.json"
    planning_ready.write_text('{"status":"fixture"}\n', encoding="utf-8")
    output_contract = {"case_count": 597, "status": "PASS"}
    bundle = {
        "status": "VALIDATED",
        "preprocessing_status": "PASS",
        "contract_version": "1.0.0",
        "phase": "PREPROCESSING_ONLY",
        "run_id": run.name,
        "committed_run_dir": str(run.resolve()),
        "training_status": "NOT_STARTED",
        "training_performed": False,
        "checkpoint_count": 0,
        "oof_prediction_count": 0,
        "result_count": 0,
        "planning_ready": _record(planning_ready),
        "planning_bound_hashes": {"splits_final": "a" * 64},
        "output_contract": output_contract,
    }
    bundle_path = run / "PREPROCESSING_BUNDLE.json"
    _write_json(bundle_path, bundle)
    ready_path = tmp_path / "PREPROCESS_READY.json"
    ready = {
        "status": "COMMITTED",
        "preprocessing_status": "PASS",
        "contract_version": "1.0.0",
        "phase": "PREPROCESSING_ONLY",
        "training_status": "NOT_STARTED",
        "training_performed": False,
        "checkpoint_count": 0,
        "oof_prediction_count": 0,
        "result_count": 0,
        "run_id": run.name,
        "run_dir": str(run.resolve()),
        "run_receipt": _record(bundle_path),
        "validated_bundle": bundle,
    }
    _write_json(ready_path, ready)

    planning = {
        "raw_source_paths": {name: str(path.resolve()) for name, path in sources.items()},
        "bound_hashes": {"splits_final": "a" * 64},
    }
    validated = validate_preprocess_ready(
        ready_path,
        output_validator=lambda _run: output_contract,
        planning_validator=lambda _path: planning,
    )
    assert {
        binding["policy"] for binding in validated["raw_source_bindings"].values()
    } == {"PLANNING_RECEIPT_BOUND_DIRECTORY_SYMLINK"}
    assert Path(validated["raw_source_bindings"]["imagesTr"]["target_path"]) == sources[
        "imagesTr"
    ].resolve()

    wrong = tmp_path / "wrong-images"
    wrong.mkdir()
    if simulated_links is not None:
        simulated_links[raw_dataset / "imagesTr"] = wrong.resolve()
    else:
        (raw_dataset / "imagesTr").unlink()
        (raw_dataset / "imagesTr").symlink_to(wrong, target_is_directory=True)
    with pytest.raises(ContractError, match="planning raw_source_paths"):
        validate_preprocess_ready(
            ready_path,
            output_validator=lambda _run: output_contract,
            planning_validator=lambda _path: planning,
        )


def test_planning_gate_binds_committed_pass_and_frozen_petct_contract(
    tmp_path: Path,
) -> None:
    ready, run, _ = _planning_fixture(tmp_path)

    validated = _test_planning_validator(ready)

    assert validated["planning_run_dir"] == str(run.resolve())
    assert validated["dataset"] == {
        "id": 901,
        "folder": "Dataset901_PSMA_M0_AutoPETVNorm",
        "case_count": 597,
        "channel_names": {"0": "CT", "1": "PET"},
    }
    assert validated["preprocess_api"] == {
        "dataset_ids": [901],
        "plans_identifier": "nnUNetPlans",
        "configurations": ["3d_fullres"],
        "num_processes": [4],
    }
    assert validated["plan_contract"]["normalization_schemes"] == [
        "CTNormalization",
        "ZScoreNormalization",
    ]
    assert validated["plan_contract"]["use_mask_for_norm"] == [False, False]
    assert set(validated["bound_hashes"]) == {
        "planning_ready",
        "planning_bundle",
        "dataset_json",
        "dataset_fingerprint",
        "nnunet_plans",
        "splits_final",
    }


@pytest.mark.parametrize(
    "target,key,value,error",
    [
        ("ready", "status", "VALIDATED", "COMMITTED"),
        ("ready", "planning_status", "FAIL", "planning_status"),
        ("ready", "phase", "PREPROCESSING_ONLY", "PLANNING_ONLY"),
        ("bundle", "phase", "PREPROCESSING_ONLY", "PLANNING_ONLY"),
        ("bundle", "preprocessing_status", "PASS", "NOT_STARTED"),
    ],
)
def test_planning_gate_rejects_state_drift(
    tmp_path: Path, target: str, key: str, value: str, error: str
) -> None:
    ready, _, published = _planning_fixture(tmp_path)
    payload = json.loads(ready.read_text(encoding="utf-8"))
    if target == "ready":
        payload[key] = value
    else:
        payload["validated_bundle"][key] = value
        bundle_path = Path(payload["run_receipt"]["path"])
        _write_json(bundle_path, payload["validated_bundle"])
        payload["run_receipt"] = _record(bundle_path)
    _write_json(ready, payload)

    with pytest.raises(ContractError, match=error):
        _test_planning_validator(ready)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda bundle: bundle["dataset"].update(id=902),
            "dataset",
        ),
        (
            lambda bundle: bundle["fingerprint_contract"].update(case_count=596),
            "597",
        ),
        (
            lambda bundle: bundle["dataset_contract"]["derived_channel_names"].update(
                {"1": "CT"}
            ),
            "CT/PET",
        ),
        (
            lambda bundle: bundle["plan_contract"].update(
                normalization_schemes=["CTNormalization", "CTNormalization"]
            ),
            "normalization",
        ),
        (
            lambda bundle: bundle["plan_contract"].update(
                use_mask_for_norm=[False, True]
            ),
            "mask",
        ),
    ],
)
def test_planning_gate_rejects_dataset_or_plan_drift(
    tmp_path: Path, mutation, error: str
) -> None:
    ready, _, _ = _planning_fixture(tmp_path)
    payload = json.loads(ready.read_text(encoding="utf-8"))
    mutation(payload["validated_bundle"])
    bundle_path = Path(payload["run_receipt"]["path"])
    _write_json(bundle_path, payload["validated_bundle"])
    payload["run_receipt"] = _record(bundle_path)
    _write_json(ready, payload)

    with pytest.raises(ContractError, match=error):
        _test_planning_validator(ready)


def test_planning_gate_rejects_tampered_plans_hash(tmp_path: Path) -> None:
    ready, run, _ = _planning_fixture(tmp_path)
    plans = (
        run
        / "nnUNet_preprocessed"
        / "Dataset901_PSMA_M0_AutoPETVNorm"
        / "nnUNetPlans.json"
    )
    plans.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ContractError, match="hash"):
        _test_planning_validator(ready)


def test_staging_is_run_scoped_and_refuses_nonempty_or_existing_destination(
    tmp_path: Path,
) -> None:
    ready, _, _ = _planning_fixture(tmp_path)
    runs = tmp_path / "preprocess_runs"
    runs.mkdir()
    staging = runs / ".partial-pre-1"
    final = runs / "pre-1"
    staging.mkdir()

    staged = stage_preprocessing_run(
        ready,
        staging,
        final,
        "pre-1",
        planning_validator=_test_planning_validator,
        raw_source_stager=lambda source, destination: destination.mkdir(),
    )

    assert staged["status"] == "STAGED"
    assert staged["run_id"] == "pre-1"
    assert not final.exists()
    assert (
        staging
        / "nnUNet_preprocessed"
        / "Dataset901_PSMA_M0_AutoPETVNorm"
        / "nnUNetPlans.json"
    ).is_file()
    assert (
        staging / "nnUNet_raw" / "Dataset901_PSMA_M0_AutoPETVNorm" / "imagesTr"
    ).is_dir()

    second = runs / ".partial-pre-2"
    second.mkdir()
    (second / "foreign.txt").write_text("owned", encoding="utf-8")
    with pytest.raises(ContractError, match="empty"):
        stage_preprocessing_run(
            ready,
            second,
            runs / "pre-2",
            "pre-2",
            planning_validator=_test_planning_validator,
        )

    third = runs / ".partial-pre-3"
    third.mkdir()
    (runs / "pre-3").mkdir()
    with pytest.raises(FileExistsError, match="destination"):
        stage_preprocessing_run(
            ready,
            third,
            runs / "pre-3",
            "pre-3",
            planning_validator=_test_planning_validator,
        )


def test_inventory_requires_exact_597_triplets_gt_and_injected_load_contract() -> None:
    config_names, gt_names = _valid_inventory()
    calls: list[str] = []

    def injected_load(identifier: str) -> dict:
        calls.append(identifier)
        return {
            "data_shape": [2, 8, 8, 8],
            "seg_shape": [1, 8, 8, 8],
            "properties_type": "dict",
        }

    receipt = validate_preprocessed_inventory(
        config_names,
        gt_names,
        load_case_hook=injected_load,
        hook_kind="TEST_INJECTED",
    )

    assert receipt["case_count"] == 597
    assert receipt["artifact_counts"] == {
        ".b2nd": 597,
        "_seg.b2nd": 597,
        ".pkl": 597,
        "gt_segmentations": 597,
    }
    assert receipt["one_case_load"]["status"] == "PASS"
    assert receipt["one_case_load"]["hook_kind"] == "TEST_INJECTED"
    assert receipt["one_case_load"]["official_nnunet_load_claimed"] is False
    assert calls == ["psma_0000"]


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda config, gt: config.pop(), "triplet|597"),
        (lambda config, gt: gt.pop(), "ground-truth|597"),
        (lambda config, gt: config.append("foreign.npy"), "unexpected"),
    ],
)
def test_inventory_rejects_missing_or_foreign_outputs(mutation, error: str) -> None:
    config_names, gt_names = _valid_inventory()
    mutation(config_names, gt_names)

    with pytest.raises(ContractError, match=error):
        validate_preprocessed_inventory(
            config_names,
            gt_names,
            load_case_hook=lambda identifier: {
                "data_shape": [2, 8, 8, 8],
                "seg_shape": [1, 8, 8, 8],
                "properties_type": "dict",
            },
            hook_kind="TEST_INJECTED",
        )


def test_bundle_and_fixed_receipt_keep_training_and_results_not_started(
    tmp_path: Path,
) -> None:
    ready, _, _ = _planning_fixture(tmp_path)
    run = tmp_path / "preprocess_runs" / "pre-1"
    run.mkdir(parents=True)
    bundle_path = run / "PREPROCESSING_BUNDLE.json"
    inventory = {
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
    bundle = build_preprocessing_bundle(
        ready,
        run_id="pre-1",
        committed_run_dir=run,
        inventory=inventory,
        planning_validator=_test_planning_validator,
    )
    _write_json(bundle_path, bundle)
    fixed = tmp_path / "manifests" / "PREPROCESS_READY.json"

    published = publish_preprocess_ready(
        run,
        bundle_path,
        fixed,
        output_validator=lambda _: inventory,
        planning_validator=_test_planning_validator,
    )

    assert published["status"] == "COMMITTED"
    assert published["preprocessing_status"] == "PASS"
    assert published["training_status"] == "NOT_STARTED"
    assert published["checkpoint_count"] == 0
    assert published["oof_prediction_count"] == 0
    assert published["result_count"] == 0
    before = fixed.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_preprocess_ready(
            run,
            bundle_path,
            fixed,
            output_validator=lambda _: inventory,
            planning_validator=_test_planning_validator,
        )
    assert fixed.read_bytes() == before


def test_shell_wrapper_calls_only_frozen_official_preprocess_api() -> None:
    project = Path(__file__).resolve().parents[1]
    shell = (project / "scripts" / "baseline" / "run_petct_m0_preprocess.sh").read_text(
        encoding="utf-8"
    )

    assert "flock" in shell
    assert "PLANNING_READY.json" in shell
    assert "PREPROCESSING_BUNDLE.json" in shell
    assert "PREPROCESS_READY.json" in shell
    assert "mktemp -d" in shell
    assert "commit-run" in shell
    assert "publish-preprocess-ready" in shell
    assert "nnunetv2.experiment_planning.plan_and_preprocess_api" in shell
    assert "preprocess([901], 'nnUNetPlans', ['3d_fullres'], [4]" in shell
    assert "nnUNetv2_plan_and_preprocess" not in shell
    assert "nnUNetv2_train" not in shell
    assert "oof" not in shell.lower()
