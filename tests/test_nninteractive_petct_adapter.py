from __future__ import annotations

import json
import ast
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts" / "comparators"))

import nninteractive_petct_adapter as adapter  # noqa: E402


MODEL_FOLDER = PROJECT / "models" / "nnInteractive" / "nnInteractive_v1.0"
CONFIG = PROJECT / "configs" / "petct_external_comparators.json"
MODEL_INFO = {
    "model_folder": str(MODEL_FOLDER),
    "checkpoint_path": str(MODEL_FOLDER / adapter.CHECKPOINT_RELATIVE_PATH),
    "checkpoint_sha256": adapter.CHECKPOINT_SHA256,
    "checkpoint_bytes": 411387150,
    "license_path": str(MODEL_FOLDER / adapter.LICENSE_RELATIVE_PATH),
    "license_sha256": adapter.LICENSE_SHA256,
    "license_id": adapter.LICENSE_ID,
    "source_commit": adapter.SOURCE_COMMIT,
}


class FakeSession:
    supports_initial_label = True
    supported_interactions = {"scribble": True}
    license = adapter.LICENSE_ID

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.target: np.ndarray | None = None

    def set_image(self, image: np.ndarray) -> None:
        self.calls.append(("set_image", image.shape, image.dtype))

    def set_target_buffer(self, target_buffer: np.ndarray) -> None:
        self.calls.append(("set_target_buffer", target_buffer.shape, target_buffer.dtype))
        self.target = target_buffer

    def add_initial_seg_interaction(
        self,
        initial_seg: np.ndarray,
        run_prediction: bool = False,
        override_capability_checks: bool = False,
    ) -> None:
        self.calls.append(
            ("add_initial_seg_interaction", initial_seg.copy(), run_prediction, override_capability_checks)
        )
        assert self.target is not None
        self.target[:] = initial_seg

    def add_scribble_interaction(
        self,
        scribble_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
        interaction_bbox: list[list[int]] | None = None,
    ) -> None:
        self.calls.append(
            (
                "add_scribble_interaction",
                scribble_image.copy(),
                include_interaction,
                run_prediction,
                override_capability_checks,
                interaction_bbox,
            )
        )
        assert self.target is not None
        # A native output that intentionally drops M0 makes the two output policies distinguishable.
        self.target[:] = scribble_image


def _write_nifti(path: Path, values: np.ndarray, affine: np.ndarray | None = None) -> None:
    nib.save(nib.Nifti1Image(values, np.eye(4) if affine is None else affine), str(path))


def _case_fixture(tmp_path: Path, *, split: str = "validation") -> tuple[Path, dict[str, np.ndarray]]:
    shape = (7, 6, 5)
    pet = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    m0 = np.zeros(shape, dtype=np.uint8)
    m0[1, 1, 1] = 1
    scribble = np.zeros(shape, dtype=np.uint8)
    scribble[4, 3, 2] = 1
    _write_nifti(tmp_path / "pet.nii.gz", pet)
    _write_nifti(tmp_path / "m0.nii.gz", m0)
    _write_nifti(tmp_path / "scribble.nii.gz", scribble)
    _write_nifti(tmp_path / "grid.nii.gz", np.zeros(shape, dtype=np.uint8))
    internal_partition = "val" if split == "validation" else split
    learning_split = tmp_path / "learning-split.json"
    learning_split.write_text(
        json.dumps(
            {
                "schema_version": "PETCT-LEARNING-SPLIT-v1.0",
                "status": "FROZEN_BEFORE_MODEL_SELECTION",
                "patient_count": 1,
                "case_count": 1,
                "case_counts": {
                    "train": int(internal_partition == "train"),
                    "val": int(internal_partition == "val"),
                    "test": int(internal_partition == "test"),
                },
                "patients": [
                    {
                        "patient_id": "patient-001",
                        "partition": internal_partition,
                        "case_ids": ["case-001"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": adapter.INPUT_SCHEMA,
        "records": [
            {
                "case_id": "case-001",
                "patient_id": "patient-001",
                "split": split,
                "fold": 0,
                "step": 1,
                "pet_path": "pet.nii.gz",
                # This file deliberately does not exist: CT must never be consumed by this PET-only adapter.
                "ct_path": "not-consumed-ct.nii.gz",
                "m0_path": "m0.nii.gz",
                "fg_scribble_path": "scribble.nii.gz",
                "bg_scribble_path": None,
                "original_grid_reference": "grid.nii.gz",
                "scribble_strategy": "centerline",
                "scribble_polarity": "foreground",
                "patient_split_receipt": {
                    "internal_partition": internal_partition,
                    "learning_split_sha256": adapter.sha256_file(learning_split),
                },
            }
        ],
    }
    manifest_path = tmp_path / "input.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, {"pet": pet, "m0": m0, "scribble": scribble}


def _options(
    tmp_path: Path,
    manifest: Path,
    policy: str,
    *,
    partition: str = "val",
    test_access_receipt: Path | None = None,
) -> adapter.AdapterOptions:
    return adapter.AdapterOptions(
        input_manifest=manifest,
        output_manifest=tmp_path / f"output-{policy}.json",
        output_dir=tmp_path / f"predictions-{policy}",
        model_folder=MODEL_FOLDER,
        config=CONFIG,
        output_policy=policy,
        device="cpu",
        torch_threads=1,
        partition=partition,
        learning_split=tmp_path / "learning-split.json",
        experiment_config=PROJECT / "configs" / "petct_route_a_experiment.json",
        test_access_receipt=test_access_receipt,
        run_root=tmp_path if partition == "test" else None,
    )


def test_real_v1_checkpoint_and_license_match_frozen_hashes() -> None:
    info = adapter.validate_model_folder(MODEL_FOLDER, CONFIG)
    assert info["checkpoint_sha256"] == adapter.CHECKPOINT_SHA256
    assert info["checkpoint_bytes"] == 411387150
    assert info["license_sha256"] == adapter.LICENSE_SHA256
    assert info["license_id"] == "CC BY-NC-SA 4.0"
    assert info["source_commit"] == adapter.SOURCE_COMMIT


def test_pinned_official_source_exposes_the_expected_native_session_api() -> None:
    source_path = (
        PROJECT
        / "upstream"
        / "nnInteractive"
        / "nnInteractive"
        / "inference"
        / "inference_session.py"
    )
    if not source_path.is_file():
        source_path = (
            PROJECT
            / "external_runners"
            / "nninteractive"
            / "source"
            / "nnInteractive"
            / "inference"
            / "inference_session.py"
        )
    assert source_path.is_file()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    session_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "nnInteractiveInferenceSession"
    )
    methods = {
        node.name: [argument.arg for argument in node.args.args]
        for node in session_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"image"} <= set(methods["set_image"])
    assert {"target_buffer"} <= set(methods["set_target_buffer"])
    assert {"initial_seg", "run_prediction"} <= set(methods["add_initial_seg_interaction"])
    assert {"scribble_image", "include_interaction", "run_prediction"} <= set(
        methods["add_scribble_interaction"]
    )


@pytest.mark.parametrize("policy", adapter.OUTPUT_POLICIES)
def test_small_nifti_cpu_contract_uses_only_initial_state_then_one_native_3d_scribble(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    manifest, arrays = _case_fixture(tmp_path)
    monkeypatch.setattr(adapter, "validate_model_folder", lambda *_: MODEL_INFO)
    session = FakeSession()
    output = adapter.run_adapter(_options(tmp_path, manifest, policy), session_factory=lambda *_: session)

    assert output["counts"] == {"total": 1, "complete": 1, "failed": 0}
    record = output["records"][0]
    assert record["status"] == "complete"
    assert record["pretraining_exposure"] == "KNOWN_PUBLIC_COHORT_EXPOSURE"
    assert record["exact_psma_v3_exposure"] == "UNKNOWN"
    assert record["headline_eligible"] is False
    assert record["checkpoint_sha256"] == adapter.CHECKPOINT_SHA256
    assert record["model_license_sha256"] == adapter.LICENSE_SHA256
    assert record["ct_consumed_by_model"] is False
    assert record["pet_only_image_channel"] is True
    assert record["failure_reason"] is None
    assert record["prediction_sha256"] == adapter.sha256_file(Path(record["prediction_path"]))

    call_names = [call[0] for call in session.calls]
    assert call_names == [
        "set_image",
        "set_target_buffer",
        "add_initial_seg_interaction",
        "add_scribble_interaction",
    ]
    assert session.calls[0][1] == (1, *arrays["pet"].shape)
    initial_call = session.calls[2]
    assert np.array_equal(initial_call[1], arrays["m0"])
    assert initial_call[2:] == (False, False)
    scribble_call = session.calls[3]
    assert np.array_equal(scribble_call[1], arrays["scribble"])
    assert scribble_call[2:] == (True, True, False, None)

    prediction_image = nib.load(record["prediction_path"])
    prediction = np.asarray(prediction_image.dataobj, dtype=np.uint8)
    assert prediction.shape == arrays["pet"].shape
    assert np.allclose(prediction_image.affine, np.eye(4))
    assert prediction[4, 3, 2] == 1
    assert prediction[1, 1, 1] == (1 if policy == "union_with_m0" else 0)
    assert record["m0_preservation_guaranteed"] is (policy == "union_with_m0")


def test_paired_policies_reuse_one_native_model_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, arrays = _case_fixture(tmp_path)
    monkeypatch.setattr(adapter, "validate_model_folder", lambda *_: MODEL_INFO)
    session = FakeSession()
    options = _options(tmp_path, manifest, "native_full_mask")
    options = adapter.AdapterOptions(
        **{
            **options.__dict__,
            "derived_union_output_manifest": tmp_path / "output-union_with_m0.json",
            "derived_union_output_dir": tmp_path / "predictions-union_with_m0",
        }
    )

    native = adapter.run_adapter(options, session_factory=lambda *_: session)

    assert [call[0] for call in session.calls].count("add_scribble_interaction") == 1
    union = json.loads(options.derived_union_output_manifest.read_text(encoding="utf-8"))
    assert union["output_policy"] == "union_with_m0"
    assert union["derivation"]["model_inference_calls"] == 0
    native_row = native["records"][0]
    union_row = union["records"][0]
    assert union_row["derived_from_prediction_sha256"] == native_row["prediction_sha256"]
    native_mask = np.asarray(nib.load(native_row["prediction_path"]).dataobj, dtype=np.uint8)
    union_mask = np.asarray(nib.load(union_row["prediction_path"]).dataobj, dtype=np.uint8)
    assert np.array_equal(union_mask, np.logical_or(native_mask, arrays["m0"]))


def test_test_split_is_locked_before_session_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _case_fixture(tmp_path, split="test")
    monkeypatch.setattr(adapter, "validate_model_folder", lambda *_: MODEL_INFO)
    called = False

    def factory(*_: object) -> FakeSession:
        nonlocal called
        called = True
        return FakeSession()

    with pytest.raises(adapter.AdapterError, match="consumed test-access receipt"):
        adapter.run_adapter(
            _options(tmp_path, manifest, "native_full_mask", partition="test"),
            session_factory=factory,
        )
    assert called is False
    assert not (tmp_path / "output-native_full_mask.json").exists()


def test_test_split_rejects_an_invalid_consumed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _case_fixture(tmp_path, split="test")
    monkeypatch.setattr(adapter, "validate_model_folder", lambda *_: MODEL_INFO)
    receipt = tmp_path / "invalid-consumed-receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    options = _options(
        tmp_path,
        manifest,
        "native_full_mask",
        partition="test",
        test_access_receipt=receipt,
    )
    with pytest.raises(adapter.AdapterError, match="self-hash mismatch"):
        adapter.run_adapter(options, session_factory=lambda *_: FakeSession())


def test_test_case_relabelled_as_validation_is_rejected_before_nifti_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _case_fixture(tmp_path, split="test")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][0]["split"] = "validation"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(adapter, "validate_model_folder", lambda *_: MODEL_INFO)
    monkeypatch.setattr(
        adapter,
        "_load_nifti",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("NIfTI was opened before frozen split validation")
        ),
    )

    with pytest.raises(adapter.AdapterError, match="partition differs from frozen split"):
        adapter.run_adapter(
            _options(tmp_path, manifest, "native_full_mask", partition="val"),
            session_factory=lambda *_: FakeSession(),
        )


def test_validation_rejects_test_receipt_before_manifest_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _case_fixture(tmp_path)
    options = _options(tmp_path, manifest, "native_full_mask")
    options = adapter.AdapterOptions(
        **{**options.__dict__, "test_access_receipt": tmp_path / "not-read.json"}
    )
    monkeypatch.setattr(
        adapter,
        "_load_json",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("manifest was opened before val receipt rejection")
        ),
    )

    with pytest.raises(adapter.AdapterError, match="rejects a test receipt"):
        adapter.run_adapter(options, session_factory=lambda *_: FakeSession())


def test_grid_mismatch_is_retained_as_explicit_failed_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, arrays = _case_fixture(tmp_path)
    shifted = np.eye(4)
    shifted[0, 3] = 2.0
    _write_nifti(tmp_path / "m0.nii.gz", arrays["m0"], shifted)
    monkeypatch.setattr(adapter, "validate_model_folder", lambda *_: MODEL_INFO)

    output = adapter.run_adapter(
        _options(tmp_path, manifest, "native_full_mask"), session_factory=lambda *_: FakeSession()
    )
    record = output["records"][0]
    assert output["counts"] == {"total": 1, "complete": 0, "failed": 1}
    assert record["status"] == "failed"
    assert "affine differs" in record["failure_reason"]
    assert record["prediction_sha256"] is None
    assert not Path(record["prediction_path"]).exists()


def test_existing_output_manifest_or_prediction_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _case_fixture(tmp_path)
    monkeypatch.setattr(adapter, "validate_model_folder", lambda *_: MODEL_INFO)
    options = _options(tmp_path, manifest, "native_full_mask")
    options.output_manifest.write_text("keep", encoding="utf-8")
    with pytest.raises(adapter.AdapterError, match="refusing to overwrite output manifest"):
        adapter.run_adapter(options, session_factory=lambda *_: FakeSession())
    assert options.output_manifest.read_text(encoding="utf-8") == "keep"


def test_incompatible_runtime_api_fails_closed() -> None:
    class MissingInitialLabel:
        supports_initial_label = True
        supported_interactions = {"scribble": True}
        license = adapter.LICENSE_ID

        def set_image(self, image: np.ndarray) -> None:
            pass

        def set_target_buffer(self, target_buffer: np.ndarray) -> None:
            pass

        def add_scribble_interaction(
            self, scribble_image: np.ndarray, include_interaction: bool, run_prediction: bool = True
        ) -> None:
            pass

    with pytest.raises(adapter.AdapterError, match="lacks add_initial_seg_interaction"):
        adapter.validate_session_api(MissingInitialLabel())


def test_setup_script_uses_a_separate_prefix_and_checks_native_apis_and_hashes() -> None:
    script = (PROJECT / "scripts" / "setup" / "setup_nninteractive_env.sh").read_text(encoding="utf-8")
    assert 'ENV_PREFIX="${PETCT_ROOT}/envs/nninteractive_v1"' in script
    assert "petct_nnunet_v281" in script  # checked only as a forbidden sys.path source
    assert "pip install" in script
    assert "add_initial_seg_interaction" in script
    assert "add_scribble_interaction" in script
    assert adapter.CHECKPOINT_SHA256 in script
    assert adapter.LICENSE_SHA256 in script
    assert '"schema_version": "PETCT-NNINTERACTIVE-ENV-v1.1"' in script
    assert '"status": "PASS"' in script
    assert "initialize_from_trained_model_folder" in script
    assert 'session.add_initial_seg_interaction(initial, run_prediction=False)' in script
    assert 'session.add_scribble_interaction(scribble, include_interaction=True, run_prediction=False)' in script
    assert '"model_load_smoke": "PASS"' in script
    assert '"source_bundle_sha256"' in script
    assert '"environment_freeze_sha256"' in script
    assert 'os.replace(temporary, target)' in script
    assert script.index('"${PYTHON}" "${ADAPTER_FILE}" --help') < script.index('os.replace(temporary, target)')
    assert "git " not in script


def test_setup_skip_install_mode_is_strictly_offline_and_still_rebinds_receipt() -> None:
    script = (PROJECT / "scripts" / "setup" / "setup_nninteractive_env.sh").read_text(
        encoding="utf-8"
    )
    assert 'SKIP_INSTALL="${PETCT_SKIP_INSTALL:-0}"' in script
    assert 'if [[ "${SKIP_INSTALL}" == "1" ]]; then' in script
    skip_branch_start = script.index('if [[ "${SKIP_INSTALL}" == "1" ]]; then')
    install_branch_start = script.index("\nelse\n", skip_branch_start)
    common_checks_start = script.index('\n"${PYTHON}" -m pip check', install_branch_start)
    skip_branch = script[skip_branch_start:install_branch_start]
    install_branch = script[install_branch_start:common_checks_start]
    assert "PIP_NO_INDEX=1" in skip_branch
    assert "requires an existing valid Conda environment" in skip_branch
    assert "conda-meta/history" in skip_branch
    assert '"${CONDA_EXE}" create' not in skip_branch
    assert "pip install" not in skip_branch
    assert '"${CONDA_EXE}" create' in install_branch
    assert "pip install" in install_branch
    common_tail = script[common_checks_start:]
    assert '"${PYTHON}" -m pip check' in common_tail
    assert '"${PYTHON}" -m pip freeze --all' in common_tail
    assert '"${PYTHON}" "${ADAPTER_FILE}" --help' in common_tail
    assert "initialize_from_trained_model_folder" in common_tail
    assert '"setup_mode": "VERIFY_EXISTING_NO_INSTALL"' in common_tail
    assert 'os.replace(temporary, target)' in common_tail
