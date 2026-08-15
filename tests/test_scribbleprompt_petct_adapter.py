from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts/comparators"))

import scribbleprompt_petct_adapter as adapter  # noqa: E402


def _save(path: Path, data: np.ndarray) -> None:
    affine = np.diag([2.0, 2.0, 3.0, 1.0])
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _fixture(
    tmp_path: Path, *, split: str = "validation", bg_nonempty: bool = False
) -> Path:
    shape = (12, 10, 4)
    pet = np.linspace(0.0, 20.0, np.prod(shape), dtype=np.float32).reshape(shape)
    ct = np.zeros(shape, dtype=np.float32)
    m0 = np.zeros(shape, dtype=np.uint8)
    m0[1:4, 1:4, :] = 1
    fg = np.zeros(shape, dtype=np.uint8)
    fg[7:9, 6:8, 2] = 1
    bg = np.zeros(shape, dtype=np.uint8)
    if bg_nonempty:
        bg[0, 0, 2] = 1
    for name, value in (("pet", pet), ("grid", ct), ("m0", m0), ("fg", fg), ("bg", bg)):
        _save(tmp_path / (name + ".nii.gz"), value)
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
                        "case_ids": ["case/001"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "schema_version": adapter.INPUT_SCHEMA,
        "records": [
            {
                "case_id": "case/001",
                "patient_id": "patient-001",
                "split": split,
                "fold": 0,
                "step": 1,
                "pet_path": "pet.nii.gz",
                # ScribblePrompt is PET-only; this deliberately need not exist.
                "ct_path": "not-consumed-ct.nii.gz",
                "m0_path": "m0.nii.gz",
                "fg_scribble_path": "fg.nii.gz",
                "bg_scribble_path": "bg.nii.gz",
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
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest


class _FakeModel:
    def __init__(self) -> None:
        self.call = None
        self.call_count = 0

    def predict(self, **kwargs: object) -> torch.Tensor:
        self.call_count += 1
        self.call = kwargs
        image = kwargs["img"]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (1, 1, 128, 128)
        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 1.0
        assert kwargs["point_coords"] is None
        assert kwargs["point_labels"] is None
        assert kwargs["box"] is None
        scribble = kwargs["scribbles"]
        previous = kwargs["mask_input"]
        assert isinstance(scribble, torch.Tensor)
        assert isinstance(previous, torch.Tensor)
        assert scribble.shape == (1, 2, 128, 128)
        assert torch.count_nonzero(scribble[:, 1]) == 0
        assert previous.shape == (1, 1, 128, 128)
        result = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
        result[:, :, 70:100, 70:100] = 1.0
        return result


def _patch_provenance(monkeypatch: pytest.MonkeyPatch, fake: _FakeModel) -> None:
    monkeypatch.setattr(
        adapter,
        "verify_checkpoint",
        lambda path: {"path": str(path), "bytes": 3, "sha256": "a" * 64},
    )
    monkeypatch.setattr(
        adapter,
        "verify_source",
        lambda path: {
            "path": str(path),
            "pinned_commit": adapter.SOURCE_COMMIT,
            "required_file_sha256": {},
            "required_files_bundle_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(adapter, "load_official_model", lambda *args: fake)


@pytest.mark.parametrize("policy", adapter.OUTPUT_POLICIES)
def test_cpu_adapter_restores_grid_and_declares_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    manifest = _fixture(tmp_path)
    fake = _FakeModel()
    _patch_provenance(monkeypatch, fake)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"abc")
    source = tmp_path / "source"
    source.mkdir()
    output_manifest = tmp_path / (policy + ".json")
    output_dir = tmp_path / (policy + "-predictions")

    rc = adapter.run(
        input_manifest=manifest,
        output_manifest=output_manifest,
        output_dir=output_dir,
        checkpoint=checkpoint,
        source_root=source,
        output_policy=policy,
        device_request="cpu",
        partition="val",
        learning_split=tmp_path / "learning-split.json",
        experiment_config=PROJECT / "configs" / "petct_route_a_experiment.json",
        test_access_receipt=None,
        run_root=None,
    )
    assert rc == 0
    payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == adapter.OUTPUT_SCHEMA
    assert payload["output_policy"] == policy
    assert payload["output_policy_was_explicit"] is True
    assert payload["prompt_contract"]["oracle_slice_selection"] is False
    row = payload["records"][0]
    assert row["prompted_axial_slice_index"] == 2
    assert row["prompt_budget"]["foreground_scribbles"] == 1
    assert row["prompt_budget"]["background_scribbles"] == 0
    assert row["input_sha256"]["ct_path"] is None
    prediction = nib.load(row["prediction_path"])
    assert prediction.shape == (12, 10, 4)
    data = np.asarray(prediction.dataobj)
    m0 = np.asarray(nib.load(str(tmp_path / "m0.nii.gz")).dataobj)
    assert np.array_equal(data[:, :, 0], m0[:, :, 0])
    if policy == "union_with_m0":
        assert np.all(data[m0 > 0] == 1)
        assert row["m0_preservation_guaranteed"] is True
    else:
        assert not np.array_equal(data[:, :, 2], m0[:, :, 2])
        assert row["m0_preservation_guaranteed"] is False


def test_paired_policies_reuse_one_native_model_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _fixture(tmp_path)
    fake = _FakeModel()
    _patch_provenance(monkeypatch, fake)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"abc")
    source = tmp_path / "source"
    source.mkdir()
    native_manifest = tmp_path / "native.json"
    union_manifest = tmp_path / "union.json"

    rc = adapter.run(
        input_manifest=manifest,
        output_manifest=native_manifest,
        output_dir=tmp_path / "native-predictions",
        checkpoint=checkpoint,
        source_root=source,
        output_policy="native_slice_replace",
        device_request="cpu",
        partition="val",
        learning_split=tmp_path / "learning-split.json",
        experiment_config=PROJECT / "configs" / "petct_route_a_experiment.json",
        test_access_receipt=None,
        run_root=None,
        derived_union_output_manifest=union_manifest,
        derived_union_output_dir=tmp_path / "union-predictions",
    )

    assert rc == 0
    assert fake.call_count == 1
    native = json.loads(native_manifest.read_text(encoding="utf-8"))
    union = json.loads(union_manifest.read_text(encoding="utf-8"))
    assert union["output_policy"] == "union_with_m0"
    assert union["derivation"]["model_inference_calls"] == 0
    native_row = native["records"][0]
    union_row = union["records"][0]
    assert union_row["derived_from_prediction_sha256"] == native_row["prediction_sha256"]
    native_mask = np.asarray(nib.load(native_row["prediction_path"]).dataobj) > 0
    union_mask = np.asarray(nib.load(union_row["prediction_path"]).dataobj) > 0
    m0 = np.asarray(nib.load(tmp_path / "m0.nii.gz").dataobj) > 0
    assert np.array_equal(union_mask, native_mask | m0)


def test_test_split_is_fail_closed_before_model_or_output(tmp_path: Path) -> None:
    manifest = _fixture(tmp_path, split="test")
    output_manifest = tmp_path / "out.json"
    with pytest.raises(adapter.AdapterError, match="consumed test-access receipt"):
        adapter.run(
            input_manifest=manifest,
            output_manifest=output_manifest,
            output_dir=tmp_path / "predictions",
            checkpoint=tmp_path / "missing.pt",
            source_root=tmp_path / "missing-source",
            output_policy="native_slice_replace",
            device_request="cpu",
            partition="test",
            learning_split=tmp_path / "learning-split.json",
            experiment_config=PROJECT / "configs" / "petct_route_a_experiment.json",
            test_access_receipt=None,
            run_root=tmp_path,
        )
    assert not output_manifest.exists()
    assert not (tmp_path / "predictions").exists()


def test_test_case_relabelled_as_validation_is_rejected_before_image_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _fixture(tmp_path, split="test")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["records"][0]["split"] = "validation"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        adapter,
        "_resolve_input_path",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("image was opened before frozen split validation")
        ),
    )

    with pytest.raises(adapter.AdapterError, match="partition differs from frozen split"):
        adapter.run(
            input_manifest=manifest,
            output_manifest=tmp_path / "out.json",
            output_dir=tmp_path / "predictions",
            checkpoint=tmp_path / "missing.pt",
            source_root=tmp_path / "missing-source",
            output_policy="native_slice_replace",
            device_request="cpu",
            partition="val",
            learning_split=tmp_path / "learning-split.json",
            experiment_config=PROJECT / "configs" / "petct_route_a_experiment.json",
            test_access_receipt=None,
            run_root=None,
        )


def test_validation_rejects_test_receipt_before_input_manifest_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _fixture(tmp_path)
    monkeypatch.setattr(
        adapter,
        "_read_json",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("manifest was opened before val receipt rejection")
        ),
    )
    with pytest.raises(adapter.AdapterError, match="rejects a test receipt"):
        adapter.run(
            input_manifest=manifest,
            output_manifest=tmp_path / "out.json",
            output_dir=tmp_path / "predictions",
            checkpoint=tmp_path / "missing.pt",
            source_root=tmp_path / "missing-source",
            output_policy="native_slice_replace",
            device_request="cpu",
            partition="val",
            learning_split=tmp_path / "learning-split.json",
            experiment_config=PROJECT / "configs" / "petct_route_a_experiment.json",
            test_access_receipt=tmp_path / "not-read.json",
            run_root=None,
        )


def test_nonempty_negative_prompt_becomes_explicit_failure_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _fixture(tmp_path, bg_nonempty=True)
    fake = _FakeModel()
    _patch_provenance(monkeypatch, fake)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"abc")
    source = tmp_path / "source"
    source.mkdir()
    output_manifest = tmp_path / "out.json"
    rc = adapter.run(
        input_manifest=manifest,
        output_manifest=output_manifest,
        output_dir=tmp_path / "predictions",
        checkpoint=checkpoint,
        source_root=source,
        output_policy="native_slice_replace",
        device_request="cpu",
        partition="val",
        learning_split=tmp_path / "learning-split.json",
        experiment_config=PROJECT / "configs" / "petct_route_a_experiment.json",
        test_access_receipt=None,
        run_root=None,
    )
    assert rc == 1
    payload = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert payload["failed_count"] == 1
    assert payload["records"][0]["status"] == "failed"
    assert payload["records"][0]["input_sha256"]["pet_path"]
    assert (
        "negative/background prompt must be empty"
        in payload["records"][0]["error"]["message"]
    )


def test_no_clobber_and_multi_slice_scribble_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _fixture(tmp_path)
    fg_path = tmp_path / "fg.nii.gz"
    fg = np.asarray(nib.load(str(fg_path)).dataobj)
    fg[1, 1, 1] = 1
    _save(fg_path, fg)
    fake = _FakeModel()
    _patch_provenance(monkeypatch, fake)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"abc")
    source = tmp_path / "source"
    source.mkdir()
    output_manifest = tmp_path / "out.json"
    rc = adapter.run(
        input_manifest=manifest,
        output_manifest=output_manifest,
        output_dir=tmp_path / "predictions",
        checkpoint=checkpoint,
        source_root=source,
        output_policy="native_slice_replace",
        device_request="cpu",
        partition="val",
        learning_split=tmp_path / "learning-split.json",
        experiment_config=PROJECT / "configs" / "petct_route_a_experiment.json",
        test_access_receipt=None,
        run_root=None,
    )
    assert rc == 1
    with pytest.raises(adapter.AdapterError, match="refusing existing output manifest"):
        adapter.run(
            input_manifest=manifest,
            output_manifest=output_manifest,
            output_dir=tmp_path / "other-predictions",
            checkpoint=checkpoint,
            source_root=source,
            output_policy="native_slice_replace",
            device_request="cpu",
            partition="val",
            learning_split=tmp_path / "learning-split.json",
            experiment_config=PROJECT / "configs" / "petct_route_a_experiment.json",
            test_access_receipt=None,
            run_root=None,
        )


def test_real_checkpoint_and_pinned_source_load_on_cpu_when_present() -> None:
    checkpoint = (
        PROJECT / "models/ScribblePrompt/ScribblePrompt_unet_v1_nf192_res128.pt"
    )
    source = PROJECT / "upstream/ScribblePrompt"
    if not checkpoint.is_file() or not source.is_dir():
        pytest.skip(
            "official ScribblePrompt source/checkpoint is not available locally"
        )
    adapter.verify_checkpoint(checkpoint)
    adapter.verify_source(source)
    model = adapter.load_official_model(source, checkpoint, "cpu")
    assert model.input_size == (128, 128)
