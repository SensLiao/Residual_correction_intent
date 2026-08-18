from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "data"))

import build_petct_residual_manifest as residual_builder  # noqa: E402
from common.petct_learning import sha256_file  # noqa: E402


def _write_nifti(path: Path, array: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))


def _main_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    source_partition: str = "train",
    selected_partitions: tuple[str, ...] = ("train",),
):
    shape = (5, 5, 1)
    gt = np.zeros(shape, dtype=np.uint8)
    gt[2, 2, 0] = 1
    m0 = np.zeros(shape, dtype=np.uint8)
    arrays = {
        "ct": np.zeros(shape, dtype=np.float32),
        "pet": np.ones(shape, dtype=np.float32),
        "gt": gt,
        "m0": m0,
    }
    paths = {}
    for name, array in arrays.items():
        path = tmp_path / (name + ".nii.gz")
        _write_nifti(path, array)
        paths[name] = path
    source = {
        "case_id": "case-a",
        "patient_id": "patient-a",
        "held_out_fold": 0,
        "partition": source_partition,
        "ct_path": str(paths["ct"]),
        "pet_path": str(paths["pet"]),
        "gt_path": str(paths["gt"]),
        "ct_bytes": paths["ct"].stat().st_size,
        "pet_bytes": paths["pet"].stat().st_size,
        "gt_bytes": paths["gt"].stat().st_size,
        "ct_sha256": sha256_file(paths["ct"]),
        "pet_sha256": sha256_file(paths["pet"]),
        "gt_sha256": sha256_file(paths["gt"]),
    }
    provenance = {
        "input_ct_sha256": sha256_file(paths["ct"]),
        "input_pet_sha256": sha256_file(paths["pet"]),
        "input_gt_sha256": sha256_file(paths["gt"]),
    }
    validated = {
        "cases": {
            "case-a": {"patient_id": "patient-a", "held_out_fold": 0}
        }
    }
    monkeypatch.setattr(residual_builder, "load_jsonl", lambda path: [source])
    monkeypatch.setattr(
        residual_builder,
        "validate_patient_folds",
        lambda rows: {"case_count": 597, "patient_count": 378},
    )
    oof_ready = tmp_path / "OOF_READY.json"
    oof_ready.write_text('{"status":"fixture"}\n', encoding="utf-8")
    case_manifest = tmp_path / "cases.jsonl"
    case_manifest.write_text(json.dumps(source) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        residual_builder, "validate_oof_ready_receipt_only", lambda path: validated
    )
    monkeypatch.setattr(
        residual_builder, "resolve_oof_mask", lambda accepted, case: paths["m0"]
    )
    monkeypatch.setattr(
        residual_builder,
        "build_natural_oof_binding_from_validated",
        lambda *args, **kwargs: provenance,
    )
    monkeypatch.setattr(
        residual_builder,
        "validate_oof_case_leaf",
        lambda *args, **kwargs: {
            "case_id": "case-a",
            "patient_id": "patient-a",
            "oof_ready_sha256": "0" * 64,
            "binding_sha256": "b" * 64,
            "inputs": {
                name: {
                    "path": str(paths[name]),
                    "bytes": paths[name].stat().st_size,
                    "sha256": sha256_file(paths[name]),
                }
                for name in ("ct", "pet", "gt")
            },
            "m0": {
                "path": str(paths["m0"]),
                "bytes": paths["m0"].stat().st_size,
                "sha256": sha256_file(paths["m0"]),
            },
            "foreground_probability": {
                "path": str(paths["m0"]),
                "bytes": paths["m0"].stat().st_size,
                "sha256": sha256_file(paths["m0"]),
            },
        },
    )

    def validate_split(path, rows, experiment_config):
        assert experiment_config["marker"] == "config-consumed"
        return {}, {
            "case_to_partition": {"case-a": source_partition},
            "split_sha256": "1" * 64,
            "algorithm": "stable-patient-hash-v1",
            "seed": 20260717,
            "target_patient_counts": {"train": 264, "val": 57, "test": 57},
            "patient_counts": {"train": 264, "val": 57, "test": 57},
            "case_counts": {"train": 483, "val": 57, "test": 57},
        }

    monkeypatch.setattr(
        residual_builder, "load_and_validate_learning_split", validate_split
    )
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps({"marker": "config-consumed"}), encoding="utf-8")
    output_dir = tmp_path / "residuals"
    output_manifest = tmp_path / "residuals.jsonl"
    ready_receipt = tmp_path / "RESIDUAL_READY.json"
    argv = [
        "--oof-ready",
        str(oof_ready),
        "--learning-split",
        str(tmp_path / "split.json"),
        "--experiment-config",
        str(config_path),
        "--case-manifest",
        str(case_manifest),
        "--partitions",
        *selected_partitions,
        "--output-dir",
        str(output_dir),
        "--output-manifest",
        str(output_manifest),
        "--ready-receipt",
        str(ready_receipt),
    ]
    return argv, output_dir, output_manifest, ready_receipt


def test_main_publishes_residual_bundle_with_split_receipt(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    argv, output_dir, output_manifest, ready_receipt = _main_fixture(tmp_path, monkeypatch)
    assert residual_builder.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["cases"] == 1
    row = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert Path(row["fn_path"]).parent == output_dir
    assert Path(row["fp_path"]).parent == output_dir
    assert Path(row["fn_path"]).is_file()
    assert Path(row["fp_path"]).is_file()
    assert row["learning_split_receipt"]["algorithm"] == "stable-patient-hash-v1"
    assert row["learning_split_receipt"]["case_counts"] == {
        "train": 483,
        "val": 57,
        "test": 57,
    }
    ready = json.loads(ready_receipt.read_text(encoding="utf-8"))
    assert ready["schema_version"] == "PETCT-FN-FP-RESIDUAL-READY-v2.0"
    assert ready["cohort"]["fp_positive"]["case_count"] >= 0
    assert ready["cohort"]["zero_fp"]["case_count"] >= 0
    assert ready["status"] == "PASS"
    assert ready["residual_manifest"]["sha256"] == sha256_file(output_manifest)
    assert ready["cohort"]["selected_source"]["case_count"] == 1
    assert ready["cohort"]["generated"]["case_ids"] == ["case-a"]
    assert ready["cohort"]["fn_positive"]["case_ids"] == ["case-a"]
    assert ready["cohort"]["zero_fn"]["case_count"] == 0


def test_main_residual_write_failure_cleans_all_staging(
    tmp_path: Path, monkeypatch
) -> None:
    argv, output_dir, output_manifest, ready_receipt = _main_fixture(tmp_path, monkeypatch)
    original = residual_builder.write_binary_nifti
    calls = 0

    def fail_second(path, mask, reference):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected FP write failure")
        original(path, mask, reference)

    monkeypatch.setattr(residual_builder, "write_binary_nifti", fail_second)
    with pytest.raises(RuntimeError, match="injected FP write failure"):
        residual_builder.main(argv)
    assert not output_dir.exists()
    assert not output_manifest.exists()
    assert not ready_receipt.exists()
    assert not list(tmp_path.glob(".*.partial"))


def test_locked_test_partition_is_filtered_before_gt_is_opened(
    tmp_path: Path, monkeypatch
) -> None:
    argv, _, output_manifest, ready_receipt = _main_fixture(
        tmp_path,
        monkeypatch,
        source_partition="test",
        selected_partitions=("train", "val"),
    )
    original_load = residual_builder.nib.load

    def reject_test_gt(path, *args, **kwargs):
        if Path(path).name == "gt.nii.gz":
            raise AssertionError("locked test GT was opened")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(residual_builder.nib, "load", reject_test_gt)
    assert residual_builder.main(argv) == 0
    assert output_manifest.read_text(encoding="utf-8") == ""
    ready = json.loads(ready_receipt.read_text(encoding="utf-8"))
    assert ready["cohort"]["selected_source"]["case_count"] == 0
    assert ready["cohort"]["excluded"]["reasons"]["PARTITION_NOT_SELECTED"]["case_ids"] == ["case-a"]


@pytest.mark.parametrize(
    "partition,receipt_args",
    [
        ("test", []),
        ("val", ["--test-access-receipt", "not-allowed.json"]),
    ],
)
def test_partition_gate_precedes_every_manifest_or_volume_read(
    tmp_path: Path, monkeypatch, partition: str, receipt_args: list[str]
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("data was read before partition authorization")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(residual_builder, "load_jsonl", forbidden)
    monkeypatch.setattr(residual_builder, "validate_oof_ready_receipt_only", forbidden)
    monkeypatch.setattr(residual_builder.nib, "load", forbidden)
    monkeypatch.setattr(residual_builder.np, "load", forbidden, raising=False)
    argv = [
        "--oof-ready", str(tmp_path / "OOF_READY.json"),
        "--learning-split", str(tmp_path / "split.json"),
        "--experiment-config", str(tmp_path / "experiment.json"),
        "--case-manifest", str(tmp_path / "cases.jsonl"),
        "--partitions", partition,
        "--output-dir", str(tmp_path / "residuals"),
        "--output-manifest", str(tmp_path / "residuals.jsonl"),
        "--ready-receipt", str(tmp_path / "RESIDUAL_READY.json"),
        *receipt_args,
    ]
    with pytest.raises(SystemExit):
        residual_builder.main(argv)


def test_regular_source_path_rejects_symlink_before_resolve(
    tmp_path: Path, monkeypatch
) -> None:
    link = tmp_path / "link.nii.gz"
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve
    monkeypatch.setattr(
        Path, "is_symlink", lambda self: self == link or original_is_symlink(self)
    )

    def guarded_resolve(self, *args, **kwargs):
        if self == link:
            raise AssertionError("symlink was resolved before it was rejected")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(RuntimeError, match="non-symlink"):
        residual_builder._regular_source_path(link, label="GT")


def test_oof_mask_rejects_symlink_before_resolve(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "oof"
    run_dir.mkdir()
    link = run_dir / "mask.nii.gz"
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve
    monkeypatch.setattr(
        Path, "is_symlink", lambda self: self == link or original_is_symlink(self)
    )

    def guarded_resolve(self, *args, **kwargs):
        if self == link:
            raise AssertionError("OOF symlink was resolved before it was rejected")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(RuntimeError, match="non-symlink"):
        residual_builder.resolve_oof_mask(
            {"run_dir": str(run_dir)},
            {"mask": {"path": "mask.nii.gz", "bytes": 1, "sha256": "0" * 64}},
        )


def test_natural_binding_v6_schema_resolves_training_hashes(tmp_path: Path) -> None:
    """Regression (2026-08-18 KeyError): v6 OOF per-case records carry no
    training sha fields; the natural binding must resolve them from the v6
    validated envelope (checkpoints / splits_final) and live trainer-root
    files instead of the legacy per-case schema."""
    from baseline.validate_petct_m0_oof import (
        build_natural_oof_binding_from_validated,
    )
    from common.petct_mainline_lineage import M0_V6_OOF_SCHEMA

    run_dir = tmp_path / "oof_run"
    (run_dir / "masks").mkdir(parents=True)
    (run_dir / "probabilities").mkdir()
    m0 = run_dir / "masks" / "case-a.nii.gz"
    _write_nifti(m0, np.zeros((5, 5, 1), dtype=np.uint8))
    prob = run_dir / "probabilities" / "case-a.npz"
    prob.write_bytes(b"prob")

    trainer_root = tmp_path / "trainer_root"
    fold_dir = trainer_root / "fold_0"
    fold_dir.mkdir(parents=True)
    checkpoint = fold_dir / "checkpoint_final.pth"
    checkpoint.write_bytes(b"ckpt")
    plans = trainer_root / "plans.json"
    plans.write_text('{"marker": "plans"}\n', encoding="utf-8")
    dataset_json = trainer_root / "dataset.json"
    dataset_json.write_text('{"marker": "dataset"}\n', encoding="utf-8")
    splits_file = tmp_path / "splits_final.json"
    splits_file.write_text("{}", encoding="utf-8")
    ready = run_dir / "M0_V6_FIVEFOLD_OOF_READY.json"
    ready.write_text("{}", encoding="utf-8")

    def file_record(path: Path) -> dict:
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    checkpoints = [
        {"fold": fold, "checkpoint": file_record(checkpoint)} for fold in range(5)
    ]
    validated = {
        "status": "PASS",
        "schema_version": M0_V6_OOF_SCHEMA,
        "patient_excluded": True,
        "ready_path": str(ready),
        "ready_sha256": sha256_file(ready),
        "run_dir": str(run_dir),
        "checkpoints": checkpoints,
        "splits_final": file_record(splits_file),
        "cases": {
            "case-a": {
                "patient_id": "patient-a",
                "held_out_fold": 0,
                "mask": {
                    "path": "masks/case-a.nii.gz",
                    "bytes": m0.stat().st_size,
                    "sha256": sha256_file(m0),
                },
                "foreground_probability": {
                    "path": "probabilities/case-a.npz",
                    "bytes": prob.stat().st_size,
                    "sha256": sha256_file(prob),
                },
                "input_ct_sha256": "a" * 64,
                "input_pet_sha256": "b" * 64,
                "input_gt_sha256": "c" * 64,
            }
        },
    }
    binding = build_natural_oof_binding_from_validated(
        validated,
        ready_path=ready,
        case_id="case-a",
        patient_id="patient-a",
        m0_path=m0,
        leaf_binding=None,
    )
    assert binding["checkpoint_sha256"] == sha256_file(checkpoint)
    assert binding["plans_sha256"] == sha256_file(plans)
    assert binding["dataset_json_sha256"] == sha256_file(dataset_json)
    assert binding["splits_final_sha256"] == sha256_file(splits_file)
    assert len(binding["source_tree_sha256"]) == 64
    assert binding["preprocess_ready_sha256"] is None
    assert binding["full_train_ready_sha256"] is None
    assert binding["fold_receipt_sha256"] is None
    assert binding["input_ct_sha256"] == "a" * 64
    assert len(binding["binding_sha256"]) == 64
