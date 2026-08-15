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

from evaluation.evaluate_petct_m0_oof import (  # noqa: E402
    _load_image,
    _resolve_oof_mask,
    evaluate_m0_oof,
)
import baseline.validate_petct_m0_oof as oof_contract  # noqa: E402


class _FakeOfficialEvaluator:
    def __init__(self, *, overlap_threshold: float, connectivity: int) -> None:
        assert overlap_threshold == 0.1
        assert connectivity == 18
        self.rows: list[dict[str, Any]] = []

    def __call__(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        _case_name: str,
        *,
        spacing: tuple[float, ...],
        suv: np.ndarray | None,
    ) -> dict[str, Any]:
        assert suv is not None
        if ground_truth.any():
            metrics = {
                "dsc": 0.75,
                "f1": 0.8,
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "fpv": 0.0,
                "fnv": 0.0,
            }
        else:
            prediction_count = int(prediction.sum())
            metrics = {
                "dsc": float("nan"),
                "f1": float("nan"),
                "tp": 0,
                "fp": int(prediction_count > 0),
                "fn": 0,
                "fpv": prediction_count * float(np.prod(spacing)) / 1000.0,
                "fnv": float("nan"),
            }
        self.rows.append(metrics)
        return metrics

    def aggregate(self, *, weighted: bool) -> dict[str, float]:
        assert weighted is False
        assert self.rows
        return {"dsc": 0.75, "f1_aggregated": 0.8}


def _write_nifti(path: Path, *, positive: bool) -> None:
    data = np.zeros((2, 2, 2), dtype=np.uint8)
    if positive:
        data[0, 0, 0] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    display = path if relative_to is None else path.relative_to(relative_to)
    return {
        "path": display.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    run_dir = tmp_path / "oof-run"
    run_dir.mkdir()
    source_rows: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any]] = {}
    partitions = {
        "case_0": "train",
        "case_1": "val",
        "case_2": "test",
        "case_3": "train",
        "case_4": "train",
    }
    for fold in range(5):
        case_id = f"case_{fold}"
        patient_id = f"patient_{fold}"
        prediction = run_dir / "masks" / f"{case_id}.nii.gz"
        _write_nifti(prediction, positive=True)
        probability = run_dir / "probabilities" / f"{case_id}.npz"
        probability.parent.mkdir(parents=True, exist_ok=True)
        probability.write_bytes(f"probability:{case_id}".encode("utf-8"))
        ct = tmp_path / "ct" / f"{case_id}.nii.gz"
        pet = tmp_path / "pet" / f"{case_id}.nii.gz"
        _write_nifti(ct, positive=False)
        _write_nifti(pet, positive=True)
        gt = tmp_path / "gt" / f"{case_id}.nii.gz"
        _write_nifti(gt, positive=fold != 1)
        source_rows.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "held_out_fold": fold,
                "gt_path": str(gt),
                "ct_path": str(ct),
                "pet_path": str(pet),
                "gt_bytes": gt.stat().st_size,
                "ct_bytes": ct.stat().st_size,
                "pet_bytes": pet.stat().st_size,
                "gt_sha256": hashlib.sha256(gt.read_bytes()).hexdigest(),
                "ct_sha256": hashlib.sha256(ct.read_bytes()).hexdigest(),
                "pet_sha256": hashlib.sha256(pet.read_bytes()).hexdigest(),
            }
        )
        cases[case_id] = {
            "case_id": case_id,
            "patient_id": patient_id,
            "held_out_fold": fold,
            "mask": _record(prediction, relative_to=run_dir),
            "foreground_probability": _record(probability, relative_to=run_dir),
            "input_ct_bytes": ct.stat().st_size,
            "input_pet_bytes": pet.stat().st_size,
            "input_gt_bytes": gt.stat().st_size,
            "input_ct_sha256": hashlib.sha256(ct.read_bytes()).hexdigest(),
            "input_pet_sha256": hashlib.sha256(pet.read_bytes()).hexdigest(),
            "input_gt_sha256": hashlib.sha256(gt.read_bytes()).hexdigest(),
        }
    case_manifest = tmp_path / "cases.jsonl"
    case_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8"
    )
    oof_ready = tmp_path / "OOF_READY.json"
    learning_split = tmp_path / "learning_split.json"
    experiment_config = tmp_path / "experiment.json"
    official_metrics = tmp_path / "metrics.py"
    oof_ready.write_text("{}\n", encoding="utf-8")
    learning_split.write_text("{}\n", encoding="utf-8")
    experiment_config.write_text("{}\n", encoding="utf-8")
    official_metrics.write_text("# injected test evaluator\n", encoding="utf-8")

    def ready_validator(_path: Path) -> dict[str, Any]:
        return {
            "status": "PASS",
            "patient_excluded": True,
            "run_dir": str(run_dir),
            "ready_path": str(_path.resolve()),
            "ready_sha256": hashlib.sha256(_path.read_bytes()).hexdigest(),
            "cases": cases,
        }

    def split_loader(
        _path: Path, _rows: list[dict[str, Any]], _experiment: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return {}, {
            "case_to_partition": partitions,
            "split_sha256": hashlib.sha256(_path.read_bytes()).hexdigest(),
        }

    return {
        "oof_ready": oof_ready,
        "case_manifest": case_manifest,
        "learning_split": learning_split,
        "experiment_config": experiment_config,
        "official_metrics": official_metrics,
        "ready_validator": ready_validator,
        "split_loader": split_loader,
        "test_gt": tmp_path / "gt" / "case_2.nii.gz",
    }


def _evaluate(tmp_path: Path, **overrides: Any):
    fixture = _fixture(tmp_path)
    kwargs = {
        key: fixture[key]
        for key in (
            "oof_ready",
            "case_manifest",
            "learning_split",
            "experiment_config",
            "official_metrics",
            "ready_validator",
            "split_loader",
        )
    }
    kwargs.update(
        {
            "rows_path": tmp_path / "rows.jsonl",
            "summary_path": tmp_path / "summary.json",
            "metric_evaluator_class": _FakeOfficialEvaluator,
        }
    )
    kwargs.update(overrides)
    return fixture, evaluate_m0_oof(**kwargs)


def test_default_partitions_exclude_test_and_serialize_empty_gt_as_null(
    tmp_path: Path,
) -> None:
    fixture, (rows, summary) = _evaluate(tmp_path)

    assert fixture["test_gt"].exists()
    assert {row["partition"] for row in rows} == {"train", "val"}
    assert len(rows) == 4
    empty = next(row for row in rows if row["case_id"] == "case_1")
    assert empty["official_metric_eligible"] is False
    assert empty["official_metric_ineligibility_reason"] == "EMPTY_GT"
    assert empty["official_metric_denominators"] == {
        "gt_voxels": 0,
        "gt_lesions": 0,
    }
    assert empty["dice"] is None
    assert empty["dmm_f1"] is None
    assert empty["fnv_ml"] is None
    assert empty["empty_gt_false_positive"] is True
    assert summary["official_autoPETV"]["eligible_case_count"] == 3
    assert summary["official_autoPETV"]["ineligible_empty_gt_case_count"] == 1
    diagnostics = summary["empty_gt_false_positive_diagnostics"]
    assert diagnostics["false_positive_case_count"] == 1
    assert diagnostics["false_positive_lesion_count"] == 1
    assert (
        summary["positive_gt_patient_clustered"]["dice"][
            "defined_episode_count"
        ]
        == 3.0
    )
    assert "NaN" not in (tmp_path / "rows.jsonl").read_text(encoding="utf-8")
    assert "NaN" not in (tmp_path / "summary.json").read_text(encoding="utf-8")


def test_unselected_test_gt_has_zero_hash_and_zero_nifti_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    test_gt = fixture["test_gt"].resolve()
    original_sha256 = oof_contract._sha256
    original_load = nib.load

    def guarded_sha256(path: Path) -> str:
        if Path(path).resolve() == test_gt:
            raise AssertionError("unselected test GT was hashed")
        return original_sha256(path)

    def guarded_load(path: str | Path, *args: Any, **kwargs: Any):
        if Path(path).resolve() == test_gt:
            raise AssertionError("unselected test GT was opened")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(oof_contract, "_sha256", guarded_sha256)
    monkeypatch.setattr(nib, "load", guarded_load)
    kwargs = {
        key: fixture[key]
        for key in (
            "oof_ready",
            "case_manifest",
            "learning_split",
            "experiment_config",
            "official_metrics",
            "ready_validator",
            "split_loader",
        )
    }
    rows, _ = evaluate_m0_oof(
        **kwargs,
        rows_path=tmp_path / "guarded-rows.jsonl",
        summary_path=tmp_path / "guarded-summary.json",
        metric_evaluator_class=_FakeOfficialEvaluator,
    )
    assert "case_2" not in {row["case_id"] for row in rows}


def test_test_partition_requires_explicit_selection_and_access_gate(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    kwargs = {
        key: fixture[key]
        for key in (
            "oof_ready",
            "case_manifest",
            "learning_split",
            "experiment_config",
            "official_metrics",
            "ready_validator",
            "split_loader",
        )
    }
    with pytest.raises(RuntimeError, match="consumed test-access receipt"):
        evaluate_m0_oof(
            **kwargs,
            rows_path=tmp_path / "blocked-rows.jsonl",
            summary_path=tmp_path / "blocked-summary.json",
            partitions=("test",),
            metric_evaluator_class=_FakeOfficialEvaluator,
        )
    assert not (tmp_path / "blocked-rows.jsonl").exists()

    receipt = tmp_path / "consumed-test-access.json"
    receipt.write_text('{"status":"fixture"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="rejects a test receipt"):
        evaluate_m0_oof(
            **kwargs,
            rows_path=tmp_path / "val-blocked-rows.jsonl",
            summary_path=tmp_path / "val-blocked-summary.json",
            partitions=("val",),
            test_access_receipt=receipt,
            metric_evaluator_class=_FakeOfficialEvaluator,
        )

    formal_run = tmp_path / "formal-run"
    formal_run.mkdir()
    validated_access: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def validate_test_access(
        selected: tuple[str, ...], **access: Any
    ) -> dict[str, Any]:
        validated_access.append((selected, access))
        assert selected == ("test",)
        assert access["receipt_path"] == receipt
        assert access["run_root"] == formal_run
        assert all(
            Path(path).resolve().is_relative_to(formal_run.resolve())
            for path in access["output_paths"]
        )
        return {"status": "CONSUMED", "receipt_sha256": "a" * 64}

    rows, summary = evaluate_m0_oof(
        **kwargs,
        rows_path=formal_run / "test-rows.jsonl",
        summary_path=formal_run / "test-summary.json",
        partitions=("test",),
        test_access_receipt=receipt,
        run_root=formal_run,
        test_access_validator=validate_test_access,
        metric_evaluator_class=_FakeOfficialEvaluator,
    )
    assert [row["case_id"] for row in rows] == ["case_2"]
    assert summary["selected_partitions"] == ["test"]
    assert summary["test_access"]["required"] is True
    assert summary["test_access"]["consumed_receipt_sha256"] == "a" * 64
    assert summary["test_access"]["bound_run_root"] == str(formal_run.resolve())
    assert len(validated_access) == 1


def test_all_empty_gt_partition_has_null_official_aggregate_and_zero_denominator(
    tmp_path: Path,
) -> None:
    _, (rows, summary) = _evaluate(tmp_path, partitions=("val",))

    assert len(rows) == 1
    official = summary["official_autoPETV"]
    assert official["dsc"] is None
    assert official["dmm_f1_aggregated"] is None
    assert official["eligible_case_count"] == 0
    assert official["denominators"] == {
        "dsc_cases": 0,
        "dmm_tp": 0,
        "dmm_fp": 0,
        "dmm_fn": 0,
        "dmm_gt_lesions": 0,
    }
    json.dumps(summary, allow_nan=False)


def test_load_image_rejects_symlink_before_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "linked-image.nii.gz"
    original_resolve = Path.resolve
    original_is_symlink = Path.is_symlink

    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self == link or original_is_symlink(self),
    )

    def guarded_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if self == link:
            raise AssertionError("resolve called before symlink rejection")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(RuntimeError, match="missing or not a regular file"):
        _load_image(link, label="image")


def test_resolve_oof_mask_rejects_symlink_before_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "linked-oof.nii.gz"
    original_resolve = Path.resolve
    original_is_symlink = Path.is_symlink

    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: self == link or original_is_symlink(self),
    )

    def guarded_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if self == link:
            raise AssertionError("resolve called before symlink rejection")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        _resolve_oof_mask(tmp_path, {"path": link.name})
