"""Contract tests for the expanded baseline set (SW-FastEdit, SwinUNETR,
PRISM) — structure/contract only; no GPU, no MONAI, no scientific result."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from baseline.train_petct_swinunetr_fold import (  # noqa: E402
    SwinUNETRError,
    load_fivefold_split,
    train_fold,
)
from comparators.prism_petct_adapter import (  # noqa: E402
    CHECKPOINT_SCHEMA as PRISM_CHECKPOINT_SCHEMA,
    PrismAdapterError,
    validate_input_manifest as validate_prism_manifest,
    validate_prism_checkpoint,
)
from comparators.swfastedit_petct_adapter import (  # noqa: E402
    SWFastEditError,
    encode_guidance_signal,
    pet_percentile_clip,
    validate_input_manifest as validate_swfastedit_manifest,
)


# ---------------------------------------------------------------- SW-FastEdit

def test_guidance_signal_sphere_contract() -> None:
    volume = encode_guidance_signal(
        (8, 8, 8), [[4, 4, 4]], sigma=1.0, with_disks=False
    )
    assert volume.shape == (8, 8, 8)
    assert volume[4, 4, 4] == pytest.approx(1.0)  # min-max normalized peak
    assert volume[4, 4, 5] > 0.0  # gaussian falloff
    assert volume[0, 0, 0] == 0.0


def test_guidance_signal_disks_threshold() -> None:
    volume = encode_guidance_signal((8, 8, 8), [[4, 4, 4]], with_disks=True)
    assert set(np.unique(volume)) <= {0.0, 1.0}
    assert volume.sum() >= 7  # approximately radius-3 disk per source comment


def test_guidance_signal_mirrors_upstream_point_policy() -> None:
    # Negative points are skipped, positive overflow is clamped (upstream
    # transforms.py behavior preserved exactly).
    volume = encode_guidance_signal((8, 8, 8), [[4, 4, 4], [-1, 2, 2], [12, 3, 3]])
    assert volume[4, 4, 4] == pytest.approx(1.0)
    assert volume[7, 3, 3] >= 0.0  # clamped, no error


def test_pet_percentile_clip() -> None:
    volume = np.arange(100, dtype=np.float32).reshape(10, 10, 1)
    clipped = pet_percentile_clip(volume)
    assert clipped.min() >= float(np.percentile(volume, 0.05)) - 1e-4
    assert clipped.max() <= float(np.percentile(volume, 99.95)) + 1e-4


def _write_manifest(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"records": records}))
        stream.write("\n")


def test_swfastedit_manifest_validation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    record = {
        "case_id": "c1",
        "patient_id": "p1",
        "split": "test",
        "fold": 0,
        "step": 1,
        "pet_path": "pet.nii.gz",
        "fg_scribble_path": "fg.nii.gz",
        "original_grid_reference": "ref.nii.gz",
        "scribble_strategy": "centerline",
        "scribble_polarity": "foreground",
    }
    _write_manifest(manifest, [record])
    rows = validate_swfastedit_manifest(manifest)
    assert len(rows) == 1
    broken = dict(record)
    broken.pop("patient_id")
    broken_manifest = tmp_path / "broken.jsonl"
    _write_manifest(broken_manifest, [broken])
    with pytest.raises(SWFastEditError):
        validate_swfastedit_manifest(broken_manifest)


def test_swfastedit_manifest_rejects_bad_polarity(tmp_path: Path) -> None:
    manifest = tmp_path / "bad-polarity.jsonl"
    record = {
        "case_id": "c1",
        "patient_id": "p1",
        "split": "test",
        "fold": 0,
        "step": 1,
        "pet_path": "pet.nii.gz",
        "fg_scribble_path": "fg.nii.gz",
        "original_grid_reference": "ref.nii.gz",
        "scribble_strategy": "centerline",
        "scribble_polarity": "signed",  # invalid
    }
    _write_manifest(manifest, [record])
    with pytest.raises(SWFastEditError):
        validate_swfastedit_manifest(manifest)


# ---------------------------------------------------------------- SwinUNETR

def _fivefold_split(tmp_path: Path) -> Path:
    split = [
        {"train": [f"t{i}" for i in range(5)], "val": [f"v{i}" for i in range(2)]}
        for _ in range(5)
    ]
    path = tmp_path / "splits_final.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    return path


def test_fivefold_split_list_of_dicts(tmp_path: Path) -> None:
    path = _fivefold_split(tmp_path)
    folds, _ = load_fivefold_split(path)
    assert len(folds) == 5
    assert folds[0]["train"][0] == "t0"


def test_fivefold_split_rejects_string_key_variant(tmp_path: Path) -> None:
    path = tmp_path / "string-key.json"
    path.write_text(
        json.dumps({f"fold_{index}": {"train": [], "val": []} for index in range(5)}),
        encoding="utf-8",
    )
    with pytest.raises(SwinUNETRError):
        load_fivefold_split(path)


def test_fivefold_split_rejects_wrong_fold_count(tmp_path: Path) -> None:
    path = tmp_path / "three-folds.json"
    path.write_text(json.dumps([{"train": [], "val": []}] * 3), encoding="utf-8")
    with pytest.raises(SwinUNETRError):
        load_fivefold_split(path)


def test_fivefold_split_rejects_non_string_ids(tmp_path: Path) -> None:
    path = tmp_path / "int-ids.json"
    path.write_text(
        json.dumps([{"train": [1, 2], "val": []}] * 5), encoding="utf-8"
    )
    with pytest.raises(SwinUNETRError):
        load_fivefold_split(path)


def test_train_fold_structure_receipt(tmp_path: Path) -> None:
    split = [{"train": ["a", "b"], "val": ["c"]} for _ in range(5)]
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(split), encoding="utf-8")
    checkpoint = train_fold(
        fold=0,
        split=split,
        preprocessed_dir=tmp_path,
        output_dir=tmp_path / "out",
        splits_path=splits_path,
        plans_path=None,
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        seed=3407,
        device="cpu",
    )
    assert checkpoint["schema_version"] == "PETCT-SWINUNETR-CHECKPOINT-v1.0"
    assert checkpoint["fold"] == 0
    assert checkpoint["train_cases"] == ["a", "b"]
    assert checkpoint["status"].startswith("STRUCTURE_READY")


# ---------------------------------------------------------------- PRISM

def _prism_checkpoint(path: Path, *, warm_start="from_scratch") -> None:
    torch.save(
        {
            "schema_version": PRISM_CHECKPOINT_SCHEMA,
            "warm_start": warm_start,
            "learning_split_sha256": "a" * 64,
            "splits_m0_v6_sha256": "b" * 64,
        },
        path,
    )


def test_prism_checkpoint_missing_points_to_plan(tmp_path: Path) -> None:
    missing = tmp_path / "absent.pt"
    with pytest.raises(PrismAdapterError) as info:
        validate_prism_checkpoint(missing)
    assert "docs/11-EXTERNAL-BASELINE-EXPANSION.md" in str(info.value)


def test_prism_checkpoint_accepts_from_scratch(tmp_path: Path) -> None:
    path = tmp_path / "prism.pt"
    _prism_checkpoint(path)
    checkpoint = validate_prism_checkpoint(path)
    assert checkpoint["warm_start"] == "from_scratch"


def test_prism_checkpoint_rejects_warm_start(tmp_path: Path) -> None:
    path = tmp_path / "warm.pt"
    _prism_checkpoint(path, warm_start="sam_med3d")
    with pytest.raises(PrismAdapterError):
        validate_prism_checkpoint(path)


def test_prism_checkpoint_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "other.pt"
    torch.save({"schema_version": "SOMETHING-ELSE-v9"}, path)
    with pytest.raises(PrismAdapterError):
        validate_prism_checkpoint(path)


def test_prism_manifest_validation(tmp_path: Path) -> None:
    manifest = tmp_path / "prism-manifest.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "case_id": "c1",
                "patient_id": "p1",
                "pet_path": "pet.nii.gz",
                "ct_path": "ct.nii.gz",
                "m0_path": "m0.nii.gz",
                "fg_scribble_path": "fg.nii.gz",
                "original_grid_reference": "ref.nii.gz",
            }
        ],
    )
    rows = validate_prism_manifest(manifest)
    assert len(rows) == 1
