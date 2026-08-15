from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
for directory in (SCRIPTS, SCRIPTS / "comparators"):
    sys.path.insert(0, str(directory))

from build_petct_external_comparator_manifest import (  # noqa: E402
    FORBIDDEN_RECORD_FIELDS,
    ManifestError,
    build_comparator_input,
    canonical_json_sha256,
    main,
)
from common.petct_learning import sha256_file  # noqa: E402


def _save_nifti(path: Path, values: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(values, np.eye(4)), str(path))


def _fixture(tmp_path: Path):
    shape = (6, 7, 2)
    ct_path = tmp_path / "ct.nii.gz"
    pet_path = tmp_path / "pet.nii.gz"
    m0_path = tmp_path / "m0.nii.gz"
    _save_nifti(ct_path, np.zeros(shape, dtype=np.float32))
    _save_nifti(pet_path, np.ones(shape, dtype=np.float32))
    _save_nifti(m0_path, np.zeros(shape, dtype=np.uint8))
    visible = tmp_path / "visible.npz"
    np.savez_compressed(visible, scribble=np.ones((2, 2), dtype=np.uint8))
    coordinates = [[2, 3, 1], [3, 3, 1]]
    episode = {
        "case_id": "case-001",
        "patient_id": "patient-001",
        "episode_id": "episode-001",
        "partition": "val",
        "held_out_fold": 2,
        "strategy": "centerline",
        "strategy_mode": "primary",
        "coordinates_xyz": coordinates,
        "ct_path": str(ct_path.resolve()),
        "pet_path": str(pet_path.resolve()),
        "m0_path": str(m0_path.resolve()),
        "ct_sha256": sha256_file(ct_path),
        "pet_sha256": sha256_file(pet_path),
        "m0_sha256": sha256_file(m0_path),
        "scribble_generation": {
            "coordinate_sha256": canonical_json_sha256(coordinates)
        },
    }
    tensor = {
        "episode_id": "episode-001",
        "patient_id": "patient-001",
        "partition": "val",
        "strategy": "centerline",
        "visible_npz": str(visible.resolve()),
        "visible_sha256": sha256_file(visible),
        "source_evaluation": {
            "m0_path": str(m0_path.resolve()),
            "m0_sha256": sha256_file(m0_path),
            "scribble_coordinates_xyz": coordinates,
        },
    }
    source = {
        "case_id": "case-001",
        "patient_id": "patient-001",
        "ct_path": str(ct_path.resolve()),
        "pet_path": str(pet_path.resolve()),
    }
    provenance = {
        "experiment_config_sha256": "1" * 64,
        "case_manifest_sha256": "2" * 64,
        "learning_split_sha256": "3" * 64,
        "oof_ready_sha256": "4" * 64,
        "natural_episode_manifest_sha256": "5" * 64,
        "natural_tensor_manifest_sha256": "6" * 64,
    }
    return episode, tensor, source, provenance


def test_builds_original_grid_truth_firewalled_manifest(tmp_path: Path) -> None:
    episode, tensor, source, provenance = _fixture(tmp_path)
    manifest_path = tmp_path / "input.json"
    scribble_dir = tmp_path / "scribbles"
    document = build_comparator_input(
        natural_rows=[episode],
        tensor_rows=[tensor],
        source_rows=[source],
        case_to_partition={"case-001": "val"},
        selected_partition="val",
        output_manifest=manifest_path,
        scribble_dir=scribble_dir,
        provenance=provenance,
    )

    assert document["schema_version"] == "PETCT-EXTERNAL-COMPARATOR-INPUT-v1.0"
    assert document["partition"] == "validation"
    assert document["record_count"] == 1
    record = document["records"][0]
    assert record["split"] == "validation"
    assert record["step"] == 1
    assert record["bg_scribble_path"] is None
    assert not (FORBIDDEN_RECORD_FIELDS & set(record))
    scribble_image = nib.load(record["fg_scribble_path"])
    scribble = np.asarray(scribble_image.dataobj)
    assert scribble.shape == (6, 7, 2)
    assert {tuple(index) for index in np.argwhere(scribble > 0)} == {
        (2, 3, 1),
        (3, 3, 1),
    }
    assert record["same_frozen_scribble_receipt"]["raster_sha256"] == sha256_file(
        Path(record["fg_scribble_path"])
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == document


def test_tensor_must_bind_the_same_frozen_scribble(tmp_path: Path) -> None:
    episode, tensor, source, provenance = _fixture(tmp_path)
    tensor["source_evaluation"]["scribble_coordinates_xyz"] = [[1, 1, 1]]
    with pytest.raises(ManifestError, match="do not share one scribble"):
        build_comparator_input(
            natural_rows=[episode],
            tensor_rows=[tensor],
            source_rows=[source],
            case_to_partition={"case-001": "val"},
            selected_partition="val",
            output_manifest=tmp_path / "input.json",
            scribble_dir=tmp_path / "scribbles",
            provenance=provenance,
        )
    assert not (tmp_path / "input.json").exists()
    assert not (tmp_path / "scribbles").exists()


def test_requires_one_primary_scribble_per_case(tmp_path: Path) -> None:
    episode, tensor, source, provenance = _fixture(tmp_path)
    second_episode = dict(episode, episode_id="episode-002")
    second_tensor = dict(tensor, episode_id="episode-002")
    with pytest.raises(ManifestError, match="exactly one frozen primary scribble"):
        build_comparator_input(
            natural_rows=[episode, second_episode],
            tensor_rows=[tensor, second_tensor],
            source_rows=[source],
            case_to_partition={"case-001": "val"},
            selected_partition="val",
            output_manifest=tmp_path / "input.json",
            scribble_dir=tmp_path / "scribbles",
            provenance=provenance,
        )


def test_no_clobber_for_manifest_or_scribble_directory(tmp_path: Path) -> None:
    episode, tensor, source, provenance = _fixture(tmp_path)
    manifest_path = tmp_path / "input.json"
    scribble_dir = tmp_path / "scribbles"
    build_comparator_input(
        natural_rows=[episode],
        tensor_rows=[tensor],
        source_rows=[source],
        case_to_partition={"case-001": "val"},
        selected_partition="val",
        output_manifest=manifest_path,
        scribble_dir=scribble_dir,
        provenance=provenance,
    )
    with pytest.raises(FileExistsError, match="output manifest"):
        build_comparator_input(
            natural_rows=[episode],
            tensor_rows=[tensor],
            source_rows=[source],
            case_to_partition={"case-001": "val"},
            selected_partition="val",
            output_manifest=manifest_path,
            scribble_dir=tmp_path / "scribbles-2",
            provenance=provenance,
        )


def test_cli_test_partition_requires_consumed_receipt_before_any_input_read(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run" / "input.json"
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--oof-ready",
                str(tmp_path / "missing-oof.json"),
                "--case-manifest",
                str(tmp_path / "missing-cases.jsonl"),
                "--learning-split",
                str(tmp_path / "missing-split.json"),
                "--natural-episode-manifest",
                str(tmp_path / "missing-episodes.jsonl"),
                "--natural-tensor-manifest",
                str(tmp_path / "missing-tensors.jsonl"),
                "--experiment-config",
                str(tmp_path / "missing-config.json"),
                "--partition",
                "test",
                "--scribble-dir",
                str(tmp_path / "run" / "scribbles"),
                "--output-manifest",
                str(output),
            ]
        )
    assert not output.exists()


def test_cli_validation_rejects_test_receipt_before_any_input_read(
    tmp_path: Path,
) -> None:
    output = tmp_path / "input.json"
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--oof-ready",
                str(tmp_path / "missing-oof.json"),
                "--case-manifest",
                str(tmp_path / "missing-cases.jsonl"),
                "--learning-split",
                str(tmp_path / "missing-split.json"),
                "--natural-episode-manifest",
                str(tmp_path / "missing-episodes.jsonl"),
                "--natural-tensor-manifest",
                str(tmp_path / "missing-tensors.jsonl"),
                "--experiment-config",
                str(tmp_path / "missing-config.json"),
                "--partition",
                "val",
                "--scribble-dir",
                str(tmp_path / "scribbles"),
                "--output-manifest",
                str(output),
                "--test-access-receipt",
                str(tmp_path / "must-not-be-read.json"),
            ]
        )
    assert not output.exists()
