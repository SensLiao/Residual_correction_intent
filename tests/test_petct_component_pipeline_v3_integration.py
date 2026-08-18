"""Integration checks for the PET/CT v3 component producer/target chain."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_components import (  # noqa: E402
    COMPONENT_DESCRIPTOR_FIELDS,
    cue_hit_component_key,
    cue_hit_component_position,
    enumerate_components,
)
from common.petct_program_learning import _load_components  # noqa: E402
from data.materialize_petct_component_candidates import (  # noqa: E402
    main as candidate_main,
    materialize_candidate_record,
    physical_crop_resample_2d,
)
from data.materialize_petct_component_targets import (  # noqa: E402
    join_audit_source_evaluations,
    main as target_main,
    materialize_target_record,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_nifti(path: Path, array: np.ndarray, spacing_xyz: tuple[float, ...]) -> None:
    affine = np.diag([*spacing_xyz, 1.0])
    nib.save(nib.Nifti1Image(array.astype(np.uint8), affine), str(path))


def _strict_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-finite JSON constant: %s" % value)
        ),
    )


def _pipeline_fixture(tmp_path: Path):
    shape = (12, 12, 5)  # native NIfTI x,y,z
    spacing_xyz = (2.0, 3.0, 5.0)
    center_z = 2
    m0 = np.zeros(shape, dtype=np.uint8)
    gt = np.zeros(shape, dtype=np.uint8)
    authorized = np.zeros(shape, dtype=np.uint8)
    gt[3:9, 4:8, center_z] = 1
    m0[3:5, 4:8, center_z] = 1
    m0[7:9, 4:8, center_z] = 1
    authorized[5:7, 4:8, center_z] = 1
    cue_coordinates = [[5, 5, center_z]]

    m0_path = tmp_path / "m0.nii.gz"
    gt_path = tmp_path / "gt.nii.gz"
    authorized_path = tmp_path / "authorized.nii.gz"
    _save_nifti(m0_path, m0, spacing_xyz)
    _save_nifti(gt_path, gt, spacing_xyz)
    _save_nifti(authorized_path, authorized, spacing_xyz)

    output_size = 9
    field_mm = 24.0
    center_xy = np.asarray([5.0, 5.0])
    cue = np.zeros(shape, dtype=np.uint8)
    cue[5, 5, center_z] = 1
    crop_kwargs = {
        "center_xy": center_xy,
        "spacing_xy": spacing_xyz[:2],
        "field_mm": field_mm,
        "output_size": output_size,
        "order": 0,
    }
    visible_m0 = physical_crop_resample_2d(m0[:, :, center_z], **crop_kwargs).astype(
        np.uint8
    )
    visible_cue = physical_crop_resample_2d(cue[:, :, center_z], **crop_kwargs).astype(
        np.uint8
    )
    assert visible_cue.any()
    visible_path = tmp_path / "visible.npz"
    np.savez_compressed(
        visible_path,
        visual=np.zeros((17, output_size, output_size), dtype=np.float32),
        m0=visible_m0,
        scribble=visible_cue,
        cue_fg=visible_cue,
        cue_bg=np.zeros_like(visible_cue),
        spacing_xy=np.asarray([field_mm / output_size] * 2, dtype=np.float32),
    )
    row = {
        "episode_id": "ep-multi-positive",
        "patient_id": "patient-1",
        "partition": "train",
        "operation": "ADD",
        "goal": "ADD_SAME_COMPLETE",
        "center_z": center_z,
        "visible_npz": str(visible_path.resolve()),
        "visible_sha256": _sha256_file(visible_path),
        "geometry": {
            "crop_center_xy_voxel": center_xy.tolist(),
            "crop_field_mm": field_mm,
            "output_size_px": output_size,
            "original_spacing_xy": list(spacing_xyz[:2]),
            "output_spacing_xy": [field_mm / output_size] * 2,
            "image_interpolation": "linear",
            "mask_interpolation": "nearest",
        },
        "source_evaluation": {
            "m0_path": str(m0_path.resolve()),
            "m0_sha256": _sha256_file(m0_path),
            "gt_path": str(gt_path.resolve()),
            "gt_sha256": _sha256_file(gt_path),
            "authorized_path": str(authorized_path.resolve()),
            "authorized_sha256": _sha256_file(authorized_path),
            "center_z": center_z,
            "scribble_coordinates_xyz": cue_coordinates,
        },
    }
    manifest = tmp_path / "learning.jsonl"
    manifest.write_text(json.dumps(row, allow_nan=False) + "\n", encoding="utf-8")
    return row, manifest, visible_m0


def test_xyz_axis2_and_anisotropic_physical_euclidean_distance():
    mask = np.zeros((8, 9, 6), dtype=np.uint8)
    mask[4, 5, 3:5] = 1
    spacing = np.asarray([2.0, 3.0, 5.0])
    enumeration = enumerate_components(
        mask,
        episode_id="axis-distance",
        m_sha256="mask-sha",
        spacing_xyz=spacing,
        cue_voxels=np.asarray([[1, 1, 1]]),
        prompted_z=4,
    )
    component = enumeration.components[0]
    assert component.position == 0
    assert component.z_span == 2
    assert component.prompted_slice_mask.shape == mask.shape[:2]
    assert component.prompted_slice_mask[4, 5] == 1
    assert component.prompted_slice_overlap == 1
    assert component.distance_from_cue_mm == pytest.approx(np.sqrt(6**2 + 12**2 + 10**2))
    assert np.isfinite(component.descriptor_vector()).all()


def test_cue_hit_requires_true_label_membership_not_bounding_box():
    mask = np.zeros((7, 7, 4), dtype=np.uint8)
    mask[1:6, 1:6, 2] = 1
    mask[2:5, 2:5, 2] = 0  # bounding-box hole
    enumeration = enumerate_components(
        mask,
        episode_id="cue-hole",
        m_sha256="mask-sha",
        spacing_xyz=np.ones(3),
        cue_voxels=np.asarray([[3, 3, 2]]),
        prompted_z=2,
    )
    assert enumeration.components[0].cue_overlap_voxels == 0
    assert cue_hit_component_position(enumeration, np.asarray([[3, 3, 2]]), mask) is None
    assert cue_hit_component_key(enumeration, np.asarray([[3, 3, 2]]), mask) is None
    assert cue_hit_component_position(enumeration, np.asarray([[1, 3, 2]]), mask) == 0
    assert cue_hit_component_key(enumeration, np.asarray([[1, 3, 2]]), mask) == (
        enumeration.components[0].component_key
    )


def test_candidate_to_target_pipeline_shape_axis_and_multi_positive(tmp_path: Path):
    row, manifest, visible_m0 = _pipeline_fixture(tmp_path)
    candidates_dir = tmp_path / "candidates"
    candidate_summary = tmp_path / "candidate-summary.jsonl"
    assert candidate_main(
        [
            "--learning-manifest",
            str(manifest),
            "--output",
            str(candidates_dir),
            "--summary",
            str(candidate_summary),
        ]
    ) == 0

    candidate_path = candidates_dir / (row["episode_id"] + ".json")
    candidate_text = candidate_path.read_text(encoding="utf-8")
    assert "NaN" not in candidate_text and "Infinity" not in candidate_text
    assert "gt_path" not in candidate_text and "authorized_path" not in candidate_text
    candidate = _strict_json(candidate_path)
    assert candidate["axis_order"] == "xyz"
    assert candidate["axial_axis"] == 2
    assert candidate["prompted_z"] == row["center_z"]
    assert candidate["descriptor_order"] == list(COMPONENT_DESCRIPTOR_FIELDS)
    vectors = np.asarray(
        [component["descriptor_vector"] for component in candidate["components"]],
        dtype=np.float32,
    )
    masks = np.asarray(
        [component["prompted_slice_mask"] for component in candidate["components"]],
        dtype=np.uint8,
    )
    assert vectors.shape == (2, 7)
    assert masks.shape == (2, *visible_m0.shape)
    assert np.array_equal(np.logical_or.reduce(masks).astype(np.uint8), visible_m0)
    assert [component["position"] for component in candidate["components"]] == [0, 1]
    assert [component["candidate_position"] for component in candidate["components"]] == [
        0,
        1,
    ]
    assert len({component["component_key"] for component in candidate["components"]}) == 2
    assert np.isfinite(vectors).all()
    loaded_vectors, valid, loaded_masks, loaded_keys, cue_hit = _load_components(
        {row["episode_id"]: candidate}, row["episode_id"], {}
    )
    assert loaded_vectors.shape == (2, 7)
    assert valid.tolist() == [True, True]
    assert np.array_equal(loaded_masks, masks)
    assert loaded_keys == [
        component["component_key"] for component in candidate["components"]
    ]
    assert cue_hit is None

    targets_dir = tmp_path / "targets"
    target_summary = tmp_path / "target-summary.jsonl"
    assert target_main(
        [
            "--learning-manifest",
            str(manifest),
            "--candidate-summary",
            str(candidate_summary),
            "--output",
            str(targets_dir),
            "--summary",
            str(target_summary),
        ]
    ) == 0
    target_path = targets_dir / (row["episode_id"] + ".json")
    target_text = target_path.read_text(encoding="utf-8")
    assert "NaN" not in target_text and "Infinity" not in target_text
    target = _strict_json(target_path)
    assert target["pointer_targets"] == [0, 1]
    assert target["pointer_target_positions"] == [0, 1]
    assert target["pointer_target_component_keys"] == [
        component["component_key"] for component in candidate["components"]
    ]
    assert target["candidate_component_count"] == candidate["component_count"]
    summary_row = _strict_json(candidate_summary)
    assert summary_row["component_keys"] == [
        component["component_key"] for component in candidate["components"]
    ]


def test_audit_manifest_restores_source_evaluation_without_widening_visible_row(
    tmp_path: Path,
):
    row, _, _ = _pipeline_fixture(tmp_path)
    audit_source = dict(row)
    public_row = dict(row)
    source_evaluation = public_row.pop("source_evaluation")
    public_row["evaluation_document"] = str(tmp_path / "evaluation-document.json")
    audit_row = {
        "episode_id": row["episode_id"],
        "source_record": audit_source,
        "source_record_sha256": hashlib.sha256(
            json.dumps(
                audit_source,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }

    joined = join_audit_source_evaluations([public_row], [audit_row])

    assert joined[0]["source_evaluation"] == source_evaluation
    assert "source_evaluation" not in public_row
    broken = dict(audit_row)
    broken["source_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="audit source record hash"):
        join_audit_source_evaluations([public_row], [broken])


def test_candidate_join_fails_closed_on_hash_count_or_key_mismatch(tmp_path: Path):
    row, manifest, _ = _pipeline_fixture(tmp_path)
    candidates_dir = tmp_path / "candidates"
    candidate_summary_path = tmp_path / "candidate-summary.jsonl"
    candidate_main(
        [
            "--learning-manifest",
            str(manifest),
            "--output",
            str(candidates_dir),
            "--summary",
            str(candidate_summary_path),
        ]
    )
    summary = _strict_json(candidate_summary_path)
    for field, bad_value in (
        ("candidate_sha256", "0" * 64),
        ("component_count", 99),
        ("component_keys", ["wrong-key"]),
    ):
        corrupted = dict(summary)
        corrupted[field] = bad_value
        with pytest.raises(ValueError):
            materialize_target_record(row, corrupted)


def test_remove_candidate_persists_exact_zero_based_cue_hit(tmp_path: Path):
    row, _, _ = _pipeline_fixture(tmp_path)
    source = row["source_evaluation"]
    full_m0 = (np.asarray(nib.load(source["m0_path"]).dataobj) > 0).astype(np.uint8)
    spacing_xyz = (2.0, 3.0, 5.0)
    center_z = 2
    center_xy = np.asarray([3.0, 5.0])
    cue_coordinate = [3, 5, center_z]
    cue = np.zeros_like(full_m0)
    cue[tuple(cue_coordinate)] = 1
    geometry = dict(row["geometry"])
    geometry["crop_center_xy_voxel"] = center_xy.tolist()
    crop_kwargs = {
        "center_xy": center_xy,
        "spacing_xy": spacing_xyz[:2],
        "field_mm": geometry["crop_field_mm"],
        "output_size": geometry["output_size_px"],
        "order": 0,
    }
    visible_m0 = physical_crop_resample_2d(
        full_m0[:, :, center_z], **crop_kwargs
    ).astype(np.uint8)
    visible_cue = physical_crop_resample_2d(
        cue[:, :, center_z], **crop_kwargs
    ).astype(np.uint8)
    visible_path = Path(row["visible_npz"])
    np.savez_compressed(
        visible_path,
        visual=np.zeros((17, *visible_m0.shape), dtype=np.float32),
        m0=visible_m0,
        scribble=visible_cue,
        cue_fg=np.zeros_like(visible_cue),
        cue_bg=visible_cue,
        spacing_xy=np.asarray(geometry["output_spacing_xy"], dtype=np.float32),
    )
    row["operation"] = "REMOVE"
    row["goal"] = "REMOVE_NEW_COMPLETE"
    row["geometry"] = geometry
    row["visible_sha256"] = _sha256_file(visible_path)
    source["scribble_coordinates_xyz"] = [cue_coordinate]
    candidate = materialize_candidate_record(row)
    assert candidate["cue_hit_component_position"] == 0
    assert candidate["components"][0]["cue_overlap_voxels"] == 1
    _, _, masks, keys, cue_hit = _load_components(
        {row["episode_id"]: candidate}, row["episode_id"], {}
    )
    assert masks.shape == (2, *visible_m0.shape)
    assert keys[cue_hit] == candidate["components"][0]["component_key"]


def test_candidate_materializer_refuses_test_partition(tmp_path: Path):
    row, _, _ = _pipeline_fixture(tmp_path)
    row["partition"] = "test"
    with pytest.raises(ValueError, match="non train/val"):
        materialize_candidate_record(row)
