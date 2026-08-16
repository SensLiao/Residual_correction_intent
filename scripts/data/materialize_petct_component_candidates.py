#!/usr/bin/env python3
"""Materialize inference-visible PET/CT v3 component candidates.

The input is the learning-tensor manifest, not the earlier episode-builder
manifest. That makes the candidate masks reuse the exact frozen crop geometry
of ``visible_npz``. The script opens only the current mask, cue coordinates,
visible bundle, and geometry; it never opens GT, authorized residuals, or an
evaluation bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_components import (  # noqa: E402
    COMPONENT_DESCRIPTOR_FIELDS,
    ENUMERATION_VERSION,
    cue_hit_component_position,
    enumerate_components,
)
from data.materialize_petct_learning_tensors import (  # noqa: E402
    physical_crop_resample_2d,
)

CANDIDATE_SCHEMA_VERSION = "PETCT-COMPONENT-CANDIDATES-v1.0"
_VISIBLE_KEYS = {"visual", "m0", "scribble", "cue_fg", "cue_bg", "spacing_xy"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError("non-standard JSON numeric constant: %s" % value)


def _reject_duplicate_pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON object key: %s" % key)
        output[key] = value
    return output


def _loads_strict(payload: str, label: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid strict JSON in %s: %s" % (label, error)) from error


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = _loads_strict(line, "%s line %d" % (path, line_number))
            if not isinstance(value, dict):
                raise ValueError("JSONL row must be an object: %s line %d" % (path, line_number))
            rows.append(value)
    if not rows:
        raise ValueError("learning manifest is empty: %s" % path)
    return rows


def _write_json_exclusive(path: Path, value: Mapping[str, Any], *, jsonl: bool = False) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        if jsonl:
            raise ValueError("use _write_jsonl_exclusive for JSONL")
        json.dump(
            value,
            stream,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        stream.write("\n")


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )


def _verified_regular_file(raw: Any, expected_sha256: Any, label: str) -> Path:
    path = Path(str(raw))
    if path.is_symlink():
        raise ValueError("%s must not be a symlink: %s" % (label, path))
    path = path.resolve()
    if not path.is_file():
        raise ValueError("missing %s: %s" % (label, path))
    expected = str(expected_sha256 or "")
    actual = _sha256_file(path)
    if not expected or actual != expected:
        raise ValueError("%s sha256 mismatch: %s" % (label, path))
    return path


def _load_nifti_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required to read NIfTI masks") from None
    image = nib.load(str(path))
    array = np.asarray(image.dataobj)
    if array.ndim != 3:
        raise ValueError("state mask must be a 3D [X,Y,Z] NIfTI: %s" % path)
    binary = (array > 0).astype(np.uint8)
    spacing_xyz = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if spacing_xyz.shape != (3,) or not np.all(np.isfinite(spacing_xyz)):
        raise ValueError("state mask spacing is invalid: %s" % path)
    if np.any(spacing_xyz <= 0):
        raise ValueError("state mask spacing is non-positive: %s" % path)
    return binary, spacing_xyz


def _load_visible_bundle(row: Mapping[str, Any], episode_id: str) -> Dict[str, np.ndarray]:
    path = _verified_regular_file(
        row.get("visible_npz"), row.get("visible_sha256"), "visible_npz for %s" % episode_id
    )
    with np.load(str(path), allow_pickle=False) as bundle:
        if set(bundle.files) != _VISIBLE_KEYS:
            raise ValueError("visible_npz schema mismatch for %s" % episode_id)
        output = {name: np.asarray(bundle[name]) for name in _VISIBLE_KEYS}
    for name in ("m0", "scribble", "cue_fg", "cue_bg"):
        value = output[name]
        if value.ndim != 2 or not np.all(np.isin(value, [0, 1])):
            raise ValueError("visible %s must be a binary 2D mask for %s" % (name, episode_id))
    shapes = {output[name].shape for name in ("m0", "scribble", "cue_fg", "cue_bg")}
    if len(shapes) != 1:
        raise ValueError("visible mask shapes disagree for %s" % episode_id)
    return output


def _validated_geometry(
    row: Mapping[str, Any],
    spacing_xyz: np.ndarray,
    visible_shape: tuple[int, int],
    episode_id: str,
) -> tuple[np.ndarray, float, int]:
    geometry = row.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("learning geometry missing for %s" % episode_id)
    center_xy = np.asarray(geometry.get("crop_center_xy_voxel"), dtype=np.float64)
    source_spacing_xy = np.asarray(geometry.get("original_spacing_xy"), dtype=np.float64)
    field_mm = float(geometry.get("crop_field_mm"))
    output_size = int(geometry.get("output_size_px"))
    if center_xy.shape != (2,) or not np.all(np.isfinite(center_xy)):
        raise ValueError("invalid crop_center_xy_voxel for %s" % episode_id)
    if source_spacing_xy.shape != (2,) or not np.allclose(
        source_spacing_xy, spacing_xyz[:2], atol=1e-5, rtol=0
    ):
        raise ValueError("learning/NIfTI xy spacing mismatch for %s" % episode_id)
    if not np.isfinite(field_mm) or field_mm <= 0:
        raise ValueError("invalid crop_field_mm for %s" % episode_id)
    if output_size < 2 or visible_shape != (output_size, output_size):
        raise ValueError("visible crop/output_size mismatch for %s" % episode_id)
    if geometry.get("mask_interpolation") != "nearest":
        raise ValueError("component masks require frozen nearest interpolation")
    return center_xy, field_mm, output_size


def materialize_candidate_record(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Create one candidate record without opening any label-lane artifact."""

    episode_id = str(row.get("episode_id") or "")
    if not episode_id:
        raise ValueError("learning row lacks episode_id")
    partition = str(row.get("partition") or "")
    if partition not in ("train", "val"):
        raise ValueError("component candidates refuse non train/val row: %s" % episode_id)
    operation = str(row.get("operation") or "")
    if operation not in ("ADD", "REMOVE"):
        raise ValueError("invalid operation for %s" % episode_id)
    source = row.get("source_evaluation")
    if not isinstance(source, dict):
        raise ValueError("source_evaluation mapping missing for %s" % episode_id)
    m0_path = _verified_regular_file(
        source.get("m0_path"), source.get("m0_sha256"), "m0 for %s" % episode_id
    )
    m_sha256 = str(source["m0_sha256"])
    full_mask, spacing_xyz = _load_nifti_binary(m0_path)
    visible = _load_visible_bundle(row, episode_id)
    visible_shape = tuple(int(value) for value in visible["m0"].shape)
    center_xy, field_mm, output_size = _validated_geometry(
        row, spacing_xyz, visible_shape, episode_id
    )

    coordinates = np.asarray(source.get("scribble_coordinates_xyz"))
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) == 0:
        raise ValueError("scribble_coordinates_xyz is invalid for %s" % episode_id)
    if not np.all(np.isfinite(coordinates)) or not np.all(coordinates == np.floor(coordinates)):
        raise ValueError("scribble coordinates must be finite integers for %s" % episode_id)
    coordinates = np.unique(coordinates.astype(np.int64), axis=0)
    if np.any(coordinates < 0) or np.any(coordinates >= np.asarray(full_mask.shape)[None]):
        raise ValueError("scribble coordinate outside NIfTI volume for %s" % episode_id)
    z_values = np.unique(coordinates[:, 2])
    if len(z_values) != 1:
        raise ValueError("component contract requires one axial cue slice for %s" % episode_id)
    prompted_z = int(z_values[0])
    if not np.allclose(center_xy, coordinates[:, :2].mean(axis=0), atol=1e-6, rtol=0):
        raise ValueError("crop center disagrees with source cue centroid for %s" % episode_id)
    declared_centers = (row.get("center_z"), source.get("center_z"))
    if any(int(value) != prompted_z for value in declared_centers):
        raise ValueError("center_z disagrees with xyz cue axis 2 for %s" % episode_id)

    raw_cue_slice = np.zeros(full_mask.shape[:2], dtype=np.uint8)
    raw_cue_slice[tuple(coordinates[:, :2].T)] = 1
    expected_visible_cue = physical_crop_resample_2d(
        raw_cue_slice,
        center_xy=center_xy,
        spacing_xy=spacing_xyz[:2],
        field_mm=field_mm,
        output_size=output_size,
        order=0,
    )
    expected_visible_cue = (np.asarray(expected_visible_cue) > 0.5).astype(np.uint8)
    if not np.array_equal(expected_visible_cue, np.asarray(visible["scribble"], dtype=np.uint8)):
        raise ValueError("source cue does not reconstruct visible scribble for %s" % episode_id)

    fg_any = bool(visible["cue_fg"].any())
    bg_any = bool(visible["cue_bg"].any())
    if (operation == "ADD" and (not fg_any or bg_any)) or (
        operation == "REMOVE" and (fg_any or not bg_any)
    ):
        raise ValueError("operation disagrees with visible signed cue for %s" % episode_id)
    signed_cue = visible["cue_fg"] if operation == "ADD" else visible["cue_bg"]
    if not np.array_equal(expected_visible_cue, np.asarray(signed_cue, dtype=np.uint8)):
        raise ValueError("source cue does not reconstruct visible signed cue for %s" % episode_id)

    enumeration = enumerate_components(
        full_mask,
        episode_id=episode_id,
        m_sha256=m_sha256,
        spacing_xyz=spacing_xyz,
        cue_voxels=coordinates,
        prompted_z=prompted_z,
    )
    exact_cue_overlap = sum(
        component.cue_overlap_voxels for component in enumeration.components
    )
    cue_hit_position = None
    if operation == "ADD" and exact_cue_overlap != 0:
        raise ValueError("ADD cue must lie outside current M for %s" % episode_id)
    if operation == "REMOVE":
        cue_hit_position = cue_hit_component_position(
            enumeration, coordinates, full_mask
        )
        if cue_hit_position is None or exact_cue_overlap == 0:
            raise ValueError("REMOVE cue must hit true current-M membership for %s" % episode_id)

    components = []
    central_masks = []
    for component in enumeration.components:
        if component.prompted_slice_mask is None:
            raise RuntimeError("enumeration omitted prompted slice mask")
        central_mask = physical_crop_resample_2d(
            component.prompted_slice_mask,
            center_xy=center_xy,
            spacing_xy=spacing_xyz[:2],
            field_mm=field_mm,
            output_size=output_size,
            order=0,
        )
        central_mask = (np.asarray(central_mask) > 0.5).astype(np.uint8)
        if central_mask.shape != visible_shape:
            raise RuntimeError("component crop shape mismatch for %s" % episode_id)
        component_record = component.as_dict()
        # ``candidate_position`` is the explicit producer/consumer field;
        # ``position`` remains the component-enumeration vocabulary.
        component_record["candidate_position"] = int(component.position)
        component_record["descriptor_vector"] = component.descriptor_vector()
        component_record["prompted_slice_mask"] = central_mask.tolist()
        components.append(component_record)
        central_masks.append(central_mask)

    visible_m0 = np.asarray(visible["m0"], dtype=np.uint8)
    union = (
        np.logical_or.reduce(central_masks).astype(np.uint8)
        if central_masks
        else np.zeros(visible_shape, dtype=np.uint8)
    )
    if not np.array_equal(union, visible_m0):
        raise ValueError(
            "component masks do not reconstruct visible crop under frozen geometry for %s"
            % episode_id
        )
    component_keys = [str(component["component_key"]) for component in components]
    geometry_binding = {
        "visible_sha256": str(row["visible_sha256"]),
        "crop_center_xy_voxel": center_xy.tolist(),
        "crop_field_mm": field_mm,
        "output_size_px": output_size,
        "original_spacing_xy": spacing_xyz[:2].tolist(),
        "mask_interpolation": "nearest",
    }
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "partition": partition,
        "operation": operation,
        "m_sha256": m_sha256,
        "enumeration_version": ENUMERATION_VERSION,
        "enumeration_key": enumeration.key(),
        "axis_order": "xyz",
        "axial_axis": 2,
        "prompted_z": prompted_z,
        "cue_coordinate_count": int(len(coordinates)),
        "cue_hit_component_position": cue_hit_position,
        "descriptor_order": list(COMPONENT_DESCRIPTOR_FIELDS),
        "component_count": len(components),
        "component_keys_sha256": _sha256_json(component_keys),
        "central_masks_available": True,
        "central_mask_space": "visible_crop",
        "central_mask_shape_khw": [len(components), *visible_shape],
        "geometry_binding": geometry_binding,
        "geometry_binding_sha256": _sha256_json(geometry_binding),
        "components": components,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        "--learning-manifest",
        dest="episodes",
        type=Path,
        required=True,
        help="learning tensor manifest (jsonl)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.episodes.is_file():
        parser.error("missing learning manifest: %s" % args.episodes)
    if os.path.lexists(str(args.output)):
        parser.error("output already exists: %s" % args.output)
    if os.path.lexists(str(args.summary)):
        parser.error("summary already exists: %s" % args.summary)

    try:
        rows = _load_jsonl(args.episodes)
        episode_ids = [str(row.get("episode_id") or "") for row in rows]
        if any(not value for value in episode_ids) or len(set(episode_ids)) != len(episode_ids):
            raise ValueError("learning manifest episode_id values must be non-empty and unique")
        args.output.mkdir(parents=True, exist_ok=False)
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        for row in rows:
            record = materialize_candidate_record(row)
            episode_id = str(record["episode_id"])
            output_path = (args.output / (episode_id + ".json")).resolve()
            _write_json_exclusive(output_path, record)
            component_keys = [
                str(component["component_key"]) for component in record["components"]
            ]
            summary_rows.append(
                {
                    "schema_version": CANDIDATE_SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "partition": str(record["partition"]),
                    "operation": str(record["operation"]),
                    "matched_state_group_id": str(row.get("matched_state_group_id") or ""),
                    "candidate_path": str(output_path),
                    "candidate_sha256": _sha256_file(output_path),
                    "m_sha256": str(record["m_sha256"]),
                    "enumeration_version": str(record["enumeration_version"]),
                    "component_count": int(record["component_count"]),
                    "component_keys": component_keys,
                    "component_keys_sha256": str(record["component_keys_sha256"]),
                    "descriptor_order": list(COMPONENT_DESCRIPTOR_FIELDS),
                    "geometry_binding_sha256": str(record["geometry_binding_sha256"]),
                }
            )
        _write_jsonl_exclusive(args.summary, summary_rows)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "episodes": len(summary_rows),
                "output": str(args.output.resolve()),
                "summary": str(args.summary.resolve()),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
