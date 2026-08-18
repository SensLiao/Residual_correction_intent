#!/usr/bin/env python3
"""Create physically separated visible/evaluation 2.5D learning tensors.

The trusted episode manifest may point to GT and authorized targets, but those
arrays are never written into the model-visible NPZ.  Inference code opens only
visible_npz; offline supervision/evaluation opens evaluation_npz separately.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Sequence, Tuple
from uuid import uuid4

import nibabel as nib
import numpy as np
from scipy import ndimage

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import (  # noqa: E402
    LearningContractError,
    load_jsonl,
    sha256_file,
    validate_manifest_rows_against_frozen_learning_split,
    validate_partitions,
)
from common.petct_route_a_core import (  # noqa: E402
    ContractError,
    intent_slots_from_goal,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _remove_owned_output(path: Path, *, directory: bool) -> None:
    if not _path_lexists(path):
        return
    if directory and path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _validate_output_layout(
    directory_outputs: Sequence[Path], file_outputs: Sequence[Path]
) -> None:
    tagged = [(Path(path).resolve(), True) for path in directory_outputs]
    tagged.extend((Path(path).resolve(), False) for path in file_outputs)
    for index, (first, first_is_dir) in enumerate(tagged):
        for second, second_is_dir in tagged[index + 1 :]:
            if first == second:
                raise RuntimeError("output paths must be distinct")
            nested = first in second.parents or second in first.parents
            if nested and (first_is_dir or second_is_dir):
                raise RuntimeError("file outputs must not be nested in output directories")


@contextmanager
def staged_output_bundle(
    *, directory_outputs: Sequence[Path], file_outputs: Sequence[Path]
) -> Iterator[dict[Path, Path]]:
    """Stage separated NPZ roots and publish the manifest as the commit marker."""

    directories = [Path(path).resolve() for path in directory_outputs]
    files = [Path(path).resolve() for path in file_outputs]
    _validate_output_layout(directories, files)
    tagged = [(path, True) for path in directories] + [(path, False) for path in files]
    for final, _ in tagged:
        final.parent.mkdir(parents=True, exist_ok=True)
        if _path_lexists(final):
            raise FileExistsError("refusing existing output path: %s" % final)
    staged = {
        final: final.with_name(
            ".%s.%d.%s.partial" % (final.name, os.getpid(), uuid4().hex)
        )
        for final, _ in tagged
    }
    if any(_path_lexists(stage) for stage in staged.values()):
        raise FileExistsError("generated staging path already exists")
    created: list[tuple[Path, bool]] = []
    try:
        for final, is_directory in tagged:
            if is_directory:
                staged[final].mkdir()
                created.append((staged[final], True))
    except Exception:
        for path, is_directory in reversed(created):
            _remove_owned_output(path, directory=is_directory)
        raise
    try:
        yield staged
    except Exception:
        for final, is_directory in tagged:
            _remove_owned_output(staged[final], directory=is_directory)
        raise
    committed: list[tuple[Path, bool]] = []
    try:
        for final, is_directory in tagged:
            if _path_lexists(final):
                raise FileExistsError("output appeared during staging: %s" % final)
            os.rename(staged[final], final)
            committed.append((final, is_directory))
    except Exception:
        for final, is_directory in tagged:
            _remove_owned_output(staged[final], directory=is_directory)
        for final, is_directory in reversed(committed):
            _remove_owned_output(final, directory=is_directory)
        raise


def normalize_ct(volume: np.ndarray) -> np.ndarray:
    value = np.asarray(volume, dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError("CT contains NaN or Inf")
    return (np.clip(value, -1000.0, 1000.0) / 1000.0).astype(np.float32)


def normalize_pet(volume: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    value = np.asarray(volume, dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError("PET contains NaN or Inf")
    transformed = np.log1p(np.maximum(value, 0.0))
    foreground = transformed[transformed > 0]
    if foreground.size:
        median = float(np.median(foreground))
        q25, q75 = np.percentile(foreground, [25, 75])
        scale = max(float(q75 - q25), 1e-6)
    else:
        median, scale = 0.0, 1.0
    normalized = (transformed - median) / scale
    return normalized.astype(np.float32), {"log_median": median, "log_iqr": scale}


def axial_stack(volume: np.ndarray, center_z: int, radius: int = 2) -> np.ndarray:
    if volume.ndim != 3 or not 0 <= center_z < volume.shape[2]:
        raise ValueError("invalid 3D volume or center_z")
    indices = [min(max(center_z + offset, 0), volume.shape[2] - 1) for offset in range(-radius, radius + 1)]
    return np.stack([volume[:, :, index] for index in indices], axis=0)


def scribble_mask(shape: Sequence[int], coordinates: Sequence[Sequence[int]]) -> np.ndarray:
    output = np.zeros(tuple(shape), dtype=np.uint8)
    if not coordinates:
        raise ValueError("scribble coordinates are empty")
    for raw in coordinates:
        if len(raw) != 3:
            raise ValueError("scribble coordinate must be xyz")
        coord = tuple(int(value) for value in raw)
        if any(value < 0 or value >= shape[axis] for axis, value in enumerate(coord)):
            raise ValueError("scribble coordinate is out of bounds")
        output[coord] = 1
    return output


def physical_crop_resample_2d(
    array: np.ndarray,
    *,
    center_xy: Sequence[float],
    spacing_xy: Sequence[float],
    field_mm: float,
    output_size: int,
    order: int,
) -> np.ndarray:
    """Sample a fixed physical square around the scribble into one fixed grid."""

    value = np.asarray(array)
    squeeze = value.ndim == 2
    if squeeze:
        value = value[None]
    if value.ndim != 3:
        raise ValueError("crop input must be [C,H,W] or [H,W]")
    spacing = tuple(float(v) for v in spacing_xy)
    if len(spacing) != 2 or any(v <= 0 for v in spacing):
        raise ValueError("spacing_xy must contain two positive values")
    if field_mm <= 0 or output_size < 2 or order not in (0, 1):
        raise ValueError("invalid crop field, output size or interpolation order")
    offsets_mm = ((np.arange(output_size) + 0.5) / output_size - 0.5) * field_mm
    x = float(center_xy[0]) + offsets_mm / spacing[0]
    y = float(center_xy[1]) + offsets_mm / spacing[1]
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    output = np.stack(
        [
            ndimage.map_coordinates(
                channel,
                [grid_x, grid_y],
                order=order,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            for channel in value
        ],
        axis=0,
    )
    return output[0] if squeeze else output


def _load_aligned(paths: Sequence[Path]):
    images = [nib.load(str(path)) for path in paths]
    reference = images[0]
    for image in images[1:]:
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-3, rtol=0):
            raise RuntimeError("episode NIfTI geometry mismatch")
    return images


def _verified_source_path(row, path_key: str, hash_key: str) -> Path:
    raw = Path(str(row[path_key]))
    if raw.is_symlink():
        raise RuntimeError("source artifact must be a non-symlink: %s" % path_key)
    path = raw.resolve()
    if not path.is_file():
        raise RuntimeError("missing regular source artifact: %s" % path_key)
    if sha256_file(path) != row.get(hash_key):
        raise RuntimeError("source artifact hash mismatch: %s" % path_key)
    return path


def materialize_episode(
    row,
    *,
    staged_visible_root: Path,
    staged_evaluation_root: Path,
    final_visible_root: Path,
    final_evaluation_root: Path,
    field_mm: float,
    output_size: int,
    expected_spacing: float,
    config_sha256: str,
):
    if row.get("experiment_config_sha256") != config_sha256:
        raise RuntimeError("episode manifest was built with a different config")
    # Natural single-round rows have no matched-state grouping; the R13
    # identity attaches episode_family_id downstream in the program-manifest
    # materializer.  Controlled-lane rows still carry the field through.
    matched_state_group_id = str(row.get("matched_state_group_id") or "").strip()
    goal = str(row.get("goal") or "")
    try:
        operation, target, scope = intent_slots_from_goal(goal)
    except ContractError as exc:
        raise RuntimeError("episode has unsupported six-class goal") from exc
    if row.get("operation") != operation:
        raise RuntimeError("episode operation differs from canonical goal")
    paths = [
        _verified_source_path(row, path_key, hash_key)
        for path_key, hash_key in (
            ("ct_path", "ct_sha256"),
            ("pet_path", "pet_sha256"),
            ("m0_path", "m0_sha256"),
            ("gt_path", "gt_sha256"),
            ("authorized_path", "authorized_sha256"),
        )
    ]
    for document_key, hash_key in (
        ("visible_document", "visible_document_sha256"),
        ("evaluation_document", "evaluation_document_sha256"),
    ):
        _verified_source_path(row, document_key, hash_key)
    ct_image, pet_image, m0_image, gt_image, authorized_image = _load_aligned(paths)
    ct = normalize_ct(np.asarray(ct_image.dataobj))
    pet, pet_stats = normalize_pet(np.asarray(pet_image.dataobj))
    m0 = (np.asarray(m0_image.dataobj) > 0).astype(np.uint8)
    gt = (np.asarray(gt_image.dataobj) > 0).astype(np.uint8)
    authorized = (np.asarray(authorized_image.dataobj) > 0).astype(np.uint8)
    legal_residual = (gt > 0) & ~(m0 > 0) if operation == "ADD" else (m0 > 0) & ~(gt > 0)
    if not authorized.any() or np.any((authorized > 0) & ~legal_residual):
        raise RuntimeError(
            "authorized target must be a non-empty subset of the operation residual"
        )
    scribble = scribble_mask(gt.shape, row["coordinates_xyz"])
    if np.any(scribble & ~authorized):
        raise RuntimeError("scribble lies outside authorized target")
    z_values = {int(coord[2]) for coord in row["coordinates_xyz"]}
    if len(z_values) != 1:
        raise RuntimeError("current 2.5D contract requires one axial scribble slice")
    center_z = next(iter(z_values))
    cue_fg = scribble if operation == "ADD" else np.zeros_like(scribble)
    cue_bg = scribble if operation == "REMOVE" else np.zeros_like(scribble)
    raw_visual = np.concatenate(
        [
            axial_stack(pet, center_z),
            axial_stack(ct, center_z),
            axial_stack(m0, center_z),
            cue_fg[:, :, center_z][None],
            cue_bg[:, :, center_z][None],
        ],
        axis=0,
    ).astype(np.float32)
    center_xy = np.mean(
        np.asarray([[coord[0], coord[1]] for coord in row["coordinates_xyz"]]),
        axis=0,
    )
    original_spacing_xy = np.asarray(
        ct_image.header.get_zooms()[:2], dtype=np.float32
    )
    visual = np.concatenate(
        [
            physical_crop_resample_2d(
                raw_visual[:10],
                center_xy=center_xy,
                spacing_xy=original_spacing_xy,
                field_mm=field_mm,
                output_size=output_size,
                order=1,
            ),
            physical_crop_resample_2d(
                raw_visual[10:],
                center_xy=center_xy,
                spacing_xy=original_spacing_xy,
                field_mm=field_mm,
                output_size=output_size,
                order=0,
            ),
        ],
        axis=0,
    ).astype(np.float32)
    m0_slice = physical_crop_resample_2d(
        m0[:, :, center_z],
        center_xy=center_xy,
        spacing_xy=original_spacing_xy,
        field_mm=field_mm,
        output_size=output_size,
        order=0,
    ).astype(np.uint8)
    scribble_slice = physical_crop_resample_2d(
        scribble[:, :, center_z],
        center_xy=center_xy,
        spacing_xy=original_spacing_xy,
        field_mm=field_mm,
        output_size=output_size,
        order=0,
    ).astype(np.uint8)
    cue_fg_slice = scribble_slice if operation == "ADD" else np.zeros_like(scribble_slice)
    cue_bg_slice = scribble_slice if operation == "REMOVE" else np.zeros_like(scribble_slice)
    target_slice = physical_crop_resample_2d(
        authorized[:, :, center_z],
        center_xy=center_xy,
        spacing_xy=original_spacing_xy,
        field_mm=field_mm,
        output_size=output_size,
        order=0,
    ).astype(np.uint8)
    gt_slice = physical_crop_resample_2d(
        gt[:, :, center_z],
        center_xy=center_xy,
        spacing_xy=original_spacing_xy,
        field_mm=field_mm,
        output_size=output_size,
        order=0,
    ).astype(np.uint8)
    if not scribble_slice.any() or np.any(scribble_slice & ~target_slice):
        raise RuntimeError("resampling lost the scribble or moved it outside target")
    source_authorized_slice = authorized[:, :, center_z]
    source_coordinates = np.argwhere(source_authorized_slice > 0)
    crop_limit = field_mm / 2.0 - expected_spacing / 2.0
    source_offsets = np.abs(
        (source_coordinates - center_xy[None]) * original_spacing_xy[None]
    )
    if not len(source_coordinates) or np.any(source_offsets > crop_limit + 1e-6):
        raise RuntimeError("authorized COMPLETE/LOCAL target exceeds frozen crop")
    episode_id = str(row["episode_id"])
    visible_staged_path = staged_visible_root / (episode_id + ".npz")
    evaluation_staged_path = staged_evaluation_root / (episode_id + ".npz")
    visible_final_path = final_visible_root / visible_staged_path.name
    evaluation_final_path = final_evaluation_root / evaluation_staged_path.name
    spacing_xy = np.asarray(
        [expected_spacing, expected_spacing], dtype=np.float32
    )
    np.savez_compressed(
        str(visible_staged_path),
        visual=visual,
        m0=m0_slice,
        scribble=scribble_slice,
        cue_fg=cue_fg_slice,
        cue_bg=cue_bg_slice,
        spacing_xy=spacing_xy,
    )
    np.savez_compressed(
        str(evaluation_staged_path),
        target=target_slice,
        authorized=target_slice,
        gt=gt_slice,
    )
    provenance = {
        key: row[key]
        for key in (
            "scribble_generation",
            "official_source_provenance",
            "strategy_mode",
            "strategy_salt",
            "strategy_assignment",
            "seed",
            "m0_provenance",
            "residual_kind",
            "residual_path",
            "residual_sha256",
            "residual_voxels",
            "residual_mask_sha256",
            "fn_path",
            "fn_sha256",
            "fp_path",
            "fp_sha256",
            "residual_contract",
        )
        if key in row
    }
    materialized = {
        "case_id": str(row["case_id"]),
        "episode_id": episode_id,
        "patient_id": str(row["patient_id"]),
        "partition": str(row["partition"]),
        "held_out_fold": int(row["held_out_fold"]),
        "strategy": str(row["strategy"]),
        "goal": goal,
        "operation": operation,
        "target": target,
        "scope": scope,
        "visible_npz": str(visible_final_path),
        "visible_sha256": sha256_file(visible_staged_path),
        "evaluation_npz": str(evaluation_final_path),
        "evaluation_sha256": sha256_file(evaluation_staged_path),
        "center_z": center_z,
        "normalization": {
            "ct": "clip[-1000,1000]/1000",
            "pet": "log1p+positive-voxel-median/IQR",
            **pet_stats,
        },
        "geometry": {
            "crop_center_xy_voxel": center_xy.tolist(),
            "crop_field_mm": field_mm,
            "output_size_px": output_size,
            "original_spacing_xy": original_spacing_xy.tolist(),
            "output_spacing_xy": spacing_xy.tolist(),
            "image_interpolation": "linear",
            "mask_interpolation": "nearest",
        },
        "experiment_config_sha256": config_sha256,
        "learning_split_sha256": str(row["learning_split_sha256"]),
        "test_access_receipt_sha256": row.get("test_access_receipt_sha256"),
        "source_evaluation": {
            "gt_path": str(paths[3]),
            "gt_sha256": str(row["gt_sha256"]),
            "m0_path": str(paths[2]),
            "m0_sha256": str(row["m0_sha256"]),
            "authorized_path": str(paths[4]),
            "authorized_sha256": str(row["authorized_sha256"]),
            "center_z": center_z,
            "scribble_coordinates_xyz": row["coordinates_xyz"],
        },
        **provenance,
    }
    if matched_state_group_id:
        materialized["matched_state_group_id"] = matched_state_group_id
    return materialized


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--visible-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("train", "val", "test"),
        required=True,
        help="explicit frozen learning partitions present in the episode manifest",
    )
    add_leaf_test_access_arguments(parser)
    args = parser.parse_args(argv)
    if len(args.partitions) != len(set(args.partitions)):
        parser.error("--partitions must not contain duplicates")
    roots = [path.resolve() for path in (args.visible_root, args.evaluation_root)]
    output_manifest = args.output_manifest.resolve()
    selected_partitions = set(args.partitions)
    # Receipt validation is intentionally first: do not inspect the episode
    # manifest, source documents, NIfTI truth, or output locations beforehand.
    try:
        test_access = enforce_partition_access(
            selected_partitions,
            receipt_path=args.test_access_receipt,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=args.run_root,
            output_paths=(*roots, output_manifest),
        )
    except TestAccessError as error:
        parser.error(str(error))
    test_access_sha256 = (
        None if test_access is None else str(test_access["receipt_sha256"])
    )
    with args.experiment_config.open("r", encoding="utf-8") as stream:
        experiment_config = json.load(stream)
    tensor_config = experiment_config["learning_tensor_normalization"]
    field_mm = float(tensor_config["crop_field_mm"])
    output_size = int(tensor_config["output_size_px"])
    expected_spacing = float(tensor_config["output_spacing_mm"])
    if not np.isclose(field_mm / output_size, expected_spacing, atol=1e-9, rtol=0):
        parser.error("output spacing does not equal crop_field_mm/output_size_px")
    config_sha256 = sha256_file(args.experiment_config)
    if roots[0] == roots[1] or roots[0] in roots[1].parents or roots[1] in roots[0].parents:
        parser.error("visible and evaluation roots must be physically disjoint")
    if any(_path_lexists(path) for path in (*roots, output_manifest)):
        parser.error("output paths must not already exist")
    source_rows = load_jsonl(args.episode_manifest)
    try:
        validate_manifest_rows_against_frozen_learning_split(
            source_rows,
            args.learning_split,
            require_episode_id=True,
            allowed_partitions=selected_partitions,
        )
    except LearningContractError as error:
        parser.error(str(error))
    for row_number, row in enumerate(source_rows, start=1):
        expected_receipt = (
            test_access_sha256 if row.get("partition") == "test" else None
        )
        if row.get("test_access_receipt_sha256") != expected_receipt:
            parser.error(
                "episode row %d test-access receipt provenance mismatch" % row_number
            )
    output_rows = []
    with staged_output_bundle(
        directory_outputs=roots, file_outputs=[output_manifest]
    ) as staged:
        for row in source_rows:
            output_rows.append(
                materialize_episode(
                    row,
                    staged_visible_root=staged[roots[0]],
                    staged_evaluation_root=staged[roots[1]],
                    final_visible_root=roots[0],
                    final_evaluation_root=roots[1],
                    field_mm=field_mm,
                    output_size=output_size,
                    expected_spacing=expected_spacing,
                    config_sha256=config_sha256,
                )
            )
        validate_partitions(output_rows)
        with staged[output_manifest].open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for row in output_rows:
                stream.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
    print(json.dumps({"episodes": len(output_rows), "visible_root": str(args.visible_root), "evaluation_root": str(args.evaluation_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
