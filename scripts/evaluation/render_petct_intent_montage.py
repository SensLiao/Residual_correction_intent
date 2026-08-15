#!/usr/bin/env python3
"""Render a deterministic GT-free 2.5D PET/CT intent-probe montage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


SLICE_OFFSETS = (-2, -1, 0, 1, 2)
HEADER_HEIGHT = 32
FOOTER_HEIGHT = 24
M0_COLOR = (0, 255, 255)
SCRIBBLE_COLOR = (255, 0, 255)
OPAQUE_EPISODE_PATTERN = re.compile(r"ep-[0-9a-f]{6,64}")


class MontageContractError(RuntimeError):
    """Raised when a montage would violate the blind-input contract."""


def _binary_mask(mask: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 3:
        raise MontageContractError(f"{name} must be a 3D mask")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise MontageContractError(f"{name} must be binary")
    if not np.all(np.isfinite(array)) or not np.all(np.isin(array, [0, 1])):
        raise MontageContractError(f"{name} must be binary with finite values 0/1")
    return array.astype(np.bool_, copy=False)


def _numeric_volume(volume: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(volume)
    if array.ndim != 3:
        raise MontageContractError(f"{name} must be a 3D volume")
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise MontageContractError(f"{name} must contain finite numeric voxels")
    return array.astype(np.float32, copy=False)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"shape": list(contiguous.shape), "dtype": str(contiguous.dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + contiguous.tobytes()).hexdigest()


def _resize_gray(array: np.ndarray, tile_size: int) -> np.ndarray:
    values = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    resampling = getattr(Image, "Resampling", Image)
    image = Image.fromarray(values, mode="L").resize(
        (tile_size, tile_size), resample=resampling.BILINEAR
    )
    return np.asarray(image)


def _resize_mask(mask: np.ndarray, tile_size: int) -> np.ndarray:
    values = mask.astype(np.uint8) * 255
    resampling = getattr(Image, "Resampling", Image)
    image = Image.fromarray(values, mode="L").resize(
        (tile_size, tile_size), resample=resampling.NEAREST
    )
    return np.asarray(image) > 0


def _orient_axial(array: np.ndarray) -> np.ndarray:
    return np.rot90(array, k=1)


def _base_rgb(row: str, pet: np.ndarray, ct: np.ndarray) -> np.ndarray:
    if row == "PET":
        gray = np.clip(pet * 255.0, 0, 255).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)
    if row == "CT":
        gray = np.clip(ct * 255.0, 0, 255).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)
    ct_rgb = np.repeat(np.clip(ct * 255.0, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2).astype(np.float32)
    heat = np.zeros_like(ct_rgb)
    heat[:, :, 0] = 255.0 * pet
    heat[:, :, 1] = 160.0 * np.sqrt(np.clip(pet, 0, 1))
    alpha = (0.60 * np.clip(pet, 0, 1))[:, :, None]
    return np.clip(ct_rgb * (1.0 - alpha) + heat * alpha, 0, 255).astype(np.uint8)


def _render_tile(
    *,
    row: str,
    pet_slice: np.ndarray,
    ct_slice: np.ndarray,
    m0_slice: np.ndarray,
    scribble_slice: np.ndarray,
    tile_size: int,
    offset: int,
    padded: bool,
) -> Image.Image:
    pet_resized = _resize_gray(_orient_axial(pet_slice), tile_size) / 255.0
    ct_resized = _resize_gray(_orient_axial(ct_slice), tile_size) / 255.0
    rgb = _base_rgb(row, pet_resized, ct_resized)

    m0_oriented = _orient_axial(m0_slice)
    if np.any(m0_oriented):
        eroded = ndimage.binary_erosion(
            m0_oriented, structure=np.ones((3, 3), dtype=bool), border_value=0
        )
        contour = m0_oriented & ~eroded
    else:
        contour = m0_oriented
    scribble_display = ndimage.binary_dilation(
        _orient_axial(scribble_slice),
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    )
    rgb[_resize_mask(contour, tile_size)] = M0_COLOR
    rgb[_resize_mask(scribble_display, tile_size)] = SCRIBBLE_COLOR

    tile = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    label = f"{row} z{offset:+d}" + (" PAD" if padded else "")
    text_width = draw.textbbox((0, 0), label, font=font)[2]
    draw.rectangle((0, 0, text_width + 4, 12), fill=(0, 0, 0))
    draw.text((2, 1), label, fill=(255, 255, 255), font=font)
    return tile


def render_petct_intent_montage(
    pet: np.ndarray,
    ct: np.ndarray,
    m0: np.ndarray,
    scribble: np.ndarray,
    *,
    episode_id: str,
    output_path: Path,
    tile_size: int = 192,
    ct_window_hu: tuple[float, float] = (-150.0, 250.0),
    pet_upper_percentile: float = 99.5,
) -> dict[str, Any]:
    """Render a five-slice PET/CT/M0/FG packet without any GT-derived layer."""
    pet_array = _numeric_volume(pet, name="PET")
    ct_array = _numeric_volume(ct, name="CT")
    m0_mask = _binary_mask(m0, name="M0")
    scribble_mask = _binary_mask(scribble, name="scribble")
    shapes = {pet_array.shape, ct_array.shape, m0_mask.shape, scribble_mask.shape}
    if len(shapes) != 1:
        raise MontageContractError(
            f"PET/CT/M0/scribble shape mismatch: {sorted(shapes)}"
        )
    if not np.any(scribble_mask):
        raise MontageContractError("scribble must be non-empty")
    scribble_slices = np.flatnonzero(np.any(scribble_mask, axis=(0, 1)))
    if len(scribble_slices) != 1:
        raise MontageContractError("scribble must occupy a single axial slice")
    if np.any(scribble_mask & m0_mask):
        raise MontageContractError("positive FN scribble must remain outside current M0")
    if not OPAQUE_EPISODE_PATTERN.fullmatch(episode_id):
        raise MontageContractError(
            "episode_id must be an opaque ep-<hex> token, not a source identifier"
        )
    if tile_size < 32:
        raise MontageContractError("tile_size must be at least 32 pixels")
    ct_low, ct_high = (float(ct_window_hu[0]), float(ct_window_hu[1]))
    if not ct_low < ct_high:
        raise MontageContractError("invalid CT window")
    if not 90.0 <= float(pet_upper_percentile) <= 100.0:
        raise MontageContractError("PET upper percentile must be in [90, 100]")

    output_path = Path(output_path).resolve()
    if output_path.stem != episode_id or output_path.suffix.casefold() != ".png":
        raise MontageContractError("output filename must equal the opaque episode_id")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing montage: {output_path}")

    center_slice = int(scribble_slices[0])
    slice_indices: list[int | None] = []
    padded_offsets: list[int] = []
    for offset in SLICE_OFFSETS:
        index = center_slice + offset
        if 0 <= index < pet_array.shape[2]:
            slice_indices.append(index)
        else:
            slice_indices.append(None)
            padded_offsets.append(offset)

    positive_pet = pet_array[pet_array > 0]
    pet_upper = (
        float(np.percentile(positive_pet, pet_upper_percentile))
        if positive_pet.size
        else 1.0
    )
    if not np.isfinite(pet_upper) or pet_upper <= 0:
        pet_upper = 1.0
    pet_norm = np.clip(pet_array, 0.0, pet_upper) / pet_upper
    ct_norm = np.clip(ct_array, ct_low, ct_high)
    ct_norm = (ct_norm - ct_low) / (ct_high - ct_low)

    canvas = Image.new(
        "RGB",
        (len(SLICE_OFFSETS) * tile_size, HEADER_HEIGHT + 3 * tile_size + FOOTER_HEIGHT),
        color=(16, 16, 16),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    header = f"{episode_id} | PSMA PET/CT | M0 cyan | FG scribble magenta"
    draw.text((6, 10), header, fill=(255, 255, 255), font=font)

    zero_float = np.zeros(pet_array.shape[:2], dtype=np.float32)
    zero_mask = np.zeros(pet_array.shape[:2], dtype=bool)
    for column, (offset, index) in enumerate(zip(SLICE_OFFSETS, slice_indices)):
        if index is None:
            pet_slice = zero_float
            ct_slice = zero_float
            m0_slice = zero_mask
            scribble_slice = zero_mask
        else:
            pet_slice = pet_norm[:, :, index]
            ct_slice = ct_norm[:, :, index]
            m0_slice = m0_mask[:, :, index]
            scribble_slice = scribble_mask[:, :, index]
        for row_index, row_name in enumerate(("PET", "CT", "FUSION")):
            tile = _render_tile(
                row=row_name,
                pet_slice=pet_slice,
                ct_slice=ct_slice,
                m0_slice=m0_slice,
                scribble_slice=scribble_slice,
                tile_size=tile_size,
                offset=offset,
                padded=index is None,
            )
            canvas.paste(
                tile,
                (column * tile_size, HEADER_HEIGHT + row_index * tile_size),
            )
    footer_y = HEADER_HEIGHT + 3 * tile_size + 7
    draw.text(
        (6, footer_y),
        "BLIND INPUT: no GT, residual, component, target, or gold intent",
        fill=(220, 220, 220),
        font=font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("xb") as stream:
            canvas.save(stream, format="PNG", optimize=False, compress_level=9)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            output_path.unlink()
        except FileNotFoundError:
            pass
        raise
    png_bytes = output_path.read_bytes()
    return {
        "status": "COMMITTED",
        "contract_version": "PETCT-MONTAGE-v1.0",
        "episode_id": episode_id,
        "path": str(output_path),
        "sha256": hashlib.sha256(png_bytes).hexdigest(),
        "view": "axial-overview-2p5d",
        "center_slice": center_slice,
        "slice_offsets": list(SLICE_OFFSETS),
        "slice_indices": slice_indices,
        "padded_offsets": padded_offsets,
        "tile_size": tile_size,
        "image_size": list(canvas.size),
        "ct_window_hu": [ct_low, ct_high],
        "pet_scale": {
            "method": f"volume-positive-p{pet_upper_percentile:g}",
            "lower": 0.0,
            "upper": pet_upper,
        },
        "overlay": {
            "m0_contour_rgb": list(M0_COLOR),
            "foreground_scribble_rgb": list(SCRIBBLE_COLOR),
            "scribble_display_dilation_pixels": 1,
        },
        "input_hashes": {
            "pet": _array_sha256(pet_array),
            "ct": _array_sha256(ct_array),
            "m0": _array_sha256(m0_mask),
            "scribble": _array_sha256(scribble_mask),
        },
        "contains_gt": False,
        "contains_gold_intent": False,
    }


def _load_volume(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    path = Path(path)
    if path.suffix.casefold() == ".npy":
        return np.load(path), None
    try:
        import nibabel as nib
    except ImportError as exc:
        raise MontageContractError("nibabel is required for NIfTI inputs") from exc
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj), np.asarray(image.affine)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pet", type=Path, required=True)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--m0", type=Path, required=True)
    parser.add_argument("--scribble", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=192)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    arrays: dict[str, np.ndarray] = {}
    affines: dict[str, np.ndarray] = {}
    for name in ("pet", "ct", "m0", "scribble"):
        array, affine = _load_volume(getattr(args, name))
        arrays[name] = array
        if affine is not None:
            affines[name] = affine
    if affines:
        reference = next(iter(affines.values()))
        if any(
            not np.allclose(reference, affine, atol=1e-3, rtol=0)
            for affine in affines.values()
        ):
            raise MontageContractError("PET/CT/M0/scribble affine mismatch")
    receipt = render_petct_intent_montage(
        arrays["pet"],
        arrays["ct"],
        arrays["m0"],
        arrays["scribble"],
        episode_id=args.episode_id,
        output_path=args.output,
        tile_size=args.tile_size,
    )
    sys.stdout.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
