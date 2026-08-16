#!/usr/bin/env python3
"""Render presentation-ready PET/CT Editor validation figures.

The figure contract is deliberately strict:

* every intent is represented by one validation case with a separately
  measured one-round Editor prediction;
* GT, M0, and Editor M1 use the same filled-mask colour;
* only additions (green), removals (red), and the white scribble are added;
* the displayed Dice is computed over the whole native axial slice, even
  though the image panel is zoomed to a common region of interest;
* the two Editor panels show the exact same prediction; only the right panel
  adds the scribble overlay, so it is not a no-scribble model ablation;
* the five-round teacher-forced ceiling is kept on a separate cohort figure.

The output is explanatory validation evidence. It is not locked-test evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


PAGE_BG = (10, 14, 20)
CARD = (20, 27, 36)
CARD_EDGE = (49, 61, 75)
INK = (246, 249, 252)
MUTED = (166, 177, 190)
MASK = (0, 174, 214)
ADD = (0, 196, 122)
REMOVE = (239, 72, 72)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

GOAL_TEXT = {
    "ADD_SAME_LOCAL": "ADD · SAME LESION · LOCAL",
    "REMOVE_SAME_LOCAL": "REMOVE · SAME COMPONENT · LOCAL",
    "ADD_SAME_COMPLETE": "ADD · SAME LESION · COMPLETE",
    "REMOVE_SAME_COMPLETE": "REMOVE · SAME COMPONENT · COMPLETE",
    "ADD_NEW_COMPLETE": "ADD · NEW LESION · COMPLETE",
    "REMOVE_NEW_COMPLETE": "REMOVE · NEW FALSE POSITIVE · COMPLETE",
}

# Presentation examples are fixed so reruns cannot silently cherry-pick a new
# case as metrics change. Each row was checked against the five-round corpus,
# the one-round manifest, the Editor prediction, and the visible/eval packets.
PREFERRED_CASES = {
    "ADD_SAME_LOCAL": (
        "psma_b361fb9c2455deb9_2018-11-19",
        "petct-df89539d4a495c3cfc8ba76a",
    ),
    "REMOVE_SAME_LOCAL": (
        "psma_ecbe2f11374632fa_2020-09-26",
        "petct-7068850a91bd1bdb29ce31bd",
    ),
    "ADD_SAME_COMPLETE": (
        "psma_55a7e47e7d24b12c_2015-01-04",
        "petct-2ddf33372b40527c821fb72b",
    ),
    "REMOVE_SAME_COMPLETE": (
        "psma_3959d1c381a5bcd6_2015-11-12",
        "petct-fddc32807055263fcac78549",
    ),
    "ADD_NEW_COMPLETE": (
        "psma_fad59ccacf4b88f2_2022-03-26",
        "petct-85eaf44de04d80666c199cd5",
    ),
    "REMOVE_NEW_COMPLETE": (
        "psma_3f391a1e890184b2_2020-05-09",
        "petct-7210b79329d599c36cd9ede7",
    ),
}

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
)
FONT_BOLD_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


class FigureContractError(RuntimeError):
    """Raised when source evidence cannot support the intended figure."""


@dataclass
class CaseData:
    case_id: str
    episode_id: str
    goal: str
    operation: str
    center_z: int
    round_centers: tuple[int, ...]
    pet: np.ndarray
    gt: np.ndarray
    m0: np.ndarray
    m1: np.ndarray
    oracle5: np.ndarray
    scribble: np.ndarray
    scribble_points: tuple[tuple[int, int], ...]
    editor_added: np.ndarray
    editor_removed: np.ndarray
    oracle_added: np.ndarray
    oracle_removed: np.ndarray
    crop_box: tuple[int, int, int, int]
    dice_m0: float
    dice_m1: float
    dice_oracle5: float
    target_recovery: float
    target_voxels: int
    changed_voxels: int
    full3d_dice_m0: float
    full3d_dice_oracle5: float


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = FONT_BOLD_CANDIDATES if bold else FONT_CANDIDATES
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _binary(array: np.ndarray, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    values = np.asarray(array)
    expected = shape if shape is not None else (256, 256)
    if values.shape != expected:
        raise FigureContractError(f"{name} must be {expected}, got {values.shape}")
    if not np.all(np.isfinite(values)) or not np.all(np.isin(values, (0, 1))):
        raise FigureContractError(f"{name} must be finite binary data")
    return values.astype(bool, copy=False)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    denom = float(aa.sum() + bb.sum())
    # Presentation convention: empty prediction and empty GT are a perfect
    # empty match. REMOVE_NEW_COMPLETE is additionally labelled in the manifest.
    return 2.0 * float(np.logical_and(aa, bb).sum()) / denom if denom else 1.0


def _inverse_crop_binary(
    crop: np.ndarray,
    *,
    output_shape: tuple[int, int],
    center_xy: Iterable[float],
    original_spacing_xy: Iterable[float],
    field_mm: float,
) -> np.ndarray:
    """Map a 2-D prediction crop back to its native axial slice."""
    value = np.asarray(crop)
    if value.ndim != 2:
        raise FigureContractError("prediction crop must be 2-D")
    x, y = np.meshgrid(
        np.arange(output_shape[0], dtype=float),
        np.arange(output_shape[1], dtype=float),
        indexing="ij",
    )
    spacing = np.asarray(tuple(original_spacing_xy), dtype=float)
    center = np.asarray(tuple(center_xy), dtype=float)
    crop_x = ((x - center[0]) * spacing[0] / field_mm + 0.5) * value.shape[0] - 0.5
    crop_y = ((y - center[1]) * spacing[1] / field_mm + 0.5) * value.shape[1] - 0.5
    return (
        ndimage.map_coordinates(
            value.astype(np.float32),
            [crop_x, crop_y],
            order=0,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        > 0.5
    )


def _normalize_pet(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if not np.any(np.isfinite(values)):
        raise FigureContractError("PET slice contains no finite values")
    values = np.log1p(np.clip(np.nan_to_num(values, nan=0.0), 0.0, None))
    positive = values[values > 0]
    source = positive if positive.size >= 16 else values[np.isfinite(values)]
    lo, hi = np.percentile(source, (1.0, 99.7))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(source.min()), float(source.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    return (np.clip((values - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _roi_box(masks: Iterable[np.ndarray], *, min_side: int = 64, padding: int = 14) -> tuple[int, int, int, int]:
    masks = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise FigureContractError("ROI masks do not share a native slice shape")
    union = np.logical_or.reduce(masks)
    points = np.argwhere(union)
    if not len(points):
        side = min(min_side, min(shape))
        cx, cy = shape[0] // 2, shape[1] // 2
    else:
        low, high = points.min(axis=0), points.max(axis=0) + 1
        side = max(int(high[0] - low[0]), int(high[1] - low[1])) + 2 * padding
        side = max(min_side, side)
        side = min(side, min(shape))
        cx = int(round((low[0] + high[0]) / 2))
        cy = int(round((low[1] + high[1]) / 2))
    x0 = max(0, min(shape[0] - side, cx - side // 2))
    y0 = max(0, min(shape[1] - side, cy - side // 2))
    return x0, x0 + side, y0, y0 + side


def _target_recovery(changed: np.ndarray, authorized: np.ndarray) -> tuple[float, int, int]:
    changed = np.asarray(changed, dtype=bool)
    authorized = np.asarray(authorized, dtype=bool)
    denominator = int(authorized.sum())
    if not denominator:
        raise FigureContractError("prompted slice has no authorized target voxels")
    corrected = int(np.logical_and(changed, authorized).sum())
    return corrected / denominator, denominator, int(changed.sum())


def _full5_rows(corpus: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(corpus):
        if row.get("partition") == "val":
            grouped[str(row["case_id"])].append(row)
    result: dict[str, list[dict[str, Any]]] = {}
    for case_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: int(row["round"]))
        if [int(row["round"]) for row in ordered] == [1, 2, 3, 4, 5]:
            result[case_id] = ordered
    return result


def _load_cases(
    corpus: Path,
    one_round_manifest: Path,
    editor_eval_dir: Path,
    eval5_root: Path,
    visible5_root: Path,
) -> list[CaseData]:
    manifests = {str(row["episode_id"]): row for row in _read_jsonl(one_round_manifest)}
    rounds = _full5_rows(corpus)
    cases: list[CaseData] = []

    for goal, (case_id, episode_id) in PREFERRED_CASES.items():
        if case_id not in rounds:
            raise FigureContractError(f"{case_id}: not a complete validation R1-R5 case")
        rows = rounds[case_id]
        first, last = rows[0], rows[-1]
        if str(first["goal"]) != goal:
            raise FigureContractError(f"{case_id}: R1 goal changed from {goal}")
        manifest = manifests.get(episode_id)
        if manifest is None or str(manifest.get("case_id")) != case_id:
            raise FigureContractError(f"{case_id}: one-round manifest is missing or mismatched")

        center_z = int(first["frozen_target_stats"]["center_z"])
        source_eval = manifest.get("source_evaluation", {})
        if int(source_eval.get("center_z", -1)) != center_z:
            raise FigureContractError(f"{case_id}: one-round and five-round center_z disagree")
        expected_coords = tuple(tuple(int(v) for v in p) for p in first["coordinates_xyz"])
        observed_coords = tuple(
            tuple(int(v) for v in p)
            for p in source_eval.get("scribble_coordinates_xyz", ())
        )
        if expected_coords != observed_coords:
            raise FigureContractError(f"{case_id}: one-round and five-round scribbles disagree")

        editor_path = editor_eval_dir / f"{episode_id}.npz"
        evaluation_path = eval5_root / f"{case_id}::r1.npz"
        visible_path = visible5_root / f"{case_id}::r1.npz"
        for path in (editor_path, evaluation_path, visible_path):
            if not path.exists():
                raise FigureContractError(f"missing required evidence: {path}")

        with np.load(editor_path) as editor, np.load(evaluation_path) as evaluation, np.load(
            visible_path
        ) as visible:
            crop_m0 = _binary(editor["m0"], name=f"{case_id} Editor M0")
            crop_m1 = _binary(editor["m1"], name=f"{case_id} Editor M1")
            crop_delta = _binary(editor["delta"], name=f"{case_id} Editor delta")
            crop_scribble = _binary(editor["scribble"], name=f"{case_id} Editor scribble")
            if not np.array_equal(crop_delta, crop_m0 ^ crop_m1):
                raise FigureContractError(f"{case_id}: stored Editor delta != M0 xor M1")
            if not np.array_equal(crop_m0, _binary(visible["m0"], name=f"{case_id} visible M0")):
                raise FigureContractError(f"{case_id}: visible and Editor M0 are not aligned")
            if not np.array_equal(
                crop_scribble, _binary(visible["scribble"], name=f"{case_id} visible scribble")
            ):
                raise FigureContractError(f"{case_id}: visible and Editor scribble are not aligned")
            _binary(evaluation["gt"], name=f"{case_id} crop GT")
            _binary(evaluation["authorized"], name=f"{case_id} crop authorized")

        m0_image = nib.load(str(first["state_path"]))
        gt_image = nib.load(str(first["gt_path"]))
        pet_image = nib.load(str(first["pet_path"]))
        authorized_image = nib.load(str(first["authorized_path"]))
        m0_volume = np.asarray(m0_image.dataobj) > 0
        gt_volume = np.asarray(gt_image.dataobj) > 0
        pet_volume = np.asarray(pet_image.dataobj)
        authorized_volume = np.asarray(authorized_image.dataobj) > 0
        if not (m0_volume.shape == gt_volume.shape == pet_volume.shape == authorized_volume.shape):
            raise FigureContractError(f"{case_id}: full-volume shapes disagree")

        geometry = manifest.get("geometry", {})
        native_delta = _inverse_crop_binary(
            crop_delta,
            output_shape=m0_volume.shape[:2],
            center_xy=geometry["crop_center_xy_voxel"],
            original_spacing_xy=geometry["original_spacing_xy"],
            field_mm=float(geometry["crop_field_mm"]),
        )
        m1_volume = m0_volume.copy()
        operation = str(first["operation"])
        if operation == "ADD":
            if np.any(crop_m0 & ~crop_m1):
                raise FigureContractError(f"{case_id}: ADD prediction removed crop pixels")
            m1_volume[:, :, center_z] |= native_delta
        elif operation == "REMOVE":
            if np.any(crop_m1 & ~crop_m0):
                raise FigureContractError(f"{case_id}: REMOVE prediction added crop pixels")
            m1_volume[:, :, center_z] &= ~native_delta
        else:
            raise FigureContractError(f"{case_id}: unknown operation {operation}")

        # The final teacher-forced state is the R5 pre-state plus the exact R5
        # authorised correction. This is the perfect-correction ceiling.
        oracle_volume = np.asarray(nib.load(str(last["state_path"])).dataobj) > 0
        last_authorized = np.asarray(nib.load(str(last["authorized_path"])).dataobj) > 0
        oracle_volume = (
            oracle_volume | last_authorized
            if str(last["operation"]) == "ADD"
            else oracle_volume & ~last_authorized
        )

        scribble = np.zeros(m0_volume.shape[:2], dtype=bool)
        for x, y, z in observed_coords:
            if z != center_z:
                raise FigureContractError(f"{case_id}: scribble left the prompted slice")
            scribble[x, y] = True

        gt = gt_volume[:, :, center_z]
        m0 = m0_volume[:, :, center_z]
        m1 = m1_volume[:, :, center_z]
        oracle5 = oracle_volume[:, :, center_z]
        editor_added = ~m0 & m1
        editor_removed = m0 & ~m1
        oracle_added = ~m0 & oracle5
        oracle_removed = m0 & ~oracle5
        recovery, target_voxels, changed_voxels = _target_recovery(
            m0 ^ m1, authorized_volume[:, :, center_z]
        )
        crop_box = _roi_box(
            (gt, m0, m1, oracle5, scribble), min_side=64, padding=16
        )

        cases.append(
            CaseData(
                case_id=case_id,
                episode_id=episode_id,
                goal=goal,
                operation=operation,
                center_z=center_z,
                round_centers=tuple(
                    int(row["frozen_target_stats"]["center_z"]) for row in rows
                ),
                pet=_normalize_pet(pet_volume[:, :, center_z]),
                gt=gt,
                m0=m0,
                m1=m1,
                oracle5=oracle5,
                scribble=scribble,
                scribble_points=tuple((x, y) for x, y, _ in observed_coords),
                editor_added=editor_added,
                editor_removed=editor_removed,
                oracle_added=oracle_added,
                oracle_removed=oracle_removed,
                crop_box=crop_box,
                dice_m0=_dice(m0, gt),
                dice_m1=_dice(m1, gt),
                dice_oracle5=_dice(oracle5, gt),
                target_recovery=recovery,
                target_voxels=target_voxels,
                changed_voxels=changed_voxels,
                full3d_dice_m0=_dice(m0_volume, gt_volume),
                full3d_dice_oracle5=_dice(oracle_volume, gt_volume),
            )
        )
    return cases


def _crop(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, x1, y0, y1 = box
    return np.asarray(array)[x0:x1, y0:y1]


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image)
    return np.asarray(
        Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").resize(
            (size, size), resample=resampling.NEAREST
        )
    ) > 0


def _panel_image(
    case: CaseData,
    mask: np.ndarray,
    *,
    added: np.ndarray | None = None,
    removed: np.ndarray | None = None,
    scribble_points: tuple[tuple[int, int], ...] | None = None,
    size: int = 820,
) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image)
    base = Image.fromarray(_crop(case.pet, case.crop_box), mode="L").resize(
        (size, size), resample=resampling.BICUBIC
    )
    rgb = np.repeat(np.asarray(base, dtype=np.uint8)[:, :, None], 3, axis=2).astype(np.float32)

    def blend(layer_mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
        if not np.any(layer_mask):
            return
        rgb[layer_mask] = rgb[layer_mask] * (1.0 - alpha) + np.asarray(color) * alpha

    blend(_resize_mask(_crop(mask, case.crop_box), size), MASK, 0.52)
    if added is not None:
        blend(_resize_mask(_crop(added, case.crop_box), size), ADD, 0.88)
    if removed is not None:
        blend(_resize_mask(_crop(removed, case.crop_box), size), REMOVE, 0.88)
    result = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    if scribble_points:
        x0, x1, y0, y1 = case.crop_box
        side_x, side_y = x1 - x0, y1 - y0
        # NIfTI array coordinates are (x, y); PIL drawing coordinates are
        # (horizontal=y, vertical=x).
        points = [
            (
                int(round((y - y0 + 0.5) / side_y * size)),
                int(round((x - x0 + 0.5) / side_x * size)),
            )
            for x, y in scribble_points
        ]
        stroke = ImageDraw.Draw(result)
        if len(points) == 1:
            x, y = points[0]
            stroke.ellipse((x - 7, y - 7, x + 7, y + 7), fill=BLACK)
            stroke.ellipse((x - 4, y - 4, x + 4, y + 4), fill=WHITE)
        else:
            stroke.line(points, fill=BLACK, width=16, joint="curve")
            stroke.line(points, fill=WHITE, width=8, joint="curve")
    return result


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, size: int, bold: bool = False, fill=INK) -> None:
    draw.text(xy, text, font=_font(size, bold=bold), fill=fill)


def _case_panels(case: CaseData) -> tuple[tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None, tuple[tuple[int, int], ...] | None], ...]:
    """Return the four case panels under the presentation contract.

    The final two panels deliberately share the same Editor prediction and
    delta. The only difference is whether the observed scribble is drawn.
    """
    return (
        ("GROUND TRUTH", case.gt, None, None, None),
        ("M0 · INITIAL", case.m0, None, None, None),
        (
            "EDITOR RESULT",
            case.m1,
            case.editor_added,
            case.editor_removed,
            None,
        ),
        (
            "EDITOR RESULT + SCRIBBLE",
            case.m1,
            case.editor_added,
            case.editor_removed,
            case.scribble_points,
        ),
    )


def _render_case(case: CaseData, index: int, output: Path) -> None:
    canvas = Image.new("RGB", (3840, 2160), PAGE_BG)
    draw = ImageDraw.Draw(canvas)
    _draw_label(draw, (120, 70), f"CASE {index:02d}  {GOAL_TEXT[case.goal]}", size=60, bold=True)
    _draw_label(
        draw,
        (122, 160),
        f"One measured validation edit · prompted slice z={case.center_z}",
        size=31,
        fill=MUTED,
    )

    # Minimal semantic legend: mask is constant; only the delta changes colour.
    legend_x = 2830
    for offset, color, text in (
        (0, MASK, "segmentation mask"),
        (310, ADD, "added"),
        (505, REMOVE, "removed"),
    ):
        x = legend_x + offset
        draw.rounded_rectangle((x, 105, x + 34, 139), radius=7, fill=color)
        _draw_label(draw, (x + 47, 102), text, size=26, fill=MUTED)

    panel_size = 820
    panel_y = 330
    xs = (120, 1060, 2000, 2940)
    panels = _case_panels(case)
    for x, (label, mask, added, removed, scribble) in zip(xs, panels):
        draw.rounded_rectangle(
            (x - 8, panel_y - 8, x + panel_size + 8, panel_y + panel_size + 8),
            radius=22,
            fill=CARD,
            outline=CARD_EDGE,
            width=3,
        )
        canvas.paste(
            _panel_image(
                case,
                mask,
                added=added,
                removed=removed,
                scribble_points=scribble,
                size=panel_size,
            ),
            (x, panel_y),
        )
        box = draw.textbbox((0, 0), label, font=_font(33, bold=True))
        _draw_label(
            draw,
            (x + (panel_size - (box[2] - box[0])) // 2, panel_y - 67),
            label,
            size=33,
            bold=True,
        )

    metric_y = 1285
    draw.rounded_rectangle(
        (120, metric_y, 3720, 1800),
        radius=34,
        fill=CARD,
        outline=CARD_EDGE,
        width=3,
    )
    _draw_label(draw, (190, metric_y + 55), "WHOLE-SLICE DICE", size=31, bold=True, fill=MUTED)
    _draw_label(
        draw,
        (190, metric_y + 112),
        f"{case.dice_m0:.3f}  →  {case.dice_m1:.3f}",
        size=70,
        bold=True,
    )
    _draw_label(
        draw,
        (195, metric_y + 210),
        "M0                         Editor R1",
        size=27,
        fill=MUTED,
    )

    divider_x = 2090
    draw.line((divider_x, metric_y + 55, divider_x, metric_y + 410), fill=CARD_EDGE, width=3)
    _draw_label(draw, (2180, metric_y + 55), "TARGET ERROR RECOVERED AFTER R1", size=31, bold=True, fill=MUTED)
    _draw_label(
        draw,
        (2180, metric_y + 112),
        f"{case.target_recovery:.1%}",
        size=82,
        bold=True,
        fill=ADD if case.target_recovery >= 0.5 else REMOVE,
    )
    _draw_label(
        draw,
        (2185, metric_y + 215),
        f"{int(round(case.target_recovery * case.target_voxels))} / {case.target_voxels} authorized target voxels",
        size=27,
        fill=MUTED,
    )

    canvas.save(output, optimize=False, compress_level=6)


def _ladder_summary(rows_path: Path) -> tuple[list[float], list[float], list[int], list[int]]:
    rows = list(_read_jsonl(rows_path))
    if not rows:
        raise FigureContractError("oracle ladder rows are empty")
    base = np.asarray([row["d0"] for row in rows], dtype=float)
    two_d, three_d = [float(np.mean(base))], [float(np.mean(base))]
    n_two, n_three = [len(base)], [len(base)]
    for round_index in range(1, 6):
        for key, values, counts in (
            (f"d2d_r{round_index}", two_d, n_two),
            (f"d3d_r{round_index}", three_d, n_three),
        ):
            observed = np.asarray(
                [row.get(key, math.nan) for row in rows if row.get(key) is not None],
                dtype=float,
            )
            observed = observed[np.isfinite(observed)]
            values.append(float(np.mean(observed)))
            counts.append(int(observed.size))
    return two_d, three_d, n_two, n_three


def _render_ladder(rows_path: Path, output: Path) -> None:
    two_d, _, n_two, _ = _ladder_summary(rows_path)
    canvas = Image.new("RGB", (3840, 2160), PAGE_BG)
    draw = ImageDraw.Draw(canvas)
    _draw_label(draw, (180, 105), "FIVE-ROUND PERFECT-CORRECTION CEILING", size=74, bold=True)
    _draw_label(
        draw,
        (185, 205),
        "Cohort mean full-volume Dice after each perfect single-slice correction",
        size=34,
        fill=MUTED,
    )

    left, top, right, bottom = 330, 430, 3470, 1640
    y_min, y_max = min(two_d) - 0.03, max(two_d) + 0.03
    y_min = math.floor(y_min * 20) / 20
    y_max = math.ceil(y_max * 20) / 20
    for value in np.linspace(y_min, y_max, 6):
        y = int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, y, right, y), fill=CARD_EDGE, width=2)
        _draw_label(draw, (185, y - 20), f"{value:.2f}", size=28, fill=MUTED)
    draw.line((left, top, left, bottom), fill=MUTED, width=4)
    draw.line((left, bottom, right, bottom), fill=MUTED, width=4)

    def point(round_index: int, value: float) -> tuple[int, int]:
        return (
            int(left + round_index / 5 * (right - left)),
            int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top)),
        )

    points = [point(index, value) for index, value in enumerate(two_d)]
    draw.line(points, fill=MASK, width=14, joint="curve")
    for index, (x, y) in enumerate(points):
        draw.ellipse((x - 18, y - 18, x + 18, y + 18), fill=MASK, outline=WHITE, width=4)
        _draw_label(draw, (x - 52, y - 83), f"{two_d[index]:.3f}", size=32, bold=True)
        _draw_label(draw, (x - 25, bottom + 45), "M0" if index == 0 else f"R{index}", size=30, bold=True)
    _draw_label(draw, (1540, 1780), "CORRECTION ROUND", size=31, bold=True, fill=MUTED)

    draw.rounded_rectangle((330, 1880, 3470, 2040), radius=28, fill=CARD, outline=CARD_EDGE, width=3)
    _draw_label(
        draw,
        (390, 1925),
        f"R5 ceiling: {two_d[-1]:.3f}    ·    observed cases: {' / '.join(map(str, n_two))}",
        size=35,
        bold=True,
    )
    _draw_label(draw, (1960, 1931), "VAL oracle · not Editor performance · not locked TEST", size=28, fill=MUTED)
    canvas.save(output, optimize=False, compress_level=6)


def _case_manifest(case: CaseData, index: int, filename: str) -> dict[str, Any]:
    return {
        "index": index,
        "filename": filename,
        "case_id": case.case_id,
        "episode_id": case.episode_id,
        "partition": "val",
        "goal": case.goal,
        "operation": case.operation,
        "round_centers_z": list(case.round_centers),
        "display_slice_z": case.center_z,
        "display_view": "zoomed ROI from the round-1 native axial PET slice",
        "display_metric": "whole native axial-slice Dice at round-1 z",
        "dice_m0": case.dice_m0,
        "dice_editor_r1": case.dice_m1,
        "dice_oracle_r5_same_slice_context_only": case.dice_oracle5,
        "target_recovery_r1_operable": case.target_recovery,
        "target_voxels_r1_operable": case.target_voxels,
        "changed_voxels_r1_native_slice": case.changed_voxels,
        "full3d_dice_m0_context": case.full3d_dice_m0,
        "full3d_dice_oracle_r5_context": case.full3d_dice_oracle5,
        "empty_gt_slice_convention": bool(not case.gt.any()),
        "panel_contract": (
            "the final two panels use the same one-round Editor prediction; "
            "only the right panel adds the observed scribble overlay"
        ),
        "oracle_warning": "R5 is shown only in the separate cohort ceiling figure",
    }


def render_all(args: argparse.Namespace) -> list[Path]:
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        raise FigureContractError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(
        Path(args.corpus),
        Path(args.one_round_manifest),
        Path(args.editor_eval_dir),
        Path(args.eval5_root),
        Path(args.visible5_root),
    )
    if [case.goal for case in cases] != list(GOAL_TEXT):
        raise FigureContractError("six-intent output order changed")

    outputs: list[Path] = []
    manifest_cases: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        filename = f"{index:02d}-{case.goal.lower().replace('_', '-')}.png"
        path = output / filename
        _render_case(case, index, path)
        outputs.append(path)
        manifest_cases.append(_case_manifest(case, index, filename))

    ladder_path = output / "07-five-round-oracle-ceiling.png"
    _render_ladder(Path(args.ladder_rows), ladder_path)
    outputs.append(ladder_path)

    manifest_path = output / "figure-manifest.json"
    manifest = {
        "schema_version": "PETCT-EDITOR-PRESENTATION-FIGURES-v2.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "VAL_ILLUSTRATION_NOT_LOCKED_TEST",
        "selection": (
            "fixed six-case set; complete R1-R5 validation trajectory, exact one-round alignment, "
            "one case per intent, chosen for presentation visibility rather than performance estimation"
        ),
        "colour_contract": {
            "mask": "cyan fill for GT, M0, and Editor M1",
            "added": "green",
            "removed": "red",
            "scribble": "white; no semantic legend colour",
        },
        "metric_contract": {
            "whole_slice_dice": (
                "Dice over the complete native axial slice at the round-1 prompted z; "
                "the displayed image is only a zoomed ROI"
            ),
            "target_error_recovered": (
                "fraction of the authorized target residual on the round-1 writable slice "
                "that the measured one-round Editor changed"
            ),
            "oracle_r5": (
                "teacher-forced perfect-correction trajectory shown only on the separate "
                "cohort ceiling figure; never displayed as a case-level model output"
            ),
            "case_panels": (
                "GT, M0, Editor result without overlay, and the same Editor result with "
                "scribble overlay; the final pair is not a model ablation"
            ),
        },
        "cases": manifest_cases,
        "outputs": [path.name for path in outputs],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.append(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--one-round-manifest", required=True)
    parser.add_argument("--editor-eval-dir", required=True)
    parser.add_argument("--eval5-root", required=True)
    parser.add_argument("--visible5-root", required=True)
    parser.add_argument("--ladder-rows", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    render_all(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
