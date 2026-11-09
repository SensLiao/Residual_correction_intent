"""The examination cache: normalised native-grid arrays written once per examination, and their reader.

Layout of a cache directory (``SIRB_PREPROCESSED/<data.cache.directory>``)::

    inputs/<examination>/pet.npy, ct.npy, initial_segmentation.npy,
                         initial_probability.npy (when the raw data has it), whole_volume_frame.npy, meta.json
                                                                what a deployed model may read
    labels/<examination>/gt.npy, label_meta.json               ground truth: training and scoring only

Arrays are ZYX on the native PET grid, plain ``.npy`` files that are memory-mapped when read: PET and CT normalised
and stored as float16, the initial probability as float16, the initial segmentation and the ground truth as uint8.
Normalisation, once per examination and never per window:

* CT: clip to [-1000, 1000] HU, divide by 1000;
* PET: ``log1p(max(SUV, 0))``, then ``(x - median) / IQR`` over the examination's voxels with SUV > 0 (with too few
  such voxels or a degenerate IQR the standard deviation, then 1, replaces the IQR); the rule taken, the median, the
  scale and the value of zero SUV (``pad_value``) go to ``meta.json``.

The initial segmentation is ``initial probability > 0.5`` (a tie is background), or the supplied mask, whose
disagreement with that rule is counted and must stay within ``data.initial_segmentation.max_disagreement`` of their
union. ``whole_volume_frame.npy`` holds PET, CT and the initial probability reduced to the 96 x 64 x 64 whole-volume
grid (cell means).

:class:`ExaminationCache` reads a cache. The split recorded in ``meta.json`` is checked against an allow-list on
every access: the test set is read only by a cache opened with ``splits=("test",)``. SIRB reads PET in fixed
reference coordinates: ``z = (s_i / s*) x + (mu_i - mu*) / s*`` with the examination's own median ``mu_i`` and scale
``s_i`` and the training-set reference ``(mu*, s*)`` of ``data.pet_reference``
(:func:`pet_reference_from_examinations`), so the same SUV gives the same input on every scanner; zero SUV becomes
``-mu* / s*``.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.io import TargetExistsError, write_json_atomic, write_npy_atomic
from .geometry import affine_xyz_from_zyx, validate_affine_xyz

__all__ = ["CACHE_SCHEMA", "FRAME_CHANNELS", "INITIAL_SEGMENTATION_RULE", "SPLITS", "ExaminationCache",
           "ExaminationCacheError", "ExaminationInputs", "LabelAccessError", "NonFiniteExaminationError",
           "PetReference", "cache_directory", "cached_examinations", "normalize_ct", "normalize_pet",
           "pet_reference_from_examinations", "pet_reference_from_config", "pet_to_reference",
           "threshold_initial_probability", "whole_volume_frame", "write_examination_cache"]

#: Schemas of ``meta.json`` and ``label_meta.json``, with their version.
CACHE_SCHEMA, LABEL_SCHEMA, SCHEMA_VERSION = "sirb-examination-cache", "sirb-examination-labels", 1
INPUTS_FOLDER, LABELS_FOLDER = "inputs", "labels"
META_FILE, LABEL_META_FILE = "meta.json", "label_meta.json"
#: Cached input arrays, in the order they are written.
INPUT_ARRAYS = ("pet", "ct", "initial_probability", "initial_segmentation")
PROBABILITY_ARRAY, SEGMENTATION_ARRAY, LABEL_ARRAY = "initial_probability", "initial_segmentation", "gt"
#: The whole-volume frame: channel order and file (its meta key is the file's stem).
FRAME_CHANNELS = ("pet", "ct", "initial_probability")
FRAME_FILE, FRAME_META_KEY = "whole_volume_frame.npy", "whole_volume_frame"
#: The initial segmentation rule: probability strictly greater than 0.5.
SEGMENTATION_THRESHOLD = 0.5
INITIAL_SEGMENTATION_RULE = f"strict_greater_than_{SEGMENTATION_THRESHOLD}"
#: Provenance key of the ground-truth file; it is written under ``labels/`` only.
LABEL_PROVENANCE_KEY = "ground_truth"
#: Values of padding outside the field of view in the whole-volume frame (CT: air; no initial probability).
PAD_CT, PAD_PROBABILITY = -1.0, 0.0
#: Split names an examination cache may record; reading is restricted further by an allow-list.
SPLITS = ("train", "val", "test")
DEFAULT_READ_SPLITS = ("train", "val")
#: Cells along an axis whose direction cosines exceed this are not orthogonal (whole-volume frame planning).
AXIS_ORTHOGONALITY_ATOL = 1e-4
_SHAPE_EPS = 1e-6


class ExaminationCacheError(RuntimeError):
    """A cached examination is missing, inconsistent, or outside the splits the cache was opened for."""


class LabelAccessError(ExaminationCacheError):
    """Ground truth was requested from a cache opened without labels."""


class NonFiniteExaminationError(ExaminationCacheError):
    """PET, CT or the initial probability holds NaN or Inf (before or after the float16 cast)."""


# ------------------------------------------------------------------------------------------- normalisation
def normalize_ct(ct_hu, normalization: Mapping[str, Any]) -> np.ndarray:
    """CT clipped to ``ct_clip`` HU and divided by ``ct_divisor``, float32."""
    low, high = (float(v) for v in normalization["ct_clip"])
    return (np.clip(np.asarray(ct_hu).astype(np.float32), low, high)
            / float(normalization["ct_divisor"])).astype(np.float32)


def normalize_pet(pet_suv, normalization: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    """``log1p(max(SUV, 0))``, then ``(x - median) / max(IQR, floor)`` over the examination's SUV > 0 voxels.

    Fewer than ``pet_min_valid_voxels`` such voxels or an IQR below ``pet_min_iqr`` fall back to the standard
    deviation, then to scale 1; the rule taken is returned with the statistics.
    """
    pet_suv = np.asarray(pet_suv)
    logged = np.log1p(np.clip(pet_suv.astype(np.float32), 0.0, None))
    positive = logged[pet_suv > 0]
    floor = float(normalization["pet_min_iqr"])
    stats = {"valid_voxels": int(positive.size), "rule": "median_iqr"}
    if positive.size >= int(normalization["pet_min_valid_voxels"]):
        q25, median, q75 = (float(v) for v in np.percentile(positive, (25, 50, 75)))
        scale = q75 - q25
        if scale < floor:
            scale, stats["rule"] = float(positive.std()), "median_std_fallback"
    else:
        median = float(np.median(positive)) if positive.size else 0.0
        scale, stats["rule"] = (float(positive.std()) if positive.size else 0.0), "median_std_fallback"
    if not np.isfinite(scale) or scale < floor:
        scale, stats["rule"] = 1.0, "unit_scale_fallback"
    stats.update(median=median, scale=scale, pad_value=float((0.0 - median) / scale))
    return ((logged - median) / scale).astype(np.float32), stats


def threshold_initial_probability(probability, rule: str = INITIAL_SEGMENTATION_RULE) -> np.ndarray:
    """The initial segmentation of a probability map: probability > 0.5 (:data:`INITIAL_SEGMENTATION_RULE`)."""
    if rule != INITIAL_SEGMENTATION_RULE:
        raise ExaminationCacheError(f"unknown initial segmentation rule actual={rule!r} "
                                    f"expected={INITIAL_SEGMENTATION_RULE!r}")
    return np.asarray(probability) > SEGMENTATION_THRESHOLD


# ------------------------------------------------------------------------------------------- PET reference
@dataclass(frozen=True)
class PetReference:
    """The fixed PET reference ``(mu*, s*)``: the training-set median of the examinations' PET medians and scales."""
    mu: float
    s: float

    def __post_init__(self):
        if not (np.isfinite(self.mu) and np.isfinite(self.s) and self.s > 0):
            raise ValueError(f"PET reference actual=(mu={self.mu}, s={self.s}), expected finite mu and s > 0")

    @property
    def pad_value(self) -> float:
        """The PET input of zero SUV (outside the body and in padding)."""
        return float(-self.mu / self.s)


def pet_reference_from_config(config: Mapping[str, Any]) -> PetReference:
    """``data.pet_reference`` of a merged configuration."""
    values = config["data"]["pet_reference"]
    return PetReference(float(values["mu"]), float(values["s"]))


def pet_reference_coefficients(median: float, scale: float, reference: PetReference) -> tuple[float, float]:
    """(a, b) with ``z = a x + b`` in float64: ``a = s_i / s*``, ``b = (mu_i - mu*) / s*``."""
    if not (np.isfinite(median) and np.isfinite(scale) and scale > 0):
        raise ExaminationCacheError(f"cached PET statistics actual=(median={median}, scale={scale}), expected finite "
                                    "values and scale > 0")
    return float(scale) / reference.s, (float(median) - reference.mu) / reference.s


def pet_to_reference(cached_pet, median: float, scale: float, reference: PetReference) -> np.ndarray:
    """Cached PET (normalised with the examination's own median and scale) in the reference coordinates, float32.

    The affine runs in float32 on the float32 values of the cache.
    """
    a, b = pet_reference_coefficients(median, scale, reference)
    return (np.float32(a) * np.asarray(cached_pet, dtype=np.float32) + np.float32(b)).astype(np.float32)


def pet_reference_from_examinations(metas: Sequence[Mapping[str, Any]]) -> PetReference:
    """The reference from training-set ``meta.json`` documents: per patient the median of its examinations' PET
    medians and scales, then the median over patients."""
    by_patient: dict[str, list[tuple[float, float]]] = {}
    for meta in metas:
        if meta.get("split") != "train":
            raise ExaminationCacheError(f"the PET reference uses training examinations only; "
                                        f"{meta.get('examination_id')} split={meta.get('split')!r}")
        statistics = meta["pet_normalization"]
        by_patient.setdefault(str(meta["patient_id"]), []).append((float(statistics["median"]),
                                                                   float(statistics["scale"])))
    if not by_patient:
        raise ExaminationCacheError("no training examination given")
    patient_mu = [float(np.median([m for m, _ in rows])) for _, rows in sorted(by_patient.items())]
    patient_s = [float(np.median([s for _, s in rows])) for _, rows in sorted(by_patient.items())]
    return PetReference(float(np.median(patient_mu)), float(np.median(patient_s)))


# ------------------------------------------------------------------------------------------- whole-volume frame
@dataclass(frozen=True)
class _FrameGrid:
    """The 3 mm grid of an examination as far as the whole-volume frame needs it (ZYX order)."""
    native_shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    fov_shape: tuple[int, int, int]
    scale: tuple[float, float, float]      # 3 mm cell step in native index units


def _frame_grid(native_shape, affine_zyx, spacing_mm) -> _FrameGrid:
    """The 3 mm grid keeps the native axis directions, is corner-aligned with the native grid and covers its whole
    field of view: along each axis ``(n + 0.5) * s_native = (r + 0.5) * s_grid``."""
    shape = tuple(int(v) for v in native_shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ExaminationCacheError(f"native shape must be three positive ints, got {shape}")
    a = validate_affine_xyz(affine_zyx)
    target = tuple(float(v) for v in spacing_mm)
    if len(target) != 3 or min(target) <= 0 or not np.isfinite(target).all():
        raise ExaminationCacheError(f"grid spacing must be three positive numbers, got {target}")
    native_spacing = tuple(float(np.linalg.norm(a[:3, k])) for k in range(3))
    directions = [a[:3, k] / native_spacing[k] for k in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            cosine = float(abs(np.dot(directions[i], directions[j])))
            if cosine > AXIS_ORTHOGONALITY_ATOL:
                raise ExaminationCacheError(f"native axes {i},{j} are not orthogonal: |cos|={cosine:.3e}")
    scale = tuple(target[k] / native_spacing[k] for k in range(3))
    fov = tuple(max(1, int(math.ceil(shape[k] * native_spacing[k] / target[k] - _SHAPE_EPS))) for k in range(3))
    return _FrameGrid(shape, target, fov, scale)


def _frame_cell_index(grid: _FrameGrid, axis: int, size: int) -> np.ndarray:
    """The frame cell of every native index along ``axis``: the native centre's 3 mm grid position as a fraction of
    the field of view, times the frame size."""
    native = np.arange(grid.native_shape[axis], dtype=np.float64)
    grid_index = (native + 0.5) / grid.scale[axis] - 0.5
    position = (grid_index + 0.5) / grid.fov_shape[axis]
    return np.clip(np.floor(position * size).astype(np.int64), 0, size - 1)


def _reduce_to_frame(native, grid: _FrameGrid, frame_shape: Sequence[int], *, empty_value: float) -> np.ndarray:
    """Mean of the native voxels in each frame cell (cells cover the field of view); an empty cell gets
    ``empty_value``. Separable: one axis at a time."""
    out = np.asarray(native, dtype=np.float32)
    for axis in range(3):
        size = int(frame_shape[axis])
        cells = _frame_cell_index(grid, axis, size)          # non-decreasing in the native index
        starts = np.searchsorted(cells, np.arange(size), side="left")
        stops = np.searchsorted(cells, np.arange(size), side="right")
        shape = list(out.shape)
        shape[axis] = size
        reduced = np.full(shape, np.float32(empty_value), dtype=np.float32)
        moved_in, moved_out = np.moveaxis(out, axis, 0), np.moveaxis(reduced, axis, 0)
        for cell in range(size):
            if stops[cell] > starts[cell]:
                moved_out[cell] = moved_in[starts[cell]:stops[cell]].mean(axis=0)
        out = reduced
    return out


def _frame_key(grid: _FrameGrid, frame_shape: Sequence[int]) -> dict:
    return {"channels": list(FRAME_CHANNELS), "shape": [int(v) for v in frame_shape],
            "grid_spacing_mm": [float(v) for v in grid.spacing_mm]}


def whole_volume_frame(pet, ct, probability, affine_zyx, pet_pad_value: float, *, grid_spacing_mm=(3.0, 3.0, 3.0),
                       shape=(96, 64, 64)) -> np.ndarray:
    """[3, *shape] float32: cached PET, CT and initial probability reduced to the whole-volume frame (cell means)."""
    grid = _frame_grid(np.shape(pet), affine_zyx, grid_spacing_mm)
    pads = {"pet": float(pet_pad_value), "ct": PAD_CT, "initial_probability": PAD_PROBABILITY}
    volumes = {"pet": pet, "ct": ct, "initial_probability": probability}
    return np.stack([_reduce_to_frame(volumes[name], grid, shape, empty_value=pads[name]) for name in FRAME_CHANNELS])


# ------------------------------------------------------------------------------------------- writing
def _refuse_non_finite(examination_id: str, arrays: Mapping[str, Any]) -> None:
    for name, array in arrays.items():
        array = np.asarray(array)
        count = int(array.size - np.count_nonzero(np.isfinite(array)))
        if count:
            raise NonFiniteExaminationError(f"{examination_id}: {name} holds {count} non-finite voxels (NaN or Inf) "
                                            f"of {array.size}")


def _new_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ExaminationCacheError(f"refusing to overwrite an existing cached examination: {path}") from error


def _save_array(path: Path, array: np.ndarray) -> None:
    try:
        write_npy_atomic(path, array, exclusive=True)
    except TargetExistsError as refusal:
        raise ExaminationCacheError(f"refusing to overwrite a cached array: {path}") from refusal


def _write_json(path: Path, document: Mapping) -> None:
    write_json_atomic(path, document, indent=2, sort_keys=True, ensure_ascii=True)


def _absent_probability(shape) -> np.ndarray:
    blank = np.full(tuple(int(v) for v in shape), PAD_PROBABILITY, dtype=np.float16)
    blank.flags.writeable = False
    return blank


def write_examination_cache(directory: str | os.PathLike, *, examination_id: str, patient_id: str, split: str,
                            pet_suv, ct_hu, affine_zyx, normalization: Mapping[str, Any], provenance: Mapping[str, Any],
                            initial_probability=None, initial_segmentation=None, ground_truth=None,
                            segmentation_rule: str = INITIAL_SEGMENTATION_RULE, max_disagreement: float | None = None,
                            whole_volume_frame_grid: Mapping[str, Any] | None = None) -> dict:
    """Normalise one examination once and store its native-grid arrays (module docstring). Returns the meta
    document, with the label meta under ``"label"``.

    ``initial_segmentation`` is stored as given; with ``initial_probability`` its disagreement with the threshold rule
    is counted and refused above ``max_disagreement`` (a fraction of their union). Without it the rule defines the
    mask. An examination without ``initial_probability`` needs ``initial_segmentation``; it is cached without the
    probability (``initial_segmentation.probability_cached: false``) and that frame channel holds 0.
    ``whole_volume_frame_grid`` (``grid_spacing_mm``, ``shape``) adds the whole-volume frame. ``provenance`` maps
    file roles to raw-relative paths; the ground-truth entry goes to the label meta only. Every refusal happens before
    anything is written; an examination already in the cache is refused.
    """
    if split not in SPLITS:
        raise ExaminationCacheError(f"{examination_id}: split actual={split!r} expected one of {SPLITS}")
    has_probability = initial_probability is not None
    if not has_probability and initial_segmentation is None:
        raise ExaminationCacheError(f"{examination_id}: an examination cached without the initial probability needs "
                                    "its initial segmentation")
    pet_suv, ct_hu = np.asarray(pet_suv), np.asarray(ct_hu)
    shape = tuple(int(v) for v in pet_suv.shape)
    named = [("ct", ct_hu)] + ([("initial probability", initial_probability)] if has_probability else []) + (
        [("gt", ground_truth)] if ground_truth is not None else [])
    for name, array in named:
        if tuple(np.shape(array)) != shape:
            raise ExaminationCacheError(f"{examination_id}: {name} shape {tuple(np.shape(array))} differs from PET "
                                        f"{shape}")
    _refuse_non_finite(examination_id, {"pet": pet_suv, "ct": ct_hu,
                                        **({"initial probability": initial_probability} if has_probability
                                           else {})})
    if has_probability and (float(np.min(initial_probability)) < 0.0 or float(np.max(initial_probability)) > 1.0):
        raise ExaminationCacheError(f"{examination_id}: initial probability outside [0,1]: "
                                    f"min={np.min(initial_probability)} "
                                    f"max={np.max(initial_probability)}")
    affine = validate_affine_xyz(affine_zyx)
    pet, pet_stats = normalize_pet(pet_suv, normalization)
    ruled = threshold_initial_probability(initial_probability, segmentation_rule) if has_probability else None
    if initial_segmentation is not None and tuple(np.shape(initial_segmentation)) != shape:
        raise ExaminationCacheError(f"{examination_id}: initial segmentation shape "
                                    f"{tuple(np.shape(initial_segmentation))} differs "
                                    f"from PET {shape}")
    mask = ruled if initial_segmentation is None else np.asarray(initial_segmentation) > 0
    if initial_segmentation is not None and has_probability:
        if max_disagreement is None:
            raise ExaminationCacheError(f"{examination_id}: a supplied initial segmentation needs the disagreement "
                                        "limit")
        union = int((mask | ruled).sum())
        fraction = float((mask != ruled).sum()) / union if union else 0.0
        if fraction > float(max_disagreement):
            raise ExaminationCacheError(f"{examination_id}: initial segmentation disagrees with the threshold rule: "
                                        f"fraction_of_union={fraction:.6f} allowed={float(max_disagreement)} "
                                        "(axis order of the probability file?)")
    arrays = {"pet": pet.astype(np.float16), "ct": normalize_ct(ct_hu, normalization).astype(np.float16)}
    if has_probability:
        arrays[PROBABILITY_ARRAY] = np.asarray(initial_probability, dtype=np.float16)
    arrays[SEGMENTATION_ARRAY] = mask.astype(np.uint8)
    # What training reads is the float16 copy: an absurd but finite SUV can still overflow it to Inf.
    _refuse_non_finite(examination_id, {f"cached {name}": arrays[name] for name in ("pet", "ct", PROBABILITY_ARRAY)
                                        if name in arrays})
    root = Path(directory)
    input_dir = root / INPUTS_FOLDER / examination_id
    _new_directory(input_dir)
    for name, array in arrays.items():
        _save_array(input_dir / f"{name}.npy", array)
    meta = {"schema": CACHE_SCHEMA, "version": SCHEMA_VERSION, "examination_id": examination_id,
            "patient_id": patient_id, "split": split, "shape_zyx": list(shape), "affine_zyx": affine.tolist(),
            "pet_normalization": pet_stats,
            "initial_segmentation": {
                "rule": segmentation_rule,
                "source": "threshold_rule" if initial_segmentation is None else "supplied_mask",
                "voxels_disagreeing_with_rule": int((mask != ruled).sum()) if has_probability else None,
                "probability_cached": has_probability},
            "provenance": {key: value for key, value in provenance.items() if key != LABEL_PROVENANCE_KEY}}
    if whole_volume_frame_grid is not None:
        grid = _frame_grid(shape, affine, whole_volume_frame_grid["grid_spacing_mm"])
        key = _frame_key(grid, whole_volume_frame_grid["shape"])
        probability = arrays[PROBABILITY_ARRAY] if has_probability else _absent_probability(shape)
        frames = whole_volume_frame(arrays["pet"], arrays["ct"], probability, affine, float(pet_stats["pad_value"]),
                                    grid_spacing_mm=key["grid_spacing_mm"], shape=key["shape"])
        _save_array(input_dir / FRAME_FILE, frames)
        meta[FRAME_META_KEY] = key
    label_meta = None
    if ground_truth is not None:
        label_dir = root / LABELS_FOLDER / examination_id
        _new_directory(label_dir)
        truth = np.asarray(ground_truth) > 0
        _save_array(label_dir / f"{LABEL_ARRAY}.npy", truth.astype(np.uint8))
        label_meta = {"schema": LABEL_SCHEMA, "version": SCHEMA_VERSION, "examination_id": examination_id,
                      "split": split, "gt_positive": bool(truth.any()), "gt_voxels": int(truth.sum()),
                      # An initial segmentation identical to a non-empty ground truth was made from the labels.
                      "initial_segmentation_equals_gt": bool(truth.any() and np.array_equal(mask, truth)),
                      "provenance": {key: value for key, value in provenance.items() if key == LABEL_PROVENANCE_KEY}}
        _write_json(label_dir / LABEL_META_FILE, label_meta)
    _write_json(input_dir / META_FILE, meta)
    return dict(meta, label=label_meta)


# ------------------------------------------------------------------------------------------- reading
@dataclass(frozen=True)
class ExaminationInputs:
    """Everything a deployed model may read of one examination (ZYX, native PET grid). No ground truth.

    ``pet`` is float32 in the reference coordinates when the cache was opened with a PET reference, else the cached
    float16; ``ct`` is the cached float16 (memory-mapped); ``initial_probability`` is None unless asked for.
    """
    examination_id: str
    patient_id: str
    split: str
    affine_zyx: np.ndarray
    pet: np.ndarray
    ct: np.ndarray
    initial_segmentation: np.ndarray
    pet_pad_value: float
    initial_probability: np.ndarray | None = None

    @property
    def affine_xyz(self) -> np.ndarray:
        return affine_xyz_from_zyx(self.affine_zyx)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.pet.shape)


class ExaminationCache:
    """Read side of an examination cache (module docstring).

    ``splits`` is the allow-list of recorded splits; ``labels=False`` refuses every ground-truth request;
    ``pet_reference`` returns PET in the reference coordinates; ``read_initial_probability`` also reads the initial
    probability.
    """

    def __init__(self, directory: str | os.PathLike, *, splits: Sequence[str] = DEFAULT_READ_SPLITS,
                 pet_reference: PetReference | None = None, labels: bool = True, mmap: bool = True,
                 read_initial_probability: bool = False):
        self.directory = Path(directory)
        self.inputs_root, self.labels_root = self.directory / INPUTS_FOLDER, self.directory / LABELS_FOLDER
        self.splits = tuple(splits)
        unknown = sorted(set(self.splits) - set(SPLITS))
        if not self.splits or unknown:
            raise ExaminationCacheError(f"splits actual={self.splits}, expected a non-empty subset of {SPLITS}")
        if pet_reference is not None and not isinstance(pet_reference, PetReference):
            raise TypeError("pet_reference must be a PetReference")
        self.pet_reference, self.labels = pet_reference, bool(labels)
        self.mmap_mode = "r" if mmap else None
        self.read_initial_probability = bool(read_initial_probability)
        self._meta: dict[str, dict] = {}

    def meta(self, examination_id: str) -> dict:
        """``meta.json`` of the examination (shared between calls: read it, do not modify it)."""
        meta = self._meta.get(examination_id)
        if meta is None:
            path = self.inputs_root / examination_id / META_FILE
            if not path.is_file():
                raise ExaminationCacheError(f"no cached examination {examination_id!r} under {self.inputs_root}")
            meta = self._meta[examination_id] = json.loads(path.read_text(encoding="utf-8"))
        if meta["split"] not in self.splits:
            raise ExaminationCacheError(f"{examination_id}: split actual={meta['split']!r} allowed={self.splits}")
        return meta

    def examinations(self, split: str) -> list[str]:
        """Ids of the cached examinations of ``split`` (which must be allowed), sorted."""
        if split not in self.splits:
            raise ExaminationCacheError(f"split actual={split!r} allowed={self.splits}")
        return [meta["examination_id"] for meta in cached_examinations(self.directory, split)]

    def _load(self, folder: Path, name: str) -> np.ndarray:
        return np.load(folder / f"{name}.npy", mmap_mode=self.mmap_mode, allow_pickle=False)

    def inputs(self, examination_id: str) -> ExaminationInputs:
        meta = self.meta(examination_id)
        folder = self.inputs_root / examination_id
        pet = self._load(folder, "pet")
        pad = float(meta["pet_normalization"]["pad_value"])
        if self.pet_reference is not None:
            statistics = meta["pet_normalization"]
            pet = pet_to_reference(pet, float(statistics["median"]), float(statistics["scale"]), self.pet_reference)
            pad = self.pet_reference.pad_value
        probability = None
        if self.read_initial_probability:
            if not meta["initial_segmentation"]["probability_cached"]:
                raise ExaminationCacheError(f"{examination_id}: cached without the initial probability")
            probability = self._load(folder, PROBABILITY_ARRAY)
        return ExaminationInputs(examination_id, meta["patient_id"], meta["split"],
                                 np.asarray(meta["affine_zyx"], dtype=np.float64), pet, self._load(folder, "ct"),
                                 np.asarray(self._load(folder, SEGMENTATION_ARRAY)) > 0, pad, probability)

    def ground_truth(self, examination_id: str) -> np.ndarray:
        """Training and scoring only: the ground truth as a bool ZYX array."""
        if not self.labels:
            raise LabelAccessError("this cache was opened without labels")
        self.meta(examination_id)
        return np.load(self.labels_root / examination_id / f"{LABEL_ARRAY}.npy", mmap_mode=self.mmap_mode,
                       allow_pickle=False) > 0

    def label_meta(self, examination_id: str) -> dict:
        if not self.labels:
            raise LabelAccessError("this cache was opened without labels")
        self.meta(examination_id)
        return json.loads((self.labels_root / examination_id / LABEL_META_FILE).read_text(encoding="utf-8"))


def cached_examinations(directory: str | os.PathLike, split: str) -> list[dict]:
    """``meta.json`` of every cached examination of ``split``, in examination-id order."""
    metas = [json.loads(path.read_text(encoding="utf-8"))
             for path in sorted((Path(directory) / INPUTS_FOLDER).glob(f"*/{META_FILE}"))]
    return [meta for meta in metas if meta["split"] == split]


def cache_directory(config: Mapping[str, Any]) -> Path:
    """The cache directory of a merged configuration: ``paths.preprocessed / data.cache.directory``."""
    return Path(config["paths"]["preprocessed"]) / config["data"]["cache"]["directory"]
