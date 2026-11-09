"""The examination cache: normalised native-grid arrays written once per examination, and their reader.

Layout of a cache directory (``SIRB_PREPROCESSED/<data.cache.directory>``)::

    inputs/<examination>/pet.npy, ct.npy, initial_segmentation.npy,
                         initial_probability.npy (when the raw data has it), meta.json
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
union.

:class:`ExaminationCache` reads a cache. The split recorded in ``meta.json`` is checked against an allow-list on
every access: the test set is read only by a cache opened with ``splits=("test",)``. SIRB reads PET in fixed
reference coordinates: ``z = (s_i / s*) x + (mu_i - mu*) / s*`` with the examination's own median ``mu_i`` and scale
``s_i`` and the training-set reference ``(mu*, s*)`` of ``data.pet_reference``
(:func:`pet_reference_from_examinations`), so the same SUV gives the same input on every scanner; zero SUV becomes
``-mu* / s*``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..utils.io import TargetExistsError, write_json_atomic, write_npy_atomic
from .geometry import affine_xyz_from_zyx, validate_affine_xyz

__all__ = ["CACHE_SCHEMA", "INITIAL_SEGMENTATION_RULE", "SPLITS", "ExaminationCache",
           "ExaminationCacheError", "ExaminationInputs", "LabelAccessError", "NonFiniteExaminationError",
           "PetReference", "cache_directory", "cached_examinations", "normalize_ct", "normalize_pet",
           "pet_reference_from_examinations", "pet_reference_from_config", "pet_to_reference",
           "threshold_initial_probability", "write_examination_cache"]

#: Schemas of ``meta.json`` and ``label_meta.json``, with their version.
CACHE_SCHEMA, LABEL_SCHEMA, SCHEMA_VERSION = "sirb-examination-cache", "sirb-examination-labels", 1
INPUTS_FOLDER, LABELS_FOLDER = "inputs", "labels"
META_FILE, LABEL_META_FILE = "meta.json", "label_meta.json"
#: Cached input arrays, in the order they are written.
INPUT_ARRAYS = ("pet", "ct", "initial_probability", "initial_segmentation")
PROBABILITY_ARRAY, SEGMENTATION_ARRAY, LABEL_ARRAY = "initial_probability", "initial_segmentation", "gt"
#: The initial segmentation rule: probability strictly greater than 0.5.
SEGMENTATION_THRESHOLD = 0.5
INITIAL_SEGMENTATION_RULE = f"strict_greater_than_{SEGMENTATION_THRESHOLD}"
#: Provenance key of the ground-truth file; it is written under ``labels/`` only.
LABEL_PROVENANCE_KEY = "ground_truth"
#: Split names an examination cache may record; reading is restricted further by an allow-list.
SPLITS = ("train", "val", "test")
DEFAULT_READ_SPLITS = ("train", "val")


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


def write_examination_cache(directory: str | os.PathLike, *, examination_id: str, patient_id: str, split: str,
                            pet_suv, ct_hu, affine_zyx, normalization: Mapping[str, Any], provenance: Mapping[str, Any],
                            initial_probability=None, initial_segmentation=None, ground_truth=None,
                            segmentation_rule: str = INITIAL_SEGMENTATION_RULE,
                            max_disagreement: float | None = None) -> dict:
    """Normalise one examination once and store its native-grid arrays (module docstring). Returns the meta
    document, with the label meta under ``"label"``.

    ``initial_segmentation`` is stored as given; with ``initial_probability`` its disagreement with the threshold rule
    is counted and refused above ``max_disagreement`` (a fraction of their union). Without it the rule defines the
    mask. An examination without ``initial_probability`` needs ``initial_segmentation``; it is cached without the
    probability (``initial_segmentation.probability_cached: false``). ``provenance`` maps file roles to the paths
    read (relative to their folder); the ground-truth entry goes to the label meta only. Every refusal happens before
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
