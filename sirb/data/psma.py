"""The PSMA PET/CT data in nnU-Net raw format: examinations, patient-level splits and scribble drawing styles.

Raw layout under ``SIRB_RAW`` (the folder names, channel suffixes and the probability key are ``data.raw`` of
``sirb/configs/data/psma.json``)::

    imagesTr/<examination>_0000.nii.gz           CT (HU)
    imagesTr/<examination>_0001.nii.gz           PET (SUV)
    labelsTr/<examination>.nii.gz                ground truth (lesion voxels > 0)
    initial_segmentations/<examination>.nii.gz   initial segmentation (voxels > 0)
    initial_segmentations/<examination>.npz      optional: the foreground probability of the initial segmentation
                                                 model (float, voxel order of the NIfTI files)

An examination id is ``psma_<patient>_<date>``; its patient id is the id without the date. The split lists in
``splits/`` are patient-level: no patient has examinations in two of the training, validation and test sets.

Every scribble of an examination is drawn in one of the three styles of the official simulator; the style belongs to
the examination (:func:`assign_drawing_styles`). The training set is divided by patient into two halves
(:func:`patient_half`) whose self-generated states are refreshed in turn.
"""

from __future__ import annotations

import filecmp
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = ["DRAWING_STYLES", "PATIENT_HALVES", "RAW_FILES", "SPLIT_NAMES", "RawExamination", "RawLayout",
           "RawDataError", "SplitError", "assign_drawing_styles", "check_patient_level", "patient_half", "patient_id",
           "raw_files", "read_drawing_styles", "read_raw_examination", "read_split", "splits_directory", "xyz_to_zyx",
           "zyx_affine_from_xyz"]

#: The split lists shipped in ``splits/``; ``timing`` lists validation examinations.
SPLIT_NAMES = ("train", "val", "test", "timing")
#: The scribble styles of the official autoPET V simulator, in the order styles are dealt.
DRAWING_STYLES = ("centerline", "random", "boundary")
#: Salt of the style assignment (the examinations are ordered by SHA-256 of "<salt>|<examination id>").
STYLE_SALT = "PETCT-OFFICIAL-EQUAL-THIRDS-v1"
#: The training patients are divided in two halves by the SHA-256 of the patient id.
PATIENT_HALVES = 2
#: Images, labels and initial segmentations must share the PET affine up to this many millimetres.
AFFINE_ATOL_MM = 1e-4


class RawDataError(ValueError):
    """A raw examination is missing, inconsistent or malformed."""


class SplitError(ValueError):
    """A split list or a style assignment is malformed or breaks the patient-level separation."""


# ------------------------------------------------------------------------------------------- examination ids
def patient_id(examination_id: str) -> str:
    """The patient of an examination: the id without its trailing ``_<date>``."""
    examination = str(examination_id)
    patient, separator, date = examination.rpartition("_")
    if not separator or not patient or not date:
        raise SplitError(f"examination id actual={examination!r}, expected <patient>_<date>")
    return patient


def patient_half(patient: str) -> int:
    """0 or 1: the SHA-256 of the patient id (UTF-8) as an integer, modulo :data:`PATIENT_HALVES`. A patient id, never
    an examination id, so all examinations of a patient fall in one half."""
    if not isinstance(patient, str) or not patient:
        raise SplitError(f"patient id actual={patient!r}, expected a non-empty string")
    return int(hashlib.sha256(patient.encode("utf-8")).hexdigest(), 16) % PATIENT_HALVES


# ------------------------------------------------------------------------------------------- splits
def splits_directory(config: Mapping[str, Any], repository: str | os.PathLike) -> Path:
    """``data.splits`` of a merged configuration; a relative path is relative to the repository folder."""
    path = Path(config["data"]["splits"])
    return path if path.is_absolute() else Path(repository) / path


def read_split(splits_directory: str | os.PathLike, name: str) -> list[str]:
    """The examination ids of one split list (one id per line, sorted, unique)."""
    if name not in SPLIT_NAMES:
        raise SplitError(f"split actual={name!r}, expected one of {SPLIT_NAMES}")
    path = Path(splits_directory) / f"{name}.txt"
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if ids != sorted(set(ids)):
        raise SplitError(f"{path}: examination ids must be sorted and unique")
    for examination in ids:
        patient_id(examination)
    return ids


def check_patient_level(splits: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    """The patients of each split, after checking that no patient is in two splits and no examination repeats."""
    patients = {name: {patient_id(examination) for examination in ids} for name, ids in splits.items()}
    names = sorted(patients)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            shared = patients[first] & patients[second]
            if shared:
                raise SplitError(f"splits {first!r} and {second!r} share {len(shared)} patients, "
                                 f"first: {sorted(shared)[0]}")
    return patients


# ------------------------------------------------------------------------------------------- drawing styles
def assign_drawing_styles(examination_ids: Sequence[str]) -> dict[str, str]:
    """One drawing style per examination: the examinations are ordered by the SHA-256 of
    ``"<salt>|<examination id>"`` (ties by the id) and :data:`DRAWING_STYLES` are dealt in turn along that order.

    The result depends on the list the rule is applied to; the style files in ``splits/`` were made from the whole
    validation and test lists. The returned dict follows the order of ``examination_ids``.
    """
    if isinstance(examination_ids, (str, bytes)) or not examination_ids:
        raise SplitError("the style assignment needs a non-empty list of examination ids")
    ids = [str(examination) for examination in examination_ids]
    if any(not examination for examination in ids) or len(set(ids)) != len(ids):
        raise SplitError("the examination ids of a style assignment must be non-empty and unique")
    ranked = sorted(ids, key=lambda examination: (
        hashlib.sha256(f"{STYLE_SALT}|{examination}".encode("utf-8")).hexdigest(), examination))
    by_examination = {examination: DRAWING_STYLES[index % len(DRAWING_STYLES)]
                      for index, examination in enumerate(ranked)}
    counts = [sum(style == wanted for style in by_examination.values()) for wanted in DRAWING_STYLES]
    if max(counts) - min(counts) > 1:
        raise SplitError(f"the style assignment is not balanced by thirds: {counts}")
    return {examination: by_examination[examination] for examination in ids}


def read_drawing_styles(splits_directory: str | os.PathLike, split: str) -> dict[str, str]:
    """``<split>_styles.json``: examination id -> drawing style, for the validation and test sets."""
    path = Path(splits_directory) / f"{split}_styles.json"
    styles = json.loads(path.read_text(encoding="utf-8"))
    unknown = sorted({style for style in styles.values() if style not in DRAWING_STYLES})
    if unknown:
        raise SplitError(f"{path}: unknown drawing styles {unknown}; registered {DRAWING_STYLES}")
    return {str(examination): str(style) for examination, style in styles.items()}


# ------------------------------------------------------------------------------------------- raw examinations
@dataclass(frozen=True)
class RawLayout:
    """Folder and file names of the raw dataset (``data.raw``)."""
    images: str = "imagesTr"
    labels: str = "labelsTr"
    initial_segmentations: str = "initial_segmentations"
    ct_suffix: str = "_0000"
    pet_suffix: str = "_0001"
    extension: str = ".nii.gz"
    probability_key: str = "foreground_probability"

    @classmethod
    def from_config(cls, block: Mapping[str, Any]) -> "RawLayout":
        unknown = sorted(set(block) - set(cls.__dataclass_fields__))
        if unknown:
            raise RawDataError(f"data.raw has unknown keys {unknown}")
        return cls(**{key: str(value) for key, value in block.items()})


#: The raw files of an examination, in the order they are checked; the initial probability may be absent.
RAW_FILES = ("ct", "pet", "ground_truth", "initial_segmentation", "initial_probability")


def raw_files(raw_root: str | os.PathLike, examination_id: str, layout: RawLayout = RawLayout()) -> dict[str, Path]:
    """Paths of the raw files of one examination (:data:`RAW_FILES`)."""
    root = Path(raw_root)
    return {"ct": root / layout.images / f"{examination_id}{layout.ct_suffix}{layout.extension}",
            "pet": root / layout.images / f"{examination_id}{layout.pet_suffix}{layout.extension}",
            "ground_truth": root / layout.labels / f"{examination_id}{layout.extension}",
            "initial_segmentation": root / layout.initial_segmentations / f"{examination_id}{layout.extension}",
            "initial_probability": root / layout.initial_segmentations / f"{examination_id}.npz"}


def xyz_to_zyx(array: np.ndarray) -> np.ndarray:
    """A NIfTI voxel array (x, y, z) in the ZYX order of every SIRB array, C-contiguous."""
    if array.ndim != 3:
        raise RawDataError(f"expected a 3D array, got ndim={array.ndim}")
    return np.ascontiguousarray(np.transpose(array, (2, 1, 0)))


def zyx_affine_from_xyz(affine_xyz) -> np.ndarray:
    """The voxel-to-world affine of the transposed (ZYX) array: index (z, y, x) maps where (x, y, z) mapped.

    Written as the product with a permutation matrix, so signed zeros of the file's affine come out as that product
    gives them.
    """
    a = np.asarray(affine_xyz, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise RawDataError("affine must be a finite 4x4 matrix")
    if not np.allclose(a[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise RawDataError(f"affine last row must be [0,0,0,1], got {a[3].tolist()}")
    if abs(float(np.linalg.det(a[:3, :3]))) < 1e-12:
        raise RawDataError("singular voxel-to-world affine")
    permutation = np.zeros((4, 4))
    permutation[0, 2] = permutation[1, 1] = permutation[2, 0] = permutation[3, 3] = 1.0
    return a @ permutation


@dataclass(frozen=True)
class RawExamination:
    """One raw examination in ZYX order on the native PET grid."""
    examination_id: str
    pet_suv: np.ndarray                      # float32
    ct_hu: np.ndarray                        # float32
    ground_truth: np.ndarray                 # bool
    initial_segmentation: np.ndarray         # bool
    initial_probability: np.ndarray | None   # float32 or None
    affine_zyx: np.ndarray                   # float64 4 x 4, ZYX index -> world mm
    provenance: dict = field(default_factory=dict)


def read_raw_examination(raw_root: str | os.PathLike, examination_id: str, layout: RawLayout = RawLayout(), *,
                         read_probability: bool = True) -> RawExamination:
    """Read the CT, PET, ground truth, initial segmentation and (when present) its probability of one examination.

    The CT, ground truth and initial segmentation must have the PET's shape and affine (within
    :data:`AFFINE_ATOL_MM`), the probability its shape. Arrays come back in ZYX order with the matching affine. The
    provenance holds the raw-relative path of every file read; two files with the same bytes (an image pointing at
    the ground-truth file, say) are refused.
    """
    import nibabel as nib  # noqa: PLC0415  (only the raw reader needs nibabel)

    files = raw_files(raw_root, examination_id, layout)
    sources = list(RAW_FILES[:4])
    for name in sources:
        if not files[name].is_file():
            raise RawDataError(f"{examination_id}: missing {name} file {files[name]}")
    images = {name: nib.load(os.fspath(files[name])) for name in ("ct", "pet", "ground_truth")}
    reference = images["pet"]
    for name, image in images.items():
        if image.shape != reference.shape:
            raise RawDataError(f"{examination_id}: {name} shape {image.shape} differs from PET {reference.shape}")
        if not np.allclose(image.affine, reference.affine, atol=AFFINE_ATOL_MM):
            raise RawDataError(f"{examination_id}: {name} affine differs from PET beyond {AFFINE_ATOL_MM} mm")
    probability_xyz = None
    if read_probability and files["initial_probability"].is_file():
        with np.load(files["initial_probability"], allow_pickle=False) as archive:
            probability_xyz = archive[layout.probability_key]
        if probability_xyz.shape != reference.shape:
            raise RawDataError(f"{examination_id}: initial probability shape {probability_xyz.shape} differs from "
                               f"PET {reference.shape}")
        sources.append("initial_probability")
    mask_image = nib.load(os.fspath(files["initial_segmentation"]))
    if mask_image.shape != reference.shape:
        raise RawDataError(f"{examination_id}: initial segmentation shape {mask_image.shape} differs from PET "
                           f"{reference.shape}")
    if not np.allclose(mask_image.affine, reference.affine, atol=AFFINE_ATOL_MM):
        raise RawDataError(f"{examination_id}: initial segmentation affine differs from PET beyond {AFFINE_ATOL_MM} "
                           f"mm: max_abs_difference={float(np.abs(mask_image.affine - reference.affine).max()):.6g}")
    duplicated = {a: b for i, a in enumerate(sources) for b in sources[i + 1:]
                  if filecmp.cmp(files[a], files[b], shallow=False)}
    if duplicated:
        raise RawDataError(f"{examination_id}: raw files have the same bytes: {duplicated}")
    root = Path(raw_root)
    provenance = {name: files[name].relative_to(root).as_posix() for name in sources}
    return RawExamination(
        examination_id=str(examination_id),
        pet_suv=xyz_to_zyx(np.asarray(reference.dataobj, dtype=np.float32)),
        ct_hu=xyz_to_zyx(np.asarray(images["ct"].dataobj, dtype=np.float32)),
        ground_truth=xyz_to_zyx(np.asarray(images["ground_truth"].dataobj) > 0),
        initial_segmentation=xyz_to_zyx(np.asarray(mask_image.dataobj) > 0),
        initial_probability=None if probability_xyz is None else xyz_to_zyx(np.asarray(probability_xyz,
                                                                                       dtype=np.float32)),
        affine_zyx=zyx_affine_from_xyz(reference.affine),
        provenance=provenance)
