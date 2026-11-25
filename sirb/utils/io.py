"""File input and output: atomic writes, JSON, NumPy arrays and NIfTI images.

An atomic write goes to a hidden temporary file next to its target, is flushed and fsync'ed, and is published with
``os.replace``: a reader sees either no file or the complete file, and a failed write leaves no temporary file
behind. ``exclusive=True`` refuses a target that exists before the write or that appears before publication.

NIfTI arrays keep the voxel order of the file (x, y, z for the autoPET V images); converting them to the model's
axis order is the job of ``sirb.data``. ``nibabel`` is imported only by the two NIfTI functions.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import os
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping

import numpy as np

__all__ = ["NIFTI_SUFFIXES", "TargetExistsError", "atomic_write", "json_default", "read_json", "read_nifti",
           "read_npy", "read_npz", "write_json_atomic", "write_nifti", "write_npy_atomic", "write_npz_atomic"]

#: Suffix of the temporary file of an atomic write.
PARTIAL_SUFFIX = ".partial"
#: File name endings :func:`write_nifti` accepts; ``.nii.gz`` is gzip-compressed.
NIFTI_SUFFIXES = (".nii.gz", ".nii")
#: gzip level of ``.nii.gz`` files (nibabel's default level).
NIFTI_GZIP_LEVEL = 1
#: qform and sform code of written NIfTI headers: 1, scanner coordinates, as SimpleITK writes them, so that
#: nibabel- and ITK-based readers agree on the geometry.
NIFTI_XFORM_CODE = 1


class TargetExistsError(FileExistsError):
    """An exclusive write found its target on disk, before writing or before publishing."""


def _refuse_existing(target: Path) -> None:
    if target.exists():
        raise TargetExistsError(f"refusing to overwrite an existing file: {target}")


def atomic_write(path: str | os.PathLike, write: Callable[[BinaryIO], Any], *, exclusive: bool = False) -> Path:
    """Publish what ``write`` puts into a binary stream as ``path`` (module doc). Returns the target path."""
    target = Path(path)
    if exclusive:
        _refuse_existing(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}{PARTIAL_SUFFIX}")
    try:
        with temporary.open("wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            _refuse_existing(target)
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    return target


def json_default(value: Any) -> Any:
    """``default=`` of ``json.dumps``: a NumPy scalar becomes the Python number; anything else (arrays included) is
    refused, because an array in a JSON row is a bug upstream rather than a list to write."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def write_json_atomic(path: str | os.PathLike, document: Any, *, indent: int | None = 2, sort_keys: bool = True,
                      ensure_ascii: bool = True, allow_nan: bool = False, exclusive: bool = False) -> Path:
    """One JSON document and a final newline, UTF-8, published atomically. NaN and infinity are refused unless
    ``allow_nan``; NumPy scalars are written as numbers (:func:`json_default`)."""
    text = json.dumps(document, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii, allow_nan=allow_nan,
                      default=json_default)
    data = (text + "\n").encode("utf-8")
    return atomic_write(path, lambda stream: stream.write(data), exclusive=exclusive)


def read_json(path: str | os.PathLike) -> Any:
    """The JSON document in a UTF-8 file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_npy_atomic(path: str | os.PathLike, array: Any, *, exclusive: bool = False) -> Path:
    """``np.save`` of one array (no pickle), published atomically."""
    data = np.asarray(array)
    return atomic_write(path, lambda stream: np.save(stream, data, allow_pickle=False), exclusive=exclusive)


def read_npy(path: str | os.PathLike, *, mmap_mode: str | None = None) -> np.ndarray:
    """``np.load`` of one ``.npy`` array (no pickle); ``mmap_mode="r"`` maps it instead of reading it."""
    return np.load(os.fspath(path), mmap_mode=mmap_mode, allow_pickle=False)


def write_npz_atomic(path: str | os.PathLike, arrays: Mapping[str, Any], *, compressed: bool = True,
                     exclusive: bool = False) -> Path:
    """``np.savez_compressed`` (or ``np.savez``) of named arrays, published atomically. The name must end in .npz."""
    if not os.fspath(path).endswith(".npz"):
        raise ValueError(f"an npz archive needs a file name ending in .npz: {path}")
    save = np.savez_compressed if compressed else np.savez
    return atomic_write(path, lambda stream: save(stream, **arrays), exclusive=exclusive)


def read_npz(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Every array of an ``.npz`` archive (no pickle), read into memory."""
    with np.load(os.fspath(path), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def read_nifti(path: str | os.PathLike, *, dtype: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """``(array, affine)`` of a NIfTI image: the voxel array in the file's axis order (scaled by the header's slope
    and intercept, cast to ``dtype`` when given) and the 4 x 4 voxel-to-world affine as float64."""
    import nibabel as nib  # noqa: PLC0415  (only the NIfTI functions need nibabel)

    image = nib.load(os.fspath(path))
    array = np.asarray(image.dataobj) if dtype is None else np.asarray(image.dataobj, dtype=dtype)
    return array, np.array(image.affine, dtype=np.float64)


def write_nifti(path: str | os.PathLike, array: Any, affine: Any, *, dtype: Any = None,
                exclusive: bool = False) -> Path:
    """Write ``array`` (file axis order) with its 4 x 4 ``affine`` as ``.nii.gz`` or ``.nii``, published atomically.

    A boolean array is stored as uint8; ``dtype`` stores another type. qform and sform both hold ``affine`` with code
    :data:`NIFTI_XFORM_CODE`. Writing the same array and affine twice gives the same bytes (the gzip header carries
    no time stamp or file name).
    """
    import nibabel as nib  # noqa: PLC0415  (only the NIfTI functions need nibabel)

    name = os.fspath(path)
    if not name.endswith(NIFTI_SUFFIXES):
        raise ValueError(f"a NIfTI file name must end in .nii.gz or .nii: {path}")
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"the affine must be 4 x 4, not {matrix.shape}")
    data = np.asarray(array)
    if dtype is not None:
        data = data.astype(dtype, copy=False)
    elif data.dtype == np.bool_:
        data = data.astype(np.uint8)
    image = nib.Nifti1Image(data, matrix, dtype=dtype)
    image.header.set_qform(matrix, code=NIFTI_XFORM_CODE)
    image.header.set_sform(matrix, code=NIFTI_XFORM_CODE)
    payload = image.to_bytes()
    if name.endswith(".gz"):
        payload = gzip.compress(payload, compresslevel=NIFTI_GZIP_LEVEL, mtime=0)
    return atomic_write(path, lambda stream: stream.write(payload), exclusive=exclusive)
