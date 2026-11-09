"""Physical grids, exact cell overlap, trilinear sampling at world points, and the scribble's source representatives.

Conventions used everywhere in SIRB:

* Arrays are ZYX. Every affine maps an XYZ index to world millimetres (NiBabel's voxel-centre convention), so
  ``index_xyz_to_world(index_zyx[::-1], affine_xyz)`` is the centre of a voxel.
* Grid cells follow PyTorch's ``align_corners=False`` convention: cell ``i`` covers ``[i - 0.5, i + 0.5]``.
* The network works on a grid of fixed spacing (3 mm) whose cells are corner-aligned with the native PET grid and
  cover the whole native field of view (:func:`examination_grids`); a stride-2, kernel-3, padding-1 convolution
  keeps feature cell 0 on input cell 0 (:func:`feature_affine_from_conv`).
* Exact cell overlap is supported for orthogonal grids that share their axis directions up to a signed
  permutation; other rotations and shear raise :class:`GeometryError`.

``require`` and ``solve`` check validity either at once or, inside :func:`deferred_checks`, after one device read
at the end of an inference pass (a per-window check on a GPU tensor would wait for the GPU every time).
"""

from __future__ import annotations

import contextvars
import math
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
import torch
from scipy import ndimage
from scipy.sparse import csr_matrix
from torch.nn import functional as F

__all__ = ["CONNECTIVITY_18", "MAX_SOURCES", "CellOverlapOperator", "ExaminationGrids", "GeometryError",
           "SourceRepresentatives",
           "affine_xyz_from_zyx", "affine_zyx_from_xyz", "cell_overlap_operator", "deferred_checks",
           "examination_bounds_xyz", "examination_grids", "feature_affine_from_conv", "flip_affine_xyz", "floor_integer_boundary",
           "grid_world_points", "index_xyz_to_world", "nearest_cell_xyz", "require", "sample_world_features",
           "scribble_representatives", "sliding_window_origins", "solve", "spacing_xyz_mm", "support_contains_world",
           "transform_representatives", "triangle_window", "validate_affine_xyz", "voxel_volume_ml",
           "window_affine_xyz", "world_to_index_xyz"]

#: At most this many sources (groups of scribble voxels) per scribble.
MAX_SOURCES = 16
#: The 3D 18-connectivity of lesions and error components (faces and edges, not corners).
CONNECTIVITY_18 = ndimage.generate_binary_structure(3, 2)
#: 1 mL = 1000 mm^3.
MM3_PER_ML = 1000.0
#: Tie tolerance of squared physical distances when representatives are chosen and voxels are grouped.
TIE_ATOL_MM2 = 1e-9
TIE_RTOL = 1e-12
#: Tolerance of the cell-support test (index units) and of exact-integer ties after an affine solve.
SUPPORT_TOLERANCE = 1e-7
INTEGER_TIE_TOLERANCE = 1e-10
#: Native axes whose directions have |cos| above this are not orthogonal (3 mm grid planning).
AXIS_ORTHOGONALITY_ATOL = 1e-4
#: Guard against ceil() of a float that is an integer up to rounding noise.
_SHAPE_EPS = 1e-6


class GeometryError(ValueError):
    """A physical mapping cannot be represented by the requested operator."""


# ------------------------------------------------------------------------------------------- deferred checks
_ACTIVE_CHECKS = contextvars.ContextVar("sirb_deferred_checks", default=None)
#: torch.linalg.solve's own failure text, kept verbatim for the deferred singular-matrix check.
SINGULAR_SOLVE_MESSAGE = "torch.linalg.solve: The solver failed because the input matrix is singular."


class DeferredChecks:
    """Checks recorded during one pass, in the order they were made."""

    def __init__(self) -> None:
        self._items: list[tuple[torch.Tensor, type, object]] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, ok: torch.Tensor, error: type, message) -> None:
        """``ok``: a boolean tensor that must be all True; ``message``: text, or a callable made at raise time."""
        self._items.append((ok.detach().all(), error, message))

    def raise_first_failure(self) -> None:
        """One device read for every recorded check; raise the first failing one, else clear."""
        items, self._items = self._items, []
        if not items:
            return
        flags = torch.stack([ok for ok, _, _ in items]).cpu().tolist()
        for passed, (_, error, message) in zip(flags, items):
            if not passed:
                raise error(message() if callable(message) else message)


@contextmanager
def deferred_checks():
    """Record :func:`require` / :func:`solve` checks of this pass; read them with ``raise_first_failure``.

    If the pass stops with an ordinary error, a recorded check that had already failed is raised instead (with that
    error as its context), as the immediate check would have stopped the pass first.
    """
    checks = DeferredChecks()
    token = _ACTIVE_CHECKS.set(checks)
    try:
        yield checks
    except Exception:
        _ACTIVE_CHECKS.reset(token)
        token = None
        checks.raise_first_failure()
        raise
    finally:
        if token is not None:
            _ACTIVE_CHECKS.reset(token)


def require(ok: torch.Tensor, error: type, message) -> None:
    """Raise ``error(message)`` unless the boolean tensor ``ok`` is all True (deferred inside a pass)."""
    checks = _ACTIVE_CHECKS.get()
    if checks is None:
        if not bool(ok.all()):
            raise error(message() if callable(message) else message)
    else:
        checks.add(ok, error, message)


def solve(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``torch.linalg.solve(a, b)``; inside a pass the singular check waits for the pass's single read."""
    checks = _ACTIVE_CHECKS.get()
    if checks is None:
        return torch.linalg.solve(a, b)
    result, info = torch.linalg.solve_ex(a, b, check_errors=False)
    checks.add(info == 0, torch.linalg.LinAlgError, SINGULAR_SOLVE_MESSAGE)
    return result


# ------------------------------------------------------------------------------------------- affines
def validate_affine_xyz(affine) -> np.ndarray:
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    a = np.asarray(affine, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise GeometryError(f"affine_xyz actual={a.shape}, expected=finite (4,4)")
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise GeometryError(f"affine last row actual={a[3].tolist()}, expected=[0,0,0,1]")
    if abs(np.linalg.det(a[:3, :3])) < 1e-12:
        raise GeometryError("affine_xyz is singular")
    return a


def spacing_xyz_mm(affine) -> np.ndarray:
    return np.linalg.norm(validate_affine_xyz(affine)[:3, :3], axis=0)


def voxel_volume_ml(affine) -> float:
    """Volume of one voxel in mL: |det| of the 3 x 3 block / 1000 (valid for any rotation, not a product of
    spacings)."""
    return float(abs(np.linalg.det(validate_affine_xyz(affine)[:3, :3])) / MM3_PER_ML)


def affine_xyz_from_zyx(affine_zyx) -> np.ndarray:
    """The XYZ-index affine of an affine written for ZYX indices (and back: the permutation is its own inverse)."""
    return validate_affine_xyz(affine_zyx)[:, [2, 1, 0, 3]]


def affine_zyx_from_xyz(affine_xyz) -> np.ndarray:
    return validate_affine_xyz(affine_xyz)[:, [2, 1, 0, 3]]


def index_xyz_to_world(index_xyz, affine_xyz) -> np.ndarray:
    a = validate_affine_xyz(affine_xyz)
    p = np.asarray(index_xyz, dtype=np.float64)
    if p.shape[-1:] != (3,):
        raise GeometryError(f"index trailing shape actual={p.shape}, expected=3")
    return p @ a[:3, :3].T + a[:3, 3]


def world_to_index_xyz(world_xyz, affine_xyz) -> np.ndarray:
    a = validate_affine_xyz(affine_xyz)
    p = np.asarray(world_xyz, dtype=np.float64)
    if p.shape[-1:] != (3,):
        raise GeometryError(f"world trailing shape actual={p.shape}, expected=3")
    return np.linalg.solve(a[:3, :3], (p - a[:3, 3]).reshape(-1, 3).T).T.reshape(p.shape)


def floor_integer_boundary(value):
    """Floor with exact-integer tie semantics despite affine solve rounding."""
    value = np.asarray(value, dtype=np.float64)
    nearest = np.rint(value)
    return np.floor(np.where(np.abs(value - nearest) <= INTEGER_TIE_TOLERANCE, nearest, value)).astype(np.int64)


def nearest_cell_xyz(index_xyz):
    """Index of the nearest cell centre; half ties go up, independent of the grid's rotation."""
    return floor_integer_boundary(np.asarray(index_xyz) + .5)


def _shape_zyx(shape) -> tuple[int, int, int]:
    values = tuple(int(v) for v in shape)
    if len(values) != 3 or min(values) < 1:
        raise GeometryError(f"shape_zyx actual={values}, expected=three positive integers")
    return values


def _triple_zyx(value) -> tuple[float, float, float]:
    if np.isscalar(value):
        return (float(value),) * 3
    values = tuple(float(v) for v in value)
    if len(values) != 3:
        raise GeometryError(f"kernel/stride/padding actual={values}, expected=scalar or zyx triple")
    return values


def feature_affine_from_conv(input_affine_xyz, *, kernel_size=3, stride=2, padding=1, dilation=1):
    """Affine of a convolution's output: output centre j lies at input index ``stride*j - pad + dilation*(k-1)/2``.

    Tuple arguments follow Conv3d's ZYX order. One affine or a batch of affines; torch inputs keep device and dtype.
    """
    kernel = np.array(_triple_zyx(kernel_size)[::-1])
    steps = np.array(_triple_zyx(stride)[::-1])
    pads = np.array(_triple_zyx(padding)[::-1])
    dilations = np.array(_triple_zyx(dilation)[::-1])
    if np.any(kernel < 1) or np.any(steps <= 0) or np.any(dilations <= 0):
        raise GeometryError("kernel, stride and dilation must be positive")
    transform = np.eye(4)
    transform[:3, :3] = np.diag(steps)
    transform[:3, 3] = -pads + dilations * (kernel - 1) / 2
    if isinstance(input_affine_xyz, torch.Tensor):
        return input_affine_xyz @ torch.as_tensor(transform, device=input_affine_xyz.device,
                                                  dtype=input_affine_xyz.dtype)
    return np.asarray(input_affine_xyz, dtype=np.float64) @ transform


def window_affine_xyz(grid_affine_xyz, origin_zyx: Sequence[int]) -> np.ndarray:
    """Affine of the window of the 3 mm grid whose first cell is ``origin_zyx``."""
    a = validate_affine_xyz(grid_affine_xyz).copy()
    origin = np.asarray(tuple(origin_zyx)[::-1], dtype=np.float64)
    a[:3, 3] = index_xyz_to_world(origin, a)
    return a


def examination_bounds_xyz(shape_zyx, affine_xyz) -> np.ndarray:
    """[2, 3] lower and upper world corners (XYZ, mm) of the box around the native voxel cells."""
    corners = np.array(list(product(*[(-.5, n - .5) for n in tuple(shape_zyx)[::-1]])))
    world = index_xyz_to_world(corners, affine_xyz)
    return np.stack((world.min(0), world.max(0)))


def flip_affine_xyz(affine_xyz, shape_zyx, axis_zyx: int = 2) -> np.ndarray:
    """Affine of the array flipped along ``axis_zyx``: every voxel keeps its world position."""
    shape = _shape_zyx(shape_zyx)
    axis_xyz = 2 - int(axis_zyx)
    if axis_xyz not in (0, 1, 2):
        raise GeometryError("axis_zyx must be 0,1,2")
    transform = np.eye(4)
    transform[axis_xyz, axis_xyz] = -1
    transform[axis_xyz, 3] = shape[int(axis_zyx)] - 1
    return validate_affine_xyz(affine_xyz) @ transform


# ------------------------------------------------------------------------------------------- sampling
def grid_world_points(shape_zyx, affine_xyz, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """XYZ world coordinates of every cell centre, shaped ZYX3, or BZYX3 for a batch of affines."""
    shape = _shape_zyx(shape_zyx)
    affine = torch.as_tensor(affine_xyz, device=device, dtype=dtype)
    if affine.shape[-2:] != (4, 4) or affine.ndim not in (2, 3):
        raise GeometryError(f"affine tensor actual={tuple(affine.shape)}, expected=(4,4) or (B,4,4)")
    zz, yy, xx = torch.meshgrid(*(torch.arange(n, device=affine.device, dtype=dtype) for n in shape), indexing="ij")
    points = torch.stack((xx, yy, zz), dim=-1)
    if affine.ndim == 2:
        return points @ affine[:3, :3].T + affine[:3, 3]
    return torch.einsum("zyxi,bji->bzyxj", points, affine[:, :3, :3]) + affine[:, None, None, None, :3, 3]


def _batch_affine(affine, batch: int, device, dtype) -> torch.Tensor:
    value = torch.as_tensor(affine, device=device, dtype=dtype)
    if value.ndim == 2:
        value = value.unsqueeze(0).expand(batch, -1, -1)
    if tuple(value.shape) != (batch, 4, 4):
        raise GeometryError(f"affine batch actual={tuple(value.shape)}, expected={(batch, 4, 4)}")
    require(torch.isfinite(value), GeometryError, "non-finite feature/support affine")
    return value


def sample_world_features(feature: torch.Tensor, feature_affine_xyz, world_points_xyz, *, support_affine_xyz=None,
                          support_shape_zyx=None, support_valid_zyx=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Trilinear samples ``[B, N, C]`` of a feature map at world points, and their validity ``[B, N]``.

    ``feature`` is BCZYX; points are XYZ world millimetres, ``[N, 3]`` or ``[B, N, 3]``. A point is valid when it
    lies inside the support grid's cell extent (by default the feature grid itself) and, with
    ``support_valid_zyx``, in a valid support cell. Valid points near the input edge may lie beyond the last
    stride-reduced feature centre and read the border value. Invalid points are moved to zero before sampling and
    their samples are zero. Gradients flow to ``feature``.
    """
    if feature.ndim != 5 or not feature.is_floating_point():
        raise GeometryError(f"feature actual={tuple(feature.shape)}, expected=floating BCZYX")
    batch, channels = feature.shape[:2]
    points = torch.as_tensor(world_points_xyz, device=feature.device, dtype=torch.float64)
    if points.ndim == 2:
        points = points.unsqueeze(0).expand(batch, -1, -1)
    if points.ndim != 3 or points.shape[0] != batch or points.shape[-1] != 3:
        raise GeometryError(f"points actual={tuple(points.shape)}, expected=(N,3) or (B,N,3)")
    feature_affine = _batch_affine(feature_affine_xyz, batch, feature.device, points.dtype)
    if (support_affine_xyz is None) != (support_shape_zyx is None):
        raise GeometryError("support affine and shape must be supplied together")
    support_affine = _batch_affine(feature_affine_xyz if support_affine_xyz is None else support_affine_xyz, batch,
                                   feature.device, points.dtype)
    shape = _shape_zyx(feature.shape[2:] if support_shape_zyx is None else support_shape_zyx)
    finite = torch.isfinite(points).all(dim=-1)
    safe_points = torch.where(finite[..., None], points, torch.zeros_like(points))
    support_index = solve(support_affine[:, :3, :3],
                          (safe_points - support_affine[:, None, :3, 3]).transpose(1, 2)).transpose(1, 2)
    support_size_xyz = torch.tensor(shape[::-1], device=feature.device, dtype=points.dtype)
    valid = finite & ((support_index >= -.5 - SUPPORT_TOLERANCE)
                      & (support_index <= support_size_xyz - .5 + SUPPORT_TOLERANCE)).all(dim=-1)
    if support_valid_zyx is not None:
        mask = torch.as_tensor(support_valid_zyx, device=feature.device, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0).expand(batch, -1, -1, -1)
        if mask.ndim == 5 and mask.shape[1] == 1:
            mask = mask[:, 0]
        if tuple(mask.shape) != (batch, *shape):
            raise GeometryError(f"support_valid actual={tuple(mask.shape)}, expected={(batch, *shape)}")
        shifted = support_index + .5
        shifted = torch.where(torch.abs(shifted - torch.round(shifted)) <= INTEGER_TIE_TOLERANCE,
                              torch.round(shifted), shifted)
        pick = torch.floor(shifted).to(torch.long)
        pick = torch.maximum(pick, torch.zeros_like(pick))
        pick = torch.minimum(pick, support_size_xyz.to(torch.long) - 1)
        rows = torch.arange(batch, device=feature.device)[:, None]
        valid = valid & mask[rows, pick[..., 2], pick[..., 1], pick[..., 0]]
    index = solve(feature_affine[:, :3, :3],
                  (safe_points - feature_affine[:, None, :3, 3]).transpose(1, 2)).transpose(1, 2)
    index = torch.where(valid[..., None], index, torch.zeros_like(index))
    size = torch.tensor(feature.shape[2:][::-1], device=feature.device, dtype=points.dtype)
    grid = (2 * (index + .5) / size - 1).to(feature.dtype)[:, :, None, None, :]
    sampled = F.grid_sample(feature, grid, mode="bilinear", padding_mode="border",
                            align_corners=False)[:, :, :, 0, 0].transpose(1, 2)
    return torch.where(valid[..., None], sampled, torch.zeros_like(sampled)), valid


def support_contains_world(world_xyz, affine_xyz, shape_zyx, valid_zyx=None) -> np.ndarray:
    """Whether world points lie inside a grid's cell extent (and, with ``valid_zyx``, in a valid cell)."""
    points = np.asarray(world_xyz, dtype=np.float64)
    finite = np.isfinite(points).all(axis=-1)
    safe = np.where(finite[..., None], points, 0)
    u = world_to_index_xyz(safe, affine_xyz)
    shape = np.array(_shape_zyx(shape_zyx)[::-1])
    valid = finite & ((u >= -.5 - SUPPORT_TOLERANCE) & (u <= shape - .5 + SUPPORT_TOLERANCE)).all(axis=-1)
    if valid_zyx is not None:
        mask = np.asarray(valid_zyx, dtype=bool)
        if tuple(mask.shape) != tuple(shape[::-1]):
            raise GeometryError("case valid mask has a different shape")
        pick = np.clip(nearest_cell_xyz(u), 0, shape - 1)
        valid &= mask[pick[..., 2], pick[..., 1], pick[..., 0]]
    return valid


# ------------------------------------------------------------------------------------------- exact cell overlap
def _orthogonal_directions(affine_xyz):
    a = validate_affine_xyz(affine_xyz)
    spacing = np.linalg.norm(a[:3, :3], axis=0)
    directions = a[:3, :3] / spacing
    gram = directions.T @ directions
    if not np.allclose(gram, np.eye(3), atol=1e-6, rtol=0):
        raise GeometryError(f"shear/non-orthogonal affine Gram actual={gram.tolist()}, expected=identity")
    return a, spacing, directions


def _axis_interval_overlap(native_centres, native_spacing, lows, highs, denominators):
    """CSR rows of native-cell intersection length / ``denominators[row]``.

    Target cells are explicit intervals in the common direction basis. Only a positive-length intersection (beyond
    float tolerance) is stored, so a hit always means positive physical volume and never a shared face.
    """
    order = np.argsort(native_centres)
    left = native_centres[order] - native_spacing / 2
    right = native_centres[order] + native_spacing / 2
    rows, columns, values = [], [], []
    widths = np.asarray(highs) - np.asarray(lows)
    tolerance = 1e-10 * max(native_spacing, float(np.max(np.abs(widths), initial=0)), 1)
    for row, (low, high) in enumerate(zip(lows, highs)):
        start = int(np.searchsorted(right, low, side="right"))
        stop = int(np.searchsorted(left, high, side="left"))
        for sorted_column in range(start, stop):
            overlap = min(right[sorted_column], high) - max(left[sorted_column], low)
            if overlap > tolerance:
                rows.append(row)
                columns.append(int(order[sorted_column]))
                values.append(overlap / denominators[row])
    return csr_matrix((values, (rows, columns)), shape=(len(lows), len(native_centres)), dtype=np.float64)


def _axis_overlap(native_centres, native_spacing, target_centres, target_spacing):
    """CSR rows of cell-intersection length / FULL target-cell length."""
    target_centres = np.asarray(target_centres, dtype=np.float64)
    return _axis_interval_overlap(native_centres, native_spacing, target_centres - target_spacing / 2,
                                  target_centres + target_spacing / 2, np.full(len(target_centres), target_spacing))


@dataclass(frozen=True)
class CellOverlapOperator:
    """Three sparse one-dimensional overlap operators (target cell x native cell volume fractions).

    Each axis keeps only the contiguous native index range with a nonzero weight, so a window-sized target reads a
    native box, not the whole examination. Dropped columns are all zero; results are identical.
    """
    source_axes_for_target_xyz: tuple[int, int, int]
    weights_xyz: tuple[csr_matrix, csr_matrix, csr_matrix]
    native_shape_zyx: tuple[int, int, int]
    target_shape_zyx: tuple[int, int, int]
    source_ranges_xyz: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None

    @classmethod
    def cropped(cls, source_axes, weights, native_shape, target_shape):
        ranges, kept = [], []
        for weight in weights:
            if weight.nnz:
                low, high = int(weight.indices.min()), int(weight.indices.max()) + 1
            else:
                low = high = 0
            ranges.append((low, high))
            kept.append(csr_matrix(weight[:, low:high]))
        return cls(tuple(source_axes), tuple(kept), tuple(native_shape), tuple(target_shape), tuple(ranges))

    @property
    def nbytes(self):
        return sum(w.data.nbytes + w.indices.nbytes + w.indptr.nbytes for w in self.weights_xyz)

    def apply(self, native_zyx) -> np.ndarray:
        """Volume fraction of each target cell covered by the native array (FP64, NumPy)."""
        array = np.asarray(native_zyx)
        if tuple(array.shape) != self.native_shape_zyx:
            raise GeometryError(f"source shape actual={array.shape}, expected={self.native_shape_zyx}")
        array = np.transpose(np.transpose(array, (2, 1, 0)), self.source_axes_for_target_xyz)
        if self.source_ranges_xyz is not None:
            if any(high <= low for low, high in self.source_ranges_xyz):
                return np.zeros(self.target_shape_zyx, dtype=np.float64)
            array = array[tuple(slice(low, high) for low, high in self.source_ranges_xyz)]
        array = array.astype(np.float64, copy=False)
        for axis, weight in enumerate(self.weights_xyz):
            moved = np.moveaxis(array, axis, 0)
            shape = (weight.shape[0], *moved.shape[1:])
            array = np.moveaxis(weight.dot(moved.reshape(moved.shape[0], -1)).reshape(shape), 0, axis)
        return np.transpose(array, (2, 1, 0))


class OverlapWeightCache:
    """Least-recently-used cache of overlap operators, capped in sparse-weight bytes (256 MiB)."""

    def __init__(self, max_bytes=256 * 1024 ** 2):
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.max_bytes = int(max_bytes)
        self._weights = OrderedDict()
        self.bytes = 0

    def get(self, key):
        value = self._weights.get(key)
        if value is not None:
            self._weights.move_to_end(key)
        return value

    def put(self, key, value):
        if key in self._weights:
            self.bytes -= self._weights.pop(key).nbytes
        if value.nbytes > self.max_bytes:
            return
        while self._weights and self.bytes + value.nbytes > self.max_bytes:
            self.bytes -= self._weights.popitem(last=False)[1].nbytes
        self._weights[key] = value
        self.bytes += value.nbytes


_OVERLAP_WEIGHTS = OverlapWeightCache()


@dataclass(frozen=True)
class _CommonBasis:
    """Native axes expressed along a target frame's orthogonal directions."""
    source_axes: tuple[int, int, int]
    signs: tuple[float, float, float]
    native_spacing_xyz: np.ndarray
    target_spacing_xyz: np.ndarray
    native_origin: np.ndarray
    target_origin: np.ndarray

    def native_centres(self, target_axis, native_shape_zyx):
        source_axis = self.source_axes[target_axis]
        count = _shape_zyx(native_shape_zyx)[::-1][source_axis]
        return (self.native_origin[target_axis] + self.signs[target_axis]
                * self.native_spacing_xyz[source_axis] * np.arange(count))


def _common_basis(native_affine_xyz, target_affine_xyz) -> _CommonBasis:
    na, ns, nd = _orthogonal_directions(native_affine_xyz)
    ta, ts, td = _orthogonal_directions(target_affine_xyz)
    orientation = td.T @ nd
    source_axes = tuple(int(v) for v in np.argmax(np.abs(orientation), axis=1))
    permutation = np.zeros((3, 3))
    for target_axis, source_axis in enumerate(source_axes):
        permutation[target_axis, source_axis] = np.sign(orientation[target_axis, source_axis])
    if len(set(source_axes)) != 3 or not np.allclose(orientation, permutation, atol=1e-6, rtol=0):
        raise GeometryError(f"relative rotation unsupported: native-to-target directions actual={orientation.tolist()}, "
                            "expected=signed permutation; no approximate occupancy fallback")
    signs = tuple(float(permutation[axis, source]) for axis, source in enumerate(source_axes))
    return _CommonBasis(source_axes, signs, ns, ts, td.T @ na[:3, 3], td.T @ ta[:3, 3])


def cell_overlap_operator(native_affine_xyz, native_shape_zyx, target_affine_xyz, target_shape_zyx, *,
                          cache=_OVERLAP_WEIGHTS) -> CellOverlapOperator:
    """The exact volume overlap of native cells with the cells of a target grid (:class:`CellOverlapOperator`)."""
    native_affine, target_affine = validate_affine_xyz(native_affine_xyz), validate_affine_xyz(target_affine_xyz)
    native_shape = _shape_zyx(native_shape_zyx)
    target_shape = _shape_zyx(target_shape_zyx)
    key = (native_shape, tuple(native_affine.ravel()), target_shape, tuple(target_affine.ravel()), "orthogonal-cell")
    cached = None if cache is None else cache.get(key)
    if cached is not None:
        return cached
    basis = _common_basis(native_affine, target_affine)
    weights = []
    for axis, source_axis in enumerate(basis.source_axes):
        target_centres = basis.target_origin[axis] + basis.target_spacing_xyz[axis] * np.arange(target_shape[::-1][axis])
        weights.append(_axis_overlap(basis.native_centres(axis, native_shape), basis.native_spacing_xyz[source_axis],
                                     target_centres, basis.target_spacing_xyz[axis]))
    operator = CellOverlapOperator.cropped(basis.source_axes, weights, native_shape, target_shape)
    if cache is not None:
        cache.put(key, operator)
    return operator


# ------------------------------------------------------------------------------------------- windows
def triangle_window(window_size: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Separable triangular weights of one window (its overlapping predictions are merged with them)."""
    p = int(window_size)
    if p < 2 or p % 2:
        raise GeometryError(f"window_size actual={p}, expected=even integer >=2")
    axis = 1 - torch.abs(2 * (torch.arange(p, device=device, dtype=dtype) + .5) / p - 1)
    return axis[:, None, None] * axis[None, :, None] * axis[None, None, :]


def sliding_window_origins(shape_zyx, window_size: int) -> tuple[tuple[int, int, int], ...]:
    """Window origins with stride ``window_size / 2`` and a final window at each end (no ground truth or scribble)."""
    p = int(window_size)
    if p < 2 or p % 2:
        raise GeometryError(f"window_size actual={p}, expected=positive even integer")
    axes = []
    for n in _shape_zyx(shape_zyx):
        last = max(0, n - p)
        starts = list(range(0, last + 1, p // 2))
        if starts[-1] != last:
            starts.append(last)
        axes.append(starts)
    return tuple(product(*axes))


# ------------------------------------------------------------------------------------------- 3 mm grid
@dataclass(frozen=True)
class ExaminationGrids:
    """The 3 mm grid and the whole-volume grid of one examination (they depend on its header only).

    ``grid_shape_zyx`` covers the native field of view; ``padded_shape_zyx`` pads it up to one window per axis. The
    whole-volume cells cover exactly the field of view (padding excluded).
    """
    grid_affine_xyz: np.ndarray
    grid_shape_zyx: tuple[int, int, int]
    padded_shape_zyx: tuple[int, int, int]
    whole_volume_affine_xyz: np.ndarray
    whole_volume_shape_zyx: tuple[int, int, int]


def _validate_grid_affine(affine) -> np.ndarray:
    a = np.asarray(affine, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise GeometryError("affine must be a finite 4x4 matrix")
    if not np.allclose(a[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
        raise GeometryError(f"affine last row must be [0,0,0,1], got {a[3].tolist()}")
    determinant = float(np.linalg.det(a[:3, :3]))
    if abs(determinant) < 1e-12:
        raise GeometryError(f"singular voxel-to-world affine, det={determinant}")
    return a


def _plan_grid(native_shape, native_affine, spacing_mm, min_shape):
    """(3 mm grid affine, field-of-view shape, padded shape), all in the native array's axis order.

    The grid keeps the native axis directions and is corner-aligned with the native grid, so along each axis
    ``(n + 0.5) * s_native = (r + 0.5) * s_grid``; it covers the whole native field of view and is padded only up
    to ``min_shape``.
    """
    shape = tuple(int(v) for v in native_shape)
    if len(shape) != 3 or min(shape) < 1:
        raise GeometryError(f"native shape must be three positive ints, got {shape}")
    a = _validate_grid_affine(native_affine)
    target = tuple(float(v) for v in spacing_mm)
    if len(target) != 3 or min(target) <= 0 or not np.isfinite(target).all():
        raise GeometryError(f"grid spacing must be three positive numbers, got {target}")
    native_spacing = tuple(float(np.linalg.norm(a[:3, k])) for k in range(3))
    directions = [a[:3, k] / native_spacing[k] for k in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            cosine = float(abs(np.dot(directions[i], directions[j])))
            if cosine > AXIS_ORTHOGONALITY_ATOL:
                raise GeometryError(f"native axes {i},{j} are not orthogonal: |cos|={cosine:.3e} "
                                    f"allowed={AXIS_ORTHOGONALITY_ATOL:.1e}")
    scale = tuple(target[k] / native_spacing[k] for k in range(3))
    fov = tuple(int(math.ceil(shape[k] * native_spacing[k] / target[k] - _SHAPE_EPS)) for k in range(3))
    fov = tuple(max(1, v) for v in fov)
    padded = tuple(max(fov[k], int(min_shape[k])) for k in range(3))
    grid = np.eye(4)
    for k in range(3):
        grid[:3, k] = directions[k] * target[k]
    first_centre_native = np.array([0.5 * scale[k] - 0.5 for k in range(3)])
    grid[:3, 3] = a[:3, 3] + a[:3, :3] @ first_centre_native
    return grid, fov, padded


def examination_grids(native_shape_zyx, native_affine_xyz, window_size: int, *, grid_spacing_mm=(3., 3., 3.),
                      whole_volume_shape_zyx=(96, 64, 64)) -> ExaminationGrids:
    """The 3 mm grid (corner-aligned with the native grid, padded to one window) and the whole-volume grid."""
    grid_zyx, fov, padded = _plan_grid(native_shape_zyx, affine_zyx_from_xyz(native_affine_xyz), grid_spacing_mm,
                                       (int(window_size),) * 3)
    grid = affine_xyz_from_zyx(grid_zyx)
    whole_shape = _shape_zyx(whole_volume_shape_zyx)
    transform = np.eye(4)
    factor = np.asarray(fov[::-1]) / np.asarray(whole_shape[::-1])
    transform[:3, :3] = np.diag(factor)
    transform[:3, 3] = .5 * factor - .5
    return ExaminationGrids(grid, tuple(fov), tuple(padded), grid @ transform, whole_shape)


# ------------------------------------------------------------------------------------------- source representatives
@dataclass(frozen=True)
class SourceRepresentatives:
    """The representatives of a scribble's sources and the scribble window around the first one.

    Slot j of ``world_xyz_mm`` / ``native_index_zyx`` / ``source_valid`` is source j; at most :data:`MAX_SOURCES`.
    The scribble window is the window of the 3 mm grid centred on the first representative; its level-2 features
    are pooled over the source voxels inside it.
    """
    world_xyz_mm: np.ndarray
    native_index_zyx: np.ndarray
    source_valid: np.ndarray
    window_origin_zyx: tuple[int, int, int]
    window_affine_xyz: np.ndarray
    window_valid: np.ndarray


def scribble_representatives(scribble_zyx, native_affine_xyz, grid_affine_xyz, window_size: int, *,
                             max_sources=MAX_SOURCES, native_valid_zyx=None) -> SourceRepresentatives:
    """Representatives of a non-empty scribble: the voxel closest to its centroid, then farthest-point sampling.

    Distances are physical (world millimetres); ties go to the first voxel in ``np.argwhere`` (ZYX) order.
    """
    scribble = np.asarray(scribble_zyx, dtype=bool)
    candidates = np.argwhere(scribble)  # np.argwhere's ZYX order is the tie order.
    if not len(candidates):
        raise GeometryError("source representatives require a non-empty scribble")
    if native_valid_zyx is not None and not np.asarray(native_valid_zyx, dtype=bool)[tuple(candidates.T)].all():
        raise GeometryError("the scribble includes invalid native voxels")
    count = int(max_sources)
    if count < 1:
        raise GeometryError("max_sources must be positive")
    world = index_xyz_to_world(candidates[:, ::-1], native_affine_xyz)
    distances = ((world - world.mean(axis=0)) ** 2).sum(axis=1)
    first = int(np.flatnonzero(np.isclose(distances, distances.min(), atol=TIE_ATOL_MM2, rtol=TIE_RTOL))[0])
    selected = [first]
    minimum = ((world - world[first]) ** 2).sum(axis=1)
    while len(selected) < min(count, len(candidates)):
        minimum[selected] = -1
        maximum = minimum.max()
        chosen = int(np.flatnonzero(np.isclose(minimum, maximum, atol=TIE_ATOL_MM2, rtol=TIE_RTOL))[0])
        selected.append(chosen)
        minimum = np.minimum(minimum, ((world - world[chosen]) ** 2).sum(axis=1))
    source_world = np.zeros((count, 3), dtype=np.float64)
    source_indices = np.zeros((count, 3), dtype=np.int64)
    valid = np.arange(count) < len(selected)
    source_world[valid] = world[selected]
    source_indices[valid] = candidates[selected]
    first_u = world_to_index_xyz(world[first], grid_affine_xyz)
    origin_xyz = floor_integer_boundary(first_u - (int(window_size) - 1) / 2 + .5)
    origin_zyx = tuple(int(v) for v in origin_xyz[::-1])
    window = window_affine_xyz(grid_affine_xyz, origin_zyx)
    window_valid = valid & support_contains_world(source_world, window, (int(window_size),) * 3)
    return SourceRepresentatives(source_world, source_indices, valid, origin_zyx, window, window_valid)


def transform_representatives(representatives: SourceRepresentatives, native_shape_zyx, window_size: int, *,
                              flip_lr: bool = False) -> SourceRepresentatives:
    """Representatives of a left-right flipped state: the same voxels, never selected again.

    The flip changes native indices and the scribble window's axis; world positions stay, because the affines flip
    too.
    """
    if not flip_lr:
        return representatives
    indices = representatives.native_index_zyx.copy()
    valid = representatives.source_valid
    indices[valid, 2] = int(native_shape_zyx[2]) - 1 - indices[valid, 2]
    return SourceRepresentatives(representatives.world_xyz_mm.copy(), indices, valid.copy(),
                                 representatives.window_origin_zyx,
                                 flip_affine_xyz(representatives.window_affine_xyz, (int(window_size),) * 3),
                                 representatives.window_valid.copy())
