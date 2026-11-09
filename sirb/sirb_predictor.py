"""SIRBPredictor: correct the mask of one examination with SIRB, one scribble at a time.

Usage, in the style of SAM 2's predictors::

    predictor = SIRBPredictor.from_config(config, "checkpoints/sirb_psma.pt", mode="standard", device="cuda")
    predictor.set_examination(Examination(pet_zyx, ct_zyx, affine_xyz, initial_segmentation_zyx))
    prediction = predictor.predict(scribble_zyx, "addition")   # prediction.updated_mask is the new current mask
    predictor.reset()                                          # back to the initial segmentation

``predict`` keeps the interaction history of the examination (earlier scribbles, previous mask);
:meth:`SIRBPredictor.predict_state` is the low-level call on one :class:`~sirb.data.observations.State`, used by
``sirb.interaction.session``, which builds the states itself.

One prediction with every window (the computation of the paper; ``local_update`` below restricts step 2):

1. The scribble's sources: representatives and grouping (``sirb.data.geometry``, ``sirb.modeling.source_encoder``).
2. For each orientation (the examination as it is and, with ``flip_average``, flipped left-right): the state context
   (whole-volume grid, scribble window, sources); the 14 input channels of the 3 mm grid; per editable native voxel
   its scribble distance and neighbourhood code; then every window of the 3 mm grid (stride half a window) that holds
   editable valid native voxels, ``window_batch`` windows per forward pass, and native refinement of their editable
   voxels, ``voxel_chunk`` voxels at a time. The windows' native role logits are merged with triangular weights.
3. The two orientations' logits are averaged, one softmax gives the target probability, and the update rule
   (``sirb.interaction.update_rule``) applies it.

Everything is computed on the predictor's device; the left-right flipped orientation flips the arrays on the device,
reuses the sources and flips only the scribble window's affine. The 0.5 decision is taken on the device, and
editable voxels whose probability lies within ``RECHECK_BAND`` of the threshold are decided again with the CPU
softmax, so every decision equals the CPU decision.

Two inference modes (:data:`MODES`), chosen with ``from_config(..., mode=...)``:

* ``"standard"``: ``local_update`` on, ``flip_average`` on, FP32, no compile.
* ``"fast"``: ``local_update`` on, ``flip_average`` off, bfloat16, ``torch.compile``.

Both compute only near the scribble and apply only the change components connected to it. The options below can also
be set one by one, in a configuration's ``predictor`` section or as keywords; ``local_update=False`` computes every
window of the examination.

Options (they change outputs and are reported with them):

* ``flip_average``: average the logits of both orientations; off runs one pass.
* ``local_update``: run first only the windows near the scribble (the world box of its 3^3 dilation,
  the seed voxels, widened by ``near_margin_mm``), decide only the voxels whose covering windows have all run, and
  apply only the accepted change components (18-connected) that touch a seed voxel. Windows are added while a kept
  component touches an undecided editable voxel or a seed voxel is undecided, and all windows run once more than
  ``LOCAL_FALLBACK_FRACTION`` of them were needed. The result is the full result restricted to the scribble-connected
  change components.
* ``precision``: ``"fp32"``, or ``"bf16"`` (the window forward pass under bfloat16 autocast; the state context,
  neighbourhood codes, native refinement, sampling and merging stay FP32).
* ``compile``: ``torch.compile`` of the local encoder, the role decoder and native refinement. They are compiled on
  a copy of the model; the given model is only moved to the device and set to evaluation mode, as without compile.

``window_batch`` and ``voxel_chunk`` set memory and time, but a window's outputs can change in the last bits with the
batch it runs in (on the CPU too), and matrix kernels can depend on the number of voxels per chunk on a GPU, so
reproducing a run needs the same values. The local update therefore runs windows only in the batches of the full
computation. The predictor reads no file and no environment variable; runtime switches such as TF32 belong to the
caller.
"""

from __future__ import annotations

import copy
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from .build_sirb import build_sirb, model_settings
from .data import geometry, observations
from .data.observations import Augmentation, DeviceState, State
from .data.roles import ROLE_COUNT, TARGET_ERROR, correction_sign
from .interaction.session import Examination, InteractionHistory
from .interaction.update_rule import TARGET_THRESHOLD, accept, apply, editable_domain
from .modeling.native_refinement import FEATURE_WIDTH
from .modeling.sirb import SIRB
from .modeling.source_encoder import group_scribble

__all__ = ["DEFAULT_MODE", "LOCAL_FALLBACK_FRACTION", "MODES", "PRECISIONS", "PREDICTOR_OPTIONS", "RECHECK_BAND",
           "Prediction", "SIRBPredictor"]

PRECISIONS = ("fp32", "bf16")
#: Keyword options of SIRBPredictor that a configuration's ``predictor`` section may set.
PREDICTOR_OPTIONS = ("local_update", "flip_average", "precision", "compile", "near_margin_mm", "window_batch",
                     "voxel_chunk")
#: The inference modes of :meth:`SIRBPredictor.from_config`: the options each one sets.
MODES = {
    "standard": {"local_update": True, "flip_average": True, "precision": "fp32", "compile": False,
                 "near_margin_mm": 0.0},
    "fast": {"local_update": True, "flip_average": False, "precision": "bf16", "compile": True,
             "near_margin_mm": 0.0},
}
DEFAULT_MODE = "standard"
#: Device probabilities this close to TARGET_THRESHOLD are decided again with the CPU softmax.
RECHECK_BAND = 1e-5
#: Local update: once more than this fraction of the windows (both orientations) was needed, all windows run.
LOCAL_FALLBACK_FRACTION = 0.5
#: The three associations of a three-term floating-point sum (x, y, z components).
SUM_ORDERS = ("xy_z", "xz_y", "yz_x")
#: Grid points per distance volume compared with the brute-force distance.
DISTANCE_SPOT_CHECK = 512
#: Seed voxels of the local update: the 3^3 dilation of the scribble.
SEED_STRUCTURE = np.ones((3, 3, 3), dtype=bool)
#: Columns of the window outputs sampled at native voxels: decoder features, coarse role logits, triangle weight.
_COARSE = slice(FEATURE_WIDTH, FEATURE_WIDTH + ROLE_COUNT)
_WEIGHT = FEATURE_WIDTH + ROLE_COUNT


# ------------------------------------------------------------------------------------------- scribble distance
def combine_squares(order: str, x, y, z):
    """The three-term sum ``x + y + z`` in one association (the operands broadcast)."""
    if order == "xy_z":
        return (x + y) + z
    if order == "xz_y":
        return (x + z) + y
    if order == "yz_x":
        return x + (y + z)
    raise ValueError(f"sum order must be one of {SUM_ORDERS}, got {order!r}")


def detect_sum_order(device, *, shapes=((4096, 37, 3), (97, 513, 3), (1, 1000, 3), (65536, 3, 3)), seed=7):
    """The association that ``(d * d).sum(-1)`` uses for a contiguous last dimension of three on ``device``.

    Returns one of SUM_ORDERS when exactly one association reproduces every probe element bit for bit, else None
    (the distance then keeps the brute force).
    """
    generator = torch.Generator().manual_seed(seed)
    possible = set(SUM_ORDERS)
    for shape in shapes:
        mantissa = torch.rand(shape, generator=generator, dtype=torch.float64) * 2 - 1
        scale = 10.0 ** torch.randint(-4, 5, shape, generator=generator).to(torch.float64)
        d = (mantissa * scale).to(device)
        reference = (d * d).sum(-1)
        squares = d * d
        possible &= {order for order in SUM_ORDERS
                     if torch.equal(combine_squares(order, squares[..., 0], squares[..., 1], squares[..., 2]),
                                    reference)}
    return next(iter(possible)) if len(possible) == 1 else None


def axis_aligned(affine_xyz) -> bool:
    """True when every off-diagonal entry of the 3 x 3 block is exactly zero (no rotation, no axis permutation)."""
    a = np.asarray(affine_xyz, dtype=np.float64)[:3, :3]
    return bool(np.all(a[~np.eye(3, dtype=bool)] == 0))


def axis_world(shape_zyx, affine_xyz, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """World coordinate of every index along each axis, (x[X], y[Y], z[Z]), computed as ``grid_world_points`` does.

    With zero off-diagonal entries each world coordinate depends on its own index only, so three lines give the same
    FP64 values as the full grid.
    """
    affine = torch.as_tensor(affine_xyz, device=device, dtype=torch.float64)
    lines = []
    for axis, count in enumerate(tuple(shape_zyx)[::-1]):
        line = torch.zeros((count, 3), dtype=torch.float64, device=device)
        line[:, axis] = torch.arange(count, dtype=torch.float64, device=device)
        lines.append((line @ affine[:3, :3].T + affine[:3, 3])[:, axis].contiguous())
    return tuple(lines)


def nearest_scribble_distance_axes(axes_xyz, scribble_world: torch.Tensor, order: str) -> torch.Tensor:
    """FP64 Euclidean distance (Z, Y, X) from every point of an axis-aligned grid to its nearest scribble voxel.

    Scribble voxels are taken row by row (same y and z world values), where only dx varies: a sum is non-decreasing
    in each term for every association, so the row minimum of dx^2 gives the row minimum of the whole sum, and sqrt
    is monotone and correctly rounded, so the minimum of the square roots is the square root of the minimum. The
    result equals the brute force of ``observations.normalized_scribble_distance`` bit for bit.
    """
    gx, gy, gz = axes_xyz
    rows, inverse = torch.unique(scribble_world[:, 1:], dim=0, return_inverse=True)
    best = None
    for row in range(rows.shape[0]):
        sx = scribble_world[inverse == row, 0]
        dx = gx[:, None] - sx[None, :]
        tx = (dx * dx).min(dim=1).values
        dy = gy - rows[row, 0]
        dz = gz - rows[row, 1]
        d2 = combine_squares(order, tx[None, None, :], (dy * dy)[None, :, None], (dz * dz)[:, None, None])
        best = d2 if best is None else torch.minimum(best, d2)
    return torch.sqrt(best)


def normalised_distance(nearest: torch.Tensor, native_shape_zyx, native_affine_xyz) -> torch.Tensor:
    """log1p(d / 3 mm) / log1p(examination diagonal / 3 mm) of an FP64 distance, as FP32 (the formula of
    ``observations.normalized_scribble_distance``)."""
    diagonal = float(np.linalg.norm(np.array(tuple(native_shape_zyx)[::-1])
                                    * geometry.spacing_xyz_mm(native_affine_xyz)))
    value = torch.log1p(nearest / observations.DISTANCE_SCALE_MM) / math.log1p(diagonal / observations.DISTANCE_SCALE_MM)
    return value.to(torch.float32)


# ------------------------------------------------------------------------------------------- positive overlap
def support_matrices(operator, device):
    """0/1 FP32 matrices of a cell-overlap operator's stored positive weights (None when an axis range is empty)."""
    if operator.source_ranges_xyz is not None and any(high <= low for low, high in operator.source_ranges_xyz):
        return None
    return tuple(torch.as_tensor((weight.toarray() > 0).astype(np.float32), device=device)
                 for weight in operator.weights_xyz)


def positive_hits(operator, supports, volumes: torch.Tensor) -> torch.Tensor:
    """``observations.apply_overlap(operator, v) > 0`` for each boolean volume of ``volumes`` (C, Z, Y, X) at once.

    Stored overlap weights are positive and their products far above the FP64 underflow, so the overlap is positive
    exactly where some voxel of the stored support is set; the 0/1 products here are small integers, exact in FP32
    (and TF32).
    """
    count = volumes.shape[0]
    if supports is None:
        return torch.zeros((count, *operator.target_shape_zyx), dtype=torch.bool, device=volumes.device)
    array = volumes.permute(0, 3, 2, 1)
    array = array.permute(0, *(1 + int(axis) for axis in operator.source_axes_for_target_xyz))
    if operator.source_ranges_xyz is not None:
        array = array[(slice(None), *(slice(low, high) for low, high in operator.source_ranges_xyz))]
    array = array.to(torch.float32)
    for axis, support in enumerate(supports):
        array = torch.movedim(torch.tensordot(support, array, dims=([1], [axis + 1])), 0, axis + 1)
    return array.permute(0, 3, 2, 1) > 0


# ------------------------------------------------------------------------------------------- sampling
def sample_at_points(feature: torch.Tensor, feature_affine_xyz: torch.Tensor, world_xyz: torch.Tensor, *,
                     support_affine_xyz: torch.Tensor, support_size_xyz: torch.Tensor, support_size_long: torch.Tensor,
                     feature_size_xyz: torch.Tensor, support_valid_zyx: torch.Tensor):
    """``geometry.sample_world_features`` with its size tensors passed in, so a window loop copies nothing from the
    host. Every operation, its order and precision are those of that function."""
    batch = feature.shape[0]
    points = torch.as_tensor(world_xyz, device=feature.device, dtype=torch.float64)
    if points.ndim == 2:
        points = points.unsqueeze(0).expand(batch, -1, -1)

    def batched(affine):
        value = torch.as_tensor(affine, device=feature.device, dtype=points.dtype)
        if value.ndim == 2:
            value = value.unsqueeze(0).expand(batch, -1, -1)
        geometry.require(torch.isfinite(value), geometry.GeometryError, "non-finite feature/support affine")
        return value
    feature_affine, support_affine = batched(feature_affine_xyz), batched(support_affine_xyz)
    finite = torch.isfinite(points).all(dim=-1)
    safe_points = torch.where(finite[..., None], points, torch.zeros_like(points))
    support_index = geometry.solve(support_affine[:, :3, :3],
                                   (safe_points - support_affine[:, None, :3, 3]).transpose(1, 2)).transpose(1, 2)
    valid = finite & ((support_index >= -.5 - geometry.SUPPORT_TOLERANCE)
                      & (support_index <= support_size_xyz - .5 + geometry.SUPPORT_TOLERANCE)).all(dim=-1)
    mask = support_valid_zyx
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).expand(batch, -1, -1, -1)
    if mask.ndim == 5 and mask.shape[1] == 1:
        mask = mask[:, 0]
    shifted = support_index + .5
    shifted = torch.where(torch.abs(shifted - torch.round(shifted)) <= geometry.INTEGER_TIE_TOLERANCE,
                          torch.round(shifted), shifted)
    pick = torch.floor(shifted).to(torch.long)
    pick = torch.maximum(pick, torch.zeros_like(pick))
    pick = torch.minimum(pick, support_size_long - 1)
    rows = torch.arange(batch, device=feature.device)[:, None]
    valid = valid & mask[rows, pick[..., 2], pick[..., 1], pick[..., 0]]
    index = geometry.solve(feature_affine[:, :3, :3],
                           (safe_points - feature_affine[:, None, :3, 3]).transpose(1, 2)).transpose(1, 2)
    index = torch.where(valid[..., None], index, torch.zeros_like(index))
    grid = (2 * (index + .5) / feature_size_xyz - 1).to(feature.dtype)[:, :, None, None, :]
    sampled = F.grid_sample(feature, grid, mode="bilinear", padding_mode="border",
                            align_corners=False)[:, :, :, 0, 0].transpose(1, 2)
    return torch.where(valid[..., None], sampled, torch.zeros_like(sampled)), valid


# ------------------------------------------------------------------------------------------- local update rules
def seed_voxels(scribble_zyx) -> np.ndarray:
    """The 3^3 dilation of the scribble: a change component counts as touching the scribble through these voxels."""
    return ndimage.binary_dilation(np.asarray(scribble_zyx, dtype=bool), structure=SEED_STRUCTURE)


def world_box(affine_xyz, voxels_zyx, margin_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) XYZ world corners of the voxel centres of ``voxels_zyx``, widened by ``margin_mm``."""
    index = np.argwhere(voxels_zyx)
    affine = np.asarray(affine_xyz)
    world = index[:, ::-1] @ affine[:3, :3].T + affine[:3, 3]
    return world.min(axis=0) - margin_mm, world.max(axis=0) + margin_mm


def scribble_connected(accepted: torch.Tensor, scribble: torch.Tensor) -> torch.Tensor:
    """The accepted change components (18-connected) that contain or touch (3^3) a scribble voxel."""
    if not bool(accepted.any()):
        return torch.zeros_like(accepted)
    seeds = F.max_pool3d(scribble[None, None].to(torch.float32), 3, stride=1, padding=1)[0, 0] > 0
    found = torch.nonzero(accepted | seeds)
    low = torch.clamp(found.min(dim=0).values - 1, min=0)
    high = found.max(dim=0).values + 2
    lz, ly, lx = (int(v) for v in low.tolist())
    hz, hy, hx = (int(v) for v in high.tolist())
    crop = accepted[lz:hz, ly:hy, lx:hx].cpu().numpy()
    seed_crop = seeds[lz:hz, ly:hy, lx:hx].cpu().numpy()
    labels, _ = ndimage.label(crop, structure=geometry.CONNECTIVITY_18)
    keep = np.unique(labels[seed_crop & crop])
    keep = keep[keep > 0]
    kept = torch.zeros_like(accepted)
    if keep.size:
        kept[lz:hz, ly:hy, lx:hx] = torch.as_tensor(np.isin(labels, keep), device=accepted.device)
    return kept


def undecided_neighbours(kept: torch.Tensor, region: torch.Tensor, complete: torch.Tensor) -> torch.Tensor:
    """Editable voxels not yet decided that are kept or 18-adjacent to a kept voxel."""
    kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=kept.device)
    kernel[0, 0, ::2, ::2, ::2] = 0                     # the 18-neighbourhood: no corners
    grown = F.conv3d(kept[None, None].to(torch.float32), kernel, padding=1)[0, 0] > 0
    return grown & region & ~complete


# ------------------------------------------------------------------------------------------- data classes
@dataclass(frozen=True)
class Prediction:
    """The result of one prediction.

    ``updated_mask`` is the mask after the update rule and ``applied`` the voxels the correction changed (both bool
    ZYX). With ``target_probability=True`` the prediction also holds the target probability (float32 ZYX, zero
    outside ``decided``) and ``decided``, the voxels it was computed for: the editable domain, or with the local
    update the editable voxels whose covering windows all ran. ``meta`` holds JSON values about the computation.
    """

    updated_mask: np.ndarray
    applied: np.ndarray
    target_probability: np.ndarray | None
    decided: np.ndarray | None
    meta: dict


@dataclass(frozen=True)
class _Frames:
    grid_affine_xyz: np.ndarray
    grid_shape_zyx: tuple
    whole_volume_affine_xyz: np.ndarray
    whole_volume_shape_zyx: tuple

    def flipped(self) -> "_Frames":
        return _Frames(geometry.flip_affine_xyz(self.grid_affine_xyz, self.grid_shape_zyx), self.grid_shape_zyx,
                       geometry.flip_affine_xyz(self.whole_volume_affine_xyz, self.whole_volume_shape_zyx),
                       self.whole_volume_shape_zyx)


@dataclass(frozen=True)
class _View:
    """The scalar fields of one orientation of a state that the input builders read (``DeviceState.state``)."""
    shape_zyx: tuple
    affine_xyz: np.ndarray
    correction_type: int
    previous_mask_known: bool
    pet_pad_value: float
    augmentation: Augmentation

    @classmethod
    def of(cls, state: State) -> "_View":
        return cls(state.shape_zyx, state.affine_xyz, int(state.correction_type), bool(state.previous_mask_known),
                   state.pet_pad_value, state.augmentation)

    def flipped(self) -> "_View":
        flip = Augmentation(flip_lr=True)
        return _View(self.shape_zyx, geometry.flip_affine_xyz(self.affine_xyz, self.shape_zyx), self.correction_type,
                     self.previous_mask_known, self.pet_pad_value * flip.pet_scale + flip.pet_shift, flip)


@dataclass
class _GridConstants:
    affine_xyz: np.ndarray
    shape_zyx: tuple
    image: torch.Tensor          # (2, Z, Y, X) FP32
    centre_valid: torch.Tensor   # (Z, Y, X) cell centre inside a valid native voxel
    pick_flat: torch.Tensor      # (Z, Y, X) flat index of the nearest native voxel
    operator: Any
    supports: tuple | None
    coverage: torch.Tensor       # (Z, Y, X) cells that overlap valid native voxels
    axes: tuple | None           # FP64 world axis lines when the grid is axis aligned


@dataclass
class _WindowConstants:
    origins: tuple
    affines: tuple               # (4, 4) numpy per window
    affine_device: torch.Tensor  # (W, 4, 4) FP64
    candidate_flat: torch.Tensor  # flat index of each window's valid native voxels, windows in order
    candidate_window: torch.Tensor


@dataclass
class _PassConstants:
    frames: _Frames
    pet: torch.Tensor
    ct: torch.Tensor
    valid: torch.Tensor
    affine_xyz: np.ndarray
    affine: torch.Tensor
    grids: dict
    windows: _WindowConstants
    native_axes: tuple | None


@dataclass
class _ExaminationConstants:
    """Everything that depends on the examination's images and header only, for both orientations."""
    pet: np.ndarray
    ct: np.ndarray
    valid: np.ndarray
    affine_xyz: np.ndarray
    pet_pad_value: float
    unflipped: _PassConstants
    flipped: _PassConstants | None

    def serves(self, state: State, flip_average: bool) -> bool:
        if (self.flipped is not None) != flip_average or float(state.pet_pad_value) != self.pet_pad_value \
                or not np.array_equal(state.affine_xyz, self.affine_xyz):
            return False
        for name, array in (("pet", state.pet_zyx), ("ct", state.ct_zyx), ("valid", state.valid_zyx)):
            cached = getattr(self, name)
            if array is cached:
                continue
            if array.shape != cached.shape or array.dtype != cached.dtype or not np.array_equal(array, cached):
                return False
            setattr(self, name, array)       # the same values: later rounds find the new object at once
        return True


@dataclass
class _VoxelInputs:
    """The window-independent native-refinement inputs of one orientation, per native voxel."""
    strides: torch.Tensor
    affine: torch.Tensor
    correction_type: int
    previous_mask_known: float
    spacing_xyz: torch.Tensor
    volumes: tuple
    distance: torch.Tensor       # (count,) FP32
    code: torch.Tensor | None    # (count, 8) FP32


@dataclass
class _Built:
    """One orientation's inputs while its windows run: context, 3 mm grid channels and voxel inputs."""
    context: Any
    grid: tuple
    inputs: _VoxelInputs
    contexts: dict


def _grid_key(affine_xyz, shape_zyx):
    return tuple(int(n) for n in shape_zyx), np.asarray(affine_xyz, dtype=np.float64).tobytes()


# ------------------------------------------------------------------------------------------- one orientation
class _Pass:
    """One orientation of one prediction: its editable region, windows, and the merged logits of the windows run.

    Windows always run in the batches of the full computation (``window_batch`` consecutive windows with editable
    voxels), and their contributions are added per voxel in window order: as soon as every window that covers a voxel
    has run, its merged logits equal those of the full computation bit for bit.
    """

    def __init__(self, predictor: "SIRBPredictor", view: _View, constants: _PassConstants, arrays: dict, sources,
                 scribble_window_affine_xyz, *, flip: bool, track: bool) -> None:
        self.predictor, self.view, self.constants, self.arrays = predictor, view, constants, arrays
        self.sources, self.scribble_window_affine_xyz, self.flip, self.track = (sources, scribble_window_affine_xyz,
                                                                                flip, track)
        device = predictor.device
        self.device_state = predictor._pass_device_state(view, constants, arrays, flip)
        self.region = editable_domain(self.device_state.mask, view.correction_type, self.device_state.valid)
        self.shape = view.shape_zyx
        self.count = int(np.prod(self.shape))
        self.parts, flat = self._window_parts()
        self.covering = (torch.bincount(flat.to(torch.int64), minlength=self.count).to(torch.int32)
                         if track else None)
        del flat
        self.order = sorted(self.parts)
        batch = predictor.window_batch
        self.groups = [self.order[i:i + batch] for i in range(0, len(self.order), batch)]
        self.group_of = {window: i // batch for i, window in enumerate(self.order)}
        self.position = 0                    # windows order[:position] are added to values / weights
        self.computed: set[int] = set()
        self.done = torch.zeros(self.count, dtype=torch.int32, device=device) if track else None
        self.values = torch.zeros((self.count, 3), dtype=torch.float32, device=device)
        self.weights = torch.zeros(self.count, dtype=torch.float32, device=device)
        self.store: dict[int, list] = {}     # windows run ahead of the order: copies of their contributions
        self.built: _Built | None = None
        self.builds = 0
        self.origins = np.asarray(constants.windows.origins, dtype=np.float64)
        self.inverse = np.linalg.inv(np.asarray(constants.frames.grid_affine_xyz, dtype=np.float64))

    def _window_parts(self):
        windows = self.constants.windows
        keep = self.region.reshape(-1)[windows.candidate_flat]
        flat = windows.candidate_flat[keep]
        owner = windows.candidate_window[keep]
        counts = torch.bincount(owner.to(torch.int64), minlength=len(windows.origins)).tolist()
        parts = {window: part for window, (n, part) in enumerate(zip(counts, torch.split(flat, counts))) if n}
        return parts, flat

    def _zyx(self, flat: torch.Tensor) -> torch.Tensor:
        plane = self.shape[1] * self.shape[2]
        z = flat // plane
        rest = flat - z * plane
        y = rest // self.shape[2]
        return torch.stack((z, y, rest - y * self.shape[2]), dim=1)

    def _ensure(self) -> _Built:
        if self.built is not None:
            return self.built
        predictor = self.predictor
        if self.device_state is None:
            self.device_state = predictor._pass_device_state(self.view, self.constants, self.arrays, self.flip)
        if self.parts is None:
            self.parts, _ = self._window_parts()
        context = predictor._context(self.device_state, self.view, self.constants, self.sources,
                                     self.scribble_window_affine_xyz)
        grid_shape = tuple(max(int(n), predictor.window_size) for n in self.constants.frames.grid_shape_zyx)
        grid = predictor._observation(self.device_state, self.view, self.constants,
                                      self.constants.frames.grid_affine_xyz, grid_shape)
        with geometry.deferred_checks() as checks:
            inputs = predictor._voxel_inputs(self.device_state, self.view, self.constants, self.region)
            checks.raise_first_failure()
        self.built = _Built(context, grid, inputs, {})
        self.builds += 1
        return self.built

    def release(self) -> None:
        """Drop the inputs of the windows; the merged logits, counts and stored windows stay."""
        self.device_state = self.built = self.parts = None

    def run(self, windows) -> int:
        """Run the batches that hold any of ``windows`` and are not yet done; returns the number of windows run."""
        wanted = sorted({self.group_of[w] for w in windows if w in self.group_of and w not in self.computed})
        if not wanted:
            return 0
        built = self._ensure()
        ran = 0
        with geometry.deferred_checks() as checks:
            for group in wanted:
                batch = self.groups[group]
                in_order = self.position < len(self.order) and self.order[self.position] == batch[0]

                def add(window, flat, contribution, weight):
                    self.values.index_put_((flat,), contribution, accumulate=True)
                    self.weights.index_put_((flat,), weight, accumulate=True)

                def keep(window, flat, contribution, weight):
                    self.store.setdefault(window, []).append((flat.clone(), contribution.clone(), weight.clone()))
                self.predictor._run_windows(built, self.constants, [(w, self._zyx(self.parts[w])) for w in batch],
                                            add if in_order else keep)
                for window in batch:
                    self.computed.add(window)
                    if self.track:
                        self.done.index_add_(0, self.parts[window], torch.ones_like(self.parts[window],
                                                                                    dtype=torch.int32))
                if in_order:
                    self.position += len(batch)
                ran += len(batch)
                self._fold()
            checks.raise_first_failure()
        return ran

    def _fold(self) -> None:
        while self.position < len(self.order) and self.order[self.position] in self.computed:
            for flat, contribution, weight in self.store.pop(self.order[self.position], []):
                self.values.index_put_((flat,), contribution, accumulate=True)
                self.weights.index_put_((flat,), weight, accumulate=True)
            self.position += 1

    def complete(self) -> torch.Tensor:
        """Voxels whose covering windows have all run (Z, Y, X); local update only."""
        return ((self.covering > 0) & (self.done == self.covering)).reshape(self.shape)

    def logits(self, ready: torch.Tensor) -> torch.Tensor:
        """(Z, Y, X, 3) merged native role logits at ``ready`` voxels, zero elsewhere."""
        values, weights = self.values, self.weights
        if self.store:
            values, weights = values.clone(), weights.clone()
            for window in sorted(self.store):     # every stored window comes after the added ones
                for flat, contribution, weight in self.store[window]:
                    values.index_put_((flat,), contribution, accumulate=True)
                    weights.index_put_((flat,), weight, accumulate=True)
        ready = ready.reshape(-1)
        result = torch.zeros((self.count, 3), dtype=torch.float32, device=self.predictor.device)
        result[ready] = values[ready] / weights[ready][:, None]
        return result.reshape(*self.shape, 3)

    def windows_for_world_box(self, low_xyz, high_xyz) -> list[int]:
        """Windows whose cells meet the world box (a superset is harmless)."""
        corners = np.array([[x, y, z, 1.0] for x in (low_xyz[0], high_xyz[0]) for y in (low_xyz[1], high_xyz[1])
                            for z in (low_xyz[2], high_xyz[2])])
        u = (self.inverse @ corners.T).T[:, :3]
        lo, hi = u.min(axis=0)[::-1], u.max(axis=0)[::-1]
        start = self.origins - .5 - geometry.SUPPORT_TOLERANCE
        stop = self.origins + self.predictor.window_size - .5 + geometry.SUPPORT_TOLERANCE
        return [int(w) for w in np.flatnonzero(np.all((stop >= lo) & (start <= hi), axis=1))]


# ------------------------------------------------------------------------------------------- predictor
class SIRBPredictor:
    """SIRB inference on one examination (module docstring).

    ``grid_spacing_mm`` and ``whole_volume_shape`` are those the model was trained with (its configuration's model
    section; :meth:`from_config` reads them). The constructor applies no mode: each option is a keyword argument,
    and their defaults compute every window. :meth:`from_config` starts from a mode of :data:`MODES`.
    """

    def __init__(self, model: SIRB, *, grid_spacing_mm, whole_volume_shape, device="cpu", local_update: bool = False,
                 flip_average: bool = True, precision: str = "fp32", compile: bool = False,
                 near_margin_mm: float = 0.0, window_batch: int = 4, voxel_chunk: int = 262144) -> None:
        if not isinstance(model, SIRB):
            raise TypeError(f"model must be a SIRB, got {type(model).__name__}")
        for name, value in (("local_update", local_update), ("flip_average", flip_average), ("compile", compile)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, got {value!r}")
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        for name, value in (("window_batch", window_batch), ("voxel_chunk", voxel_chunk)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        margin = float(near_margin_mm)
        if not math.isfinite(margin) or margin < 0:
            raise ValueError(f"near_margin_mm must be a finite number >= 0, got {near_margin_mm!r}")
        spacing = tuple(float(v) for v in grid_spacing_mm)
        shape = tuple(int(v) for v in whole_volume_shape)
        if len(spacing) != 3 or len(shape) != 3:
            raise ValueError(f"grid_spacing_mm and whole_volume_shape need three ZYX values; got {spacing}, {shape}")
        self.device = torch.device(device)
        self.local_update, self.flip_average, self.precision = local_update, flip_average, precision
        self.compiled, self.near_margin_mm = compile, margin
        self.window_batch, self.voxel_chunk = window_batch, voxel_chunk
        self.grid_spacing_mm, self.whole_volume_shape = spacing, shape
        model = model.to(self.device).eval()
        if compile:
            model = copy.deepcopy(model)
            model.local_encoder = torch.compile(model.local_encoder)
            model.role_decoder = torch.compile(model.role_decoder)
            refinement = model.native_refinement
            refinement.forward = torch.compile(type(refinement).forward.__get__(refinement), dynamic=True)
        self.model = model
        self.window_size = int(model.window_size)
        self.sum_order = detect_sum_order(self.device)
        self.distance_fallbacks = 0
        self._case: _ExaminationConstants | None = None
        self._supports: dict = {}
        self._triangle_window: torch.Tensor | None = None
        self._examination: Examination | None = None
        self._history: InteractionHistory | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any], checkpoint=None, *, mode: str = DEFAULT_MODE, device="cpu",
                    **options) -> "SIRBPredictor":
        """The predictor of a configuration: ``build_sirb(config, checkpoint)`` with the options of ``mode`` (a key of
        :data:`MODES`), then those of the configuration's ``predictor`` section, then the keyword ``options``, each
        overriding the one before."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}, got {mode!r}")
        settings = model_settings(config)
        merged = dict(MODES[mode])
        merged.update(config.get("predictor", {}))
        merged.update(options)
        unknown = sorted(set(merged) - set(PREDICTOR_OPTIONS))
        if unknown:
            raise ValueError(f"unknown predictor options {unknown}; known: {list(PREDICTOR_OPTIONS)}")
        model = build_sirb(config, checkpoint, device=device)
        return cls(model, grid_spacing_mm=settings["grid_spacing_mm"], whole_volume_shape=settings["whole_volume_shape"],
                   device=device, **merged)

    @property
    def options(self) -> dict:
        """The options of this predictor, as a configuration's ``predictor`` section writes them."""
        return {"local_update": self.local_update, "flip_average": self.flip_average, "precision": self.precision,
                "compile": self.compiled, "near_margin_mm": self.near_margin_mm, "window_batch": self.window_batch,
                "voxel_chunk": self.voxel_chunk}

    # ------------------------------------------------------------------ high level
    def set_examination(self, examination: Examination) -> None:
        """Start correcting ``examination`` from its initial segmentation."""
        if not isinstance(examination, Examination):
            raise TypeError(f"examination must be an Examination, got {type(examination).__name__}")
        self._examination = examination
        self._history = InteractionHistory(examination.initial_segmentation_zyx)

    def reset(self) -> None:
        """Back to the examination's initial segmentation with no interaction history."""
        if self._examination is None:
            raise RuntimeError("set_examination must be called first")
        self._history = InteractionHistory(self._examination.initial_segmentation_zyx)

    @property
    def mask(self) -> np.ndarray:
        """The current mask (a copy)."""
        if self._history is None:
            raise RuntimeError("set_examination must be called first")
        return self._history.mask.copy()

    def predict(self, scribble_zyx, correction_type, *, target_probability: bool = False) -> Prediction:
        """Correct the current mask with one scribble of ``correction_type`` (``"addition"``/``"removal"`` or
        +1/-1); the updated mask becomes the current mask. An empty scribble changes nothing and makes the previous
        mask unknown, as a round without a valid scribble does."""
        if self._history is None:
            raise RuntimeError("set_examination must be called first")
        scribble = np.asarray(scribble_zyx, dtype=bool)
        sign = correction_sign(correction_type)
        if not scribble.any():
            self._history.skip()
            mask = self._history.mask
            return Prediction(mask.copy(), np.zeros(mask.shape, dtype=bool), None, None, {"empty_scribble": True})
        state = self._history.state(self._examination, scribble, sign)
        prediction = self.predict_state(state, target_probability=target_probability)
        self._history.record(scribble, sign, prediction.updated_mask)
        return prediction

    # ------------------------------------------------------------------ low level
    @torch.no_grad()
    def predict_state(self, state: State, *, target_probability: bool = False) -> Prediction:
        """One prediction on ``state``; nothing is remembered between calls except the examination's constants."""
        if not isinstance(state, State):
            raise TypeError(f"state must be a sirb.data.observations.State, got {type(state).__name__}")
        if state.augmentation != Augmentation():
            raise ValueError("the predictor takes states without augmentation")
        if not state.scribble_zyx.any():
            mask = np.asarray(state.mask_zyx, dtype=bool)
            return Prediction(mask.copy(), np.zeros(mask.shape, dtype=bool), None, None, {"empty_scribble": True})
        self.model.eval()
        started = time.perf_counter()
        case = self._examination_constants(state)
        representatives = geometry.scribble_representatives(state.scribble_zyx, state.affine_xyz,
                                                            case.unflipped.frames.grid_affine_xyz, self.window_size,
                                                            native_valid_zyx=state.valid_zyx)
        sources = group_scribble(state.scribble_zyx, state.affine_xyz, representatives,
                                 native_valid_zyx=state.valid_zyx)
        arrays = self._round_arrays(state)
        view = _View.of(state)
        orientations = [(view, case.unflipped, representatives.window_affine_xyz, False)]
        if self.flip_average:
            orientations.append((view.flipped(), case.flipped,
                                 geometry.flip_affine_xyz(representatives.window_affine_xyz, (self.window_size,) * 3),
                                 True))
        if self.local_update:
            applied, decided, probability, meta = self._local_update(state, arrays, sources, orientations)
        else:
            applied, decided, probability, meta = self._full_update(arrays, sources, orientations)
        updated = apply(arrays["mask"], applied, view.correction_type)
        result = Prediction(updated.cpu().numpy(), applied.cpu().numpy(),
                            torch.where(decided, probability, torch.zeros_like(probability)).cpu().numpy()
                            if target_probability else None,
                            decided.cpu().numpy() if target_probability else None,
                            dict(meta, applied_voxels=int(applied.sum()), seconds=time.perf_counter() - started))
        return result

    def _decide(self, fused: torch.Tensor, region: torch.Tensor):
        """(decision, target probability, rechecked voxels): the CPU-softmax 0.5 decision at every region voxel."""
        probability = fused.softmax(-1)[..., TARGET_ERROR]
        near = region & ((probability - TARGET_THRESHOLD).abs() < RECHECK_BAND)
        decision = accept(probability)
        rechecked = int(near.sum())
        if rechecked:
            rows = fused[near].cpu()
            cpu_probability = rows.softmax(-1)[..., TARGET_ERROR]
            decision[near] = accept(cpu_probability).to(self.device)
            probability[near] = cpu_probability.to(self.device)
        return decision, probability, rechecked

    def _full_update(self, arrays, sources, orientations):
        logits, passes = [], []
        for view, constants, window_affine, flip in orientations:
            one = _Pass(self, view, constants, arrays, sources, window_affine, flip=flip, track=False)
            one.run(one.order)
            editable = one.region.reshape(-1)
            uncovered = int((one.weights[editable] <= 0).sum())
            if uncovered:
                raise ValueError(f"zero window coverage at {uncovered} editable native voxels")
            logits.append(one.logits(one.region))
            passes.append((len(one.order), one.region))
            del one
        region = passes[0][1]
        fused = logits[0] if len(logits) == 1 else (logits[0] + torch.flip(logits[1], dims=[2])) / 2
        decision, probability, rechecked = self._decide(fused, region)
        return (region & decision, region, probability,
                {"windows": [count for count, _ in passes], "rechecked_voxels": rechecked})

    def _local_update(self, state, arrays, sources, orientations):
        passes = [_Pass(self, view, constants, arrays, sources, window_affine, flip=flip, track=True)
                  for view, constants, window_affine, flip in orientations]
        for one in passes:
            one.release()
        low, high = world_box(state.affine_xyz, seed_voxels(state.scribble_zyx), self.near_margin_mm)
        need = [one.windows_for_world_box(low, high) for one in passes]
        total = sum(len(one.order) for one in passes)
        region = passes[0].region
        scribble = arrays["scribble"]
        seeds = F.max_pool3d(scribble[None, None].to(torch.float32), 3, stride=1, padding=1)[0, 0] > 0
        expansions, fallback, stalled = 0, False, False
        while True:
            ran = 0
            for one, windows in zip(passes, need):
                ran += one.run(windows)
                one.release()
            if sum(len(one.computed) for one in passes) > LOCAL_FALLBACK_FRACTION * total and not fallback:
                fallback = True
                for one in passes:
                    ran += one.run(one.order)
                    one.release()
            if len(passes) == 1:
                complete = passes[0].complete()
                fused = passes[0].logits(passes[0].region & complete)
            else:
                first, second = passes
                complete = first.complete() & torch.flip(second.complete(), dims=[2])
                fused = (first.logits(first.region & first.complete())
                         + torch.flip(second.logits(second.region & second.complete()), dims=[2])) / 2
            decided = region & complete
            decision, probability, rechecked = self._decide(fused, decided)
            accepted = decided & decision
            del fused, decision
            kept = scribble_connected(accepted, scribble)
            undecided_seeds = seeds & region & ~complete
            touching = undecided_neighbours(kept, region, complete) | undecided_seeds
            if not bool(touching.any()):
                break
            if expansions and not ran:          # nothing new could run: stop and report it
                stalled = True
                break
            expansions += 1
            index = torch.nonzero(touching).cpu().numpy()
            affine = np.asarray(state.affine_xyz)
            world = index[:, ::-1] @ affine[:3, :3].T + affine[:3, 3]
            low, high = world.min(axis=0), world.max(axis=0)
            need = [one.windows_for_world_box(low, high) for one in passes]
        return kept, decided, probability, {
            "windows": [len(one.order) for one in passes], "windows_computed": [len(one.computed) for one in passes],
            "expansions": expansions, "fallback_all_windows": fallback, "stalled": stalled,
            "rechecked_voxels": rechecked}

    # ------------------------------------------------------------------ examination constants
    def _examination_constants(self, state: State) -> _ExaminationConstants:
        if self._case is not None and self._case.serves(state, self.flip_average):
            return self._case
        self._case = None
        grids = geometry.examination_grids(state.shape_zyx, state.affine_xyz, self.window_size,
                                           grid_spacing_mm=self.grid_spacing_mm,
                                           whole_volume_shape_zyx=self.whole_volume_shape)
        frames = _Frames(grids.grid_affine_xyz, grids.padded_shape_zyx, grids.whole_volume_affine_xyz,
                         grids.whole_volume_shape_zyx)
        # np.array copies: cached volumes are read-only memory maps, which torch cannot wrap without a warning.
        pet = torch.as_tensor(np.array(state.pet_zyx, dtype=np.float32), device=self.device)
        ct = torch.as_tensor(np.array(state.ct_zyx, dtype=np.float32), device=self.device)
        valid = torch.as_tensor(np.array(state.valid_zyx, dtype=bool), device=self.device)
        view = _View.of(state)
        unflipped = self._pass_constants(view, frames, pet, ct, valid)
        flipped = None
        if self.flip_average:
            flipped = self._pass_constants(view.flipped(), frames.flipped(), torch.flip(pet, dims=[2]),
                                           torch.flip(ct, dims=[2]), torch.flip(valid, dims=[2]))
        self._case = _ExaminationConstants(state.pet_zyx, state.ct_zyx, state.valid_zyx,
                                           np.array(state.affine_xyz, dtype=np.float64), float(state.pet_pad_value),
                                           unflipped, flipped)
        return self._case

    def _static_device_state(self, view: _View, pet, ct, valid) -> DeviceState:
        """A DeviceState with the examination's images only (the per-round arrays empty)."""
        empty = torch.zeros_like(valid)
        affine = torch.as_tensor(view.affine_xyz, dtype=torch.float64, device=self.device)
        return DeviceState(view, self.device, pet, ct, empty, empty, empty, empty, empty, empty, valid,
                           np.asarray(view.affine_xyz, dtype=np.float64), affine,
                           torch.zeros((0, 3), dtype=torch.float64, device=self.device))

    def _support(self, operator):
        entry = self._supports.get(id(operator))
        if entry is None or entry[0] is not operator:
            if len(self._supports) > 32:
                self._supports.clear()
            entry = (operator, support_matrices(operator, self.device))
            self._supports[id(operator)] = entry
        return entry[1]

    def _grid_constants(self, device_state: DeviceState, affine_xyz, shape_zyx) -> _GridConstants:
        shape = tuple(int(n) for n in shape_zyx)
        image = observations.image_channels(device_state, affine_xyz, shape)
        points = geometry.grid_world_points(shape, affine_xyz, device=self.device, dtype=torch.float64)
        centre_valid, pick = observations.support_valid(points, device_state)
        native = device_state.shape_zyx
        pick_flat = (pick[..., 2] * native[1] + pick[..., 1]) * native[2] + pick[..., 0]
        del points
        operator = geometry.cell_overlap_operator(device_state.affine_xyz, native, affine_xyz, shape)
        supports = self._support(operator)
        coverage = positive_hits(operator, supports, device_state.valid[None])[0]
        axes = axis_world(shape, affine_xyz, self.device) if axis_aligned(affine_xyz) else None
        return _GridConstants(np.asarray(affine_xyz, dtype=np.float64), shape, image, centre_valid, pick_flat,
                              operator, supports, coverage, axes)

    def _pass_constants(self, view: _View, frames: _Frames, pet, ct, valid) -> _PassConstants:
        device_state = self._static_device_state(view, pet, ct, valid)
        p = self.window_size
        grid_shape = tuple(max(int(n), p) for n in frames.grid_shape_zyx)
        grids = {}
        for affine, shape in ((frames.grid_affine_xyz, grid_shape),
                              (frames.whole_volume_affine_xyz, frames.whole_volume_shape_zyx)):
            grids[_grid_key(affine, shape)] = self._grid_constants(device_state, affine, shape)
        origins = geometry.sliding_window_origins(frames.grid_shape_zyx, p)
        affines = tuple(geometry.window_affine_xyz(frames.grid_affine_xyz, origin) for origin in origins)
        native = view.shape_zyx
        strides = torch.as_tensor([native[1] * native[2], native[2], 1], dtype=torch.int64, device=self.device)
        flats, owners = [], []
        for first in range(0, len(affines), self.window_batch):
            index, counts = observations.window_native_voxels_batch(device_state, valid,
                                                                    affines[first:first + self.window_batch], (p,) * 3,
                                                                    chunk_size=self.voxel_chunk)
            flats.append((index * strides).sum(-1))
            owners.append(torch.repeat_interleave(
                torch.arange(first, first + len(counts), device=self.device, dtype=torch.int32),
                torch.as_tensor(counts, device=self.device)))
        windows = _WindowConstants(tuple(origins), affines,
                                   torch.as_tensor(np.stack(affines), dtype=torch.float64, device=self.device),
                                   torch.cat(flats), torch.cat(owners))
        native_axes = axis_world(native, view.affine_xyz, self.device) if axis_aligned(view.affine_xyz) else None
        return _PassConstants(frames, pet, ct, valid, np.asarray(view.affine_xyz, dtype=np.float64),
                              device_state.affine, grids, windows, native_axes)

    # ------------------------------------------------------------------ per-round arrays and inputs
    def _round_arrays(self, state: State) -> dict:
        def put(array):
            return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.bool, device=self.device)
        return {"mask": put(state.mask_zyx), "scribble": put(state.scribble_zyx),
                "earlier_addition": put(state.earlier_addition_scribbles_zyx),
                "earlier_removal": put(state.earlier_removal_scribbles_zyx),
                "last_added": put(state.last_added), "last_removed": put(state.last_removed)}

    def _pass_device_state(self, view: _View, constants: _PassConstants, arrays: dict, flip: bool) -> DeviceState:
        arrays = {name: torch.flip(value, dims=[2]) if flip else value for name, value in arrays.items()}
        index = torch.nonzero(arrays["scribble"]).flip(-1).to(torch.float64)
        scribble_world = index @ constants.affine[:3, :3].T + constants.affine[:3, 3]
        return DeviceState(view, self.device, constants.pet, constants.ct, arrays["mask"], arrays["scribble"],
                           arrays["earlier_addition"], arrays["earlier_removal"], arrays["last_added"],
                           arrays["last_removed"], constants.valid, constants.affine_xyz, constants.affine,
                           scribble_world)

    def _distance_volume(self, device_state: DeviceState, axes, shape_zyx, affine_xyz) -> torch.Tensor:
        """``observations.normalized_scribble_distance`` at every point of a grid (Z, Y, X), FP32."""
        native = device_state.shape_zyx
        if axes is None or self.sum_order is None or device_state.scribble_world.shape[0] == 0:
            points = geometry.grid_world_points(shape_zyx, affine_xyz, device=self.device, dtype=torch.float64)
            return observations.normalized_scribble_distance(points, device_state.scribble_world, native,
                                                             device_state.affine_xyz)
        value = normalised_distance(nearest_scribble_distance_axes(axes, device_state.scribble_world, self.sum_order),
                                    native, device_state.affine_xyz)
        if not self._distance_spot_check(device_state, value, shape_zyx, affine_xyz):
            self.distance_fallbacks += 1
            points = geometry.grid_world_points(shape_zyx, affine_xyz, device=self.device, dtype=torch.float64)
            return observations.normalized_scribble_distance(points, device_state.scribble_world, native,
                                                             device_state.affine_xyz)
        return value

    def _distance_spot_check(self, device_state, value, shape_zyx, affine_xyz) -> bool:
        count = int(np.prod(shape_zyx))
        generator = torch.Generator(device="cpu").manual_seed(count % 2147483647)
        pick = torch.randint(0, count, (DISTANCE_SPOT_CHECK,), generator=generator).to(self.device)
        z = pick // (shape_zyx[1] * shape_zyx[2])
        y = (pick // shape_zyx[2]) % shape_zyx[1]
        x = pick % shape_zyx[2]
        index = torch.stack((x, y, z), dim=-1).to(torch.float64)
        affine = torch.as_tensor(affine_xyz, device=self.device, dtype=torch.float64)
        world = index @ affine[:3, :3].T + affine[:3, 3]
        brute = observations.normalized_scribble_distance(world, device_state.scribble_world, device_state.shape_zyx,
                                                          device_state.affine_xyz)
        return bool(torch.equal(brute, value.reshape(-1)[pick]))

    def _observation(self, device_state: DeviceState, view: _View, constants: _PassConstants, affine_xyz, shape_zyx):
        """(image [1,2,...], 12 state channels [1,12,...], coverage [1,1,...]) of a grid: the values of
        ``observations.image_channels`` and ``observations.state_channels``."""
        grid = constants.grids.get(_grid_key(affine_xyz, shape_zyx))
        if grid is None:
            grid = self._grid_constants(device_state, affine_xyz, shape_zyx)
        valid = device_state.valid
        shape = grid.shape_zyx
        mask_label = torch.where(grid.centre_valid, device_state.mask.reshape(-1)[grid.pick_flat],
                                 torch.zeros_like(grid.centre_valid)).to(torch.float32)
        covered = observations.apply_overlap(grid.operator, device_state.mask & valid).to(torch.float32)
        hits = positive_hits(grid.operator, grid.supports,
                             torch.stack((device_state.earlier_addition_scribbles & valid,
                                          device_state.earlier_removal_scribbles & valid,
                                          device_state.scribble & valid, device_state.last_added & valid,
                                          device_state.last_removed & valid))).to(torch.float32)
        zero = torch.zeros(shape, dtype=torch.float32, device=self.device)
        distance = torch.where(grid.centre_valid, self._distance_volume(device_state, grid.axes, shape,
                                                                        grid.affine_xyz), zero)
        addition = view.correction_type == 1
        scribble_hit = hits[2]
        known = grid.coverage.to(torch.float32) * float(view.previous_mask_known)
        channels = torch.stack((mask_label, covered, hits[0], hits[1], scribble_hit if addition else zero,
                                zero if addition else scribble_hit, distance if addition else zero,
                                zero if addition else distance, hits[3], hits[4], known, known))
        return grid.image[None], channels[None], grid.coverage[None, None]

    def _context(self, device_state, view, constants, sources, scribble_window_affine_xyz):
        frames = constants.frames
        whole_image, whole_state, whole_valid = self._observation(device_state, view, constants,
                                                                  frames.whole_volume_affine_xyz,
                                                                  frames.whole_volume_shape_zyx)
        window_image, window_state, window_valid = self._observation(device_state, view, constants,
                                                                     scribble_window_affine_xyz,
                                                                     (self.window_size,) * 3)
        bounds = torch.as_tensor(geometry.examination_bounds_xyz(view.shape_zyx, view.affine_xyz),
                                 dtype=torch.float64, device=self.device)
        return self.model.prepare_context(
            torch.cat((whole_image, whole_state), 1),
            torch.as_tensor(frames.whole_volume_affine_xyz, dtype=torch.float64, device=self.device),
            window_image, window_state,
            torch.as_tensor(scribble_window_affine_xyz, dtype=torch.float64, device=self.device), sources,
            torch.tensor([view.correction_type], device=self.device), bounds, whole_volume_valid=whole_valid,
            window_valid=window_valid)

    def _voxel_inputs(self, device_state: DeviceState, view: _View, constants: _PassConstants, region) -> _VoxelInputs:
        """Scribble distance and neighbourhood code of every editable native voxel (the values of
        ``observations.PassVoxelInputs``)."""
        shape = view.shape_zyx
        count = int(np.prod(shape))
        strides = torch.as_tensor([shape[1] * shape[2], shape[2], 1], dtype=torch.int64, device=self.device)
        affine = device_state.affine
        distance = None
        if constants.native_axes is not None and self.sum_order is not None:
            distance = self._distance_volume(device_state, constants.native_axes, shape, view.affine_xyz).reshape(-1)
        voxels = torch.nonzero(region & device_state.valid)
        if distance is None:
            distance = torch.zeros(count, dtype=torch.float32, device=self.device)
            for start in range(0, voxels.shape[0], self.voxel_chunk):
                part = voxels[start:start + self.voxel_chunk]
                world = part.flip(-1).to(torch.float64) @ affine[:3, :3].T + affine[:3, 3]
                distance[(part * strides).sum(-1)] = observations.normalized_scribble_distance(
                    world, device_state.scribble_world, shape, view.affine_xyz)
        offsets = torch.as_tensor(observations.NEIGHBOURHOOD_OFFSETS_ZYX, dtype=torch.int64, device=self.device)
        native_shape = torch.as_tensor(shape, dtype=torch.int64, device=self.device)
        origin = torch.zeros(3, dtype=torch.int64, device=self.device)
        code = None
        for start in range(0, voxels.shape[0], self.voxel_chunk):
            part = voxels[start:start + self.voxel_chunk]
            value = self.model.encode_neighbourhood(observations.neighbourhood_values(
                part, offsets, native_shape, origin, shape, device_state.pet, device_state.ct, device_state.mask,
                device_state.valid))
            if code is None:
                code = torch.zeros((count, value.shape[-1]), dtype=torch.float32, device=self.device)
            code[(part * strides).sum(-1)] = value.to(torch.float32)
        spacing = torch.as_tensor(geometry.spacing_xyz_mm(view.affine_xyz), dtype=torch.float32, device=self.device)
        volumes = (device_state.pet, device_state.ct, device_state.mask, device_state.scribble,
                   device_state.earlier_addition_scribbles, device_state.earlier_removal_scribbles,
                   device_state.last_added, device_state.last_removed)
        return _VoxelInputs(strides, affine, view.correction_type, float(view.previous_mask_known), spacing, volumes,
                            distance, code)

    def _window_inputs(self, inputs: _VoxelInputs, index: torch.Tensor, window_affine):
        flat = (index * inputs.strides).sum(-1)
        world = index.flip(-1).to(torch.float64) @ inputs.affine[:3, :3].T + inputs.affine[:3, 3]
        picked = tuple(volume.reshape(-1)[flat].to(torch.float32) for volume in inputs.volumes)
        values = observations.voxel_values(picked, inputs.distance[flat], correction_type=inputs.correction_type,
                                           previous_mask_known=inputs.previous_mask_known,
                                           spacing_xyz=inputs.spacing_xyz, world_xyz=world,
                                           window_affine_xyz=window_affine)
        return flat, world, values, None if inputs.code is None else inputs.code[flat]

    def _triangle(self) -> torch.Tensor:
        if self._triangle_window is None:
            self._triangle_window = geometry.triangle_window(self.window_size, device=self.device)[None, None]
        return self._triangle_window

    def _run_windows(self, built: _Built, constants: _PassConstants, jobs, sink) -> None:
        """One batched forward pass of ``jobs`` [(window, editable native voxels ZYX)], then native refinement of
        each window's voxels; ``sink(window, flat, weighted logits, weights)`` receives every chunk."""
        batch, p = len(jobs), self.window_size
        if batch not in built.contexts:
            built.contexts[batch] = self.model.expand_context(built.context, batch)
        blocks = [(slice(None), slice(None), slice(z, z + p), slice(y, y + p), slice(x, x + p))
                  for z, y, x in (constants.windows.origins[window] for window, _ in jobs)]
        image, state, valid = (torch.cat([volume[block] for block in blocks]) for volume in built.grid)
        affine = constants.windows.affine_device[[window for window, _ in jobs]]
        autocast = (torch.autocast(self.device.type, dtype=torch.bfloat16) if self.precision == "bf16"
                    else nullcontext())
        with autocast:
            output = self.model.forward_window(image, state, affine, built.contexts[batch])
        stack = torch.cat((output["features"].float(), output["coarse_role_logits"].float(),
                           self._triangle().expand(batch, -1, -1, -1, -1)), 1)
        field = output["source_role_logits"]["6mm"]
        source_valid = output["source_valid"]
        b, sources, _, sz, sy, sx = field.shape
        safe_field = torch.where(source_valid[:, :, None, None, None, None], field,
                                 torch.zeros_like(field)).reshape(b, sources * ROLE_COUNT, sz, sy, sx)
        support_size = torch.tensor((p, p, p), dtype=torch.float64, device=self.device)
        support_long = support_size.to(torch.long)
        stack_size = torch.tensor(stack.shape[2:][::-1], dtype=torch.float64, device=self.device)
        field_size = torch.tensor(safe_field.shape[2:][::-1], dtype=torch.float64, device=self.device)
        for item, (window, index) in enumerate(jobs):
            covered, positive = [], []
            for start in range(0, len(index), self.voxel_chunk):
                part = index[start:start + self.voxel_chunk]
                flat, world, values, code = self._window_inputs(built.inputs, part, affine[item])
                sampled, inside = sample_at_points(
                    stack[item:item + 1], output["feature_affine"][item], world, support_affine_xyz=affine[item],
                    support_size_xyz=support_size, support_size_long=support_long, feature_size_xyz=stack_size,
                    support_valid_zyx=valid[item:item + 1])
                role_logits, source_inside = sample_at_points(
                    safe_field[item:item + 1], output["source_role_affines"]["6mm"][item], world,
                    support_affine_xyz=affine[item], support_size_xyz=support_size, support_size_long=support_long,
                    feature_size_xyz=field_size, support_valid_zyx=valid[item:item + 1])
                role_logits = role_logits.reshape(1, -1, sources, ROLE_COUNT).float()
                logits = self.model.refine_native(sampled[..., :FEATURE_WIDTH], sampled[..., _COARSE], values[None],
                                                  neighbourhood_code=code[None],
                                                  source_vectors=output["source_vectors"][item:item + 1],
                                                  source_role_logits=role_logits,
                                                  source_valid=source_valid[item:item + 1])
                inside = inside & source_inside
                logits = torch.where(inside[..., None], logits, torch.zeros_like(logits))[0].float()
                weight = sampled[0, :, _WEIGHT]
                covered.append(inside.all())
                positive.append((weight > 0).all())
                sink(window, flat, logits * weight[:, None], weight)
            geometry.require(torch.stack(covered), ValueError, "native voxels of a window lie outside its cells")
            geometry.require(torch.stack(positive), ValueError, "non-positive triangle weights inside a window")
