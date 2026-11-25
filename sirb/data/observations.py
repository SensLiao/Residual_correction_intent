"""The state of one correction round and the network inputs built from it.

A :class:`State` holds an examination's normalised PET and CT on the native PET grid, the current mask, the scribble
with its correction type and the interaction history (earlier scribbles of each correction type and the previous
mask). From it SIRB reads:

* 14 input channels on a grid of 3 mm cells (a window of the grid, or the whole-volume grid): PET and CT
  (:func:`image_channels`) and the 12 state channels (:func:`state_channels`): the mask label of the nearest native
  voxel, the covered fraction (the cell's volume fraction inside the mask, from the exact cell overlap), the earlier
  addition and removal scribbles, the current scribble and its normalised distance map in the channel of its
  correction type, the voxels added and removed by the previous correction, and twice whether the previous mask is
  known (:data:`STATE_CHANNELS`);
* for native refinement, 20 values per editable native voxel (:data:`VOXEL_CHANNELS`, :func:`voxel_values`) and the
  108 values of its 3^3 native neighbourhood (PET, CT, mask and validity, :func:`neighbourhood_values`), built
  chunk by chunk (:class:`WindowVoxelInputs` for one window, :class:`PassVoxelInputs` for a whole inference pass).

No ground truth enters any input. Values are computed with torch on the device of a :class:`DeviceState`, one
implementation for training and inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import product

import numpy as np
import torch
from scipy.spatial import cKDTree

from .geometry import (GeometryError, cell_overlap_operator, flip_affine_xyz, grid_world_points, index_xyz_to_world,
                       solve, spacing_xyz_mm, validate_affine_xyz, world_to_index_xyz)
from .roles import correction_sign

__all__ = ["DISTANCE_SCALE_MM", "IMAGE_CHANNELS", "NEIGHBOURHOOD_WIDTH", "STATE_CHANNELS", "VOXEL_CHANNELS",
           "VOXEL_WIDTH", "WHOLE_VOLUME_CHANNELS", "Augmentation", "DeviceState", "PassVoxelInputs", "State",
           "WindowVoxelInputs", "augment_state", "draw_augmentation", "image_channels", "neighbourhood_values",
           "normalized_scribble_distance", "previous_mask_from_change", "scribble_world_points", "state_channels",
           "voxel_values", "window_native_voxels", "window_native_voxels_batch"]

IMAGE_CHANNELS = ("pet", "ct")
STATE_CHANNELS = ("mask", "covered_fraction", "earlier_addition_scribbles", "earlier_removal_scribbles",
                  "addition_scribble", "removal_scribble", "addition_distance", "removal_distance", "last_added",
                  "last_removed", "previous_mask_known_addition", "previous_mask_known_removal")
#: The whole-volume encoder reads all 14 channels; the local encoder reads IMAGE_CHANNELS in its image stream and
#: STATE_CHANNELS in its state stream.
WHOLE_VOLUME_CHANNELS = IMAGE_CHANNELS + STATE_CHANNELS
#: The 20 values of one native voxel: its own image, mask and interaction values, the correction type, its offset
#: from the centre of its 3 mm window cell (in cells) and the native voxel spacing.
VOXEL_CHANNELS = ("pet", "ct", "mask", "addition_scribble", "removal_scribble", "earlier_addition_scribbles",
                  "earlier_removal_scribbles", "last_added", "last_removed", "previous_mask_known_addition",
                  "previous_mask_known_removal", "correction_type", "addition_distance", "removal_distance",
                  "offset_x", "offset_y", "offset_z", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm")
VOXEL_WIDTH = len(VOXEL_CHANNELS)
#: 3^3 native index offsets (dz, dy, dx), lexicographic; the centre is entry 13.
NEIGHBOURHOOD_OFFSETS_ZYX = tuple(product((-1, 0, 1), repeat=3))
#: Channel-major neighbourhood groups, each with the 27 offsets above.
NEIGHBOURHOOD_CHANNELS = ("pet", "ct", "mask", "valid")
NEIGHBOURHOOD_WIDTH = len(NEIGHBOURHOOD_CHANNELS) * len(NEIGHBOURHOOD_OFFSETS_ZYX)
#: One native voxel on each side; a 3^3 neighbourhood never needs more.
NEIGHBOURHOOD_HALO = 1
#: Default bound on per-voxel temporaries (world coordinates, distance queries).
DEFAULT_CHUNK = 65536
#: Points per brute-force distance block; bounds the (points x scribble voxels) temporary.
DISTANCE_PAIR_BUDGET = 1 << 22
#: Distance normalisation: log1p(d / 3 mm) / log1p(examination diagonal / 3 mm).
DISTANCE_SCALE_MM = 3.
#: Same tolerances as geometry.support_contains_world and geometry.floor_integer_boundary.
SUPPORT_TOLERANCE = 1e-7
INTEGER_TIE_TOLERANCE = 1e-10


# ------------------------------------------------------------------------------------------- state
@dataclass(frozen=True)
class Augmentation:
    """One draw of the training augmentation, shared by both members of a pair."""
    flip_lr: bool = False
    pet_scale: float = 1.
    pet_shift: float = 0.
    ct_shift: float = 0.


def draw_augmentation(rng: np.random.Generator) -> Augmentation:
    """Left-right flip with probability 0.5, PET scale in [0.9, 1.1], PET shift in [-0.1, 0.1], CT shift in
    [-0.02, 0.02]."""
    return Augmentation(bool(rng.random() < .5), float(rng.uniform(.9, 1.1)), float(rng.uniform(-.1, .1)),
                        float(rng.uniform(-.02, .02)))


@dataclass(frozen=True)
class State:
    """Everything SIRB reads in one correction round, on the native PET grid (ZYX arrays).

    PET and CT are normalised once before construction. ``previous_mask_zyx`` is only the mask before the last
    correction that was actually applied; ``None`` means the previous mask is unknown (round 1, or after a round
    without a valid scribble). A known previous mask identical to the current one gives empty last changes with
    both previous-mask-known channels set.
    """
    pet_zyx: np.ndarray
    ct_zyx: np.ndarray
    mask_zyx: np.ndarray
    scribble_zyx: np.ndarray
    affine_xyz: np.ndarray
    correction_type: int | str
    earlier_addition_scribbles_zyx: np.ndarray | None = None
    earlier_removal_scribbles_zyx: np.ndarray | None = None
    previous_mask_zyx: np.ndarray | None = None
    valid_zyx: np.ndarray | None = None
    examination_id: str = ""
    patient_id: str = ""
    pet_pad_value: float = 0.
    augmentation: Augmentation = Augmentation()

    def __post_init__(self):
        pet, ct = np.asarray(self.pet_zyx), np.asarray(self.ct_zyx)
        if pet.ndim != 3 or min(pet.shape) < 1 or ct.shape != pet.shape:
            raise ValueError("State requires matching native PET/CT ZYX volumes")
        if not np.isfinite(pet).all() or not np.isfinite(ct).all():
            raise ValueError("State image values must be finite")
        object.__setattr__(self, "pet_zyx", pet)
        object.__setattr__(self, "ct_zyx", ct)
        object.__setattr__(self, "affine_xyz", validate_affine_xyz(self.affine_xyz))
        object.__setattr__(self, "correction_type", correction_sign(self.correction_type))
        for name in ("mask_zyx", "scribble_zyx", "earlier_addition_scribbles_zyx", "earlier_removal_scribbles_zyx",
                     "previous_mask_zyx", "valid_zyx"):
            value = getattr(self, name)
            if value is None:
                if name == "previous_mask_zyx":
                    continue
                value = np.ones(pet.shape, dtype=bool) if name == "valid_zyx" else np.zeros(pet.shape, dtype=bool)
            value = np.asarray(value, dtype=bool)
            if value.shape != pet.shape:
                raise ValueError(f"State {name} actual={value.shape}, expected={pet.shape}")
            object.__setattr__(self, name, value)
        if np.any(self.scribble_zyx & ~self.valid_zyx):
            raise GeometryError("the scribble contains invalid native voxels")
        spacing = spacing_xyz_mm(self.affine_xyz)
        directions = self.affine_xyz[:3, :3] / spacing
        if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-6, rtol=0):
            raise GeometryError("physical distances require orthogonal native axes; the affine has shear")

    @property
    def shape_zyx(self):
        return tuple(int(v) for v in self.mask_zyx.shape)

    @cached_property
    def last_added(self):
        """Voxels the previous correction added (empty when the previous mask is unknown)."""
        if self.previous_mask_zyx is None:
            return np.zeros(self.shape_zyx, dtype=bool)
        return self.mask_zyx & ~self.previous_mask_zyx

    @cached_property
    def last_removed(self):
        """Voxels the previous correction removed (empty when the previous mask is unknown)."""
        if self.previous_mask_zyx is None:
            return np.zeros(self.shape_zyx, dtype=bool)
        return self.previous_mask_zyx & ~self.mask_zyx

    @property
    def previous_mask_known(self):
        return self.previous_mask_zyx is not None

    @cached_property
    def scribble_world_xyz(self):
        return index_xyz_to_world(np.argwhere(self.scribble_zyx)[:, ::-1], self.affine_xyz)

    @cached_property
    def scribble_tree(self):
        return cKDTree(self.scribble_world_xyz) if len(self.scribble_world_xyz) else None

    @property
    def examination_diagonal_mm(self):
        """Diagonal of the outer voxel-cell box, in its own orthogonal direction basis."""
        return float(np.linalg.norm(np.array(self.shape_zyx[::-1]) * spacing_xyz_mm(self.affine_xyz)))

    def distances_mm(self, world_xyz):
        """Euclidean distance (mm) from world points to the nearest scribble voxel centre (zero without a scribble)."""
        points = np.asarray(world_xyz, dtype=np.float64)
        if self.scribble_tree is None:
            return np.zeros(points.shape[:-1], dtype=np.float64)
        return self.scribble_tree.query(points.reshape(-1, 3), workers=1)[0].reshape(points.shape[:-1])

    def distances_normalized(self, world_xyz):
        """:meth:`distances_mm` as log1p(d / 3 mm) / log1p(examination diagonal / 3 mm), float32."""
        if self.scribble_tree is None:
            return np.zeros(np.asarray(world_xyz).shape[:-1], dtype=np.float32)
        return (np.log1p(self.distances_mm(world_xyz) / DISTANCE_SCALE_MM)
                / np.log1p(self.examination_diagonal_mm / DISTANCE_SCALE_MM)).astype(np.float32)


def previous_mask_from_change(mask_zyx, previous_change_zyx):
    """The previous mask from the stored change of the last correction ({-1, 0, +1}; None when unknown)."""
    if previous_change_zyx is None:
        return None
    mask = np.asarray(mask_zyx, dtype=bool)
    change = np.asarray(previous_change_zyx)
    if change.shape != mask.shape or not np.isin(change, [-1, 0, 1]).all():
        raise ValueError("the stored change must match the native mask and be -1/0/+1")
    if np.any((change == 1) & ~mask) or np.any((change == -1) & mask):
        raise ValueError("the stored change contradicts the current mask")
    return (mask & ~(change == 1)) | (change == -1)


def augment_state(state: State, augmentation: Augmentation) -> State:
    """The state with the intensity augmentation and, with ``flip_lr``, every array and the affine flipped left-right.

    World positions do not change under the flip (the affine flips too), so source representatives are kept.
    """
    if state.augmentation != Augmentation():
        raise ValueError("the state is already augmented; augmentation must be applied once")
    arrays = {name: getattr(state, name) for name in
              ("mask_zyx", "scribble_zyx", "earlier_addition_scribbles_zyx", "earlier_removal_scribbles_zyx",
               "previous_mask_zyx", "valid_zyx")}
    pet = (state.pet_zyx.astype(np.float32) * augmentation.pet_scale + augmentation.pet_shift)
    ct = state.ct_zyx.astype(np.float32) + augmentation.ct_shift
    affine = state.affine_xyz
    if augmentation.flip_lr:
        arrays = {name: None if value is None else np.flip(value, axis=2) for name, value in arrays.items()}
        pet, ct = np.flip(pet, axis=2), np.flip(ct, axis=2)
        affine = flip_affine_xyz(affine, state.shape_zyx)
    return replace(state, pet_zyx=pet, ct_zyx=ct, affine_xyz=affine, **arrays,
                   pet_pad_value=state.pet_pad_value * augmentation.pet_scale + augmentation.pet_shift,
                   augmentation=augmentation)


# ------------------------------------------------------------------------------------------- device state
def _floor_integer_boundary(value: torch.Tensor) -> torch.Tensor:
    nearest = torch.round(value)
    return torch.floor(torch.where(torch.abs(value - nearest) <= INTEGER_TIE_TOLERANCE, nearest, value))


def scribble_world_points(state: State, device) -> torch.Tensor:
    """FP64 world XYZ of every scribble voxel centre."""
    affine = torch.as_tensor(state.affine_xyz, dtype=torch.float64, device=device)
    index = torch.as_tensor(np.argwhere(np.asarray(state.scribble_zyx, dtype=bool))[:, ::-1].copy(),
                            dtype=torch.float64, device=device)
    return index @ affine[:3, :3].T + affine[:3, 3]


@dataclass
class DeviceState:
    """The arrays of one :class:`State` on one device (uploaded once per state)."""
    state: State
    device: torch.device
    pet: torch.Tensor
    ct: torch.Tensor
    mask: torch.Tensor
    scribble: torch.Tensor
    earlier_addition_scribbles: torch.Tensor
    earlier_removal_scribbles: torch.Tensor
    last_added: torch.Tensor
    last_removed: torch.Tensor
    valid: torch.Tensor
    affine_xyz: np.ndarray
    affine: torch.Tensor
    scribble_world: torch.Tensor

    @property
    def shape_zyx(self):
        return self.state.shape_zyx

    @classmethod
    def build(cls, state: State, device) -> "DeviceState":
        device = torch.device(device)

        def put(array, dtype):
            # Contiguous, writable source: state arrays may be read-only memory maps or flipped views; nothing here
            # ever writes into the uploaded tensors.
            array = np.ascontiguousarray(array)
            if not array.flags.writeable:
                array = array.copy()
            return torch.as_tensor(array, dtype=dtype, device=device)

        valid = put(state.valid_zyx, torch.bool)
        scribble = put(state.scribble_zyx, torch.bool)
        affine = torch.as_tensor(state.affine_xyz, dtype=torch.float64, device=device)
        scribble_world = scribble_world_points(state, device)
        return cls(state, device, put(state.pet_zyx, torch.float32), put(state.ct_zyx, torch.float32),
                   put(state.mask_zyx, torch.bool), scribble, put(state.earlier_addition_scribbles_zyx, torch.bool),
                   put(state.earlier_removal_scribbles_zyx, torch.bool), put(state.last_added, torch.bool),
                   put(state.last_removed, torch.bool), valid, np.asarray(state.affine_xyz, dtype=np.float64),
                   affine, scribble_world)


def world_to_index(points: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    flat = (points.reshape(-1, 3) - affine[:3, 3]).T
    return torch.linalg.solve(affine[:3, :3], flat).T.reshape(points.shape)


def support_valid(points: torch.Tensor, device_state: DeviceState) -> tuple[torch.Tensor, torch.Tensor]:
    """(inside the native extent and valid at the nearest native voxel, that voxel's XYZ index)."""
    finite = torch.isfinite(points).all(dim=-1)
    safe = torch.where(finite[..., None], points, torch.zeros_like(points))
    u = world_to_index(safe, device_state.affine)
    size = torch.tensor(device_state.shape_zyx[::-1], device=points.device, dtype=torch.float64)
    inside = finite & ((u >= -.5 - SUPPORT_TOLERANCE) & (u <= size - .5 + SUPPORT_TOLERANCE)).all(dim=-1)
    pick = _floor_integer_boundary(u + .5).to(torch.long)
    pick = torch.minimum(torch.clamp(pick, min=0), size.to(torch.long) - 1)
    valid = inside & device_state.valid[pick[..., 2], pick[..., 1], pick[..., 0]]
    return valid, pick


def normalized_scribble_distance(points: torch.Tensor, scribble_world: torch.Tensor, shape_zyx,
                                 affine_xyz) -> torch.Tensor:
    """Distance from world points to the nearest scribble voxel centre, log1p(d/3 mm)/log1p(diagonal/3 mm), FP32.

    Exact Euclidean distance by brute force in FP64 blocks; zeros when the state has no scribble voxel.
    """
    shape = points.shape[:-1]
    if scribble_world.shape[0] == 0:
        return torch.zeros(shape, dtype=torch.float32, device=points.device)
    flat = points.reshape(-1, 3).to(torch.float64)
    block = max(1, DISTANCE_PAIR_BUDGET // max(1, scribble_world.shape[0]))
    nearest = torch.empty(flat.shape[0], dtype=torch.float64, device=points.device)
    for start in range(0, flat.shape[0], block):
        difference = flat[start:start + block, None, :] - scribble_world[None, :, :]
        nearest[start:start + block] = torch.sqrt((difference * difference).sum(-1)).min(dim=1).values
    diagonal = float(np.linalg.norm(np.array(tuple(shape_zyx)[::-1]) * spacing_xyz_mm(affine_xyz)))
    value = torch.log1p(nearest / DISTANCE_SCALE_MM) / math.log1p(diagonal / DISTANCE_SCALE_MM)
    return value.to(torch.float32).reshape(shape)


def _scribble_distance(points: torch.Tensor, device_state: DeviceState) -> torch.Tensor:
    return normalized_scribble_distance(points, device_state.scribble_world, device_state.shape_zyx,
                                        device_state.affine_xyz)


# ------------------------------------------------------------------------------------------- grid channels
def _trilinear(volume: torch.Tensor, u_xyz: torch.Tensor) -> torch.Tensor:
    """Trilinear interpolation at XYZ index coordinates with edge extension and FP64 weights."""
    size = torch.tensor(volume.shape[::-1], device=u_xyz.device, dtype=torch.float64)
    c = torch.minimum(torch.clamp(u_xyz, min=0.), size - 1)
    low = torch.floor(c)
    frac = c - low
    low = low.to(torch.long)
    high = torch.minimum(low + 1, size.to(torch.long) - 1)
    out = torch.zeros(u_xyz.shape[:-1], dtype=torch.float64, device=u_xyz.device)
    for dz in (0, 1):
        z = high[..., 2] if dz else low[..., 2]
        wz = frac[..., 2] if dz else 1 - frac[..., 2]
        for dy in (0, 1):
            y = high[..., 1] if dy else low[..., 1]
            wy = frac[..., 1] if dy else 1 - frac[..., 1]
            for dx in (0, 1):
                x = high[..., 0] if dx else low[..., 0]
                wx = frac[..., 0] if dx else 1 - frac[..., 0]
                out = out + wz * wy * wx * volume[z, y, x].to(torch.float64)
    return out


def image_channels(device_state: DeviceState, target_affine_xyz, target_shape_zyx) -> torch.Tensor:
    """PET and CT on a grid, (2, Z, Y, X) FP32; cells outside the valid native extent take the state's pad values."""
    state = device_state.state
    points = grid_world_points(target_shape_zyx, target_affine_xyz, device=device_state.device, dtype=torch.float64)
    u = world_to_index(points, device_state.affine)
    valid, _ = support_valid(points, device_state)
    channels = []
    for volume, pad in ((device_state.pet, state.pet_pad_value), (device_state.ct, -1. + state.augmentation.ct_shift)):
        sampled = _trilinear(volume, u).to(torch.float32)
        channels.append(torch.where(valid, sampled, torch.full_like(sampled, float(np.float32(pad)))))
    return torch.stack(channels)


_DENSE_WEIGHTS: dict = {}


def apply_overlap(operator, native: torch.Tensor) -> torch.Tensor:
    """``CellOverlapOperator.apply`` in FP64 torch on the native tensor's device."""
    if tuple(native.shape) != tuple(operator.native_shape_zyx):
        raise GeometryError(f"source shape actual={tuple(native.shape)}, expected={operator.native_shape_zyx}")
    device = native.device
    key = (id(operator), str(device))
    entry = _DENSE_WEIGHTS.get(key)
    if entry is None or entry[0] is not operator:
        weights = tuple(torch.as_tensor(w.toarray(), dtype=torch.float64, device=device) for w in operator.weights_xyz)
        if len(_DENSE_WEIGHTS) > 64:
            _DENSE_WEIGHTS.clear()
        entry = (operator, weights)
        _DENSE_WEIGHTS[key] = entry
    weights = entry[1]
    array = native.permute(2, 1, 0).permute(*operator.source_axes_for_target_xyz)
    if operator.source_ranges_xyz is not None:
        if any(high <= low for low, high in operator.source_ranges_xyz):
            return torch.zeros(operator.target_shape_zyx, dtype=torch.float64, device=device)
        array = array[tuple(slice(low, high) for low, high in operator.source_ranges_xyz)]
    array = array.to(torch.float64)
    for axis, weight in enumerate(weights):
        array = torch.movedim(torch.tensordot(weight, array, dims=([1], [axis])), 0, axis)
    return array.permute(2, 1, 0)


def state_channels(device_state: DeviceState, target_affine_xyz, target_shape_zyx):
    """The 12 state channels (12, Z, Y, X) FP32 and the cells that overlap valid native voxels (Z, Y, X) bool.

    The mask label is that of the nearest valid native voxel; the covered fraction and every scribble, history and
    last-change channel come from the exact physical cell overlap (a channel marks any positive overlap); the
    distance map is the exact distance to the nearest scribble voxel centre.
    """
    state = device_state.state
    shape = tuple(int(v) for v in target_shape_zyx)
    operator = cell_overlap_operator(device_state.affine_xyz, state.shape_zyx, target_affine_xyz, shape)
    points = grid_world_points(shape, target_affine_xyz, device=device_state.device, dtype=torch.float64)
    centre_valid, pick = support_valid(points, device_state)
    mask_label = torch.where(centre_valid, device_state.mask[pick[..., 2], pick[..., 1], pick[..., 0]],
                             torch.zeros_like(centre_valid)).to(torch.float32)
    valid = device_state.valid
    coverage = apply_overlap(operator, valid) > 0

    def hit(mask):
        return (apply_overlap(operator, mask & valid) > 0).to(torch.float32)

    zero = torch.zeros(shape, dtype=torch.float32, device=device_state.device)
    covered_fraction = apply_overlap(operator, device_state.mask & valid).to(torch.float32)
    earlier_addition, earlier_removal = (hit(device_state.earlier_addition_scribbles),
                                         hit(device_state.earlier_removal_scribbles))
    scribble_hit = hit(device_state.scribble)
    distances = torch.where(centre_valid, _scribble_distance(points, device_state), zero)
    addition_scribble, removal_scribble = (scribble_hit, zero) if state.correction_type == 1 else (zero, scribble_hit)
    addition_distance, removal_distance = (distances, zero) if state.correction_type == 1 else (zero, distances)
    previous_known = coverage.to(torch.float32) * float(state.previous_mask_known)
    channels = torch.stack((mask_label, covered_fraction, earlier_addition, earlier_removal, addition_scribble,
                            removal_scribble, addition_distance, removal_distance, hit(device_state.last_added),
                            hit(device_state.last_removed), previous_known, previous_known))
    return channels, coverage


# ------------------------------------------------------------------------------------------- native voxels
def _native_box(state, window_affine_xyz, window_shape_zyx):
    """(low_xyz, high_xyz) of the native index box around one window's cell extent, or None if empty.

    One extra index of margin keeps voxels at the tolerance boundary in the box.
    """
    shape = np.asarray(window_shape_zyx, dtype=np.float64)[::-1]
    corners = np.array(list(product(*[(-.5, n - .5) for n in shape])))
    native_u = world_to_index_xyz(index_xyz_to_world(corners, window_affine_xyz), state.affine_xyz)
    native_shape_xyz = np.asarray(state.shape_zyx[::-1])
    low = np.clip(np.floor(native_u.min(axis=0)).astype(np.int64) - 1, 0, native_shape_xyz)
    high = np.clip(np.ceil(native_u.max(axis=0)).astype(np.int64) + 2, 0, native_shape_xyz)
    return None if np.any(high <= low) else (low, high)


def _inside_window(index_zyx: torch.Tensor, native_affine: torch.Tensor, window_matrix: torch.Tensor,
                   window_translation: torch.Tensor, size_xyz: torch.Tensor) -> torch.Tensor:
    """Whether native voxels lie inside a window's cell extent (the support_contains_world rule).

    ``window_translation`` is the window origin (3,) or one origin per voxel (N, 3) when voxels of several windows
    that share ``window_matrix`` are tested together; each row is the same subtraction either way.
    """
    xyz = index_zyx.flip(-1).to(torch.float64)
    world = xyz @ native_affine[:3, :3].T + native_affine[:3, 3]
    u = solve(window_matrix, (world - window_translation).T).T
    return ((u >= -.5 - SUPPORT_TOLERANCE) & (u <= size_xyz - .5 + SUPPORT_TOLERANCE)).all(dim=-1)


def window_native_voxels(state, region_zyx, window_affine_xyz, window_shape_zyx, *, chunk_size=DEFAULT_CHUNK,
                         device_state=None) -> np.ndarray:
    """ZYX indices (int64, lexicographic order) of the valid native voxels of ``region_zyx`` inside one window.

    Only the native box around the window is enumerated; its exact extent test runs ``chunk_size`` voxels at a
    time, on ``device_state``'s device when it is this state's, else on the CPU.
    """
    region = np.asarray(region_zyx, dtype=bool)
    if region.shape != state.shape_zyx:
        raise GeometryError(f"region actual={region.shape}, expected={state.shape_zyx}")
    shape = np.asarray(window_shape_zyx, dtype=np.float64)[::-1]
    box_bounds = _native_box(state, window_affine_xyz, window_shape_zyx)
    if box_bounds is None:
        return np.zeros((0, 3), dtype=np.int64)
    low, high = box_bounds
    step = int(chunk_size)
    if step < 1:
        raise ValueError("chunk_size must be positive")
    box = tuple(slice(int(lo), int(hi)) for lo, hi in zip(low[::-1], high[::-1]))
    reuse = device_state is not None and device_state.state is state
    device = torch.device(device_state.device) if reuse else torch.device("cpu")
    valid = device_state.valid[box] if reuse else torch.as_tensor(np.ascontiguousarray(state.valid_zyx[box]))
    inside = torch.as_tensor(np.ascontiguousarray(region[box]), device=device) & valid
    candidates = torch.nonzero(inside) + torch.as_tensor(low[::-1].copy(), dtype=torch.int64, device=device)
    native_affine = torch.as_tensor(state.affine_xyz, dtype=torch.float64, device=device)
    window = torch.as_tensor(np.asarray(window_affine_xyz, dtype=np.float64), dtype=torch.float64, device=device)
    size = torch.as_tensor(shape.copy(), dtype=torch.float64, device=device)
    keep = torch.zeros(candidates.shape[0], dtype=torch.bool, device=device)
    for start in range(0, candidates.shape[0], step):
        keep[start:start + step] = _inside_window(candidates[start:start + step], native_affine, window[:3, :3],
                                                  window[:3, 3], size)
    return candidates[keep].cpu().numpy().astype(np.int64)


def window_native_voxels_batch(device_state, region, window_affines, window_shape_zyx, *,
                               chunk_size=DEFAULT_CHUNK) -> tuple[torch.Tensor, list[int]]:
    """The native voxels of several windows, found together and kept on the device.

    ``region`` is a bool tensor on the state's device. Returns ``(index, counts)``: the voxels of window k are
    ``index[sum(counts[:k]):sum(counts[:k+1])]``, each window with the set and order of
    :func:`window_native_voxels`. Reading the counts is the only host wait.
    """
    state = device_state.state
    device = torch.device(device_state.device)
    if tuple(region.shape) != state.shape_zyx:
        raise GeometryError(f"region actual={tuple(region.shape)}, expected={state.shape_zyx}")
    step = int(chunk_size)
    if step < 1:
        raise ValueError("chunk_size must be positive")
    windows = [np.asarray(affine, dtype=np.float64) for affine in window_affines]
    size = torch.as_tensor(np.asarray(window_shape_zyx, dtype=np.float64)[::-1].copy(), dtype=torch.float64,
                           device=device)
    pieces, box_dims, lows, sizes = [], [], [], []
    for affine in windows:
        bounds = _native_box(state, affine, window_shape_zyx)
        if bounds is None:  # an empty box holds no voxel; its placeholder dimensions are never read
            box_dims.append((1, 1, 1))
            lows.append((0, 0, 0))
            sizes.append(0)
            continue
        low, high = bounds
        box = tuple(slice(int(lo), int(hi)) for lo, hi in zip(low[::-1], high[::-1]))
        piece = (region[box] & device_state.valid[box]).reshape(-1)
        pieces.append(piece)
        box_dims.append(tuple(int(v) for v in (high - low)[::-1]))
        lows.append(tuple(int(v) for v in low[::-1]))
        sizes.append(int(piece.numel()))
    if not pieces:
        return torch.zeros((0, 3), dtype=torch.int64, device=device), [0] * len(windows)
    position = torch.nonzero(torch.cat(pieces)).squeeze(1)
    ends = torch.as_tensor(np.cumsum(sizes), dtype=torch.int64, device=device)
    starts = ends - torch.as_tensor(sizes, dtype=torch.int64, device=device)
    owner_all = torch.searchsorted(ends, position, right=True)
    local = position - starts[owner_all]
    dims = torch.as_tensor(box_dims, dtype=torch.int64, device=device)[owner_all]
    plane = dims[:, 1] * dims[:, 2]
    candidates = torch.stack((local // plane, (local % plane) // dims[:, 2], local % dims[:, 2]), dim=1)
    candidates = candidates + torch.as_tensor(lows, dtype=torch.int64, device=device)[owner_all]
    native_affine = device_state.affine
    matrices = torch.as_tensor(np.stack([a[:3, :3] for a in windows]), dtype=torch.float64, device=device)
    translations = torch.as_tensor(np.stack([a[:3, 3] for a in windows]), dtype=torch.float64, device=device)
    shared = all(np.array_equal(a[:3, :3], windows[0][:3, :3]) for a in windows)
    keep = torch.zeros(candidates.shape[0], dtype=torch.bool, device=device)
    for start in range(0, candidates.shape[0], step):
        part, owner = candidates[start:start + step], owner_all[start:start + step]
        if shared:
            keep[start:start + step] = _inside_window(part, native_affine, matrices[0], translations[owner], size)
        else:  # windows with different axes (not produced by one sliding grid): one solve per window
            for k in torch.unique(owner).tolist():
                rows = owner == k
                keep[start:start + step][rows] = _inside_window(part[rows], native_affine, matrices[k],
                                                                translations[k], size)
    counts = torch.bincount(owner_all[keep], minlength=len(windows)).tolist()
    return candidates[keep], counts


def voxel_values(picked, distance, *, correction_type, previous_mask_known, spacing_xyz, world_xyz,
                 window_affine_xyz):
    """The 20 :data:`VOXEL_CHANNELS` of native voxels (FP32); one formula for training and inference.

    ``picked``: the voxels' (PET, CT, mask, scribble, earlier addition and removal scribbles, last added, last
    removed) as FP32 vectors; ``distance``: their normalised scribble distance; ``world_xyz``: their FP64 world
    positions; the offsets are measured in cells of the window grid ``window_affine_xyz``.
    """
    pet, ct, mask, scribble, earlier_addition, earlier_removal, last_added, last_removed = picked
    count = pet.shape[0]
    device = pet.device
    zero = torch.zeros(count, dtype=torch.float32, device=device)
    addition_scribble, removal_scribble = (scribble, zero) if correction_type == 1 else (zero, scribble)
    addition_distance, removal_distance = (distance, zero) if correction_type == 1 else (zero, distance)
    window = torch.as_tensor(window_affine_xyz, dtype=torch.float64, device=device)
    window_u = solve(window[:3, :3], (world_xyz - window[:3, 3]).T).T
    offset = (window_u - _floor_integer_boundary(window_u + .5)).to(torch.float32)
    known = torch.full((count,), previous_mask_known, dtype=torch.float32, device=device)
    sign = torch.full((count,), float(correction_type), dtype=torch.float32, device=device)
    scalars = torch.stack((pet, ct, mask, addition_scribble, removal_scribble, earlier_addition, earlier_removal,
                           last_added, last_removed, known, known, sign, addition_distance, removal_distance), dim=1)
    return torch.cat((scalars, offset, spacing_xyz.expand(count, 3)), dim=1)


def neighbourhood_values(index, offsets, native_shape, origin, crop_shape_zyx, pet, ct, mask, valid):
    """Channel-major [PET x27, CT x27, mask x27, valid x27] (FP32) of native voxels ``index`` (N, 3).

    The arrays are a crop starting at ``origin`` with shape ``crop_shape_zyx`` (a window's voxel box plus the halo,
    or the whole examination with origin 0). Neighbours outside the examination or not valid are zero with
    validity 0, so a missing neighbour differs from a real zero.
    """
    neighbour = index[:, None, :] + offsets[None]
    inside = ((neighbour >= 0) & (neighbour < native_shape)).all(dim=-1)
    crop = torch.as_tensor(crop_shape_zyx, dtype=torch.int64, device=index.device)
    local = torch.minimum(torch.clamp(neighbour - origin, min=0), crop - 1)
    _, height, width = (int(v) for v in crop_shape_zyx)
    flat = (local[..., 0] * height + local[..., 1]) * width + local[..., 2]
    valid = inside & valid.reshape(-1)[flat]
    zero = torch.zeros((), dtype=torch.float32, device=index.device)
    pet = torch.where(valid, pet.reshape(-1)[flat], zero)
    ct = torch.where(valid, ct.reshape(-1)[flat], zero)
    mask = torch.where(valid, mask.reshape(-1)[flat].to(torch.float32), zero)
    return torch.cat((pet, ct, mask, valid.to(torch.float32)), dim=1)


class WindowVoxelInputs:
    """The native-refinement inputs of chosen native voxels of one window, rebuilt chunk by chunk.

    Per voxel only its ZYX index and normalised scribble distance are stored; world positions, the 20 voxel values
    and the 108 neighbourhood values are rebuilt for each chunk (inside the caller's checkpointed block during
    training). The neighbourhood reads the complete current state, including neighbours outside the window.
    """

    def __init__(self, state, index_zyx, *, device, neighbourhood: bool, chunk_size=DEFAULT_CHUNK,
                 device_state=None):
        index = np.asarray(index_zyx, dtype=np.int64).reshape(-1, 3)
        shape = np.asarray(state.shape_zyx, dtype=np.int64)
        if len(index) and (index.min() < 0 or np.any(index >= shape)):
            raise GeometryError("native voxel index lies outside the native array")
        if len(index) and not state.valid_zyx[tuple(index.T)].all():
            raise GeometryError("native voxels must be valid")
        self.device = torch.device(device)
        self.neighbourhood = bool(neighbourhood)
        self.correction_type = int(state.correction_type)
        self.previous_mask_known = float(state.previous_mask_known)
        if len(index):
            low = np.maximum(index.min(axis=0) - NEIGHBOURHOOD_HALO, 0)
            high = np.minimum(index.max(axis=0) + NEIGHBOURHOOD_HALO + 1, shape)
        else:
            low, high = np.zeros(3, dtype=np.int64), np.ones(3, dtype=np.int64)
        crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(low, high))
        self.crop_origin_zyx = tuple(int(v) for v in low)
        self.crop_shape_zyx = tuple(int(v) for v in high - low)

        reuse = (device_state is not None and device_state.state is state
                 and torch.device(device_state.device) == self.device)
        if reuse:
            def take(tensor, dtype):
                return tensor[crop].to(dtype).contiguous()
            self.pet, self.ct = take(device_state.pet, torch.float32), take(device_state.ct, torch.float32)
            (self.mask, self.scribble, self.earlier_addition_scribbles, self.earlier_removal_scribbles,
             self.last_added, self.last_removed, self.valid) = (take(t, torch.bool) for t in (
                device_state.mask, device_state.scribble, device_state.earlier_addition_scribbles,
                device_state.earlier_removal_scribbles, device_state.last_added, device_state.last_removed,
                device_state.valid))
            scribble_world = device_state.scribble_world
        else:
            def put(volume, dtype):
                # An owned copy of the small crop: state arrays may be read-only views or flips.
                return torch.as_tensor(np.array(np.asarray(volume)[crop], copy=True, order="C"),
                                       dtype=dtype, device=self.device)
            self.pet, self.ct = put(state.pet_zyx, torch.float32), put(state.ct_zyx, torch.float32)
            (self.mask, self.scribble, self.earlier_addition_scribbles, self.earlier_removal_scribbles,
             self.last_added, self.last_removed, self.valid) = (put(v, torch.bool) for v in (
                state.mask_zyx, state.scribble_zyx, state.earlier_addition_scribbles_zyx,
                state.earlier_removal_scribbles_zyx, state.last_added, state.last_removed, state.valid_zyx))
            scribble_world = scribble_world_points(state, self.device)
        self.index = torch.as_tensor(index, dtype=torch.int64, device=self.device)
        self.affine = torch.as_tensor(state.affine_xyz, dtype=torch.float64, device=self.device)
        step = int(chunk_size)
        if step < 1:
            raise ValueError("chunk_size must be positive")
        # Exact scribble distance per voxel (normalized_scribble_distance, FP64 blocks).
        self.distance = torch.zeros(len(index), dtype=torch.float32, device=self.device)
        for start in range(0, len(index), step):
            world = self.world(start, start + step)
            self.distance[start:start + step] = normalized_scribble_distance(world, scribble_world, state.shape_zyx,
                                                                             state.affine_xyz)
        self.spacing_xyz = torch.as_tensor(spacing_xyz_mm(state.affine_xyz), dtype=torch.float32, device=self.device)
        self._origin = torch.as_tensor(low, dtype=torch.int64, device=self.device)
        self._native_shape = torch.as_tensor(shape, dtype=torch.int64, device=self.device)
        self._offsets = torch.as_tensor(NEIGHBOURHOOD_OFFSETS_ZYX, dtype=torch.int64, device=self.device)

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def _flat(self, local_zyx: torch.Tensor) -> torch.Tensor:
        _, height, width = self.crop_shape_zyx
        return (local_zyx[..., 0] * height + local_zyx[..., 1]) * width + local_zyx[..., 2]

    def world(self, start: int, stop: int) -> torch.Tensor:
        xyz = self.index[start:stop].flip(-1).to(torch.float64)
        return xyz @ self.affine[:3, :3].T + self.affine[:3, 3]

    def voxel_values(self, start: int, stop: int, world_xyz: torch.Tensor, window_affine_xyz) -> torch.Tensor:
        """The 20 :data:`VOXEL_CHANNELS` of voxels [start, stop) (FP32)."""
        flat = self._flat(self.index[start:stop] - self._origin)
        picked = tuple(volume.reshape(-1)[flat].to(torch.float32) for volume in (
            self.pet, self.ct, self.mask, self.scribble, self.earlier_addition_scribbles,
            self.earlier_removal_scribbles, self.last_added, self.last_removed))
        return voxel_values(picked, self.distance[start:stop], correction_type=self.correction_type,
                            previous_mask_known=self.previous_mask_known, spacing_xyz=self.spacing_xyz,
                            world_xyz=world_xyz, window_affine_xyz=window_affine_xyz)

    def neighbourhood_values(self, start: int, stop: int) -> torch.Tensor:
        """Channel-major [PET x27, CT x27, mask x27, valid x27] (FP32) of voxels [start, stop)."""
        return neighbourhood_values(self.index[start:stop], self._offsets, self._native_shape, self._origin,
                                    self.crop_shape_zyx, self.pet, self.ct, self.mask, self.valid)

    def chunk(self, start: int, stop: int, window_affine_xyz):
        """(world FP64, voxel values, neighbourhood values or None) of voxels [start, stop)."""
        world = self.world(start, stop)
        values = self.voxel_values(start, stop, world, window_affine_xyz)
        return world, values, self.neighbourhood_values(start, stop) if self.neighbourhood else None


class PassVoxelInputs:
    """The window-independent native-refinement inputs of one state's inference pass, computed once per voxel.

    A native voxel lies in up to eight overlapping windows. Its scribble distance and its neighbourhood code
    ``encoder(neighbourhood values)`` do not depend on the window, so they are computed once here (``chunk_size``
    voxels at a time, with the formulas of :class:`WindowVoxelInputs`) and every window gathers them; the
    window-dependent offsets are still computed per window. Both live in whole-examination arrays indexed by the
    voxel's flat ZYX position.
    """

    def __init__(self, device_state, region, *, encoder=None, chunk_size=DEFAULT_CHUNK):
        state = device_state.state
        self.state, self.device = state, torch.device(device_state.device)
        step = int(chunk_size)
        if step < 1:
            raise ValueError("chunk_size must be positive")
        shape = state.shape_zyx
        self.strides = torch.as_tensor([shape[1] * shape[2], shape[2], 1], dtype=torch.int64, device=self.device)
        self.affine = device_state.affine
        self.correction_type = int(state.correction_type)
        self.previous_mask_known = float(state.previous_mask_known)
        self.spacing_xyz = torch.as_tensor(spacing_xyz_mm(state.affine_xyz), dtype=torch.float32, device=self.device)
        self.volumes = (device_state.pet, device_state.ct, device_state.mask, device_state.scribble,
                        device_state.earlier_addition_scribbles, device_state.earlier_removal_scribbles,
                        device_state.last_added, device_state.last_removed)
        native_shape = torch.as_tensor(shape, dtype=torch.int64, device=self.device)
        origin = torch.zeros(3, dtype=torch.int64, device=self.device)
        offsets = torch.as_tensor(NEIGHBOURHOOD_OFFSETS_ZYX, dtype=torch.int64, device=self.device)
        count = int(np.prod(shape))
        self.distance = torch.zeros(count, dtype=torch.float32, device=self.device)
        self.code = None
        voxels = torch.nonzero(region & device_state.valid)
        for start in range(0, voxels.shape[0], step):
            part = voxels[start:start + step]
            flat = (part * self.strides).sum(-1)
            self.distance[flat] = normalized_scribble_distance(self.world(part), device_state.scribble_world, shape,
                                                               state.affine_xyz)
            if encoder is not None:
                code = encoder(neighbourhood_values(part, offsets, native_shape, origin, shape, device_state.pet,
                                                    device_state.ct, device_state.mask, device_state.valid))
                if self.code is None:  # FP32 store, also when the encoder ran under a reduced-precision autocast
                    self.code = torch.zeros((count, code.shape[-1]), dtype=torch.float32, device=self.device)
                self.code[flat] = code.to(torch.float32)

    def world(self, index: torch.Tensor) -> torch.Tensor:
        """FP64 world XYZ of native voxels (the WindowVoxelInputs.world formula)."""
        return index.flip(-1).to(torch.float64) @ self.affine[:3, :3].T + self.affine[:3, 3]

    def window_inputs(self, index: torch.Tensor, window_affine_xyz):
        """(world, voxel values, neighbourhood code or None) of the native voxels ``index`` (N, 3) of one window."""
        flat = (index * self.strides).sum(-1)
        world = self.world(index)
        picked = tuple(volume.reshape(-1)[flat].to(torch.float32) for volume in self.volumes)
        values = voxel_values(picked, self.distance[flat], correction_type=self.correction_type,
                              previous_mask_known=self.previous_mask_known, spacing_xyz=self.spacing_xyz,
                              world_xyz=world, window_affine_xyz=window_affine_xyz)
        return world, values, None if self.code is None else self.code[flat]
