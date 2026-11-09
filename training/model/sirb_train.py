"""SIRBTrain: SIRB during training, one view at a time (the counterpart of SAM 2's ``SAM2Train``).

For one view (``sirb.data.pairs.TrainingView``: one state of a unit in one window) :meth:`SIRBTrain.forward`

1. builds the state context (``SIRB.prepare_context``) from the 14 input channels on the whole-volume grid and on the
   scribble window, and the scribble's sources; :class:`SharedStateContexts` builds it once for both windows of a
   state;
2. runs the window (``SIRB.forward_window``) on its 14 input channels;
3. picks the supervised voxels, the editable valid native voxels inside the window, and refines them in chunks of
   ``voxel_chunk`` voxels (``SIRB.native_role_logits``), keeping only the per-chunk sums of the native CE and Dice
   loss (``training.loss_fns.native_statistics``); with ``recompute_native_voxels`` each chunk is recomputed in the
   backward pass (memory only);
4. adds the source CE of the 12 mm and 6 mm source role logits and returns the view total and monitoring readings.

The variants change what a view supervises (the model switches change the network itself, ``sirb.modeling.sirb``):

* No scribble conditioning: the window of every view is centred on a valid native voxel drawn uniformly, independent
  of the scribble (one draw per unit and window index, :func:`scribble_independent_window`); there is no source
  CE;
* No correct-tissue role: correct-tissue voxels are not supervised (they become ``IGNORE`` before every loss);
* Merged non-target roles: other errors and correct tissue share one likelihood in both cross-entropies
  (``training.loss_fns.role_log_likelihoods``);
* Coarse roles only: nothing changes here.

Ground truth reaches only the labels of the losses; the model reads the state.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from sirb.data.geometry import (GeometryError, examination_bounds_xyz, examination_grids, flip_affine_xyz,
                                index_xyz_to_world, window_affine_xyz)
from sirb.data.observations import DeviceState, WindowVoxelInputs, image_channels, state_channels, window_native_voxels
from sirb.data.pairs import RoleLabels, _origin_at_point
from sirb.data.roles import CORRECT_TISSUE, IGNORE, ROLE_COUNT, TARGET_ERROR
from sirb.interaction.update_rule import TARGET_THRESHOLD
from sirb.modeling.role_decoder import LEVELS
from sirb.modeling.sirb import SIRB, StateContext
from training.loss_fns import (LOSS_WEIGHT_KEYS, NATIVE_BINS, native_ce_and_dice, native_statistics,
                               role_log_likelihoods, source_ce, source_labels_for_points, view_total)

__all__ = ["SCRIBBLE_INDEPENDENT_WINDOW_STREAM", "SIRBTrain", "SharedStateContexts", "plain_readings",
           "scribble_independent_window", "stream_word"]

#: Recent device copies of view states (a view's state, the other state of its unit and their flipped twins).
DEVICE_STATE_CACHE = 4
#: The random stream of the scribble-independent windows of the No scribble conditioning variant.
SCRIBBLE_INDEPENDENT_WINDOW_STREAM = "scribble_independent_window"


def stream_word(name: str) -> int:
    """The 64-bit seed word of a named random stream: the first 8 bytes of SHA-256(name), little-endian."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"stream name actual={name!r}, expected a non-empty string")
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")


def scribble_independent_window(view, *, grid_spacing_mm, whole_volume_shape):
    """The view with its window centred on a valid native voxel of its examination, drawn uniformly with
    ``SeedSequence([source seed, stream word, window index])``; both states of a unit get the same window."""
    state = view.state
    flip = bool(view.augmentation.flip_lr)
    native_affine = flip_affine_xyz(state.affine_xyz, state.shape_zyx) if flip else state.affine_xyz
    grid = examination_grids(state.shape_zyx, native_affine, view.window_size, grid_spacing_mm=grid_spacing_mm,
                             whole_volume_shape_zyx=whole_volume_shape).grid_affine_xyz
    valid = state.valid_zyx
    flat = None if bool(valid.all()) else np.flatnonzero(valid)
    if flat is not None and not len(flat):
        raise GeometryError("the examination has no valid native voxel")
    count = valid.size if flat is None else len(flat)
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(view.source_seed), stream_word(SCRIBBLE_INDEPENDENT_WINDOW_STREAM), int(view.window_index)]))
    chosen = int(rng.integers(count))
    centre = tuple(int(v) for v in np.unravel_index(chosen if flat is None else int(flat[chosen]), valid.shape))
    world = index_xyz_to_world(np.asarray(centre[::-1], dtype=np.float64)[None], state.affine_xyz)[0]
    origin = _origin_at_point(world, grid, view.window_size)
    affine = window_affine_xyz(grid, origin)
    if flip:
        affine = flip_affine_xyz(affine, (view.window_size,) * 3)
    return replace(view, window_origin_zyx=origin, window_affine_xyz=affine)


def plain_readings(value):
    """Readings as JSON values: tensors become numbers or lists (read after the view's backward pass)."""
    if torch.is_tensor(value):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, dict):
        return {key: plain_readings(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_readings(item) for item in value]
    return value


class SIRBTrain(nn.Module):
    """SIRB with the supervision of its training views (module docstring).

    ``loss_weights`` holds :data:`training.loss_fns.LOSS_WEIGHT_KEYS`; ``dice_epsilon`` is the Dice loss epsilon;
    ``grid_spacing_mm`` and ``whole_volume_shape`` are the model section's. ``voxel_chunk`` and
    ``recompute_native_voxels`` change memory and time only.
    """

    def __init__(self, model: SIRB, *, grid_spacing_mm, whole_volume_shape, loss_weights: Mapping[str, float],
                 dice_epsilon: float, merged_non_target_roles: bool, voxel_chunk: int,
                 recompute_native_voxels: bool) -> None:
        super().__init__()
        if not isinstance(model, SIRB):
            raise TypeError(f"model must be a SIRB, got {type(model).__name__}")
        if sorted(loss_weights) != sorted(LOSS_WEIGHT_KEYS):
            raise ValueError(f"loss weights actual={sorted(loss_weights)}, expected {sorted(LOSS_WEIGHT_KEYS)}")
        for name, value in (("merged_non_target_roles", merged_non_target_roles),
                            ("recompute_native_voxels", recompute_native_voxels)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, got {value!r}")
        if isinstance(voxel_chunk, bool) or not isinstance(voxel_chunk, int) or voxel_chunk < 1:
            raise ValueError(f"voxel_chunk must be a positive integer, got {voxel_chunk!r}")
        self.model = model
        self.grid_spacing_mm = tuple(float(v) for v in grid_spacing_mm)
        self.whole_volume_shape = tuple(int(v) for v in whole_volume_shape)
        self.loss_weights = {key: float(loss_weights[key]) for key in LOSS_WEIGHT_KEYS}
        self.dice_epsilon = float(dice_epsilon)
        self.merged_non_target_roles = merged_non_target_roles
        self.voxel_chunk = voxel_chunk
        self.recompute_native_voxels = recompute_native_voxels
        self._device_states: OrderedDict[int, DeviceState] = OrderedDict()
        self._relabelled: tuple[Any, RoleLabels] | None = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def window_size(self) -> int:
        return int(self.model.window_size)

    # ------------------------------------------------------------------ inputs
    def device_state(self, state) -> DeviceState:
        """The state's arrays on the model's device, uploaded once per state object."""
        key = id(state)
        entry = self._device_states.get(key)
        if entry is None or entry.state is not state or entry.device != self.device:
            entry = DeviceState.build(state, self.device)
            self._device_states[key] = entry
            while len(self._device_states) > DEVICE_STATE_CACHE:
                self._device_states.popitem(last=False)
        else:
            self._device_states.move_to_end(key)
        return entry

    def _affine(self, affine_xyz) -> torch.Tensor:
        return torch.as_tensor(np.asarray(affine_xyz, dtype=np.float64), dtype=torch.float64, device=self.device)

    @staticmethod
    def observe(device_state: DeviceState, affine_xyz, shape_zyx):
        """(image channels [1,2,...], state channels [1,12,...], valid cells [1,1,...]) of a grid."""
        image = image_channels(device_state, affine_xyz, shape_zyx)
        channels, valid = state_channels(device_state, affine_xyz, shape_zyx)
        return image[None], channels[None], valid[None, None]

    def whole_volume_grid(self, view) -> tuple[np.ndarray, tuple[int, int, int]]:
        """(affine, shape) of the whole-volume grid of the view's state; for a flipped state the grid of the original
        examination, flipped."""
        state = view.state
        flip = bool(view.augmentation.flip_lr)
        native_affine = flip_affine_xyz(state.affine_xyz, state.shape_zyx) if flip else state.affine_xyz
        grids = examination_grids(state.shape_zyx, native_affine, self.window_size,
                                  grid_spacing_mm=self.grid_spacing_mm, whole_volume_shape_zyx=self.whole_volume_shape)
        affine = grids.whole_volume_affine_xyz
        if flip:
            affine = flip_affine_xyz(affine, grids.whole_volume_shape_zyx)
        return affine, grids.whole_volume_shape_zyx

    def view_context(self, view) -> StateContext:
        """The state context of a view's state (with gradients)."""
        state = view.state
        device_state = self.device_state(state)
        whole_affine, whole_shape = self.whole_volume_grid(view)
        whole_image, whole_channels, whole_valid = self.observe(device_state, whole_affine, whole_shape)
        window_affine = view.representatives.window_affine_xyz
        window_image, window_channels, window_valid = self.observe(device_state, window_affine,
                                                                   (self.window_size,) * 3)
        return self.model.prepare_context(
            torch.cat((whole_image, whole_channels), 1), self._affine(whole_affine), window_image, window_channels,
            self._affine(window_affine), view.sources, torch.tensor([state.correction_type], device=self.device),
            self._affine(examination_bounds_xyz(state.shape_zyx, state.affine_xyz)), whole_volume_valid=whole_valid,
            window_valid=window_valid)

    def shared_contexts(self) -> "SharedStateContexts":
        """A training iteration in which the two windows of a state share its context."""
        return SharedStateContexts(self)

    # ------------------------------------------------------------------ variants
    def prepare_view(self, view):
        """The view as the variant supervises it (module docstring)."""
        if not self.model.scribble_conditioning:
            view = scribble_independent_window(view, grid_spacing_mm=self.grid_spacing_mm,
                                       whole_volume_shape=self.whole_volume_shape)
        if not self.model.correct_tissue_role:
            if self._relabelled is None or self._relabelled[0] is not view.roles:
                labels = np.asarray(view.labels)
                ignored = np.where(labels == CORRECT_TISSUE, np.asarray(IGNORE, dtype=labels.dtype), labels)
                self._relabelled = (view.roles, RoleLabels(ignored, tuple(view.roles.touched_components)))
            view = replace(view, roles=self._relabelled[1])
        return view

    # ------------------------------------------------------------------ one view
    def forward(self, view, *, monitor: bool = True, context: StateContext | None = None):
        """(view total, readings) of one view; ``monitor=False`` keeps the readings to what needs no device read.

        ``context`` is the state's context when the caller shares it between windows (:class:`SharedStateContexts`).
        """
        if context is None:
            context = self.view_context(view)
        view = self.prepare_view(view)
        state = view.state
        device_state = self.device_state(state)
        window_affine = np.asarray(view.window_affine_xyz, dtype=np.float64)
        shape = (self.window_size,) * 3
        image, channels, valid = self.observe(device_state, window_affine, shape)
        affine = self._affine(window_affine)
        output = self.model.forward_window(image, channels, affine, context, diagnostics=monitor)
        labels = np.asarray(view.labels)
        index = window_native_voxels(state, labels >= 0, window_affine, shape, chunk_size=self.voxel_chunk,
                                     device_state=device_state)
        readings: dict[str, Any] = {"correction_type": int(state.correction_type), "native_voxels": int(len(index)),
                                    "view_kind": view.view_kind}
        if not len(index):
            # A zero connected to the window outputs: the view still counts in the mean over the iteration's views.
            zero = output["coarse_role_logits"].sum() * 0 + sum(value.sum() * 0
                                                               for value in output["source_role_logits"].values())
            if monitor:
                readings.update(role_voxels=[0] * ROLE_COUNT, accepted_voxels=[0] * ROLE_COUNT)
            return zero, readings
        selected = labels[tuple(index.T)]
        point_labels = source_labels_for_points(state, index, selected, view.targets)
        bins = selected + ((selected == CORRECT_TISSUE) & point_labels.shell).astype(np.int64)
        counts = np.bincount(bins, minlength=NATIVE_BINS)
        role_labels = torch.as_tensor(selected, device=self.device, dtype=torch.long)
        bin_labels = torch.as_tensor(bins, device=self.device, dtype=torch.long)
        voxels = WindowVoxelInputs(state, index, device=self.device, neighbourhood=True, chunk_size=self.voxel_chunk,
                                   device_state=device_state)
        fine = LEVELS[1]

        def block(features, coarse, role_logits, source_vectors, start, stop):
            outputs = dict(output, features=features, coarse_role_logits=coarse, source_vectors=source_vectors,
                           source_role_logits=dict(output["source_role_logits"], **{fine: role_logits}))
            world, values, neighbourhood = voxels.chunk(start, stop, window_affine)
            logits, inside = self.model.native_role_logits(outputs, world, values[None], window_affine_xyz=affine,
                                                           window_valid=valid, neighbourhood=neighbourhood[None])
            likelihoods = role_log_likelihoods(logits[0], merged_non_target_roles=self.merged_non_target_roles)
            statistics = native_statistics(likelihoods, role_labels[start:stop], bin_labels[start:stop])
            accepted = likelihoods[:, TARGET_ERROR].exp() >= TARGET_THRESHOLD
            accepted = torch.stack([(accepted & (role_labels[start:stop] == role)).sum()
                                    for role in range(ROLE_COUNT)]).detach()
            return statistics, accepted, inside[0].all()

        totals, accepted, covered = [], [], []
        recompute = self.recompute_native_voxels and self.model.training and torch.is_grad_enabled()
        for start in range(0, len(voxels), self.voxel_chunk):
            arguments = (output["features"], output["coarse_role_logits"], output["source_role_logits"][fine],
                         output["source_vectors"], start, min(start + self.voxel_chunk, len(voxels)))
            if recompute:
                statistics, counted, inside = checkpoint(block, *arguments, use_reentrant=False)
            else:
                statistics, counted, inside = block(*arguments)
            totals.append(statistics)
            accepted.append(counted)
            covered.append(inside)
        if not bool(torch.stack(covered).all()):
            raise GeometryError("supervised native voxels lie outside the valid cells of their window")
        native_ce, dice = native_ce_and_dice(torch.stack(totals).sum(0), counts, epsilon=self.dice_epsilon)
        source, source_readings = None, {}
        if self.model.scribble_conditioning:
            rng = np.random.default_rng(np.random.SeedSequence([int(view.source_seed), int(view.state_index),
                                                                int(view.window_index)]))
            source, source_readings = source_ce(output["source_role_logits"], output["source_role_affines"],
                                                point_labels, rng=rng, support_affine_xyz=affine,
                                                support_shape_zyx=shape, support_valid_zyx=valid,
                                                merged_non_target_roles=self.merged_non_target_roles, monitor=monitor)
        loss = view_total(native_ce, source, dice, weights=self.loss_weights)
        if monitor:
            readings.update(role_voxels=[int((selected == role).sum()) for role in range(ROLE_COUNT)],
                            accepted_voxels=torch.stack(accepted).sum(0),
                            correct_tissue_shell_voxels=int(counts[NATIVE_BINS - 1]),
                            sources=int(point_labels.source_valid.sum()), source_ce=source_readings,
                            loss_terms={"native_ce": native_ce.detach(),
                                        "source_ce": None if source is None else source.detach(),
                                        "dice": None if dice is None else dice.detach()})
        return loss, readings


# ------------------------------------------------------------------------------------------- shared state contexts
def _cut_graph(value, cut: dict):
    """``value`` with every tensor that requires grad replaced by a detached leaf; ``cut`` maps id -> (root, leaf).

    Walks dataclasses (the state context), tuples, lists and dicts; one leaf per root tensor.
    """
    if torch.is_tensor(value):
        if not value.requires_grad:
            return value
        if id(value) not in cut:
            cut[id(value)] = (value, value.detach().requires_grad_())
        return cut[id(value)][1]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{f.name: _cut_graph(getattr(value, f.name), cut) for f in dataclasses.fields(value)})
    if type(value) in (tuple, list):
        return type(value)(_cut_graph(item, cut) for item in value)
    if type(value) is dict:
        return {key: _cut_graph(item, cut) for key, item in value.items()}
    return value


class _SharedContext:
    """One state's context, given to its windows as detached leaves; their gradients go back in one backward pass."""

    def __init__(self, view, context: StateContext) -> None:
        self.state, self.sources, self.representatives = view.state, view.sources, view.representatives
        self.flip_lr = bool(view.augmentation.flip_lr)
        cut: dict = {}
        self.context = _cut_graph(context, cut)
        self.roots, self.leaves = zip(*cut.values()) if cut else ((), ())
        self.views = 0
        self.backward_passes = 0

    def serves(self, view) -> bool:
        return (view.state is self.state and view.sources is self.sources
                and view.representatives is self.representatives and bool(view.augmentation.flip_lr) == self.flip_lr)

    def count_backward(self, _gradient) -> None:
        self.backward_passes += 1

    def backward(self) -> None:
        """Backpropagate the windows' accumulated context gradients through the context once."""
        if self.backward_passes != self.views:
            raise RuntimeError(f"a shared state context was released after the backward pass of {self.backward_passes} "
                               f"of its {self.views} views; each view needs its backward pass before the next view")
        pairs = [(root, leaf.grad) for root, leaf in zip(self.roots, self.leaves) if leaf.grad is not None]
        if pairs:
            torch.autograd.backward([root for root, _ in pairs], [gradient for _, gradient in pairs])


class SharedStateContexts:
    """One training iteration in which the two windows of a state share the state's context.

    A context depends only on the state, its flip, its sources and the weights, which change only at the optimizer
    step after all views. It is computed once per state: each window's forward pass reads detached leaf copies, each
    window's backward pass accumulates into them, and when the next state starts (or :meth:`finish` runs after the
    last view's backward pass) the summed gradient goes back through the context once. This equals building the
    context per window up to float rounding. Views must alternate forward and backward passes.
    """

    def __init__(self, train_model: SIRBTrain) -> None:
        self.train_model = train_model
        self.contexts_built = 0
        self._shared: _SharedContext | None = None

    def view_loss(self, view, *, monitor: bool = True):
        shared = self._shared
        if shared is None or not shared.serves(view):
            self._release()
            shared = self._shared = _SharedContext(view, self.train_model.view_context(view))
            self.contexts_built += 1
        loss, readings = self.train_model(view, monitor=monitor, context=shared.context)
        shared.views += 1
        if loss.requires_grad:
            loss.register_hook(shared.count_backward)
        return loss, readings

    def _release(self) -> None:
        shared, self._shared = self._shared, None
        if shared is not None:
            shared.backward()

    def finish(self) -> None:
        """After the last view's backward pass: the last state's context backward pass."""
        self._release()

    def discard(self) -> None:
        """On failure: drop the context and its graph without a backward pass."""
        self._shared = None

    def __enter__(self) -> "SharedStateContexts":
        return self

    def __exit__(self, *_) -> None:
        self.discard()
