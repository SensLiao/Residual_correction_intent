"""Training views of state pairs and scribble pairs, and how the units of each training iteration are chosen.

Every training iteration trains on two units of four views each:

* a state-pair unit: the two states of a stored state pair (same scribble, two masks), each seen in two windows of
  the 3 mm grid: the scribble window (centred on the scribble's first source representative) and a window centred on
  a target-error voxel, drawn from the half of the target error near the scribble on even ``iteration + unit`` and
  from the far half otherwise (preferring voxels the scribble window does not show). With the configured quota one
  state-pair request per block of iterations becomes a target-exhausted unit, whose second state has the whole target
  error corrected (:func:`target_exhausted_views`);
* a scribble-pair unit: one state of a stored pair with its own scribble and a second scribble that the simulated
  user draws on one of its other-error components (same mask, two scribbles). Up to eight components are tried in an
  order drawn with probability growing with the square root of their volume; the first that gives a valid second
  scribble whose target error is disjoint from the first and a window showing both targets is used. Both windows show
  both targets: the first at the middle of the feasible window origins, the second the feasible origin (from 16
  draws) that maximises the smaller of the two target fractions seen in either window. When no component works the
  unit falls back to the stored state pair (:func:`scribble_pair_views`).

A view is one state in one window (:class:`TrainingView`): the augmented state (one left-right flip and intensity
draw per unit, :func:`sirb.data.observations.draw_augmentation`), the window, the roles on the native grid, the
scribble's source representatives and sources, and the seed of the source-CE samples.

Which stored states a unit uses (:class:`TrainingCourse`, :class:`PoolSampler`): per block of eight iterations each
request type gets eight source slots, four ground-truth-corrected and four model states (earlier-version states, two
drawn naturally and two weighted by their recoverable gap, until the self-generated states of a patient half replace
two of them), in an order drawn per block. Within a pool a row is drawn patient -> examination -> state uniformly
(or by the gap table), then a pair type by ``data.state_pairs.type_mix``. All draws come from named random streams
(:class:`RandomStreams`) named by :func:`stream_name` after their purpose, the unit kind and the pool, so the same seed
and pool names give the same course.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .geometry import (GeometryError, SourceRepresentatives, examination_grids, flip_affine_xyz,
                       floor_integer_boundary, index_xyz_to_world, scribble_representatives,
                       transform_representatives, window_affine_xyz, world_to_index_xyz)
from .observations import Augmentation, State, augment_state, draw_augmentation, previous_mask_from_change
from .preprocessing import ExaminationCache, ExaminationInputs, PetReference
from .roles import Roles, correction_sign, make_roles
from .states import (PAIR_TYPES, StatePair, choose_pair_type, load_training_state_pair, read_pool_rows,
                     resolve_pair_path)

__all__ = ["SCRIBBLE_PAIR", "STATE_PAIR", "TARGET_EXHAUSTED", "GapTable", "PairViews", "PoolSampler", "RandomStreams",
           "RoleLabels", "TrainingCourse", "TrainingPairSampler", "TrainingView", "UnitContext", "UnitRequest",
           "attach_sources", "compact_views", "error_component_order", "exhausted_mask", "mixing_slots",
           "pair_states", "prepare_unit", "read_training_pools", "scribble_pair_views", "state_pair_views",
           "stream_name", "target_exhausted_views"]

#: Unit kinds and modes.
STATE_PAIR, SCRIBBLE_PAIR = "state_pair", "scribble_pair"
UNIT_KINDS = (STATE_PAIR, SCRIBBLE_PAIR)
NORMAL, TARGET_EXHAUSTED = "normal", "target_exhausted"
#: View kinds (a scribble-pair unit without a usable second scribble falls back to its state pair).
VIEW_STATE_PAIR, VIEW_TARGET_EXHAUSTED = "state_pair", "target_exhausted"
VIEW_SCRIBBLE_PAIR, VIEW_STATE_PAIR_FALLBACK = "scribble_pair", "state_pair_fallback"
#: Scribble-pair search: other-error components tried, and random draws of the second window.
SCRIBBLE_PAIR_CANDIDATES = 16 // 2
SECOND_WINDOW_DRAWS = 16
#: The per-unit seeds, each drawn from its own named stream (:func:`stream_name`).
SEED_NAMES = ("window", "scribble", "augmentation", "source_samples", "error_order")
_SEED_HIGH = np.iinfo(np.int64).max


def stream_name(purpose: str, kind: str, pool: str | None = None) -> str:
    """Name of a random stream: ``<purpose>-<unit kind>`` or ``<purpose>-<unit kind>-<pool>``. Purposes: ``source``
    (pool order of a block), ``target_exhausted`` (target-exhausted positions of a block), ``row`` (rows of a pool) and
    the per-unit seeds :data:`SEED_NAMES`."""
    return f"{purpose}-{kind}" if pool is None else f"{purpose}-{kind}-{pool}"


# ------------------------------------------------------------------------------------------- views
@dataclass(frozen=True)
class RoleLabels:
    """The part of a view's roles the training step reads: role labels (int8, roles.IGNORE outside the editable
    domain) and the touched error components."""
    labels: np.ndarray
    touched_components: tuple


@dataclass(frozen=True)
class TrainingView:
    """One state of a unit in one window of the 3 mm grid (module docstring).

    ``roles`` are full :class:`sirb.data.roles.Roles` until :func:`compact_views` replaces them by
    :class:`RoleLabels`; ``sources`` are the scribble's sources (:func:`sirb.modeling.source_encoder.group_scribble`);
    ``targets`` is what the ``view_targets`` hook of :func:`attach_sources` computed per state (the source targets of
    the loss); ``source_seed`` seeds the source-CE samples with ``(state_index, window_index)``.
    """
    state: State
    window_origin_zyx: tuple[int, int, int]
    window_affine_xyz: np.ndarray
    roles: Roles | RoleLabels
    representatives: SourceRepresentatives
    window_size: int
    augmentation: Augmentation
    state_index: int
    window_index: int
    sources: Any = None
    targets: Any = None
    source_seed: int = 0
    view_kind: str = ""

    @property
    def labels(self) -> np.ndarray:
        return self.roles.labels


@dataclass(frozen=True)
class PairViews:
    """The four views of one unit: (state 0, window 0), (state 0, window 1), (state 1, window 0), (state 1,
    window 1)."""
    views: tuple
    kind: str
    diagnostics: dict

    def __iter__(self):
        return iter(self.views)

    def __getitem__(self, index):
        return self.views[index]

    def __len__(self):
        return len(self.views)


def _check_same_examination(before: State, after: State) -> None:
    if before.shape_zyx != after.shape_zyx or not np.allclose(before.affine_xyz, after.affine_xyz, atol=1e-8, rtol=0):
        raise ValueError("the states of a pair must share the native grid")
    if before.examination_id != after.examination_id or before.patient_id != after.patient_id:
        raise ValueError("the states of a pair must share the examination and the patient")
    if before.correction_type != after.correction_type:
        raise ValueError("the states of a state pair have different correction types")
    if not np.array_equal(before.scribble_zyx, after.scribble_zyx):
        raise ValueError("the states of a state pair must share the scribble")
    for name in ("pet_zyx", "ct_zyx", "earlier_addition_scribbles_zyx", "earlier_removal_scribbles_zyx", "valid_zyx"):
        left, right = getattr(before, name), getattr(after, name)
        if left is not right and not np.array_equal(left, right):
            raise ValueError(f"the states of a pair must share {name}")


def _target_world(state: State, roles: Roles) -> np.ndarray:
    return index_xyz_to_world(np.argwhere(roles.target_error)[:, ::-1], state.affine_xyz)


def _target_visibility(state: State, roles: Roles, origin, grid_affine_xyz, window_size: int) -> np.ndarray:
    """Which target-error voxel centres lie inside the window (centres 0..P-1, not the padded cell faces)."""
    points = _target_world(state, roles)
    if not len(points):
        return np.zeros(0, dtype=bool)
    u = world_to_index_xyz(points, window_affine_xyz(grid_affine_xyz, origin))
    return ((u >= -1e-7) & (u <= window_size - 1 + 1e-7)).all(axis=1)


def _origin_at_point(world_xyz, grid_affine_xyz, window_size: int) -> tuple[int, int, int]:
    u = world_to_index_xyz(np.asarray(world_xyz), grid_affine_xyz)
    xyz = floor_integer_boundary(u - (window_size - 1) / 2 + .5)
    return tuple(int(v) for v in xyz[::-1])


def _state_windows(before: State, roles: Roles, representatives: SourceRepresentatives, grid_affine_xyz,
                   window_size: int, rng, iteration: int, unit: int):
    first = representatives.window_origin_zyx
    points = _target_world(before, roles)
    if not len(points):
        return (first, first), {"second_window_half": "empty_target", "duplicate_window": True}
    distance = before.distances_mm(points)
    ordered = np.argsort(distance, kind="stable")
    split = (len(ordered) + 1) // 2                      # equal halves, the odd extra voxel near
    halves = (ordered[:split], ordered[split:])
    wanted = int((int(iteration) + int(unit)) % 2)
    visible = _target_visibility(before, roles, first, grid_affine_xyz, window_size)
    selected, selected_half = None, wanted
    for half, unseen_only in ((wanted, True), (wanted, False), (1 - wanted, False)):
        pool = halves[half]
        if unseen_only:
            pool = pool[~visible[pool]]
        if len(pool):
            selected, selected_half = pool, half
            break
    chosen = int(selected[int(rng.integers(len(selected)))])
    second = _origin_at_point(points[chosen], grid_affine_xyz, window_size)
    return (first, second), {"second_window_half": "near" if selected_half == 0 else "far",
                             "duplicate_window": first == second, "first_visible_target_voxels": int(visible.sum()),
                             "target_voxels": len(points)}


def _build_views(states, ground_truth, origins, representatives, grid_affine_xyz, window_size: int,
                 augmentation: Augmentation) -> tuple:
    truth = np.flip(ground_truth, axis=2) if augmentation.flip_lr else np.asarray(ground_truth)
    views = []
    for state_index, (state, representative) in enumerate(zip(states, representatives)):
        augmented = augment_state(state, augmentation)
        roles = make_roles(augmented.mask_zyx, truth, augmented.scribble_zyx, augmented.correction_type,
                           valid_zyx=augmented.valid_zyx)
        representative = transform_representatives(representative, state.shape_zyx, window_size,
                                                   flip_lr=augmentation.flip_lr)
        for window_index, origin in enumerate(origins):
            affine = window_affine_xyz(grid_affine_xyz, origin)
            if augmentation.flip_lr:
                affine = flip_affine_xyz(affine, (window_size,) * 3)
            views.append(TrainingView(augmented, origin, affine, roles, representative, window_size, augmentation,
                                      state_index, window_index))
    return tuple(views)


def state_pair_views(before: State, after: State, ground_truth, *, grid_affine_xyz, window_size: int,
                     rng: np.random.Generator, iteration: int = 0, unit: int = 0,
                     augmentation: Augmentation | None = None,
                     window_rng: np.random.Generator | None = None) -> PairViews:
    """The four views of a state pair (module docstring); the windows come from ``before``."""
    _check_same_examination(before, after)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    p = int(window_size)
    representatives = scribble_representatives(before.scribble_zyx, before.affine_xyz, grid_affine_xyz, p,
                                               native_valid_zyx=before.valid_zyx)
    roles = make_roles(before.mask_zyx, ground_truth, before.scribble_zyx, before.correction_type,
                       valid_zyx=before.valid_zyx)
    origins, diagnostics = _state_windows(before, roles, representatives, grid_affine_xyz, p,
                                          rng if window_rng is None else window_rng, iteration, unit)
    augmentation = draw_augmentation(rng) if augmentation is None else augmentation
    views = _build_views((before, after), ground_truth, origins, (representatives, representatives), grid_affine_xyz,
                         p, augmentation)
    diagnostics.update(examination_id=before.examination_id, patient_id=before.patient_id,
                       target_before=int(roles.target_error.sum()), target_after=int(views[2].roles.target_error.sum()),
                       origins_zyx=[list(v) for v in origins])
    return PairViews(views, VIEW_STATE_PAIR, diagnostics)


def exhausted_mask(before: State, ground_truth) -> np.ndarray:
    """The mask with the whole target error of the scribble corrected: M | T (addition), M - T (removal)."""
    roles = make_roles(before.mask_zyx, ground_truth, before.scribble_zyx, before.correction_type,
                       valid_zyx=before.valid_zyx)
    if correction_sign(before.correction_type) == 1:
        return before.mask_zyx | roles.target_error
    return before.mask_zyx & ~roles.target_error


def target_exhausted_views(before: State, ground_truth, *, grid_affine_xyz, window_size: int, rng, iteration=0,
                           unit=0, augmentation=None, window_rng=None) -> PairViews:
    """A state-pair unit whose second state has the scribble's whole target error corrected (no target error left;
    its previous mask is unknown)."""
    after = replace(before, mask_zyx=exhausted_mask(before, ground_truth), previous_mask_zyx=None)
    unit_views = state_pair_views(before, after, ground_truth, grid_affine_xyz=grid_affine_xyz,
                                  window_size=window_size, rng=rng, iteration=iteration, unit=unit,
                                  augmentation=augmentation, window_rng=window_rng)
    return PairViews(unit_views.views, VIEW_TARGET_EXHAUSTED, dict(unit_views.diagnostics, previous_mask_known=False))


# ------------------------------------------------------------------------------------------- scribble pairs
def error_component_order(component_ids, volumes, rng: np.random.Generator) -> np.ndarray:
    """The components in the order of an exponential race with rate sqrt(volume): sampling without replacement with
    probability growing with the square root of the volume (ties by id)."""
    ids = np.asarray(component_ids, dtype=np.int64)
    volumes = np.asarray(volumes, dtype=np.float64)
    if ids.shape != volumes.shape or np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError("component volumes must be positive finite values matching the ids")
    if not len(ids):
        return ids.copy()
    u = np.asarray(rng.random(len(ids)), dtype=np.float64)
    zeros = u == 0                      # Generator.random is [0, 1): redraw an exact zero, bias-free
    while zeros.any():
        u[zeros] = rng.random(int(zeros.sum()))
        zeros = u == 0
    keys = -np.log(u) / np.sqrt(volumes)
    return ids[np.lexsort((ids, keys))]


def _common_window_interval(world_a, world_b, grid_affine_xyz, window_size: int):
    """Integer window origins whose window shows both points (in the grid's index basis), or None."""
    a, b = (world_to_index_xyz(v, grid_affine_xyz) for v in (world_a, world_b))
    low = np.ceil(np.maximum(a, b) - (int(window_size) - 1) - 1e-8).astype(np.int64)
    high = np.floor(np.minimum(a, b) + 1e-8).astype(np.int64)
    return None if np.any(low > high) else (low, high)


def _nearest_feasible_pair(points_a, points_b, grid_affine_xyz, window_size: int):
    """The feasible window origins of the closest (Chebyshev) pair of target voxels of the two scribbles."""
    ua = world_to_index_xyz(points_a, grid_affine_xyz)
    ub = world_to_index_xyz(points_b, grid_affine_xyz)
    distances, nearest = cKDTree(ub).query(ua, p=np.inf, workers=1)
    candidates = np.flatnonzero(distances <= window_size - 1 + 1e-8)
    for index in candidates[np.argsort(distances[candidates], kind="stable")]:
        interval = _common_window_interval(points_a[index], points_b[nearest[index]], grid_affine_xyz, window_size)
        if interval is not None:
            return interval
    return None


def _scribble_pair_windows(state_a, roles_a, state_b, roles_b, grid_affine_xyz, window_size: int, rng):
    interval = _nearest_feasible_pair(_target_world(state_a, roles_a), _target_world(state_b, roles_b),
                                      grid_affine_xyz, window_size)
    if interval is None:
        return None
    low, high = interval

    def coverage(origin):
        return (_target_visibility(state_a, roles_a, origin, grid_affine_xyz, window_size),
                _target_visibility(state_b, roles_b, origin, grid_affine_xyz, window_size))

    first = tuple(int(v) for v in ((low + high) // 2)[::-1])
    visible_a, visible_b = coverage(first)
    if not visible_a.any() or not visible_b.any():
        raise GeometryError("the middle window origin failed the exact two-target check")
    first_union = np.concatenate((visible_a, visible_b))

    def score(va, vb):
        return min(float((visible_a | va).mean()), float((visible_b | vb).mean()))

    # Candidates: the first window itself (not distinct), then the random draws in order.
    best_key, best, best_new, attempts, distinct_candidates = (score(visible_a, visible_b), False, 1), first, 0, 0, 0
    if np.any(low != high):
        for attempt in range(SECOND_WINDOW_DRAWS):
            origin = tuple(int(v) for v in rng.integers(low, high + 1)[::-1])
            va, vb = coverage(origin)
            attempts = attempt + 1
            if not va.any() or not vb.any():
                continue
            distinct = origin != first
            distinct_candidates += int(distinct)
            key = (score(va, vb), distinct, -attempt)
            if key > best_key:
                best_key, best = key, origin
                best_new = int((np.concatenate((va, vb)) & ~first_union).sum())
    second = best
    reason = ("distinct_window" if second != first else "single_feasible_origin" if bool(np.all(low == high))
              else "no_distinct_alternative")
    va2, vb2 = coverage(second)
    return (first, second), {
        "second_window_draws": attempts, "single_feasible_origin": bool(np.all(low == high)),
        "duplicate_window": first == second, "new_target_voxels_in_second_window": best_new,
        "visible_target_voxels": [[int(visible_a.sum()), int(visible_b.sum())], [int(va2.sum()), int(vb2.sum())]],
        "visible_target_fraction": [[float(visible_a.mean()), float(visible_b.mean())],
                                    [float(va2.mean()), float(vb2.mean())]],
        "second_window_reason": reason,
        "second_window_min_coverage": min(float((visible_a | va2).mean()), float((visible_b | vb2).mean())),
        "distinct_candidates": distinct_candidates}


def scribble_pair_views(state: State, ground_truth, *, grid_affine_xyz, window_size: int, rng: np.random.Generator,
                        draw_scribble: Callable[[np.ndarray], np.ndarray | None], fallback_before: State,
                        fallback_after: State, component_rng: np.random.Generator, iteration: int = 0, unit: int = 1,
                        augmentation: Augmentation | None = None,
                        window_rng: np.random.Generator | None = None) -> PairViews:
    """The four views of a scribble pair on ``state`` (module docstring), or of the state pair
    ``(fallback_before, fallback_after)`` of the same examination when no other-error component gives one.

    ``draw_scribble(component)`` returns the simulated user's scribble on that error component, or None.
    """
    _check_same_examination(fallback_before, fallback_after)
    if state.examination_id != fallback_before.examination_id or state.patient_id != fallback_before.patient_id:
        raise ValueError("the fallback must be the same examination and patient")
    if state.shape_zyx != fallback_before.shape_zyx or not np.allclose(state.affine_xyz, fallback_before.affine_xyz,
                                                                     atol=1e-8, rtol=0):
        raise ValueError("the fallback must keep the native grid")
    p = int(window_size)
    roles_a = make_roles(state.mask_zyx, ground_truth, state.scribble_zyx, state.correction_type,
                         valid_zyx=state.valid_zyx)
    representatives_a = scribble_representatives(state.scribble_zyx, state.affine_xyz, grid_affine_xyz, p,
                                                 native_valid_zyx=state.valid_zyx)
    candidates = np.unique(roles_a.components[roles_a.other_errors])
    candidates = candidates[candidates != 0]
    volumes = np.bincount(roles_a.components.reshape(-1))
    candidates = error_component_order(candidates, volumes[candidates], component_rng)[:SCRIBBLE_PAIR_CANDIDATES]
    attempts, failures = 0, []
    for candidate in candidates:
        attempts += 1
        component = roles_a.components == candidate
        scribble = draw_scribble(component)
        if scribble is None:
            failures.append("no_scribble_drawn")
            continue
        scribble = np.asarray(scribble, dtype=bool)
        if scribble.shape != state.shape_zyx or not scribble.any():
            failures.append("scribble_shape_or_empty")
            continue
        state_b = replace(state, scribble_zyx=scribble)
        roles_b = make_roles(state_b.mask_zyx, ground_truth, scribble, state_b.correction_type,
                             valid_zyx=state_b.valid_zyx)
        if not roles_b.target_error.any() or np.any(roles_a.target_error & roles_b.target_error):
            failures.append("target_empty_or_shared")
            continue
        windows = _scribble_pair_windows(state, roles_a, state_b, roles_b, grid_affine_xyz, p,
                                         rng if window_rng is None else window_rng)
        if windows is None:
            failures.append("no_common_window")
            continue
        origins, diagnostics = windows
        representatives_b = scribble_representatives(scribble, state.affine_xyz, grid_affine_xyz, p,
                                                     native_valid_zyx=state.valid_zyx)
        augmentation = draw_augmentation(rng) if augmentation is None else augmentation
        views = _build_views((state, state_b), ground_truth, origins, (representatives_a, representatives_b),
                             grid_affine_xyz, p, augmentation)
        diagnostics.update(candidates_tried=attempts, failures=failures, fallback=False,
                           examination_id=state.examination_id, patient_id=state.patient_id,
                           origins_zyx=[list(v) for v in origins])
        return PairViews(views, VIEW_SCRIBBLE_PAIR, diagnostics)
    fallback = state_pair_views(fallback_before, fallback_after, ground_truth, grid_affine_xyz=grid_affine_xyz,
                                window_size=p, rng=rng, iteration=iteration, unit=unit, augmentation=augmentation,
                                window_rng=window_rng)
    return PairViews(fallback.views, VIEW_STATE_PAIR_FALLBACK,
                     dict(fallback.diagnostics, candidates_tried=attempts, failures=failures, fallback=True))


# ------------------------------------------------------------------------------------------- sources, compaction
def attach_sources(unit: PairViews, *, source_seed: int, view_kind: str,
                   view_targets: Callable[[Roles, Any, State], Any] | None = None) -> PairViews:
    """Attach to every view its scribble's sources (grouped on the augmented state, once per state), the result of
    ``view_targets(roles, sources, state)`` (once per state), the source-CE seed and the view kind."""
    from ..modeling.source_encoder import group_scribble  # noqa: PLC0415  (grouping lives with the source encoder)

    per_state: dict[int, tuple] = {}
    views = []
    for view in unit.views:
        key = id(view.state)
        if key not in per_state:
            sources = group_scribble(view.state.scribble_zyx, view.state.affine_xyz, view.representatives,
                                     native_valid_zyx=view.state.valid_zyx)
            targets = None if view_targets is None else view_targets(view.roles, sources, view.state)
            per_state[key] = (sources, targets)
        sources, targets = per_state[key]
        views.append(replace(view, sources=sources, targets=targets, source_seed=int(source_seed),
                             view_kind=view_kind))
    return PairViews(tuple(views), unit.kind, dict(unit.diagnostics, view_kind=view_kind))


#: State arrays that the two states of one unit often hold with equal values.
_SHAREABLE = ("pet_zyx", "ct_zyx", "valid_zyx", "earlier_addition_scribbles_zyx", "earlier_removal_scribbles_zyx",
              "scribble_zyx", "mask_zyx", "previous_mask_zyx")
#: Role labels (-1..2) fit one signed byte.
ROLE_LABEL_DTYPE = np.int8


def compact_views(unit: PairViews) -> PairViews:
    """The unit as a worker sends it: role labels as one int8 array per state, arrays equal between the unit's two
    states held once, both windows of a state sharing one state object. Every value a training step reads is
    unchanged."""
    states, labels = {}, {}
    first = None
    for view in unit.views:
        if id(view.state) in states:
            continue
        state = view.state
        if first is None:
            first = state
        else:
            shared = {}
            for name in _SHAREABLE:
                mine, theirs = getattr(state, name), getattr(first, name)
                if (mine is not None and theirs is not None and mine is not theirs and mine.shape == theirs.shape
                        and mine.dtype == theirs.dtype and np.array_equal(mine, theirs)):
                    shared[name] = theirs
            state = replace(state, **shared) if shared else state
        states[id(view.state)] = state
    for view in unit.views:
        if id(view.roles) not in labels:
            labels[id(view.roles)] = RoleLabels(np.asarray(view.roles.labels).astype(ROLE_LABEL_DTYPE),
                                                tuple(view.roles.touched_components))
    views = tuple(replace(view, state=states[id(view.state)], roles=labels[id(view.roles)]) for view in unit.views)
    return PairViews(views, unit.kind, unit.diagnostics)


# ------------------------------------------------------------------------------------------- units
def pair_states(pair: StatePair, inputs: ExaminationInputs) -> tuple[State, State, int]:
    """(before, after, real side) of a stored pair on its examination. Only the real state (the before state of a
    natural pair, the after state of a synthetic bridge) knows its previous mask, from the stored previous change."""
    direction = correction_sign(pair.correction_type)
    before_mask, after_mask = np.asarray(pair.mask_before, dtype=bool), np.asarray(pair.mask_after, dtype=bool)
    if before_mask.shape != after_mask.shape:
        raise ValueError(f"stored pair mask shapes differ: {before_mask.shape} vs {after_mask.shape}")
    wrong = (before_mask & ~after_mask) if direction == 1 else (after_mask & ~before_mask)
    if np.any(wrong):
        raise ValueError("the stored pair's change contradicts its correction type")
    real_side = pair.real_side
    states = []
    for index, mask in enumerate((pair.mask_before, pair.mask_after)):
        previous = previous_mask_from_change(mask, pair.previous_change) if index == real_side else None
        states.append(State(inputs.pet, inputs.ct, mask, pair.scribble, inputs.affine_xyz, direction,
                            pair.earlier_addition_scribbles, pair.earlier_removal_scribbles, previous, None,
                            inputs.examination_id, inputs.patient_id, float(inputs.pet_pad_value)))
    return states[0], states[1], real_side


@dataclass(frozen=True)
class UnitContext:
    """What a unit worker needs (picklable): the cache, the pools, the training roster, the grids and the drawing
    styles of the simulated user. Build it with :meth:`from_config`."""
    cache_directory: str
    pet_reference: PetReference | None
    pool_directories: Mapping[str, str]
    roster: Mapping[str, str]
    window_size: int
    grid_spacing_mm: tuple[float, float, float]
    whole_volume_shape: tuple[int, int, int]
    styles: tuple[str, ...]

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, cache_directory, pool_directories: Mapping[str, Any],
                    roster: Mapping[str, str], pet_reference: PetReference | None) -> "UnitContext":
        model, data = config["model"], config["data"]
        return cls(str(cache_directory), pet_reference, {name: str(path) for name, path in pool_directories.items()},
                   dict(roster), int(model["window_size"]), tuple(float(v) for v in model["grid_spacing_mm"]),
                   tuple(int(v) for v in model["whole_volume_shape"]), tuple(data["drawing_styles"]))


def _validate_row(row: Mapping[str, Any], roster: Mapping[str, str]) -> dict:
    if row.get("split") != "train":
        raise ValueError(f"a training pool row has split={row.get('split')!r}, allowed 'train'")
    examination, patient = str(row["examination_id"]), str(row["patient_id"])
    for value in (examination, patient):
        if not value or value in (".", "..") or "/" in value or "\\" in value:
            raise ValueError(f"unsafe identifier {value!r}")
    if roster.get(examination) != patient:
        raise ValueError(f"examination/patient {examination!r}/{patient!r} is not in the training roster")
    for key in ("path", "pair_id", "state_id", "pair_type"):
        if key not in row:
            raise ValueError(f"a pool row misses {key!r}")
    return dict(row)


_SIMULATOR = None


def _simulator_scribble(error_zyx, ground_truth, correction_type: int, style: str):
    """The simulated user's scribble on one error component (None when the official simulator draws nothing)."""
    global _SIMULATOR
    from ..interaction.scribble_simulator import NoScribbleDrawn, ScribbleSimulator  # noqa: PLC0415

    if _SIMULATOR is None:
        _SIMULATOR = ScribbleSimulator()
    try:
        drawn = _SIMULATOR.draw_on_error(error_zyx, ground_truth, correction_type, style)
    except NoScribbleDrawn:
        return None
    return drawn.scribble_zyx if drawn.valid else None


def _scribble_drawer(draw: Callable, ground_truth, correction_type: int, styles: Sequence[str],
                     rng: np.random.Generator) -> Callable[[np.ndarray], np.ndarray | None]:
    def draw_component(component):
        style = styles[int(rng.integers(len(styles)))]
        scribble = draw(np.ascontiguousarray(component), np.ascontiguousarray(ground_truth), correction_type, style)
        return scribble if scribble is not None and np.asarray(scribble).any() else None
    return draw_component


def prepare_unit(context: UnitContext, request: "UnitRequest", iteration: int, *,
                 draw_scribble: Callable | None = None,
                 view_targets: Callable[[Roles, Any, State], Any] | None = None) -> PairViews:
    """The four compacted views of one unit (top-level, picklable: workers may run it).

    ``draw_scribble(error_zyx, ground_truth_zyx, correction_type, style) -> scribble | None`` defaults to the
    official simulator (:class:`sirb.interaction.scribble_simulator.ScribbleSimulator`); ``view_targets`` is passed
    to :func:`attach_sources`.
    """
    if request.kind not in UNIT_KINDS or request.mode not in (NORMAL, TARGET_EXHAUSTED) or (
            request.kind == SCRIBBLE_PAIR and request.mode != NORMAL):
        raise ValueError(f"unit kind/mode actual={request.kind!r}/{request.mode!r}")
    if int(iteration) < 1:
        raise ValueError("training iterations count from 1")
    row = _validate_row(request.row, context.roster)
    if request.source not in context.pool_directories:
        raise ValueError(f"pool {request.source!r} has no registered directory")
    path = resolve_pair_path(row, context.pool_directories[request.source])
    examination = row["examination_id"]
    cache = ExaminationCache(context.cache_directory, splits=("train",), pet_reference=context.pet_reference)
    meta = cache.meta(examination)                       # identity and split before any array is opened
    if meta["examination_id"] != examination or meta["patient_id"] != row["patient_id"]:
        raise ValueError(f"cached examination {meta['examination_id']}/{meta['patient_id']} differs from the row")
    pair = load_training_state_pair(path)
    for name, expected in (("examination_id", examination), ("patient_id", row["patient_id"]),
                           ("split", row["split"]), ("pair_id", row["pair_id"])):
        if getattr(pair, name) != expected:
            raise ValueError(f"stored pair {name} actual={getattr(pair, name)!r}, expected={expected!r}")
    inputs = cache.inputs(examination)
    truth = cache.ground_truth(examination)
    before, after, real_side = pair_states(pair, inputs)
    if truth.shape != before.shape_zyx:
        raise ValueError(f"ground truth shape actual={truth.shape}, expected={before.shape_zyx}")
    grids = examination_grids(inputs.shape_zyx, before.affine_xyz, context.window_size,
                              grid_spacing_mm=context.grid_spacing_mm,
                              whole_volume_shape_zyx=context.whole_volume_shape)
    seeds = request.seeds
    window_rng = np.random.default_rng(int(seeds["window"]))
    augmentation = draw_augmentation(np.random.default_rng(int(seeds["augmentation"])))
    common = dict(grid_affine_xyz=grids.grid_affine_xyz, window_size=context.window_size, rng=window_rng,
                  iteration=int(iteration), augmentation=augmentation, window_rng=window_rng)
    if request.kind == STATE_PAIR and request.mode == TARGET_EXHAUSTED:
        unit, view_kind = target_exhausted_views(before, truth, unit=0, **common), VIEW_TARGET_EXHAUSTED
    elif request.kind == STATE_PAIR:
        unit, view_kind = state_pair_views(before, after, truth, unit=0, **common), VIEW_STATE_PAIR
    else:
        draw = _scribble_drawer(_simulator_scribble if draw_scribble is None else draw_scribble, truth,
                                correction_sign(pair.correction_type), context.styles,
                                np.random.default_rng(int(seeds["scribble"])))
        unit = scribble_pair_views((before, after)[real_side], truth, draw_scribble=draw, fallback_before=before,
                                   fallback_after=after, unit=1,
                                   component_rng=np.random.default_rng(int(seeds["error_order"])), **common)
        view_kind = VIEW_SCRIBBLE_PAIR if unit.kind == VIEW_SCRIBBLE_PAIR else VIEW_STATE_PAIR_FALLBACK
    unit = attach_sources(unit, source_seed=int(seeds["source_samples"]), view_kind=view_kind,
                          view_targets=view_targets)
    compact = compact_views(unit)
    diagnostics = dict(unit.diagnostics, source=request.source, pair_id=pair.pair_id, real_side=real_side,
                       mode=request.mode, pet_normalization=copy.deepcopy(meta["pet_normalization"]))
    return PairViews(compact.views, unit.kind, diagnostics)


# ------------------------------------------------------------------------------------------- sampling
class RandomStreams:
    """Named, restorable random streams: stream ``name`` is ``default_rng(SeedSequence([seed, *w]))`` with ``w`` the
    first 16 bytes of SHA-256(name) as four little-endian uint32 words."""

    def __init__(self, seed: int = 3407):
        self.seed = seed
        self.streams: dict[str, np.random.Generator] = {}

    def get(self, name: str) -> np.random.Generator:
        if name not in self.streams:
            words = np.frombuffer(hashlib.sha256(name.encode()).digest()[:16], dtype="<u4").tolist()
            self.streams[name] = np.random.default_rng(np.random.SeedSequence([self.seed, *words]))
        return self.streams[name]

    def state_dict(self) -> dict:
        return {"seed": self.seed,
                "streams": {k: copy.deepcopy(v.bit_generator.state) for k, v in self.streams.items()}}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.seed, self.streams = state["seed"], {}
        for name, value in state["streams"].items():
            self.get(name).bit_generator.state = copy.deepcopy(value)


def mixing_slots(config: Mapping[str, Any], iteration: int) -> list[str]:
    """The pools of the eight slots of a block (``data.mixing``) for the refreshes before ``iteration``.

    The ground-truth-corrected pool takes its slots; each patient half's latest self-generated pool takes
    ``self_generated_slots_per_half``; the earlier-version pools share the rest equally, in their listed order.
    Without earlier-version pools (an empty list) the ground-truth-corrected pool takes the rest as well.
    """
    data = config["data"]
    mixing, generated = data["mixing"], data["self_generated"]
    latest: dict[int, str] = {}
    for point in generated["refresh"]:
        if int(point["iteration"]) < int(iteration):
            latest[int(point["patient_half"])] = generated["name"].format(half=int(point["patient_half"]),
                                                                          iteration=int(point["iteration"]))
    block = int(mixing["block_iterations"])
    fixed = int(mixing["ground_truth_corrected_slots"])
    per_half = int(mixing["self_generated_slots_per_half"])
    earlier = list(mixing["earlier_version_pools"])
    rest = block - fixed - per_half * len(latest)
    if rest < 0 or (earlier and rest % len(earlier)):
        raise ValueError(f"data.mixing cannot fill {block} slots: fixed={fixed} self-generated="
                         f"{per_half * len(latest)} earlier-version pools={earlier}")
    slots = [mixing["ground_truth_corrected_pool"]] * (fixed if earlier else fixed + rest)
    for pool in earlier:
        slots += [pool] * (rest // len(earlier))
    for half in sorted(latest):
        slots += [latest[half]] * per_half
    return slots


class TrainingCourse:
    """The pool and the mode of every request: per block of ``data.mixing.block_iterations`` iterations a
    permutation of :func:`mixing_slots` (stream ``source-<kind>``), and per block of
    ``data.views.target_exhausted_block`` iterations the positions of the target-exhausted state-pair requests
    (stream of the null quota). Only the current block is kept; :meth:`state_dict` restores a course exactly."""

    def __init__(self, config: Mapping[str, Any], streams: RandomStreams):
        self.config, self.streams = config, streams
        views = config["data"]["views"]
        self.block = int(config["data"]["mixing"]["block_iterations"])
        self.null_block = int(views["target_exhausted_block"])
        self.null_per_block = {STATE_PAIR: int(views["target_exhausted_per_block"]), SCRIBBLE_PAIR: 0}
        self.blocks: dict[tuple, list[str]] = {}
        self.null_blocks: dict[tuple, list[int]] = {}

    def source(self, iteration: int, kind: str) -> str:
        if kind not in UNIT_KINDS:
            raise ValueError(f"unknown unit kind {kind!r}")
        block = (int(iteration) - 1) // self.block
        key = (block, kind)
        if key not in self.blocks:
            slots = mixing_slots(self.config, int(iteration))
            permutation = self.streams.get(stream_name("source", kind)).permutation(self.block)
            self.blocks = {k: v for k, v in self.blocks.items() if k[0] == block}
            self.blocks[key] = [slots[i] for i in permutation]
        return self.blocks[key][(int(iteration) - 1) % self.block]

    def mode(self, iteration: int, kind: str) -> str:
        if kind not in UNIT_KINDS:
            raise ValueError(f"unknown unit kind {kind!r}")
        if int(iteration) < 1:
            raise ValueError("training iterations count from 1")
        block = (int(iteration) - 1) // self.null_block
        key = (block, kind)
        if key not in self.null_blocks:
            stream = self.streams.get(stream_name("target_exhausted", kind))
            positions = sorted(int(v) for v in stream.choice(self.null_block, size=self.null_per_block[kind],
                                                             replace=False))
            self.null_blocks = {k: v for k, v in self.null_blocks.items() if k[0] == block}
            self.null_blocks[key] = positions
        hit = (int(iteration) - 1) % self.null_block in self.null_blocks[key]
        return TARGET_EXHAUSTED if hit and kind == STATE_PAIR else NORMAL

    def state_dict(self) -> dict:
        return {"blocks": [(list(k), v) for k, v in self.blocks.items()],
                "null_blocks": [(list(k), list(v)) for k, v in self.null_blocks.items()]}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.blocks = {tuple(k): v for k, v in state["blocks"]}
        self.null_blocks = {tuple(k): list(v) for k, v in state["null_blocks"]}


@dataclass(frozen=True)
class GapTable:
    """States drawn in proportion to frequency x recoverable gap: the natural draw probability (patient, then
    examination, then state uniform) times the occurrences the state stands for, times its gap. States without a
    positive gap are never drawn here."""
    keys: tuple
    cumulative: np.ndarray
    counts: Mapping[str, int]

    def draw(self, rng: np.random.Generator) -> tuple[str, str, str]:
        index = int(np.searchsorted(self.cumulative, rng.random() * self.cumulative[-1], side="right"))
        return self.keys[min(index, len(self.keys) - 1)]


def gap_weighted_table(tree: Mapping[str, Mapping[str, Mapping[str, Mapping[str, dict]]]]) -> GapTable:
    """The :class:`GapTable` of a pool tree (patient -> examination -> state -> {pair type: row})."""
    keys, weights = [], []
    counts = {"states": 0, "states_with_positive_gap": 0, "states_without_gap": 0, "states_with_zero_gap": 0}
    for patient, examinations in tree.items():
        for examination, states in examinations.items():
            for state, by_type in states.items():
                counts["states"] += 1
                gaps = {row.get("recoverable_gap") for row in by_type.values()}
                occurrences = {int(row.get("state_occurrences", 1)) for row in by_type.values()}
                if len(gaps) != 1 or len(occurrences) != 1:
                    raise ValueError(f"state {state}: its rows disagree on the gap {sorted(map(str, gaps))} or the "
                                     f"occurrences {sorted(occurrences)}")
                (gap,), (occurrence,) = gaps, occurrences
                if gap is None or not math.isfinite(float(gap)):
                    counts["states_without_gap"] += 1
                    continue
                if not float(gap) > 0.0:
                    counts["states_with_zero_gap"] += 1
                    continue
                natural = 1.0 / (len(tree) * len(examinations) * len(states))
                keys.append((patient, examination, state))
                weights.append(natural * occurrence * float(gap))
                counts["states_with_positive_gap"] += 1
    if not keys:
        raise ValueError(f"gap-weighted pool without any positive gap (pool counts {counts})")
    return GapTable(tuple(keys), np.cumsum(np.asarray(weights, dtype=np.float64)), counts)


class PoolSampler:
    """Rows of one pool: patient, then examination, then state uniformly (or by the gap table), then a pair type by
    the mix among the types the state has."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], *, pair_mix: Mapping[str, float], gap_weighted=False):
        if not rows:
            raise ValueError("a training pool is empty; it cannot be replaced by another pool")
        self.pair_mix, self.gap_weighted = dict(pair_mix), bool(gap_weighted)
        self.patients: dict[str, dict] = {}
        for row in rows:
            if row.get("split") != "train":
                raise ValueError(f"a training pool holds split={row.get('split')!r}")
            by_type = self.patients.setdefault(row["patient_id"], {}).setdefault(row["examination_id"], {}) \
                .setdefault(row["state_id"], {})
            if row["pair_type"] in by_type:
                raise ValueError(f"two {row['pair_type']} pairs of state {row['state_id']}")
            by_type[row["pair_type"]] = dict(row)
        self.gap_table = gap_weighted_table(self.patients) if self.gap_weighted else None

    def draw(self, rng: np.random.Generator) -> dict:
        if self.gap_weighted:
            patient, examination, state = self.gap_table.draw(rng)
            by_type = self.patients[patient][examination][state]
        else:
            patients = sorted(self.patients)
            patient = self.patients[patients[int(rng.integers(len(patients)))]]
            examinations = sorted(patient)
            states = patient[examinations[int(rng.integers(len(examinations)))]]
            ids = sorted(states)
            by_type = states[ids[int(rng.integers(len(ids)))]]
        pair_type, _ = choose_pair_type(set(by_type), self.pair_mix, rng)
        return by_type[pair_type]


@dataclass(frozen=True)
class UnitRequest:
    """One reserved unit: its kind, mode, pool, row and per-unit seeds (``window``, ``scribble``,
    ``augmentation``, ``source_samples``, ``error_order``)."""
    kind: str
    mode: str
    source: str
    row: Mapping[str, Any]
    seeds: Mapping[str, int] = field(default_factory=dict)


def reserve_unit_seeds(streams: RandomStreams, kind: str, source: str) -> dict[str, int]:
    """The per-unit seeds :data:`SEED_NAMES`, one draw of each stream ``<seed>-<kind>-<pool>``."""
    return {name: int(streams.get(stream_name(name, kind, source)).integers(0, _SEED_HIGH)) for name in SEED_NAMES}


class TrainingPairSampler:
    """Reserves the two units of every training iteration: pools and modes from the course, rows from the pool
    samplers (stream ``row-<kind>-<pool>``) and the per-unit seeds. ``state_dict`` of the streams and the course
    after a reservation restores the sampler exactly."""

    def __init__(self, config: Mapping[str, Any], pools: Mapping[str, Sequence[Mapping[str, Any]]],
                 streams: RandomStreams):
        self.config, self.streams = config, streams
        self.course = TrainingCourse(config, streams)
        sources = config["data"]["state_sources"]
        mix = config["data"]["state_pairs"]["type_mix"]
        self.samplers = {name: PoolSampler(rows, pair_mix=mix,
                                           gap_weighted=sources.get(name, {}).get("sampling") == "gap_weighted")
                         for name, rows in pools.items()}

    def reserve(self, iteration: int) -> dict[str, UnitRequest]:
        sources = {kind: self.course.source(iteration, kind) for kind in UNIT_KINDS}
        modes = {kind: self.course.mode(iteration, kind) for kind in UNIT_KINDS}
        rows = {}
        for kind, source in sources.items():
            if source not in self.samplers:
                raise ValueError(f"pool {source!r} is needed at iteration {iteration} but was not given")
            rows[kind] = self.samplers[source].draw(self.streams.get(stream_name("row", kind, source)))
        seeds = {kind: reserve_unit_seeds(self.streams, kind, source) for kind, source in sources.items()}
        return {kind: UnitRequest(kind, modes[kind], sources[kind], rows[kind], seeds[kind]) for kind in UNIT_KINDS}


def read_training_pools(config: Mapping[str, Any], *, roster: Mapping[str, str],
                        self_generated: Mapping[str, tuple[Any, Any]] | None = None
                        ) -> tuple[dict[str, list[dict]], dict[str, Path]]:
    """(rows, directory) of every pool of ``data.state_sources`` (directories relative to ``paths.preprocessed``)
    and of the given self-generated pools ({name: (directory, manifest)}); every row is checked against the
    training roster."""
    preprocessed = Path(config["paths"]["preprocessed"])
    rows, directories = {}, {}
    entries = {name: (preprocessed / source["directory"], preprocessed / source["directory"] / source["manifest"])
               for name, source in config["data"]["state_sources"].items()}
    entries.update({name: (Path(directory), Path(manifest))
                    for name, (directory, manifest) in (self_generated or {}).items()})
    for name, (directory, manifest) in entries.items():
        rows[name] = [_validate_row(row, roster) for row in read_pool_rows(manifest)]
        directories[name] = directory
    return rows, directories
