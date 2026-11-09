"""Training states: the three state sources, state pairs and the stored state-pair format.

A state is the current mask of one correction round with the scribble drawn on it, the scribble's correction type
and the earlier scribbles (:class:`ScribbleState`). SIRB is trained on state pairs (:class:`StatePair`): two states
that share the examination, the scribble, its correction type and the earlier scribbles and differ by one correct
correction of the scribble's type. The roles are recomputed on both states, never derived by hand, so a correction
that cuts the target error turns its far part from target error into other errors.

The three state sources:

* ground-truth-corrected states (:func:`generate_ground_truth_corrected_states`): trajectories that start from the
  initial segmentation; in each round the simulated user draws a scribble on the current mask and the target error is
  corrected with the ground truth (all of it, or with probability ``partial_correction_probability`` a geodesic ball
  around the scribble holding a sampled fraction of it), which gives the next state. The first trajectories use the
  simulated user as it is, one per drawing style; later trajectories draw the other correction type whenever that
  type still has errors (the simulated user is deterministic, so they would repeat the first ones);
* earlier-version states: five-round interactions of three earlier SIRB versions, read from a pool in the format
  below (:func:`read_pool_rows`, :func:`load_state_pair`); the networks that produced them are not part of this
  repository;
* self-generated states (:func:`states_from_rollout`): the five-round interaction of a frozen copy of the current
  model, refreshed during training. They also store the change the update rule wrote in the round before.

Every state yields at most one state pair of each type (:func:`build_state_pairs`): ``peripheral`` corrects the part
of the target error geodesically farthest from the scribble (5-30 % of it); ``other_error`` corrects whole components
of the other errors; ``bridge`` cuts the target error at a thin neck, so its far part becomes other errors (with
``synthetic_bridges`` a short error path to another error component may be injected instead, training set only). A
type that does not exist for a state is logged with its reason, never made up.

Stored format of a state pool (the same for all three sources)::

    pairs/<examination>/<pair id>.npz     one state pair: packed bit volumes and a JSON record
    pairs.jsonl                           one row per pair (pool.json with "rows" for a self-generated pool)

A record stores the attributes of :class:`StatePair` under their own names: each volume as ``<name>_bits`` (one bit
per voxel), the previous change as ``previous_change_added_bits`` and ``previous_change_removed_bits``, every other
attribute in the JSON record ``record_json``. Correction types are stored as ``"addition"`` and ``"removal"``. Arrays
are ZYX bool volumes on the native PET grid.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import json
import math
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from ..utils.io import TargetExistsError, write_json_atomic, write_npz_atomic
from .psma import DRAWING_STYLES
from .roles import Roles, correction_sign, make_roles

__all__ = ["ADDITION", "CORRECTION_ROUNDS", "INTERACTION_FLAG_KEYS", "PAIR_TYPES", "REMOVAL",
           "SOURCE_INITIAL_SEGMENTATION", "SOURCE_SELF_GENERATED", "PairAlgebraViolation", "PairInfeasible",
           "ScribbleState", "StateError", "StatePair", "append_rows", "build_state_pairs", "choose_pair_type",
           "correction_name", "generate_ground_truth_corrected_states", "load_state_pair", "load_training_state_pair",
           "publish_pool", "read_pool_rows", "resolve_pair_path", "save_state_pair", "stable_hash", "state_pair_row",
           "states_from_rollout", "write_state_pairs"]

#: Stored correction types.
ADDITION, REMOVAL = "addition", "removal"
#: Correction rounds of one interaction.
CORRECTION_ROUNDS = 5
#: Pair types, in the order they are built and sampled.
PAIR_PERIPHERAL, PAIR_OTHER_ERROR, PAIR_BRIDGE = "peripheral", "other_error", "bridge"
PAIR_TYPES = (PAIR_PERIPHERAL, PAIR_OTHER_ERROR, PAIR_BRIDGE)
NATURAL, SYNTHETIC = "natural", "synthetic"
#: Attempts per pair type and state; a failure that resampling cannot change stops earlier.
MAX_PAIR_ATTEMPTS = 10
#: How the mask of a state was made: the initial segmentation, a full or partial correction of the target error with
#: the ground truth, or the model's own interaction (stored values).
SOURCE_INITIAL_SEGMENTATION = "initial_segmentation"
SOURCE_FULL_CORRECTION = "full_target_correction"
SOURCE_PARTIAL_CORRECTION = "partial_target_correction"
SOURCE_SELF_GENERATED = "self_generated"
#: Whether the scribble touches one error component of its type, none, or several (stored values).
TOUCHES_ONE, TOUCHES_NONE, TOUCHES_SEVERAL = "one_error", "no_error", "several_errors"
STATUS_OK = "ok"
#: How the scribble of a state was drawn (not stored).
SELECTION_SIMULATED_USER = "simulated_user"
SELECTION_OTHER_TYPE = "simulated_user_other_correction_type"
#: Lesions and errors are 18-connected; an injected bridge path is 6-connected (a 3D line is only 26-connected).
CONNECTIVITY_RANK = {6: 1, 18: 2, 26: 3}
MAIN_CONNECTIVITY, BRIDGE_PATH_CONNECTIVITY = 18, 6
#: A cut of area a between solids of volume V scales like V ** (2/3) (the neck test is size free).
NECK_AREA_EXPONENT = 2.0 / 3.0
#: Bytes of SHA-256 that seed the per-examination and per-state generators (never Python's salted hash()).
STABLE_HASH_BYTES = 8

# Reasons a pair type does not exist for a state.
REASON_NO_TARGET = "scribble_touches_no_error"
REASON_NO_PERIPHERAL_VOXELS = "target_has_no_voxel_beyond_the_scribble"
REASON_NO_OTHER_ERROR = "no_other_error_of_the_type"
REASON_TARGET_TOO_SHALLOW = "target_depth_below_two"
REASON_NO_THIN_NECK = "target_has_no_thin_neck"
REASON_NO_BRIDGE_PARTNER = "no_other_error_within_gap"
REASON_EMPTY_DOWN_SET = "bridge_cut_left_no_down_voxel"

# ------------------------------------------------------------------------------------------- stored format
#: Schema and version of a stored state pair (also in every manifest row).
PAIR_SCHEMA, PAIR_SCHEMA_VERSION = "sirb-state-pair", 1
PAIRS_FOLDER, PAIRS_MANIFEST, POOL_MANIFEST = "pairs", "pairs.jsonl", "pool.json"
_SHAPE_DTYPE = "<i8"
#: The volumes of a stored pair, in their order in the archive (stored as ``<name>_bits``).
_PAIR_VOLUMES = ("mask_before", "mask_after", "correction", "scribble", "earlier_addition_scribbles",
                 "earlier_removal_scribbles")
#: The two bit volumes of the previous change: voxels the update rule added and removed in the round before.
_CHANGE_PARTS = ("previous_change_added_bits", "previous_change_removed_bits")
#: The attributes of the JSON record of a stored pair.
_PAIR_FIELDS = ("pair_id", "state_id", "examination_id", "patient_id", "split", "state_source", "pair_type",
                "synthetic", "correction_type", "transition_counts", "target_status_before", "target_status_after",
                "attempts", "fallback_from", "status", "interaction_flags")
#: Keys of the interaction flags: round (1-based), correction type of the round before (+1, -1, 0 for none),
#: whether the scribble repeats that round's scribble, whether that round changed nothing.
INTERACTION_FLAG_KEYS = ("round_index", "previous_correction_type", "scribble_repeats_previous",
                         "previous_correction_empty")
_FLAG_ROUND, _FLAG_PREVIOUS_TYPE, _FLAG_REPEATS, _FLAG_EMPTY = INTERACTION_FLAG_KEYS
#: Transition counts on the voxels editable in both states (``down``: target error before, not after; ``up`` the
#: reverse, always 0 for a correct correction).
_COUNT_KEYS = ("common", "down", "up", "stable", "target_before", "target_after", "correction_voxels")


class StateError(RuntimeError):
    """A state, a state pair or a stored record is malformed or was asked for outside its split."""


class PairInfeasible(StateError):
    """This pair type does not exist for this state (logged, never made up)."""

    def __init__(self, reason: str, detail: str = "", *, retryable: bool = False) -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail, self.retryable = reason, detail, retryable


class PairAlgebraViolation(StateError):
    """A built pair is not one correct correction of the scribble's type."""


# ------------------------------------------------------------------------------------------- small helpers
def _binary(array, name: str) -> np.ndarray:
    a = np.asarray(array)
    if a.ndim != 3:
        raise StateError(f"{name} must be a 3D array, got ndim={a.ndim}")
    if a.dtype != np.bool_ and not np.all((a == 0) | (a == 1)):
        raise StateError(f"{name} must be binary")
    return a.astype(bool, copy=False)


def _same_shape(*arrays) -> None:
    shapes = {np.shape(a) for a in arrays}
    if len(shapes) != 1:
        raise StateError(f"arrays must share one shape, got {sorted(shapes)}")


def _frozen(volume: np.ndarray) -> np.ndarray:
    """Mark a volume as read-only before it enters a record: records share volumes."""
    volume.setflags(write=False)
    return volume


def _structure(connectivity: int) -> np.ndarray:
    return ndimage.generate_binary_structure(3, CONNECTIVITY_RANK[connectivity])


def _check_rng(rng) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise StateError(f"rng must be a numpy.random.Generator, got {type(rng).__name__}")
    return rng


def stable_hash(text: str) -> int:
    """Process-independent integer of a string (the first 8 bytes of its SHA-256, big-endian)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:STABLE_HASH_BYTES], "big")


def correction_name(correction_type) -> str:
    """``"addition"`` or ``"removal"`` (accepts +1/-1, "+"/"-", "addition"/"removal")."""
    return ADDITION if correction_sign(correction_type) == 1 else REMOVAL


def _apply(mask, region, correction_type) -> np.ndarray:
    """The plain set update of the update rule: ``mask | region`` (addition), ``mask & ~region`` (removal)."""
    from ..interaction.update_rule import apply  # noqa: PLC0415  (the update rule is not needed to read pools)

    return np.asarray(apply(mask, region, correction_sign(correction_type)), dtype=bool)


def _earlier_scribbles(earlier, shape) -> tuple[np.ndarray, np.ndarray]:
    from ..interaction.update_rule import accumulate_scribbles  # noqa: PLC0415

    addition, removal = accumulate_scribbles(earlier, shape=tuple(int(v) for v in shape))
    return np.asarray(addition, dtype=bool), np.asarray(removal, dtype=bool)


def _signed_error(mask, truth, correction_type) -> np.ndarray:
    """Errors of a correction type: missed ground truth (addition) or mask outside it (removal)."""
    return (truth & ~mask) if correction_sign(correction_type) == 1 else (mask & ~truth)


def _status(roles: Roles) -> str:
    count = len(roles.touched_components)
    return TOUCHES_NONE if not count else (TOUCHES_SEVERAL if count > 1 else TOUCHES_ONE)


def bounding_box(region, margin=0) -> tuple[slice, ...]:
    """Smallest box holding every True voxel of a non-empty region, grown by ``margin`` and clipped to the array."""
    occupied = np.asarray(region, dtype=bool)
    grow = np.broadcast_to(np.asarray(margin), (occupied.ndim,))
    box = []
    for axis in range(occupied.ndim):
        present = np.flatnonzero(occupied.any(axis=tuple(a for a in range(occupied.ndim) if a != axis)))
        box.append(slice(max(int(present[0]) - int(grow[axis]), 0),
                         min(int(present[-1]) + int(grow[axis]) + 1, occupied.shape[axis])))
    return tuple(box)


def _embed(shape, box, part) -> np.ndarray:
    volume = np.zeros(tuple(shape), dtype=bool)
    volume[box] = part
    return volume


def _volume_from_indices(indices, shape) -> np.ndarray:
    volume = np.zeros(tuple(int(v) for v in shape), dtype=bool)
    rows = np.asarray(indices)
    if len(rows):
        volume[tuple(rows.T)] = True
    return volume


def _values_equal(left, right) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and left.shape == right.shape
                and bool(np.array_equal(left, right)))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return dict(left) == dict(right)
    return bool(left == right)


class _ArrayRecord:
    """Field-wise equality for frozen records that hold numpy volumes."""

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(_values_equal(getattr(self, f.name), getattr(other, f.name)) for f in fields(self))

    __hash__ = None  # type: ignore[assignment]


# ------------------------------------------------------------------------------------------- records
@dataclass(frozen=True, eq=False)
class ScribbleState(_ArrayRecord):
    """One state with its scribble. Holds no ground truth and no role map.

    ``previous_change`` (int8: +1 added, -1 removed by the update rule in the round before), ``interaction_flags``
    and ``previous_scribble`` exist for self-generated states only (None elsewhere).
    """
    state_id: str
    examination_id: str
    patient_id: str
    split: str
    trajectory_index: int
    round_index: int
    correction_type: str
    style: str
    selection_policy: str
    state_source: str
    mask: np.ndarray
    scribble: np.ndarray
    earlier_addition_scribbles: np.ndarray
    earlier_removal_scribbles: np.ndarray
    target_status: str
    previous_change: np.ndarray | None = None
    interaction_flags: Mapping[str, Any] | None = None
    previous_scribble: np.ndarray | None = None

    @property
    def has_history(self) -> bool:
        return self.previous_change is not None or self.interaction_flags is not None


@dataclass(frozen=True, eq=False)
class StatePair(_ArrayRecord):
    """Two states that differ by one correct correction ``correction`` of the scribble's correction type.

    For a synthetic bridge ``mask_before`` is the injected state and ``mask_after`` the real one; the history (if
    any) belongs to the real state only (:attr:`real_side`).
    """
    pair_id: str
    state_id: str
    examination_id: str
    patient_id: str
    split: str
    state_source: str
    pair_type: str
    synthetic: bool
    mask_before: np.ndarray | None
    mask_after: np.ndarray | None
    correction: np.ndarray | None
    correction_type: str
    scribble: np.ndarray | None
    earlier_addition_scribbles: np.ndarray | None
    earlier_removal_scribbles: np.ndarray | None
    transition_counts: Mapping[str, int]
    target_status_before: str
    target_status_after: str
    attempts: int
    fallback_from: str | None
    status: str
    previous_change: np.ndarray | None = None
    interaction_flags: Mapping[str, Any] | None = None
    previous_scribble: np.ndarray | None = None

    @property
    def has_history(self) -> bool:
        """Whether the pair comes from a model's own interaction and carries the round before."""
        return self.previous_change is not None or self.interaction_flags is not None

    @property
    def real_side(self) -> int:
        """0 when ``mask_before`` is the state's own (real) mask, 1 for a synthetic bridge."""
        return 1 if self.synthetic else 0


# ------------------------------------------------------------------------------------------- ground-truth-corrected
def geodesic_levels(region, seeds, connectivity: int = MAIN_CONNECTIVITY):
    """BFS distance from ``seeds`` inside ``region`` on the region's bounding box: ``(box, levels)`` with 0 on the
    seeds, k on voxels first reached at step k and -1 elsewhere. Neighbours differ by at most one level."""
    inside, start = _binary(region, "region"), _binary(seeds, "seeds")
    _same_shape(inside, start)
    if not (inside & start).any():
        raise StateError("geodesic levels need at least one seed inside the region")
    box = bounding_box(inside)
    area = inside[box]
    structure = _structure(connectivity)
    levels = np.full(area.shape, -1, dtype=np.int32)
    frontier = start[box] & area
    visited = frontier.copy()
    levels[frontier] = 0
    level = 0
    while True:
        frontier = ndimage.binary_dilation(frontier, structure=structure) & area & ~visited
        if not frontier.any():
            return box, levels
        level += 1
        levels[frontier] = level
        visited |= frontier


def _partial_target(target, seeds, fraction: float) -> np.ndarray:
    """Geodesic ball around the scribble's voxels in the target holding at least ``fraction`` of it (whole levels)."""
    box, levels = geodesic_levels(target, seeds)
    counts = np.bincount(levels[levels >= 0])
    need = fraction * int(counts.sum())
    cut = min(int(np.searchsorted(np.cumsum(counts), need, side="left")), len(counts) - 1)
    return _embed(target.shape, box, (levels >= 0) & (levels <= cut))


def _same_content(left: ScribbleState, right: ScribbleState) -> bool:
    return (left.correction_type == right.correction_type and left.style == right.style
            and all(np.array_equal(getattr(left, name), getattr(right, name))
                    for name in ("mask", "scribble", "earlier_addition_scribbles", "earlier_removal_scribbles")))


def _config_int(config: Mapping[str, Any], key: str) -> int:
    value = config[key]
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise StateError(f"{key} must be a positive integer, got {value!r}")
    return int(value)


def _config_probability(config: Mapping[str, Any], key: str) -> float:
    value = float(config[key])
    if not 0.0 <= value <= 1.0:
        raise StateError(f"{key} must lie in [0, 1], got {value}")
    return value


def _config_fraction_range(config: Mapping[str, Any], key: str) -> tuple[float, float]:
    low, high = (float(v) for v in config[key])
    if not 0.0 < low <= high <= 1.0:
        raise StateError(f"{key} must satisfy 0 < low <= high <= 1, got {(low, high)}")
    return low, high


def _draw_training_scribble(mask, truth, *, style: str, simulator, other_type: bool):
    """The simulated user's scribble; with ``other_type`` the other correction type's scribble when that type still
    has errors. Returns (scribble, selection policy)."""
    main = simulator.draw(mask, truth, style)
    if not other_type or not main.valid:
        return main, SELECTION_SIMULATED_USER
    other = simulator.draw_correction_type(mask, truth, -int(main.correction_type), style)
    return (other, SELECTION_OTHER_TYPE) if other.valid else (main, SELECTION_SIMULATED_USER)


def generate_ground_truth_corrected_states(*, examination_id: str, patient_id: str, split: str, initial_segmentation,
                                           ground_truth, simulator, state_config: Mapping[str, Any],
                                           styles: Sequence[str], seed: int,
                                           splits: Sequence[str] = ("train", "val")
                                           ) -> tuple[list[ScribbleState], list[dict]]:
    """The ground-truth-corrected states of one examination and a log (module docstring).

    ``simulator`` draws scribbles: ``draw(mask, ground_truth, style)`` and ``draw_correction_type(mask, ground_truth,
    correction_type, style)`` return objects with ``valid``, ``correction_type`` (+1/-1) and ``scribble_zyx``
    (:class:`sirb.interaction.scribble_simulator.ScribbleSimulator`). ``state_config``: ``trajectories_per_examination``
    (trajectory t uses style ``styles[t % len(styles)]``; t >= len(styles) draws the other correction type),
    ``states_per_examination`` (an upper bound), ``partial_correction_probability`` and
    ``partial_correction_fraction_range``. A trajectory stops when no error is left, when a scribble touches no error
    (that state is kept) or when the bound is reached. Randomness: ``default_rng([seed, stable_hash(examination_id),
    trajectory])``. Examinations outside ``splits`` are refused.
    """
    if split not in tuple(splits):
        raise StateError(f"ground-truth-corrected states: split actual={split!r} allowed={tuple(splits)}")
    state0 = _frozen(_binary(initial_segmentation, "initial segmentation").copy())
    truth = _binary(ground_truth, "ground truth")
    _same_shape(state0, truth)
    styles = tuple(styles)
    unknown = [style for style in styles if style not in DRAWING_STYLES]
    if not styles or unknown:
        raise StateError(f"styles actual={styles} registered={DRAWING_STYLES}")
    if int(seed) < 0:
        raise StateError(f"seed must be a non-negative integer, got {seed!r}")
    trajectories = _config_int(state_config, "trajectories_per_examination")
    limit = _config_int(state_config, "states_per_examination")
    partial_probability = _config_probability(state_config, "partial_correction_probability")
    fraction_range = _config_fraction_range(state_config, "partial_correction_fraction_range")

    states: list[ScribbleState] = []
    log: list[dict] = []
    no_history = tuple(_frozen(channel) for channel in _earlier_scribbles([], state0.shape))
    stopped = False
    for trajectory in range(trajectories):
        if stopped:
            break
        rng = np.random.default_rng([int(seed), stable_hash(examination_id), trajectory])
        style = styles[trajectory % len(styles)]
        mask, source = state0, SOURCE_INITIAL_SEGMENTATION
        earlier: list[tuple[int, np.ndarray]] = []
        for round_index in range(CORRECTION_ROUNDS):
            where = {"examination_id": examination_id, "trajectory_index": trajectory, "round_index": round_index,
                     "style": style}
            if len(states) >= limit:
                log.append({"event": "state_limit_reached", "limit": limit, **where})
                stopped = True
                break
            drawn, policy = _draw_training_scribble(mask, truth, style=style, simulator=simulator,
                                                    other_type=trajectory >= len(styles))
            if not drawn.valid:
                log.append({"event": "no_error_left", **where})
                break
            correction_type = int(correction_sign(drawn.correction_type))
            scribble = _frozen(np.asarray(drawn.scribble_zyx, dtype=bool))
            roles = make_roles(mask, truth, scribble, correction_type)
            status = _status(roles)
            addition_history, removal_history = no_history if not earlier else tuple(
                _frozen(channel) for channel in _earlier_scribbles(earlier, mask.shape))
            state = ScribbleState(
                state_id=f"{examination_id}.seed{int(seed)}.traj{trajectory:02d}.round{round_index}",
                examination_id=examination_id, patient_id=patient_id, split=split, trajectory_index=trajectory,
                round_index=round_index, correction_type=correction_name(correction_type), style=style,
                selection_policy=policy, state_source=source, mask=mask, scribble=scribble,
                earlier_addition_scribbles=addition_history, earlier_removal_scribbles=removal_history,
                target_status=status)
            twin = next((s for s in states if s.round_index == round_index and _same_content(s, state)), None)
            states.append(state)
            where = {"state_id": state.state_id, **where}
            log.append({"event": "state_generated", "correction_type": state.correction_type, "state_source": source,
                        "target_status": status, "target_voxels": int(roles.target_error.sum()),
                        "other_error_voxels": int(roles.other_errors.sum()),
                        "scribble_voxels": int(scribble.sum()), **where})
            if twin is not None:
                log.append({"event": "duplicate_state", "duplicate_of": twin.state_id, **where})
            if status == TOUCHES_NONE:
                log.append({"event": "scribble_touches_no_error", **where})
                break
            if status == TOUCHES_SEVERAL:
                log.append({"event": "scribble_touches_several_errors",
                            "components": len(roles.touched_components), **where})
            corrected, requested = roles.target_error, None
            if rng.random() < partial_probability:
                requested = float(rng.uniform(*fraction_range))
                corrected = _partial_target(roles.target_error, scribble & roles.target_error, requested)
            full = bool(np.array_equal(corrected, roles.target_error))
            following = _frozen(_apply(mask, corrected, correction_type))
            for side in (1, -1):
                before, after = _signed_error(mask, truth, side), _signed_error(following, truth, side)
                if (after & ~before).any():
                    raise PairAlgebraViolation(f"a ground-truth correction created errors of type {side:+d}: "
                                               f"before_voxels={int(before.sum())} after_voxels={int(after.sum())} "
                                               f"({state.state_id})")
            source = SOURCE_FULL_CORRECTION if full else SOURCE_PARTIAL_CORRECTION
            log.append({"event": "target_corrected", "next_state_source": source, "requested_fraction": requested,
                        "realised_fraction": float(corrected.sum() / roles.target_error.sum()),
                        "corrected_voxels": int(corrected.sum()), **where})
            earlier.append((correction_type, scribble))
            mask = following
    return states, log


# ------------------------------------------------------------------------------------------- self-generated
@dataclass(frozen=True)
class _PreviousRound:
    """The round before: its correction type and scribble (None when nothing was drawn) and the int8 change
    the update rule wrote."""
    correction_type: str | None
    scribble: np.ndarray | None
    change: np.ndarray


def _signed_change(changed, type_name: str) -> np.ndarray:
    return _binary(changed, "change").astype(np.int8) * np.int8(correction_sign(type_name))


def _previous_round(masks, scribbles, type_names, round_index: int) -> _PreviousRound | None:
    """The memory of the round before ``round_index`` (0-based) of a stored interaction: None at round 0; else that
    round's correction type, scribble and the change it wrote (``masks[r] XOR masks[r-1]`` signed by its type).
    A change that is not the update of that round's type, or a change after a round without a scribble, is
    refused."""
    if round_index == 0:
        return None
    earlier, now = masks[round_index - 1], masks[round_index]
    if earlier is None or now is None:
        raise StateError(f"round {round_index}: masks r-1 and r are needed, got "
                         f"{[earlier is not None, now is not None]}")
    earlier, now = _binary(earlier, "mask"), _binary(now, "mask")
    added, removed = int((now & ~earlier).sum()), int((earlier & ~now).sum())
    scribble, type_name = scribbles[round_index - 1], type_names[round_index - 1]
    if scribble is None or not type_name:
        if added or removed:
            raise StateError(f"round {round_index}: the mask changed after a round without a scribble: "
                             f"added={added} removed={removed}")
        return _PreviousRound(None, None, np.zeros(now.shape, dtype=np.int8))
    wrong = removed if type_name == ADDITION else added
    if wrong:
        raise StateError(f"round {round_index}: masks r-1 and r are not the update of round {round_index - 1} "
                         f"(type {type_name!r}): added={added} removed={removed}")
    return _PreviousRound(type_name, _binary(scribble, "scribble"), _signed_change(now ^ earlier, type_name))


def _interaction_fields(round_index: int, previous: _PreviousRound | None, scribble, type_name: str):
    """(previous change, interaction flags) of round ``round_index`` (0-based)."""
    flags = dict.fromkeys(INTERACTION_FLAG_KEYS)
    flags.update({_FLAG_ROUND: int(round_index) + 1, _FLAG_PREVIOUS_TYPE: 0, _FLAG_REPEATS: False,
                  _FLAG_EMPTY: False})
    if previous is None:
        return None, flags
    flags[_FLAG_EMPTY] = not bool(previous.change.any())
    if previous.correction_type is None:
        return previous.change, flags
    flags.update({_FLAG_PREVIOUS_TYPE: int(correction_sign(previous.correction_type)),
                  _FLAG_REPEATS: bool(previous.correction_type == type_name and np.any(previous.scribble & scribble))})
    return previous.change, flags


def _type_name_or_none(correction_type) -> str | None:
    if correction_type is None or (isinstance(correction_type, str) and not correction_type):
        return None
    return correction_name(correction_type)


def states_from_rollout(*, examination_id: str, patient_id: str, split: str, ground_truth, masks: Sequence,
                        scribbles: Sequence, correction_types: Sequence, style: str, trajectory_index: int,
                        run_tag: str, splits: Sequence[str] = ("train",)) -> tuple[list[ScribbleState], list[dict]]:
    """Self-generated states of one five-round interaction of a model.

    ``masks[r]`` is the mask before round r (``masks[0]`` the initial segmentation), ``scribbles[r]`` and
    ``correction_types[r]`` the simulated user's scribble on it (None when none was drawn). Round 0 is skipped: its
    state is the initial segmentation with the simulated user's scribble, which the ground-truth-corrected states
    already hold. Every state stores the previous change and the interaction flags; the earlier scribbles are those
    of the strictly earlier rounds. Ids: ``<examination>.<run_tag>.traj<t>.round<r>``. Training set only.
    """
    if split not in tuple(splits):
        raise StateError(f"self-generated states: split actual={split!r} allowed={tuple(splits)}")
    if style not in DRAWING_STYLES:
        raise StateError(f"style actual={style!r} registered={DRAWING_STYLES}")
    if not (len(scribbles) == len(correction_types) == CORRECTION_ROUNDS) or len(masks) < 1:
        raise StateError(f"interaction record: masks={len(masks)} scribbles={len(scribbles)} "
                         f"correction types={len(correction_types)} expected rounds={CORRECTION_ROUNDS}")
    truth = _binary(ground_truth, "ground truth")
    type_names = [_type_name_or_none(value) for value in correction_types]
    states: list[ScribbleState] = []
    log: list[dict] = []
    earlier: list[tuple[int, np.ndarray]] = []
    for round_index in range(CORRECTION_ROUNDS):
        where = {"examination_id": examination_id, "trajectory_index": int(trajectory_index),
                 "round_index": round_index, "style": style}
        mask = masks[round_index] if round_index < len(masks) else None
        scribble, type_name = scribbles[round_index], type_names[round_index]
        has_scribble = scribble is not None and bool(type_name) and bool(np.asarray(scribble).any())
        reason = ("mask_missing" if mask is None else "no_scribble" if not has_scribble
                  else "initial_segmentation_round" if round_index == 0 else None)
        if reason is None:
            current = _frozen(_binary(mask, "mask").copy())
            drawn = _frozen(_binary(scribble, "scribble").copy())
            _same_shape(current, drawn, truth)
            roles = make_roles(current, truth, drawn, type_name)
            addition_history, removal_history = (_frozen(channel)
                                                 for channel in _earlier_scribbles(earlier, current.shape))
            previous = _previous_round(masks, scribbles, type_names, round_index)
            change, flags = _interaction_fields(round_index, previous, drawn, type_name)
            state = ScribbleState(
                state_id=f"{examination_id}.{run_tag}.traj{int(trajectory_index):02d}.round{round_index}",
                examination_id=examination_id, patient_id=patient_id, split=split,
                trajectory_index=int(trajectory_index), round_index=round_index, correction_type=type_name, style=style,
                selection_policy=SELECTION_SIMULATED_USER, state_source=SOURCE_SELF_GENERATED, mask=current,
                scribble=drawn, earlier_addition_scribbles=addition_history,
                earlier_removal_scribbles=removal_history, target_status=_status(roles),
                previous_change=None if change is None else _frozen(change.copy()),
                interaction_flags=MappingProxyType(dict(flags)),
                previous_scribble=None if previous is None or previous.scribble is None
                else _frozen(previous.scribble.copy()))
            states.append(state)
            log.append({"event": "state_generated", "state_id": state.state_id, "correction_type": type_name,
                        "state_source": SOURCE_SELF_GENERATED, "target_status": state.target_status,
                        "target_voxels": int(roles.target_error.sum()),
                        "other_error_voxels": int(roles.other_errors.sum()), "scribble_voxels": int(drawn.sum()),
                        **where})
        else:
            log.append({"event": "round_skipped", "reason": reason, **where})
        if has_scribble:
            earlier.append((int(correction_sign(type_name)), _binary(scribble, "scribble")))
    return states, log


# ------------------------------------------------------------------------------------------- pair factory
@dataclass(frozen=True)
class _PairContext:
    """Roles of one state, computed once and shared by its pair builders; ``levels`` (on the target's bounding box)
    is the geodesic distance from the scribble's voxels inside the target error."""
    state_id: str
    truth: np.ndarray
    roles: Roles
    scribble_target: np.ndarray
    box: tuple | None
    levels: np.ndarray | None
    max_level: int


def _pair_context(state: ScribbleState, ground_truth) -> _PairContext:
    truth = _binary(ground_truth, "ground truth")
    roles = make_roles(state.mask, truth, state.scribble, state.correction_type)
    scribble_target = state.scribble & roles.target_error
    if not scribble_target.any():
        return _PairContext(state.state_id, truth, roles, scribble_target, None, None, -1)
    box, levels = geodesic_levels(roles.target_error, scribble_target)
    return _PairContext(state.state_id, truth, roles, scribble_target, box, levels, int(levels.max()))


def _context(state: ScribbleState, ground_truth, context: _PairContext | None) -> _PairContext:
    ready = context if context is not None else _pair_context(state, ground_truth)
    if context is not None:
        truth = _binary(ground_truth, "ground truth")
        if context.state_id != state.state_id:
            raise StateError(f"pair context state actual={context.state_id!r} expected={state.state_id!r}")
        if context.truth is not truth and not np.array_equal(context.truth, truth):
            raise StateError("the pair context was built for another ground truth")
    if ready.box is None:
        raise PairInfeasible(REASON_NO_TARGET, f"target_status={_status(ready.roles)} "
                                               f"scribble_voxels={int(state.scribble.sum())}")
    return ready


def _transition_masks(before: Roles, after: Roles) -> dict[str, np.ndarray]:
    """Role transitions on the voxels editable in both states; ``stable`` means an identical role."""
    common = before.editable & after.editable
    return {"common": common, "down": common & before.target_error & ~after.target_error,
            "up": common & ~before.target_error & after.target_error,
            "stable": common & (before.labels == after.labels)}


def assemble_pair(state: ScribbleState, ground_truth, *, pair_type: str, synthetic: bool, mask_before, mask_after,
                  correction, roles_before: Roles | None = None, roles_after: Roles | None = None) -> StatePair:
    """Recompute the roles on both states, fill the transition counts and check the pair is one correct correction:
    ``mask_after`` is ``mask_before`` corrected by ``correction`` (errors of the scribble's type only, never the
    scribble's own voxels), the target error survives and never grows."""
    if pair_type not in PAIR_TYPES:
        raise StateError(f"pair type actual={pair_type!r} registered={PAIR_TYPES}")
    truth = _binary(ground_truth, "ground truth")
    type_name = correction_name(state.correction_type)
    _same_shape(truth, mask_before, mask_after, correction, state.scribble)
    before = roles_before if roles_before is not None else make_roles(mask_before, truth, state.scribble, type_name)
    after = roles_after if roles_after is not None else make_roles(mask_after, truth, state.scribble, type_name)
    size = int(correction.sum())
    expected = _apply(mask_before, correction, type_name)
    if size == 0 or not np.array_equal(expected, mask_after):
        raise PairAlgebraViolation(f"the after state is not the correction of the before state: correction_voxels="
                                   f"{size} mismatching_voxels={int((expected ^ mask_after).sum())}")
    outside_error = int((correction & ~before.errors).sum())
    on_scribble = int((correction & state.scribble).sum())
    if outside_error or on_scribble:
        raise PairAlgebraViolation(f"a correction must be errors of the scribble's type and spare the scribble: "
                                   f"voxels_outside_error={outside_error} voxels_on_scribble={on_scribble}")
    if not before.target_error.any():
        raise PairInfeasible(REASON_NO_TARGET, f"target_status={_status(before)}")
    if not after.target_error.any():
        raise PairAlgebraViolation(f"the scribble lost its target error: target_before="
                                   f"{int(before.target_error.sum())} target_after=0")
    masks = _transition_masks(before, after)
    counts = {name: int(masks[name].sum()) for name in ("common", "down", "up", "stable")}
    counts.update({"target_before": int(before.target_error.sum()), "target_after": int(after.target_error.sum()),
                   "correction_voxels": size})
    if counts["up"]:
        raise PairAlgebraViolation(f"a correct correction cannot grow the target error: up_voxels={counts['up']} "
                                   f"pair_type={pair_type}")
    if pair_type == PAIR_BRIDGE and not counts["down"]:
        raise PairInfeasible(REASON_EMPTY_DOWN_SET, f"correction_voxels={size} "
                                                    f"target_after={counts['target_after']}", retryable=True)
    origin = SYNTHETIC if synthetic else NATURAL
    return StatePair(
        pair_id=f"{state.state_id}.pair-{pair_type}-{origin}", state_id=state.state_id,
        examination_id=state.examination_id, patient_id=state.patient_id, split=state.split,
        state_source=state.state_source, pair_type=pair_type, synthetic=bool(synthetic),
        mask_before=_frozen(mask_before), mask_after=_frozen(mask_after), correction=_frozen(correction),
        correction_type=type_name, scribble=state.scribble,
        earlier_addition_scribbles=state.earlier_addition_scribbles,
        earlier_removal_scribbles=state.earlier_removal_scribbles, transition_counts=MappingProxyType(counts),
        target_status_before=_status(before), target_status_after=_status(after), attempts=1, fallback_from=None,
        status=STATUS_OK, previous_change=state.previous_change, interaction_flags=state.interaction_flags,
        previous_scribble=state.previous_scribble)


def _natural_pair(state: ScribbleState, context: _PairContext, pair_type: str, correction) -> StatePair:
    return assemble_pair(state, context.truth, pair_type=pair_type, synthetic=False, mask_before=state.mask,
                         mask_after=_apply(state.mask, correction, state.correction_type), correction=correction,
                         roles_before=context.roles)


def build_peripheral_pair(state: ScribbleState, ground_truth, *, pair_config, rng, context=None) -> StatePair:
    """Correct the geodesically farthest part of the target error: whole levels are peeled from the far end while
    they fit into the sampled share of it, then the missing voxels of the next level farthest from the scribble's
    centre. The scribble's own level is never touched."""
    ready = _context(state, ground_truth, context)
    low, high = _config_fraction_range(pair_config, "peripheral_fraction_range")
    if ready.max_level < 1:
        raise PairInfeasible(REASON_NO_PERIPHERAL_VOXELS, f"target_voxels={int(ready.roles.target_error.sum())} "
                                                          f"scribble_target_voxels={int(ready.scribble_target.sum())}")
    fraction = float(_check_rng(rng).uniform(low, high))
    counts = np.bincount(ready.levels[ready.levels >= 0])
    wanted = min(max(1, int(math.floor(fraction * int(counts.sum()) + 0.5))), int(counts[1:].sum()))
    level, taken = ready.max_level, 0
    while level >= 1 and taken + int(counts[level]) <= wanted:
        taken += int(counts[level])
        level -= 1
    chosen = ready.levels > level
    if taken < wanted:
        shell = np.argwhere(ready.levels == level)
        centre = np.argwhere(ready.levels == 0).mean(axis=0)
        order = np.lexsort((*shell.T[::-1], -np.linalg.norm(shell - centre, axis=1)))
        picked = shell[order[: wanted - taken]]
        chosen = chosen.copy()
        chosen[picked[:, 0], picked[:, 1], picked[:, 2]] = True
    correction = _embed(state.mask.shape, ready.box, chosen)
    return _natural_pair(state, ready, PAIR_PERIPHERAL, correction)


def build_other_error_pair(state: ScribbleState, ground_truth, *, pair_config, rng, context=None) -> StatePair:
    """Correct whole other-error components in random order until the sampled share of the other errors is reached
    (whole components keep the target error as it is)."""
    ready = _context(state, ground_truth, context)
    low, high = _config_fraction_range(pair_config, "other_errors_fraction_range")
    other = ready.roles.other_errors
    if not other.any():
        raise PairInfeasible(REASON_NO_OTHER_ERROR, f"error_voxels={int(ready.roles.errors.sum())} "
                                                    f"target_voxels={int(ready.roles.target_error.sum())}")
    fraction = float(_check_rng(rng).uniform(low, high))
    box = bounding_box(other)
    labels, count = ndimage.label(other[box], _structure(MAIN_CONNECTIVITY))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    chosen, covered = [], 0
    for component in rng.permutation(np.arange(1, count + 1)):
        chosen.append(int(component))
        covered += int(sizes[component])
        if covered >= fraction * int(other.sum()):
            break
    correction = _embed(state.mask.shape, box, np.isin(labels, chosen))
    return _natural_pair(state, ready, PAIR_OTHER_ERROR, correction)


def build_natural_bridge_pair(state: ScribbleState, ground_truth, *, pair_config, rng, context=None) -> StatePair:
    """Cut the target error at a thin neck. For a level k every 18-connected component u of ``levels >= k`` hangs on
    the rest through its level-k voxels c only, so correcting c detaches ``far = u - c``. (k, u) is a neck when
    ``|c| <= bridge_neck_ratio_max * min(|near|, |far|) ** (2/3)`` and both sides hold at least
    ``bridge_min_side_voxels`` voxels; one neck is drawn at random."""
    ready = _context(state, ground_truth, context)
    ratio_max = float(pair_config["bridge_neck_ratio_max"])
    min_side = _config_int(pair_config, "bridge_min_side_voxels")
    if not ratio_max > 0:
        raise StateError(f"bridge_neck_ratio_max must be positive, got {ratio_max}")
    if ready.max_level < 2:
        raise PairInfeasible(REASON_TARGET_TOO_SHALLOW, f"max_geodesic_level={ready.max_level} required>=2")
    structure = _structure(MAIN_CONNECTIVITY)
    total = int((ready.levels >= 0).sum())
    necks: list[tuple[int, int]] = []
    best = math.inf
    for level in range(1, ready.max_level):
        labels, count = ndimage.label(ready.levels >= level, structure)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        cuts = np.bincount(labels[ready.levels == level], minlength=count + 1)
        for component in range(1, count + 1):
            cut, far = int(cuts[component]), int(sizes[component] - cuts[component])
            near = total - int(sizes[component])
            if min(near, far) < min_side:
                continue
            ratio = cut / min(near, far) ** NECK_AREA_EXPONENT
            best = min(best, ratio)
            if ratio <= ratio_max:
                necks.append((level, component))
    if not necks:
        raise PairInfeasible(REASON_NO_THIN_NECK, f"thinnest_neck_ratio={best:.3f} allowed<={ratio_max} "
                                                  f"min_side_voxels={min_side} target_voxels={total}")
    level, component = necks[int(_check_rng(rng).integers(len(necks)))]
    labels, _ = ndimage.label(ready.levels >= level, structure)
    correction = _embed(state.mask.shape, ready.box, (labels == component) & (ready.levels == level))
    return _natural_pair(state, ready, PAIR_BRIDGE, correction)


def _face_offsets() -> np.ndarray:
    offsets = np.argwhere(_structure(BRIDGE_PATH_CONNECTIVITY)) - 1
    return offsets[np.any(offsets != 0, axis=1)].astype(np.int64)


def build_synthetic_bridge_pair(state: ScribbleState, ground_truth, *, pair_config, rng, context=None,
                                splits: Sequence[str] = ("train",)) -> StatePair:
    """Inject the shortest 6-connected error path (at most ``synthetic_bridge_max_gap_voxels`` voxels) that ties the
    target error to a nearby other-error component; ``mask_before`` is the injected state, ``mask_after`` the real
    one and the correction removes the path. Training set only."""
    if state.split not in tuple(splits):
        raise StateError(f"synthetic bridge: split actual={state.split!r} allowed={tuple(splits)}")
    ready = _context(state, ground_truth, context)
    max_gap = _config_int(pair_config, "synthetic_bridge_max_gap_voxels")
    _check_rng(rng)
    other = ready.roles.other_errors
    if not other.any():
        raise PairInfeasible(REASON_NO_OTHER_ERROR, f"target_voxels={int(ready.roles.target_error.sum())}")
    mask, truth = state.mask, ready.truth
    legal = (mask & truth) if state.correction_type == ADDITION else (~mask & ~truth)
    box = bounding_box(ready.roles.target_error, margin=max_gap + 1)
    legal_box = legal[box] & ~state.scribble[box]
    face = _structure(BRIDGE_PATH_CONNECTIVITY)
    distance = np.zeros(legal_box.shape, dtype=np.int32)
    frontier = ready.roles.target_error[box]
    for step in range(1, max_gap + 1):
        frontier = ndimage.binary_dilation(frontier, structure=face) & legal_box & (distance == 0)
        if not frontier.any():
            break
        distance[frontier] = step
    labels, _ = ndimage.label(other[box], _structure(MAIN_CONNECTIVITY))
    ends = np.argwhere((distance > 0) & ndimage.binary_dilation(labels > 0, structure=face))
    offsets = _face_offsets()
    limit = np.asarray(distance.shape, dtype=np.int64)
    rows = []
    for offset in offsets:
        neighbours = ends + offset
        inside = np.all((neighbours >= 0) & (neighbours < limit), axis=1)
        touched = np.zeros(len(ends), dtype=np.int64)
        touched[inside] = labels[tuple(neighbours[inside].T)]
        hit = touched > 0
        rows.append(np.column_stack([touched[hit], distance[tuple(ends[hit].T)], ends[hit]]))
    table = np.concatenate(rows) if rows else np.zeros((0, 5), dtype=np.int64)
    if len(table) == 0:
        raise PairInfeasible(REASON_NO_BRIDGE_PARTNER, f"max_gap_voxels={max_gap} other_voxels={int(other.sum())} "
                                                       f"reachable_legal_voxels={int((distance > 0).sum())}")
    table = table[np.lexsort(tuple(table[:, column] for column in range(4, -1, -1)))]
    shortest = table[np.unique(table[:, 0], return_index=True)[1]]
    partner = shortest[int(rng.integers(len(shortest)))]
    length = int(partner[1])
    path = [partner[2:].astype(np.int64)]
    for step in range(length - 1, 0, -1):
        for neighbour in path[-1] + offsets:
            if np.all(neighbour >= 0) and np.all(neighbour < limit) and distance[tuple(neighbour)] == step:
                path.append(neighbour)
                break
        else:
            raise PairAlgebraViolation(f"bridge backtracking lost the path at step={step} of length={length}")
    correction = _embed(mask.shape, box, _volume_from_indices(np.asarray(path), legal_box.shape))
    injected = _apply(mask, correction, -correction_sign(state.correction_type))
    return assemble_pair(state, truth, pair_type=PAIR_BRIDGE, synthetic=True, mask_before=injected, mask_after=mask,
                         correction=correction, roles_after=ready.roles)


def build_state_pairs(state: ScribbleState, ground_truth, *, pair_config: Mapping[str, Any], synthetic_bridges: bool,
                      rng: np.random.Generator, synthetic_splits: Sequence[str] = ("train",)
                      ) -> tuple[dict[str, StatePair], list[dict]]:
    """Every pair type that exists for the state -> (pairs by type, log).

    Each type gets at most :data:`MAX_PAIR_ATTEMPTS` attempts; a failure resampling cannot change ends them. With
    ``synthetic_bridges`` each bridge attempt is synthetic with ``synthetic_bridge_probability``; when the drawn
    origin is impossible the other one is used.
    """
    _check_rng(rng)
    if synthetic_bridges and state.split not in tuple(synthetic_splits):
        raise StateError(f"synthetic bridge: split actual={state.split!r} allowed={tuple(synthetic_splits)}")
    synthetic_probability = _config_probability(pair_config, "synthetic_bridge_probability")
    context = _pair_context(state, ground_truth)
    builders: dict[str, dict[str, Callable[..., StatePair]]] = {
        PAIR_PERIPHERAL: {NATURAL: build_peripheral_pair}, PAIR_OTHER_ERROR: {NATURAL: build_other_error_pair},
        PAIR_BRIDGE: {NATURAL: build_natural_bridge_pair}}
    if synthetic_bridges:
        builders[PAIR_BRIDGE][SYNTHETIC] = functools.partial(build_synthetic_bridge_pair,
                                                             splits=tuple(synthetic_splits))
    pairs: dict[str, StatePair] = {}
    log: list[dict] = []
    for pair_type in PAIR_TYPES:
        origins = builders[pair_type]
        closed: dict[str, str] = {}
        attempts = 0
        while attempts < MAX_PAIR_ATTEMPTS and len(closed) < len(origins):
            attempts += 1
            drawn = NATURAL
            if SYNTHETIC in origins and rng.random() < synthetic_probability:
                drawn = SYNTHETIC
            origin = drawn if drawn not in closed else next(o for o in origins if o not in closed)
            row = {"state_id": state.state_id, "pair_type": pair_type, "origin": origin,
                   "origin_switched_from": drawn if drawn != origin else None, "attempt": attempts}
            try:
                pair = origins[origin](state, ground_truth, pair_config=pair_config, rng=rng, context=context)
            except PairInfeasible as failure:
                log.append({"event": "pair_attempt_failed", "reason": failure.reason, "detail": failure.detail,
                            "retryable": failure.retryable, **row})
                if not failure.retryable:
                    closed[origin] = failure.reason
                continue
            pairs[pair_type] = replace(pair, attempts=attempts)
            log.append({"event": "pair_built", "pair_id": pair.pair_id,
                        "transition_counts": dict(pair.transition_counts), **row})
            break
        if pair_type not in pairs:
            log.append({"event": "pair_type_infeasible", "state_id": state.state_id, "pair_type": pair_type,
                        "attempts": attempts, "reasons": dict(closed)})
    return pairs, log


def choose_pair_type(available: Iterable[str], mix: Mapping[str, float], rng: np.random.Generator
                     ) -> tuple[str, str | None]:
    """Draw a pair type by ``mix`` -> (type used, type drawn first or None). When the drawn type does not exist for
    the state the draw is repeated over the available types with the mix renormalised. Order follows
    :data:`PAIR_TYPES` (iterating a set of strings is salted per process)."""
    _check_rng(rng)
    available = set(available)
    if set(mix) != set(PAIR_TYPES) or not available <= set(PAIR_TYPES):
        raise StateError(f"pair types: mix={sorted(mix)} available={sorted(available)} registered={PAIR_TYPES}")
    weights = np.asarray([float(mix[name]) for name in PAIR_TYPES], dtype=np.float64)
    if np.any(weights < 0) or not weights.sum() > 0:
        raise StateError(f"the pair-type mix must be non-negative with a positive sum, got {dict(mix)}")
    drawn = PAIR_TYPES[int(rng.choice(len(PAIR_TYPES), p=weights / weights.sum()))]
    if drawn in available:
        return drawn, None
    usable = [index for index, name in enumerate(PAIR_TYPES) if name in available and weights[index] > 0]
    if not usable:
        raise StateError(f"no available pair type has a positive weight: available={sorted(available)} "
                         f"mix={dict(mix)}")
    share = weights[usable] / weights[usable].sum()
    return PAIR_TYPES[usable[int(rng.choice(len(usable), p=share))]], drawn


# ------------------------------------------------------------------------------------------- stored format: records
def _pack(volume) -> np.ndarray:
    return np.packbits(np.ascontiguousarray(volume, dtype=bool))


def _signed_parts(change, shape, path) -> tuple[np.ndarray, np.ndarray]:
    signed = np.asarray(change)
    if tuple(signed.shape) != tuple(shape):
        raise StateError(f"previous change shape actual={tuple(signed.shape)} expected={tuple(shape)} ({path})")
    if not np.issubdtype(signed.dtype, np.integer) or int(np.abs(signed).max(initial=0)) > 1:
        raise StateError(f"previous change: dtype={signed.dtype} max |value|={int(np.abs(signed).max(initial=0))} "
                         f"expected an integer map of +1, -1, 0 ({path})")
    return signed > 0, signed < 0


def save_state_pair(path: str | os.PathLike, pair: StatePair) -> None:
    """Write one state pair (``.npz``): its shape, its JSON record and one bit per voxel for each volume (and for the
    previous change and scribble when the pair has them). An existing file is refused."""
    target = Path(path)
    if target.suffix != ".npz":
        raise StateError(f"record suffix actual={target.suffix!r} expected '.npz'")
    volumes = [getattr(pair, name) for name in _PAIR_VOLUMES]
    _same_shape(*volumes)
    record = {name: getattr(pair, name) for name in _PAIR_FIELDS}
    record["transition_counts"] = dict(pair.transition_counts)
    record["interaction_flags"] = None if pair.interaction_flags is None else dict(pair.interaction_flags)
    shape = np.shape(volumes[0])
    arrays = {"shape": np.asarray(shape, dtype=_SHAPE_DTYPE),
              "record_json": np.asarray(json.dumps({"schema": PAIR_SCHEMA, "version": PAIR_SCHEMA_VERSION, **record},
                                                   sort_keys=True, ensure_ascii=False, allow_nan=False))}
    for name, volume in zip(_PAIR_VOLUMES, volumes):
        arrays[f"{name}_bits"] = _pack(_binary(volume, name))
    if pair.previous_change is not None:
        for key, part in zip(_CHANGE_PARTS, _signed_parts(pair.previous_change, shape, target)):
            arrays[key] = _pack(part)
    if pair.previous_scribble is not None:
        previous = _binary(pair.previous_scribble, "previous scribble")
        _same_shape(previous, volumes[0])
        arrays["previous_scribble_bits"] = _pack(previous)
    try:
        write_npz_atomic(target, arrays, exclusive=True)
    except TargetExistsError as refusal:
        raise StateError(f"refusing to overwrite an existing record: {target}") from refusal


def _unpack(archive, key: str, shape, path) -> np.ndarray:
    size = int(np.prod(shape))
    bits = archive[key]
    if bits.size != (size + 7) // 8:
        raise StateError(f"{key} packed length actual={bits.size} expected={(size + 7) // 8} ({path})")
    return _frozen(np.unpackbits(bits, count=size).reshape(shape).view(np.bool_))


#: The volumes a training unit reads: both masks, the scribble, the earlier scribbles and (when stored) the previous
#: change and scribble; the correction stays packed on disk.
TRAINING_VOLUMES = ("mask_before", "mask_after", "scribble", "earlier_addition_scribbles",
                    "earlier_removal_scribbles", "previous_change", "previous_scribble")


def load_state_pair(path: str | os.PathLike, *, volumes: Sequence[str] | None = None) -> StatePair:
    """Read one stored state pair; volumes come back read-only. A volume left out of ``volumes`` (default: all) is not
    decoded and comes back as None."""
    names = list(_PAIR_VOLUMES) + ["previous_change", "previous_scribble"]
    wanted = set(names if volumes is None else volumes)
    if not wanted <= set(names):
        raise StateError(f"volumes requested={sorted(wanted)} stored={names} ({path})")
    with np.load(Path(path), allow_pickle=False) as archive:
        record = json.loads(str(archive["record_json"][()]))
        schema, version = record.pop("schema", None), record.pop("version", None)
        if (schema, version) != (PAIR_SCHEMA, PAIR_SCHEMA_VERSION):
            raise StateError(f"record schema actual={(schema, version)} expected={(PAIR_SCHEMA, PAIR_SCHEMA_VERSION)} "
                             f"({path})")
        shape = tuple(int(v) for v in archive["shape"])
        values = {name: _unpack(archive, f"{name}_bits", shape, path) if name in wanted else None
                  for name in _PAIR_VOLUMES}
        change = previous_scribble = None
        if all(key in archive.files for key in _CHANGE_PARTS) and "previous_change" in wanted:
            added, removed = (_unpack(archive, key, shape, path) for key in _CHANGE_PARTS)
            change = _frozen(added.astype(np.int8) - removed.astype(np.int8))
        if "previous_scribble_bits" in archive.files and "previous_scribble" in wanted:
            previous_scribble = _unpack(archive, "previous_scribble_bits", shape, path)
    flags, stored_counts = record.pop("interaction_flags"), record.pop("transition_counts")
    counts = MappingProxyType({key: int(stored_counts[key]) for key in _COUNT_KEYS})
    return StatePair(**record, **values, transition_counts=counts, previous_change=change,
                     interaction_flags=None if flags is None else MappingProxyType(dict(flags)),
                     previous_scribble=previous_scribble)


def load_training_state_pair(path: str | os.PathLike) -> StatePair:
    """The stored pair as a training unit reads it (:data:`TRAINING_VOLUMES`)."""
    return load_state_pair(path, volumes=TRAINING_VOLUMES)


# ------------------------------------------------------------------------------------------- stored format: pools
def state_pair_row(pair: StatePair, path: str | os.PathLike, *, root: str | os.PathLike | None = None) -> dict:
    """The manifest row of a stored pair: its record (without the volumes), its shape, the number of voxels the
    previous change holds (None without one) and ``path`` relative to ``root``."""
    target = Path(path)
    shown = target.resolve().relative_to(Path(root).resolve()) if root is not None else target
    row = {name: getattr(pair, name) for name in _PAIR_FIELDS}
    row.update(schema=PAIR_SCHEMA, version=PAIR_SCHEMA_VERSION, synthetic=bool(pair.synthetic),
               attempts=int(pair.attempts), shape=[int(v) for v in pair.mask_before.shape],
               transition_counts={key: int(pair.transition_counts[key]) for key in _COUNT_KEYS},
               interaction_flags=None if pair.interaction_flags is None else dict(pair.interaction_flags),
               previous_change_voxels=None if pair.previous_change is None
               else int(np.count_nonzero(pair.previous_change)), path=shown.as_posix())
    return row


def write_state_pairs(root: str | os.PathLike, states: Sequence[ScribbleState], ground_truth, *, seed: int,
                      pair_config: Mapping[str, Any], synthetic_bridges: bool,
                      synthetic_splits: Sequence[str] = ("train",), counters: collections.Counter | None = None
                      ) -> tuple[list[dict], list[dict]]:
    """Build and store every pair of the states of ONE examination under ``root/pairs/<examination>/``.

    The pair generator of a state is ``default_rng([seed, stable_hash(state_id)])``. Returns (manifest rows, log);
    ``counters`` is updated in place.
    """
    root = Path(root)
    counters = collections.Counter() if counters is None else counters
    rows: list[dict] = []
    log: list[dict] = []
    for state in states:
        counters[f"state_source_{state.state_source}"] += 1
        counters[f"target_status_{state.target_status}"] += 1
        rng = np.random.default_rng([int(seed), stable_hash(state.state_id)])
        pairs, pair_log = build_state_pairs(state, ground_truth, pair_config=pair_config,
                                            synthetic_bridges=synthetic_bridges, rng=rng,
                                            synthetic_splits=synthetic_splits)
        log.extend(pair_log)
        counters["states_without_pairs"] += int(not pairs)
        for pair in pairs.values():
            path = root / PAIRS_FOLDER / state.examination_id / f"{pair.pair_id}.npz"
            save_state_pair(path, pair)
            rows.append(state_pair_row(pair, path, root=root))
            counters[f"pair_{pair.pair_type}_{SYNTHETIC if pair.synthetic else NATURAL}"] += 1
    counters["states"] += len(states)
    return rows, log


def append_rows(path: str | os.PathLike, rows: Iterable[Mapping[str, Any]]) -> None:
    """Append one JSON line per row (sorted keys, UTF-8, LF); the file is created even for no row."""
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_pool_rows(manifest: str | os.PathLike) -> list[dict]:
    """The pair rows of a pool manifest: ``pairs.jsonl``, or a JSON document with ``"rows"`` (``pool.json``)."""
    path = Path(manifest)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict):
        document = document.get("rows")
    if not isinstance(document, list):
        raise StateError(f"{path}: a pool manifest holds a list of rows (JSONL, or JSON with 'rows')")
    return [dict(row) for row in document]


def publish_pool(directory: str | os.PathLike, rows: Sequence[Mapping[str, Any]], *, identity: Mapping[str, Any],
                 examinations: Sequence[str]) -> Path:
    """Write ``pool.json`` (identity, examinations, rows) of a complete self-generated pool; returns its path."""
    target = Path(directory) / POOL_MANIFEST
    write_json_atomic(target, {"identity": dict(identity), "examinations": sorted(examinations), "rows": list(rows)},
                      indent=2, sort_keys=False, ensure_ascii=False, exclusive=True)
    return target


def resolve_pair_path(row: Mapping[str, Any], root: str | os.PathLike) -> Path:
    """The pair file of a manifest row, which must lie inside the pool directory ``root`` and exist."""
    base = Path(root).resolve()
    path = Path(row["path"])
    path = path.resolve() if path.is_absolute() else (base / path).resolve()
    try:
        path.relative_to(base)
    except ValueError as error:
        raise StateError(f"pair path actual={path}, expected inside the pool directory {base}") from error
    if not path.is_file():
        raise StateError(f"pair file is missing: {path}")
    return path
