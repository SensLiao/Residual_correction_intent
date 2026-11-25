"""The auxiliary model: a rule-based corrector, and the auxiliary-model states of its five-round interactions.

The auxiliary model needs no weights. It starts from the initial segmentation and, in every round, answers the
simulated user's scribble with an imperfect correction (:func:`imperfect_correction`): the ground truth gives the
scribble's target error T, one of seven outcomes turns T into the region A the round writes, and the update rule
applies A with the scribble's correction type (an addition adds A, a removal removes it). The outcomes
(:data:`OUTCOMES`, drawn by ``outcome_mix``):

* ``complete``: A = T;
* ``partial_near``: the part of T reached first from the scribble, where a step costs more the more the PET uptake
  differs from the uptake under the scribble; a sampled fraction of T. The far part, or a part of different uptake,
  stays;
* ``partial_split``: T is ordered along its principal axis, starting at the scribble's end; either the far end
  stays or a middle band stays, which leaves the corrected target in two pieces (elongated targets only);
* ``spill_uptake``: T and the voxels next to it reached first when a step costs more the more the uptake differs
  from that of T; the spill holds a sampled fraction of ``|T|``;
* ``spill_boundary``: T and one dilation of T inside a random box on its boundary;
* ``extra_region``: T and a second region a few voxels away, grown from a seed (for an addition one of the
  highest-uptake voxels nearby that are not lesion, for a removal a correctly segmented lesion voxel nearby);
* ``no_change``: A is empty, so the next scribble is drawn on the same mask.

A never leaves the editable domain of the scribble's correction type, and every voxel of A outside T is a voxel the
mask labelled correctly: an imperfect correction leaves part of its target, or creates errors, and does nothing
else. The sampled fraction is checked on the region that was built; an outcome that cannot be built for a target
(or misses its fraction ``max_attempts`` times) is replaced by ``complete`` and logged with the reason.

:func:`auxiliary_rollout` runs the five rounds of one trajectory and :func:`generate_auxiliary_model_states` turns
the trajectories of one examination into states with :func:`sirb.data.states.states_from_rollout`, so the states
carry the previous change and the interaction flags like the self-generated states. Everything is configured by
``data.auxiliary_model``. Arrays are ZYX on the native PET grid. Training set only. CPU only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

from ..interaction.update_rule import apply
from .psma import DRAWING_STYLES
from .roles import correction_sign, make_roles
from .state_operations import (
    StateError,
    binary_volume,
    bounding_box,
    connectivity_structure,
    embed_region,
    configured_fraction_range,
    configured_positive_integer,
    readonly_volume,
    require_generator,
    require_same_shape,
)
from .states import CORRECTION_ROUNDS, MAIN_CONNECTIVITY, ScribbleState, correction_name, stable_hash, states_from_rollout

__all__ = ["OUTCOMES", "SOURCE_AUXILIARY_MODEL", "Correction", "auxiliary_rollout",
           "generate_auxiliary_model_states", "imperfect_correction"]

#: Stored state source of the auxiliary-model states.
SOURCE_AUXILIARY_MODEL = "auxiliary_model"
#: The outcomes of one round, in the order ``outcome_mix`` is read.
COMPLETE, PARTIAL_NEAR, PARTIAL_SPLIT = "complete", "partial_near", "partial_split"
SPILL_UPTAKE, SPILL_BOUNDARY, EXTRA_REGION, NO_CHANGE = "spill_uptake", "spill_boundary", "extra_region", "no_change"
OUTCOMES = (COMPLETE, PARTIAL_NEAR, PARTIAL_SPLIT, SPILL_UPTAKE, SPILL_BOUNDARY, EXTRA_REGION, NO_CHANGE)
#: The two ways ``partial_split`` leaves a part of the target.
SPLIT_FAR_END, SPLIT_MIDDLE = "far_end", "middle"
#: The dilation of ``spill_boundary`` is 6-connected.
BOUNDARY_CONNECTIVITY = 6

REASON_NO_TARGET = "scribble_touches_no_error"
REASON_TARGET_TOO_SMALL = "target_too_small"
REASON_SCRIBBLE_COVERS_TARGET = "scribble_covers_the_target"
REASON_NOT_ELONGATED = "target_not_elongated"
REASON_NO_ROOM = "no_voxel_to_spill_into"
REASON_NO_REGION_NEARBY = "no_candidate_region_nearby"
REASON_ATTEMPTS = "fraction_missed_in_every_attempt"


class _Infeasible(Exception):
    """The outcome cannot be built for this target (no further attempt can change that)."""


class _Missed(Exception):
    """This attempt missed its fraction; another draw may hit it."""


@dataclass(frozen=True)
class Correction:
    """The imperfect correction of one round.

    ``region`` is what the update rule writes. ``outcome`` is the outcome that was built and ``drawn`` the one drawn
    first (they differ when ``drawn`` could not be built; ``reason`` says why). ``requested`` is the sampled fraction
    (None for the outcomes without one); ``corrected_fraction`` is ``|region & target| / |target|`` and
    ``spill_fraction`` is ``|region - target| / |target|`` (both 0 without a target).
    """
    region: np.ndarray
    outcome: str
    drawn: str
    requested: float | None
    corrected_fraction: float
    spill_fraction: float
    target_voxels: int
    attempts: int
    variant: str | None = None
    reason: str | None = None

    def to_log(self) -> dict:
        return {"outcome": self.outcome, "drawn_outcome": self.drawn, "requested_fraction": self.requested,
                "corrected_fraction": self.corrected_fraction, "spill_fraction": self.spill_fraction,
                "target_voxels": self.target_voxels, "attempts": self.attempts, "variant": self.variant,
                "reason": self.reason}


# ------------------------------------------------------------------------------------------- configuration
def _int_range(config: Mapping[str, Any], key: str, *, minimum: int) -> tuple[int, int]:
    low, high = config[key]
    if any(isinstance(v, bool) or int(v) != v for v in (low, high)) or not minimum <= int(low) <= int(high):
        raise StateError(f"{key} must be two integers with {minimum} <= low <= high, got {config[key]!r}")
    return int(low), int(high)


def _positive(config: Mapping[str, Any], key: str, *, minimum: float = 0.0) -> float:
    value = float(config[key])
    if not np.isfinite(value) or value < minimum:
        raise StateError(f"{key} must be a finite number >= {minimum}, got {config[key]!r}")
    return value


def _outcome_mix(config: Mapping[str, Any]) -> np.ndarray:
    mix = config["outcome_mix"]
    if set(mix) != set(OUTCOMES):
        raise StateError(f"outcome_mix keys actual={sorted(mix)} expected={sorted(OUTCOMES)}")
    weights = np.asarray([float(mix[name]) for name in OUTCOMES], dtype=np.float64)
    if np.any(weights < 0) or not weights.sum() > 0:
        raise StateError(f"outcome_mix must be non-negative with a positive sum, got {dict(mix)}")
    return weights / weights.sum()


# ------------------------------------------------------------------------------------------- growth
def _offsets(connectivity: int) -> np.ndarray:
    offsets = np.argwhere(connectivity_structure(connectivity)) - 1
    return offsets[np.any(offsets != 0, axis=1)]


def _first_reached(domain, seeds, uptake, *, reference: float, weight: float, count: int) -> np.ndarray:
    """The ``count`` voxels of ``domain`` (seeds excluded) reached first from ``seeds``.

    Paths are 18-connected and stay inside ``domain | seeds``; entering a voxel costs ``1 + weight * |uptake -
    reference|``, so the growth prefers near voxels of similar uptake and the chosen set hangs on the seeds. Ties
    are taken in array order. Fewer voxels come back when fewer can be reached. All arrays share one (boxed) shape.
    """
    from skimage.graph import MCP  # noqa: PLC0415  (needed for generation only)

    free = domain & ~seeds
    if count < 1 or not free.any() or not seeds.any():
        return np.zeros(domain.shape, dtype=bool)
    costs = np.where(free, 1.0 + weight * np.abs(uptake.astype(np.float64) - reference), -1.0)
    costs[seeds] = 1.0
    reached, _ = MCP(costs, offsets=_offsets(MAIN_CONNECTIVITY)).find_costs(np.argwhere(seeds))
    cost = reached[free]
    order = np.argsort(cost, kind="stable")[:count]
    order = order[np.isfinite(cost[order])]
    chosen = np.zeros(domain.shape, dtype=bool)
    chosen[tuple(np.argwhere(free)[order].T)] = True
    return chosen


def _median(uptake, region) -> float:
    return float(np.median(np.asarray(uptake[region], dtype=np.float64)))


def _voxel_count(fraction: float, total: int) -> int:
    return int(np.floor(fraction * total + 0.5))


def _check_fraction(realised: float, requested: float, tolerance: float) -> None:
    if abs(realised - requested) > tolerance:
        raise _Missed(f"realised={realised:.3f} requested={requested:.3f} tolerance={tolerance}")


# ------------------------------------------------------------------------------------------- the outcomes
@dataclass(frozen=True)
class _Round:
    """One round as the outcome builders see it: the target error, the scribble's voxels in it, the voxels a spill may
    take (editable and correctly labelled) and the PET."""
    target: np.ndarray
    seeds: np.ndarray
    spillable: np.ndarray
    uptake: np.ndarray
    spacing: np.ndarray
    size: int


def _partial_near(now: _Round, config, rng):
    seeds = int(now.seeds.sum())
    if now.size < 2:
        raise _Infeasible(REASON_TARGET_TOO_SMALL)
    if seeds >= now.size:
        raise _Infeasible(REASON_SCRIBBLE_COVERS_TARGET)
    requested = float(rng.uniform(*configured_fraction_range(config, "partial_fraction_range")))
    count = min(max(_voxel_count(requested, now.size), seeds), now.size - 1)
    _check_fraction(count / now.size, requested, float(config["fraction_tolerance"]))
    box = bounding_box(now.target)
    grown = _first_reached(now.target[box], now.seeds[box], now.uptake[box],
                           reference=_median(now.uptake, now.seeds), weight=float(config["uptake_weight"]),
                           count=count - seeds)
    return embed_region(now.target.shape, box, grown | now.seeds[box]), requested, None


def _partial_split(now: _Round, config, rng):
    if now.size < configured_positive_integer(config, "split_min_voxels"):
        raise _Infeasible(REASON_TARGET_TOO_SMALL)
    voxels = np.argwhere(now.target)
    centred = (voxels - voxels.mean(axis=0)) * now.spacing
    values, vectors = np.linalg.eigh(centred.T @ centred / len(voxels))
    if np.sqrt(values[2] / max(values[1], 1e-12)) < float(config["split_min_elongation"]):
        raise _Infeasible(REASON_NOT_ELONGATED)
    order = np.argsort(centred @ vectors[:, 2], kind="stable")
    rank = np.empty(len(voxels), dtype=np.int64)
    rank[order] = np.arange(len(voxels))
    if rank[now.seeds[tuple(voxels.T)]].mean() > (len(voxels) - 1) / 2:      # the scribble's end comes first
        order = order[::-1]
    requested = float(rng.uniform(*configured_fraction_range(config, "partial_fraction_range")))
    variant = SPLIT_MIDDLE if rng.random() < float(config["split_middle_probability"]) else SPLIT_FAR_END
    count = min(max(_voxel_count(requested, now.size), 2 if variant == SPLIT_MIDDLE else 1), now.size - 1)
    _check_fraction(count / now.size, requested, float(config["fraction_tolerance"]))
    if variant == SPLIT_MIDDLE:
        left = now.size - count                                              # voxels of the band that stays
        start = int(rng.integers(1, count))                                  # 1 <= start, start + left <= size - 1
        kept = np.concatenate([order[:start], order[start + left:]])
    else:
        kept = order[:count]
    region = np.zeros(now.target.shape, dtype=bool)
    region[tuple(voxels[kept].T)] = True
    return region, requested, variant


def _spill_uptake(now: _Round, config, rng):
    requested = float(rng.uniform(*configured_fraction_range(config, "spill_fraction_range")))
    count = max(_voxel_count(requested, now.size), 1)
    box = bounding_box(now.target, margin=2 + 2 * int(np.ceil(count ** (1.0 / 3.0))))
    spill = _first_reached(now.spillable[box], now.target[box], now.uptake[box],
                           reference=_median(now.uptake, now.target), weight=float(config["uptake_weight"]),
                           count=count)
    if not spill.any():
        raise _Infeasible(REASON_NO_ROOM)
    _check_fraction(int(spill.sum()) / now.size, requested, float(config["fraction_tolerance"]))
    return embed_region(now.target.shape, box, spill | now.target[box]), requested, None


def _spill_boundary(now: _Round, config, rng):
    half_low, half_high = _int_range(config, "boundary_box_half_voxels", minimum=1)
    step_low, step_high = _int_range(config, "boundary_dilation_iterations", minimum=1)
    _, high = configured_fraction_range(config, "spill_fraction_range")
    box = bounding_box(now.target, margin=step_high)
    target = now.target[box]
    structure = connectivity_structure(BOUNDARY_CONNECTIVITY)
    if not (ndimage.binary_dilation(target, structure=structure) & now.spillable[box]).any():
        raise _Infeasible(REASON_NO_ROOM)
    boundary = np.argwhere(target & ~ndimage.binary_erosion(target, structure=structure))
    centre = boundary[int(rng.integers(len(boundary)))]
    half = rng.integers(half_low, half_high + 1, size=3)
    window = tuple(slice(max(int(c - h), 0), int(c + h) + 1) for c, h in zip(centre, half))
    steps = int(rng.integers(step_low, step_high + 1))
    spill = np.zeros(target.shape, dtype=bool)
    spill[window] = (ndimage.binary_dilation(target[window], structure=structure, iterations=steps)
                     & now.spillable[box][window])
    realised = int(spill.sum()) / now.size
    if not spill.any() or realised > high + float(config["fraction_tolerance"]):
        raise _Missed(f"spill_fraction={realised:.3f} allowed<={high}")
    return embed_region(now.target.shape, box, spill | target), None, None


def _extra_region(now: _Round, config, rng, *, addition: bool):
    gap_low, gap_high = _int_range(config, "extra_region_gap_voxels", minimum=2)
    requested = float(rng.uniform(*configured_fraction_range(config, "spill_fraction_range")))
    count = max(_voxel_count(requested, now.size), 1)
    box = bounding_box(now.target, margin=gap_high + 2 * int(np.ceil(count ** (1.0 / 3.0))))
    distance = ndimage.distance_transform_edt(~now.target[box])
    apart = now.spillable[box] & (distance >= gap_low)
    candidates = np.argwhere(apart & (distance <= gap_high))
    if not len(candidates):
        raise _Infeasible(REASON_NO_REGION_NEARBY)
    uptake = now.uptake[box]
    if addition:                                                 # one of the highest-uptake voxels that are not lesion
        values = np.asarray(uptake[tuple(candidates.T)], dtype=np.float64)
        top = max(1, int(np.ceil(float(config["extra_region_top_uptake_fraction"]) * len(candidates))))
        candidates = candidates[np.argsort(-values, kind="stable")[:top]]
    seed = np.zeros(apart.shape, dtype=bool)
    seed[tuple(candidates[int(rng.integers(len(candidates)))])] = True
    grown = _first_reached(apart, seed, uptake, reference=float(uptake[seed][0]),
                           weight=float(config["uptake_weight"]), count=count - 1)
    extra = grown | seed
    _check_fraction(int(extra.sum()) / now.size, requested, float(config["fraction_tolerance"]))
    return embed_region(now.target.shape, box, extra | now.target[box]), requested, None


def imperfect_correction(mask, ground_truth, scribble, correction_type, pet, *, config: Mapping[str, Any],
                         rng: np.random.Generator, spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0)) -> Correction:
    """The auxiliary model's correction of one scribble on ``mask`` (module docstring).

    ``pet`` is the normalised PET the model reads; ``spacing_zyx`` (mm) only shapes the principal axis of
    ``partial_split``. ``config`` is ``data.auxiliary_model``. The returned region lies in the editable domain of the
    correction type, and outside the target error it holds only voxels the mask labelled correctly.
    """
    require_generator(rng)
    current = binary_volume(mask, "mask")
    truth = binary_volume(ground_truth, "ground truth")
    drawn_on = binary_volume(scribble, "scribble")
    uptake = np.asarray(pet)
    require_same_shape(current, truth, drawn_on, uptake)
    sign = correction_sign(correction_type)
    roles = make_roles(current, truth, drawn_on, sign)
    target = roles.target_error
    size = int(target.sum())
    drawn = OUTCOMES[int(rng.choice(len(OUTCOMES), p=_outcome_mix(config)))]
    attempts_allowed = configured_positive_integer(config, "max_attempts")
    _positive(config, "fraction_tolerance")
    _positive(config, "uptake_weight")
    empty = np.zeros(current.shape, dtype=bool)

    def result(region, outcome, requested=None, attempts=1, variant=None, reason=None) -> Correction:
        inside = int((region & target).sum())
        return Correction(readonly_volume(region), outcome, drawn, requested, inside / size if size else 0.0,
                          (int(region.sum()) - inside) / size if size else 0.0, size, attempts, variant, reason)

    if not size:
        return result(empty, NO_CHANGE, reason=REASON_NO_TARGET)
    if drawn == NO_CHANGE:
        return result(empty, NO_CHANGE)
    if drawn == COMPLETE:
        return result(target, COMPLETE)
    now = _Round(target, drawn_on & target, roles.correct_tissue, uptake, np.asarray(spacing_zyx, dtype=np.float64),
                 size)
    builders = {PARTIAL_NEAR: _partial_near, PARTIAL_SPLIT: _partial_split, SPILL_UPTAKE: _spill_uptake,
                SPILL_BOUNDARY: _spill_boundary,
                EXTRA_REGION: lambda *arguments: _extra_region(*arguments, addition=sign == 1)}
    reason = REASON_ATTEMPTS
    for attempt in range(1, attempts_allowed + 1):
        try:
            region, requested, variant = builders[drawn](now, config, rng)
        except _Infeasible as failure:
            reason = str(failure)
            break
        except _Missed:
            continue
        outside = region & ~target
        partial = drawn in (PARTIAL_NEAR, PARTIAL_SPLIT)
        if (outside & ~roles.correct_tissue).any() or (partial and (outside.any() or not (target & ~region).any())) \
                or (not partial and ((target & ~region).any() or not outside.any())):
            raise StateError(f"the {drawn} outcome broke its contract: region={int(region.sum())} target={size} "
                             f"outside_target={int(outside.sum())} "
                             f"outside_correct_tissue={int((outside & ~roles.correct_tissue).sum())}")
        return result(region, drawn, requested, attempt, variant)
    return result(target, COMPLETE, attempts=attempts_allowed if reason == REASON_ATTEMPTS else 1, reason=reason)


# ------------------------------------------------------------------------------------------- rollouts and states
def auxiliary_rollout(*, initial_segmentation, ground_truth, pet, simulator, style: str, config: Mapping[str, Any],
                      rng: np.random.Generator, spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0)
                      ) -> tuple[list[np.ndarray], list, list, list[dict]]:
    """One five-round interaction of the auxiliary model -> (masks, scribbles, correction types, log).

    ``masks[r]`` is the mask before round r (``masks[0]`` the initial segmentation, six masks in all);
    ``scribbles[r]`` and ``correction_types[r]`` (+1/-1) are the simulated user's scribble on it, None once no error
    is left. ``simulator.draw(mask, ground_truth, style)`` is
    :meth:`sirb.interaction.scribble_simulator.ScribbleSimulator.draw`.
    """
    mask = readonly_volume(binary_volume(initial_segmentation, "initial segmentation").copy())
    truth = binary_volume(ground_truth, "ground truth")
    masks, scribbles, types, log = [mask], [], [], []
    for round_index in range(CORRECTION_ROUNDS):
        drawn = simulator.draw(mask, truth, style)
        if not drawn.valid:
            scribbles.append(None)
            types.append(None)
            masks.append(mask)
            log.append({"event": "no_error_left", "round_index": round_index, "style": style})
            continue
        sign = int(correction_sign(drawn.correction_type))
        scribble = readonly_volume(np.asarray(drawn.scribble_zyx, dtype=bool))
        correction = imperfect_correction(mask, truth, scribble, sign, pet, config=config, rng=rng,
                                          spacing_zyx=spacing_zyx)
        mask = readonly_volume(apply(mask, correction.region, sign))
        scribbles.append(scribble)
        types.append(sign)
        masks.append(mask)
        log.append({"event": "auxiliary_correction", "round_index": round_index, "style": style,
                    "correction_type": correction_name(sign), **correction.to_log()})
    return masks, scribbles, types, log


def generate_auxiliary_model_states(*, examination_id: str, patient_id: str, split: str, initial_segmentation,
                                    ground_truth, pet, simulator, model_config: Mapping[str, Any],
                                    styles: Sequence[str], spacing_zyx: Sequence[float] = (1.0, 1.0, 1.0),
                                    splits: Sequence[str] = ("train",)) -> tuple[list[ScribbleState], list[dict]]:
    """The auxiliary-model states of one examination and a log.

    ``model_config`` is ``data.auxiliary_model``: ``seed``, ``trajectories_per_examination`` (trajectory t draws in
    style ``styles[t % len(styles)]``) and the outcome settings of :func:`imperfect_correction`. Every trajectory is
    one :func:`auxiliary_rollout`; its masks before rounds 2 to 5 become states (the first round's state is the
    initial segmentation, which the ground-truth-corrected states hold). Randomness:
    ``default_rng([seed, stable_hash("<examination>.auxiliary_model"), trajectory])``, so the same configuration
    gives the same states. Ids: ``<examination>.auxiliary-seed<seed>.traj<t>.round<r>``.
    """
    if split not in tuple(splits):
        raise StateError(f"auxiliary-model states: split actual={split!r} allowed={tuple(splits)}")
    styles = tuple(styles)
    unknown = [style for style in styles if style not in DRAWING_STYLES]
    if not styles or unknown:
        raise StateError(f"styles actual={styles} registered={DRAWING_STYLES}")
    seed = int(model_config["seed"])
    if seed < 0:
        raise StateError(f"seed must be a non-negative integer, got {model_config['seed']!r}")
    trajectories = configured_positive_integer(model_config, "trajectories_per_examination")
    truth = binary_volume(ground_truth, "ground truth")
    states: list[ScribbleState] = []
    log: list[dict] = []
    for trajectory in range(trajectories):
        rng = np.random.default_rng([seed, stable_hash(f"{examination_id}.{SOURCE_AUXILIARY_MODEL}"), trajectory])
        style = styles[trajectory % len(styles)]
        masks, scribbles, types, rollout_log = auxiliary_rollout(
            initial_segmentation=initial_segmentation, ground_truth=truth, pet=pet, simulator=simulator, style=style,
            config=model_config, rng=rng, spacing_zyx=spacing_zyx)
        found, state_log = states_from_rollout(
            examination_id=examination_id, patient_id=patient_id, split=split, ground_truth=truth, masks=masks,
            scribbles=scribbles, correction_types=types, style=style, trajectory_index=trajectory,
            run_tag=f"auxiliary-seed{seed}", splits=splits, state_source=SOURCE_AUXILIARY_MODEL)
        states.extend(found)
        log.extend(dict(row, examination_id=examination_id, trajectory_index=trajectory) for row in rollout_log)
        log.extend(state_log)
    return states, log
