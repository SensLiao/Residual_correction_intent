"""The losses of one training view, and the source targets they need.

A view is one state of a state pair or scribble pair in one training window (``sirb.data.pairs.TrainingView``). Its
supervised voxels U are the editable valid native voxels inside the window, labelled target error, other errors or
correct tissue (``sirb.data.roles``). Three terms:

* native CE: the cross-entropy of the native role logits over U, the mean over the roles present in U of each role's
  mean negative log-likelihood (NLL). Correct tissue is itself the mean of its present strata: the shell (the correct
  tissue 26-adjacent to the target error on the complete native grid) and the rest.
* Dice loss: ``1 - (2 sum_U p t + eps) / (sum_U p + |T| + eps)`` with ``p`` the target probability, ``t`` the
  target-error indicator and ``|T|`` the target-error voxels in U; there is no Dice term when ``|T| = 0``.
* source CE, the cross-entropy of source-conditioned role prediction at 12 mm and 6 mm. For source ``j`` the target
  error ``T_j`` is the union of the error components its scribble voxels touch and its other errors are the remaining
  errors; correct tissue is shared by all sources. A source's term is the mean over its present roles, ``T_j`` being
  the mean of its near and far strata (``T_j`` split into equal halves by the distance to the whole scribble on the
  complete native grid, the odd voxel near). Each stratum is sampled from U uniformly without replacement: at most K/2
  points of each ``T_j`` stratum and at most K points of the other errors and of the correct tissue, K the number of
  feature cells of the level in one window. A level's term is the mean over the valid sources; the source CE is the
  mean of the two levels.

The view total is ``w_native * native CE + w_source * source CE + w_dice * Dice loss`` (:func:`view_total`).

With merged non-target roles (a variant of the paper) the NLL of other errors and of correct tissue both become
``-log(p_other + p_correct)`` in the native CE and the source CE; the bins, strata and weights do not change.

The native CE and Dice loss are reduced over the whole view from per-chunk sums (:func:`native_statistics`,
:func:`native_ce_and_dice`), so the voxel chunk changes memory use only. Labels are read only here; nothing in this
module is a model input. :func:`source_targets` is the ``view_targets`` hook of ``sirb.data.pairs.prepare_unit``: it
runs in the unit workers on the full roles of each state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from sirb.data.geometry import MAX_SOURCES, index_xyz_to_world, sample_world_features, world_to_index_xyz
from sirb.data.roles import CORRECT_TISSUE, OTHER_ERRORS, ROLE_COUNT, TARGET_ERROR
from sirb.modeling.role_decoder import LEVELS

__all__ = ["LOSS_WEIGHT_KEYS", "NATIVE_BINS", "SAMPLING_STRATA", "SHELL_BIN", "SourceLabels", "SourceTargets",
           "correct_tissue_shell", "native_ce_and_dice", "native_statistics", "role_log_likelihoods", "source_ce",
           "source_labels_for_points", "source_targets", "split_near_far", "touched_components", "view_total"]

#: Native CE bins: target error, other errors, correct tissue (rest), correct tissue (shell).
NATIVE_BINS = 4
SHELL_BIN = 3
#: The correct-tissue shell: correct tissue 26-adjacent to the target error.
SHELL_STRUCTURE = ndimage.generate_binary_structure(3, 3)
#: The sampled strata of one source, in the order they are drawn from the view's generator.
SAMPLING_STRATA = ("target_near", "target_far", "other_errors", "correct_tissue")
#: The weights of the view total (``train.loss`` of a configuration).
LOSS_WEIGHT_KEYS = ("native_ce", "source_ce", "dice")

NONFINITE_SOURCE_LOGITS = "non-finite source role logits"


# ------------------------------------------------------------------------------------------- likelihoods
def role_log_likelihoods(logits: torch.Tensor, *, merged_non_target_roles: bool) -> torch.Tensor:
    """Log-likelihood of each role ``[..., 3]`` from role logits.

    With ``merged_non_target_roles`` the other-errors and correct-tissue columns both hold ``log(p_other +
    p_correct)``; these columns are lookups of the NLL of each label, not a normalised distribution. The target column
    is the ordinary log-softmax either way.
    """
    if not merged_non_target_roles:
        return F.log_softmax(logits, dim=-1)
    log_probability = F.log_softmax(logits.float(), dim=-1)
    non_target = torch.logsumexp(log_probability[..., OTHER_ERRORS:], dim=-1)
    return torch.stack((log_probability[..., TARGET_ERROR], non_target, non_target), dim=-1)


# ------------------------------------------------------------------------------------------- native CE, Dice loss
def native_statistics(log_likelihoods: torch.Tensor, labels: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """The sums of one chunk of supervised voxels: the NLL sum of each of the :data:`NATIVE_BINS` bins, the sum of the
    target probability and its sum over the target-error voxels (``[NATIVE_BINS + 2]``)."""
    nll = -log_likelihoods.gather(1, labels[:, None]).squeeze(1)
    sums = log_likelihoods.new_zeros(NATIVE_BINS).scatter_add(0, bins, nll)
    target_probability = log_likelihoods[:, TARGET_ERROR].exp()
    target = labels == TARGET_ERROR
    return torch.cat((sums, torch.stack((target_probability.sum(), (target_probability * target).sum()))))


def native_ce_and_dice(totals: torch.Tensor, counts, *, epsilon: float) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(native CE, Dice loss or ``None`` without target-error voxels) of a view from the sums of all its chunks
    (:func:`native_statistics`) and its voxel count per bin."""
    counts = torch.as_tensor(counts, device=totals.device, dtype=totals.dtype)
    present = counts > 0
    means = totals[:NATIVE_BINS] / counts.clamp(min=1)
    tissue_present = present[CORRECT_TISSUE:].to(totals.dtype)
    tissue = (means[CORRECT_TISSUE:] * tissue_present).sum() / tissue_present.sum().clamp(min=1)
    roles = torch.stack((means[TARGET_ERROR], means[OTHER_ERRORS], tissue))
    role_present = torch.stack((present[TARGET_ERROR], present[OTHER_ERRORS],
                                present[CORRECT_TISSUE:].any())).to(totals.dtype)
    native_ce = (roles * role_present).sum() / role_present.sum().clamp(min=1)
    if not bool(present[TARGET_ERROR]):
        return native_ce, None
    dice = 1 - (2 * totals[NATIVE_BINS + 1] + epsilon) / (totals[NATIVE_BINS] + counts[TARGET_ERROR] + epsilon)
    return native_ce, dice


def view_total(native_ce: torch.Tensor, source_ce: torch.Tensor | None, dice: torch.Tensor | None, *,
               weights: Mapping[str, float]) -> torch.Tensor:
    """``w_native * native CE + w_source * source CE + w_dice * Dice loss``; a ``None`` term is left out (no source CE
    without scribble conditioning, no Dice loss without target-error voxels)."""
    total = weights["native_ce"] * native_ce
    if source_ce is not None:
        total = total + weights["source_ce"] * source_ce
    if dice is not None:
        total = total + weights["dice"] * dice
    return total


# ------------------------------------------------------------------------------------------- source targets
@dataclass(frozen=True)
class SourceTargets:
    """The target error of every source of one (augmented) state, on its complete native grid.

    ``target_flat`` holds the sorted flat native indices of the whole scribble's target error T. Bit ``j`` of
    ``target_bits[i]`` is set when that voxel lies in ``T_j`` and bit ``j`` of ``far_bits[i]`` when it lies in the far
    stratum of ``T_j``. ``shell_flat`` holds the sorted flat indices of the correct-tissue shell.
    """
    target_flat: np.ndarray
    target_bits: np.ndarray
    far_bits: np.ndarray
    source_valid: np.ndarray
    shape_zyx: tuple
    shell_flat: np.ndarray

    def _lookup(self, values: np.ndarray, flat_index) -> np.ndarray:
        flat_index = np.asarray(flat_index, dtype=np.int64)
        if not len(self.target_flat):
            return np.zeros(len(flat_index), dtype=np.int64)
        position = np.minimum(np.searchsorted(self.target_flat, flat_index), len(self.target_flat) - 1)
        hit = self.target_flat[position] == flat_index
        return np.where(hit, values[position], 0)

    def bits_at(self, flat_index) -> np.ndarray:
        """``T_j`` bit masks at flat native indices (0 outside T)."""
        return self._lookup(self.target_bits, flat_index)

    def far_bits_at(self, flat_index) -> np.ndarray:
        """Far-stratum bit masks at flat native indices (0 outside T and in the near strata)."""
        return self._lookup(self.far_bits, flat_index)

    def shell_at(self, flat_index) -> np.ndarray:
        """Whether each flat native index lies in the correct-tissue shell."""
        flat_index = np.asarray(flat_index, dtype=np.int64)
        if not len(self.shell_flat):
            return np.zeros(len(flat_index), dtype=bool)
        position = np.minimum(np.searchsorted(self.shell_flat, flat_index), len(self.shell_flat) - 1)
        return self.shell_flat[position] == flat_index

    def _volume(self, flat: np.ndarray) -> np.ndarray:
        mask = np.zeros(int(np.prod(self.shape_zyx)), dtype=bool)
        mask[flat] = True
        return mask.reshape(self.shape_zyx)

    def target_mask(self, j: int) -> np.ndarray:
        """``T_j`` as a full-volume boolean array."""
        return self._volume(self.target_flat[(self.target_bits >> j) & 1 == 1])

    def far_mask(self, j: int) -> np.ndarray:
        """The far stratum of ``T_j`` as a full-volume boolean array."""
        return self._volume(self.target_flat[((self.target_bits >> j) & 1 == 1) & ((self.far_bits >> j) & 1 == 1)])

    def shell_mask(self) -> np.ndarray:
        """The correct-tissue shell as a full-volume boolean array."""
        return self._volume(self.shell_flat)


def touched_components(roles, sources, affine_xyz) -> list[np.ndarray]:
    """For each of the :data:`MAX_SOURCES` source slots, the error components its scribble voxels touch (empty for an
    invalid slot)."""
    shape = np.asarray(roles.components.shape)
    index = np.rint(world_to_index_xyz(sources.voxel_world_xyz, affine_xyz)).astype(np.int64)[:, ::-1]
    if len(index) and (np.any(index < 0) or np.any(index >= shape)):
        raise ValueError("scribble voxels of the sources lie outside the native array of these roles")
    components = roles.components[tuple(index.T)] if len(index) else np.zeros(0, dtype=np.int64)
    touched = []
    for j in range(MAX_SOURCES):
        if not sources.source_valid[j]:
            touched.append(np.zeros(0, dtype=np.int64))
            continue
        labels = np.unique(components[sources.voxel_source == j])
        touched.append(labels[labels != 0])
    return touched


def split_near_far(distance_mm) -> tuple[np.ndarray, np.ndarray]:
    """(near, far) positions of equal halves by distance, stable order, the odd extra position near."""
    order = np.argsort(np.asarray(distance_mm), kind="stable")
    split = (len(order) + 1) // 2
    return order[:split], order[split:]


def correct_tissue_shell(roles) -> np.ndarray:
    """The correct tissue 26-adjacent to the target error, on the complete native grid (empty without target)."""
    return roles.correct_tissue & ndimage.binary_dilation(roles.target_error, structure=SHELL_STRUCTURE)


def source_targets(roles, sources, state) -> SourceTargets:
    """``T_j``, its near and far strata and the correct-tissue shell of every valid source of ``state``.

    ``roles`` are the full roles of ``state`` (``sirb.data.roles.Roles``) and ``sources`` its scribble's sources
    (``sirb.modeling.source_encoder.ScribbleSources``).
    """
    if tuple(roles.target_error.shape) != tuple(state.shape_zyx):
        raise ValueError(f"roles shape {roles.target_error.shape} differs from the state {state.shape_zyx}")
    touched = touched_components(roles, sources, state.affine_xyz)
    target_index = np.argwhere(roles.target_error)                    # ZYX lexicographic, the flat order
    flat = (np.ravel_multi_index(tuple(target_index.T), roles.target_error.shape).astype(np.int64)
            if len(target_index) else np.zeros(0, dtype=np.int64))
    components = roles.components[tuple(target_index.T)] if len(target_index) else np.zeros(0, dtype=np.int64)
    bits = np.zeros(len(flat), dtype=np.int64)
    for j, labels in enumerate(touched):
        if len(labels):
            bits |= np.isin(components, labels).astype(np.int64) << j
    distance = (state.distances_mm(index_xyz_to_world(target_index[:, ::-1], state.affine_xyz))
                if len(target_index) else np.zeros(0))
    far = np.zeros(len(flat), dtype=np.int64)
    splits: dict[bytes, np.ndarray] = {}
    for j in range(MAX_SOURCES):
        in_target = ((bits >> j) & 1) == 1
        if not sources.source_valid[j] or not in_target.any():
            continue
        key = in_target.tobytes()
        if key not in splits:                                        # the same T_j gives the same strata
            positions = np.flatnonzero(in_target)
            splits[key] = positions[split_near_far(distance[positions])[1]]
        far[splits[key]] |= np.int64(1) << j
    shell = np.flatnonzero(correct_tissue_shell(roles).reshape(-1)).astype(np.int64)
    return SourceTargets(flat, bits, far, np.asarray(sources.source_valid, dtype=bool).copy(),
                         tuple(int(v) for v in roles.target_error.shape), shell)


# ------------------------------------------------------------------------------------------- source CE
@dataclass(frozen=True)
class SourceLabels:
    """The labels of the source CE at the supervised voxels U of one view (CPU arrays)."""
    world_xyz: np.ndarray       # [n, 3] float64 voxel centres
    labels: np.ndarray          # [n] role of each voxel for the whole scribble
    target_bits: np.ndarray     # [n] bit j: the voxel lies in T_j
    far_bits: np.ndarray        # [n] bit j: the voxel lies in the far stratum of T_j
    source_valid: np.ndarray    # [MAX_SOURCES]
    shell: np.ndarray           # [n] the voxel lies in the correct-tissue shell


def source_labels_for_points(state, index_zyx, labels, targets: SourceTargets) -> SourceLabels:
    """The :class:`SourceLabels` of native voxels ``index_zyx`` of ``state`` with role ``labels``."""
    index = np.asarray(index_zyx, dtype=np.int64).reshape(-1, 3)
    world = index_xyz_to_world(index[:, ::-1], state.affine_xyz) if len(index) else np.zeros((0, 3))
    flat = (np.ravel_multi_index(tuple(index.T), state.shape_zyx).astype(np.int64) if len(index)
            else np.zeros(0, dtype=np.int64))
    if tuple(targets.shape_zyx) != tuple(state.shape_zyx):
        raise ValueError("the source targets belong to another native grid")
    bits, far, shell = targets.bits_at(flat), targets.far_bits_at(flat), targets.shell_at(flat)
    roles = np.asarray(labels, dtype=np.int64)
    if np.any(bits[roles != TARGET_ERROR]):
        raise ValueError("source target bits found outside the target error of the whole scribble")
    if np.any(shell & (roles != CORRECT_TISSUE)):
        raise ValueError("correct-tissue shell voxels found outside the correct tissue")
    return SourceLabels(world, roles, bits, far, np.asarray(targets.source_valid, dtype=bool), shell)


def _draw(rng: np.random.Generator, positions: np.ndarray, cap: int) -> np.ndarray:
    if len(positions) <= cap:
        return np.asarray(positions, dtype=np.int64)
    chosen = rng.choice(len(positions), size=cap, replace=False)
    return np.sort(np.asarray(positions, dtype=np.int64)[chosen])


def source_ce(role_logits: Mapping[str, torch.Tensor], role_affines: Mapping[str, torch.Tensor], labels: SourceLabels,
              *, rng: np.random.Generator, support_affine_xyz, support_shape_zyx, support_valid_zyx=None,
              merged_non_target_roles: bool, monitor: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
    """(source CE, monitoring readings) of one view (module docstring).

    ``role_logits[level]`` are the source role logits ``[1, 16, 3, Z, Y, X]`` of :data:`LEVELS` with their affines
    ``role_affines[level]``; the window (``support_*``) bounds the sampled voxels. Sources whose ``T_j`` are equal
    share one draw.
    """
    valid_sources = [j for j in range(MAX_SOURCES) if labels.source_valid[j]]
    if not valid_sources:
        raise ValueError("the source CE needs at least one valid source (an empty scribble changes nothing)")
    readings: dict[str, Any] = {}
    level_terms = []
    is_target = labels.labels == TARGET_ERROR
    is_error = is_target | (labels.labels == OTHER_ERRORS)
    is_tissue = labels.labels == CORRECT_TISSUE
    target_positions = np.flatnonzero(is_target)
    patterns: dict[bytes, list[int]] = {}
    for j in valid_sources:
        in_target = ((labels.target_bits[target_positions] >> j) & 1).astype(np.uint8)
        far = ((labels.far_bits[target_positions] >> j) & 1).astype(np.uint8) * in_target
        patterns.setdefault(np.stack((in_target, far)).tobytes(), []).append(j)
    for level in LEVELS:
        logits = role_logits[level]
        if logits.ndim != 6 or tuple(logits.shape[:3]) != (1, MAX_SOURCES, ROLE_COUNT):
            raise ValueError(f"{level} source role logits actual={tuple(logits.shape)}, expected [1,16,3,Z,Y,X]")
        affine = role_affines[level]
        affine = affine[0] if affine.ndim == 3 else affine
        cap = int(np.prod(logits.shape[3:]))                            # feature cells of the level in one window
        terms: dict[int, torch.Tensor] = {}
        level_reading: dict[str, Any] = {"cap": cap, "patterns": len(patterns), "sampled": {}}
        for key, sources in patterns.items():
            columns = np.frombuffer(key, dtype=np.uint8).reshape(2, -1).astype(bool)
            source_target = target_positions[columns[0]]
            far_positions = target_positions[columns[0] & columns[1]]
            near_positions = target_positions[columns[0] & ~columns[1]]
            other = is_error.copy()
            other[source_target] = False                                  # target voxels outside T_j are other errors
            draws = {"target_near": _draw(rng, near_positions, cap // 2),
                     "target_far": _draw(rng, far_positions, cap // 2),
                     "other_errors": _draw(rng, np.flatnonzero(other), cap),
                     "correct_tissue": _draw(rng, np.flatnonzero(is_tissue), cap)}
            points = np.concatenate([draws[name] for name in SAMPLING_STRATA])
            if not len(points):
                raise ValueError("a view with supervised voxels gave no source CE sample")
            world = torch.as_tensor(labels.world_xyz[points], dtype=torch.float64, device=logits.device)
            fields = logits[:, sources].reshape(1, len(sources) * ROLE_COUNT, *logits.shape[3:])
            sampled, inside = sample_world_features(fields, affine, world, support_affine_xyz=support_affine_xyz,
                                                    support_shape_zyx=support_shape_zyx,
                                                    support_valid_zyx=support_valid_zyx)
            if not bool(inside.all()):
                raise ValueError(f"{level} source CE sample outside the window")
            sampled = sampled[0].reshape(len(points), len(sources), ROLE_COUNT).float()
            if not bool(torch.isfinite(sampled).all()):
                raise FloatingPointError(NONFINITE_SOURCE_LOGITS)
            bounds = np.cumsum([0] + [len(draws[name]) for name in SAMPLING_STRATA])
            segments = {name: slice(int(bounds[i]), int(bounds[i + 1])) for i, name in enumerate(SAMPLING_STRATA)}
            likelihoods = role_log_likelihoods(sampled, merged_non_target_roles=merged_non_target_roles)
            for local, j in enumerate(sources):
                strata = [-(likelihoods[segments[name], local, TARGET_ERROR]).mean()
                          for name in ("target_near", "target_far") if len(draws[name])]
                role_means = []
                if strata:
                    role_means.append(torch.stack(strata).mean())
                if len(draws["other_errors"]):
                    role_means.append(-(likelihoods[segments["other_errors"], local, OTHER_ERRORS]).mean())
                if len(draws["correct_tissue"]):
                    role_means.append(-(likelihoods[segments["correct_tissue"], local, CORRECT_TISSUE]).mean())
                terms[j] = torch.stack(role_means).mean()
            if monitor:
                level_reading["sampled"][str(sources[0])] = {name: int(len(draws[name])) for name in SAMPLING_STRATA}
        level_term = torch.stack([terms[j] for j in valid_sources]).mean()
        level_terms.append(level_term)
        if monitor:
            level_reading["ce"] = level_term.detach()
            readings[level] = level_reading
    return torch.stack(level_terms).mean(), readings
