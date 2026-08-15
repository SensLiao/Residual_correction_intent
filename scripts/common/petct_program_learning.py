#!/usr/bin/env python3
"""v3 program learning support: label-free dataset, matched-state losses.

This module adds the v3 learning surface next to the frozen v2
``petct_learning.py`` (never modifies it):

  * ``InferenceEpisodeDataset`` — a label-free visible-lane dataset.  It
    derives operation authority from the signed cue channels and refuses
    rows whose manifest disagrees; it never opens an evaluation bundle.
  * ``ProgramEpisodeDataset`` — the 13-channel editor dataset used by the
    J6/J7/J8/J9 representation ladder.
  * ``GroupedBatchSampler`` — batches are built from whole matched-state
    groups so the same-operation margin loss sees its siblings.
  * ``matched_family_margin_loss`` — SCEP 5.7 margin over same-operation
    matched pairs.
  * ``multi_positive_pointer_loss`` — SCEP 5.6 multi-positive ADD_SAME
    pointer objective.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from common.petct_program_contract import (
    GOAL_TO_FAMILY,
    NEW_CUE_SENTINEL,
    ProgramContractError,
    family_ids,
    operation_from_cue_sign,
)

ADD_FAMILY_INDEX = {family: index for index, family in enumerate(family_ids("ADD"))}
REMOVE_FAMILY_INDEX = {family: index for index, family in enumerate(family_ids("REMOVE"))}


class LearningContractError(ValueError):
    """Mirrors the frozen v2 error type for v3 fail-closed checks."""


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise LearningContractError("missing manifest: %s" % path)
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise LearningContractError(
                    "manifest line %d is not JSON: %s" % (line_number, error)
                ) from error
    return rows


def _load_visible_bundle(row: Mapping[str, Any], episode_id: str) -> Dict[str, np.ndarray]:
    visible_path = Path(row["visible_npz"])
    if not visible_path.is_file():
        raise LearningContractError("missing visible_npz: %s" % visible_path)
    if _sha256_file(visible_path) != row.get("visible_sha256"):
        raise LearningContractError("visible_npz hash mismatch: %s" % episode_id)
    with np.load(str(visible_path), allow_pickle=False) as bundle:
        required = {"visual", "m0", "scribble", "cue_fg", "cue_bg", "spacing_xy"}
        if set(bundle.files) != required:
            raise LearningContractError("visible tensor bundle schema mismatch")
        return {name: np.asarray(bundle[name], dtype=np.float32) for name in required}


def operation_id_from_row(row: Mapping[str, Any], cue_fg: np.ndarray, cue_bg: np.ndarray) -> Tuple[str, int]:
    """Derive operation authority from cue channels and cross-check the row."""

    operation = operation_from_cue_sign(bool(cue_fg.any()), bool(cue_bg.any()))
    row_operation = row.get("operation")
    if row_operation is not None and str(row_operation) != operation:
        raise LearningContractError(
            "manifest operation disagrees with signed cue for episode %s"
            % row.get("episode_id")
        )
    return operation, 0 if operation == "ADD" else 1


def goal_to_family_id(goal: str, operation: str) -> int:
    family = GOAL_TO_FAMILY[goal]
    return (ADD_FAMILY_INDEX if operation == "ADD" else REMOVE_FAMILY_INDEX)[family]


def _load_components(
    candidates: Optional[Mapping[str, Any]], episode_id: str, cache: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load [K,D] descriptor vectors, [K] validity, and [K,H,W] central masks."""

    if episode_id in cache:
        return cache[episode_id]
    if candidates is None:
        raise LearningContractError("component candidates sidecar is required")
    record = candidates.get(episode_id)
    if record is None:
        raise LearningContractError("missing component candidates for %s" % episode_id)
    descriptor_names = (
        "log_volume", "z_span", "bbox_span_z", "centroid_x_mm",
        "centroid_y_mm", "cue_overlap_voxels", "distance_from_cue_mm",
    )
    rows = []
    for component in record["components"]:
        rows.append([float(component[name]) for name in descriptor_names])
    vectors = np.asarray(rows, dtype=np.float32)
    masks = None
    if record.get("central_masks_available"):
        masks = np.stack(
            [np.asarray(c["prompted_slice_mask"], dtype=np.uint8) for c in record["components"]],
            axis=0,
        )
    result = (vectors, np.ones(len(vectors), dtype=bool), masks)
    cache[episode_id] = result
    return result


class InferenceEpisodeDataset(Dataset):
    """Label-free visible-lane dataset for compiler inference/diagnostics.

    Never opens an evaluation bundle; the label/evaluator lane is joined by
    the evaluator via content hash after predictions exist.
    """

    def __init__(
        self,
        manifest: Path,
        partition: str,
        candidates: Optional[Mapping[str, Any]] = None,
        family_gold: Optional[Mapping[str, str]] = None,
    ):
        rows = load_jsonl(manifest)
        self.rows = [row for row in rows if row.get("partition") == partition]
        if not self.rows:
            raise LearningContractError("partition %s is empty" % partition)
        self.candidates = candidates
        # family_gold is the label-only join used ONLY at training time; the
        # inference path never supplies it and therefore never reads goals.
        self.family_gold = family_gold
        self._component_cache: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        episode_id = str(row["episode_id"])
        bundle = _load_visible_bundle(row, episode_id)
        visual = bundle["visual"]
        operation, operation_id = operation_id_from_row(
            row, bundle["cue_fg"], bundle["cue_bg"]
        )
        if visual.shape != (17, bundle["m0"].shape[0], bundle["m0"].shape[1]):
            raise LearningContractError("visible arrays have inconsistent shapes")
        vectors, valid, _ = _load_components(
            self.candidates, episode_id, self._component_cache
        )
        output: Dict[str, Any] = {
            "episode_id": episode_id,
            "patient_id": str(row["patient_id"]),
            "group_id": str(row.get("matched_state_group_id") or episode_id),
            "operation": operation,
            "operation_id": torch.tensor(operation_id, dtype=torch.long),
            "visual": torch.from_numpy(visual),
            "spacing_xy": torch.from_numpy(bundle["spacing_xy"]),
            "component_vectors": torch.from_numpy(vectors),
            "component_mask": torch.from_numpy(valid),
        }
        if self.family_gold is not None:
            goal = self.family_gold.get(episode_id)
            if goal is None:
                raise LearningContractError("missing family gold for %s" % episode_id)
            output["goal"] = str(goal)
            output["family_gold"] = torch.tensor(
                goal_to_family_id(str(goal), operation), dtype=torch.long
            )
        return output


class ProgramEpisodeDataset(Dataset):
    """13-channel editor dataset for the J6-J9 representation ladder.

    ``editor_condition`` selects the ladder rung:
      * ``spatial_only``   — 12-channel input, NULL program conditioning;
      * ``flat_action``    — 13-channel input, gold flat call (family only);
      * ``typed_program``  — 13-channel input, typed call (family+operand);
      * ``continuous``     — 13-channel input, continuous state readout;
      * ``null_program``   — 13-channel input, exact NULL conditioning.

    ``call_source`` is ``gold`` or ``predicted``; predicted calls come from a
    frozen inference artifact.  NEW_CUE episodes always carry an all-zero
    selected-component channel (bitwise zero, never a GT map).
    """

    def __init__(
        self,
        manifest: Path,
        partition: str,
        candidates: Optional[Mapping[str, Any]] = None,
        pointer_targets: Optional[Mapping[str, List[int]]] = None,
        predicted_calls: Optional[Mapping[str, Mapping[str, Any]]] = None,
        *,
        editor_condition: str = "spatial_only",
        call_source: str = "gold",
        program_dropout: float = 0.0,
        seed: int = 3407,
        load_evaluation: bool = True,
    ):
        if editor_condition not in (
            "spatial_only", "flat_action", "typed_program", "continuous", "null_program",
        ):
            raise LearningContractError("unknown editor condition: %s" % editor_condition)
        if call_source not in ("gold", "predicted"):
            raise LearningContractError("call_source must be gold or predicted")
        if not 0.0 <= float(program_dropout) < 1.0:
            raise LearningContractError("program_dropout must be in [0,1)")
        rows = load_jsonl(manifest)
        self.rows = [row for row in rows if row.get("partition") == partition]
        if not self.rows:
            raise LearningContractError("partition %s is empty" % partition)
        self.candidates = candidates
        self.pointer_targets = pointer_targets or {}
        self.predicted_calls = predicted_calls or {}
        self.editor_condition = editor_condition
        self.call_source = call_source
        self.program_dropout = float(program_dropout)
        self.load_evaluation = bool(load_evaluation)
        self.rng = np.random.default_rng(seed)
        self._component_cache: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        episode_id = str(row["episode_id"])
        bundle = _load_visible_bundle(row, episode_id)
        visual = bundle["visual"]
        operation, operation_id = operation_id_from_row(
            row, bundle["cue_fg"], bundle["cue_bg"]
        )
        goal = str(row["goal"])
        gold_family_id = goal_to_family_id(goal, operation)
        gold_operand_key: Optional[str] = None
        if operation == "ADD" and goal == "ADD_NEW_COMPLETE":
            gold_operand_key = NEW_CUE_SENTINEL
        vectors, valid, masks = _load_components(
            self.candidates, episode_id, self._component_cache
        )
        selected_channel = np.zeros_like(bundle["m0"])
        if operation == "REMOVE":
            # Deterministic cue-hit binding supplies the operand.
            cue_overlaps = vectors[:, 5]
            candidate_indices = np.flatnonzero(valid)
            if len(candidate_indices):
                hit = candidate_indices[int(np.argmax(cue_overlaps[candidate_indices]))]
                gold_operand_key = f"component_{hit}"
                if masks is not None:
                    selected_channel = masks[hit]
        elif operation == "ADD" and goal != "ADD_NEW_COMPLETE":
            targets = self.pointer_targets.get(episode_id, [])
            if not targets:
                raise LearningContractError("missing pointer targets for %s" % episode_id)
            first = int(targets[0])
            gold_operand_key = f"component_{first}"
            if masks is not None:
                selected_channel = masks[first]
        signed_cue = visual[15:16] - visual[16:17]
        editor_visual = np.concatenate([visual[:10], visual[12:13], signed_cue], axis=0)
        if self.editor_condition in ("flat_action", "typed_program", "continuous", "null_program"):
            editor_visual = np.concatenate([editor_visual, selected_channel[None]], axis=0)
        null_condition = self.editor_condition in ("spatial_only", "null_program")
        family_id = gold_family_id
        operand_mode = 0 if gold_operand_key != NEW_CUE_SENTINEL else 1
        if self.editor_condition == "typed_program":
            if self.call_source == "predicted":
                call = self.predicted_calls.get(episode_id)
                if call is None:
                    raise LearningContractError("missing predicted call for %s" % episode_id)
                family_id = goal_to_family_id(str(call["goal"]), str(call["operation"]))
                operand_mode = 0 if call.get("operand") != NEW_CUE_SENTINEL else 1
        elif self.editor_condition == "flat_action":
            operand_mode = 0
        else:
            family_id = -1
            operand_mode = 2
        if self.program_dropout and self.rng.random() < self.program_dropout:
            family_id = -1
            operand_mode = 2
        active = 1 if not null_condition else 0
        support_mode = 0 if not null_condition else 1
        output = {
            "episode_id": episode_id,
            "patient_id": str(row["patient_id"]),
            "operation": operation,
            "operation_id": torch.tensor(operation_id, dtype=torch.long),
            "goal": goal,
            "family_gold": torch.tensor(gold_family_id, dtype=torch.long),
            "visual": torch.from_numpy(editor_visual),
            "family_id": torch.tensor(family_id, dtype=torch.long),
            "operand_mode": torch.tensor(operand_mode, dtype=torch.long),
            "support_mode": torch.tensor(support_mode, dtype=torch.long),
            "active": torch.tensor(active, dtype=torch.long),
            "m0": torch.from_numpy(bundle["m0"][None]),
        }
        if self.editor_condition in ("flat_action", "typed_program", "continuous", "null_program"):
            output["selected_component"] = torch.from_numpy(selected_channel[None])
        if self.load_evaluation:
            evaluation_path = Path(row["evaluation_npz"])
            if not evaluation_path.is_file():
                raise LearningContractError(
                    "missing evaluation_npz: %s" % evaluation_path
                )
            if _sha256_file(evaluation_path) != row.get("evaluation_sha256"):
                raise LearningContractError(
                    "evaluation_npz hash mismatch: %s" % episode_id
                )
            with np.load(str(evaluation_path), allow_pickle=False) as bundle_eval:
                required = {"target", "gt", "authorized"}
                if set(bundle_eval.files) != required:
                    raise LearningContractError(
                        "evaluation tensor bundle schema mismatch"
                    )
                target = np.asarray(bundle_eval["target"], dtype=np.float32)
                gt = np.asarray(bundle_eval["gt"], dtype=np.float32)
                authorized = np.asarray(bundle_eval["authorized"], dtype=np.float32)
            output["target"] = torch.from_numpy(target[None])
            output["gt"] = torch.from_numpy(gt[None])
            output["authorized"] = torch.from_numpy(authorized[None])
        return output


class GroupedBatchSampler(Sampler):
    """Build batches from whole matched-state groups (same-operation triplets).

    Rows are grouped by ``matched_state_group_id``; groups are shuffled and
    packed so every batch contains complete groups (last batch may be
    smaller).  The margin loss then always sees sibling rows of the same
    image/cue/operation context.
    """

    def __init__(self, group_ids: Sequence[str], batch_size: int, seed: int):
        if batch_size < 1:
            raise LearningContractError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.seed = seed
        by_group: Dict[str, List[int]] = {}
        for index, group_id in enumerate(group_ids):
            by_group.setdefault(str(group_id), []).append(index)
        self.groups: List[List[int]] = list(by_group.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        order = list(range(len(self.groups)))
        rng.shuffle(order)
        batches: List[List[int]] = []
        current: List[int] = []
        for group_index in order:
            group = self.groups[group_index]
            if current and len(current) + len(group) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(group)
        if current:
            batches.append(current)
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return max(1, int(np.ceil(sum(len(g) for g in self.groups) / self.batch_size)))


def matched_family_margin_loss(
    family_logits: Tensor,
    family_targets: Tensor,
    group_ids: List[str],
    operation_ids: Tensor,
    margin: float,
) -> Tensor:
    """SCEP 5.7 same-operation matched-state margin loss.

    For every pair (i,j) inside one matched-state group the loss asks each
    row to prefer its own family over its sibling's family by ``margin``.
    Pairs across groups or across operations are never compared.
    """

    if margin < 0 or not np.isfinite(margin):
        raise LearningContractError("margin must be finite and non-negative")
    losses: List[Tensor] = []
    group_order: Dict[str, List[int]] = {}
    for index, group_id in enumerate(group_ids):
        group_order.setdefault(str(group_id), []).append(index)
    for indices in group_order.values():
        if len(indices) < 2:
            continue
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                if operation_ids[i] != operation_ids[j]:
                    raise LearningContractError(
                        "matched group mixes operations; same-operation groups only"
                    )
                yi = int(family_targets[i])
                yj = int(family_targets[j])
                pair = (
                    margin
                    - (family_logits[i, yi] - family_logits[i, yj])
                    - (family_logits[j, yj] - family_logits[j, yi])
                )
                losses.append(torch.clamp(pair, min=0.0))
    if not losses:
        return torch.zeros((), dtype=family_logits.dtype, device=family_logits.device)
    return torch.stack(losses).mean()


def multi_positive_pointer_loss(
    pointer_logits: Tensor,
    pointer_mask: Tensor,
    pointer_targets: Sequence[Sequence[int]],
) -> Tensor:
    """SCEP 5.6 multi-positive pointer objective (ADD existing-object rows)."""

    if pointer_logits.shape[0] != len(pointer_targets):
        raise LearningContractError("pointer logits and targets must share batch size")
    losses: List[Tensor] = []
    for index, targets in enumerate(pointer_targets):
        if not targets:
            raise LearningContractError("pointer target set must be non-empty")
        valid = pointer_mask[index].to(dtype=torch.bool)
        if not valid.any():
            raise LearningContractError("no valid pointer candidates for a row")
        if any(int(t) >= pointer_logits.shape[1] or not bool(valid[int(t)]) for t in targets):
            raise LearningContractError("pointer target outside valid candidates")
        log_probs = torch.log_softmax(pointer_logits[index], dim=-1)
        losses.append(-torch.logsumexp(log_probs[[int(t) for t in targets]], dim=-1))
    return torch.stack(losses).mean()
