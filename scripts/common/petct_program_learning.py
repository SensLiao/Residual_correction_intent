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
from torch.utils.data._utils.collate import default_collate

from common.petct_program_contract import (
    GOAL_TO_FAMILY,
    NEW_CUE_SENTINEL,
    ProgramContractError,
    family_to_id,
    family_ids,
    operation_from_cue_sign,
    render_goal,
    validate_legal_call,
)

ADD_FAMILY_INDEX = {family: index for index, family in enumerate(family_ids("ADD"))}
REMOVE_FAMILY_INDEX = {family: index for index, family in enumerate(family_ids("REMOVE"))}

COMPONENT_DESCRIPTOR_NAMES = (
    "log_volume",
    "z_span",
    "prompted_slice_overlap",
    "centroid_dx_mm",
    "centroid_dy_mm",
    "cue_overlap_voxels",
    "distance_from_cue_mm",
)

# This is deliberately an allowlist rather than a denylist.  Adding any
# field to an inference manifest is therefore a reviewed contract change.
# Reviewed addition (R13 mainline, 2026-08-18): the program-manifest
# materializer stamps rows with the R13 identity and binds the visible
# candidate artifacts; all six fields are non-label-derived and pass the
# visible firewall before publication.
INFERENCE_MANIFEST_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "partition",
        "operation",
        "visible_npz",
        "visible_sha256",
        "geometry",
        "center_z",
        "candidate_json",
        "candidate_sha256",
        "dataset_id",
        "source_m0_lineage",
        "round_index",
        "scribble_count",
    }
)
INFERENCE_MANIFEST_REQUIRED_FIELDS = frozenset(
    {"episode_id", "partition", "operation", "visible_npz", "visible_sha256"}
)


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


def goal_to_local_family_id(goal: str, operation: str) -> int:
    """Operation-conditioned 0..2 id consumed by the compiler head."""

    try:
        family = GOAL_TO_FAMILY[goal]
        return (ADD_FAMILY_INDEX if operation == "ADD" else REMOVE_FAMILY_INDEX)[family]
    except (KeyError, ProgramContractError):
        raise LearningContractError(
            "goal %s is not legal for operation %s" % (goal, operation)
        ) from None


# Backwards-compatible name; this module's id is intentionally local.  The
# editor must use ``family_to_id`` (global 0..5) instead.
goal_to_family_id = goal_to_local_family_id


def _validate_inference_row(row: Mapping[str, Any]) -> None:
    keys = set(row)
    missing = INFERENCE_MANIFEST_REQUIRED_FIELDS - keys
    unexpected = keys - INFERENCE_MANIFEST_ALLOWED_FIELDS
    if missing:
        raise LearningContractError(
            "inference manifest row is missing fields: %s" % sorted(missing)
        )
    if unexpected:
        raise LearningContractError(
            "inference manifest contains non-visible fields: %s" % sorted(unexpected)
        )
    if row.get("schema_version") != "PETCT-PROGRAM-INFERENCE-MANIFEST-v1.0":
        raise LearningContractError("unknown inference manifest schema")
    if row.get("partition") not in ("train", "val"):
        raise LearningContractError("inference partition is not train/val")
    if row.get("operation") not in ("ADD", "REMOVE"):
        raise LearningContractError("inference operation is invalid")
    if row.get("partition") == "test":
        raise LearningContractError(
            "generic v3 dataset entrypoints may not consume the locked test partition"
        )


def load_label_manifest(
    path: Path, *, require_matched_groups: bool = True
) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(path)
    labels: Dict[str, Dict[str, Any]] = {}
    matched_groups: Dict[str, List[Tuple[str, str]]] = {}
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if not episode_id or episode_id in labels:
            raise LearningContractError("label manifest has missing/duplicate episode_id")
        if row.get("schema_version") != "PETCT-PROGRAM-LABEL-MANIFEST-v1.0":
            raise LearningContractError("unknown label manifest schema")
        required = {
            "case_id", "patient_id", "partition", "goal", "operation",
            "evaluation_npz", "evaluation_sha256",
            "learning_split_sha256",
        }
        if require_matched_groups:
            required.add("matched_state_group_id")
        if not required.issubset(row):
            raise LearningContractError(
                "label manifest row %s is incomplete" % episode_id
            )
        if row["partition"] not in ("train", "val"):
            raise LearningContractError(
                "generic v3 training labels may contain only train/val rows"
            )
        operation = str(row["operation"])
        goal = str(row["goal"])
        goal_to_local_family_id(goal, operation)
        group_id = str(row.get("matched_state_group_id") or "").strip()
        if require_matched_groups:
            if not group_id:
                raise LearningContractError("label manifest has an empty matched group id")
            matched_groups.setdefault(group_id, []).append((operation, goal))
        labels[episode_id] = dict(row)
    for group_id, entries in matched_groups.items():
        operations = {operation for operation, _ in entries}
        if len(entries) != 3 or len(operations) != 1:
            raise LearningContractError(
                "matched group %s must be one same-operation triplet" % group_id
            )
        operation = next(iter(operations))
        expected_goals = {
            render_goal(operation, family) for family in family_ids(operation)
        }
        if {goal for _, goal in entries} != expected_goals:
            raise LearningContractError(
                "matched group %s lacks the exact operation family triplet" % group_id
            )
    return labels


def validate_training_split(
    labels: Mapping[str, Mapping[str, Any]], learning_split: Path
) -> str:
    """Bind every training label to the independent frozen patient split."""

    if not learning_split.is_file():
        raise LearningContractError("missing learning split: %s" % learning_split)
    try:
        document = json.loads(learning_split.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LearningContractError("invalid learning split: %s" % error) from error
    if (
        document.get("schema_version") != "PETCT-LEARNING-SPLIT-v1.0"
        or document.get("status") != "FROZEN_BEFORE_MODEL_SELECTION"
        or document.get("split_unit") != "patient"
    ):
        raise LearningContractError("learning split is not the frozen patient authority")
    split_sha = _sha256_file(learning_split)
    authority: Dict[str, Tuple[str, frozenset[str]]] = {}
    for entry in document.get("patients", []):
        patient = str(entry.get("patient_id") or "").casefold()
        partition = str(entry.get("partition") or "")
        cases = frozenset(str(value) for value in entry.get("case_ids", []))
        if not patient or partition not in ("train", "val", "test") or not cases:
            raise LearningContractError("learning split contains an invalid patient row")
        if patient in authority:
            raise LearningContractError("learning split contains a duplicate patient")
        authority[patient] = (partition, cases)
    for episode_id, row in labels.items():
        patient = str(row["patient_id"]).casefold()
        case_id = str(row["case_id"])
        partition = str(row["partition"])
        if patient not in authority:
            raise LearningContractError("label patient absent from frozen split: %s" % episode_id)
        expected_partition, cases = authority[patient]
        if case_id not in cases or partition != expected_partition:
            raise LearningContractError("label row disagrees with frozen split: %s" % episode_id)
        if partition not in ("train", "val"):
            raise LearningContractError("training entrypoint refuses partition %s" % partition)
        if str(row["learning_split_sha256"]).lower() != split_sha.lower():
            raise LearningContractError("label row has stale learning split hash")
    return split_sha


def validate_program_manifest_receipt(
    receipt_path: Path,
    inference_manifest: Path,
    label_manifest: Path,
    learning_split: Path,
) -> str:
    """Verify the immutable physical-lane publication before training."""

    if not receipt_path.is_file():
        raise LearningContractError("missing program manifest receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LearningContractError("invalid program manifest receipt") from error
    if (
        receipt.get("schema_version") != "PETCT-PROGRAM-MANIFEST-READY-v1.0"
        or receipt.get("status") != "PASS"
        or receipt.get("locked_test_present") is not False
    ):
        raise LearningContractError("program manifest receipt is not PASS/train-val only")
    expected = {
        "inference": (inference_manifest, _sha256_file(inference_manifest)),
        "labels": (label_manifest, _sha256_file(label_manifest)),
    }
    for role, (path, digest) in expected.items():
        record = receipt.get("outputs", {}).get(role, {})
        if (
            Path(str(record.get("path") or "")).resolve() != path.resolve()
            or str(record.get("sha256") or "").lower() != digest.lower()
        ):
            raise LearningContractError("program receipt does not bind %s manifest" % role)
    split_sha = _sha256_file(learning_split)
    split_record = receipt.get("learning_split", {})
    if (
        Path(str(split_record.get("path") or "")).resolve() != learning_split.resolve()
        or str(split_record.get("sha256") or "").lower() != split_sha.lower()
    ):
        raise LearningContractError("program receipt does not bind frozen learning split")
    return split_sha


def _load_components(
    candidates: Optional[Mapping[str, Any]], episode_id: str, cache: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str], Optional[int]]:
    """Load canonical descriptors, masks, stable keys and cue-hit position."""

    if episode_id in cache:
        return cache[episode_id]
    if candidates is None:
        raise LearningContractError("component candidates sidecar is required")
    record = candidates.get(episode_id)
    if record is None:
        raise LearningContractError("missing component candidates for %s" % episode_id)
    rows = []
    component_keys: List[str] = []
    components = record.get("components")
    if not isinstance(components, list):
        raise LearningContractError("candidate component list is invalid")
    for position, component in enumerate(components):
        if int(component.get("candidate_position", -1)) != position:
            raise LearningContractError("candidate positions must be contiguous and zero-based")
        key = str(component.get("component_key") or "")
        if not key or key in component_keys:
            raise LearningContractError("candidate component keys must be unique")
        component_keys.append(key)
        try:
            descriptor = [float(component[name]) for name in COMPONENT_DESCRIPTOR_NAMES]
        except (KeyError, TypeError, ValueError) as error:
            raise LearningContractError(
                "candidate descriptor contract mismatch for %s" % episode_id
            ) from error
        if not np.all(np.isfinite(descriptor)):
            raise LearningContractError("candidate descriptors must be finite")
        rows.append(descriptor)
    vectors = np.asarray(rows, dtype=np.float32).reshape(
        len(rows), len(COMPONENT_DESCRIPTOR_NAMES)
    )
    masks = None
    if record.get("central_masks_available"):
        if not components:
            masks = np.zeros((0, 0, 0), dtype=np.uint8)
        else:
            masks = np.stack(
                [np.asarray(c["prompted_slice_mask"], dtype=np.uint8) for c in components],
                axis=0,
            )
            if masks.ndim != 3 or not np.all(np.isin(masks, [0, 1])):
                raise LearningContractError("central component masks must be binary [K,H,W]")
    cue_hit = record.get("cue_hit_component_position")
    if cue_hit is not None:
        cue_hit = int(cue_hit)
        if not 0 <= cue_hit < len(components):
            raise LearningContractError("cue-hit component position is invalid")
    result = (
        vectors,
        np.ones(len(vectors), dtype=bool),
        masks,
        component_keys,
        cue_hit,
    )
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
        training_labels: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ):
        rows = load_jsonl(manifest)
        for row in rows:
            _validate_inference_row(row)
        self.rows = [row for row in rows if row.get("partition") == partition]
        if not self.rows:
            raise LearningContractError("partition %s is empty" % partition)
        self.candidates = candidates
        # family_gold is the label-only join used ONLY at training time; the
        # inference path never supplies it and therefore never reads goals.
        self.training_labels = training_labels
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
        vectors, valid, _, _, _ = _load_components(
            self.candidates, episode_id, self._component_cache
        )
        output: Dict[str, Any] = {
            "episode_id": episode_id,
            "operation": operation,
            "operation_id": torch.tensor(operation_id, dtype=torch.long),
            "visual": torch.from_numpy(visual),
            "spacing_xy": torch.from_numpy(bundle["spacing_xy"]),
            "component_vectors": torch.from_numpy(vectors),
            "component_mask": torch.from_numpy(valid),
        }
        if self.training_labels is not None:
            label = self.training_labels.get(episode_id)
            if label is None:
                raise LearningContractError("missing training label for %s" % episode_id)
            if str(label["partition"]) != str(row["partition"]):
                raise LearningContractError("visible/label partition mismatch")
            if str(label["operation"]) != operation:
                raise LearningContractError("visible/label operation mismatch")
            goal = str(label["goal"])
            output["patient_id"] = str(label["patient_id"])
            output["group_id"] = str(
                label.get("matched_state_group_id")
                or label.get("episode_family_id")
                or episode_id
            )
            output["goal"] = goal
            output["family_gold"] = torch.tensor(
                goal_to_local_family_id(goal, operation), dtype=torch.long
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
        labels: Mapping[str, Mapping[str, Any]],
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
        for row in rows:
            _validate_inference_row(row)
        self.rows = [row for row in rows if row.get("partition") == partition]
        if not self.rows:
            raise LearningContractError("partition %s is empty" % partition)
        self.candidates = candidates
        self.labels = labels
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
        label = self.labels.get(episode_id)
        if label is None:
            raise LearningContractError("missing editor label for %s" % episode_id)
        if str(label["partition"]) != str(row["partition"]):
            raise LearningContractError("visible/label partition mismatch")
        if str(label["operation"]) != operation:
            raise LearningContractError("visible/label operation mismatch")
        goal = str(label["goal"])
        gold_family = GOAL_TO_FAMILY.get(goal)
        if gold_family is None:
            raise LearningContractError("unknown gold goal: %s" % goal)
        validate_legal_call(
            operation,
            gold_family,
            NEW_CUE_SENTINEL if gold_family == "CREATE_NEW" else "placeholder",
        )
        gold_family_id = family_to_id(gold_family)
        gold_operand_key: Optional[str] = None
        gold_operand_position: Optional[int] = None
        if operation == "ADD" and gold_family == "CREATE_NEW":
            gold_operand_key = NEW_CUE_SENTINEL
        vectors, valid, masks, component_keys, cue_hit_position = _load_components(
            self.candidates, episode_id, self._component_cache
        )
        if operation == "REMOVE":
            if cue_hit_position is None:
                raise LearningContractError(
                    "REMOVE episode lacks deterministic cue-hit component"
                )
            gold_operand_position = cue_hit_position
            gold_operand_key = component_keys[cue_hit_position]
        elif operation == "ADD" and gold_family != "CREATE_NEW":
            targets = self.pointer_targets.get(episode_id, [])
            if not targets:
                raise LearningContractError("missing pointer targets for %s" % episode_id)
            first = int(targets[0])
            if not 0 <= first < len(component_keys):
                raise LearningContractError("gold pointer target is outside candidates")
            gold_operand_position = first
            gold_operand_key = component_keys[first]

        chosen_family = gold_family
        chosen_operand = gold_operand_key
        chosen_position = gold_operand_position
        if self.call_source == "predicted" and self.editor_condition in (
            "typed_program", "continuous"
        ):
            prediction = self.predicted_calls.get(episode_id)
            if prediction is None:
                raise LearningContractError("missing predicted call for %s" % episode_id)
            if prediction.get("decision", "PREDICT") != "PREDICT":
                raise LearningContractError("abstained call cannot train the predicted-call editor")
            call = prediction.get("call", prediction)
            predicted_operation = str(call.get("operation") or "")
            chosen_family = str(call.get("family") or "")
            chosen_operand = str(call.get("operand") or "")
            if predicted_operation != operation:
                raise LearningContractError("predicted call changed signed-cue operation")
            validate_legal_call(operation, chosen_family, chosen_operand)
            if call.get("goal") is not None and str(call["goal"]) != render_goal(
                operation, chosen_family
            ):
                raise LearningContractError("predicted goal is inconsistent with legal call")
            if chosen_operand == NEW_CUE_SENTINEL:
                chosen_position = None
            else:
                try:
                    chosen_position = component_keys.index(chosen_operand)
                except ValueError:
                    raise LearningContractError(
                        "predicted operand is absent from candidates"
                    ) from None
            if operation == "REMOVE" and chosen_position != cue_hit_position:
                raise LearningContractError(
                    "REMOVE predicted operand is not the deterministic cue-hit component"
                )

        selected_channel = np.zeros_like(bundle["m0"], dtype=np.float32)
        if chosen_position is not None:
            if masks is None:
                raise LearningContractError(
                    "selected operand has no crop-aligned central component mask"
                )
            selected_channel = np.asarray(masks[chosen_position], dtype=np.float32)
            if selected_channel.shape != bundle["m0"].shape:
                raise LearningContractError("selected component mask geometry mismatch")
        signed_cue = visual[15:16] - visual[16:17]
        editor_visual = np.concatenate([visual[:10], visual[12:13], signed_cue], axis=0)
        if self.editor_condition in ("flat_action", "typed_program", "continuous", "null_program"):
            editor_visual = np.concatenate([editor_visual, selected_channel[None]], axis=0)
        null_condition = self.editor_condition in ("spatial_only", "null_program")
        family_id = gold_family_id
        operand_mode = 0 if chosen_operand != NEW_CUE_SENTINEL else 1
        if self.editor_condition == "typed_program":
            family_id = family_to_id(chosen_family)
        elif self.editor_condition == "flat_action":
            operand_mode = 0
        else:
            family_id = -1
            operand_mode = 2
        if self.program_dropout and self.rng.random() < self.program_dropout:
            family_id = -1
            operand_mode = 2
            null_condition = True
        active = 1 if not null_condition else 0
        support_mode = 0 if not null_condition else 1
        output = {
            "episode_id": episode_id,
            "patient_id": str(label["patient_id"]),
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
        if self.editor_condition == "continuous":
            output["compiler_visual"] = torch.from_numpy(visual)
            output["spacing_xy"] = torch.from_numpy(bundle["spacing_xy"])
            output["component_vectors"] = torch.from_numpy(vectors)
            output["component_mask"] = torch.from_numpy(valid)
        if self.editor_condition in ("flat_action", "typed_program", "continuous", "null_program"):
            output["selected_component"] = torch.from_numpy(selected_channel[None])
        if self.load_evaluation:
            evaluation_path = Path(label["evaluation_npz"])
            if not evaluation_path.is_file():
                raise LearningContractError(
                    "missing evaluation_npz: %s" % evaluation_path
                )
            if _sha256_file(evaluation_path) != label.get("evaluation_sha256"):
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
        bad_sizes = {key: len(value) for key, value in by_group.items() if len(value) != 3}
        if bad_sizes:
            raise LearningContractError(
                "same-operation matched groups must contain exactly three families: %s"
                % bad_sizes
            )
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


def program_collate(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pad variable component counts and collate every other field normally."""

    if not batch:
        raise LearningContractError("cannot collate an empty batch")
    if not all("component_vectors" in row and "component_mask" in row for row in batch):
        return default_collate(list(batch))
    dimensions = {int(row["component_vectors"].shape[-1]) for row in batch}
    if dimensions != {len(COMPONENT_DESCRIPTOR_NAMES)}:
        raise LearningContractError("component descriptor dimension drift")
    maximum = max(1, max(int(row["component_vectors"].shape[0]) for row in batch))
    vectors = torch.zeros(
        len(batch), maximum, len(COMPONENT_DESCRIPTOR_NAMES), dtype=torch.float32
    )
    mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
    stripped: List[Dict[str, Any]] = []
    for index, row in enumerate(batch):
        count = int(row["component_vectors"].shape[0])
        if row["component_mask"].shape != (count,):
            raise LearningContractError("component vector/mask count mismatch")
        if count:
            vectors[index, :count] = row["component_vectors"].to(dtype=torch.float32)
            mask[index, :count] = row["component_mask"].to(dtype=torch.bool)
        stripped.append(
            {
                key: value
                for key, value in row.items()
                if key not in ("component_vectors", "component_mask")
            }
        )
    result = default_collate(stripped)
    result["component_vectors"] = vectors
    result["component_mask"] = mask
    return result


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
