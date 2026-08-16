#!/usr/bin/env python3
"""Shared manifest, dataset, loss and reproducibility utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from common.petct_models import (
    EDITOR_PRIMARY_ARCHITECTURE_ID,
    OPERATION_TO_ID,
    TARGET_TO_ID,
    SCOPE_TO_ID,
    editor_condition_ids,
)
from common.petct_route_a_core import EDITOR_CONDITIONS


class LearningContractError(RuntimeError):
    pass


EXPERIMENT_CONFIG_SCHEMA = "PETCT-ROUTE-A-EXPERIMENT-v2.0"
# v2.1 (D-2026-08-06-02): same structure as v2.0, adds the fifth P2T input arm
# `operation_control` to p2t.simple_first_input_arms; carries derived_from + its
# own sha. The frozen v2 config must keep loading byte-identically.
EXPERIMENT_CONFIG_SCHEMAS = (
    EXPERIMENT_CONFIG_SCHEMA,
    "PETCT-ROUTE-A-EXPERIMENT-v2.1",
)
P2T_CHECKPOINT_SCHEMA = "PETCT-P2T-CHECKPOINT-v2.0"
P2T_PREDICTION_SCHEMA = "PETCT-P2T-PREDICTION-v2.0"
P2T_METRICS_SCHEMA = "PETCT-P2T-METRICS-v2.0"
EDITOR_CHECKPOINT_SCHEMA = "PETCT-EDITOR-CHECKPOINT-v1.3"
INTERVENTION_SCHEMA = "PETCT-INTENT-INTERVENTION-v1.1"
P2T_CHECKPOINT_CRITERION = (
    "max val_patient_balanced_six_class_joint_macro_f1; tie-break min val_loss"
)
EDITOR_CHECKPOINT_CRITERION = "min validation operation-aware authorized-target loss"
SHUFFLE_ALGORITHM = "six-legal-joint-label-preserving-non-identity-derangement-v2"
NULL_SEMANTICS = "zero-condition-vector-and-explicit-conditioner-bypass"
OPERATION_ONLY_SEMANTICS = "preserve-gold-operation-and-neutralize-target-scope"
WRONG_SCOPE_SEMANTICS = (
    "flip LOCAL and COMPLETE only when target=SAME; retain target; "
    "mark all NEW episodes INELIGIBLE"
)
WRONG_OPERATION_SEMANTICS = (
    "flip only the ADD/REMOVE conditioning token; retain the gold execution "
    "operation for physical set algebra and scoring; never construct a gold "
    "target or confirmatory utility row"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_experiment_config(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise LearningContractError(
            "cannot read frozen experiment config: %s" % error
        ) from error
    if not isinstance(value, dict):
        raise LearningContractError("experiment config must be a JSON object")
    if value.get("schema_version") not in EXPERIMENT_CONFIG_SCHEMAS:
        raise LearningContractError("unsupported experiment config schema")
    return value


def _required_mapping(
    value: Mapping[str, Any], key: str, context: str
) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, dict):
        raise LearningContractError("%s.%s must be an object" % (context, key))
    return nested


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LearningContractError("%s must be a positive integer" % name)
    return int(value)


def _finite_number(value: Any, name: str, *, strictly_positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LearningContractError("%s must be numeric" % name)
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if strictly_positive else result < 0):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise LearningContractError("%s must be finite and %s" % (name, qualifier))
    return result


def load_training_contract(config: Mapping[str, Any], component: str) -> Dict[str, Any]:
    if component not in ("p2t", "editor"):
        raise LearningContractError("unknown training component: %s" % component)
    section = _required_mapping(config, component, "config")
    training = _required_mapping(section, "training", component)
    optimizer = training.get("optimizer")
    if optimizer != "AdamW":
        raise LearningContractError("%s.training.optimizer must be AdamW" % component)
    seeds_raw = training.get("seeds")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise LearningContractError(
            "%s.training.seeds must be a non-empty list" % component
        )
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds_raw
    ):
        raise LearningContractError(
            "%s.training.seeds must contain non-negative integers" % component
        )
    seeds = [int(seed) for seed in seeds_raw]
    if len(set(seeds)) != len(seeds):
        raise LearningContractError("%s.training.seeds must be unique" % component)
    expected_criterion = (
        P2T_CHECKPOINT_CRITERION if component == "p2t" else EDITOR_CHECKPOINT_CRITERION
    )
    criterion = training.get("checkpoint_criterion")
    if criterion != expected_criterion:
        raise LearningContractError(
            "%s.training.checkpoint_criterion must be %r"
            % (component, expected_criterion)
        )
    contract: Dict[str, Any] = {
        "epochs": _positive_int(
            training.get("epochs"), "%s.training.epochs" % component
        ),
        "batch_size": _positive_int(
            training.get("batch_size"), "%s.training.batch_size" % component
        ),
        "learning_rate": _finite_number(
            training.get("learning_rate"),
            "%s.training.learning_rate" % component,
            strictly_positive=True,
        ),
        "weight_decay": _finite_number(
            training.get("weight_decay"),
            "%s.training.weight_decay" % component,
            strictly_positive=False,
        ),
        "seeds": seeds,
        "optimizer": optimizer,
        "checkpoint_criterion": criterion,
    }
    if component == "p2t":
        loss = _required_mapping(training, "loss", "p2t.training")
        loss_keys = {
            "joint_loss_weight": "joint_cross_entropy_weight",
            "operation_loss_weight": "operation_cross_entropy_weight",
            "target_loss_weight": "target_cross_entropy_weight",
            "scope_loss_weight": "scope_cross_entropy_weight",
        }
        for contract_key, config_key in loss_keys.items():
            contract[contract_key] = _finite_number(
                loss.get(config_key),
                "p2t.training.loss.%s" % config_key,
                strictly_positive=False,
            )
        if contract["joint_loss_weight"] <= 0:
            raise LearningContractError("p2t joint loss weight must be positive")
    else:
        loss = _required_mapping(training, "loss", "editor.training")
        for key in (
            "bce_loss_weight",
            "dice_loss_weight",
            "unauthorized_delta_loss_weight",
        ):
            contract[key] = _finite_number(
                loss.get(key),
                "editor.training.loss.%s" % key,
                strictly_positive=False,
            )
            if contract[key] <= 0:
                raise LearningContractError(
                    "editor.training.loss.%s must be positive" % key
                )
        expected_formula = (
            "BCEWithLogits(delta_logits,target_delta) + "
            "softDice(delta_logits,target_delta) + "
            "mean(sigmoid(delta_logits)*(1-authorized_delta))"
        )
        if loss.get("formula") != expected_formula:
            raise LearningContractError("editor loss formula differs from v2")
        expected_semantics = (
            "the same unauthorized-delta formula penalizes ADD outside authorized "
            "FN and REMOVE outside authorized FP"
        )
        if loss.get("operation_semantics") != expected_semantics:
            raise LearningContractError("editor loss operation semantics differ from v2")
    return contract


def load_editor_fusion_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the professor-directed simple-first editor contract."""

    editor = _required_mapping(config, "editor", "config")
    if editor.get("primary_architecture_id") != EDITOR_PRIMARY_ARCHITECTURE_ID:
        raise LearningContractError("editor primary architecture is not the v2 simple-first model")
    fusion = _required_mapping(editor, "fusion_plan", "editor")
    if fusion.get("primary") != "simple structured conditioning":
        raise LearningContractError("editor fusion_plan.primary must remain simple")
    deferred = fusion.get("deferred_ablations")
    expected_deferred = ["FiLM", "concat", "gated", "intent-image cross_attention"]
    if deferred != expected_deferred:
        raise LearningContractError("editor deferred fusion list changed")
    return {
        "primary_architecture_id": EDITOR_PRIMARY_ARCHITECTURE_ID,
        "deferred_ablations": list(deferred),
        "fusion_plan": "simple-first-current-campaign; listed ablations deferred",
    }


def load_editor_architecture_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Canonical v2 name; the old function name remains an internal alias only."""

    return load_editor_fusion_contract(config)


def select_frozen_seed(contract: Mapping[str, Any], requested_seed: int) -> int:
    if isinstance(requested_seed, bool) or not isinstance(requested_seed, int):
        raise LearningContractError(
            "--seed must be an integer from the frozen registry"
        )
    seeds = contract.get("seeds")
    if not isinstance(seeds, list) or requested_seed not in seeds:
        raise LearningContractError(
            "--seed %s is not present in the frozen seed registry" % requested_seed
        )
    return int(requested_seed)


def validate_frozen_override(name: str, requested: Any, frozen: Any) -> None:
    if requested is None:
        return
    if isinstance(frozen, float):
        matches = isinstance(requested, (int, float)) and math.isclose(
            float(requested), frozen, rel_tol=0.0, abs_tol=0.0
        )
    else:
        matches = requested == frozen
    if not matches:
        raise LearningContractError(
            "%s=%r differs from frozen config value %r" % (name, requested, frozen)
        )


def load_p2t_evaluation_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    p2t = _required_mapping(config, "p2t", "config")
    evaluation = _required_mapping(p2t, "evaluation", "p2t")
    required = evaluation.get("required_diagnostics")
    expected = {
        "six_class_joint_confusion_matrix",
        "operation_macro_f1",
        "joint_decoded_target_macro_f1",
        "joint_decoded_scope_macro_f1",
        "invalid_combination_rate",
        "per_polarity",
        "per_strategy",
        "per_goal",
        "patient_balanced_metrics",
    }
    if (
        not isinstance(required, list)
        or any(not isinstance(value, str) for value in required)
        or not expected.issubset(set(required))
    ):
        raise LearningContractError(
            "p2t.evaluation.required_diagnostics omits required diagnostics"
        )
    alpha = _finite_number(
        evaluation.get("alpha"), "p2t.evaluation.alpha", strictly_positive=True
    )
    if alpha >= 1:
        raise LearningContractError("p2t.evaluation.alpha must be less than 1")
    primary_arm = p2t.get("primary_end_to_end_arm")
    ablations = p2t.get("simple_first_input_arms")
    if (
        not isinstance(ablations, list)
        or not ablations
        or any(not isinstance(value, str) for value in ablations)
        or len(set(ablations)) != len(ablations)
        or primary_arm not in ablations
    ):
        raise LearningContractError(
            "p2t primary_end_to_end_arm must occur in simple_first_input_arms"
        )
    bootstrap_seed = evaluation.get("bootstrap_seed")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise LearningContractError(
            "p2t.evaluation.bootstrap_seed must be a non-negative integer"
        )
    candidate = _required_mapping(
        p2t, "candidate_confirmatory_contrast", "p2t"
    )
    expected_candidate = {
        "treatment": "full",
        "comparator": "no_M0",
        "metric": "six_class_patient_balanced_joint_macro_f1",
        "alternative": "greater",
        "minimum_absolute_effect": (
            "PENDING_PRETEST_FEASIBILITY_AND_POWER_JUSTIFICATION"
        ),
        "decision_rule": (
            "freeze effect threshold before locked test; then require p-value "
            "and point-estimate criteria"
        ),
    }
    if candidate != expected_candidate:
        raise LearningContractError(
            "p2t candidate confirmatory contrast differs from pre-freeze v2"
        )
    confirmatory_gate = p2t.get("confirmatory_execution_gate")
    if confirmatory_gate != (
        "BLOCKED_UNTIL_ERROR_ATLAS_FEASIBILITY_AND_EFFECT_FREEZE"
    ):
        raise LearningContractError("p2t confirmatory execution gate differs from v2")
    return {
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_samples": _positive_int(
            evaluation.get("bootstrap_samples"), "p2t.evaluation.bootstrap_samples"
        ),
        "alpha": alpha,
        "required_diagnostics": list(required),
        "primary_end_to_end_arm": str(primary_arm),
        "ablation_inputs": [str(value) for value in ablations],
        "validation_role": "DESCRIPTIVE_FEASIBILITY_ONLY",
        "candidate_confirmatory_contrast": dict(candidate),
        "confirmatory_execution_gate": str(confirmatory_gate),
        "confirmatory_execution_allowed": False,
    }


def load_intent_intervention_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    editor = _required_mapping(config, "editor", "config")
    interventions = _required_mapping(editor, "intent_interventions", "editor")
    if interventions.get("shuffle_algorithm") != SHUFFLE_ALGORITHM:
        raise LearningContractError(
            "editor intent shuffle algorithm is unsupported or not frozen"
        )
    seed = interventions.get("shuffle_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise LearningContractError(
            "editor intent shuffle seed must be a non-negative integer"
        )
    if interventions.get("null_semantics") != NULL_SEMANTICS:
        raise LearningContractError("editor NULL semantics are not the required bypass")
    if interventions.get("operation_only_semantics") != OPERATION_ONLY_SEMANTICS:
        raise LearningContractError(
            "editor operation-only semantics are not the required controlled ablation"
        )
    if interventions.get("wrong_scope_semantics") != WRONG_SCOPE_SEMANTICS:
        raise LearningContractError("editor wrong-scope semantics are not frozen")
    if interventions.get("wrong_operation_semantics") != WRONG_OPERATION_SEMANTICS:
        raise LearningContractError("editor wrong-operation semantics are not frozen")
    return {
        "shuffle_algorithm": SHUFFLE_ALGORITHM,
        "shuffle_seed": int(seed),
        "null_semantics": NULL_SEMANTICS,
        "operation_only_semantics": OPERATION_ONLY_SEMANTICS,
        "wrong_scope_semantics": WRONG_SCOPE_SEMANTICS,
        "wrong_operation_semantics": WRONG_OPERATION_SEMANTICS,
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise LearningContractError(
                    "manifest line %d is not an object" % line_number
                )
            rows.append(value)
    if not rows:
        raise LearningContractError("manifest is empty")
    return rows


def load_frozen_learning_split_index(path: Path) -> Dict[str, Any]:
    """Load the split itself as the independent case/patient/partition authority."""

    raw = Path(path)
    if raw.is_symlink():
        raise LearningContractError("learning split must not be a symlink")
    resolved = raw.resolve()
    try:
        with resolved.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LearningContractError("cannot read frozen learning split: %s" % error) from error
    if not isinstance(document, dict):
        raise LearningContractError("learning split must be a JSON object")
    if document.get("schema_version") != "PETCT-LEARNING-SPLIT-v1.0":
        raise LearningContractError("unsupported learning split schema")
    if document.get("status") != "FROZEN_BEFORE_MODEL_SELECTION":
        raise LearningContractError("learning split is not frozen before model selection")
    patient_rows = document.get("patients")
    if not isinstance(patient_rows, list) or not patient_rows:
        raise LearningContractError("learning split patients must be a non-empty list")
    case_to_patient: Dict[str, str] = {}
    case_to_partition: Dict[str, str] = {}
    patient_to_partition: Dict[str, str] = {}
    case_counts = {"train": 0, "val": 0, "test": 0}
    for index, raw_row in enumerate(patient_rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise LearningContractError("learning split patient row %d is invalid" % index)
        patient_id = str(raw_row.get("patient_id") or "").casefold()
        partition = str(raw_row.get("partition") or "")
        case_ids = raw_row.get("case_ids")
        if (
            not patient_id
            or patient_id in patient_to_partition
            or partition not in case_counts
            or not isinstance(case_ids, list)
            or not case_ids
        ):
            raise LearningContractError("learning split patient row %d is invalid" % index)
        normalized_cases = [str(case_id) for case_id in case_ids]
        if (
            any(not case_id for case_id in normalized_cases)
            or len(normalized_cases) != len(set(normalized_cases))
        ):
            raise LearningContractError("learning split patient cases are invalid")
        patient_to_partition[patient_id] = partition
        for case_id in normalized_cases:
            if case_id in case_to_partition:
                raise LearningContractError("case appears more than once in learning split: %s" % case_id)
            case_to_patient[case_id] = patient_id
            case_to_partition[case_id] = partition
            case_counts[partition] += 1
    if document.get("patient_count") != len(patient_to_partition):
        raise LearningContractError("learning split patient_count differs from its rows")
    if document.get("case_count") != len(case_to_partition):
        raise LearningContractError("learning split case_count differs from its rows")
    if document.get("case_counts") != case_counts:
        raise LearningContractError("learning split case_counts differs from its rows")
    return {
        "path": str(resolved),
        "split_sha256": sha256_file(resolved),
        "case_to_patient": case_to_patient,
        "case_to_partition": case_to_partition,
        "patient_to_partition": patient_to_partition,
        "case_counts": case_counts,
    }


def validate_manifest_rows_against_frozen_learning_split(
    rows: Sequence[Mapping[str, Any]],
    learning_split: Path,
    *,
    require_episode_id: bool,
    allowed_partitions: Iterable[str],
    require_split_hash: bool = True,
) -> Dict[str, Any]:
    """Reject relabelled case/patient/partition rows before any tensor is opened."""

    index = load_frozen_learning_split_index(learning_split)
    allowed = set(allowed_partitions)
    if not allowed or not allowed.issubset({"train", "val", "test"}):
        raise LearningContractError("allowed_partitions must be an explicit subset")
    if not rows:
        raise LearningContractError("manifest is empty")
    episode_ids = set()
    observed_counts = {partition: 0 for partition in sorted(allowed)}
    for row_number, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise LearningContractError("manifest row %d is not an object" % row_number)
        case_id = str(raw_row.get("case_id") or "")
        patient_id = str(raw_row.get("patient_id") or "").casefold()
        partition = str(raw_row.get("partition") or "")
        if not case_id or not patient_id or partition not in allowed:
            raise LearningContractError(
                "manifest row %d has invalid case_id/patient_id/partition" % row_number
            )
        expected_patient = index["case_to_patient"].get(case_id)
        expected_partition = index["case_to_partition"].get(case_id)
        if expected_patient is None:
            raise LearningContractError("manifest case is absent from frozen split: %s" % case_id)
        if patient_id != expected_patient:
            raise LearningContractError(
                "manifest patient differs from frozen split for case %s" % case_id
            )
        if partition != expected_partition:
            raise LearningContractError(
                "manifest partition differs from frozen split for case %s" % case_id
            )
        if require_split_hash and raw_row.get("learning_split_sha256") != index["split_sha256"]:
            raise LearningContractError(
                "manifest row does not bind the frozen learning split hash"
            )
        episode_id = str(raw_row.get("episode_id") or "")
        if require_episode_id:
            if not episode_id:
                raise LearningContractError("manifest row requires episode_id")
            if episode_id in episode_ids:
                raise LearningContractError("duplicate episode_id: %s" % episode_id)
            episode_ids.add(episode_id)
        observed_counts[partition] += 1
    return {
        "learning_split_sha256": index["split_sha256"],
        "row_count": len(rows),
        "episode_count": len(episode_ids),
        "observed_partition_counts": observed_counts,
    }


def validate_partitions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    patient_partition = {}
    episode_ids = set()
    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        episode = str(row.get("episode_id") or "")
        patient = str(row.get("patient_id") or "")
        partition = str(row.get("partition") or "")
        if not episode or not patient or partition not in counts:
            raise LearningContractError(
                "episode_id, patient_id and train/val/test partition are required"
            )
        if episode in episode_ids:
            raise LearningContractError("duplicate episode_id: %s" % episode)
        episode_ids.add(episode)
        if patient in patient_partition and patient_partition[patient] != partition:
            raise LearningContractError(
                "patient crosses learning partitions: %s" % patient
            )
        patient_partition[patient] = partition
        counts[partition] += 1
    return {"patient_count": len(patient_partition), "episode_counts": counts}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_input_ablation(visual: np.ndarray, ablation: str) -> np.ndarray:
    array = np.asarray(visual, dtype=np.float32).copy()
    if array.ndim != 3 or array.shape[0] != 17:
        raise LearningContractError("P2T visual tensor must have shape [17,H,W]")
    if ablation == "full":
        return array
    if ablation == "no_M0":
        array[10:15] = 0
    elif ablation == "polarity_blind":
        support = np.maximum(array[15], array[16])
        array[15] = support
        array[16] = support
    elif ablation == "geometry_only":
        array[0:10] = 0
    elif ablation == "operation_control":
        # The `operation` control group.  `no_M0` and `polarity_blind` each close only
        # one ADD/REMOVE path, so neither is a control: the gold construction forces
        # scribble ⊆ GT\M0 for ADD and scribble ⊆ M0\GT for REMOVE, which keeps the
        # operation readable from cue-vs-M0 containment whenever M0 survives.
        array[10:15] = 0
        support = np.maximum(array[15], array[16])
        array[15] = support
        array[16] = support
    else:
        raise LearningContractError("unknown input ablation: %s" % ablation)
    return array


class EpisodeDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        partition: str,
        input_ablation: str = "full",
        editor_condition: Optional[str] = None,
        predicted_intents: Optional[Mapping[str, Mapping[str, str]]] = None,
        intervention_intents: Optional[Mapping[str, Mapping[str, str]]] = None,
        load_evaluation: bool = True,
    ):
        rows = load_jsonl(manifest)
        validate_partitions(rows)
        self.rows = [row for row in rows if row["partition"] == partition]
        if not self.rows:
            raise LearningContractError("partition %s is empty" % partition)
        self.input_ablation = input_ablation
        self.editor_condition = editor_condition
        self.predicted_intents = dict(predicted_intents or {})
        self.intervention_intents = dict(intervention_intents or {})
        self.load_evaluation = bool(load_evaluation)
        if editor_condition is not None and editor_condition not in EDITOR_CONDITIONS:
            raise LearningContractError(
                "unknown editor condition: %s" % editor_condition
            )
        if editor_condition == "same_weight_shuffled" and not self.intervention_intents:
            raise LearningContractError(
                "same_weight_shuffled requires a frozen intervention manifest"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        visible_path = Path(row["visible_npz"])
        if not visible_path.is_file():
            raise LearningContractError("missing visible_npz: %s" % visible_path)
        if sha256_file(visible_path) != row.get("visible_sha256"):
            raise LearningContractError("visible_npz hash mismatch: %s" % visible_path)
        with np.load(str(visible_path), allow_pickle=False) as bundle:
            required = {
                "visual",
                "m0",
                "scribble",
                "cue_fg",
                "cue_bg",
                "spacing_xy",
            }
            if set(bundle.files) != required:
                raise LearningContractError("visible tensor bundle schema mismatch")
            visual = apply_input_ablation(bundle["visual"], self.input_ablation)
            m0 = np.asarray(bundle["m0"], dtype=np.float32)
            scribble = np.asarray(bundle["scribble"], dtype=np.float32)
            cue_fg = np.asarray(bundle["cue_fg"], dtype=np.float32)
            cue_bg = np.asarray(bundle["cue_bg"], dtype=np.float32)
            spacing = np.asarray(bundle["spacing_xy"], dtype=np.float32)
        geometry = row.get("geometry")
        if not isinstance(geometry, dict):
            raise LearningContractError("manifest row omits geometry contract")
        manifest_spacing = np.asarray(
            geometry.get("output_spacing_xy"), dtype=np.float32
        )
        if manifest_spacing.shape != (2,) or not np.array_equal(
            manifest_spacing, spacing
        ):
            raise LearningContractError(
                "visible spacing_xy differs from manifest output_spacing_xy"
            )
        if not np.all(np.isfinite(visual)):
            raise LearningContractError("visual tensor contains NaN or Inf")
        if (
            m0.ndim != 2
            or scribble.shape != m0.shape
            or cue_fg.shape != m0.shape
            or cue_bg.shape != m0.shape
            or visual.shape != (17, *m0.shape)
        ):
            raise LearningContractError("visible arrays have inconsistent 2D shapes")
        if not np.all(np.isfinite(m0)) or not np.all(np.isin(m0, [0, 1])):
            raise LearningContractError("M0 must be a finite binary mask")
        if (
            not np.all(np.isfinite(scribble))
            or not np.all(np.isin(scribble, [0, 1]))
            or not scribble.any()
        ):
            raise LearningContractError(
                "scribble must be a non-empty finite binary mask"
            )
        for name, cue in (("cue_fg", cue_fg), ("cue_bg", cue_bg)):
            if not np.all(np.isfinite(cue)) or not np.all(np.isin(cue, [0, 1])):
                raise LearningContractError(f"{name} must be a finite binary mask")
        if np.any(cue_fg * cue_bg) or not np.array_equal(
            (cue_fg + cue_bg) > 0, scribble > 0
        ):
            raise LearningContractError("FG/BG cue channels must be disjoint and cover scribble")
        if (
            spacing.shape != (2,)
            or not np.all(np.isfinite(spacing))
            or np.any(spacing <= 0)
        ):
            raise LearningContractError("spacing_xy must be two positive finite values")
        target = gt = authorized = None
        evaluation_path = Path(row["evaluation_npz"])
        if self.load_evaluation:
            if not evaluation_path.is_file():
                raise LearningContractError(
                    "missing evaluation_npz: %s" % evaluation_path
                )
            if sha256_file(evaluation_path) != row.get("evaluation_sha256"):
                raise LearningContractError(
                    "evaluation_npz hash mismatch: %s" % evaluation_path
                )
            with np.load(str(evaluation_path), allow_pickle=False) as bundle:
                required = {"target", "gt", "authorized"}
                if set(bundle.files) != required:
                    raise LearningContractError(
                        "evaluation tensor bundle schema mismatch"
                    )
                # `target_mask`, not `target`: a few lines below, `target` is rebound to the
                # gold TARGET slot label ("SAME"/"NEW").  Sharing the name silently replaced
                # this mask with a str, so every editor batch (the only caller that sets
                # load_evaluation=True) died at `target[None]`.
                target_mask = np.asarray(bundle["target"], dtype=np.float32)
                gt = np.asarray(bundle["gt"], dtype=np.float32)
                authorized = np.asarray(bundle["authorized"], dtype=np.float32)
            for name, array in (
                ("target", target_mask),
                ("gt", gt),
                ("authorized", authorized),
            ):
                if (
                    array.shape != m0.shape
                    or not np.all(np.isfinite(array))
                    or not np.all(np.isin(array, [0, 1]))
                ):
                    raise LearningContractError(
                        "%s must be a same-shape finite binary mask" % name
                    )
            if not np.array_equal(target_mask, authorized) or not authorized.any():
                raise LearningContractError(
                    "target must equal non-empty authorized residual delta"
                )
            if np.any(scribble > authorized):
                raise LearningContractError(
                    "scribble must lie inside authorized target"
                )
        operation = str(row["operation"])
        target = str(row["target"])
        scope = str(row["scope"])
        if operation not in ("ADD", "REMOVE"):
            raise LearningContractError("illegal gold operation")
        if target not in ("SAME", "NEW") or scope not in (
            "LOCAL",
            "COMPLETE",
        ) or (target == "NEW" and scope == "LOCAL"):
            raise LearningContractError("illegal gold target/scope")
        if operation == "ADD":
            if not np.array_equal(cue_fg, scribble) or cue_bg.any():
                raise LearningContractError("ADD requires positive FG cue only")
            if self.load_evaluation and (
                np.any(authorized > gt) or np.any(authorized * m0)
            ):
                raise LearningContractError("ADD target must be a subset of GT\\M0")
        else:
            if not np.array_equal(cue_bg, scribble) or cue_fg.any():
                raise LearningContractError("REMOVE requires negative BG cue only")
            if self.load_evaluation and (
                np.any(authorized > m0) or np.any(authorized * gt)
            ):
                raise LearningContractError("REMOVE target must be a subset of M0\\GT")
        intent_mode = "CORRECT"
        if self.editor_condition in (
            "visual_state_only",
            "spatial_only",
            "same_weight_NULL",
        ):
            intent_mode = "NULL"
        elif self.editor_condition == "scribble_plus_operation":
            intent_mode = "OPERATION_ONLY"
        # The editor contract remains PET5+CT5+central-M0+scribble.  P2T alone
        # consumes the full five-slice M0 state.  Both views come from the same
        # immutable 17-channel visible tensor.
        if self.editor_condition is not None:
            signed_cue = visual[15:16] - visual[16:17]
            visual = np.concatenate([visual[:10], visual[12:13], signed_cue], axis=0)
        # The crop was already fixed around the immutable scribble upstream.
        # This arm hides the prompt from the model; it is not a prompt-free ROI.
        if self.editor_condition == "visual_state_only":
            visual[11] = 0
        elif self.editor_condition == "intent_only":
            visual[11] = 0
        elif self.editor_condition == "same_weight_wrong_scope":
            if target == "NEW":
                raise LearningContractError(
                    "wrong-scope intervention is undefined for NEW_COMPLETE"
                )
            scope = "COMPLETE" if scope == "LOCAL" else "LOCAL"
        elif self.editor_condition == "same_weight_shuffled":
            shuffled = self.intervention_intents.get(str(row["episode_id"]))
            if not shuffled:
                raise LearningContractError("missing frozen shuffled intent")
            if (
                shuffled.get("operation"),
                shuffled.get("target"),
                shuffled.get("scope"),
            ) == (operation, target, scope):
                raise LearningContractError(
                    "shuffled intent did not change the slot pair"
                )
            operation, target, scope = (
                shuffled["operation"],
                shuffled["target"],
                shuffled["scope"],
            )
        elif self.editor_condition == "predicted_slots":
            prediction = self.predicted_intents.get(str(row["episode_id"]))
            if not prediction:
                raise LearningContractError("missing predicted intent for episode")
            operation, target, scope = (
                prediction["operation"],
                prediction["target"],
                prediction["scope"],
            )
        operation_id, target_id, scope_id = editor_condition_ids(
            operation, target, scope, intent_mode
        )
        output = {
            "episode_id": str(row["episode_id"]),
            "patient_id": str(row["patient_id"]),
            "strategy": str(row.get("strategy") or "UNSPECIFIED"),
            "input_ablation": self.input_ablation,
            "visual": torch.from_numpy(visual),
            "operation_gold": torch.tensor(OPERATION_TO_ID[str(row["operation"])]),
            "target_gold": torch.tensor(TARGET_TO_ID[str(row["target"])]),
            "scope_gold": torch.tensor(SCOPE_TO_ID[str(row["scope"])]),
            "operation_id": torch.tensor(operation_id),
            "target_id": torch.tensor(target_id),
            "scope_id": torch.tensor(scope_id),
            "m0": torch.from_numpy(m0[None]),
            "scribble": torch.from_numpy(scribble[None]),
            "cue_fg": torch.from_numpy(cue_fg[None]),
            "cue_bg": torch.from_numpy(cue_bg[None]),
            "spacing_xy": torch.from_numpy(spacing),
            "evaluation_npz": str(evaluation_path),
            "visible_sha256": str(row["visible_sha256"]),
            "evaluation_sha256": str(row["evaluation_sha256"]),
        }
        if self.load_evaluation:
            output.update(
                {
                    "target": torch.from_numpy(target_mask[None]),
                    "gt": torch.from_numpy(gt[None]),
                    "authorized": torch.from_numpy(authorized[None]),
                }
            )
        return output


def soft_dice_loss(logits: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denom = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + epsilon) / (denom + epsilon)).mean()


def editor_loss(
    logits: Tensor,
    target: Tensor,
    authorized: Tensor,
    *,
    bce_loss_weight: float,
    dice_loss_weight: float,
    unauthorized_delta_loss_weight: float,
) -> Dict[str, Tensor]:
    weights = {
        "bce": float(bce_loss_weight),
        "dice": float(dice_loss_weight),
        "unauthorized": float(unauthorized_delta_loss_weight),
    }
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise LearningContractError("editor loss weights must be finite and positive")
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    dice_loss = soft_dice_loss(logits, target)
    unauthorized = (torch.sigmoid(logits) * (1.0 - authorized)).mean()
    total = (
        weights["bce"] * bce
        + weights["dice"] * dice_loss
        + weights["unauthorized"] * unauthorized
    )
    return {
        "total": total,
        "bce": bce,
        "dice": dice_loss,
        "unauthorized": unauthorized,
        "bce_loss_weight": torch.as_tensor(weights["bce"], device=logits.device),
        "dice_loss_weight": torch.as_tensor(weights["dice"], device=logits.device),
        "unauthorized_delta_loss_weight": torch.as_tensor(
            weights["unauthorized"], device=logits.device
        ),
    }


def macro_f1(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]
) -> float:
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2.0 * tp / denom)
    return float(np.mean(scores))


def confusion_matrix_diagnostic(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
    label_names: Sequence[str],
) -> Dict[str, Any]:
    if len(y_true) != len(y_pred) or not y_true:
        raise LearningContractError("confusion vectors must be non-empty and aligned")
    if len(labels) != len(label_names) or len(set(labels)) != len(labels):
        raise LearningContractError(
            "confusion labels and names must be unique and aligned"
        )
    index = {int(label): position for position, label in enumerate(labels)}
    counts = [[0 for _ in labels] for _ in labels]
    for truth, prediction in zip(y_true, y_pred):
        if int(truth) not in index or int(prediction) not in index:
            raise LearningContractError("confusion vector contains an unknown label")
        counts[index[int(truth)]][index[int(prediction)]] += 1
    return {
        "labels": [str(name) for name in label_names],
        "rows_are_gold": True,
        "columns_are_predicted": True,
        "counts": counts,
        "episode_count": len(y_true),
    }


def legal_joint_ids(operation: Tensor, target: Tensor, scope: Tensor) -> Tensor:
    """Map the three binary slots to the six supported joint goals."""

    if operation.shape != target.shape or target.shape != scope.shape:
        raise LearningContractError("intent slot tensors must share shape")
    joint = torch.full_like(operation, -1)
    joint[(operation == 0) & (target == 0) & (scope == 0)] = 0
    joint[(operation == 1) & (target == 0) & (scope == 0)] = 1
    joint[(operation == 0) & (target == 0) & (scope == 1)] = 2
    joint[(operation == 1) & (target == 0) & (scope == 1)] = 3
    joint[(operation == 0) & (target == 1) & (scope == 1)] = 4
    joint[(operation == 1) & (target == 1) & (scope == 1)] = 5
    if torch.any(joint < 0):
        raise LearningContractError(
            "predictor gold is outside the six-class support (including NEW_LOCAL)"
        )
    return joint


def patient_balanced_macro_f1_summary(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    patient_ids: Sequence[str],
    labels: Sequence[int],
) -> Dict[str, Any]:
    """Return the D-2026-08-14-04 patient-balanced macro-F1 estimand.

    Confusion counts are pooled within each patient and class before F1 is
    computed.  Undefined patient/class cells (``2*TP + FP + FN == 0``) are
    skipped.  Defined patient F1 values are averaged equally within each
    class, then defined class means are averaged equally.  The support for a
    class is therefore the number of patients with a defined F1 for it.
    """

    if not (len(y_true) == len(y_pred) == len(patient_ids)) or len(y_true) == 0:
        raise LearningContractError("metric vectors must be non-empty and aligned")
    label_values = [int(label) for label in labels]
    if not label_values or len(set(label_values)) != len(label_values):
        raise LearningContractError("metric labels must be non-empty and unique")
    allowed = set(label_values)
    truth = [int(value) for value in y_true]
    prediction = [int(value) for value in y_pred]
    if any(value not in allowed for value in truth + prediction):
        raise LearningContractError("metric vectors contain a label outside labels")

    grouped: Dict[str, List[int]] = {}
    for index, patient in enumerate(patient_ids):
        grouped.setdefault(str(patient), []).append(index)

    per_class_f1: Dict[str, Optional[float]] = {}
    per_class_support: Dict[str, int] = {}
    defined_class_scores: List[float] = []
    for label in label_values:
        patient_scores: List[float] = []
        for indices in grouped.values():
            tp = sum(
                truth[index] == label and prediction[index] == label
                for index in indices
            )
            fp = sum(
                truth[index] != label and prediction[index] == label
                for index in indices
            )
            fn = sum(
                truth[index] == label and prediction[index] != label
                for index in indices
            )
            denom = 2 * tp + fp + fn
            if denom > 0:
                patient_scores.append(2.0 * tp / denom)
        key = str(label)
        per_class_support[key] = len(patient_scores)
        if patient_scores:
            class_score = float(np.mean(patient_scores))
            per_class_f1[key] = class_score
            defined_class_scores.append(class_score)
        else:
            per_class_f1[key] = None

    if not defined_class_scores:
        raise LearningContractError("metric has no defined patient/class cells")
    return {
        "estimate": float(np.mean(defined_class_scores)),
        "patient_count": len(grouped),
        "defined_class_count": len(defined_class_scores),
        "per_class_f1": per_class_f1,
        "per_class_support": per_class_support,
    }


def patient_balanced_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    patient_ids: Sequence[str],
    labels: Sequence[int],
) -> float:
    """Return the D14 score while preserving the historical scalar API."""

    return float(
        patient_balanced_macro_f1_summary(y_true, y_pred, patient_ids, labels)[
            "estimate"
        ]
    )


def patient_cluster_bootstrap_macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    patient_ids: Sequence[str],
    labels: Sequence[int],
    *,
    seed: int,
    samples: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    point_summary = patient_balanced_macro_f1_summary(
        y_true, y_pred, patient_ids, labels
    )
    grouped: Dict[str, List[int]] = {}
    for index, patient in enumerate(patient_ids):
        grouped.setdefault(str(patient), []).append(index)
    patients = sorted(grouped)
    if len(patients) < 2 or samples < 1:
        raise LearningContractError(
            "cluster bootstrap needs >=2 patients and >=1 sample"
        )
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        selected = rng.choice(patients, size=len(patients), replace=True)
        truth, prediction, synthetic_patients = [], [], []
        for draw_index, patient in enumerate(selected):
            for row_index in grouped[str(patient)]:
                truth.append(y_true[row_index])
                prediction.append(y_pred[row_index])
                synthetic_patients.append("%d:%s" % (draw_index, patient))
        draws.append(
            patient_balanced_macro_f1(truth, prediction, synthetic_patients, labels)
        )
    return {
        "estimate": point_summary["estimate"],
        "ci_low": float(np.quantile(draws, alpha / 2)),
        "ci_high": float(np.quantile(draws, 1 - alpha / 2)),
        "patient_count": float(len(patients)),
        "bootstrap_samples": float(samples),
        "seed": float(seed),
        "defined_class_count": point_summary["defined_class_count"],
        "per_class_f1": point_summary["per_class_f1"],
        "per_class_support": point_summary["per_class_support"],
    }


def encode_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def encode_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _commit_staged_files_exclusive(staged: Sequence[tuple[Path, Path]]) -> None:
    committed: List[Path] = []
    try:
        if any(target.exists() for _, target in staged):
            raise FileExistsError("one or more output paths already exist")
        for temporary, target in staged:
            # The temporary is in the target directory, so a hard-link publish
            # is atomic and refuses to replace a concurrent writer's file.
            os.link(str(temporary), str(target))
            committed.append(target)
            temporary.unlink()
    except Exception:
        for target in reversed(committed):
            try:
                target.unlink()
            except FileNotFoundError:
                pass
        for temporary, _ in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def write_bytes_bundle_exclusive(payloads: Mapping[Path, bytes]) -> None:
    normalized = [(Path(path), payload) for path, payload in payloads.items()]
    if not normalized:
        raise LearningContractError("atomic output bundle must not be empty")
    resolved = [str(path.resolve()) for path, _ in normalized]
    if len(set(resolved)) != len(resolved):
        raise LearningContractError("atomic output bundle paths must be unique")
    if any(not isinstance(payload, bytes) for _, payload in normalized):
        raise LearningContractError("atomic output bundle values must be bytes")
    if any(path.exists() for path, _ in normalized):
        raise FileExistsError("one or more output paths already exist")
    staged: List[tuple[Path, Path]] = []
    try:
        for target, payload in normalized:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".%s." % target.name,
                suffix=".tmp",
                dir=str(target.parent),
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((temporary, target))
        _commit_staged_files_exclusive(staged)
    except Exception:
        for temporary, _ in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def write_json_exclusive(path: Path, payload: Any) -> None:
    write_bytes_bundle_exclusive({path: encode_json(payload)})


def write_jsonl_exclusive(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_bytes_bundle_exclusive({path: encode_jsonl(rows)})


def torch_save_exclusive(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError("output already exists: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        _commit_staged_files_exclusive([(temporary, path)])
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
