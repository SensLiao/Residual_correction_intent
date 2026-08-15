#!/usr/bin/env python3
"""Run one frozen editor checkpoint under a declared intent intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import (  # noqa: E402
    EDITOR_CHECKPOINT_SCHEMA,
    INTERVENTION_SCHEMA,
    NULL_SEMANTICS,
    P2T_CHECKPOINT_SCHEMA,
    P2T_METRICS_SCHEMA,
    P2T_PREDICTION_SCHEMA,
    EpisodeDataset,
    LearningContractError,
    encode_jsonl,
    load_editor_architecture_contract,
    load_experiment_config,
    load_intent_intervention_contract,
    load_jsonl,
    load_p2t_evaluation_contract,
    load_training_contract,
    sha256_file,
    validate_manifest_rows_against_frozen_learning_split,
    write_bytes_bundle_exclusive,
)
from common.petct_models import (  # noqa: E402
    EDITOR_PRIMARY_ARCHITECTURE_ID,
    LEGAL_GOALS,
    OPERATION_TO_ID,
    P2T_PRIMARY_ARCHITECTURE_ID,
    TARGET_TO_ID,
    ResidualEditorUNet2D,
    SCOPE_TO_ID,
)
from common.petct_route_a_core import (  # noqa: E402
    EDITOR_CONDITIONS,
    ContractError,
    expected_editor_checkpoint_condition as _expected_editor_checkpoint_condition,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)


LEGAL_PREDICTIONS = {
    "ADD_SAME_LOCAL": ("ADD", "SAME", "LOCAL"),
    "REMOVE_SAME_LOCAL": ("REMOVE", "SAME", "LOCAL"),
    "ADD_SAME_COMPLETE": ("ADD", "SAME", "COMPLETE"),
    "REMOVE_SAME_COMPLETE": ("REMOVE", "SAME", "COMPLETE"),
    "ADD_NEW_COMPLETE": ("ADD", "NEW", "COMPLETE"),
    "REMOVE_NEW_COMPLETE": ("REMOVE", "NEW", "COMPLETE"),
}
EDITOR_TRAINING_STATUS = "TRAINED_WHEN_THIS_SCRIPT_IS_EXECUTED"
WRONG_SCOPE_ELIGIBILITY = "SAME_ONLY_SCOPE_FLIP_DESCRIPTIVE"


def validate_editor_checkpoint_training_binding(
    *,
    checkpoint,
    training_manifest: Path,
    training_manifest_sha256: str,
    learning_split: Path,
    learning_split_sha256: str,
) -> None:
    """Require the checkpoint to bind the immutable training-side artifacts."""

    if checkpoint.get("status") != EDITOR_TRAINING_STATUS:
        raise LearningContractError("editor checkpoint training status is invalid")
    if (
        checkpoint.get("manifest_sha256") != training_manifest_sha256
        or checkpoint.get("training_manifest_sha256")
        != training_manifest_sha256
    ):
        raise LearningContractError(
            "editor checkpoint was not trained against the supplied training manifest"
        )
    for field in ("manifest", "training_manifest"):
        raw = Path(str(checkpoint.get(field) or ""))
        if raw.is_symlink() or not raw.is_file():
            raise LearningContractError(
                f"editor checkpoint {field} must be a non-symlink regular file"
            )
        if raw.resolve() != training_manifest.resolve():
            raise LearningContractError(
                f"editor checkpoint {field} differs from --training-manifest"
            )
    if checkpoint.get("learning_split_sha256") != learning_split_sha256:
        raise LearningContractError(
            "editor checkpoint was not trained against this frozen learning split"
        )
    raw_split = Path(str(checkpoint.get("learning_split") or ""))
    if raw_split.is_symlink() or not raw_split.is_file():
        raise LearningContractError(
            "editor checkpoint learning_split must be a non-symlink regular file"
        )
    if raw_split.resolve() != learning_split.resolve():
        raise LearningContractError(
            "editor checkpoint learning_split differs from --learning-split"
        )


def expected_editor_checkpoint_condition(condition: str) -> str:
    try:
        return _expected_editor_checkpoint_condition(condition)
    except ContractError as error:
        raise LearningContractError(str(error)) from error


def execution_operation_for_condition(
    condition: str,
    *,
    gold_operation: str,
    predicted: Mapping[str, object] | None = None,
    intervention: Mapping[str, object] | None = None,
) -> str:
    """Return the operation that deterministically composes M1.

    ``predicted_slots`` is the actual end-to-end arm, so its predicted operation
    controls the physical action. Same-weight and OOD interventions change only
    the learned conditioning token and retain cue/gold operation algebra.
    """

    if gold_operation not in {"ADD", "REMOVE"}:
        raise LearningContractError("editor row has an invalid gold operation")
    operation = gold_operation
    if condition == "predicted_slots":
        operation = str((predicted or {}).get("operation") or "")
    if operation not in {"ADD", "REMOVE"}:
        raise LearningContractError("editor condition resolved an invalid operation")
    return operation


def apply_operation_delta(
    logits: torch.Tensor,
    m0: torch.Tensor,
    operations: Sequence[str],
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one delta head under ADD union or REMOVE subtraction algebra."""

    if logits.shape != m0.shape or len(operations) != logits.shape[0]:
        raise ValueError("logits, M0 and operations must be batch-aligned")
    current = m0 > 0
    activated = torch.sigmoid(logits) >= float(threshold)
    delta = torch.zeros_like(activated)
    corrected = current.clone()
    for index, operation in enumerate(operations):
        if operation == "ADD":
            delta[index] = activated[index] & ~current[index]
            corrected[index] = current[index] | delta[index]
        elif operation == "REMOVE":
            delta[index] = activated[index] & current[index]
            corrected[index] = current[index] & ~delta[index]
        else:
            raise ValueError("operation must be ADD or REMOVE")
    return delta, corrected


def conditioned_slots_for_condition(
    condition: str,
    *,
    gold_operation: str,
    gold_target: str,
    gold_scope: str,
    predicted: Mapping[str, object] | None = None,
    intervention: Mapping[str, object] | None = None,
) -> tuple[str, str, str]:
    """Describe the slots actually exposed to the editor embedding."""

    if condition in {"visual_state_only", "spatial_only", "same_weight_NULL"}:
        return "NULL", "NULL", "NULL"
    if condition == "scribble_plus_operation":
        return gold_operation, "NULL", "NULL"
    if condition == "predicted_slots":
        source = predicted or {}
        return (
            str(source.get("operation") or ""),
            str(source.get("target") or ""),
            str(source.get("scope") or ""),
        )
    if condition == "same_weight_shuffled":
        source = intervention or {}
        return (
            str(source.get("operation") or ""),
            str(source.get("target") or ""),
            str(source.get("scope") or ""),
        )
    if condition == "same_weight_wrong_scope":
        if gold_target != "SAME":
            raise LearningContractError(
                "same_weight_wrong_scope is INELIGIBLE for NEW episodes; "
                "the intervention may flip scope only and must not change target"
            )
        if gold_scope not in {"LOCAL", "COMPLETE"}:
            raise LearningContractError(
                "same_weight_wrong_scope requires a legal SAME scope"
            )
        flipped_scope = "COMPLETE" if gold_scope == "LOCAL" else "LOCAL"
        return gold_operation, gold_target, flipped_scope
    if condition == "wrong_operation_OOD":
        flipped = "REMOVE" if gold_operation == "ADD" else "ADD"
        return flipped, gold_target, gold_scope
    return gold_operation, gold_target, gold_scope


def partition_rows_for_condition(
    condition: str, rows: Sequence[Mapping[str, object]]
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    """Return the explicit episode eligibility partition for one condition."""

    source = list(rows)
    if condition != "same_weight_wrong_scope":
        return source, []
    eligible: list[Mapping[str, object]] = []
    ineligible: list[Mapping[str, object]] = []
    for row in source:
        target = str(row.get("target") or "")
        scope = str(row.get("scope") or "")
        if target == "SAME" and scope in {"LOCAL", "COMPLETE"}:
            eligible.append(row)
        elif target == "NEW" and scope == "COMPLETE":
            ineligible.append(row)
        else:
            raise LearningContractError(
                "same_weight_wrong_scope eligibility encountered an illegal target/scope"
            )
    if not eligible:
        raise LearningContractError(
            "same_weight_wrong_scope has no eligible SAME episodes"
        )
    return eligible, ineligible


def episode_id_set_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    identifiers = sorted(str(row.get("episode_id") or "") for row in rows)
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise LearningContractError("condition eligibility has empty/duplicate episode_id")
    payload = json.dumps(identifiers, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def conditioned_slot_ids(
    operation: str, target: str, scope: str
) -> tuple[int, int, int]:
    """Encode one declared editor intervention and reject partial/illegal NULLs."""

    try:
        operation_id = OPERATION_TO_ID[operation]
        target_id = TARGET_TO_ID[target]
        scope_id = SCOPE_TO_ID[scope]
    except KeyError as error:
        raise LearningContractError(
            "editor intervention contains an unknown intent slot"
        ) from error
    all_null = operation == target == scope == "NULL"
    operation_only = (
        operation in {"ADD", "REMOVE"}
        and target == "NULL"
        and scope == "NULL"
    )
    legal_joint = (
        operation in {"ADD", "REMOVE"}
        and target in {"SAME", "NEW"}
        and scope in {"LOCAL", "COMPLETE"}
        and not (target == "NEW" and scope == "LOCAL")
    )
    if not (all_null or operation_only or legal_joint):
        raise LearningContractError(
            "editor intervention resolved an illegal operation/target/scope tuple"
        )
    return operation_id, target_id, scope_id


def dataset_condition_for_inference(condition: str) -> str:
    """Keep visual ablations in the dataset; apply slot interventions once here."""

    if condition in {
        "same_weight_wrong_scope",
        "same_weight_shuffled",
        "predicted_slots",
        "wrong_operation_OOD",
    }:
        return "scribble_plus_intent"
    return condition


def validate_editor_architecture_receipt(
    *, checkpoint, architecture_contract, model_parameter_count: int
) -> int:
    if checkpoint.get("architecture_id") != EDITOR_PRIMARY_ARCHITECTURE_ID:
        raise LearningContractError(
            "editor checkpoint is not the frozen simple operation-conditioned architecture"
        )
    if checkpoint.get("architecture_id") != architecture_contract["primary_architecture_id"]:
        raise LearningContractError(
            "editor checkpoint architecture differs from the frozen config"
        )
    if checkpoint.get("deferred_fusion_ablations") != architecture_contract[
        "deferred_ablations"
    ] or checkpoint.get("fusion_plan") != architecture_contract["fusion_plan"]:
        raise LearningContractError("editor checkpoint deferred fusion plan differs from config")
    parameter_count = checkpoint.get("parameter_count")
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count <= 0
    ):
        raise LearningContractError(
            "editor checkpoint omits positive trainable parameter_count"
        )
    if parameter_count != model_parameter_count:
        raise LearningContractError(
            "editor checkpoint parameter_count does not match architecture"
        )
    return parameter_count


def validate_predicted_slots_receipt(
    *,
    p2t_checkpoint,
    p2t_checkpoint_sha256,
    p2t_metrics,
    predictions_rows,
    predictions_sha256,
    expected_episode_ids,
    visible_by_episode,
    partition,
    manifest_sha256,
    experiment_config_sha256,
    training_contract,
    evaluation_contract,
):
    if evaluation_contract["primary_end_to_end_arm"] != "full":
        raise LearningContractError(
            "predicted_slots requires p2t.primary_end_to_end_arm to be full"
        )
    if p2t_checkpoint.get("schema_version") != P2T_CHECKPOINT_SCHEMA:
        raise LearningContractError("unsupported P2T checkpoint schema")
    if p2t_checkpoint.get("architecture_id") != P2T_PRIMARY_ARCHITECTURE_ID:
        raise LearningContractError(
            "predicted_slots requires the frozen primary P2T architecture_id"
        )
    training_manifest_sha256 = p2t_checkpoint.get("manifest_sha256")
    if (
        not isinstance(training_manifest_sha256, str)
        or len(training_manifest_sha256) != 64
    ):
        raise LearningContractError("P2T checkpoint omits its training manifest hash")
    if p2t_checkpoint.get("experiment_config_sha256") != experiment_config_sha256:
        raise LearningContractError(
            "P2T checkpoint does not bind the experiment config"
        )
    if (
        p2t_checkpoint.get("input_ablation") != "full"
        or p2t_checkpoint.get("arm_role") != "primary"
    ):
        raise LearningContractError(
            "predicted_slots requires the primary full-input P2T checkpoint"
        )
    if p2t_checkpoint.get("seed") not in training_contract["seeds"]:
        raise LearningContractError(
            "P2T checkpoint seed is outside the frozen registry"
        )
    if (
        p2t_checkpoint.get("checkpoint_criterion")
        != training_contract["checkpoint_criterion"]
    ):
        raise LearningContractError("P2T checkpoint criterion differs from config")
    hyperparameters = p2t_checkpoint.get("hyperparameters")
    training_keys = (
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "joint_loss_weight",
        "operation_loss_weight",
        "target_loss_weight",
        "scope_loss_weight",
    )
    if not isinstance(hyperparameters, dict) or any(
        hyperparameters.get(key) != training_contract[key] for key in training_keys
    ):
        raise LearningContractError("P2T checkpoint hyperparameters differ from config")

    expected_metrics = {
        "schema_version": P2T_METRICS_SCHEMA,
        "partition": partition,
        "checkpoint_schema_version": P2T_CHECKPOINT_SCHEMA,
        "checkpoint_sha256": p2t_checkpoint_sha256,
        "checkpoint_criterion": training_contract["checkpoint_criterion"],
        "checkpoint_seed": p2t_checkpoint["seed"],
        "manifest_sha256": manifest_sha256,
        "training_manifest_sha256": training_manifest_sha256,
        "inference_manifest_sha256": manifest_sha256,
        "experiment_config_sha256": experiment_config_sha256,
        "predictions_sha256": predictions_sha256,
        "prediction_schema_version": P2T_PREDICTION_SCHEMA,
        "input_ablation": "full",
        "architecture_id": P2T_PRIMARY_ARCHITECTURE_ID,
        "arm_role": "primary",
        "legal_goals": list(LEGAL_GOALS),
    }
    if any(p2t_metrics.get(key) != value for key, value in expected_metrics.items()):
        raise LearningContractError(
            "P2T metrics receipt does not bind checkpoint/predictions/partition/config"
        )
    if p2t_metrics.get("episode_count") != len(predictions_rows):
        raise LearningContractError(
            "P2T metrics episode_count does not match predictions"
        )
    if len(predictions_rows) != len(expected_episode_ids):
        raise LearningContractError(
            "P2T prediction episode count differs from partition"
        )
    confusion = p2t_metrics.get("confusion_diagnostic")
    if not isinstance(confusion, dict) or not {
        "operation",
        "target",
        "scope",
        "joint_goal",
    }.issubset(confusion):
        raise LearningContractError("P2T metrics omit confusion diagnostics")
    per_strategy = p2t_metrics.get("per_strategy_diagnostic")
    if not isinstance(per_strategy, dict):
        raise LearningContractError("P2T metrics omit per-strategy diagnostics")

    observed_episode_ids = set()
    observed_strategies = set()
    for row in predictions_rows:
        episode = str(row.get("episode_id") or "")
        if not episode or episode in observed_episode_ids:
            raise LearningContractError(
                "duplicate or missing episode_id in P2T predictions"
            )
        observed_episode_ids.add(episode)
        expected_row = {
            "schema_version": P2T_PREDICTION_SCHEMA,
            "partition": partition,
            "checkpoint_schema_version": P2T_CHECKPOINT_SCHEMA,
            "checkpoint_sha256": p2t_checkpoint_sha256,
            "learning_manifest_sha256": manifest_sha256,
            "training_manifest_sha256": training_manifest_sha256,
            "inference_manifest_sha256": manifest_sha256,
            "experiment_config_sha256": experiment_config_sha256,
            "input_ablation": "full",
            "architecture_id": P2T_PRIMARY_ARCHITECTURE_ID,
            "arm_role": "primary",
            "visible_npz_sha256": visible_by_episode.get(episode),
        }
        if any(row.get(key) != value for key, value in expected_row.items()):
            raise LearningContractError("P2T prediction row provenance mismatch")
        if LEGAL_PREDICTIONS.get(row.get("goal")) != (
            row.get("operation"),
            row.get("target"),
            row.get("scope"),
        ):
            raise LearningContractError(
                "P2T prediction has an illegal or inconsistent goal/slot pair"
            )
        strategy = str(row.get("strategy") or "")
        if not strategy:
            raise LearningContractError("P2T prediction omits scribble strategy")
        observed_strategies.add(strategy)
    if observed_episode_ids != set(expected_episode_ids):
        raise LearningContractError(
            "P2T predictions must exactly cover the selected partition"
        )
    if observed_strategies != set(per_strategy):
        raise LearningContractError(
            "P2T per-strategy diagnostics do not cover prediction strategies"
        )
    if any(
        not isinstance(per_strategy[strategy], dict)
        or per_strategy[strategy].get("episode_count")
        != sum(str(row.get("strategy")) == strategy for row in predictions_rows)
        for strategy in observed_strategies
    ):
        raise LearningContractError(
            "P2T per-strategy episode counts do not match predictions"
        )


def validate_intervention_receipt(
    *,
    intervention_rows,
    source_rows,
    partition,
    manifest_sha256,
    experiment_config_sha256,
    intervention_contract,
):
    source_by_episode = {str(row["episode_id"]): row for row in source_rows}
    expected_episode_ids = set(source_by_episode)
    observed_recipients = set()
    observed_donors = set()
    for row in intervention_rows:
        episode = str(row.get("episode_id") or "")
        donor = str(row.get("source_episode_id") or "")
        if episode in observed_recipients or not episode:
            raise LearningContractError("duplicate or missing intervention recipient")
        if donor in observed_donors or not donor:
            raise LearningContractError(
                "intervention donors must be unique and non-empty"
            )
        observed_recipients.add(episode)
        observed_donors.add(donor)
        if episode not in source_by_episode or donor not in source_by_episode:
            raise LearningContractError(
                "intervention donor/recipient is outside partition"
            )
        if episode == donor:
            raise LearningContractError("intervention manifest is not a derangement")
        expected_provenance = {
            "schema_version": INTERVENTION_SCHEMA,
            "algorithm": intervention_contract["shuffle_algorithm"],
            "seed": intervention_contract["shuffle_seed"],
            "partition": partition,
            "permutation_size": len(source_rows),
            "source_manifest_sha256": manifest_sha256,
            "experiment_config_sha256": experiment_config_sha256,
        }
        if any(row.get(key) != value for key, value in expected_provenance.items()):
            raise LearningContractError(
                "intervention provenance differs from frozen config"
            )
        recipient_row = source_by_episode[episode]
        donor_row = source_by_episode[donor]
        recipient_joint = (
            str(recipient_row["operation"]),
            str(recipient_row["target"]),
            str(recipient_row["scope"]),
        )
        donor_joint = (
            str(donor_row["operation"]),
            str(donor_row["target"]),
            str(donor_row["scope"]),
        )
        if (
            row.get("original_operation"),
            row.get("original_target"),
            row.get("original_scope"),
        ) != recipient_joint:
            raise LearningContractError(
                "intervention original slots differ from learning manifest"
            )
        if (
            row.get("operation"),
            row.get("target"),
            row.get("scope"),
        ) != donor_joint:
            raise LearningContractError(
                "intervention slots do not match declared donor"
            )
        if donor_joint == recipient_joint or row.get("changed") is not True:
            raise LearningContractError(
                "intervention did not change the joint slot label"
            )
        donor_visible_sha256 = str(donor_row.get("visible_sha256") or "")
        if not donor_visible_sha256:
            raise LearningContractError("learning manifest donor omits visible hash")
        if row.get("source_visible_npz_sha256") != donor_visible_sha256:
            raise LearningContractError("intervention donor visible hash mismatch")
    if observed_recipients != expected_episode_ids:
        raise LearningContractError("intervention recipients do not cover partition")
    if observed_donors != expected_episode_ids:
        raise LearningContractError(
            "intervention donors are not a without-replacement permutation"
        )
    original_counts = Counter(
        (
            str(row["operation"]),
            str(row["target"]),
            str(row["scope"]),
        )
        for row in source_rows
    )
    donor_counts = Counter(
        (
            str(row["operation"]),
            str(row["target"]),
            str(row["scope"]),
        )
        for row in intervention_rows
    )
    if original_counts != donor_counts:
        raise LearningContractError("intervention changed joint-label marginals")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        help=(
            "tensor manifest used to train the checkpoint; defaults to --manifest "
            "for within-lane inference"
        ),
    )
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=["val", "test"], required=True)
    parser.add_argument("--condition", choices=EDITOR_CONDITIONS, required=True)
    parser.add_argument("--p2t-checkpoint", type=Path)
    parser.add_argument("--p2t-predictions", type=Path)
    parser.add_argument("--p2t-metrics", type=Path)
    parser.add_argument("--intervention-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--device", default="cuda")
    add_leaf_test_access_arguments(parser)
    args = parser.parse_args()
    training_manifest = args.training_manifest or args.manifest
    try:
        test_access = enforce_partition_access(
            args.partition,
            receipt_path=args.test_access_receipt,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=args.run_root,
            output_paths=(args.output_dir, args.output_manifest),
        )
    except TestAccessError as error:
        parser.error(str(error))
    test_access_sha256 = (
        None if test_access is None else str(test_access["receipt_sha256"])
    )
    if args.output_dir.exists() or args.output_manifest.exists():
        parser.error("output path already exists")
    for input_path, label in (
        (args.manifest, "inference manifest"),
        (training_manifest, "training manifest"),
        (args.learning_split, "learning split"),
        (args.experiment_config, "experiment config"),
        (args.checkpoint, "editor checkpoint"),
    ):
        if input_path.is_symlink() or not input_path.is_file():
            parser.error(f"{label} must be a non-symlink regular file")
    try:
        experiment_config = load_experiment_config(args.experiment_config)
        editor_training = load_training_contract(experiment_config, "editor")
        architecture_contract = load_editor_architecture_contract(experiment_config)
        p2t_training = load_training_contract(experiment_config, "p2t")
        p2t_evaluation = load_p2t_evaluation_contract(experiment_config)
        intervention_contract = load_intent_intervention_contract(experiment_config)
        frozen_threshold = float(experiment_config["editor"]["inference_threshold"])
    except (LearningContractError, KeyError, TypeError, ValueError) as error:
        parser.error("invalid frozen experiment config: %s" % error)
    try:
        manifest_rows = load_jsonl(args.manifest)
        training_manifest_rows = (
            manifest_rows
            if training_manifest == args.manifest
            else load_jsonl(training_manifest)
        )
        validate_manifest_rows_against_frozen_learning_split(
            manifest_rows,
            args.learning_split,
            require_episode_id=True,
            allowed_partitions={"train", "val", "test"},
        )
        validate_manifest_rows_against_frozen_learning_split(
            training_manifest_rows,
            args.learning_split,
            require_episode_id=True,
            allowed_partitions={"train", "val", "test"},
        )
    except LearningContractError as error:
        parser.error(str(error))
    if args.threshold is not None and not np.isclose(
        args.threshold, frozen_threshold, atol=0, rtol=0
    ):
        parser.error("--threshold differs from the frozen experiment config")
    args.threshold = frozen_threshold
    if not 0.0 < args.threshold < 1.0:
        parser.error("threshold must be strictly between 0 and 1")
    p2t_inputs = (args.p2t_checkpoint, args.p2t_predictions, args.p2t_metrics)
    if args.condition != "predicted_slots" and any(
        value is not None for value in p2t_inputs
    ):
        parser.error("P2T receipt inputs are valid only for predicted_slots")
    if args.condition == "same_weight_shuffled" and args.intervention_manifest is None:
        parser.error("same_weight_shuffled requires --intervention-manifest")
    if (
        args.condition != "same_weight_shuffled"
        and args.intervention_manifest is not None
    ):
        parser.error("--intervention-manifest is valid only for same_weight_shuffled")
    predicted = {}
    predicted_rows = []
    if args.condition == "predicted_slots":
        if any(value is None for value in p2t_inputs):
            parser.error(
                "predicted_slots requires --p2t-checkpoint, --p2t-predictions "
                "and --p2t-metrics"
            )
        predicted_rows = load_jsonl(args.p2t_predictions)
        predicted = {row["episode_id"]: row for row in predicted_rows}
        if len(predicted) != len(predicted_rows):
            parser.error("duplicate episode_id in P2T predictions")
    interventions = {}
    intervention_rows = []
    if args.condition == "same_weight_shuffled":
        intervention_rows = load_jsonl(args.intervention_manifest)
        interventions = {row["episode_id"]: row for row in intervention_rows}
        if len(interventions) != len(intervention_rows):
            parser.error("duplicate episode_id in intervention manifest")
    manifest_sha256 = sha256_file(args.manifest)
    training_manifest_sha256 = sha256_file(training_manifest)
    experiment_config_sha256 = sha256_file(args.experiment_config)
    editor_checkpoint_sha256 = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != EDITOR_CHECKPOINT_SCHEMA:
        parser.error("unsupported editor checkpoint schema")
    try:
        validate_editor_checkpoint_training_binding(
            checkpoint=checkpoint,
            training_manifest=training_manifest,
            training_manifest_sha256=training_manifest_sha256,
            learning_split=args.learning_split,
            learning_split_sha256=sha256_file(args.learning_split),
        )
    except LearningContractError as error:
        parser.error(str(error))
    if checkpoint.get("experiment_config_sha256") != experiment_config_sha256:
        parser.error("checkpoint was not trained against this experiment config")
    if checkpoint.get("seed") not in editor_training["seeds"]:
        parser.error("editor checkpoint seed is outside the frozen registry")
    if (
        checkpoint.get("checkpoint_criterion")
        != editor_training["checkpoint_criterion"]
    ):
        parser.error("editor checkpoint criterion differs from the frozen config")
    checkpoint_hyperparameters = checkpoint.get("hyperparameters")
    if not isinstance(checkpoint_hyperparameters, dict) or any(
        checkpoint_hyperparameters.get(key) != editor_training[key]
        for key in (
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "optimizer",
        )
    ):
        parser.error("editor checkpoint hyperparameters differ from the frozen config")
    if checkpoint.get("input_ablation") != "full":
        parser.error("editor checkpoint must use the full frozen visual input")
    checkpoint_condition = checkpoint.get("condition")
    expected_checkpoint_condition = expected_editor_checkpoint_condition(args.condition)
    if checkpoint_condition != expected_checkpoint_condition:
        parser.error(
            "condition %s requires checkpoint trained as %s; observed %s"
            % (args.condition, expected_checkpoint_condition, checkpoint_condition)
        )
    if args.condition in (
        "visual_state_only",
        "spatial_only",
        "scribble_plus_operation",
        "same_weight_NULL",
    ):
        semantics = checkpoint.get("special_condition_semantics")
        if not isinstance(semantics, dict):
            parser.error("editor checkpoint omits deterministic special semantics")
        if semantics.get("NULL") != NULL_SEMANTICS:
            parser.error(
                "editor checkpoint NULL semantics are not deterministic bypass"
            )
        if semantics.get("OPERATION_ONLY") != intervention_contract[
            "operation_only_semantics"
        ]:
            parser.error(
                "editor checkpoint operation-only semantics are not deterministic"
            )
    dataset = EpisodeDataset(
        args.manifest,
        args.partition,
        editor_condition=dataset_condition_for_inference(args.condition),
        load_evaluation=False,
    )
    try:
        eligible_rows, ineligible_rows = partition_rows_for_condition(
            args.condition, dataset.rows
        )
        dataset.rows = eligible_rows
        ineligible_episode_set_sha256 = episode_id_set_sha256(ineligible_rows)
    except LearningContractError as error:
        parser.error(str(error))
    learning_split_sha256 = sha256_file(args.learning_split)
    if {row.get("learning_split_sha256") for row in dataset.rows} != {
        learning_split_sha256
    }:
        parser.error("editor manifest differs from the receipt-bound learning split")
    expected_episode_ids = {str(row["episode_id"]) for row in dataset.rows}
    manifest_row_by_episode = {
        str(row["episode_id"]): row for row in dataset.rows
    }
    p2t_checkpoint_sha256 = None
    p2t_architecture_id = None
    p2t_predictions_sha256 = None
    p2t_metrics_sha256 = None
    intervention_manifest_sha256 = None
    if args.condition == "predicted_slots":
        with args.p2t_metrics.open("r", encoding="utf-8") as stream:
            p2t_metrics = json.load(stream)
        if not isinstance(p2t_metrics, dict):
            parser.error("P2T metrics receipt must be a JSON object")
        p2t_checkpoint_sha256 = sha256_file(args.p2t_checkpoint)
        p2t_predictions_sha256 = sha256_file(args.p2t_predictions)
        p2t_metrics_sha256 = sha256_file(args.p2t_metrics)
        p2t_checkpoint = torch.load(
            args.p2t_checkpoint, map_location="cpu", weights_only=True
        )
        visible_by_episode = {
            str(row["episode_id"]): str(row["visible_sha256"]) for row in dataset.rows
        }
        try:
            validate_predicted_slots_receipt(
                p2t_checkpoint=p2t_checkpoint,
                p2t_checkpoint_sha256=p2t_checkpoint_sha256,
                p2t_metrics=p2t_metrics,
                predictions_rows=predicted_rows,
                predictions_sha256=p2t_predictions_sha256,
                expected_episode_ids=expected_episode_ids,
                visible_by_episode=visible_by_episode,
                partition=args.partition,
                manifest_sha256=manifest_sha256,
                experiment_config_sha256=experiment_config_sha256,
                training_contract=p2t_training,
                evaluation_contract=p2t_evaluation,
            )
            p2t_architecture_id = p2t_checkpoint["architecture_id"]
        except LearningContractError as error:
            parser.error(str(error))
    if args.condition == "same_weight_shuffled":
        intervention_manifest_sha256 = sha256_file(args.intervention_manifest)
        try:
            validate_intervention_receipt(
                intervention_rows=intervention_rows,
                source_rows=dataset.rows,
                partition=args.partition,
                manifest_sha256=manifest_sha256,
                experiment_config_sha256=experiment_config_sha256,
                intervention_contract=intervention_contract,
            )
        except LearningContractError as error:
            parser.error(str(error))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device(args.device)
    model = ResidualEditorUNet2D().to(device)
    try:
        parameter_count = validate_editor_architecture_receipt(
            checkpoint=checkpoint,
            architecture_contract=architecture_contract,
            model_parameter_count=model.parameter_count(),
        )
    except LearningContractError as error:
        parser.error(str(error))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".%s." % args.output_dir.name,
            suffix=".tmp",
            dir=str(args.output_dir.parent),
        )
    )
    rows = []
    conditioning_semantics = {
        "same_weight_NULL": NULL_SEMANTICS,
        "scribble_plus_operation": intervention_contract[
            "operation_only_semantics"
        ],
        "same_weight_wrong_scope": intervention_contract["wrong_scope_semantics"],
        "same_weight_shuffled": intervention_contract["shuffle_algorithm"],
        "wrong_operation_OOD": "flip-ADD-REMOVE-preserve-target-scope-stress-only",
    }.get(args.condition, "declared-slot-condition")
    try:
        with torch.no_grad():
            for batch in loader:
                episode = batch["episode_id"][0]
                source_row = manifest_row_by_episode[episode]
                gold_operation = str(source_row.get("operation") or "")
                intervention = interventions.get(episode, {})
                prediction_intent = predicted.get(episode, {})
                execution_operation = execution_operation_for_condition(
                    args.condition,
                    gold_operation=gold_operation,
                    predicted=prediction_intent,
                    intervention=intervention,
                )
                conditioned_operation, conditioned_target, conditioned_scope = (
                    conditioned_slots_for_condition(
                        args.condition,
                        gold_operation=gold_operation,
                        gold_target=str(source_row.get("target") or ""),
                        gold_scope=str(source_row.get("scope") or ""),
                        predicted=prediction_intent,
                        intervention=intervention,
                    )
                )
                operation_id, target_id, scope_id = conditioned_slot_ids(
                    conditioned_operation,
                    conditioned_target,
                    conditioned_scope,
                )
                logits = model(
                    batch["visual"].to(device),
                    torch.tensor([operation_id], device=device),
                    torch.tensor([target_id], device=device),
                    torch.tensor([scope_id], device=device),
                )
                delta, corrected = apply_operation_delta(
                    logits,
                    batch["m0"].to(device),
                    [execution_operation],
                    args.threshold,
                )
                staged_output = staging_dir / (episode + ".npz")
                final_output = args.output_dir / (episode + ".npz")
                cue = (
                    batch["cue_fg"][0, 0].numpy()
                    - batch["cue_bg"][0, 0].numpy()
                ).astype(np.int8)
                np.savez_compressed(
                    str(staged_output),
                    delta=delta[0, 0].cpu().numpy().astype(np.uint8),
                    m1=corrected[0, 0].cpu().numpy().astype(np.uint8),
                    m0=batch["m0"][0, 0].numpy().astype(np.uint8),
                    cue=cue,
                    scribble=batch["scribble"][0, 0].numpy().astype(np.uint8),
                    spacing_xy=batch["spacing_xy"][0].numpy(),
                )
                prediction_sha256 = sha256_file(staged_output)
                rows.append(
                    {
                        "episode_id": episode,
                        "patient_id": batch["patient_id"][0],
                        "condition": args.condition,
                        "conditioning_semantics": conditioning_semantics,
                        "gold_operation": gold_operation,
                        "gold_target": str(source_row.get("target") or ""),
                        "gold_scope": str(source_row.get("scope") or ""),
                        "execution_operation": execution_operation,
                        "conditioning_operation": conditioned_operation,
                        "conditioning_target": conditioned_target,
                        "conditioning_scope": conditioned_scope,
                        "execution_operation_matches_gold": (
                            execution_operation == gold_operation
                        ),
                        "conditioning_operation_matches_gold": (
                            conditioned_operation == gold_operation
                        ),
                        "condition_eligibility": (
                            WRONG_SCOPE_ELIGIBILITY
                            if args.condition == "same_weight_wrong_scope"
                            else "ALL_LEGAL_EPISODES"
                        ),
                        "ineligible_episode_count": len(ineligible_rows),
                        "ineligible_episode_set_sha256": (
                            ineligible_episode_set_sha256
                        ),
                        "ood_stress_only": args.condition == "wrong_operation_OOD",
                        "checkpoint": str(args.checkpoint.resolve()),
                        "checkpoint_schema_version": EDITOR_CHECKPOINT_SCHEMA,
                        "checkpoint_sha256": editor_checkpoint_sha256,
                        "checkpoint_status": EDITOR_TRAINING_STATUS,
                        "learning_manifest_sha256": manifest_sha256,
                        "learning_manifest": str(args.manifest.resolve()),
                        "training_manifest_sha256": training_manifest_sha256,
                        "training_manifest": str(training_manifest.resolve()),
                        "inference_manifest_sha256": manifest_sha256,
                        "inference_manifest": str(args.manifest.resolve()),
                        "experiment_config_sha256": experiment_config_sha256,
                        "learning_split_sha256": learning_split_sha256,
                        "test_access_receipt_sha256": test_access_sha256,
                        "partition": args.partition,
                        "checkpoint_condition": checkpoint_condition,
                        "architecture_id": model.architecture_id,
                        "deferred_fusion_ablations": checkpoint[
                            "deferred_fusion_ablations"
                        ],
                        "parameter_count": parameter_count,
                        "threshold": args.threshold,
                        "prediction_npz": str(final_output.resolve()),
                        "prediction_npz_sha256": prediction_sha256,
                        "prediction_manifest": str(args.output_manifest.resolve()),
                        "evaluation_npz": batch["evaluation_npz"][0],
                        "evaluation_npz_sha256": batch["evaluation_sha256"][0],
                        "visible_npz": str(
                            Path(
                                str(manifest_row_by_episode[episode]["visible_npz"])
                            ).resolve()
                        ),
                        "visible_npz_sha256": batch["visible_sha256"][0],
                        "p2t_checkpoint_sha256": p2t_checkpoint_sha256,
                        "p2t_architecture_id": p2t_architecture_id,
                        "p2t_input_ablation": (
                            "full" if args.condition == "predicted_slots" else None
                        ),
                        "p2t_predictions_sha256": (p2t_predictions_sha256),
                        "p2t_metrics_sha256": (p2t_metrics_sha256),
                        "intervention_manifest_sha256": (intervention_manifest_sha256),
                        "intervention_source_episode_id": intervention.get(
                            "source_episode_id"
                        ),
                    }
                )
        os.rename(str(staging_dir), str(args.output_dir))
        try:
            write_bytes_bundle_exclusive({args.output_manifest: encode_jsonl(rows)})
        except Exception:
            shutil.rmtree(args.output_dir)
            raise
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    print(
        json.dumps(
            {"condition": args.condition, "predictions": len(rows)}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
