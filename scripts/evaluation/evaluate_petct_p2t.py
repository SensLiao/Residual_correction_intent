#!/usr/bin/env python3
"""Evaluate P2T on a frozen partition and write predictions plus joint metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import (  # noqa: E402
    EpisodeDataset,
    LearningContractError,
    P2T_CHECKPOINT_SCHEMA,
    P2T_METRICS_SCHEMA,
    P2T_PREDICTION_SCHEMA,
    confusion_matrix_diagnostic,
    decode_flat_baseline,
    encode_json,
    encode_jsonl,
    load_experiment_config,
    load_jsonl,
    load_p2t_evaluation_contract,
    load_training_contract,
    macro_f1,
    patient_balanced_macro_f1,
    patient_cluster_bootstrap_macro_f1,
    sha256_file,
    sha256_bytes,
    validate_frozen_override,
    validate_manifest_rows_against_frozen_learning_split,
    write_bytes_bundle_exclusive,
)
from common.petct_models import (  # noqa: E402
    LEGAL_GOALS,
    LEGAL_GOAL_SLOTS,
    P2T_ARCHITECTURE_IDS,
    P2T_PRIMARY_ARCHITECTURE_ID,
    build_p2t_model,
    decode_joint_goal,
    p2t_architecture_contract,
    validate_p2t_architecture_selection,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)
from common.petct_mainline_lineage import (  # noqa: E402
    LineageContractError,
    validate_r13_data_ready,
)


OPERATIONS = ("ADD", "REMOVE")
TARGETS = ("SAME", "NEW")
SCOPES = ("LOCAL", "COMPLETE")
P2T_PAIRED_EVALUATION_ROW_SCHEMA = "PETCT-P2T-PAIRED-EVAL-ROW-v2.0"
P2T_RELEASE_PREDICTION_FIELDS = frozenset(
    {
        "schema_version",
        "episode_id",
        "patient_id",
        "partition",
        "strategy",
        "goal",
        "operation",
        "target",
        "scope",
        "input_ablation",
        "architecture_id",
        "arm_role",
        "visible_npz_sha256",
        "checkpoint_schema_version",
        "checkpoint_sha256",
        "learning_manifest_sha256",
        "training_manifest_sha256",
        "inference_manifest_sha256",
        "experiment_config_sha256",
        "learning_split_sha256",
        "test_access_receipt_sha256",
    }
)
decode_legal_joint = decode_joint_goal


def _validate_prediction_rows_for_release(
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Fail closed if a future edit leaks truth or evaluation-side paths."""

    for index, row in enumerate(rows):
        fields = set(row)
        if fields != P2T_RELEASE_PREDICTION_FIELDS:
            missing = sorted(P2T_RELEASE_PREDICTION_FIELDS - fields)
            extra = sorted(fields - P2T_RELEASE_PREDICTION_FIELDS)
            raise RuntimeError(
                "P2T release prediction field allowlist mismatch at row "
                f"{index}: missing={missing}, extra={extra}"
            )
        forbidden = sorted(
            field
            for field in fields
            if field.lower().startswith("gold_")
            or field.lower().endswith("_true")
            or field.lower() == "gt"
            or field.lower().startswith("gt_")
            or ("evaluation" in field.lower() and "path" in field.lower())
        )
        if forbidden:
            raise RuntimeError(
                f"P2T release prediction contains truth/evaluation fields: {forbidden}"
            )


def _per_strategy_diagnostics(
    strategies,
    patient_ids,
    operation_true,
    operation_pred,
    target_true,
    target_pred,
    scope_true,
    scope_pred,
    joint_true,
    joint_pred,
):
    diagnostics = {}
    for strategy in sorted(set(strategies)):
        indices = [index for index, value in enumerate(strategies) if value == strategy]
        rt = [target_true[index] for index in indices]
        rp = [target_pred[index] for index in indices]
        st = [scope_true[index] for index in indices]
        sp = [scope_pred[index] for index in indices]
        jt = [joint_true[index] for index in indices]
        jp = [joint_pred[index] for index in indices]
        patients = [patient_ids[index] for index in indices]
        ot = [operation_true[index] for index in indices]
        op = [operation_pred[index] for index in indices]
        diagnostics[strategy] = {
            "episode_count": len(indices),
            "patient_count": len(set(patients)),
            "episode_level_macro_f1": {
                "operation": macro_f1(ot, op, [0, 1]),
                "target": macro_f1(rt, rp, [0, 1]),
                "scope": macro_f1(st, sp, [0, 1]),
                "joint_goal": macro_f1(jt, jp, list(range(6))),
            },
            "patient_balanced_macro_f1": {
                "operation": patient_balanced_macro_f1(ot, op, patients, [0, 1]),
                "target": patient_balanced_macro_f1(rt, rp, patients, [0, 1]),
                "scope": patient_balanced_macro_f1(st, sp, patients, [0, 1]),
                "joint_goal": patient_balanced_macro_f1(
                    jt, jp, patients, list(range(6))
                ),
            },
            "joint_confusion_matrix": confusion_matrix_diagnostic(
                jt, jp, list(range(6)), LEGAL_GOALS
            ),
        }
    return diagnostics


def _per_goal_diagnostics(joint_true, joint_pred):
    diagnostics = {}
    for goal_id, goal in enumerate(LEGAL_GOALS):
        indices = [index for index, value in enumerate(joint_true) if value == goal_id]
        diagnostics[goal] = {
            "gold_episode_count": len(indices),
            "correct_episode_count": sum(
                joint_pred[index] == goal_id for index in indices
            ),
            "recall": (
                0.0
                if not indices
                else sum(joint_pred[index] == goal_id for index in indices)
                / len(indices)
            ),
        }
    return diagnostics


def accepted_experiment_config_shas(
    config_sha256: str, config: dict
) -> frozenset:
    """Config digests a manifest row may carry: the config itself and, when the
    config is a derived child (v2.1 + operation_control arm, D-2026-08-06-02),
    its frozen parent digest. Manifests are frozen under the parent config;
    deriving a child must not orphan them (D-2026-08-09-01)."""
    accepted = {config_sha256}
    parent_sha = (config.get("derived_from") or {}).get("parent_sha256")
    if isinstance(parent_sha, str) and parent_sha:
        accepted.add(parent_sha)
    return frozenset(accepted)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        help=(
            "tensor manifest used to train the checkpoint; defaults to --manifest "
            "for within-lane evaluation"
        ),
    )
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--r13-data-ready", type=Path, required=True)
    parser.add_argument("--partition", choices=["val", "test"], required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--paired-evaluation-rows",
        type=Path,
        required=True,
        help=(
            "truth-bearing controlled-evaluation JSONL; kept separate from the "
            "normal prediction manifest"
        ),
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--architecture-id",
        choices=P2T_ARCHITECTURE_IDS,
        help="optional assertion against the checkpoint architecture_id",
    )
    add_leaf_test_access_arguments(parser)
    parser.add_argument(
        "--bootstrap-seed", type=int, help="optional frozen-config assertion"
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, help="optional frozen-config assertion"
    )
    args = parser.parse_args()
    training_manifest = args.training_manifest or args.manifest
    for input_path, label in (
        (args.manifest, "inference manifest"),
        (training_manifest, "training manifest"),
        (args.experiment_config, "experiment config"),
        (args.learning_split, "learning split"),
        (args.checkpoint, "checkpoint"),
        (args.r13_data_ready, "R13 data-ready receipt"),
    ):
        if input_path.is_symlink() or not input_path.is_file():
            parser.error(f"{label} must be a non-symlink regular file")
    try:
        data_ready = validate_r13_data_ready(
            args.r13_data_ready,
            rich_tensor_manifest=training_manifest,
            learning_split=args.learning_split,
            experiment_config=args.experiment_config,
        )
    except LineageContractError as error:
        parser.error(str(error))
    try:
        test_access = enforce_partition_access(
            args.partition,
            receipt_path=args.test_access_receipt,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=args.run_root,
            output_paths=(
                args.predictions,
                args.paired_evaluation_rows,
                args.metrics,
            ),
        )
    except TestAccessError as error:
        parser.error(str(error))
    test_access_sha256 = (
        None if test_access is None else str(test_access["receipt_sha256"])
    )
    if any(
        path.exists() or path.is_symlink()
        for path in (args.predictions, args.paired_evaluation_rows, args.metrics)
    ):
        parser.error("output path already exists")
    try:
        config = load_experiment_config(args.experiment_config)
        training = load_training_contract(config, "p2t")
        evaluation = load_p2t_evaluation_contract(config)
        validate_frozen_override(
            "--bootstrap-seed", args.bootstrap_seed, evaluation["bootstrap_seed"]
        )
        validate_frozen_override(
            "--bootstrap-samples",
            args.bootstrap_samples,
            evaluation["bootstrap_samples"],
        )
    except LearningContractError as error:
        parser.error(str(error))
    experiment_config_sha256 = sha256_file(args.experiment_config)
    accepted_config_shas = accepted_experiment_config_shas(
        experiment_config_sha256, config
    )
    learning_split_sha256 = sha256_file(args.learning_split)
    try:
        inference_manifest_rows = load_jsonl(args.manifest)
        training_manifest_rows = (
            inference_manifest_rows
            if training_manifest == args.manifest
            else load_jsonl(training_manifest)
        )
        validate_manifest_rows_against_frozen_learning_split(
            inference_manifest_rows,
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
    for rows_label, manifest_rows in (
        ("inference", inference_manifest_rows),
        ("training", training_manifest_rows),
    ):
        if not manifest_rows:
            parser.error(f"{rows_label} manifest is empty")
        if any(
            row.get("learning_split_sha256") != learning_split_sha256
            for row in manifest_rows
        ):
            parser.error(
                f"{rows_label} manifest differs from the supplied frozen learning split"
            )
        if any(
            row.get("experiment_config_sha256") not in accepted_config_shas
            for row in manifest_rows
        ):
            parser.error(
                f"{rows_label} manifest differs from the supplied experiment config"
            )
    training_manifest_sha256 = sha256_file(training_manifest)
    inference_manifest_sha256 = sha256_file(args.manifest)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != P2T_CHECKPOINT_SCHEMA:
        parser.error("unsupported P2T checkpoint schema")
    if checkpoint.get("manifest_sha256") != training_manifest_sha256:
        parser.error(
            "checkpoint was not trained against the supplied training manifest hash"
        )
    checkpoint_manifest_path = Path(str(checkpoint.get("manifest") or ""))
    if checkpoint_manifest_path.is_symlink():
        parser.error("checkpoint training manifest path must not be a symlink")
    if checkpoint_manifest_path.resolve() != training_manifest.resolve():
        parser.error("checkpoint training manifest path differs from --training-manifest")
    if checkpoint.get("learning_split_sha256") != learning_split_sha256:
        parser.error("checkpoint was not trained against this frozen learning split")
    checkpoint_split_path = Path(str(checkpoint.get("learning_split") or ""))
    if checkpoint_split_path.is_symlink():
        parser.error("checkpoint learning split path must not be a symlink")
    if checkpoint_split_path.resolve() != args.learning_split.resolve():
        parser.error("checkpoint learning split path differs from --learning-split")
    if checkpoint.get("experiment_config_sha256") != experiment_config_sha256:
        parser.error("checkpoint was not trained against this experiment config")
    if (
        checkpoint.get("source_m0_lineage") != data_ready["source_m0_lineage"]
        or checkpoint.get("r13_data_ready_sha256")
        != sha256_file(args.r13_data_ready)
    ):
        parser.error("checkpoint is not bound to this R13 data-ready receipt")
    baseline_arm = checkpoint.get("baseline_arm")
    if baseline_arm not in ("J1", "J2"):
        parser.error("checkpoint baseline_arm must be J1 or J2")
    expected_baseline_weights = (
        {"joint": 1.0, "operation": 0.0, "target": 0.0, "scope": 0.0}
        if baseline_arm == "J1"
        else {"joint": 0.0, "operation": 1.0 / 3.0, "target": 1.0 / 3.0, "scope": 1.0 / 3.0}
    )
    if checkpoint.get("baseline_loss_weights") != expected_baseline_weights:
        parser.error("checkpoint baseline loss contract is invalid")
    input_ablation = checkpoint.get("input_ablation")
    if input_ablation not in evaluation["ablation_inputs"]:
        parser.error("checkpoint input ablation is not in the frozen config")
    # Narrow compatibility only: the checkpoint has already passed the exact
    # current-schema check above and was loaded with weights_only=True.  This is
    # not a promise to accept arbitrary historical checkpoint formats.
    architecture_id = checkpoint.get("architecture_id", P2T_PRIMARY_ARCHITECTURE_ID)
    if architecture_id not in P2T_ARCHITECTURE_IDS:
        parser.error("checkpoint P2T architecture_id is unsupported")
    if args.architecture_id is not None and args.architecture_id != architecture_id:
        parser.error("--architecture-id differs from the checkpoint architecture_id")
    try:
        expected_arm_role = validate_p2t_architecture_selection(
            config, architecture_id, input_ablation
        )
    except ValueError as error:
        parser.error(str(error))
    if checkpoint.get("arm_role") != expected_arm_role:
        parser.error("checkpoint arm role is inconsistent with its input ablation")
    if architecture_id != P2T_PRIMARY_ARCHITECTURE_ID:
        expected_architecture_contract = p2t_architecture_contract(
            architecture_id,
            use_relative_geometry=input_ablation != "no_relpos",
        )
        if checkpoint.get("architecture_contract") != expected_architecture_contract:
            parser.error("secondary checkpoint architecture contract is incomplete")
    if checkpoint.get("checkpoint_criterion") != training["checkpoint_criterion"]:
        parser.error("checkpoint criterion differs from the frozen config")
    if checkpoint.get("seed") not in training["seeds"]:
        parser.error("checkpoint seed is outside the frozen seed registry")
    checkpoint_hyperparameters = checkpoint.get("hyperparameters")
    expected_checkpoint_training = (
        {**training, "epochs": 1}
        if checkpoint.get("smoke_one_epoch") is True
        else training
    )
    if not isinstance(checkpoint_hyperparameters, dict) or any(
        checkpoint_hyperparameters.get(key) != expected_checkpoint_training[key]
        for key in (
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
    ):
        parser.error("checkpoint hyperparameters differ from the frozen config")
    dataset = EpisodeDataset(
        args.manifest, args.partition, input_ablation, load_evaluation=False
    )
    manifest_rows_by_episode = {
        str(row["episode_id"]): row for row in dataset.rows
    }
    if len(manifest_rows_by_episode) != len(dataset.rows):
        parser.error("evaluation manifest contains duplicate episode_id values")
    if {
        row.get("learning_split_sha256") for row in dataset.rows
    } != {learning_split_sha256}:
        parser.error("evaluation manifest differs from the receipt-bound learning split")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    device = torch.device(args.device)
    model = build_p2t_model(
        architecture_id, use_relative_geometry=input_ablation != "no_relpos"
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    recorded_parameter_count = checkpoint.get("parameter_count")
    if (
        recorded_parameter_count is not None
        and recorded_parameter_count != parameter_count
    ):
        parser.error("checkpoint parameter_count differs from its architecture")
    if (
        architecture_id != P2T_PRIMARY_ARCHITECTURE_ID
        and recorded_parameter_count != parameter_count
    ):
        parser.error("secondary checkpoint must record its exact parameter_count")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    rows = []
    paired_rows = []
    raw_illegal_predictions = []
    strategies = []
    patient_ids = []
    operation_true, operation_pred, target_true, target_pred, scope_true, scope_pred, joint_true, joint_pred = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    configured_strategies = config.get("scribble", {}).get("strategies")
    if not isinstance(configured_strategies, list) or not configured_strategies:
        parser.error("scribble.strategies must be frozen in the experiment config")
    with torch.no_grad():
        for batch in loader:
            output = model(batch["visual"].to(device), batch["spacing_xy"].to(device))
            decoded = decode_flat_baseline(output, baseline_arm)
            jp_tensor = decoded["joint_id"]
            op_tensor = decoded["operation_id"]
            rp_tensor = decoded["target_id"]
            sp_tensor = decoded["scope_id"]
            raw_scope_tensor = decoded["raw_scope_id"]
            raw_illegal_tensor = decoded["raw_illegal"]
            jp = jp_tensor.cpu().tolist()
            op = op_tensor.cpu().tolist()
            rp = rp_tensor.cpu().tolist()
            sp = sp_tensor.cpu().tolist()
            raw_sp = raw_scope_tensor.cpu().tolist()
            raw_illegal = raw_illegal_tensor.cpu().tolist()
            ot = batch["operation_gold"].tolist()
            rt = batch["target_gold"].tolist()
            st = batch["scope_gold"].tolist()
            for index, episode in enumerate(batch["episode_id"]):
                strategy = str(batch["strategy"][index])
                if strategy not in configured_strategies:
                    parser.error(
                        "episode %s has an unfrozen or missing scribble strategy"
                        % episode
                    )
                gold_slots = (ot[index], rt[index], st[index])
                try:
                    legal_gold = LEGAL_GOAL_SLOTS.index(gold_slots)
                except ValueError:
                    parser.error(f"episode {episode} has an illegal gold slot triple")
                rows.append(
                    {
                        "schema_version": P2T_PREDICTION_SCHEMA,
                        "episode_id": episode,
                        "patient_id": batch["patient_id"][index],
                        "partition": args.partition,
                        "strategy": strategy,
                        "goal": LEGAL_GOALS[jp[index]],
                        "operation": OPERATIONS[op[index]],
                        "target": TARGETS[rp[index]],
                        "scope": SCOPES[sp[index]],
                        "input_ablation": input_ablation,
                        "architecture_id": architecture_id,
                        "arm_role": expected_arm_role,
                        "visible_npz_sha256": batch["visible_sha256"][index],
                        "checkpoint_schema_version": P2T_CHECKPOINT_SCHEMA,
                        "checkpoint_sha256": checkpoint_sha256,
                        "learning_manifest_sha256": inference_manifest_sha256,
                        "training_manifest_sha256": training_manifest_sha256,
                        "inference_manifest_sha256": inference_manifest_sha256,
                        "experiment_config_sha256": experiment_config_sha256,
                        "learning_split_sha256": learning_split_sha256,
                        "test_access_receipt_sha256": test_access_sha256,
                    }
                )
                source_row = manifest_rows_by_episode[str(episode)]
                paired_rows.append(
                    {
                        "schema_version": P2T_PAIRED_EVALUATION_ROW_SCHEMA,
                        "episode_id": str(episode),
                        "patient_id": str(batch["patient_id"][index]),
                        "matched_state_group_id": source_row.get(
                            "matched_state_group_id"
                        ),
                        "partition": args.partition,
                        "strategy": strategy,
                        "gold_joint_id": int(legal_gold),
                        "predicted_joint_id": int(jp[index]),
                        "baseline_arm": baseline_arm,
                        "raw_predicted_scope_id": int(raw_sp[index]),
                        "slot_projection_applied": bool(raw_illegal[index]),
                        "gold_operation_id": int(ot[index]),
                        "predicted_operation_id": int(op[index]),
                        "checkpoint_seed": int(checkpoint["seed"]),
                        "input_ablation": input_ablation,
                        "architecture_id": architecture_id,
                        "checkpoint_sha256": checkpoint_sha256,
                        "learning_manifest_sha256": inference_manifest_sha256,
                        "training_manifest_sha256": training_manifest_sha256,
                        "visible_npz_sha256": str(batch["visible_sha256"][index]),
                        "evaluation_npz_sha256": str(
                            batch["evaluation_sha256"][index]
                        ),
                        "experiment_config_sha256": experiment_config_sha256,
                        "learning_split_sha256": learning_split_sha256,
                        "test_access_receipt_sha256": test_access_sha256,
                    }
                )
                raw_illegal_predictions.append(bool(raw_illegal[index]))
                strategies.append(strategy)
                patient_ids.append(str(batch["patient_id"][index]))
                operation_true.append(ot[index])
                operation_pred.append(op[index])
                target_true.append(rt[index])
                target_pred.append(rp[index])
                scope_true.append(st[index])
                scope_pred.append(sp[index])
                joint_true.append(legal_gold)
                joint_pred.append(jp[index])
    _validate_prediction_rows_for_release(rows)
    predictions_payload = encode_jsonl(rows)
    predictions_sha256 = sha256_bytes(predictions_payload)
    paired_rows_payload = encode_jsonl(paired_rows)
    paired_rows_sha256 = sha256_bytes(paired_rows_payload)
    metrics = {
        "schema_version": P2T_METRICS_SCHEMA,
        "partition": args.partition,
        "episode_count": len(rows),
        "checkpoint_schema_version": P2T_CHECKPOINT_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "source_m0_lineage": data_ready["source_m0_lineage"],
        "r13_data_ready_sha256": sha256_file(args.r13_data_ready),
        "checkpoint_criterion": training["checkpoint_criterion"],
        "checkpoint_seed": checkpoint["seed"],
        "smoke_one_epoch": checkpoint.get("smoke_one_epoch") is True,
        "baseline_arm": baseline_arm,
        "raw_invalid_slot_combination_count": int(sum(raw_illegal_predictions)),
        "raw_invalid_slot_combination_rate": (
            float(sum(raw_illegal_predictions) / len(raw_illegal_predictions))
            if raw_illegal_predictions else None
        ),
        # manifest_sha256 remains the manifest actually inferred on for
        # downstream editor compatibility.  The two explicit fields preserve
        # cross-lane provenance without pretending the checkpoint was trained
        # on natural OOF tensors.
        "manifest_sha256": inference_manifest_sha256,
        "training_manifest_sha256": training_manifest_sha256,
        "inference_manifest_sha256": inference_manifest_sha256,
        "experiment_config_sha256": experiment_config_sha256,
        "learning_split_sha256": learning_split_sha256,
        "test_access_receipt_sha256": test_access_sha256,
        "predictions_sha256": predictions_sha256,
        "prediction_schema_version": P2T_PREDICTION_SCHEMA,
        "paired_evaluation_rows_schema": P2T_PAIRED_EVALUATION_ROW_SCHEMA,
        "paired_evaluation_rows_sha256": paired_rows_sha256,
        "paired_evaluation_row_count": len(paired_rows),
        "input_ablation": input_ablation,
        "architecture_id": architecture_id,
        "parameter_count": parameter_count,
        "architecture_comparison_role": (
            "descriptive_secondary_architecture_baseline"
            if expected_arm_role == "secondary_architecture"
            else "frozen_primary_or_input_ablation"
        ),
        "arm_role": expected_arm_role,
        "legal_goals": list(LEGAL_GOALS),
        "primary_unit": "patient",
        "joint_goal_macro_f1": patient_cluster_bootstrap_macro_f1(
            joint_true,
            joint_pred,
            patient_ids,
            list(range(6)),
            seed=evaluation["bootstrap_seed"],
            samples=evaluation["bootstrap_samples"],
            alpha=evaluation["alpha"],
        ),
        "episode_level_diagnostic": {
            "operation_macro_f1": macro_f1(operation_true, operation_pred, [0, 1]),
            "target_macro_f1": macro_f1(target_true, target_pred, [0, 1]),
            "scope_macro_f1": macro_f1(scope_true, scope_pred, [0, 1]),
            "joint_goal_macro_f1": macro_f1(
                joint_true, joint_pred, list(range(6))
            ),
        },
        "patient_balanced_diagnostic": {
            "operation_macro_f1": patient_balanced_macro_f1(
                operation_true, operation_pred, patient_ids, [0, 1]
            ),
            "target_macro_f1": patient_balanced_macro_f1(
                target_true, target_pred, patient_ids, [0, 1]
            ),
            "scope_macro_f1": patient_balanced_macro_f1(
                scope_true, scope_pred, patient_ids, [0, 1]
            ),
            "joint_goal_macro_f1": patient_balanced_macro_f1(
                joint_true, joint_pred, patient_ids, list(range(6))
            ),
        },
        "confusion_diagnostic": {
            "operation": confusion_matrix_diagnostic(
                operation_true, operation_pred, [0, 1], OPERATIONS
            ),
            "target": confusion_matrix_diagnostic(
                target_true, target_pred, [0, 1], TARGETS
            ),
            "scope": confusion_matrix_diagnostic(
                scope_true, scope_pred, [0, 1], SCOPES
            ),
            "joint_goal": confusion_matrix_diagnostic(
                joint_true, joint_pred, list(range(6)), LEGAL_GOALS
            ),
        },
        "per_strategy_diagnostic": _per_strategy_diagnostics(
            strategies,
            patient_ids,
            operation_true,
            operation_pred,
            target_true,
            target_pred,
            scope_true,
            scope_pred,
            joint_true,
            joint_pred,
        ),
        "per_goal_diagnostic": _per_goal_diagnostics(joint_true, joint_pred),
        "evaluation_contract": {
            "bootstrap_seed": evaluation["bootstrap_seed"],
            "bootstrap_samples": evaluation["bootstrap_samples"],
            "alpha": evaluation["alpha"],
            "required_diagnostics": evaluation["required_diagnostics"],
        },
        "execution_required_for_values": True,
    }
    write_bytes_bundle_exclusive(
        {
            args.predictions: predictions_payload,
            args.paired_evaluation_rows: paired_rows_payload,
            args.metrics: encode_json(metrics),
        }
    )
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
