from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "editor"))

from infer_petct_residual_editor import (  # noqa: E402
    EDITOR_TRAINING_STATUS,
    apply_operation_delta,
    conditioned_slot_ids,
    conditioned_slots_for_condition,
    dataset_condition_for_inference,
    execution_operation_for_condition,
    expected_editor_checkpoint_condition,
    partition_rows_for_condition,
    validate_editor_architecture_receipt,
    validate_editor_checkpoint_training_binding,
    validate_intervention_receipt,
    validate_predicted_slots_receipt,
)
from common.petct_learning import (  # noqa: E402
    INTERVENTION_SCHEMA,
    P2T_CHECKPOINT_CRITERION,
    P2T_CHECKPOINT_SCHEMA,
    P2T_METRICS_SCHEMA,
    P2T_PREDICTION_SCHEMA,
    SHUFFLE_ALGORITHM,
    LearningContractError,
)
from common.petct_models import (  # noqa: E402
    LEGAL_GOALS,
    P2T_DEFERRED_CROSS_ATTENTION_ARCHITECTURE_ID,
    P2T_PRIMARY_ARCHITECTURE_ID,
)
from train_petct_residual_editor import TRAINABLE  # noqa: E402


def _p2t_receipt():
    training_manifest_hash = "t" * 64
    manifest_hash = "m" * 64
    config_hash = "e" * 64
    checkpoint_hash = "c" * 64
    predictions_hash = "p" * 64
    training = {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "optimizer": "AdamW",
        "joint_loss_weight": 1.0,
        "operation_loss_weight": 0.25,
        "target_loss_weight": 0.25,
        "scope_loss_weight": 0.25,
        "checkpoint_criterion": P2T_CHECKPOINT_CRITERION,
        "seeds": [3407],
    }
    evaluation = {"primary_end_to_end_arm": "full"}
    checkpoint = {
        "schema_version": P2T_CHECKPOINT_SCHEMA,
        "manifest_sha256": training_manifest_hash,
        "experiment_config_sha256": config_hash,
        "input_ablation": "full",
        "architecture_id": P2T_PRIMARY_ARCHITECTURE_ID,
        "arm_role": "primary",
        "seed": 3407,
        "checkpoint_criterion": P2T_CHECKPOINT_CRITERION,
        "hyperparameters": {
            key: training[key]
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
        },
    }
    predictions = [
        {
            "schema_version": P2T_PREDICTION_SCHEMA,
            "episode_id": "a",
            "partition": "test",
            "strategy": "centerline",
            "goal": "ADD_SAME_LOCAL",
            "operation": "ADD",
            "target": "SAME",
            "scope": "LOCAL",
            "checkpoint_schema_version": P2T_CHECKPOINT_SCHEMA,
            "checkpoint_sha256": checkpoint_hash,
            "learning_manifest_sha256": manifest_hash,
            "training_manifest_sha256": training_manifest_hash,
            "inference_manifest_sha256": manifest_hash,
            "experiment_config_sha256": config_hash,
            "input_ablation": "full",
            "architecture_id": P2T_PRIMARY_ARCHITECTURE_ID,
            "arm_role": "primary",
            "visible_npz_sha256": "va",
        },
        {
            "schema_version": P2T_PREDICTION_SCHEMA,
            "episode_id": "b",
            "partition": "test",
            "strategy": "boundary",
            "goal": "REMOVE_NEW_COMPLETE",
            "operation": "REMOVE",
            "target": "NEW",
            "scope": "COMPLETE",
            "checkpoint_schema_version": P2T_CHECKPOINT_SCHEMA,
            "checkpoint_sha256": checkpoint_hash,
            "learning_manifest_sha256": manifest_hash,
            "training_manifest_sha256": training_manifest_hash,
            "inference_manifest_sha256": manifest_hash,
            "experiment_config_sha256": config_hash,
            "input_ablation": "full",
            "architecture_id": P2T_PRIMARY_ARCHITECTURE_ID,
            "arm_role": "primary",
            "visible_npz_sha256": "vb",
        },
    ]
    metrics = {
        "schema_version": P2T_METRICS_SCHEMA,
        "partition": "test",
        "episode_count": 2,
        "checkpoint_schema_version": P2T_CHECKPOINT_SCHEMA,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_criterion": P2T_CHECKPOINT_CRITERION,
        "checkpoint_seed": 3407,
        "manifest_sha256": manifest_hash,
        "training_manifest_sha256": training_manifest_hash,
        "inference_manifest_sha256": manifest_hash,
        "experiment_config_sha256": config_hash,
        "predictions_sha256": predictions_hash,
        "prediction_schema_version": P2T_PREDICTION_SCHEMA,
        "input_ablation": "full",
        "architecture_id": P2T_PRIMARY_ARCHITECTURE_ID,
        "arm_role": "primary",
        "legal_goals": list(LEGAL_GOALS),
        "confusion_diagnostic": {
            "operation": {},
            "target": {},
            "scope": {},
            "joint_goal": {},
        },
        "per_strategy_diagnostic": {
            "centerline": {"episode_count": 1},
            "boundary": {"episode_count": 1},
        },
    }
    arguments = {
        "p2t_checkpoint": checkpoint,
        "p2t_checkpoint_sha256": checkpoint_hash,
        "p2t_metrics": metrics,
        "predictions_rows": predictions,
        "predictions_sha256": predictions_hash,
        "expected_episode_ids": {"a", "b"},
        "visible_by_episode": {"a": "va", "b": "vb"},
        "partition": "test",
        "manifest_sha256": manifest_hash,
        "experiment_config_sha256": config_hash,
        "training_contract": training,
        "evaluation_contract": evaluation,
    }
    return arguments


def test_predicted_slots_receipt_accepts_bound_primary_full_arm() -> None:
    validate_predicted_slots_receipt(**_p2t_receipt())


def test_predicted_slots_checkpoint_load_remains_weights_only() -> None:
    source = (SCRIPTS / "editor" / "infer_petct_residual_editor.py").read_text(
        encoding="utf-8"
    )
    assert (
        'torch.load(args.checkpoint, map_location="cpu", weights_only=True)'
        in source
    )
    assert source.count('map_location="cpu", weights_only=True') >= 2


def test_raw_torch_version_breaks_weights_only_load(tmp_path: Path) -> None:
    """Document the trap: an uncast ``torch.__version__`` is unloadable.

    ``TorchVersion`` is not an allowlisted global, so a checkpoint carrying it
    cannot be read back by the fail-closed loaders above.  Every trainer must
    therefore store the runtime block as primitives.
    """
    path = tmp_path / "raw_version.pth"
    torch.save({"runtime": {"torch": torch.__version__}}, path)
    with pytest.raises(Exception, match="TorchVersion"):
        torch.load(path, map_location="cpu", weights_only=True)

    path = tmp_path / "cast_version.pth"
    torch.save({"runtime": {"torch": str(torch.__version__)}}, path)
    assert torch.load(path, map_location="cpu", weights_only=True)["runtime"][
        "torch"
    ] == str(torch.__version__)


@pytest.mark.parametrize(
    "trainer",
    ("editor/train_petct_residual_editor.py", "p2t/train_petct_p2t.py"),
)
def test_trainers_store_primitive_runtime_versions(trainer: str) -> None:
    """Both trainers must cast the runtime versions, not just the P2T one.

    The editor trainer diverged here once and made all four editor checkpoints
    unreadable by ``infer_petct_residual_editor.py``.
    """
    source = (SCRIPTS / trainer).read_text(encoding="utf-8")
    assert '"torch": str(torch.__version__)' in source
    assert '"torch": torch.__version__,' not in source


def test_predicted_slots_receipt_rejects_schema_hash_and_episode_count_drift() -> None:
    arguments = _p2t_receipt()
    arguments["p2t_checkpoint"]["schema_version"] = "unexpected"
    with pytest.raises(LearningContractError, match="checkpoint schema"):
        validate_predicted_slots_receipt(**arguments)

    arguments = _p2t_receipt()
    arguments["p2t_metrics"]["checkpoint_sha256"] = "wrong"
    with pytest.raises(LearningContractError, match="does not bind"):
        validate_predicted_slots_receipt(**arguments)

    arguments = _p2t_receipt()
    arguments["p2t_metrics"]["episode_count"] = 1
    with pytest.raises(LearningContractError, match="episode_count"):
        validate_predicted_slots_receipt(**arguments)


def test_predicted_slots_receipt_rejects_non_primary_or_ablated_checkpoint() -> None:
    arguments = _p2t_receipt()
    arguments["p2t_checkpoint"]["input_ablation"] = "no_M0"
    arguments["p2t_checkpoint"]["arm_role"] = "ablation"
    with pytest.raises(LearningContractError, match="primary full-input"):
        validate_predicted_slots_receipt(**arguments)


def test_predicted_slots_receipt_rejects_secondary_architecture_even_if_role_is_forged_primary() -> (
    None
):
    arguments = _p2t_receipt()
    arguments["p2t_checkpoint"][
        "architecture_id"
    ] = P2T_DEFERRED_CROSS_ATTENTION_ARCHITECTURE_ID
    with pytest.raises(LearningContractError, match="primary P2T architecture_id"):
        validate_predicted_slots_receipt(**arguments)


def test_predicted_slots_receipt_rejects_training_or_inference_manifest_drift() -> None:
    arguments = _p2t_receipt()
    arguments["p2t_metrics"]["training_manifest_sha256"] = "x" * 64
    with pytest.raises(LearningContractError, match="does not bind"):
        validate_predicted_slots_receipt(**arguments)

    arguments = _p2t_receipt()
    arguments["predictions_rows"][0]["inference_manifest_sha256"] = "x" * 64
    with pytest.raises(LearningContractError, match="provenance mismatch"):
        validate_predicted_slots_receipt(**arguments)


def _intervention_receipt():
    labels = {
        "a": ("ADD", "SAME", "LOCAL"),
        "b": ("REMOVE", "SAME", "LOCAL"),
        "c": ("ADD", "SAME", "COMPLETE"),
        "d": ("REMOVE", "SAME", "COMPLETE"),
        "e": ("ADD", "NEW", "COMPLETE"),
        "f": ("REMOVE", "NEW", "COMPLETE"),
    }
    visible = {episode: f"v{episode}" for episode in labels}
    episodes = tuple(labels)
    source_rows = [
        {
            "episode_id": episode,
            "operation": labels[episode][0],
            "target": labels[episode][1],
            "scope": labels[episode][2],
            "visible_sha256": visible[episode],
        }
        for episode in episodes
    ]
    donor = {
        episode: episodes[(index + 1) % len(episodes)]
        for index, episode in enumerate(episodes)
    }
    rows = []
    for episode in episodes:
        donor_episode = donor[episode]
        rows.append(
            {
                "schema_version": INTERVENTION_SCHEMA,
                "algorithm": SHUFFLE_ALGORITHM,
                "seed": 20260717,
                "partition": "test",
                "permutation_size": len(episodes),
                "source_manifest_sha256": "m" * 64,
                "experiment_config_sha256": "e" * 64,
                "episode_id": episode,
                "source_episode_id": donor_episode,
                "original_operation": labels[episode][0],
                "original_target": labels[episode][1],
                "original_scope": labels[episode][2],
                "operation": labels[donor_episode][0],
                "target": labels[donor_episode][1],
                "scope": labels[donor_episode][2],
                "source_visible_npz_sha256": visible[donor_episode],
                "changed": True,
            }
        )
    arguments = {
        "intervention_rows": rows,
        "source_rows": source_rows,
        "partition": "test",
        "manifest_sha256": "m" * 64,
        "experiment_config_sha256": "e" * 64,
        "intervention_contract": {
            "shuffle_algorithm": SHUFFLE_ALGORITHM,
            "shuffle_seed": 20260717,
        },
    }
    return arguments


def test_intervention_receipt_validates_true_donor_permutation() -> None:
    validate_intervention_receipt(**_intervention_receipt())


def test_intervention_receipt_rejects_donor_reuse() -> None:
    arguments = _intervention_receipt()
    rows = copy.deepcopy(arguments["intervention_rows"])
    rows[2]["source_episode_id"] = rows[0]["source_episode_id"]
    arguments["intervention_rows"] = rows
    with pytest.raises(LearningContractError, match="donors must be unique"):
        validate_intervention_receipt(**arguments)


def test_operation_only_is_an_inference_intervention_on_the_rich_checkpoint() -> None:
    assert "visual_state_only" in TRAINABLE
    assert (
        expected_editor_checkpoint_condition("visual_state_only")
        == "visual_state_only"
    )
    assert "scribble_plus_operation" not in TRAINABLE
    assert (
        expected_editor_checkpoint_condition("scribble_plus_operation")
        == "scribble_plus_intent"
    )
    assert "oracle_slots" not in TRAINABLE
    assert expected_editor_checkpoint_condition("oracle_slots") == (
        "scribble_plus_intent"
    )
    assert expected_editor_checkpoint_condition("predicted_slots") == (
        "scribble_plus_intent"
    )


def test_operation_algebra_supports_add_and_remove_in_one_batch() -> None:
    logits = torch.full((2, 1, 2, 2), -20.0)
    logits[0, 0, 0, 1] = 20.0
    logits[1, 0, 1, 0] = 20.0
    m0 = torch.tensor(
        [
            [[[1, 0], [0, 0]]],
            [[[1, 1], [1, 0]]],
        ],
        dtype=torch.float32,
    )
    delta, m1 = apply_operation_delta(logits, m0, ["ADD", "REMOVE"], 0.5)
    assert delta[0, 0, 0, 1]
    assert m1[0, 0, 0, 0] and m1[0, 0, 0, 1]
    assert delta[1, 0, 1, 0]
    assert not m1[1, 0, 1, 0]
    assert m1[1, 0, 0, 0] and m1[1, 0, 0, 1]


def test_slot_interventions_are_legal_and_ood_changes_only_conditioning() -> None:
    assert conditioned_slots_for_condition(
        "scribble_plus_operation",
        gold_operation="REMOVE",
        gold_target="NEW",
        gold_scope="COMPLETE",
    ) == ("REMOVE", "NULL", "NULL")
    with pytest.raises(LearningContractError, match="INELIGIBLE"):
        conditioned_slots_for_condition(
            "same_weight_wrong_scope",
            gold_operation="REMOVE",
            gold_target="NEW",
            gold_scope="COMPLETE",
        )
    assert conditioned_slots_for_condition(
        "same_weight_wrong_scope",
        gold_operation="REMOVE",
        gold_target="SAME",
        gold_scope="COMPLETE",
    ) == ("REMOVE", "SAME", "LOCAL")
    assert conditioned_slots_for_condition(
        "wrong_operation_OOD",
        gold_operation="ADD",
        gold_target="SAME",
        gold_scope="LOCAL",
    ) == ("REMOVE", "SAME", "LOCAL")
    assert conditioned_slot_ids("REMOVE", "NULL", "NULL") == (1, 2, 2)
    with pytest.raises(LearningContractError, match="illegal"):
        conditioned_slot_ids("NULL", "SAME", "LOCAL")
    assert dataset_condition_for_inference("wrong_operation_OOD") == (
        "scribble_plus_intent"
    )
    prediction = {"operation": "REMOVE", "target": "SAME", "scope": "LOCAL"}
    assert execution_operation_for_condition(
        "predicted_slots", gold_operation="ADD", predicted=prediction
    ) == "REMOVE"
    assert execution_operation_for_condition(
        "same_weight_shuffled",
        gold_operation="ADD",
        intervention=prediction,
    ) == "ADD"
    assert execution_operation_for_condition(
        "wrong_operation_OOD", gold_operation="ADD"
    ) == "ADD"
    assert (
        expected_editor_checkpoint_condition("same_weight_NULL")
        == "scribble_plus_intent"
    )


def test_wrong_scope_episode_eligibility_is_same_only_without_target_repair() -> None:
    rows = [
        {"episode_id": "same", "target": "SAME", "scope": "LOCAL"},
        {
            "episode_id": "new",
            "target": "NEW",
            "scope": "COMPLETE",
        },
    ]
    eligible, ineligible = partition_rows_for_condition(
        "same_weight_wrong_scope", rows
    )
    assert [row["episode_id"] for row in eligible] == ["same"]
    assert [row["episode_id"] for row in ineligible] == ["new"]
    assert eligible[0]["target"] == "SAME"


def test_editor_architecture_receipt_rejects_contract_and_parameter_mismatch() -> None:
    architecture_contract = {
        "primary_architecture_id": "simple_operation_conditioned_residual_unet_v2",
        "deferred_ablations": [
            "FiLM",
            "concat",
            "gated",
            "intent-image cross_attention",
        ],
        "fusion_plan": "simple-first-current-campaign; listed ablations deferred",
    }
    checkpoint = {
        "architecture_id": "simple_operation_conditioned_residual_unet_v2",
        "deferred_fusion_ablations": architecture_contract["deferred_ablations"],
        "fusion_plan": architecture_contract["fusion_plan"],
        "parameter_count": 123,
    }
    assert (
        validate_editor_architecture_receipt(
            checkpoint=checkpoint,
            architecture_contract=architecture_contract,
            model_parameter_count=123,
        )
        == 123
    )
    drifted = copy.deepcopy(checkpoint)
    drifted["architecture_id"] = "legacy_editor"
    with pytest.raises(LearningContractError, match="not the frozen simple"):
        validate_editor_architecture_receipt(
            checkpoint=drifted,
            architecture_contract=architecture_contract,
            model_parameter_count=123,
        )
    with pytest.raises(LearningContractError, match="parameter_count"):
        validate_editor_architecture_receipt(
            checkpoint=checkpoint,
            architecture_contract=architecture_contract,
            model_parameter_count=124,
        )


def test_editor_checkpoint_binds_distinct_training_manifest(
    tmp_path: Path,
) -> None:
    training_manifest = tmp_path / "training.jsonl"
    learning_split = tmp_path / "split.json"
    training_manifest.write_text("{}\n", encoding="utf-8")
    learning_split.write_text("{}\n", encoding="utf-8")
    checkpoint = {
        "status": EDITOR_TRAINING_STATUS,
        "manifest": str(training_manifest.resolve()),
        "manifest_sha256": "a" * 64,
        "training_manifest": str(training_manifest.resolve()),
        "training_manifest_sha256": "a" * 64,
        "learning_split": str(learning_split.resolve()),
        "learning_split_sha256": "b" * 64,
    }

    validate_editor_checkpoint_training_binding(
        checkpoint=checkpoint,
        training_manifest=training_manifest,
        training_manifest_sha256="a" * 64,
        learning_split=learning_split,
        learning_split_sha256="b" * 64,
    )

    drifted = copy.deepcopy(checkpoint)
    drifted["training_manifest_sha256"] = "c" * 64
    with pytest.raises(LearningContractError, match="training manifest"):
        validate_editor_checkpoint_training_binding(
            checkpoint=drifted,
            training_manifest=training_manifest,
            training_manifest_sha256="a" * 64,
            learning_split=learning_split,
            learning_split_sha256="b" * 64,
        )


def test_editor_training_checkpoint_declares_execution_status() -> None:
    source = (SCRIPTS / "editor" / "train_petct_residual_editor.py").read_text(
        encoding="utf-8"
    )
    assert '"status": EDITOR_TRAINING_STATUS' in source
    assert '"training_manifest_sha256": sha256_file(args.manifest)' in source
