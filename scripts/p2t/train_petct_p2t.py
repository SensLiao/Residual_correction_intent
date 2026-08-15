#!/usr/bin/env python3
"""Train P2T for six joint intents plus three binary auxiliary slots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import (  # noqa: E402
    EpisodeDataset,
    LearningContractError,
    P2T_CHECKPOINT_SCHEMA,
    legal_joint_ids,
    load_experiment_config,
    load_jsonl,
    load_p2t_evaluation_contract,
    load_training_contract,
    patient_balanced_macro_f1,
    seed_everything,
    select_frozen_seed,
    sha256_file,
    torch_save_exclusive,
    validate_frozen_override,
    validate_manifest_rows_against_frozen_learning_split,
)
from common.petct_models import (  # noqa: E402
    P2T_ARCHITECTURE_IDS,
    P2T_PRIMARY_ARCHITECTURE_ID,
    build_p2t_model,
    p2t_architecture_contract,
    validate_p2t_architecture_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="optional frozen-config assertion")
    parser.add_argument(
        "--batch-size", type=int, help="optional frozen-config assertion"
    )
    parser.add_argument(
        "--learning-rate", type=float, help="optional frozen-config assertion"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="member of p2t.training.seeds"
    )
    parser.add_argument(
        "--architecture-id",
        choices=P2T_ARCHITECTURE_IDS,
        default=P2T_PRIMARY_ARCHITECTURE_ID,
        help=(
            "frozen P2T architecture; the default is the professor-directed "
            "simple signed-scribble/state-pooling v2 model"
        ),
    )
    parser.add_argument(
        "--input-ablation",
        # operation_control only resolves against the v2.1 config; the frozen v2
        # config rejects it below, so offering it here cannot mislabel a run.
        choices=[
            "full",
            "no_M0",
            "polarity_blind",
            "geometry_only",
            "operation_control",
        ],
        default="full",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    try:
        config = load_experiment_config(args.experiment_config)
        training = load_training_contract(config, "p2t")
        evaluation = load_p2t_evaluation_contract(config)
        seed = select_frozen_seed(training, args.seed)
        validate_frozen_override("--epochs", args.epochs, training["epochs"])
        validate_frozen_override(
            "--batch-size", args.batch_size, training["batch_size"]
        )
        validate_frozen_override(
            "--learning-rate", args.learning_rate, training["learning_rate"]
        )
        if args.input_ablation not in evaluation["ablation_inputs"]:
            raise LearningContractError(
                "input ablation is not present in frozen p2t.ablation_inputs"
            )
        arm_role = validate_p2t_architecture_selection(
            config, args.architecture_id, args.input_ablation
        )
    except (LearningContractError, ValueError) as error:
        parser.error(str(error))
    try:
        split_validation = validate_manifest_rows_against_frozen_learning_split(
            load_jsonl(args.manifest),
            args.learning_split,
            require_episode_id=True,
            allowed_partitions={"train", "val"},
        )
    except LearningContractError as error:
        parser.error(str(error))
    seed_everything(seed)
    device = torch.device(args.device)
    train = EpisodeDataset(
        args.manifest, "train", args.input_ablation, load_evaluation=False
    )
    val = EpisodeDataset(
        args.manifest, "val", args.input_ablation, load_evaluation=False
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train,
        batch_size=training["batch_size"],
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(val, batch_size=training["batch_size"], shuffle=False)
    model = build_p2t_model(args.architecture_id, use_relative_geometry=True).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    best = None
    history = []
    for epoch in range(training["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            visual = batch["visual"].to(device)
            output = model(visual, batch["spacing_xy"].to(device))
            operation_gold = batch["operation_gold"].to(device)
            target_gold = batch["target_gold"].to(device)
            scope_gold = batch["scope_gold"].to(device)
            joint_gold = legal_joint_ids(operation_gold, target_gold, scope_gold)
            loss = training["joint_loss_weight"] * torch.nn.functional.cross_entropy(
                output["joint_logits"], joint_gold
            )
            loss = loss + training[
                "operation_loss_weight"
            ] * torch.nn.functional.cross_entropy(
                output["operation_logits"], operation_gold
            )
            loss = loss + training[
                "target_loss_weight"
            ] * torch.nn.functional.cross_entropy(
                output["target_logits"], target_gold
            )
            loss = loss + training[
                "scope_loss_weight"
            ] * torch.nn.functional.cross_entropy(output["scope_logits"], scope_gold)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite P2T training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * visual.shape[0]
        model.eval()
        val_loss = 0.0
        val_true, val_pred, val_patients = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                visual = batch["visual"].to(device)
                output = model(visual, batch["spacing_xy"].to(device))
                operation_gold = batch["operation_gold"].to(device)
                target_gold = batch["target_gold"].to(device)
                scope_gold = batch["scope_gold"].to(device)
                joint_gold = legal_joint_ids(operation_gold, target_gold, scope_gold)
                loss = training[
                    "joint_loss_weight"
                ] * torch.nn.functional.cross_entropy(
                    output["joint_logits"], joint_gold
                )
                loss = loss + training[
                    "operation_loss_weight"
                ] * torch.nn.functional.cross_entropy(
                    output["operation_logits"], operation_gold
                )
                loss = loss + training[
                    "target_loss_weight"
                ] * torch.nn.functional.cross_entropy(
                    output["target_logits"], target_gold
                )
                loss = loss + training[
                    "scope_loss_weight"
                ] * torch.nn.functional.cross_entropy(
                    output["scope_logits"], scope_gold
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite P2T validation loss")
                val_loss += float(loss) * visual.shape[0]
                val_true.extend(joint_gold.cpu().tolist())
                val_pred.extend(output["joint_logits"].argmax(dim=1).cpu().tolist())
                val_patients.extend(list(batch["patient_id"]))
        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(train),
            "val_loss": val_loss / len(val),
            "val_patient_balanced_six_class_joint_macro_f1": patient_balanced_macro_f1(
                val_true, val_pred, val_patients, [0, 1, 2, 3, 4, 5]
            ),
        }
        history.append(row)
        if best is None or (
            row["val_patient_balanced_six_class_joint_macro_f1"],
            -row["val_loss"],
        ) > (best["val_patient_balanced_six_class_joint_macro_f1"], -best["val_loss"]):
            best = dict(row)
            best["state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    state_dict = best.pop("state_dict")
    checkpoint = {
        "schema_version": P2T_CHECKPOINT_SCHEMA,
        "status": "TRAINED_WHEN_THIS_SCRIPT_IS_EXECUTED",
        "seed": seed,
        "seed_registry": training["seeds"],
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
        "input_ablation": args.input_ablation,
        "architecture_id": args.architecture_id,
        "architecture_contract": p2t_architecture_contract(
            args.architecture_id,
            use_relative_geometry=True,
        ),
        "parameter_count": parameter_count,
        "arm_role": arm_role,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "learning_split": str(args.learning_split.resolve()),
        "learning_split_sha256": split_validation["learning_split_sha256"],
        "experiment_config": str(args.experiment_config.resolve()),
        "experiment_config_sha256": sha256_file(args.experiment_config),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "checkpoint_criterion": training["checkpoint_criterion"],
        "checkpoint_metric": "val_patient_balanced_six_class_joint_macro_f1",
        "best_val_patient_balanced_six_class_joint_macro_f1": best[
            "val_patient_balanced_six_class_joint_macro_f1"
        ],
        "state_dict": state_dict,
        "history": history,
        "source_sha256": {
            "train_entrypoint": sha256_file(Path(__file__)),
            "models": sha256_file(SCRIPTS_ROOT / "common" / "petct_models.py"),
            "learning": sha256_file(SCRIPTS_ROOT / "common" / "petct_learning.py"),
        },
        "runtime": {
            "python": sys.version,
            # Cast TorchVersion to a primitive so the evaluator can retain its
            # fail-closed ``weights_only=True`` checkpoint load.
            "torch": str(torch.__version__),
            "torch_cuda": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
        },
    }
    torch_save_exclusive(args.output, checkpoint)
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "best_epoch": best["epoch"],
                "architecture_id": args.architecture_id,
                "arm_role": arm_role,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
