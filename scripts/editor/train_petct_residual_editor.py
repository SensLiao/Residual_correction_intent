#!/usr/bin/env python3
"""Train one declared residual-editor representation condition."""

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
    EDITOR_CHECKPOINT_SCHEMA,
    EpisodeDataset,
    LearningContractError,
    editor_loss,
    load_editor_architecture_contract,
    load_experiment_config,
    load_intent_intervention_contract,
    load_jsonl,
    load_training_contract,
    seed_everything,
    select_frozen_seed,
    sha256_file,
    torch_save_exclusive,
    validate_frozen_override,
    validate_manifest_rows_against_frozen_learning_split,
)
from common.petct_models import ResidualEditorUNet2D  # noqa: E402
from common.petct_route_a_core import EDITOR_TRAINING_CONDITIONS  # noqa: E402


TRAINABLE = EDITOR_TRAINING_CONDITIONS
EDITOR_TRAINING_STATUS = "TRAINED_WHEN_THIS_SCRIPT_IS_EXECUTED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--condition", choices=TRAINABLE, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="optional frozen-config assertion")
    parser.add_argument(
        "--batch-size", type=int, help="optional frozen-config assertion"
    )
    parser.add_argument(
        "--learning-rate", type=float, help="optional frozen-config assertion"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="member of editor.training.seeds"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    try:
        config = load_experiment_config(args.experiment_config)
        declared_conditions = tuple(config["editor"].get("training_conditions") or ())
        if declared_conditions != TRAINABLE:
            raise LearningContractError(
                "editor.training_conditions must match the four distinct local contracts"
            )
        training = load_training_contract(config, "editor")
        architecture_contract = load_editor_architecture_contract(config)
        intervention_contract = load_intent_intervention_contract(config)
        seed = select_frozen_seed(training, args.seed)
        validate_frozen_override("--epochs", args.epochs, training["epochs"])
        validate_frozen_override(
            "--batch-size", args.batch_size, training["batch_size"]
        )
        validate_frozen_override(
            "--learning-rate", args.learning_rate, training["learning_rate"]
        )
    except LearningContractError as error:
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
    train = EpisodeDataset(args.manifest, "train", editor_condition=args.condition)
    val = EpisodeDataset(args.manifest, "val", editor_condition=args.condition)
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train,
        batch_size=training["batch_size"],
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(val, batch_size=training["batch_size"], shuffle=False)
    device = torch.device(args.device)
    model = ResidualEditorUNet2D().to(device)
    if model.architecture_id != architecture_contract["primary_architecture_id"]:
        parser.error("runtime editor architecture differs from the frozen config")
    parameter_count = model.parameter_count()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    best = None
    history = []
    for epoch in range(training["epochs"]):
        model.train()
        train_total = 0.0
        for batch in train_loader:
            logits = model(
                batch["visual"].to(device),
                batch["operation_id"].to(device),
                batch["target_id"].to(device),
                batch["scope_id"].to(device),
            )
            losses = editor_loss(
                logits,
                batch["target"].to(device),
                batch["authorized"].to(device),
                bce_loss_weight=training["bce_loss_weight"],
                dice_loss_weight=training["dice_loss_weight"],
                unauthorized_delta_loss_weight=training[
                    "unauthorized_delta_loss_weight"
                ],
            )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("non-finite editor training loss")
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            train_total += float(losses["total"].detach()) * logits.shape[0]
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["visual"].to(device),
                    batch["operation_id"].to(device),
                    batch["target_id"].to(device),
                    batch["scope_id"].to(device),
                )
                val_loss = editor_loss(
                    logits,
                    batch["target"].to(device),
                    batch["authorized"].to(device),
                    bce_loss_weight=training["bce_loss_weight"],
                    dice_loss_weight=training["dice_loss_weight"],
                    unauthorized_delta_loss_weight=training[
                        "unauthorized_delta_loss_weight"
                    ],
                )["total"]
                if not torch.isfinite(val_loss):
                    raise RuntimeError("non-finite editor validation loss")
                val_total += float(val_loss) * logits.shape[0]
        row = {
            "epoch": epoch,
            "train_loss": train_total / len(train),
            "val_loss": val_total / len(val),
        }
        history.append(row)
        if best is None or row["val_loss"] < best["val_loss"]:
            best = dict(row)
            best["state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    state = best.pop("state_dict")
    checkpoint = {
        "schema_version": EDITOR_CHECKPOINT_SCHEMA,
        "status": EDITOR_TRAINING_STATUS,
        "condition": args.condition,
        "architecture_id": model.architecture_id,
        "deferred_fusion_ablations": architecture_contract["deferred_ablations"],
        "fusion_plan": architecture_contract["fusion_plan"],
        "parameter_count": parameter_count,
        "seed": seed,
        "seed_registry": training["seeds"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "training_manifest": str(args.manifest.resolve()),
        "training_manifest_sha256": sha256_file(args.manifest),
        "learning_split": str(args.learning_split.resolve()),
        "learning_split_sha256": split_validation["learning_split_sha256"],
        "experiment_config": str(args.experiment_config.resolve()),
        "experiment_config_sha256": sha256_file(args.experiment_config),
        "input_ablation": "full",
        "hyperparameters": {
            key: training[key]
            for key in (
                "epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "optimizer",
                "bce_loss_weight",
                "dice_loss_weight",
                "unauthorized_delta_loss_weight",
            )
        },
        "checkpoint_criterion": training["checkpoint_criterion"],
        "special_condition_semantics": {
            "NULL": intervention_contract["null_semantics"],
            "OPERATION_ONLY": intervention_contract["operation_only_semantics"],
        },
        "best": best,
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
        "state_dict": state,
    }
    torch_save_exclusive(args.output, checkpoint)
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "condition": args.condition,
                "architecture_id": model.architecture_id,
                "parameter_count": parameter_count,
                "best_epoch": best["epoch"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
