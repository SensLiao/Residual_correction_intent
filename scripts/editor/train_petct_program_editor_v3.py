#!/usr/bin/env python3
"""Train the v3 program-conditioned editor (J6-J9 representation ladder).

J6 spatial-only (12ch, NULL conditioning), J7 flat-action (13ch gold family
call + program dropout), J8 continuous readout control, J9 typed program
(13ch, legal call with family+operand).  Conditioning dimension and
injection location are shared across the ladder; only the representation
changes.  Training targets come from the label/evaluator lane; the
selected-component channel is inference-visible only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_program_learning import (  # noqa: E402
    LearningContractError,
    ProgramEpisodeDataset,
    _sha256_file,
    load_label_manifest,
    program_collate,
    validate_program_manifest_receipt,
    validate_training_split,
)
from common.petct_mainline_lineage import (  # noqa: E402
    validate_r13_training_binding,
)
from common.petct_program_models import (  # noqa: E402
    PROGRAM_EDITOR_ARCHITECTURE_ID,
    ProgramCompilerNet,
    ProgramEditorUNet2D,
)

CHECKPOINT_SCHEMA = "PETCT-PROGRAM-EDITOR-CHECKPOINT-v1.0"

ARM_CONDITION = {
    "J6": "spatial_only",
    "J7": "flat_action",
    "J8": "continuous",
    "J9": "typed_program",
}


def _load_candidates(candidates_dir: Path) -> dict:
    if not candidates_dir.is_dir():
        raise LearningContractError("missing candidates dir: %s" % candidates_dir)
    records = {}
    for path in sorted(candidates_dir.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            record = json.load(stream)
        records[str(record["episode_id"])] = record
    return records


def _load_pointer_targets(targets_dir: Path) -> dict:
    if not targets_dir.is_dir():
        raise LearningContractError("missing pointer targets dir: %s" % targets_dir)
    targets = {}
    for path in sorted(targets_dir.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            record = json.load(stream)
        targets[str(record["episode_id"])] = [
            int(value) for value in record["pointer_targets"]
        ]
    return targets


def _load_predicted_calls(calls_path: Path) -> dict:
    if not calls_path.is_file():
        raise LearningContractError("missing predicted calls: %s" % calls_path)
    calls = {}
    for line in calls_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        calls[str(record["episode_id"])] = record
    return calls


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denom = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + epsilon) / (denom + epsilon)).mean()


def _tree_sha256(directory: Path) -> str:
    records = []
    for path in sorted(value for value in directory.rglob("*") if value.is_file()):
        records.append((path.relative_to(directory).as_posix(), _sha256_file(path)))
    return hashlib.sha256(
        json.dumps(records, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _load_frozen_compiler(path: Path, *, split_sha: str, device: torch.device):
    if not path.is_file():
        raise LearningContractError("missing compiler checkpoint: %s" % path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "PETCT-PROGRAM-COMPILER-CHECKPOINT-v1.0"
        or checkpoint.get("architecture_id") != "matched_legal_component_program_v1"
        or checkpoint.get("learning_split_sha256") != split_sha
    ):
        raise LearningContractError("continuous control compiler checkpoint is incompatible")
    include_repair = bool(checkpoint.get("hyperparameters", {}).get("include_repair", True))
    compiler = ProgramCompilerNet(include_repair=include_repair).to(device)
    compiler.load_state_dict(checkpoint["state_dict"], strict=True)
    compiler.eval()
    for parameter in compiler.parameters():
        parameter.requires_grad_(False)
    return compiler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--lineage-receipt", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pointer-targets", type=Path, default=None)
    parser.add_argument("--predicted-calls", type=Path, default=None)
    parser.add_argument("--compiler-checkpoint", type=Path, default=None)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["J6", "J7", "J8", "J9"], required=True)
    parser.add_argument("--call-source", choices=["gold", "predicted"], default="gold")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dataset-mode", choices=["matched", "natural"], default="natural"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.call_source != "gold":
        parser.error(
            "editor training is gold-call only; inject predicted calls at evaluation "
            "with the same frozen editor checkpoint"
        )
    if args.arm == "J8" and args.compiler_checkpoint is None:
        parser.error("J8 requires --compiler-checkpoint for frozen state embeddings")
    with args.experiment_config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema_version") != "PETCT-ROUTE-A-EXPERIMENT-v3.0":
        parser.error("experiment config is not v3")
    training = config["editor"]["training"]
    if args.seed not in training["seeds"]:
        parser.error("seed %d is not in editor.training.seeds" % args.seed)
    loss_config = config["editor"]["training"]["loss"]
    dropout = float(config["editor"]["program_dropout"])
    lineage = validate_r13_training_binding(
        args.lineage_receipt,
        args.manifest_receipt,
        args.episodes,
        args.labels,
    )
    labels = load_label_manifest(
        args.labels, require_matched_groups=args.dataset_mode == "matched"
    )
    split_sha = validate_training_split(labels, args.learning_split)
    validate_program_manifest_receipt(
        args.manifest_receipt, args.episodes, args.labels, args.learning_split
    )
    candidates = _load_candidates(args.candidates)
    pointer_targets = _load_pointer_targets(args.pointer_targets) if args.pointer_targets else {}
    predicted_calls = _load_predicted_calls(args.predicted_calls) if args.predicted_calls else {}
    condition = ARM_CONDITION[args.arm]
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train = ProgramEpisodeDataset(
        args.episodes, labels, "train", candidates, pointer_targets, predicted_calls,
        editor_condition=condition, call_source=args.call_source,
        program_dropout=dropout, seed=args.seed,
    )
    val = ProgramEpisodeDataset(
        args.episodes, labels, "val", candidates, pointer_targets, predicted_calls,
        editor_condition=condition, call_source=args.call_source,
        program_dropout=0.0, seed=args.seed,
    )
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=program_collate,
    )
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False, collate_fn=program_collate
    )
    model = ProgramEditorUNet2D(
        visual_channels=12 if condition == "spatial_only" else 13,
        conditioner="continuous" if condition == "continuous" else "program",
    ).to(device)
    frozen_compiler = (
        _load_frozen_compiler(
            args.compiler_checkpoint, split_sha=split_sha, device=device
        )
        if condition == "continuous"
        else None
    )
    parameter_count = model.parameter_count()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=training["weight_decay"]
    )
    best = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            visual = batch["visual"].to(device)
            family = batch["family_id"].to(device)
            operand = batch["operand_mode"].to(device)
            support = batch["support_mode"].to(device)
            state_embedding = None
            if frozen_compiler is not None:
                with torch.no_grad():
                    state_embedding = frozen_compiler(
                        batch["compiler_visual"].to(device),
                        batch["spacing_xy"].to(device),
                        batch["operation_id"].to(device),
                        batch["component_vectors"].to(device),
                        batch["component_mask"].to(device),
                    )["embedding"]
            logits = model(
                visual,
                family,
                operand,
                support,
                state_embedding=state_embedding,
                active_mask=batch["active"].to(device),
            )
            target = batch["authorized"] if "authorized" in batch else None
            if target is None:
                raise LearningContractError("editor training requires label-lane targets")
            target = target.to(device)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            dice = soft_dice_loss(logits, target)
            spill = (torch.sigmoid(logits) * (1.0 - target)).mean()
            loss = (
                float(loss_config["bce_loss_weight"]) * bce
                + float(loss_config["dice_loss_weight"]) * dice
                + float(loss_config["spill_loss_weight"]) * spill
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite editor training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * visual.shape[0]
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                visual = batch["visual"].to(device)
                state_embedding = None
                if frozen_compiler is not None:
                    state_embedding = frozen_compiler(
                        batch["compiler_visual"].to(device),
                        batch["spacing_xy"].to(device),
                        batch["operation_id"].to(device),
                        batch["component_vectors"].to(device),
                        batch["component_mask"].to(device),
                    )["embedding"]
                logits = model(
                    visual,
                    batch["family_id"].to(device),
                    batch["operand_mode"].to(device),
                    batch["support_mode"].to(device),
                    state_embedding=state_embedding,
                    active_mask=batch["active"].to(device),
                )
                target = batch["authorized"].to(device)
                val_loss += float(
                    torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                    + soft_dice_loss(logits, target)
                ) * visual.shape[0]
        row = {"epoch": epoch, "train_loss": train_loss / len(train), "val_loss": val_loss / len(val)}
        history.append(row)
        if best is None or row["val_loss"] < best["row"]["val_loss"]:
            best = {"row": dict(row)}
            best["state_dict"] = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    state_dict = best.pop("state_dict")
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "TRAINED_WHEN_THIS_SCRIPT_IS_EXECUTED",
        "arm": args.arm,
        "call_source": args.call_source,
        "seed": args.seed,
        "seed_registry": training["seeds"],
        "dataset_mode": args.dataset_mode,
        "source_m0_lineage": lineage["source_m0_lineage"],
        "lineage_receipt": str(args.lineage_receipt.resolve()),
        "lineage_receipt_sha256": _sha256_file(args.lineage_receipt),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": training["weight_decay"],
            "optimizer": training["optimizer"],
            "program_dropout": dropout,
            "loss_weights": loss_config,
        },
        "architecture_id": PROGRAM_EDITOR_ARCHITECTURE_ID,
        "parameter_count": parameter_count,
        "episodes_manifest": str(args.episodes.resolve()),
        "episodes_sha256": _sha256_file(args.episodes),
        "labels_manifest": str(args.labels.resolve()),
        "labels_sha256": _sha256_file(args.labels),
        "learning_split": str(args.learning_split.resolve()),
        "learning_split_sha256": split_sha,
        "manifest_receipt": str(args.manifest_receipt.resolve()),
        "manifest_receipt_sha256": _sha256_file(args.manifest_receipt),
        "candidates_tree_sha256": _tree_sha256(args.candidates),
        "pointer_targets_tree_sha256": (
            _tree_sha256(args.pointer_targets) if args.pointer_targets else None
        ),
        "compiler_checkpoint": (
            None if args.compiler_checkpoint is None else str(args.compiler_checkpoint.resolve())
        ),
        "compiler_checkpoint_sha256": (
            None if args.compiler_checkpoint is None else _sha256_file(args.compiler_checkpoint)
        ),
        "experiment_config": str(args.experiment_config.resolve()),
        "experiment_config_sha256": _sha256_file(args.experiment_config),
        "best": best["row"],
        "state_dict": state_dict,
        "history": history,
        "source_sha256": {
            "train_entrypoint": _sha256_file(Path(__file__)),
            "program_models": _sha256_file(SCRIPTS_ROOT / "common" / "petct_program_models.py"),
            "program_learning": _sha256_file(SCRIPTS_ROOT / "common" / "petct_program_learning.py"),
        },
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "torch_cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        },
    }
    torch.save(checkpoint, args.output)
    print(json.dumps({"checkpoint": str(args.output), "arm": args.arm, "best": best["row"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
