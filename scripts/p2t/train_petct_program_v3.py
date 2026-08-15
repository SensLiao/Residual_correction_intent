#!/usr/bin/env python3
"""Train the v3 program compiler (J0/J3/J4/J5 mechanism arms).

Family CE + optional same-operation matched-margin loss + optional
multi-positive ADD_SAME pointer loss.  J0 disables the group loss, J3 uses a
frozen shuffled-group placebo, J4 uses true matched groups, J5 adds the
pointer.  J1/J2 baselines reuse the frozen v2 architecture trained on the
same controlled corpus (separate launcher), per config v3 ``compiler.arms``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_program_contract import (  # noqa: E402
    ARCHITECTURE_ID,
    ProgramContractError,
)
from common.petct_program_learning import (  # noqa: E402
    GroupedBatchSampler,
    InferenceEpisodeDataset,
    LearningContractError,
    goal_to_family_id,
    load_jsonl,
    matched_family_margin_loss,
    multi_positive_pointer_loss,
)
from common.petct_program_models import ProgramCompilerNet  # noqa: E402

CHECKPOINT_SCHEMA = "PETCT-PROGRAM-COMPILER-CHECKPOINT-v1.0"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if record["pointer_targets"]:
            targets[str(record["episode_id"])] = [
                int(value) for value in record["pointer_targets"]
            ]
    return targets


def _patient_balanced_family_f1(
    true_ids: list, pred_ids: list, patients: list, classes: list
) -> float:
    per_patient: dict = {}
    for true_id, pred_id, patient in zip(true_ids, pred_ids, patients):
        per_patient.setdefault(str(patient), {"tp": {}, "fp": {}, "fn": {}})
        slot = per_patient[str(patient)]
        if true_id == pred_id:
            slot["tp"][true_id] = slot["tp"].get(true_id, 0) + 1
        else:
            slot["fp"][pred_id] = slot["fp"].get(pred_id, 0) + 1
            slot["fn"][true_id] = slot["fn"].get(true_id, 0) + 1
    macro = []
    for class_id in classes:
        patient_f1s = []
        for slot in per_patient.values():
            tp = slot["tp"].get(class_id, 0)
            fp = slot["fp"].get(class_id, 0)
            fn = slot["fn"].get(class_id, 0)
            denom = 2 * tp + fp + fn
            patient_f1s.append((2 * tp / denom) if denom else 1.0)
        macro.append(float(np.mean(patient_f1s)))
    return float(np.mean(macro))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pointer-targets", type=Path, default=None)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["J0", "J3", "J4", "J5"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.arm in ("J5",) and args.pointer_targets is None:
        parser.error("J5 requires --pointer-targets")
    with args.experiment_config.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema_version") != "PETCT-ROUTE-A-EXPERIMENT-v3.0":
        parser.error("experiment config is not v3")
    arm_config = config["compiler"]["arms"][args.arm]
    if arm_config.get("baseline_reuse"):
        parser.error("%s is a frozen-v2 reuse arm; use the v2 trainer" % args.arm)
    training = config["compiler"]["training"]
    if args.seed not in training["seeds"]:
        parser.error("seed %d is not in compiler.training.seeds" % args.seed)
    loss_config = config["compiler"]["loss"]
    margin = float(loss_config["matched_margin"])
    lambda_group = 0.0 if arm_config["group_loss"] == "disabled" else float(
        loss_config["matched_margin_weight"]
    )
    lambda_pointer = float(loss_config["pointer_weight"]) if arm_config["pointer"] else 0.0
    rows = load_jsonl(args.episodes)
    rows = [row for row in rows if row.get("partition") in ("train", "val")]
    if not rows:
        parser.error("no train/val rows in episodes manifest")
    candidates = _load_candidates(args.candidates)
    pointer_targets = _load_pointer_targets(args.pointer_targets) if args.pointer_targets else {}
    # Label-only join built at TRAIN time only; the dataset itself never
    # reads goals from the manifest rows.
    family_gold = {str(row["episode_id"]): str(row["goal"]) for row in rows}
    group_shuffle = {}
    if args.arm == "J3":
        rng = np.random.default_rng(int(arm_config["group_shuffle_seed"]))
        group_ids = sorted({str(row["matched_state_group_id"]) for row in rows})
        shuffled = list(group_ids)
        rng.shuffle(shuffled)
        group_shuffle = dict(zip(group_ids, shuffled))
    train = InferenceEpisodeDataset(args.episodes, "train", candidates, family_gold)
    val = InferenceEpisodeDataset(args.episodes, "val", candidates, family_gold)
    train_rows = [row for row in rows if row["partition"] == "train"]
    effective_groups = [
        group_shuffle.get(str(row["matched_state_group_id"]), str(row["matched_state_group_id"]))
        for row in train_rows
    ]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train_loader = DataLoader(
        train,
        batch_sampler=GroupedBatchSampler(effective_groups, args.batch_size, args.seed),
    )
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)
    model = ProgramCompilerNet().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
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
            output = model(
                visual,
                batch["spacing_xy"].to(device),
                batch["operation_id"].to(device),
                batch["component_vectors"].to(device),
                batch["component_mask"].to(device),
            )
            family_logits = output["family_logits"]
            family_gold = batch["family_gold"].to(device)
            loss = torch.nn.functional.cross_entropy(family_logits, family_gold)
            if lambda_group:
                loss = loss + lambda_group * matched_family_margin_loss(
                    family_logits,
                    family_gold,
                    list(batch["group_id"]),
                    batch["operation_id"].to(device),
                    margin,
                )
            if lambda_pointer:
                add_same_rows = [
                    index
                    for index, (operation, goal) in enumerate(zip(batch["operation"], batch["goal"]))
                    if operation == "ADD" and goal != "ADD_NEW_COMPLETE"
                ]
                if add_same_rows:
                    loss = loss + lambda_pointer * multi_positive_pointer_loss(
                        output["pointer_logits"][add_same_rows],
                        batch["component_mask"][add_same_rows].to(device),
                        [
                            pointer_targets.get(str(batch["episode_id"][index]), [])
                            for index in add_same_rows
                        ],
                    )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite compiler training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * visual.shape[0]
        model.eval()
        val_true, val_pred, val_patients = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                output = model(
                    batch["visual"].to(device),
                    batch["spacing_xy"].to(device),
                    batch["operation_id"].to(device),
                    batch["component_vectors"].to(device),
                    batch["component_mask"].to(device),
                )
                family_gold = batch["family_gold"]
                val_true.extend(family_gold.tolist())
                val_pred.extend(output["family_logits"].argmax(dim=1).cpu().tolist())
                val_patients.extend(list(batch["patient_id"]))
        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(train),
            "val_patient_balanced_family_macro_f1": _patient_balanced_family_f1(
                val_true, val_pred, val_patients, [0, 1, 2]
            ),
        }
        history.append(row)
        score = (row["val_patient_balanced_family_macro_f1"], -row["train_loss"])
        if best is None or score > best["score"]:
            best = {"score": score, "row": dict(row)}
            best["state_dict"] = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    state_dict = best.pop("state_dict")
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "status": "TRAINED_WHEN_THIS_SCRIPT_IS_EXECUTED",
        "arm": args.arm,
        "seed": args.seed,
        "seed_registry": training["seeds"],
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": training["weight_decay"],
            "optimizer": training["optimizer"],
            "matched_margin": margin,
            "lambda_group": lambda_group,
            "lambda_pointer": lambda_pointer,
            "include_repair": config["grammar"]["include_repair"],
        },
        "group_shuffle_receipt": (
            None
            if args.arm != "J3"
            else {
                "shuffle_seed": int(arm_config["group_shuffle_seed"]),
                "mapping_sha256": _sha256_bytes_hook(group_shuffle),
            }
        ),
        "architecture_id": ARCHITECTURE_ID,
        "parameter_count": parameter_count,
        "episodes_manifest": str(args.episodes.resolve()),
        "episodes_sha256": _sha256_file(args.episodes),
        "experiment_config": str(args.experiment_config.resolve()),
        "experiment_config_sha256": _sha256_file(args.experiment_config),
        "best": best["row"],
        "state_dict": state_dict,
        "history": history,
        "source_sha256": {
            "train_entrypoint": _sha256_file(Path(__file__)),
            "program_contract": _sha256_file(SCRIPTS_ROOT / "common" / "petct_program_contract.py"),
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
    print(
        json.dumps(
            {"checkpoint": str(args.output), "arm": args.arm, "best": best["row"]},
            sort_keys=True,
        )
    )
    return 0


def _sha256_bytes_hook(mapping: dict) -> str:
    payload = json.dumps(mapping, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
