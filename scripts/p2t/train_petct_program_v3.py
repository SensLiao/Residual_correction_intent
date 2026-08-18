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
)
from common.petct_learning import patient_balanced_macro_f1_summary  # noqa: E402
from common.petct_program_learning import (  # noqa: E402
    GroupedBatchSampler,
    InferenceEpisodeDataset,
    LearningContractError,
    goal_to_local_family_id,
    load_label_manifest,
    load_jsonl,
    matched_family_margin_loss,
    multi_positive_pointer_loss,
    program_collate,
    validate_program_manifest_receipt,
    validate_training_split,
)
from common.petct_mainline_lineage import (  # noqa: E402
    validate_r13_training_binding,
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


def _tree_sha256(directory: Path) -> str:
    records = []
    for path in sorted(value for value in directory.rglob("*") if value.is_file()):
        records.append((path.relative_to(directory).as_posix(), _sha256_file(path)))
    return hashlib.sha256(
        json.dumps(records, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _build_placebo_groups(
    labels: dict, *, partition: str, seed: int
) -> tuple[dict[str, str], str]:
    """Shuffle membership, not names, while preserving one row per family."""

    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    receipt_rows = []
    for operation in ("ADD", "REMOVE"):
        by_group: dict[str, dict[int, str]] = {}
        for episode_id, row in labels.items():
            if row["partition"] != partition or row["operation"] != operation:
                continue
            local = goal_to_local_family_id(str(row["goal"]), operation)
            group = str(row["matched_state_group_id"])
            if local in by_group.setdefault(group, {}):
                raise LearningContractError("duplicate family inside matched group")
            by_group[group][local] = episode_id
        groups = sorted(by_group)
        if len(groups) < 2:
            raise LearningContractError("J3 placebo needs at least two groups per operation")
        if any(set(by_group[group]) != {0, 1, 2} for group in groups):
            raise LearningContractError("matched group lacks one of three legal families")
        base = list(range(len(groups)))
        rng.shuffle(base)
        # Family 0 anchors a randomized target ordering; the other families
        # are non-zero cyclic shifts.  This guarantees changed membership,
        # including the two-group edge case where three independent random
        # permutations could all coincide and merely rename intact groups.
        shifts = (0, 1, 2 if len(groups) > 2 else 1)
        permutations = [base[shift:] + base[:shift] for shift in shifts]
        for target_index, target_group in enumerate(groups):
            placebo_id = "placebo|%s|%06d" % (operation, target_index)
            source_groups = []
            for family in range(3):
                source_group = groups[permutations[family][target_index]]
                episode_id = by_group[source_group][family]
                assignment[episode_id] = placebo_id
                source_groups.append(source_group)
            receipt_rows.append(
                {"placebo_group": placebo_id, "source_groups_by_family": source_groups}
            )
            if len(set(source_groups)) < 2:
                raise LearningContractError("J3 placebo did not change group membership")
    if not assignment:
        raise LearningContractError("J3 placebo assignment is empty")
    digest = hashlib.sha256(
        json.dumps(receipt_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return assignment, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--manifest-receipt", type=Path, required=True)
    parser.add_argument("--lineage-receipt", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pointer-targets", type=Path, default=None)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=["J0", "J3", "J4", "J5", "J9C"], required=True
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    # R13 mainline is natural single-round (D-2026-08-17-01 cutover);
    # "matched" serves only the legacy v2 controlled lane / J3 placebo.
    parser.add_argument(
        "--dataset-mode", choices=["matched", "natural"], default="natural"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.arm in ("J5", "J9C") and args.pointer_targets is None:
        parser.error("pointer-enabled compiler arms require --pointer-targets")
    if args.dataset_mode == "natural" and args.arm not in ("J0", "J9C"):
        parser.error("the single-round natural mainline supports J0 or J9C")
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
    visible_ids = {str(row["episode_id"]) for row in rows}
    if visible_ids != set(labels):
        raise LearningContractError("visible and label manifests do not have exact coverage")
    group_override: dict[str, str] = {}
    group_shuffle_sha = None
    if args.arm == "J3":
        group_override, group_shuffle_sha = _build_placebo_groups(
            labels,
            partition="train",
            seed=int(arm_config["group_shuffle_seed"]),
        )
    train = InferenceEpisodeDataset(args.episodes, "train", candidates, labels)
    val = InferenceEpisodeDataset(args.episodes, "val", candidates, labels)
    train_rows = [row for row in rows if row["partition"] == "train"]
    effective_groups = []
    if args.dataset_mode == "matched":
        effective_groups = [
            group_override.get(
                str(row["episode_id"]),
                "%s|%s" % (
                    labels[str(row["episode_id"])]["matched_state_group_id"],
                    labels[str(row["episode_id"])]["operation"],
                ),
            )
            for row in train_rows
        ]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if args.dataset_mode == "matched":
        train_loader = DataLoader(
            train,
            batch_sampler=GroupedBatchSampler(
                effective_groups, args.batch_size, args.seed
            ),
            collate_fn=program_collate,
        )
    else:
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=program_collate,
        )
    val_loader = DataLoader(
        val, batch_size=args.batch_size, shuffle=False, collate_fn=program_collate
    )
    model = ProgramCompilerNet(
        include_repair=bool(config["grammar"]["include_repair"])
    ).to(device)
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
                    [
                        group_override.get(str(episode_id), str(group_id))
                        for episode_id, group_id in zip(
                            batch["episode_id"], batch["group_id"]
                        )
                    ],
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
        val_loss = 0.0
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
                predicted = output["family_logits"].argmax(dim=1).cpu()
                operation_offset = batch["operation_id"].cpu() * 3
                val_true.extend((family_gold.cpu() + operation_offset).tolist())
                val_pred.extend((predicted + operation_offset).tolist())
                val_patients.extend(list(batch["patient_id"]))
                val_loss += float(
                    torch.nn.functional.cross_entropy(
                        output["family_logits"], family_gold.to(device), reduction="sum"
                    )
                )
        metric = patient_balanced_macro_f1_summary(
            val_true, val_pred, val_patients, list(range(6))
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(train),
            "val_loss": val_loss / len(val),
            "val_patient_balanced_family_macro_f1": metric["estimate"],
            "val_per_family_support": metric["per_class_support"],
        }
        history.append(row)
        score = (row["val_patient_balanced_family_macro_f1"], -row["val_loss"])
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
                "membership_sha256": group_shuffle_sha,
            }
        ),
        "architecture_id": ARCHITECTURE_ID,
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


if __name__ == "__main__":
    raise SystemExit(main())
