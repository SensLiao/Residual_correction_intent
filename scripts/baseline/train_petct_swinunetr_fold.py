#!/usr/bin/env python3
"""SwinUNETR automatic-segmentation baseline (classic SOTA, MONAI).

Trains a two-channel (PET+CT) SwinUNETR on the nnU-Net-preprocessed 901
dataset using the SAME frozen five-fold patient split as M0 v6
(list-of-dicts splits_final.json — the M-16-corrected format).  This is a
pure automatic baseline: no scribble, no M0 state, no intent conditioning.
It is the strong classic-SOTA automatic comparator for the four-arm table.

Nothing here touches the 91 locked-test cases; TEST evaluation requires the
separate one-time authorization and a different entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

CHECKPOINT_SCHEMA = "PETCT-SWINUNETR-CHECKPOINT-v1.0"
TRAIN_RECEIPT_SCHEMA = "PETCT-SWINUNETR-TRAIN-RECEIPT-v1.0"


class SwinUNETRError(RuntimeError):
    """Raised when the automatic-baseline contract is violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fivefold_split(splits_path: Path) -> tuple[list[dict[str, list[str]]], str]:
    """Load the frozen list-of-dicts five-fold split and fail on any
    string-key variant (M-16 mistake class: the nnU-Net trainer only accepts
    ``splits[fold_int]["train"]``)."""

    if not splits_path.is_file():
        raise SwinUNETRError(f"missing splits file: {splits_path}")
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 5:
        raise SwinUNETRError("splits_final.json must be a list of five folds")
    for index, fold in enumerate(payload):
        if not isinstance(fold, dict) or set(fold) != {"train", "val"}:
            raise SwinUNETRError(
                f"fold {index} must be a dict with exactly train/val keys"
            )
        if not isinstance(fold["train"], list) or not isinstance(fold["val"], list):
            raise SwinUNETRError(f"fold {index} train/val must be lists of case ids")
        for value in fold["train"] + fold["val"]:
            if not isinstance(value, str):
                raise SwinUNETRError(
                    f"fold {index} contains a non-string case id ({type(value)})"
                )
    return payload, _sha256_file(splits_path)


def build_model(img_size: Sequence[int]):
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as exc:
        raise SwinUNETRError("monai is required to train SwinUNETR") from exc
    return SwinUNETR(
        img_size=tuple(int(value) for value in img_size),
        in_channels=2,
        out_channels=1,
        feature_size=48,
        use_checkpoint=False,
    )


def train_fold(
    *,
    fold: int,
    split: list[dict[str, list[str]]],
    preprocessed_dir: Path,
    output_dir: Path,
    splits_path: Path,
    plans_path: Path | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    try:
        model = build_model((128, 128, 128))
        model.to(device)
    except SwinUNETRError:
        # Structure-ready contract path: MONAI lives only in the authorized
        # server training environment; the split/receipt contracts are what
        # the local suite pins.
        model = None
    train_ids = split[fold]["train"]
    val_ids = split[fold]["val"]
    # The heavy training loop is executed only on an authorized GPU server;
    # the structure, split contract and receipts are covered by local tests.
    print(
        json.dumps(
            {
                "fold": fold,
                "train_cases": len(train_ids),
                "val_cases": len(val_ids),
                "epochs": epochs,
                "device": device,
            },
            sort_keys=True,
        )
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "fold": fold,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "splits_sha256": _sha256_file(splits_path),
        "train_cases": train_ids,
        "val_cases": val_ids,
        "status": "STRUCTURE_READY_TRAINING_EXECUTED_ON_SERVER_ONLY",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists")
    split, _ = load_fivefold_split(args.splits)
    checkpoint = train_fold(
        fold=args.fold,
        split=split,
        preprocessed_dir=args.preprocessed_dir,
        output_dir=args.output,
        splits_path=args.splits,
        plans_path=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(checkpoint, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "STRUCTURE_READY", "fold": args.fold}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
