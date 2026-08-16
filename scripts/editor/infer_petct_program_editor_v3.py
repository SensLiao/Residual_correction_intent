#!/usr/bin/env python3
"""Run a frozen v3 editor from compiler calls without opening label lanes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_program_contract import (  # noqa: E402
    NEW_CUE_SENTINEL,
    family_to_id,
    validate_legal_call,
)
from common.petct_program_learning import (  # noqa: E402
    LearningContractError,
    _load_components,
    _load_visible_bundle,
    _sha256_file,
    load_jsonl,
    operation_id_from_row,
    program_collate,
)
from common.petct_program_models import (  # noqa: E402
    NULL_FAMILY_ID,
    ProgramCompilerNet,
    ProgramEditorUNet2D,
)

RECEIPT_SCHEMA = "PETCT-PROGRAM-EDITOR-PREDICTIONS-v1.0"


def _unique(rows, role):
    result = {}
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if not episode_id or episode_id in result:
            raise LearningContractError("%s has missing/duplicate episode_id" % role)
        result[episode_id] = dict(row)
    return result


def _load_candidates(directory: Path):
    if not directory.is_dir():
        raise LearningContractError("missing candidate directory")
    return _unique(
        [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))],
        "candidates",
    )


def _verify_call_receipt(path: Path, predictions: Path) -> dict:
    if not path.is_file():
        raise LearningContractError("missing compiler prediction receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    schema = receipt.get("schema_version")
    if schema == "PETCT-PROGRAM-PREDICTIONS-v1.0":
        artifact_path = receipt.get("predictions_path")
        artifact_sha = receipt.get("predictions_sha256")
        if receipt.get("label_lane_opened") is not False:
            raise LearningContractError("compiler receipt opened a label lane")
        receipt["program_source"] = "predicted_compiler"
    elif schema == "PETCT-PROGRAM-ORACLE-CALLS-READY-v1.0":
        artifact_path = receipt.get("calls_path")
        artifact_sha = receipt.get("calls_sha256")
        if receipt.get("label_lane_opened") is not True or receipt.get("oracle_calls") is not True:
            raise LearningContractError("oracle receipt does not declare label access")
        receipt["program_source"] = "gold_oracle_ceiling"
    else:
        raise LearningContractError("unknown program call receipt schema")
    if (
        receipt.get("status") != "PASS"
        or Path(str(artifact_path or "")).resolve() != predictions.resolve()
        or artifact_sha != _sha256_file(predictions)
    ):
        raise LearningContractError("program call receipt is invalid or stale")
    return receipt


class _EditorInferenceDataset(Dataset):
    def __init__(self, manifest: Path, partition: str, candidates, predictions, arm: str):
        rows = load_jsonl(manifest)
        forbidden = {
            "patient_id", "case_id", "goal", "matched_state_group_id",
            "evaluation_npz", "evaluation_sha256",
        }
        if any(forbidden & set(row) for row in rows):
            raise LearningContractError("editor inference manifest contains label-lane fields")
        self.rows = [row for row in rows if row.get("partition") == partition]
        if not self.rows:
            raise LearningContractError("editor inference partition is empty")
        self.candidates = candidates
        self.predictions = predictions
        self.arm = arm
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        episode_id = str(row["episode_id"])
        if episode_id not in self.predictions or episode_id not in self.candidates:
            raise LearningContractError("editor inference join is incomplete")
        prediction = self.predictions[episode_id]
        bundle = _load_visible_bundle(row, episode_id)
        operation, operation_id = operation_id_from_row(
            row, bundle["cue_fg"], bundle["cue_bg"]
        )
        vectors, valid, masks, keys, cue_hit = _load_components(
            self.candidates, episode_id, self.cache
        )
        decision = str(prediction.get("decision") or "")
        selected = np.zeros_like(bundle["m0"], dtype=np.float32)
        family_id = NULL_FAMILY_ID
        operand_mode = 2
        active = 0
        if decision == "PREDICT":
            family = str(prediction.get("family") or "")
            operand = str(prediction.get("operand") or "")
            if str(prediction.get("operation")) != operation:
                raise LearningContractError("compiler call changed cue-signed operation")
            validate_legal_call(operation, family, operand)
            family_id = family_to_id(family)
            operand_mode = 1 if operand == NEW_CUE_SENTINEL else 0
            active = 1
            if operand != NEW_CUE_SENTINEL:
                try:
                    position = keys.index(operand)
                except ValueError:
                    raise LearningContractError("compiler operand absent from candidates") from None
                if operation == "REMOVE" and position != cue_hit:
                    raise LearningContractError("REMOVE operand is not deterministic cue-hit")
                if masks is None:
                    raise LearningContractError("selected operand lacks crop-aligned mask")
                selected = np.asarray(masks[position], dtype=np.float32)
                if selected.shape != bundle["m0"].shape:
                    raise LearningContractError("selected component geometry mismatch")
        elif decision != "ABSTAIN":
            raise LearningContractError("unknown compiler decision")
        signed_cue = bundle["visual"][15:16] - bundle["visual"][16:17]
        visual = np.concatenate(
            [bundle["visual"][:10], bundle["visual"][12:13], signed_cue], axis=0
        )
        if self.arm != "J6":
            visual = np.concatenate([visual, selected[None]], axis=0)
        if self.arm in ("J6", "J8"):
            family_for_editor = NULL_FAMILY_ID
            operand_for_editor = 2
        elif self.arm == "J7":
            family_for_editor = family_id
            operand_for_editor = 0 if active else 2
        else:
            family_for_editor = family_id
            operand_for_editor = operand_mode
        output = {
            "episode_id": episode_id,
            "decision": decision,
            "visual": torch.from_numpy(visual),
            "m0": torch.from_numpy(bundle["m0"][None]),
            "selected_component": torch.from_numpy(selected[None]),
            "operation_id": torch.tensor(operation_id, dtype=torch.long),
            "family_id": torch.tensor(family_for_editor, dtype=torch.long),
            "operand_mode": torch.tensor(operand_for_editor, dtype=torch.long),
            "support_mode": torch.tensor(0 if active else 1, dtype=torch.long),
            "active": torch.tensor(bool(active), dtype=torch.bool),
        }
        if self.arm == "J8":
            output.update(
                {
                    "compiler_visual": torch.from_numpy(bundle["visual"]),
                    "spacing_xy": torch.from_numpy(bundle["spacing_xy"]),
                    "component_vectors": torch.from_numpy(vectors),
                    "component_mask": torch.from_numpy(valid),
                }
            )
        return output


def _load_compiler(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "PETCT-PROGRAM-COMPILER-CHECKPOINT-v1.0":
        raise LearningContractError("invalid compiler checkpoint")
    model = ProgramCompilerNet(
        include_repair=bool(checkpoint["hyperparameters"]["include_repair"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--program-predictions", type=Path, required=True)
    parser.add_argument("--program-receipt", type=Path, required=True)
    parser.add_argument("--editor-checkpoint", type=Path, required=True)
    parser.add_argument("--compiler-checkpoint", type=Path, default=None)
    parser.add_argument("--partition", choices=("train", "val"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    outputs = (args.output_dir, args.output_manifest, args.receipt)
    if any(path.exists() or path.is_symlink() for path in outputs):
        parser.error("output already exists")
    program_receipt = _verify_call_receipt(
        args.program_receipt, args.program_predictions
    )
    predictions = _unique(load_jsonl(args.program_predictions), "program predictions")
    candidates = _load_candidates(args.candidates)
    checkpoint = torch.load(args.editor_checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "PETCT-PROGRAM-EDITOR-CHECKPOINT-v1.0"
        or checkpoint.get("episodes_sha256") != _sha256_file(args.episodes)
        or checkpoint.get("candidates_tree_sha256")
        != program_receipt.get("candidates_tree_sha256")
    ):
        raise LearningContractError("editor checkpoint is not bound to inference manifest")
    arm = str(checkpoint.get("arm") or "")
    if arm not in ("J6", "J7", "J8", "J9"):
        raise LearningContractError("unknown editor arm")
    if arm == "J8" and args.compiler_checkpoint is None:
        parser.error("J8 requires --compiler-checkpoint")
    device = torch.device(args.device)
    if arm == "J8":
        compiler_sha = _sha256_file(args.compiler_checkpoint)
        if checkpoint.get("compiler_checkpoint_sha256") != compiler_sha:
            raise LearningContractError(
                "J8 inference compiler differs from the frozen training compiler"
            )
        if program_receipt.get("program_source") == "predicted_compiler" and (
            program_receipt.get("checkpoint_sha256") != compiler_sha
        ):
            raise LearningContractError(
                "J8 continuous features and program predictions use different compilers"
            )
    editor = ProgramEditorUNet2D(
        visual_channels=12 if arm == "J6" else 13,
        conditioner="continuous" if arm == "J8" else "program",
    ).to(device)
    editor.load_state_dict(checkpoint["state_dict"], strict=True)
    editor.eval()
    compiler = _load_compiler(args.compiler_checkpoint, device) if arm == "J8" else None
    dataset = _EditorInferenceDataset(
        args.episodes, args.partition, candidates, predictions, arm
    )
    expected_ids = [str(row["episode_id"]) for row in dataset.rows]
    if set(predictions) != set(expected_ids):
        raise LearningContractError("program predictions require exact partition coverage")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=program_collate
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    with torch.no_grad():
        for batch in loader:
            state_embedding = None
            if compiler is not None:
                state_embedding = compiler(
                    batch["compiler_visual"].to(device),
                    batch["spacing_xy"].to(device),
                    batch["operation_id"].to(device),
                    batch["component_vectors"].to(device),
                    batch["component_mask"].to(device),
                )["embedding"]
            logits = editor(
                batch["visual"].to(device),
                batch["family_id"].to(device),
                batch["operand_mode"].to(device),
                batch["support_mode"].to(device),
                state_embedding=state_embedding,
                active_mask=batch["active"].to(device),
            )
            delta, _ = ProgramEditorUNet2D.apply_program_operation(
                logits,
                batch["m0"].to(device),
                batch["selected_component"].to(device),
                batch["operation_id"].to(device),
            )
            # ABSTAIN is an identity edit even if a model bias is nonzero.
            delta = delta & batch["active"].to(device)[:, None, None, None]
            for index, episode_id in enumerate(batch["episode_id"]):
                path = args.output_dir / ("%s.npz" % episode_id)
                np.savez_compressed(path, delta=delta[index, 0].cpu().numpy().astype(np.uint8))
                records.append(
                    {
                        "episode_id": str(episode_id),
                        "decision": str(batch["decision"][index]),
                        "delta_path": str(path.resolve()),
                        "delta_sha256": _sha256_file(path),
                    }
                )
    if [row["episode_id"] for row in records] != expected_ids:
        raise LearningContractError("editor output order/coverage mismatch")
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("x", encoding="utf-8", newline="\n") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "PASS",
        "arm": arm,
        "partition": args.partition,
        "episode_count": len(records),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": _sha256_file(args.output_manifest),
        "editor_checkpoint_sha256": _sha256_file(args.editor_checkpoint),
        "compiler_predictions_sha256": _sha256_file(args.program_predictions),
        "compiler_prediction_receipt_sha256": _sha256_file(args.program_receipt),
        "compiler_checkpoint_sha256": program_receipt.get("checkpoint_sha256"),
        "program_source": program_receipt["program_source"],
        "label_lane_opened": program_receipt["program_source"] == "gold_oracle_ceiling",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "PASS", "episodes": len(records), "arm": arm}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
