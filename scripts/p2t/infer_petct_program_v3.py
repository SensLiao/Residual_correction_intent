#!/usr/bin/env python3
"""Run label-free v3 compiler inference and emit immutable legal calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_program_contract import (  # noqa: E402
    GRAMMAR_VERSION,
    SCHEMA_VERSION,
    protected_refs_policy,
    render_goal,
)
from common.petct_program_learning import (  # noqa: E402
    InferenceEpisodeDataset,
    LearningContractError,
    _sha256_file,
    program_collate,
)
from common.petct_program_models import (  # noqa: E402
    LegalCallCompiler,
    ProgramCompilerNet,
)

PREDICTION_RECEIPT_SCHEMA = "PETCT-PROGRAM-PREDICTIONS-v1.0"


def _tree_sha256(directory: Path) -> str:
    records = []
    for path in sorted(value for value in directory.rglob("*") if value.is_file()):
        records.append((path.relative_to(directory).as_posix(), _sha256_file(path)))
    return hashlib.sha256(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_candidates(directory: Path) -> dict[str, dict[str, Any]]:
    if not directory.is_dir():
        raise LearningContractError("missing candidates directory")
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        episode_id = str(record.get("episode_id") or "")
        if not episode_id or episode_id in records:
            raise LearningContractError("candidate records have missing/duplicate episode_id")
        records[episode_id] = record
    if not records:
        raise LearningContractError("candidate directory is empty")
    return records


def _load_schema(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise LearningContractError("missing program JSON schema")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_json(instance: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError:
        raise RuntimeError("jsonschema is required for program inference") from None
    jsonschema.Draft202012Validator(schema).validate(instance)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "val"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=SCRIPTS_ROOT.parent / "schemas" / "petct_program_v1.schema.json",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output.resolve() == args.receipt.resolve():
        parser.error("prediction and receipt paths must differ")
    if any(path.exists() or path.is_symlink() for path in (args.output, args.receipt)):
        parser.error("output already exists")
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if not args.checkpoint.is_file():
        parser.error("missing compiler checkpoint")

    candidates = _load_candidates(args.candidates)
    candidate_tree_sha = _tree_sha256(args.candidates)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema_version") != "PETCT-PROGRAM-COMPILER-CHECKPOINT-v1.0"
        or checkpoint.get("architecture_id") != "matched_legal_component_program_v1"
        or checkpoint.get("episodes_sha256") != _sha256_file(args.episodes)
        or checkpoint.get("candidates_tree_sha256") != candidate_tree_sha
    ):
        raise LearningContractError("checkpoint is not bound to inference inputs")
    include_repair = bool(checkpoint["hyperparameters"]["include_repair"])
    device = torch.device(args.device)
    model = ProgramCompilerNet(include_repair=include_repair).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    compiler = LegalCallCompiler(include_repair=include_repair)
    dataset = InferenceEpisodeDataset(
        args.episodes, args.partition, candidates, training_labels=None
    )
    expected_ids = [str(row["episode_id"]) for row in dataset.rows]
    if set(expected_ids) - set(candidates):
        raise LearningContractError("candidate coverage is incomplete")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=program_collate,
    )
    schema = _load_schema(args.schema)
    checkpoint_sha = _sha256_file(args.checkpoint)
    manifest_sha = _sha256_file(args.episodes)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                batch["visual"].to(device),
                batch["spacing_xy"].to(device),
                batch["operation_id"].to(device),
                batch["component_vectors"].to(device),
                batch["component_mask"].to(device),
            )
            for index, episode_id in enumerate(batch["episode_id"]):
                episode_id = str(episode_id)
                operation = str(batch["operation"][index])
                candidate = candidates[episode_id]
                components = candidate.get("components", [])
                valid = batch["component_mask"][index].to(dtype=torch.bool)
                family_logits = outputs["family_logits"][index]
                if operation == "REMOVE" and not include_repair:
                    family_logits = family_logits[:2]
                pointer_probs = None
                if bool(valid.any()):
                    pointer_probs = torch.softmax(
                        outputs["pointer_logits"][index][valid.to(device)], dim=0
                    ).cpu()
                    # Valid candidates are contiguous by contract; keep a
                    # hard assertion before the compiler interprets indices.
                    if not bool(valid[: len(components)].all()) or bool(valid[len(components):].any()):
                        raise LearningContractError("padded candidate mask is not contiguous")
                compiled = compiler(
                    operation,
                    family_logits.cpu(),
                    pointer_probs,
                    components,
                    cue_hit_component_position=candidate.get(
                        "cue_hit_component_position"
                    ),
                )
                call = compiled["call"]
                goal = render_goal(operation, str(call["family"]))
                protected = dict(protected_refs_policy(operation, str(call["operand"])))
                typed_trace = list(compiled["typed_trace"])
                typed_trace.insert(
                    -1, {"step": "PROTECT", "protected_refs": protected}
                )
                confidence = float(torch.softmax(family_logits, dim=0).max().item())
                prediction = {
                    "schema_version": SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "decision": "PREDICT",
                    "operation": operation,
                    "family": str(call["family"]),
                    "operand": str(call["operand"]),
                    "goal": goal,
                    "protected_refs": protected,
                    "confidence": confidence,
                    "typed_trace": typed_trace,
                    "audit": {
                        "grammar_version": GRAMMAR_VERSION,
                        "enumeration_version": str(candidate["enumeration_version"]),
                        "m_sha256": str(candidate["m_sha256"]),
                        "manifest_sha256": manifest_sha,
                        "checkpoint_hash": checkpoint_sha,
                        "visibility_lane": "inference_visible",
                    },
                }
                _validate_json(prediction, schema)
                predictions.append(prediction)
    if [row["episode_id"] for row in predictions] != expected_ids:
        raise LearningContractError("prediction output coverage/order mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        for prediction in predictions:
            stream.write(json.dumps(
                prediction, ensure_ascii=False, sort_keys=True, allow_nan=False
            ))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "schema_version": PREDICTION_RECEIPT_SCHEMA,
        "status": "PASS",
        "partition": args.partition,
        "prediction_count": len(predictions),
        "predictions_path": str(args.output.resolve()),
        "predictions_sha256": _sha256_file(args.output),
        "episodes_sha256": manifest_sha,
        "candidates_tree_sha256": candidate_tree_sha,
        "checkpoint_sha256": checkpoint_sha,
        "schema_sha256": _sha256_file(args.schema),
        "label_lane_opened": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": "PASS", "predictions": len(predictions)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
