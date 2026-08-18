#!/usr/bin/env python3
"""B2: five-round 2D trajectory ceiling on R13 VAL (gold program, real editor).

This runner measures the 1-5 round trajectory ceiling of the frozen
effect-val editors on the VAL partition: the ceiling program is gold while
the state advance is the ACTUAL editor output restricted to the prompted
plane (never a GT-perfect correction).

Round 1 consumes the natural-lane oracle calls (label-derived gold program,
same artifact as the B1 gold-call ceiling pass) on the frozen episode crop.
Rounds 2-5 regenerate an official-simulator scribble on the CURRENT state
FN/FP residual, derive a residual-driven gold call, run the editor on the
runtime crop, and rebuild the prompted axial slice back into the 3D state
(plane-edit reconstruction).

Per round the runner records delta Dice in the ``2d_prompted_plane`` domain
on two aligned grids: crop-space (the editor I/O domain; round 1
cross-checks the B1 gold pass) and state-grid (the prompted axial slice of
the advancing volume).  Output is one sealed per-arm trajectory JSON.

Boundaries: VAL only (locked TEST rows are refused at the case-join guard);
editor/compiler checkpoints and all lane artifacts are read-only inputs; the
official simulator is sha-pinned.  The per-episode loop and the policy
functions live in ``petct_b2_trajectory_engine.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_mainline_lineage import (  # noqa: E402
    MAINLINE_SOURCE,
    validate_r13_lineage_receipt,
)
from common.petct_program_learning import (  # noqa: E402
    LearningContractError,
    _sha256_file,
    load_jsonl,
)
from common.petct_program_models import (  # noqa: E402
    ProgramCompilerNet,
    ProgramEditorUNet2D,
)
from common.petct_w21_test_access import (  # noqa: E402
    OFFICIAL_SIMULATOR_SHA256,
    PROTOCOL,
)
from evaluation.petct_b2_trajectory_engine import (  # noqa: E402,F401
    POLICY_ROUND_1,
    POLICY_ROUNDS_2_PLUS,
    _arm_conditioning,
    _gold_call_residual_driven,
    _mean_defined,
    _patient_balanced_mean,
    _require_arm_from_checkpoint,
    _require_val_case,
    run_episode,
)
from evaluation.run_petct_w21_official_test import (  # noqa: E402
    _load_module,
    _regular,
    _seal,
    _write_json_exclusive,
)

try:  # flat import when scripts/editor precedes the package path
    import infer_petct_program_editor_v3 as editor_infer  # noqa: E402
except ImportError:  # pragma: no cover - package-style fallback
    # plain dotted import (not ImportFrom) keeps the R13 audit's
    # namespace-package resolver quiet — scripts/editor has no __init__.py
    import editor.infer_petct_program_editor_v3 as editor_infer  # noqa: E402,F401

ARM_SCHEMA = "PETCT-B2-TRAJECTORY-CEILING-ARM-v1.0"
TRAJECTORY_ROUNDS = int(PROTOCOL["correction_rounds"])
EDITOR_CHECKPOINT_SCHEMA = "PETCT-PROGRAM-EDITOR-CHECKPOINT-v1.0"
COMPILER_CHECKPOINT_SCHEMA = "PETCT-PROGRAM-COMPILER-CHECKPOINT-v1.0"


def _tree_sha256(directory: Path) -> str:
    records = []
    for path in sorted(value for value in directory.rglob("*") if value.is_file()):
        records.append((path.relative_to(directory).as_posix(), _sha256_file(path)))
    return hashlib.sha256(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_oracle_calls(calls_path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(calls_path):
        episode_id = str(row.get("episode_id") or "")
        if not episode_id or episode_id in result:
            raise LearningContractError(
                "oracle calls have missing/duplicate episode_id"
            )
        result[episode_id] = dict(row)
    return result


def _load_episodes(episodes_path: Path, partition: str) -> dict[str, dict[str, Any]]:
    forbidden = {
        "patient_id",
        "case_id",
        "goal",
        "matched_state_group_id",
        "evaluation_npz",
        "evaluation_sha256",
    }
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(episodes_path):
        if any(key in row for key in forbidden):
            raise LearningContractError("episodes manifest contains label-lane fields")
        if str(row.get("partition") or "") != partition:
            continue
        episode_id = str(row.get("episode_id") or "")
        if not episode_id or episode_id in result:
            raise LearningContractError(
                "episodes manifest has missing/duplicate episode_id"
            )
        for key in ("operation", "visible_npz", "visible_sha256"):
            if not row.get(key):
                raise LearningContractError(
                    "episode row is incomplete: %s" % episode_id
                )
        result[episode_id] = dict(row)
    if not result:
        raise LearningContractError("episodes partition is empty")
    return result


def _load_rich(rich_path: Path, partition: str) -> dict[str, dict[str, Any]]:
    required_evaluation = (
        "gt_path",
        "gt_sha256",
        "m0_path",
        "m0_sha256",
        "authorized_path",
        "authorized_sha256",
        "center_z",
        "scribble_coordinates_xyz",
    )
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(rich_path):
        if str(row.get("partition") or "") != partition:
            continue
        episode_id = str(row.get("episode_id") or "")
        if not episode_id or episode_id in result:
            raise LearningContractError(
                "rich manifest has missing/duplicate episode_id"
            )
        source = row.get("source_evaluation")
        if not isinstance(source, Mapping) or any(
            not source.get(key) for key in required_evaluation
        ):
            raise LearningContractError("rich row lacks hash-bound evaluation sources")
        for key in (
            "case_id",
            "patient_id",
            "strategy",
            "geometry",
            "center_z",
            "evaluation_npz",
            "evaluation_sha256",
            "operation",
            "goal",
        ):
            if key not in row or row[key] is None:
                raise LearningContractError("rich row is incomplete: %s" % episode_id)
        result[episode_id] = dict(row)
    return result


def _load_case_manifest(case_manifest_path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(case_manifest_path):
        case_id = str(row.get("case_id") or "")
        if not case_id or case_id in result:
            raise LearningContractError("case manifest has missing/duplicate case_id")
        result[case_id] = dict(row)
    if not result:
        raise LearningContractError("case manifest is empty")
    return result


def _load_editor_bundle(
    checkpoint_path: Path,
    episodes_path: Path,
    candidates_dir: Path,
    lineage: Mapping[str, Any],
    device: torch.device,
) -> tuple[ProgramEditorUNet2D, dict[str, Any], str]:
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    arm = _require_arm_from_checkpoint(str(checkpoint.get("arm") or ""))
    candidates_tree_sha = _tree_sha256(candidates_dir)
    if (
        checkpoint.get("schema_version") != EDITOR_CHECKPOINT_SCHEMA
        or checkpoint.get("episodes_sha256") != _sha256_file(episodes_path)
        or checkpoint.get("candidates_tree_sha256") != candidates_tree_sha
        or checkpoint.get("source_m0_lineage") != MAINLINE_SOURCE
        or checkpoint.get("lineage_receipt_sha256") != lineage["receipt_sha256"]
    ):
        raise LearningContractError(
            "editor checkpoint is not bound to trajectory inputs"
        )
    model = ProgramEditorUNet2D(
        visual_channels=12 if arm == "J6" else 13,
        conditioner="continuous" if arm == "J8" else "program",
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint, arm


def _load_compiler_bundle(
    checkpoint_path: Path, device: torch.device
) -> ProgramCompilerNet:
    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema_version") != COMPILER_CHECKPOINT_SCHEMA:
        raise LearningContractError("compiler checkpoint schema is not the frozen v1.0")
    model = ProgramCompilerNet(
        include_repair=bool(checkpoint["hyperparameters"]["include_repair"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--rich-manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--oracle-calls", type=Path, required=True)
    parser.add_argument("--oracle-receipt", type=Path, required=True)
    parser.add_argument("--editor-checkpoint", type=Path, required=True)
    parser.add_argument("--compiler-checkpoint", type=Path, default=None)
    parser.add_argument("--lineage-receipt", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "val"), default="val")
    parser.add_argument(
        "--official-simulator",
        type=Path,
        default=SCRIPTS_ROOT.parent
        / "upstream"
        / "autoPETV"
        / "interactive"
        / "simulate_scribbles.py",
    )
    parser.add_argument("--field-mm", type=float, default=64.0)
    parser.add_argument("--output-size", type=int, default=128)
    parser.add_argument("--expected-spacing", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    try:
        lineage = validate_r13_lineage_receipt(args.lineage_receipt)
        simulator_path = _regular(args.official_simulator, label="official simulator")
        if _sha256_file(simulator_path) != OFFICIAL_SIMULATOR_SHA256:
            raise LearningContractError("official simulator is not the pinned file")
        simulator = _load_module(simulator_path, "b2_official_simulator")
        verified_oracle = editor_infer._verify_call_receipt(
            args.oracle_receipt, args.oracle_calls
        )
        if verified_oracle.get("program_source") != "gold_oracle_ceiling":
            raise LearningContractError("trajectory oracle calls must be gold")
        if verified_oracle.get("natural_lane") is not True:
            raise LearningContractError(
                "trajectory oracle receipt is not the natural-lane variant"
            )
        oracle_calls = _load_oracle_calls(args.oracle_calls)
        episodes = _load_episodes(args.episodes, args.partition)
        rich_rows = _load_rich(args.rich_manifest, args.partition)
        if set(episodes) != set(rich_rows):
            raise LearningContractError(
                "episodes and rich manifest disagree on episodes"
            )
        candidates = editor_infer._load_candidates(args.candidates)
        case_manifest = _load_case_manifest(args.case_manifest)
        device = torch.device(args.device)
        editor_model, editor_checkpoint, arm = _load_editor_bundle(
            args.editor_checkpoint, args.episodes, args.candidates, lineage, device
        )
        compiler = None
        if arm == "J8":
            if args.compiler_checkpoint is None:
                parser.error("J8 requires --compiler-checkpoint")
            compiler = _load_compiler_bundle(args.compiler_checkpoint, device)
            if editor_checkpoint.get("compiler_checkpoint_sha256") != _sha256_file(
                args.compiler_checkpoint
            ):
                raise LearningContractError(
                    "J8 inference compiler differs from the frozen training compiler"
                )
        model_config = {
            "field_mm": args.field_mm,
            "output_size": args.output_size,
            "expected_spacing": args.expected_spacing,
            "trajectory_rounds": TRAJECTORY_ROUNDS,
        }
        episode_records = []
        case_cache: dict[str, Any] = {}
        component_cache: dict[str, Any] = {}
        for episode_id in sorted(episodes):
            episode_records.append(
                run_episode(
                    episode_id=episode_id,
                    row=episodes[episode_id],
                    rich=rich_rows[episode_id],
                    case_manifest=case_manifest,
                    oracle_calls=oracle_calls,
                    candidates=candidates,
                    simulator=simulator,
                    editor_model=editor_model,
                    compiler=compiler,
                    arm=arm,
                    model_config=model_config,
                    device=device,
                    case_cache=case_cache,
                    component_cache=component_cache,
                )
            )
        per_round_crop: dict[int, list[tuple[str, float]]] = {}
        per_round_state: dict[int, list[tuple[str, float]]] = {}
        for episode in episode_records:
            for record in episode["rounds"]:
                round_index = int(record["round"])
                for bucket, key in (
                    (per_round_crop, "crop_plane"),
                    (per_round_state, "state_plane"),
                ):
                    delta = record[key]["delta_dice"]
                    if delta is not None:
                        bucket.setdefault(round_index, []).append(
                            (str(episode["patient_id"]), float(delta))
                        )

        def aggregate(bucket):
            return {
                str(round_index): _mean_defined(
                    [value for _, value in bucket.get(round_index, [])]
                )
                for round_index in range(1, TRAJECTORY_ROUNDS + 1)
            }

        payload = _seal(
            {
                "schema_version": ARM_SCHEMA,
                "arm": arm,
                "partition": args.partition,
                "source_m0_lineage": MAINLINE_SOURCE,
                "lineage_receipt_sha256": lineage["receipt_sha256"],
                "protocol": {
                    "trajectory_rounds": TRAJECTORY_ROUNDS,
                    "state_semantics": (
                        "state_0=M0; state_i=plane-edit reconstruction after i "
                        "actual editor corrections (never GT substitution)"
                    ),
                    "round_1_gold_call_policy": POLICY_ROUND_1,
                    "rounds_2_to_5_gold_call_policy": POLICY_ROUNDS_2_PLUS,
                    "rounds_2_to_5_gold_call_rule": {
                        "REMOVE": "DELETE_COMPONENT at the deterministic cue-hit component",
                        "ADD": "COMPLETE_EXISTING at max scribble-overlap component "
                        "(ties: nearest cue distance then position); CREATE_NEW "
                        "when nothing overlaps",
                    },
                    "simulator": {
                        "source": "official autoPETV simulate_scribbles.py (sha-pinned)",
                        "seed": int(PROTOCOL["simulator_seed"]),
                        "residual": "current-state FN/FP at every round",
                    },
                    "domain": "2d_prompted_plane",
                    "checkpoint_lock": "effect-val frozen editor weights; no retraining",
                },
                "checkpoint_bindings": {
                    "editor_checkpoint_sha256": _sha256_file(args.editor_checkpoint),
                    "compiler_checkpoint_sha256": (
                        _sha256_file(args.compiler_checkpoint)
                        if args.compiler_checkpoint is not None
                        else None
                    ),
                    "oracle_receipt_sha256": _sha256_file(args.oracle_receipt),
                },
                "episodes": episode_records,
                "aggregates": {
                    "episode_count": len(episode_records),
                    "patient_count": len(
                        {str(episode["patient_id"]) for episode in episode_records}
                    ),
                    "per_round_mean_crop_delta_dice": aggregate(per_round_crop),
                    "per_round_patient_balanced_mean_crop_delta_dice": _patient_balanced_mean(
                        per_round_crop
                    ),
                    "per_round_mean_state_delta_dice": aggregate(per_round_state),
                    "per_round_patient_balanced_mean_state_delta_dice": _patient_balanced_mean(
                        per_round_state
                    ),
                    "per_round_defined_episode_counts": {
                        str(round_index): len(per_round_crop.get(round_index, []))
                        for round_index in range(1, TRAJECTORY_ROUNDS + 1)
                    },
                },
            },
            "arm_sha256",
        )
    except LearningContractError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(args.output, payload)
    print(json.dumps({"status": "PASS", "arm": arm, "episodes": len(episode_records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
