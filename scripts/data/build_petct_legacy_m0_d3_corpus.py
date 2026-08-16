#!/usr/bin/env python3
"""Build the legacy-M0 D3 corpus as matched scribble variants of D2 states.

D3 is not a 15-round trajectory.  Every accepted D2 state remains frozen while
centerline, random and boundary cues are generated in parallel on the exact D2
authorised target slice.  The source state, residual, intent and target are never
re-derived by this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common.petct_learning import (  # noqa: E402
    LearningContractError,
    load_jsonl,
    sha256_file,
    validate_manifest_rows_against_frozen_learning_split,
)
from common.petct_route_a_core import intent_slots_from_goal  # noqa: E402
from data.build_petct_scribble_dataset import (  # noqa: E402
    apply_strategy_identity_policy,
    resolve_scribble_generation_contract,
    staged_output_bundle,
    validate_official_simulator_provenance,
)
from data.build_petct_scribble_episode import (  # noqa: E402
    DEFAULT_RUNTIME_MANIFEST,
    STRATEGIES,
    EpisodeContractError,
    ResidualCueIneligibleError,
    build_episode_documents,
    canonical_intent_frame,
    generate_residual_scribble,
    load_official_simulator,
    publish_episode_documents,
)


DATASET_ID = "legacy_m0_D3_five_round_three_strategy"
SCHEMA_VERSION = "PETCT-LEGACY-M0-D3-v1.0"
READY_SCHEMA_VERSION = "PETCT-LEGACY-M0-D3-READY-v1.0"
NEAR_DUPLICATE_JACCARD_DEFAULT = 0.80
STRATEGY_ORDER = tuple(STRATEGIES)


def _hex_digest(value: Any, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 64:
        raise RuntimeError(f"{label} must be a SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise RuntimeError(f"{label} must be a SHA-256 digest") from exc
    return text


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _mask_sha256(mask: np.ndarray) -> str:
    binary = np.asarray(mask, dtype=np.bool_)
    header = json.dumps(
        {"shape": list(binary.shape), "axis_order": "xyz"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + np.packbits(binary.reshape(-1)).tobytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def d3_episode_id(source_d2_record_id: str, strategy: str) -> str:
    """Return a deterministic opaque identifier for one matched D3 variant."""

    if not source_d2_record_id or strategy not in STRATEGY_ORDER:
        raise ValueError("source D2 record id and canonical strategy are required")
    digest = hashlib.sha256(
        f"{SCHEMA_VERSION}|{source_d2_record_id}|{strategy}".encode("utf-8")
    ).hexdigest()
    return "petct-legacy-d3-" + digest[:24]


def d3_state_id(source: Mapping[str, Any]) -> str:
    source_id = str(source.get("episode_id") or "")
    state_sha256 = _hex_digest(source.get("state_sha256"), label="state sha256")
    authorized_sha256 = _hex_digest(
        source.get("authorized_sha256"), label="authorised sha256"
    )
    digest = hashlib.sha256(
        f"{SCHEMA_VERSION}|{source_id}|{state_sha256}|{authorized_sha256}".encode(
            "utf-8"
        )
    ).hexdigest()
    return "legacy-m0-state-" + digest[:24]


def fixed_authorized_slice(
    authorized: np.ndarray, *, center_z: int
) -> np.ndarray:
    """Freeze cue generation to the original D2 target and axial interaction plane."""

    mask = np.asarray(authorized) > 0
    if mask.ndim != 3 or not 0 <= center_z < mask.shape[2]:
        raise RuntimeError("invalid D2 authorised target or frozen center_z")
    output = np.zeros_like(mask, dtype=np.uint8)
    output[:, :, center_z] = mask[:, :, center_z]
    if not output.any():
        raise RuntimeError("D2 authorised target is empty on frozen center_z")
    return output


def _coordinate_set(record: Mapping[str, Any]) -> frozenset[tuple[int, int, int]]:
    coordinates = record.get("coordinates_xyz")
    if not isinstance(coordinates, list) or not coordinates:
        raise RuntimeError("generated scribble has no coordinates")
    return frozenset(tuple(int(value) for value in coord) for coord in coordinates)


def annotate_geometry_similarity(
    records: Sequence[dict[str, Any]], *, near_duplicate_jaccard: float
) -> dict[str, Any]:
    """Annotate, retain and count exact/near duplicate strategy geometries."""

    if not 0 < near_duplicate_jaccard < 1:
        raise ValueError("near-duplicate Jaccard threshold must be between 0 and 1")
    by_strategy = {str(record["strategy"]): record for record in records}
    if len(by_strategy) != len(records):
        raise RuntimeError("a D3 state contains duplicate strategy labels")
    exact_pairs: list[list[str]] = []
    near_pairs: list[dict[str, Any]] = []
    for record in records:
        record["validity_flags"] = {
            "source_state_frozen": True,
            "authorized_target_frozen": True,
            "center_z_frozen": True,
            "strategy_fallback": False,
            "geometry_exact_duplicate_of": [],
            "geometry_near_duplicate_of": [],
        }
    for left_index, left_strategy in enumerate(STRATEGY_ORDER):
        if left_strategy not in by_strategy:
            continue
        left = _coordinate_set(by_strategy[left_strategy])
        for right_strategy in STRATEGY_ORDER[left_index + 1 :]:
            if right_strategy not in by_strategy:
                continue
            right = _coordinate_set(by_strategy[right_strategy])
            jaccard = len(left & right) / len(left | right)
            if jaccard == 1.0:
                exact_pairs.append([left_strategy, right_strategy])
                by_strategy[left_strategy]["validity_flags"][
                    "geometry_exact_duplicate_of"
                ].append(right_strategy)
                by_strategy[right_strategy]["validity_flags"][
                    "geometry_exact_duplicate_of"
                ].append(left_strategy)
            elif jaccard >= near_duplicate_jaccard:
                near_pairs.append(
                    {
                        "strategies": [left_strategy, right_strategy],
                        "coordinate_jaccard": jaccard,
                    }
                )
                by_strategy[left_strategy]["validity_flags"][
                    "geometry_near_duplicate_of"
                ].append(right_strategy)
                by_strategy[right_strategy]["validity_flags"][
                    "geometry_near_duplicate_of"
                ].append(left_strategy)
    for record in records:
        flags = record["validity_flags"]
        flags["geometry_unique_among_generated_variants"] = not (
            flags["geometry_exact_duplicate_of"]
        )
    return {
        "exact_pairs": exact_pairs,
        "near_pairs": near_pairs,
        "distinct_geometry_count": len({_coordinate_set(record) for record in records}),
    }


def attach_geometry_audit(
    records: Sequence[dict[str, Any]], similarity: Mapping[str, Any]
) -> None:
    """Keep D3 geometry-quality provenance in materialized learning rows."""

    for record in records:
        generation = record.get("scribble_generation")
        if not isinstance(generation, dict):
            raise RuntimeError("generated D3 row lacks scribble_generation metadata")
        generation["validity_flags"] = dict(record["validity_flags"])
        generation["triplet_geometry_similarity"] = dict(similarity)


def _verified_source_path(
    row: Mapping[str, Any], path_key: str, hash_key: str, cache: dict[Path, str]
) -> Path:
    raw = Path(str(row.get(path_key) or ""))
    if raw.is_symlink():
        raise RuntimeError(f"{path_key} must be a non-symlink file")
    path = raw.resolve()
    if not path.is_file():
        raise RuntimeError(f"missing source artifact: {path_key}")
    expected = _hex_digest(row.get(hash_key), label=hash_key)
    observed = cache.get(path)
    if observed is None:
        observed = sha256_file(path)
        cache[path] = observed
    if observed != expected:
        raise RuntimeError(f"source artifact hash mismatch: {path_key}")
    return path


def _load_binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    value = np.asarray(image.dataobj)
    if value.ndim != 3 or not np.all(np.isfinite(value)):
        raise RuntimeError(f"invalid 3D binary source mask: {path}")
    return image, value > 0


def _same_geometry(reference: nib.Nifti1Image, other: nib.Nifti1Image) -> bool:
    return other.shape == reference.shape and np.allclose(
        other.affine, reference.affine, atol=1e-3, rtol=0
    )


def _validate_source_state(
    source: Mapping[str, Any], *, hash_cache: dict[Path, str]
) -> dict[str, Any]:
    required_paths = {
        key: _verified_source_path(source, key, sha_key, hash_cache)
        for key, sha_key in (
            ("ct_path", "ct_sha256"),
            ("pet_path", "pet_sha256"),
            ("state_path", "state_sha256"),
            ("m0_path", "m0_sha256"),
            ("gt_path", "gt_sha256"),
            ("fn_path", "fn_sha256"),
            ("fp_path", "fp_sha256"),
            ("authorized_path", "authorized_sha256"),
        )
    }
    if (
        required_paths["state_path"] != required_paths["m0_path"]
        or source.get("state_sha256") != source.get("m0_sha256")
    ):
        raise RuntimeError("D2 current state and materialisation M0 are not identical")
    operation, target, scope = intent_slots_from_goal(str(source.get("goal") or ""))
    if (
        source.get("operation") != operation
        or source.get("target") != target
        or source.get("scope") != scope
    ):
        raise RuntimeError("D2 intent slots differ from the canonical goal")
    state_image, state = _load_binary(required_paths["state_path"])
    gt_image, gt = _load_binary(required_paths["gt_path"])
    fn_image, fn = _load_binary(required_paths["fn_path"])
    fp_image, fp = _load_binary(required_paths["fp_path"])
    authorized_image, authorized = _load_binary(required_paths["authorized_path"])
    if not all(
        _same_geometry(gt_image, image)
        for image in (state_image, fn_image, fp_image, authorized_image)
    ):
        raise RuntimeError("D2 state/GT/residual/authorised geometry mismatch")
    expected_fn = gt & ~state
    expected_fp = state & ~gt
    if not np.array_equal(fn, expected_fn) or not np.array_equal(fp, expected_fp):
        raise RuntimeError("D2 residual masks differ from the frozen current state")
    legal_residual = expected_fn if operation == "ADD" else expected_fp
    if not authorized.any() or np.any(authorized & ~legal_residual):
        raise RuntimeError("D2 authorised target is not a non-empty legal residual subset")
    stats = source.get("frozen_target_stats")
    if not isinstance(stats, Mapping):
        raise RuntimeError("D2 state lacks frozen_target_stats")
    center_z = int(stats.get("center_z", -1))
    if int(stats.get("authorized_voxels", -1)) != int(authorized.sum()):
        raise RuntimeError("D2 frozen authorised voxel count changed")
    if stats.get("target") != target:
        raise RuntimeError("D2 frozen target label changed")
    original_coordinates = source.get("coordinates_xyz")
    if not isinstance(original_coordinates, list) or not original_coordinates:
        raise RuntimeError("D2 source scribble is missing")
    if {int(coord[2]) for coord in original_coordinates} != {center_z}:
        raise RuntimeError("D2 source scribble differs from frozen center_z")
    return {
        "paths": required_paths,
        "authorized": authorized,
        "center_z": center_z,
        "operation": operation,
        "target": target,
        "scope": scope,
        "legal_residual_voxels": int(legal_residual.sum()),
        "legal_residual_mask_sha256": _mask_sha256(legal_residual),
    }


def _attempt_id(source_d2_record_id: str, strategy: str) -> str:
    return f"{d3_episode_id(source_d2_record_id, strategy)}::attempt"


def _base_variant(
    *,
    source: Mapping[str, Any],
    validated: Mapping[str, Any],
    strategy: str,
    scribble: Mapping[str, Any],
    generation: Mapping[str, Any],
    simulator_provenance: Mapping[str, Any],
    expected_oof_ready_sha256: str,
    source_d2_manifest_sha256: str,
) -> dict[str, Any]:
    source_id = str(source["episode_id"])
    episode_id = d3_episode_id(source_id, strategy)
    record = dict(scribble)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "legacy_m0_version": (
            "PETCT-M0-OOF-v1.0@" + expected_oof_ready_sha256[:12]
        ),
        "source_D2_record_id": source_id,
        "source_D2_manifest_sha256": source_d2_manifest_sha256,
        "source_D2_strategy": str(source.get("strategy") or ""),
        "case_id": str(source["case_id"]),
        "patient_id": str(source["patient_id"]).casefold(),
        "study_id": str(source["case_id"]),
        "partition": str(source["partition"]),
        "held_out_fold": int(source["held_out_fold"]),
        "round": int(source["round"]),
        "round_id": int(source["round"]),
        "state_id": d3_state_id(source),
        "episode_id": episode_id,
        "attempt_id": _attempt_id(source_id, strategy),
        "generation_status": "GENERATED",
        "goal": str(source["goal"]),
        "operation": str(validated["operation"]),
        "target": str(validated["target"]),
        "scope": str(validated["scope"]),
        "cue_polarity": str(record["polarity"]),
        "strategy": strategy,
        "scribble_strategy": strategy,
        "requested_strategy": str(record["requested_strategy"]),
        "effective_strategy": str(record["effective_strategy"]),
        "strategy_fallback": bool(record["strategy_fallback"]),
        "strategy_mode": "all",
        "strategy_salt": str(generation["strategy_salt"]),
        "strategy_assignment": "parallel_matched_alternatives_per_D2_state",
        "seed": int(generation["seed"]),
        "scribble_seed": int(generation["seed"]),
        "coordinates_xyz": record["coordinates_xyz"],
        "coordinate_count": int(record["coordinate_count"]),
        "coordinate_sha256": str(record["coordinate_sha256"]),
        "scribble_density_mode": str(record["scribble_density_mode"]),
        "fallback_mode": str(record["fallback_mode"]),
        "center_z": int(validated["center_z"]),
        "frozen_target_stats": dict(source["frozen_target_stats"]),
        "target_stats": dict(source["frozen_target_stats"]),
        "m0_provenance": dict(source["m0_provenance"]),
        "learning_split_sha256": str(source["learning_split_sha256"]),
        "experiment_config_sha256": str(source["experiment_config_sha256"]),
        "test_access_receipt_sha256": None,
        "official_source_provenance": dict(simulator_provenance),
        "scribble_generation": {
            **dict(generation),
            "dataset_id": DATASET_ID,
            "source_D2_record_id": source_id,
            "state_policy": "D2_CURRENT_STATE_FROZEN",
            "target_policy": "D2_AUTHORIZED_TARGET_FROZEN",
            "center_z_policy": "D2_CENTER_Z_FROZEN",
            "selected_strategy": strategy,
            "requested_strategy": record["requested_strategy"],
            "effective_strategy": record["effective_strategy"],
            "strategy_fallback": record["strategy_fallback"],
            "fallback_reason": record["fallback_reason"],
            "strategy_audit": record["strategy_audit"],
            "cue_contract_version": record["contract_version"],
            "simulator_entrypoint": record["simulator_entrypoint"],
            "scribble_source": "FROZEN_D2_AUTHORIZED_TARGET_CENTER_SLICE",
            "scribble_source_mask_sha256": record["residual_sha256"],
            "scribble_source_voxels": record["residual_voxels"],
            "coordinate_count": record["coordinate_count"],
            "coordinate_sha256": record["coordinate_sha256"],
            "official_source_provenance": dict(simulator_provenance),
        },
        "residual_kind": "FN" if validated["operation"] == "ADD" else "FP",
        "residual_path": str(
            validated["paths"][
                "fn_path" if validated["operation"] == "ADD" else "fp_path"
            ]
        ),
        "residual_sha256": str(
            source["fn_sha256"]
            if validated["operation"] == "ADD"
            else source["fp_sha256"]
        ),
        "residual_voxels": int(
            validated["legal_residual_voxels"]
        ),
        "residual_mask_sha256": str(validated["legal_residual_mask_sha256"]),
        "residual_contract": {
            "source": "D2_FROZEN_CURRENT_STATE",
            "scribble_substrate": "D2_FROZEN_AUTHORIZED_TARGET_CENTER_SLICE",
        },
        "d3_lineage": {
            "source_D2_record_id": source_id,
            "image_reference": {
                "ct_path": str(validated["paths"]["ct_path"]),
                "pet_path": str(validated["paths"]["pet_path"]),
            },
            "M0_reference": {
                "current_state_path": str(validated["paths"]["state_path"]),
                "current_state_sha256": str(source["state_sha256"]),
                "legacy_oof_ready_sha256": expected_oof_ready_sha256,
            },
            "scribble_reference": {
                "strategy": strategy,
                "seed": int(generation["seed"]),
                "coordinate_sha256": str(record["coordinate_sha256"]),
            },
            "target_reference": {
                "authorized_path": str(validated["paths"]["authorized_path"]),
                "authorized_sha256": str(source["authorized_sha256"]),
            },
        },
        **{
            key: str(validated["paths"][key])
            for key in (
                "ct_path",
                "pet_path",
                "state_path",
                "m0_path",
                "gt_path",
                "fn_path",
                "fp_path",
                "authorized_path",
            )
        },
        **{
            key: source[key]
            for key in (
                "ct_sha256",
                "pet_sha256",
                "state_sha256",
                "m0_sha256",
                "gt_sha256",
                "fn_sha256",
                "fp_sha256",
                "authorized_sha256",
            )
        },
    }


def _legacy_provenance(
    source: Mapping[str, Any], *, expected_oof_ready_sha256: str
) -> None:
    provenance = source.get("m0_provenance")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("D2 source lacks legacy M0 provenance")
    if (
        provenance.get("contract_version") != "PETCT-M0-OOF-v1.0"
        or provenance.get("kind") != "patient_excluded_oof"
        or provenance.get("oof_ready_sha256") != expected_oof_ready_sha256
    ):
        raise RuntimeError("D2 source is not bound to the audited legacy M0 OOF receipt")


def _source_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("partition")),
        str(row.get("patient_id")).casefold(),
        str(row.get("case_id")),
        int(row.get("round", -1)),
        str(row.get("episode_id")),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d2-manifest", type=Path, required=True)
    parser.add_argument("--expected-d2-manifest-sha256", required=True)
    parser.add_argument("--legacy-oof-ready-sha256", required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--official-simulator", type=Path, required=True)
    parser.add_argument(
        "--official-runtime-manifest", type=Path, default=DEFAULT_RUNTIME_MANIFEST
    )
    parser.add_argument("--official-commit")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=("train", "val"),
        required=True,
    )
    parser.add_argument("--visible-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--ready-receipt", type=Path, required=True)
    parser.add_argument(
        "--near-duplicate-jaccard",
        type=float,
        default=NEAR_DUPLICATE_JACCARD_DEFAULT,
    )
    args = parser.parse_args(argv)

    if len(args.partitions) != len(set(args.partitions)):
        parser.error("--partitions must not contain duplicates")
    expected_d2_sha256 = _hex_digest(
        args.expected_d2_manifest_sha256, label="expected D2 manifest sha256"
    )
    expected_oof_sha256 = _hex_digest(
        args.legacy_oof_ready_sha256, label="legacy OOF ready sha256"
    )
    if sha256_file(args.d2_manifest) != expected_d2_sha256:
        parser.error("D2 manifest differs from the predeclared audited SHA-256")
    selected_partitions = set(args.partitions)
    if selected_partitions != {"train", "val"}:
        parser.error("legacy D3 must be built from the complete train+val D2 feasibility set")
    if not 0 < args.near_duplicate_jaccard < 1:
        parser.error("--near-duplicate-jaccard must be between 0 and 1")

    with args.experiment_config.open("r", encoding="utf-8") as stream:
        experiment_config = json.load(stream)
    generation = resolve_scribble_generation_contract(
        experiment_config,
        official_commit=args.official_commit,
        strategy_mode="all",
        strategy_salt=None,
        seed=args.seed,
    )
    experiment_config_sha256 = sha256_file(args.experiment_config)
    source_rows = sorted(load_jsonl(args.d2_manifest), key=_source_sort_key)
    try:
        split_validation = validate_manifest_rows_against_frozen_learning_split(
            source_rows,
            args.learning_split,
            require_episode_id=True,
            allowed_partitions=selected_partitions,
        )
    except LearningContractError as exc:
        parser.error(str(exc))
    if any(row.get("experiment_config_sha256") != experiment_config_sha256 for row in source_rows):
        parser.error("D2 source rows differ from the selected experiment config")
    if any(row.get("partition") == "test" for row in source_rows):
        parser.error("future locked test data is outside the legacy D3 task")
    source_ids = [str(row.get("episode_id") or "") for row in source_rows]
    if len(source_ids) != len(set(source_ids)):
        parser.error("D2 source record ids are not unique")

    simulator = load_official_simulator(
        args.official_simulator,
        expected_commit=generation["official_commit"],
        expected_sha256=generation["simulator_file_sha256"],
        runtime_manifest=args.official_runtime_manifest,
    )
    simulator_provenance = getattr(simulator, "_petct_official_provenance", None)
    validate_official_simulator_provenance(simulator_provenance, generation)

    visible_root = args.visible_root.resolve()
    evaluation_root = args.evaluation_root.resolve()
    output_manifest = args.output_manifest.resolve()
    exclusions_path = args.exclusions.resolve()
    ready_receipt = args.ready_receipt.resolve()
    final_directories = [visible_root, evaluation_root]
    final_files = [exclusions_path, output_manifest, ready_receipt]
    hash_cache: dict[Path, str] = {}
    generated_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    triplet_audits: list[dict[str, Any]] = []

    with staged_output_bundle(
        directory_outputs=final_directories, file_outputs=final_files
    ) as staged:
        for source in source_rows:
            source_id = str(source["episode_id"])
            _legacy_provenance(
                source, expected_oof_ready_sha256=expected_oof_sha256
            )
            validated = _validate_source_state(source, hash_cache=hash_cache)
            cue_substrate = fixed_authorized_slice(
                validated["authorized"], center_z=int(validated["center_z"])
            )
            variants: list[dict[str, Any]] = []
            for strategy in STRATEGY_ORDER:
                try:
                    scribble = generate_residual_scribble(
                        cue_substrate,
                        operation=str(validated["operation"]),
                        strategy=strategy,
                        simulator=simulator,
                        upstream_commit=str(generation["official_commit"]),
                        seed=int(generation["seed"]),
                        minimum_best_slice_pixels=int(
                            generation["minimum_best_slice_pixels"]
                        ),
                    )
                except ResidualCueIneligibleError as exc:
                    exclusions.append(
                        {
                            "source_D2_record_id": source_id,
                            "case_id": source["case_id"],
                            "patient_id": source["patient_id"],
                            "partition": source["partition"],
                            "round": source["round"],
                            "scribble_strategy": strategy,
                            "attempt_id": _attempt_id(source_id, strategy),
                            "generation_status": "EXCLUDED",
                            "reason": "FROZEN_AUTHORIZED_SLICE_CUE_INELIGIBLE",
                            "reason_detail": str(exc),
                        }
                    )
                    continue
                except EpisodeContractError as exc:
                    exclusions.append(
                        {
                            "source_D2_record_id": source_id,
                            "case_id": source["case_id"],
                            "patient_id": source["patient_id"],
                            "partition": source["partition"],
                            "round": source["round"],
                            "scribble_strategy": strategy,
                            "attempt_id": _attempt_id(source_id, strategy),
                            "generation_status": "FAILED",
                            "reason": "OFFICIAL_CUE_GENERATION_FAILED",
                            "reason_detail": str(exc),
                        }
                    )
                    continue
                disposition = apply_strategy_identity_policy(
                    scribble,
                    strategy_mode="all",
                    context=f"D3 matched attempt {_attempt_id(source_id, strategy)}",
                )
                if disposition is not None:
                    exclusions.append(
                        {
                            "source_D2_record_id": source_id,
                            "case_id": source["case_id"],
                            "patient_id": source["patient_id"],
                            "partition": source["partition"],
                            "round": source["round"],
                            "scribble_strategy": strategy,
                            "attempt_id": _attempt_id(source_id, strategy),
                            "generation_status": "EXCLUDED",
                            "reason": disposition["reason"],
                            "reason_detail": disposition["detail"],
                            "effective_strategy": disposition["effective_strategy"],
                        }
                    )
                    continue
                coordinates = _coordinate_set(scribble)
                if {coord[2] for coord in coordinates} != {int(validated["center_z"])}:
                    raise RuntimeError("generated D3 scribble changed the frozen center_z")
                if any(not validated["authorized"][coord] for coord in coordinates):
                    raise RuntimeError("generated D3 scribble escaped the frozen authorised target")
                variants.append(
                    _base_variant(
                        source=source,
                        validated=validated,
                        strategy=strategy,
                        scribble=scribble,
                        generation=generation,
                        simulator_provenance=simulator_provenance,
                        expected_oof_ready_sha256=expected_oof_sha256,
                        source_d2_manifest_sha256=expected_d2_sha256,
                    )
                )
            similarity = annotate_geometry_similarity(
                variants, near_duplicate_jaccard=args.near_duplicate_jaccard
            )
            attach_geometry_audit(variants, similarity)
            triplet_audits.append(
                {
                    "source_D2_record_id": source_id,
                    "generated_strategy_count": len(variants),
                    "generated_strategies": [row["strategy"] for row in variants],
                    **similarity,
                }
            )
            for row in variants:
                episode_id = str(row["episode_id"])
                visible, evaluation = build_episode_documents(
                    episode_id=episode_id,
                    lane="natural",
                    patient_group_hash=hashlib.sha256(
                        (
                            "PETCT-PATIENT-GROUP-v2|" + str(row["patient_id"])
                        ).encode("utf-8")
                    ).hexdigest(),
                    montage_reference=f"legacy-m0-d3-visible/{episode_id}.npz",
                    m0_provenance=dict(row["m0_provenance"]),
                    scribble_record={
                        **dict(row["scribble_generation"]),
                        "polarity": row["cue_polarity"],
                        "strategy": row["strategy"],
                        "requested_strategy": row["requested_strategy"],
                        "effective_strategy": row["effective_strategy"],
                        "strategy_fallback": row["strategy_fallback"],
                        "fallback_reason": None,
                        "strategy_audit": "OFFICIAL_PRIMITIVE_CALL_AUDITED",
                        "seed": row["scribble_seed"],
                        "coordinates_xyz": row["coordinates_xyz"],
                        "coordinate_count": row["coordinate_count"],
                        "coordinate_sha256": row["coordinate_sha256"],
                        "scribble_density_mode": row["scribble_density_mode"],
                        "fallback_mode": row["fallback_mode"],
                        "validity_flags": row["validity_flags"],
                    },
                    source_case_id=str(row["case_id"]),
                    source_patient_id=str(row["patient_id"]),
                    residual_sha256=str(
                        row["scribble_generation"]["scribble_source_mask_sha256"]
                    ),
                    residual_voxels=int(
                        row["scribble_generation"]["scribble_source_voxels"]
                    ),
                    gold_intent=canonical_intent_frame(str(row["goal"])),
                )
                receipt = publish_episode_documents(
                    visible,
                    evaluation,
                    visible_root=staged[visible_root],
                    eval_root=staged[evaluation_root],
                )
                row["visible_document"] = str(visible_root / f"{episode_id}.json")
                row["visible_document_sha256"] = receipt["visible_sha256"]
                row["evaluation_document"] = str(
                    evaluation_root / f"{episode_id}.json"
                )
                row["evaluation_document_sha256"] = receipt["eval_sha256"]
                generated_rows.append(row)

        generated_attempts = {str(row["attempt_id"]) for row in generated_rows}
        excluded_attempts = {str(row["attempt_id"]) for row in exclusions}
        requested_attempts = {
            _attempt_id(str(source["episode_id"]), strategy)
            for source in source_rows
            for strategy in STRATEGY_ORDER
        }
        if generated_attempts & excluded_attempts:
            raise RuntimeError("a D3 attempt is both generated and excluded")
        if len(excluded_attempts) != len(exclusions):
            raise RuntimeError("a D3 attempt has multiple exclusion records")
        if generated_attempts | excluded_attempts != requested_attempts:
            raise RuntimeError("D3 attempt denominator is not closed")
        validate_manifest_rows_against_frozen_learning_split(
            generated_rows,
            args.learning_split,
            require_episode_id=True,
            allowed_partitions=selected_partitions,
        )
        patient_partitions: dict[str, set[str]] = defaultdict(set)
        for row in generated_rows:
            patient_partitions[str(row["patient_id"]).casefold()].add(
                str(row["partition"])
            )
        leakage = {
            patient: sorted(partitions)
            for patient, partitions in patient_partitions.items()
            if len(partitions) > 1
        }
        if leakage:
            raise RuntimeError("D3 patient leakage detected")

        _write_jsonl(staged[exclusions_path], exclusions)
        _write_jsonl(staged[output_manifest], generated_rows)
        generated_by_strategy = Counter(row["strategy"] for row in generated_rows)
        excluded_by_strategy = Counter(row["scribble_strategy"] for row in exclusions)
        reasons = Counter(row["reason"] for row in exclusions)
        complete = sum(audit["generated_strategy_count"] == 3 for audit in triplet_audits)
        partial = sum(audit["generated_strategy_count"] in {1, 2} for audit in triplet_audits)
        zero = sum(audit["generated_strategy_count"] == 0 for audit in triplet_audits)
        exact_pair_count = sum(len(audit["exact_pairs"]) for audit in triplet_audits)
        near_pair_count = sum(len(audit["near_pairs"]) for audit in triplet_audits)
        ready = {
            "schema_version": READY_SCHEMA_VERSION,
            "status": "COMMITTED",
            "phase": "LEGACY_M0_D3_MATCHED_STRATEGY_CORPUS_GENERATION",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_id": DATASET_ID,
            "experimental_boundary": "legacy-M0 train/validation feasibility only",
            "future_locked_test_touched": False,
            "source": {
                "d2_manifest": str(args.d2_manifest.resolve()),
                "d2_manifest_sha256": expected_d2_sha256,
                "d2_state_count": len(source_rows),
                "patient_count": len(
                    {str(row["patient_id"]).casefold() for row in source_rows}
                ),
                "case_count": len({str(row["case_id"]) for row in source_rows}),
                "round_counts": dict(
                    sorted(Counter(str(row["round"]) for row in source_rows).items())
                ),
                "legacy_oof_ready_sha256": expected_oof_sha256,
                "experiment_config_sha256": experiment_config_sha256,
                "learning_split_sha256": split_validation[
                    "learning_split_sha256"
                ],
            },
            "generation": {
                "contract": generation,
                "official_source_provenance": simulator_provenance,
                "requested_attempts": len(requested_attempts),
                "generated_by_strategy": dict(generated_by_strategy),
                "excluded_by_strategy": dict(excluded_by_strategy),
                "complete_triplets": complete,
                "partial_triplets": partial,
                "zero_strategy_states": zero,
                "complete_distinct_geometry_triplets": sum(
                    audit["generated_strategy_count"] == 3
                    and audit["distinct_geometry_count"] == 3
                    for audit in triplet_audits
                ),
                "exact_duplicate_pairs_retained_and_flagged": exact_pair_count,
                "near_duplicate_pairs_retained_and_flagged": near_pair_count,
                "near_duplicate_coordinate_jaccard_threshold": args.near_duplicate_jaccard,
                "exclusion_reasons": dict(reasons),
                "final_record_count": len(generated_rows),
            },
            "split_validation": {
                "allowed_partitions": sorted(selected_partitions),
                "patient_leakage_count": 0,
                "patient_variants_and_rounds_kept_together": True,
            },
            "outputs": {
                "manifest": str(output_manifest),
                "manifest_sha256": sha256_file(staged[output_manifest]),
                "exclusions": str(exclusions_path),
                "exclusions_sha256": sha256_file(staged[exclusions_path]),
                "visible_documents": str(visible_root),
                "evaluation_documents": str(evaluation_root),
            },
        }
        _write_json(staged[ready_receipt], ready)
    print(json.dumps(ready, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
