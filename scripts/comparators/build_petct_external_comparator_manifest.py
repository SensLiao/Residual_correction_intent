#!/usr/bin/env python3
"""Build the truth-firewalled input manifest for external PET/CT comparators.

The construction plane reads the already frozen natural OOF episode and tensor
receipts, but the published model-input manifest contains only PET, CT, M0 and
the exact AutoPET-V foreground scribble raster.  GT, residuals, authorized
targets and intent labels are deliberately not copied into the model plane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import nibabel as nib
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1]
for directory in (
    SCRIPTS,
    SCRIPTS / "common",
    SCRIPTS / "data",
    SCRIPTS / "baseline",
    SCRIPTS / "orchestration",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from baseline.validate_petct_m0_oof import (  # noqa: E402
    build_natural_oof_binding_from_validated,
    validate_oof_ready_receipt_only,
)
from common.petct_learning import (  # noqa: E402
    load_jsonl,
    sha256_file,
    validate_manifest_rows_against_frozen_learning_split,
)
from common.petct_test_access import (  # noqa: E402
    TestAccessError,
    add_leaf_test_access_arguments,
    enforce_partition_access,
)
from data.validate_petct_learning_split import (  # noqa: E402
    load_and_validate_learning_split,
)
from orchestration.validate_petct_route_a_receipt_pipeline import (  # noqa: E402
    validate_natural_episode_rows,
    validate_tensor_rows,
)


SCHEMA_VERSION = "PETCT-EXTERNAL-COMPARATOR-INPUT-v1.0"
RECEIPT_VERSION = "PETCT-EXTERNAL-COMPARATOR-INPUT-RECEIPT-v1.0"
PARTITION_TO_PUBLIC = {"val": "validation", "test": "test"}
STRATEGIES = {"centerline", "random", "boundary"}
FORBIDDEN_RECORD_FIELDS = {
    "gt_path",
    "label_path",
    "target_path",
    "authorized_path",
    "fn_path",
    "fp_path",
    "fn_residual_path",
    "gold_intent",
    "intent_target",
    "target",
    "scope",
    "goal",
}


class ManifestError(RuntimeError):
    """Raised when a comparator input would violate the frozen data contract."""


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _regular(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ManifestError(f"missing regular {label}: {resolved}")
    return resolved


def _verified_path(row: Mapping[str, Any], path_key: str, hash_key: str) -> Path:
    raw = row.get(path_key)
    expected = row.get(hash_key)
    if not isinstance(raw, str) or not raw or not isinstance(expected, str):
        raise ManifestError(f"episode requires {path_key} and {hash_key}")
    path = _regular(Path(raw), label=path_key)
    if sha256_file(path) != expected:
        raise ManifestError(f"episode source hash mismatch: {path_key}")
    return path


def _load_3d(path: Path, *, label: str) -> nib.spatialimages.SpatialImage:
    try:
        image = nib.load(str(path))
    except Exception as exc:
        raise ManifestError(f"cannot read {label} NIfTI {path}: {exc}") from exc
    if len(image.shape) != 3:
        raise ManifestError(f"{label} must be 3D, got {image.shape}")
    return image


def _same_grid(
    reference: nib.spatialimages.SpatialImage,
    candidate: nib.spatialimages.SpatialImage,
    *,
    label: str,
) -> None:
    if reference.shape != candidate.shape:
        raise ManifestError(
            f"{label} shape differs from frozen original grid: "
            f"{candidate.shape} != {reference.shape}"
        )
    if not np.allclose(reference.affine, candidate.affine, rtol=0.0, atol=1e-4):
        raise ManifestError(f"{label} affine differs from frozen original grid")


def _coordinates(
    raw: Any, shape: Sequence[int]
) -> tuple[list[list[int]], np.ndarray]:
    if not isinstance(raw, list) or not raw:
        raise ManifestError("natural episode has no frozen scribble coordinates")
    normalized: list[list[int]] = []
    seen: set[tuple[int, int, int]] = set()
    mask = np.zeros(tuple(int(value) for value in shape), dtype=np.uint8)
    for item in raw:
        if not isinstance(item, list) or len(item) != 3:
            raise ManifestError("scribble coordinates must be xyz integer triples")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in item):
            raise ManifestError("scribble coordinates must be exact integer indices")
        coordinate = tuple(int(value) for value in item)
        if any(value < 0 or value >= mask.shape[axis] for axis, value in enumerate(coordinate)):
            raise ManifestError(f"scribble coordinate is out of bounds: {coordinate}")
        if coordinate in seen:
            raise ManifestError(f"scribble coordinate is duplicated: {coordinate}")
        seen.add(coordinate)
        normalized.append(list(coordinate))
        mask[coordinate] = 1
    if len({coordinate[2] for coordinate in seen}) != 1:
        raise ManifestError("frozen POC scribble must occupy exactly one axial slice")
    return sorted(normalized), mask


def _save_binary_mask(
    mask: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    target: Path,
) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, header=header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    if qform is not None:
        image.set_qform(qform, int(qcode))
    if sform is not None:
        image.set_sform(sform, int(scode))
    nib.save(image, str(target))


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    if not safe or safe in {".", ".."}:
        raise ManifestError(f"unsafe case identifier: {value!r}")
    return safe


def _assert_record_firewall(record: Mapping[str, Any]) -> None:
    leaked = FORBIDDEN_RECORD_FIELDS & set(record)
    if leaked:
        raise ManifestError(f"truth-plane fields leaked into comparator input: {sorted(leaked)}")


def build_comparator_input(
    *,
    natural_rows: Sequence[Mapping[str, Any]],
    tensor_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    case_to_partition: Mapping[str, str],
    selected_partition: str,
    output_manifest: Path,
    scribble_dir: Path,
    provenance: Mapping[str, Any],
    expected_strategy_mode: str = "primary",
) -> dict[str, Any]:
    """Materialize one safe, original-grid scribble per selected natural episode."""

    if selected_partition not in PARTITION_TO_PUBLIC:
        raise ManifestError("external comparator partition must be val or test")
    output_manifest = output_manifest.resolve()
    scribble_dir = scribble_dir.resolve()
    if output_manifest.exists() or output_manifest.is_symlink():
        raise FileExistsError(f"refusing to overwrite output manifest: {output_manifest}")
    if scribble_dir.exists() or scribble_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite scribble directory: {scribble_dir}")
    if (
        output_manifest == scribble_dir
        or output_manifest in scribble_dir.parents
        or scribble_dir in output_manifest.parents
    ):
        raise ManifestError("manifest and scribble directory must be physically disjoint")

    sources = {str(row.get("case_id") or ""): row for row in source_rows}
    if "" in sources or len(sources) != len(source_rows):
        raise ManifestError("source case manifest has empty or duplicate case_id")
    tensors = {str(row.get("episode_id") or ""): row for row in tensor_rows}
    if "" in tensors or len(tensors) != len(tensor_rows):
        raise ManifestError("natural tensor manifest has empty or duplicate episode_id")
    selected = [row for row in natural_rows if row.get("partition") == selected_partition]
    if not selected:
        raise ManifestError(f"natural manifest has no {selected_partition} episodes")

    seen_cases: set[str] = set()
    stage_parent = scribble_dir.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{scribble_dir.name}.", dir=str(stage_parent)))
    manifest_stage: Path | None = None
    moved = False
    records: list[dict[str, Any]] = []
    try:
        for row in sorted(selected, key=lambda value: str(value.get("case_id"))):
            case_id = str(row.get("case_id") or "")
            patient_id = str(row.get("patient_id") or "").casefold()
            episode_id = str(row.get("episode_id") or "")
            if not case_id or not patient_id or not episode_id:
                raise ManifestError("natural episode requires case, patient and episode identifiers")
            if case_id in seen_cases:
                raise ManifestError(
                    "external one-step input requires exactly one frozen primary scribble per case"
                )
            seen_cases.add(case_id)
            if case_to_partition.get(case_id) != selected_partition:
                raise ManifestError("episode partition differs from the frozen patient split")
            source = sources.get(case_id)
            tensor = tensors.get(episode_id)
            if source is None or tensor is None:
                raise ManifestError("episode is missing its source-case or tensor receipt")
            if str(source.get("patient_id") or "").casefold() != patient_id:
                raise ManifestError("episode patient differs from source case manifest")
            if str(tensor.get("patient_id") or "").casefold() != patient_id:
                raise ManifestError("episode patient differs from natural tensor receipt")
            if tensor.get("partition") != selected_partition:
                raise ManifestError("natural tensor partition differs from episode partition")
            if row.get("strategy_mode") != expected_strategy_mode:
                raise ManifestError("external comparator requires the frozen primary strategy lane")
            strategy = str(row.get("strategy") or "")
            if strategy not in STRATEGIES or tensor.get("strategy") != strategy:
                raise ManifestError("episode/tensor scribble strategy is invalid or changed")

            for key in ("ct_path", "pet_path"):
                if row.get(key) != source.get(key):
                    raise ManifestError(f"episode {key} differs from source case manifest")
            ct_path = _verified_path(row, "ct_path", "ct_sha256")
            pet_path = _verified_path(row, "pet_path", "pet_sha256")
            m0_path = _verified_path(row, "m0_path", "m0_sha256")
            reference = _load_3d(ct_path, label="CT/original grid")
            pet_image = _load_3d(pet_path, label="PET")
            m0_image = _load_3d(m0_path, label="M0")
            _same_grid(reference, pet_image, label="PET")
            _same_grid(reference, m0_image, label="M0")

            coordinates, scribble = _coordinates(row.get("coordinates_xyz"), reference.shape)
            generation = row.get("scribble_generation")
            if not isinstance(generation, Mapping):
                raise ManifestError("natural episode lacks scribble generation receipt")
            coordinate_sha = canonical_json_sha256(coordinates)
            if generation.get("coordinate_sha256") != coordinate_sha:
                raise ManifestError("frozen scribble coordinate hash changed")
            source_eval = tensor.get("source_evaluation")
            if not isinstance(source_eval, Mapping):
                raise ManifestError("natural tensor lacks source_evaluation receipt")
            if source_eval.get("scribble_coordinates_xyz") != coordinates:
                raise ManifestError("natural tensor and external input do not share one scribble")
            if source_eval.get("m0_path") != str(m0_path) or source_eval.get("m0_sha256") != row.get("m0_sha256"):
                raise ManifestError("natural tensor M0 differs from external comparator M0")

            visible_path = _regular(Path(str(tensor.get("visible_npz") or "")), label="natural visible tensor")
            if sha256_file(visible_path) != tensor.get("visible_sha256"):
                raise ManifestError("natural visible tensor hash changed")
            filename = f"{_safe_name(case_id)}__{_safe_name(episode_id)}__fg.nii.gz"
            staged_path = staging / filename
            final_path = scribble_dir / filename
            _save_binary_mask(scribble, reference, staged_path)
            scribble_sha = sha256_file(staged_path)

            fold = row.get("held_out_fold")
            if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
                raise ManifestError("episode held_out_fold must be in [0,4]")
            record = {
                "case_id": case_id,
                "patient_id": patient_id,
                "split": PARTITION_TO_PUBLIC[selected_partition],
                "fold": int(fold),
                "step": 1,
                "pet_path": str(pet_path),
                "ct_path": str(ct_path),
                "m0_path": str(m0_path),
                "fg_scribble_path": str(final_path),
                "bg_scribble_path": None,
                "original_grid_reference": str(ct_path),
                "scribble_strategy": strategy,
                "scribble_polarity": "foreground",
                "episode_id": episode_id,
                "prompt_budget": {
                    "rounds": 1,
                    "foreground_scribbles": 1,
                    "background_scribbles": 0,
                    "clicks": 0,
                    "boxes": 0,
                },
                "same_frozen_scribble_receipt": {
                    "coordinates_sha256": coordinate_sha,
                    "raster_sha256": scribble_sha,
                    "coordinate_count": int(scribble.sum()),
                    "natural_visible_npz_sha256": str(tensor["visible_sha256"]),
                    "natural_episode_manifest_sha256": str(
                        provenance["natural_episode_manifest_sha256"]
                    ),
                    "natural_tensor_manifest_sha256": str(
                        provenance["natural_tensor_manifest_sha256"]
                    ),
                },
                "input_sha256": {
                    "pet": str(row["pet_sha256"]),
                    "ct": str(row["ct_sha256"]),
                    "m0": str(row["m0_sha256"]),
                    "fg_scribble": scribble_sha,
                },
                "patient_split_receipt": {
                    "internal_partition": selected_partition,
                    "learning_split_sha256": str(provenance["learning_split_sha256"]),
                },
                "oof_state_receipt": {
                    "kind": "patient_excluded_oof",
                    "oof_ready_sha256": str(provenance["oof_ready_sha256"]),
                    "held_out_fold": int(fold),
                    "m0_sha256": str(row["m0_sha256"]),
                },
            }
            _assert_record_firewall(record)
            records.append(record)

        patient_partitions: dict[str, set[str]] = {}
        for record in records:
            patient_partitions.setdefault(record["patient_id"], set()).add(record["split"])
        if any(len(partitions) != 1 for partitions in patient_partitions.values()):
            raise ManifestError("a patient crosses comparator input partitions")

        document = {
            "schema_version": SCHEMA_VERSION,
            "receipt_schema_version": RECEIPT_VERSION,
            "status": "FROZEN_INPUT_READY",
            "partition": PARTITION_TO_PUBLIC[selected_partition],
            "record_count": len(records),
            "patient_count": len(patient_partitions),
            "step_budget": 1,
            "scribble_policy": "same frozen AutoPET-V foreground scribble raster for every method",
            "model_input_firewall": {
                "allowed": ["PET", "CT", "patient-excluded OOF M0", "foreground scribble"],
                "forbidden": sorted(FORBIDDEN_RECORD_FIELDS),
                "truth_plane_separate": True,
            },
            "provenance": dict(provenance),
            "records": records,
        }
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_stage = tempfile.mkstemp(
            prefix=f".{output_manifest.name}.", suffix=".tmp", dir=str(output_manifest.parent)
        )
        manifest_stage = Path(raw_stage)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(staging, scribble_dir)
        moved = True
        os.link(str(manifest_stage), str(output_manifest))
        manifest_stage.unlink()
        return document
    except Exception:
        if manifest_stage is not None and manifest_stage.exists():
            manifest_stage.unlink()
        if moved and scribble_dir.exists():
            shutil.rmtree(scribble_dir)
        elif staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-ready", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--learning-split", type=Path, required=True)
    parser.add_argument("--natural-episode-manifest", type=Path, required=True)
    parser.add_argument("--natural-tensor-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--partition", choices=("val", "test"), required=True)
    parser.add_argument("--scribble-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    add_leaf_test_access_arguments(parser)
    args = parser.parse_args(argv)

    # This gate runs before resolving or opening any case, episode, tensor, or
    # image input.  Test outputs must remain under the one receipt-bound Route-A
    # run root; validation explicitly rejects a supplied test receipt.
    try:
        access_receipt = enforce_partition_access(
            args.partition,
            receipt_path=args.test_access_receipt,
            experiment_config=args.experiment_config,
            learning_split=args.learning_split,
            run_root=args.run_root,
            output_paths=(args.output_manifest, args.scribble_dir),
        )
    except TestAccessError as exc:
        parser.error(str(exc))

    inputs = {
        "oof_ready": _regular(args.oof_ready, label="OOF_READY"),
        "case_manifest": _regular(args.case_manifest, label="case manifest"),
        "learning_split": _regular(args.learning_split, label="learning split"),
        "natural_episode_manifest": _regular(
            args.natural_episode_manifest, label="natural episode manifest"
        ),
        "natural_tensor_manifest": _regular(
            args.natural_tensor_manifest, label="natural tensor manifest"
        ),
        "experiment_config": _regular(args.experiment_config, label="experiment config"),
    }
    with inputs["experiment_config"].open("r", encoding="utf-8") as stream:
        experiment_config = json.load(stream)
    source_rows = load_jsonl(inputs["case_manifest"])
    _, split = load_and_validate_learning_split(
        inputs["learning_split"], source_rows, experiment_config
    )
    oof = validate_oof_ready_receipt_only(inputs["oof_ready"])
    natural_rows = load_jsonl(inputs["natural_episode_manifest"])
    selected_natural_rows = [
        row for row in natural_rows if row.get("partition") == args.partition
    ]
    natural = validate_natural_episode_rows(
        selected_natural_rows,
        config_sha256=sha256_file(inputs["experiment_config"]),
        split_sha256=split["split_sha256"],
        oof_ready_sha256=oof["ready_sha256"],
        case_to_partition=split["case_to_partition"],
    )
    validate_manifest_rows_against_frozen_learning_split(
        selected_natural_rows,
        inputs["learning_split"],
        require_episode_id=True,
        allowed_partitions={args.partition},
    )
    selected_episode_ids = set(natural["episode_ids"])
    tensor_rows = [
        row
        for row in load_jsonl(inputs["natural_tensor_manifest"])
        if row.get("episode_id") in selected_episode_ids
    ]
    validate_tensor_rows(
        tensor_rows,
        expected_episode_ids=set(natural["episode_ids"]),
        config_sha256=sha256_file(inputs["experiment_config"]),
        split_sha256=split["split_sha256"],
    )
    for row in selected_natural_rows:
        expected = build_natural_oof_binding_from_validated(
            oof,
            ready_path=inputs["oof_ready"],
            case_id=str(row["case_id"]),
            patient_id=str(row["patient_id"]),
            m0_path=Path(str(row["m0_path"])),
        )
        if row.get("m0_provenance") != expected:
            raise ManifestError("natural episode OOF binding changed before comparator input")

    provenance = {
        "experiment_config_sha256": sha256_file(inputs["experiment_config"]),
        "case_manifest_sha256": sha256_file(inputs["case_manifest"]),
        "learning_split_sha256": split["split_sha256"],
        "oof_ready_sha256": oof["ready_sha256"],
        "natural_episode_manifest_sha256": sha256_file(
            inputs["natural_episode_manifest"]
        ),
        "natural_tensor_manifest_sha256": sha256_file(
            inputs["natural_tensor_manifest"]
        ),
        "test_access_receipt_sha256": (
            sha256_file(args.test_access_receipt)
            if access_receipt is not None
            else None
        ),
    }
    document = build_comparator_input(
        natural_rows=selected_natural_rows,
        tensor_rows=tensor_rows,
        source_rows=source_rows,
        case_to_partition=split["case_to_partition"],
        selected_partition=args.partition,
        output_manifest=args.output_manifest,
        scribble_dir=args.scribble_dir,
        provenance=provenance,
        expected_strategy_mode=str(experiment_config["scribble"]["primary_strategy_mode"]),
    )
    print(
        json.dumps(
            {
                "status": document["status"],
                "partition": document["partition"],
                "records": document["record_count"],
                "patients": document["patient_count"],
                "output_manifest": str(args.output_manifest.resolve()),
                "output_manifest_sha256": sha256_file(args.output_manifest.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
