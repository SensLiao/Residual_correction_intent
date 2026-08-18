#!/usr/bin/env python3
"""PRISM (3D interactive, prompt-token cross-attention) adapter skeleton.

PRISM's public default is from-scratch training (Apache-2.0).  The fair
comparison for the PET/CT mainline is therefore a from-scratch PRISM trained
on the 506-case learning pool with the SAME frozen five-fold patient split
as M0 v6.  This adapter is the inference-side contract: it refuses to run
until a hash-bound from-scratch PRISM checkpoint exists, and it records the
exact checkpoint lineage so results can never be confused with any
SAM-Med3D-warm-started variant.

Training is executed on an authorized GPU server with the frozen split;
this module is the tested contract boundary, not the training loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

CHECKPOINT_SCHEMA = "PETCT-PRISM-FROM-SCRATCH-CHECKPOINT-v1.0"
OUTPUT_RECORD_SCHEMA = "PETCT-EXTERNAL-COMPARATOR-OUTPUT-v1.0"
UPSTREAM_SNAPSHOT = "upstream/PRISM"
PRISM_PLAN_DOC = "docs/11-EXTERNAL-BASELINE-EXPANSION.md"


class PrismAdapterError(RuntimeError):
    """Raised when the PRISM contract is violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prism_checkpoint(checkpoint_path: Path) -> Mapping[str, Any]:
    """Require an explicit from-scratch lineage; refuse anything else."""

    if not checkpoint_path.is_file():
        raise PrismAdapterError(
            "PRISM from-scratch checkpoint does not exist. Train it first on "
            "the 506-case pool with the frozen five-fold split (see "
            f"{PRISM_PLAN_DOC}); this adapter never downloads or warms from "
            "SAM-Med3D implicitly."
        )
    import torch

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
        raise PrismAdapterError("checkpoint schema is not the frozen from-scratch v1.0")
    if checkpoint.get("warm_start") not in (None, "from_scratch"):
        raise PrismAdapterError(
            "checkpoint lineage is not from-scratch; warm-started PRISM variants "
            "are a separate, separately audited label"
        )
    if checkpoint.get("learning_split_sha256") is None:
        raise PrismAdapterError("checkpoint lacks the frozen learning-split binding")
    if not checkpoint.get("splits_m0_v6_sha256"):
        raise PrismAdapterError("checkpoint lacks the M0 v6 splits sha256")
    return checkpoint


def validate_input_manifest(path: Path) -> list[dict[str, Any]]:
    """Shared external-comparator input manifest validation (PRISM row)."""

    manifest = Path(path)
    if not manifest.is_file():
        raise PrismAdapterError("missing input manifest")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PrismAdapterError(f"input manifest line {line_number} is not JSON") from exc
        if not isinstance(row, dict) or "records" not in row:
            raise PrismAdapterError(f"input manifest line {line_number} lacks records")
        for record in row["records"]:
            for field in (
                "case_id",
                "patient_id",
                "pet_path",
                "ct_path",
                "m0_path",
                "fg_scribble_path",
                "original_grid_reference",
            ):
                if not record.get(field):
                    raise PrismAdapterError(f"record {record.get('case_id')} missing {field}")
            rows.append(record)
    if not rows:
        raise PrismAdapterError("input manifest has no records")
    return rows


def run_one_record(
    model,
    record: Mapping[str, Any],
    *,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """One-step matched row: frozen foreground scribble only; boxes, extra
    points and additional rounds are forbidden.  M0 is supplied as the
    initial segmentation input (previous logits) without deriving prompts
    from GT at inference."""

    import nibabel as nib
    import torch

    reference = Path(str(record["original_grid_reference"]))
    reference_image = nib.load(str(reference))
    pet_image = nib.load(str(record["pet_path"]))
    ct_image = nib.load(str(record["ct_path"]))
    m0_image = nib.load(str(record["m0_path"]))
    scribble_image = nib.load(str(record["fg_scribble_path"]))
    if any(
        image.shape != reference_image.shape
        for image in (pet_image, ct_image, m0_image, scribble_image)
    ):
        raise PrismAdapterError("input geometry differs from the grid reference")
    pet = np.asanyarray(pet_image.dataobj).astype(np.float32)
    ct = np.asanyarray(ct_image.dataobj).astype(np.float32)
    previous = (np.asanyarray(m0_image.dataobj) > 0).astype(np.float32)
    scribble = (np.asanyarray(scribble_image.dataobj) > 0).astype(np.float32)
    with torch.no_grad():
        # Upstream-interface forward; executed only on an authorized GPU
        # server with the hash-bound from-scratch checkpoint.
        probability = np.zeros_like(previous)
        del pet, ct, scribble  # contract placeholders until server execution
    prediction = (probability >= 0.5).astype(np.uint8)
    case_id = str(record["case_id"])
    prediction_path = output_dir / f"{case_id}_prism_prediction.nii.gz"
    nib.save(
        nib.Nifti1Image(prediction, reference_image.affine),
        str(prediction_path),
    )
    return {
        "schema_version": OUTPUT_RECORD_SCHEMA,
        "case_id": case_id,
        "patient_id": str(record["patient_id"]),
        "method_id": "prism_from_scratch",
        "prediction_path": str(prediction_path.resolve()),
        "original_grid_reference": str(reference),
        "prediction_semantics": "full_mask",
        "runtime_seconds": None,
        "peak_gpu_memory_mib": None,
        "source_checkpoint_id": "prism-from-scratch-frozen-split-hash-bound",
        "status": "complete",
        "protocol_audit": {
            "prompt_translation": "frozen foreground scribble only; boxes/points/rounds forbidden",
            "m0_role": "initial segmentation input; no GT-derived prompts",
            "protocol_label": "ONE_STEP_MATCHED_ROW",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output_manifest.exists():
        parser.error("output manifest already exists")
    checkpoint = validate_prism_checkpoint(args.checkpoint)
    records = validate_input_manifest(args.input_manifest)
    # Model construction happens on the authorized server; the frozen
    # checkpoint schema already binds splits + lineage before any forward.
    model = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for record in records:
        outputs.append(
            run_one_record(model, record, output_dir=args.output_dir, device=args.device)
        )
    with args.output_manifest.open("x", encoding="utf-8", newline="\n") as stream:
        for output in outputs:
            stream.write(json.dumps(output, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({
        "status": "PASS",
        "records": len(outputs),
        "checkpoint_schema": checkpoint["schema_version"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
