#!/usr/bin/env python3
"""Sliding Window FastEdit (Hadlich et al., ISBI 2024) native-click adapter.

Runs the upstream AutoPET2-Submission interactive DynUNet under its NATIVE
paired-click protocol.  This is a separately labelled native-click table; it
is never compressed into the matched one-scribble comparison and it is not a
bidirectional baseline (full-mask output, POSITIVE_ONLY semantics follow the
frozen external-comparator contract).

Checkpoint: the interactive weights are published at
    https://bwsyncandshare.kit.edu/s/Yky4x6PQbtxLj2H
(pretrained on tumor-only AutoPET volumes).  The adapter refuses to run
without a hash-bound checkpoint; downloading is a separate, explicitly
authorized act and is not performed by this process (NO_DOWNLOADS policy).
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

CHECKPOINT_SCHEMA = "PETCT-SWFASTEDIT-CHECKPOINT-v1.0"
OUTPUT_RECORD_SCHEMA = "PETCT-EXTERNAL-COMPARATOR-OUTPUT-v1.0"
KNOWN_CHECKPOINT_URL = "https://bwsyncandshare.kit.edu/s/Yky4x6PQbtxLj2H"
PET_PERCENTILE_CLIP = (0.05, 99.95)
SLIDING_WINDOW = (128, 128, 128)
SLIDING_OVERLAP = 0.25
GUIDANCE_SIGMA = 1.0
GUIDANCE_DISK_THRESHOLD = 0.1


class SWFastEditError(RuntimeError):
    """Raised when the native-click contract is violated."""


def _regular(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise SWFastEditError(f"{label} must be a non-symlink regular file: {raw}")
    return raw.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular(path, label="file").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_guidance_signal(
    shape: Sequence[int],
    coordinates: Sequence[Sequence[int]],
    *,
    sigma: float = GUIDANCE_SIGMA,
    with_disks: bool = True,
    disk_threshold: float = GUIDANCE_DISK_THRESHOLD,
) -> np.ndarray:
    """Pinned-source guidance contract (upstream transforms.py, exact):

    impulse points -> MONAI GaussianFilter(sigma) -> min-max normalize to
    [0,1] -> with disks: threshold at >0.1 (sigma=1 gives approximately
    radius-3 disks, otherwise a cube).  Upstream skips negative points and
    clamps positive overflow; this adapter mirrors that behavior exactly.
    """

    from scipy.ndimage import gaussian_filter

    volume = np.zeros(tuple(int(value) for value in shape), dtype=np.float32)
    for raw in coordinates:
        if len(raw) != 3:
            raise SWFastEditError("guidance coordinate must be xyz")
        coord = tuple(int(value) for value in raw)
        if any(value < 0 for value in coord):
            continue  # upstream skips negative guidance points
        coord = tuple(
            min(value, volume.shape[axis] - 1) for axis, value in enumerate(coord)
        )
        volume[coord] = 1.0
    if not np.any(volume):
        return volume
    encoded = gaussian_filter(volume, sigma=sigma, mode="constant")
    low, high = float(encoded.min()), float(encoded.max())
    if high > low:
        encoded = (encoded - low) / (high - low)
    if with_disks:
        encoded = (encoded > disk_threshold).astype(np.float32)
    return encoded


def pet_percentile_clip(volume: np.ndarray) -> np.ndarray:
    """Paper training contract: PET percentile clip [0.05, 99.95]."""

    values = np.asarray(volume, dtype=np.float32)
    if not np.any(np.isfinite(values)):
        raise SWFastEditError("PET volume has no finite values")
    low, high = np.percentile(values, PET_PERCENTILE_CLIP)
    return np.clip(values, float(low), float(high))


def validate_input_manifest(path: Path) -> list[dict[str, Any]]:
    """Validate the frozen external-comparator input manifest (one record per
    case, OOF state, correction step, and frozen scribble)."""

    manifest = _regular(path, label="input manifest")
    rows = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SWFastEditError(f"input manifest line {line_number} is not JSON") from exc
        if not isinstance(row, dict) or "records" not in row:
            raise SWFastEditError(f"input manifest line {line_number} lacks records")
        for record in row["records"]:
            for field in (
                "case_id",
                "patient_id",
                "split",
                "fold",
                "step",
                "pet_path",
                "fg_scribble_path",
                "original_grid_reference",
                "scribble_strategy",
                "scribble_polarity",
            ):
                if record.get(field) is None or record.get(field) == "":
                    raise SWFastEditError(f"record {record.get('case_id')} missing {field}")
            if record["scribble_polarity"] not in ("foreground", "background"):
                raise SWFastEditError("scribble_polarity must be foreground/background")
            rows.append(record)
    if not rows:
        raise SWFastEditError("input manifest has no records")
    return rows


def load_checkpoint(checkpoint_path: Path, device: str):
    """Load the three-channel interactive DynUNet.  MONAI is imported lazily
    so contract tests stay environment-free; the checkpoint must be
    hash-bound by the caller's runtime admission receipt."""

    checkpoint = _regular(checkpoint_path, label="checkpoint")
    try:
        import torch
        from monai.networks.nets import DynUNet
    except ImportError as exc:
        raise SWFastEditError("torch + monai are required to run SW-FastEdit") from exc
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    raw = state.get("model") if isinstance(state, Mapping) else None
    if raw is None and isinstance(state, Mapping):
        raw = state.get("state_dict")
    if raw is None:
        raw = state
    model = DynUNet(
        spatial_dims=3,
        in_channels=3,  # PET + tumor guidance + background guidance
        out_channels=1,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2, 2],
        filters=[32, 64, 128, 256, 320, 320],
        dropout=None,
        norm_name=("INSTANCE", {"affine": True}),
        deep_supervision=False,
        res_block=False,
    )
    incompatible = model.load_state_dict(raw, strict=False)
    if incompatible.missing_keys:
        raise SWFastEditError(f"checkpoint is missing keys: {incompatible.missing_keys[:3]}")
    model.to(device)
    model.eval()
    return model


def run_one_record(
    model,
    record: Mapping[str, Any],
    *,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Run the native-click protocol for one input record (one correction
    step).  Output is a full mask restored to the original grid."""

    import nibabel as nib
    import torch

    pet_path = _regular(Path(str(record["pet_path"])), label="PET")
    fg_path = _regular(Path(str(record["fg_scribble_path"])), label="FG scribble")
    reference = _regular(
        Path(str(record["original_grid_reference"])), label="grid reference"
    )
    pet_image = nib.load(str(pet_path))
    reference_image = nib.load(str(reference))
    if pet_image.shape != reference_image.shape:
        raise SWFastEditError("PET and grid reference shapes differ")
    fg_image = nib.load(str(fg_path))
    if fg_image.shape != reference_image.shape:
        raise SWFastEditError("scribble and grid reference shapes differ")
    fg_mask = np.asanyarray(fg_image.dataobj) > 0
    tumor_points = np.argwhere(fg_mask).astype(np.int64)
    bg_mask = np.zeros(reference_image.shape, dtype=np.uint8)
    if record.get("bg_scribble_path"):
        bg_image = nib.load(str(_regular(Path(str(record["bg_scribble_path"])), label="BG scribble")))
        if bg_image.shape != reference_image.shape:
            raise SWFastEditError("BG scribble and grid reference shapes differ")
        bg_mask = np.asanyarray(bg_image.dataobj) > 0
    background_points = np.argwhere(bg_mask).astype(np.int64)
    if not len(tumor_points):
        raise SWFastEditError("native-click protocol requires at least one tumor point")

    pet = pet_percentile_clip(np.asanyarray(pet_image.dataobj))
    tumor_channel = encode_guidance_signal(
        pet.shape, tumor_points.tolist()
    )
    background_channel = encode_guidance_signal(
        pet.shape, background_points.tolist()
    )
    input_tensor = (
        torch.from_numpy(
            np.stack(
                [pet, tumor_channel, background_channel], axis=0
            ).astype(np.float32)[None]
        )
        .to(device)
    )
    with torch.no_grad():
        try:
            from monai.inferers import SlidingWindowInferer
        except ImportError as exc:
            raise SWFastEditError("monai inferers are required") from exc
        inferer = SlidingWindowInferer(
            roi_size=SLIDING_WINDOW,
            sw_batch_size=1,
            overlap=SLIDING_OVERLAP,
            mode="gaussian",
        )
        logits = inferer(input_tensor, model)
    probability = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
    prediction = (probability >= 0.5).astype(np.uint8)
    case_id = str(record["case_id"])
    step = int(record["step"])
    prediction_path = output_dir / f"{case_id}_step{step}_prediction.nii.gz"
    nib.save(
        nib.Nifti1Image(prediction, reference_image.affine),
        str(prediction_path),
    )
    return {
        "schema_version": OUTPUT_RECORD_SCHEMA,
        "case_id": case_id,
        "patient_id": str(record["patient_id"]),
        "method_id": "sw_fastedit",
        "prediction_path": str(prediction_path.resolve()),
        "original_grid_reference": str(reference),
        "prediction_semantics": "full_mask",
        "runtime_seconds": None,
        "peak_gpu_memory_mib": None,
        "source_checkpoint_id": "native-click-protocol-checkpoint-hash-bound",
        "status": "complete",
        "interaction_audit": {
            "interaction_rounds": int(record["step"]),
            "tumor_point_count": int(len(tumor_points)),
            "background_point_count": int(len(background_points)),
            "total_point_count": int(len(tumor_points) + len(background_points)),
            "guidance_encoding": (
                "sigma-1 gaussian, disks thresholded at >0.1 "
                "(pinned-source contract; documented paper/source difference)"
            ),
            "protocol_label": "NATIVE_CLICK_SEPARATELY_LABELLED_TABLE",
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
    records = validate_input_manifest(args.input_manifest)
    model = load_checkpoint(args.checkpoint, args.device)
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
    print(json.dumps({"status": "PASS", "records": len(outputs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
