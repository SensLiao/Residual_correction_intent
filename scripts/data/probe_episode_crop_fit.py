#!/usr/bin/env python3
"""Classify every episode manifest row with the frozen materializer itself.

D-2026-08-13 (Director option A, generation-side alignment): the frozen
materializer fails closed when an authorized COMPLETE/LOCAL target exceeds the
frozen crop (`materialize_petct_learning_tensors.py` raise at the
"authorized COMPLETE/LOCAL target exceeds frozen crop" assertion).  This probe
splits an episode manifest without touching the frozen contract:

- each row is passed to the frozen `materialize_episode` with per-row temporary
  staging directories (the classification IS the frozen assertion, not a copy
  of it);
- "authorized COMPLETE/LOCAL target exceeds frozen crop" -> excluded;
- any other exception -> ABORT (an unexpected defect; never silently excluded);
- success -> kept (temporary outputs deleted).

Exclusion records keep the original row plus a reporting-only offset figure.
The offset figure mirrors the frozen check formula for documentation only; the
classification itself comes from the materializer raising.

Outputs are written atomically only after the whole pass succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Sequence

import nibabel as nib
import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(DATA_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_ROOT))

from common.petct_learning import load_jsonl, sha256_file  # noqa: E402
from materialize_petct_learning_tensors import (  # noqa: E402
    _verified_source_path,
    materialize_episode,
)

CROP_ERROR_MARKER = "authorized COMPLETE/LOCAL target exceeds frozen crop"


def crop_offset_report(
    row: Dict, field_mm: float, expected_spacing: float
) -> Dict:
    """Reporting-only mirror of the frozen crop-check formula.

    Classification never uses this function; it documents by how much an
    excluded row overshot the frozen limit.  The authorized NIfTI is geometry
    aligned to the CT (asserted by the frozen materializer), so its header
    zooms equal the CT zooms the frozen check uses.
    """

    authorized_path = _verified_source_path(
        row, "authorized_path", "authorized_sha256"
    )
    authorized_image = nib.load(str(authorized_path))
    authorized = (np.asarray(authorized_image.dataobj) > 0).astype(np.uint8)
    z_values = {int(coord[2]) for coord in row["coordinates_xyz"]}
    if len(z_values) != 1:
        raise RuntimeError("report requires one axial scribble slice")
    center_z = next(iter(z_values))
    center_xy = np.mean(
        np.asarray([[coord[0], coord[1]] for coord in row["coordinates_xyz"]]),
        axis=0,
    )
    original_spacing_xy = np.asarray(
        authorized_image.header.get_zooms()[:2], dtype=np.float32
    )
    source_coordinates = np.argwhere(authorized[:, :, center_z] > 0)
    crop_limit = field_mm / 2.0 - expected_spacing / 2.0
    report = {"crop_limit_mm": float(crop_limit)}
    if not len(source_coordinates):
        report["max_authorized_offset_mm"] = None
        report["overshoot_mm"] = None
        return report
    source_offsets = np.abs(
        (source_coordinates - center_xy[None]) * original_spacing_xy[None]
    )
    max_offset = float(source_offsets.max())
    report["max_authorized_offset_mm"] = max_offset
    report["overshoot_mm"] = float(max(max_offset - crop_limit, 0.0))
    return report


def classify_row(
    row: Dict,
    *,
    staged_visible_root: Path,
    staged_evaluation_root: Path,
    field_mm: float,
    output_size: int,
    expected_spacing: float,
    config_sha256: str,
) -> str:
    """Return "keep" | "crop_excluded" | error message (any other exception)."""

    episode_id = str(row["episode_id"])
    try:
        materialize_episode(
            row,
            staged_visible_root=staged_visible_root,
            staged_evaluation_root=staged_evaluation_root,
            final_visible_root=staged_visible_root,
            final_evaluation_root=staged_evaluation_root,
            field_mm=field_mm,
            output_size=output_size,
            expected_spacing=expected_spacing,
            config_sha256=config_sha256,
        )
    except RuntimeError as exc:
        if CROP_ERROR_MARKER in str(exc):
            return "crop_excluded"
        return "other_error: %s" % exc
    except Exception as exc:  # noqa: BLE001 - probe must fail closed
        return "other_error: %s" % exc
    finally:
        for staged_root in (staged_visible_root, staged_evaluation_root):
            staged = Path(staged_root) / (episode_id + ".npz")
            if staged.is_file():
                staged.unlink()
    return "keep"


def write_jsonl_atomic(path: Path, rows: Sequence[Dict]) -> None:
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args(argv)

    if any(Path(p).exists() for p in (args.output_manifest, args.exclusions)):
        parser.error("output paths must not already exist")
    args.temp_root.mkdir(parents=True, exist_ok=True)

    with args.experiment_config.open("r", encoding="utf-8") as stream:
        experiment_config = json.load(stream)
    tensor_config = experiment_config["learning_tensor_normalization"]
    field_mm = float(tensor_config["crop_field_mm"])
    output_size = int(tensor_config["output_size_px"])
    expected_spacing = float(tensor_config["output_spacing_mm"])
    if not np.isclose(field_mm / output_size, expected_spacing, atol=1e-9, rtol=0):
        parser.error("output spacing does not equal crop_field_mm/output_size_px")
    config_sha256 = sha256_file(args.experiment_config)

    rows = load_jsonl(args.episode_manifest)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    for row_number, row in enumerate(rows, start=1):
        if row.get("partition") == "test":
            parser.error(
                "episode row %d has partition=test; locked test must stay "
                "untouched" % row_number
            )

    visible_tmp = Path(tempfile.mkdtemp(dir=args.temp_root, prefix="visible_"))
    evaluation_tmp = Path(
        tempfile.mkdtemp(dir=args.temp_root, prefix="evaluation_")
    )
    kept: list[Dict] = []
    excluded: list[Dict] = []
    try:
        for row in rows:
            verdict = classify_row(
                row,
                staged_visible_root=visible_tmp,
                staged_evaluation_root=evaluation_tmp,
                field_mm=field_mm,
                output_size=output_size,
                expected_spacing=expected_spacing,
                config_sha256=config_sha256,
            )
            if verdict == "keep":
                kept.append(row)
            elif verdict == "crop_excluded":
                report = crop_offset_report(row, field_mm, expected_spacing)
                excluded.append(
                    {"row": row, "report": report, "error": CROP_ERROR_MARKER}
                )
            else:
                print("PROBE ABORT: row %s -> %s" % (row.get("episode_id"), verdict))
                return 1
    finally:
        for tmp in (visible_tmp, evaluation_tmp):
            try:
                tmp.rmdir()
            except OSError:
                pass

    write_jsonl_atomic(args.output_manifest, kept)
    write_jsonl_atomic(args.exclusions, excluded)
    print(
        json.dumps(
            {
                "total": len(rows),
                "kept": len(kept),
                "crop_excluded": len(excluded),
                "output_manifest": str(args.output_manifest),
                "exclusions": str(args.exclusions),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
