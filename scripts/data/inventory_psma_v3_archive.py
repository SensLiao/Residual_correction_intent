#!/usr/bin/env python3
"""Create a read-only, provenance-preserving inventory of PSMA v3.

The script deliberately does not extract the archive or infer patient splits. It
only runs after the download workflow has produced DOWNLOAD_VERIFIED.done.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


EXPECTED_ARCHIVE_BYTES = 20_588_970_455
VERIFIED_SENTINEL = "DOWNLOAD_VERIFIED.done"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory a fully verified PSMA-PET-CT Lesions v3 archive."
    )
    parser.add_argument("archive", type=Path, help="Path to the verified ZIP archive")
    parser.add_argument("output_dir", type=Path, help="Directory for JSON and CSV manifests")
    return parser.parse_args()


def normalized_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    suffix = PurePosixPath(name).suffix.lower()
    return suffix or "<none>"


def entry_class(name: str) -> str:
    lower = name.lower()
    if lower.endswith("/"):
        return "directory"
    if lower.endswith(".nii.gz") or lower.endswith(".nii"):
        if "label" in lower or "seg" in lower or "mask" in lower:
            return "candidate_label"
        if "_0000." in lower or "ct" in PurePosixPath(lower).name:
            return "candidate_ct"
        if "_0001." in lower or "pet" in PurePosixPath(lower).name:
            return "candidate_pet"
        return "unclassified_nifti"
    if lower.endswith((".csv", ".json", ".txt", ".yaml", ".yml")):
        return "metadata"
    return "other"


def path_is_safe(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    archive = args.archive.resolve()
    output_dir = args.output_dir.resolve()

    if not archive.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive}")
    sentinel = archive.parent / VERIFIED_SENTINEL
    if not sentinel.is_file():
        raise RuntimeError(
            f"refusing to inspect an unverified archive; missing {sentinel}"
        )
    actual_bytes = archive.stat().st_size
    if actual_bytes != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"archive size changed after verification: {actual_bytes} "
            f"!= {EXPECTED_ARCHIVE_BYTES}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "psma_v3_archive_inventory.json"
    csv_path = output_dir / "psma_v3_archive_entries.csv"

    suffix_counts: Counter[str] = Counter()
    top_level_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    seen_names: Counter[str] = Counter()
    unsafe_names: list[str] = []
    rows: list[dict[str, object]] = []
    total_compressed = 0
    total_uncompressed = 0

    with ZipFile(archive, "r") as zip_file:
        for index, info in enumerate(zip_file.infolist()):
            name = info.filename
            path = PurePosixPath(name)
            safe = path_is_safe(name)
            if not safe:
                unsafe_names.append(name)

            suffix = normalized_suffix(name)
            category = entry_class(name)
            top_level = path.parts[0] if path.parts else "<empty>"
            suffix_counts[suffix] += 1
            top_level_counts[top_level] += 1
            class_counts[category] += 1
            seen_names[name] += 1
            total_compressed += info.compress_size
            total_uncompressed += info.file_size

            rows.append(
                {
                    "index": index,
                    "name": name,
                    "safe_relative_path": safe,
                    "is_directory": info.is_dir(),
                    "class": category,
                    "suffix": suffix,
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                }
            )

    duplicate_names = sorted(name for name, count in seen_names.items() if count > 1)
    summary = {
        "dataset": "PSMA-PET-CT Lesions v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "verified_sentinel": str(sentinel),
        "archive_bytes": actual_bytes,
        "entry_count": len(rows),
        "total_compressed_entry_bytes": total_compressed,
        "total_uncompressed_entry_bytes": total_uncompressed,
        "unsafe_entry_count": len(unsafe_names),
        "unsafe_entries": unsafe_names,
        "duplicate_name_count": len(duplicate_names),
        "duplicate_names": duplicate_names,
        "top_level_counts": dict(sorted(top_level_counts.items())),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "interpretation_guard": (
            "candidate_ct/pet/label classes are filename-based inventory hints only; "
            "patient grouping and modality identity must be confirmed from official metadata"
        ),
    }

    if unsafe_names:
        raise RuntimeError(
            f"archive contains {len(unsafe_names)} unsafe paths; no manifest written"
        )

    write_json_atomic(json_path, summary)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"inventory_json={json_path}")
    print(f"entries_csv={csv_path}")
    print(f"entry_count={len(rows)}")
    print(f"unsafe_entry_count={len(unsafe_names)}")
    print(f"duplicate_name_count={len(duplicate_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
