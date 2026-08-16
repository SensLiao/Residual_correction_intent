#!/usr/bin/env python3
"""Verify and safely extract the PSMA-PET-CT Lesions v3 ZIP release.

The tool is deliberately limited to archive provenance, filesystem safety, and
CT/PET/label filename pairing. It does not infer patients, repeats, splits,
lesion components, or research eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from zipfile import BadZipFile, ZipFile, ZipInfo


EXPECTED_ARCHIVE_BYTES = 20_588_970_455
EXPECTED_ARCHIVE_MD5 = "156c136aea40541275d97cc6bfae9f39"
EXPECTED_ENTRY_COUNT = 1_795
EXPECTED_FILE_COUNT = 1_795
EXPECTED_TRIPLET_COUNT = 597
EXPECTED_DATASET_ROOT = "PSMA-PET-CT-Lesions_v3"
EXPECTED_METADATA_PATHS = (
    "PSMA-PET-CT-Lesions_v3/dataset_fingerprint.json",
    "PSMA-PET-CT-Lesions_v3/splits_final.json",
    "PSMA-PET-CT-Lesions_v3/dataset.json",
    "PSMA-PET-CT-Lesions_v3/psma_metadata.csv",
)
COPY_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_FREE_SPACE_RESERVE_BYTES = 5 * 1024**3
MANIFEST_NAME = "SHA256-MANIFEST.json"
TOOL_CONTRACT_VERSION = "1.0.0"


class ArchiveSafetyError(RuntimeError):
    """The archive cannot be handled without risking an unsafe filesystem write."""


class ArchiveContractError(RuntimeError):
    """The archive does not match the frozen release contract."""


def _identity_from_stat(status: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _path_identity(path: Path) -> tuple[int, int, int, int, int]:
    return _identity_from_stat(path.stat())


def _identity_payload(identity: tuple[int, int, int, int, int]) -> dict[str, int]:
    device, inode, size, modified_ns, changed_ns = identity
    return {
        "device": device,
        "inode": inode,
        "bytes": size,
        "modified_ns": modified_ns,
        "changed_ns": changed_ns,
    }


def _is_symlink(info: ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return info.create_system == 3 and stat.S_ISLNK(unix_mode)


def _validate_member(info: ZipInfo) -> PurePosixPath:
    name = info.orig_filename
    if not name or "\\" in name or "\x00" in name:
        raise ArchiveSafetyError(f"unsafe archive path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ArchiveSafetyError(f"unsafe archive path: {name!r}")
    if any(":" in component for component in path.parts):
        raise ArchiveSafetyError(f"unsafe archive path: {name!r}")
    if _is_symlink(info):
        raise ArchiveSafetyError(f"symbolic link is not allowed: {name!r}")
    if info.create_system == 3:
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ArchiveSafetyError(
                f"special filesystem entry is not allowed: {name!r}"
            )
    return path


def _safe_target(staging: Path, member: PurePosixPath) -> Path:
    base = staging.resolve(strict=False)
    candidate = staging.joinpath(*member.parts).resolve(strict=False)
    base_text = os.path.normcase(str(base))
    candidate_text = os.path.normcase(str(candidate))
    try:
        common = os.path.commonpath([base_text, candidate_text])
    except ValueError as exc:
        raise ArchiveSafetyError(
            f"archive member resolves outside staging: {member.as_posix()!r}"
        ) from exc
    if common != base_text:
        raise ArchiveSafetyError(
            f"archive member resolves outside staging: {member.as_posix()!r}"
        )
    return candidate


def _nifti_stem(filename: str) -> str | None:
    lower = filename.casefold()
    if lower.endswith(".nii.gz"):
        return filename[:-7]
    if lower.endswith(".nii"):
        return filename[:-4]
    return None


def _triplet_role(path: PurePosixPath) -> tuple[str, str, str] | None:
    """Return dataset root, case id, and role for a direct nnU-Net-style member."""
    if len(path.parts) != 3:
        return None
    stem = _nifti_stem(path.name)
    if stem is None:
        return None
    dataset_root, parent, _ = path.parts
    parent = parent.casefold()
    stem_folded = stem.casefold()
    if parent == "imagestr" and stem_folded.endswith("_0000"):
        return dataset_root, stem[:-5], "ct"
    if parent == "imagestr" and stem_folded.endswith("_0001"):
        return dataset_root, stem[:-5], "pet"
    if parent == "labelstr":
        return dataset_root, stem, "label"
    return None


def _inspect_zip(archive: ZipFile, archive_path: Path) -> dict:
    seen: dict[str, str] = {}
    triplets: dict[tuple[str, str], dict[str, list[str]]] = {}
    dataset_roots: set[str] = set()
    unclassified_nifti: list[str] = []
    other_files: list[str] = []
    entry_count = 0
    file_count = 0
    uncompressed_bytes = 0
    compressed_bytes = 0

    for info in archive.infolist():
        path = _validate_member(info)
        canonical = path.as_posix().casefold().rstrip("/")
        if canonical in seen:
            raise ArchiveSafetyError(
                "duplicate archive path after case normalization: "
                f"{seen[canonical]!r} and {info.filename!r}"
            )
        seen[canonical] = info.filename
        entry_count += 1
        compressed_bytes += info.compress_size
        uncompressed_bytes += info.file_size
        if info.is_dir():
            continue
        file_count += 1
        role = _triplet_role(path)
        if role is not None:
            dataset_root, case_id, modality = role
            dataset_roots.add(dataset_root)
            by_role = triplets.setdefault((dataset_root, case_id), {})
            by_role.setdefault(modality, []).append(path.as_posix())
        elif _nifti_stem(path.name) is not None:
            unclassified_nifti.append(path.as_posix())
        else:
            other_files.append(path.as_posix())

    required = {"ct", "pet", "label"}
    complete_keys: list[tuple[str, str]] = []
    incomplete: list[dict] = []
    conflicts: list[dict] = []
    for (dataset_root, case_id), by_role in sorted(triplets.items()):
        present = set(by_role)
        missing = required - present
        duplicate_roles = {
            role: paths for role, paths in sorted(by_role.items()) if len(paths) != 1
        }
        if duplicate_roles:
            conflicts.append(
                {
                    "dataset_root": dataset_root,
                    "case_id": case_id,
                    "duplicate_roles": duplicate_roles,
                }
            )
        if missing:
            incomplete.append(
                {
                    "dataset_root": dataset_root,
                    "case_id": case_id,
                    "present": sorted(present),
                    "missing": sorted(missing),
                }
            )
        if not missing and not duplicate_roles:
            complete_keys.append((dataset_root, case_id))

    return {
        "archive": str(archive_path.resolve()),
        "entry_count": entry_count,
        "file_count": file_count,
        "compressed_entry_bytes": compressed_bytes,
        "uncompressed_entry_bytes": uncompressed_bytes,
        "dataset_roots": sorted(dataset_roots),
        "other_files": sorted(other_files),
        "triplets": {
            "complete_count": len(complete_keys),
            "complete_ids": [case_id for _, case_id in complete_keys],
            "complete_keys": [
                {"dataset_root": root, "case_id": case_id}
                for root, case_id in complete_keys
            ],
            "incomplete": incomplete,
            "conflicts": conflicts,
            "unclassified_nifti": sorted(unclassified_nifti),
            "role_counts": {
                role: sum(len(paths.get(role, [])) for paths in triplets.values())
                for role in sorted(required)
            },
        },
        "scope_guard": (
            "filename-level CT/PET/label pairing only; patient/repeat mapping and "
            "medical-image QA remain separate P0 steps"
        ),
    }


def inspect_archive(archive_path: Path | str) -> dict:
    """Inspect all ZIP members and fail closed on unsafe filesystem semantics."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive_path}")
    try:
        with archive_path.open("rb") as stream, ZipFile(stream, "r") as archive:
            return _inspect_zip(archive, archive_path)
    except BadZipFile as exc:
        raise ArchiveContractError(f"invalid ZIP archive: {archive_path}") from exc


def _hash_stream(stream: BinaryIO) -> tuple[int, str, str]:
    # FDAT publishes an MD5 release identifier; SHA-256 is computed alongside it
    # and is the cryptographic identity recorded for local audit.
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    stream.seek(0)
    for chunk in iter(lambda: stream.read(COPY_CHUNK_BYTES), b""):
        md5.update(chunk)
        sha256.update(chunk)
        size += len(chunk)
    stream.seek(0)
    return size, md5.hexdigest(), sha256.hexdigest()


def _prepare_archive(stream: BinaryIO, archive_path: Path) -> dict:
    actual_size, actual_md5, actual_sha256 = _hash_stream(stream)
    try:
        with ZipFile(stream, "r") as archive:
            report = _inspect_zip(archive, archive_path)
    except BadZipFile as exc:
        raise ArchiveContractError(f"invalid ZIP archive: {archive_path}") from exc
    stream.seek(0)
    return {
        **report,
        "archive_bytes": actual_size,
        "archive_md5": actual_md5,
        "archive_sha256": actual_sha256,
    }


def _validate_contract(
    report: dict,
    *,
    expected_size: int | None,
    expected_md5: str | None,
    expected_entries: int | None,
    expected_file_count: int | None,
    expected_triplets: int | None,
    expected_root: str | None,
    expected_metadata_paths: tuple[str, ...] | None,
) -> dict:
    if expected_size is not None and report["archive_bytes"] != expected_size:
        raise ArchiveContractError(
            f"archive size mismatch: {report['archive_bytes']} != {expected_size}"
        )
    if (
        expected_md5 is not None
        and report["archive_md5"].casefold() != expected_md5.casefold()
    ):
        raise ArchiveContractError(
            f"archive MD5 mismatch: {report['archive_md5']} != {expected_md5}"
        )
    conflicts = report["triplets"]["conflicts"]
    if conflicts:
        raise ArchiveContractError(
            f"duplicate triplet role detected: {len(conflicts)} case(s)"
        )
    incomplete = report["triplets"]["incomplete"]
    if incomplete:
        raise ArchiveContractError(f"incomplete triplets detected: {len(incomplete)}")
    unclassified = report["triplets"]["unclassified_nifti"]
    if unclassified:
        raise ArchiveContractError(
            f"unclassified NIfTI members detected: {len(unclassified)}"
        )
    if expected_root is not None and report["dataset_roots"] != [expected_root]:
        raise ArchiveContractError(
            f"dataset root mismatch: {report['dataset_roots']} != {[expected_root]}"
        )
    if expected_metadata_paths is not None and report["other_files"] != sorted(
        expected_metadata_paths
    ):
        raise ArchiveContractError(
            "metadata member mismatch: "
            f"{report['other_files']} != {sorted(expected_metadata_paths)}"
        )
    if expected_entries is not None and report["entry_count"] != expected_entries:
        raise ArchiveContractError(
            f"entry count mismatch: {report['entry_count']} != {expected_entries}"
        )
    if expected_file_count is not None and report["file_count"] != expected_file_count:
        raise ArchiveContractError(
            f"file count mismatch: {report['file_count']} != {expected_file_count}"
        )
    if (
        expected_triplets is not None
        and report["triplets"]["complete_count"] != expected_triplets
    ):
        raise ArchiveContractError(
            "complete triplet count mismatch: "
            f"{report['triplets']['complete_count']} != {expected_triplets}"
        )
    return {**report, "contract_status": "PASS"}


def verify_release(
    archive_path: Path | str,
    *,
    expected_size: int = EXPECTED_ARCHIVE_BYTES,
    expected_md5: str = EXPECTED_ARCHIVE_MD5,
    expected_entries: int = EXPECTED_ENTRY_COUNT,
    expected_file_count: int = EXPECTED_FILE_COUNT,
    expected_triplets: int = EXPECTED_TRIPLET_COUNT,
    expected_root: str | None = EXPECTED_DATASET_ROOT,
    expected_metadata_paths: tuple[str, ...] | None = EXPECTED_METADATA_PATHS,
) -> dict:
    """Verify one opened file against the frozen release and triplet contract."""
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive_path}")
    initial_identity = _path_identity(archive_path)
    if initial_identity[2] != expected_size:
        raise ArchiveContractError(
            f"archive size mismatch: {initial_identity[2]} != {expected_size}"
        )
    with archive_path.open("rb") as stream:
        handle_identity = _identity_from_stat(os.fstat(stream.fileno()))
        if handle_identity != initial_identity:
            raise ArchiveSafetyError("archive path changed while it was opened")
        report = _prepare_archive(stream, archive_path)
        if _identity_from_stat(os.fstat(stream.fileno())) != initial_identity:
            raise ArchiveSafetyError("archive changed during verification")
        if _path_identity(archive_path) != initial_identity:
            raise ArchiveSafetyError("archive path changed during verification")
    return _validate_contract(
        report,
        expected_size=expected_size,
        expected_md5=expected_md5,
        expected_entries=expected_entries,
        expected_file_count=expected_file_count,
        expected_triplets=expected_triplets,
        expected_root=expected_root,
        expected_metadata_paths=expected_metadata_paths,
    )


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise ArchiveSafetyError(f"no existing parent for destination: {path}")
        candidate = candidate.parent
    return candidate


def _copy_and_hash(source: BinaryIO, target: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    written = 0
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        for chunk in iter(lambda: source.read(COPY_CHUNK_BYTES), b""):
            output.write(chunk)
            digest.update(chunk)
            written += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    return written, digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    for directory, _, _ in os.walk(root, topdown=False):
        _fsync_directory(Path(directory))


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if os.path.lexists(path):
        raise ArchiveSafetyError(f"manifest already exists: {path}")
    if os.path.lexists(temporary):
        raise ArchiveSafetyError(f"temporary manifest already exists: {temporary}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tool_sha256() -> str:
    digest = hashlib.sha256()
    with Path(__file__).resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(
    archive_path: Path | str,
    destination: Path | str,
    *,
    free_space_reserve_bytes: int = DEFAULT_FREE_SPACE_RESERVE_BYTES,
    expected_size: int | None = None,
    expected_md5: str | None = None,
    expected_entries: int | None = None,
    expected_file_count: int | None = None,
    expected_triplets: int | None = None,
    expected_root: str | None = None,
    expected_metadata_paths: tuple[str, ...] | None = None,
) -> Path:
    """Verify and extract one open file, then atomically publish data plus manifest."""
    archive_path = Path(archive_path)
    destination = Path(destination)
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive_path}")
    if os.path.lexists(destination):
        raise ArchiveSafetyError(f"destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise ArchiveSafetyError(
            f"destination parent must already exist: {destination.parent}"
        )
    staging = destination.with_name(destination.name + ".partial")
    if os.path.lexists(staging):
        raise ArchiveSafetyError(f"partial destination already exists: {staging}")

    initial_identity = _path_identity(archive_path)
    if expected_size is not None and initial_identity[2] != expected_size:
        raise ArchiveContractError(
            f"archive size mismatch: {initial_identity[2]} != {expected_size}"
        )

    with archive_path.open("rb") as stream:
        handle_identity = _identity_from_stat(os.fstat(stream.fileno()))
        if handle_identity != initial_identity:
            raise ArchiveSafetyError("archive path changed while it was opened")
        report = _prepare_archive(stream, archive_path)
        verification_report = _validate_contract(
            report,
            expected_size=expected_size,
            expected_md5=expected_md5,
            expected_entries=expected_entries,
            expected_file_count=expected_file_count,
            expected_triplets=expected_triplets,
            expected_root=expected_root,
            expected_metadata_paths=expected_metadata_paths,
        )

        required_bytes = (
            verification_report["uncompressed_entry_bytes"] + free_space_reserve_bytes
        )
        disk_root = _nearest_existing_parent(destination.parent)
        free_bytes = shutil.disk_usage(disk_root).free
        if free_bytes < required_bytes:
            raise ArchiveSafetyError(
                f"insufficient free space: {free_bytes} < required {required_bytes}"
            )

        staging.mkdir(mode=0o700, parents=False)
        files: list[dict] = []
        try:
            stream.seek(0)
            with ZipFile(stream, "r") as archive:
                for info in archive.infolist():
                    member = _validate_member(info)
                    target = _safe_target(staging, member)
                    if info.is_dir():
                        target.mkdir(mode=0o700, parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    with archive.open(info, "r") as source:
                        written, sha256 = _copy_and_hash(source, target)
                    if written != info.file_size:
                        raise ArchiveContractError(
                            f"extracted size mismatch for {info.filename}: "
                            f"{written} != {info.file_size}"
                        )
                    files.append(
                        {
                            "path": member.as_posix(),
                            "bytes": written,
                            "sha256": sha256,
                            "zip_crc32": f"{info.CRC:08x}",
                        }
                    )
        except BadZipFile as exc:
            raise ArchiveContractError(
                f"archive changed or failed CRC during extraction: {archive_path}"
            ) from exc

        if _identity_from_stat(os.fstat(stream.fileno())) != initial_identity:
            raise ArchiveSafetyError("archive changed during extraction")
        if _path_identity(archive_path) != initial_identity:
            raise ArchiveSafetyError("archive path changed before publish")

        verification_hash = _canonical_json_sha256(verification_report)
        manifest_staging = staging / MANIFEST_NAME
        manifest_payload = {
            "status": "PASS",
            "archive": {
                "path": str(archive_path.resolve()),
                "bytes": report["archive_bytes"],
                "md5": report["archive_md5"],
                "sha256": report["archive_sha256"],
                "source_identity": _identity_payload(initial_identity),
            },
            "destination": str(destination.resolve()),
            "verification_report": verification_report,
            "verification_report_sha256": verification_hash,
            "contract_expected": {
                "archive_bytes": expected_size,
                "archive_md5": expected_md5,
                "entry_count": expected_entries,
                "file_count": expected_file_count,
                "triplet_count": expected_triplets,
                "dataset_root": expected_root,
                "metadata_paths": (
                    list(expected_metadata_paths)
                    if expected_metadata_paths is not None
                    else None
                ),
            },
            "tool": {
                "name": Path(__file__).name,
                "contract_version": TOOL_CONTRACT_VERSION,
                "sha256": _tool_sha256(),
            },
            "file_count": len(files),
            "files": files,
            "scope_guard": (
                "extraction identity only; patient mapping, NIfTI geometry, SUV, label, "
                "and eligibility checks are not implied"
            ),
        }
        _write_json_atomic(manifest_staging, manifest_payload)
        if _identity_from_stat(os.fstat(stream.fileno())) != initial_identity:
            raise ArchiveSafetyError("archive changed before publish")
        if _path_identity(archive_path) != initial_identity:
            raise ArchiveSafetyError("archive path changed before publish")
        if os.path.lexists(destination):
            raise ArchiveSafetyError(
                f"destination appeared before publish: {destination}"
            )
        _fsync_tree_directories(staging)
        os.rename(staging, destination)
        try:
            _fsync_directory(destination.parent)
        except OSError as exc:
            raise ArchiveSafetyError(
                "destination published but parent directory fsync failed; "
                "state=PUBLISHED_DURABILITY_UNCONFIRMED; manual audit required: "
                f"{destination}"
            ) from exc

    return destination / MANIFEST_NAME


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify the frozen PSMA v3 release")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--report", type=Path)
    extract = subparsers.add_parser(
        "extract", help="verify, safely extract, and write per-file SHA-256"
    )
    extract.add_argument("archive", type=Path)
    extract.add_argument("destination", type=Path)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "verify":
        report = verify_release(args.archive)
        if args.report:
            _write_json_atomic(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    manifest = safe_extract(
        args.archive,
        args.destination,
        expected_size=EXPECTED_ARCHIVE_BYTES,
        expected_md5=EXPECTED_ARCHIVE_MD5,
        expected_entries=EXPECTED_ENTRY_COUNT,
        expected_file_count=EXPECTED_FILE_COUNT,
        expected_triplets=EXPECTED_TRIPLET_COUNT,
        expected_root=EXPECTED_DATASET_ROOT,
        expected_metadata_paths=EXPECTED_METADATA_PATHS,
    )
    print(json.dumps({"status": "PASS", "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
