from __future__ import annotations

import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "data"))

import psma_v3_preflight as preflight  # noqa: E402
from psma_v3_preflight import (  # noqa: E402
    ArchiveContractError,
    ArchiveSafetyError,
    inspect_archive,
    safe_extract,
    verify_release,
)


def _write_release_zip(path: Path, *, incomplete: bool = False) -> dict[str, bytes]:
    members = {
        "PSMA-PET-CT-Lesions_v3/imagesTr/psma_case_0000.nii.gz": b"ct",
        "PSMA-PET-CT-Lesions_v3/imagesTr/psma_case_0001.nii.gz": b"pet",
        "PSMA-PET-CT-Lesions_v3/labelsTr/psma_case.nii.gz": b"label",
        "PSMA-PET-CT-Lesions_v3/metadata.csv": b"case,patient\n",
    }
    if incomplete:
        members.pop("PSMA-PET-CT-Lesions_v3/imagesTr/psma_case_0001.nii.gz")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return members


def test_inspect_archive_finds_complete_ct_pet_label_triplet(tmp_path: Path) -> None:
    archive = tmp_path / "psma.zip"
    _write_release_zip(archive)

    report = inspect_archive(archive)

    assert report["entry_count"] == 4
    assert report["triplets"]["complete_count"] == 1
    assert report["triplets"]["incomplete"] == []
    assert report["triplets"]["complete_ids"] == ["psma_case"]


def test_inspect_archive_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", b"escape")

    with pytest.raises(ArchiveSafetyError, match="unsafe archive path"):
        inspect_archive(archive)


@pytest.mark.parametrize(
    "member_name",
    [
        "/absolute.txt",
        "C:/windows-drive.txt",
        "C:drive-relative.txt",
        "D:../escape.txt",
        "folder/D:drive-relative.txt",
        "folder/file.txt:alternate-stream",
        r"folder\backslash.txt",
    ],
)
def test_inspect_archive_rejects_non_posix_relative_paths(
    tmp_path: Path, member_name: str
) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        if "\\" in member_name:
            output.writestr(member_name.replace("\\", "/"), b"unsafe")
        else:
            output.writestr(member_name, b"unsafe")
    if "\\" in member_name:
        archive.write_bytes(
            archive.read_bytes().replace(
                member_name.replace("\\", "/").encode(), member_name.encode()
            )
        )

    with pytest.raises(ArchiveSafetyError):
        inspect_archive(archive)


def test_inspect_archive_rejects_case_insensitive_duplicate_paths(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("root/file.txt", b"first")
        output.writestr("ROOT/FILE.TXT", b"second")

    with pytest.raises(ArchiveSafetyError, match="duplicate archive path"):
        inspect_archive(archive)


def test_inspect_archive_rejects_symbolic_links(tmp_path: Path) -> None:
    archive = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("root/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(link, b"target")

    with pytest.raises(ArchiveSafetyError, match="symbolic link"):
        inspect_archive(archive)


def test_verify_release_checks_size_md5_entry_and_triplet_contract(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "psma.zip"
    _write_release_zip(archive)
    expected_md5 = hashlib.md5(archive.read_bytes()).hexdigest()

    report = verify_release(
        archive,
        expected_size=archive.stat().st_size,
        expected_md5=expected_md5,
        expected_entries=4,
        expected_triplets=1,
        expected_file_count=4,
        expected_metadata_paths=("PSMA-PET-CT-Lesions_v3/metadata.csv",),
    )

    assert report["contract_status"] == "PASS"
    assert report["archive_md5"] == expected_md5


def test_verify_release_rejects_unexpected_metadata_layout(tmp_path: Path) -> None:
    archive = tmp_path / "psma.zip"
    _write_release_zip(archive)

    with pytest.raises(ArchiveContractError, match="metadata member mismatch"):
        verify_release(
            archive,
            expected_size=archive.stat().st_size,
            expected_md5=hashlib.md5(archive.read_bytes()).hexdigest(),
            expected_entries=4,
            expected_triplets=1,
            expected_file_count=4,
            expected_metadata_paths=("PSMA-PET-CT-Lesions_v3/dataset.json",),
        )


def test_verify_release_blocks_incomplete_triplets(tmp_path: Path) -> None:
    archive = tmp_path / "psma.zip"
    _write_release_zip(archive, incomplete=True)

    with pytest.raises(ArchiveContractError, match="incomplete triplets"):
        verify_release(
            archive,
            expected_size=archive.stat().st_size,
            expected_md5=hashlib.md5(archive.read_bytes()).hexdigest(),
            expected_entries=3,
            expected_triplets=1,
        )


def test_verify_release_does_not_join_modalities_across_dataset_roots(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "cross-root.zip"
    members = {
        "rootA/imagesTr/psma_case_0000.nii.gz": b"ct",
        "rootB/imagesTr/psma_case_0001.nii.gz": b"pet",
        "rootC/labelsTr/psma_case.nii.gz": b"label",
    }
    with zipfile.ZipFile(archive, "w") as output:
        for name, payload in members.items():
            output.writestr(name, payload)

    with pytest.raises(ArchiveContractError, match="incomplete triplets"):
        verify_release(
            archive,
            expected_size=archive.stat().st_size,
            expected_md5=hashlib.md5(archive.read_bytes()).hexdigest(),
            expected_entries=3,
            expected_triplets=1,
            expected_root=None,
        )


def test_verify_release_rejects_duplicate_modality_for_one_case(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "duplicate-role.zip"
    members = {
        "PSMA-PET-CT-Lesions_v3/imagesTr/psma_case_0000.nii": b"ct-a",
        "PSMA-PET-CT-Lesions_v3/imagesTr/psma_case_0000.nii.gz": b"ct-b",
        "PSMA-PET-CT-Lesions_v3/imagesTr/psma_case_0001.nii.gz": b"pet",
        "PSMA-PET-CT-Lesions_v3/labelsTr/psma_case.nii.gz": b"label",
    }
    with zipfile.ZipFile(archive, "w") as output:
        for name, payload in members.items():
            output.writestr(name, payload)

    with pytest.raises(ArchiveContractError, match="duplicate triplet role"):
        verify_release(
            archive,
            expected_size=archive.stat().st_size,
            expected_md5=hashlib.md5(archive.read_bytes()).hexdigest(),
            expected_entries=4,
            expected_triplets=1,
        )


def test_safe_extract_refuses_nonempty_destination(tmp_path: Path) -> None:
    archive = tmp_path / "psma.zip"
    _write_release_zip(archive)
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(ArchiveSafetyError, match="destination already exists"):
        safe_extract(archive, destination)

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "do not overwrite"


def test_safe_extract_refuses_destination_path_even_if_exists_is_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "psma.zip"
    _write_release_zip(archive)
    destination = tmp_path / "reserved-destination"
    real_lexists = preflight.os.path.lexists

    def simulated_broken_symlink(path) -> bool:
        if Path(path) == destination:
            return True
        return real_lexists(path)

    monkeypatch.setattr(preflight.os.path, "lexists", simulated_broken_symlink)

    with pytest.raises(ArchiveSafetyError, match="destination already exists"):
        safe_extract(archive, destination)


def test_safe_extract_writes_sha256_manifest(tmp_path: Path) -> None:
    archive = tmp_path / "psma.zip"
    members = _write_release_zip(archive)
    destination = tmp_path / "extracted"

    manifest_path = safe_extract(
        archive,
        destination,
        expected_size=archive.stat().st_size,
        expected_md5=hashlib.md5(archive.read_bytes()).hexdigest(),
        expected_entries=4,
        expected_file_count=4,
        expected_triplets=1,
        expected_root="PSMA-PET-CT-Lesions_v3",
        expected_metadata_paths=("PSMA-PET-CT-Lesions_v3/metadata.csv",),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path == destination / "SHA256-MANIFEST.json"
    assert manifest["status"] == "PASS"
    assert len(manifest["files"]) == len(members)
    by_name = {row["path"]: row for row in manifest["files"]}
    for name, payload in members.items():
        assert (destination / name).read_bytes() == payload
        assert by_name[name]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["archive"]["md5"] == hashlib.md5(archive.read_bytes()).hexdigest()
    assert (
        manifest["archive"]["sha256"]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert manifest["archive"]["bytes"] == archive.stat().st_size
    assert manifest["verification_report_sha256"]
    assert manifest["contract_expected"] == {
        "archive_bytes": archive.stat().st_size,
        "archive_md5": hashlib.md5(archive.read_bytes()).hexdigest(),
        "entry_count": 4,
        "file_count": 4,
        "triplet_count": 1,
        "dataset_root": "PSMA-PET-CT-Lesions_v3",
        "metadata_paths": ["PSMA-PET-CT-Lesions_v3/metadata.csv"],
    }
    assert manifest["tool"]["contract_version"]
    assert len(manifest["tool"]["sha256"]) == 64


def test_safe_extract_fsyncs_nested_directories_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "psma.zip"
    _write_release_zip(archive)
    destination = tmp_path / "extracted"
    staging = destination.with_name("extracted.partial")
    calls: list[Path] = []

    monkeypatch.setattr(preflight, "_fsync_directory", lambda path: calls.append(path))

    safe_extract(archive, destination)

    assert staging in calls
    assert staging / "PSMA-PET-CT-Lesions_v3" / "imagesTr" in calls
    assert staging / "PSMA-PET-CT-Lesions_v3" / "labelsTr" in calls


def test_safe_extract_reports_post_publish_directory_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "psma.zip"
    _write_release_zip(archive)
    destination = tmp_path / "extracted"
    real_fsync_directory = preflight._fsync_directory

    def fail_after_publish(path: Path) -> None:
        if path == destination.parent:
            raise OSError("simulated parent fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(preflight, "_fsync_directory", fail_after_publish)

    with pytest.raises(
        ArchiveSafetyError, match="published but parent directory fsync failed"
    ):
        safe_extract(archive, destination)

    assert destination.is_dir()
    assert (destination / "SHA256-MANIFEST.json").is_file()


def test_safe_extract_does_not_publish_if_manifest_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "psma.zip"
    _write_release_zip(archive)
    destination = tmp_path / "extracted"

    def fail_manifest(*args, **kwargs):
        raise OSError("simulated manifest failure")

    monkeypatch.setattr(preflight, "_write_json_atomic", fail_manifest)

    with pytest.raises(OSError, match="simulated manifest failure"):
        safe_extract(archive, destination)

    assert not destination.exists()
    assert destination.with_name("extracted.partial").exists()


def test_safe_extract_blocks_bad_crc_without_publishing(tmp_path: Path) -> None:
    archive = tmp_path / "bad-crc.zip"
    members = {
        "root/imagesTr/case_0000.nii.gz": b"ct-payload",
        "root/imagesTr/case_0001.nii.gz": b"pet-payload",
        "root/labelsTr/case.nii.gz": b"label-payload",
    }
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        for name, payload in members.items():
            output.writestr(name, payload)
    with zipfile.ZipFile(archive, "r") as source:
        info = source.getinfo("root/imagesTr/case_0000.nii.gz")
    raw = bytearray(archive.read_bytes())
    payload_offset = (
        info.header_offset + 30 + len(info.filename.encode("utf-8")) + len(info.extra)
    )
    raw[payload_offset] ^= 0xFF
    archive.write_bytes(raw)
    destination = tmp_path / "extracted"

    with pytest.raises(ArchiveContractError, match="CRC"):
        safe_extract(archive, destination)

    assert not destination.exists()
    assert destination.with_name("extracted.partial").exists()


def test_safe_extract_blocks_insufficient_space_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "psma.zip"
    _write_release_zip(archive)
    destination = tmp_path / "extracted"
    monkeypatch.setattr(
        preflight.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )

    with pytest.raises(ArchiveSafetyError, match="insufficient free space"):
        safe_extract(archive, destination, free_space_reserve_bytes=1)

    assert not destination.exists()
    assert not destination.with_name("extracted.partial").exists()


def test_safe_extract_blocks_archive_path_replacement_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "psma.zip"
    _write_release_zip(archive)
    destination = tmp_path / "extracted"
    actual_identity = preflight._path_identity
    calls = 0

    def simulate_path_replacement(path):
        nonlocal calls
        calls += 1
        identity = actual_identity(path)
        if calls >= 2:
            return (*identity[:-1], identity[-1] + 1)
        return identity

    monkeypatch.setattr(preflight, "_path_identity", simulate_path_replacement)

    with pytest.raises(ArchiveSafetyError, match="archive path changed"):
        safe_extract(archive, destination)

    assert not destination.exists()
