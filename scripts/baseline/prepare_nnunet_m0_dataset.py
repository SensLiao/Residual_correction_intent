#!/usr/bin/env python3
"""Create and validate the transactional PSMA M0 nnU-Net planning contract.

The official PSMA v3 source is treated as read-only by this workflow. This
helper changes only derived nnU-Net metadata from CT/CT to CT/PET and
fail-closes unless an isolated, planning-only run matches the autoPET V
normalization semantics and its complete evidence chain remains intact.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import re
import site
import sys
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


EXPECTED_SOURCE_CHANNELS = {"0": "CT", "1": "CT"}
DERIVED_CHANNELS = {"0": "CT", "1": "PET"}
EXPECTED_LABELS = {"background": 0, "tumor": 1}
EXPECTED_FILE_ENDING = ".nii.gz"
EXPECTED_TRAINING_CASES = 597
DERIVED_DATASET_NAME = "PSMA_M0_AutoPETVNorm"
DATASET_ID = 901
DATASET_FOLDER = "Dataset901_PSMA_M0_AutoPETVNorm"
PLANS_NAME = "nnUNetPlans"
EXPECTED_DATA_IDENTIFIER = "nnUNetPlans_3d_fullres"
EXPECTED_NORMALIZATION_SCHEMES = [
    "CTNormalization",
    "ZScoreNormalization",
]
EXPECTED_USE_MASK_FOR_NORM = [False, False]
EXPECTED_CHANNEL_KEYS = ["0", "1"]
EXPECTED_NNUNET_VERSION = "2.8.1"
EXPECTED_NNUNET_COMMIT = "468cf803df9b267150ae2b6c0c59b8ac84f16227"
EXPECTED_NNUNET_SOURCE_FILE_COUNT = 202
EXPECTED_NNUNET_SOURCE_TREE_SHA256 = (
    "474cbdc319d78638bfcd512f5feb5cbdbb8b57db9230ce55fd4323902c075b92"
)
EXPECTED_AUDIT_VERSION = "1.2.0"
EXPECTED_AUDIT_TOOL_SHA256 = (
    "9faf2f717c64a81386ef64de215d350dc470c2ebbdb86d47efd20c2604270d86"
)
EXPECTED_AUTOPETV_PLANS_SHA256 = (
    "4b99541110d0a99b6c6d0f44ce380e7aaa7d5559dea77c93cabd7759f7f16396"
)
EXPECTED_AUTOPETV_DATASET_SHA256 = (
    "c884d1153c98f317ce9edcce547af170a778bb243cff64ebb9a82030b6cc74ed"
)
EXPECTED_AUTOPETV_COMMIT = "4a2026866bfacc812492cfc7e6a8c54ac3c4f703"
EXPECTED_ARCHIVE_BYTES = 20_588_970_455
EXPECTED_ARCHIVE_MD5 = "156c136aea40541275d97cc6bfae9f39"
EXPECTED_ARCHIVE_SHA256 = (
    "e0d7e5ceba493e8686cbac0af1709a3e112db90b769b580c4085733651904858"
)
EXPECTED_EXTRACTION_MANIFEST_SHA256 = (
    "64b1431740f05ccb1aa7b681b8959be860679a8161964d35004d1634f0bfd09e"
)
EXPECTED_MIGRATION_RECEIPT_SHA256 = (
    "799912427f0e11de8cc15c69a5b3db66db83257331c879cdb3752cdf46f38ecb"
)
EXPECTED_PRE_MIGRATION_EXTRACT_ROOT = (
    "/mnt/HDD4/workspace/honor_degree/Honor_codes/Dataset/AutoPET_V/PSMA_v3/extracted"
)
EXPECTED_MIGRATION_DATA_MAPPING = (
    "Honor_codes/Dataset/AutoPET_V/PSMA_v3 -> "
    "projects/petct_textual_intent/data/PSMA_v3"
)
EXPECTED_EXTRACTED_FILE_COUNT = 1_795
EXPECTED_PATIENTS = 378
EXPECTED_POSITIVE_CASES = 539
EXPECTED_EMPTY_CASES = 58
EXPECTED_PYTHON_EXECUTABLE_SHA256 = (
    "0fd3e7f99c756a61a3b4b10d7ebd7236983c91a96df673b78588c6afff6f6e60"
)
EXPECTED_CONDA_EXPLICIT_NORMALIZED_SHA256 = (
    "5b3ed1ef6ee6f6171f9beb72ed49f5484b6df0c199922a8eaf51c3035037ea59"
)
EXPECTED_PIP_FREEZE_NORMALIZED_SHA256 = (
    "23a26b97db36e9df8d8ba0120e25f9e8640da82ff8cf2264e1213075ba36f59f"
)
CONTRACT_VERSION = "2.0.0"
EXPECTED_EVIDENCE_LABELS = {
    "source_dataset_json",
    "source_splits",
    "source_metadata",
    "extraction_manifest",
    "migration_receipt",
    "audit_pointer",
    "audit_completion",
    "audit_report",
    "audit_csv",
    "audit_tool",
    "planning_gate_tool",
    "autopetv_official_plans",
    "autopetv_official_dataset",
    "python_executable",
    "environment_receipt",
    "conda_explicit",
    "pip_freeze",
    "environment_hash_receipt",
    "forbidden_symlinks",
}
EXPECTED_ARTIFACT_LABELS = {
    "derived_dataset_json",
    "preprocessed_dataset_json",
    "dataset_fingerprint",
    "nnunet_plans",
    "splits_final",
    "runtime_identity",
    "preflight_receipt",
    "run_owner",
    "live_conda_snapshot",
    "live_pip_snapshot",
}


class ContractError(RuntimeError):
    """Source metadata or generated plans violate the frozen M0 contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"required file is missing: {path}")
    return {
        "path": display_path if display_path is not None else str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    linked = False
    try:
        # Hard-link publication is atomic and fails if the destination exists.
        # Unlike os.replace, it can never overwrite another successful receipt.
        os.link(temporary, path)
        linked = True
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite: {path}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            if not linked:
                raise


def write_run_owner(run_id: str, run_root: Path, output_path: Path) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ContractError("unsafe planning run_id")
    run_root = run_root.resolve()
    if output_path.resolve().parent != run_root:
        raise ContractError("run owner record must be inside the staging root")
    payload = {
        "status": "OWNED",
        "contract_version": CONTRACT_VERSION,
        "run_id": run_id,
        "staging_dir_name": run_root.name,
        "owner_token": uuid4().hex,
    }
    _write_json_exclusive(output_path, payload)
    return payload


def commit_run_directory(
    staging_dir: Path, final_dir: Path, run_receipt_path: Path
) -> dict[str, Any]:
    """Atomically publish a run directory without replacing any destination."""

    if sys.platform != "linux":
        raise ContractError("atomic planning-run commit requires Linux renameat2")
    staging = staging_dir.resolve()
    final = final_dir.resolve()
    if staging.parent != final.parent:
        raise ContractError("staging/final run directories must be siblings")
    if staging.stat().st_dev != final.parent.stat().st_dev:
        raise ContractError("staging/final run directories are on different filesystems")
    if os.path.lexists(final):
        raise FileExistsError(f"refusing existing committed run: {final}")
    run_id = final.name
    if staging.name != f".partial-{run_id}":
        raise ContractError("staging directory does not match final run identity")
    owner_path = staging / "RUN_OWNER.json"
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    if owner.get("status") != "OWNED" or owner.get("run_id") != run_id:
        raise ContractError("staging run owner identity mismatch")
    receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    if run_receipt_path.resolve().parent != staging:
        raise ContractError("planning bundle must be inside staging")
    if receipt.get("status") != "VALIDATED" or receipt.get("run_id") != run_id:
        raise ContractError("planning bundle is not validated for this run")
    if Path(receipt.get("committed_run_dir", "")).resolve() != final:
        raise ContractError("planning bundle final directory mismatch")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(staging),
            -100,
            os.fsencode(final),
            1,
        )
        rename_method = "libc.renameat2(RENAME_NOREPLACE)"
    else:
        syscall_numbers = {"x86_64": 316, "aarch64": 276}
        machine = os.uname().machine
        syscall_number = syscall_numbers.get(machine)
        if syscall_number is None:
            raise ContractError(
                f"renameat2 syscall number is not pinned for architecture: {machine}"
            )
        syscall = libc.syscall
        syscall.restype = ctypes.c_long
        result = syscall(
            ctypes.c_long(syscall_number),
            ctypes.c_int(-100),
            ctypes.c_char_p(os.fsencode(staging)),
            ctypes.c_int(-100),
            ctypes.c_char_p(os.fsencode(final)),
            ctypes.c_uint(1),
        )
        rename_method = f"syscall({syscall_number}, RENAME_NOREPLACE)"
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"refusing existing committed run: {final}")
        raise OSError(error, os.strerror(error), str(final))
    _fsync_directory(final.parent)
    return {
        "status": "COMMITTED",
        "run_id": run_id,
        "run_dir": str(final),
        "rename_semantics": rename_method,
    }


def source_tree_manifest(source_root: Path) -> list[dict[str, Any]]:
    """Return a deterministic manifest for the code that nnU-Net imports.

    Runtime caches and repository metadata are intentionally excluded. The
    pinned manifest covers packaging metadata plus every regular package file.
    """

    source_root = source_root.resolve()
    package_root = source_root / "nnunetv2"
    required = [source_root / "pyproject.toml", source_root / "setup.py"]
    if not package_root.is_dir():
        raise ContractError(f"missing nnunetv2 source package: {package_root}")
    files = list(required)
    for path in package_root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ContractError(f"nnU-Net source symlink is not allowed: {path}")
        if path.is_file():
            files.append(path)
    manifest = []
    for path in sorted(set(files), key=lambda item: item.relative_to(source_root).as_posix()):
        if not path.is_file():
            raise ContractError(f"required nnU-Net source file is missing: {path}")
        manifest.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return manifest


def source_tree_sha256(source_root: Path) -> str:
    encoded = json.dumps(
        source_tree_manifest(source_root),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_derived_dataset_json(source: dict) -> dict:
    if source.get("channel_names") != EXPECTED_SOURCE_CHANNELS:
        raise ContractError(
            "source channel_names do not match the verified PSMA v3 CT/CT contract"
        )
    if source.get("labels") != EXPECTED_LABELS:
        raise ContractError("source labels do not match the verified PSMA v3 contract")
    if source.get("file_ending") != EXPECTED_FILE_ENDING:
        raise ContractError("source file_ending does not match the PSMA v3 contract")
    if source.get("numTraining") != EXPECTED_TRAINING_CASES:
        raise ContractError("source numTraining does not equal 597")

    derived = copy.deepcopy(source)
    derived["channel_names"] = copy.deepcopy(DERIVED_CHANNELS)
    derived["name"] = DERIVED_DATASET_NAME
    derived["description"] = (
        "PSMA v3 automatic M0; derived metadata using autoPET V CT/PET "
        "normalization semantics"
    )
    return derived


def validate_derived_dataset_json(source: dict, derived: dict) -> dict:
    expected = build_derived_dataset_json(source)
    if derived != expected:
        raise ContractError("derived dataset.json differs from the frozen CT/PET contract")
    return {
        "source_channel_names": copy.deepcopy(EXPECTED_SOURCE_CHANNELS),
        "derived_channel_names": copy.deepcopy(DERIVED_CHANNELS),
        "expected_3d_fullres_normalization": list(
            EXPECTED_NORMALIZATION_SCHEMES
        ),
    }


def validate_plan_normalization(
    plans: dict, *, configuration: str = "3d_fullres"
) -> dict:
    try:
        config = plans["configurations"][configuration]
    except (KeyError, TypeError) as exc:
        raise ContractError(f"missing plans configuration: {configuration}") from exc
    normalization_schemes = config.get("normalization_schemes")
    use_mask_for_norm = config.get("use_mask_for_norm")
    if (
        normalization_schemes != EXPECTED_NORMALIZATION_SCHEMES
        or use_mask_for_norm != EXPECTED_USE_MASK_FOR_NORM
    ):
        raise ContractError(
            "normalization contract mismatch: "
            f"{normalization_schemes}/{use_mask_for_norm} != "
            f"{EXPECTED_NORMALIZATION_SCHEMES}/{EXPECTED_USE_MASK_FOR_NORM}"
        )
    return {
        "configuration": configuration,
        "normalization_schemes": list(normalization_schemes),
        "use_mask_for_norm": list(use_mask_for_norm),
    }


def validate_plan_contract(
    plans: dict, *, configuration: str = "3d_fullres"
) -> dict[str, Any]:
    if plans.get("dataset_name") != DATASET_FOLDER:
        raise ContractError(
            f"plans dataset_name mismatch: {plans.get('dataset_name')} != {DATASET_FOLDER}"
        )
    if plans.get("plans_name") != PLANS_NAME:
        raise ContractError(
            f"plans_name mismatch: {plans.get('plans_name')} != {PLANS_NAME}"
        )
    if plans.get("experiment_planner_used") != "ExperimentPlanner":
        raise ContractError("plans experiment_planner_used mismatch")
    image_reader_writer = plans.get("image_reader_writer")
    if not isinstance(image_reader_writer, str) or not image_reader_writer:
        raise ContractError("plans image_reader_writer is missing")
    intensity = plans.get("foreground_intensity_properties_per_channel")
    if not isinstance(intensity, dict) or sorted(intensity) != EXPECTED_CHANNEL_KEYS:
        raise ContractError(
            "plans channel contract mismatch: expected exactly channels 0/1"
        )
    normalization = validate_plan_normalization(plans, configuration=configuration)
    config = plans["configurations"][configuration]
    if config.get("data_identifier") != EXPECTED_DATA_IDENTIFIER:
        raise ContractError(
            "plans data_identifier mismatch: "
            f"{config.get('data_identifier')} != {EXPECTED_DATA_IDENTIFIER}"
        )
    if config.get("preprocessor_name") != "DefaultPreprocessor":
        raise ContractError("plans preprocessor_name mismatch")
    return {
        "dataset_name": DATASET_FOLDER,
        "plans_name": PLANS_NAME,
        "channel_keys": list(EXPECTED_CHANNEL_KEYS),
        "data_identifier": EXPECTED_DATA_IDENTIFIER,
        "experiment_planner_used": "ExperimentPlanner",
        "preprocessor_name": "DefaultPreprocessor",
        "image_reader_writer": image_reader_writer,
        **normalization,
    }


def validate_fingerprint(fingerprint: dict) -> dict[str, Any]:
    spacings = fingerprint.get("spacings")
    shapes = fingerprint.get("shapes_after_crop")
    intensity = fingerprint.get("foreground_intensity_properties_per_channel")
    if not isinstance(spacings, list) or len(spacings) != EXPECTED_TRAINING_CASES:
        raise ContractError("fingerprint must contain spacings for exactly 597 cases")
    if not isinstance(shapes, list) or len(shapes) != EXPECTED_TRAINING_CASES:
        raise ContractError("fingerprint must contain shapes for exactly 597 cases")
    if not isinstance(intensity, dict) or sorted(intensity) != EXPECTED_CHANNEL_KEYS:
        raise ContractError(
            "fingerprint channel contract mismatch: expected exactly channels 0/1"
        )
    if "median_relative_size_after_cropping" not in fingerprint:
        raise ContractError("fingerprint is missing median_relative_size_after_cropping")
    return {
        "case_count": EXPECTED_TRAINING_CASES,
        "channel_keys": list(EXPECTED_CHANNEL_KEYS),
    }


def validate_autopetv_reference_plan(plans: dict) -> dict[str, Any]:
    if plans.get("dataset_name") != "Dataset998_AutoPETV":
        raise ContractError("official autoPET V reference dataset_name mismatch")
    if plans.get("plans_name") != PLANS_NAME:
        raise ContractError("official autoPET V reference plans_name mismatch")
    intensity = plans.get("foreground_intensity_properties_per_channel")
    if not isinstance(intensity, dict) or sorted(intensity) != ["0", "1", "2", "3"]:
        raise ContractError("official autoPET V reference must contain four channels")
    try:
        config = plans["configurations"]["3d_fullres"]
    except (KeyError, TypeError) as exc:
        raise ContractError("official autoPET V reference lacks 3d_fullres") from exc
    expected_schemes = [
        "CTNormalization",
        "ZScoreNormalization",
        "ZScoreNormalization",
        "ZScoreNormalization",
    ]
    expected_masks = [False, False, False, False]
    if config.get("normalization_schemes") != expected_schemes:
        raise ContractError("official autoPET V normalization reference changed")
    if config.get("use_mask_for_norm") != expected_masks:
        raise ContractError("official autoPET V mask-for-normalization reference changed")
    return {
        "dataset_name": "Dataset998_AutoPETV",
        "configuration": "3d_fullres",
        "channel_count": 4,
        "normalization_schemes": expected_schemes,
        "use_mask_for_norm": expected_masks,
    }


def validate_autopetv_reference_dataset(dataset: dict) -> dict[str, Any]:
    expected_channels = {"0": "CT", "1": "PET", "2": "FG", "3": "BG"}
    if dataset.get("channel_names") != expected_channels:
        raise ContractError("official autoPET V channel-role contract changed")
    if dataset.get("labels") != EXPECTED_LABELS:
        raise ContractError("official autoPET V label contract changed")
    if dataset.get("numTraining") != 1611:
        raise ContractError("official autoPET V training-case count changed")
    if dataset.get("file_ending") != EXPECTED_FILE_ENDING:
        raise ContractError("official autoPET V file ending changed")
    if dataset.get("name") != "AutoPETV":
        raise ContractError("official autoPET V dataset name changed")
    return {
        "dataset_name": "AutoPETV",
        "channel_names": expected_channels,
        "labels": dict(EXPECTED_LABELS),
        "num_training": 1611,
        "expected_upstream_commit": EXPECTED_AUTOPETV_COMMIT,
        "git_commit_verified_on_server": False,
    }


def capture_runtime_identity(
    nnunet_source: Path,
    env_receipt_path: Path,
    python_executable: Path,
    output_path: Path,
) -> dict[str, Any]:
    import nnunetv2
    import torch

    source = nnunet_source.resolve()
    package_import = Path(nnunetv2.__file__).resolve()
    tree_manifest = source_tree_manifest(source)
    payload = {
        "status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "python": sys.version,
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "site_packages": site.getsitepackages(),
        "pythonpath": list(sys.path),
        "nnunetv2_version": importlib_metadata.version("nnunetv2"),
        "nnunetv2_import": str(package_import),
        "torch_version": torch.__version__,
        "torch_import": str(Path(torch.__file__).resolve()),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "expected_upstream_commit": EXPECTED_NNUNET_COMMIT,
        "provenance": "content-manifest-no-git-required-on-server",
        "git_commit_verified_on_server": False,
        "nnunet_source_root": str(source),
        "nnunet_source_file_count": len(tree_manifest),
        "nnunet_source_tree_sha256": source_tree_sha256(source),
        "nnunet_source_manifest": tree_manifest,
    }
    env_receipt = json.loads(env_receipt_path.read_text(encoding="utf-8"))
    payload["validation"] = validate_runtime_identity(
        payload, env_receipt, source, python_executable
    )
    _write_json_exclusive(output_path, payload)
    return payload


def validate_runtime_identity(
    runtime: dict,
    env_receipt: dict,
    nnunet_source: Path,
    python_executable: Path,
) -> dict[str, Any]:
    source = nnunet_source.resolve()
    expected_prefix = python_executable.resolve().parent.parent
    if runtime.get("status") != "PASS":
        raise ContractError("runtime identity is not PASS")
    if runtime.get("nnunetv2_version") != EXPECTED_NNUNET_VERSION:
        raise ContractError("live nnU-Net version mismatch")
    if Path(runtime.get("sys_prefix", "")).resolve() != expected_prefix:
        raise ContractError("live Python prefix does not own the planner executable")
    imported = Path(runtime.get("nnunetv2_import", "")).resolve()
    if not imported.is_relative_to(source):
        raise ContractError("live nnU-Net import is outside the pinned source tree")
    if runtime.get("cuda_available") is not True:
        raise ContractError("live PET/CT environment has no CUDA")
    torch_import = Path(runtime.get("torch_import", "")).resolve()
    if not torch_import.is_relative_to(expected_prefix):
        raise ContractError("live torch import is outside the isolated PET/CT environment")
    forbidden_prefixes = (
        "/mnt/HDD4/workspace/honor_degree/conda_envs/rl_nnunet",
        "/mnt/HDD4/workspace/honor_degree/Honor_codes/experiments/rl_nnunet",
    )
    for entry in runtime.get("pythonpath", []):
        if entry and any(str(entry).startswith(prefix) for prefix in forbidden_prefixes):
            raise ContractError("IAC path leaked into live runtime sys.path")
    if env_receipt.get("nnunetv2") != EXPECTED_NNUNET_VERSION:
        raise ContractError("environment receipt nnU-Net version mismatch")
    if env_receipt.get("expected_nnunet_commit") != EXPECTED_NNUNET_COMMIT:
        raise ContractError("environment receipt expected commit mismatch")
    if Path(env_receipt.get("conda_prefix", "")).resolve() != expected_prefix:
        raise ContractError("environment receipt prefix mismatch")
    receipt_import = Path(env_receipt.get("nnunetv2_import", "")).resolve()
    if receipt_import != imported or not receipt_import.is_relative_to(source):
        raise ContractError("environment receipt import path mismatch")
    observed_count = runtime.get("nnunet_source_file_count")
    observed_tree = runtime.get("nnunet_source_tree_sha256")
    if observed_count != EXPECTED_NNUNET_SOURCE_FILE_COUNT:
        raise ContractError("nnU-Net source manifest file count mismatch")
    if observed_tree != EXPECTED_NNUNET_SOURCE_TREE_SHA256:
        raise ContractError("nnU-Net source tree hash mismatch")
    if source_tree_sha256(source) != EXPECTED_NNUNET_SOURCE_TREE_SHA256:
        raise ContractError("nnU-Net source tree changed after runtime capture")
    return {
        "nnunetv2_version": EXPECTED_NNUNET_VERSION,
        "expected_upstream_commit": EXPECTED_NNUNET_COMMIT,
        "git_commit_verified_on_server": False,
        "source_file_count": observed_count,
        "source_tree_sha256": observed_tree,
        "conda_prefix": str(expected_prefix),
        "import_path": str(imported),
    }


def resolve_audit_complete(pointer_path: Path, audits_root: Path) -> Path:
    if not pointer_path.is_file():
        raise ContractError(f"audit pointer is missing: {pointer_path}")
    fields: dict[str, str] = {}
    for line in pointer_path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            raise ContractError("audit pointer has malformed content")
        key, value = line.split("=", 1)
        if key in fields or key not in {"run_dir", "completed_at"} or not value:
            raise ContractError("audit pointer has unexpected or duplicate fields")
        fields[key] = value
    if set(fields) != {"run_dir", "completed_at"}:
        raise ContractError("audit pointer must bind run_dir and completed_at")
    root = audits_root.resolve()
    run_dir = Path(fields["run_dir"]).resolve()
    if run_dir.parent != root or not run_dir.name.startswith("psma_v3_nifti_audit_"):
        raise ContractError("audit pointer escapes or does not identify a full audit run")
    completion = run_dir / "AUDIT_COMPLETE.json"
    if not completion.is_file():
        raise ContractError("audit pointer target has no AUDIT_COMPLETE.json")
    return completion


def _validate_extraction_manifest(
    manifest_path: Path,
    source_dataset_root: Path,
    migration_receipt_path: Path,
    *,
    verify_all_files: bool = True,
) -> dict[str, Any]:
    if _sha256(manifest_path) != EXPECTED_EXTRACTION_MANIFEST_SHA256:
        raise ContractError("extraction manifest content hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise ContractError("extraction manifest is not PASS")
    archive = manifest.get("archive", {})
    if archive.get("bytes") != EXPECTED_ARCHIVE_BYTES:
        raise ContractError("extraction manifest archive byte count mismatch")
    if str(archive.get("md5", "")).lower() != EXPECTED_ARCHIVE_MD5:
        raise ContractError("extraction manifest archive MD5 mismatch")
    if str(archive.get("sha256", "")).lower() != EXPECTED_ARCHIVE_SHA256:
        raise ContractError("extraction manifest archive SHA-256 mismatch")
    if manifest.get("file_count") != EXPECTED_EXTRACTED_FILE_COUNT:
        raise ContractError("extraction manifest file count mismatch")
    expected = manifest.get("contract_expected", {})
    if expected.get("triplet_count") != EXPECTED_TRAINING_CASES:
        raise ContractError("extraction manifest triplet count mismatch")
    if expected.get("dataset_root") != source_dataset_root.name:
        raise ContractError("extraction manifest dataset root mismatch")
    historical_destination = str(Path(manifest.get("destination", "")).resolve())
    if historical_destination != EXPECTED_PRE_MIGRATION_EXTRACT_ROOT:
        raise ContractError("extraction manifest historical destination mismatch")
    if _sha256(migration_receipt_path) != EXPECTED_MIGRATION_RECEIPT_SHA256:
        raise ContractError("project migration receipt hash mismatch")
    migration = json.loads(migration_receipt_path.read_text(encoding="utf-8"))
    project_root = source_dataset_root.parents[3].resolve()
    expected_current_destination = project_root / "data/PSMA_v3/extracted"
    if migration.get("status") != "PASS":
        raise ContractError("project migration receipt is not PASS")
    if Path(migration.get("new_project_root", "")).resolve() != project_root:
        raise ContractError("project migration receipt new root mismatch")
    if migration.get("moved", {}).get("data") != EXPECTED_MIGRATION_DATA_MAPPING:
        raise ContractError("project migration data mapping mismatch")
    if source_dataset_root.resolve().parent != expected_current_destination:
        raise ContractError("current extraction root does not match migration target")
    destination = source_dataset_root.resolve().parent
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != EXPECTED_EXTRACTED_FILE_COUNT:
        raise ContractError("extraction manifest file list mismatch")
    by_path: dict[str, dict] = {}
    for entry in files:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path, str) or path in by_path:
            raise ContractError("extraction manifest contains invalid/duplicate paths")
        by_path[path] = entry
    for name in ("dataset.json", "splits_final.json", "psma_metadata.csv"):
        relative = f"{source_dataset_root.name}/{name}"
        entry = by_path.get(relative)
        if entry is None:
            raise ContractError(f"extraction manifest lacks {relative}")
        current = source_dataset_root / name
        if entry.get("bytes") != current.stat().st_size or entry.get("sha256") != _sha256(current):
            raise ContractError(f"extracted metadata changed: {relative}")
    if verify_all_files:
        expected_paths = set(by_path)
        actual_paths: set[str] = set()
        for path in destination.rglob("*"):
            if path == manifest_path:
                continue
            if path.is_symlink():
                raise ContractError(f"symlink appeared in extracted source: {path}")
            if path.is_file():
                actual_paths.add(path.relative_to(destination).as_posix())
        if actual_paths != expected_paths:
            missing = sorted(expected_paths - actual_paths)[:5]
            extra = sorted(actual_paths - expected_paths)[:5]
            raise ContractError(
                f"extracted source path set changed: missing={missing}, extra={extra}"
            )
        for relative in sorted(expected_paths):
            pure = Path(relative)
            if pure.is_absolute() or ".." in pure.parts:
                raise ContractError(f"unsafe path in extraction manifest: {relative}")
            current = (destination / pure).resolve()
            if not current.is_relative_to(destination) or not current.is_file():
                raise ContractError(f"unsafe/missing extracted file: {relative}")
            record = by_path[relative]
            if current.stat().st_size != record.get("bytes"):
                raise ContractError(f"extracted file byte-size changed: {relative}")
            if _sha256(current) != record.get("sha256"):
                raise ContractError(f"extracted file hash changed: {relative}")
    return {
        "archive_sha256": archive.get("sha256"),
        "archive_md5": EXPECTED_ARCHIVE_MD5,
        "archive_bytes": EXPECTED_ARCHIVE_BYTES,
        "file_count": EXPECTED_EXTRACTED_FILE_COUNT,
        "triplet_count": EXPECTED_TRAINING_CASES,
        "all_file_hashes_reverified": verify_all_files,
        "historical_destination": historical_destination,
        "current_destination": str(destination),
        "migration_receipt_sha256": EXPECTED_MIGRATION_RECEIPT_SHA256,
        "manifest_sha256": EXPECTED_EXTRACTION_MANIFEST_SHA256,
    }


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_stat_snapshot(
    extraction_manifest_path: Path, source_dataset_root: Path
) -> list[dict[str, Any]]:
    manifest = json.loads(extraction_manifest_path.read_text(encoding="utf-8"))
    destination = source_dataset_root.resolve().parent
    snapshot: list[dict[str, Any]] = []
    for entry in sorted(manifest.get("files", []), key=lambda item: item["path"]):
        relative = Path(entry["path"])
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination) or not path.is_file() or path.is_symlink():
            raise ContractError(f"unsafe source file while snapshotting: {relative}")
        status = path.stat()
        snapshot.append(
            {
                "path": relative.as_posix(),
                "device": status.st_dev,
                "inode": status.st_ino,
                "bytes": status.st_size,
                "modified_ns": status.st_mtime_ns,
                "changed_ns": status.st_ctime_ns,
            }
        )
    if len(snapshot) != EXPECTED_EXTRACTED_FILE_COUNT:
        raise ContractError("source stat snapshot file count mismatch")
    return snapshot


def _validate_source_stat_snapshot(
    expected_snapshot: list[dict[str, Any]],
    extraction_manifest_path: Path,
    source_dataset_root: Path,
) -> str:
    current = _source_stat_snapshot(extraction_manifest_path, source_dataset_root)
    if current != expected_snapshot:
        raise ContractError("PSMA source changed between preflight and postflight")
    return _canonical_json_sha256(current)


def _normalized_snapshot(path: Path) -> tuple[int, str]:
    lines = sorted(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
    return len(lines), digest


def _validate_audit(
    completion_path: Path, source_dataset_root: Path
) -> tuple[dict[str, Any], Path, Path]:
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "COMMITTED" or completion.get("audit_status") != "PASS":
        raise ContractError("full NIfTI audit is not a committed PASS")
    if completion.get("audit_version") != EXPECTED_AUDIT_VERSION:
        raise ContractError("full NIfTI audit version mismatch")
    if completion.get("tool_sha256") != EXPECTED_AUDIT_TOOL_SHA256:
        raise ContractError("full NIfTI audit tool hash mismatch")
    outputs = completion.get("outputs", {})
    report_path = completion_path.parent / "psma_v3_nifti_audit.json"
    csv_path = completion_path.parent / "psma_v3_case_audit.csv"
    for path in (report_path, csv_path):
        record = outputs.get(path.name)
        if not isinstance(record, dict):
            raise ContractError(f"audit completion lacks {path.name}")
        _verify_record(path, record, label=f"audit output {path.name}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("audit_version") != EXPECTED_AUDIT_VERSION:
        raise ContractError("full NIfTI audit report identity mismatch")
    if Path(report.get("dataset_root", "")).resolve() != source_dataset_root.resolve():
        raise ContractError("full NIfTI audit dataset root mismatch")
    source_hashes = report.get("source_hashes", {})
    source_files = {
        "dataset_json_sha256": source_dataset_root / "dataset.json",
        "splits_final_sha256": source_dataset_root / "splits_final.json",
        "psma_metadata_sha256": source_dataset_root / "psma_metadata.csv",
    }
    for key, path in source_files.items():
        if source_hashes.get(key) != _sha256(path):
            raise ContractError(f"source metadata changed since full audit: {path.name}")
    summary = report.get("summary", {})
    expected_summary = {
        "case_count": EXPECTED_TRAINING_CASES,
        "patient_count": EXPECTED_PATIENTS,
        "positive_label_count": EXPECTED_POSITIVE_CASES,
        "empty_label_count": EXPECTED_EMPTY_CASES,
        "unreadable_label_count": 0,
        "invalid_label_count": 0,
        "failed_case_count": 0,
        "connectivity": 18,
    }
    for key, expected_value in expected_summary.items():
        if summary.get(key) != expected_value:
            raise ContractError(f"full NIfTI audit summary mismatch: {key}")
    split = report.get("split_audit", {})
    zero_fields = (
        "patient_overlap_total",
        "case_overlap_total",
        "incomplete_coverage_folds",
        "train_not_val_complement_folds",
        "val_cases_not_exactly_once",
        "patients_in_multiple_val_folds",
        "unknown_case_count",
    )
    if split.get("fold_count") != 5 or split.get("fold_count_matches") is not True:
        raise ContractError("full NIfTI audit split fold count mismatch")
    if any(split.get(field) != 0 for field in zero_fields):
        raise ContractError("full NIfTI audit split isolation mismatch")
    return (
        {
            **expected_summary,
            "fold_count": 5,
            "split_isolation": "PASS",
        },
        report_path,
        csv_path,
    )


def write_derived_dataset(
    source_path: Path, target_path: Path, receipt_path: Path | None = None
) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    derived = build_derived_dataset_json(source)
    _write_json_exclusive(target_path, derived)
    if receipt_path is None:
        return {
            "status": "PASS",
            "contract_version": CONTRACT_VERSION,
            **validate_derived_dataset_json(source, derived),
        }
    return validate_derived_dataset_file(source_path, target_path, receipt_path)


def validate_derived_dataset_file(
    source_path: Path, target_path: Path, receipt_path: Path
) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    derived = json.loads(target_path.read_text(encoding="utf-8"))
    validation = validate_derived_dataset_json(source, derived)
    receipt = {
        "status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "source_dataset_json": str(source_path.resolve()),
        "source_dataset_json_sha256": _sha256(source_path),
        "derived_dataset_json": str(target_path.resolve()),
        "derived_dataset_json_sha256": _sha256(target_path),
        **validation,
        "source_access_intent": "read-only",
        "source_files_modified_by_run": False,
        "filesystem_immutability_claimed": False,
    }
    _write_json_exclusive(receipt_path, receipt)
    return receipt


def validate_plans_file(plans_path: Path, receipt_path: Path) -> dict:
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    validation = validate_plan_contract(plans)
    receipt = {
        "status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "plans": str(plans_path.resolve()),
        "plans_sha256": _sha256(plans_path),
        **validation,
    }
    _write_json_exclusive(receipt_path, receipt)
    return receipt


def _relative_file_record(run_root: Path, path: Path) -> dict[str, Any]:
    resolved_root = run_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ContractError(f"run artifact escapes staging root: {path}")
    if os.name == "posix":
        descriptor = os.open(resolved, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    relative = resolved.relative_to(resolved_root).as_posix()
    return _file_record(resolved, display_path=relative)


def _validate_env_evidence(env_receipt_path: Path) -> tuple[dict, dict[str, Path]]:
    env = json.loads(env_receipt_path.read_text(encoding="utf-8"))
    base = env_receipt_path.parent
    marker_path = base / "ENV_READY.done"
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ContractError("missing atomic environment completion marker")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        marker.get("schema_version") != "PETCT-NNUNET-ENV-MARKER-v1.0"
        or marker.get("status") != "ENVIRONMENT_EVIDENCE_COMPLETE"
    ):
        raise ContractError("environment marker is not complete")
    bundle_path = Path(str(marker.get("bundle_path") or "")).resolve()
    bundle_root = (base / "evidence-bundles").resolve()
    if (
        bundle_path.is_symlink()
        or not bundle_path.is_dir()
        or not bundle_path.is_relative_to(bundle_root)
    ):
        raise ContractError("environment marker points outside evidence-bundles")
    bundle_manifest = bundle_path / "bundle.json"
    if bundle_manifest.is_symlink() or not bundle_manifest.is_file():
        raise ContractError("environment evidence bundle manifest is missing")
    if marker.get("bundle_manifest_sha256") != _sha256(bundle_manifest):
        raise ContractError("environment bundle manifest hash mismatch")
    bundle = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    unsigned_bundle = {
        key: value for key, value in bundle.items() if key != "bundle_sha256"
    }
    if (
        bundle.get("schema_version")
        != "PETCT-NNUNET-ENV-EVIDENCE-BUNDLE-v1.0"
        or bundle.get("status") != "ENVIRONMENT_EVIDENCE_COMPLETE"
        or bundle.get("bundle_sha256") != _canonical_json_sha256(unsigned_bundle)
        or marker.get("bundle_sha256") != bundle.get("bundle_sha256")
    ):
        raise ContractError("environment evidence bundle contract is invalid")
    bundle_records = bundle.get("files")
    if not isinstance(bundle_records, list) or not bundle_records:
        raise ContractError("environment evidence bundle file inventory is missing")
    bundle_by_name = {}
    for record in bundle_records:
        if not isinstance(record, Mapping):
            raise ContractError("environment evidence bundle record is invalid")
        name = str(record.get("name") or "")
        path = bundle_path / name
        if (
            not name
            or name in bundle_by_name
            or Path(name).name != name
            or path.is_symlink()
            or not path.is_file()
            or record.get("sha256") != _sha256(path)
            or record.get("bytes") != path.stat().st_size
        ):
            raise ContractError("environment evidence bundle file changed: %s" % name)
        bundle_by_name[name] = path
    files = {
        "environment_receipt": env_receipt_path,
        "conda_explicit": base / "petct_nnunet_v281.conda-explicit.txt",
        "pip_freeze": base / "petct_nnunet_v281.pip-freeze.txt",
        "environment_hash_receipt": base / "petct_nnunet_v281.receipt-sha256.txt",
        "forbidden_symlinks": base / "petct_nnunet_v281.forbidden-symlinks.txt",
        "completion_marker": marker_path,
        "bundle_manifest": bundle_manifest,
    }
    for label, path in files.items():
        if not path.is_file():
            raise ContractError(f"missing environment evidence {label}: {path}")
    if files["forbidden_symlinks"].stat().st_size != 0:
        raise ContractError("IAC symlink evidence is not empty")
    expected_hashes: dict[str, str] = {}
    for line in files["environment_hash_receipt"].read_text(encoding="utf-8").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2:
            raise ContractError("environment hash receipt is malformed")
        expected_hashes[Path(fields[1].strip()).name] = fields[0].lower()
    for key in ("environment_receipt", "conda_explicit", "pip_freeze"):
        path = files[key]
        if expected_hashes.get(path.name) != _sha256(path):
            raise ContractError(f"environment evidence hash mismatch: {path.name}")
        bundle_copy = bundle_by_name.get(path.name)
        if bundle_copy is None or _sha256(bundle_copy) != _sha256(path):
            raise ContractError(f"environment compatibility file differs from bundle: {path.name}")
    marker_receipt = Path(str(marker.get("receipt_path") or "")).resolve()
    bundled_receipt = bundle_by_name.get("petct_nnunet_v281.json")
    if (
        bundled_receipt is None
        or marker_receipt != bundled_receipt.resolve()
        or marker.get("receipt_sha256") != _sha256(bundled_receipt)
    ):
        raise ContractError("environment marker receipt binding is invalid")
    return env, files


def write_preflight_receipt(
    *,
    source_dataset_root: Path,
    extraction_manifest_path: Path,
    migration_receipt_path: Path,
    audit_pointer_path: Path,
    audits_root: Path,
    env_receipt_path: Path,
    live_conda_snapshot_path: Path,
    live_pip_snapshot_path: Path,
    nnunet_source: Path,
    autopetv_plans_path: Path,
    autopetv_dataset_path: Path,
    audit_tool_path: Path,
    python_executable: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate all code/data/environment evidence before importing nnU-Net."""

    source_dataset_root = source_dataset_root.resolve()
    nnunet_source = nnunet_source.resolve()
    python_executable = python_executable.resolve()
    expected_prefix = python_executable.parent.parent
    if Path(sys.prefix).resolve() != expected_prefix:
        raise ContractError("preflight Python prefix mismatch")
    if _sha256(python_executable) != EXPECTED_PYTHON_EXECUTABLE_SHA256:
        raise ContractError("preflight Python executable hash mismatch")
    forbidden_prefixes = (
        "/mnt/HDD4/workspace/honor_degree/conda_envs/rl_nnunet",
        "/mnt/HDD4/workspace/honor_degree/Honor_codes/experiments/rl_nnunet",
    )
    for entry in sys.path:
        if entry and any(str(entry).startswith(prefix) for prefix in forbidden_prefixes):
            raise ContractError("IAC path leaked into live PET/CT Python sys.path")
    if importlib_metadata.version("nnunetv2") != EXPECTED_NNUNET_VERSION:
        raise ContractError("live nnU-Net distribution version mismatch before import")
    nnunet_spec = importlib.util.find_spec("nnunetv2")
    if nnunet_spec is None or nnunet_spec.origin is None:
        raise ContractError("live nnU-Net module cannot be resolved before import")
    if not Path(nnunet_spec.origin).resolve().is_relative_to(nnunet_source):
        raise ContractError("live nnU-Net resolution is outside pinned source before import")
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or torch_spec.origin is None:
        raise ContractError("live torch module cannot be resolved before import")
    if not Path(torch_spec.origin).resolve().is_relative_to(expected_prefix):
        raise ContractError("live torch resolution is outside isolated environment before import")

    tree_manifest = source_tree_manifest(nnunet_source)
    tree_hash = source_tree_sha256(nnunet_source)
    if len(tree_manifest) != EXPECTED_NNUNET_SOURCE_FILE_COUNT:
        raise ContractError("nnU-Net source file count mismatch before import")
    if tree_hash != EXPECTED_NNUNET_SOURCE_TREE_SHA256:
        raise ContractError("nnU-Net source hash mismatch before import")

    if _sha256(autopetv_plans_path) != EXPECTED_AUTOPETV_PLANS_SHA256:
        raise ContractError("official autoPET V plans hash mismatch before planning")
    if _sha256(autopetv_dataset_path) != EXPECTED_AUTOPETV_DATASET_SHA256:
        raise ContractError("official autoPET V dataset hash mismatch before planning")
    official_plan = validate_autopetv_reference_plan(
        json.loads(autopetv_plans_path.read_text(encoding="utf-8"))
    )
    official_dataset = validate_autopetv_reference_dataset(
        json.loads(autopetv_dataset_path.read_text(encoding="utf-8"))
    )
    if _sha256(audit_tool_path) != EXPECTED_AUDIT_TOOL_SHA256:
        raise ContractError("deployed full-audit tool hash mismatch before planning")

    before_snapshot = _source_stat_snapshot(
        extraction_manifest_path, source_dataset_root
    )
    extraction_contract = _validate_extraction_manifest(
        extraction_manifest_path,
        source_dataset_root,
        migration_receipt_path,
        verify_all_files=True,
    )
    after_snapshot = _source_stat_snapshot(
        extraction_manifest_path, source_dataset_root
    )
    if before_snapshot != after_snapshot:
        raise ContractError("PSMA source changed during full preflight hash verification")
    snapshot_hash = _canonical_json_sha256(after_snapshot)

    completion_path = resolve_audit_complete(audit_pointer_path, audits_root)
    audit_contract, audit_report_path, audit_csv_path = _validate_audit(
        completion_path, source_dataset_root
    )
    env_receipt, env_files = _validate_env_evidence(env_receipt_path)
    if env_receipt.get("nnunetv2") != EXPECTED_NNUNET_VERSION:
        raise ContractError("environment receipt nnU-Net version mismatch")
    if env_receipt.get("expected_nnunet_commit") != EXPECTED_NNUNET_COMMIT:
        raise ContractError("environment receipt expected commit mismatch")
    if Path(env_receipt.get("conda_prefix", "")).resolve() != expected_prefix:
        raise ContractError("environment receipt prefix mismatch")
    receipt_import = Path(env_receipt.get("nnunetv2_import", "")).resolve()
    if not receipt_import.is_relative_to(nnunet_source):
        raise ContractError("environment receipt import path is outside pinned source")
    if importlib_metadata.version("torch") != env_receipt.get("torch"):
        raise ContractError("live torch distribution differs from environment receipt")

    live_conda_count, live_conda_hash = _normalized_snapshot(live_conda_snapshot_path)
    saved_conda_count, saved_conda_hash = _normalized_snapshot(env_files["conda_explicit"])
    live_pip_count, live_pip_hash = _normalized_snapshot(live_pip_snapshot_path)
    saved_pip_count, saved_pip_hash = _normalized_snapshot(env_files["pip_freeze"])
    if not (
        live_conda_hash
        == saved_conda_hash
        == EXPECTED_CONDA_EXPLICIT_NORMALIZED_SHA256
        and live_conda_count == saved_conda_count
    ):
        raise ContractError("live Conda package snapshot differs from frozen evidence")
    if not (
        live_pip_hash
        == saved_pip_hash
        == EXPECTED_PIP_FREEZE_NORMALIZED_SHA256
        and live_pip_count == saved_pip_count
    ):
        raise ContractError("live pip package snapshot differs from frozen evidence")

    evidence_paths: dict[str, Path] = {
        "source_dataset_json": source_dataset_root / "dataset.json",
        "source_splits": source_dataset_root / "splits_final.json",
        "source_metadata": source_dataset_root / "psma_metadata.csv",
        "extraction_manifest": extraction_manifest_path,
        "migration_receipt": migration_receipt_path,
        "audit_pointer": audit_pointer_path,
        "audit_completion": completion_path,
        "audit_report": audit_report_path,
        "audit_csv": audit_csv_path,
        "audit_tool": audit_tool_path,
        "planning_gate_tool": Path(__file__).resolve(),
        "autopetv_official_plans": autopetv_plans_path,
        "autopetv_official_dataset": autopetv_dataset_path,
        "python_executable": python_executable,
        **env_files,
    }
    receipt = {
        "status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "phase": "PRE_IMPORT_PREFLIGHT",
        "nnunet_import_authorized": True,
        "planning_authorized": True,
        "dataset": {
            "id": DATASET_ID,
            "folder": DATASET_FOLDER,
            "source_release": "PSMA-PET-CT-Lesions_v3",
            "scope": "PSMA v3 only",
        },
        "interaction_scope": "scribbles-only downstream",
        "nnunet_source": {
            "expected_upstream_commit": EXPECTED_NNUNET_COMMIT,
            "git_commit_verified_on_server": False,
            "file_count": len(tree_manifest),
            "tree_sha256": tree_hash,
        },
        "environment": {
            "python_executable_sha256": EXPECTED_PYTHON_EXECUTABLE_SHA256,
            "nnunetv2_version": EXPECTED_NNUNET_VERSION,
            "conda_package_count": live_conda_count,
            "conda_snapshot_sha256": live_conda_hash,
            "pip_package_count": live_pip_count,
            "pip_snapshot_sha256": live_pip_hash,
            "iac_path_isolation": "PASS",
        },
        "official_autopetv_reference": {
            "dataset": official_dataset,
            "plan": official_plan,
        },
        "extraction_contract": extraction_contract,
        "audit_contract": audit_contract,
        "source_stat_snapshot_sha256": snapshot_hash,
        "source_stat_snapshot": after_snapshot,
        "evidence": {
            label: _file_record(path) for label, path in evidence_paths.items()
        },
        "artifacts": {
            "live_conda_snapshot": _relative_file_record(
                output_path.parent, live_conda_snapshot_path
            ),
            "live_pip_snapshot": _relative_file_record(
                output_path.parent, live_pip_snapshot_path
            ),
        },
    }
    _write_json_exclusive(output_path, receipt)
    return receipt


def write_planning_bundle(
    *,
    run_id: str,
    run_root: Path,
    committed_run_dir: Path,
    source_dataset_root: Path,
    derived_dataset_root: Path,
    preprocessed_dataset_root: Path,
    extraction_manifest_path: Path,
    migration_receipt_path: Path,
    audit_pointer_path: Path,
    audits_root: Path,
    env_receipt_path: Path,
    preflight_receipt_path: Path,
    runtime_identity_path: Path,
    nnunet_source: Path,
    autopetv_plans_path: Path,
    autopetv_dataset_path: Path,
    audit_tool_path: Path,
    python_executable: Path,
    run_owner_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Validate an isolated plan-only run and write its run-scoped receipt."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ContractError("unsafe planning run_id")
    run_root = run_root.resolve()
    committed_run_dir = committed_run_dir.resolve()
    source_dataset_root = source_dataset_root.resolve()
    derived_dataset_root = derived_dataset_root.resolve()
    preprocessed_dataset_root = preprocessed_dataset_root.resolve()
    python_executable = python_executable.resolve()
    if not derived_dataset_root.is_relative_to(run_root):
        raise ContractError("derived dataset is outside the isolated run")
    if not preprocessed_dataset_root.is_relative_to(run_root):
        raise ContractError("preprocessed planning output is outside the isolated run")
    if committed_run_dir.parent != run_root.parent:
        raise ContractError("staging and committed run must share one parent/filesystem")
    if run_root.stat().st_dev != committed_run_dir.parent.stat().st_dev:
        raise ContractError("staging and committed run are on different filesystems")
    expected_root_entries = {
        "nnUNet_raw",
        "nnUNet_preprocessed",
        "nnUNet_results",
        "RUN_OWNER.json",
        "live_conda_snapshot.txt",
        "live_pip_snapshot.txt",
        "PREFLIGHT.json",
        "nnunet_runtime_identity.json",
    }
    actual_root_entries = {item.name for item in run_root.iterdir()}
    if actual_root_entries != expected_root_entries:
        raise ContractError("isolated planning run root whitelist mismatch")
    if {item.name for item in derived_dataset_root.parent.iterdir()} != {DATASET_FOLDER}:
        raise ContractError("isolated nnUNet_raw contains an unexpected dataset")
    if {item.name for item in preprocessed_dataset_root.parent.iterdir()} != {
        DATASET_FOLDER
    }:
        raise ContractError("isolated nnUNet_preprocessed contains an unexpected dataset")
    results_root = run_root / "nnUNet_results"
    if not results_root.is_dir() or any(results_root.iterdir()):
        raise ContractError("training/results side effects appeared during planning")
    owner = json.loads(run_owner_path.read_text(encoding="utf-8"))
    if owner.get("status") != "OWNED" or owner.get("run_id") != run_id:
        raise ContractError("run owner record mismatch")

    source_json_path = source_dataset_root / "dataset.json"
    source_splits_path = source_dataset_root / "splits_final.json"
    source_metadata_path = source_dataset_root / "psma_metadata.csv"
    source = json.loads(source_json_path.read_text(encoding="utf-8"))
    derived_json_path = derived_dataset_root / "dataset.json"
    if derived_json_path.is_symlink() or not derived_json_path.is_file():
        raise ContractError("derived dataset.json must be a regular file")
    derived = json.loads(derived_json_path.read_text(encoding="utf-8"))
    dataset_contract = validate_derived_dataset_json(source, derived)
    expected_raw_entries = {"dataset.json", "imagesTr", "labelsTr"}
    if {item.name for item in derived_dataset_root.iterdir()} != expected_raw_entries:
        raise ContractError("derived raw dataset contains unexpected entries")
    for name in ("imagesTr", "labelsTr"):
        link = derived_dataset_root / name
        if not link.is_symlink():
            raise ContractError(f"derived raw {name} must be a directory symlink")
        if link.resolve() != (source_dataset_root / name).resolve():
            raise ContractError(f"derived raw {name} symlink target mismatch")

    preprocessed_json_path = preprocessed_dataset_root / "dataset.json"
    fingerprint_path = preprocessed_dataset_root / "dataset_fingerprint.json"
    plans_path = preprocessed_dataset_root / "nnUNetPlans.json"
    preprocessed_splits_path = preprocessed_dataset_root / "splits_final.json"
    expected_planning_files = {
        "dataset.json",
        "dataset_fingerprint.json",
        "nnUNetPlans.json",
        "splits_final.json",
    }
    actual_planning_entries = {item.name for item in preprocessed_dataset_root.iterdir()}
    if actual_planning_entries != expected_planning_files:
        raise ContractError(
            "plan-only output whitelist mismatch: "
            f"{sorted(actual_planning_entries)} != {sorted(expected_planning_files)}"
        )
    for item in preprocessed_dataset_root.iterdir():
        if item.is_symlink() or not item.is_file():
            raise ContractError("plan-only outputs must be regular files, never symlinks/directories")
    preprocessed_dataset = json.loads(preprocessed_json_path.read_text(encoding="utf-8"))
    validate_derived_dataset_json(source, preprocessed_dataset)
    if _sha256(preprocessed_json_path) != _sha256(derived_json_path):
        raise ContractError("preprocessed dataset.json is not byte-identical to derived metadata")
    if _sha256(preprocessed_splits_path) != _sha256(source_splits_path):
        raise ContractError("planning split copy differs from audited source split")
    fingerprint_contract = validate_fingerprint(
        json.loads(fingerprint_path.read_text(encoding="utf-8"))
    )
    plan_contract = validate_plan_contract(
        json.loads(plans_path.read_text(encoding="utf-8"))
    )

    if _sha256(autopetv_plans_path) != EXPECTED_AUTOPETV_PLANS_SHA256:
        raise ContractError("official autoPET V reference plan hash mismatch")
    if _sha256(autopetv_dataset_path) != EXPECTED_AUTOPETV_DATASET_SHA256:
        raise ContractError("official autoPET V reference dataset hash mismatch")
    autopetv_plan_contract = validate_autopetv_reference_plan(
        json.loads(autopetv_plans_path.read_text(encoding="utf-8"))
    )
    autopetv_dataset_contract = validate_autopetv_reference_dataset(
        json.loads(autopetv_dataset_path.read_text(encoding="utf-8"))
    )
    if _sha256(audit_tool_path) != EXPECTED_AUDIT_TOOL_SHA256:
        raise ContractError("deployed full-audit tool differs from the approved tool")
    extraction_contract = _validate_extraction_manifest(
        extraction_manifest_path,
        source_dataset_root,
        migration_receipt_path,
        verify_all_files=False,
    )
    completion_path = resolve_audit_complete(audit_pointer_path, audits_root)
    audit_contract, audit_report_path, audit_csv_path = _validate_audit(
        completion_path, source_dataset_root
    )
    env_receipt, env_files = _validate_env_evidence(env_receipt_path)
    preflight = json.loads(preflight_receipt_path.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "PASS"
        or preflight.get("contract_version") != CONTRACT_VERSION
        or preflight.get("phase") != "PRE_IMPORT_PREFLIGHT"
        or preflight.get("planning_authorized") is not True
    ):
        raise ContractError("pre-import preflight receipt is not valid")
    source_snapshot_hash = _validate_source_stat_snapshot(
        preflight.get("source_stat_snapshot", []),
        extraction_manifest_path,
        source_dataset_root,
    )
    if source_snapshot_hash != preflight.get("source_stat_snapshot_sha256"):
        raise ContractError("preflight source snapshot hash mismatch")
    runtime = json.loads(runtime_identity_path.read_text(encoding="utf-8"))
    runtime_contract = validate_runtime_identity(
        runtime, env_receipt, nnunet_source, python_executable
    )
    if runtime.get("validation") != runtime_contract:
        raise ContractError("runtime was not validated before planning")
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        raise ContractError("pinned PET/CT Python is missing or not executable")
    if _sha256(python_executable) != EXPECTED_PYTHON_EXECUTABLE_SHA256:
        raise ContractError("pinned PET/CT Python changed after preflight")

    evidence_paths: dict[str, Path] = {
        "source_dataset_json": source_json_path,
        "source_splits": source_splits_path,
        "source_metadata": source_metadata_path,
        "extraction_manifest": extraction_manifest_path,
        "migration_receipt": migration_receipt_path,
        "audit_pointer": audit_pointer_path,
        "audit_completion": completion_path,
        "audit_report": audit_report_path,
        "audit_csv": audit_csv_path,
        "audit_tool": audit_tool_path,
        "planning_gate_tool": Path(__file__).resolve(),
        "autopetv_official_plans": autopetv_plans_path,
        "autopetv_official_dataset": autopetv_dataset_path,
        "python_executable": python_executable,
        **env_files,
    }
    evidence = {label: _file_record(path) for label, path in evidence_paths.items()}
    if preflight.get("evidence") != evidence:
        raise ContractError("preflight evidence chain changed before postflight")
    expected_preflight_artifacts = {
        "live_conda_snapshot": _relative_file_record(
            run_root, run_root / "live_conda_snapshot.txt"
        ),
        "live_pip_snapshot": _relative_file_record(
            run_root, run_root / "live_pip_snapshot.txt"
        ),
    }
    if preflight.get("artifacts") != expected_preflight_artifacts:
        raise ContractError("preflight live environment artifacts changed")
    artifacts = {
        "derived_dataset_json": _relative_file_record(run_root, derived_json_path),
        "preprocessed_dataset_json": _relative_file_record(
            run_root, preprocessed_json_path
        ),
        "dataset_fingerprint": _relative_file_record(run_root, fingerprint_path),
        "nnunet_plans": _relative_file_record(run_root, plans_path),
        "splits_final": _relative_file_record(run_root, preprocessed_splits_path),
        "runtime_identity": _relative_file_record(run_root, runtime_identity_path),
        "preflight_receipt": _relative_file_record(run_root, preflight_receipt_path),
        "run_owner": _relative_file_record(run_root, run_owner_path),
        **expected_preflight_artifacts,
    }
    _fsync_directory(derived_dataset_root)
    _fsync_directory(preprocessed_dataset_root)
    _fsync_directory(run_root)
    bundle = {
        "status": "VALIDATED",
        "planning_status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "phase": "PLANNING_ONLY",
        "preprocessing_status": "NOT_STARTED",
        "preprocessing_performed": False,
        "training_status": "NOT_STARTED",
        "training_performed": False,
        "run_id": run_id,
        "committed_run_dir": str(committed_run_dir),
        "dataset": {
            "id": DATASET_ID,
            "folder": DATASET_FOLDER,
            "metadata_name": DERIVED_DATASET_NAME,
            "source_release": "PSMA-PET-CT-Lesions_v3",
            "scope": "PSMA v3 only",
        },
        "interaction_scope": "scribbles-only downstream",
        "source_access_intent": "read-only",
        "source_files_modified_by_run": False,
        "filesystem_immutability_claimed": False,
        "planner_command": {
            "executable": str(python_executable),
            "arguments": [
                "-m",
                "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
                "-d",
                str(DATASET_ID),
                "--verify_dataset_integrity",
                "-npfp",
                "4",
                "--no_pp",
                "--clean",
            ],
        },
        "dataset_contract": dataset_contract,
        "fingerprint_contract": fingerprint_contract,
        "plan_contract": plan_contract,
        "official_autopetv_reference": {
            "dataset": autopetv_dataset_contract,
            "plan": autopetv_plan_contract,
        },
        "extraction_contract": extraction_contract,
        "audit_contract": audit_contract,
        "runtime_contract": runtime_contract,
        "preflight_contract": {
            "phase": "PRE_IMPORT_PREFLIGHT",
            "nnunet_import_authorized": True,
            "source_stat_snapshot_sha256": source_snapshot_hash,
            "all_source_file_hashes_reverified_before_import": True,
        },
        "evidence": evidence,
        "artifacts": artifacts,
    }
    _write_json_exclusive(receipt_path, bundle)
    return bundle


def _verify_record(path: Path, record: dict, *, label: str) -> None:
    expected = record.get("sha256")
    if not isinstance(expected, str):
        raise ContractError(f"{label} record has no sha256")
    actual = _sha256(path)
    if actual != expected:
        raise ContractError(f"{label} hash mismatch: {actual} != {expected}")
    if "bytes" in record and path.stat().st_size != record["bytes"]:
        raise ContractError(f"{label} byte-size mismatch")


def publish_planning_ready(
    run_dir: Path, run_receipt_path: Path, ready_receipt_path: Path
) -> dict[str, Any]:
    """Publish the one fixed planning-ready receipt after an atomic run rename."""

    run_dir = run_dir.resolve()
    run_receipt_path = run_receipt_path.resolve()
    if not run_receipt_path.is_relative_to(run_dir):
        raise ContractError("run receipt must be inside the committed run directory")
    receipt_bytes = run_receipt_path.read_bytes()
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    bundle = json.loads(receipt_bytes)
    required_top_level = {
        "status": "VALIDATED",
        "planning_status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "phase": "PLANNING_ONLY",
        "preprocessing_status": "NOT_STARTED",
        "preprocessing_performed": False,
        "training_status": "NOT_STARTED",
        "training_performed": False,
        "interaction_scope": "scribbles-only downstream",
        "source_files_modified_by_run": False,
        "filesystem_immutability_claimed": False,
    }
    for key, expected in required_top_level.items():
        if bundle.get(key) != expected:
            raise ContractError(f"run receipt critical field mismatch: {key}")
    if Path(bundle.get("committed_run_dir", "")).resolve() != run_dir:
        raise ContractError("committed_run_dir does not match the published run")
    if bundle.get("run_id") != run_dir.name:
        raise ContractError("run receipt run_id does not match the run directory")
    dataset = bundle.get("dataset", {})
    if dataset != {
        "id": DATASET_ID,
        "folder": DATASET_FOLDER,
        "metadata_name": DERIVED_DATASET_NAME,
        "source_release": "PSMA-PET-CT-Lesions_v3",
        "scope": "PSMA v3 only",
    }:
        raise ContractError("run receipt dataset identity mismatch")

    artifacts = bundle.get("artifacts")
    evidence = bundle.get("evidence")
    if not isinstance(artifacts, dict) or not isinstance(evidence, dict):
        raise ContractError("run receipt is missing artifact/evidence records")
    if set(artifacts) != EXPECTED_ARTIFACT_LABELS:
        raise ContractError("run receipt artifact allowlist mismatch")
    if set(evidence) != EXPECTED_EVIDENCE_LABELS:
        raise ContractError("run receipt evidence allowlist mismatch")

    artifact_paths: dict[str, Path] = {}
    for label, record in artifacts.items():
        relative = Path(record.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"unsafe artifact path in run receipt: {relative}")
        artifact = (run_dir / relative).resolve()
        if not artifact.is_relative_to(run_dir):
            raise ContractError(f"artifact escapes committed run: {artifact}")
        artifact_paths[label] = artifact
        _verify_record(artifact, record, label=f"artifact {label}")
    evidence_paths: dict[str, Path] = {}
    for label, record in evidence.items():
        evidence_path = Path(record.get("path", "")).resolve()
        evidence_paths[label] = evidence_path
        _verify_record(evidence_path, record, label=f"evidence {label}")

    expected_root_entries = {
        "nnUNet_raw",
        "nnUNet_preprocessed",
        "nnUNet_results",
        "RUN_OWNER.json",
        "live_conda_snapshot.txt",
        "live_pip_snapshot.txt",
        "PREFLIGHT.json",
        "nnunet_runtime_identity.json",
        "PLANNING_BUNDLE.json",
    }
    if {item.name for item in run_dir.iterdir()} != expected_root_entries:
        raise ContractError("committed planning run root whitelist mismatch")
    if any((run_dir / "nnUNet_results").iterdir()):
        raise ContractError("committed planning run contains training/results outputs")
    raw_dataset_root = run_dir / "nnUNet_raw" / DATASET_FOLDER
    preprocessed_root = run_dir / "nnUNet_preprocessed" / DATASET_FOLDER
    if {item.name for item in (run_dir / "nnUNet_raw").iterdir()} != {DATASET_FOLDER}:
        raise ContractError("committed raw root contains an unexpected dataset")
    if {item.name for item in (run_dir / "nnUNet_preprocessed").iterdir()} != {
        DATASET_FOLDER
    }:
        raise ContractError("committed preprocessed root contains an unexpected dataset")
    if {item.name for item in raw_dataset_root.iterdir()} != {
        "dataset.json",
        "imagesTr",
        "labelsTr",
    }:
        raise ContractError("committed derived raw dataset whitelist mismatch")
    if {item.name for item in preprocessed_root.iterdir()} != {
        "dataset.json",
        "dataset_fingerprint.json",
        "nnUNetPlans.json",
        "splits_final.json",
    }:
        raise ContractError("committed plan-only output whitelist mismatch")

    source_dataset_root = evidence_paths["source_dataset_json"].parent
    if evidence_paths["planning_gate_tool"] != Path(__file__).resolve():
        raise ContractError("published planning gate tool path mismatch")
    for name in ("imagesTr", "labelsTr"):
        link = raw_dataset_root / name
        if not link.is_symlink() or link.resolve() != (source_dataset_root / name).resolve():
            raise ContractError(f"published derived raw {name} link mismatch")
    for item in preprocessed_root.iterdir():
        if item.is_symlink() or not item.is_file():
            raise ContractError("published plan-only outputs must be regular files")
    source = json.loads(evidence_paths["source_dataset_json"].read_text(encoding="utf-8"))
    derived = json.loads(artifact_paths["derived_dataset_json"].read_text(encoding="utf-8"))
    preprocessed_dataset = json.loads(
        artifact_paths["preprocessed_dataset_json"].read_text(encoding="utf-8")
    )
    dataset_contract = validate_derived_dataset_json(source, derived)
    validate_derived_dataset_json(source, preprocessed_dataset)
    if _sha256(artifact_paths["derived_dataset_json"]) != _sha256(
        artifact_paths["preprocessed_dataset_json"]
    ):
        raise ContractError("published derived/preprocessed dataset metadata differs")
    if _sha256(evidence_paths["source_splits"]) != _sha256(
        artifact_paths["splits_final"]
    ):
        raise ContractError("published split copy differs from audited source")
    fingerprint_contract = validate_fingerprint(
        json.loads(artifact_paths["dataset_fingerprint"].read_text(encoding="utf-8"))
    )
    plan_contract = validate_plan_contract(
        json.loads(artifact_paths["nnunet_plans"].read_text(encoding="utf-8"))
    )
    if bundle.get("dataset_contract") != dataset_contract:
        raise ContractError("published dataset contract was rewritten")
    if bundle.get("fingerprint_contract") != fingerprint_contract:
        raise ContractError("published fingerprint contract was rewritten")
    if bundle.get("plan_contract") != plan_contract:
        raise ContractError("published plan contract was rewritten")

    if _sha256(evidence_paths["autopetv_official_plans"]) != EXPECTED_AUTOPETV_PLANS_SHA256:
        raise ContractError("published official autoPET V plans hash mismatch")
    if _sha256(evidence_paths["autopetv_official_dataset"]) != EXPECTED_AUTOPETV_DATASET_SHA256:
        raise ContractError("published official autoPET V dataset hash mismatch")
    official = {
        "dataset": validate_autopetv_reference_dataset(
            json.loads(
                evidence_paths["autopetv_official_dataset"].read_text(encoding="utf-8")
            )
        ),
        "plan": validate_autopetv_reference_plan(
            json.loads(
                evidence_paths["autopetv_official_plans"].read_text(encoding="utf-8")
            )
        ),
    }
    if bundle.get("official_autopetv_reference") != official:
        raise ContractError("published official autoPET V contract was rewritten")

    extraction_contract = _validate_extraction_manifest(
        evidence_paths["extraction_manifest"],
        source_dataset_root,
        evidence_paths["migration_receipt"],
        verify_all_files=False,
    )
    if bundle.get("extraction_contract") != extraction_contract:
        raise ContractError("published extraction contract was rewritten")
    completion = resolve_audit_complete(
        evidence_paths["audit_pointer"], evidence_paths["audit_pointer"].parent
    )
    if completion != evidence_paths["audit_completion"]:
        raise ContractError("published audit pointer target changed")
    audit_contract, report_path, csv_path = _validate_audit(
        completion, source_dataset_root
    )
    if report_path != evidence_paths["audit_report"] or csv_path != evidence_paths["audit_csv"]:
        raise ContractError("published audit output paths changed")
    if bundle.get("audit_contract") != audit_contract:
        raise ContractError("published audit contract was rewritten")

    env_receipt, _ = _validate_env_evidence(evidence_paths["environment_receipt"])
    runtime = json.loads(artifact_paths["runtime_identity"].read_text(encoding="utf-8"))
    runtime_contract = validate_runtime_identity(
        runtime,
        env_receipt,
        Path(runtime["nnunet_source_root"]),
        evidence_paths["python_executable"],
    )
    if runtime.get("validation") != runtime_contract:
        raise ContractError("published runtime lacks pre-planning validation")
    if bundle.get("runtime_contract") != runtime_contract:
        raise ContractError("published runtime contract was rewritten")
    preflight = json.loads(artifact_paths["preflight_receipt"].read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "PASS"
        or preflight.get("phase") != "PRE_IMPORT_PREFLIGHT"
        or preflight.get("evidence") != evidence
    ):
        raise ContractError("published preflight evidence is incomplete")
    if preflight.get("artifacts") != {
        "live_conda_snapshot": artifacts["live_conda_snapshot"],
        "live_pip_snapshot": artifacts["live_pip_snapshot"],
    }:
        raise ContractError("published preflight environment artifacts are incomplete")
    source_snapshot_hash = _validate_source_stat_snapshot(
        preflight.get("source_stat_snapshot", []),
        evidence_paths["extraction_manifest"],
        source_dataset_root,
    )
    if source_snapshot_hash != preflight.get("source_stat_snapshot_sha256"):
        raise ContractError("published source snapshot changed")
    owner = json.loads(artifact_paths["run_owner"].read_text(encoding="utf-8"))
    if owner.get("status") != "OWNED" or owner.get("run_id") != run_dir.name:
        raise ContractError("published run owner mismatch")

    expected_command = {
        "executable": str(evidence_paths["python_executable"]),
        "arguments": [
            "-m",
            "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
            "-d",
            str(DATASET_ID),
            "--verify_dataset_integrity",
            "-npfp",
            "4",
            "--no_pp",
            "--clean",
        ],
    }
    if bundle.get("planner_command") != expected_command:
        raise ContractError("published planner command contract mismatch")
    expected_preflight_contract = {
        "phase": "PRE_IMPORT_PREFLIGHT",
        "nnunet_import_authorized": True,
        "source_stat_snapshot_sha256": source_snapshot_hash,
        "all_source_file_hashes_reverified_before_import": True,
    }
    if bundle.get("preflight_contract") != expected_preflight_contract:
        raise ContractError("published preflight contract was rewritten")
    if _sha256(run_receipt_path) != receipt_sha256:
        raise ContractError("run receipt changed during fixed-receipt publication")

    _fsync_directory(run_dir)
    _fsync_directory(run_dir.parent)
    published = {
        "status": "COMMITTED",
        "planning_status": "PASS",
        "contract_version": CONTRACT_VERSION,
        "phase": "PLANNING_ONLY",
        "preprocessing_status": "NOT_STARTED",
        "preprocessing_performed": False,
        "training_status": "NOT_STARTED",
        "training_performed": False,
        "run_id": bundle.get("run_id"),
        "run_dir": str(run_dir),
        "run_receipt": {
            "path": str(run_receipt_path),
            "bytes": len(receipt_bytes),
            "sha256": receipt_sha256,
        },
        "validated_bundle": bundle,
    }
    _write_json_exclusive(ready_receipt_path, published)
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write-dataset-json")
    write.add_argument("source", type=Path)
    write.add_argument("target", type=Path)
    write.add_argument("--receipt", type=Path)
    validate_dataset = subparsers.add_parser("validate-dataset-json")
    validate_dataset.add_argument("source", type=Path)
    validate_dataset.add_argument("target", type=Path)
    validate_dataset.add_argument("--receipt", type=Path, required=True)
    validate = subparsers.add_parser("validate-plans")
    validate.add_argument("plans", type=Path)
    validate.add_argument("--receipt", type=Path, required=True)
    runtime = subparsers.add_parser("capture-runtime")
    runtime.add_argument("nnunet_source", type=Path)
    runtime.add_argument("--env-receipt", type=Path, required=True)
    runtime.add_argument("--python-executable", type=Path, required=True)
    runtime.add_argument("--output", type=Path, required=True)
    audit = subparsers.add_parser("resolve-audit")
    audit.add_argument("pointer", type=Path)
    audit.add_argument("audits_root", type=Path)
    owner = subparsers.add_parser("write-run-owner")
    owner.add_argument("run_id")
    owner.add_argument("run_root", type=Path)
    owner.add_argument("output", type=Path)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--source-dataset-root", type=Path, required=True)
    preflight.add_argument("--extraction-manifest", type=Path, required=True)
    preflight.add_argument("--migration-receipt", type=Path, required=True)
    preflight.add_argument("--audit-pointer", type=Path, required=True)
    preflight.add_argument("--audits-root", type=Path, required=True)
    preflight.add_argument("--env-receipt", type=Path, required=True)
    preflight.add_argument("--live-conda-snapshot", type=Path, required=True)
    preflight.add_argument("--live-pip-snapshot", type=Path, required=True)
    preflight.add_argument("--nnunet-source", type=Path, required=True)
    preflight.add_argument("--autopetv-plans", type=Path, required=True)
    preflight.add_argument("--autopetv-dataset", type=Path, required=True)
    preflight.add_argument("--audit-tool", type=Path, required=True)
    preflight.add_argument("--python-executable", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    bundle = subparsers.add_parser("validate-planning-bundle")
    bundle.add_argument("--run-id", required=True)
    bundle.add_argument("--run-root", type=Path, required=True)
    bundle.add_argument("--committed-run-dir", type=Path, required=True)
    bundle.add_argument("--source-dataset-root", type=Path, required=True)
    bundle.add_argument("--derived-dataset-root", type=Path, required=True)
    bundle.add_argument("--preprocessed-dataset-root", type=Path, required=True)
    bundle.add_argument("--extraction-manifest", type=Path, required=True)
    bundle.add_argument("--migration-receipt", type=Path, required=True)
    bundle.add_argument("--audit-pointer", type=Path, required=True)
    bundle.add_argument("--audits-root", type=Path, required=True)
    bundle.add_argument("--env-receipt", type=Path, required=True)
    bundle.add_argument("--preflight-receipt", type=Path, required=True)
    bundle.add_argument("--runtime-identity", type=Path, required=True)
    bundle.add_argument("--nnunet-source", type=Path, required=True)
    bundle.add_argument("--autopetv-plans", type=Path, required=True)
    bundle.add_argument("--autopetv-dataset", type=Path, required=True)
    bundle.add_argument("--audit-tool", type=Path, required=True)
    bundle.add_argument("--python-executable", type=Path, required=True)
    bundle.add_argument("--run-owner", type=Path, required=True)
    bundle.add_argument("--receipt", type=Path, required=True)
    commit = subparsers.add_parser("commit-run")
    commit.add_argument("staging_dir", type=Path)
    commit.add_argument("final_dir", type=Path)
    commit.add_argument("run_receipt", type=Path)
    publish = subparsers.add_parser("publish-planning-ready")
    publish.add_argument("run_dir", type=Path)
    publish.add_argument("run_receipt", type=Path)
    publish.add_argument("ready_receipt", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "write-dataset-json":
        receipt = write_derived_dataset(args.source, args.target, args.receipt)
    elif args.command == "validate-dataset-json":
        receipt = validate_derived_dataset_file(
            args.source, args.target, args.receipt
        )
    elif args.command == "validate-plans":
        receipt = validate_plans_file(args.plans, args.receipt)
    elif args.command == "capture-runtime":
        receipt = capture_runtime_identity(
            args.nnunet_source,
            args.env_receipt,
            args.python_executable,
            args.output,
        )
    elif args.command == "resolve-audit":
        completion = resolve_audit_complete(args.pointer, args.audits_root)
        print(str(completion))
        return 0
    elif args.command == "write-run-owner":
        receipt = write_run_owner(args.run_id, args.run_root, args.output)
    elif args.command == "preflight":
        receipt = write_preflight_receipt(
            source_dataset_root=args.source_dataset_root,
            extraction_manifest_path=args.extraction_manifest,
            migration_receipt_path=args.migration_receipt,
            audit_pointer_path=args.audit_pointer,
            audits_root=args.audits_root,
            env_receipt_path=args.env_receipt,
            live_conda_snapshot_path=args.live_conda_snapshot,
            live_pip_snapshot_path=args.live_pip_snapshot,
            nnunet_source=args.nnunet_source,
            autopetv_plans_path=args.autopetv_plans,
            autopetv_dataset_path=args.autopetv_dataset,
            audit_tool_path=args.audit_tool,
            python_executable=args.python_executable,
            output_path=args.output,
        )
    elif args.command == "validate-planning-bundle":
        receipt = write_planning_bundle(
            run_id=args.run_id,
            run_root=args.run_root,
            committed_run_dir=args.committed_run_dir,
            source_dataset_root=args.source_dataset_root,
            derived_dataset_root=args.derived_dataset_root,
            preprocessed_dataset_root=args.preprocessed_dataset_root,
            extraction_manifest_path=args.extraction_manifest,
            migration_receipt_path=args.migration_receipt,
            audit_pointer_path=args.audit_pointer,
            audits_root=args.audits_root,
            env_receipt_path=args.env_receipt,
            preflight_receipt_path=args.preflight_receipt,
            runtime_identity_path=args.runtime_identity,
            nnunet_source=args.nnunet_source,
            autopetv_plans_path=args.autopetv_plans,
            autopetv_dataset_path=args.autopetv_dataset,
            audit_tool_path=args.audit_tool,
            python_executable=args.python_executable,
            run_owner_path=args.run_owner,
            receipt_path=args.receipt,
        )
    elif args.command == "commit-run":
        receipt = commit_run_directory(
            args.staging_dir, args.final_dir, args.run_receipt
        )
    else:
        receipt = publish_planning_ready(
            args.run_dir, args.run_receipt, args.ready_receipt
        )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
