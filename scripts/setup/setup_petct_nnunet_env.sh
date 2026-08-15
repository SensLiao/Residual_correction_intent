#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common/petct_m0_common.sh"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "Missing Conda executable: ${CONDA_EXE}" >&2
  exit 2
fi
if [[ ! -f "${NNUNET_SOURCE}/pyproject.toml" ]]; then
  echo "Missing pinned nnU-Net source: ${NNUNET_SOURCE}" >&2
  exit 3
fi

SKIP_INSTALL="${PETCT_SKIP_INSTALL:-0}"
if [[ "${SKIP_INSTALL}" != "0" && "${SKIP_INSTALL}" != "1" ]]; then
  echo "PETCT_SKIP_INSTALL must be 0 or 1" >&2
  exit 4
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p no:cacheprovider"
AUTOPETV_PROTOCOL_ROOT="${PETCT_AUTOPETV_PROTOCOL_ROOT:-${PETCT_ROOT}/external_runners/autopetv_protocol}"
AUTOPETV_PROTOCOL_MANIFEST="${PETCT_AUTOPETV_PROTOCOL_MANIFEST:-${PETCT_ROOT}/protocols/autopetv_protocol_runtime.json}"
export AUTOPETV_PROTOCOL_ROOT AUTOPETV_PROTOCOL_MANIFEST
EVIDENCE_ROOT="${EXP_ROOT}/envs"
mkdir -p "${EVIDENCE_ROOT}"
EVIDENCE_STAGE="$(mktemp -d "${EVIDENCE_ROOT}/.petct_nnunet_v281.evidence.XXXXXX")"
export PETCT_ENV_EVIDENCE_STAGE="${EVIDENCE_STAGE}"
cleanup_stage() {
  rm -rf -- "${EVIDENCE_STAGE}"
}
trap cleanup_stage EXIT

if [[ ! -f "${CONDA_ENV}/conda-meta/history" ]]; then
  if [[ -e "${CONDA_ENV}" ]]; then
    echo "Conda target exists but has no conda-meta history: ${CONDA_ENV}" >&2
    exit 5
  fi
  if [[ "${SKIP_INSTALL}" == "1" ]]; then
    echo "Verify-only mode requires an existing Conda environment: ${CONDA_ENV}" >&2
    exit 6
  fi
  "${CONDA_EXE}" create --prefix "${CONDA_ENV}" --no-default-packages \
    "python=3.10" pip -y
fi

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  # An install/refresh mutates the environment.  Invalidate the public marker
  # atomically before that mutation; a failed refresh must never retain READY.
  "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

marker = Path("/mnt/HDD4/zlei0805/honor_degree/projects/petct_textual_intent/nnunet/envs/ENV_READY.done")
marker.parent.mkdir(parents=True, exist_ok=True)
temporary = marker.with_name(".ENV_READY.mutation-in-progress.tmp")
temporary.write_text(
    json.dumps(
        {
            "schema_version": "PETCT-NNUNET-ENV-MARKER-v1.0",
            "status": "ENVIRONMENT_MUTATION_IN_PROGRESS_NOT_READY",
        },
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
os.replace(temporary, marker)
PY
  "${PYTHON}" -m pip install --disable-pip-version-check \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0"
  "${PYTHON}" -m pip install --disable-pip-version-check \
    "pytest==8.3.4" \
    "connected-components-3d==4.0.0"
  "${PYTHON}" -m pip install --disable-pip-version-check -e "${NNUNET_SOURCE}"
  export PETCT_ENV_SETUP_MODE="INSTALL_OR_REFRESH"
else
  export PETCT_ENV_SETUP_MODE="VERIFY_EXISTING_NO_INSTALL"
fi

"${PYTHON}" - <<'PY'
import importlib.metadata as metadata
import importlib.util
import hashlib
import json
import os
import site
import sys
from pathlib import Path
import torch
import nnunetv2
import cc3d

workspace = Path("/mnt/HDD4/zlei0805/honor_degree")
project = workspace / "projects/petct_textual_intent"
env = project / "envs/petct_nnunet_v281"
autopetv_raw = Path(os.environ["AUTOPETV_PROTOCOL_ROOT"])
if autopetv_raw.is_symlink():
    raise SystemExit("AutoPET V server protocol root must not be a symlink")
autopetv = autopetv_raw.resolve()
autopetv_manifest_path = Path(os.environ["AUTOPETV_PROTOCOL_MANIFEST"]).resolve()
forbidden = (
    str(workspace / "conda_envs/rl_nnunet"),
    str(workspace / "Honor_codes/experiments/rl_nnunet"),
)
receipt = {
    "schema_version": "PETCT-NNUNET-ENV-v1.1",
    "status": "PREFLIGHT_PASS_PENDING_ATOMIC_EVIDENCE_PUBLICATION",
    "setup_mode": os.environ["PETCT_ENV_SETUP_MODE"],
    "environment_kind": "fresh-conda-prefix",
    "conda_prefix": str(Path(sys.prefix).resolve()),
    "base_prefix": str(Path(sys.base_prefix).resolve()),
    "python": __import__("sys").version,
    "torch": torch.__version__,
    "torch_import": str(Path(torch.__file__).resolve()),
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "nnunetv2": metadata.version("nnunetv2"),
    "nnunetv2_import": str(Path(nnunetv2.__file__).resolve()),
    "connected_components_3d": metadata.version("connected-components-3d"),
    "cc3d_import": str(Path(cc3d.__file__).resolve()),
    "cc3d_distribution_root": str(
        Path(metadata.distribution("connected-components-3d").locate_file("")).resolve()
    ),
    "site_packages": site.getsitepackages(),
    "pythonpath": sys.path,
    "forbidden_iac_prefixes": list(forbidden),
    "expected_nnunet_commit": "468cf803df9b267150ae2b6c0c59b8ac84f16227",
    "expected_nnunet_runtime_tree_sha256": "02d8e7578634022737245d29e8be46e5badb7f56b1a38c805b7a9b42d0e76cf4",
}
if receipt["nnunetv2"] != "2.8.1":
    raise SystemExit(f"nnunetv2 version mismatch: {receipt['nnunetv2']}")
if receipt["connected_components_3d"] != "4.0.0":
    raise SystemExit(
        "connected-components-3d version mismatch: "
        + receipt["connected_components_3d"]
    )
if Path(sys.prefix).resolve() != env.resolve():
    raise SystemExit(f"Conda prefix mismatch: {sys.prefix} != {env}")
expected_source = project / "upstream/nnUNet"

def resolve_head(source):
    git_dir = source / ".git"
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                oid, name = line.split(" ", 1)
                if name == ref:
                    return oid
    raise SystemExit(f"cannot resolve source HEAD without git: {source}")

runtime_files = [
    expected_source / "pyproject.toml",
    expected_source / "setup.py",
    *expected_source.joinpath("nnunetv2").rglob("*.py"),
]
runtime_files = sorted(
    runtime_files, key=lambda path: path.relative_to(expected_source).as_posix()
)
tree_digest = hashlib.sha256()
for path in runtime_files:
    relative = path.relative_to(expected_source).as_posix()
    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    tree_digest.update(relative.encode("utf-8"))
    tree_digest.update(b"\0")
    tree_digest.update(file_sha256.encode("ascii"))
    tree_digest.update(b"\n")
observed_tree_sha256 = tree_digest.hexdigest()
if observed_tree_sha256 != receipt["expected_nnunet_runtime_tree_sha256"]:
    raise SystemExit("nnUNet runtime source tree differs from the pinned tree manifest")

# Server deployments intentionally omit Git metadata.  In that case the exact
# runtime-tree digest is the executable identity and binds this snapshot to the
# canonical upstream commit declared above.  If Git metadata is present, both
# identities must agree; an incomplete or unexpected .git entry fails closed.
git_dir = expected_source / ".git"
if git_dir.exists():
    if not git_dir.is_dir():
        raise SystemExit("nnUNet .git entry is not a directory")
    observed_commit = resolve_head(expected_source)
    if observed_commit != receipt["expected_nnunet_commit"]:
        raise SystemExit(
            f"nnUNet source commit mismatch: {observed_commit} != "
            + receipt["expected_nnunet_commit"]
        )
    source_identity_mode = "git-head-and-pinned-runtime-tree"
else:
    observed_commit = receipt["expected_nnunet_commit"]
    source_identity_mode = "pinned-runtime-tree-without-git-metadata"
receipt["nnunet_source_commit"] = observed_commit
receipt["nnunet_runtime_tree_sha256"] = observed_tree_sha256
receipt["nnunet_runtime_tree_file_count"] = len(runtime_files)
receipt["nnunet_source_identity_mode"] = source_identity_mode
receipt["nnunet_git_metadata_present"] = git_dir.is_dir()
resolved_import = Path(receipt["nnunetv2_import"])
if not resolved_import.is_relative_to(expected_source):
    raise SystemExit(
        f"nnunetv2 import is not pinned to source: {resolved_import} != {expected_source}"
    )
torch_import = Path(receipt["torch_import"])
if not torch_import.is_relative_to(env):
    raise SystemExit(f"torch import is outside isolated Conda env: {torch_import}")
cc3d_import = Path(receipt["cc3d_import"])
if not cc3d_import.is_relative_to(env):
    raise SystemExit(f"cc3d import is outside isolated Conda env: {cc3d_import}")
cc3d_distribution_root = Path(receipt["cc3d_distribution_root"])
if not cc3d_distribution_root.is_relative_to(env):
    raise SystemExit(
        "connected-components-3d distribution is outside isolated Conda env: "
        f"{cc3d_distribution_root}"
    )
for entry in sys.path:
    if entry and any(entry.startswith(prefix) for prefix in forbidden):
        raise SystemExit(f"IAC path leaked into PET/CT sys.path: {entry}")
if not receipt["cuda_available"]:
    raise SystemExit("CUDA is not available")

def import_official(path, module_name, required_callable, expected_sha256):
    path = path.resolve()
    if not path.is_relative_to(autopetv.resolve()) or not path.is_file():
        raise SystemExit(f"official AutoPET module is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot create import spec for official AutoPET module: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SystemExit(f"official AutoPET import preflight failed for {path}: {exc}")
    if not callable(getattr(module, required_callable, None)):
        raise SystemExit(
            f"official AutoPET module lacks callable {required_callable}: {path}"
        )
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_sha256 != expected_sha256:
        raise SystemExit(f"official AutoPET hash mismatch: {path}")
    return {
        "path": str(path),
        "sha256": observed_sha256,
        "expected_sha256": expected_sha256,
        "required_callable": required_callable,
        "import_status": "PASS",
    }

if autopetv_manifest_path.is_symlink() or not autopetv_manifest_path.is_file():
    raise SystemExit("AutoPET V minimal runtime manifest is missing")
autopetv_manifest = json.loads(autopetv_manifest_path.read_text(encoding="utf-8"))
if (
    autopetv_manifest.get("schema_version")
    != "PETCT-AUTOPETV-PROTOCOL-RUNTIME-v2.0"
    or autopetv_manifest.get("status")
    != "FROZEN_MINIMAL_RUNTIME_SIX_CLASS_POLARITY_ADAPTER_NOT_EXECUTED"
    or autopetv_manifest.get("upstream_commit")
    != "4a2026866bfacc812492cfc7e6a8c54ac3c4f703"
    or autopetv_manifest.get("license") != "Apache-2.0"
):
    raise SystemExit("AutoPET V minimal runtime manifest contract mismatch")
records = {
    record.get("path"): record
    for record in autopetv_manifest.get("files", [])
    if isinstance(record, dict)
}
expected_autopetv = {
    "simulator": (
        "interactive/simulate_scribbles.py",
        "petct_setup_autopetv_simulator",
        "simulate_scribble_from_label",
        "a2124e8aa4207e53ac93259214a35b7cf74626f83ab164e519769f86557d7cd2",
    ),
    "metrics": (
        "metrics.py",
        "petct_setup_autopetv_metrics",
        "MetricEvaluator",
        "93e303219deb46b10fc5e5532873a42745aec1ecd6f78335f36cebba62104b83",
    ),
}
expected_runtime_paths = {contract[0] for contract in expected_autopetv.values()}
if set(records) != {*expected_runtime_paths, "LICENSE"}:
    raise SystemExit("AutoPET V minimal runtime file inventory mismatch")
runtime_entries = list(autopetv.rglob("*"))
if any(path.is_symlink() for path in runtime_entries):
    raise SystemExit("AutoPET V server protocol root contains a symlink")
observed_runtime_files = {
    path.relative_to(autopetv).as_posix()
    for path in runtime_entries
    if path.is_file()
}
if observed_runtime_files != {*expected_runtime_paths, "LICENSE"}:
    raise SystemExit(
        "AutoPET V server protocol root contains files outside the frozen minimal package"
    )
observed_runtime_directories = {
    path.relative_to(autopetv).as_posix()
    for path in runtime_entries
    if path.is_dir()
}
if observed_runtime_directories != {"interactive"}:
    raise SystemExit("AutoPET V server protocol directory inventory mismatch")
receipt["official_autopetv_preflight"] = {}
for receipt_key, (
    relative,
    module_name,
    callable_name,
    expected_sha256,
) in expected_autopetv.items():
    record = records[relative]
    if (
        record.get("sha256") != expected_sha256
        or record.get("required_callable") != callable_name
    ):
        raise SystemExit(f"AutoPET V manifest binding mismatch: {relative}")
    receipt["official_autopetv_preflight"][receipt_key] = import_official(
        autopetv / relative,
        module_name,
        callable_name,
        expected_sha256,
    )
license_path = autopetv / "LICENSE"
license_expected = "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6"
if (
    records["LICENSE"].get("sha256") != license_expected
    or license_path.is_symlink()
    or not license_path.is_file()
    or hashlib.sha256(license_path.read_bytes()).hexdigest() != license_expected
):
    raise SystemExit("AutoPET V minimal runtime license mismatch")
receipt["official_autopetv_preflight"]["LICENSE"] = {
    "path": str(license_path),
    "sha256": license_expected,
    "license": "Apache-2.0",
}
receipt["autopetv_source_commit"] = autopetv_manifest["upstream_commit"]
receipt["autopetv_runtime_manifest"] = {
    "path": str(autopetv_manifest_path),
    "sha256": hashlib.sha256(autopetv_manifest_path.read_bytes()).hexdigest(),
    "schema_version": autopetv_manifest["schema_version"],
}
out = Path(os.environ["PETCT_ENV_EVIDENCE_STAGE"]) / "petct_nnunet_v281.json"
out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
PY

"${PYTHON}" -m pytest \
  "${PETCT_ROOT}/tests/test_psma_v3_preflight.py" \
  "${PETCT_ROOT}/tests/test_audit_psma_v3_dataset.py" \
  -q -p no:cacheprovider

"${CONDA_EXE}" list --prefix "${CONDA_ENV}" --explicit \
  > "${EVIDENCE_STAGE}/petct_nnunet_v281.conda-explicit.txt"
"${PYTHON}" -m pip freeze --all \
  > "${EVIDENCE_STAGE}/petct_nnunet_v281.pip-freeze.txt"
if find "${CONDA_ENV}" -type l -printf '%p -> %l\n' \
  | grep -E 'conda_envs/rl_nnunet|Honor_codes/experiments/rl_nnunet' \
  > "${EVIDENCE_STAGE}/petct_nnunet_v281.forbidden-symlinks.txt"; then
  echo "IAC symlink leaked into isolated PET/CT Conda environment" >&2
  exit 7
fi

"${PYTHON}" - <<'PY'
import hashlib
import json
import os
import shutil
from pathlib import Path

stage = Path(os.environ["PETCT_ENV_EVIDENCE_STAGE"]).resolve()
root = Path("/mnt/HDD4/zlei0805/honor_degree/projects/petct_textual_intent/nnunet/envs").resolve()
setup_mode = os.environ["PETCT_ENV_SETUP_MODE"]

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

hash_bound_names = [
    "petct_nnunet_v281.json",
    "petct_nnunet_v281.conda-explicit.txt",
    "petct_nnunet_v281.pip-freeze.txt",
]
hash_receipt_name = "petct_nnunet_v281.receipt-sha256.txt"
(stage / hash_receipt_name).write_text(
    "".join(
        f"{sha256_file(stage / name)}  {root / name}\n" for name in hash_bound_names
    ),
    encoding="utf-8",
)
evidence_names = [
    *hash_bound_names,
    hash_receipt_name,
    "petct_nnunet_v281.forbidden-symlinks.txt",
]
for name in evidence_names:
    path = stage / name
    if not path.is_file():
        path.write_text("", encoding="utf-8")
records = [
    {"name": name, "sha256": sha256_file(stage / name), "bytes": (stage / name).stat().st_size}
    for name in evidence_names
]
core = {
    "schema_version": "PETCT-NNUNET-ENV-EVIDENCE-BUNDLE-v1.0",
    "status": "ENVIRONMENT_EVIDENCE_COMPLETE",
    "setup_mode": setup_mode,
    "files": records,
}
canonical = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
bundle_sha256 = hashlib.sha256(canonical).hexdigest()
bundle = {**core, "bundle_sha256": bundle_sha256}
(stage / "bundle.json").write_text(
    json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
bundle_manifest_sha256 = sha256_file(stage / "bundle.json")
final_bundle = root / "evidence-bundles" / bundle_sha256
final_bundle.parent.mkdir(parents=True, exist_ok=True)
if final_bundle.exists():
    existing = final_bundle / "bundle.json"
    if not existing.is_file() or sha256_file(existing) != bundle_manifest_sha256:
        raise SystemExit("existing environment evidence bundle conflicts with this bundle")
    shutil.rmtree(stage)
else:
    os.rename(stage, final_bundle)

# Compatibility artifacts are individually atomically replaced from the now
# immutable bundle.  They are not an acceptance signal; ENV_READY.done is.
for name in evidence_names:
    target = root / name
    temporary = root / ("." + name + ".tmp")
    shutil.copyfile(final_bundle / name, temporary)
    os.replace(temporary, target)

receipt = final_bundle / "petct_nnunet_v281.json"
marker_payload = {
    "schema_version": "PETCT-NNUNET-ENV-MARKER-v1.0",
    "status": "ENVIRONMENT_EVIDENCE_COMPLETE",
    "setup_mode": setup_mode,
    "bundle_path": str(final_bundle),
    "bundle_sha256": bundle_sha256,
    "bundle_manifest_sha256": bundle_manifest_sha256,
    "receipt_path": str(receipt),
    "receipt_sha256": sha256_file(receipt),
}
marker = root / "ENV_READY.done"
temporary_marker = root / ".ENV_READY.done.tmp"
temporary_marker.write_text(
    json.dumps(marker_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary_marker, marker)
print(json.dumps(marker_payload, sort_keys=True))
PY
