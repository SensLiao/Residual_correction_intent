#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/petct_m0_common.sh"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "Missing Conda executable: ${CONDA_EXE}" >&2
  exit 2
fi
if [[ ! -f "${NNUNET_SOURCE}/pyproject.toml" ]]; then
  echo "Missing pinned nnU-Net source: ${NNUNET_SOURCE}" >&2
  exit 3
fi

if [[ ! -f "${CONDA_ENV}/conda-meta/history" ]]; then
  if [[ -e "${CONDA_ENV}" ]]; then
    echo "Conda target exists but has no conda-meta history: ${CONDA_ENV}" >&2
    exit 4
  fi
  "${CONDA_EXE}" create --prefix "${CONDA_ENV}" --no-default-packages \
    "python=3.10" pip -y
fi

"${PYTHON}" -m pip install --disable-pip-version-check \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0"
"${PYTHON}" -m pip install --disable-pip-version-check "pytest==8.3.4"
"${PYTHON}" -m pip install --disable-pip-version-check -e "${NNUNET_SOURCE}"

"${PYTHON}" - <<'PY'
import importlib.metadata as metadata
import json
import site
import sys
from pathlib import Path
import torch
import nnunetv2

workspace = Path("/mnt/HDD4/workspace/honor_degree")
project = workspace / "projects/petct_textual_intent"
env = project / "envs/petct_nnunet_v281"
forbidden = (
    str(workspace / "conda_envs/rl_nnunet"),
    str(workspace / "Honor_codes/experiments/rl_nnunet"),
)
receipt = {
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
    "site_packages": site.getsitepackages(),
    "pythonpath": sys.path,
    "forbidden_iac_prefixes": list(forbidden),
    "expected_nnunet_commit": "468cf803df9b267150ae2b6c0c59b8ac84f16227",
}
if receipt["nnunetv2"] != "2.8.1":
    raise SystemExit(f"nnunetv2 version mismatch: {receipt['nnunetv2']}")
if Path(sys.prefix).resolve() != env.resolve():
    raise SystemExit(f"Conda prefix mismatch: {sys.prefix} != {env}")
expected_source = project / "upstream/nnUNet"
resolved_import = Path(receipt["nnunetv2_import"])
if not resolved_import.is_relative_to(expected_source):
    raise SystemExit(
        f"nnunetv2 import is not pinned to source: {resolved_import} != {expected_source}"
    )
torch_import = Path(receipt["torch_import"])
if not torch_import.is_relative_to(env):
    raise SystemExit(f"torch import is outside isolated Conda env: {torch_import}")
for entry in sys.path:
    if entry and any(entry.startswith(prefix) for prefix in forbidden):
        raise SystemExit(f"IAC path leaked into PET/CT sys.path: {entry}")
if not receipt["cuda_available"]:
    raise SystemExit("CUDA is not available")
out = project / "nnunet/envs/petct_nnunet_v281.json"
out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
PY

"${PYTHON}" -m pytest \
  "${PETCT_ROOT}/tests/test_psma_v3_preflight.py" \
  "${PETCT_ROOT}/tests/test_audit_psma_v3_dataset.py" \
  -q

"${CONDA_EXE}" list --prefix "${CONDA_ENV}" --explicit \
  > "${EXP_ROOT}/envs/petct_nnunet_v281.conda-explicit.txt"
"${PYTHON}" -m pip freeze --all \
  > "${EXP_ROOT}/envs/petct_nnunet_v281.pip-freeze.txt"
if find "${CONDA_ENV}" -type l -printf '%p -> %l\n' \
  | grep -E 'conda_envs/rl_nnunet|Honor_codes/experiments/rl_nnunet' \
  > "${EXP_ROOT}/envs/petct_nnunet_v281.forbidden-symlinks.txt"; then
  echo "IAC symlink leaked into isolated PET/CT Conda environment" >&2
  exit 5
fi
sha256sum \
  "${EXP_ROOT}/envs/petct_nnunet_v281.json" \
  "${EXP_ROOT}/envs/petct_nnunet_v281.conda-explicit.txt" \
  "${EXP_ROOT}/envs/petct_nnunet_v281.pip-freeze.txt" \
  > "${EXP_ROOT}/envs/petct_nnunet_v281.receipt-sha256.txt"

touch "${EXP_ROOT}/envs/ENV_READY.done"
