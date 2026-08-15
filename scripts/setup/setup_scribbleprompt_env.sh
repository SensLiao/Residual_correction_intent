#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PETCT_ROOT:-/mnt/HDD4/workspace/honor_degree/projects/petct_textual_intent}"
CONDA_EXE="${CONDA_EXE:-/mnt/HDD4/workspace/miniconda3/bin/conda}"
ENV_PREFIX="${PETCT_SCRIBBLEPROMPT_ENV:-${PROJECT_ROOT}/envs/scribbleprompt_v1}"
CORE_ENV="${PETCT_CORE_ENV:-${PROJECT_ROOT}/envs/petct_nnunet_v281}"
SOURCE_ROOT="${PETCT_SCRIBBLEPROMPT_SOURCE:-${PROJECT_ROOT}/external_runners/scribbleprompt/source}"
CHECKPOINT="${PETCT_SCRIBBLEPROMPT_CHECKPOINT:-${PROJECT_ROOT}/models/ScribblePrompt/ScribblePrompt_unet_v1_nf192_res128.pt}"
RECEIPT_DIR="${PETCT_SCRIBBLEPROMPT_RECEIPT_DIR:-${PROJECT_ROOT}/records/environments}"
EXPECTED_CHECKPOINT_SHA256="43f57ee8fa8ec529c31be281e06749f9e629b30157bbbcc9baf200cddec1acbe"

# Keep Conda state on the project disk and suppress channel-notice reads.  The
# server's shared Conda currently has a zero-byte notice cache; allowing the
# default notice fetch makes an otherwise valid `conda create` abort before it
# reaches the solver.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${PROJECT_ROOT}/.conda_pkgs}"
export CONDA_NUMBER_CHANNEL_NOTICES=0
mkdir -p "${CONDA_PKGS_DIRS}"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "Missing Conda executable: ${CONDA_EXE}" >&2
  exit 2
fi
if [[ "$(readlink -m "${ENV_PREFIX}")" == "$(readlink -m "${CORE_ENV}")" ]]; then
  echo "ScribblePrompt must use an independent prefix, not the core PET/CT environment" >&2
  exit 3
fi
for required in \
  "${SOURCE_ROOT}/setup.py" \
  "${SOURCE_ROOT}/scribbleprompt/models/unet.py" \
  "${SOURCE_ROOT}/scribbleprompt/models/network.py" \
  "${SOURCE_ROOT}/LICENSE" \
  "${CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required ScribblePrompt artifact: ${required}" >&2
    exit 4
  fi
done
ACTUAL_CHECKPOINT_SHA256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
if [[ "${ACTUAL_CHECKPOINT_SHA256}" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
  echo "ScribblePrompt checkpoint SHA256 mismatch" >&2
  exit 5
fi

if [[ ! -f "${ENV_PREFIX}/conda-meta/history" ]]; then
  if [[ -e "${ENV_PREFIX}" ]]; then
    echo "Environment target exists but is not a Conda prefix: ${ENV_PREFIX}" >&2
    exit 6
  fi
  "${CONDA_EXE}" create --prefix "${ENV_PREFIX}" --no-default-packages \
    "python=3.10.18" pip -y
fi

PYTHON="${ENV_PREFIX}/bin/python"
"${PYTHON}" -m pip install --disable-pip-version-check "pip==24.3.1"
"${PYTHON}" -m pip install --disable-pip-version-check \
  --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.6.0" "torchvision==0.21.0"
"${PYTHON}" -m pip install --disable-pip-version-check \
  "numpy==1.26.4" \
  "nibabel==5.3.2" \
  "opencv-python==4.10.0.84" \
  "pytest==8.3.5"
"${PYTHON}" -m pip install --disable-pip-version-check --no-deps -e "${SOURCE_ROOT}"
"${PYTHON}" -m pip check

mkdir -p "${RECEIPT_DIR}"
export PROJECT_ROOT ENV_PREFIX CORE_ENV SOURCE_ROOT CHECKPOINT RECEIPT_DIR EXPECTED_CHECKPOINT_SHA256
"${PYTHON}" - <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path

import nibabel
import numpy
import torch

project = Path(os.environ["PROJECT_ROOT"]).resolve()
prefix = Path(os.environ["ENV_PREFIX"]).resolve()
core = Path(os.environ["CORE_ENV"]).resolve()
source = Path(os.environ["SOURCE_ROOT"]).resolve()
checkpoint = Path(os.environ["CHECKPOINT"]).resolve()
receipt_dir = Path(os.environ["RECEIPT_DIR"]).resolve()

if Path(sys.prefix).resolve() != prefix:
    raise SystemExit(f"wrong active prefix: {sys.prefix} != {prefix}")
if prefix == core:
    raise SystemExit("independent environment invariant failed")
if not Path(torch.__file__).resolve().is_relative_to(prefix):
    raise SystemExit("torch imported outside the ScribblePrompt prefix")
if torch.__version__ != "2.6.0+cu124":
    raise SystemExit(f"unexpected torch build: {torch.__version__}")
if numpy.__version__ != "1.26.4":
    raise SystemExit(f"unexpected numpy version: {numpy.__version__}")
if nibabel.__version__ != "5.3.2":
    raise SystemExit(f"unexpected nibabel version: {nibabel.__version__}")

sys.path.insert(0, str(source))
from scribbleprompt import ScribblePromptUNet

ScribblePromptUNet.weights = {"v1": checkpoint}
model = ScribblePromptUNet(version="v1", device="cpu")
model.model.eval()

digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
if digest != os.environ["EXPECTED_CHECKPOINT_SHA256"]:
    raise SystemExit("checkpoint changed during environment setup")
receipt = {
    "schema_version": "PETCT-SCRIBBLEPROMPT-ENV-v1.0",
    "environment_kind": "independent-conda-prefix",
    "prefix": str(prefix),
    "core_prefix_modified": False,
    "python": sys.version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "numpy": numpy.__version__,
    "nibabel": nibabel.__version__,
    "opencv_python": metadata.version("opencv-python"),
    "scribbleprompt": metadata.version("scribbleprompt"),
    "source_root": str(source),
    "source_commit": "182c44975f77749b559974ce8db558c8bde57788",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": digest,
    "official_unet_loaded_on_cpu": True,
}
path = receipt_dir / "scribbleprompt_v1.json"
path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
PY

"${PYTHON}" "${PROJECT_ROOT}/scripts/comparators/scribbleprompt_petct_adapter.py" --help >/dev/null
"${CONDA_EXE}" list --prefix "${ENV_PREFIX}" --explicit \
  > "${RECEIPT_DIR}/scribbleprompt_v1.conda-explicit.txt"
"${PYTHON}" -m pip freeze --all \
  > "${RECEIPT_DIR}/scribbleprompt_v1.pip-freeze.txt"
sha256sum \
  "${RECEIPT_DIR}/scribbleprompt_v1.json" \
  "${RECEIPT_DIR}/scribbleprompt_v1.conda-explicit.txt" \
  "${RECEIPT_DIR}/scribbleprompt_v1.pip-freeze.txt" \
  > "${RECEIPT_DIR}/scribbleprompt_v1.receipt-sha256.txt"
