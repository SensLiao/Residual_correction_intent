#!/usr/bin/env bash
set -euo pipefail

# Isolated nnInteractive comparator environment. This never modifies the core PET/CT env.
PETCT_ROOT="${PETCT_ROOT:-/mnt/HDD4/zlei0805/honor_degree/projects/petct_textual_intent}"
CONDA_EXE="${CONDA_EXE:-/mnt/HDD3/Zhenghong/anaconda3/bin/conda}"
ENV_PREFIX="${PETCT_ROOT}/envs/nninteractive_v1"
SOURCE_ROOT="${PETCT_NNINTERACTIVE_SOURCE:-${PETCT_ROOT}/external_runners/nninteractive/source}"
MODEL_ROOT="${PETCT_ROOT}/models/nnInteractive/nnInteractive_v1.0"
PYTHON="${ENV_PREFIX}/bin/python"
CHECKPOINT="${MODEL_ROOT}/fold_0/checkpoint_final.pth"
LICENSE_FILE="${MODEL_ROOT}/LICENSE"
SOURCE_LICENSE_FILE="${SOURCE_ROOT}/LICENSE"
CONFIG_FILE="${PETCT_ROOT}/configs/petct_external_comparators.json"
ADAPTER_FILE="${PETCT_ROOT}/scripts/comparators/nninteractive_petct_adapter.py"
ENV_FREEZE="${PETCT_ROOT}/envs/nninteractive_v1.freeze.txt"
READY_RECEIPT="${PETCT_ROOT}/envs/nninteractive_v1.READY.json"
SOURCE_COMMIT="bbe12fdccc876cb2d4e0a47133811e362608e000"
EXPECTED_CHECKPOINT_SHA256="b3ac4421f85457bbd1aa0d87f5e67bcb7bc8e2ce6b824b6ac45077cc5d630ea9"
EXPECTED_LICENSE_SHA256="4f60f5747c5506020923866690c2a41a3c74ffa85b7371eac2b02e23185f91d5"
EXPECTED_SOURCE_LICENSE_SHA256="3888a43f438f1834561474a1b16cfd0b11037c7a225935a8b89558aac550167e"
SKIP_INSTALL="${PETCT_SKIP_INSTALL:-0}"

if [[ "${SKIP_INSTALL}" != "0" && "${SKIP_INSTALL}" != "1" ]]; then
  echo "PETCT_SKIP_INSTALL must be exactly 0 or 1" >&2
  exit 2
fi

# Keep package state off the nearly-full root filesystem.  Suppressing Conda
# channel notices also avoids the shared server's corrupt zero-byte notice
# cache without mutating any user-global Conda files.
if [[ "${SKIP_INSTALL}" == "0" ]]; then
  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${PETCT_ROOT}/.conda_pkgs}"
  export CONDA_NUMBER_CHANNEL_NOTICES=0
  mkdir -p "${CONDA_PKGS_DIRS}"
fi

for required in \
  "${SOURCE_ROOT}/pyproject.toml" \
  "${SOURCE_LICENSE_FILE}" \
  "${SOURCE_ROOT}/client/pyproject.toml" \
  "${SOURCE_ROOT}/client/LICENSE" \
  "${MODEL_ROOT}/dataset.json" \
  "${MODEL_ROOT}/plans.json" \
  "${MODEL_ROOT}/inference_session_class.json" \
  "${CHECKPOINT}" \
  "${LICENSE_FILE}" \
  "${CONFIG_FILE}" \
  "${ADAPTER_FILE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required local nnInteractive artifact: ${required}" >&2
    exit 2
  fi
done
if [[ "${SKIP_INSTALL}" == "0" && ! -x "${CONDA_EXE}" ]]; then
  echo "Missing Conda executable: ${CONDA_EXE}" >&2
  exit 2
fi

observed_checkpoint_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
observed_license_sha256="$(sha256sum "${LICENSE_FILE}" | awk '{print $1}')"
observed_source_license_sha256="$(sha256sum "${SOURCE_LICENSE_FILE}" | awk '{print $1}')"
if [[ "${observed_checkpoint_sha256}" != "${EXPECTED_CHECKPOINT_SHA256}" ]]; then
  echo "nnInteractive checkpoint hash mismatch" >&2
  exit 3
fi
if [[ "${observed_license_sha256}" != "${EXPECTED_LICENSE_SHA256}" ]]; then
  echo "nnInteractive model license hash mismatch" >&2
  exit 4
fi
if [[ "${observed_source_license_sha256}" != "${EXPECTED_SOURCE_LICENSE_SHA256}" ]]; then
  echo "nnInteractive source license hash mismatch" >&2
  exit 5
fi
if [[ "$(awk 'NF {print; exit}' "${LICENSE_FILE}")" != "CC BY-NC-SA 4.0" ]]; then
  echo "Unexpected nnInteractive model license identifier" >&2
  exit 6
fi

# A failed rebuild must never leave a stale READY receipt that a launcher can accept.
rm -f "${READY_RECEIPT}"

if [[ "${SKIP_INSTALL}" == "1" ]]; then
  # Strict receipt-only refresh: an already-installed, regular Conda prefix is
  # mandatory. No Conda command, installer, editable install, or network-capable
  # dependency resolution is reachable from this branch.
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PIP_NO_INDEX=1
  export PYTHONNOUSERSITE=1
  if [[ ! -d "${ENV_PREFIX}" || -L "${ENV_PREFIX}" \
    || ! -f "${ENV_PREFIX}/conda-meta/history" || -L "${ENV_PREFIX}/conda-meta/history" \
    || ! -x "${PYTHON}" ]]; then
    echo "PETCT_SKIP_INSTALL=1 requires an existing valid Conda environment: ${ENV_PREFIX}" >&2
    exit 7
  fi
else
  if [[ ! -f "${ENV_PREFIX}/conda-meta/history" ]]; then
    if [[ -e "${ENV_PREFIX}" ]]; then
      echo "Environment target exists but is not a valid Conda prefix: ${ENV_PREFIX}" >&2
      exit 7
    fi
    "${CONDA_EXE}" create --prefix "${ENV_PREFIX}" --no-default-packages python=3.10 pip -y
  fi

  "${PYTHON}" -m pip install --disable-pip-version-check \
    --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0"
  "${PYTHON}" -m pip install --disable-pip-version-check \
    "nnunetv2==2.8.1" \
    "acvl-utils>=0.2.3,<0.3" \
    "batchgenerators>=0.25.1" \
    blosc2 \
    "fastapi>=0.110" \
    huggingface_hub \
    httpx \
    nibabel \
    numpy \
    "uvicorn[standard]>=0.27"
  "${PYTHON}" -m pip install --disable-pip-version-check --no-deps -e "${SOURCE_ROOT}/client"
  "${PYTHON}" -m pip install --disable-pip-version-check --no-deps -e "${SOURCE_ROOT}"
fi

"${PYTHON}" -m pip check

# Freeze the exact isolated environment before the smoke. This is an execution
# artifact, not a shared/global Conda mutation.
LC_ALL=C "${PYTHON}" -m pip freeze --all | LC_ALL=C sort > "${ENV_FREEZE}"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PIP_NO_INDEX=1 PYTHONNOUSERSITE=1
unset PYTHONPATH || true
"${PYTHON}" "${ADAPTER_FILE}" --help >/dev/null

export PETCT_ROOT ENV_PREFIX SOURCE_ROOT MODEL_ROOT CONFIG_FILE ADAPTER_FILE ENV_FREEZE READY_RECEIPT
export SOURCE_COMMIT EXPECTED_CHECKPOINT_SHA256 EXPECTED_LICENSE_SHA256 EXPECTED_SOURCE_LICENSE_SHA256
export SKIP_INSTALL
export CUDA_VISIBLE_DEVICES="${PETCT_NNINTERACTIVE_SMOKE_GPU:-0}"
"${PYTHON}" - <<'PY'
import gc
import hashlib
import importlib.metadata as metadata
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_bundle_sha256(root: Path) -> tuple[str, int]:
    records = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root)
        if any(part == "__pycache__" or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        records.append(f"{sha256(path)}  {rel.as_posix()}")
    payload = ("\n".join(records) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(records)


root = Path(os.environ["PETCT_ROOT"]).resolve()
env = Path(os.environ["ENV_PREFIX"]).resolve()
source = Path(os.environ["SOURCE_ROOT"]).resolve()
model = Path(os.environ["MODEL_ROOT"]).resolve()
checkpoint = model / "fold_0/checkpoint_final.pth"
license_file = model / "LICENSE"
source_license = source / "LICENSE"
config_file = Path(os.environ["CONFIG_FILE"]).resolve()
adapter_file = Path(os.environ["ADAPTER_FILE"]).resolve()
environment_freeze = Path(os.environ["ENV_FREEZE"]).resolve()
session_source = Path(inspect.getsourcefile(nnInteractiveInferenceSession)).resolve()
if Path(sys.prefix).resolve() != env:
    raise SystemExit(f"Conda prefix mismatch: {sys.prefix} != {env}")
if not session_source.is_relative_to(source):
    raise SystemExit(f"nnInteractive import is not pinned to local source: {session_source}")
core_env = root / "envs/petct_nnunet_v281"
if any(entry and Path(entry).is_relative_to(core_env) for entry in sys.path):
    raise SystemExit("Core PET/CT environment leaked into nnInteractive sys.path")
if sha256(checkpoint) != os.environ["EXPECTED_CHECKPOINT_SHA256"]:
    raise SystemExit("checkpoint hash changed after environment setup")
if sha256(license_file) != os.environ["EXPECTED_LICENSE_SHA256"]:
    raise SystemExit("license hash changed after environment setup")
if sha256(source_license) != os.environ["EXPECTED_SOURCE_LICENSE_SHA256"]:
    raise SystemExit("source license hash changed after environment setup")
initial_params = inspect.signature(nnInteractiveInferenceSession.add_initial_seg_interaction).parameters
scribble_params = inspect.signature(nnInteractiveInferenceSession.add_scribble_interaction).parameters
if not {"initial_seg", "run_prediction"} <= set(initial_params):
    raise SystemExit("installed initial-label API is incompatible")
if not {"scribble_image", "include_interaction", "run_prediction"} <= set(scribble_params):
    raise SystemExit("installed scribble API is incompatible")

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA is unavailable in the isolated nnInteractive environment")

# Real server-side smoke: load the pinned checkpoint into the actual inference
# session (which performs the official CUDA warmup), then exercise the native
# initial-M0 and scribble state path on a deterministic synthetic volume without
# producing a scientific prediction or reading project evaluation data.
smoke_device = torch.device("cuda:0")
session = nnInteractiveInferenceSession(
    device=smoke_device,
    use_torch_compile=False,
    verbose=False,
    torch_n_threads=1,
    do_autozoom=False,
    enable_undo=False,
)
session.initialize_from_trained_model_folder(
    str(model), use_fold=0, checkpoint_name="checkpoint_final.pth"
)
if session.license != "CC BY-NC-SA 4.0":
    raise SystemExit(f"runtime model license mismatch: {session.license!r}")
if session.supports_initial_label is not True or session.supported_interactions.get("scribble") is not True:
    raise SystemExit("loaded checkpoint does not expose initial-label plus scribble support")

shape = (24, 24, 24)
image = np.linspace(0.25, 1.25, num=int(np.prod(shape)), dtype=np.float32).reshape((1, *shape))
target = np.zeros(shape, dtype=np.uint8)
initial = np.zeros(shape, dtype=np.uint8)
initial[8:12, 8:12, 8:12] = 1
scribble = np.zeros(shape, dtype=np.uint8)
scribble[14, 14, 14] = 1
session.set_image(image)
session.set_target_buffer(target)
session.add_initial_seg_interaction(initial, run_prediction=False)
session.add_scribble_interaction(scribble, include_interaction=True, run_prediction=False)
if not np.array_equal(target, initial):
    raise SystemExit("initial-M0 API smoke did not preserve the supplied target buffer")

source_bundle_hash, source_bundle_files = source_bundle_sha256(source)
receipt = {
    "schema_version": "PETCT-NNINTERACTIVE-ENV-v1.1",
    "status": "PASS",
    "conda_prefix": str(env),
    "python": sys.version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "smoke_device": str(smoke_device),
    "gpu_name": torch.cuda.get_device_name(smoke_device),
    "nninteractive": metadata.version("nnInteractive"),
    "nninteractive_client": metadata.version("nninteractive-client"),
    "nninteractive_source": str(session_source),
    "source_commit": os.environ["SOURCE_COMMIT"],
    "source_bundle_sha256": source_bundle_hash,
    "source_bundle_file_count": source_bundle_files,
    "source_license": "Apache-2.0",
    "source_license_sha256": sha256(source_license),
    "checkpoint_sha256": sha256(checkpoint),
    "license": "CC BY-NC-SA 4.0",
    "license_sha256": sha256(license_file),
    "model_metadata_sha256": {
        name: sha256(model / name)
        for name in ("dataset.json", "plans.json", "inference_session_class.json")
    },
    "config_sha256": sha256(config_file),
    "adapter_sha256": sha256(adapter_file),
    "environment_freeze_sha256": sha256(environment_freeze),
    "model_load_smoke": "PASS",
    "initial_m0_api_smoke": "PASS",
    "scribble_api_smoke": "PASS",
    "adapter_cli_smoke": "PASS",
    "synthetic_only": True,
    "scientific_prediction_produced": False,
    "network_policy_at_runtime": "NO_DOWNLOADS",
    "setup_mode": "VERIFY_EXISTING_NO_INSTALL" if os.environ["SKIP_INSTALL"] == "1" else "INSTALL_OR_REFRESH",
}
if receipt["nninteractive"] != "2.5.1" or receipt["nninteractive_client"] != "2.5.1":
    raise SystemExit(f"nnInteractive source version mismatch: {receipt}")
target = Path(os.environ["READY_RECEIPT"]).resolve()
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
print(json.dumps(receipt, indent=2, sort_keys=True))

del session
gc.collect()
torch.cuda.empty_cache()
PY

echo "nnInteractive v1 comparator environment ready: ${ENV_PREFIX}"
