#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/mnt/HDD4/zlei0805/honor_degree"
PETCT_ROOT="${WORKSPACE}/projects/petct_textual_intent"
EXP_ROOT="${PETCT_ROOT}/nnunet"
DATA_ARCHIVE_ROOT="${PETCT_ROOT}/data/PSMA_v3"
ARCHIVE="${DATA_ARCHIVE_ROOT}/PSMA-PET-CT_Lesions_v3.zip"
EXTRACT_ROOT="${DATA_ARCHIVE_ROOT}/extracted"
SOURCE_DATASET="${EXTRACT_ROOT}/PSMA-PET-CT-Lesions_v3"
DATASET_ID="901"
DATASET_NAME="Dataset901_PSMA_M0_AutoPETVNorm"

CONDA_EXE="/mnt/HDD3/Zhenghong/anaconda3/bin/conda"
CONDA_ENV="${PETCT_ROOT}/envs/petct_nnunet_v281"
PYTHON="${CONDA_ENV}/bin/python"
NNUNET_SOURCE="${PETCT_ROOT}/upstream/nnUNet"
AUTOPETV_SOURCE="${PETCT_ROOT}/upstream/autoPETV"
PSMA_PREPROCESSING_SOURCE="${PETCT_ROOT}/upstream/tcia-psma-pet-ct-preprocessing"

export TMPDIR="${PETCT_ROOT}/.tmp"
export PIP_CACHE_DIR="${PETCT_ROOT}/.pip_cache"
export CONDA_PKGS_DIRS="${PETCT_ROOT}/.conda_pkgs"
export XDG_CACHE_HOME="${PETCT_ROOT}/.cache"
export HF_HOME="${PETCT_ROOT}/.cache/huggingface"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true
export nnUNet_raw="${EXP_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${EXP_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${EXP_ROOT}/nnUNet_results"

mkdir -p \
  "${TMPDIR}" \
  "${PIP_CACHE_DIR}" \
  "${CONDA_PKGS_DIRS}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${PETCT_ROOT}/envs" \
  "${PETCT_ROOT}/quarantine" \
  "${PETCT_ROOT}/receipts" \
  "${EXP_ROOT}/upstream_receipts" \
  "${EXP_ROOT}/manifests" \
  "${EXP_ROOT}/splits" \
  "${EXP_ROOT}/audits" \
  "${EXP_ROOT}/envs" \
  "${EXP_ROOT}/logs" \
  "${EXP_ROOT}/oof_predictions" \
  "${EXP_ROOT}/oof_probabilities" \
  "${EXP_ROOT}/evaluation" \
  "${nnUNet_raw}" \
  "${nnUNet_preprocessed}" \
  "${nnUNet_results}"
