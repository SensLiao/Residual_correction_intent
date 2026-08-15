#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/HDD4/zlei0805/honor_degree/projects/petct_textual_intent"
DESTINATION="${PROJECT_ROOT}/data/PSMA_v3"
ARCHIVE="PSMA-PET-CT_Lesions_v3.zip"
SOURCE_URL="https://fdat.uni-tuebingen.de/api/records/g27kx-86t35/files/PSMA-PET-CT_Lesions_v3.zip/content"
EXPECTED_BYTES="20588970455"
EXPECTED_MD5="156c136aea40541275d97cc6bfae9f39"

export TMPDIR="${PROJECT_ROOT}/.tmp"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.pip_cache"
export CONDA_PKGS_DIRS="${PROJECT_ROOT}/.conda_pkgs"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

mkdir -p \
  "${TMPDIR}" \
  "${PIP_CACHE_DIR}" \
  "${CONDA_PKGS_DIRS}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${DESTINATION}"
cd "${DESTINATION}"

on_error() {
  local exit_code=$?
  {
    date --iso-8601=seconds
    echo "exit_code=${exit_code}"
  } > DOWNLOAD.failed
  exit "${exit_code}"
}
trap on_error ERR

{
  echo "dataset=PSMA-PET-CT-Lesions v3"
  echo "record=https://fdat.uni-tuebingen.de/records/g27kx-86t35"
  echo "doi=10.57754/FDAT.g27kx-86t35"
  echo "license=CC BY-NC 4.0"
  echo "source_url=${SOURCE_URL}"
  echo "expected_bytes=${EXPECTED_BYTES}"
  echo "expected_md5=${EXPECTED_MD5}"
  echo "started_at=$(date --iso-8601=seconds)"
} > download_manifest.txt

echo "[$(date --iso-8601=seconds)] starting/resuming ${ARCHIVE}"
curl \
  --fail \
  --location \
  --continue-at - \
  --retry 30 \
  --retry-connrefused \
  --retry-delay 10 \
  --retry-max-time 0 \
  --speed-limit 1024 \
  --speed-time 120 \
  --output "${ARCHIVE}" \
  "${SOURCE_URL}"

actual_bytes=$(stat --format='%s' "${ARCHIVE}")
if [[ "${actual_bytes}" != "${EXPECTED_BYTES}" ]]; then
  echo "size mismatch: expected ${EXPECTED_BYTES}, got ${actual_bytes}" >&2
  exit 2
fi

echo "${EXPECTED_MD5}  ${ARCHIVE}" | md5sum --check --strict
unzip -t "${ARCHIVE}" > unzip_test.log

{
  echo "verified_at=$(date --iso-8601=seconds)"
  echo "actual_bytes=${actual_bytes}"
  echo "md5=${EXPECTED_MD5}"
  echo "zip_test=pass"
} >> download_manifest.txt

touch DOWNLOAD_VERIFIED.done
echo "[$(date --iso-8601=seconds)] download, size, MD5, and ZIP integrity verified"
