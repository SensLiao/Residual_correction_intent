#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/HDD4/zlei0805/honor_degree/projects/petct_textual_intent"
DATA_DIR="${PROJECT_ROOT}/data/PSMA_v3"
ARCHIVE="${DATA_DIR}/PSMA-PET-CT_Lesions_v3.zip"
VERIFIED="${DATA_DIR}/DOWNLOAD_VERIFIED.done"
FAILED="${DATA_DIR}/DOWNLOAD.failed"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
ROUTE_DIR="${PROJECT_ROOT}/route_a"
OUTPUT_DIR="${ROUTE_DIR}/manifests"
DONE="${OUTPUT_DIR}/PSMA_V3_ARCHIVE_INVENTORY.done"

export TMPDIR="${PROJECT_ROOT}/.tmp"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.pip_cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${OUTPUT_DIR}" "${ROUTE_DIR}/logs"

echo "[$(date --iso-8601=seconds)] waiting for verified PSMA v3 archive"
while [[ ! -f "${VERIFIED}" && ! -f "${FAILED}" ]]; do
  sleep 30
done

if [[ -f "${FAILED}" ]]; then
  echo "[$(date --iso-8601=seconds)] download failure sentinel detected" >&2
  cat "${FAILED}" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] verified sentinel detected; starting read-only inventory"
python3 "${SCRIPT_DIR}/inventory_psma_v3_archive.py" "${ARCHIVE}" "${OUTPUT_DIR}"
touch "${DONE}"
echo "[$(date --iso-8601=seconds)] archive inventory complete"
