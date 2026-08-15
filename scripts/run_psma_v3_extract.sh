#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/petct_m0_common.sh"

if [[ -e "${EXTRACT_ROOT}" ]]; then
  echo "Extraction destination already exists: ${EXTRACT_ROOT}" >&2
  exit 2
fi

"${PYTHON}" "${SCRIPT_DIR}/psma_v3_preflight.py" extract \
  "${ARCHIVE}" \
  "${EXTRACT_ROOT}"

test -f "${EXTRACT_ROOT}/SHA256-MANIFEST.json"
test -d "${SOURCE_DATASET}/imagesTr"
test -d "${SOURCE_DATASET}/labelsTr"
touch "${EXP_ROOT}/manifests/EXTRACTION_READY.done"
