#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/petct_m0_common.sh"
exec "${PYTHON}" "${SCRIPT_DIR}/build_petct_pilot3_contract.py" "$@"
