#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <stage-script> <log-file> <state-prefix>" >&2
  exit 2
fi

STAGE_SCRIPT="$1"
LOG_FILE="$2"
STATE_PREFIX="$3"

if [[ ! -x "${STAGE_SCRIPT}" ]]; then
  echo "Stage script is not executable: ${STAGE_SCRIPT}" >&2
  exit 3
fi
for marker in "${STATE_PREFIX}.running" "${STATE_PREFIX}.done" "${STATE_PREFIX}.fail"; do
  if [[ -e "${marker}" ]]; then
    echo "Refusing existing state marker: ${marker}" >&2
    exit 4
  fi
done

mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${STATE_PREFIX}")"
printf 'started_at=%s\nhost=%s\nscript=%s\n' \
  "$(date --iso-8601=seconds)" "$(hostname)" "${STAGE_SCRIPT}" \
  > "${STATE_PREFIX}.running"

set +e
"${STAGE_SCRIPT}" > >(tee -a "${LOG_FILE}") 2>&1
rc=$?
set -e

printf 'finished_at=%s\nexit_code=%s\n' "$(date --iso-8601=seconds)" "${rc}" \
  >> "${STATE_PREFIX}.running"
if [[ ${rc} -eq 0 ]]; then
  mv "${STATE_PREFIX}.running" "${STATE_PREFIX}.done"
else
  mv "${STATE_PREFIX}.running" "${STATE_PREFIX}.fail"
fi
exit "${rc}"
