#!/usr/bin/env bash
set -euo pipefail

# Gaussian may call External with either "EIn EOu" or "operation EIn EOu".
if [[ $# -eq 2 ]]; then
  ein="$1"
  eou="$2"
elif [[ $# -ge 3 ]]; then
  ein="$2"
  eou="$3"
else
  echo "Usage: horm_external.sh [operation] EIn EOu" >&2
  exit 2
fi

exec "${MLIP_GAUSSIAN_PYTHON:-python}" -m mlip_gaussian_handoff.external "$ein" "$eou"
