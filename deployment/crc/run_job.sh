#!/usr/bin/env bash
set -euo pipefail

: "${BP_REPO:?Set BP_REPO to the checkout path on CRC}"
: "${BP_PYTHON:?Set BP_PYTHON to the tested environment Python executable}"
: "${BP_MANIFEST:?Set BP_MANIFEST to the audited dataset manifest}"
: "${BP_RUN_ROOT:?Set BP_RUN_ROOT to restricted storage outside the checkout}"
: "${BP_CONFIG:?Set BP_CONFIG to a reviewed configuration path}"
: "${BP_RUN_ID:?Set a unique BP_RUN_ID}"
: "${BP_SEED:?Set the predeclared seed}"
if [[ ! "$BP_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo 'BP_RUN_ID must contain only letters, digits, underscores and hyphens' >&2
  exit 2
fi
cd "$BP_REPO"
"$BP_PYTHON" -m bp_training preflight --config "$BP_CONFIG" \
  --manifest "$BP_MANIFEST" --seed "$BP_SEED" --device cuda --run-root "$BP_RUN_ROOT"
resume=()
if [[ "${BP_RESUME:-0}" == 1 ]]; then resume=(--resume); fi
exec "$BP_PYTHON" -m bp_training train --config "$BP_CONFIG" \
  --manifest "$BP_MANIFEST" --seed "$BP_SEED" --device cuda \
  --run-dir "$BP_RUN_ROOT/$BP_RUN_ID" "${resume[@]}"
