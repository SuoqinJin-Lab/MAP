#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

run_id="toy-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-$$}"
echo "[toy] root=storage/toy_runs/${run_id}"
python scripts/toy_e2e.py \
  --root "storage/toy_runs/${run_id}" \
  --frozen-models "storage/frozen_models" \
  "$@"
