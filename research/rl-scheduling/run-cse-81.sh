#!/usr/bin/env bash
# Uses existing local metadata and Python packages only. Never installs or syncs.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON:-$repo_root/.venv/bin/python}"
exec "$python_bin" -m ygaraty.simulation.cse_experiment \
  --config "$repo_root/public/research/rl-scheduling/cse-81.json" "$@"
