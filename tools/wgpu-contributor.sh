#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m ygaraty.backends.webgpu.cli contribute --runtime wgpu "$@"
