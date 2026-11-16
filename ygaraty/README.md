# Ygaraty implementation

This directory contains the Python implementations and browser assets. There
are no forwarding modules or separately installed core/backend distributions.

- `compiler`, `analysis`, `runtime`: Tape graphs, lowering, analysis and execution.
- `training`, `models`, `datasets`: training algorithms, models and data helpers.
- `backends/webgpu`: server, model preparation, contributor engine and browser assets.
- `backend_contracts`, `inference_support`: shared interfaces and inference utilities.
- `reference`, `verification`, `parity`: optional reference and validation helpers.

The root exports and `tape.py` expose the classes implemented in these modules.
Install from the repository root with `pip install -e '.[server,prepare]'`.
See [Python usage](../docs/python-api.md).
