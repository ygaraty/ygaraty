# WebGPU backend modules

Configuration and immutable preparation are dependency-light. `factory`, `session`
and `handles` implement the public contracts. `server`, `http_api` and `cli`
compose the single-session relay host; optional dependencies are in the server
extra. `_legacy` retains audited batching, artifact and contributor behavior.
Browser assets are package resources. See [the user guide](../../../docs/README.md) for launch, scoped credentials
and contributor setup. Tests live in `tests/webgpu/` at the repository root.

`launch` provides the product entry point. `sharing` owns the anonymous SSH
subprocess and coordinates Uvicorn startup/shutdown: never expose a port until
this server has acquired it, never silently fall back to local links, and always
close the tunnel on failure. The SSH event parser accepts only registration
events and provider HTTPS hostnames. Tunnel tests replace all network/process
operations; live smoke checks must be run explicitly with synthetic data.
