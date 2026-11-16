# Browser inference core

This directory contains matching TypeScript sources and generated ES modules.
`inference_schema`/`inference_contracts` define the shared operation ABI;
`inference_layout` validates byte bounds; `inference_plan` admits immutable plans,
executes compositions and tracks failed slots. `webgpu_kernels` owns shared GPU
primitives; the Qwen, dense and Kolibri modules bind family-specific resources.
Qwen3-MoE uses the Kolibri adapter. `webgpu_contracts` and `sdk_tensor` supply
their public type/value dependencies.

These modules have no private imports. `session_contributor.html` and
`qwen35_contributor.mjs` provide opt-in browser contribution. `session_chat.html`
and the handwritten `public_chat.mjs` provide private-link authentication,
readiness polling, app connection settings and streaming chat. The public server
serves these assets directly; `public_chat.mjs` is not generated TypeScript. Run `python tools/check_offline.py`
from the package directory for regenerated-module and fake-device verification.
Real GPU parity and performance still require an authorized host.
