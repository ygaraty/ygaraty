# Audited compatibility engine

Artifact validation, tokenization, contributor protocol, conversation batching
and bounded ASGI transport extracted from the existing WebGPU coordinator.
Imports are public-local; storage is local-file-only. The public session adapter
owns authorization, admission, lifecycle and accounting around this engine.

This is the public compatibility engine used by the WebGPU backend. Changes to
kernels, batching or protocol semantics should include corresponding regression
checks in [the WebGPU tests](../../../../tests/webgpu/README.md).
`TokenMailbox` adds bounded delivery
and shutdown drains pending inbox work. Control and data use separate pools.

Tests use fake engines and synthetic packets. No numerical or performance
claim follows from them; authorized-host integration remains required.

Shared chat formatting, contributor history, throughput and HTTP helpers are
provided by `ygaraty.inference_support`. The old
helper modules only re-export symbols for compatibility. Bundle validation and
browser protocol behavior remain WebGPU-specific.
