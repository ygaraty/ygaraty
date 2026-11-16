"""Transitive SDK implementation assets served alongside sdk.mjs."""

from importlib.resources import files

SDK_ASSETS = (
    "webgpu_kernels.mjs",
    "inference_plan.mjs",
    "inference_schema.mjs",
    "inference_contracts.mjs",
    "inference_layout.mjs",
    "webgpu_contracts.mjs",
    "training_gpu.mjs",
    "factorization_gpu.mjs",
    "convolution.mjs",
    "sdk_capture.mjs",
    "sdk_tensor.mjs",
    "sdk_frames.mjs",
    "sdk_cpu.mjs",
    "sdk_admission.mjs",
    "sdk_compute.mjs",
    "sdk_cache.mjs",
    "sdk_relay.mjs",
    "sdk_worker.mjs",
    "sdk_capabilities.mjs",
)

BROWSER_ROOT = files("ygaraty.backends.webgpu").joinpath("browser")
