# A100 WebGPU and WebLLM results (experiment 123, public projection)

| | |
|---|---|
| **Conclusion** | Found: WebGPU 29–31 tok/s versus WebLLM 23 (measured, A100, Qwen3.5-9B). |
| **Status** | executed |
| **ID** | DO-67 / 123 |
| **Date** | 2026-11-12 |
| **Evidence** | measured-remote; [curated WebGPU evidence](evidence/browser/four-engines-a100-2026-11-12.json), source ee2aef1 |

This is the WebGPU projection of the original multi-engine study. The complete
comparison is retained in the private research archive; these numbers are not a
new measurement. The original experiment ID is unchanged.

One A100 SXM4 80 GB ran Qwen3.5-9B. Our coalesced kernel decoded 29–31 tokens/s
with a short prompt and 30.6 at 1K context; the previous default achieved 24–26
and 24.6 respectively. WebLLM 0.2.88 q4f32 achieved 23.2 and 20.8. Its q4f16
build could not run because Chrome's NVIDIA Vulkan adapter lacked shader-f16.
The measured copy roof was 725 GB/s; GPU work took 20.6 ms of a 32–34 ms step.

Aggregate WebGPU throughput was 29.85, 23.15, 42.19 and 76.78 tokens/s at
1, 2, 4 and 8 conversations. The batch path used the rows kernel instead of the
single-conversation coalesced kernel. Two conversations were slower than one.
This motivated the batched-matvec study in [125](125-batched-matvec-a100.md).

Limits: short prompts, one rented card and differing quantization formats.
Detailed instrumentation adds overhead; use the baseline runs for throughput.
These historical measurements do not validate the newly extracted coordinator.
