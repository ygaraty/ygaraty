# On the 680M, matrix-vector products are 94% of decode GPU time and run at 71% of the copy roof; attention stays under 3 ms at 4k

| | |
|---|---|
| **Conclusion** | Found: matvecs are 94% of 680M decode GPU time, at 71% of copy bandwidth; steady (measured) |
| **Status** | executed |
| **ID** | WS-31 (experiment 119) |
| **Category** | browser |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | `qwen35_webgpu bench` reworked: a fresh slot and prefill per variant, contexts 64, 1,024, 4,096; per operation bytes moved and GPU ms (detail pass) as GB/s and share of the measured copy peak; S5-nano, Qwen3.5-9B and tiny Kolibri on the 680M |
| **Result** | Steady (base and base-again 0.8% and 0.1% apart). Copy roof 62.3 GB/s (llama.cpp's effective decode bandwidth in experiment 118: 62.7). Qwen3.5-9B: step 131.8 ms at 64 positions, 135.5 at 4,096; GPU 125.5–127.8 ms, of which matrix-vector products 119.6 ms (94%) at 44.4 GB/s (Q4_K 71%, Q6_K 74%, Q5_K 68% of the roof); attention 0.26 → 1.47 → 2.74 ms at 64, 1k, 4k; the gated delta update 2.4 ms (49%); host (expansion + encode) 2.5 ms. S5-nano the same shape: matvecs 93% of 72–75 ms at 42.5 GB/s. On this GPU the gap to native is the matrix-vector kernels, nothing else. |
| **Evidence** | measured-local; curated `public/research/evidence/browser/webgpu-roofline-680m-2026-11-12.json`; raw `runs/experiments/WS-31/omarchy-20261112-1354/` (reports, logs); driver `runs/ws30/ws31.sh` |

## Question

Experiment 115 was unsteady because its variants ran at growing contexts, and it gave times, not efficiencies. Which operations are furthest from the bandwidth roof, at which context?

## Prediction

Matrix-vector products at 20–35% of peak, attention far below at short context, norms and adds dominated by launch cost.

## Gate (frozen 2026-11-12, before the main runs)

Diagnostic: **Found**. `qwen35_webgpu bench --contexts 64,1024,4096` on the 680M, Qwen3.5-9B Q4_K_M then S5-nano Q4_K_M, one run each: per context a fresh slot prefilled to it, 16 profiled decode steps and 4 detail steps (a timestamped pass per dispatch), then base again at 64 in a new slot; the first step of each variant dropped.

- **Valid** if base-64 and base-again-64 median steps are within 5%; otherwise unsteady.
- Roof: the copy probe's median (256 MiB, read + write, GPU time).
- Reported per context: step ms, GPU ms, host ms; per operation GPU ms, bound bytes, GB/s and share of the copy roof. Bound bytes are exact for matrix-vector products (the weight binding is the tensor) and an upper bound for operations that bind whole caches (attention, state updates): their GB/s is reported but not read as efficiency.

## Protocol

Bench change and an `analyze` script; checked on the tiny model first. The same bench runs unchanged in DO-67.

## Log

- 2026-11-12: Added at user request ("to close the gap with: https://github.com/mlc-ai/web-llm - plan next set of webgpu worker optimization experiments"), in the plan [DO_webgpu_speed.md](DO_webgpu_speed.md).
- 2026-11-12: bench reworked (`--contexts`: per context a fresh slot prefilled to it; detail records each dispatch's bound bytes, `bytesByName`; a 256 MiB copy probe gives the roof). Rehearsed on S5-nano (64, 1024): steady, 0.8% apart. Gate frozen.
- 2026-11-12 13:54–13:59: main run (`runs/ws30/ws31.sh omarchy-20261112-1354`, under watch.py, stall 900 s): both models exit 0. Results as above.

## Results

Qwen3.5-9B Q4_K_M, one stage, 680M (copy roof 62.3 GB/s):

| Context | Step ms | GPU ms | Matvec ms (share) | Matvec GB/s | Attention ms | Delta ms | Host ms |
|---|---|---|---|---|---|---|---|
| 64 | 131.8 | 125.5 | 119.6 (95%) | 44.4 | 0.26 | 2.36 | 2.7 |
| 1,024 | 134.4 | 126.7 | 119.6 (94%) | 44.4 | 1.47 | 2.37 | 2.4 |
| 4,096 | 135.5 | 128.0 | 119.7 (94%) | 44.4 | 2.74 | 2.38 | 2.5 |

Per format at 64 positions: Q4_K 65.3 ms, 2,742 MB, 44.0 GB/s (71%); Q6_K 33.9 ms, 1,498 MB, 46.3 GB/s (74%); Q5_K 20.4 ms, 828 MB, 42.6 GB/s (68%).

## What it taught

- On a bandwidth-bound GPU only the matrix-vector kernels matter: 94% of GPU time. Raising them from 71% to 95% of the roof would take Qwen3.5-9B from about 7.6 to about 10 tokens per second, close to llama.cpp's 12.3 (which also reads at about 63 GB/s). WS-34 is the experiment for this GPU; host work (2.5 ms), attention (2% at 4k) and the small operations do not move it.
- Attention grows linearly with context (0.26 → 2.74 ms from 64 to 4,096 positions) but stays small on Qwen3.5 (8 of 32 layers attend). Kolibri's attention was much larger in experiment 115 (0.21 ms per layer at short context on the A100), so WS-35 matters there.
- The bench runs the matrix-vector kernel at one row per thread unless told otherwise; the chat path uses the 2,4,8 default (experiment 104), which is why the chat path in experiment 118 was faster (124.8 against 131.8 ms). WS-34's comparisons must name the kernel setting.

## Limits

- One run per model; one GPU. The A100 equivalent runs in DO-67.
- Bound bytes are exact for matrix-vector products only; for attention, norms and state updates they count whole bound buffers, so those GB/s are not efficiencies.
- The copy roof counts read and write; a read-only roof may be a little higher.
- The bench used one row per thread (see above).

## Reproduce

```bash
python -m ygaraty.experiments.qwen35_webgpu bench --gguf runs/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf \
  --config runs/models/qwen3.5-9b/config.json --tokenizer runs/models/qwen3.5-9b/tokenizer.json \
  --contexts 64,1024,4096 --out OUT
```
