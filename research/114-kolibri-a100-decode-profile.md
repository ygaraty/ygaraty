# On one A100, a Kolibri-1 token's GPU work is 28–38 ms of about 100 ms; merging expert chunks saves 9.5 ms, bitwise identical

| | |
|---|---|
| **Conclusion** | Found: GPU work is 28–38 ms of a ~100 ms token; merging expert chunks saves 9.5 ms GPU, bitwise identical (measured) |
| **Status** | executed |
| **ID** | DO-63 (experiment 114) |
| **Category** | browser |
| **Where** | remote |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | Kolibri-1 Q4_K (GGUF SHA-256 510f8b5f…aec8c), one stage of 50 layers (`qwen35_webgpu pipeline --splits 0,50 --profile`), headless Chrome on one Vast.ai A100 80 GB; arms C (expert chunks in separate buffers, `--no-merge-experts`, as in experiment 113) and M (fusion step 1: a layer's chunks merged into one buffer); both with the uniform pool |
| **Result** | Found. Per decode token (medians, C vs M): step 108 vs 99 ms; submission (encode to GPU done) 63 vs 55 ms; JavaScript encode 11 vs 9 ms; GPU pass 37.7 vs 28.2 ms; dispatches 1688 vs 1388 (expert products 400 vs 100). About 45 ms per token lies outside the submission (the plan's per-token expansion and validation in JavaScript before encoding, and the token's read-back). All 4 runs valid (32/32 tokens, logits within 2.8e-4); M's logits bitwise identical to C's (32/32 files). Parallel GGUF download 271 MB/s (165 s for 44.8 GB). |
| **Evidence** | `runs/pods-data/do63/` (gate.json, C1/M2/M3/C4 reports, logs and logits); machine clone 6e60083; curated: `public/research/evidence/browser/kolibri-a100-decode-speed-2026-11-12.json`; code committed as 430c583 |

## Question

Experiment [113](113-kolibri-single-a100-contributor.md) measured about 138 ms of stage time per decode token on one A100, against a memory-bandwidth floor of about 1–2 ms. Where does it go: JavaScript encoding of about 1,750 dispatches, the GPU's own execution, or waiting? Which operations dominate? And does fusion step 1 (one expert dispatch per projection per layer instead of one per 128 MiB chunk: about 8 → 2 per layer) change the time, with identical output?

## Prediction

Per-dispatch costs dominate: the GPU pass is a minority of the wall time, and JavaScript encoding plus submission overhead the majority. M saves the expert dispatches (about 300 per token) and a few ms per token; logits bitwise identical to C (the same products in the same order; checked on a tiny model on the workstation).

## Gate (frozen 2026-11-12, before renting)

Diagnostic: the verdict is **Found**, with one correctness check.

- Runs: order C M M C, each `pipeline --splits 0,50 --generate 32 --kv f16 --wire f16 --profile` (19 prompt tokens, as in 113).
- Valid run: 32/32 tokens equal to the NumPy reference (near ties excepted) and logits within 2e-3 of the largest (113's gate 2).
- **Correctness:** every M run's logits bitwise identical to the C runs'. If not, report it as a fault of the merge (and stop using M).
- Report per arm (median over decode steps, then over runs): wall ms per token, JavaScript encode ms, GPU pass ms (timestamp queries), dispatches and submissions per token, dispatches by operation; the difference M − C.

## Budget

One Vast.ai A100 80 GB (about $1.27–1.61/h): setup about 15 min, the GGUF from Spaces about 35 min (the presigned URL is valid until 2026-11-12 23:35 CET), four runs of about 18 min each (the NumPy reference included). About 2 h: about $3. User: "yes, parallel and start do-63" (2026-11-12), after the proposal: cap **$6**, pool watchdog **180 min**.

## Protocol

- Code: the working tree after 113 with the profiler (`webgpu_kernels.mts`: `profiling`, `takeProfile`, timestamp queries when the adapter has them), the uniform pool, the Kolibri expert merge (`kolibri_gpu.mts`, `mergeExperts`) and the harness flags (`qwen35_webgpu.py --profile`, `--no-merge-experts`). Committed and pushed before renting (the machine clones `main`).
- Checked on the workstation first (2026-11-12, Radeon 680M): a tiny random Kolibri (3 layers, 8 experts, Q4_K experts, chunks forced to 2 experts with `YGARATY_KOLIBRI_EXPERT_CHUNK_BYTES=150000`): both arms 16/16 tokens equal to the reference, logits bitwise identical between arms (16/16 files), expert dispatches 18 → 6 per token; the profile reads JavaScript encode 0.7 ms, GPU pass 1.1 ms, wall 3.6 ms per token (86 dispatches). The uniform pool touches every family: WS-27's S5-nano chat sessions (1 and 2 workers, `local --chat --logit-digests`) rerun on this code are bitwise identical to WS-27's run 1 (prompt ids, tokens, all 96 digests). WebGPU unit tests: 132 of 134 pass; the 2 failures (`chat-connection-ui.test.mjs`) fail on clean HEAD too.
- `contributor_fleet up` (A100 80 GB, `--min-ram 64 --disk 200`), `fetch` (the GGUF in 16 parallel byte ranges, `range_fetch`, SHA-256 checked against 510f8b5f…aec8c), then `runs/pods-data/do63/runs.sh` (config and tokenizer, `tokenizers`, the four runs C M M C); copy each report back as soon as it ends; `down`.

## Log

- 2026-11-12: Added at user request ("profile so we learn, but also start expert-fusion kernel to test locally initially"). Profiler, uniform pool and fusion step 1 implemented and checked locally (Protocol). Budget not yet approved.
- 2026-11-12: budget approved (cap $6, watchdog 180 min); gate frozen as written. Added `range_fetch` (parallel byte ranges with retries and a stall timeout; tested locally against a ranged server that drops connections) for the GGUF download.
- 2026-11-12 02:39 CET: rented `do63`: Vast.ai A100 SXM4 80 GB, North Carolina, $1.634/h, watchdog 179 min (05:38). GGUF in 165 s by `range_fetch` (16 ranges, 271 MB/s; one connection gave 23 MB/s in 113), SHA-256 510f8b5f…aec8c matched.
- 02:44–03:28 CET: runs C1 M2 M3 C4 (`runs.sh`, each exit 0, about 11 min each). Gate by script (`runs/pods-data/do63/gate.json`): all valid; **merge bitwise identical**. Found as in Result. Reading: GPU time is under a third of the step; the rest is JavaScript before and during encoding (the plan expansion runs every token), the submit-to-done wait, and the token read-back. Merging cut GPU time by a quarter (fewer, fuller expert dispatches; each chunk's dispatch launched workgroups that exited at once).

- 2026-11-12 05:06 CET: machine deleted after DO-66; `pods ls` 0. The shared rental (DO-63 to DO-66) cost $4.01 (cap $6).
- 2026-11-12: written up as experiment 114.

## Limits

- One machine, one prompt (19 tokens), 32 generated tokens: short context; full-attention cost grows with positions.
- Two runs per arm; the step medians spread (M 91.7 and 105.9 ms), so the step difference is indicative; the GPU-time difference (37.7 vs 28.2 ms, both runs per arm within 0.4 ms) is firm.
- The harness step (`pipeline`) excludes a chat session's overhead: experiment 113's chat turns ran 125–174 ms against about 105 ms here.
- The 45 ms outside the submission was located by subtraction here; DO-65 measured it directly (plan expansion).

## Reproduce

```bash
# rented 80 GB GPU, clone at 6e60083 or later
python -m ygaraty.experiments.contributor_fleet up do63 --gpus 1 --gpu any-80gb --min-ram 64 --disk 200 --max-hourly 2 --minutes 180
python -m ygaraty.experiments.contributor_fleet setup do63
python -m ygaraty.experiments.contributor_fleet fetch do63 --url-file ~/.config/ygaraty/kolibri-gguf.url \
  --dest /workspace/kolibri-1/kolibri-1-Q4_K.gguf --sha256 510f8b5f...aec8c   # full hash in experiment 113
# on the machine: runs/pods-data/do63/runs.sh (C M M C: pipeline --splits 0,50 --generate 32 --kv f16 --wire f16 --profile [--no-merge-experts])
```
