# One rented A100 80 GB serves all of Kolibri-1 as a single WebGPU contributor, reference-exact, at 125–174 ms per token

| | |
|---|---|
| **Conclusion** | Works: one rented A100 80 GB serves all of Kolibri-1 in one Chrome page; reference-exact, 125–174 ms/token (measured) |
| **Status** | executed |
| **ID** | DO-55 (experiment 113) |
| **Category** | browser |
| **Where** | remote |
| **Priority** | high |
| **Date** | Added 2026-11-11; run 2026-11-11 23:14 to 2026-11-12 01:48 CET |
| **Path** | Kolibri-1 Q4_K (our GGUF, converted outside this experiment), one stage (layers 0–50, `--splits 0,50`), headless Chrome on one Vast.ai 80 GB GPU (A100 or H100), coordinator and chat driver on the same machine |
| **Result** | All three gates pass. Gate 1: one page held 60 GiB (76 GiB too), 0 corrupt. Gate 2: 32/32 greedy tokens equal to the NumPy reference, logits within 3.7e-4 of the largest (limit 2e-3), kernels 4.1e-7 (limit 1e-5). Gate 3: 3 chat turns of 256 tokens, no error; first token 0.52–0.84 s; 125, 174, 157 ms per token; load 16 min; peak GPU allocation 45.2 GB (budget 49). Descriptive: through production (relay, coordinator far away) the same contributor served coherent replies at about 285 ms per token, about 145 ms of it the per-token round trip; cold loads from Spaces are slow (one piece at a time, no stall timeout). Cost $3.27 of $10. |
| **Evidence** | `runs/pods-data/do55/` (capacity-60/76.json, gate2/{gate2.json,pipeline-report.json,pipeline.log,kernels-report.json,model.json}, gate3/{report.json,serve.log,serve.events.jsonl}); code fa8d44b (machine clone 9a09720); GGUF SHA-256 510f8b5f…aec8c; curated: `evidence/browser/kolibri-single-a100-2026-11-12.json` |

## Question

The one-contributor plan (Kolibri-1, 49 GB at 4096 positions) assumes a single browser page can hold and run the whole model. Can it, on a rented data-center GPU: does Chrome's WebGPU run there, does one page's device hold about 50 GB, does the single stage reproduce the NumPy reference, and how fast does it serve chat turns?

This is also DO-48's first GPU contact: the Kolibri shaders have never been compiled by a browser. Related: [DO-48](DO-48.md) (multi-contributor Kolibri; its numerical gate is reused here), historical large-GPU contributor notes.

## Prediction

The NVIDIA Vulkan driver runs Chrome's WebGPU on the A100 as on the RTX 3090s (DO-36); one page holds at least 60 GiB; the stage matches the reference (a WGSL compile error at first contact is likely and would be fixed and rerun). Decode is bandwidth-bound: a few tens of ms per token for about 3 GB of active weights per token, with no link round trips.

## Inputs (from outside this experiment)

- The converted `kolibri-1-Q4_K.gguf` and its `record` output (SHA-256, size), from the conversion already under way; the GGUF placed on the rented machine and checked against that SHA-256 before any run. The historical transfer used Spaces; deployment credentials and transfer orchestration are not part of this public protocol.
- `config.json` and `tokenizer.json` at the pinned revision `abfe85f` (fetched by `ygaraty model fetch` from the record).

## Gate (frozen 2026-11-11, before renting)

1. **Capacity:** allocate, fill and verify 60 GiB in the browser: 60 GiB allocated, filled and checked, 0 corrupt samples, device not lost.
2. **Numerics (DO-48's proposed gate):** `qwen35_webgpu pipeline --splits 0,50 --generate 32 --kv f16 --wire f16` against the NumPy reference (`kolibri_gguf`): every greedy token equal to the reference's except near ties (reference margin below 0.05), logits within 2e-3. `kernels` within 1e-5 of the largest output.
3. **Serving:** `qwen35_session local --chat --splits 0,50 --capacity 4096 --generate 256 --record-turn-ids`, the chat driver's three turns of 256 tokens (`chat_cycles.py`-style prompts of 500 words): all complete without error. Reported, not gated: load time (cold download from the coordinator; warm from Chrome's cache), time to first token, ms per token, peak GPU allocation against the 49 GB budget.
- **Works** if 1–3 pass. **Failed gate** naming the step otherwise. A shader fix found at step 2 is committed, pushed, and the step rerun once as a new frozen attempt (logged).

## Historical protocol

On one A100 SXM4 80 GB with at least 64 GB host memory and 200 GB disk,
verify browser capacity first, then the pinned model's kernel and pipeline
numerics, then three chat turns with the coordinator on the same host. The
pipeline uses one stage (layers 0–50), 32 generated tokens, f16 KV and f16 wire.
Preserve each stage's logs and reports before releasing the machine.

The gates above describe the historical harness settings, not supported current
CLI commands. Private rental, transfer and teardown instructions are archived
outside this repository. Actual cost and runtime are retained with the results.

## Log

- 2026-11-11: The 49 GB single-worker configuration and capacity probe were prepared; the probe passed a 2 GiB workstation check. Model conversion was performed separately.

- 2026-11-11 23:14 CET: rented `do55` server: Vast.ai A100 SXM4 80 GB, Montana US, $1.274/h, 1 TB host RAM, 208 CPUs, 200 GB disk; watchdog 239 min (at most $5.08). The machine cloned `main` at 9a09720 (upstream moved past fa8d44b; DO-55's code unchanged). Setup: Chrome 157.0.8103.78, Vulkan device `NVIDIA A100-SXM4-80GB`; f16 probe exact (1.000000): **Chrome's WebGPU runs on the data-center A100**.
- **Gate 1 passes:** `probe --gb 60`: 60.0 of 60 GiB in 60 buffers of 1 GiB, 0 corrupt samples, device not lost (`runs/pods-data/do55/capacity-60.json`). Descriptive: 76 GiB also held, 0 corrupt (`capacity-76.json`). Adapter limits: maxBufferSize 4 GiB, maxStorageBufferBindingSize 2 GiB.
- Prepared for gates 2–3: config, tokenizer at `abfe85f` fetched and verified on the machine; `tokenizers` 0.23.3 installed; `serve55.py` copied. Waiting for the GGUF's presigned URL and SHA-256.
- 23:40 CET: the presigned GGUF URL arrived (0600 file, not read here); the GGUF's SHA-256 has not (the repo's record is still planned). Protocol change before gate 2, logged: the GGUF is downloaded without an expected hash; its SHA-256 is recorded on the machine and `kolibri_convert record` re-derives the record there (structure and metadata checks), to be compared with the preparation host's record. Gate 2's criteria unchanged.
- 23:46 CET: A descriptive relay session used one contributor for layers 0–50, loading the published single-worker pieces while the GGUF downloaded. It was stopped before the numerical gate needed the GPU.
- 00:08 CET: GGUF downloaded from Spaces (presigned, ~24 MB/s): 44,762,823,200 bytes, SHA-256 `510f8b5fdeefd8105437cb00ac8fed9dd0ba0b05931b8a480df0360b868aec8c` (to compare with the preparation host's record).
- 00:12–00:14 CET, gate 2 part 1 (beside the loading production contributor; `gate2.sh record kernels`): `record`: 50 layers, tensor types F32 401, Q4_K 100, Q6_K 352, 41.19 GiB in layers + 0.50 GiB embedding/head, chat format thinking off, `compatible: yes` (first attempt exit 1: `jinja2` missing for the chat-template check, installed, rerun). **`kernels` passes:** Kolibri's WGSL compiled in Chrome for the first time; every case's max |err| at most 4.1e-7 of the largest output (limit 1e-5), Q4_K experts, Q6_K attention, shared expert and head, F32 router (`runs/pods-data/do55/gate2/`). The `pipeline` step waits for the GPU.
- 00:15–00:38 CET, production (descriptive): the first contributor's load stalled at 29.6 of 44.8 GB from 00:17 (one piece's Spaces connection at ~170 KB/s; the page fetches pieces one at a time, `spaces_download.mjs` has no stall timeout). A leave and rejoin found the slot still held (panel: "All contributor slots are occupied") until about 5 min after the leave; my retry loop then re-joined after its own successful join (every join first stops the machine's browsers) and killed it: a harness fault. The session was cancelled and recreated.
- 00:38–00:55 CET: joined once; Chrome's cache reused (about 30 GB from disk, the rest from Spaces at ~17 MB/s); **serving at 00:55:07** (1 of 1), 17 min after the join. User's three turns in one chat: 97 tokens, first token 1.28 s, ≈3.5 tok/s; 17 tokens, 0.81 s, ≈3.4 tok/s; 164 tokens, 0.81 s, ≈3.5 tok/s (about 285 ms per token); relay, prompt via the coordinator. Replies coherent ("good replies, compatible with what I would expect"). The per-token time likely includes one Montana–coordinator round trip per token; gate 3 (coordinator on the machine) separates it.
- 01:10 CET: contributor left (`leave`); gate 2's `pipeline` step started (19 prompt tokens, 32 generated, one stage 0–50).
- 01:10–01:28 CET, gate 2 part 2 (`gate2.sh pipeline`, exit 0, 17 min with the NumPy reference): one stage of all 50 layers in one Chrome page, 19 prompt tokens, 32 greedy tokens, f16 cache and wire. **Gate 2 passes** (computed by script, `runs/pods-data/do55/gate2/gate2.json`): 32/32 tokens equal to the reference (smallest margin 0.23, no near ties); logits max |error| 3.7e-4 of the largest |logit| (limit 2e-3; "2e-3" read as WS-12/experiment 101 defines it, relative to the largest logit; absolute max 4.3e-3); kernels worst 4.1e-7 (limit 1e-5). Descriptive: harness decode median **86 ms per token** (no coordinator), against about 285 ms per token through production.
- 01:31 CET: gate 3 started (`serve55.py` under xvfb-run: `local --chat --splits 0,50`, coordinator on the machine).
- 01:31–01:47 CET, gate 3 (`serve55.py`, exit 0): load 972 s (hashing 1157 weight ranges on the machine, then the page's download from the local coordinator, its SHA-256 checks and GPU upload; `loadMs` 930 s); turns of 38, 47, 47 prompt tokens, 256 tokens each, no error, first token 0.84 / 0.52 / 0.61 s, **125 / 174 / 157 ms per token**; stage step compute median 137.6 ms, wait 7 ms; peak GPU allocation 45,209,217,856 bytes (42.1 GiB; plan budget 49 GB). **Gate 3 passes.**
- 01:48 CET: Measurement completed. Cost: 2 h 34 min × $1.274/h = **$3.27**.
- **Works** (all three gates). Findings for follow-up: (1) relay sessions with one contributor pay a coordinator round trip per token (285 against ~145 ms per token here); (2) cold loads fetch pieces one at a time with no stall timeout (one slow connection stalled a load at 29.6 of 44.8 GB); (3) a contributor that leaves mid-load holds its slot about 5 min; (4) about 138 ms of GPU work per token, against a ~1–2 ms memory-bandwidth floor: dispatch count is the next lever.

## Results

| Gate | Measure | Result | Limit | Pass |
|---|---|---|---|---|
| 1 | GiB allocated, filled and checked in one page | 60.0 (76.0 also) | 60, 0 corrupt | yes |
| 2 | greedy tokens equal to the NumPy reference | 32/32 (smallest margin 0.23) | all, near ties excepted | yes |
| 2 | logits max \|error\| / largest \|logit\| | 3.7e-4 (absolute 4.3e-3) | 2e-3 | yes |
| 2 | kernels max \|error\| / largest output | 4.1e-7 | 1e-5 | yes |
| 3 | chat turns of 256 tokens without error | 3/3 | 3/3 | yes |

| Serving (descriptive) | Load | First token | ms per token | Note |
|---|---|---|---|---|
| Gate 3, coordinator on the machine | 16 min (972 s) | 0.84, 0.52, 0.61 s | 125, 174, 157 | step compute median 137.6 ms, wait 7 ms; peak 45.2 GB |
| Gate 2 harness (no coordinator) | — | — | 86 (median) | 19-token prompt, 32 tokens |
| Production (relay, user's turns) | 17 min after a rejoin (about 30 GB from Chrome's cache) | 0.81–1.28 s | about 285 (≈3.5 tok/s) | coherent replies |

Cost: $3.27 (2 h 34 min of one Vast.ai A100 SXM4 80 GB at $1.274/h).

## What it taught

- Chrome's WebGPU runs on a data-center A100 in a Vast.ai container (the NVIDIA Vulkan driver mounted), and one page can use almost the whole 80 GB card. Kolibri's shaders, never compiled by a browser before this run, are correct at the first attempt.
- A one-contributor relay session pays a round trip to the coordinator per token: about 285 ms per token against about 140–150 ms with the coordinator on the machine. Keeping the token loop on the contributor would roughly halve the per-token time.
- Cold loads are slow and fragile: the page fetches pieces one at a time, with no stall timeout (`spaces_download.mjs`); one slow Spaces connection held a load at 29.6 of 44.8 GB. Parallel fetches with a stall timeout are the fix.
- A contributor that leaves mid-load holds its panel slot for about 5 minutes; a rejoin is refused meanwhile.
- About 138 ms of GPU work per token, against a memory-bandwidth floor of about 1–2 ms for 2.2 GB of active weights: the fixed cost per GPU call (at least about 0.15 ms each, about a thousand calls per token) dominates.
- Harness faults (logged): `jinja2` missing for the record step's chat-template check; a retry loop that re-joined after its own successful join and stopped that contributor (every `join` first stops the machine's browsers). An early size report read the GGUF mid-download (42.7 GB); the complete file is 44,762,823,200 bytes.

## Reproduction scope

The original rental and deployment commands are archived outside this repository;
they are not supported public commands. Preserve the pinned model revision,
GGUF checksum, hardware/runtime details, gates and prompt sizes above when
reproducing the measurements. The [public research helpers](src/ygaraty_webgpu_research/README.md)
and [contributor guide](../docs/contributing-gpu.md) describe the available code
and current worker setup. The historical serving harness is not included.

## Limits

One GPU model and host, one run per gate (no repeats); Chrome only; 4096 positions; gate 3's prompts were 38–47 tokens (chat_cycles' `prompt(0, t, 500)`, as in WS-27/28), so long-prompt prefill was not measured; one conversation at a time (no batching). The production run is descriptive (user's own three turns, no fixed prompts).
