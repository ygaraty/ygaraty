# Two 16-layer stages cut the Qwen WebGPU pipeline's time per token 57%, from 580 to 250 ms, same tokens

| | |
|---|---|
| **Conclusion** | **Works:** two 16-layer stages cut time per token 57% (577 → 246 ms) against four, same tokens (measured, Vast.ai). |
| **Status** | executed |
| **ID** | DO-39 |
| **Category** | browser |
| **Where** | workstation (a 2-stage check on the 680M), then Vast.ai and DigitalOcean |
| **Date** | Added and run 2026-11-07 |
| **Path** | [Experiment 92](092-qwen35-webgpu-four-contributors.md)'s pipeline (Qwen3.5-9B Q4_K_M in headless Chrome, hidden states relayed through an https coordinator in Amsterdam), one conversation. Arm A: four stages of 8 layers (`--splits 0,8,16,24,32`); arm B: two stages of 16 (`--splits 0,16,32`). Four RTX 3060 12 GB on Vast.ai in Poland (arm B used two of them), coordinator on a DigitalOcean s-4vcpu-8gb in `ams3` |
| **Result** | **Gate met at both lengths.** Median time per token (median of 3 runs per arm): 24-token prompt 577 → 246 ms (−57%), 8,192-token prompt 589 → 255 ms (−57%); tokens identical in all 12 runs of each length; peak application GPU memory 3.16 GB (under 4.0 GB). First token: 1.80 → 1.10 s at 24 tokens, 23.6 → 29.2 s at 8,192 (24% slower, as predicted). The cut is twice the predicted 20–25%: compute per token halved too, because a stage's step takes about the same time with 16 layers (63–85 ms) as with 8 (60–97 ms). Cost about $0.40 of the $1.50 budget |
| **Evidence** | Measured: [evidence/browser/qwen35-two-stages-2026-11-07.json](evidence/browser/qwen35-two-stages-2026-11-07.json) (gate, per-run times, per-stage compute, wait and memory, tokens); raw reports in `runs/experiments/DO-39/remote/main/` and the local check in `runs/experiments/DO-39/local-20261107-125647-splits2/` (not in git). Code 64cefcc; driver `runs/experiments/DO-39/drive_remote.py`, gate `runs/experiments/DO-39/analyze.py` |

## Question

In [experiment 92](092-qwen35-webgpu-four-contributors.md) about 270 of the 580 ms per token was the relay: each token crosses four
handoffs (A → B → C → D, then the token back to A), each through the coordinator.
Eight layers use 1.2–2.0 GB of application GPU memory, so 16 fit in 4 GB
(about 2.4–3.2 GB). With two stages, each token crosses two handoffs. Does that
cut the time per token, and what does it do to time to first token?

Raised in conversation 2026-11-07.

## Prediction

Relay time roughly halves (about 270 → 135 ms) while compute per token stays
about the same (about 300 ms, now on two GPUs instead of four): about 440 ms per
token, a cut of about 20–25%. Prefill pipelines across only two machines, so
first token at 8,192 tokens may get slower than 37 s. Memory per stage 2.4–3.2 GB,
within the 4 GB budget but with less headroom; the 8 GB Quadro would hold one.

## Gate (frozen 2026-11-07, before any machine was made; the user's yes)

On the same machines and coordinator, interleaved arms (4 stages,
2 stages; ABBAAB), the 24- and 8,192-token prompts with 128 tokens: tokens
identical across arms; **works** if the median time per token falls by at least
15%, with every contributor's application GPU memory under 4,000,000,000 bytes.

## Protocol

- **Machines:** experiment 93's setup: pool `do39gpu`, 4 Vast.ai GPUs with 8 GB or more (`any-8gb` in Poland, `NVIDIA_DRIVER_CAPABILITIES=all`, WebGPU probe at setup; all four were RTX 3060 12 GB), and pool `do39coord`, the coordinator on DigitalOcean `s-4vcpu-8gb` in Amsterdam (`ams3`).
- **Arms:** A = 4 stages (`--splits 0,8,16,24,32`, browsers on server, w0, w1, w2); B = 2 stages (`--splits 0,16,32`, browsers on server and w0; w1 and w2 idle). One conversation, no batching.
- **Prompts:** the 24-token default prompt and the 8,192-token prompt (`--text public/research/048-factorization-protein-map-webgpu.md --prompt-tokens 8192`), 128 tokens each. Order per length: A B B A A B.
- **Measured:** median inter-token ms per run (the gate takes the median over each arm's 3 runs), first-token time, per-stage application GPU memory (the page's own allocation count, as in experiment 92).
- **Cost:** about $0.37/h for both pools; budget $1.50, both deleted at 2 h (watchdogs `--minutes 120`).
- Before machines: a local 2-stage run on the 680M (`--splits 0,16,32`).
- One warmup per arm (8 tokens) before the measured runs, so each browser profile has its stage's weights cached.

## Log

- 2026-11-07: Added.
- 2026-11-07: Gate frozen as proposed, on the user's yes ("Yes: $1.50, delete at 2 h"). Started (commit 64cefcc).
- 12:56: Local check on the 680M, `--splits 0,16,32`, 16 tokens: both stages admitted, first token after 0.66 s, 143 ms per token, peaks 2.75 and 3.01 GB.
- 12:58: Pools made: 4 RTX 3060 12 GB in Poland ($0.299/h), the coordinator in `ams3` ($0.071/h). All four passed the WebGPU probe (NVIDIA, Ampere).
- 13:02–13:18: **First warmup timed out** (0 tokens, 900 s): w0 (192.0.2.1, listed in Poland) never finished its self-test; the coordinator's send to it broke. Its connect time to the coordinator was 270 ms, against 30 ms from the others, so it was deleted by its recorded ID, its host added to the run's `avoid_machines`, and replaced by an RTX 3060 at the same address as server and w2 (30 ms). Not a measured run.
- 13:26, 13:31: Warmups: A 583 ms per token, B 187 ms (8 tokens each).
- 13:34–14:01: The 12 measured runs, all finished with 128 tokens; `analyze.py`: gate met.
- 14:02: Both pools deleted (`pods down do39gpu`, `pods down do39coord`); `pods ls`: 0 machines. About 63 minutes, about $0.40 at listed prices.

## Results

| Prompt | Run | Arm | Time per token, median | p95 | First token |
|---|---|---|---:|---:|---:|
| 24 | 1 | A | 577.1 ms | 681.6 ms | 1.71 s |
| 24 | 2 | B | 246.3 | 330.4 | 1.20 |
| 24 | 3 | B | 245.9 | 331.2 | 1.04 |
| 24 | 4 | A | 581.6 | 703.4 | 1.82 |
| 24 | 5 | A | 575.4 | 697.4 | 1.80 |
| 24 | 6 | B | 249.1 | 329.4 | 1.10 |
| 8,192 | 1 | A | 589.1 | 714.2 | 23.60 |
| 8,192 | 2 | B | 255.1 | 325.7 | 28.93 |
| 8,192 | 3 | B | 253.1 | 329.3 | 29.34 |
| 8,192 | 4 | A | 596.7 | 750.0 | 30.62 |
| 8,192 | 5 | A | 589.5 | 709.8 | 21.86 |
| 8,192 | 6 | B | 259.6 | 330.7 | 29.24 |

| Gate item | 24 tokens | 8,192 tokens |
|---|---|---|
| Time per token, A → B (median of 3) | 577.1 → 246.3 ms, **−57.3%** | 589.5 → 255.1 ms, **−56.7%** |
| Tokens identical (6 runs, both arms) | yes | yes |
| Peak application GPU memory | A: 1.07–1.96 GB; B: 2.75, 3.01 GB | A: 1.15–2.04 GB; B: 2.90, **3.16 GB** |
| First token (median) | 1.80 → 1.10 s | 23.6 → 29.2 s |

**Gate met:** at least 15% at both lengths, tokens identical, every contributor under 4,000,000,000 bytes.

Per stage, median over each arm's runs (decode steps; "wait" is the time a stage waited for its next input):

| Prompt | Arm | Stage (layers) | Compute per step | Wait per step |
|---|---|---|---:|---:|
| 24 | A | A (0–7), B (8–15), C (16–23), D (24–31) | 68.0, 59.9, 60.3, 95.5 ms | 498, 507, 508, 480 ms |
| 24 | B | A (0–15), B (16–31) | 63.1, 77.5 ms | 179, 165 ms |
| 8,192 | A | A, B, C, D | 71.0, 62.9, 63.8, 97.3 ms | 508, 513, 516, 497 ms |
| 8,192 | B | A, B | 64.2, 85.2 ms | 184, 168 ms |

Summed compute per token: arm A about 284 ms (24 tokens) and 295 ms (8,192); arm B 141 and 149 ms. The rest of the time per token, the relay and its turnarounds, is about 293 ms in arm A against 106 ms in arm B: about 73 ms per handoff with four, 53 ms with two.

## What it taught

- **Fewer stages win twice.** The relay time fell from about 293 to 106 ms (two handoffs instead of four, each also a little cheaper), as predicted. The surprise is compute. A 16-layer stage's decode step takes 63–85 ms, about the same as an 8-layer one's 60–97 ms. So a step's time is mostly fixed cost per stage (submitting, reading back, waking up), not per layer, and halving the stages halved the summed compute too. Experiment 93 found the related effect from the other side: busier GPUs keep their clocks up.
- **First token is slower with fewer stages.** At 8,192 tokens the prompt is pipelined across two machines instead of four: 29.2 s against 23.6 s (+24%). Decode, which a chat spends most of its time in, more than repays that after about 17 tokens.
- **Memory fits.** 16 layers peak at 2.75–3.16 GB of application GPU memory, under the 4 GB budget, so a 4 GB contributor can still hold a stage of 16.
- **A listed location is not a latency.** Vast.ai listed the first w0 in Poland, yet it was 270 ms from Amsterdam and could not finish the self-test downloads in 15 minutes. A connect-time check to the coordinator at setup would catch such a host before any session.

## Limits

- **Same site.** All four GPUs were in Poland, three at one address; arm B's two stages were both at that address. Over home links between countries the relay share is larger, so fewer stages should matter more, but that is not measured here.
- **One conversation.** With several conversations ([experiment 93](093-qwen35-interleaved-conversations.md)), four stages keep more GPUs busy; whether two stages still win on throughput is not tested.
- **Only RTX 3060s.** A slower contributor (the Quadro RTX 4000 of experiment 92) holding 16 layers would make its stage the bottleneck.
- **Two stages, not one.** One 32-layer stage (no relay hops but the token's) was not tried; it needs about 6 GB.
- Cost is from listed prices times time, not from invoices.

## Reproduce

Workstation check (the GGUF in `runs/models/qwen3.5-9b/`):

```bash
.venv/bin/python -m ygaraty.experiments.qwen35_session local --gguf runs/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf \
  --config runs/models/qwen3.5-9b/config.json --tokenizer runs/models/qwen3.5-9b/tokenizer.json --out DIR \
  --splits 0,16,32 --generate 16
```

Rented (about $0.37/h; watchdogs delete the pools):

```bash
YGARATY_VAST_ENV=NVIDIA_DRIVER_CAPABILITIES=all python -m ygaraty.experiments.pods up do39gpu --provider vast \
  --location PL --server any-8gb --workers 3 --worker-size any-8gb --minutes 120 --max-hourly 0.8 --no-collect
python -m ygaraty.experiments.pods up do39coord --provider digitalocean --location ams3 --server s-4vcpu-8gb \
  --workers 0 --minutes 120 --max-hourly 0.3 --no-collect
python -m ygaraty.experiments.pods ship do39gpu && python -m ygaraty.experiments.pods ship do39coord
# coordinator: bash /root/remote.sh qwen35-coordinator-setup; each GPU host: bash /root/remote.sh qwen35-contributor-setup
# check each GPU host's connect time to the coordinator (curl -w '%{time_connect}'); replace any far one
python runs/experiments/DO-39/drive_remote.py runs/experiments/DO-39/<out>/warmup --only p24-1A p24-2B --generate 8
python runs/experiments/DO-39/drive_remote.py runs/experiments/DO-39/<out>/main
python runs/experiments/DO-39/analyze.py runs/experiments/DO-39/<out>/main
python -m ygaraty.experiments.pods down do39gpu && python -m ygaraty.experiments.pods down do39coord && python -m ygaraty.experiments.pods ls
```
