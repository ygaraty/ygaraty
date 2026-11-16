# Eight interleaved conversations get 9.3x the Qwen WebGPU pipeline's throughput on the same four GPUs, every token unchanged

| | |
|---|---|
| **Conclusion** | **Works:** eight batched conversations give 9.3x one conversation's tokens/s on the same four GPUs, tokens unchanged (measured, Vast.ai). |
| **Status** | executed |
| **ID** | DO-38 (added as DO-37; renumbered when that ID was taken upstream) |
| **Category** | browser |
| **Where** | workstation (code, correctness, screen on the 680M), then Vast.ai and DigitalOcean |
| **Date** | Added and run 2026-11-07 |
| **Path** | [Experiment 92](092-qwen35-webgpu-four-contributors.md)'s pipeline (Qwen3.5-9B Q4_K_M, four WebGPU contributors of 8 layers each, hidden states relayed through an https coordinator) serving N conversations at once, each with its own caches on every stage. Arm I: each stage serves the waiting messages one at a time. Arm B: a stage runs the waiting decode tokens of different conversations as one batch and sends their outputs in one message. Four RTX 3060 12 GB in Poland (Vast.ai), coordinator in Amsterdam (DigitalOcean); 8 chat prompts, 128 tokens each, N = 1, 2, 4, 6, 8, 3 repeats per arm |
| **Result** | **Gate passes in both arms.** Steady total throughput, median of 3: arm I 1.74 / 4.24 / 9.15 / 9.51 / 9.71 tokens/s at N = 1 / 2 / 4 / 6 / 8 (5.6x at N = 8), arm B 1.74 / 4.28 / 8.91 / 11.58 / **16.17 (9.3x)**; the slowest conversation's median time per token at N = 8: arm I 816 ms (1.43x the 573 ms of one conversation), arm B 610 ms (1.06x). **Tokens: all 121 conversations of the multi-conversation runs (12,098 tokens) equal their runs alone.** Cost about $0.62 at listed prices, budget $2 |
| **Evidence** | Measured: [evidence/browser/qwen35-interleaved-conversations-2026-11-07.json](evidence/browser/qwen35-interleaved-conversations-2026-11-07.json) (gate table, per-run throughput, per-conversation latency, per-stage compute and wait); raw reports in `runs/experiments/DO-38/rented-20261107/`, local runs in `runs/experiments/DO-38/omarchy-20261107-*` (not in git). Code bc2ffd1; driver `runs/experiments/DO-38/drive_remote.py`, analysis `analyze.py`; tests `tests/test_qwen35_session.py` |

## Question

In [experiment 92](092-qwen35-webgpu-four-contributors.md) one conversation got 580 ms per token, but each stage computed for only
42–93 ms of it; stage A waited a median 491 ms per step. If several
conversations pass through the pipeline at once, each stage can work on one
conversation while the others are elsewhere. How much total throughput does the
same set of four GPUs give, and what does it cost each conversation?

Related: [DO-21](DO-21.md) asks the same for native Phi-4 (staged batching);
this is the WebGPU Qwen counterpart. Raised in conversation 2026-11-07 (would two
copies of every layer be twice as fast?).

## Prediction

Total tokens/s rises almost linearly until the slowest stage is busy all the
time: about 580 / 93 ≈ 6 conversations, roughly 10 tokens/s against 1.7.
Beyond that, batching the waiting conversations' tokens into one step (decode is
bound by reading the weights) raises the ceiling further. Each conversation's
time per token stays near 580 ms until saturation, then grows. Memory: each
conversation needs its own KV cache and DeltaNet state per stage (64 MiB of
float16 KV per stage at 8,192 positions); short contexts make that small.

## Gate

**Frozen 2026-11-07, before any measured run** (the user approved the plan and
a $2 budget for the remote run).

- **Metric.** Total throughput is the steady rate: tokens arriving between the
  moment every conversation has its first token and the moment the first
  conversation ends, divided by that window (`throughput.steady_tokens_per_s`
  in the report; for one conversation it is 1 / mean gap). Per-conversation
  latency is its median time between tokens.
- **Arms.** I: interleaving only (each stage serves one conversation at a
  time, in arrival order). B: interleaving plus batching (`--batch`: a stage
  runs the waiting decode tokens of different conversations as one batch).
- **Correctness (both arms):** every conversation's generated tokens equal
  the same conversation's tokens run alone (`--conversations 1
  --first-conversation c`) on the same machines. Any difference fails the arm.
- **Decision (remote run, per arm):** **works** if, at the best N of
  1, 2, 4, 6, 8, the median over 3 repeats of the steady throughput is at
  least 3x the N = 1 median, and every conversation's median time per token
  at that N is at most 1.5x the N = 1 median. Otherwise **failed gate**,
  reporting the best ratio.
- The local screen on the 680M (all four stages on one GPU, the relay emulated
  with `--relay-delay`) predicts and checks correctness; it is not the gate.

## Protocol

1. **Local correctness** (680M, relay delay 0 and 68 ms): N = 8 conversations,
   32 tokens each, both arms; each conversation alone (8 single runs).
2. **Local screen** (680M): `--relay-delay 68` (experiment 92's 270 ms of relay
   per token over 4 hand-offs), 64 tokens per conversation, N = 1, 2, 4, 6, 8,
   arms I and B, 2 repeats in alternating order.
3. **Remote** (budget $2, deleted within 2 hours of the first machine): 4
   Vast.ai GPUs (8 GB or more, Europe) and a DigitalOcean s-4vcpu-8gb
   coordinator in ams3, as experiment 92. The 8 conversations' chat prompts
   (`MESSAGES` in `qwen35_session.py`), 128 tokens each. N = 1, 2, 4, 6, 8,
   arms I and B, 3 repeats, interleaved order (I B B I I B per N); single runs
   of conversations 1-7 for the correctness check (conversation 0's are the
   N = 1 runs).

## Log

- 2026-11-07: Added.
- 2026-11-07: Built: per-conversation slots in `Stage` (KV caches, DeltaNet and
  convolution states, positions) and `runBatch` (one decode token per slot:
  the matrices through the matrix-vector kernel, whose per-token sums match a
  single token's; convolution, delta rule, RoPE, cache and attention per token
  on its slot); conversation IDs on every message, bundled receive
  (`recv?all=1`) and bundled sends (`hiddens`, `tokens`); coordinator options
  `--conversations`, `--first-conversation`, `--batch`, `--relay-delay`, and a
  `throughput` and `conversations` section in the report. Smoke test (680M, 2
  conversations, batch, 8 tokens): completed. Gate and protocol frozen.
- 2026-11-07 00:59: Local correctness and screen (680M, code eafa7aa+dirty,
  `runs/experiments/DO-38/omarchy-20261107-005919-local/`, under watch.py,
  stall 900 s, max 3,600 s; all 32 runs finished). **Correctness: 124 of 124
  conversations equal their run alone**, both arms, relay delay 0 and 68 ms.
  Screen (relay 68 ms per hand-off, 64 tokens, 2 repeats; all four stages
  share one GPU, so throughput is capped by the sum of the stages' compute):
  arm I 2.28 tokens/s at N = 1, 3.59 / 6.28 / 7.89 / 7.86 at N = 2 / 4 / 6 / 8
  (3.46x at N = 6, the slowest conversation's median 1.75x); arm B no better
  (6.39 at N = 4, 6.3 at N = 6 and 8). Found: a batched step cost more than
  its tokens one at a time (2 tokens about 100 ms, a single token 32 ms):
  the reused matrix-vector kernel reads every token's inputs from global
  memory in every thread. Added a batch kernel (`batchShader`: the same
  per-thread work items, sums and reduction as the matrix-vector kernel, so
  the same per-token results, with each round's inputs staged once in
  workgroup memory for 16 rows); arm B's local runs repeated with it in
  `omarchy-20261107-013126-local-batch2/`.
- 2026-11-07 01:31: The staged batch kernel's first rerun (`…-013126-local-batch2/`)
  failed on every batched run: 20 KiB of workgroup memory, over the 16 KiB
  default limit (labeled failed; the self-test now runs the batched decode
  path, so a device that cannot is refused at admission). Fixed and rerun
  (`…-local-batch3/`): 108 of 108 conversations equal, but arm B slower
  still (2.55x at N = 4, 2.0x at N = 6). Kernel benchmark on the 680M
  (`…-kernels-default`, `…-kernels-batch`), ffn_up, 1 / 2 / 4 / 8 tokens:
  matrix-vector 0.82 / 1.99 / 2.13 / 3.90 ms, the staged kernel 0.82 / 3.65 /
  3.65 / 7.20 ms. On this GPU a decode matrix product is bound by
  dequantization arithmetic, not by memory (34 GB/s), so batching saves
  little work here; the staged kernel was removed and arm B uses the
  matrix-vector kernel in groups of 4 tokens (the code of the first local
  run). On the rented RTX 3060s a stage's step took 40–65 ms for weights
  readable in about 3 ms (experiment 92), so per-call overhead dominates
  there, which batching shares; the remote run decides arm B. Check with the
  final code (`…-final-check/`): self-test passed, 8 batched conversations
  with relay 68 ms, 8 of 8 equal to their runs alone, 6.81 tokens/s.
- 2026-11-07 01:02 UTC: Remote run approved by the user: budget $2,
  every machine deleted 2 hours after creation (watchdogs at `pods up`). Code
  bc2ffd1. Renumbered from DO-37 (that ID was taken upstream at the same time,
  for the ConvNet migration); runs and pools renamed (`runs/experiments/DO-38/`,
  `do38gpu`, `do38coord`). Driver `runs/experiments/DO-38/drive_remote.py`:
  the 30 timed runs first, then the 7 single-conversation references.
- 2026-11-07 01:02–01:19 UTC: Pool (`pods up`, watchdogs at 03:01 UTC): Vast.ai
  RTX 3060 12 GB x 4 in Poland (one site, one public address), $0.075/h each;
  DigitalOcean s-4vcpu-8gb in ams3 coordinator. w0 failed its WebGPU probe
  twice: the first host (Poland) and its replacement (Romania, driver
  595.91.07) had no NVIDIA EGL in the container (`libEGL_nvidia.so.0` and its
  GLVND vendor file missing), so Chrome could not create a GPU context; both
  deleted by recorded ID and replaced by an RTX 3060 in Poland (driver
  595.99.02, like the others). Warm-up session (8 tokens): weights downloaded
  in 114–179 s per stage, 593 ms per token. The 37 runs started at 01:19 UTC
  under watch.py (stall 900 s, max 5,700 s), output
  `runs/experiments/DO-38/rented-20261107/`.
- 2026-11-07 01:19–01:40 UTC: All 37 runs finished (the timed 30, then the 7
  single-conversation references); none failed. Pools deleted at 01:41 UTC
  with `pods down` by recorded ID; `pods ls` shows 0. Cost about $0.62 at
  listed prices (four RTX 3060s and the coordinator for 100 min, plus about
  11 min of the two failed w0 hosts); Vast.ai storage and bandwidth not
  included.

## Results

Steady total throughput (tokens arriving between the moment every
conversation has its first token and the moment the first one ends, over that
window) and the slowest conversation's median time per token, median of 3
repeats; ratios against N = 1 of the same arm (`analyze.py`):

| Arm | N | Tokens/s | x N = 1 | Slowest conversation, ms per token | x N = 1 | Mean batch (largest stage) | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| I | 1 | 1.74 | 1.00 | 573 | 1.00 | 1.00 | |
| I | 2 | 4.24 | 2.44 | 508 | 0.89 | 1.00 | |
| I | 4 | 9.15 | 5.27 | 439 | 0.77 | 1.00 | pass |
| I | 6 | 9.51 | 5.47 | 641 | 1.12 | 1.00 | pass |
| I | 8 | 9.71 | 5.59 | 816 | 1.43 | 1.00 | pass |
| B | 1 | 1.74 | 1.00 | 575 | 1.00 | 1.00 | |
| B | 2 | 4.28 | 2.46 | 507 | 0.88 | 1.00 | |
| B | 4 | 8.91 | 5.11 | 621 | 1.08 | 1.43 | pass |
| B | 6 | 11.58 | 6.65 | 579 | 1.01 | 2.03 | pass |
| B | 8 | **16.17** | **9.29** | 610 | 1.06 | 1.69 | pass |

- **Correctness.** Every conversation in every multi-conversation run, both
  arms (121 conversations, 12,098 tokens), generated exactly the tokens it
  generated alone on the same machines (conversation 0 against an N = 1 run,
  1–7 against `single-c1`…`single-c7`). Locally on the 680M: 124 of 124, and
  108 of 108 with the removed kernel, 8 of 8 with the final code.
- **Arm I saturates at about 9.5–9.7 tokens/s** from N = 4: the stages are
  then never idle, and the slowest conversation's latency grows (816 ms at
  N = 8).
- **Arm B keeps scaling** to 16.2 tokens/s at N = 8 (13.2–17.4 across the 3
  repeats), with the slowest conversation at 610 ms, about one conversation's
  latency.
- **Per-stage costs** (N = 8): one decode token 22–41 ms of compute per
  stage, a batch of 2 tokens 47–62 ms, of 3 tokens 48–63 ms; each stage waited
  a median 60–80 ms per step for its next messages. With one conversation, a
  token took 60–89 ms of compute per stage and the stages waited 470–500 ms.

## What it taught

- **The pipeline was idle, not slow.** One conversation leaves each GPU
  waiting about 85% of the time; interleaving fills it. Two copies of every
  layer would double throughput; interleaving on one copy gives 5.6x, and
  batching 9.3x.
- **Batching gained in the relay, not in the GPUs.** On the RTX 3060s a batch
  of 2 decode tokens costs about twice one token's compute (47–62 ms against
  22–26), so batching saved little GPU work. Its gain is in transport: the
  outputs of several conversations travel as one message through the
  coordinator (`hiddens`, `tokens`), and the relay's per-message cost, not
  compute, limits arm I.
- **A busy GPU is a faster GPU.** One token's compute per stage fell from
  60–89 ms with one conversation to 22–41 ms with eight. A GPU that idles
  between tokens drops its clocks; that is why each conversation got faster
  as others joined (573 → 439 ms at N = 4 in arm I).
- **The 680M screen underestimated the gain** (3.46x against 5.6x and 9.3x):
  all four stages share one GPU there, so throughput is capped by the sum of
  the stages' compute; and on the 680M a decode matrix product is bound by
  dequantization arithmetic (34 GB/s), so batching saved nothing there.
- **A staged batch kernel was slower** (input rounds in workgroup memory with
  two barriers each: 3.65 ms for 2–4 tokens against 1.99–2.13 for the
  matrix-vector kernel) and was removed. Batching uses the matrix-vector
  kernel in groups of 4 tokens, whose per-token sums equal a single token's,
  which is why batching keeps every token exactly.
- **Rented containers can lack NVIDIA's EGL.** Two of five Vast.ai hosts had
  no `libEGL_nvidia.so.0` in the container, so Chrome had no GPU context; the
  setup's WebGPU probe caught both before any session.

## Limits

- Short prompts (24–32 tokens) and 128 tokens per conversation; long
  prompts, whose prefill would occupy a stage for seconds, were not mixed in.
- All four GPUs at one site in Poland, the coordinator in Amsterdam: shorter
  links than experiment 92's Poland/Spain/Sweden mix (its single conversation
  was 580 ms per token, here 573). Throughput limited by the relay will
  differ on other links.
- The same RTX 3060 model on every stage; experiment 92's slower Quadro RTX
  4000 would saturate earlier.
- Conversations all start together and run to 128 tokens; arrivals and
  departures over time were not tested. The chat page (`--chat`) still serves
  one conversation.
- 3 repeats per point; arm B's N = 8 throughput ranged 13.2–17.4 tokens/s.

## Reproduce

Workstation (correctness and screen; the GGUF in `runs/models/qwen3.5-9b/`):

```bash
.venv/bin/python -m unittest tests.test_qwen35_session
bash runs/experiments/DO-38/omarchy-20261107-005919-local/drive.sh runs/experiments/DO-38/<new-dir>
.venv/bin/python runs/experiments/DO-38/analyze.py runs/experiments/DO-38/<new-dir>
# one run by hand:
.venv/bin/python -m ygaraty.experiments.qwen35_session local --gguf runs/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf \
  --config runs/models/qwen3.5-9b/config.json --tokenizer runs/models/qwen3.5-9b/tokenizer.json --out DIR \
  --conversations 8 --batch --relay-delay 68 --generate 64
```

Rented (about $0.37/h; watchdogs delete the pools):

```bash
YGARATY_VAST_ENV=NVIDIA_DRIVER_CAPABILITIES=all python -m ygaraty.experiments.pods up do38gpu --provider vast \
  --location PL --server any-8gb --workers 3 --worker-size any-8gb --minutes 120 --max-hourly 0.8 --no-collect
python -m ygaraty.experiments.pods up do38coord --provider digitalocean --location ams3 --server s-4vcpu-8gb \
  --workers 0 --minutes 120 --max-hourly 0.3 --no-collect
python -m ygaraty.experiments.pods ship do38coord && python -m ygaraty.experiments.pods ship do38gpu
# coordinator: bash /root/remote.sh qwen35-coordinator-setup; each GPU host: bash /root/remote.sh qwen35-contributor-setup
python runs/experiments/DO-38/drive_remote.py runs/experiments/DO-38/<out> --only warmup --generate 8
python runs/experiments/DO-38/drive_remote.py runs/experiments/DO-38/<out>
python runs/experiments/DO-38/analyze.py runs/experiments/DO-38/<out>
python -m ygaraty.experiments.pods down do38gpu && python -m ygaraty.experiments.pods down do38coord && python -m ygaraty.experiments.pods ls
```
