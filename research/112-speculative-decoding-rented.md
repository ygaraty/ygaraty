# On rented GPUs, exactly checked speculative decoding cuts edits 65% and code 73%, bit for bit; a summary ran 4% slower without drafting

| | |
|---|---|
| **Conclusion** | **Failed gate, narrowly:** edits −65%, code −73%, bit for bit, but summary +4.1% with no drafts sent (measured, Vast.ai). |
| **Status** | executed |
| **ID** | DO-25 (number added on finish) |
| **Category** | transport |
| **Where** | remote |
| **Date** | Added 2026-11-05; rewritten 2026-11-09; run 2026-11-10 |
| **Path** | [Experiment 97](097-qwen35-webrtc-direct-links.md)'s pipeline: Qwen3.5-9B Q4_K_M in headless Chrome, two stages of 16 layers (`--splits 0,16,32`), direct WebRTC links (`--route webrtc`), on two Vast.ai RTX 3060 12 GB; coordinator on a DigitalOcean s-4vcpu-8gb in `ams3`. Correctness first on the workstation with S5-nano. Not on the simulation-only machine |
| **Result** | **Correctness, admission, memory and both primary lines met; one of four "never slower" lines missed.** Medians of 3 runs, ms per token, plain → drafting: edit 281 → 99 (−65%, line −30%), code 291 → 78 (−73%), held-out rename 285 → 129 (−55%), chat 288 → 282 (−2%), notes 287 → 276 (−4%), summary 286 → 298 (+4.1%, line +3%). The summary's drafting arm sent no draft (1.00 tokens per pass): its loss is stage A's single-token compute (74, 74, 57 ms against 64–69 in plain runs; 50–70 across this host's plain runs). Tokens and logits digests equal plain decoding's in all 36 runs; 12-case self-test on both hosts; peak 2.78 / 3.05 GB per stage. Stage A an RTX 3060 in Poland, stage B an RTX 3060 Ti in Norway (protocol amendment), direct link 58 ms RTT. About $0.43 |
| **Evidence** | Measured: [evidence/browser/speculative-decoding-rented-2026-11-10.json](evidence/browser/speculative-decoding-rented-2026-11-10.json) (per-run times, medians, drafting record, single-token compute per run, links, gate lines); raw reports in `runs/experiments/DO-25/remote-20261110-1047/` (not in git); `runs/experiments/DO-25/drive_remote.py`, `gate.py`. Code e346637; the drafting rule is [experiment 111](111-cautious-cost-aware-drafting.md)'s |

## Question

Each generated token costs one pass through both stages and back. On experiment 97's pipeline that pass takes 83 ms per token. Most of it is compute (stage A 36–38 ms, stage B 39–40 ms per step), and the direct link adds only 2–3 ms. Decoding one position at a time reads every weight once per token. Prefill shows how much cheaper many positions per pass are: an 8,192-token prompt takes 24.3 s, about 3 ms per position.

Speculative decoding proposes k draft tokens and checks them in **one** pass of k + 1 positions. With greedy checking, the accepted tokens are the tokens plain greedy decoding gives. Every accepted token beyond the first saves a whole pass.

**With exactly checked speculative decoding, how much lower is one conversation's time per token on experiment 97's pipeline, with the same tokens? And how much of the gain survives when eight conversations are batched?**

Drafts, cheapest first:
- **N-gram (prompt lookup) drafts.** The next k tokens after the longest earlier match of the last n tokens, in the prompt and the text so far. No model, no traffic; strong when the answer repeats its input (edits, summaries, code with names).
- **The model's own multi-token prediction (MTP) head, if the checkpoint has one.** Experiment 92 left the "speculative/MTP paths" out, and the GGUF loader ignores "extra vision/MTP tensors". If Qwen3.5-9B's GGUF carries an MTP head, it drafts with the same tokenizer and no second model. It sits after the last layer, so stage B can draft on its own, with no extra crossing. Check its presence first (`ygaraty model inspect`); if absent, this arm is dropped, not replaced.

Batching (experiment 93) already fills a pass with eight conversations' tokens, so speculation should gain less there. That is the third question.

## The part specific to Qwen3.5: rolling back recurrent state

A rejected draft must be undone. For full-attention layers that means truncating the KV cache to the accepted length. Qwen3.5's Gated DeltaNet layers instead carry a recurrent state and a convolution state, which every token updates in place. To undo a rejection, each such layer keeps its state as it stood after each of the k + 1 positions of a verify pass, and restores the snapshot at the accepted length. The snapshots' memory must fit under the 4 GB cap, measured per stage.

## Prediction

- **N-gram drafts, k = 4, one conversation:**
  - 1.5–2.5 tokens accepted per pass on edit and summary prompts, under 1.2 on open chat;
  - a pass of 5 positions costs 1.1–1.4× a single-position pass (decode is bound by reading weights);
  - so time per token 30–50% lower on edit and summary prompts, under 10% on chat.
- **MTP drafts, if present:** 1.6–2.2 tokens per pass on every prompt kind, chat included, because the head predicts from the model's own state.
- **Eight conversations batched:** the gain falls to under 15%, because a batched pass already does more arithmetic per weight read.
- **Memory:** the DeltaNet snapshots for k = 4 add under 100 MB per stage.

## Gate

Frozen 2026-11-10 10:22 CET, before any rented run; it replaces the proposal above (never frozen), after the
workstation findings in the Log. Contingent on WS-20 passing its own gate first; if WS-20 fails, nothing is
rented.

Setup: experiment 110's (two Vast.ai RTX 3060 hosts on different machines, direct WebRTC links, coordinator with
STUN/TURN on a DigitalOcean s-4vcpu-8gb in `ams3`), Qwen3.5-9B Q4_K_M, `--splits 0,16,32`, one conversation, 64
tokens, the default kernel (`--matvec 2,4,8`), `--logit-digests` in every run. Arms per prompt: T1 plain
decoding; S3 `--draft ngram:n=3,k=8,min=2,adapt=cost` (WS-20's rule). Prompts
`runs/experiments/DO-25/prompts/{edit,summary,chat,code,rename,notes}.txt` (sha256 prefixes c63c93a4, 48eb8c1e,
bd24e7c1, be791a4b, 6c31debf, b91d5954), passed with `--prompt-file`. Three repeats, order T1 S3 / S3 T1 / T1 S3,
prompts in that order. An arm's value is the median of its three runs' (last − first token) / 63.

1. **Correctness (required):** every run's 64 tokens and logits digests equal the first T1 run's of its prompt.
2. **Admission and memory (required):** every contributor passes the self-test; peak application GPU memory
   per stage under 4.0 GB.
3. **Primary:** S3 ≤ 0.70 × T1 on edit and code.
4. **Never slower:** S3 ≤ 1.03 × T1 on summary, chat, rename and notes (1.03, not WS-20's 1.02: network
   jitter on rented links).

All lines pass → **Works**. Computed by `runs/experiments/DO-25/gate.py`. Not run, and why: the MTP arm (no MTP
tensors in the GGUF), a fixed-k sweep (the cost rule picks the number of drafts per pass; fixed k lost 2–3× on
summaries locally), and speculation inside batched decoding (not built: a batched step carries one token per
conversation).

## Protocol

1. **Inventory (free).** List the Qwen3.5-9B GGUF's tensors and metadata. Record whether MTP tensors are present, their shapes, and the head's input (final hidden state, embedding of the last token, or both).
2. **Code:**
   - a verify pass of k + 1 positions per stage. It is a short prefill chunk, so it reuses the prefill kernels: decode-shaped for the full-attention cache, chunked for the DeltaNet layers.
   - DeltaNet state snapshots per position and their restore; KV truncation for full-attention layers;
   - greedy acceptance at stage B, which returns the accepted tokens and the next draft;
   - draft sources: `--draft ngram:n=3,k=4` and, if present, `--draft mtp:k=…`;
   - batching: speculation per conversation inside experiment 93's batched step.
3. **Workstation (S5-nano, two local contributors):**
   - speculative output equals plain greedy output for random prompts and every k;
   - after a rejection, every stage's caches and DeltaNet states equal those of a session that never saw the rejected tokens, bit for bit.
4. **Prompts:**
   - experiment 93's 8 chat prompts;
   - 2 edit prompts ("rewrite this paragraph with one change");
   - 2 summary prompts over a 1,024-token passage;
   - 128 new tokens each.

   The new prompts are fixed in the evidence file.
5. **Rented runs (as frozen in the Gate):** experiment 110's setup; a rehearsal (S3 on edit, 16 tokens), then
   `runs/experiments/DO-25/drive_remote.py OUT` (36 runs; a browser launch whose ssh times out is retried once)
   and `runs/experiments/DO-25/gate.py OUT`.

## Trigger

On the workstation:
```bash
git pull && pip install -e . && python -m unittest tests.test_qwen35_speculative     # equals greedy; state rollback
.venv/bin/python -m ygaraty.experiments.qwen35_session local --gguf runs/models/s5-nano/S5-nano-Q4_K_M.gguf \
  --config runs/models/s5-nano/config.json --tokenizer runs/models/s5-nano/tokenizer.json --out DIR \
  --splits 0,16,32 --generate 32 --route webrtc --draft ngram:n=3,k=4
```

Rented (experiment 97's commands; watchdogs delete the pools):
```bash
YGARATY_VAST_ENV=NVIDIA_DRIVER_CAPABILITIES=all python -m ygaraty.experiments.pods up do25gpu --provider vast \
  --location PL --server any-8gb --workers 3 --worker-size any-8gb --minutes 150 --max-hourly 0.8 --no-collect
python -m ygaraty.experiments.pods up do25coord --provider digitalocean --location ams3 --server s-4vcpu-8gb \
  --workers 0 --minutes 150 --max-hourly 0.3 --no-collect
python -m ygaraty.experiments.pods ship do25gpu && python -m ygaraty.experiments.pods ship do25coord
# coordinator: bash /root/remote.sh qwen35-coordinator-setup && bash /root/remote.sh turn-serve
# each GPU host: bash /root/remote.sh qwen35-contributor-setup; pick two hosts near the coordinator
python runs/experiments/DO-25/drive_remote.py OUT/main --hosts server w1 \
  --drafts none,ngram:n=3,k=2,ngram:n=3,k=4,ngram:n=3,k=8,mtp --repeats 3
python runs/experiments/DO-25/drive_remote.py OUT/batched --hosts server w1 --conversations 8 --batch \
  --drafts none,ngram:n=3,k=4 --repeats 3
python runs/experiments/DO-25/analyze.py OUT
python -m ygaraty.experiments.pods down do25gpu && python -m ygaraty.experiments.pods down do25coord && python -m ygaraty.experiments.pods ls
```

To build: `--draft`, the verify pass, the DeltaNet snapshots, `tests.test_qwen35_speculative`, and `drive_remote.py`/`analyze.py` (from DO-40's).

## Cost and limits

- **Cost:** about $1–1.50 of Vast.ai and DigitalOcean time, like experiment 97 ($0.75). Confirm a budget and teardown deadline before provisioning.
- **Limits:**
  - Greedy decoding only; sampling needs speculative rejection sampling, a separate question.
  - Acceptance depends on the prompts, so the result is per prompt kind.
  - One host pair.

## Log

- 2026-11-05: Added, for the Phi-4 PyTorch pipeline of experiments 81–82, from a review of recent inference research (speculative decoding; BloomBee's pruned draft trees).
- 2026-11-09: Rewritten for the current pipeline: Qwen3.5-9B on two WebGPU stages with direct WebRTC links (experiment 97, 83 ms per token). Added the MTP-head arm, the DeltaNet state rollback, and the batched arm (experiment 93). Not run.
- 2026-11-09 21:30 CET: inventory (Protocol step 1): the Qwen3.5-9B config declares one MTP layer, but
  unsloth's Q4_K_M GGUF has none (427 tensors, blocks 0–31, no `mtp`/`nextn` tensors or metadata): **the
  MTP arm is dropped**. S5-nano's vocabulary is identical to Qwen3.5-9B's (248,044 entries).
- 2026-11-09 21:40–21:55 CET: workstation code and screen (not the gate; no remote run). Built: `--draft
  ngram:n=N,k=K[,exact=0]` and `--logit-digests` in `qwen35_session`; `speculative.mjs` (drafts, acceptance,
  the first stage's group tracking); in the stage, a checking pass of 1 + k rows (`heads: "all"`), one
  DeltaNet snapshot per pass and, after a rejection, restore plus replay of the kept rows at the start of
  the next pass (at most k + 1 rows; no per-position snapshots). `tests/webgpu/speculative.test.mjs`: plain
  greedy's tokens and the stage state after every pass, over 200 random prompts × k ∈ {1, 2, 4, 8}.
  Screen on the Radeon 680M, both stages on the one GPU, relay route, 64 tokens
  (`runs/experiments/DO-25/q9-*`, prompts in `runs/experiments/DO-25/prompts/`):

  | Prompt (Qwen3.5-9B) | Arm | Tokens per pass | Drafts kept | ms per token | Tokens; logits |
  |---|---|---|---|---|---|
  | edit | plain | 1 | — | 186 | — |
  | edit | k = 4, exact | 3.50 | 45/56 | 174 | same; bit-identical |
  | edit | k = 4, GEMM | 3.50 | 45/56 | 154 | same; rounding differs |
  | edit | k = 8, GEMM | 4.85 | 50/68 | 115 (−38%) | same; rounding differs |
  | summary (740 tokens) | plain | 1 | — | 221 | — |
  | summary | k = 8, exact | 1.37 | 17/247 | 630 | same; bit-identical |
  | summary | k = 8, GEMM | 1.37 | 17/247 | 452 | same |
  | chat | plain | 1 | — | 185 | — |
  | chat | k = 8, exact | 1.00 | 0/117 | 398 | same; bit-identical |
  | chat | k = 8, GEMM | 1.00 | 0/117 | 333 | same |

  S5-nano (code prompt, 32 tokens): k = 4 exact gave the same tokens and bit-identical logits digests.
  Snapshots added 26–35 MB per stage (prediction "under 100 MB" holds for one snapshot per pass).
- **Findings that change the plan:**
  1. **Correctness is solved exactly.** With the matrix-vector kernel at every row count (`exact`), a
     checking pass reproduces plain decoding bit for bit, logits included, through every rollback. The
     proposed near-tie excuse (margin 0.5) is unnecessary for the exact arm; the gate can require identical
     logits digests.
  2. **The premise "a pass of k + 1 positions costs about one decode step" is false with today's kernels.**
     Stage compute per pass grows almost linearly with rows: 9B, 59/79 ms for 1 row, 277/362 ms for 5
     (exact), 210–314 ms for 5–9 rows on the prefill GEMM. Experiment 93 measured the same on the RTX 3060s
     (2 rows about 2× one). The matrix-vector kernel shares dequantization but re-reads the input per row;
     the small-row tiled kernel is no cheaper; only the 16+-row GEMM is efficient per row, with a high
     fixed cost.
  3. **So speculation only pays at very high acceptance:** −38% on the edit prompt (4.85 tokens per pass), but
     2× slower on chat (no draft kept) and 2–3× slower on the summary, where the n-gram drafts mostly come
     from 1-token matches.
- **Next, before any rented run:** (a) draft only from matches of at least 2–3 tokens and stop drafting after
  rejections (adaptive k), so chat and summary fall back to plain decoding at no cost; (b) the real lever is a
  multi-row decode kernel whose cost is nearly flat to 8–16 rows (it would also make batching save GPU work,
  experiment 93); (c) then freeze the gate with the exact arm (identical logits digests) and measure on the
  RTX 3060 pair, where the pass-cost ratio may differ from the 680M.
- 2026-11-09 22:06 CET: adaptive drafting (`ngram:n=3,k=8,min=2,adapt=1`: drafts only from matches of 2+
  tokens; all kept: 2 more, up to k; any rejected: 1 fewer; at 0, none for 16 tokens). Same screen:

  | Prompt | Plain | Adaptive, exact | Adaptive, GEMM | Fixed k = 8 (exact / GEMM) |
  |---|---|---|---|---|
  | edit | 186 | 163 (−12%), 3.94 tokens per pass | 117 (−37%) | — / 115 |
  | summary | 221 | 260 (+18%), 1.11 | 242 (+10%) | 630 / 452 |
  | chat | 185 | 187 (no drafts) | 185 | 398 / 333 |

  ms per token; tokens identical in every run, logits bit-identical on the exact arm. Chat no longer pays for
  drafting; the summary's loss falls from 2–3× to 10–18%. The kernel question moved to [WS-15](104-matvec-rows-per-thread.md).
- 2026-11-09 22:12 CET: [experiment 104](104-matvec-rows-per-thread.md) (WS-15): a matrix-vector kernel with
  R rows per thread keeps outputs bit-identical and brings 8 rows to 1.3–1.7× one row (R = 4, VT = 8), against
  3.5–4.3× today; it failed its frozen gate only because R = 4 slows single tokens. A checking pass of up to 8
  rows on it would cost about 1.5–2× a decode step instead of about 4–5×.
- 2026-11-09 22:22 CET: [experiment 105](105-matvec-policy-in-stages.md) (WS-16): with experiment 104's kernel
  chosen by token count, adaptive exact drafting gave edit −50%, summary +11%, chat −4% against plain decoding,
  logits bit-identical in every run. Remaining before a rented run: summaries (drafting that backs off on
  alternating hits and misses), and the contributors' self-test covering the new kernel.
- 2026-11-09 22:53 CET: [experiment 107](107-speculative-backoff-rerun.md) (WS-17): with a faster back-off and
  admission on the new kernel, 3 runs per arm: edit −32%, summary +4.6%, chat −3.7%; plain decoding −3.5 to
  −3.7% with `--matvec 2,4,8`; identical logits in all 27 runs. Next: a cost-aware drafting rule (draft only
  when recent acceptance pays for the longer pass).
- 2026-11-10 00:10 CET: [experiment 109](109-cost-aware-drafting.md) (WS-19): cost-aware drafting (`adapt=cost`)
  gave edit −64%, held-out code edit −73%, chat unchanged, summary +15% against plain on the same kernel; logits
  bit-identical in 36 runs. Next: a pessimistic acceptance prior and the replay's rows in the cost.
- 2026-11-10 10:22 CET: gate frozen for the rented run (see Gate), with WS-20's revised rule and the rows-per-thread
  kernel as the default. Budget $1, teardown 2 h (approved by the user). Runs only if WS-20 passes.
- 2026-11-10 10:37 CET: WS-20 passed its gate; code e346637 pushed. Pools do25gpu (Vast.ai, two RTX 3060 12 GB,
  machines 20938 and 17972, Poland, behind one public IP) and do25coord (DigitalOcean s-4vcpu-8gb, `ams3`);
  $0.22/h in all; watchdogs delete both at about 12:37. Setup started.
- 10:47–10:52: setup done (e483d3c… e346637 on all hosts; TURN on the coordinator). Rehearsal (S3 on edit, 16
  tokens) on the two Polish hosts: finished, but they share one public IP (192.0.2.3), so their "direct"
  link ran through the coordinator's TURN server (srflx–relay, 64 ms RTT), about 250 ms per plain token against
  experiment 110's 104. Kept as `edit/R3-1-turn-pair`; not experiment 110's setup. All three Polish offers on
  Vast.ai were behind that IP.
- **Protocol amendment, 11:05, before any main run:** stage B moves to a Vast.ai RTX 3060 Ti 8 GB in Norway
  (pool do25b, machine of offer 155565, public IP different from stage A's); stage A stays on the Polish RTX 3060
  (machine 20938). The gate, arms, prompts and order are unchanged. The link is direct (srflx–srflx) but 58 ms
  RTT, not experiment 110's 1–2 ms, and the stages are a 3060 and a 3060 Ti: absolute times are not comparable
  with experiment 110; the arm ratios are what the gate reads. The idle Polish host w0 stays in its pool until
  teardown (pools are deleted whole). Rehearsal on the new pair: finished, 16 tokens, 12-case self-test on both,
  link opened in under 1 s.
- 11:04–11:58: 36 runs under the watchdog (stall 600 s, limit 85 min), all finished; no launch retry needed.
- 11:59: gate computed: failed on one line (summary +4.1% against 1.03). Pools deleted (`down --no-fetch`; the
  reports were already fetched by the driver); `pods ls`: 0 machines. Cost about $0.43 (do25gpu $0.15/h and
  do25coord $0.07/h for about 1.4 h, do25b $0.12/h for about 1 h).

## Results

Median of 3 runs, ms per token (each run in brackets), stage A RTX 3060 (Poland), stage B RTX 3060 Ti (Norway),
direct link 58 ms RTT:

| Prompt | T1 plain | S3 drafting | S3 tokens per pass | Drafting in S3's first run |
|---|---|---|---|---|
| edit | 281 [284, 278, 281] | **99** [99, 95, 99] (−65%) | 3.32 | 8 passes with drafts, 44 of 56 kept |
| code | 291 [291, 294, 280] | **78** [78, 80, 78] (−73%) | 4.50 | 8 passes, 49 of 55 kept |
| rename | 285 [285, 316, 282] | **129** [129, 200, 128] (−55%) | 2.74 | 11 passes, 40 of 68 kept |
| chat | 288 [288, 291, 288] | 282 [282, 280, 284] (−2%) | 1.00 | none (no matches) |
| notes | 287 [287, 295, 283] | 276 [270, 276, 328] (−4%) | 1.07 | 2 passes, 4 of 12 kept |
| summary | 286 [286, 283, 290] | 298 [298, 298, 282] (**+4.1%**) | 1.00 | none sent |

Gate (`gate.json`): bit-exactness, admission and memory pass on every prompt; edit and code pass (≤ 0.70);
chat, rename and notes pass (≤ 1.03); **summary fails** (1.041).

Single-token compute on stage A, median per run (ms): summary T1 69, 65, 64; summary S3 74, 74, 57. Across all
plain runs on this host: 50–70. Drafting runs' single-token steps were faster (33–40 ms on edit and code),
consistent with the GPU clocking down while it waits through the 58 ms link between steps.

## What it taught

- **On real links, the gain is larger than on one local GPU.** Each plain token pays a round trip of the 58 ms
  link; a pass that returns 3–4.5 tokens pays it once: edits −65% and code −73% here, against −61% and −69% in
  experiment 111. The rule measured this pair's pass costs itself; nothing was tuned for it.
- **Correctness holds end to end on two different GPUs.** Logits digests identical to plain decoding's in all
  36 runs, across a 3060 and a 3060 Ti, with replays inside the checking pass.
- **The miss is not drafting.** The summary's drafting runs sent no draft, so their passes were plain steps;
  the 4% is stage A's single-token compute, which varies about ±15% between plain runs on this host. A 3%
  tolerance at 3 runs per arm was tighter than this host's run-to-run spread; a rerun would need more repeats
  or a tolerance set from measured spread, frozen in advance.
- **Vast.ai's Polish 3060s now share one public IP.** Two of them cannot link directly (hairpin NAT): the link
  ran through TURN at 64 ms RTT. Check the hosts' public IPs before a run that needs direct links.

## Limits

- One host pair, a 3060 and a 3060 Ti, 58 ms apart; experiment 110's 1–2 ms pair was not available, so absolute
  times are not comparable with it.
- 64 tokens, six prompts, one conversation; greedy decoding only. Batched speculation and the MTP arm were not
  run (see Gate).

## Reproduce

Pools as experiment 110 (check that the two GPU hosts have different public IPs), then:
```bash
python runs/experiments/DO-25/drive_remote.py OUT --hosts POOL_A:HOST POOL_B:HOST
python runs/experiments/DO-25/gate.py OUT
```
