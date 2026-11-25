# Relay backpressure keeps a near-full-context prefill safe with 4-call mailboxes, at the same speed and with the same tokens

| | |
|---|---|
| **Conclusion** | Works, now default: 4-call relay mailboxes with backpressure, no drop, same tokens, no slowdown (prefill −3.6%, decode +0.3%; measured-local, tiny Inkling) |
| **Status** | executed |
| **ID** | WS-51 (report 157) |
| **Category** | distributed engine |
| **Where** | workstation (local coordinator and native contributors; delays emulated) |
| **Priority** | med |
| **Date** | Added 2026-11-22; started 2026-11-23; finished 2026-11-23 |
| **Path** | The coordinator relay (`_legacy/qwen_protocol.py`: `BoundedMailbox`, `/api/send`, `/api/recv`) and the contributor's `handOn` (`qwen35_contributor.mjs`) |
| **Result** | With each relay mailbox bounded at 4 calls (64–192 KB here) and stage 1 slowed 200 ms per prefill call, a 3,919-token prompt (context 4,096) answered with no 429, no lost stage and the baseline's exact 32 tokens; 19 sends waited and the fullest mailbox peaked at 81% of its bound, where the baseline's grew to 1.98 MB (15 times it). Without a slow stage, medians of 3 runs: prefill 0.911 s against 0.945 s (−3.6%), decode 25.18 against 25.11 ms per token (+0.3%). A 6 s-per-call stage made the coordinator answer "retry" 4 times; the contributor resent and the answer was unchanged. The Qwen3.5 S5-nano arm was not run (no model on this machine) |
| **Evidence** | measured-local: `runs/experiments/WS-51/omarchy-20261123-2215/` (`gate.json`; `nodrop/`, `same/`, `speed/`, `retry/`) |

## Question

In report 156's arm B, a long prompt dropped stage A with `429 Stage mailbox capacity reached`. The fast first stage pushed its whole prefill (1.2 MB per 64-token call on Inkling's `f32` wire) into stage B's 32 MiB mailbox before B read any. The fix sizes each mailbox for a whole-context prefill of its inputs: 158–315 MiB per stage for Inkling at 8k positions. That memory grows with context, stages and sessions, and a full mailbox still drops the contributor.

Can the relay apply backpressure instead, so a sender waits (or retries) when the next stage is behind and no mailbox ever needs a whole prefill?

Candidates:
- the server answers `send` with a retry-after while a mailbox is over a small bound, and the contributor retries the same call; or
- a credit window per stage pair: a sender holds at most N calls in flight, and `recv` returns credits.

## Prediction

A window of 2–4 calls keeps every stage busy (a call takes longer to compute than to relay), so prefill time is unchanged within 5%. Mailbox memory is bounded at N calls per stage, under 10 MB for Inkling. No stage is dropped on prompts up to the full context.

## Gate (frozen 2026-11-23, before any run)

Implemented variant: the coordinator holds a `send` of hidden states until the next stage's mailbox has room (at most 5 s), else answers `429` with `Retry-After: 0`, and the contributor sends the same numbered call again. The mailbox bound is `YGARATY_MAILBOX_CALLS` (default 4) of the stage's largest input calls; `0` is DO-95's code (whole-context mailboxes, no waiting) and is the baseline arm.

Setup: tiny `inkling2_loop` GGUF (`models/inkling-2-tiny/Inkling-2-tiny.gguf`, from `tiny_gguf.py`), bundle re-prepared at 4,096 positions with report 156's splits 0,2,4,6 and `f32` wire; local `ygaraty serve` and `ygaraty contribute --runtime node --num-workers 3` (Dawn, 680M); requests to `/v1/chat/completions`, streamed, greedy.

1. **No drop:** with the window at 4 calls and stage 1 slowed by 200 ms per prefill call (`?slowStage=1&slowMs=200`), a prompt of at least 3,800 tokens (near the 4,096 context) prefills and answers with no `429` drop, no stage lost and no session reset. Backpressure must actually engage (stage 1 or 2 mailbox `waits` > 0), and no mailbox's peak may exceed its 4-call bound plus one call.
2. **Same tokens:** each answer (32 greedy tokens, compared as decoded text and token count) is identical to the baseline arm's on the same prompt.
3. **Same speed:** without the slow stage, 3 runs per arm on the same machine, alternating arms, a fresh prompt per run pair (so no prefix cache): median prefill time (request to first streamed token) and median decode ms per token each within 5% of the baseline.

Not run: the Qwen3.5 S5-nano `f16` arm of the proposed gate. No S5-nano GGUF or bundle is on this machine, and inputs are not downloaded for a run (logged as a limit).

## Protocol

Implement the retry-after variant first (smallest change, compatible with WebRTC peers that bypass the relay). Local coordinator plus 3 native contributors on the workstation (as report 156's tiny-model session). Delay emulation: a contributor-side sleep per call, or `tc` on loopback. Then decide whether the per-context mailbox sizing (`mailbox_bytes`) can return to a small fixed bound.

Related: report 156 (DO-95, where the 429 appeared), report 092 (Qwen3.5 four contributors, f16 wire).

## Log

- 2026-11-22: Added after report 156.
- 2026-11-23: Implemented at 12f33ea+dirty (on top of DO-95's uncommitted tree): `BoundedMailbox.room`/`wait_room`, `SendBusy`, `mailbox_calls()` (`YGARATY_MAILBOX_CALLS`, default 4; 0 is DO-95's whole-context mailboxes), mailbox stats in `/api/status`; a busy send's number is no longer taken (before, a refused send could never be retried: `send` recorded `n` before the put). Contributor: `429` with `Retry-After` is resent at once, no deadline; `?slowStage=K&slowMs=N` for local tests. `tests/webgpu/test_relay_backpressure.py` (6 tests) passes.
- 2026-11-23: Gate frozen (above). Dawn `webgpu@0.6.2` reinstalled offline from the npm cache into `runs/tools/dawn` (report 156's copy was in tmpfs `/tmp`). Bundle re-prepared at 4,096 positions. Driver `runs/experiments/WS-51/tools/arm.py`, one serve + `contribute --runtime node --num-workers 3` per run.
- 2026-11-23: No-drop run under watch.py (stall 300 s, max 1,200 s): finished. Its own `log_429` field read 1, a false match on a client port (`:54290`); the driver's count was fixed and `gate.py` recounts from the logs (0). Then the batch (same-tokens baseline, 3 alternating speed pairs; watch.py stall 300 s, max 2,400 s): finished. Contributors "left session" at about 15 s in every arm: the driver's teardown after the answer.
- 2026-11-23: Extra, not gated: stage 1 slowed 6 s per call on a 609-token prompt to exercise the resend path (`retry/`).

## Results

All three gate items pass on the tiny Inkling arm (`gate.json`, computed by `tools/gate.py`). Bundle: `Inkling-2-tiny.gguf` (sha256 81364507…), splits 0,2,4,6, `f32` wire, chunk 64, context 4,096. Mailbox bounds at 4 calls: 64, 128 and 192 KiB for stages A, B and C (inputs of 1, 2 and 3 vectors of 64 floats).

| Gate | Backpressure (4 calls) | Baseline (DO-95 code) | Pass |
|---|---|---|---|
| 1. No drop, stage B slowed 200 ms per prefill call, 3,919-token prompt | 32 tokens; no 429, no lost stage, no error; stage B mailbox: 19 waits, peak 106,631 B of 131,072 (0.81) | (not gated) peak 1,983,632 B in stage B's mailbox, 32 MiB bound | Yes |
| 2. Same answers (text and token counts) | 4 of 4 prompts identical | — | Yes |
| 3. Prefill, median of 3 | 0.911 s (0.925, 0.911, 0.902) | 0.945 s (0.990, 0.896, 0.945) | Yes, −3.6% |
| 3. Decode, median of 3 | 25.18 ms/token (25.18, 25.46, 25.01) | 25.11 (25.10, 25.93, 25.11) | Yes, +0.3% |

Backpressure engages even with no slow stage: stage B (the loop core, 3 visits per layer) is slower than A at prefill, and A's sends waited 19 times per prompt in every speed run, without changing prefill time. With the slow stage, prefill took 13.19 s against the baseline's 13.11 s (+0.6%, one run each, not gated): the slow stage sets the pace either way.

Resend path (extra): stage B slowed 6 s per call, 609-token prompt. The coordinator answered `429 Retry-After: 0` 4 times; the contributor resent the same numbered calls, and the answer and prefill (60.20 s against 60.20 s) matched the baseline.

Memory at Inkling 2 Mini's size, from the same rule: 4 calls of 64 tokens × 4 vectors × 2,304 × 4 bytes = 9.4 MB for the largest stage's mailbox, instead of 158–315 MiB at 8k positions (report 156).

## Limits

- One tiny model on one machine (680M, Dawn): all three stages share the GPU, and relay hops are loopback. The Qwen3.5 S5-nano `f16` arm of the proposed gate was not run, since that model is not on this machine. Real links add latency per send, and a 4-call window may then be too small to keep a stage busy; WAN timing needs a remote run (DO-97 could carry it).
- Only the relay is bounded. The sender's own send chain is not: a fast stage still computes ahead and keeps its outputs in its own process (here about 2 MB). Bounding that needs the compute loop to wait on its sends.
- A send held by the coordinator occupies one of its request threads (128 by default) for up to 5 s; one per contributor at most, since each contributor's sends are serial.
- The emulated delay is a sleep before each prefill call, not slower kernels.
- WebRTC direct links bypass the relay and are unchanged.

## Reproduce

At 12f33ea plus this working tree (DO-95's uncommitted changes and WS-51's), on a machine with a WebGPU-capable GPU and Node:

```bash
D=runs/experiments/WS-51/<host>-<timestamp>
(mkdir -p runs/tools/dawn && cd runs/tools/dawn && echo '{"private":true}' > package.json && npm install webgpu@0.6.2)
ygaraty model prepare --gguf Inkling-2-tiny.gguf --config inkling-2-mini/hf/config.json \
  --tokenizer inkling-2-mini/hf/tokenizer.json --output $D/bundle-4k --splits 0,2,4,6 --context 4096 --generate 32
mkdir -p $D/keys  # owner.key and contributor.key: one random token each
python runs/experiments/WS-51/tools/prompts.py $D/bundle-4k/tokenizer.json $D/prompts
python runs/experiments/WS-51/tools/arm.py $D/bundle-4k/bundle.json $D/keys $D/nodrop/run --calls 4 --slow-ms 200 --prompt $D/prompts/nodrop.txt
bash runs/experiments/WS-51/tools/batch.sh $D
python runs/experiments/WS-51/tools/gate.py $D
```

`Inkling-2-tiny.gguf` comes from `runs/experiments/DO-95/tools/tiny_gguf.py` on the real GGUF. Input hashes: bundle.json f9cd6c39…, prompts nodrop 892d14e0…, speed1 4214b9dc…, speed2 ffa668a0…, speed3 cfc43331….
