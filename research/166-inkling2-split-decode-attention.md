# Splitting single-token attention over position blocks halves Inkling 2 Mini's A100 decode at 1k and cuts it to a third at 8k, with the same tokens

| | |
|---|---|
| **Conclusion** | Works, now default: split attention 34.7 vs 65.9 ms per token at 1k, 36.8 vs 107.4 at 8k (measured, A100) |
| **Status** | executed |
| **ID** | DO-101 (report 166) |
| **Category** | browser |
| **Where** | workstation (680M: correctness against NumPy and report 156's answers); remote (one Vast.ai A100 SXM4 40 GB) |
| **Priority** | high |
| **Date** | Added 2026-11-24; started 2026-11-24; finished 2026-11-24 |
| **Path** | `inkling2_gpu.mts` `FAST_DECODE.attn` (default on): for one-token steps, `ATTN_SPLIT` (one workgroup per head and 128-position block: block max, sum and weighted values) then `ATTN_MERGE` (one workgroup per head: the blocks plus the zero sink logit). Prefill keeps `ATTENTION`. Against `attn=0` (report 165's code) on the same A100 |
| **Result** | Item 1 passes: 64 greedy tokens identical to report 156's arm A on all 5 prompts; math check within 3.1e-5 (bound 2e-3). Item 2 passes: per-token time after the 1k prompt (median of 39 steady steps, alternating twice) **34.4 and 34.9 ms against 65.7 and 66.0**, ratio **0.526** (bound ≤ 0.75). At 8k: 36.8 against 107.4 ms (0.343). GPU per step at 1k 30.4 ms (from 62.8 in report 165); attention is now 5.6 ms of it. Tokens identical between arms in every timing run |
| **Evidence** | measured-remote: `runs/experiments/DO-101/vast-do101b-20261124/` (`gate.json` by `tools/gate.py`; `do101/` timing and GPU profiles; DO-98 gate outputs); local: `omarchy-20261124-split/` (`check/`, `check-long/`) |

## Question

Report 165 (DO-99) found that at 1k positions Inkling 2 Mini's A100 decode step is GPU-bound: 63 ms of GPU per step, of which the attention kernel is 38 ms over 48 dispatches. `ATTENTION` runs one 128-thread workgroup per head for a decoded token, which walks every position in blocks of 128 with two tree reductions per block: few workgroups on a 108-SM GPU, and a serial walk that grows with positions (report 162: 75 ms per token at 1k, 110 at 8k). Does spreading the blocks over workgroups (flash-decoding) cut time per token, with the same tokens?

## Prediction

Attention drops from 38 ms to a few ms per step at 1k, so the GPU step falls to about 28–30 ms. Host encoding (about 31 ms per step, report 165) then sets the pace: about 35–40 ms per token at 1k (about 0.55× report 165's 68.9 ms). At 8k the gain is larger (report 162: 110 ms).

## Gate (frozen 2026-11-24, before any A100 timing)

1. **Same tokens:** 64 greedy tokens identical to report 156's arm A on all 5 prompts; math check (forced routing, last-position logits) within 2e-3 of NumPy on all 5.
2. **Faster:** per-token time after the 1k prompt on the same A100 (run + read, median of steady steps, alternating `attn=1`/`attn=0` twice, as report 165), new at most 0.75× old. Reported: 8k the same way, GPU per kernel at 1k, first-step time.

## Protocol

1. Workstation: synthetic models against NumPy, the tiny GGUF (on/off tokens), and DO-96's `check.sh` on chat and long (2,761 positions, so many blocks) on the 680M.
2. One rented A100 (DO-98's kit plus DO-99's tools): the timing above, then the 5-prompt gate (`a100.sh gate`). Budget and teardown deadline from the user first.

Related: [165 (DO-99)](165-inkling2-decode-attention-not-host.md), [162 (DO-98)](162-inkling2-fewer-dispatches-a100.md), DO-100 (direct links, which gains when tokens are faster).

## Log

- 2026-11-24: Added at the user's request from report 165 ("Add, run before DO-100").
- 2026-11-24: Implemented `ATTN_SPLIT`/`ATTN_MERGE` behind `FAST_DECODE.attn`. Synthetic models 9/9 (within 1.3e-5); tiny GGUF fully resident, 200 steps (positions to 227, two blocks on global layers): identical 201 tokens on and off. Gate frozen above. Real-model check on the 680M (chat, long) running in `omarchy-20261124-split`.
- 2026-11-24: **680M checks.** DO-96's `check.sh` (`omarchy-20261124-split/check`): synthetic models 9/9; chat forced, token by token (so every step uses the split kernel), last logits within 1.4e-5, worst 3.6e-5, 27/27, 0 routing flips; 64 greedy chat tokens as arm A. The long prompt's greedy run hit the 680M's known prefill limit at residency 6–17 ("Not enough memory for command submission", in prefill, which this change does not touch); rerun at residency 8–15 (`tools/long_check.sh`, `check-long/`): 64 greedy tokens as arm A, 2,761 positions, so up to 22 blocks.
- 2026-11-24: User's budget $1.50, 2 h. First A100 (`do101a`, Vast.ai 62362714, US) downloaded Node at 2 MB/s, so it was deleted by its recorded ID after about 24 minutes (about $0.23; my Hugging Face speed test there lacked `-L` and measured only the redirect). Second A100 (`do101b`, 62366085, Iowa, $0.457/h, 126 MB/s from Hugging Face): GGUF in 5 minutes (sha256 OK), timing and gate in 17 minutes, deleted by its recorded ID after 0.31 h (about $0.14). About **$0.37** in all.
- 2026-11-24: **Gate** (`tools/gate.py vast-do101b-20261124`): 1 PASS (5/5 prompts identical; forced last logits 2.4e-5 to 3.1e-5). 2 PASS (0.526). The A100's forced checks prefill each whole prompt, so they do not run the split kernel; the greedy runs and the 680M's token-by-token check do.

## Results

| After the prompt | `attn=0` (ms per token) | `attn=1` (ms per token) | Ratio |
|---|---:|---:|---:|
| 1k positions (two runs each) | 65.7, 66.0 | 34.4, 34.9 | 0.526 |
| 8k positions | 107.4 | 36.8 | 0.343 |

Medians of steady steps (run + read, step 1 excluded). First steps (pipeline compilation) 160–650 ms in both arms. GPU at 1k (pass profile): 30.4 ms per step. Per kernel (detail mode): `ink_mix` 5.7 ms, `ink_attn_split` 5.6, `ink_ctrl_mix` 3.6, expert and matrix kernels about 3 each.

## Conclusion

The decode attention kernel was the A100's limit at long context, as report 165 found. Spreading the 128-position blocks over workgroups makes time per token nearly flat in context length: 34.7 ms at 1k and 36.8 at 8k, against 65.9 and 107.4 before. Against report 156's one-GPU 160 ms per token at 1k, Inkling 2 Mini's decode on an A100 is now 4.6 times faster. Now GPU (about 30 ms) and host encoding (about 31 ms, report 165) are about equal, so the next gains need both: fewer dispatches or cheaper encoding, and the remaining small kernels.

## Limits

- One A100 host for timing; two runs per arm at 1k and one at 8k.
- Block size fixed at 128 positions; not tuned. The merge reads all blocks serially per head (up to 67 at 8k), which is cheap at these sizes.
- Prefill still uses `ATTENTION` (one workgroup per head and token), which is already parallel over tokens.

## Reproduce

Workstation: `bash runs/experiments/DO-96/tools/check.sh OUT chat`, `bash runs/experiments/DO-101/tools/long_check.sh OUT`; `INK_FAST=attn=0` (via `DO-96/tools/flags.mjs`) selects the old kernel.

A100: DO-98's kit plus DO-99's `host_profile.mjs` and `decode_profile.mjs` and `DO-101/tools/a100.sh`; `DO-96/tools/a100.sh setup`, `DO-101/tools/a100.sh timing`, `DO-101/tools/a100.sh gate`, then `python3 runs/experiments/DO-101/tools/gate.py RUN_DIR`.
