# A faster draft back-off brings summaries to +4.6% but costs edits a third of their gain; every logit identical in 27 runs

| | |
|---|---|
| **Conclusion** | **Failed gate:** edits −32%, chat −4%, but summaries +4.6% with speculative decoding (measured, workstation, 3 runs per arm). |
| **Status** | executed |
| **ID** | WS-17 |
| **Category** | browser |
| **Where** | workstation (Radeon 680M, Chrome WebGPU) |
| **Date** | Added 2026-11-09 |
| **Path** | As [experiment 105](105-matvec-policy-in-stages.md) (WS-16): Qwen3.5-9B Q4_K_M, two stages (0,16,32) on the one GPU, relay route, `qwen35_session local`, 64 tokens, DO-25's edit, summary and chat prompts. Changed since 105: the drafter backs off to the number of drafts just kept (not one fewer) and doubles its pause while retries fail; admission's self-test runs the session's `--matvec` kernel, with two new cases (8 and 9 tokens) |
| **Result** | **Eleven of twelve lines met; summary speculative missed.** Medians of 3 runs, ms per token (T0 → T1 → S1): edit 187 → 180 → 127 (−32%, line −25%), summary 187 → 180 → 195 (+4.6%, line 0%), chat 186 → 179 → 179 (−3.7%). Tokens and logits digests equal T0's in all 27 runs; every contributor passed the 12-case self-test, run on the session's kernel. Repeats agreed within ±1 ms. Against experiment 105, the new back-off cut the summary's loss (+11% → +4.6%) and the edit's gain (−50% → −32%; 3.94 → 2.03 tokens per pass) |
| **Evidence** | Measured: [evidence/browser/speculative-backoff-rerun-2026-11-09.json](evidence/browser/speculative-backoff-rerun-2026-11-09.json) (per-run ms per token, medians, tokens per pass, gate lines); raw reports in `runs/experiments/WS-17/omarchy-20261109-2237/` (not in git); `runs/experiments/WS-17/run.sh`, `gate.py`. Logic tests `tests/webgpu/speculative.test.mjs` (alternating hits and misses: adaptive wastes 24 drafts against fixed k's 457) |

## Question

Experiment 105 failed its gate: edits −50% with speculative decoding, but summaries +11%, and plain decoding
only −4% (line −5%), from one run per arm. With a drafter that backs off faster when hits and misses
alternate, and three runs per arm, is speculative decoding with `--matvec 2,4,8` faster on edits and no slower
on summaries and chat, with plain decoding no slower and every logit identical?

## Prediction

Edit about −45%; summary within −5 to +3%; chat about −4%; plain −3 to −5%.

## Gate

Frozen 2026-11-09 22:38 CET, before any run.

Arms per prompt: T0 plain, today's kernels; T1 plain, `--matvec 2,4,8`; S1 `--matvec 2,4,8 --draft
ngram:n=3,k=8,min=2,adapt=1`. Three repeats, order per prompt T0 T1 S1 / S1 T0 T1 / T1 S1 T0, prompts edit,
summary, chat in turn within each repeat. Time per token: (last token time − first) / 63; an arm's value is
the median of its three runs.

1. **Bit-exactness (required):** every run's 64 tokens and logits digests equal the first T0 run's of its
   prompt.
2. **Speculative:** S1 ≤ 0.75 × T0 on edit; S1 ≤ 1.00 × T0 on summary and on chat.
3. **No regression:** T1 ≤ 1.00 × T0 on each prompt. (Experiment 105's −5% line was a speed prediction for
   the kernel alone; this experiment asks whether the combination can be adopted. T1 against T0 is reported.)
4. **Admission:** every run's contributors pass the self-test, which in T1 and S1 runs the new kernel.

All lines pass → **Works**: `--matvec 2,4,8` with adaptive drafting is ready for DO-25's rented run.
Computed by `runs/experiments/WS-17/gate.py`.

## Protocol

As experiment 105's, with `runs/experiments/WS-17/run.sh OUT` (the order above) and
`runs/experiments/WS-17/gate.py OUT`.

## Log

- 2026-11-09 22:38 CET: added; gate frozen. Drafter change and self-test cases in place; S5-nano smoke
  run with `--matvec 2,4,8` (`runs/experiments/DO-25/smoke-selftest-2235`): self-test passed on both stages
  with the 12 cases (worst 5.7e-7), session finished.
- 22:37: runs started, `runs/experiments/WS-17/omarchy-20261109-2237`.
- 22:53: 27 runs finished; gate computed: failed on the summary's speculative line only.

## Results

Median of 3 runs, ms per token (each run in brackets):

| Prompt | T0 plain, today | T1 plain, `--matvec 2,4,8` | S1 speculative | S1 tokens per pass |
|---|---|---|---|---|
| edit | 187 [187, 188, 187] | 180 [180, 179, 180] (−3.7%) | **127** [127, 126, 127] (−32.3%) | 2.03 |
| summary | 187 [188, 187, 186] | 180 [180, 179, 180] (−3.5%) | **195** [196, 194, 195] (+4.6%) | 1.02 |
| chat | 186 [186, 186, 186] | 179 [178, 179, 180] (−3.7%) | 179 [179, 179, 179] (−3.7%) | 1.00 |

Every run gave T0's tokens and logits digests; every contributor passed the self-test (12 cases, on the new
kernel in T1 and S1).

## What it taught

- **The new kernel is safe to adopt for plain decoding:** −3.5 to −3.7% on all three prompts, bit-identical,
  and admitted by a self-test that runs it.
- **One back-off rule cannot serve both edits and summaries.** Backing off to the drafts just kept stops
  long accepted runs early (edit: 3.94 → 2.03 tokens per pass), while a summary still pays for occasional
  1-draft passes (1.02 tokens per pass, +4.6%). A rule that weighs a pass's measured cost against the expected
  number of kept drafts (draft only when the recent acceptance rate makes a longer pass pay) is the next step.
- **Three repeats were enough here:** runs agreed within ±1 ms, so the ±5% lines are measurable on this
  setup.

## Limits

- One GPU with both stages; relay route; three prompts, 64 tokens each.

## Reproduce

`runs/experiments/WS-17/run.sh OUT PYTHON PROFILES`, then `python runs/experiments/WS-17/gate.py OUT`.

