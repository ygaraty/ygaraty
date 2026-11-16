# Cost-aware drafting cuts edits 64% and a held-out code edit 73%, bit for bit, but still loses 15% on a summary

| | |
|---|---|
| **Conclusion** | **Failed gate:** edits −64%, held-out code −73%, chat unchanged, but summary +15% (measured, workstation, 3 runs per arm). |
| **Status** | executed |
| **ID** | WS-19 |
| **Category** | browser |
| **Where** | workstation (Radeon 680M, Chrome WebGPU) |
| **Date** | Added 2026-11-09 |
| **Path** | As experiment 107 (WS-17): Qwen3.5-9B Q4_K_M, two stages on the one GPU, relay route, 64 tokens, `--matvec 2,4,8`; the drafter `ngram:n=3,k=8,min=2,adapt=cost` (`speculative.mjs` `CostModel`) |
| **Result** | **Eleven of twelve lines met; summary missed.** Medians of 3 runs, ms per token (T0 → T1 → S2): edit 187 → 180 → 67 (−64%, line −35%), held-out code 187 → 180 → 51 (−73%; 5.25 tokens per pass), chat 186 → 179 → 180, summary 188 → 179 → 206 (S2 1.15× T1; line 1.02×). Tokens and logits digests equal T0's in all 36 runs; every contributor passed the 12-case self-test |
| **Evidence** | Measured: [evidence/browser/cost-aware-drafting-2026-11-09.json](evidence/browser/cost-aware-drafting-2026-11-09.json) (per-run times, medians, tokens per pass, drafting record, gate lines); raw reports in `runs/experiments/WS-19/omarchy-20261109-2349/` (not in git); `runs/experiments/WS-19/run.sh`, `gate.py`; tests `tests/webgpu/speculative.test.mjs` |

## Question

Experiments 105 and 107 showed that a fixed back-off rule trades edit gains for summary losses. A rule that
measures each pass's time by its rows and the acceptance of each draft position (per match length), and drafts
the number that maximizes expected tokens per millisecond (none when that is plain decoding), should keep both.
**Is speculative decoding with it faster on edits and code edits, and never slower than plain decoding on the
same kernel on summaries and chat?**

## Prediction

Edit −40 to −50% against today's plain; held-out code edit −20 to −40%; summary and chat within ±2% of plain
on the same kernel. Simulated (`tests/webgpu/speculative.test.mjs`): 1.03× plain when drafts rarely hold, 0.47×
when hits and misses alternate, 0.25× when they always hold.

## Gate

Frozen 2026-11-09 23:43 CET, before any GPU run of the rule. The rule was designed after seeing experiments
105 and 107's edit, summary and chat results; the code prompt (`runs/experiments/DO-25/prompts/code.txt`,
153 tokens, sha256 prefix be791a4b) is held out: written now, never run before.

Arms per prompt: T0 plain, today's kernels; T1 plain, `--matvec 2,4,8`; S2 `--matvec 2,4,8 --draft
ngram:n=3,k=8,min=2,adapt=cost`. Three repeats, order per prompt T0 T1 S2 / S2 T0 T1 / T1 S2 T0, prompts edit,
summary, chat, code in turn. An arm's value is the median of its three runs' (last − first token) / 63.

1. **Bit-exactness (required):** every run's 64 tokens and logits digests equal the first T0 run's of its prompt.
2. **Edit:** S2 ≤ 0.65 × T0.
3. **Never slower:** S2 ≤ 1.02 × T1 on summary, chat and code.
4. **Admission:** every contributor passes the self-test.

All lines pass → **Works**: the rule and kernel go to DO-25's rented run. Computed by
`runs/experiments/WS-19/gate.py`.

## Protocol

`runs/experiments/WS-19/run.sh OUT PYTHON PROFILES`, then `runs/experiments/WS-19/gate.py OUT`.

## Log

- 2026-11-09 23:43 CET: added; gate frozen. Rule in `speculative.mjs` (`adapt: "cost"`), its tests, and
  `--draft ...,adapt=cost`. Waiting for WS-18 to free the GPU.
- 23:49: runs started, `runs/experiments/WS-19/omarchy-20261109-2349`.
- 2026-11-10 00:10 CET: 36 runs finished; gate computed: failed on the summary line only.

## Results

Median of 3 runs, ms per token (each run in brackets):

| Prompt | T0 plain, today | T1 plain, `--matvec 2,4,8` | S2 cost-aware | S2 tokens per pass |
|---|---|---|---|---|
| edit | 187 [188, 186, 187] | 180 [180, 180, 182] | **67** [67, 67, 67] (−64%) | 3.94 |
| code (held out) | 187 [187, 187, 187] | 180 [180, 180, 179] | **51** [51, 52, 51] (−73%) | 5.25 |
| chat | 186 [187, 186, 185] | 179 [179, 179, 179] | 180 [180, 179, 180] | 1.00 |
| summary | 188 [188, 188, 188] | 179 [179, 180, 179] | 206 [206, 211, 201] (+15% on T1) | 1.12 |

Drafting in the first S2 run of each prompt: summary 45 passes without drafts, 11 with 6–7 drafts (7 kept in
all) and 11 replays; edit 8 passes with 7 drafts (46 kept); code 8 passes with 6–7 (51 kept).

## What it taught

- **Measured cost makes long drafts pay where they hold:** with passes of up to 8 rows costing about one
  group of the new kernel, edits and code edits gain 64–73%, against 32–50% for the earlier rules.
- **The summary loss comes from an optimistic start and an uncounted replay.** Every draft position starts at
  a 50% acceptance estimate (one of two pseudo-counts), so the first passes draft 7; each rejection updates only
  the first failed position and forces a replay of the kept rows into the next pass, which the cost model does
  not count. Eleven failed 7-draft passes and their replays cost the 15%. Fixes to test: a pessimistic prior
  (start short and grow), and the replay's rows in the expected cost.
- The held-out code prompt behaved like the edit prompt, so the gain is not specific to the prompt the rule was
  designed on.

## Limits

- Both stages on one 680M; relay route; four prompts of 64 tokens.

## Reproduce

`runs/experiments/WS-19/run.sh OUT PYTHON PROFILES`, then `python runs/experiments/WS-19/gate.py OUT`.

