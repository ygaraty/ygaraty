# Cautious cost-aware drafting keeps edits 61% and code 69% faster, bit for bit, and stops losing on summaries

| | |
|---|---|
| **Conclusion** | **Works:** summary +0.6%, edits −61%, code −69%, held-out rename −50%, bit for bit (measured, workstation, 3 runs per arm). |
| **Status** | executed |
| **ID** | WS-20 |
| **Category** | browser |
| **Where** | workstation (Radeon 680M, Chrome WebGPU) |
| **Date** | 2026-11-10 |
| **Path** | As experiment 109 (WS-19): Qwen3.5-9B Q4_K_M, two stages on the one GPU, relay route, 64 tokens, the rows-per-thread kernel (now the default, `--matvec 2,4,8`); the revised drafter `ngram:n=3,k=8,min=2,adapt=cost` |
| **Result** | **Every gate line met.** Medians of 3 runs, ms per token, plain → revised drafting: edit 179.7 → 70.0 (−61%, line −35%), code 180.1 → 55.6 (−69%), summary 179.9 → 181.0 (+0.6%, line +2%), chat 179.8 → 179.4, held-out rename 181.0 → 90.4 (−50%), held-out notes 180.6 → 184.1 (+1.9%, on the line). Tokens and logits digests equal plain decoding's in all 36 runs; every contributor passed the 12-case self-test. Against experiment 109's rule: the summary's +15% is gone; edits gave back 3 ms (67 → 70) |
| **Evidence** | Measured: [evidence/browser/cautious-cost-aware-drafting-2026-11-10.json](evidence/browser/cautious-cost-aware-drafting-2026-11-10.json) (per-run times, medians, tokens per pass, drafting record, the offline simulation's predictions, gate lines); raw reports in `runs/experiments/WS-20/omarchy-20261110-1015/` (not in git); `runs/experiments/WS-20/run.sh`, `gate.py`. Code e346637 (`speculative.mjs`, `qwen35_contributor.mjs`); tests `tests/webgpu/speculative.test.mjs` (8, including the pessimistic start) |

## Question

Experiment 109's cost-aware rule cut edits 64% but lost 15% on a summary. Its acceptance estimates started at
50% per draft position, and a pass after a rejection reran the kept rows as a separate stage run, which the rule
did not count. **With a pessimistic start, acceptance learned from drafts that are never sent, and the replay
inside the checking pass, is the summary no slower than plain decoding while edits keep most of their gain, bit
for bit?**

Changes against 109 (`speculative.mjs`, `qwen35_contributor.mjs`):
- **Shadow drafts.** At every opportunity the drafter records the n-gram drafts it found, sent or not, and
  scores each position against the tokens that follow. Acceptance comes from every opportunity at no GPU cost.
- **Pessimistic prior.** Position 1 starts at 25% (2 pseudo-observations); each later position is shrunk
  towards the previous one's estimate instead of a fixed 50%. No exploration probes.
- **The replay inside the pass.** After a rejection, every stage restores the state saved before the rejected
  pass and runs `[kept rows | token | drafts]` as one run (at most k + 1 rows). In 109 the kept rows were a
  run of their own: 166 ms on stage A for 3 replayed rows plus the token, against 116 ms for one run of 8 rows.
- **The replay's cost in the rule.** Drafts that may fail are charged the next pass's extra group of rows.

An offline simulation on 109's recorded greedy tokens and measured run times (the real `Drafter` in node; it
reproduces experiment 107's rule within 2 ms) predicted: edit 75, code 58, summary 180, chat 180 ms per token,
against 180 plain; with ±10% jitter on pass times, the summary's worst of 20 seeds 182.7.

## Prediction

Edit −55 to −65%, code −60 to −70%, summary and chat within ±2% of plain; the held-out rename prompt gains
like code; the held-out notes prompt sends few or no drafts.

## Gate

Frozen 2026-11-10 10:14 CET, before any GPU run of the revised rule. The rule was designed on 109's four
prompts; two prompts are held out, written now and never run before:
`runs/experiments/DO-25/prompts/rename.txt` (sha256 prefix 6c31debf) and `notes.txt` (b91d5954).

Arms per prompt: T1 plain decoding (the default kernel); S3 `--draft ngram:n=3,k=8,min=2,adapt=cost`. Three
repeats, order T1 S3 / S3 T1 / T1 S3, prompts edit, summary, chat, code, rename, notes in turn. An arm's value is
the median of its three runs' (last − first token) / 63.

1. **Bit-exactness (required):** every run's 64 tokens and logits digests equal the first T1 run's of its prompt.
2. **Edits:** S3 ≤ 0.65 × T1 on edit and code.
3. **Never slower:** S3 ≤ 1.02 × T1 on summary, chat, rename and notes.
4. **Admission:** every contributor passes the self-test.

All lines pass → **Works**: the rule goes to DO-25's rented run. Computed by `runs/experiments/WS-20/gate.py`.

## Protocol

`runs/experiments/WS-20/run.sh OUT PY PROFILES` (36 runs, about 25 minutes), under the experiments skill's
watchdog; then `gate.py OUT`.

## Log

- 2026-11-10 10:14 CET: added and gate frozen; run.sh and gate.py written.
- 2026-11-10 10:15–10:40 CET: rehearsal (edit, 32 tokens, S3 and T1: same tokens and digests; replays inside the
  pass). Then the 36 runs in `runs/experiments/WS-20/omarchy-20261110-1015/` under the watchdog (stall 300 s,
  limit 1 h): all finished. Code e483d3c+dirty (speculative.mjs 292d207c, qwen35_contributor.mjs 79648ac6).
  Gate passed: edit −61%, code −69%, rename −50%, summary +0.6%, chat −0.3%, notes +2.0% (on the line);
  identical tokens and digests in all 36 runs; self-test 12 cases on every contributor.

## Results

Median of 3 runs, ms per token (each run in brackets):

| Prompt | T1 plain | S3 revised drafting | S3 tokens per pass | Drafting in S3's first run |
|---|---|---|---|---|
| edit | 179.7 [180, 180, 179] | **70.0** [69, 70, 70] (−61%) | 3.50 | 9 passes with drafts, 45 of 57 kept, 2 replays |
| code | 180.1 [180, 180, 180] | **55.6** [56, 56, 55] (−69%) | 4.50 | 8 passes, 49 of 55 kept, 1 replay |
| rename (held out) | 181.0 [181, 181, 183] | **90.4** [90, 90, 91] (−50%) | 2.74 | 11 passes, 40 of 69 kept, 6 replays |
| summary | 179.9 [181, 180, 180] | 181.0 [181, 181, 181] (+0.6%) | 1.00 | none sent |
| chat | 179.8 [180, 179, 180] | 179.4 [179, 180, 179] (−0.2%) | 1.00 | none (no matches) |
| notes (held out) | 180.6 [181, 180, 181] | 184.1 [179, 185, 184] (+1.9%) | 1.17 | 6 passes, 9 of 32 kept, 6 replays |

All gate lines pass (`gate.json`); the notes line passes at 1.019 against 1.02.

## What it taught

- **Learning acceptance from unsent drafts removes the cost of finding out.** On this kernel any draft makes a
  pass about 1.7× a plain step, so one failed probe on the summary would already cost 2.3% of the run. Scoring
  the drafts the rule would have sent, against the tokens that follow, costs nothing: the summary sent no draft
  at all and ran at plain speed (109: +15%).
- **The replay belongs inside the pass.** In 109 a rejection's kept rows ran as a separate stage run (166 ms on
  stage A for 3 rows plus the token, against 116 ms for one run of 8 rows). Restoring the saved state and running
  `[kept rows | token | drafts]` as one run kept every logits digest identical.
- **An offline simulation of the drafter is a good screen.** Driving the real `Drafter` with recorded greedy
  tokens and measured run times reproduced 107's rule within 2 ms and predicted this rule's edit, code, summary
  and chat within 5 ms; it also showed that 109's summary loss depended on timing noise near break-even.
- **What is left: prompts whose drafts hold about a third of the time.** The held-out notes prompt drafted in 6
  passes and kept 9 of 32 drafts: +1.9%, on the line. A stricter break-even margin would trade some of the edit
  gain for it.

## Limits

- Both stages on one 680M, relay route, 64 tokens; six prompts. The rule was designed on four of them (two held
  out). The pass-cost ratios are this GPU's; the rule measures them per conversation.
- One run order; the notes line passed by 0.1%.

## Reproduce

```bash
bash runs/experiments/WS-20/run.sh OUT .venv/bin/python PROFILES   # 36 qwen35_session local runs, about 25 minutes
python runs/experiments/WS-20/gate.py OUT
node --test tests/webgpu/speculative.test.mjs
```
