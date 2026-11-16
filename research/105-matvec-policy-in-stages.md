# Rows per thread by token count halve edit time with speculative decoding, bit for bit, but plain decode gains only 4% and summaries lose 11%

| | |
|---|---|
| **Conclusion** | **Failed gate:** bit-exact and edits −50%, but plain decode only −4% and summaries +11% (measured, workstation). |
| **Status** | executed |
| **ID** | WS-16 |
| **Category** | browser |
| **Where** | workstation (Radeon 680M, Chrome WebGPU) |
| **Date** | Added 2026-11-09 |
| **Path** | Qwen3.5-9B Q4_K_M, two stages (0,16,32) on the one GPU, relay route, `qwen35_session local`, 64 tokens; today's kernels against `--matvec 2,4,8` (experiment 104's kernel: 2 rows per thread for one token, 4 rows and up to 8 tokens otherwise); plain decoding and DO-25's adaptive drafting (`ngram:n=3,k=8,min=2,adapt=1`, exact) |
| **Result** | **Bit-exactness met; two of the three speed lines missed.** All 12 runs gave T0's 64 tokens and the same logits digest for every token. Plain decoding with the new kernel: 180 / 179 / 178 ms per token against 187 / 187 / 186 (−4% on each prompt; line −5%). Speculative with the new kernel: edit 94 ms (−50%, 3.94 tokens per pass; line −25%), summary 208 ms (+11%; line 0%), chat 179 ms (−4%). With today's kernels the same drafting gave −13% / +36% / 0% |
| **Evidence** | Measured: [evidence/browser/matvec-policy-in-stages-2026-11-09.json](evidence/browser/matvec-policy-in-stages-2026-11-09.json) (per-arm ms per token, tokens per pass, allocations, gate lines); raw reports in `runs/experiments/WS-16/omarchy-20261109-2215/` (not in git); gate `runs/experiments/WS-16/gate.py` |

## Question

[Experiment 104](104-matvec-rows-per-thread.md) failed its gate because no single variant was fast at both one
and eight tokens; choosing the variant by token count (decided after seeing its numbers) would pass. In whole
stages, does that choice make plain decoding faster and DO-25's checking passes cheap enough that speculative
decoding gains on edits without losing on summaries and chat, with the same logits bit for bit?

## Prediction

Plain decoding 8–15% faster (the products are most of a step; attention, DeltaNet and the head unchanged).
Speculative, edit: −35 to −50% against today's plain; summary and chat within ±5%.

## Gate

Frozen 2026-11-09 22:15 CET, before any run with `--matvec`.

Arms per prompt (DO-25's fixed edit, summary and chat prompts): T0 plain, today's kernels; T1 plain,
`--matvec 2,4,8`; S1 adaptive exact drafting with `--matvec 2,4,8`; S0 adaptive exact drafting, today's
kernels. Order T0 T1 S1 S0 (edit), S0 S1 T1 T0 (summary), T0 T1 S1 S0 (chat). Time per token: (last token
time − first) / 63, from the report.

1. **Bit-exactness (required):** in every arm, the 64 tokens equal T0's and every emitted token's logits digest
   equals T0's.
2. **Plain decode:** T1 ≤ 0.95 × T0 on each of the three prompts.
3. **Speculative:** S1 ≤ 0.75 × T0 on the edit prompt; S1 ≤ 1.00 × T0 on the summary and on chat.

All three lines pass → **Works**; the policy becomes a candidate default once the contributors' self-test
covers the new kernel. Otherwise Failed gate. Computed by `runs/experiments/WS-16/gate.py`.

## Protocol

```bash
S="python -m ygaraty.experiments.qwen35_session local --gguf $M/Qwen3.5-9B-Q4_K_M.gguf --config $M/config.json \
   --tokenizer $M/tokenizer.json --splits 0,16,32 --generate 64 --logit-digests"
$S --prompt "$(cat runs/experiments/DO-25/prompts/edit.txt)" --out OUT/edit/T0                       # T0
$S ... --matvec 2,4,8 --out OUT/edit/T1                                                              # T1
$S ... --matvec 2,4,8 --draft ngram:n=3,k=8,min=2,adapt=1 --out OUT/edit/S1                          # S1
$S ... --draft ngram:n=3,k=8,min=2,adapt=1 --out OUT/edit/S0                                         # S0
python runs/experiments/WS-16/gate.py OUT
```

## Log

- 2026-11-09 22:15 CET: added; gate frozen. `--matvec SINGLE,ROWS,TOKENS` in `qwen35_session` (fixture
  `matvec`, set on the contributor's `Gpu`); `Gpu.matvec.single` in `webgpu_kernels.mts`.
- 22:15: runs started, `runs/experiments/WS-16/omarchy-20261109-2215`.
- 22:22: twelve runs finished in the frozen order; gate computed: failed (plain −4% on each prompt; summary
  speculative +11%).

## Results

ms per token over 64 tokens (change against T0):

| Prompt | T0 plain, today | T1 plain, `--matvec 2,4,8` | S1 speculative, `--matvec 2,4,8` | S0 speculative, today |
|---|---|---|---|---|
| edit | 187 | 180 (−4%) | **94 (−50%)**, 3.94 tokens per pass | 163 (−13%) |
| summary | 187 | 179 (−4%) | 208 (+11%), 1.11 | 254 (+36%) |
| chat | 186 | 178 (−4%) | 179 (−4%), no drafts | 187 (0%) |

Tokens and logits digests equal T0's in every run (bit-exactness line met on all three prompts).

## What it taught

- **The new kernel makes speculative checking cheap enough to matter:** the same drafting went from −13% to
  −50% on the edit prompt and from +36% to +11% on the summary, with identical logits.
- **Plain decoding gains only 4% in a whole stage**, against up to 30% for one product in experiment 104: the
  R = 2 gain is largest on `ffn_up` and small on the others, and attention, DeltaNet, norms, the head and the
  relay's per-token overhead are unchanged.
- **Summaries still lose:** at 1.11 tokens per pass, drafts that fail cost more than the few that hold. The
  adaptive drafter backs off only after a rejection streak; summaries alternate short hits and misses.
- The plain summary took 187 ms here against 221 ms in DO-25's earlier screen: run-to-run drift on this
  shared GPU is about that size, so single runs per arm limit the precision of the ±5% lines.

## Limits

- One run per arm and prompt; both stages on one GPU; relay route.
- The contributors' self-test does not yet run the new kernel (it checks today's); the outputs are identical,
  but admission should test the kernel a contributor will use before it becomes a default.
- One host, the 680M; the RTX 3060's balance may differ.

## Reproduce

As the Protocol, with `runs/experiments/WS-16/gate.py OUT`.
