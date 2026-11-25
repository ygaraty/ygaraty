# Counted on broad code, S5-nano's Portuguese-English-code vocabulary keeps all three: 10% smaller, faster decode, same answers

| | |
|---|---|
| **Conclusion** | Works: 121,937 of 248,070 tokens, GGUF −9.8%, bits/byte within +0.61% in all three, 8/8 answers same, decode −14% (measured, 680M) |
| **Status** | executed |
| **ID** | M-10 (report 176) |
| **Category** | browser |
| **Where** | this machine (Radeon 680M, S5-nano whole on the GPU) |
| **Priority** | med |
| **Date** | Added 2026-11-25; finished 2026-11-25 |
| **Path** | Report 175's pruning of S5-nano with the code count from `codeparrot/github-code-clean` (revision c48d40f9…; shard 0, sha256 c05e4cad…, up to 4 MB in each of 16 languages, 62 MB) instead of this repository; code held out from shard 1 (sha256 2e1b2e8b…, 497 files, 1.5 MB). Pruned GGUF `../models/s5-nano-m10/S5-nano-Q4_K_M.m10.gguf` (2.51 GB, sha256 2a0dcaf2…) |
| **Result** | Kept 121,937 tokens (49%); GGUF 2.78 → 2.51 GB (−272 MB, −9.8%). Item 1 passes: no fully-covered held-out word encodes differently (extra tokens: Portuguese +0.09%, English +0.13%, code +0.36%). Item 2 passes: bits per byte Portuguese 0.898 → 0.903 (+0.52%), English 0.790 → 0.795 (+0.61%), code 0.339 → 0.341 (+0.45%); top-1 agreement 99.83%, 99.93%, 99.85%. Item 3 passes: all 8 greedy answers (Portuguese, English, Python, JavaScript, Java, Go) identical. Item 4 passes: decode at about 1k positions 50.8 against 59.3 ms per token (medians of 47.0/54.5 and 55.7/62.8; −14.3%, bound −5%), about 20 against 17 tokens/s |
| **Evidence** | measured-local: `runs/experiments/M-10/` (`count/`, `qual/item2.json`, `serve/` answers and timing per run, `bundles/pruned`, `tools/corpus_code.py`) |

## Question

Report 175 (M-9) kept 80,036 of 248,070 tokens and passed for Portuguese and English, but failed on code (bits per byte +5.75%, top-1 agreement 97.97%, a JavaScript answer changed), with code counted on only 465 files of this repository. Counted on a broad code sample instead, does code pass while the file stays meaningfully smaller and decode faster?

## Prediction

TBD. More code tokens are kept (perhaps 90–110k in all), so the size and speed gain shrinks from 13% to perhaps 9–11%.

## Gate (frozen 2026-11-25, before running)

Report 175's gate, items unchanged, with: the code count from `codeparrot/github-code-clean` shard 0 (up to 4 MB per language, the 16 languages above) instead of this repository; code held-out text from shard 1 (a mixed sample, about 1.5 MB, files from the same 16 languages); the same Portuguese and English corpora and held-out texts as report 175; eight chat prompts (report 175's six plus "Write a Java method that checks whether a string is a palindrome." and "Write a Go function that sums a slice of integers.").

1. Tokenization unchanged for every fully-covered held-out word (Portuguese, English, code).
2. Per text (Portuguese, English, code): top-1 agreement ≥ 98% and bits per byte within +2% on a 4,096-token held-out prefix.
3. All eight greedy chat answers (64 tokens) identical to the full model's.
4. Decode at about 1k positions on the 680M at least 5% faster (medians of two alternating runs per arm).

## Protocol

As report 175 (`runs/experiments/M-9/tools`), with a code corpus script for the new source.

Related: [175 (M-9)](175-s5-nano-portuguese-english-code-vocabulary.md), [163 (M-8)](163-inkling2-english-vocabulary.md).

## Log

- 2026-11-25: Added at the user's request, from report 175.
- 2026-11-25: User: "run m10". Source chosen (ungated, pinned) and gate frozen above, before running.
- 2026-11-25: **Code corpus**: 16 languages from shard 0 (Python 4.46 MB, JavaScript 4.15, TypeScript 4.11, Java 4.01, C 4.03, C++ 4.02, C# 4.00, Go 4.03, Rust 2.85, PHP 4.04, Ruby 4.00, Shell 3.32, SQL 4.01, HTML 4.01, CSS 4.01, Markdown 4.01); Go is labelled `GO` in this dataset (the first pass missed it and was redone before counting). Count over Portuguese, English and code: 44.7M tokens over 166 MB, 135,668 distinct, kept 121,937 (seen twice, bytes, special, closure).
- 2026-11-25: **Items 1–4** as report 175's tools (`runs/experiments/M-9/tools/qual_s5.mjs`, `serve_run.sh` with `PROMPTS`/`NAMES`): see Result. Order full, pruned, pruned, full; the 1,049-token timing prompt has the same length in both tokenizations.
- 2026-11-25: **Gate**: 1 PASS, 2 PASS, 3 PASS, 4 PASS.

## Conclusion

Report 175's code failure came from counting code on a narrow sample. Counted on a broad 16-language sample, the kept vocabulary grows from 80k to 122k tokens and code behaves like the original (bits per byte +0.45%, every answer unchanged), while Portuguese and English stay within +0.61%. The model is 9.8% smaller and decodes faster on the 680M; the measured −14% exceeds what the smaller head alone explains (about −10% of bytes read) and the runs spread by 13–16%, so the real gain is probably near 10%. A pruned S5-nano is a reasonable default for Portuguese, English and code users, with a full-vocabulary fallback for other scripts.

## Limits

- One 4,096-token held-out sample per text; code held-out text from one shard of one dataset.
- Speed on the 680M only, two runs per arm with a wide spread; not timed on an A100 or in a multi-stage session.
- Scripts other than Latin (and emoji or rare symbols) now split into byte tokens; not measured.

## Reproduce

`runs/tools/m8venv/bin/python -I runs/experiments/M-10/tools/corpus_code.py SHARD0.parquet SHARD1.parquet runs/experiments/M-10/corpus`; then report 175's steps with `runs/experiments/M-10/count` (code list from `corpus/code_*.txt`, held-out `corpus/code_heldout.txt`) and `PROMPTS=runs/experiments/M-10/prompts.json NAMES=pt1,pt2,en1,en2,code1,code2,java1,go1,t1k bash runs/experiments/M-9/tools/serve_run.sh BUNDLE PORT OUT`.
