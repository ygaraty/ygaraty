# A Portuguese-English-code vocabulary makes S5-nano 13% smaller and about 13% faster, keeps Portuguese and English, but costs code

| | |
|---|---|
| **Conclusion** | Failed gate on code: bits/byte +5.75%, one code answer changed; Portuguese +0.85%, English +1.34%, decode −13% (measured, 680M) |
| **Status** | executed |
| **ID** | M-9 (report 175) |
| **Category** | browser |
| **Where** | this machine (Radeon 680M, S5-nano whole on the GPU) |
| **Priority** | med |
| **Date** | Added 2026-11-25; finished 2026-11-25 |
| **Path** | M-8's Kyse pruning on `maritaca-ai/S5-nano-GGUF` Q4_K_M (sha256 c7040c07…): kept 80,036 of 248,070 tokens (32%), rows of the tied `token_embd` (Q6_K) sliced, pruned tokenizer; GGUF 2.78 → 2.42 GB (−361 MB, −13%) |
| **Result** | Item 1 passes: no fully-covered held-out word encodes differently (extra tokens: Portuguese +0.16%, English +0.19%, code +2.7%). Item 2 **fails on code**: bits per byte Portuguese 0.898 → 0.906 (+0.85%), English 0.790 → 0.801 (+1.34%), code 0.502 → 0.531 (**+5.75%**, bound 2%); top-1 agreement 99.76%, 99.88%, **97.97%** (bound 98%). Item 3 **fails on code**: 5 of 6 chat answers identical; the JavaScript one starts "```js" instead of "```javascript". Item 4 passes: decode at about 1k positions 49.0 ms against 56.3 ms (medians, alternating runs; −13%, bound −5%), about 20 against 18 tokens/s |
| **Evidence** | measured-local: `runs/experiments/M-9/` (`count/` kept set and tokcheck, `qual/` logit runs and `item2.json`, `serve/` answers and timing per run, `bundles/`); pruned GGUF `../models/s5-nano-m9/S5-nano-Q4_K_M.m9.gguf` (sha256 3519ebbd…) |

## Question

S5-nano (Maritaca, Qwen3.5 architecture, 33 layers, hidden 2,560) carries a 248,320-token vocabulary, and its embedding doubles as the output head, read in full on every decoded token. Its users write Portuguese, English and code. If the vocabulary keeps only what those need (Latin scripts, code, digits, punctuation), how much smaller and faster does decoding get, and are the answers and text quality unchanged?

Builds on [163 (M-8)](163-inkling2-english-vocabulary.md): the same method on Inkling 2 Mini kept 30% of the tokens, made the file 4% smaller and decode about 7% faster with the same chat answers, but failed on English text (+3.4% bits per byte), because 15 rare words in a 4k-token sample split into more pieces. Here the head is a larger share of the model (18.8% against about 4%), so the speed gain should be larger, and the token count should come from a much larger sample so rare words keep their tokens.

## Prediction

TBD until the count. If about 80–100k tokens cover the three corpora, the file shrinks by about 330–350 MB (12%) and bandwidth-bound decode gets about 10% faster; Portuguese with accents and code identifiers are the risk for text quality.

## Gate (frozen 2026-11-25, before running)

Corpora: Portuguese (a public Portuguese Wikipedia sample plus a chat/instruction set; exact source TBD, pinned by hash), English (Kyse's corpus, as M-8), code (as M-8); each with a held-out part not used for counting.

1. **Tokenization:** on held-out text, every word whose pieces are all kept encodes exactly as before; report the share of held-out bytes that encode differently (expected only words containing dropped tokens).
2. **Text quality**, per language and for code, on held-out text through the GPU path: top-1 agreement with the full model ≥ 98% and bits per byte within +2% (M-8's bound).
3. **Answers:** greedy answers to a fixed set of chat prompts (Portuguese, English, code; DO-47's three included) identical to the full model's.
4. **Speed:** decode tokens/s at 1k positions on the same GPU (the 680M; an A100 if a rental is approved), pruned against full, alternating runs: at least 5% faster. Report file size and load time.

## Protocol

1. Fetch and pin the Portuguese corpus; reuse M-8's English and code. Count with S5-nano's tokenizer over the counting parts; choose the kept set by coverage with the BPE closure (M-8's fix).
2. Build the pruned GGUF and tokenizer; check items 1–3 on the 680M (`tokcheck.py`, `qual.mjs`).
3. Speed on the 680M (whole model resident), alternating full and pruned, then optionally an A100.

Related: [163 (M-8)](163-inkling2-english-vocabulary.md), [DO-42](DO-42.md) (S5-nano serving), DO-47 (S5-nano screen).

## Log

- 2026-11-25: Added at the user's request ("could we test s5-nano latin only (portuguese, english, coding) kyse prunning? and test tk s"). Counted from the GGUF: 248,320 tokens, tied embedding Q6_K 521 MB of 2.77 GB.
- 2026-11-25: Gate frozen as proposed (user: "Freeze and start"); local only.
- 2026-11-25: **Corpora.** Portuguese: `wikimedia/wikipedia` 20231101.pt shard 1 of 6 (revision b04c8d1c…, file sha256 5e9b4476…): its first 17,619 articles (40 MB) counted, its last 578 (1.5 MB) held out; `dominguesm/alpaca-data-pt-br` (revision 99a2a14c…, sha256 ebdefad4…, 51,759 rows, 17.9 MB) counted (`tools/corpus_pt.py`). English: Kyse's corpus as M-8 (WikiText-2 test held out). Code: M-8's lists, of which only 465 of 1,762 files still exist after the repository's reorganization; M-8's held-out code. Counted 26.8M tokens over 107 MB; kept tokens seen at least twice plus bytes, special tokens and the BPE closure.
- 2026-11-25: **Items 1–2** (`runs/experiments/M-8/tools/tokcheck.py`, `prep_texts.py`; `tools/qual_s5.mjs`: the Qwen3.5 stage of a one-stage prepared bundle, logits for every position, about 50 s per 4,096-token text): see Result. Both quality runs use the full model; the pruned model's logits are the full model's restricted to the kept rows (the body is unchanged), so the pruned side is the kept-only softmax on the pruned tokenization.
- 2026-11-25: **Items 3–4** (`tools/serve_run.sh`: each bundle served here with one native contributor; six chat prompts with 64 greedy tokens, and a 1,049-token Portuguese prompt timed with 1- and 33-token requests; order full, pruned, pruned, full): decode 51.6 and 61.0 ms (full), 51.1 and 46.9 ms (pruned). Bundles: full 3.22 GB of weights, pruned 2.51 GB.
- 2026-11-25: **Gate**: 1 PASS, 2 FAIL (code), 3 FAIL (code), 4 PASS.

## Conclusion

For Portuguese and English the pruned S5-nano behaves like the original (bits per byte within 1.4%, the same answers), and it is 13% smaller and about 13% faster to decode, because the dropped rows are part of the output head read on every token. Code is where it fails: about 2% of code tokens the full model would predict are gone (top-1 agreement 97.97%), and a code answer changed (a fenced block's language tag). The code counting corpus was this repository's 465 remaining files, much narrower than the code users ask about. A rerun with a broad code sample (several languages) in the count is the natural next experiment; it would keep more tokens and give back part of the speed gain.

## Limits

- One held-out sample per language (4,096 tokens); code held-out text from a single repository.
- Speed on the 680M only, two runs per arm, with an 18% spread between the two full runs.
- Text quality from teacher-forced logits; answers on six prompts.

## Reproduce

`runs/tools/m8venv/bin/python -I runs/experiments/M-9/tools/corpus_pt.py PTWIKI.parquet ALPACA.parquet runs/experiments/M-9/corpus`; `runs/experiments/M-8/tools/prune.py TOKENIZER.json runs/experiments/M-9/count/pruned pt.list english.list code.list`; `tokcheck.py`, `prune_gguf.py`, `ygaraty model prepare` (full and pruned, `--context 4608 --generate 256`); `prep_texts.py`, `node runs/experiments/M-9/tools/qual_s5.mjs BUNDLE TOKENS.i32 KEPT.json OUT.json`; `bash runs/experiments/M-9/tools/serve_run.sh BUNDLE PORT OUT`.
