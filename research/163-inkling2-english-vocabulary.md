# An English-and-code vocabulary makes Inkling 2 Mini 4% smaller and about 7% faster with the same chat answers, but costs English text 3.4% bits per byte

| | |
|---|---|
| **Conclusion** | Failed gate: 60k-token vocabulary costs English +3.4% bits/byte (bound 2%); code −0.1%, decode −6.7% (measured, 680M and Vast.ai) |
| **Status** | executed |
| **ID** | M-8 (report 163) |
| **Category** | browser |
| **Where** | workstation (quality, CPU reference); remote (speed, with DO-95's stage) |
| **Priority** | med |
| **Date** | Added 2026-11-22; started 2026-11-24; finished 2026-11-24 |
| **Path** | Kyse's English-vocabulary pruning applied to ggml-org's `Inkling-2-Mini-IQ4_NL.gguf`. Rows of `token_embd` and `output` are sliced: they are `Q8_0`, so nothing is requantized. A new tokenizer comes with the new IDs |
| **Result** | Kept 60,205 of 200,058 tokens (30.1%); GGUF 16.93 GB (−0.70 GB, −4.0%). Item 1 passes: no covered held-out word encodes differently. Item 2 **fails** on English: top-1 agreement 99.9% (≥ 98%) but bits per byte 0.640 → 0.662, **+3.4%** (≤ 2%); code passes (100%, −0.1%). Item 3 passes: all prompts and answers use kept tokens. Item 4 passes at the timer's resolution: decode at 1k 70 vs 75 ms per token on the same A100 (−6.7%, bound −5%) |
| **Evidence** | measured-local (680M, WebGPU logits) and measured-remote (Vast.ai A100): `runs/experiments/M-8/` (`gate.json` by `tools/gate.py`; `count/`, `qual/`, `pruned-gguf/`), A100 files in `runs/experiments/DO-98/vast-do98a-20261124/m8.*` |

## Question

Builds on [156 (DO-95)](156-inkling2-mini-one-gpu-three-contributors.md), finished 2026-11-22: it provides the checked NumPy reference, the WebGPU stage, and arm A's A100 setup for the speed measurement.

Inkling 2 Mini's vocabulary has 200,058 tokens (rows padded to 201,024). Of those, 103,320 (51.6%) are ASCII. The rest are CJK (10.8%), accented Latin (7.5%), Cyrillic (6.7%), Arabic (5.2%), Hangul (3.9%), Indic scripts (about 7.5%) and others. This was counted from the GGUF token list on 2026-11-22.

Kyse cut Qwen3.5's 248k tokens to 52k for English. That made the 0.8B model 27% smaller and 40% faster, for +1.5% bits/byte (Kyse README, "English vocabulary").

Inkling's embedding and output head are only 0.98 GB of 17.6 GB, so the expected gain is smaller. Does it still pay?

## Prediction

Keeping about 50k tokens saves 0.74 GB (4% of the file). It also removes most of the output head's 0.49 GB, which is read on every token. That head is about 12% of the roughly 3.9 GB read per token (44 `IQ4_NL` expert visits, 4 `Q8_0` visits, 48 `Q8_0` attention visits, and the loop controllers).

So decoding on one GPU should be about 9% faster. In DO-95's three-worker split, worker 1 goes from 4.1 to 3.7 GB and worker 3 from 6.1 to 5.7 GB.

Quality should cost about what it did for Kyse (+1–2% bits/byte on English), as long as the corpus covers code and tool calls.

## Gate (frozen 2026-11-24, before counting)

Counting corpus: Kyse's `english.txt` (WikiText-2 train, Dolly-15k, TinyStories valid; sha256 51ed2d93…) plus local code and markdown: llama.cpp's sources in `models/src/llama.cpp` (C, C++, Python, Markdown) and this repository's `ygaraty/`, `tests/`, `research/` and `docs/`. Held out (never counted): WikiText-2 test (sha256 bbf94c53…) and the `pods/` repository's Python and README as code. Kept set as in the Protocol (count ≥ 2, the 256 byte tokens, every special or control token, the BPE-merge closure).

Measured on the 680M with the WebGPU stage (report 156's, checked against NumPy), first 4,096 tokens of each held-out text. The pruned model's logits are the full model's restricted to kept rows (the same weights minus head rows), so quality is computed from full-model logits:

1. **Tokenization unchanged:** on the whole held-out English and code, every maximal span the original tokenizer encodes with kept tokens only encodes identically (mapped IDs) with the pruned tokenizer. Any difference fails.
2. **Quality:** top-1 agreement (argmax over kept tokens = argmax over all tokens, at the pruned tokenization's positions) ≥ 98% on each held-out text; bits per UTF-8 byte (pruned: kept-vocabulary softmax on the pruned tokenization; full: full softmax on the original tokenization) rises by at most 2% on each.
3. **Chat:** greedy answers to DO-95's 5 prompts identical: report 156's 64-token arm A answers use kept tokens only, and at every answer position the kept-vocabulary argmax equals the full argmax (checked from the logits of a teacher-forced run).
4. **Speed (remote, decides worth it):** decode time per token on one A100 with the pruned GGUF at least 5% lower than the full GGUF on the same machine (median of 2 runs, 1k positions). If not, keep the full vocabulary.

## Protocol

1. **Count tokens** with the Inkling tokenizer (llama.cpp `llama-tokenize`, or `tokenizers` with `tokenizer.json`) over an English + code corpus. Kyse's corpus (WikiText-2 train, Dolly-15k, TinyStories) plus code, JSON and markdown, and Inkling chat-template transcripts with thinking and tool calls. Record the corpus hashes.
2. **Keep:**
   - tokens seen at least twice;
   - all 256 byte tokens;
   - every special or control token, including the image placeholder `<|unused_200054|>` and the role, content and end tokens;
   - every intermediate token the BPE merges pass through to build a kept token, as in Kyse.
3. **Write a pruned GGUF.** Slice the `token_embd.weight` and `output.weight` rows. Rewrite `tokenizer.ggml.tokens`, `token_type` and `merges` (keeping only merges whose result is kept), the EOS/pad IDs, and `vocab_size` and `vocab_size_unpadded`.
4. **Quality.** Run the NumPy reference (`inkling2_gguf`) on both models over held-out English and code. Report top-1 agreement, KL, and bits/byte.
5. **Speed.** Rerun DO-95 arm A with the pruned GGUF on the same rented GPU and prompts.

Questions:
- Can llama.cpp run the pruned GGUF unchanged (it checks that the tokenizer size equals `vocab_size_unpadded`)? If so, it can serve as a second check.
- How does this interact with the DSpark draft model's head (`mtp.safetensors`), which is out of scope until speculative decoding is tackled?

## Log

- 2026-11-22: Added. Vocabulary composition counted from the GGUF token list. Gain estimates are from tensor sizes (analysis, not measured).
- 2026-11-24: Queued at the user's request ("queue do62, do54, do47, do97 and m8"). Order M-8, DO-97, DO-54, DO-62, DO-47; each remote run up to $3 and 150 minutes of machines, gate frozen before renting. Queued first (local quality part now; speed part on DO-98's A100 kit).
- 2026-11-24: Started. Corpus downloaded with Kyse's code (`runs/experiments/M-8/tools/corpus.py`, user approved the download): `english.txt` 44,858,696 bytes, sha256 51ed2d933baeba66…; `wiki.test.txt` 1,287,656 bytes, bbf94c53a05abe9e…. Gate frozen above before counting.
- 2026-11-24: Counting (`tools/prune.py`, 1,763 files, 77.8 MB, 23.95 M tokens; lists in `runs/experiments/M-8/count/`): 59,500 distinct tokens, 51,459 seen at least twice. The first build kept only the two parts of the merge that builds each kept token, which is not the protocol's "every intermediate token the BPE merges pass through": gate item 1 then found 119 of 256,832 covered held-out English words encoded differently (e.g. " episodes"). Fixed to the protocol as written: BPE is replayed on every distinct counted word whose encoding is all kept, and every token it creates is kept (+469 tokens). Only the counting corpus was used; the gate is unchanged. Result: **60,205 kept** of 200,058 (30.1%), 149,968 merges. **Gate item 1 passes**: 0 of 256,932 covered held-out English words and 0 of 46,734 covered held-out code words differ; held-out text needs 0.23% (English) and 0.18% (code) more tokens.
- 2026-11-24: Gate item 3 passes without a GPU run: the 5 prompts' tokens (18–2,761) and all 320 tokens of report 156's arm A answers are kept, so the inputs are unchanged and each answer token (the full argmax) is also the kept-vocabulary argmax (`count/chat_check.json`).
- 2026-11-24: Pruned GGUF (`tools/prune_gguf.py`): checked on the tiny GGUF (5,070 random kept ids) by the NumPy reference: pruned logits equal the full model's at every kept id (max difference 0.0). Full model: 16,929,872,608 bytes (−0.70 GB, −4.0%), sha256 242d4d2f…7e07, manifest `pruned-gguf/chat/manifest.json` (vocab 60,205). Gate item 4 runs on DO-98's A100 (same machine, new kernels): the A100 rebuilds the file from its own download with the same script, kept list and gguf 0.19.0, and must match that sha256.
- 2026-11-24: Gate item 2 runs on the 680M (`tools/qual.mjs`, `qual.sh`): the first attempt lost the device on submission memory (`qual/watch-oom1`); rerun with layers 8–15 resident (the others streamed), which changes where weights live, not the arithmetic.
- 2026-11-24: The resident 8–15 rerun finished wiki/full (4,096 tokens, 1,319 s) and was then stopped by watch.py as `stalled-idle` during wiki/pruned (`qual/watch-falsestall`): a false stall, since `qual.mjs` reports progress on stderr and the watcher saw neither output nor the node child's CPU. Restarted for the remaining three runs, skipping finished outputs, with progress on stdout (`--progress "chunk [0-9]+"`, stall 600 s).
- 2026-11-24: Item 4 on DO-98's A100 (`do98a`, cost shared, about $0.66 in all): the pruned GGUF rebuilt from the machine's own download matched the workstation's sha256 (242d4d2f…); decode at 1k positions pruned 70, 70 ms against full 80, 70 (alternating); the pruned answer equals the full answer mapped to the new ids.
- 2026-11-24: Item 2 runs finished (wiki full 1,319 s; the others after the false-stall restart). Gate computed by `tools/gate.py`: items 1, 3, 4 pass, item 2 fails on English. Decomposition: on the pruned tokens, the kept-vocabulary softmax costs slightly less than the full softmax (0.662 against 0.664 bits per byte), so dropping head rows is not the cost; the whole rise comes from the 15 of 4,095 English positions whose original tokens are not kept (rare words split into more pieces: 4,106 predictions instead of 4,095).

## Results

| Gate item | English (WikiText-2 test, 4,096 tokens) | Code (`pods/`, 4,096 tokens) | Pass |
|---|---|---|---|
| 1. Tokenization unchanged | 0 of 256,932 covered words differ | 0 of 46,734 | Yes |
| 2. Top-1 agreement ≥ 98% | 99.9% | 100% | Yes |
| 2. Bits per byte rise ≤ 2% | 0.6400 → 0.6620, +3.4% | 0.2867 → 0.2864, −0.1% | **No** (English) |
| 3. Chat answers | 5 prompts and 320 answer tokens all kept | | Yes |
| 4. Decode ≥ 5% faster, same A100 | 70 (70, 70) vs 75 (80, 70) ms per token | | Yes (−6.7%) |

The speed gain is close to the prediction (about 9%), and the size gain matches it (−0.70 GB). Kyse's vocabulary cost Qwen3.5 +1.5% on the whole WikiText-2 test; here a 16.7 KB sample with 15 rare-word splits costs +3.4%.

## Limits

- The quality sample is small: the first 4,096 tokens (16.7 KB) of WikiText-2 test, and 11.5 KB of code. A few rare words decide the English figure; the whole test set would give a steadier number, and could pass or fail.
- The timer reports 10 ms steps; item 4's −6.7% rests on one full-vocabulary run at 80 ms.
- The pruned model's quality was computed from the full model's logits (exact, as the pruned head is the full head's kept rows), on the 680M's WebGPU stage, which report 156 checked against NumPy.
- Other languages still encode byte by byte but were not measured.

## Reproduce

`python runs/experiments/M-8/tools/corpus.py runs/experiments/M-8/corpus` (Kyse's sources), then `prune.py` with `count/english.list` and `count/code.list`, `tokcheck.py`, `prep_texts.py`, `bash tools/qual.sh runs/experiments/M-8/qual` (680M), `prune_gguf.py` and `manifest.py` for the GGUF, DO-98's kit for item 4, and `python tools/gate.py runs/experiments/M-8/qual RUN_DIR`. Python with `tokenizers`, `pyarrow`, `gguf==0.19.0`.
