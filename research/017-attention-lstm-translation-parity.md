# Full attention-LSTM translation: local parity

| | |
|---|---|
| **Date** | 2026-10-22 |
| **Path** | ZeroMQ, in-memory, 2 workers |
| **Result** | 19.2M-parameter attention LSTM on a real SQL minibatch; all 141 checks passed (max update error 8.51e-13). |
| **Evidence** | `runs/lstm-translation-verified/verification.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

Implemented `ygaraty/models/lstm_translation.py`: stacked bidirectional encoder, learned
hidden/cell bridges for every layer, stacked decoder, source attention, tied
input/output embedding and vocabulary cross-entropy. The graph uses existing
rank-two matrix, gate, slice, concatenation, embedding and softmax operations;
workers do not call PyTorch. Source recurrence stops at each example's true
length, reverse recurrence aligns outputs to source positions, encoder padding
is zero and attention masks it. Decoder recurrence runs through teacher-forced
inputs; padded labels do not contribute to the token-weighted batch loss.

**Verified locally on the real pilot checkpoint:** 19,211,136 parameters,
two encoder and decoder layers, embedding 256, hidden 512, vocabulary 16,000.
The two real pairs came from SQL dataset `en-pt-br/v1`, using the original pilot's
SentencePiece tokenizer. Two Python/ZeroMQ workers executed 17,581 tasks in
12.97 seconds of distributed execution (oracle preparation excluded), with zero
retries. All **141 checks** passed at the existing `rtol=2e-7, atol=2e-9` budgets:
encoder outputs, decoder outputs, attention, logits, loss, all 33 named parameter
gradients, both Adam moments and every parameter update. Adam comparisons use
update deltas, so a large starting weight cannot hide an inaccurate update.

| Comparison | Maximum absolute error |
| --- | ---: |
| Encoder outputs | 8.83e-16 |
| Decoder outputs | 5.55e-16 |
| Attention | 2.22e-16 |
| Logits | 3.38e-14 |
| Loss | 1.33e-15 |
| Gradients | 4.86e-15 |
| Adam first moment | 4.87e-16 |
| Adam second moment | 2.85e-18 |
| Adam parameter updates | 8.51e-13 |

The independent oracle uses the pilot's packed PyTorch `nn.LSTM` architecture.
`lstm_translation_oracle.Float64Translator` overrides only attention arithmetic:
the pilot explicitly casts attention to float32 even after `.double()`, whereas
this oracle preserves float64. The original pilot model is unchanged. This test
uses eval mode (dropout off), no label smoothing or clipping, and fresh Adam
moments with learning rate .001, betas (.9,.999), epsilon 1e-8. It is not a resume
of the pilot's mixed-precision/dropout/clipped training run.

Reproduce using the model-only checkpoint exported from our existing pilot:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
ygaraty experiment translation_demo --architecture lstm \
  --checkpoint runs/lstm-translation-pilot-input/model.pt \
  --tokenizer ../tinyad/benchmarks/translation/colab-pilot/tokenizer.model \
  --pairs assets/data/en_ptbr_minibatch.json --workers 2 --lr .001 \
  --output runs/lstm-translation-repeat
```

For SQL selection, replace `--pairs` with `--dataset en-pt-br --dataset-version v1
--batch-size 2` and set `YGARATY_DATASET_URL`. `--architecture lstm` requires an
explicit model checkpoint and matching tokenizer; the checkpoint dictionary
contains `config` and `model`. Source/export hashes and the model-only extraction
boundary are recorded in `runs/lstm-translation-pilot-input/provenance.json`.

Evidence: `runs/lstm-translation-verified/verification.json`, saved checkpoint
and manifest in its `checkpoint/` directory, and the concise
`runs/lstm-translation-summary.json`. Unit tests compare one- and two-layer
models, unsorted source lengths, repeated tokens, padding, target causality,
all gradients and tiled Adam. `tests/test_lstm_translation.py` uses tighter
numerical checks on the small cases. The regression suite passed 34 tests,
with 25 optional external-service tests skipped; log:
`runs/lstm-translation-regression.log`.

**Boundary:** this is a deterministic one-minibatch local correctness milestone.
Peak live coordinator tensor payload was 2,032,933,904 bytes (~1.89 GiB), not
whole-process RSS. Import, activations, optimizer state and execution still use
the in-memory parity harness; this is not a bounded-memory capacity result.
The small SQL-fed HTTP/browser recipe described above now covers multiple updates.
Full-size continuous training/resume and dropout/clipping training validation
remain follow-up work. No AX41 was rented.
