# English → Brazilian Portuguese Transformer: one-minibatch parity

| | |
|---|---|
| **Date** | 2026-10-22 |
| **Path** | ZeroMQ, in-memory, 3 workers |
| **Result** | 19.7M-parameter Transformer, one real minibatch; all 511 tensor comparisons against PyTorch passed. |
| **Evidence** | `runs/translation-verified-checkpoint/verification.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

Ygaraty can now lower our existing pre-norm encoder–decoder Transformer to a
forward/backward tape and execute one real bilingual minibatch through Python
ZeroMQ workers. The browser UI also offers this fixed Transformer parity recipe (see below).

```sh
pip install -e '.[translation]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  ygaraty experiment translation_demo \
  --output runs/translation-check --workers 3 --tile 512
```

The defaults use the existing OPUS-distilled **19,661,824-parameter** checkpoint
and its 16,000-token SentencePiece tokenizer in the sibling `tinyad` directory.
Use `--checkpoint`, `--tokenizer`, and `--pairs` to select other local assets.
The paired JSON fixture contains two hand-authored English/Brazilian Portuguese
sentences; it is a correctness fixture, not a corpus or translation-quality test.
FLORES evaluation examples are not used for this update.

`ygaraty/verification/translation_reference.py` imports the checkpoint into the original PyTorch
`TransformerTranslator`, tokenizes the pairs, appends source EOS, and creates
shifted BOS/target/EOS decoder inputs and labels. It runs separately and exits
before the distributed computation. Workers import no PyTorch code.

`ygaraty/models/transformer.py` lowers each batch item and attention head into explicit
rank-two operations. The graph preserves shared parameters, tied input/output
embeddings, learned positions, pre-norm residual blocks, ReLU feed-forward layers,
source/target padding masks, and causal decoder masking. Embedding PAD lookups
have zero gradient, while the tied output projection still contributes to that
same parameter row. PAD labels are ignored; loss uses 0.1 label smoothing and
normalizes over all non-PAD target tokens in the minibatch.

`ygaraty/experiments/translation_demo.py` tiles matmuls and parameter operations, uses the existing
Ygaraty scheduler and wire protocol, and launches ordinary `ygaraty/runtime/zmq_worker.py` processes.
Workers compute forward operations, backward operations, and Adam updates.
Assembly and transpose data movement happen in the coordinator. Live arrays are
released after their last consumer. Every logit, loss, parameter gradient, updated
weight, and Adam moment is compared elementwise against PyTorch (rtol 2e-7,
atol 2e-9); any mismatch aborts the run.

Outputs:

- `verification.json`: checkpoint/tokenizer hashes, settings, all errors and worker statistics.
- `plan.json` and `events.jsonl`: task graph, assignments, attempts, and completions.
- `checkpoint/`: **actual worker-computed** updated parameters and Adam moments,
  one NumPy file per tensor, plus the model/tokenizer manifest.
- `reference/`: independent PyTorch inputs/results for auditing.

The check disables dropout, uses float64, and performs a **fresh Adam** step
(beta1=.9, beta2=.98, epsilon=1e-8, learning rate=1e-4, no clipping/weight decay).
It does not resume the checkpoint's original AdamW training state. The coordinator
and oracle hold full live tensors; this milestone is not a 128-MiB worker or
out-of-core capacity claim. Allow roughly 1.3 GiB of disk for verification assets
and the resulting checkpoint. No service on port 8090 is restarted.

Before long translation training: stream the paired corpus, tile vocabulary loss
and normalization reductions for larger sequences, extend the browser recipe to general datasets and configurations, reproduce the intended stochastic
training/optimizer policy, add durable resume and held-out PT-BR evaluation.

Verified on 2026-10-22 with the full checkpoint: 2 sentence pairs, 15 non-PAD
labels, 11,092 tasks (6,923 numerical worker tasks), 3 workers, no retries.
All **511 comparisons passed**. Maximum absolute errors: logits 2.40e-14,
loss 8.88e-16, gradients 8.61e-13, updated parameters 8.98e-12. Distributed
execution plus comparisons/checkpoint export took 8.15 seconds (excludes the
separate PyTorch oracle and graph construction). See
`runs/translation-verified-checkpoint/verification.json`. All 381 saved parameter/
Adam-state tensors also passed a separate disk read-back comparison. The local
regression suite passed 27 tests; 24 external Redis/PostgreSQL tests were skipped.
