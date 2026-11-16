# Byte-level language model with embeddings and Adam under 128 MiB

| | |
|---|---|
| **Date** | not recorded |
| **Path** | ZeroMQ, storage-backed (Redis), Docker memory caps |
| **Result** | 18.4M-parameter model on Shakespeare, three Adam updates; weights, moments and state matched the reference within 5.39e-15. |
| **Evidence** | `runs/language-adam-capacity-128m/` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

Train on the existing Shakespeare text, with every compute client constrained to
128 MiB and swap disabled:

```sh
ygaraty experiment stream_demo --model language --steps 3 --workers 3 --memory-mib 128 --corpus ../tinyad/data/shakespeare/input.txt --output runs/language-adam
```

Use any text file with `--corpus`, or add `--small` for a tiny model. The reader
uses raw byte IDs (vocabulary 256), so no tokenizer or full corpus is loaded into
RAM. It seeks to one short window plus its next-token targets per batch lane.
Windows advance each update, carry hidden/cell state, and detach gradients at
window boundaries. Each lane spans an equal contiguous portion of the corpus;
at the end of complete windows all lanes restart with cleared state. Trailing
bytes insufficient for another complete window are omitted. The driver copies
the corpus into the run directory. Without `--corpus`, the language recipe uses
a deterministic cyclic next-token task for tests.

The language model is **embedding → single-layer LSTM → vocabulary projection**.
The default recipe has batch 2, three timesteps, embedding width 128, hidden width
2,048, vocabulary 256, and tile width 256. Its 18,391,296 float64 parameters use
**140.31 MiB**. Adam's two moments add **280.63 MiB** in Redis; gradients, forward
intermediates, and overlapping old/new versions require additional storage.
Redis remains the larger storage tier; each worker handles only task blocks.

- Embedding tasks read selected table rows and column ranges. Backward scatters
  and sums contributions, including repeated IDs within and across timesteps.
- Adam updates first and second moments per block, uses bias correction by
  completed update number, and applies epsilon outside the square root. Defaults
  are beta1=.9, beta2=.999, epsilon=1e-8, and learning rate=.001.
- Global gradient clipping defaults to norm 1. Each gradient block contributes a
  scalar squared norm; workers reduce those scalars and apply one shared scale
  before moment updates. The loss already averages batch and window length.
- Every embedding row's moments decay and its weight receives the corresponding
  Adam update, including rows absent from the current window. This is dense Adam
  semantics implemented with streamed blocks, not a sparse-Adam approximation.
- The coordinator publishes weights and both moment versions together after the
  complete update. Previous committed versions remain available until then.
  Export streams `parameters`, `optimizer_state`, and `recurrent_state` to block
  files, with optimizer settings and the next window index in the manifest.
- Workers stream task audit records to `.tasks.jsonl` files and assemble their
  final reports incrementally, avoiding an in-memory history proportional to
  the number of training tasks.

For custom dimensions or hyperparameters, use the existing
`ygaraty/experiments/redis_seed.py --model-config` / `ygaraty/runtime/stored_server.py` / `ygaraty/runtime/stored_worker.py` workflow.
The JSON recipe uses `model: "lstm"`, `language_model: true`, `optimizer: "adam"`,
`input_dim` for embedding width, `hidden_dim`, `classes: 256`, `batch`, `window`,
`tile_size`, `seed`, and an optional `corpus` path visible to the importer,
coordinator, and verifier. Optional `learning_rate`, `beta1`, `beta2`, `epsilon`,
and `clip_norm` customize the optimizer; `clip_norm: 0` disables clipping.

The independent verifier implements manual embedding/LSTM backpropagation and
Adam equations, streams old moments from scratch files, and compares every
exported parameter, moment, and recurrent-state value. Tests cover repeated
IDs, finite differences, three successive bias-corrected updates, clipping,
corpus advancement and reset, three-worker numerical comparisons, and rejection
of corrupted optimizer values even with recomputed checksums.

Measured Shakespeare run (`runs/language-adam-capacity-128m`): three workers
completed three windows and Adam updates, with no retries or OOMs. All 18,391,296
weights, 36,782,592 moment values, and final recurrent states matched the
independent streaming reference; maximum absolute checkpoint error was
5.39e-15. Peak worker RSS was 53.3–53.4 MiB with 2 MiB of tracked tensor buffers.
The coordinator peaked at 79.8 MiB RSS, read only 24 tensor bytes (three losses),
and wrote no tensor arrays. Import, export, and verification also passed under
128 MiB with swap disabled. The verifier's filesystem cache reached the cgroup
limit and was reclaimed with zero OOM events. Cleanup left zero Redis tensor
blocks and manifests. All 32 tests passed against real Redis.

This is an executable training prototype. The capacity demonstration is too
short to assess language quality. The output cross-entropy still processes a
complete batch of vocabulary logits; that small activation must fit the worker's
task budget. The independent verifier holds short-window activations in RAM.
There is no resume loader, generation CLI, weight tying, scheduler, or weight
decay yet. The exported state is block-streamed, but exporting it does not by
itself implement crash recovery or corpus-integrity validation on resume.
