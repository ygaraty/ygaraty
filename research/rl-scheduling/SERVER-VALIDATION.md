# Server handoff: tile epilogues and Transformer applicability

Use the commit containing this file on a machine authorized for numerical execution.
The originating workspace remains simulation-only. This handoff does not deploy
anything or start a remote job.

## What is ready

The repository contains the offline grouping audit, `tile_epilogue.transform`,
structural tests and the small
[`artifacts/tile-epilogue-v1.zip`](artifacts/tile-epilogue-v1.zip) bundle. The bundle
contains pruning baselines, candidate graph/certificate envelopes, timing records,
cost tables, checksums, source and the verifier. No numerical inputs, credentials,
model weights or real training results are included.

The hypothetical `offline_tile_epilogue` operation has **no numerical runtime
implementation**. Loading these candidates into a production executor is not a
supported test. A small explicit numerical adapter is still needed, as below.
No flag currently enables this epilogue rewrite in the Transformer runtime.
`YGARATY_PRUNE_PLAN=1` enables pruning only.

Existing evidence:

| Change | Workload | Evidence |
|---|---|---|
| Pruning | Tiny Transformer | Experiment 54 reports about 4% measured gain; same losses |
| Tile reduction + bias + ReLU | Three MLP graphs | 8.88%, 15.31%, 9.39% modeled gain with +2 ms/tile wrapper |
| Tile epilogues | Transformer | Not yet evaluated; no claimed gain |

See [TILE-EPILOGUE-AUDIT.md](TILE-EPILOGUE-AUDIT.md) for contracts and limitations.
The modeled results use float64 transport; do not extrapolate them to bf16 or
float32 serialization boundaries.

## First check: does the Transformer expose this pattern?

Inspect the actual compiled plan on the server before building an adapter for it.
`ygaraty/runtime/transformer_kernels.py` already fuses a full layer into `tf_layer`;
its feed-forward activations stay inside that task. `tf_layer_grad` recomputes the
forward activations locally. The owned path adds stateful ownership boundaries.
The MLP reduction/assembly opportunity may therefore not exist in this runtime.

Record the exact model, compiler settings, graph hash, ownership, dtype/wire
format, task counts and declared outputs. Count explicit
`add_many → assemble → add_bias → relu` patterns. Check every removed value's
consumers, including backward consumers, diagnostics and persistent state.

If no pattern is exposed, report **not applicable** and stop the epilogue experiment
for that plan. Do not defuse layers or cross owned-state boundaries merely to
create matches. A kernel-internal Transformer optimization is a separate study.
The current rewrite rejects owned, replicated and cache-affinity plans.

## Verify the shipped MLP evidence first

From the repository root, using the server's already provisioned environment:

```bash
unzip public/research/rl-scheduling/artifacts/tile-epilogue-v1.zip -d /absolute/path/to/test-output
PYTHONPATH=. .venv/bin/python /absolute/path/to/test-output/tile-epilogue-v1/verify.py
.venv/bin/python -m unittest discover -s tests -p 'test_tile_epilogue.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_*grouping.py' -v
```

Choose a fresh extraction directory. Check `checksums.json` against extracted
files and record the repository revision. Expected verifier result: 28 plans,
24 structural rewrite checks, 672 unique rollouts and 48 aggregate summaries.
These commands perform structural checks only.

Snapshots are gzip JSON envelopes: load `envelope['plan']`, not the entire envelope,
through `simulation.reference.load_plan` using a temporary JSON file. Use each
`<workload>-prune.json.gz` as the control. Candidate names include memory limit and
wrapper cost; wrapper cost affects timing assumptions, not numerical semantics.
The archived plans avoid needing the unavailable legacy source files just to test
these exact candidates. Regenerating a corpus still requires those sources.

## Numerical adapter to implement on the execution server

Keep it opt-in and local. For ordinary tasks use the existing kernels and normal
region semantics. For `offline_tile_epilogue`:

1. Read the original reduction descriptor in `attrs.original_reduction`.
2. Pass its partial inputs, in their exact recorded order and multiplicity, to the
   original `add_many` kernel. Preserve zero initialization and its finite checks.
3. Apply the original `add_bias` kernel to the completed tile and the final input,
   which is the sliced row bias. Then apply the original `relu` kernel.
4. Pack `[preactivation; activation]` by rows, without casting or changing either
   value. The output shape is twice the tile height. Preserve stage-level checks.
5. Let the two existing metadata assemblies select the top and bottom regions.
   Never omit preactivation: the backward graph reads it under its original ID.

Do not execute activation on partial products, reassociate reductions, use a
BLAS epilogue that changes rounding, or silently drop intermediate values needed
by an external consumer. `runtime.local.execute_plan` directly calls `execute`;
it needs an explicit adapter/injection to recognize the synthetic operation.
Do not register it globally as a production kernel for this experiment.

Compare pruning alone with pruning plus epilogues on identical independently
copied valid input arrays. Start with the archived MLPs, float64 and seeds 0–9;
add zero, signed, small/large finite and supported noncontiguous fixtures.
Preserve semantic constants and valid labels. Record fixture generation and input
hashes. Compare every declared root, plus full preactivation/activation and their
backward consumers where available. Check input immutability and exact shape,
dtype, finite status and contiguous logical-element bytes. Any tolerance-based
acceptance must be separately justified and declared before running that test.

The simulation's MLP output contract includes loss, parameter gradients and
parameters; it does not include an optimizer update. If integrating into a
multi-update workload, additionally compare updated parameters, optimizer state,
step counters and loss over multiple updates. Matching loss alone is insufficient.

## Memory and timing acceptance

Reject any packed tile exceeding its 1 or 8 MiB tensor allowance or the 8 MiB
packed-output limit. MLP-77 is intentionally rejected at 1 MiB; its modeled
allowance is 2,494,464 bytes. These are per-task tensor estimates, not process RSS.
Measure actual native/NumPy memory separately and include cache/concurrency when
assessing a worker-wide limit. Avoid measurements that count only Python allocations.

After numerical checks pass, benchmark pruning alone against the rewrite using
identical output retention and input copies, fixed backend/thread settings, five
warm-ups and at least 30 paired trials in alternating or randomized order. Include
packing and output reconstruction inside the timed interval. Keep initialization,
fixture copying and correctness comparisons outside both timed intervals. Record
rewrite cost separately and report amortization across repeated plan executions.

Follow [LOCAL-VALIDATION.md](LOCAL-VALIDATION.md): proposed acceptance is at least
2% median paired gain and a positive lower 95% confidence bound, with numerical
checks passing. Report memory regressions. Sequential local timing is not proof
of distributed speedup. Test actual runtime serialization/ownership only as a
separate authorized runtime experiment with an adapter that implements this
packed-pair protocol; do not reuse simulator timing as measured evidence.

## Return a compact report

Record applicable/not-applicable patterns, source revision and graph hashes,
input recipes/hashes, exact output contracts, per-output error checks, raw paired
timings, confidence-interval method, measured memory and environment settings.
Classify each workload as blocked, numerically failed, performance inconclusive,
or validated on the tested backend. Compress large artifacts and keep the original
plans as controls. Do not claim Transformer gains from the shipped MLP simulations.
