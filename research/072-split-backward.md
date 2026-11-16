# Owners' weight gradients off the chain don't shorten the update: +1.4% at band 16, −2.1% at band 64

| | |
|---|---|
| **Conclusion** | **Rejected:** the split backward is bit-identical but saves nothing, +1.4% per update at band 16 (measured, 3 runs, workstation). |
| **Status** | executed (stage 1); stage 2 skipped by the gate |
| **ID** | DO-9 (experiment 72) |
| **Category** | transport |
| **Where** | workstation (omarchy) |
| **Date** | Added 2026-11-01; run 2026-11-03 |
| **Path** | Stage 1: the 1.06M recipe, 8 Python workers, emulated 100/20 Mb/s and 50 ms links, handoffs on, 12 updates. Arms A (combined backward, band 16), B (split, band 16), C0 (combined, band 64), C (split, band 64), 3 rotated runs each |
| **Result** | Median seconds per update over updates 2–10: A 3.010, B 3.051 (+1.4%, ranges overlap), C0 4.329, C 4.238 (−2.1%, ranges apart). Neither reaches the gate's 3%, so stage 2 on the pods is skipped. Losses are bit-identical update by update in all 12 runs. The chain trace shows why: the split takes 17 ms of kernel time off the chain per update (85 → 68 ms), but the median upload grows 8 ms (96 → 104) and the chain doesn't shorten (2,745 → 2,768 ms). Free |
| **Evidence** | [`evidence/transport/do9-split-backward-2026-11-03.json`](evidence/transport/do9-split-backward-2026-11-03.json); raw runs, captures and `drive.sh`/`analyze.py` in `runs/experiments/DO-9/omarchy-20261103-161702/` (local, not in git). Code: `YGARATY_SPLIT_BACKWARD` (default off) in `ygaraty/runtime/transformer_kernels.py` and `ygaraty/transports/http/worker_execution.py`; `tests/test_split_backward.py` |

## Question

Zero-bubble pipeline parallelism (Qi et al., ICLR 2024, arXiv:2401.10241) splits each backward pass into two parts:
- the gradient with respect to the stage's input, which the next stage waits for;
- the gradient with respect to its weights, which nobody waits for.

It schedules the second part into idle time, keeps synchronous semantics, and reports up to 23–31% more throughput than 1F1B.

Our owners already sit idle 3–5.5 s per update (L1). Today `tf_owned_grad` (`ygaraty/runtime/transformer_kernels.py`) computes `dx`, `dmem` **and** every weight gradient before it returns `dx`, so the weight part lies on the chain.

**If an owner returns `dx` first and computes its weight gradients afterwards, how much shorter is the update, with bit-identical losses?**

## Prediction, recorded before any code

**Small on our CPU pools.** The paper's gains come from compute-bound GPU pipelines; ours is latency-bound.
- **30.3M model:** owners compute 0.3–0.6 s per update over 14–34 tasks (L1). The weight part of a backward task is about half of it, roughly 10–15 ms per backward hop over about 10 backward hops: **2–4% of 4.6 s**.
- **1.06M model:** a kernel is 17 ms of a hop of about 190 ms (experiments 51, 58): **1–3%**.

The split matters more when compute per hop grows: larger bands (DO-10) or GPU workers with bigger batches (DO-11). Arm C tests that.

## What had to be built (as planned; see the Log for what was built instead)

1. **Kernels.** `layer_backward` in `transformer_kernels.py` becomes two functions:
   - `layer_backward_input`: returns `dx` and `dmem` and stashes what the weight gradients need. Those are the recomputed forward intermediates the current function already holds, and they stay in the owner's memory, keyed by band.
   - `layer_backward_weights`: computes the packed weight gradients from the stash, in the same arithmetic order as today.
2. **New ops.**
   - `tf_owned_grad_x` returns the same matrix as `tf_owned_grad` today.
   - `tf_owned_grad_w` writes `layer['grads'][band]` and returns a 1×1 token.
   - `tf_owned_step` depends on every band's `tf_owned_grad_w`.
3. **Planner** (`ygaraty/compiler/sequence_transformer.py`, `lower_tf_layer`): behind `YGARATY_SPLIT_BACKWARD=1` (default off), emit both ops per band. The `_w` task depends only on its `_x` task and is owned by the same worker.
4. **Scheduler and worker.** An owner runs ready `_x` tasks before ready `_w` tasks. Use the eager-claim ordering: give `_w` a lower priority. Verify that the result upload of `_x` starts before `_w` runs.
5. **Memory.** The stash holds one band's intermediates until its `_w` runs. Report the owner's peak memory in both arms. At 30.3M, one band's intermediates are a few MB.
6. **Tests.**
   - A float64 parity test: the split ops give the same `dx`, `dmem` and packed gradients, bit for bit, as `layer_backward` on random inputs, encoder and decoder.
   - A plan test: `tf_owned_step` waits for all `_w` tasks.
   - A worker test: `_x` is uploaded before `_w` starts.

## Design

**Stage 1, workstation, 1.06M model, 12 updates.** Steady window is updates 2 to 10, as in experiment 58.

| Arm | Split backward | Band | Notes |
|---|---|---|---|
| A | no | 16 | experiment 59's best (handoffs on, non-blocking) |
| B | yes | 16 | |
| C | yes, against C0 without | 64 (4× the work per hop) | tests whether the gain grows with compute per hop |

- 3 runs per arm in rotated order (A B C0 C, B C C0 A, C C0 A B), plus one captured run per arm for `chain_trace`.
- Wait for a 1-minute load under 3 before each run.

**Stage 2, only if B beats A by at least 3% beyond the spread:**
- Experiment 61's pool (8 c-2 weak, 8 c-8 strong, nyc3), handoffs on.
- Arms A and B, ABBAAB, 22 updates. About $2.

## Gate

- **Correctness gate:** losses bit-identical update by update between A and B, and between C0 and C. Owners' published weights equal after the run.
- **Counts as a lever:** B at least 3% faster than A over three repeats, outside both arms' ranges. Otherwise report it as measured and small, and skip stage 2.
- **Chain trace:** the median backward hop falls by the owner's weight-gradient compute time, within 5 ms.

## Trigger on the workstation

```bash
git pull && pip install -e . && python -m unittest tests.test_split_backward   # the new tests
cd relay && go build -o ../relay-server ./cmd/relay && cd ..
# per run (fresh relay per run, as in experiment 58):
./relay-server -addr 127.0.0.1:PORT -token TOKEN -max-mib 4096 &
YGARATY_HANDOFF=1 YGARATY_SPLIT_BACKWARD=$SPLIT \
ygaraty experiment http_profile --recipe transformer --learning-rate 0.0005 --workers-kind python --tiles 512 --max-tasks 16 --repeats 1 \
  --relay-url http://127.0.0.1:PORT --relay-token TOKEN --version 1m-seed2026-v2k --embedding 128 --hidden 384 --layers 2 --heads 4 --band $BAND \
  --max-task-mib 512 --workers 8 --float32-wire --activation-wire bf16 --owned-layers --owned-vocab 4 --pushed-inputs --eager-claims --handoff \
  --length-window 16 --batch-tokens 3072 --dataset-url sqlite:///$PWD/runs/datasets/opus100-en-pt/opus100-en-pt.sqlite \
  --store-url postgresql:///ygaraty_tiny --steps 12 --rtt-ms 50 --down-mbps 100 --up-mbps 20 --output runs/experiments/DO-9/r$R-$ARM
ygaraty experiment chain_trace runs/experiments/DO-9/cap-A-cap runs/experiments/DO-9/cap-B-cap
```

**Stage 2:** experiment 61's Reproduce section, with `YGARATY_SPLIT_BACKWARD=1` exported on the coordinator and every worker for arm B.

## Cost and limits

- **Cost:** free for stage 1, about half a day to build plus 40 minutes of runs; stage 2 about $2.
- **Limits:**
  - Only owned layers are split. Vocabulary slices keep their combined backward.
  - The gain is bounded by compute per hop. On CPU pools it is small by construction, so this mostly tests whether the split is free to keep for the GPU experiment.

## Log

- 2026-11-01: Added (first committed as a numbered plan file; renumbered to a track ID, since numbers mark finished experiments).
- 2026-11-03: Stage 1 started on the workstation (omarchy), at `abb1716+dirty`. **Built differently from "What has to be built", gate unchanged:** the split happens on the worker, not in the plan. `tf_owned_grad` with `YGARATY_SPLIT_BACKWARD=1` computes `dx` and `dmem` and keeps each weight matrix's gradient as its pair of factors (`layer_backward(..., defer=True)`). The worker hands its results to the eager poster, whose threads upload them, and then multiplies the factors (`OwnedLayers.flush`, `worker.weight_grads` in captures). `tf_owned_step` flushes first in any case. So there are no new tasks, no extra claim per band, and the plan and the coordinator are unchanged. Separate `_w` tasks would have cost a claim round trip each, and the coordinator would only have made them ready after `_x`'s commit. Tests in `tests/test_split_backward.py`: deferred products equal the combined backward bit for bit (encoder, decoder, final norm); two owned updates are bit-identical, whether flushed per task or only by the step; a retried band replaces its pending products; results reach the poster before the flush.
- 2026-11-03: Bug found and fixed before any run: DO-5's `self.hints` (the n-gram hint mode, `coordination/lstm.py`) overwrote `Job.hints()`, the workers' cache hints. Every translation job failed at its first claim with a cache message ("'str' object is not callable"), on `main` since 5e6f1b2, the pods' `serve34` included. Renamed to `hint_mode`; regression check in `tests/test_transformer_job.py`.
- 2026-11-03: Rehearsal (3 updates, A and B, band 16): both complete, losses identical (9.780902862548828, 7.765143394470215). Captured split run: 32 `worker.weight_grads` spans of about 4 ms, against about 13 ms for the rest of an owned backward task. Main matrix: `runs/experiments/DO-9/omarchy-20261103-161702/drive.sh main "<order>" 12`, the orders A B C0 C / B C C0 A / C C0 A B, under watch.py (stall 600 s, maximum 3,600 s).
- 2026-11-03: Main matrix finished (watch.py state `finished`, all 12 runs rc 0, 16:21–16:31), then one captured run per arm at band 16. Results below; stage 2 skipped by the gate.

## Results

| Arm | Split | Band | Runs (s per update, median of updates 2–10) | Median |
|---|---|---|---|---|
| A | no | 16 | 2.980, 3.030, 3.010 | 3.010 |
| B | yes | 16 | 3.051, 3.059, 2.968 | 3.051 |
| C0 | no | 64 | 4.329, 4.301, 4.331 | 4.329 |
| C | yes | 64 | 4.238, 4.275, 4.237 | 4.238 |

- **Correctness gate: passed.** Losses are bit-identical update by update, B against A and C against C0, in all 12 runs.
- **Lever: no.** B is 1.4% *slower* than A, within the runs' spread. C is 2.1% faster than C0, with ranges apart, which is under the 3% bar. Stage 2 is skipped.
- **Chain trace (captured runs, band 16, updates 2–11): failed.** Both arms have 15 hops on the chain and a median hop of 168 ms. Kernels on the chain fall from 85 to 68 ms, but the median handed upload grows from 96 to 104 ms, and the chain goes from 2,745 to 2,768 ms. The gate asked for the median hop to fall by the weight-gradient time.
- **Pool check:** arm A's 3.010 s reproduces experiment 59's 3.018 s for the same setup.

## What it taught

- **On a CPU worker, "off the chain" isn't off the critical resource.** The weight products run while the result uploads, and the upload takes longer by about what the products took: bf16 encoding, hashing and the socket compete with the matmuls for the worker's cores and the GIL. The work leaves the kernel span but not the hop.
- **It helps a little only when compute per hop grows.** At band 64 (4× the work per hop) the split saves 2.1%, beyond the spread. On GPU owners, where the upload doesn't share the compute device, the overlap should be real (DO-11). The switch stays, default off.
- **A split in the worker beats a split in the plan.** Deferring the products inside `tf_owned_grad` costs no extra claims or tasks; separate `_w` tasks would each have needed a claim round trip, and the coordinator would have made them ready only after `_x` committed.
- **Bug found:** DO-5's `self.hints` (the n-gram hint mode) overwrote `Job.hints()`. Every translation job with cache messages had failed at its first claim since 5e6f1b2, the pods' `serve34` included. Fixed in cec07b1.

## Limits

- One machine, the 1.06M model, 3 runs per arm, 12 updates. The 30.3M model's backward tasks carry more weight work per hop (DO-9's prediction: 2–4%), but the same contention would apply on CPU pods.
- Only owned layers are split. Vocabulary slices keep their combined backward.
- The flush runs right after a reply's results go to the poster, not when the worker is idle. An idle-time flush might avoid some contention, but it was not tried.
- The worker threads use one BLAS thread each (`SINGLE_THREAD`, as in experiment 23). Pods' workers use 4 threads on the strong machines, which could change the contention.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_split_backward
cd relay && go build -o ../relay-server ./cmd/relay && cd ..
D=runs/experiments/DO-9/<host>-<timestamp>   # copy drive.sh and analyze.py from the original run directory
bash $D/drive.sh $D/main "A1:0:16 B1:1:16 C01:0:64 C1:1:64 B2:1:16 C2:1:64 C02:0:64 A2:0:16 C3:1:64 C03:0:64 A3:0:16 B3:1:16" 12
.venv/bin/python $D/analyze.py $D/main
for a in A:0 B:1; do YGARATY_CAPTURE_DIR=$PWD/$D/cap/cap-${a%%:*}-cap bash $D/drive.sh $D/cap "${a%%:*}:${a##*:}:16" 12; done
.venv/bin/python -m ygaraty.experiments.chain_trace $D/cap/cap-A-cap $D/cap/cap-B-cap
```

`drive.sh` runs, per item `NAME:SPLIT:BAND`, a fresh relay and DO-9's trigger command above with `YGARATY_SPLIT_BACKWARD=SPLIT` and `--band BAND`. It waits for a 1-minute load under 3 before each run.

## Merge reconciliation

- 2026-11-01: Local DO-5 reassigned to DO-9 to avoid a different upstream experiment. Original pre-merge file retained in the local merge backup.
