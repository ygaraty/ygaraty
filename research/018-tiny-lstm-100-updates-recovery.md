# Tiny LSTM: 100 distributed updates with worker and coordinator failures

| | |
|---|---|
| **Date** | not recorded (re-run 2026-10-24, see package-reorganization-check) |
| **Path** | ZeroMQ, in-memory, 2 workers |
| **Result** | 100/100 updates matched PyTorch through a killed worker and a coordinator crash/resume; final weight error 1.57e-14. |
| **Evidence** | `runs/tiny-lstm-100/verification.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

`ygaraty/experiments/tiny_lstm_training.py` tests the complete two-layer attention-LSTM architecture
with vocabulary 64, embedding 8 and hidden size 16: **16,104 parameters**. Eight
fixed English–Portuguese toy pairs use a small shared word vocabulary. Batch size
is two; seeded shuffling visits each example once per epoch. The test rebuilds
the graph for each batch, preserves Adam moments and advances its bias-correction
step. Existing tiled operations are grouped into chains of up to eight operations
and dispatched to two persistent Python/ZeroMQ workers.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
ygaraty experiment tiny_lstm_training --output runs/tiny-lstm-repeat
```

Use a new output directory. The command runs all 100 updates and both intentional
faults automatically. It requires the optional translation dependencies (PyTorch)
for the independent oracle, but workers use Ygaraty kernels only. It takes about
two minutes on the tested local machine. The runner uses POSIX process signals.

Every update compares logits, masked loss, all gradients, parameter-update deltas,
weights and both Adam moments against PyTorch at the unchanged
`rtol=2e-7, atol=2e-9` tolerances. It additionally runs a separate **uninterrupted**
PyTorch trajectory from the initial weights and compares weights/moments at steps
50 and 100. This catches errors that could otherwise be hidden by restarting the
per-step oracle from a faulty Ygaraty checkpoint.

Faults exercised:

- At update 25, stop a worker before dispatching a leased task, then kill that
  process. Its lease expires, a replacement worker joins, and the task is retried.
  A scheduler probe also confirms the killed worker's stale lease is rejected.
- After saving update 50, abruptly exit the coordinator during update 51 after
  ten numerical task results, without saving that partial update. The parent
  terminates orphan workers and launches a new coordinator process from checkpoint
  50. Update 51 is recomputed, continuing through update 100.

Checkpoints atomically publish model weights, Adam moments, completed update count,
next batch cursor, model/optimizer settings, shuffle seed and dataset checksum.
Files are flushed/fsynced before rename, followed by directory fsync. Resume
rejects mismatching dataset/configuration/cursor metadata. Dropout is disabled;
shuffling derives from seed plus epoch, so no evolving RNG state is required.
This validates the specified crash boundary, not every possible filesystem,
checkpoint/journal or host failure. It is not a production exactly-once guarantee.

Observed result in `runs/tiny-lstm-100/verification.json`: **100/100 updates pass**,
with exactly one injected task retry and contiguous committed steps 1–100. The
final resumed weights differ from uninterrupted PyTorch by at most **1.57e-14**.
Maximum per-step gradient error is **1.83e-15**, and Adam update-delta error is
**1.53e-14**. Total elapsed time was approximately **110 seconds**, including
oracle checks and the injected failures. First/last training-batch losses were
4.386 and 0.649; these are different minibatches, not a held-out quality metric.

Artifact size is **under 1 MB**: an initial state, the latest two rolling
checkpoints, the toy corpus/token vocabulary, 100 compact metric records, fault
proofs and a final report. No full plans, intermediate tensors, PyTorch snapshots
per step or task traces are saved. The runner enforces a 5 MiB final-artifact
ceiling. Checkpoints are written every update and older ones removed after atomic
publication; transient checkpoint disk use remains small too.

`tests/test_tiny_lstm_training.py` additionally checks checkpoint round-trip,
nonzero Adam moment preservation, metadata rejection and epoch coverage. This
harness is a local float64 lifecycle test with in-memory tensors; it does not
establish large-model memory bounds, browser recovery, production SQL training
or translation quality. Those remain subsequent AX41 milestone work.
