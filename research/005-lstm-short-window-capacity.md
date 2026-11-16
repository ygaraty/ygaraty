# Single-layer LSTM with short windows under 128 MiB

| | |
|---|---|
| **Date** | not recorded |
| **Path** | ZeroMQ, storage-backed (Redis), Docker memory caps |
| **Result** | 17.9M-parameter (137 MiB) LSTM, two windows; all final weights matched within 5.56e-17; workers peaked near 62 MiB RSS. |
| **Evidence** | `runs/lstm-capacity-128m/` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

Run the same capacity demonstration for an LSTM:

```sh
ygaraty experiment stream_demo --model lstm --steps 2 --workers 3 --memory-mib 128 --output runs/lstm-capacity
```

Add `--small` for a quick end-to-end smoke test. The large recipe uses dense
128-dimensional inputs, 2,048 hidden units, 32 output classes, batch 2, a
three-timestep window, and 256 × 256 weight tiles. Its 17,899,552 float64
parameters occupy **136.56 MiB**, larger than any client's 128 MiB limit.

The cell follows `tinyad/src/lstm.c`: packed gates are **forget, input, output,
candidate**, with forget bias 2. Each timestep computes input and recurrent
projections, gate activations, cell/hidden updates, and output logits. Loss is
cross-entropy averaged over batch rows and timesteps. Backward unrolls the
window and sums each shared parameter's gradient contributions before its SGD
update. Parameters are immutable versions, so updates cannot change weights
still needed by another backward task.

Hidden and cell values carry into the next window as Redis references. Their
gradient history stops at the boundary (truncated backpropagation through time).
The coordinator reads only the scalar loss; workers fetch the blocks needed for
their task. Gate activations, slicing/scattering, elementwise state operations,
and shared-weight gradient sums are tiled too. Saved forward blocks live until
their final backward consumer finishes. Carry blocks survive until the next
window completes; the final carry is exported with the checkpoint. Namespace
cleanup releases everything after export.

`ygaraty/verification/lstm_reference.py` separately implements the cell equations and manual BPTT.
It reads weights by block, keeps the short window's small activations in memory,
and forms parameter-gradient blocks on demand. `ygaraty/verification/lstm_verify.py` independently
replays all windows and compares every exported final parameter. Reference
weights between updates live in scratch files, never in a full in-memory model.

`tests/test_lstm.py` checks every timestep's hidden state, cell state, and logits,
all parameter gradients, and uneven tile boundaries against that reference.
It also checks every small-model parameter with finite differences. With
`REDIS_TEST_URL` set, a three-worker integration test compares the actual stored
outputs and gradients over two windows, including state carry and final SGD
weights. `ygaraty/runtime/stored_server.py --retain-verification-tensors` retains these tensors
for inspection; this diagnostic mode increases Redis usage and is off by default.

Measured run (`runs/lstm-capacity-128m`): all 17,899,552 final weights matched
within 5.56e-17 absolute error after two windows; losses were 3.39977 and 1.71027.
All three workers contributed (4,414–4,440 tasks each). Peak worker RSS was
61.3–61.7 MiB, with 1.5 MiB of tracked tensor buffers; coordinator RSS was
64.5 MiB and it read 16 tensor bytes total. Importer, exporter, and verifier
also completed under the same 128 MiB cap. The verifier's filesystem page cache
reached the cgroup limit and was reclaimed, with zero OOM events. Redis cleanup
left zero tensor blocks or manifests. The complete 28-test suite passed against
real Redis, including the multi-worker output/gradient comparison.

Scope of `--model lstm`: this is a Python implementation of the native cell's
equations, not a C tape importer. This recipe uses dense inputs and SGD. The
language recipe below adds embeddings, Adam, and clipping. Arbitrary row-wise
sequence resets and variable-length masks are not included.
The demonstration repeats a deterministic input window while carrying state;
it tests computation and memory capacity, not language-model quality. Exported
checkpoints now include final recurrent state and the next window index, as well
as model weights. There is no checkpoint-resume command yet. The independent verifier holds short-window activations in RAM,
and cross-entropy needs batch logits to fit one task's budget.
