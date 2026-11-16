# Streaming import and verification under 128 MiB container limits (MLP)

| | |
|---|---|
| **Date** | not recorded |
| **Path** | ZeroMQ, storage-backed (Redis), Docker memory caps |
| **Result** | 25.2M-parameter (192 MiB) MLP trained with every client capped at 128 MiB; all parameters matched the streaming reference. |
| **Evidence** | `runs/stream-capacity/` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

The capacity-tested path uses `ygaraty/models/synthetic.py` to build a **shape-only tape**.
Its JSON recipe describes dimensions, initialization seed, and tile layout.
`ygaraty/experiments/redis_seed.py --model-config config.json` generates and publishes one block at
a time. No full parameter array is allocated. Initial values are reproducible
from the recipe; the block layout is part of the initialization seed scheme.
Recipes support the two-layer ReLU MLP and the single-layer LSTM below;
arbitrary legacy Python tape factories remain the explicit in-memory option.

`ygaraty/verification/stream_verify.py` independently recomputes the selected model's forward pass, cross-entropy,
backward pass, and SGD updates. It does not execute the distributed tape's kernel
or scheduling implementation. It streams initial weight blocks from the recipe,
and stores intermediate expected parameter versions as disk blocks. After a
training step, the preceding reference version is removed. For the last step,
each expected block is compared numerically against its exported counterpart,
including shape, coverage, file-size, checksum, and path checks. No full model or
full weight matrix is reconstructed. Activations still fit in RAM; larger batches
would require further activation partitioning. Scratch disk usage can reach two
parameter versions and is removed on success or failure.

Build the reproducible Linux runtime and run the hard-limit demonstration from
this directory (requires Docker):

```sh
docker build -f assets/docker/streaming.Dockerfile -t ygaraty-streaming:local .
ygaraty experiment stream_demo --steps 2 --workers 3 --memory-mib 128 --output runs/stream-capacity
```

The driver itself handles metadata only. It starts an isolated Redis container
and separate importer, coordinator, worker, exporter, and verifier containers.
Every client has `--memory 128m --memory-swap 128m`, which imposes a 128 MiB cgroup
limit with swap disabled. Each client records the limits it sees; the driver
checks them against Docker inspection, exit status, and OOM-kill status. Redis is
the deliberately larger storage tier (2 GiB container limit, 1.5 GiB maxmemory),
not a compute client. Containers and their network are removed after the run;
checkpoints and reports remain on disk. Add `--small` for a fast smoke test.

The default capacity recipe uses batch 16 and a 2048 → 8192 → 1024 MLP. Its
25,175,040 float64 parameters occupy **192.07 MiB**, exceeding every client's
**128 MiB** limit. In the verified two-step run, all clients exited successfully,
all final parameters matched the independent streaming reference, and cleanup
removed every Redis tensor block. Peak tracked tensor buffers were 0.5 MiB for
import/export, 2–3 MiB per worker, 5.5 MiB for verification, and 16 bytes for the
coordinator. Observed process RSS peaks were about 48–58 MiB. The verifier's disk
page cache reached the cgroup ceiling and was reclaimed without an OOM; RSS and
cgroup memory charges are different measures.

Outputs include `summary.json`, per-client memory reports, Docker limit/exit
records, the model recipe, a sharded checkpoint, and `verification.json`. Reported
RSS includes interpreter/library overhead. Tracked NumPy buffers exclude hidden
library temporaries, while the enforced container limit covers the client as a
whole. The large proof run used three workers and verified every parameter;
unit tests separately compare the streaming reference with untiled autodiff on
an uneven small network and reject a corrupted checkpoint even when its checksum
has been recomputed.
