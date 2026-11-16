# PostgreSQL larger than the database memory cap

| | |
|---|---|
| **Date** | 2026-10-22 |
| **Path** | ZeroMQ, storage-backed (PostgreSQL), Docker memory caps |
| **Result** | 40.2M parameters with a 256 MiB database cap; two updates verified (6.20e-15) before stopping for disk safety; heavy repeated I/O. |
| **Evidence** | `runs/postgres-capacity-256m/summary.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

The capacity driver can cap the **whole database container**, including its Linux
page cache, separately from clients. Limiting `shared_buffers` alone would not
prove that tensors cannot stay cached.

```sh
ygaraty experiment stream_demo \
  --output runs/postgres-capacity-256m --model language \
  --corpus runs/store-comparison/1-postgres-durable/corpus.txt \
  --storage-profile postgres-durable \
  --database-memory-mib 256 --shared-buffers-mib 32 \
  --memory-mib 128 --hidden-dim 3072 --tile-size 512 \
  --workers 3 --steps 3 --require-out-of-cache
```

This uses a single-layer float64 LSTM with embeddings and Adam, batch 2,
three-token windows, and 512-element tiles. The flag requires weights alone to
exceed the database memory cap, measured block-device reads during training,
and file-cache refaults. Database and clients have swap disabled. PostgreSQL
retains synchronous commits, fsync, and full-page writes.

`storage-snapshots.json` records cgroup memory limits, peaks, memory-pressure
counters, file-cache refaults, and block-device I/O at each phase. The summary
reports training I/O separately from a subsequent explicit PostgreSQL checkpoint
that flushes remaining dirty buffers. PostgreSQL buffer reads/hits and timed I/O
are supplementary counters: a PostgreSQL buffer miss can still hit the OS cache.
The cgroup I/O counters measure Docker's Linux guest block device, **not physical
host SSD traffic**; the host may cache the virtual disk. Memory peaks include
startup and import. Training timings exclude import, export, verification, and
the explicit database checkpoint. This is a capacity experiment on one machine,
not a projection for large-scale distributed training.

Measured on 2026-10-22 (`runs/postgres-capacity-256m`): 40,153,344
parameters, 306.35 MiB of weights plus 612.69 MiB of Adam state. PostgreSQL
had a 256 MiB container cap and 32 MiB shared buffers; three workers and every
other client had 128 MiB caps. **The run was interrupted during the third window
for host disk safety.** The last committed two-update checkpoint was exported,
and the disposable database volume was removed before verification.

| Measurement | Observed |
|---|---:|
| Completed update times | 171.06 s; 285.57 s |
| Completed-window throughput (12 tokens) | 0.0263 tokens/s |
| Database guest-device reads through interruption | 73.38 GiB |
| Database guest-device writes through interruption | 65.38 GiB |
| Database directory at interruption | 13.64 GiB |
| Worker peak RSS (three workers) | 74.57; 77.49; 74.07 MiB |
| Coordinator peak RSS | 74.62 MiB |
| Worker tracked tensor-buffer peak | 8 MiB |
| Database cgroup peak, including file cache | 258.71 MiB |
| Database OOM kills / swap allowed | 0 / 0 |

I/O deltas exclude import but include two complete updates **and the partial
third window**, plus a brief pause/resume for counter collection. They cannot
be divided by completed-update time to claim a disk bandwidth. The kernel's
recorded memory peak slightly exceeded the configured 256 MiB cap during
reclaim; the cap was enforced and no OOM kill occurred. Millions of file-cache
refaults and guest-device reads confirm that the workload did not fit the
container's caches. This demonstrates capacity, but also severe repeated I/O
and substantial transient table-space growth. It is not a matched comparison
with the earlier, smaller cached benchmark.

The rescued two-update checkpoint passed independent block-streamed verification
of all 40,153,344 weights, 80,306,688 Adam values, and recurrent state. Maximum
absolute discrepancy was 6.20e-15. Verification also ran under 128 MiB, with no
OOM kills. The saved `summary.json` explicitly marks the interrupted run; raw
I/O counters are in `interrupted-resources.json`. Temporary database storage was
removed, restoring approximately 15 GiB of free host space. The exported
checkpoint remains available (about 919 MiB).
