# Reducing storage traffic while preserving exact results

| | |
|---|---|
| **Date** | 2026-10-22 |
| **Path** | ZeroMQ, storage-backed (PostgreSQL), Docker memory caps |
| **Result** | Transpose views and LZ4 compression: 9% faster training, 24% fewer reads, 35% less database disk; identical checkpoint checksums. |
| **Evidence** | `runs/storage-optimization-comparison.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

The storage worker now publishes transposes as **metadata views** of immutable
blocks. Rectangular edge tiles carry an orientation bit; reads verify the
physical payload checksum and then apply that view. Double transposes cancel,
and reference counting keeps blocks alive after their original manifest is
removed. Deploy matching worker/store code together: older readers do not
understand the orientation field. Export materializes each logical block and
computes its export checksum, preserving the existing checkpoint format.

PostgreSQL now uses lossless LZ4 TOAST compression, particularly useful for zero
Adam initialization and sparse/zero-heavy intermediate blocks. More frequent,
unthrottled block/TOAST autovacuum makes dead space reusable sooner; it does not
promise to shrink files after every update. Publication no longer rewrites an
already-cleared expiration field. Durability, float64 precision, immutable
versions, checksums, leases, and transactional reference cleanup remain enabled.
Initializing an existing schema updates the payload policy without rewriting
existing blocks. This requires PostgreSQL with LZ4 support (tested on the same
PostgreSQL 17 image used for the earlier measurements).

A matched two-update comparison on 2026-10-22 used hidden size 2048, input size
128, vocabulary 256, batch 2, window 3, tile 512, embeddings and Adam. Weights
alone occupied 140.31 MiB, exceeding PostgreSQL's entire 128 MiB container cap
(shared buffers 16 MiB). Three workers and all other clients had 128 MiB caps,
with swap disabled. Both versions had the same 5,852 tasks and input corpus.

| Measurement | Before | Optimized | Reduction |
|---|---:|---:|---:|
| Training time | 101.65 s | 92.29 s | 9.2% |
| Database directory after training | 5.75 GiB | 3.77 GiB | 34.5% |
| Guest block-device reads during training | 34.18 GiB | 25.89 GiB | 24.2% |
| Guest block-device writes during training | 23.44 GiB | 18.87 GiB | 19.5% |
| Worker tensor payload reads | 6.89 GiB | 6.07 GiB | 11.9% |
| Worker tensor payload writes | 4.13 GiB | 3.31 GiB | 19.9% |
| Highest worker RSS | 80.33 MiB | 84.20 MiB | — |

Both full checkpoints passed the independent reference, with maximum absolute
error 1.11e-14. **Every exported block checksum matched between the two runs**,
including weights, Adam state, and recurrent state. All 45 regression tests
passed, including transpose view lifetime/GC, rectangular slicing, double
transpose, and bit-for-bit lossless compression. Neither run had an OOM kill.
The optimized database still had 6.22 million file-cache refaults, confirming
that this was an out-of-cache workload.

This is one trial per version, baseline first; timing is sensitive to host load.
Guest-device counters are not physical host SSD telemetry. Training excludes
import/export/reference verification and the explicit post-training database
checkpoint; that flush added 1.59 seconds and about 174 MiB of writes in the
optimized run (recorded separately in the summary). Directory sizes are phase
snapshots, not peaks. The improvements do not eliminate substantial intermediate
traffic or establish long-run steady-state disk consumption.

The driver now checks a 4 GiB host free-space reserve before starting and while
waiting for clients (`--min-free-disk-mib`). Crossing it aborts the benchmark and
removes its disposable containers/volumes. This guard is polled, not a hard disk
quota, and does not promise checkpoint recovery from an aborted run. The
coordinator writes completed-step metrics incrementally. Database volumes are
removed after export/cleanup, before reference verification allocates scratch.

Results: `runs/storage-optimization-comparison.json`, with individual runs at
`runs/optimization-baseline-code/runs/case` and `runs/postgres-optimized-128m`.
The baseline source snapshot restores materialized transposes, uncompressed
payloads, original autovacuum settings, and unconditional expiration updates;
the benchmark harness and all other computation are shared.
