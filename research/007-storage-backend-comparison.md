# PostgreSQL tensor storage and Redis/PostgreSQL comparison

| | |
|---|---|
| **Date** | 2026-10-22 |
| **Path** | ZeroMQ, storage-backed (Redis and PostgreSQL) |
| **Result** | Memory-only Redis 2.19 tokens/s; durable Redis 0.77 and PostgreSQL 0.81 (overlapping ranges, no clear winner). Two runs each. |
| **Evidence** | `runs/store-comparison/comparison.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

PostgreSQL can replace Redis for the complete tensor store: tensor payloads,
versioned manifests, shared-block ownership, leases, and cleanup. The same
coordinator/worker protocol and numerical kernels work with either backend.
Use `--store-url postgresql://USER:PASSWORD@HOST:5432/DATABASE` on the seed,
server, worker, and export scripts. `--redis-url` remains a compatibility alias;
script filenames have not changed. Install the updated requirements (including
`psycopg[binary]`) or rebuild `ygaraty-streaming:local` (`--rebuild-image`).

`ygaraty/storage/postgres.py` implements `TensorStore`. Each block is an immutable BYTEA
payload with a checksum. Manifests reference blocks through a relational table;
assembly shares existing blocks without copying tensor arrays. Transactions
publish manifests and their references together. Run/lease locks fence expired
attempts and namespace cleanup; block locks protect shared ownership during
publication and deletion. Workers fetch binary query results one block at a
time. The shared `BlockTensorStore` implements bounded slice reads for both
backends. PostgreSQL uses lossless LZ4 payload compression (`STORAGE EXTENDED`)
and more frequent autovacuum of block and TOAST tables. The earlier benchmark
results below used the original uncompressed implementation.

Only the importer creates tables. Its database role needs schema/table creation
rights; worker roles need the relevant table permissions. This prototype uses a
single database/schema. Initialization upgrades the payload storage policy for
new writes without rewriting existing blocks; general schema migrations, separate
role provisioning, and database sharding are not implemented. Tensor metadata is durable, but the live
scheduler is still in memory: this change does not implement coordinator restart
or checkpoint-resume loading.

Unpublished PostgreSQL blocks have a one-hour expiration. `collect_expired()`
reclaims them; `stats()` invokes it, including at training-step boundaries. It
can also be invoked by external maintenance. Expiration is not a background
PostgreSQL TTL service. Namespace cleanup leaves a closed-run tombstone and
removes only that namespace's tensors/leases. PostgreSQL VACUUM reclaims dead
row space for reuse; logical deletion does not promise a smaller database file.

Run a durable PostgreSQL LSTM/Adam demonstration with hard client memory limits:

```sh
ygaraty experiment stream_demo --storage-profile postgres-durable --model language --benchmark --steps 3 --corpus ../tinyad/data/shakespeare/input.txt --output runs/postgres-lstm
```

Omit `--benchmark` to use the existing 140.31 MiB language-model capacity recipe.
The benchmark recipe is smaller: batch 8, four timesteps, embedding width 64,
hidden width 512, vocabulary 256, and 128 × 128 tiles. It measures an LSTM with
embeddings, Adam and global clipping, not only isolated database queries.

Compare the two durable stores and the original memory-only Redis baseline:

```sh
ygaraty experiment benchmark_stores --repeats 2 --steps 3 --workers 3 --corpus ../tinyad/data/shakespeare/input.txt --output runs/store-comparison
```

Each run starts fresh isolated containers and an anonymous database volume,
then removes them after collecting artifacts. Runs execute sequentially and
reverse profile order on alternate repetitions. Clients have 128 MiB and one
CPU each, with swap disabled. Storage servers have 2 GiB and two CPUs each.
PostgreSQL 17 uses 256 MiB shared buffers with fsync, synchronous commit, and
full-page writes enabled. Redis 7 uses noeviction and a 1.5 GiB memory ceiling.
`redis-durable` uses AOF with `appendfsync always`; `redis-memory` disables both
AOF and snapshots. Automatic AOF rewrite is disabled for these short benchmarks
so rewrite timing does not distort the comparison. The durable profiles both
sync writes, but have different transaction grouping and storage formats.

Artifacts include:

- `comparison.json`: median throughput/time/traffic/disk metrics and maximum
  observed worker memory across repetitions.
- `runs.json`: all complete run summaries, including versions and durability
  settings. Every checkpoint is independently verified, including Adam moments.
- Per-run `storage-snapshots.json`: empty store, after import, after training,
  and after export/cleanup measurements.
- Worker reports: tensor payload bytes, task histories, sampled/OS peak RSS,
  tracked tensor-buffer peaks, and observed memory limits.

Training time is the sum of coordinator step durations (including initial worker
registration). It omits housekeeping after each recorded step, import, export,
and independent verification. The measured first-task-to-final-event spans
were within about 1% of these step sums in this benchmark. Transferred tensor bytes count successful worker
reads/writes. Database-interface RX/TX deltas additionally include protocol and
metadata traffic between the importer-finished and workers-finished snapshots;
they exclude ZMQ coordinator traffic and are not a complete physical-wire bill.
Disk usage comes from `du -sk` on the database data directory and includes
retained WAL/AOF, indexes, metadata and dead space. It measures allocated space
inside the container filesystem, not the host Docker disk-image size. PostgreSQL WAL generation is
measured separately by WAL insertion-position advancement, not inferred from
current WAL directory size. Disk measurements are phase snapshots, not sampled
peaks or physical write-amplification counters. Exported checkpoint files are
separate from database disk usage.

The moderate benchmark can fit much of its active data in the storage server's
cache. Results characterize this implementation on one Docker host; they do not
predict a disk-bound 10B model or a wide-area volunteer network. Cleanup is
verified logically for both backends even when their on-disk files remain large.


## Measured storage comparison (2026-10-22)

Two fresh runs per profile, reversed order in round two: 1,329,408 parameters,
three workers, three four-timestep windows, batch 8 (96 byte tokens per run).
All six runs verified every parameter, Adam moment and final recurrent state,
executed 11,679 tasks, and reclaimed all live tensor blocks/manifests.

| Backend | Median tokens/s (range) | Median DB network traffic | Median allocated DB disk after training | Maximum worker RSS |
|---|---:|---:|---:|---:|
| Redis, no persistence | 2.19 (1.64–2.74) | 1.758 GiB | 4 KiB | 53.7 MiB |
| Redis, AOF/fsync always | 0.77 (0.53–1.01) | 1.758 GiB | 680.5 MiB | 52.4 MiB |
| PostgreSQL, durable | 0.81 (0.68–0.93) | 1.814 GiB | 1385.3 MiB | 66.6 MiB |

Every run read 1,075,690,848 tensor bytes and wrote 650,965,776 tensor bytes
(1.608 GiB combined); network counters also include database protocol and
metadata. Peak tracked tensor buffers were 0.5 MiB per worker for all profiles.
The remaining RSS includes the interpreter, drivers, and native allocations.
PostgreSQL generated a median 809 MiB of WAL during training. The disk columns
include retained logs and dead/reusable space; they are not the roughly 30 MiB
live model-plus-Adam payload, nor a steady-state disk-capacity prediction.

The durable profiles' ranges overlap substantially. These two local trials do
not establish a reliable winner between PostgreSQL and durable Redis. Both were
slower than memory-only Redis. PostgreSQL is a working complete replacement with
transactional disk-backed tensor storage, with higher measured worker RSS and
short-run disk occupation in this implementation. This cached, modest-size test
does not establish throughput for a 10B model larger than the database cache.

Raw results: `runs/store-comparison/comparison.json` and
`runs/store-comparison/runs.json`, plus per-profile storage snapshots, task logs,
checkpoints, independent verification reports, and enforced memory-limit records.
