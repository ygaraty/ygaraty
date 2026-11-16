# Redis-backed, memory-bounded training path

| | |
|---|---|
| **Date** | not recorded |
| **Path** | ZeroMQ, storage-backed (Redis) |
| **Result** | Coordinator exchanges tensor references only; 50 MB MLP trained by 3 workers, final weights verified against local training. |
| **Evidence** | `runs/redis-demo/` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

`ygaraty/storage/tensors.py` defines the `TensorStore` interface and `RedisTensorStore`
implementation. The `redis_*` entry points use the same tape, numerical kernels,
and dependency scheduler as the original demo, but exchange **references only**:

```text
Python tape importer → immutable Redis blocks + metadata bundle
                                ↑              ↑
                         worker reads     worker writes
                                ↑              ↑
coordinator → task + version IDs → workers → result version IDs → coordinator
```

The coordinator never loads weights, gradients, or activations. It reads one
8-byte loss scalar per step. SGD updates execute as tiled worker tasks, and the
coordinator publishes the next parameter set only after all updates finish.
The preceding parameter versions remain live until this publication completes.

## Run a complete demonstration

Install the updated requirements, then start an isolated local Redis instance:

```sh
pip install -e .
docker run --rm -d --name tinyad-redis-demo -p 127.0.0.1:6387:6379 redis:7-alpine redis-server --save '' --appendonly no --maxmemory 512mb --maxmemory-policy noeviction
ygaraty experiment redis_demo --workers 3 --steps 2 --output runs/redis-demo
```

The default example is `ygaraty.examples.memory_mlp`: a 1024 → 4096 → 512 ReLU MLP,
64 examples, 50,368,512 bytes of float64 parameters, and 256-sized tiles. The
runner imports the model in a separate process, starts the coordinator and three
workers, exports a checkpoint one block at a time, verifies losses and every
final weight against local training, and cleans up its unique Redis namespace.
It terminates its child processes even on failure. Stop the disposable Redis
service when finished:

```sh
docker stop tinyad-redis-demo
```

For a manual run, use a fresh namespace and paths:

```sh
ygaraty experiment redis_seed --namespace my-run --tape ygaraty.examples.memory_mlp --tile-size 256 --output runs/my-run/bundle.json
ygaraty stored-server --bundle runs/my-run/bundle.json --steps 2 --min-workers 3 --output runs/my-run/training
```

In three additional terminals, change the worker name/report path for each:

```sh
ygaraty stored-worker --namespace my-run --name alice --block-size 256 --max-task-mib 32 --report runs/my-run/alice.json
```

After training, stream the final checkpoint to files and remove its Redis data:

```sh
ygaraty experiment redis_export --checkpoint runs/my-run/training/checkpoint.json --output runs/my-run/checkpoint --cleanup
```

All commands accept `--redis-url`; server/worker defaults use ZeroMQ port 5588.
Use trusted workers and a private network. The example Docker configuration is
explicitly ephemeral; it is not a durable Redis deployment.

## Storage and lifetime rules

- Each tensor version is immutable. Worker output versions contain the attempt
  token and task ID; inputs always name exact versions, never a mutable “latest”.
- Arrays are independently checksummed binary blocks. Slice reads fetch only
  intersecting blocks, then copy the requested region into a task-local array.
- Assembly publishes a manifest referencing existing blocks, without loading or
  concatenating the full matrix. Transpose streams one block at a time. Partial
  sums stream their inputs instead of retaining all partial products together.
- Every live manifest owns one reference to each distinct physical block it
  uses. Redis scripts atomically publish manifests and update reference counts.
  A block is deleted only when the last owning manifest is deleted.
- The coordinator tracks remaining consumers. It releases a tensor manifest
  after its last consumer commits, except constants and parameter versions that
  must remain available. A retry therefore retains its required inputs.
- Attempt leases fence writes. On timeout, the coordinator revokes the attempt
  and deletes its orphan manifests/staged blocks. Late workers cannot publish
  into expired attempts. Unpublished blocks also have a one-hour staging TTL;
  live published blocks do not expire.
- Namespace cleanup first closes the run, then removes only its data. It never
  uses `FLUSHDB`. A small closed-state tombstone expires after an hour so delayed
  workers remain fenced. On a coordinator crash, explicitly clean up or recover
  the run; automatic crash recovery/garbage collection is not implemented.
- `noeviction` is required for authoritative tensors. A full Redis store causes
  writes to fail; this prototype does not implement storage backpressure or
  spill-to-disk. The atomic scripts target a single Redis instance, not Cluster.

`checkpoint/manifest.json` maps each parameter to its version, shape, dtype,
block files, regions, and SHA-256 checksums. It appears only after every block
has been written. `training/checkpoint.json` contains Redis references and is
not an independent durable backup. Export before deleting Redis data. The
optimizer is SGD without momentum, so there are no additional optimizer tensors.

## What the memory measurements prove

`summary.json` includes coordinator and worker measurements. Individual worker
reports list every task, input version, fetched/written bytes, and memory usage.
Coordinator `events.jsonl` includes manifest release events.

- **Tracked tensor buffers:** distinct NumPy backing buffers allocated by storage
  reads, explicit outputs, and block writes, deduplicating array views. Each
  successful worker task must return this count to zero before acknowledging.
- **Sampled RSS:** process resident memory sampled every 5 ms, including Python,
  NumPy, allocator overhead, serialization, and numerical-library allocations.
  It is an observed peak, not an enforced limit or guaranteed instantaneous peak.
- **Task budget:** `--max-task-mib` rejects tasks whose conservative buffer
  estimate exceeds the requested size. It is admission control, not an OS RSS
  limit. Whole-tensor ReLU/loss/bias operations still require their task's arrays
  to fit. Operations exceeding the budget need further partitioning.

This Redis path deliberately has no persistent worker array cache and uses
ungrouped tasks to make lifetimes auditable. Existing cached/grouped experiments
remain available through `ygaraty/runtime/zmq_server.py` and `ygaraty/runtime/zmq_worker.py`. Original intermediate
arrays are reclaimed; their task history remains in the event log.

The training coordinator and workers never materialize the complete model.
The legacy `ygaraty/experiments/redis_seed.py --tape` importer and `ygaraty/experiments/redis_demo.py` local verifier
still materialize arrays. Use the fully streaming recipe and capacity test below
to avoid that limitation, including during import and verification.

Run all tests against an isolated Redis instance:

```sh
REDIS_TEST_URL=redis://127.0.0.1:6387/0 python -m unittest discover -s tests -v
```

Redis tests cover immutable versions, sliced fetches, shared-block lifetimes,
corruption checks, expired attempts, orphan cleanup, isolation between namespaces,
three-worker training/checkpoint export, and a worker disappearing after writing
an unacknowledged result. They skip when `REDIS_TEST_URL` is absent.
