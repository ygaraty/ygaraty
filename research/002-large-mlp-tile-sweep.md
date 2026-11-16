# Larger-network tile, cache and grouping sweep

| | |
|---|---|
| **Date** | not recorded |
| **Path** | ZeroMQ, in-memory, 3 workers |
| **Result** | Procedure for a repeated tile-size sweep (32–256) on a 164k-parameter MLP; results are not summarized here. |
| **Evidence** | `runs/large-sweep/summary.json` (when run) (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

The deterministic `ygaraty.examples.large_mlp` uses 128 examples, 256 input features,
512 hidden units, and 64 classes (164,416 trainable parameters). Run a single
comparison with configurable tiles and optional grouping-only isolation:

```sh
ygaraty experiment compare --tape ygaraty.examples.large_mlp --tile-size 128 --workers 3 --steps 3 --modes baseline cached grouped cached-grouped --output runs/large-128
```

For a repeated sweep over tile sizes 32, 64, 128, and 256:

```sh
ygaraty experiment benchmark_large --repeats 3 --output runs/large-sweep
```

The sweep shuffles tile/repeat order and rotates execution-mode order. Each run
starts fresh workers and caches and uses three training steps. `summary.csv` and
`summary.json` report median compute time and payload bytes per step, median wall
time across the second and third steps, and the observed wall-time range.
`process_wall_seconds` additionally includes startup, shutdown, and snapshot I/O
for the three-step run, excluding reference verification. All 18 original graph
values and final parameters are compared with local execution for every run.
These are local TCP experiments with three CPU worker processes, not a WAN or GPU
benchmark. Worker compute time excludes hashing, serialization, and transport;
use wall time and total payload bytes when judging whether an optimization pays.
Large sweeps retain full tensor snapshots and can occupy several GB.
