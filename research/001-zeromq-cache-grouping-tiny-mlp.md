# Caching and task grouping on the tiny MLP

| | |
|---|---|
| **Date** | not recorded |
| **Path** | ZeroMQ, in-memory, 3 workers |
| **Result** | Grouping cut tasks/step from 76 to 62; at 2×2 tiles cache metadata cost more bytes than it saved. Single run. |
| **Evidence** | `runs/comparison/comparison.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

Run a reproducible comparison that starts and cleans up its own workers:

```sh
ygaraty experiment compare --workers 3 --steps 10 --output runs/comparison
```

It runs baseline, cached, and cached-plus-grouped modes, saves logs and
`comparison.json`, and verifies every original graph value and final parameter
against local execution at `1e-12` tolerance. Use a fresh output directory.

In one local three-worker, ten-step run of the tiny MLP:

| Mode | Tasks/step | Total array bytes | Total payload bytes | Sum of step wall times |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 76 | 97,760 | 683,336 | 0.442 s |
| Cached | 76 | 87,528 | 830,612 | 0.405 s |
| Cached + grouped | 62 | 82,728 | 763,510 | 0.370 s |

All modes matched local execution. These are single-run observations, not a
speedup guarantee. At 2×2 tile sizes, cache metadata costs more bytes than the
arrays it saves; the separate counters make that overhead visible. Larger
workloads and repeated measurements are needed to assess practical performance.
