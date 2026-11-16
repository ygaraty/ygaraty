# Offline instrumentation review — 2026-10-28

The instrumentation is prepared for an external capture pilot. The review fixed correctness and overhead issues, but does not certify negligible overhead on DigitalOcean. No training, live services, sockets or remote jobs were run here.

## Findings fixed

| Priority | Finding | Resolution |
| --- | --- | --- |
| High | `Recorder.close()` could wait forever trying to enqueue its sentinel into a full queue while the writer was stalled. | One deadline covers sentinel insertion and thread join; a timeout marks capture failure. Writer failure stops accepting events, and cannot turn into a clean footer. |
| High | The extractor accepted missing update plans/batches and detected duplicate plans only after writing a valid report and partial outputs. | Validate required plan/batch/boundary sets, duplicates, per-job worker counts and coordinator identities before exporting any calibration data. |
| High | Ready events inherited the triggering parent task's lease hash from the context, giving child tasks a false attempt association. | Ready events explicitly clear the attempt hash. Planning/background-update contexts also clear inherited task/worker fields. |
| Medium | The generic decorator required a positional first argument, breaking keyword-only use of `assemble_region`. | Preserve positional/keyword calling conventions for both functions and methods. |
| Medium | Capture-off paths still hashed attempts and timestamped eager uploads. | Guard those operations; retain the original lock context manager on the disabled path. Small wrapper/context-manager overhead remains and is quantified below. |
| Medium | Lock/post-queue logging occurred inside the locks being measured, and queue timings included tracing work done after dequeue. | Emit lock-wrapper metrics after releasing that acquisition, emit post-queue metrics outside the poster condition, and capture dequeue timestamps before hashing/logging. Other scheduler events and nested reentrant scopes can still run inside outer job locks; they remain part of observer overhead. |
| Medium | Every update deep-copied already detached task descriptions again and hashed its graph on the planning/request thread. | Keep one detached snapshot; hash it on the writer. Preserve `writer_hash_ns` separately. Snapshot construction and writer/GIL contention remain real overhead. |
| Medium | Negative/nonfinite timing data and unsafe capture identifiers were insufficiently checked. | Reject invalid durations, out-of-order update boundaries, nonfinite JSON, non-UUID capture identifiers and nonpositive expected counts. Invalid captures cannot export calibration samples. |

Also preserve NumPy version even when optional BLAS introspection is unavailable, and allow platforms without Python's `resource` module to import the instrumentation.

## Local timing probe

Five repeats on the local macOS ARM64/Python 3.12.2 host, with 1,000 synthetic calls per repeat (three snapshots per repeat). No kernels or network operations: the HTTP connection is a fake object, and snapshots contain 1,000 synthetic scale-task descriptions. Times below are medians of per-repeat caller cost. All measured captures had zero drops and no writer failure.

| Probe | Before review | Reviewed |
| --- | ---: | ---: |
| Decorated no-op, capture off | 0.47 µs | 0.43 µs |
| Span context, capture off | 1.40 µs | 1.38 µs |
| Span context, capture on | 4.56 µs | 4.59 µs |
| Fake HTTP exchange with phase spans, on | 41.74 µs | 44.78 µs |
| 1,000-task snapshot, caller time | 9.57 ms | 3.87 ms |
| Snapshot including final writer drain, amortized | 9.96 ms | 5.67 ms |

The bare function call measured 0.05 µs. Small before/after differences are not confidence intervals or guaranteed regressions/improvements. JSON serialization, filesystem flush and GIL contention contribute to enabled measurements; low-contention synthetic calls do not reproduce a two-vCPU worker or the shared eight-vCPU server. The large snapshot improvement comes from removing the redundant copy and moving hashing off the caller. The remaining per-update observer effect cannot be inferred by adding these microseconds or subtracting all trace durations from wall time.

The [raw probe output](review-overhead.json) preserves samples, source-module hashes and drop/failure counts. Run the current implementation's probe entirely offline:

```bash
ygaraty experiment capture_overhead --output /absolute/path/to/new-overhead.json
```

`--baseline` is optional and loads a **trusted local Python source file**, not JSON. The pre-review draft used for the table is archived in the local parent workspace audit directory; it was not a deployed revision. Provenance scanning for an archived module is explicitly scoped to the current package tree.

## Verification and deployment limits

- 18 instrumentation tests passed, with fake HTTP, synthetic metadata, blocked socket connections, writer failure/full-queue shutdown tests, correlation tests, invalid-capture tests and keyword-call compatibility.
- The 10 existing offline simulator/reference tests passed. The shape-only planner/CLI test also passed with capture enabled. Those prior simulator/reference changes are separate from this instrumentation commit.
- No live integration or DigitalOcean performance claim is made. Use capture-off/on pairs on identical external workloads, compare steady updates 2–21 and reserve independent repeats for validation.
- A full queue or failed/truncated trace is unusable for calibration. OS CPU steal, Go relay internals, PostgreSQL waits, effective bandwidth and clock offsets still need the separate collectors in the checklist.
- Nested/concurrent spans overlap. Thread CPU measures only the calling thread. HTTP waits include remote service time. Use task/attempt and request joins plus the DAG; do not sum stage totals or fit a universal runtime multiplier.
