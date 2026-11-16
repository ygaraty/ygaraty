# Offline handoff simulation

The checkout already contains HandoffSimulation, implementing the non-blocking
worker path described by experiments 59–60. The 2026-11-01 integration retains
that implementation and makes it usable through reference replay and the new
bottleneck diagnostics. The real scheduler and worker protocol are unchanged.

## Modeled behavior

- Lease a waiting task only when every uncommitted input can be traced through
  assemble/transpose metadata to an already leased compute producer.
- Respect established ownership, one unowned handoff per claim, and at most four
  outstanding handed tasks per worker.
- Send the relay read when the handed claim is received. Wait for the last named
  producer's upload to land before releasing the consumer's input transfer.
- Continue ordinary claiming while handed reads wait; serialize all kernels on
  the contributor's modeled compute resource.
- Keep result uploads and coordinator posts. Defer an early consumer result until
  its dependencies commit; relay availability itself is not a coordinator commit.

This is the non-blocking handoff path, not the original blocking worker arm of
experiment 58. It cannot reproduce that arm exactly.

## Local changes

Preleases now use simulated time instead of the scheduler's default wall clock.
Relay publication occurs at the upload's modeled landing event; consumer reads
are sent through the simulated uplink and reserve relay bandwidth after arrival,
rather than reserving a future transfer ahead of intervening events. Waiting
coroutines are deduplicated and awakened in stable order.

The non-pushed-input branch no longer sends a second claim reply; committed
inputs are fetched through the relay separately. Cache counters, per-worker
transfer demands and task compute intervals feed the diagnostic reports.
Per-update handoff records include source task IDs, lease/read-arrival/release
times, payload bytes and wait durations.

Upload hashing (upload_hash_ms_per_mib) is charged on handed-path uploads as
in Simulation.upload; L5's version skipped it. Other nonzero upload-stage
coefficients and relay_commit are rejected, rather than silently ignored.
handoff_reads="whole" reads every named result in full (experiment 59);
"region" (experiment 60) is the default. replay_sequence (engine.py) replays
consecutive batches with owners, caches and clocks carried over. Default ordinary simulation is unchanged.

## Usage

With an existing local dataset and the matching model/batch flags:

    .venv/bin/python -m ygaraty.experiments.simulate run --dataset-url sqlite:////absolute/path/to/existing.sqlite --eager-claims --push-inputs --max-tasks 4 --handoff --diagnostics --output-json /absolute/local/handoff.json

Repeat without --handoff for the ordinary protocol. Keep all other configuration
and input hashes fixed. A dataset is unnecessary when replaying a captured plan:
set handoff: true, eager_claims: true and max_tasks >= 2 in a reference case's
simulation object. Pushed inputs are optional; match the measured configuration.

Reference replay now selects HandoffSimulation explicitly, records the handoff
setting, hashes its source along with the diagnostics module, and retains its
limitations in the case's warnings. No default costs are refitted.

## Reading the output

handoff_by_update contains per-update counts and long-poll events; handed_tasks
is the total across updates. diagnostics continues to report bandwidth/compute
demands, the compute-weighted DAG path and last-completion dependency chain.

With handoffs, a consumer can compute before its producer commits. Therefore
dependency-chain stage totals can overlap and are marked
chain_stage_totals_additive: false. Do not sum them to reconstruct update wall
time. This does not make data dependencies optional: consumer input release still
waits for the producing upload.

## Evidence and limits

The experiment 58 report records a measured 3.988 to 3.182 s/update improvement
on emulated home links. It is a historical observation, not a validation of these
changes. Its raw runs/exp58 profiles and captures are absent in this checkout.

The later L5-RESULTS.md reports that its handoff-gain gate failed, despite better
placement estimates. Its numbers belong to the original source revision and
frozen calibration split. They have not been replayed with the event-ordering
fixes here. Preserve both the old predictions and the failed gate.

Remaining approximations:

- Logical pending-region bytes instead of exact recipe block overfetch, shared
  block deduplication, framing and compression.
- Aggregate named upload timing; hashing is the only upload-stage coefficient.
- Region reads are charged only their bytes: experiment 71 found this credits
  them with a 0.15 s gain per tiny update that experiment 60 measured as none.
- No failed attempts, expiry, cancellation, lease recovery or block-hash checking.
  Timing tests cannot establish numerical or protocol fault correctness.
- No explicit relay long-poll service/connection resource or measured host load.
- Existing compute and network cost calibration limits still apply.

Validation used shape-only task graphs, fake scheduling and blocked socket/kernel
entry points: 59 focused offline tests passed, including dependency ordering,
prelease clock, monotonic event scheduling, metadata chains, reference dispatch,
multi-update diagnostics and existing simulator regressions. No workload,
browser, coordinator, worker, relay or external job was run.
