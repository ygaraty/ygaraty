# Offline simulation reference suite

Run the synthetic demonstration from the repository root:

```bash
ygaraty simulation-reference \
  public/research/simulation-references/synthetic-suite.json \
  --output /tmp/ygaraty-reference-report.json
```

This example has a known task count but no measured timing evidence. Its warnings are intentional. Replay executes the scheduler on simulated time; it does not execute tensor kernels, open a database, start servers, or fetch inputs.

## Reference format

A suite is JSON with `schema_version: 1` and a nonempty `cases` list. Each case has:

- `name`: unique case name.
- `kind`: `measured` for real evidence, or `synthetic` for fixtures.
- `plan`: local path to a versioned snapshot. Paths resolve relative to the suite.
- `cost_model`: optional local JSON path; otherwise the bundled model is used.
- `simulation`: settings accepted by `Simulation` (workers, updates, RTT, bandwidth, coordinator costs, wire size, noise, etc.). The report records resolved defaults. Seeds are specified separately.
- `seeds`: distinct nonnegative integers, default `[0, 1, 2]`.
- `reference`: `plan_sha256`, `source_revision`, `dataset_batch`, `hardware`, and `evidence` identifying the measured run. Missing provenance produces warnings. A supplied plan hash must match.
- `measured`: optional normalized metrics, as below.
- Alternatively, `profile`, `profile_result`, and `profile_rows`: local `http_profile` JSON, result index, and explicit row indices. Rows are zero-based positions in the saved `rows` list, which already excludes warmup. Select steady or publication updates deliberately. Manual metrics may supplement imports but may not overwrite them.

A matching hash establishes graph identity only when it was captured with the measured run. Do not assign today's graph hash to an old run merely to suppress a warning. Snapshot serialization includes tasks, shapes, attributes, reusable leaves, and ownership metadata; it excludes tensor payloads. It can still include token IDs in task attributes, so keep snapshots local as appropriate.

## Creating snapshots

A current offline simulation can save its graph and result:

```bash
ygaraty simulate run \
  --dataset-url sqlite:////absolute/path/to/local-dataset.sqlite \
  --owned-layers --owned-vocab 8 --float32-wire --activation-wire bf16 \
  --workers 8 --rtt-ms 50 --down-mbps 100 --up-mbps 20 \
  --plan-output /tmp/ygaraty-plan.json --output-json /tmp/ygaraty-result.json
```

Planning requires an existing absolute local SQLite file URL (four slashes before the absolute path), and rejects remote database URLs before opening a connection. This needs the local dataset/version used by the planner. Replay of the saved snapshot does not. Use `save_plan(path, plan, provenance)` to capture an existing compatible plan in Python. Measurements on another authorized machine must capture their actual graph; a current offline export does not recover historical plans. The opt-in calibration capture also exports per-update snapshots; see ../calibration-capture/README.md.

## Metric definitions

- `seconds/update`: last simulated update, compared with the median of selected measured updates.
- `tasks`: planned task count versus measured count when available.
- `worker/<stage>/seconds`: mean over workers for the last simulated update. Stages are `compute`, `claims`, `empty claims`, `sleeping`, `fetch`, `upload in claim`, `posting`, and `stream upload queue`.
- `relay/down/GB`, `relay/up/GB`: simulated tensor payload bytes divided by 1e9, excluding protocol headers. Only supply measured values with the same accounting, after correcting historical bf16 counter issues.
- `family/<name>/tasks`, `family/<name>/in_flight`, `family/<name>/queue_ms`: task count, summed lease-to-commit seconds, and mean ready-to-lease milliseconds. These are not additive shares of update wall time.

Profile import maps worker `Kernel`, `Empty claims`, and `Sleeping (no ready task)` and matching family counters. Other worker spans and server timings retain `profile-worker/...` and `profile-server/...` names and appear as **not modeled**. The legacy profiler's network spans split encode/decode time and combine transfers differently; do not equate them with simulator fetch/posting stages. Missing measurements are **missing_measurement**, never zero. Profile metrics missing in any selected row are excluded from aggregation.

Reports contain per-metric measured values, simulated medians, seed minima/maxima, signed deltas, and percentage errors. Percentage error is undefined (`null`/`n/a`) for a zero baseline; absolute error remains available. Seed ranges are scheduling variation, not uncertainty bounds for model accuracy. There is no automatic pass/fail accuracy threshold: a successful exit means replay completed, not that the simulator is calibrated.

## Cost coverage

Each operation reports modeled call count, work range, sample count, fallback use, and out-of-range calls. Warnings identify unmeasured fallback operations, zero calibration samples, constant-cost fits, unknown calibration ranges, and extrapolation. Extra work attached to rewritten tasks is included; coordinator metadata tasks are excluded from kernel coverage.

`simulate fit` now saves `work_min` and `work_max` per operation. Old cost files still load, but their ranges are unknown. `run`, `sweep`, and reference reports expose warnings; legacy `validate` remains the historical aggregate comparison. Use the reference suite for detailed diagnostics. Refit only from already-local traces and write to a separate cost file.

Reports include hashes of the suite, graph, cost model, imported profile (when used), simulator, reference runner and scheduler sources for reproducibility. Exact replay can still change with scheduler/dependency versions; record the checkout revision and environment alongside results. Neither a matching graph nor these hashes verifies dataset identity or timing provenance automatically.

## Legacy kernel-only references

A case with `mode: "kernel"` compares existing compute-only observations directly with the cost model, without replaying a network protocol. It uses `samples` (local JSON path), required `samples_sha256` (canonical JSON hash from `digest`), optional `cost_model`, `kind`, and `reference` provenance. The sample file has `schema_version: 1` and a nonempty `samples` list of `{op, work, seconds}` records; additional task/step evidence is preserved. Durations and work must be finite and nonnegative. A sample may include `extra_work: [[op, work], ...]`; its predicted duration sums all component costs, while measured duration remains attributed to the primary operation. Coverage includes the extra component calls.

The report compares per-operation mean seconds over the same samples and includes cost coverage warnings. Seed bounds are null because no scheduler is run. A kernel comparison cannot validate transport, coordinator, or full-update time, and missing hardware/source provenance warns. The local workspace's `calibration-audit/local-kernel-suite.json` contains extracted historical observations; see the workspace `SIMULATIONS.md` audit section for selection rules and limitations. The bundled synthetic replay example remains separate.


## Worker-protocol and batch-window diagnostics

The simulator now limits streamed/eager uploads to two concurrent result uploads per worker, matching `StreamPoster(uploaders=2)`. `--stream-uploaders 0` restores the old unlimited-upload approximation for a controlled comparison; it does not undo other fixes. Empty pushed-input claims also pay their response/return-trip cost. These defaults follow the worker code, not a fitted experiment-37 correction factor.

`--batch-start 1 --batches 20` selects zero-based batches 1–20 (training updates 2–21). Use an explicit local SQLite dataset URL. JSON reports include `stream_uploaders`, per-worker `stream_upload_peak`, and mean per-worker summed `stream upload queue` seconds. Queue time sums waits across results; it can overlap computation and other uploads, so it is not additive wall time or an attribution of the measured residual.

Reference-suite `simulation` objects accept `stream_uploaders` as well. Cost coefficients and the default coordinator/network rates remain unchanged. A sensitivity scenario that happens to fit an aggregate runtime is not evidence that its hypothesized mechanism caused the observed delay.


## Review and validation (2026-10-28)

The reviewed implementation fixes the following issues before committing this suite:

- Plan-cache keys include the dataset URL; reusing one process for two datasets no longer replays the first dataset's graph. Datasets are assumed immutable during a process; restart/clear the cache after modifying a dataset in place.
- Planner store/relay overrides are scoped and restored even on exceptions. Planning only accepts existing absolute SQLite paths; replay still requires no dataset or connection.
- Constructor and reference settings share typed validation. Boolean protocol flags, integer counts, numeric costs and worker-profile values are checked separately; `placement` and `profiles` are supported. Per-worker RTT now affects modeled requests/replies. Capacity placement requires captured piece metadata, and never benchmarks hardware.
- Eager claims with `max_tasks=1` follow the real worker's non-eager path. Two streamed upload slots and empty pushed-claim response timing remain covered by tests.
- Invalid/nonfinite costs and measurements are rejected before aggregation. Missing profile stages remain missing. Kernel references include explicit extra work and warn when synthetic.
- Snapshot leaves contain shapes only, excluding backend descriptors or payloads. Replay checks graph shapes, input regions, reusable values, owners and cycles before simulation. The production scheduler validates dependencies in linear time.
- A plan snapshot and result JSON cannot target the same file; a single snapshot path cannot silently overwrite multiple batches. Stalled simulations cannot report partial updates as successful.

The replica rewrite was inspected and already copies its input before mutation; its behavior is unchanged. No global cost coefficients were changed or fitted.

Validation: 18 simulator/reference tests and 18 instrumentation tests passed offline; the synthetic CLI replay and all 38 local kernel-only cases (40,118 samples) completed. Tests use shape-only planning with local SQLite fixtures, scheduler simulation, recorded JSON and mocked network/benchmark calls. No training, services, downloads, hardware benchmarks or online tests ran. Reference warnings still apply: these legacy kernel cases do not calibrate DigitalOcean transport or c-2 hardware.

## Upload stages and placement-ranking audit

The replay engine accepts the following optional simulation settings (CLI names replace underscores with hyphens):

| Setting | Default | Meaning |
|---|---:|---|
| `upload_block_kib` | 0 | Zero keeps the aggregate relay request. Positive values split each result into capped blocks; this is an approximation, not captured tile/band layout. |
| `upload_request_concurrency` | 4 | Maximum block requests in an upload wave, inside each existing uploader slot. |
| `upload_convert_ms_per_mib` | 0 | Conversion/serialization cost per MiB of decoded float64 output. |
| `upload_hash_ms_per_mib` | 0 | Hash cost per MiB of encoded wire output. |
| `upload_request_ms` | 0 | Per-block client request setup cost; requires block mode. |
| `upload_relay_ms` | 0 | Per-block service on one shared serial relay service resource; requires block mode. |

These are **sensitivity controls, not fitted coefficients**. Defaults preserve historical aggregate timing; zero means an omitted cost, not a measured zero. Worker preparation serializes on a separate CPU resource, not on the compute kernel resource. Request waves wait for their slowest reply before admitting the next wave; real pools can refill continuously. Existing uploader-slot limits still apply. The hypothetical `relay_commit` shortcut rejects nondefault upload stages rather than silently skipping them.

Reports retain `upload_stage_model`, `upload_stage_warnings`, and `upload_stages_by_update` per worker. The reference suite exposes last-update worker-mean metrics as `upload/<stage>/seconds`, or `/count` for requests. Stages include conversion, hashing, preparation queue, request setup, request travel (uplink queue/transfer plus latency), shared relay link, relay queue/service, reply and streamed uploader-slot queue/notification. Stages overlap; never sum them to reconstruct wall time. Notifications are measured to commit; reply spans include only replies completed before simulation termination. Legacy transfer and the existing posting metrics are inclusive views, not extra costs to add.

Run the evidence audit without any dataset or services:

```bash
ygaraty simulation-ranking \
  public/research/simulation-references/placement-ranking.json \
  --output /tmp/placement-ranking-report.json
```

The default audits **archived predictions**, not today's engine. Experiment 42's 4/6/8-strong pools are diagnostic cases; experiment 37 A/B/C are validation-only. The former's values are rounded report means. Experiment 37's archived predictions are the earlier local matched-window audit (13.0396/9.2489/13.2150), not the predictions printed in the original experiment report. Their reconstructed data/ownership are not proven identical to the measured runs.

The four-strong case reverses the measured preference. Six strong workers have a positive predicted gain of 0.06 s, but that lies inside the declared **0.1 s practical tie band**; measured gain is 0.48 s. This band is an explicit regression criterion, not a confidence interval. Eight strong workers predict the right ordering. Gain magnitudes and signed absolute errors are always reported, including when ordering passes.

For fresh comparisons, pass `--predictions predictions.json` with:

```json
{
  "provenance": "Describe source revision, settings, measured-window selection and replay artifact paths here",
  "seconds": {
    "exp42-4-strong-paired": 0.0
  }
}
```

This snippet illustrates the shape only: replace it with **all nine case names** from the fixture and strictly positive measured-window aggregate predictions. Incomplete coverage is rejected so validation cases cannot be silently dropped. All fresh predictions are caller-supplied; the audit does not generate missing measured plans, enforce experimental separation externally, or fit any cost. Evidence, suite and supplied-prediction files are hashed in the output. `--check` exits 1 for a ranking failure; the archived baseline intentionally fails it.

Exact captured plans, worker profiles, block layouts and raw spans for experiment 42 are absent locally. Current replay/calibration is therefore blocked on those inputs, and no new end-to-end accuracy improvement is claimed. Do not fit a global “2× upload” factor from overlapping spans that include startup/publication. Keep experiment 37 out of coefficient selection and report it separately after any future fit.

## Transport calibration: experiments 50, 53 and 54

This is one step in building a simulator that can guide our sovereign computing
infrastructure: retain failures as evidence before trusting a scaling decision.

Run from the checkout, with an unused output filename:

```bash
.venv/bin/python -m ygaraty.simulation.transport_calibration \
  public/research/simulation-references/transport-calibration.json \
  --output /absolute/local/path/transport-audit.json
```

This command reads local report aggregates, fits three scoped quantities, and
writes source hashes, signed errors and held-out comparisons. It never starts
services, executes workload kernels, or downloads data. It does **not** refit
the discrete-event simulator's default costs. Experiment 37 is untouched.

The split is retrospective and explicit in the fixture: experiment 53 jobs 0/1
fit a common additive residual on the archived placement model B; jobs 2/3 are
held out. Experiment 54 home repeat 1 fits the pruning difference; repeat 2 home
and loopback are held out. Experiment 50 home is diagnostic only, without an
independent matching repeat. These are rounded report tables, not raw records.

| Held-out case | Measured s/update | Conditional estimate | Signed error |
|---|---:|---:|---:|
| 53 job 2, single layers | 8.58 | 8.715 | +1.57% |
| 53 job 3, pairs | 8.79 | 8.785 | −0.06% |
| 54 home repeat 2, pruned | 3.83 | 3.83 | 0% |
| 54 loopback repeat 2, pruned | 0.36 | 0.21 | −41.67% |

The common placement residual is 1.555 s, not an identified request or server
cost. The starting model B already includes experiment 42's per-piece overhead;
this does not show that the current event simulator ranks placements correctly.
Predicted placement gain is 0.07 s versus 0.21 s measured; both are practical
ties at the predeclared 5% threshold. Same-pool repeats are weak generalization
evidence. No confidence interval is available.

The pruning estimate uses the measured baseline as an input and subtracts the
0.15 s home-link gain. Its loopback failure is deliberately retained: a home-link
difference is not a transferable per-task service cost. The report prose gives
loopback means inconsistent with its rounded repeat table; this audit uses the
individual table rows, without reconstructing unavailable precision.

Experiment 50's 24 s slowdown divided by 879 additional requests × 50 ms gives
0.546 exposed RTT-equivalents per additional request. This is an effective
residual including concurrency and other unseparated costs, not a measured
request latency. Do not apply it to Python workers or extrapolate it to different
batch sizes. Unique block URLs defeat browser preflight reuse even with a long
max-age; a tensor cache and a preflight cache are different mechanisms.

### Event simulator changes and limits

Mixed-precision region reads now use the intersected upload bands' precision
rather than their whole-result average. Cache residency uses decoded float64
size independently of wire precision. Reports expose eligible/saved wire bytes
and hit/miss counts per update. Existing versioned immutable-input and bounded
LRU behavior remains covered by offline tests.

These are logical-region byte counts. Full-block overfetch, compression, framing,
browser pinning and browser preflight caching still require captured manifests
and traces. Experiment 54's 45 MB prediction used an unowned decoded plan;
the owned runtime saved 1.5 MB on the wire. The factor of 30 combines plan and
precision changes and must **not** become a compression multiplier.

### Still required for calibration

The referenced `runs/experiments/048/`, `runs/pods-data/exp53/` and
`runs/exp54/` profiles are absent from this checkout. To fit event-level costs,
supply already-collected local records containing:

- Exact owned/pruned plans, shapes, upload bands, dtypes and per-update batches.
- Per-request endpoint, method, preflight/cache status, bytes, queue/start/end
  times and concurrency, plus relay counters and fallback counts.
- Tensor cache keys/versions, hits/misses, evictions, pin/unpin events, decoded
  capacity and cold/warm boundaries.
- Per-update owner placement, kernel and upload spans, coordinator service
  timing, and aligned steady-state windows.

Reserve complete runs and hardware/link configurations before a future fit.
Do not sum overlapping worker spans into update wall time. Cache parameters and
causal request overhead remain unidentified here; broad ranking reliability has
not been established.

## Track quality across revisions

Run the offline snapshot command with a new local output directory:

    python -m ygaraty.simulation.quality --output /absolute/local/new-snapshot

It writes quality.json and QUALITY.md with source/evidence hashes, signed bias,
MAPE, worst error, ranking failures and missing calibration evidence. Add
repeatable --reference-suite paths for current offline replays and --previous
pointing to an earlier quality.json for same-evidence aggregate trends.
Historical, conditional and current replay evidence remain separate, including
synthetic status and stage warnings.

Whole-run kernel refitting and the next capture contract are in
[the capture guide](../calibration-capture/NEXT-RUNS.md). Identifiable
transport/cache fitting still needs raw evidence; no global residual multiplier
is substituted for it.

## Handoff protocol replay

[HANDOFF.md](HANDOFF.md) documents the existing non-blocking handoff simulator,
its integration with reference replay and bottleneck diagnostics, event-ordering
fixes, and calibration limits. Enable --handoff in the simulator CLI, or handoff:
true in a reference case's simulation settings, with eager claims and max_tasks
at least two. Historical L5 predictions are not revalidated by the offline tests.
