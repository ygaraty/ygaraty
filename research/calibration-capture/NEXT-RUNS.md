# Next-run calibration contract

We are building the evidence needed to guide sovereign distributed computing.
More fitted coefficients alone do not establish accuracy: each candidate must
predict runs it did not see. Prepared locally on 2026-10-30. All actual workload
runs below belong on a separately authorized execution machine.

## Required evidence per run

| Scope | Retain | Resolves |
|---|---|---|
| Identity | Run/repeat ID, preassigned fit/validation role, Git revision and patch hash, hardware/BLAS/browser versions, thread counts, configured links and probes | Comparable hardware and protocol |
| Actual plan | Per-update owned/pruned graph, exact batch IDs/lengths, dtype/upload bands, placement, task count and hash | Wrong plan or precision assumptions |
| Transfers | Block hash, shape, region, transposition, decoded/payload/frame/compressed-body bytes, retries and fallback count | Full-block overfetch versus logical-region bytes |
| Requests | Queue/send/headers/body timestamps, request and task-attempt joins, status, pool limits, server processing and lock spans | Transfer versus service and queue delay |
| Browser network | CDP OPTIONS/GET/PUT, hashed resource URL, timing, total encoded bytes, cache/service-worker flags | Unique preflights and reuse |
| Tensor cache | Capacity, decoded residency, hit/miss/insert/skip/evict, version-sensitive key hash, browser pin lifecycle | Bytes actually avoided |
| Update | Unrounded boundaries, warmup/publication labels, exact steady window, losses/retries/rejections | Throughput and correctness evidence |
| Observer | Clean footer, zero drops/errors, capture-on/off paired runs, CPU/memory, clock-offset estimates | Incomplete traces and observer overhead |

Never save bearer tokens, headers, payloads or job.json. Plans can contain token
IDs; keep them with dataset artifacts. Clock anchors only approximately align
hosts. Use local monotonic durations and causal joins. Do not add overlapping
worker spans to obtain wall time.

## Next comparisons

Fix batches, wire precision and task budgets within each pair. Prefer at least
three independent repeats per cell; reserve the third whole repeat before fitting.
Alternate A/B order. Retain slow and failed updates with explicit exclusion reasons.

1. **Exp54:** pruned/unpruned tiny Transformer on loopback and home links. Verify
   task count and relay bytes. Keep loopback as a separate link holdout: transferring
   the fitted home-link delta already fails there. Separate cold, steady and publication.
2. **Exp50:** browser inline/relay with identical groups and data on both links.
   Vary RTT at fixed bandwidth, then bandwidth at fixed RTT. Capture JS and CDP
   together. Batched PUT or concurrent uploads must be separate protocol variants.
3. **Exp53:** pairs/single layers with two strong workers; reserve a new pool or
   another strong-worker count as transfer validation. Record actual owners every
   update, distinguishing decoder, pair and vocabulary owners.
4. **Cache:** cold/warm immutable inputs, cache off/bounded capacity, fixed plan
   and scheduling. Capture pressure and pins. Parameters must change version keys
   each update; only immutable inputs qualify for cross-update reuse.
5. **Observer:** capture-off/on/off pilot before the matrix. If observer cost is
   material relative to the effect, reduce capture volume and repeat. Do not
   subtract summed trace durations from wall time.

The 30.3M recipe uses updates 2–21 of 22; exp54 used 2–12 of 12. Preserve each
window definition. Keep experiment 37 out of coefficient selection.

## Prepared instrumentation

Python coordinator/workers opt in through YGARATY_CAPTURE_DIR and optionally
YGARATY_CAPTURE_REVISION. Existing bounded asynchronous recording remains.
New records include relay.upload_block (dtype/shape/region/decoded/payload bytes),
worker.input_manifest (task/hashed attempt and actual covering blocks), worker.cache
(hashed key, worker, action/residency/capacity), relay HTTP retry/normalized route
and compression flags. Plan provenance includes placement_pairs and prune_plan.
HTTP byte counts exclude HTTP/TLS headers. Cache hits count lookups, not unique transfers.

The extractor exports transport-events.json only for otherwise valid captures,
with event-family coverage. Presence alone does not prove completeness: verify
per-worker/task/attempt joins before fitting.

Browser capture is optional, bounded, per JavaScript realm, and local:

    import {startCapture, stopCapture} from './sdk.mjs';
    startCapture(32768);
    // Existing authorized harness performs its run.
    const metadata = stopCapture(); // harness saves JSON locally, with run/worker ID

It captures relay fetch-to-headers spans, body/frame/payload counts, result block
bytes, cache actions and pin counts. No automatic upload is added. Browser cache
events omit keys: detailed version reconstruction still needs claim manifests.

For preflights, attach the prepared adapter to an existing synchronous Playwright
harness session with Python capture enabled:

    from ygaraty.experiments.browser_capture import attach
    session = page.context.new_cdp_session(page)
    attach(session, worker='tab-0')

The adapter does not launch a browser. It allowlists timing/status/byte fields
and hashes resource URLs, excluding headers, bodies and error strings. OPTIONS
is a separate request; fetch includes preflight wait, so never add both durations.
CDP and JS request IDs are different domains: compare counts/aligned windows,
not ID equality. CDP -1 means unavailable. Encoded total and JS payload bytes
have different scopes. Relay/server processing still needs server-side spans.

## Offline refit and quality workflow

Extract each complete run separately using the existing calibration_capture
command. A split manifest for kernel refitting has this form:

    {"schema_version":1,"cohort":"same-SKU-BLAS-threads","runs":[
      {"run_id":"repeat-1","experiment":54,"role":"calibration","extracted":"repeat-1-extracted"},
      {"run_id":"repeat-3","experiment":54,"role":"validation","extracted":"repeat-3-extracted"}
    ]}

    .venv/bin/python -m ygaraty.simulation.refit /absolute/local/split.json --output /absolute/local/new-candidate
    .venv/bin/python -m ygaraty.simulation.quality --output /absolute/local/new-quality-snapshot

The candidate is never installed automatically. The refitter rejects incomplete
captures, duplicate runs/capture identities, exp37 training, and fused extra_work
that cannot be assigned to a single operation. Cohort selection is the caller's
responsibility: separate strong/weak workers and BLAS/thread configurations.
Validation errors, missing operations and extrapolation remain visible.

Add repeatable --reference-suite paths to quality for current shape-only replays
and per-stage/byte/update comparisons. The reference-suite README defines those
inputs. Add --previous pointing to an older quality.json for same-evidence
aggregate trends. Existing output directories are refused. Keep source, evidence,
plan, cost-model and split hashes with every result.

Track signed bias, MAPE, worst error, share within ±10%, paired ranking and gain
error, measured-stage coverage, extrapolation and missing measurements. The 10%
band is an engineering diagnostic, not statistical confidence. Separate cohorts
and metric units; never hide failed holdouts in an overall average.

## What remains unavailable

The 50/53/54 report aggregates reproduce only limited conditional corrections;
they cannot identify independent request, cache, upload-service and compute costs.
The quality snapshot keeps historical 37/42 comparisons, conditional fits and
current synthetic replay separate. Synthetic success is not real calibration.
No event-level global fit is promoted without exact raw plans/profiles.

Hybrid relay/P2P/synchronization estimates also need matching local-step and sync
measurements; changing synchronization frequency additionally needs learning-quality
evidence. Those estimates remain scenarios, not validated throughput forecasts.

The local synthetic probe (three repeats, 200 calls) measured median cache-cycle
caller costs 4.85 µs off / 288.20 µs on, and fake HTTP 1.96 µs off / 130.05 µs on.
These include local contention and are not DigitalOcean overhead estimates. Check
the local overhead JSON and run the capture-on/off pilot externally.

## Offline integration review — 2026-10-31

After merging the upstream batched relay client, 79 focused Python tests passed
across capture, calibration, SDK assets, Python worker boundaries, relay fixtures,
simulator boundaries/cache/reference replay, adaptive claims and network placement.
The two Node test scripts for capture and relay compatibility also passed. They
use fake fetch, including batched uploads, legacy fallback, and identical fixture
bytes for batched and legacy reads. The compatibility test now covers both request
modes instead of assuming that every read uses GET.

These checks started no live services or workload training. They validate offline
contracts and integration, not real-network behavior, capture overhead on deployment
hosts, or the reported simulation campaigns (which were not rerun in this review).
