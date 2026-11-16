# Does the batched relay scale over 1, 2 and 4 physical WebGPU workers and group sizes?

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open |
| **ID** | DO-2, stage 3 (number added on finish) |
| **Category** | browser |
| **Where** | remote |
| **Date** | Added 2026-10-30 |
| **Path** | Batched relay against existing relay and direct transport on 1, 2 and 4 hardware-WebGPU workers, plain and shaped links, single tiles, 5×5 and 11×11 groups |

## Question

[Experiment 50](050-browser-workers-on-the-relay.md) removed more than 99% of coordinator traffic and fixed browser cache misses, but the relay was 18% slower on emulated home links. The next experiment should test request batching, not repeat byte savings alone.

Stages 1 and 2 ran locally on 2026-10-30 and passed the gate; See [experiment 56](056-batched-relay-calls-on-home-links.md).
- **Stage 1:** 2,048 rows, 2×2 groups, float32, 2 tabs. All four arms gave identical losses with 0 rejections. Relay requests fell from 300 preflights, 111 reads and 208 writes to 5 preflights, 45 batched reads and 144 batched writes.
- **Stage 2:** emulated home links, 4 tabs on one GPU, 3 balanced repetitions. Batched relay 105.0 s per update against 157.5 for the existing relay and 133.3 direct, with bit-identical losses.
- **Not yet run:** stage 3 and any external browsers.

## Gate

Require matched numerical results under the established precision tolerance, zero unexplained rejections, no memory-budget violations and all intended workers participating. Compare end-to-end time, not merely coordinator bytes. Batched relay should beat the existing relay beyond repeat variation and avoid a material regression versus direct transport at the target scale. If it does not, record the crossover rather than declare success.

Report per-worker cache hits/misses, repeated input fetches, requests/preflights, bytes, compute/idle/transfer time, coordinator CPU/RSS and complete timing distributions. Factorization learning checks must separate data loss from regularization; decreasing total loss alone previously gave a misleading result.

## Protocol

- Add and verify batched block uploads with bounded memory, per-block integrity, correct retries and idempotency. Record requests/preflights and actual wire bytes.
- Preserve pinned input keys and the validated cache policy. Treat cache size as part of each worker's total memory budget.
- Prepare browser-accessible TLS, CORS and appropriately scoped contributor credentials before external browser use.
- Confirm the external runner exposes group size, precision and the chosen data source. CPU grouped-factorization and browser relay runs are distinct execution paths.

1. Small correctness case: direct frames, existing relay and batched relay, identical inputs/precision. Compare outputs and loss trajectories; inspect rejection/retry reasons.
2. On fixed hardware and links, compare the three transport arms with 5×5 groups and float32 wire, at least three balanced repetitions. Separate cold-cache startup from subsequent updates.
3. Only after the request-overhead gate passes, compare 1, 2 and 4 physical workers on plain and shaped links, with single tiles, 5×5 and admissible 11×11 groups. Use a bounded size ladder before 100,000 proteins.

DigitalOcean CPU Droplets can host the coordinator/relay and a CPU-worker campaign. Browser GPU scaling requires actual hardware-WebGPU workers on authorized GPU hosts; multiple tabs on one GPU are not independent GPU machines. Keep CPU and browser results in separate tables and do not substitute software rendering.

## Log

- 2026-10-30: Added. Stages 1 and 2 run and written up as [experiment 56](056-batched-relay-calls-on-home-links.md).
- 2026-11-01: Moved to its own file; stage 3 open.
