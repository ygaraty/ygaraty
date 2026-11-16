# Batched relay calls: browser workers on the relay go from 18% slower to 21% faster than direct on home links

| | |
|---|---|
| **Date** | 2026-10-30 (DO-2, local part; built and run the same day) |
| **Path** | Experiment 50's home-link run (the sharded factorization of experiment 48: 100,000 proteins, 4 WebGPU tabs, 5 x 5 tile groups, float32 wire, one emulated 100/20 Mb/s, 50 ms link per tab), with the browser SDK's relay calls batched, uploads overlapped with compute and 4 tasks per claim. Workstation (Omarchy), no cloud |
| **Result** | Yes. Median seconds per update (updates 2 to 6, median of 3 repetitions): relay 157.5 → **105.0** (−33%), direct 133.3 → 122.0 (−8.5%). The relay is now 14% faster than direct instead of 18% slower. Relay requests per run 23,700 → 4,030 (CORS preflights 11,750 → 8). Losses bit-identical across all 12 runs, 0 rejected, 0 retries. The classic arms reproduce experiment 50 (133.5 and 157.5 s) |
| **Evidence** | `runs/experiments/do2/home/{relay,direct}-{new,classic}-rep{1,2,3}/report.json`, per-run load logs `*.load` beside them (local, not in git); `relay/relay_test.go: TestBatchedPutAndPostGet, TestBatchedPutStoresNothingWhenOneBlockIsBad, TestCORSAllowsBatchedMethods`; `tests/test_browser_relay.py: test_batched_and_pipelined_relay_matches_and_needs_fewer_requests` |

## Question

Experiment 50 found that the relay takes the coordinator out of the byte path but costs a tab 18% on home links: about 880 extra round trips per tab per update, each with a CORS preflight, with the same bytes on the link. It named the fix, not built: a batched upload and uploads overlapped with compute. With those built, plus several tasks per claim, does the relay stop costing time on home links, and what do the same SDK changes do without the relay?

## What was built

- **Relay (`relay/relay.go`).**
  - `PUT /blocks`: one frame carries many blocks under the header `{"blocks":[{"key","size","check"}]}`. Every block is verified before any is stored, so a bad block stores nothing. A frame is capped at `MaxBatchBytes`, 256 MiB by default.
  - `POST /blocks/get`: reads with the keys in a JSON body.
  - Both URLs are constant, so a browser preflights each once per tab and caches it. A query-string `GET` gets a fresh preflight per key list.
  - `/status` counts requests by kind.
- **SDK (`webapp/static/sdk.mjs`).**
  - `RelayClient` batches reads and writes. If the relay lacks the batched routes, it falls back to the old calls once.
  - `BrowserWorker` has two new options, both on by default:
    - `pipeline`: result uploads run as promises while the next task computes.
    - `relayBatch`: batched reads and writes.
  - The result post carries `{claim: true, max_tasks}`, so one round trip returns the result and claims the next tasks (`Job.result_and_claim`).
  - `contributor.mjs` accepts the hash options `tasks=N`, `relaybatch=0` and `pipeline=0`.
- **Driver (`factorization_browser_training.py`).**
  - `--tasks-per-claim N`.
  - `--sdk-classic`, which turns batching and pipelining off and reproduces experiment 50's SDK.

## Design

Four arms, 6 updates each, update 1 (cold) reported apart:

| Arm | Relay | SDK |
|---|---|---|
| direct-classic | no | `--sdk-classic` (experiment 50) |
| direct-new | no | `--tasks-per-claim 4` |
| relay-classic | yes | `--sdk-classic` (experiment 50) |
| relay-new | yes | `--tasks-per-claim 4` |

- Three repetitions, with the arm order rotated each repetition.
- Before each run, the driver waited for no DiLoCo process and a 1-minute load under 3 for 60 s, then sampled the load every 30 s during the run.
- The highest 1-minute load seen in any run was 2.2. No other experiment ran during the campaign.

A first attempt at relay-new was discarded. It ran alongside another session's DiLoCo run, with a load near 25, and is kept in `runs/experiments/do2/contaminated/`.

## Results

Seconds per update. "Steady" is the median of updates 2 to 6.

| Arm | Rep 1 steady | Rep 2 steady | Rep 3 steady | **Median** | First update | Total, 6 updates |
|---|---|---|---|---|---|---|
| direct-classic | 133.3 | 135.1 | 133.0 | **133.3** | 137–139 | 802–809 s |
| direct-new | 122.9 | 121.7 | 122.0 | **122.0** | 124–127 | 734–738 s |
| relay-classic | 157.5 | 157.5 | 158.5 | **157.5** | 161–164 | 948–955 s |
| relay-new | 105.0 | 104.5 | 105.2 | **105.0** | 108–111 | 631–640 s |

The repetitions agree within 2 s, so the arms' differences (11 to 52 s) are far outside the noise.

Relay requests per run (all 4 tabs, 6 updates):

| Kind | relay-classic | relay-new |
|---|---|---|
| CORS preflights | 11,750 | 8 |
| Single reads | 2,830 | 6 |
| Batched reads | 0 | 776 |
| Single writes | 9,124 | 196 |
| Batched writes | 0 | 3,048 |
| **Total** | **23,700** | **4,034** |

That is about 990 relay requests per tab per update before the change and about 170 after.

What stayed the same:
- Every run's six losses are bit-identical: final loss 0.507600904, within 3.3e-8 of the float64 reference.
- No run had a rejected result or a retry.
- Every tile ran on the GPU.

## What it means

- **Round trips, not bytes, were the relay's cost on home links,** as experiment 50 said. Removing about 83% of the relay requests, almost all of the preflights, turned an 18% loss into a 21% gain over the direct path.
- **Why the relay now beats direct is an inference, not measured.** Overlapped uploads only help on the relay. There, a result's blocks go up while the next task computes, and the coordinator post carries only hashes. On the direct path, the result bytes travel inside the result post, so the 20 Mb/s uplink time stays on the critical path. A run with pipelining on and the other options off would confirm it.
- **The SDK changes help without the relay too, but less.** Direct-new is 8.5% faster than direct-classic, from multi-task claims and the result post that also claims the next tasks.
- **The relay is now the faster path, besides taking the coordinator out of the byte path** (experiment 50: 99.8% less coordinator traffic). Experiment 50's 18% cost is gone.

## Limits

- **Emulated links on one machine:** 4 tabs, the same GPU, and the latency proxy's 100/20 Mb/s and 50 ms. Real browsers on real home links (DO-2's cloud part, or experiment 48's stage E) were not run.
- **No ablation:** batched calls, pipelining and 4 tasks per claim change together in the new arms. Only their sum is measured, plus the direct path's share.
- **One workload:** the factorization's tile tasks, about 100 per update, with large inputs and small results. Workloads with more, smaller tasks would gain more from the batching; this was not measured.

## Reproduce

```bash
cd relay && go test ./... && go build -o relay-server ./cmd/relay-server && cd ..
python -m unittest tests.test_browser_relay
BASE="--embeddings runs/datasets/protein-sprot/sample-100000.npz --group 5 --wire float32 --updates 6 --tabs 4 --links home --timeout 3600"
python -m ygaraty.experiments.factorization_browser_training $BASE --relay --tasks-per-claim 4 --output runs/experiments/do2/home/relay-new-rep1
python -m ygaraty.experiments.factorization_browser_training $BASE --relay --sdk-classic     --output runs/experiments/do2/home/relay-classic-rep1
python -m ygaraty.experiments.factorization_browser_training $BASE --tasks-per-claim 4      --output runs/experiments/do2/home/direct-new-rep1
python -m ygaraty.experiments.factorization_browser_training $BASE --sdk-classic            --output runs/experiments/do2/home/direct-classic-rep1
```

Each run takes 10 to 16 minutes. The campaign rotated the order over three repetitions and waited for an idle machine before each run.
