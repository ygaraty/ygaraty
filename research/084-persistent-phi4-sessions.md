# Persistent sessions halve Phi-4's time between tokens on emulated links, with the same tokens

| | |
|---|---|
| **Conclusion** | **Works:** persistent sessions cut p95 inter-token latency 51% at 25 and 50 ms, same tokens (emulated links). |
| **Status** | executed |
| **ID** | DO-17 |
| **Date** | 2026-11-05 |
| **Path** | Two-owner Phi-4 pipeline (a tiny random model with Phi-4's architecture, float64 owners) on this workstation, each worker behind `latency-proxy` at an emulated round trip of 1, 10, 25 or 50 ms. Arm A: the step-by-step path (`phi4_run --handoff`, relay, float32 wire), a coordinator claim cycle per stage of every token. Arm B: persistent sessions (`--persistent`): one `phi4_loop` task per owner per session, activations and tokens handed owner to owner through the relay's named handoffs, the coordinator only watching tokens |
| **Result** | Passes the frozen gate. p95 inter-token latency fell 51.4% at 25 ms (117.0 → 56.9 ms) and 51.3% at 50 ms (214.5 → 104.5 ms), against a 30% bar; 61% at 1 ms and 52% at 10 ms. Tokens were identical in every session of all 16 jobs, with 0 retries and 0 rejected. The duplicate-delivery and stalled-owner tests pass. One step-by-step job hung in the first start and did not hang again on its rerun. Free (workstation) |
| **Evidence** | Measured, emulated links: [evidence/transport/do17-persistent-sessions-2026-11-05.json](evidence/transport/do17-persistent-sessions-2026-11-05.json); raw runs in `runs/experiments/DO-17/omarchy-20261105-132105/`. Code d314641; tests `tests/test_phi4_persistent.py` |

## Question

Experiment 83 showed the network, not compute, setting a Phi-4-style pipeline's token time: each stage of each token goes through a coordinator claim, a relay upload and a result post. **Does handing activations owner to owner, with the coordinator off every hop, remove those round trips?**

## Prediction

- Each persistent hop is one round trip: half for the upload to reach the relay, half for the relay's reply to the waiting owner. A token around 2 owners then costs 2 round trips plus compute.
- The step-by-step path costs about 4.5 round trips per token at 2 owners (the pilot), so the gain should be about 50% wherever the round trip dominates, and less at 1 ms, where compute and Python overhead count.

## Gate (frozen 2026-11-05, before the deciding run; the user's yes)

- **Arms:** A, the step-by-step path (`phi4_run --handoff`, a coordinator claim cycle per stage); B, persistent sessions (`--persistent`, `phi4_loop` owner to owner through the relay). Both use the relay, float32 wire and the same kernels.
- **Setup:** a tiny random model with Phi-4's architecture (`tests/test_phi4_kernels.reference_model`, float64 owners), 2 owners, each worker behind `latency-proxy` at an emulated round trip of 1, 10, 25 and 50 ms (no bandwidth limit), on this workstation. Real Phi-4 does not fit this workstation's CPU; the tiny model keeps compute at a few ms, so the arms differ in coordination only.
- **Sessions:** a 128-token prompt (seed 2026), 128 new tokens; per round trip, jobs in the order A B B A, each 1 warmup plus 5 measured sessions.
- **Pass:**
  - p95 inter-token latency of B at least 30% lower than A's, at both 25 and 50 ms;
  - every session's tokens identical across arms and jobs;
  - in the duplicate-delivery and stalled-owner tests (`tests/test_phi4_persistent.py`), no token is committed twice or out of order, and a stalled owner fails its task with an explicit error.
- **Reported regardless:** the 1 and 10 ms results, medians and p95s of both arms, and retries and rejections.

## Protocol

- **Code (d314641):**
  - `phi4_owner.Phi4Owner.loop` (op `phi4_loop`): an owner runs every prefill chunk and decode step of a session. It waits for its input under a relay handoff name (`loop_names`: SHA-256 of `job/session/a/stage/position`, or `.../t/position` for tokens) and publishes its output under the next owner's name.
  - Uploads and deletes go out in order on a sending thread, so an owner waits for its next input while its output is on the wire.
  - The first owner holds the prompt; the last applies the stopping rule (eos, max_new) and publishes each token. A stop sentinel travels the ring.
  - `Phi4Job(persistent=True)`: a load step, then one step per session of `compile_phi4_loop` tasks with one attempt each and a 7-day lease. A thread commits each token from its relay name, once per position.
  - Owners whose input does not come within `stall_seconds` (300 by default) fail their task, which fails the job.
- **Driver:** `runs/experiments/DO-17/omarchy-20261105-132105/drive.py` under the experiments skill's `watch.py`; `analyze.py` computes the gate into `gate.json`.

## Log

- 2026-11-04: Added at the user's request from the inference strategy discussion; renumbered from local DO-15 to DO-17.
- 2026-11-05: Implemented persistent sessions (d314641).
  - Pilot (not the deciding run): median inter-token 115 → 83 ms at 25 ms at first.
  - A synchronous upload and a delete before the compute each cost a round trip. With both on a sending thread, the delete after the send: 115 → 56 ms at 25 ms, 214 → 105 ms at 50 ms.
  - The user froze the gate and chose a local run.
- 13:21: Deciding run at d314641. The first start stopped before any job: TINY's 128-token context cannot hold 128 + 128 tokens. The driver now builds the same tiny model with a 512-token context; restarted.
- 13:22–13:52: 11 of 16 jobs completed, each with 0 retries and 0 rejected. The 12th, rtt 25's last A job (step-by-step), hung: a worker's `communicate()` reached the 1,800 s limit, and its stderr was lost.
  - `phi4_run` now reports a timed-out job's step, leases and workers' stderr.
  - Resumed (`JOBS=resume`): rtt 25's A, then rtt 50's A B B A, 600 s per job.
- 13:55–14:10: The 5 resumed jobs completed, 0 retries and 0 rejected; the hang did not recur.

## Results

| Round trip | A median | A p95 | B median | B p95 | p95 reduction | Median reduction |
|---|---|---|---|---|---|---|
| 1 ms | 14.5 ms | 15.9 ms | 6.0 ms | 6.1 ms | 61.3% | 58.9% |
| 10 ms | 49.7 ms | 51.2 ms | 23.6 ms | 24.6 ms | 52.0% | 52.6% |
| **25 ms** | 114.8 ms | 117.0 ms | 56.1 ms | 56.9 ms | **51.4%** | 51.1% |
| **50 ms** | 211.2 ms | 214.5 ms | 103.8 ms | 104.5 ms | **51.3%** | 50.9% |

Each row pools 2 jobs per arm, 10 measured sessions of 127 inter-token gaps each. Gate:

| Item | Outcome |
|---|---|
| p95 at least 30% lower at 25 and 50 ms | **Pass:** 51.4% and 51.3% |
| Tokens identical across arms and jobs | **Pass:** one token sequence in all 96 sessions |
| Duplicates and stalls | **Pass:** `tests/test_phi4_persistent.py`, 4 tests OK. A doubled publish of every name gives the reference tokens once each; a stalled owner raises "… was not published within 1 s" and drops its caches; loop tasks have one attempt |

The hung job (failed run, kept for the record): the first start's 12th job, arm A at 25 ms, never completed. Its rerun took 89 s like the other A jobs at 25 ms.

## What it taught

- **Persistent sessions cost 2 round trips per token at 2 owners, the floor for a ring through one relay.** At 50 ms, 103.8 ms is 2 × 50 plus about 4 ms of compute and Python. The step-by-step path costs about 4.2 round trips.
- **Two details decided whether the gain appeared at all.** With uploads waiting for the relay's acknowledgement, the gain was 28%. Then a delete of the consumed input, queued before the compute, delayed the next upload by a full round trip. Ordering the sends fixed both.
- **The coordinator now only watches.** It commits each token as the last owner publishes it. A dead or stalled owner fails the job within `stall_seconds`, rather than leaving it waiting for its timeout (experiment 83 lost 50 min that way).
- **The step-by-step path can hang.** It hung once in 6 runs at 25 ms on the same settings, and its cause was not captured. `phi4_run` now reports where a timed-out job stopped.

## Limits

- **Emulated links on one machine:** latency only, no bandwidth limit, loss or jitter, and every process shares one CPU.
- **A tiny model:** compute is a few ms. With real Phi-4 or TwIL per-owner compute (24–64 ms in experiment 83), the absolute saving stays near 2 round trips per owner hop, but the percentage falls.
- **2 owners only.** With N owners, the step-by-step path pays its claim cycle at every hop, so the saving should grow with N; this was not measured.
- **Coverage:**
  - No GPU, and the timing window is a single session at a time.
  - Persistent sessions have no retry: an owner failure aborts the session.
  - Prefill runs through the same ring, chunk by chunk, and was not timed separately.
- **The hang** in the step-by-step arm was not reproduced or explained.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_phi4_persistent
python3 <experiments skill>/scripts/watch.py run --dir runs/experiments/DO-17/<new> --stall 1800 --progress 'update rtt' \
  -- .venv/bin/python -u runs/experiments/DO-17/<new>/drive.py          # needs ./relay-server and ./latency-proxy
.venv/bin/python runs/experiments/DO-17/<new>/analyze.py
```
