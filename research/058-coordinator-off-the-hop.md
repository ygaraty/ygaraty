# Handoffs: with the coordinator off the hop, the tiny Transformer's update on home links is 20% shorter

| | |
|---|---|
| **Date** | 2026-11-01 (L3 of `papers/uplink-bound/NEXT_2026-10-31.md`; built and run the same night) |
| **Path** | Experiment 54's tiny-Transformer recipe (the 1.06M `1m-seed2026-v2k` model, 8 NumPy workers, owned layers and 4 vocabulary slices, float32 wire, bf16 activations, pushed inputs, eager claims, 512 tiles), 12 updates per run, on the workstation (Omarchy), no cloud. Home links: the latency proxy at 100 Mb/s down, 20 Mb/s up, 50 ms round trip, one link per worker |
| **Result** | Yes. Median seconds per update (updates 2 to 10, median of 3 runs): home links 3.99 → **3.18 s (−20.2%)**, loopback 0.389 → 0.359 s (−7.6%). The repeats agree within 0.04 s (baseline) and 0.12 s (handoffs). Losses are bit-identical in all 12 runs, with 0 retries and 0 rejected. The traced chain (15 hops) is 3.91 → 3.10 s and its median hop 223 → 188 ms, against the 150–200 ms predicted. About 27% of tasks were handed off, and every hop of the traced chains was |
| **Evidence** | `runs/exp58/r{1,2,3}-{home,loop}-h{0,1}/profile.json` and `.log`; captures `runs/exp58/cap-home-h{0,1}-cap/` (local, not in git); `relay/relay_test.go: TestHandoffWaitsForTheNamedBatch`; `tests/test_handoff.py` |

## Question

Experiment 51 traced a hop on home links at about 300 ms. About half of it passes through the coordinator:
- the result post and commit: 30 ms;
- waiting to be handed the next task: 31 ms;
- the claim reply reaching the next worker: 86 ms.

L3 asked whether the next owner can start as soon as the result reaches the relay, without those steps, while the coordinator still commits every update.

**Prediction, recorded on 31 October before any code:**
- A hop falls from about 300 ms to 150–200 ms.
- The update falls from about 4.0 to about 3.0 s.

**Gate:** losses identical update by update, and at least 15% faster over three repeats.

## What was built

A **handoff**: a task handed to a worker before its inputs are committed. Off by default: the coordinator's runtime `handoff` (`YGARATY_HANDOFF=1`) and `http-worker --handoff`, or `http_profile --handoff` for both.

- **Coordinator (`coordination/handoff.py`).** A worker the scheduler has nothing ready for can be handed a task, if every uncommitted input is being computed by a worker that names its results.
  - Owned tasks go only to their owner. An unowned task is handed out at most once per reply, and a worker holds at most 4 handed tasks.
  - The reply describes each input still in flight as a recipe: the relay names of the producing attempts, any committed blocks, and the `assemble`/`transpose` steps between them. The recipe reproduces exactly the manifest that the store's metadata operations would publish.
  - The coordinator still commits every result in dependency order. A handed task's result is held until its inputs have committed. It is accepted only if the worker read the very attempts that committed, compared by block hash; otherwise the task is retried.
  - Retrying a producer also retries the consumers that were handed its result.
- **Scheduler (`runtime/scheduler.py`).** `prelease` leases a task that is in no queue. A commit never queues a task that is already leased. A retried task goes back in the queue only once its inputs exist.
- **Relay (`relay/handoff.go`).** A batched `PUT /blocks` frame may carry a `name` and the result's block list. `POST /handoff/get` waits up to 60 s until the names it asks for are published, then answers with the block lists and all their bytes in one frame. It can add committed blocks by key in the same request.
- **Worker.**
  - Each result goes up as one named `PUT /blocks`; the name is the hash of the job, the task and the attempt, which the coordinator derives too.
  - A handed task's relay read is sent as soon as the claim reply arrives.
  - Tasks that can run now run first; handed tasks follow in lease order.
  - The result post reports the block hashes the worker read.
- **`experiment chain_trace`.** It walks a captured update's critical chain through the plan, from kernel to kernel, seeing through metadata tasks. `hop_trace` (experiment 51) follows the scheduler's ready events instead, which a handed task skips. On experiment 51's capture, `chain_trace` finds 15 hops of median 252 ms over a 4.6 s chain.

## Design

- Two arms, handoffs off (h0) and on (h1), each with 3 runs on home links and 3 on loopback.
- The order alternated: h0 h1, then h1 h0, then h0 h1.
- Before each run, the driver waited for a 1-minute load under 3 and no other training process.
- Then one captured run per arm (8 updates) for the chain trace.
- Steady is the median of updates 2 to 10, chosen because both arms share the same batches:
  - update 1 is cold;
  - update 11 is a 123-token batch (about 2 s);
  - update 12 publishes the owned state (about 6.5 s).

## Results

Seconds per update, updates 2 to 10:

| Link | Arm | Run 1 | Run 2 | Run 3 | **Median** | Updates 2–12, total |
|---|---|---|---|---|---|---|
| home | h0 | 4.014 | 3.988 | 3.973 | **3.988** | 44.2–44.5 s |
| home | h1 | 3.164 | 3.182 | 3.281 | **3.182** (−20.2%) | 36.0–36.9 s |
| loopback | h0 | 0.389 | 0.402 | 0.380 | **0.389** | 4.0–4.6 s |
| loopback | h1 | 0.363 | 0.342 | 0.359 | **0.359** (−7.6%) | 3.6–4.4 s |

Other effects:
- **First update:** 5.6–5.8 s becomes 4.8–4.9 s.
- **Publishing update:** 6.7 s becomes 6.3–6.6 s.
- **Correctness:** all six home runs and all six loopback runs give the same twelve losses to the last printed digit (`profile.json`), with 0 retries and 0 rejections.
- **Handoffs:** 1,182 to 1,197 tasks per run (27% of 4,368).
- **Coordinator traffic:** it sends 10.0 MB per update instead of 20.6–21.1. Handed tasks read their inputs from the relay, not from pushed replies.

### The chain, traced

`chain_trace` on the captured runs (updates 2 to 7):

| | h0 | h1 |
|---|---|---|
| Hops on the chain | 15 | 15 |
| Chain length | 3,907 ms | 3,103 ms |
| Median hop (producer kernel end to consumer kernel start) | 223 ms | 188 ms |
| Median upload on a chain hop | 93 ms | 98 ms |
| Upload end to the consumer's kernel | 127 ms | 73 ms |
| Chain hops handed off | 0 of 90 | 90 of 90 |

- **Where the saving is:** the producer's upload is unchanged. What shrinks is the time after it, from 127 to 73 ms.
- **Why 73 ms is not lower:** the 90 handed hops split by when the consumer's relay read went out.

| When the consumer's read went out | Hops | Reply after the upload ended |
|---|---|---|
| Before the result landed | 48 | 22 ms (the relay forwarding the bytes as they arrive) |
| After the result landed (handed the task late) | 42 | 119 ms |

## What it means

- **The coordinator-mediated part of a hop can be removed, and the update keeps its exact numbers.** The −20% passes the 15% gate, the repeats are 5 to 20 times tighter than the gap, and the losses are bit-identical.
- **The prediction held for the update, not fully for the hop.**
  - The update came in at 3.18 s against the predicted 3.0 s.
  - The hop came in at 188 ms (median, kernel to kernel) against 150–200 ms predicted, measured from a 223 ms baseline on the same footing (experiment 51's 300 ms counted differently).
  - Nearly half of the handoffs are handed out too late for the read to be waiting when the result lands. Handing tasks out earlier is the remaining lever on this hop; one quick try failed (Limits).
- **What is left of a hop is the upload, about 95 ms: a round trip plus the bytes over the 20 Mb/s uplink.** This matches experiment 51 and the paper's thesis: once the coordinator is off the path, the per-contributor uplink is what sets the hop.
- **The coordinator still decides what counts.**
  - It places the work.
  - It commits every update in order.
  - It checks each handed task against the attempts that committed.
  - It samples results for validation as before (handed owned tasks are not sampled, as owned tasks never are).

## Limits

- **Emulation, not real links.** One machine, the tiny model, the proxy's links without jitter or loss. The 30.3M model has about 25 times the bytes per hop, so the share saved will be smaller there; DigitalOcean (R1's pool) would measure it.
- **One capture per arm for the trace.** Capture adds overhead, but both arms carry it.
- **Fewer cache hits.** Handed tasks rarely hit the worker's tile cache (132 hits against 7,500–12,400) and read those inputs from the relay. The relay's per-worker download bytes were not recorded.
- **Tried and reverted: waking held-open claims after every lease,** so consumers would be handed tasks earlier. It gave 3.13 s in one run, inside the h1 spread. The late handoffs may instead come from workers busy computing or waiting on an earlier handoff; not yet measured.
- **Prototype scope:**
  - The worker blocks on a handed task's input; ready tasks run first, but tasks claimed later wait behind it.
  - A consumer handed a lost attempt's result waits up to 20 s, or until the producer's retry retries it too.
  - Only the Python worker and the relay path exist; there is no browser SDK support.

## Reproduce

```bash
cd relay && go test ./... && go build -o ../relay-server ./cmd/relay && cd ..
python -m unittest tests.test_handoff
./relay-server -addr 127.0.0.1:PORT -token TOKEN -max-mib 4096 &        # a fresh relay per run
ygaraty experiment http_profile --recipe transformer --learning-rate 0.0005 --workers-kind python --tiles 512 --max-tasks 16 --repeats 1 \
  --relay-url http://127.0.0.1:PORT --relay-token TOKEN --version 1m-seed2026-v2k --embedding 128 --hidden 384 --layers 2 --heads 4 --band 16 \
  --max-task-mib 512 --workers 8 --float32-wire --activation-wire bf16 --owned-layers --owned-vocab 4 --pushed-inputs --eager-claims \
  --length-window 16 --batch-tokens 3072 --dataset-url sqlite:///$PWD/runs/datasets/opus100-en-pt/opus100-en-pt.sqlite \
  --store-url postgresql:///ygaraty_tiny --steps 12 --rtt-ms 50 --down-mbps 100 --up-mbps 20 --handoff --output runs/exp58/NAME
ygaraty experiment chain_trace runs/exp58/cap-home-h0-cap runs/exp58/cap-home-h1-cap   # captures: YGARATY_CAPTURE_DIR=... and --steps 8
```

Drop `--handoff` for the baseline and the three shaping flags for loopback. The local drivers (`runs/exp58/drive.sh`, `campaign.sh`, `precise.py`) rotate the arms and read the steady window from `profile.json`. The campaign took about 8 minutes.
