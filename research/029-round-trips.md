# Round trips on the critical path: network latency, and three changes that remove them

| | |
|---|---|
| **Date** | 2026-10-26 |
| **Path** | HTTP path in production mode (experiment 24's job), PostgreSQL store, 4 NumPy `http-worker` processes claiming 8 tasks each, tile 512; the Go relay of experiment 28 where noted; a latency proxy (`relay/cmd/latency-proxy`) between the workers and the coordinator and relay |
| **Result** | With network latency an update is dominated by round trips along the plan's dependency chain: 25, 42 and 71 s per update at 20, 50 and 100 ms (12.8 s on loopback). Three changes remove round trips or the bytes that queue behind them: the next task in the result reply, a whole LSTM layer per task (through the relay), and a fused output layer. On loopback the best setups now take 10.5–10.8 s per update (12.8 s before). Losses equal to every printed digit; parity against PyTorch passes. The network gain of the last three changes is not measured yet. |
| **Evidence** | `runs/experiments/029/omarchy-20261026-*/`, not synced |

## Question

Experiments 27 and 28 measured one machine, where a round trip costs milliseconds. Volunteers are on the internet. What does latency do to an update, and which changes take round trips off the critical path?

## Latency

`latency-proxy` delays every byte by half a round trip in each direction, preserving order (no bandwidth limit). Workers reach the coordinator and the relay through it; the coordinator, the relay and PostgreSQL stay together, as on one server. Seconds per update, one run each, before the changes below:

| Round trip | No relay, 1 step per task | No relay, 4 steps | Relay, 1 step | Relay, 4 steps |
|---:|---:|---:|---:|---:|
| loopback | 12.8 | 12.5 | 14.9 | 13.5 |
| 20 ms | 25.3 | 20.1 | 67.1* | 49.7* |
| 50 ms | 42.1 | 30.3 | 89.3 | 53.5 |
| 100 ms | 70.8 | 47.7 | 164.4 | 94.5 |

\* before result uploads were batched per claim, which the 50 and 100 ms relay runs had.

- The added time tracks hops times round trips: about 220 hops on the critical path at 2 round trips each is about 22 s at 50 ms, most of the 29 s added. Four steps per task (experiment 27's change, worth 2.5% on loopback) saves 20–33% here.
- The relay as built makes it worse: a hop through it is a claim, a fetch, an upload and a result, four round trips instead of two.

## Three changes

- **The next task in the result reply.** A worker posts its results with `claim: true`; the coordinator records them, gives that worker the first chance at what they made ready (the next step of its chain, whose state and weights it holds), then wakes the other waiting workers (`Job.result_and_claim`). One round trip per hop instead of two. Default in `ygaraty http-worker`; `--separate-claims` restores the old protocol. The browser SDK still uses two requests.
- **A whole LSTM layer per task.** `YGARATY_LSTM_CHAIN_TASK_MIB` gives chain tasks their own admission budget instead of the 8 MiB result cap, so `YGARATY_LSTM_CHAIN_STEPS=64` runs each layer direction as one forward and one backward task (624 chain tasks per update become 18). A layer's result is about 80 MB, so this needs the relay; the setting is refused without one.
- **A fused output layer.** `YGARATY_LSTM_OUTPUT_TASK_MIB` replaces the output projection, the bias, the cross-entropy and its gradient with one `ce_fused` task per band of 65 rows, returning the logit gradient and a column of row losses; the 460 MB of logits never leave the worker. Production mode only (verification checks every logit). The loss is summed per band rather than per 16-row chunk, so it agrees with the unfused plan to rounding, not bit for bit.

The profiler now prints the plan's critical path: worker hops on the longest dependency path, times the round trips a hop costs in that protocol.

## Result on loopback

Seconds per update, 4 NumPy workers, one run each:

| Setup | Critical path | Seconds per update | Coordinator sent per update |
|---|---|---:|---:|
| No relay, 1 step per task, separate claims (before) | about 220 hops × 2 | 12.8 | 12.8 GB |
| No relay, 4 steps, result claims | 122 hops × 1 | 12.5 | 12.2 GB |
| No relay, 4 steps, result claims, fused output | 120 hops × 1 | **10.8** | 10.6 GB |
| Relay, whole layers, result claims | 54 hops × 3 | 12.7 | 14 MB |
| Relay, whole layers, result claims, fused output | 52 hops × 3 | **10.5** | 9 MB |

- Result claims cut each worker's result-request time from 2.1 to 0.6 s per update but barely moved wall time here: a loopback round trip costs milliseconds.
- Whole layers shortened the chains (backward chains 0.4–0.7 s each instead of about 1.4 s) and made the relay path as fast as the direct one.
- The fused output layer removed 650 tasks, 1.6 GB sent and 1 GB received per update, and shrank the gap between the forward and the backward chains from 5.0 to 3.3 s. It took off only 2 hops: its gain is bytes and queueing, not latency.

## Reading

- **Over a network, count round trips.** By that count the best setup is not the relay one: 120 round trips per update for the direct path with 4 steps and result claims, 156 for the relay with whole layers, because each relay hop adds a fetch and an upload. The relay pays where the coordinator's bandwidth is the limit (a larger pool, experiment 28), and its hop cost is the next thing to cut there.
- What remains on loopback is the work between the chains: the projections and weight-gradient matmuls (3 s of the update, tasks waiting 350 ms on average in the queue).

## Limits

One run per setting, one batch, one machine. The network effect of the three changes is projected from the hop counts, not measured. The proxy adds latency only; at 12 GB per update, a 1 Gb/s server link would itself take about 100 s per update without the relay.

## Running it

```bash
cd relay && go build -o ../relay-server ./cmd/relay && go build -o ../latency-proxy ./cmd/latency-proxy && cd ..
ygaraty experiment http_profile --recipe lstm --workers-kind python --tiles 512 --max-tasks 8 --steps 3 --repeats 1 \
  --learning-rate 0.001 --chain-steps 4 --output-task-mib 256 [--rtt-ms 50] [--separate-claims]
# through the relay, whole layers:  add --relay-url URL --relay-token T --chain-steps 64 --chain-task-mib 1024
```
