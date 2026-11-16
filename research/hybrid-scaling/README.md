# Offline hybrid-island capacity comparison

This study treats each learner as a 16-worker sharded island and synchronizes model-state changes between islands. Its local step input is experiment 37 B's rounded **10.1 seconds**, already including intra-island communication. It is a synchronization capacity model, not task-level replay, a measured hybrid deployment, or a convergence simulation.

Run without a dataset, workers, services, downloads or real numerical kernels:

```bash
ygaraty simulate-hybrid --output /tmp/hybrid-comparison.json --report /tmp/hybrid-comparison.md
python -m unittest discover -s tests -p test_hybrid_simulation.py -v
```

Outputs must not already exist. The JSON contains every configuration, assumptions, evidence/code checksums, timing, traffic, synchronization count, aggregate local-update throughput and scaling efficiency.

## Controlled comparison

All scenarios perform **1,024 island-local updates**. A run with N islands performs 1,024/N updates per island. This fixes aggregate local work, not optimization trajectory or global effective batch. The sweep covers 1–64 islands (16–1,024 workers), H=1/10/100, one/four relays or a direct ring, and 1/10 Gb/s per relay: 252 scenarios.

Model-state payload is 30.3M parameters × 4 bytes = 121.2 MB per island per upload/download. Links have 50 ms RTT. Two access assumptions bracket different implementations:

- **Single gateway:** all island synchronization traverses one 20 Mb/s uplink and 100 Mb/s downlink.
- **Ideal 16-way striping:** state is perfectly balanced over sixteen independent links, providing an aggregate 320/1,600 Mb/s. This is an optimistic design assumption, not measured island capacity.

Every local block of up to H steps is followed by synchronization, including the final shorter block. At 64 islands there are only 16 local steps each: H=100 therefore means one final synchronization, not an observed 100-step training round. One island has no inter-island exchange.

Relays partition tensor coordinates evenly; every island sends a shard to every relay. Each relay has one shared ingress/egress bandwidth resource. Uploads cross the island access link before entering that resource; broadcast waits for all uploads. This store-and-forward model excludes averaging compute. No inter-relay exchange is needed under the perfect tensor partition assumption. Four relays multiply provisioned aggregate capacity by four, so this is not an equal-cost comparison.

Direct P2P uses a synchronous full-duplex ring reduce-scatter/all-gather: 2(N−1) hops, each carrying P/N bytes at the slower access rate plus one-way latency. It removes relay concentration but sends approximately twice the per-island outbound bytes of centralized aggregation at large N. P2P is not “free networking.” The ring is one collective design; trees, pipelined relays and other collectives may rank differently.

## Representative results

Speedups below are against **one relay, H=1, same number of islands and same local work**. They are not speedups relative to experiment 37's original single island.

| Islands / workers | Access | Relay capacity each | Four relays | P2P ring | One relay, H=10 |
|---|---|---:|---:|---:|---:|
| 16 / 256 | Single gateway | 10 Gb/s | 1.03× | 0.70× | **4.25×** |
| 16 / 256 | Ideal striping | 10 Gb/s | 1.16× | 1.02× | **1.56×** |
| 64 / 1,024 | Ideal striping | 1 Gb/s | 3.08× | **7.18×** | 5.29× |
| 64 / 1,024 | Ideal striping | 10 Gb/s | 1.55× | 1.36× | **2.16×** |

For the 16-island single-gateway / 10 Gb/s scenario, H=1 takes 4,571 seconds for the fixed workload and H=10 about 1,076 seconds. A one-island execution takes 10,342 seconds under the fixed local-step input. This demonstrates how access-link synchronization can consume most of the theoretical parallel gain.

**Interpretation:** reducing synchronization frequency is the strongest modeled lever in the common access-limited cases. Extra relays help when shared relay bandwidth is saturated. P2P becomes attractive when distributed access capacity is high relative to centralized relay capacity, but can lose when access links dominate. Combining interventions is included in the JSON; throughput gains do not establish a preferred learning algorithm.

## What this cannot establish

- H changes the optimization algorithm. There is no simulated validation loss, calibration, convergence or time-to-quality; equal local work is not equal learned quality.
- The 10.1 s island input is measured on one historical recipe, not calibrated for replication or shared infrastructure. No fitted correction uses experiment 37; its task-level validation role is unchanged.
- More relays/P2P modify only **inter-island synchronization** here. This does not predict their effect on experiment 42's intra-island activation uploads, or repair the task simulator's upload-stage calibration gap.
- Compute/communication overlap, stragglers, churn, packet loss, NAT traversal, encryption, relay CPU, aggregation, verification, startup, checkpointing and final owner publication are omitted. RTT and throughput are identical across peers.
- Full-duplex ring and store-and-forward relay assumptions are different execution strategies. A streaming relay or tree reduction can change the comparison; relay count alone is not an architecture-independent speedup.
- Laya-sized workloads require new local-step and memory evidence. Multiplying parameters without replacing the 10.1 s input would not produce a credible Laya training estimate.

The next useful evidence is per-island state ownership and effective synchronization bandwidth, plus matched quality measurements across H. Until then, use this comparison to select candidate experiments, not as a training-time promise.
