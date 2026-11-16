# What the next changes could buy: placement on mixed pools, wider models, 8-bit activations (simulated)

| | |
|---|---|
| **Date** | 2026-10-28 |
| **Path** | The update simulator (experiment 33) on experiment 37's best setup: the 30.3M Transformer, 16 workers on home links (100 Mb/s down, 20 up, 50 ms), owned layers and 8 vocabulary slices, bf16 activations, pushed inputs, eager claims, sorted 3,072-token batches; and the 1.06M tiny model, with a cost model refit on its own runs |
| **Result** | Priced in seconds, before any machines: capacity placement makes a pool with a few fast uplinks pay. Two fast workers among 16 take the update from 9.3 s to 7.7 s (−17%), six from 9.3 to 5.8 s (−38%); first-come placement wastes them (9.3 s). A wider model is the largest lever on the nanochat KPI: width 768 (62M parameters) 49 → 33 years, width 1,024 (105M) 49 → 25 years, though each update is slower (12.7 and 16.3 s). Every result at 1 byte, an upper bound for 8-bit activations, gives 7.2 s (−23%). Recovery from a lost owner changes no speed but its cost: checkpoints every ~40 updates would cost about 12% if an owner failed every hour. The simulator's cost model is size-specific: fitted on the 30.3M kernels, it overcharges the tiny model 2.4 to 5 times on loopback. |
| **Evidence** | `ygaraty experiment price_changes placement|width|bytes|tiny` (output below) |

## Question

Experiment 37 left the 30.3M model at 10.1 s per update over DigitalOcean with 3,072-token batches (54 years by the nanochat KPI, about 108 adjusted for learning). Several changes are built but untested at that scale (capacity placement, recovery from a lost owner) or not built (8-bit activations, a wider model). Which would show a better way of running, and by how much?

## Setup

`ygaraty experiment price_changes` (new) runs the simulator over 8 consecutive batches per line and reports the mean update, real tokens per second and the KPI (`experiments/kpi.py`). Mixed pools use the simulator's worker profiles and the job's own capacity placement (commit 1027ff2): a strong worker reports 4 cores, 4 times the reference GFLOP/s and 500/100 Mb/s; a weak one 1 core, the reference GFLOP/s and 100/20 Mb/s. Identical workers take the run's links. The simulator has run 7 to 10% fast against DigitalOcean (experiment 37).

## Results

**Placement** (16 workers unless stated):

| Pool | Placement | s/update | Real tokens/s | nanochat |
|---|---|---:|---:|---:|
| 16 identical, 20 Mb/s up (today) | first-come | 9.3 | 286 | 49 years |
| 2 strong + 14 weak | first-come | 9.3 | 287 | 49 years |
| 2 strong + 14 weak | **capacity** | **7.7** | 344 | **41 years** |
| 6 strong + 10 weak | capacity | 5.8 | 463 | 30 years |
| 16 strong (every link 100 Mb/s up) | first-come | 4.5 | 596 | 23 years |
| 8 identical | first-come | 10.1 | 262 | 53 years |
| 2 strong + 6 weak | first-come | 9.1 | 292 | 48 years |
| 2 strong + 6 weak | **capacity** | **7.2** | 370 | **38 years** |

**Model width** (3 + 3 layers, 16 identical workers):

| Width · feed-forward · heads | Parameters | s/update | Real tokens/s | Useful GFLOP/s | nanochat |
|---|---:|---:|---:|---:|---:|
| 512 · 2,048 · 8 (today) | 30.3M | 9.3 | 287 | 26.0 | 49 years |
| 768 · 3,072 · 12 | 62.0M | 12.7 | 210 | 39.0 | 33 years |
| 1,024 · 4,096 · 16 | 104.6M | 16.3 | 164 | 51.3 | 25 years |

**Bytes per activation**: bf16 (today) 9.3 s, 49 years; every result at 1 byte per element 7.2 s, 368 real tokens/s, 38 years. That makes gradients, statistics and partials 1 byte as well, so it bounds what 8-bit activations alone could give.

**Recovery** (an estimate from measured costs, not a simulation). Since commits 1027ff2 and 79fd370 a lost owner costs the updates since the last publish instead of the run. A publishing update took 32 to 46 s over DigitalOcean against 10 to 15 s for a steady one, so a publish adds about 25 s. With an owner lost on average every M seconds somewhere in the pool, the best interval is about √(2 × 25 s × M) (Young's approximation) and the overhead about twice 25 s over that interval: a publish every ~420 s (about 40 updates) and 12% if M is an hour, every ~850 s and 6% if M is four hours.

**The tiny model** (`price_changes tiny`, 1.06M parameters, against its measured runs):

| Setup | Measured | 30.3M cost model | Cost model refit on tiny runs |
|---|---:|---:|---:|
| 16 workers, home links, corpus order | 6.0 s | 5.4 s (−10%) | 4.8 s (−20%) |
| 16 workers, home links, 3,072-token batches | 3.8 s | 4.3 s (+13%) | 3.5 s (−8%) |
| 4 workers, loopback, corpus order | 0.82 s | 2.0 s (+144%) | 0.8 s (−2%) |
| 4 workers, loopback, 3,072-token batches | 0.40 s | 2.0 s (+400%) | 0.6 s (+50%) |

The refit (`public/research/cost-models/cost_model_tiny.json`) is `simulate fit` on the 1,780 task timings of the two home-link runs in `runs/experiments/tiny/`.

## What it taught

- **Fast uplinks pay only with capacity placement.** The update is bound by the layer owners' uplinks; capacity placement puts the layers on the fastest ones. Two fast workers make 16 identical workers 17% faster, six make them 38% faster, and a pool of only fast uplinks reaches 4.5 s. First-come placement leaves the fast workers on vocabulary slices or idle. Real volunteers are a mixed pool (fibre next to cable and DSL), so this is the cheapest gain to confirm: it is built.
- **A wider model is the largest lever on the KPI.** Arithmetic per token grows with the width squared and the bytes each owner sends with the width, so a 1,024-wide model does twice the useful work per second at 16.3 s per update. It needs its own learning check: per FLOP, a wider model at this data size may learn differently.
- **8-bit activations are worth at most 23%.** The smallest of these levers, and the only one that needs new numerics.
- **Recovery turns a failure into an overhead of 6 to 12%** at hourly to four-hourly owner losses; a cheaper publish, such as owners publishing in turn rather than all at once, would cut it further.
- **Cost models are size-specific.** Fitted on the 30.3M kernels, the simulator charges tiny kernels too much fixed cost on loopback, where nothing else hides it; on home links, where latency dominates, it stays within 13%. Refit before trusting a different model size.

## Limits

- Simulated; each claim needs a real run. The simulator has run 7 to 26% fast on home links.
- The width rows reuse the 30.3M cost model for larger kernels, extrapolating its per-element cost; their tasks were not measured.
- Strong workers have all their advantages at once (compute, both links); which one matters is not separated.
- The recovery overhead is an estimate from three publishing updates and an assumed failure rate.

## Reproduce

```bash
export YGARATY_DATASET_URL=sqlite:///.../opus100-en-pt.sqlite
ygaraty experiment price_changes placement     # about 10 minutes
ygaraty experiment price_changes width
ygaraty experiment price_changes bytes
ygaraty experiment price_changes tiny                                                   # 30.3M cost model
ygaraty experiment price_changes tiny --cost-model public/research/cost-models/cost_model_tiny.json
```
