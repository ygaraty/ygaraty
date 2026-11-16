# Evolution strategies on home links: does a workload with no chain scale with workers?

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open |
| **ID** | WS-7 (number added on finish) |
| **Category** | training |
| **Where** | workstation |
| **Date** | Added 2026-11-03 |
| **Path** | Workstation, 1–8 NumPy workers, emulated 100/20 Mb/s, 50 ms links (experiment 59's link setup), a shared-seed ES job on the HTTP coordinator, a small model (about 5,000–20,000 parameters). Not on the simulation-only machine |

## Question

The sharded update is bound by its chain of dependent hops, and no uplink is busy for more than half an update (L1). More workers add capacity the chain cannot use.

Evolution strategies with shared random seeds (Salimans et al. 2017, from memory) have no chain:
- every worker regenerates its own perturbations from seeds, evaluates them on its own data, and uploads **one score** per perturbation;
- the coordinator broadcasts the seeds and scores, and every worker rebuilds the same update locally.

Communication per generation is a few hundred bytes per worker and one round trip, whatever the model size. ES only suits small parameter counts. Within those it is a good use of the pool: meta-learning a local rule as in [M-1](M-1.md)'s VSML arm (Kirsch & Schmidhuber 2021 meta-train with ES), learned optimizers, hyperparameters, or n-gram hint settings.

**On emulated home links, does ES throughput (perturbations evaluated per second) grow near-linearly from 1 to 8 workers, and at what evaluation time per perturbation does the per-generation round trip start to bind?**

## Prediction

- **Near-linear scaling** (at least 85% of linear at 8 workers) when evaluating a worker's share of perturbations takes at least 0.5 s per generation: one 52 ms round trip plus a gather of 8 small posts is under 10% of that.
- **Scaling collapses** below about 0.1 s of evaluation per generation, where the round trip dominates. Fix it by giving each worker more perturbations per generation, not more workers.
- **Bytes per worker per generation:** under 1 KB up, under 4 KB down at 8 workers.
- **The rebuild** costs each worker n × P multiply-adds per generation (n perturbations, P parameters): negligible at P ≤ 20,000.

## Gate

TBD, to be frozen before any run. Proposed:
- **Primary:** at 8 workers, perturbations evaluated per second at least **80%** of 8× the 1-worker rate, at the evaluation size where one worker's share takes 0.5 s per generation.
- **Required alongside:** the loss curve per generation matches a single-process reference with the same seeds exactly, in float64. Shared seeds make the update deterministic, so this is checkable bit for bit.
- **Report also:** the scaling curve over evaluation sizes (0.05, 0.1, 0.5 and 2 s per generation), bytes per generation, and the time a worker waits per generation.

## Protocol

1. **What has to be built:**
   - an `es` job in the HTTP coordinator: per generation, task k = "evaluate perturbations [i, j) of generation g with seed s"; result = scores;
   - a commit that broadcasts (g, seeds, scores);
   - a worker update rule: antithetic pairs, rank-normalized scores, Adam on the ES gradient estimate;
   - a single-process reference, `ygaraty.es.reference`, with the same seeding.
2. **Model and data:** a small MLP classifier (one hidden layer, about 10,000 parameters) on the CIFAR pilot of M-1, downsampled to 8×8 grey (local file, SHA-256 in M-1). Fitness: minus the loss on a fixed 256-example minibatch per generation, drawn by seed.
3. **Arms:** 1, 2, 4 and 8 workers × 4 evaluation sizes, set by perturbations per worker per generation. 50 generations each, 3 interleaved repeats.
4. **Links:** each worker shaped to 100/20 Mb/s and 50 ms round trip, as in experiment 59. The coordinator on the workstation.

## Trigger on the workstation

```bash
git pull && pip install -e . && python -m unittest tests.test_es_job      # job equals the single-process reference, float64
for w in 1 2 4 8; do for m in 4 8 40 160; do for rep in 1 2 3; do
  YGARATY_SHAPE=1 YGARATY_DOWN_MBPS=100 YGARATY_UP_MBPS=20 YGARATY_RTT_MS=50 \
  ygaraty experiment es_pool --workers $w --perturbations-per-worker $m --generations 50 \
    --dataset datasets/cifar10/cifar-pilot.npz --output runs/experiments/WS-7/w$w-m$m-r$rep
done; done; done
```

`es_pool` and its flags are what has to be built.

## Log

- 2026-11-03: Added, from the VSML paper (arXiv 2012.14905), with user approval. Pairs with M-1's VSML arm, whose meta-training could run this way.
