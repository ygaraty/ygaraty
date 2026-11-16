# DiLoCo over HTTP tracks the PyTorch reference to within 0.004

| | |
|---|---|
| **Date** | 2026-10-28 |
| **Path** | The tiny model (1.06M parameters, 2,000-piece vocabulary) trained with DiLoCo through the coordinator's HTTP server: a DiLoCoJob and 2 replica processes on this workstation (Ryzen 9 PRO 6950H), against `ygaraty experiment diloco_learning` in PyTorch |
| **Result** | The DiLoCo backend (commit 1553a35) now runs over HTTP (commit c20c975): workers claim a round's weights and batches and submit pseudo-gradients. With 2 replicas, 100 local steps per round and the paper's outer step, held-out loss after 100, 200 and 300 steps per replica is 5.4486, 4.9555 and 4.7067, against PyTorch's 5.4482, 4.9560 and 4.7108 from the same weights and batches: within 0.004. Two rounds over HTTP reproduce the in-process backend's weights bit for bit (tests). A local step took about 1.2 s per replica (float64 NumPy, one thread). |
| **Evidence** | `runs/experiments/tiny/diloco-http/` (not synced) |

## Question

ygaraty-48's PyTorch grid found DiLoCo with 100 local steps per round faster than plain Adam in wall-clock time on the tiny model. Volunteers need it over HTTP. Does the adapter train what the reference trains?

## What was built

- `ygaraty/webapp/diloco_job.py`: a DiLoCoJob holding the backend's Coordinator, unchanged. It deals corpus-order batches round-robin to a fixed set of M replicas (replica r trains on batches r, r+M, ...), as `diloco_learning` deals them.
- `POST /api/jobs/<id>/round` returns the recipe, the global weights and the replica's H batches as one binary frame; `POST /api/jobs/<id>/pseudo` takes the pseudo-gradient and commits the round when every member is in.
- `ygaraty/runtime/diloco_worker.py`: a replica, running the backend's Trainer (which keeps its inner Adam moments between rounds).
- `ygaraty experiment diloco_http`: a job with M local replica processes, held-out loss of the global weights after the run.
- A fix on the way (commit 2c120a3): the PyTorch reference seeded its weights by the parameters' dotted names, a job by the tape's names, so the two started from different draws. They now start identical, which this comparison needs.

## Results

| Steps per replica | Pairs | Held-out, HTTP (float64) | Held-out, PyTorch (float32) | Difference |
|---:|---:|---:|---:|---:|
| 100 | 12,800 | 5.4486 | 5.4482 | +0.0004 |
| 200 | 25,600 | 4.9555 | 4.9560 | −0.0005 |
| 300 | 38,400 | 4.7067 | 4.7108 | −0.0041 |

A round of 100 local steps took 122 to 148 s per replica over HTTP (the first includes reading the round's batches); each replica moves its weights (8.5 MB) and pseudo-gradient (8.5 MB) once per round.

## What it taught

- **The adapter adds nothing to the arithmetic**: bit-identical to the backend in tests, and within float32/float64 rounding of PyTorch over three rounds.
- **The exchange is cheap for this model**: 17 MB per replica per 100 steps, under a second each way on a 100/20 Mb/s link; the local steps (about 1.2 s each in NumPy) are the cost.

## Limits

Synchronous and trusted like the backend: membership is fixed by the first M workers to claim, and a lost member stalls the job. Requests are capped at 16 MiB, enough for the tiny model's pseudo-gradient but not the 30.3M model's (242 MB in float64), which would need the block relay. Three rounds of one setting.

## Reproduce

```bash
export YGARATY_DATASET_URL=sqlite:///.../opus100-en-pt.sqlite
ygaraty experiment diloco_http --replicas 2 --local-steps 100 --rounds 3 --output runs/experiments/tiny/diloco-http/m2-h100      # about 7 minutes
ygaraty experiment diloco_learning --replicas 2 --local-steps 100 --steps 300 --eval-every 100 --output runs/experiments/tiny/diloco-http/torch-m2-h100
python -m unittest tests.test_diloco_http
```
