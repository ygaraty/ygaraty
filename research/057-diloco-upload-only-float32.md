# Float32 for the pseudo-gradients only: 23% less time on home links and no accuracy loss, now the default

| | |
|---|---|
| **Date** | 2026-10-30 (local quality screen) and 2026-10-31 (timing on DigitalOcean); DO-1 (plan below) |
| **Path** | Experiment 52's setup: 8 replicas of the 1.71M-parameter CIFAR-10 MLP (3072-512-256-10), H=50, batch 64, 40 rounds (1,024,000 examples), outer Nesterov 0.7/0.9, inner Adam 1e-3, computation in float64. New: the two directions of the wire are set apart (`--download-dtype`, `--upload-dtype`), and `--split-seed 2026` fixes one 5,000-image validation split for every seed. Quality: four arms × seeds 2026–2028 on the workstation (loopback, 8 processes). Timing: float64 against upload-only, the same three seeds, on DigitalOcean sfo2 (two pools of 8 c-2 replicas and a g-8vcpu-32gb coordinator, links shaped to 100/20 Mb/s with 50 ms), the two arms of a seed at the same time on disjoint pools, the pool alternating by seed |
| **Result** | **Worth it; now the default.** Upload-only float32 (weights down in float64, pseudo-gradients up in float32) trains the 40 rounds in 370, 380 and 371 s against 491, 480 and 491 s for float64: **23.2% less time** (24.6, 20.7, 24.4%), faster in every seed, and 24% less time to 52% validation accuracy. Accuracy (mean of the last three evaluations) differs from float64 by +0.35, 0.00 and 0.00 points (mean +0.12), the same on the workstation and on DigitalOcean. Float32 both ways is 27% faster (experiment 52) but costs accuracy: −0.14 points here, −0.59 in experiment 52. Float32 down only passes (+0.04) but saves the least. Cost about $3.10 |
| **Evidence** | [`public/research/evidence/diloco/cifar-mlp-upload-only-float32-2026-10-31.json`](evidence/diloco/cifar-mlp-upload-only-float32-2026-10-31.json) (all 18 curves, bytes, hashes); local runs in `runs/do1-local/` (`drive.sh`, `analyze.py`), remote driver, rehearsal and machine logs in `runs/pods-data/do1/`, `do1a/`, `do1b/` (local, not in git). Code at `f478f5c`; `tests/test_diloco_wire.py` |

## Question

Experiment 52 found the float32 wire 27% faster on home links but 0.6 points less accurate, failing the frozen tolerance. Both directions were narrowed together. Which one costs the accuracy? If only one does, the other can be narrowed for free.

## Prediction

Rounding the pseudo-gradient makes an error relative to the round's small update; rounding the weights makes an error relative to the weights themselves, far larger. So the download should carry the loss, and upload-only should be free. The upload is also the slow direction (20 Mb/s), so upload-only should keep most of the time saving: about 20%.

## Quality screen (local)

Accuracy is the mean validation accuracy at rounds 32, 36 and 40, minus the float64 control's, in points. Gate (frozen in the plan below): mean ≥ −0.5, no seed below −1.0.

| Arm | Down / up | 2026 | 2027 | 2028 | Mean | Gate |
|---|---|---:|---:|---:|---:|---|
| Control | float64 / float64 | 53.09% | 52.87% | 52.29% | — | — |
| **Upload-only** | float64 / float32 | **+0.35** | **0.00** | **0.00** | **+0.12** | pass |
| Download-only | float32 / float64 | −0.12 | +0.26 | −0.03 | +0.04 | pass |
| Both | float32 / float32 | −0.26 | −0.23 | +0.07 | −0.14 | pass |

Upload-only matched float64 to two decimals in two seeds; its validation loss differed in the seventh digit (2.1918670 against 2.1918675 for seed 2027), so the arm was live and the effect is real but tiny. Download-only moves the loss in the third digit. Every run used split `5149106d`, and each seed's four arms shared their initial weights.

All arms pass here, and "both" passes too, unlike experiment 52 (−0.59). With 5,000 validation images a run's accuracy is uncertain by about ±0.7 points, so this screen and experiment 52 together leave "both" undecided. Upload-only is the only arm that never moved.

## Timing (DigitalOcean, home links)

| Seed | float64 | Upload-only | Less time | To 52% accuracy | Accuracy change |
|---|---:|---:|---:|---|---:|
| 2026 | 491.4 s | 370.5 s | 24.6% | 198 → 149 s | +0.35 |
| 2027 | 479.5 s | 380.2 s | 20.7% | 193 → 153 s | 0.00 |
| 2028 | 490.5 s | 370.9 s | 24.4% | 198 → 150 s | 0.00 |

Upload bytes: 4.37 GB against 2.19 GB per job, exactly half; download bytes unchanged. The gate (faster in all three seeds, mean gain above the 15% planned threshold) passes. The accuracies equal the local screen's to the hundredth, so the shaped links changed nothing about precision.

## What it taught

- **The accuracy loss of the float32 wire is in the download.** Narrowing only the pseudo-gradients keeps 23 of the 27 points of time saving and none of the accuracy cost.
- **Default changed.** `diloco_cifar_http` now sends pseudo-gradients in float32 unless `--upload-dtype float64` is given; weights still go down in `--wire-dtype` (float64). Plain-Adam baselines keep their wire. `scripts/remote.sh` follows the same default (`YGARATY_CIFAR_UPLOAD_DTYPE=float64` restores the old wire).
- **Two launcher bugs found on the way:** a job token starting with `-` was read as an option by the replica's argument parser (about one job in 64), locally and in `scripts/remote.sh`; now the token follows `--`.

## Also tested: batched relay uploads for R1 (no gain)

R1 of `papers/uplink-bound/NEXT_2026-10-31.md` needed a candidate fix, and L1 pointed at fewer requests per upload. `http-worker --batch-uploads` (or `YGARATY_BATCH_UPLOADS=1`) now sends each result's relay blocks in one `PUT /blocks`, which also accepts a gzipped frame now. On the tiny Transformer with 8 workers on emulated 100/20 Mb/s links (experiment 54's recipe, `runs/r1-local/`), it changed nothing: 3.96 → 3.98 s per update at 512 tiles (one block per result) and 4.45 → 4.51 s at 128 tiles (several blocks per result), losses identical within each pair. The worker already sends a result's blocks in parallel over kept-alive connections, so the round trips overlap. It is off by default, and R1 was not run.

## Limits

One model (the MLP), one link shape, 40 rounds, three seeds. The ConvNet is compute-bound and would gain little (experiment 52's predecessor). The Transformer DiLoCo path (`coordination/diloco_job.py`) has the same switch but keeps its float64 default; it was not measured.

## Reproduce

```bash
runs/do1-local/drive.sh && python runs/do1-local/analyze.py           # local quality screen, ~2.5 h
python -m ygaraty.experiments.pods up do1a --provider digitalocean --location sfo2 --server g-8vcpu-32gb --workers 8 --worker-size c-2 --minutes 150
python -m ygaraty.experiments.pods up do1b ...                         # one at a time: parallel ups collide on the VPC range
python -m ygaraty.experiments.pods ship do1a; python -m ygaraty.experiments.pods ship do1b
# on every machine: source /root/remote.sh; cifar_setup (server) or worker_packages; repository; python_env worker; go_toolchain (workers)
python runs/pods-data/do1/drive_do1.py runs/pods-data/do1/main 40 2026:home,2027:home,2028:home
python -m ygaraty.experiments.pods down do1a; python -m ygaraty.experiments.pods down do1b
```

## Plan as frozen on 2026-10-30 (DO-1)

Status: planned; no machines provisioned or jobs launched. Baseline reviewed: `31e48c7`.

Run actual workloads only on separately authorized external hosts. Prepare code, protocols and offline checks before renting machines. Pin one revision and verify every worker uses it. Before launch, resolve available instance types, region, estimated cost, total budget and teardown deadline; this file is not deployment authorization or a current price quote. Preserve compact reports and failures without exposing credentials.

**Priority: first.** [Experiment 52](052-diloco-float32-wire-matched-seeds.md) completed the both-directions comparison: float32 was 27.1% faster on shaped home links but missed the preset mean validation tolerance by 0.09 percentage points. Do not repeat that campaign unchanged.

### Prepare

The existing single wire-dtype switch does not isolate directions. Implement and test independent weight-download and pseudo-gradient-upload encodings, including endpoint negotiation, dtype validation, actual encoded-byte counters and metadata. Retain float64 computation, optimizer state and coordinator state. Keep the original float64/both-float32 behavior available as controls.

**Done 2026-10-30.** `diloco_cifar_http --download-dtype` and `--upload-dtype` (each defaults to `--wire-dtype`, so the old switch is unchanged); the round header carries `upload_dtype`, the worker encodes its pseudo-gradient in it and the job rejects a mismatch. `--split-seed` fixes one split across initialization seeds. Tests: `tests/test_diloco_wire.py` (all four arms round-trip, byte counts per direction). **Result (2026-10-31): [experiment 57](057-diloco-upload-only-float32.md).** Upload-only float32 passes the quality gate locally (+0.35, 0.00, 0.00 points) and trains 23.2% faster on DigitalOcean home links (all three seeds); it is now the default. Float32 down only passes but saves least; both directions remain opt-in. About $3.10.

### Matrix

| Arm | Global weights sent to replicas | Pseudo-gradients returned |
|---|---|---|
| Control | float64 | float64 |
| Upload-only | float64 | float32 |
| Download-only | float32 | float64 |
| Both | float32 | float32 |

Use the same CIFAR MLP, 8 replicas, H=50, batch 64 and 1,024,000 total examples (40 rounds). Start with a screening seed, then confirm all four arms with three predeclared initialization seeds. Fix one train/validation split across seeds; match data order and initialization within every four-arm comparison. This split protocol differs from historical experiment 52 and requires a fresh control.

Prioritize 100/20 Mb/s, 50 ms shaped links, where the benefit was measured. Run paired arms in balanced order on equivalent dedicated-CPU capacity; avoid sharing compute between simultaneous arms. Use an unshaped reference only after the primary matrix is clean. Record machine effects and report all failures.

### Gates and outputs

Freeze quality limits before running: proposed mean paired last-three-evaluation accuracy difference at least -0.5 percentage points versus float64, with no seed below -1.0 points. Retain these endpoints even if another checkpoint looks better. Evaluate validation at the same example counts; keep the official test set out of selection.

Report loss/accuracy curves, paired quality deltas, encoded bytes by direction, training and total elapsed time, and time to a predeclared validation target. Upload-only is a hypothesis, not a promised solution. Accept a new default only if quality passes and timing improvement exceeds observed run variation; otherwise retain an opt-in tradeoff.
