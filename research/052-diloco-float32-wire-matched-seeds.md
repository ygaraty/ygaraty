# Does the float32 DiLoCo wire shorten the round without costing accuracy? 27% faster on home links, but 0.6 points lower in all three seeds

| | |
|---|---|
| **Date** | 2026-10-29 |
| **Path** | Experiment 43's setup on DigitalOcean: 8 replicas of the 1.71M-parameter CIFAR-10 MLP (3072-512-256-10), one per c-2 Droplet (1 BLAS thread), a g-8vcpu-32gb coordinator, over the private network and over links shaped to 100/20 Mb/s with 50 ms. The only difference between the two arms of a pair is `--wire-dtype` (float64 or float32) for the global weights and pseudo-gradients; computation stays float64 |
| **Result** | On 100/20 Mb/s links the 40 rounds take 27.1% less time with the float32 wire (462.4 to 344.0 s, 473.1 to 331.5 s, 462.5 to 343.2 s; faster in all three seeds; prediction 28%). On the private network it is not shown to be faster: +1.8%, -11.1% and -0.5% (mean -3.3%, the 11% is a slow float64 group). Validation accuracy is lower with the float32 wire in all three seeds: -0.26, -0.70 and -0.80 points on the mean of the last three evaluation points (mean -0.59), which fails the frozen rule (mean at least -0.5; no seed below -1.0, which held), and -0.24, -0.64 and -1.20 at round 40. The gap grows along the run (0.00, -0.08, -0.34, -0.28, -0.69 points at rounds 4, 8, 20, 32, 40). Bytes: 29% fewer sent down, 50% fewer received up, exactly. Same data and initial weights in every pair |
| **Evidence** | [`public/research/evidence/diloco/cifar-mlp-float32-wire-matched-do-2026-10-29.json`](evidence/diloco/cifar-mlp-float32-wire-matched-do-2026-10-29.json) (all 12 curves, bytes, hashes); per-job summaries, curves, coordinator logs, the driver and the machines' logs in `runs/pods-data/exp52/` and `runs/pods-data/exp52a/`, `exp52b/` (local, not in git). Code at `5e44132` |

## Question

The float32 wire halves the model payload (weights down, pseudo-gradient up). The earlier paired run on the ConvNet (4 replicas, one seed, `public/research/evidence/diloco/convnet-float32-wire-do-2026-10-28.json`) found the accuracy within 0.52 points and the time 1% shorter on the private network, 2.4% on home links, because that model is compute-bound (0.19 s per local step). It tells nothing about a communication-bound model, and it is one seed. Does the float32 wire shorten the round of the communication-bound MLP on home links by about the share of bytes it saves, and does the validation accuracy stay within a tolerance fixed in advance, over three seeds and eight replicas?

## Why the MLP

In experiment 43 a round takes 4.6 s on the private network and 11.6 s on home links, "most of it the 13.7 MB pseudo-gradient's upload at 20 Mb/s", with 23.5 MB down and 13.7 MB up per replica per round. The float32 wire turns those into about 16.7 MB down (the 9.8 MB of pixel bytes do not change) and 6.9 MB up. This is the setting where the saving can show.

## Frozen before the runs

- **Data and split.** CIFAR-10 as in experiment 43 (`uoft-cs/cifar10`, train SHA-256 8428b53a…). The evaluation set is the 5,000 validation images that `--validation-images 5000` splits off the training data with the run's seed; the test set is not touched. The seed fixes the split, the initial weights and the data order, so both arms of a pair see the same everything. Each job records the SHA-256 of the training subset and of the initial weights, and a pair is invalid if those differ.
- **Example budget.** 8 replicas x H=50 x batch 64 x 40 rounds = 1,024,000 examples (`--examples 1024000`), outer Nesterov (learning rate 0.7, momentum 0.9), inner Adam (learning rate 1e-3), as in experiment 43. Nothing is tuned.
- **Evaluation points.** The global weights after every 4th round (rounds 0, 4, ..., 40; 11 points), evaluated after the training from copies, so the evaluation costs the curve no time.
- **Seeds.** 2026, 2027, 2028.
- **Quality tolerance.** For each seed, the accuracy is the mean over the last three evaluation points (rounds 32, 36, 40). The float32 wire passes if the mean paired difference (float32 minus float64) over the three seeds is at least -0.5 points and no seed is below -1.0 point. Five thousand images give a binomial standard error of about 0.65 points at 70% accuracy and somewhat less at the MLP's 54%, so the rule can only exclude a loss of about a point; it cannot show equality.
- **Speed measure.** Training seconds from the moment all replicas have joined to the end of round 40, and seconds to the first evaluation point at or above 52%, per seed and link. The float32 wire counts as faster on a link if it is faster in all three seeds.
- **Pairing and placement.** The two arms of a pair run at the same time on disjoint sets of 8 c-2 Droplets, each with its own coordinator, so no arm runs on a quieter day. The arm that gets the first set of Droplets alternates between seeds (float64 first set for 2026 and 2028, float32 for 2027), so a slow Droplet cannot always fall on one arm.
- **Order.** Per seed: the pair on the private network, then the pair on shaped links. Shaped links use `YGARATY_SHAPE=1` (the latency proxy, 100/20 Mb/s, real round trip topped up to 50 ms).

## Predictions, made before any run

- **Home links:** the round falls from about 11.6 s to about 8.3 s (the upload from 5.5 s to 2.7 s, the download from 1.9 s to 1.3 s), so the 40 rounds take about 28% less time. If it is under 15% the model is not the bytes-bound case I think it is.
- **Private network:** 4 to 7% faster, because compute and the server's per-round work dominate.
- **Accuracy:** the paired difference is within 0.3 points on average. Rounding the weights to float32 each round is far below the noise of Adam on batches of 64, but 40 rounds of it have not been measured on this model.
- **Same loss curve in the first rounds:** the first evaluation point differs by rounding only.

## Stages

1. **Local gate** (done, see below; free, about 5 minutes): both arms with 8 processes on this workstation for 8 rounds, seed 2026. Checks that the data hashes and initial weights match across arms, that `--wire-dtype float32` reaches the replicas through `serve-cifar` and `scripts/remote.sh diloco-worker`, and that the byte counters show the saving (about 50% up, 29% down).
2. **Bootstrap**: 16 c-2 Droplets and 2 g-8vcpu-32gb coordinators in one region that sells both (lon1 does not sell c-2; nyc3 was used for experiment 49), `pods up`, `pods ship`; a rehearsal of 2 rounds on each arm.
3. **The runs**: 3 seeds x 2 links x 2 arms = 12 jobs, in 6 concurrent pairs.
4. **Write-up**: per-seed tables, the paired differences against the frozen tolerance, the predictions against the measurements, and a compact evidence JSON.

## Local gate (done 2026-10-29)

Both arms, 8 replica processes each (16 processes at once on the 16-core workstation, which also carried other work), 8 rounds (204,800 examples), seed 2026, the frozen split and settings, `ygaraty experiment diloco_cifar_http` without `--external-workers`. Code at `1f9e9b6`.

| | float64 wire | float32 wire |
|---|---:|---:|
| Training subset SHA-256 | 5149106d... | 5149106d... (same) |
| Initial weights SHA-256 | 3533db6b... | 3533db6b... (same) |
| Validation accuracy, round 0 / 4 / 8 | 12.14 / 45.70 / 48.88% | 12.14 / 45.68 / 48.92% |
| Validation loss, round 4 / 8 | 1.5404 / 1.4469 | 1.5409 / 1.4424 |
| Bytes the coordinator sent (down to replicas) | 1.505 GB | 1.068 GB (29% less) |
| Bytes the coordinator received (up from replicas) | 0.874 GB | 0.437 GB (50% less) |
| Training seconds | 189.3 | 187.5 (1% less) |

The gate passes: the two arms see identical data and initial weights, `--wire-dtype float32` reaches the replicas, and the byte savings are the predicted 29% down and 50% up. The accuracies differ by 0.02 and 0.04 points, the losses by rounding-sized amounts. The time is not informative: on loopback with 16 processes competing for cores a round took about 24 s of compute, so the 1% says only that the float32 conversion costs nothing measurable. Whether the saving shortens a round is what the runs on shaped links test. One `ConnectionResetError` appeared in the float64 coordinator's log after round 7, at shutdown, with all 8 rounds complete and no failure reported; the float32 arm had none.

## Results (2026-10-29)

Twelve jobs in six pairs (code `5e44132`, DigitalOcean sfo2: nyc3 does not sell c-2), every job completed, none retried. Each pair ran at the same time on two groups of 8 c-2 Droplets, each with its own g-8vcpu-32gb coordinator; the float32 arm sat on group a in seed 2027 and on group b in the other two. The training subset and initial weights hash the same in both arms of each pair. A 2-round rehearsal on each group came first (seed 2026, private network).

**Training seconds, round 0 to round 40**

| Seed | Link | float64 | float32 | Change | Seconds to 52% (float64, float32) |
|---:|---|---:|---:|---:|---|
| 2026 | private | 182.4 | 185.7 | +1.8% | 74.0, 74.8 |
| 2027 | private | 193.4 | 171.9 | -11.1% | 78.5, 69.5 |
| 2028 | private | 183.3 | 182.4 | -0.5% | 74.2, 73.5 |
| 2026 | 100/20 Mb/s | 462.4 | 344.0 | -25.6% | 186.3, 139.0 |
| 2027 | 100/20 Mb/s | 473.1 | 331.5 | -29.9% | 190.3, 133.4 |
| 2028 | 100/20 Mb/s | 462.5 | 343.2 | -25.8% | 186.4, 138.0 |

On home links the float32 wire counts as faster by the frozen rule (all three seeds), by 27.1% in the mean, and reaches 52% 27.1% sooner. On the private network the three changes disagree in sign and the spread (about 6% between groups) is as large as the mean, so no gain is shown; the prediction of 4 to 7% is not supported. Across the six float64 jobs on home links group b is 2% slower than group a, and in the float32 jobs 3.5%, so a group effect of a few percent exists and the alternation did not remove it entirely.

**Validation accuracy** (5,000 images; identical on both links, the runs being deterministic, so the three seeds are three samples, not six)

| Seed | last 3 points, float64 | float32 | Difference | Round 40, float64 | float32 | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 53.09% | 52.83% | -0.26 | 52.34% | 52.10% | -0.24 |
| 2027 | 52.81% | 52.11% | -0.70 | 52.24% | 51.60% | -0.64 |
| 2028 | 52.57% | 51.77% | -0.80 | 52.88% | 51.68% | -1.20 |
| Mean | | | **-0.59** | | | -0.69 |

By the frozen rule the float32 wire fails on quality: the mean paired difference is -0.59 against a limit of -0.5. No seed's last-three mean is below -1.0 (the limit), but the round-40 value of seed 2028 is (-1.20). The sign is the same in all three seeds, and the mean difference grows along the run (0.00 at round 0, +0.01 at round 4, -0.08 at round 8, -0.34 at round 20, -0.28 at round 32, -0.69 at round 40), which looks like a small real loss accumulating through the weight rounding, not noise; the three paired differences have a mean of -0.59 and a standard error of 0.17 points (t = 3.5 on 2 degrees of freedom, two-sided p about 0.07), so three seeds and 5,000 images point to a loss but do not settle it. The prediction that the difference would be within 0.3 points on average was wrong.

**Bytes per job:** float64 7,525 MB sent and 4,370 MB received by the coordinator; float32 5,339 MB and 2,185 MB (29% and 50% fewer), in every pair.

## Reading

The 27% saving on home links is the prediction (28%) and is as large as the saving in bytes through a 20 Mb/s uplink would give. The accuracy cost is small, consistent in sign, and over the pre-set limit. The two results together make the float32 wire an option to turn on for upload-bound links when a point of accuracy at 40 rounds is acceptable, not a default. Not established: whether the loss comes from rounding the weights sent down or the pseudo-gradients sent up (only the second saves the 50% of the upload that matters on home links), whether it persists past 40 rounds, and whether it appears with the ConvNet (the earlier 4-replica run saw 0.52 points).

## Cost

Planned about $3. Spent: 18 machines (16 c-2 at $0.0625/h and 2 g-8vcpu-32gb at $0.375/h, $1.75/h) from 19:41 to about 21:33, roughly 1.9 hours, about $3.3. The 12 jobs took 40 minutes (20:49 to 21:28). The watchdog deadline was extended by 40 minutes while they ran; the machines were deleted by `pods down` when they finished (0 left in the account).

## Limits to carry into the write-up

- One model family (an MLP), eight replicas, three seeds, 5,000 validation images: the tolerance can exclude a loss of a point, not smaller ones.
- The ConvNet result already says the saving is small for a compute-bound model; this experiment does not change that.
- The latency proxy has no jitter or loss; the c-2 Droplets are shared hosts and differ day to day, which is why the arms run together.
- The group effect (a few percent) and the machines' day-to-day differences limit the private-network comparison to what the three pairs show.
- Float32 weights are rounded once per round on each side; whether that matters for much longer runs or for the 30.3M Transformer is not tested here.

## Reproduce

Two runs (`pods up exp52a|b --provider digitalocean --location sfo2 --server g-8vcpu-32gb --workers 8 --worker-size c-2`, `pods ship`, `scripts/remote.sh worker-setup` on every machine), then the driver, which for each pair starts `serve-cifar 8 40` on both coordinators with `YGARATY_CIFAR_WIRE_DTYPE` and `YGARATY_CIFAR_SEED`, starts the replicas (`YGARATY_SHAPE=1` for home links) and fetches each job's summary and curve; it is kept in `runs/pods-data/exp52/drive52.py`. The pieces are: `YGARATY_CIFAR_WIRE_DTYPE=float32 YGARATY_CIFAR_SEED=2027 YGARATY_CIFAR_EXAMPLES=1024000 bash scripts/remote.sh serve-cifar 8 40` on the coordinator and `YGARATY_SHAPE=1 bash scripts/remote.sh diloco-worker HOST:8000 JOB TOKEN NAME` on each replica.
