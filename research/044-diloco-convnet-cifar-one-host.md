# The CIFAR ConvNet on full data, one host: 4 DiLoCo replicas end where plain Adam does, 8 fall short in 10 epochs

| | |
|---|---|
| **Date** | 2026-10-28 |
| **Path** | The CIFAR DiLoCo pilot's ConvNet (`ygaraty experiment diloco_cifar`: 94,538 parameters, PyTorch 2.14.1 CPU kernels in float64, the backend's Coordinator) on the full 45,000/5,000 train/validation split, on this workstation (Ryzen 9 PRO 6950H, 16 logical CPUs): plain Adam against DiLoCo with 4 and 8 threaded replicas, H = 50, 8 PyTorch threads for each method, about 10 epochs (450k images) each, run one after another |
| **Result** | Target fixed before the runs: 60% validation accuracy. Plain Adam reached it after 131 s of training (185k images). DiLoCo with 4 replicas took 171 s (243k), and DiLoCo with 8 never did (59.9% after 460k). At 10 epochs, 4 replicas reached 67.9% and were still rising, against Adam's 65.6% (68.0% at its best, 437k images); 8 replicas reached 59.9%. Training time per image was the same for all three (about 0.7 ms), because the replicas share the host's 8 threads. So these runs measure learning per image, not speed: on one host DiLoCo cannot be faster, and with 8 replicas its 18 outer updates in 10 epochs are too few. One seed |
| **Evidence** | `runs/experiments/cifar-convnet/` (not synced): each method's `metadata.json`, `curve.jsonl` and `summary.json`; data `runs/datasets/cifar10/cifar-pilot-hf.npz` (SHA-256 136e3a1b…) |

## Question

The pilot checked the ConvNet with DiLoCo on 1,280 images (the [maturity tracker](../../docs/status/diloco.md)). [Experiment 43](043-diloco-cifar-across-machines.md) showed DiLoCo across machines with an MLP, which tops out near 54%. On all of CIFAR-10's training images, with a model that learns more, does averaging replicas every 50 steps cost accuracy, and how much?

## Data

`cifar_prepare` takes only the official binary archive, which downloaded at about 60 KB/s. Its download was stopped at 80% when the workstation ran short of memory: another process, a test using 22 GB, had filled it. The NPZ was built instead from the Hugging Face copy (`uoft-cs/cifar10`, train parquet SHA-256 8428b53a…, the file experiment 43 used), with `cifar_prepare`'s split rule: `default_rng(2026).permutation(50000)`, the first 5,000 for validation, and the test set not read. The indices are in `cifar-pilot-hf.json`. Whether the Hugging Face training order matches the official archive's, and so whether the split is the same images as `cifar_prepare` would give, is unchecked. Class counts in training are 4,470 to 4,525.

## Setup

These are the pilot's defaults: inner Adam (learning rate 1e-3), batch 64, seed 2026, pixels scaled to [-1, 1], no augmentation or batch normalization, and validation after every round.

| Method | Rounds × local steps | Replicas | Outer step | Images |
|---|---|---|---|---:|
| Plain Adam | 141 × 50 | 1 | none | 450,640 |
| DiLoCo | 35 × 50 | 4 | Nesterov 0.7, momentum 0.9 | 447,496 |
| DiLoCo | 18 × 50 | 8 | Nesterov 0.7, momentum 0.9 | 460,240 |

The replicas ran threaded in one process (`--execution threaded --threads 8`).

## Results

Validation accuracy (loss) of the global model at about equal images:

| Images | Plain Adam | DiLoCo, 4 replicas | DiLoCo, 8 replicas |
|---:|---|---|---|
| 64k | 47.50% (1.435) | 36.98% (1.712) | 29.94% (2.118), at 77k |
| 128k | 56.44% (1.227) | 49.42% (1.393) | 39.54% (1.641) |
| 192k | 60.00% (1.130) | 55.90% (1.220) | 46.56% (1.469), at 205k |
| 256k | 60.72% (1.105) | 60.56% (1.105) | 49.54% (1.381) |
| 320k | 64.18% (1.014) | 63.50% (1.021) | 54.56% (1.269), at 332k |
| 384k | 64.92% (0.989) | **66.32%** (0.951) | 56.24% (1.214) |
| 450k | 65.56% (0.952) | **67.90%** (0.910) | 59.90% (1.142), at 460k |

| | Plain Adam | DiLoCo, 4 replicas | DiLoCo, 8 replicas |
|---|---:|---:|---:|
| Training seconds to 60% | **131** | 171 | not reached |
| Images to 60% | **185k** | 243k | not reached |
| Best accuracy | 68.02% (437k images) | 67.90% (the last round) | 59.90% (the last round) |
| Training seconds, whole run | 318 | 313 | 340 |
| Elapsed, with validation | 526 | 363 | 373 |

- **Replicas cost accuracy early and none by 10 epochs, with 4 of them.** Four replicas trail Adam by 7 to 10 points over the first 2 epochs, draw level near 5.7 epochs (256k images), and are ahead after that. Adam's accuracy wobbles (66.5% at round 110, 64.9% at round 120, 68.0% at round 137), while the 4-replica curve dips by at most 0.6 points between rounds in its second half. Averaging smooths it.
- **Eight replicas need more rounds.** They used the same images in 18 rounds, so the global model moved 18 times in 10 epochs. They were still climbing at about 3 points per 70k images when the run ended. Fewer local steps per round or a longer run are the obvious next settings. With the MLP, experiment 43 saw 8 replicas reach the model's ceiling in 12 epochs at H = 50.
- **Throughput is equal by construction.** Every method trained at about 0.7 ms per image, because the replicas share 8 threads. That is why the target came first for plain Adam: it needed fewer images. Elapsed time with validation favours DiLoCo (363 s against 526), but only because Adam validated 141 times and DiLoCo 35 or 18 times, so the training seconds are the fair clock.

## What it shows, and what it does not

- The pilot's ConvNet learns CIFAR-10 to about 68% in 10 epochs on this split. That makes it a better test model than the MLP, whose ceiling is near 54%.
- At 4 replicas, DiLoCo's averaging every 50 steps costs about 30% more images to reach 60% and nothing in accuracy at 10 epochs. That is the price DiLoCo would pay per image for running replicas on separate machines, where each adds its own compute.
- This is not a speed result. Replicas on one host share its CPU; the speedup needs one machine per replica, as in experiment 43. At this experiment's historical stage the ConvNet HTTP replica was absent. It now exists; experiment 45 documents subsequent remote browser checks and throughput, not a full-data multi-host time-to-quality result.
- **One seed.** Adam's curve moves by about 2 points between neighbouring evaluations, so the 4-replica lead at the end is within that.
- **Data:** the split follows `cifar_prepare`'s rule on the Hugging Face copy, not yet verified against the official archive.

## Reproduce

```bash
python - <<'EOF'   # the NPZ from the Hugging Face parquet in runs/datasets/cifar10/hf (experiment 43), cifar_prepare's split
import numpy as np
from ygaraty.experiments.diloco_cifar_http import load_cifar
d = load_cifar(); x = d['train_images'].reshape(-1, 3, 32, 32); y = d['train_labels'].astype(np.int64)
ids = np.random.default_rng(2026).permutation(len(x)); v, t = ids[:5000], ids[5000:]
np.savez('runs/datasets/cifar10/cifar-pilot-hf.npz', train_images=x[t], train_labels=y[t], validation_images=x[v], validation_labels=y[v])
EOF
D=runs/datasets/cifar10/cifar-pilot-hf.npz O=runs/experiments/cifar-convnet
ygaraty experiment diloco_cifar --data $D --algorithm adam --rounds 141 --local-steps 50 --batch 64 --threads 8 --output $O/adam-t8
ygaraty experiment diloco_cifar --data $D --algorithm diloco --replicas 8 --rounds 18 --local-steps 50 --batch 64 --execution threaded --threads 8 --output $O/diloco-m8-h50
ygaraty experiment diloco_cifar --data $D --algorithm diloco --replicas 4 --rounds 35 --local-steps 50 --batch 64 --execution threaded --threads 8 --output $O/diloco-m4-h50
```

About 9, 6 and 6 minutes, run one at a time.
