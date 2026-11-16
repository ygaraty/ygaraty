# DiLoCo across 8 machines: CIFAR-10 trained over HTTP, identical to a single host and 7.7 times sooner to its best accuracy than one machine

| | |
|---|---|
| **Date** | 2026-10-28 |
| **Path** | The DiLoCo backend's coordinator (`ygaraty.diloco`) in a DiLoCoJob served by the HTTP application, with replicas as separate `transports/http/diloco_worker.py` processes running Ygaraty's own float64 NumPy kernels: on this workstation (8 processes, loopback), then on DigitalOcean (sfo2, one private network) with one replica per c-2 Droplet (2 dedicated vCPUs, 4 GB, 1 BLAS thread) and the coordinator on a g-8vcpu-32gb, over data-center links and over links shaped to a home connection (100/20 Mb/s, 50 ms). Model: a 1.71M-parameter MLP (3072-512-256-10) on CIFAR-10. Code at commit 2429f99 (branch `diloco-cifar`, where the driver was `diloco_cifar.py`; now `ygaraty experiment diloco_cifar_http`) |
| **Result** | Eight replicas on eight machines trained the same model the same way as eight processes on one host: test accuracy and loss agree to every printed digit at all 11 evaluation points (54.19% best, 52.69% after 40 rounds), on data-center and on home links alike. Against the same MLP on one c-2 with plain Adam, DiLoCo reached the best accuracy either run attains (about 54.2%) in 111 s against 850 s (7.7 times sooner) and 52% in 74 s against 360 s (4.9 times). Over home links a round takes 11.6 s instead of 4.6 s, most of it the 13.7 MB pseudo-gradient's upload at 20 Mb/s, and 52% comes in 186 s, still 1.9 times sooner than one machine. One seed; test-set evaluations along the curve; an MLP, not a ConvNet |
| **Evidence** | Server reports, curves and replica logs in `runs/pods-data/cifar/` (not synced); local runs in `runs/experiments/cifar/`; compact summary in [`public/research/evidence/diloco/cifar-mlp-http-2026-10-28.json`](evidence/diloco/cifar-mlp-http-2026-10-28.json) |

## Question

The DiLoCo backend had been checked in one process (the coordinator, the trainer, PyTorch parity), over HTTP on loopback with the tiny translation model ([experiment 40](040-diloco-over-http.md)), and with PyTorch convolutions on one host (the CIFAR pilot (historical internal note; not part of this repository)). Does it train a model with replicas that only meet over the network, on machines of their own, and is that faster than one machine?

## What was built

- `ygaraty/diloco/mlp.py`: an MLP image classifier as complete local steps. Its layers, ReLU and cross-entropy are operations the compiler already has, and inner Adam is the Transformer recipe's lowering. Pixels arrive as integers 0..255 and are normalized per channel inside the recipe. One-step and three-step Adam updates match PyTorch's to 1e-12 (`tests/test_diloco_mlp.py`).
- `ygaraty/experiments/diloco_cifar_http.py`: a DiLoCoJob subclass that deals seeded-shuffle CIFAR-10 batches round-robin to M replicas and evaluates the global weights on the 10,000 test images after the run. `--listen 0.0.0.0:8000 --external-workers` serves remote replicas and writes `job.json` (the token is never printed). Pixels travel as their bytes, 8 per float64 of the frame: 9.8 MB of batches per replica per round instead of 78 MB.
- `transports/http/diloco_worker.py` builds the recipe the round's header names (`kind: mlp`), so the same replica process trains either model.
- `scripts/remote.sh serve-cifar`, `diloco-worker` (with `YGARATY_SHAPE=1` for the latency proxy) and `cifar-alone`.
- Data: the Hugging Face copy of CIFAR-10 (`uoft-cs/cifar10`, parquet with PNG images; train 8428b53a…, test 841389e6… by SHA-256), decoded once into CIFAR's layout.

## Setup

Every run: batch 64, inner Adam (learning rate 1e-3, betas 0.9/0.999), seed 2026, weights from the production block initializer.

- **DiLoCo:** 8 replicas, H = 50 local steps per round, outer Nesterov (learning rate 0.7, momentum 0.9), 40 rounds: 2,000 steps per replica, 1.02M images, 20.5 epochs.
- **One machine:** the same driver with 1 replica, outer learning rate 1 and no momentum, which is plain Adam (the replica's moments persist across rounds), for 400 rounds: 20,000 steps, 25.6 epochs. It ran on its own c-2 (w8) while the others ran, so it shared no CPU with them.

The clock of every curve starts when all replicas have joined. Evaluation runs after the training, on copies of the global weights taken at every 4th round (every 4th round of 50 steps for the single machine), so it costs the curve no time.

## Results

Test accuracy (and loss) of the global weights:

| Steps per replica | Rounds | Workstation, 8 processes | DigitalOcean, 8 machines | 8 machines, home links |
|---:|---:|---|---|---|
| 0 | 0 | 11.47% (2.299) | 11.47% (2.299) | 11.47% (2.299) |
| 200 | 4 | 46.27% (1.530) at 42 s | 46.27% (1.530) at 19 s | 46.27% (1.530) at 47 s |
| 600 | 12 | 51.69% (1.373) at 123 s | 51.69% (1.373) at 56 s | 51.69% (1.373) at 140 s |
| 1,000 | 20 | 53.77% (1.362) at 203 s | 53.77% (1.362) at 93 s | 53.77% (1.362) at 232 s |
| 1,200 | 24 | **54.19%** (1.426) at 244 s | **54.19%** (1.426) at 111 s | **54.19%** (1.426) at 279 s |
| 2,000 | 40 | 52.69% (2.000) at 406 s | 52.69% (2.000) at 184 s | 52.69% (2.000) at 465 s |

The three columns are one computation: the same weights, batches and arithmetic, whichever processes or machines did the steps and whatever the links. Only the time differs. The local run sent pixels as float64 (29.6 GB over the job) and the cloud runs packed (7.5 GB). The curves agree because packing moves the same integers.

Against one machine (DigitalOcean, both on c-2):

| | 8 machines, data center | 8 machines, home links | 1 machine, plain Adam |
|---|---:|---:|---:|
| Accuracy at 111 s | 54.19% | 49.73% | 47.70% |
| Accuracy at 465 s | 52.69% | 52.69% | 52.28% |
| First at or above 52% | 74 s | 186 s | 360 s |
| Best accuracy | 54.19% at 111 s | 54.19% at 279 s | 54.30% at 850 s |
| Seconds per round of 50 steps | 4.6 | 11.6 | 3.8 |

Both methods peak at about 54% and then overfit (test loss rises from 1.36 to 2.0 for DiLoCo, 1.43 to 2.49 for one machine): this MLP, without augmentation or regularization, tops out there. DiLoCo got to the peak sooner because eight machines took eight steps for each one the single machine took, and averaging their pseudo-gradients every 50 steps lost little. To 52%, the 8 replicas needed 800 steps each, against 4,800 for one machine: 6 times fewer steps. By images seen they needed somewhat more: 410k (8.2 epochs) against 307k (6.1 epochs), so about 1.3 times the data. The best accuracy came at 12.3 epochs with DiLoCo and 14.3 with one machine.

### Where a round's time goes

| | Data center | Home links |
|---|---:|---:|
| A replica's 50 steps and its upload (mean over replicas) | 3.6 s | 9.1 s |
| The round, at the coordinator | 4.6 s | 11.6 s |
| Bytes per replica per round | 23.5 MB down, 13.7 MB up | the same |

A replica's local step on a c-2 takes about 0.07 s (0.077 s alone on w8), so 50 steps are about 3.5 s. On data-center links the rest of a round is under a second. On home links the upload alone takes about 5.5 s: the pseudo-gradient is 1.71M float64 values (13.7 MB) at 20 Mb/s. The download takes about 1.9 s: the global weights, also float64, plus the round's packed batches (23.5 MB) at 100 Mb/s. The coordinator waits for the slowest of the eight replicas.

## What it shows, and what it does not

- **The backend works across machines.** Each replica process ran alone on its machine and met the coordinator only through `POST /round` and `POST /pseudo`, and the job trained to the same result as on one host. So the coordinator's round protocol, the replicas' private Adam state, the outer Nesterov step and the binary frames hold up off loopback. On these runs the HTTP adapter is deterministic across hosts and links.
- **It beats one machine in time to accuracy here:** 4.9 to 7.7 times on data-center links, 1.9 times on home links, using 8 machines' compute against 1.
- **One seed.** The accuracies are single runs, and the targets were read off the curves afterwards, not fixed in advance.
- **The test set was used along the curve.** The CIFAR pilot's guide reserves it for final evaluation; a validation split would be the stricter protocol.
- **The baseline is not a tuned single machine.** It ran one BLAS thread on a 2-vCPU Droplet, as each replica did, so the comparison is per machine, not against the fastest one-machine setup.
- **An MLP, not a ConvNet.** This measured MLP recipe peaked around 54%; the experiment does not isolate why. At the historical revision, the ConvNet pilot used a separate PyTorch path. Later portable convolution and HTTP support are documented in experiments 44–45; those capabilities do not change this run's evidence.
- **Fixed membership.** A lost replica would stall the job, as it does in the DiLoCo backend generally. Nothing here tests failure, recovery or untrusted contributors.

## Reproduce

Locally (16 logical CPUs; about 7 and 1.5 minutes):

```bash
ygaraty experiment diloco_cifar_http --replicas 8 --local-steps 50 --rounds 40 --eval-every 4 --output runs/experiments/cifar/diloco-m8-h50
ygaraty experiment diloco_cifar_http --replicas 1 --local-steps 50 --rounds 40 --eval-every 4 --outer-learning-rate 1 --momentum 0 --output runs/experiments/cifar/adam-m1
```

The data goes in `runs/datasets/cifar10/hf/{train,test}.parquet`, downloaded from `https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet`.

On DigitalOcean (about 40 minutes and $0.60; lon1 had run out of c-2 after 7 Droplets, so this ran in sfo2):

```bash
ygaraty experiment pods up cifar --provider digitalocean --location sfo2 --server g-8vcpu-32gb --workers 9 --worker-size c-2 --minutes 90
# on every machine: scripts/remote.sh worker-setup, then (source scripts/remote.sh; cifar_setup) and on the server cifar_data
# server: ./scripts/remote.sh serve-cifar 8 40, and read job and token from its run folder's job.json
# w0..w7: ./scripts/remote.sh diloco-worker SERVER_PRIVATE_IP:8000 JOB TOKEN wN        (YGARATY_SHAPE=1 for home links)
# w8:     ./scripts/remote.sh cifar-alone 400
ygaraty experiment pods down cifar        # fetches runs/ first
```

Machines ran with `YGARATY_BRANCH=diloco-cifar`. A script that passes the token from `job.json` to the replicas without printing it did the middle steps.
