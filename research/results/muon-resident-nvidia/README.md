# NVIDIA resident Muon evidence

The protocol was frozen before measurement. All four training trials and all
optimizer cases are retained. `attempts.json` also records the first host's
startup failure. `source.json` identifies the base commit, uncommitted patch,
runtime and tool hashes. `hardware.json` contains sanitized GPU validation;
`cleanup.json` records verified teardown and provider-reported cost.

Install the public package with its `server` extra and Deno 2.9.5. The experiment
used Ubuntu 24.04, an RTX 3060 with driver 580.178.04, and Vulkan. Exact Python
versions are in `dependencies.txt`. Run from the tested source revision:

```sh
export DENO_WEBGPU_BACKEND=vulkan
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/probe_training_gpus.py --expected 1 --output /tmp/gpu-private
```

Use the verified selector in `/tmp/gpu-private/selectors-1.json` as `DRI_PRIME`
and set `VK_LOADER_LAYERS_ENABLE=VK_LAYER_MESA_device_select`. Keep these hardware
identifiers private.

```sh
python tools/benchmark_muon.py --repeats 7 --output /tmp/optimizer.json

python tools/training_e2e.py --runtime deno --precision webgpu --training-execution tasks \
  --num-workers 1 --optimizer muon --channels 8 16 16 --batch-size 16 \
  --output /tmp/correctness.json --timeout 180

python tools/training_e2e.py --runtime deno --precision webgpu --training-execution tasks \
  --num-workers 1 --optimizer muon --channels 8 16 16 --batch-size 16 \
  --local-steps 8 --rounds 6 --dataset fashion-mnist \
  --eval-examples 256 --eval-interval 2 --training-run --seed 2026 \
  --muon-execution tasks --output /tmp/2026-tasks.json --timeout 300
```

Then run seed 2026 with `--muon-execution resident`, followed by seed 2027 resident
and seed 2027 tasks, using distinct output files. Training uses loopback HTTP and
JSON tensors. These measurements compare execution modes on one GPU, not PyTorch
or distributed scaling. Momentum, normalization and final updates remain on CPU.

Only the public Ygaraty package and declared dependencies performed training;
provider/SSH scripts handled provisioning, transfer and independent teardown.

These historical timings predate full-round residency. The commands explicitly
select `--training-execution tasks` to reproduce the Muon-only execution path.
