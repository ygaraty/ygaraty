# GPU-resident NVIDIA comparison

One host, four verified RTX 3060 GPUs (12 GiB each). Three seeds: 2026, 2027,
2028. All twelve timed trials are retained. `protocol.json` pins candidate source,
settings and trial order; `summary.json` preserves individual measurements and
`medians.json` aggregates them. This tests same-host scaling, not WAN operation.

Install the public candidate with its server dependencies, Deno 2.9.5, PyTorch
2.9.1 with CUDA 12.8, Vulkan tools and Mesa's device-select layer. Dependency
versions are recorded in `dependencies.txt`. The tested snapshot is identified by its base commit, patch hash and scoped
file hashes in `source-manifest.json`. Implementation and evidence were published
in [450b147](https://github.com/ygaraty/ygaraty/commit/450b147eb37a74a0bf0f98154f0efbf46e7527dd); that commit also includes later coordinator changes, so it is not
the exact snapshot timed here. This is experimental, not a release.

Verify physical GPU isolation; keep selectors and identifiers private:

```sh
PYTHONPATH=. python tools/probe_training_gpus.py --expected 4 --output runs/private
PYTHONPATH=. python tools/check_resident_training.py --output runs/kernel-gates.json
```

Run the numerical check first (same command below, but omit `--training-run`,
use `--dataset fixture --rounds 2`, and choose a different output). Repeat with
one and four workers; the weight-error tolerance is 5e-5.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python tools/training_e2e.py --runtime deno --precision webgpu \
  --training-execution resident --num-workers 1 \
  --gpu-selectors runs/private/selectors-1.json --optimizer muon \
  --channels 8 16 16 --batch-size 16 --local-steps 8 --rounds 120 \
  --dataset fashion-mnist --eval-examples 1024 --eval-interval 12 \
  --training-run --seed 2026 --timeout 1500 \
  --output runs/2026-resident-1.json
```

For four workers, use `--num-workers 4`, `selectors-4.json`, `--rounds 30`,
`--eval-interval 3`. Keep all other settings fixed. Use the frozen arm order and
all three seeds. Every resident round must report zero CPU tensor tasks and one
readback per worker; four processes alone do not demonstrate GPU isolation.

For the reference, download the pinned [upstream optimizer](https://raw.githubusercontent.com/bentherien/muloco-1/33cc33315eacd1bb5f179d7d5d9f85e012fe58f2/muloco/pytorch/muloco.py)
to `runs/upstream-muloco.py`. Set `CUDA_VISIBLE_DEVICES` to the UUID of the same
physical GPU as the one-worker Vulkan selector, then run:

```sh
PYTHONPATH=. python tools/torch_muloco_reference.py \
  --upstream-file runs/upstream-muloco.py --verify-only \
  --output runs/torch-parity.json
PYTHONPATH=. python tools/torch_muloco_reference.py \
  --upstream-file runs/upstream-muloco.py --arm upstream \
  --seed 2026 --samples 15360 --output runs/2026-torch-upstream-1.json
```

Repeat with `--arm ygaraty-math` and the other seeds. The public helper rejects
an upstream source hash mismatch. Outer settings match our previous comparison;
the upstream inner optimizer and arithmetic differ from Ygaraty's.

Timing includes training and evaluation. Ygaraty additionally includes
contributor startup and coordinator work. PyTorch reports setup separately;
loss crossings use its training clock. Dataset loading and initial Ygaraty
coordinator evaluation are excluded. Crossings are first observed evaluations
at 1,536-sample intervals, not interpolated estimates. No thermal/cache reset.

`gpu-validation.json` records hardware checks; `attempts.json` records benchmark
validity. All twelve timed trials are retained. Session IDs and selectors are
removed from public trial reports. Cost and teardown are summarized in the
[report](../../resident-training-nvidia.md); operational notes are archived separately.
