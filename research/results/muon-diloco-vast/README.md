# Muon DiLoCo benchmark evidence

[Results and conclusions](../../muon-diloco-vast.md).

- [Ygaraty protocol and source hashes](protocol.json)
- [PyTorch protocol and upstream revision](torch-protocol.json)
- [All eight trial summaries](all-summary.json) and [medians](medians.json)
- [GPU placement verification](gpu-validation.json)
- [Setup attempts](attempts.json) and [cleanup](cleanup.json)

Individual trial JSON files retain evaluation curves and configuration. Numerical
checks are in `parity-*.json` and `torch-*-parity.json`. Dependency versions are
in `python-dependencies.txt` and `torch-dependencies.txt`.

## Reproduce

Run these commands from the repository root.

Use the candidate public source checkout, its server and optional training-torch
extras, Deno 2.9.5, and a working NVIDIA Vulkan driver. Minimal containers may
need `libvulkan1`, `vulkan-tools`, `mesa-vulkan-drivers`, `libxext6`, `libx11-6`,
`libxrender1` and `libegl1`. Run from an isolated experiment directory/host.

```bash
python tools/probe_training_gpus.py --expected 4 --output runs/gpu-private

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  python tools/training_e2e.py --runtime deno --precision webgpu \
  --optimizer muon --dataset fashion-mnist --num-workers 1 \
  --gpu-selectors runs/gpu-private/selectors-1.json \
  --channels 8 16 16 --batch-size 16 --local-steps 8 --rounds 120 \
  --seed 2026 --eval-examples 1024 --eval-interval 12 \
  --training-run --timeout 1500 --output runs/muon-1w-seed2026.json
```

For four workers use `--num-workers 4`, `selectors-4.json`, `--rounds 30`,
`--eval-interval 3`, and a different output. Repeat both arms with seed 2027 in
reverse order. First run the same commands without `--training-run`, using
`--dataset fixture --rounds 2 --local-steps 2`, to check numerical parity.
Keep the GPU probe directory private.

For the reference, install `torch==2.9.1` from the official CUDA 12.8 wheel index
in a separate environment containing the same Ygaraty wheel. Download the pinned
[upstream source](https://raw.githubusercontent.com/bentherien/muloco-1/33cc33315eacd1bb5f179d7d5d9f85e012fe58f2/muloco/pytorch/muloco.py) to `runs/upstream-muloco.py`. Select the same
physical GPU by setting `CUDA_VISIBLE_DEVICES` to its UUID from the private
probe, then run:

```bash
PYTHONPATH=. python tools/torch_muloco_reference.py \
  --upstream-file runs/upstream-muloco.py --device cuda:0 \
  --verify-only --output runs/torch-parity.json

PYTHONPATH=. python tools/torch_muloco_reference.py \
  --upstream-file runs/upstream-muloco.py --arm upstream \
  --seed 2026 --samples 15360 --output runs/torch-upstream-seed2026.json
```

Repeat with `--arm ygaraty-math` and both seeds. The frozen reference order is
math/upstream for 2026, upstream/math for 2027. The helper reports setup and
training times separately and rejects an upstream source hash mismatch.
Regenerate the combined figure with `python tools/plot_muon_reference.py
research/results/muon-diloco-vast`.
