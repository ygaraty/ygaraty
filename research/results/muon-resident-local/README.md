# Resident Muon local measurements

One Apple M1 GPU; Deno 2.9.5. No cloud resources. `protocol.json` records the
training trial order. All four trials are retained. JSON files contain measured
results; optimizer microbenchmark trials include one warmup per arm and seven
paired repetitions. The short correctness checks use synthetic images.

Reproduce from the public checkout with its training dependencies and Deno:

```sh
PYTHONPATH=. python tools/benchmark_muon.py --output /tmp/muon-kernels.json

PYTHONPATH=. python tools/training_e2e.py \
  --runtime deno --precision webgpu --training-execution tasks --num-workers 1 --optimizer muon \
  --channels 8 16 16 --batch-size 16 --local-steps 8 --rounds 6 \
  --dataset fashion-mnist --eval-examples 256 --eval-interval 2 \
  --training-run --seed 2026 --muon-execution resident \
  --output /tmp/muon-resident.json --timeout 180
```

Repeat with `--muon-execution tasks` for the original per-operation execution.
Run seed 2027 in reverse arm order. The switch is a Deno benchmark control;
normal browser and native training automatically use supported resident blocks.
The Fashion-MNIST loader uses the existing cache or downloads the public dataset.

For CPU-oracle checks, omit `--training-run`, use the default fixture dataset,
`--rounds 2 --local-steps 2`, and retain channels 8/16/16 and batch size 16.
Use `--runtime chromium --num-workers 2` for the browser check, omitting the
Deno-only execution controls. Current Chromium uses full-round residency; the
recorded source hash identifies the historical Muon-only check. Both contributors
share the same physical GPU; this is a protocol/correctness check, not GPU scaling.

Limits: 768 training examples and 256 evaluation examples per training trial;
two seeds; no thermal/cache reset. Timing includes contributor startup, training
and evaluation. Initial coordinator evaluation is outside the timed section.
These measurements do not establish convergence quality or Linux/NVIDIA speedup.

These historical timings predate full-round residency. The commands explicitly
select `--training-execution tasks` to reproduce the Muon-only execution path.
