# GPU-resident local training evidence

Apple M1, one physical GPU. Deno 2.9.5; browser correctness also checked with
Chromium. All twelve successful timed trials are retained in three series.
Each protocol records source hashes, trial order, settings and timing exclusions.
`fashion-paired` is the earlier candidate; `fashion-final` and `fashion-repeat`
include the uniform-metadata admission limit. No arithmetic changed between them.
The desktop was not isolated; no thermal or cache reset was performed.

Reproduce from the public checkout with training dependencies and Deno:

```sh
npm ci
npm run build
PYTHONPATH=. python tools/check_resident_training.py --output /tmp/resident-check.json
PYTHONPATH=. python tools/benchmark_resident_training.py \
  --num-workers 1 --samples 768 --timeout 180 --output /tmp/resident-comparison
```

Choose a new output directory for each repetition. The benchmark compares the
previous Muon-only resident path (`--training-execution tasks`) with complete
GPU-resident rounds, alternating arm order for seeds 2026 and 2027. It checks
matching configuration, sample counts, source hashes, losses and GPU counters.

For CPU-reference weight checks:

```sh
PYTHONPATH=. python tools/training_e2e.py \
  --runtime deno --precision webgpu --num-workers 1 --optimizer muon \
  --channels 8 16 16 --batch-size 16 --local-steps 8 --rounds 2 \
  --output /tmp/resident-parity.json --timeout 180
```

Repeat with `--optimizer adam`, or `--runtime chromium --num-workers 2`.
The two browser contributors share one GPU; this is not a scaling measurement.
The checks use synthetic fixtures; timed trials use Fashion-MNIST.

JSON omits session round identifiers and GPU selectors. Source hashes remain
those actually tested. Kernel gates cover 19 independent NumPy-reference cases,
hidden nonfinite values, cancellation and device loss. Raw development logs are
not included. No remote machines were used.
