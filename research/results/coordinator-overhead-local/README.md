# Coordinator overhead: local evidence

One Apple M1 GPU, Deno 2.9.5. Four contributors share that GPU; worker count
is not a physical-GPU scaling claim. All twelve unprofiled trials and the
exploratory profiles are retained. Raw operational logs and session IDs are omitted.

The baseline already has full GPU residency. Only the coordinator/HTTP path
changes. `source.json` and `paired/protocol.json` identify both package hashes,
the patch, workload and trial order. No changes were committed for this run.

## Reproduce the comparison

Use the candidate source checkout and its server/training dependencies. The
included patch reconstructs the baseline without changing your working checkout:

```sh
baseline_dir="$(mktemp -d)"
cp -R ygaraty tools "$baseline_dir/"
git -C "$baseline_dir" apply --reverse \
  "$PWD/research/results/coordinator-overhead-local/coordinator.patch"
PYTHONPATH=. python tools/benchmark_training_coordinator.py \
  --baseline-root "$baseline_dir" --output /tmp/coordinator-comparison \
  --samples 3072 --workers 1 4 --seeds 2026 2027 2028
```

Choose a new output directory for each run. The helper verifies frozen package
hashes, matching sample/configuration/dataset identities, full GPU residency,
identical accuracy and per-step/evaluation loss differences below 1e-10.
All measured differences here were zero. Initial coordinator evaluation and
dataset loading are excluded; contributor startup and later evaluation count.
There is no thermal/cache reset. Alternating order reduces but does not eliminate
noise on a nonisolated desktop.

## Reproduce the profile

```sh
PYTHONPATH=. python tools/profile_training_coordinator.py \
  --profile-output /tmp/coordinator-profile.json \
  --runtime deno --precision webgpu --num-workers 4 --optimizer muon \
  --channels 8 16 16 --batch-size 16 --local-steps 8 --rounds 3 \
  --dataset fashion-mnist --eval-examples 256 --eval-interval 3 \
  --training-run --output /tmp/coordinator-profile-trial.json

PYTHONPATH=. python tools/profile_training_coordinator.py \
  --preparation-only --profile-output /tmp/coordinator-preparation.json
```

Repeat with `PYTHONPATH="$baseline_dir"` and `--source "$baseline_dir"` for the
baseline. Live phase timers include initial evaluation, overlap across threads,
and perturb wall time. `assignment_bytes` includes time asleep waiting for work;
`_assignment` measures preparation. Do not sum inclusive scopes into percentages.
The sequential CPU profile covers four responses with batch 16 and eight local
steps; it does not measure GPU execution or end-to-end speed.

`exploratory/` retains the early baseline, HTTP-only and cached/waiting profiles.
Desktop timing varied substantially; the final matched profiles and unprofiled
paired trials support the report, not comparisons between exploratory runs.
The first concurrent CPU-profiling attempt failed due to instrumentation, then
was replaced with timers plus a separate sequential CPU profile.

The Chromium check uses two contributors, Muon, batch 16, eight local steps and
two rounds against the CPU reference. Its weight error was 1.40e-6 (limit 5e-5).
