# Coordinator overhead: profile and fixes

**Result:** local training is **1.63× faster with one contributor** and
**2.38× faster with four**, with identical recorded losses and accuracy.
Both versions use full GPU residency; the optimizer and GPU code are unchanged.

| Contributors | Previous coordinator | Updated coordinator | Speedup |
|---|---:|---:|---:|
| 1 | 11.21 s | 6.88 s | 1.63× |
| 4 | 9.34 s | 3.93 s | 2.38× |

Medians over three seeds; all twelve timed trials retained. Apple M1, one
physical GPU, Deno 2.9.5. Each trial uses 3,072 Fashion-MNIST samples, ConvNet
channels 8/16/16, batch 16 and eight local steps. Evaluation uses 256 images
every 1,536 samples. These are before/after coordinator comparisons, not
multi-GPU scaling or remote time-to-loss results.

## Measured bottlenecks and changes

The matched three-round, four-contributor profile found **2.12 s in redundant
FastAPI response conversion**. Returning protocol JSON directly removes that
pass; response encoding runs off the event loop. Actual JSON encoding remains
about 0.50 s. Checkpoint writes were only 0.05 s and remain durable.

A separate sequential CPU profile identified **1.00 s in deep copying** across
four assignment requests. The HTTP path now serializes each assignment once
and retains immutable bytes for retries. Direct Python callers still receive
isolated copies. The cache holds only the current round, at most 16 MiB per
worker; larger responses are sent without retaining their bytes.

Bounded server waits wake on join, commit or rollback, removing the steady-state
one-second client polling delay. Waiting releases the lock and rechecks the
lease after waking. Lightweight assignment status omits accumulated metrics;
the explicit status endpoint still returns them.

Inclusive profiler scopes overlap and are not additive. Exploratory timings
varied with desktop load; the table uses separate unprofiled, alternating-order
trials. Repeated plan construction and large JSON payloads remain future targets.

## Validation and limits

- All recorded local-step losses, evaluation losses and accuracies match exactly
  across six before/after pairs. Residency counters pass in every trial.
- 40 Python tests passed, one optional test skipped; 148 browser tests and
  TypeScript/generated-code checks passed.
- Two real Chromium contributors passed the CPU-reference gate: maximum weight
  error 1.40e-6, below 5e-5.
- New tests cover response isolation, cache limits, join wakeup, expired leases,
  failed publication and withholding the next round until its checkpoint is durable.
  Existing recovery and lost-acknowledgement tests also pass.

No remote resources or budget were used. NVIDIA gains remain unmeasured.
Implementation and evidence are published in [450b147](https://github.com/ygaraty/ygaraty/commit/450b147eb37a74a0bf0f98154f0efbf46e7527dd).
Recorded source hashes and the reconstruction patch identify the measured
baseline and candidate. This is experimental, not a release.

[All measurements, exact baseline patch and reproduction commands](results/coordinator-overhead-local/README.md).
