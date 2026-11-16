# Transformer minibatch computed by two browser tabs

| | |
|---|---|
| **Date** | 2026-10-22 |
| **Path** | HTTP + browser (JavaScript float64) |
| **Result** | Two browser tabs computed 6,923 numerical tasks; all 511 PyTorch checks passed (max error 9.01e-12). |
| **Evidence** | `runs/http-transformer-9c6d5f36d3464fd2ba2e0cfea3f5b525/verification.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

The webapp's **Transformer · EN → PT-BR · Adam** recipe runs the same prepared
minibatch as the Python milestone, using actual JavaScript browser computations.
It uses the existing local oracle in
`runs/translation-verified-checkpoint/reference` by default. To prepare another
trusted oracle, run `ygaraty/verification/translation_reference.py` with `--checkpoint`, `--tokenizer`,
`--pairs`, and `--output`, then supply that directory via `--translation-reference`.
The oracle is prepared with PyTorch once; it does not execute browser tasks.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ygaraty webapp --port 8092
```

Open the printed administrator URL, select the Transformer recipe and create a
job. Open its contributor link in two browsers and click **Start contributing**
in each. The dashboard shows operations, worker contributions, loss, and progress
through all 511 PyTorch tensor checks. After completion, download the verification
report from the Learning panel. The existing checkpoint endpoint exposes final
parameter and Adam-state manifests; tensors remain in the job's store until
released. **Cancel & release tensors** releases storage, including completed jobs.

This is a fixed correctness recipe: all settings come from the prepared reference,
verification is mandatory, dropout is off, and the default browser matmuls use float64 CPU
kernels. The optional WebGPU comparison mode is described below. MLP/LSTM retain optional WebGPU support.
Changing the learning rate, minibatch, or optimizer requires a newly prepared
reference. The UI hides unrelated size controls for this recipe.

Transformer tasks use binary little-endian float64 payloads: a uint32 JSON-header
length, JSON header, then packed tensor values. Shape metadata is checked on
receipt. Claim inputs and submitted results avoid decimal JSON arrays. Numerical
results are checked against NumPy for each task; complete logits, loss, gradients,
updated weights and moments are independently checked against PyTorch with
rtol=2e-7 and atol=2e-9. A failed parity check fails the job. Intermediate tensors
are released after their last consumer; assembly/transpose reuse block metadata.

The local Transformer store has a 1.5-GiB tensor-payload ceiling (not an RSS
limit). Only one unreleased Transformer job is admitted per server. This is not
a 128-MiB-worker capacity claim or a long-training implementation. Redis and
PostgreSQL remain optional storage backends, but this browser milestone is tested
with the local store. Jobs still live for the server session; the complete event
journal, task plan and verification report are saved under `runs/http-transformer-ID/`.

`tests/test_http_translation.py` runs two Node instances of the identical browser
SDK against an independent small PyTorch Transformer through actual binary HTTP.
It checks every gradient/update/moment, padding and masking, mandatory verification,
float64 enforcement, malformed frames, report authorization and cancellation.
Actual-browser validation of the full checkpoint is recorded separately.

Full-checkpoint browser verification on 2026-10-22: created job
`9c6d5f36d3464fd2ba2e0cfea3f5b525` through the actual web form and ran two in-app
browser tabs. They computed 3,782 and 3,141 numerical tasks respectively (6,923
total; 11,092 operations including metadata). All 511 PyTorch checks passed;
maximum absolute error 9.0143e-12, loss 2.6173874537584383, zero retries/rejections.
See `runs/http-transformer-9c6d5f36d3464fd2ba2e0cfea3f5b525/verification.json`
and its `dashboard.png`. This run used JavaScript float64, not WebGPU. The full
regression suite passed 28 tests with 24 external-database tests skipped.
