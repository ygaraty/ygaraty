# Qwen3-Coder Q4_K_M: direct llama.cpp experiment

Completed 2026-11-26. **Q4 llama.cpp is a promising lower-cost dedicated option,
but this is exploratory evidence, not a production release gate.** On the RTX
4090, the tested capacity boundary was 4K × 8, 16K × 4 and 32K × 2. On the same
A100, graph-enabled BF16 vLLM outperformed Q4 llama.cpp in the measured
near-full-context cells, with lower first-token latency. Eager-only comparisons
would have produced a misleading engine conclusion.

All 908 measured benchmark requests completed with HTTP success; **906/908**
contained the expected reference marker. This is **not a general accuracy score**: the marker occurs in the prefix,
and both precisions failed the independent terse arithmetic fixture. Basic tool
calls, streamed arguments, tool-result continuation, authentication rejection and
post-disconnect requests passed.

Provider-recorded total after deletion: **US$3.261**, below the US$5 cap.
Both instances and their local disks were deleted; paginated provider inventory
confirmed zero remaining experiment instances. Charges are a ledger snapshot,
not a finalized invoice. Nothing was deployed or published.

This compares Qwen3-Coder-30B-A3B-Instruct Q4_K_M on an RTX 4090 24 GB and
A100 SXM4 80 GB with BF16 on vLLM on the same A100. The original checkpoint
revision matches the GGUF source revision. Quantization, engine and (for the
4090 comparison) hardware differ; this is not an isolated engine comparison.

## Reproduction and evidence

See [protocol](qwen-coder-llamacpp-validation-plan.md) and
[evidence](evidence/qwen-coder-llamacpp-20261126/). Public source base:
`bbb5bd606326d70c5c57333ee380850ff10d05b9`; helpers were uncommitted at execution time,
with their hashes captured in provenance manifests. They are included alongside this report. The exact full-grid harness
is also saved as `benchmark-grid.py` in the evidence directory; the current
helper adds optional labeling and CUDA-graph execution for the separate subset. llama.cpp b12432,
commit `21fa9c043f0e4f574b36cd06cff4b4bce64ec76f`; vLLM 0.33.0.

Run `python3 tools/summarize_coder_backends.py research/evidence/qwen-coder-llamacpp-20261126`
to regenerate the per-cell CSV and JSON from raw trials. Each measured cell has
one warmup and up to three measured waves. Outputs are capped at 256 tokens.
Short prompts and near-full prompts are separate workloads. Actual input token
counts appear in the summary. Aggregate throughput includes prefill and decode;
per-request decode throughput excludes time to first content. TTFT includes queue
and prompt processing. These few observations do not establish tail-latency SLAs.

## Quality and limitations

The terse fixture `sum(x*x for x in [2, 3, 5])` should return 38. Q4 returned 10
in both streaming and non-streaming mode, while equivalent expanded Python,
natural-language arithmetic and direct addition returned 38. BF16/vLLM returns the same wrong answer in both modes. This fixture therefore
does not establish a quantization regression. The failed gate is retained. Performance measurements after that diagnosis are exploratory only.

The first RTX attempt silently fell back to CPU because CUDA runtime libraries
were not on the loader path. It is retained separately and excluded from GPU
results. Corrected runs explicitly require CUDA0 and full GPU offload.

The reference marker appears both in the request prefix and in the code bundle.
Reference matching therefore checks response consistency, **not long-context
retrieval accuracy**. Code-review prose is not scored for correctness. A future
quality study must hide references solely inside the context and run a broader
coding suite. These trials establish workload capacity and protocol observations,
not general coding accuracy.

Tool tests cover one automatic tool selection with exact JSON arguments and its
result continuation, in streaming and non-streaming modes. They do not cover
parallel tools or all SDK edge cases. Cancellation checks close a client stream
and check a subsequent request; they do not prove GPU work was reclaimed.

llama.cpp restarts with context × concurrency KV allocation for each cell and
uses default FP16 KV storage. vLLM uses a shared 32K/8-sequence engine; lower rows
are workload sizes, not separately configured engine limits. Prompt caching is
disabled. The initial vLLM baseline runs with `--enforce-eager` to bound startup work in this
experiment, with `VLLM_USE_FLASHINFER_SAMPLER=0` after missing CUDA build
dependencies were diagnosed; results do not represent a tuned CUDA-graph
production deployment. Cold request timing is after health readiness, separate from engine
load time and first checkpoint download. Fixed ordering and small sample counts
limit performance conclusions. No public Ygaraty adapter or Kaa production route
is exercised, and no production configuration is changed.

## RTX 4090 observations

Near-full prompts; three measured waves per cell. Throughput includes prompt processing.

| Context | Concurrent requests | Median TTFT (s) | Aggregate output tok/s |
|---:|---:|---:|---:|
| 4,096 | 1 | 0.49 | 127.5 |
| 4,096 | 2 | 0.86 | 163.5 |
| 4,096 | 4 | 1.66 | 186.9 |
| 4,096 | 8 | 2.93 | 212.5 |
| 16,384 | 1 | 2.56 | 57.9 |
| 16,384 | 2 | 4.32 | 63.3 |
| 16,384 | 4 | 7.86 | 66.0 |
| 32,768 | 1 | 6.43 | 28.9 |
| 32,768 | 2 | 10.70 | 29.9 |

16K × 8, 32K × 4 and 32K × 8 failed to allocate the FP16 KV cache.
No CPU fallback or context reduction was used to make these cells fit.

## Cost interpretation

Rental rates include the selected disk allocation: RTX 4090 $0.484444/h,
A100 $1.391781/h. Transfers are additional. A steady-state output cost estimate
is `hourly rental rate / (aggregate output tokens per second × 3600) × 1,000,000`.
This assumes continuously occupied capacity with the measured prompt/output mix;
it excludes idle time, startup, transfers and customer pricing multipliers. It is
not an API token price or a forecast for arbitrary workloads.

Memory samples are observed values, not allocator guarantees. The vLLM retry
monitor began during the second 4K cell; earlier vLLM cells have end-of-cell
GPU snapshots only. Restart load times use already-downloaded weights and may
benefit from the filesystem cache; they are not total deployment startup times.

## Baseline setup failures

Three failed vLLM startups are retained separately: missing C compiler, missing
Python headers, and missing CUDA compiler for FlashInfer sampler warmup. The
partial precompiled-kernel download was stopped after its transfer rate proved
unhelpful. The successful baseline uses system build tools/Python headers and
the supported native sampler setting. These failures are environment setup
failures, not model quality or GPU capacity failures.

The eager vLLM engine reports 14.18 GiB of KV cache, or 154,832 tokens
(about 4.73 full 32K sequences). Eight submitted 32K requests therefore exercise
scheduling/queueing and potentially preemption, not eight fully resident 32K
sequences. Successful HTTP completion must not be interpreted as a guarantee
that all eight contexts fit simultaneously. llama.cpp reserves its per-slot
KV allocations up front, so its startup OOM behavior differs.

## Same A100: Q4 llama.cpp versus eager BF16 vLLM

Near-full prompts, 256-token output cap. TTFT is median seconds. Throughput
includes prefill and decode. All cells have three measured waves except llama.cpp
32K × 8 (one wave before its benchmark deadline).

| Context | Requests | Q4 TTFT | BF16 TTFT | Q4 output tok/s | BF16 output tok/s |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 1 | 0.92 | 0.26 | 95.0 | 20.1 |
| 4,096 | 2 | 1.54 | 0.37 | 126.9 | 39.0 |
| 4,096 | 4 | 2.93 | 0.63 | 141.8 | 75.0 |
| 4,096 | 8 | 5.05 | 0.98 | 159.9 | 137.4 |
| 16,384 | 1 | 4.67 | 1.40 | 37.7 | 18.4 |
| 16,384 | 2 | 7.61 | 2.11 | 42.1 | 32.8 |
| 16,384 | 4 | 13.61 | 3.56 | 43.4 | 55.4 |
| 16,384 | 8 | 37.35 | 6.42 | 35.3 | 81.3 |
| 32,768 | 1 | 10.95 | 3.91 | 19.1 | 15.5 |
| 32,768 | 2 | 17.78 | 5.98 | 19.9 | 24.6 |
| 32,768 | 4 | 37.30 | 10.30 | 18.1 | 34.4 |
| 32,768 | 8 | 105.61 | 24.33 | 13.9 | 35.2 |

## Graph-enabled BF16 comparison (preferred over eager-only conclusions)

Same A100, native sampler, prefix caching disabled, default graph-enabled
compilation. No hand-tuned MoE configuration; the runtime warns that its default
MoE configuration may be suboptimal. Each graph cell has three measured waves.
4K × 2/4 were not repeated with graphs; use the separate eager rows for those.

| Context | Requests | Q4 TTFT (s) | BF16 graph TTFT (s) | Q4 output tok/s | BF16 graph output tok/s |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 1 | 0.92 | 0.23 | 95.0 | 115.0 |
| 4,096 | 8 | 5.05 | 0.95 | 159.9 | 366.0 |
| 16,384 | 1 | 4.67 | 1.38 | 37.7 | 70.4 |
| 16,384 | 2 | 7.61 | 2.08 | 42.1 | 89.4 |
| 16,384 | 4 | 13.61 | 3.51 | 43.4 | 104.1 |
| 16,384 | 8 | 37.35 | 6.36 | 35.3 | 120.6 |
| 32,768 | 1 | 10.95 | 3.90 | 19.1 | 40.0 |
| 32,768 | 2 | 17.78 | 5.97 | 19.9 | 44.7 |
| 32,768 | 4 | 37.30 | 10.24 | 18.1 | 46.9 |
| 32,768 | 8 | 105.61 | 21.18 | 13.9 | 46.5 |

Q4 32K × 8 has only one measured wave, so that row has weaker evidence.
Graph-enabled 32K × 8 also queues requests; it does not create cache space for
all eight full contexts simultaneously. These short campaigns do not establish
sustained production reliability or a quality equivalence between BF16 and Q4.

## Cost and cleanup

| Host | Recorded GPU + disk + transfer cost |
|---|---:|
| RTX 4090 | US$0.244 |
| A100 SXM4 80 GB | US$3.017 |
| Total | **US$3.261** |

See `costs.json` and `cleanup.json` in the evidence directory. No independent
persistent volumes were allocated. The experiment's host cleanup timers are
inactive after verified deletion. Provider billing can post delayed adjustments.

## Next step

If integrating llama.cpp, begin with an experimental Q4 backend targeting the
measured RTX 4090 configurations, preserve session authentication/concurrency
and cancellation behavior, and test through Kaa's actual route. Before releasing
profiles, run a broader coding/tool suite and hidden-reference long-context
retrieval tests. Do not infer support for untested concurrency or 64K/128K from
aggregate KV memory alone. No production profile was registered by this work.

## Final audit: reference-copy failure and harness correction

At 4K × 8, short prompts, measured wave 2, BF16 returned `0255ba218e2`
instead of `0255ba218e9618e2`. This happened once in eager mode and once with
CUDA graphs. Both failed rows remain in the summary (`all_waves_passed=false`).
Near-full-context rows passed the reference check, but that does not erase the
short-prompt failure or establish model accuracy.

The original harness stopped the failed cell but continued the wider sweep;
this was discovered during final evidence auditing, after the machines were
removed. Earlier progress updates saying no additional failures were incomplete.
All measurements remain exploratory. The current helper now stops the entire
campaign on an unexpected wave failure. The exact executed versions remain
saved as `benchmark-grid.py` and `benchmark-graphs.py` in the evidence directory;
this fix was tested locally and was not rerun on a GPU.
