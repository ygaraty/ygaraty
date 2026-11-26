# Qwen3-Coder: llama.cpp Q4_K_M versus vLLM

Status: completed on 2026-11-26. See [results](qwen-coder-llamacpp-results.md). Provider-recorded cost US$3.261; both rentals deleted and inventory verified.

## Question

Can Qwen3-Coder-30B-A3B-Instruct Q4_K_M offer useful dedicated coding inference
on one RTX 4090 24 GB, with correct tool calls and streaming? Compare customer
cost/performance with fresh vLLM BF16 trials, and run llama.cpp on the baseline
machine as well to distinguish hardware effects. Quantization remains a confound;
these results must not be described as an isolated engine comparison.

## Approved budget

US$5 total, including setup, model downloads, storage, transfer and cleanup.
One RTX 4090 for at most 90 minutes, and one A100 80 GB for at most 120 minutes.
Recheck all-in offers before renting; target ceilings $0.60/h and $1.50/h.
Those ceilings reserve $3.90 for machine time and $1.10 for ancillary charges.
Shorten or skip runs when provider charges would exceed the remaining cap.
Read-only Vast quotes: RTX 4090 $0.4077/h; A100 SXM4 $1.2000/h;
A100 PCIe $1.4430/h. Availability and exact machine suitability are not guaranteed.
No H200 planned (current minimum quote $5.0877/h).

## Reproduction requirements

Use public experiment helpers; do not import the private cloud application.
Record the exact public source revision, candidate patch SHA256, llama.cpp commit,
vLLM version/container digest, GGUF revision/SHA256 and original model revision.
Verify model source lineage before comparing quality. Historical baseline checkpoint:
Qwen/Qwen3-Coder-30B-A3B-Instruct@b2cff646eb4bb1d68355c01b18ae02e7cf42d120.
GGUF candidate: ggml-org/Qwen3-Coder-30B-A3B-Instruct-GGUF, Q4_K_M (about 18 GB).
Pin resolved full revisions and dependency versions before provisioning.
Capture driver, CUDA, device memory, RAM, disk, engine arguments, chat template,
KV cache precision, offload configuration and prefix-cache policy.
This is direct-runtime exploratory evidence, not validation of a deployed Kaa
llama.cpp backend. The existing public runtime has no assumed llama.cpp adapter.

## Trial matrix and gates

- Context per request: 4096, 16384, 32768 tokens, including generation allowance.
- Concurrent requests: 1, 2, 4, 8; record actual per-slot context, not only a global
  cache allocation. Explicitly report unfit combinations instead of reducing context.
- Independent realistic code/document prompts, deterministic seed and generation
  settings, 256-token output allowance. Record actual tokenizer counts. Use both
  short prompts and near-full-context prompts; never label a short-prompt capacity
  setting as a long-context performance measurement.
- One cold sample, one representative warmup and three measured waves per cell.
  Report individual observations; small samples do not establish production p95.
- Fixed order: ascending context and concurrency. Record cache policy; prevent
  repeated-prefix reuse from silently making comparisons easier.
- Streaming: valid SSE, text reconstruction, terminal marker, finish reason, usage
  where available, and no dropped/duplicated chunks.
- Tools: required tool selection, JSON argument/schema correctness, streamed tool
  argument assembly and tool-result continuation; deterministic fixture expectations.
- Coding/retrieval: bounded checked fixtures with exact answers or unit tests.
  Do not execute model-generated code outside an isolated sandbox.
- Cancellation: disconnect a request and verify the next request succeeds; record
  GPU reclamation separately, without inferring it from client disconnection.
- Record TTFT, total latency, output tokens/s per request, aggregate tokens/s,
  peak VRAM, ready/load time, errors, provider costs and estimated cost/token.
- Fail a configuration on an OOM, malformed protocol/tool response or wrong checked
  answer. Retain partial successes and all failures. Do not automatically publish
  new production profiles.

## Safety, evidence and teardown

Prebuild/package the harness and run offline tests before renting. Install an
independent watchdog before unattended work; count setup inside rental deadlines.
Stop repeated setup failures unless a new diagnosis justifies another attempt.
Export sanitized raw JSONL, summarized CSV, configuration manifests and report
under research/evidence/qwen-coder-llamacpp-20261126/. Keep provider identifiers,
credentials and machine addresses in a private operational record only.
Delete this experiment's machines and billable storage; verify inventory, then
record actual billed cost when available, otherwise label estimates explicitly.

## Execution refinements before trials

The pinned GGUF's `.src_sha` matches the baseline checkpoint exactly. Use
llama.cpp b12432 / 21fa9c043f0e4f574b36cd06cff4b4bce64ec76f, with release archive
hashes verified before execution. GPU layers are fully offloaded; no automatic
CPU fallback is accepted. llama.cpp allocates context × parallel slots per cell.
vLLM uses a single 32768-token / 8-sequence engine and its paged cache; smaller
rows describe the request context workload, not separately capped engines.
Use one warmup wave and three measured waves for both short and near-full prompts.
A checked fixture failure stops that engine's performance campaign for diagnosis.
The first GPU offer disappeared before creation; no allocation occurred on that
attempt. Retry is capped at 80 minutes and retains an earlier independent host
cleanup timer. Transfer charges are included in offer selection, not just rent.

### Amendment: reject CPU fallback, disable prefix caching

The first RTX smoke run silently used CPU because the CUDA runtime archive
extracts into a separate directory. Its GPU samples stayed at 1 MiB / 0% and
`--list-devices` was empty. Preserve that trial as invalid CPU-fallback evidence,
including failed arithmetic. Add the CUDA runtime library directory, require
`--device CUDA0`, disable automatic fitting and prompt caching explicitly.
The first attempt had caching enabled despite cache-reuse=0 (which only controls
KV shifting). No GPU performance result is claimed from that attempt.

### Amendment: diagnose and retain prompt-sensitive quality failure

CUDA/no-cache repeats return 10 for the terse generator-expression fixture,
but 38 for direct addition, natural-language sum of squares and expanded Python
using `value ** 2`. Step-by-step explanation also identifies the correct squared
terms. Exact echo, tool JSON and tool-result follow-up pass. Continue exploratory
performance measurement only for this exact diagnosed failure (10 in both modes),
while keeping the correctness gate failed in raw evidence and the release verdict.
Any other smoke failure still stops the engine. Compare the unchanged failing
fixture with BF16. This is not permission to claim validated model quality.

### Amendment: vLLM host compiler dependency

The first vLLM startup failed because Triton could not find a C compiler in
the minimal Ubuntu image. The CUDA numerical probe passed. Preserve the failed
startup, install build-essential, and retry the unchanged engine configuration
on the same A100. Include the missing system dependency in the setup helper.

The compiler retry exposed a second missing build dependency: `Python.h`.
Preserve that attempt, install python3-dev, and retry with the same model and
engine settings. Both system dependencies are now explicit in the helper.

The next startup reached FlashInfer sampler warmup and required nvcc. Install
the runtime-supported precompiled kernels with `vllm download-kernels`
(FlashInfer 0.7.2.post1, CUDA 13.0) instead of compiling on the rental. Preserve
the missing-kernel attempt and record the resulting package freeze.

The 1.67 GB precompiled wheel downloaded only about 128 MB in 147 seconds.
Stop that download and use vLLM’s supported `VLLM_USE_FLASHINFER_SAMPLER=0`
native sampler setting for this greedy benchmark. Record the environment and
retain the partial-download log. This baseline is explicitly not a tuned default
production deployment. No new machine is allocated.

### Optional matched CUDA-graph subset

If at least ten minutes remain after the eager baseline and projected total
spend remains below US$5, run vLLM without `--enforce-eager` for 4K × 1 and
4K × 8 only, in a separate evidence directory. Keep all other settings and
fixtures identical. This checks whether eager execution materially penalizes
the baseline; it does not replace or mix with the original full grid. Do not
extend the rental deadline.

### Extend graph-enabled comparison after material result

The graph-enabled 4K × 8 subset reaches about 367 output tokens/s versus
137 eager, materially changing the comparison. With 16 minutes remaining
on the existing rental and spending below the cap, run the 16K/32K grids
with CUDA graphs under a separate eight-minute benchmark budget. Preserve
the first graph startup log, append trials to the same labeled graph dataset,
and leave the existing rental teardown deadline unchanged.

The eight-minute graph campaign completed through 32K × 4 and stopped before
32K × 8 at its safety margin. With about eight minutes remaining on the rental,
run only that final cell under a 330-second benchmark / 360-second process
limit, preserving prior logs. No rental deadline extension.

### Final audit correction

Two BF16 short-prompt reference-copy failures were found in raw trials after
cleanup. The original harness stopped only the affected cell; it should have
stopped the campaign. Preserve these failures and subsequent exploratory data,
correct the report, and make the current helper stop the campaign on unexpected
wave failures. No additional GPU was rented to rerun the corrected harness.
