# Phi-4 runs on Ygaraty locally: kernels, KV cache and the two-worker loop give the reference's tokens

| | |
|---|---|
| **Conclusion** | **Works:** our Phi-4 path gives transformers' exact greedy tokens with one and two workers (numerical, tiny same-architecture model). |
| **Status** | executed |
| **ID** | WS-9 (experiment 79) |
| **Category** | correctness |
| **Where** | workstation (omarchy) |
| **Date** | 2026-11-04 |
| **Path** | [`inference.md`](../../inference.md) stages 1 and 2, on a tiny random `Phi3ForCausalLM` (Phi-4's architecture: grouped-query attention 8 over 2, fused q/k/v and gate/up projections, rotary θ 250,000, untied head), built locally, no weights downloaded. NumPy and PyTorch kernels, a safetensors reader, owner state with per-session KV caches, and greedy generation as a Ygaraty job through its coordinator, worker processes and relay |
| **Result** | **Logits:** within 1.7e-6 of transformers (logits up to 2.6), and cached decode at an offset within 1.2e-6; tolerance rtol 1e-4, atol 1e-5. **Cache:** chunked prefill (1, 3 and 7 tokens) plus decode equals the uncached forward within 1e-10 in float64. **Generation:** 16 tokens from 3 prompts are identical across transformers' `generate`, our kernels, and the Ygaraty loop: 1 worker, 2 workers, prefill in chunks, the torch backend over 3 sessions, and externally started workers. Every token is printed exactly once on the last owner, with 0 retries. Retried steps, gaps, wrong checkpoints and workers without the capability all fail clearly. Free |
| **Evidence** | [`evidence/correctness/ws9-phi4-local-2026-11-04.json`](evidence/correctness/ws9-phi4-local-2026-11-04.json); transport runs in `runs/experiments/WS-9/omarchy-20261104-local/` (local, not in git). Code: `ygaraty/inference/phi4_kernels.py`, `phi4_torch.py`, `phi4_checkpoint.py`, `phi4_owner.py`, `phi4_job.py`, `phi4_run.py`, the stage plan in `phi4.py`, and worker dispatch and capability in `transports/http/worker*.py` and `webapp/worker_routes.py`. Tests: `tests/test_phi4_kernels.py`, `test_phi4_checkpoint.py`, `test_phi4_loop.py` (21 with `test_phi4_plan_offline.py`) |

## Question

The GPU trial (`inference.md` stage 3) needs everything except the GPUs ready and tested beforehand: kernels, checkpoint loading, a KV cache, owner-placed stage tasks, and the autoregressive loop through Ygaraty's own execution path. All of that can be checked on a tiny model with the same architecture.

**Do our Phi-4 kernels match transformers' implementation, does cached prefill and decode match the uncached forward, and does the two-worker loop through Ygaraty's workers emit the same tokens as one worker and as the reference?**

## Prediction and gate (frozen before any code)

**Prediction:** all pass. The likely problems are rotary positions at nonzero offsets and the grouped-query head mapping.

**Gate:**
- **Reference parity:** on a tiny random `Phi3ForCausalLM` in float32, one block's output and the full model's logits match within rtol 1e-4, atol 1e-5. transformers runs its RMSNorm and rotary tables in float32 internally.
- **Cache parity:** in float64, chunked prefill of 1, 3 and 7 tokens followed by single-token decode matches the uncached full-prefix forward within rtol 1e-10, atol 1e-12, at nonzero starting positions.
- **Generation:** greedy generation of 16 tokens from 3 prompts gives the same token IDs in transformers' `generate`, in our single-process kernels, and in the Ygaraty loop with one worker and with two.
- **Robustness:** these must fail with a clear error: a retried stage task appending KV twice; out-of-order or gapped positions; an unknown session; a checkpoint whose config differs from the plan; an unsupported worker. The plan contains no backward or optimizer tasks.
- Timing is not part of this gate.

## Protocol

- **Kernels** (`phi4_kernels.py`, NumPy):
  - RMSNorm, and transformers' default rotary embedding over absolute positions.
  - Grouped-query causal attention in which query head h reads key/value head h // (H/KV), over a `LayerCache`.
  - SwiGLU with fused gate/up, the block, the embedding, the final norm and head, and greedy selection.

  `phi4_torch.py` has the same arithmetic in PyTorch on any device and dtype, for GPU owners.
- **KV cache:** an append at the cache's end, or a retry of exactly the last step (which replaces it). Anything else raises an error, so a retried task never appends twice.
- **Checkpoint reader** (`phi4_checkpoint.py`):
  - Reads a Hugging Face directory (`config.json` plus one safetensors file or shards) tensor by tensor, without torch, from F32, F16 or BF16 (BF16 through bit shifts).
  - Checks the config and every tensor's shape against the plan, and rejects sliding windows, rope scaling, tied embeddings and other architectures.
  - The manifest ID hashes `config.json` and each shard's header and size.
- **Plan** (`compile_phi4_stages`): one step is one stage task per owner. The first owner embeds the token IDs and runs its layers; the last owner runs its layers, the final norm, the head and greedy selection. The first step also loads each owner's layers once.
- **Job** (`Phi4Job`, a `Job` subclass):
  - Each step is one prefill chunk or one decode token; the coordinator commits the token, which becomes the next step's input.
  - In the following step, the last owner prints the committed token, at most once per (session, position).
  - Generation stops at EOS or the length limit; a final step prints the last token and drops every owner's caches.
  - Stage tasks go only to their named worker, and only after its hello announced the `phi4` capability.
  - Several sessions can run in one job, so only the first loads weights.
- **Driver** (`python -m ygaraty.inference.phi4_run`):
  - Starts the coordinator and worker processes, optionally with the relay, handoffs, a float32 wire and emulated links.
  - Or, with `--external`, serves on a host and port and waits for workers started on other machines; the job token goes to a mode-600 file.
  - Takes token-ID prompts or seeded synthetic prompts of fixed length (`--prompt-length`, for the measurement protocol's 128, 1,024 and 3,968 tokens), and `--repeats` sessions.

## Log

- 2026-11-04: Gate frozen and started with the user's go, on cbdc0b1.
- 2026-11-04: Kernels, and their parity with transformers and between cached and uncached paths, passed on the first run. Checked that the comparisons are real: the logit gap is 1.7e-6 on logits up to 2.6, and the generated sequences vary.
- 2026-11-04: Checkpoint reader tested against transformers' own `save_pretrained`: float32 single file, and bfloat16 in 7 shards, read bit-exactly.
- 2026-11-04: The first loop run hung. With eager claims and no relay, a worker's results wait for an uploader that only exists with a relay. The driver now enables eager claims and pushed inputs only with a relay. The first relay run hung too: the job's policy didn't carry the relay URL, so the coordinator couldn't read uploaded blocks. Fixed in the driver.
- 2026-11-04: All variants run with identical tokens. The full suite has 4 failures, none in this code: 2 in `test_http_compression`, which predate this session, and 2 DiLoCo errors (`'DiLoCoReplica' object has no attribute 'backend_factory'`), which reproduce on unmodified cbdc0b1 (another session's DiLoCo commit).

## Results

**Parity with transformers** (float32 reference, our kernels in float64):

| Check | Max abs difference | Gate |
|---|---|---|
| Block 1 output | within tolerance | rtol 1e-4, atol 1e-5: pass |
| Full-model logits (12 tokens) | 1.7e-6 (logits up to 2.6) | pass |
| Decode at position 9 after a 9-token prefill, against transformers' own cache | 1.2e-6 | pass |

**Cache parity** (float64): prefill in chunks of 1, 3 and 7 tokens, then decode to position 22, matches the uncached forward within rtol 1e-10, atol 1e-12. A retried step replaces itself and leaves the cache length unchanged. **Pass.**

**Generation:** 16 greedy tokens from each of 3 prompts are identical in every setting:

| Setting | Same tokens |
|---|---|
| transformers `generate` | (reference) |
| Our kernels in one process | yes |
| Ygaraty loop, 1 worker | yes |
| Ygaraty loop, 2 owners plus 1 spare worker (which ran nothing) | yes |
| 2 owners, prefill in chunks of 2 | yes |
| 2 owners, torch backend (CPU), 3 sessions in one job | yes |
| 2 workers started externally | yes |
| EOS at the 4th token | stops there |

Each token was printed once, on the last owner only. **Pass.**

**Robustness:** each of these fails with a clear error:
- an append that skips positions or rewrites an older step;
- an unknown session at a nonzero position;
- non-integer or out-of-vocabulary token IDs;
- a chunk past the context;
- a load from a checkpoint with another manifest or config;
- a stage on a worker that didn't load it;
- a NumPy backend asked for bfloat16;
- an owner without the `phi4` capability, which fails the job; a capable non-owner is admitted but given nothing.

Retried load, stage and print tasks are idempotent. The plan contains only `phi4_*` operations. **Pass.**

**Transport** (2 owners, 6 tokens, through the relay; not part of the gate, and the model is tiny, so these are protocol overheads):

| Setting | First token (s, includes loading) | Between tokens (s) |
|---|---|---|
| Loopback, handoffs | 0.44 | 0.004–0.005 |
| Loopback, handoffs, float32 wire | 0.44 | 0.004–0.005 |
| Emulated 50 ms, 100/20 Mb/s | 1.58 | 0.214–0.215 |
| Emulated, handoffs | 1.58 | 0.214–0.215 |

In the bfloat16 torch run, the first session's time to first token included the weight load (1.1 s); the warm sessions' first tokens took about 5 ms.

## What it taught

- **The architecture is implemented correctly where errors usually hide.** Rotary at offsets, the grouped-query mapping, and the fused projections' split order all agree with transformers. So does decoding against transformers' own cache.
- **Each token costs about 4 round trips on home links, and handoffs don't change that.** The token goes from B to the coordinator, which commits it and only then publishes the next step for A. At 50 ms that is 214 ms per token before any compute. The obvious next step is to send the token straight from B to A (a handoff of the next step's input), which would leave about 2 round trips. It needs its own commit rule.
- **Two worker-path traps:**
  - Eager claims without a relay hang silently: results wait for an uploader that doesn't exist.
  - A job's `RuntimePolicy` must carry the relay URL, or the coordinator can't read uploaded blocks.

  Both are avoided in the Phi-4 driver; the first is still a trap for other drivers.

## Limits

- **No real Phi-4 weights.** The reader checks Phi-4's config mapping against the published values, but not its real tensor names, shard layout or bf16 values. Validating real layers needs the checkpoint (about 29 GB in bf16, or one 5 GB shard for a few layers), which this run did not download.
- **No tokenizer.** Prompts and outputs are token IDs; text streaming needs Phi-4's tokenizer files.
- **No quantization.** A Q4 loader waits for a pinned format, which depends on the GPUs chosen. The torch backend runs bf16 or float16, so unquantized Phi-4 (about 29 GB) needs two GPUs of at least about 16 GB each.
- **Speed is not established.** The NumPy and torch-CPU kernels are for correctness. Torch on CUDA is untested here: no GPU.
- **Not built:** owner loss mid-session (deferred, as in `inference.md`), migration, and batching.

## Reproduce

```bash
.venv/bin/python -m unittest tests.test_phi4_kernels tests.test_phi4_checkpoint tests.test_phi4_loop tests.test_phi4_plan_offline
# A tiny checkpoint and one run (relay and emulated links optional):
.venv/bin/python -c "import sys; sys.path.insert(0,'.'); from tests.test_phi4_kernels import reference_model; reference_model().save_pretrained('runs/tiny-phi4')"
.venv/bin/python -m ygaraty.inference.phi4_run runs/tiny-phi4 --prompt 1 5 17 --max-new 6 --owners 2 --dtype float64 \
  --relay-url http://127.0.0.1:PORT --relay-token TOKEN --handoff --rtt-ms 50 --down-mbps 100 --up-mbps 20
# For the GPU trial: --external --host 0.0.0.0 --port P on the coordinator, --backend torch --dtype bfloat16,
# YGARATY_PHI4_CHECKPOINT=<local dir> on each worker.
```
