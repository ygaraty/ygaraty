# Per operation on one A100, Kolibri-1's GPU time is Q6_K matvecs 29%, attention 24%, norms 13%; variant timings confounded by context growth

| | |
|---|---|
| **Conclusion** | Found, unsteady: per op, Q6_K matvecs 29%, attention 24%, norms 13% of GPU time; variant timings confounded (measured) |
| **Status** | executed |
| **ID** | DO-64 (experiment 115) |
| **Category** | browser |
| **Where** | remote |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | The DO-63 machine (Vast.ai A100 SXM4 80 GB, North Carolina) after DO-63's four runs; Kolibri-1 Q4_K, one stage of 50 layers loaded once into one Chrome page (`qwen35_webgpu bench`), merged expert chunks; decode variants in turn without a reference |
| **Result** | Found, but the run is **unsteady** by the frozen check: base 76.9 against base-again 96.8 ms per step (26% apart). Cause: every variant decoded in the same conversation, so later variants ran at longer contexts (the full-attention layers' cost grows with positions); the matvec and batch comparisons are confounded and not used. The detail breakdown (at about 40 positions, GPU 43.3 ms per token with a pass per dispatch) stands: Q6_K matrix-vector products 12.5 ms (29%), windowed attention 10.5 ms (24%; 0.213 ms per layer), RMS norms 5.6 ms (13%; 6 per layer at 0.014–0.020 ms each: attention norm, QK-norm ×2, both sandwich norms, MLP norm), routed experts 4.5 ms (10%), router matmul 2.3, cache stores 1.4, routing 1.3, residual adds 1.3, RoPE 1.1 ms; outside the layers (embedding, head) 1.0 ms. |
| **Evidence** | `runs/pods-data/do64/` (bench.log, report.json); files hand-copied with hashes 80a5d7a9…, 4f656acb…, dec13b60… (in the report's machine log); curated: `public/research/evidence/browser/kolibri-a100-decode-speed-2026-11-12.json`; code committed as 430c583 |

## Question

Where does a decode token's time go on one A100 (JavaScript encoding, GPU work, the wait between submit and the token's read-back), which operations take the GPU time, and which levers move tokens per second? The user's interest: levers that also apply to workers that hold only layers (no embedding or head), such as the small per-layer operations: QK-norm (Kolibri: per-head RMSNorm of Q and K), sandwich norms (norms after attention and after the MLP, each with a residual add), and attention gating (Qwen3.5's output gate).

## Prediction

The GPU's own time per token is a fraction of the wall time; per layer, the small operations (norms, adds, RoPE, cache stores, router) take a share of GPU time out of proportion to their bytes, so fusing them into neighbouring kernels is worth it; batching conversations raises tokens per second several times (a step reads the same weights for every conversation).

## Gate (frozen 2026-11-12, before running)

Diagnostic: the verdict is **Found**.

- One `bench` run (`qwen35_webgpu bench`, BENCH_VARIANTS in order): base (24 tokens), detail (6 tokens, a timestamped pass per dispatch), matvec rows per thread 1, 2, 4 (16 tokens each), batch 2, 4, 8 conversations (12 steps each), base again (24 tokens). The first step of each variant is dropped (kernel compilation).
- Report per variant: wall ms per step, tokens per second, JavaScript encode ms, GPU ms (timestamps), dispatches. From detail: GPU ms by operation, and per layer shape (sliding, full) each position's median GPU ms, so QK-norm, sandwich norms, residual adds, RoPE, cache stores, attention, projections, router and experts are each costed; and the time outside layers (embedding, head).
- Valid if base and base-again medians are within 10% of each other (the machine steady); otherwise reported as unsteady.
- No correctness claim (no reference); DO-63 and experiment 113 cover numerics. The variants that change arithmetic (matvec rows) were shown bitwise identical in experiment 104 on other models; here only timing is read.

## Protocol

- Code: `main` at 6e60083 on the machine plus hand-copied `webgpu_kernels.mjs` (detail profiling), `qwen35_parity.mjs` (bench mode) and `qwen35_webgpu.py` (bench), copied only after DO-63's runs end; hashes recorded. Checked on the workstation first on a tiny Kolibri (vocabulary 128).
- Budget: within DO-63's rental and cap (user, 2026-11-12: "use the download to run more tests on finding raw performance with 1 worker"); about 20–30 min more of the machine.

## Log

- 2026-11-12: Added at user request ("use the download to run more tests on finding raw performance with 1 worker. find ideas that would also apply for workers containing only layers", "such as QK-Norm, attention gating, sandwich norm"). Bench and detail profiling built; checked on the workstation (Radeon 680M, tiny Kolibri, vocabulary 128): all variants run; base 5.4–6.6 ms per step with 0.5 ms encode and 1.1 ms GPU; batch 8 at about 5× the tokens per second of batch 1.
- 2026-11-12 03:30–03:35 CET: bench run (`bench.sh`, exit 0). Results as in Result. Lesson: decode variants must each start from the same position (re-prefill a fresh slot per variant), or position growth confounds them. For layer-only workers: per layer, the five small norm/add dispatches (QK-norm ×2, two sandwich norms, the attention norm, two adds) cost about 0.12 ms of GPU per layer, more than the routed experts (0.09 ms); the attention kernel (0.21 ms per layer at a short context) is the single largest per-layer operation.

- 2026-11-12 05:06 CET: machine deleted after DO-66; `pods ls` 0. The shared rental (DO-63 to DO-66) cost $4.01 (cap $6).
- 2026-11-12: written up as experiment 115.

## Limits

- **Unsteady** by the frozen check (base 76.9 vs base-again 96.8 ms): every variant decoded in the same conversation, so later ones ran at longer contexts. The matvec rows-per-thread and batch comparisons are not used.
- The detail pass (one timestamped pass per dispatch) inflates GPU time (43.3 against 27.9 ms per token); shares, not absolute times, are the finding.
- One run, about 40 positions; attention's share will grow with context.
- No correctness check (no reference).

## Reproduce

```bash
# on the DO-63 machine (or any rented 80 GB GPU with the GGUF), code at 430c583 or later
bash runs/pods-data/do64/bench.sh   # qwen35_webgpu bench, BENCH_VARIANTS in order
```
A rerun should re-prefill a fresh slot per variant (the bench does not yet).
