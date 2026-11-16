# Shapes picked by a single-matrix sweep make A100 decode 18.5% slower: the microbenchmark measured L2-cached reads

| | |
|---|---|
| **Conclusion** | Rejected: A100-picked shapes make decode 18.5% slower (19.95 → 23.65 ms), tokens identical (measured, A100) |
| **Status** | executed |
| **ID** | DO-74 (experiment 134) |
| **Category** | browser |
| **Where** | remote |
| **Priority** | high |
| **Date** | Added and run 2026-11-13 |
| **Path** | One rented A100 (SXM4 preferred), headless Chrome, Qwen3.5-9B Q4_K_M. (1) The `kernels` microbenchmark (`--cases decode1`: ffn_up, ffn_gate, ffn_down, attn_qkv, attn_q, the head) times all 57 coalesced shapes (threads per row 4–64, threads per item 1–8, workgroups of 64, 128 or 256) in one session, 3 rounds in alternating order, each output checked against NumPy; run twice (S1, S2). (2) The fastest shape per K-quant format from S1 becomes arm T; the bench (`--resident`, 64 and 1,024 positions, production kernel flags) compares T with today's 16,4,64 (arm B), order B T T B |
| **Result** | Sweeps: all 57 shapes correct (worst error 9.7e-7 of the largest value); on single matrices the best shapes beat today's 16,4,64 by 0–17% (Q4_K and Q5_K: 4 threads per row; Q6_K: 16,4,64 stays best), and the picks moved between S1 and S2. Bench, resident step, median of two runs per arm: 64 positions 19.95 (B) against 23.65 ms (T), **+18.5%**; 1,024 positions 22.75 against 26.25 ms, +15.4%; tokens identical in every run. Gate 1 passes, gate 2: T slower, **Rejected**; 16,4,64 stays. Today's defaults give 50 tokens per second on this A100. About $0.15 (one A100 SXM4 40 GB, $0.589/h, 15 minutes). |
| **Evidence** | measured-remote; curated `public/research/evidence/browser/matvec-retune-a100-2026-11-13.json`; raw `runs/pods-data/do74/out/` (A100) and `runs/experiments/DO-74/omarchy-20261113-1056-screen/` (680M); code 07fec07; drivers `runs/pods-data/do74/do74.sh`, `do74b.sh`, `pick.py` |

## Question

On the A100 (experiments 128 and 129, 64 positions, detail profile), the K-quant matrix-vector products take 18.8 of 28.0 profiled GPU ms per token and read at 233 (Q5_K), 257 (Q4_K) and 408 GB/s (Q6_K), against Dawn's 727 GB/s copy roof. Their shape (16 threads per row, 4 per item, 64 per workgroup) was tuned on the Radeon 680M (experiment 121). Does a shape tuned on the A100 read near the roof and shorten the step?

## Prediction

The best shapes reach 450–600 GB/s on the large matrices; the resident step at 64 positions drops from about 22 to 15–18 ms (55–65 tokens per second). If no shape passes about 450 GB/s, the kernel's structure (loads per thread, the reduction through workgroup memory) is the limit, not its shape.

## Gate (frozen 2026-11-13, before renting)

1. **Correctness:** every swept shape's output within 1e-5 of the largest reference value on every case; arm T's generated tokens identical to arm B's at 64 and 1,024 positions in every run.
2. **Decision:** step = the `resident` variant's wall ms per token at 64 positions, median of each arm's two runs. **Worth it, now default for this adapter class** if T's step is at least 10% shorter than B's (the per-adapter selection is then implemented as a follow-up; the 680M keeps 16,4,64); **Works, small gain** if 3–10% shorter; **Rejected** if under 3%. Fails gate 1: **Failed gate**.
3. **Found (reported, no threshold):** per format, the best shape's GB/s and its share of the copy probe; S1 against S2 (whether the pick is stable); the tuned arm's detail profile (what remains outside the matrix-vector products, including Q8_0 with experiment 132's fast path, unmeasured on the A100 so far).

## Protocol

- Code: `--matvec-c TPR,L[,WG][;KIND=TPR,L[,WG]...]` (per-format shapes; the workgroup size changes only scheduling, never a sum) and `kernels --sweep-matvec-c all --rounds 3`.
- Pod: `contributor_fleet up do74 --gpus 1 --gpu "A100 SXM4" --max-hourly 1.6 --minutes 60 --min-ram 64 --disk 100 --location US,CA`, setup, then `runs/pods-data/do74/do74.sh` (fetch with 3 tries, S1, pick, S2, benches B T T B, every bench with `--matvec 2,4,8 --q8-fast-up-to 8 --resident --contexts 64,1024`).
- Pick: `runs/pods-data/do74/pick.py S1/report.json` = per format, the shape with the least summed time over that format's cases; other formats keep 16,4.
- Cost: about 40 minutes (setup 15, fetch 6, sweeps 2, benches 12, slack) at $0.6–1.2/h, about $0.40–0.80; cap and teardown asked before renting.

## Log

- 2026-11-13: Added at user request ("yes", after experiment 133's report proposed the retune as next priority), from the A100 detail profiles of experiments 128 and 129.
- 2026-11-13: Code and 680M screen (`runs/experiments/DO-74/omarchy-20261113-1056-screen/`): six single shapes and the 57-shape sweep run, all outputs within 1.3e-6 of the reference; workgroups 64 and 256 give bit-identical outputs; a Q4_K-only override changes only Q4_K's outputs. On the 680M, 16,4,64 stays the fastest on 4 of 6 cases (the pick: Q4_K 32,4,64, others 16,4,64); the sweep takes 32 s.
- 2026-11-13: 680M rehearsal of the bench arms (64 positions, B then T with the pick): both exit 0, all 32 tokens identical across B and T; resident step 90.3 (B) against 91.1 ms (T), as expected on the GPU the current shape was tuned for. `runs/pods-data/do74/do74.sh` and `pick.py` written (not yet run remotely).
- 2026-11-13: User: "yes, commit, push and do A100 test": approved one A100, $2 cap, 60-minute teardown.
- 2026-11-13: started: code 07fec07; `contributor_fleet up do74` (A100 SXM4, at most $1.6/h, 60-minute watchdog), then `do74.sh` and `pick.py` copied to the machine.
- 2026-11-13 10:06 UTC: rented `do74` (Vast.ai A100 SXM4 40 GB, Alberta, $0.589/h, driver 615.71.09, Chrome 157); setup exit 0 at 07fec07. Model fetched in 70 s; S1 and S2 took 34 and 27 s; pick `16,4;Q4_K=4,2,128;Q5_K=4,1,256;Q6_K=16,4,64`. The four benches failed at once: `tokenizers` was not installed on the image (DO-73's driver had a step for it). `do74b.sh` installed it and ran B1 T1 T2 B2 with the same pick (10:12–10:19). Logs copied back, machine deleted by recorded ID at 10:20 (`pods ls`: 0).

## Results

Sweeps (GB/s on single matrices, today's 16,4,64 against the sweep's best):

| Matrix | Format | S1: today / best (shape) | S2: today / best (shape) |
|---|---|---|---|
| ffn_up | Q4_K | 162 / 183 (4,2,128) | 202 / 236 (16,2,64) |
| ffn_gate | Q4_K | 183 / 195 (4,1,256) | 172 / 177 (4,1,256) |
| ffn_down | Q6_K | 275 / 275 (16,4,64) | 258 / 258 (16,4,64) |
| attn_qkv | Q5_K | 154 / 154 (4,1,256) | 140 / 140 (4,1,128) |
| attn_q | Q4_K | 122 / 130 (4,4,128) | 126 / 126 (4,1,256) |
| output (head) | Q6_K | 423 / 423 (16,4,64) | 415 / 415 (16,4,256) |

Bench, resident step, ms per token (32 tokens per run; each run's two resident variants):

| Positions | B1 | T1 | T2 | B2 | B median | T median | T against B |
|---|---|---|---|---|---|---|---|
| 64 | 19.9, 20.0 | 23.7, 23.6 | 23.6, 23.7 | 20.0, 19.9 | 19.95 | 23.65 | +18.5% |
| 1,024 | 22.7, 22.8 | 26.3, 26.3 | 26.0, 26.2 | 22.7, 22.8 | 22.75 | 26.25 | +15.4% |

Gate (computed by script): 1 passes (worst sweep error 9.7e-7; tokens identical at both lengths in every run); 2: T is 18.5% slower, **Rejected**.

## What it taught

- **The single-matrix microbenchmark misleads on an A100.** It repeats one matrix 20 times; the Q4_K and Q5_K matrices (12–30 MB) fit in the A100's 40 MB L2, so after the first dispatch it times cached reads, where many short rows per workgroup (4 threads per row) win. In the model each matrix is read once from memory, and those shapes lose. The head (110 MB, larger than L2) picked today's shape in both sweeps. A useful sweep must rotate over more than 40 MB of weights per timing, or time the whole step.
- **The single-matrix numbers are below the in-model ones** (Q4_K 120–240 GB/s here, 257 in the model's profile): one dispatch of a 4096-column matrix is too short to fill the A100. Small matrices are latency-bound, which points at fewer, larger dispatches (fusing projections that share an input, WS-36) rather than a different shape.
- **Today's default reached 50 tokens per second** on this A100 (19.95 ms at 64 positions), against 45 (22.2 ms) in experiment 129 on another A100 host; this run also had experiment 132's Q8_0 fast path, but host and fast path were not separated.
- The microbenchmark's picks moved between two sweeps a minute apart (ffn_up: 4,2,128 then 16,2,64): differences of 5–15% between shapes are within its noise.

## Limits

- One host, two runs per arm. The detail profile (gate item 3) was not taken: the bench's `--resident` mode records no per-kernel times, so how much of the step is now Q8_0 or small kernels on the A100 is still unmeasured.
- Only the 57 shapes of today's kernel structure were tried (threads per row, threads per item, workgroup size), not new structures (several rows per thread, wider loads, weights in a different layout).
- The pick used per-format summed single-matrix time, the method that the L2 effect invalidates; a pick from whole-step timings was not tried.

## Reproduce

```bash
# on an A100 after contributor_fleet setup, at 07fec07 (tokenizers installed: uv pip install "tokenizers>=0.23,<1")
bash runs/pods-data/do74/do74.sh      # fetch, sweeps S1 and S2, pick (pick.py at /workspace), benches B T T B
# a sweep alone:
.venv/bin/python -m ygaraty.experiments.qwen35_webgpu kernels --gguf $Q/Qwen3.5-9B-Q4_K_M.gguf --config $Q/config.json --cases decode1 --matvec-c 16,4 --sweep-matvec-c all --rounds 3 --out OUT
```
