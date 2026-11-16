# Caching plan checks and bind groups halves a Kolibri-1 token on one A100, 101 → 52 ms, bitwise identical

| | |
|---|---|
| **Conclusion** | Works: host caches halve a Kolibri-1 token on one A100, 101 → 52 ms, bitwise identical (measured) |
| **Status** | executed |
| **ID** | DO-65 (experiment 116) |
| **Category** | browser |
| **Where** | remote |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | The DO-63 machine (Vast.ai A100 80 GB); Kolibri-1, one stage of 50 layers (`pipeline --splits 0,50 --generate 32 --profile`, merged experts); arms H (host caches: each plan task's contract checked once; bind groups cached by pipeline and entries) and N (`--no-host-cache`: as before) |
| **Result** | Works (gate by script): median step 51.8 ms (H) against 100.8 ms (N), ratio 0.51 (gate ≤ 0.8), every H run below every N run (51.4, 52.1 against 95.1, 106.6); plan expansion 4.4 against 42.7 ms per token, submission wall 47.1 against 58.1 ms, JavaScript encode about 10 ms in both, GPU 28.2 against 31.9 ms. All 4 runs valid (32/32 tokens, logits within 2.8e-4); logits bitwise identical to H1's (32/32 files). About 19 tokens per second in this harness. |
| **Evidence** | `runs/pods-data/do65/` (gate.json, H1 N2 N3 H4 reports, logs, logits, files.sha256: webgpu_kernels.mjs f7760362…, inference_plan.mjs ad3e6b79…, qwen35_parity.mjs 48f45477…, qwen35_webgpu.py fac5c79d…); curated: `public/research/evidence/browser/kolibri-a100-decode-speed-2026-11-12.json`; code committed as 430c583 |

## Question

[DO-63](114-kolibri-a100-decode-profile.md) (experiment 114) found about 45 ms of each ~100 ms token outside the GPU submission, and 9 ms of JavaScript encoding inside it. Two host caches target it: the plan executor checked every task's contract (a canonical serialization) every token, and every dispatch created a bind group. With both cached, how much of the step goes, with tokens and logits unchanged?

## Prediction

Plan expansion falls about 10× (tiny Kolibri on the workstation: 2.3 → 0.2 ms per token for 86 commands; Kolibri has about 1,400): the step drops by roughly 30 ms; encoding by a few ms; GPU time unchanged.

## Gate (frozen 2026-11-12, before running)

- Runs H N N H, each `pipeline --splits 0,50 --generate 32 --kv f16 --wire f16 --profile` (merged experts). Valid: 32/32 tokens equal to the reference, logits within 2e-3 of the largest.
- **Correctness:** every run's logits bitwise identical to the first H run's.
- **Works** if H's median step time is at least 20% below N's, with every H run below every N run, and correctness holds. Otherwise **Rejected** (or **Failed gate** for a correctness break).
- Reported: per arm, plan expansion ms, JavaScript encode ms, GPU ms, submission wall ms.

## Protocol

- Code: the machine's clone (6e60083) plus hand-copied `webgpu_kernels.mjs`, `inference_plan.mjs`, `qwen35_parity.mjs`, `qwen35_webgpu.py` (hashes in `/workspace/do65/files.sha256`). Checked on the workstation first: tiny Kolibri, both arms 16/16 tokens equal to the reference and bitwise identical logits (16/16); WS-27's S5-nano chat sessions (1 and 2 workers) bitwise identical to WS-27's run 1.
- Budget: user, 2026-11-12: "work on 1, 2 and 3. budget approved"; within the running rental (watchdog 05:38).

## Log

- 2026-11-12: Added at user request (items 1–3 of DO-63's findings). Gate frozen before running.
- 2026-11-12 03:36–04:20 CET: runs H1 N2 N3 H4 (`do65.sh`, each exit 0). Gate: **Works** as in Result. The bind-group cache barely shows (encode unchanged at about 10 ms); the gain is the plan expansion, whose contract checks ran every token.

- 2026-11-12 05:06 CET: machine deleted after DO-66; `pods ls` 0. The shared rental (DO-63 to DO-66) cost $4.01 (cap $6).
- 2026-11-12: written up as experiment 116.

## Limits

- One machine, one prompt, 32 tokens; two runs per arm (all four on the predicted side).
- The two caches were not separated on the A100; encode time was unchanged (about 10 ms in both arms), so the bind-group cache contributes little and the gain is the plan check.
- The caches are on by default (`gpu.hostCache`); `--no-host-cache` switches back. The contract check now runs once per task object: a plan mutated in place after its first run would not be re-checked.
- Harness step, not a chat session.

## Reproduce

```bash
# rented 80 GB GPU with the Kolibri-1 GGUF, code at 430c583 or later
bash runs/pods-data/do65/runs.sh   # H N N H: pipeline --splits 0,50 --generate 32 --kv f16 --wire f16 --profile [--no-host-cache]
```
