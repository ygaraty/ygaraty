# Greedy WebGPU decode is bitwise repeatable run to run: 5 of 5 runs identical in tokens and logits, with either matvec kernel

| | |
|---|---|
| **Conclusion** | Found: decode bitwise repeatable, 5/5 runs per arm identical in tokens and logit digests (measured, 680M) |
| **Status** | executed |
| **ID** | WS-25 (experiment 152) |
| **Category** | correctness |
| **Where** | workstation |
| **Priority** | high |
| **Date** | Added 2026-11-10; run 2026-11-11 00:25–00:36 local |
| **Path** | S5-nano Q4_K_M in headless Chromium on the Radeon 680M, two stages (`--splits 0,16,32`), `qwen35_session local` (coordinator relay route, no chat page), one fixed prompt, 128 greedy tokens, `--logit-digests`; arm K default matvec kernel, arm O `--matvec off`; order K O K O K O K O K O |
| **Result** | All 10 runs valid (`finished`, 128 tokens, 128 digests). K: 5/5 identical to K1 in token ids and logit digests at every step; O: 5/5 identical to O1; K1 and O1 identical to each other. The WebGPU decode (kernels, two stages, relay transport) is deterministic, so WS-23's run-to-run differences lay in the chat path (WS-27 then found them in the test driver). About 137 ms per token (K) and 134 (O). Gate: **Found**. Free. |
| **Evidence** | measured-local; curated `public/research/evidence/correctness/decode-repeatability-local-2026-11-11.json`; raw `runs/experiments/WS-25/omarchy-20261111-0030/main/` (`gate.json`, per-run `report.json`), rehearsals beside it; scripts `runs/experiments/WS-25/{go.sh,gate.py}`; code 64f73f3 |

## Question

[WS-23](150-one-worker-chat-local.md) found that two identical HTTP chat runs give the same text on only 7 of 15 turns, and even a fresh conversation's first turn differs; earlier runs (WS-21, WS-22, DO-50 with Phi-4 on an RTX 3090) show the same. Is the decode non-deterministic on the GPU (e.g. reduction order), or does something else differ between runs (prompt, state, sampling)? At which step do two runs first diverge, and by how much do the top two logits differ there? [DO-51](DO-51.md)'s bitwise-logit gate presupposes repeatability.

## Prediction

Repeatable in this path: the kernels have a fixed reduction order and the transport preserves order. If so, WS-23's mismatches come from the chat path.

## Gate (frozen 2026-11-11, before any run)

Diagnostic: the verdict is **Found**.

- Runs: 10 `local` sessions, in the order K O K O K O K O K O, identical arguments within an arm (same prompt, `--generate 256`, `--logit-digests`, fresh profile each run, so each one downloads its weights again).
- Valid run: report state `finished` with 256 tokens, or a stop at EOS. An invalid run is reported and not compared.
- Per arm: the number of runs whose token ids equal run 1's at every step, and whose logit digests equal run 1's at every step; for any that differ, the first diverging step (tokens and digests separately).
- **Repeatable** (per arm) means 5/5 identical tokens and digests.
- K repeatable: the GPU decode is deterministic on this path, and WS-23's mismatches lie in the chat path (next: the same check through the chat page).
- K not repeatable but O repeatable: the multi-row matvec kernel is the source.
- Both not repeatable: the source is elsewhere (another kernel, or the transport); next, narrow it down by stage.

## Protocol

- Code: the unpatched clean export of `main` at 64f73f3 (`runs/experiments/WS-24/omarchy-20261110-2316/src`), the repository's `.venv`, model files from `runs/models/s5-nano` (no downloads).
- Command per run: `python -m ygaraty.experiments.qwen35_session local --gguf S5-nano-Q4_K_M.gguf --config config.json --tokenizer tokenizer.json --splits 0,16,32 --generate 256 --logit-digests --prompt "<fixed>" --out RUN/<arm><n>` (plus `--matvec off` for O).
- Under `watch.py`; one local GPU job at a time; profiles are deleted with each run.
- Capture outside the chat path: `--logit-digests` (DO-25) gives the SHA-256 of each emitted token's logits row; a logit margin at the divergence needs the raw row (not exposed).

## Log

- 2026-11-10: Added at user request ("yes" to the proposed repeatability test). Not started.
- 2026-11-11: started at user request ("run the 60ms run and after WS-25"). Gate frozen above before any run.
- 2026-11-11 00:25, rehearsal (`runs/experiments/WS-25/omarchy-20261111-0030/rehearsal/`): the first attempt did not start (the repository `.venv` has no uvicorn; switched to the panel venv). The second ran 256 tokens, but stage B's `done` report, with 256 digests, was refused: `413 Contributor metadata exceeds 32 KiB`. The run is `failed` and the digests are lost. **Protocol amended before the main run: `--generate 128`** (about 19 KB of digests); the gate is otherwise unchanged, with 128 in place of 256.
- 2026-11-11 00:27–00:36, main run (`watch.py`, `--stall 300`, progress per run; state `finished`), K O K O K O K O K O. Gate by script (`gate.json`): every run valid (`finished`, 128 tokens, 128 digests). **K: 5/5 identical to K1 in tokens and digests. O: 5/5 identical to O1. K1 against O1: identical** (as experiments 104 and 108 claim for the matvec variants). The prediction held.
- Meaning: the WebGPU decode itself (kernels, two stages, relay transport) is deterministic. WS-23's mismatches (H1 = H2 on 7 of 15 turns, even first turns of a fresh conversation) come from something in the chat path. The driver's prompts are fixed strings (`chat_cycles.prompt`), so the candidates are: how the chat path prefills or batches a turn (call shapes that depend on message timing), state left over across a conversation reset, or the page text the hash is taken from. Next: the same digest comparison through the chat path.
- 2026-11-11: Follow-up [WS-27](154-chat-repeatability-local.md): the chat path is bitwise repeatable too; WS-23's mismatches were the test driver hashing the page's timing line with the reply.

## Results

Gate by script (`gate.py`, recomputed 2026-11-14 from the reports with identical output):

| Arm | Valid runs | Identical to run 1 (tokens and digests) | Repeatable | Median ms per token |
|---|---|---|---|---|
| K (default matvec) | 5/5 | 5/5 | yes | 136.5–137.3 |
| O (`--matvec off`) | 5/5 | 5/5 | yes | 133.4–134.0 |

K1 against O1: no diverging step in tokens or digests. Every run's token list hashes to the same value (sha256 22f70f6da2d6…), and every run's digest sequence too (e8039038b0ee…). First token after 0.41–0.47 s. **Found: repeatable** in both arms, and the two kernels agree bitwise.

## What it taught

- The WebGPU decode is deterministic on this path, and the multi-row matvec kernel is bitwise equal to the plain one (as experiments 104 and 108 claimed), so DO-51's bitwise-logit gate is meaningful.
- WS-23's apparent non-repeatability had to come from the chat path; WS-27 found it in the driver (it hashed the page's timing line with the reply).
- **Limit found:** the coordinator refuses contributor metadata over 32 KiB (413), so a `done` report with 256 logit digests was lost; runs were cut to 128 tokens. WS-27 then gave `done` reports their own, larger limit.

## Limits

- One model (S5-nano), one GPU (Radeon 680M), one prompt, 128 tokens, relay transport only.
- Not the chat path, so it does not by itself explain WS-23.
- Digests are SHA-256 of each logits row: equality is bitwise, but a logit margin at a divergence could not be measured (none occurred).
- Digests were capped by the coordinator's 32 KiB contributor-metadata limit then, about 128 steps per run.

## Reproduce

```bash
# clean export of 64f73f3 in runs/experiments/WS-24/omarchy-20261110-2316/src; panel venv (uvicorn) in go.sh
runs/experiments/WS-25/go.sh runs/experiments/WS-25/<run-dir> "K O K O K O K O K O"
.venv/bin/python runs/experiments/WS-25/gate.py runs/experiments/WS-25/<run-dir>    # writes gate.json
```

