# Grouped one-token products shorten Kolibri-1's A100 step 4.2%, tokens identical: kept opt-in

| | |
|---|---|
| **Conclusion** | Works, small gain: grouping cuts Kolibri-1's A100 step 33.0 → 31.6 ms (−4.2%), tokens identical (measured) |
| **Status** | executed |
| **ID** | DO-77 (experiment 139) |
| **Category** | browser |
| **Where** | remote (local layer screen first) |
| **Priority** | high |
| **Date** | Added and run 2026-11-13 |
| **Path** | Experiment 135's grouping (`--group-matmul`) on Kolibri-1 Q4_K (44.7 GB, same plan executor): which product groups form (attention q/k/v, the shared expert's gate/up with SwiGLU), a local `stage` screen on a few layers (bitwise identical hidden states with and without), then the bench on an 80 GB A100 |
| **Result** | Local screen (680M, layers 0..2): outputs byte-identical. A100 SXM4 80 GB, B G G B, resident: step 33.0 → 31.6 ms at 64 positions (**−4.2%**) and 96.45 → 94.75 at 1,024 (−1.8%); tokens identical in every variant. Gate: 2–5% shorter, **Works, small gain**: grouping stays opt-in for Kolibri (Qwen3.5 keeps it by default). Kolibri has fewer groups than Qwen3.5 (attention q/k/v and the shared expert's gate/up; its routed experts already run in merged dispatches), so less to gain. About $0.92 (45 minutes at $1.227/h, including the 45 GB download). |
| **Evidence** | measured-remote and measured-local; curated `public/research/evidence/browser/matvec-group-kolibri-a100-2026-11-13.json`; raw `runs/pods-data/do77/out/` and `runs/experiments/DO-77/omarchy-20261113-1304-screen/`; code 317e82c; driver `runs/pods-data/do77/do77.sh` |

## Question

Grouping is the Qwen3.5 default since experiment 136 (−11.5% on an A100); Kolibri uses the same executor but was not measured, so its default is unchanged. Does it hold there?

## Prediction

Bitwise identical; A100 step −5–10%.

## Gate (frozen 2026-11-13, before renting)

1. **Local screen (done):** stage outputs byte-identical with and without `--group-matmul` (layers 0..2, 16-token prefill and three one-token calls).
2. **A100 (80 GB):** Kolibri-1 bench `--contexts 64,1024 --resident` with today's Kolibri kernel flags, B (no grouping) and G (grouping), order B G G B: tokens identical in every variant; **Worth it, now default for Kolibri** if G's resident step at 64 positions (median of two runs) is at least 5% shorter; **Works, small gain** if 2–5%; **Rejected** if under 2%.

## Protocol

Local stage screen (layers 0..3), then one A100 80 GB (with DO-76) after budget approval.

## Log

- 2026-11-13: Added and started at user request ("all levers as experiment and start them"), from experiment [136](136-matvec-group-a100.md)'s A100 profile.
- 2026-11-13: local screen (`runs/experiments/DO-77/omarchy-20261113-1304-screen/`, Kolibri-1 layers 0..2, 680M): outputs byte-identical in all four calls (B and G); Kolibri's plan has attention q/k/v consecutive and the shared expert's gate/up followed by silu_mul, so groups form (the router product is F32 and stays separate). Gate 1 passes.
- 2026-11-13: User: "Commit, rent, $3 cap, 2h": one 80 GB A100 shared by WS-49, DO-76 and DO-77.
- 2026-11-13: the shared A100 rental (with WS-49 and DO-76) could not fetch the Kolibri GGUF: HTTP 403, the presigned URL in `~/.config/ygaraty/kolibri-gguf.url` expired at 23:35 CET on 2026-11-12. The A100 part waits for a fresh URL. Back to open.
- 2026-11-13: fresh presigned URL from the user ("signed url there. start tests"; a ranged probe returns the GGUF header); A100 part started under the earlier approval ($3 cap shared with WS-49 and DO-76, $0.55 spent), one 80 GB A100, 90-minute watchdog, `runs/pods-data/do77/do77.sh` at 317e82c.
- 2026-11-13 12:15–13:00 UTC: `do77` (Vast.ai A100 SXM4 80 GB, Czechia, $1.227/h, driver 595.91.07); Kolibri GGUF fetched with a matching SHA-256; KB1 KG1 KG2 KB2 all exit 0 (about 5 minutes each). Logs copied back; machine deleted by recorded ID at 13:00 (`pods ls`: 0).

## Results

| Positions | B1 | G1 | G2 | B2 | B median | G median | G against B |
|---|---|---|---|---|---|---|---|
| 64 | 33.1, 33.1 | 31.5, 31.7 | 31.3, 31.7 | 32.9, 32.9 | 33.0 | 31.6 | −4.2% |
| 1,024 | 96.4, 96.4 | 94.8, 94.7 | 95.0, 94.7 | 96.6, 96.5 | 96.45 | 94.75 | −1.8% |

Gate (by script): 1 passes (local outputs byte-identical); 2: tokens identical, −4.2%: **Works, small gain** (stays opt-in).

## What it taught

- Grouping is exact on Kolibri too (byte-identical locally, identical tokens on the A100); its gain scales with how much of the step is small products sharing an input: −11.5% on Qwen3.5 (experiment 136), −4.2% on Kolibri, whose routed experts were already merged (experiment 114) and whose step at 1,024 positions is dominated by other work (96 ms against 33 at 64).
- Kolibri's step grows threefold from 64 to 1,024 positions (33 → 96 ms), unlike Qwen3.5 (17.7 → 20.4): at longer contexts its attention, not its products, is the lever.

## Limits

- One A100 host, two runs per arm; no detail profile of Kolibri's step was taken here.
- Kept opt-in by the frozen gate (under 5%); `--group-matmul` (or `group_matmul` in a fixture) turns it on.

## Reproduce

```bash
# on an A100 80 GB after contributor_fleet setup and fetch of the Kolibri GGUF, at 317e82c
bash runs/pods-data/do77/do77.sh
```
