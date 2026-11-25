# Splitting Inkling 2 in three costs only 1.14× on one A100: the 4090 sessions' slowness is the 4090 hosts, not the split

| | |
|---|---|
| **Conclusion** | Found: on one A100 the 3-way split computes 41.3 vs 36.5 ms whole (1.14×); 4090 stages are 2.7× slower per visit than an A100 |
| **Status** | executed |
| **ID** | WS-55 (report 179) |
| **Category** | browser |
| **Where** | this machine (Radeon 680M, 2,304-wide tiny Inkling 2) and one Vast.ai A100 SXM4 40 GB (UK) |
| **Priority** | high |
| **Date** | Added 2026-11-25; finished 2026-11-25 |
| **Path** | The whole model through `Inkling2Stage.run` directly (report 166's tool) against the same model split 0–5 \| 6–17 \| 18–23 and run by three contributors on the same GPU (hop trace: per-stage compute per call) |
| **Result** | A100, about 1k positions: whole model 36.4 and 36.6 ms per step; split, stage compute A 5.2–5.9, B 26.8–27.1, C 8.7–9.1 ms, sum 40.7 and 42.0 ms (1.14×); stage B 0.75 ms per visit against 0.76 whole; read-back 0.4–0.55 ms per stage; token cycle with relay hops on the same machine 53–56 ms. Local (680M): contributor path about 1.11× the direct path, split 1.41×. On the rented RTX 4090s (reports 171, 174) stage B ran about 2.05 ms per visit: 2.7× the A100's, where memory bandwidth explains about 1.5× |
| **Evidence** | measured-remote: `runs/experiments/WS-55/vast-ws55b-20261125/` (`direct.*.json`, `relay-split-*` contributor logs with `HOP` lines); measured-local: `runs/experiments/WS-55/local`, `runs/experiments/WS-54/local/relay-ws55-split-*` |

## Question

On three RTX 4090s (reports 171 and 174), stage compute per decoded token was A 23, B 72–75 and C 52–54 ms (about 155 ms in all); stage B runs 36 of the 48 layer visits, about 2.05 ms per visit. The whole model on one A100 decodes in 34.7 ms (report 166), about 0.73 ms per visit. Memory bandwidth (4090 about 1.0 TB/s, A100 about 1.55) explains about 1.5×, not 2.8×. Where does the rest go: the contributor's path (options not applied, no pipelined submission, a synchronous read-back each step), the split itself (each stage's own encode and GPU work in series), or the 4090 hosts?

## Prediction

TBD.

## Gate

A diagnostic (`Found`): per visit, contributor path against direct path on the same GPU and model (local), and whole against split on one rented GPU, each from at least 20 steady steps; a fix it points to is a separate experiment.

## Protocol

1. Local (680M, 2,304-wide tiny Inkling 2, fully resident): the same one-stage bundle timed through `Inkling2Stage.run` directly and through a contributor (`hopTrace` compute per call), and the three-stage split through contributors.
2. Remote (budget from the user first): one A100, report 166's whole-model timing against the three-stage split on that GPU.

Related: [166](166-inkling2-split-decode-attention.md), [171](171-direct-links-fallback-and-relay-slow-start.md), [174](174-relay-recv-wait-slow-start.md), [165](165-inkling2-decode-attention-not-host.md).

## Log

- 2026-11-25: Added at the user's request ("Inkling stage gap").
- 2026-11-25: **Local part** (Radeon 680M, 2,304-wide tiny Inkling 2 fully resident; `runs/experiments/WS-55/local`, `runs/experiments/WS-54/local/relay-ws55-split-*`): one-token step, medians over about 60–94 steady steps.

  | Path | ms per step | Against direct |
  |---|---:|---:|
  | Whole model, `Inkling2Stage.run` directly (report 166's tool), run + read | 12.6, 13.1 | 1.00 |
  | Whole model, one contributor (`loop=relay`, `resident=off`), stage compute per call | 14.0, 14.5 | about 1.11 |
  | Split 0–2 \| 2–4 \| 4–6, three contributors, sum of stage compute (A 2.2, B 4.5, C 11.2–11.4) | 18.0, 18.1 | about 1.41 |

  Read-back is 0.16–0.19 ms per call in every case. So the contributor's path adds about 10%, and each extra stage boundary about 2 ms (its own encode, submit and wait, with no overlap across stages); together 1.4×, not the 2.8× per visit seen on the 4090s. The remainder is on the rented hosts (GPU type, CPU, or something only the real model shows), which the remote step isolates: the whole model against the same split on one A100.
- 2026-11-25: Remote step approved: one A100, budget $1, teardown 1.5 h (`tools/a100.sh`: the whole model through the direct path, through one contributor, and the 0–5 | 6–17 | 18–23 split through three contributors on the same GPU, two runs each, at about 1k positions).
- 2026-11-25: **A100 run** (`ws55b`, Vast.ai 62404177, UK, $0.441/h, 212 MB/s from Hugging Face; a first host, `ws55a`, downloaded at 2.5 MB/s and was released). Bundles prepared on the host (one stage; 0,6,18,24). Direct runs and split runs as in Result. The single-contributor runs failed twice: first the one-stage coordinator refused to start ("Stage weights exceed contribution budget"; it needs `--memory-gb`, now `MEMORY_GB` in `DO-97/tools/local.sh`), then the driver started three contributors for a one-stage session and the timing request got a 503; that data point is missing (the 680M measured the contributor path at about 1.11×). Host deleted by its recorded ID after 0.69 h, about **$0.30** of $1.

## Conclusion

The split is not what makes multi-contributor Inkling slow. On one A100 the three stages together compute only 14% more than the whole model (each extra stage boundary adds its own encode, submit and wait), and the stage holding the looped core runs at the same cost per visit as the whole model. The 4090 sessions' stage compute (about 155 ms per token) is therefore the 4090 hosts: 2.7× slower per visit than an A100 where memory bandwidth alone predicts about 1.5×. The next step is a host profile on one RTX 4090 (DO-99's tools: CPU encode against GPU time per step, and the GPU per kernel) to see whether the kernels' A100-tuned shapes or the hosts' CPUs account for the rest.

## Limits

- One A100 host; two runs per arm; the whole-model-through-one-contributor point is missing on the A100.
- The 4090 per-visit figure comes from the hop trace of reports 171 and 174 (whole decode calls), not from a profile on a 4090.

## Reproduce

Local: `node runs/experiments/DO-99/tools/host_profile.mjs MANIFEST_DIR PROMPT.tokens --steps 60 --chunk 32 --resident 0-99 --plain` and `PARAMS=hopTrace=1 bash runs/experiments/DO-97/tools/local.sh DIR relay PORT`. Remote: `runs/experiments/WS-55/tools/a100.sh setup|run` with DO-101's kit and this tree.
