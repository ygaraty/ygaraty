# A batched matvec kernel doubles two to four conversations, exactly, but both batch paths stall near 107 tokens per second

| | |
|---|---|
| **Conclusion** | Failed gate: exact, 2.2× at 2 conversations, but saturates at 107 tok/s by 32 like the old path (measured, A100) |
| **Status** | executed |
| **ID** | DO-69 (experiment 125) |
| **Category** | browser |
| **Where** | remote |
| **Priority** | high |
| **Date** | Added 2026-11-12 |
| **Path** | Qwen3.5-9B on one rented A100 80 GB (as experiment 123), after a workstation screen: a decode kernel for 2–32 tokens per step (each weight block read once for all rows; the coalesced kernel's layout, and Chrome's `chromium-experimental-subgroup-matrix` where the adapter has it), the batch limit raised from 8 to 32; arms: current `runBatch` vs the new kernel |
| **Result** | Qwen3.5-9B, tokens per second in all at 1 / 2 / 4 / 8 / 16 / 32 conversations, on an A100 SXM4 40 GB: current path 29.2 / 22.6 / 40.8 / 72.9 / 93.6 / 108.9, new kernel 28.7 / 49.1 / 72.3 / 85.5 / 100.3 / 107.0. Correctness passes: every batch row gives the lone conversation's 16 tokens and bitwise-equal logits at every size (the old path within 2e-4). Speed fails as frozen: 32 conversations give 3.7× one (needed 8×) and the new kernel is 2% under the old one at 32. Past 8 conversations both paths add about 9 ms per conversation per step, so (experiment 126: the products are arithmetic-bound). Workstation screen first (680M): 2.1× at 2, 1.1× at 8, 8/1 = 2.997× (needed 3×). About $0.51 (two hosts; the first refused SSH). |
| **Evidence** | measured-remote and local; curated `public/research/evidence/browser/batched-matvec-a100-2026-11-12.json`; raw `runs/pods-data/do69/out/` (A100) and `runs/experiments/DO-69/omarchy-20261112-222543/` (screen); code 073286b |

## Question

Experiment 123: our WebGPU path reaches 77 tokens per second at 8 conversations, and 2 conversations are slower than 1 (the batch path uses the rows kernel, not experiment 121's). How far does a kernel that reads the weights once per step for all conversations take aggregate throughput?

## Prediction

Aggregate tokens per second near N × one conversation's up to 8–16 (weights read once), at least 300 at 8 and 600 at 32 on the A100; one conversation unchanged.

## Gate (proposed; freeze before running)

TBD: correctness (each conversation's tokens equal to its single-conversation run; logits within 2e-3 of the reference), and a speed gate in aggregate tokens per second at 2, 4, 8, 16, 32 against the current path and experiment 123's llama.cpp numbers.

**A100 run (frozen 2026-11-12, after the screen, before renting).** User, 2026-11-12, after the screen's narrow speed miss: "Rent, $3 cap, 90 min". One Vast.ai A100 80 GB (SXM4 preferred, as experiment 123), Qwen3.5-9B Q4_K_M, the screen's command with `--batches 1,2,4,8,16,32 --batch-check`, arms A and B (`--matvec-b 8`), order A B B A.
1. **Correctness (B):** every row's 16 tokens equal the lone conversation's at every batch size, last logits within 2e-3 of the largest.
2. **Speed:** B's aggregate tokens per second above A's at every size from 2 to 32, and B at 32 at least 8× B at 1 (medians of two runs).
Both pass: **Works** (the kernel goes to the sessions' batch path as an option, default decided with the coordinator's batching). Otherwise **Failed gate** with the numbers. Reported beside experiment 123’s WebGPU measurements.

**Workstation screen (frozen 2026-11-12, before its first run).** Radeon 680M, Qwen3.5-9B Q4_K_M, short prompt, `bench --batches 1,2,4,8 --batch-check --matvec 2,4,8 --matvec-c 16,4`; arm A without `--matvec-b` (the current batch path, R-rows kernel), arm B with `--matvec-b 8` (new kernel); order A B B A.
1. **Correctness (B):** for every batch size, every row's 16 tokens equal those of one conversation decoded alone from the same prompt, and the last step's logits within 2e-3 of the largest (bitwise equality reported, not required: the batch's attention and delta steps are already part of A).
2. **Speed:** B's aggregate tokens per second above A's at 2, 4 and 8, and B at 8 at least 3× B at 1 (both medians of the two runs per arm).
Pass both: ask for the A100 budget. Fail: fix or stop, recorded here.

## Protocol

Workstation screen first (680M: correctness and the 2–8 curve), then one A100 run with `bench --batches 1,2,4,8,16,32`; budget asked first.

## Log

- 2026-11-12: Added at user request, from experiment [123](123-four-engines-a100.md) (WebGPU throughput improvements).

- 2026-11-12: user "Yes" to building the kernel and screening it on the workstation first, the A100 budget to be asked after it passes (my estimate: $1–1.50 for DO-69 alone, $3 cap, 90-min watchdog). Kernel `matvecB` (webgpu_kernels.mts: matvecC's layout and sum order for each of T tokens per pass, passes looping inside the workgroup), `gpu.matvecB`, `--matvec-b T`; `bench --batch-check` decodes slots 0..T-1 together and slot T alone after the same prompt.

- 2026-11-12 22:26: rehearsal (B, batches 1 and 2, `--batch-check`): rows equal 1/1 and 2/2, last logits **bitwise equal** to the lone conversation's; but 2 conversations 211.9 ms per step against 96.8 for one: the kernel always computed 8 token lanes. Fixed before the screen: the pass is the smallest of 2, 4, 8 holding the batch. Second rehearsal: B 10.3 / 17.6 / 25.0 tokens per second in all at 1 / 2 / 4, A 10.4 / 8.3 / 15.6.
- 2026-11-12 22:3x: screen started, code ee2aef1+dirty (the kernel, not yet committed), `runs/experiments/DO-69/screen.sh` into `runs/experiments/DO-69/omarchy-20261112-222543/`, watch.py stall 900 s, max 2 h. Unrelated: `npm run test:webgpu` 143/145, the 2 failures in chat UI tests (`chat-connection-ui`), which do not load the changed modules.

- 2026-11-12 22:32: screen finished (watch.py `finished`, 4.5 min). Medians of two runs per arm, tokens per second in all at 1 / 2 / 4 / 8 conversations: **A** 10.31 / 8.24 / 15.58 / 27.79; **B** 10.30 / 17.47 / 24.87 / 30.86.
  - Gate 1 (correctness) **passes**: B's rows 1/1, 2/2, 4/4, 8/8 with the lone conversation's 16 tokens in both runs, last logits bitwise equal (A: tokens equal, logits within 6.65e-5).
  - Gate 2 (speed) **fails narrowly**: B beats A at 2 (2.1×), 4 (1.6×) and 8 (1.11×), but B at 8 is 2.997× B at 1, under the frozen 3×. On the 680M the 8-token passes become arithmetic-bound (dequantization and 8 dot products per weight): 4 → 8 conversations adds only 24%.
  - Per the frozen rule, no rental without a new decision by the user.

- 2026-11-12 22:44 CET: rented `do69` (Vast.ai A100 SXM4, $0.589/h, 89-min watchdog) with `contributor_fleet up`; the container took 21 min to start and then refused SSH (no port mapping recorded, reconnect failed); deleted with `pods down do69` after about 35 min (about $0.35).
- 2026-11-12 23:1x CET: rented `do69b`: A100 SXM4 **40 GB** (same chip as experiment 123's 80 GB card, HBM2 at 1.56 TB/s instead of 2.04), $0.589/h, 75-min watchdog; setup running. Machine script `runs/pods-data/do69/do69.sh`, code 073286b.

- 2026-11-12 23:39 CET: setup and model fetch (14 min: Hugging Face at about 7 MB/s), then A1 B1 B2 A2 in 7.5 min, all exit 0; results copied to `runs/pods-data/do69/out/`; `pods down do69b`; `pods ls` shows 0 machines. Both hosts together about $0.51.

- 2026-11-13: **now the default** for Qwen3.5 and Kolibri-1 sessions (`kernel_defaults` in `qwen_protocol.py`, used by `prepared.py` and `qwen35_session.py`; `YGARATY_KERNELS=off` switches back). Local check, Qwen3.5-9B, 4 batched conversations: self-test passed (worst kernel 5.0e-7 both ways), all 4 conversations' tokens identical to the old kernels, steady 24.5 against 15.5 tokens per second (`runs/experiments/defaults-check/omarchy-20261113-014740/`).

## Results

A100 SXM4 40 GB, medians of two runs per arm, tokens per second in all (ms per step):

| Conversations | A: current path | B: batched kernel | B / A |
|---|---|---|---|
| 1 | 29.2 (34.2) | 28.7 (34.8) | 0.98 |
| 2 | 22.6 (88.6) | **49.1** (40.7) | 2.18 |
| 4 | 40.8 (98.1) | **72.3** (55.3) | 1.78 |
| 8 | 72.9 (109.8) | 85.5 (93.6) | 1.17 |
| 16 | 93.6 (171.0) | 100.3 (159.5) | 1.07 |
| 32 | 108.9 (293.8) | 107.0 (299.0) | 0.98 |

- Correctness: B's rows 1/1 … 32/32 with the lone conversation's 16 tokens in both runs, last logits bitwise equal (A: tokens equal, logits within 1.97e-4 of 22.8).
- Gate 2: B above A at 2, 4, 8, 16, not at 32; B at 32 is 3.73× B at 1 (needed 8×). **Failed gate.**

## What it taught

- Reading the weights once per step for the whole batch works and is exact: at 2–4 conversations the step barely grows (34.8 → 40.7 → 55.3 ms), where the old kernel paid about 88 ms from 2 on.
- **Corrected by experiment [126](126-batch-step-profile-680m.md):** the profile shows the matrix products still dominate (87% of a 32-conversation step on the 680M, flat at about 720 GFLOP/s): both paths are arithmetic-bound, so the reading below is wrong.
- Beyond 8 conversations, each extra conversation adds about 9 ms to the step in **both** arms (B: 93.6 → 159.5 → 299.0 ms at 8 / 16 / 32), so the matrix products are no longer the limit. The suspects are per-conversation work in `layerBatch`: attention and the delta-net state updates dispatched one slot at a time, the per-row head and argmax. These runs were not profiled (the batch variants run with profiling off); a detail profile of the 32-conversation step is the next measurement.
- The 680M screen predicted the shape (good at 2–4, flat by 8) for a different reason (arithmetic there); on the A100 the flattening comes from per-slot work.
- Prediction (at least 300 at 8 and 600 at 32) was far off: 85 and 107.

## Limits

- A100 SXM4 40 GB (HBM2, 1.56 TB/s) instead of experiment 123's 80 GB card; both arms on the same machine.
- Identical prompt and tokens in every slot (the bench's batch variant); real conversations at different positions change attention cost.
- 16 steps per size, two runs per arm; no profile of the batch steps.
- The first host refused SSH after 21 min of image pull and was deleted unused.

## Reproduce

```bash
# workstation screen (680M), code 073286b or later:
bash runs/experiments/DO-69/screen.sh runs/experiments/DO-69/<host-ts>
# A100: contributor_fleet up RUN --gpus 1 --gpu "A100 SXM4" --max-hourly 1.6 --minutes 75 --min-ram 64 --disk 100
# then copy runs/pods-data/do69/do69.sh to /workspace and run it (A1 B1 B2 A2, bench --batches 1,2,4,8,16,32 --batch-check)
```


Ownership note (2026-11-15): public WebGPU projection; full original comparison retained in the private archive. No measurements were rerun.
