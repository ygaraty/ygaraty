# Do n-gram hints cut training time on the distributed 30.3M Transformer?

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open (DO-5a executed as [experiment 73](073-hints-timing-pool.md)) |
| **ID** | DO-5 (number added on finish) |
| **Category** | training |
| **Where** | remote |
| **Date** | Added 2026-11-01 |
| **Path** | Experiment 61's pool (DigitalOcean nyc3, 8 weak c-2 + 8 strong c-8, handoffs on), the 30.3M Transformer, with and without n-gram hints |

## Question

Experiment 55 (F8 in [NEXT_2026-10-31.md](../../papers/uplink-bound/NEXT_2026-10-31.md)) found that source-filtered n-gram hints reach arm N's held-out loss after 63–68% of the pairs, at +6% compute per update: about 30% less training time. That is the only open lever on **useful compute** (learning per pair) rather than distribution overhead, and it is larger than any overhead lever left. But it was measured on the PyTorch reference model only, at 3.6% of an epoch, with two seeds. Does it hold on the distributed 30.3M model, and what does it cost per update there?

**Needs first:** n-gram hints in the distributed task graph (`distributed_translation` and its tasks). The coordinator has no hint path today; the build and a local parity check (hinted model with hints off equals the plain model, as experiment 55 checked) belong to the local session.

## Prediction

Held-out loss reached with about 30% fewer pairs, at a few percent more seconds per update (the hint lookup and the extra input). Net: roughly 25–30% less time to the same held-out loss. TBD how the hint inputs add bytes on the weak workers' uplinks.

## Gate

TBD, to freeze before the run. Proposed: arms with and without hints, 2 seeds, the same pairs budget; pass if the hinted arm reaches the plain arm's final held-out loss in at most 80% of its wall time, with no seed worse than 90%.

**DO-5a, frozen 2026-11-03 before any pool exists:**
- **Correctness:** every job completes all 22 updates with 0 retries and 0 rejected; losses are identical across repeats within each arm (the arms differ, since hints change the model).
- **Pool check:** the plain arm's mean steady seconds per update (updates 2–21) is within 10% of experiment 61's 4.64 s.
- **Decision:** the hinted arm's mean steady seconds per update is at most 1.15× the plain arm's → run the learning curves. Above 1.15× → fold the hint adds into the layer and embedding kernels first.

## Protocol

- Pool and setup of experiment 61 (handoffs on, gzip on weak workers, capacity placement `never`).
- Arms: plain, hinted (experiment 55's source-filtered hints, the same index build); seeds 2026 and 2027; interleaved.
- Budget: TBD from experiment 55's 32,768 pairs (512 updates of 64 pairs, about 40 minutes per job at 4.6 s per update).
- Record seconds per update, held-out loss at fixed pair counts, bytes per update.
- Cost: about $8 (4 jobs at about 40 minutes on a $2.9/h pool, plus setup).

## Log

- 2026-11-01: Added (high priority). Blocked on the hint path in the distributed graph.
- 2026-11-01: Hint path built (uncommitted): `models/transformer_translation.py` (`HintVector`, three injection points, `hint.*` parameters), `compiler/tape.py` (position-table gradients grouped per table), `coordination/lstm.py` (`YGARATY_HINTS`, `YGARATY_HINT_INDEX`, leave-one-out lookup per update). `tests/test_transformer_hints.py`: loss and every gradient equal `hinted_translator` (2 and 3 layers); two owned Adam updates (layers and 3 or 2 vocabulary shards) equal the plain plan. 30.3M plan with hints, shape-only: 374 → 662 tasks, critical path 20 → 24 hops; source lookup 7 ms for 64 random-token pairs. Still missing for DO-5: a held-out loss (the distributed job evaluates none), and the index on the pods' coordinator.
- 2026-11-01: Both built. Held-out loss: `ygaraty.experiments.checkpoint_heldout` loads a block checkpoint into the PyTorch reference (hinted when its model_config has hint_buckets) and reports experiment 55's `heldout` (and `heldout_off`) on the first 1,024 validation pairs; `watch RUN` scores each checkpoint as it is written (`tests/test_checkpoint_heldout.py`: a 4-wide-block checkpoint reloads exactly and its model equals the reference). On the pods: `scripts/remote.sh hint-index` builds the index in `runs/hint-index` (two shards for the merge check); `serve34` with `YGARATY_HINTS=source` builds it if missing and passes it on, and `YGARATY_HELDOUT=1` starts the scorer beside the coordinator (set `YGARATY_CHECKPOINT_EVERY` to the evaluation interval). The local index's metadata passes the coordinator's checks against the dataset's tokenizer record.
- 2026-11-01: Plan revised: timing first (DO-5a). The distributed model equals experiment 55's (same initialization, loss and gradients), so its learning per update should repeat 55's; the unknowns are the seconds per update with hints on the real pool (4 more critical-path hops, the per-update lookup, 662 tasks instead of 374) and whether float32 weights with bf16 activations keep the gain. DO-5a: experiment 61's pool, plain against hints, ABBAAB, 22 updates, plus a small-model rehearsal; about $3. If hints cost under about 15% per update, run the learning curves (about $10); if more, fold the hint adds into the layer and embedding kernels first.
- 2026-11-03: DO-5a approved: budget $5, teardown deadline 90 minutes after the pool is created (the pool watchdog's `--minutes 90`). Before provisioning, a local run found that the hint setting's `self.hints` overwrote `Job.hints()`, the workers' cache hints. Every translation job failed at its first cached claim, on the pods too. Fixed in cec07b1 (renamed to `hint_mode`). Driver `runs/pods-data/do5a/drive_do5a.py`: arm A has `YGARATY_HINTS=none`, arm B has `YGARATY_HINTS=source`, both with handoffs and gzip on the weak workers, ABBAAB, 22 updates. It is a copy of experiment 61's `drive_r1.py`. Analysis: `analyze_do5a.py`. Waiting for the DigitalOcean token on this workstation (`~/.config/ygaraty/digitalocean-dev`).
- 2026-11-03: Provisioning attempt 1: nyc3 no longer sells g-8vcpu-32gb, c-2 or c-8, so the pool was moved to sfo2 (same sizes, $2.875/h, at most $4.31 in 90 minutes). `pods up do5a` failed at its first call: `POST /v2/vpcs` HTTP 403, since the token lacks write scope. 0 machines were created and $0 spent. Waiting for a token with full read and write scope.
- 2026-11-03: Pool `do5a` up in sfo2, created about 16:50: server g-8vcpu-32gb, w0–w7 c-2, w8–w15 c-8, watchdog 90 minutes. All 17 bootstrapped in 7.5 minutes; the hint index was built on the server. Code on the pods: dc2f86f. Small-model rehearsal (6 updates, 0-A and 0-B): both complete with 0 retries and 0 rejected, and the losses part after update 1, so hints took effect. Main matrix started 17:08: `drive_do5a.py runs/pods-data/do5a/main 22` under watch.py (stall 900 s, maximum 3,000 s).
- 2026-11-03: DO-5a finished as [experiment 73](073-hints-timing-pool.md). **Failed gate:** hinted updates take 1.58× as long as plain (7.65 against 4.84 s; 915 tasks instead of 708), so time to loss comes out at 1.00–1.07× plain. The pool was deleted, cost about $1.90. Next, by the frozen rule: fold the hint adds into the layer and embedding kernels, and measure with a local capture first. The learning curves wait.
