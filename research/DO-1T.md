# Do float32 uploads shorten Transformer DiLoCo rounds without accuracy loss?

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open |
| **ID** | DO-1T (number added on finish) |
| **Category** | training |
| **Where** | remote |
| **Date** | Added 2026-10-31 |
| **Path** | Transformer DiLoCo (`coordination/diloco_job.py`), tiny model locally, then two pools of 8 workers on 100/20 Mb/s DigitalOcean links, three seeds |

## Question

Follow-up of DO-1, [experiment 57](057-diloco-upload-only-float32.md), which settled upload-only float32 for the CIFAR MLP. The Transformer DiLoCo path (`coordination/diloco_job.py`) has the same switch but keeps float64 and is unmeasured, and it is the model the main KPI (nanochat time, 15–20 s/update on home links) is about. Free local screen first (tiny model, float64 against upload-only, three seeds), then DO-1's timing setup (two pools of 8 workers, 100/20 Mb/s, three seeds, about $3–4). Same gates as DO-1 (frozen in [experiment 57](057-diloco-upload-only-float32.md): mean paired last-three-evaluation accuracy at least −0.5 points versus float64, no seed below −1.0). Check first that `scripts/remote.sh` can launch Transformer DiLoCo on pods; if not, that is about an hour's work before any spend. R1 stays held until a candidate shows a local gain.

## Gate

As in experiment 57: mean paired last-three-evaluation accuracy difference at least −0.5 points versus float64, no seed below −1.0; adopt as default only if the timing gain exceeds run-to-run variation.

## Protocol

Local screen on the tiny model (float64 against upload-only, three seeds), then DO-1's timing setup. Check first that `scripts/remote.sh` can launch Transformer DiLoCo on pods.

## Log

- 2026-10-31: Added, in the DigitalOcean plan of 30 October.
- 2026-11-01: Moved to its own file.
