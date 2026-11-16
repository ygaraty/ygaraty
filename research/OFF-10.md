# What workers need to serve a session at 4k, 8k and 16k context

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open |
| **ID** | OFF-10 |
| **Category** | memory |
| **Where** | this machine |
| **Date** | Added 2026-11-11 |
| **Path** | Offline: model geometry and memory accounting, plus the discrete-event simulator for timing; S5-nano and Qwen3.5-9B chat sessions |

## Question

**For a chat session whose context window goes from 4096 to 8192 to 16384 positions, what must each worker provide: memory per stage (weights plus KV cache plus activations), how many workers or stages, and what does it do to predicted prefill and decode time?**

Raised on 2026-11-11 while connecting the pi coding agent to an S5-nano session through `ygaraty-proxy`. The session reported a 4096-token context and a 1024-token output cap. pi's system prompt and tool definitions take about 2.3–2.6k tokens, which leaves about 500 for the conversation, so agent use needs a larger window. Before raising it we need to know what workers must have, so that a session profile asks only for workers that can hold it.

Related: [WS-22](149-long-replies-full-context-local.md) already served a full 16384-position S5-nano conversation locally (it tested behavior, not per-worker requirements); [DO-29](DO-29.md), [DO-56](DO-56.md), [DO-57](DO-57.md) and [DO-58](DO-58.md) are ways to fit more context per stage, which this baseline would size.

## Prediction

The KV cache grows linearly with context, and only in the full-attention layers (Qwen3.5's linear-attention layers keep a fixed state), so for S5-nano going from 4k to 16k adds a modest amount of memory per stage compared with the weights. The real limit is likely the per-stage memory a contributor offers (2–8 GiB), and prefill time for a 16k prompt rather than per-token decode time. `TBD` in numbers.

## Gate

`TBD`, frozen before running. Proposed: a diagnostic (`Found`) reporting, for each model × context (4096 / 8192 / 16384) × stage count, the peak memory per stage and the smallest worker memory that fits; plus predicted time to first token and ms per token for a pi-like turn (2.5k-token prompt growing to the window, 1024-token reply). The accounting counts as validated if it matches WS-22's observed S5-nano memory at 16384 within 10%, if that measurement exists in its logs.

## Protocol

- **Models:** S5-nano (Q4_K_M) and Qwen3.5-9B, geometry from local configs or GGUF headers only.
- **Memory accounting:** weights per stage under the current placement, KV bytes per position for the full-attention layers (and fixed state for linear-attention layers), activation and scratch buffers at the session's prefill chunk size. Start from `ygaraty/coordination/placement.py` and the capacity code in `ygaraty/experiments/qwen35_webgpu.py` and `ygaraty/coordinator/chat_api.py`.
- **Arms:** context 4096, 8192, 16384 positions; output cap 1024; 1, 2 and 4 stages; worker memory 2, 4, 8 GiB.
- **Timing:** `python -m ygaraty.experiments.simulate` with an explicit `--dataset-url sqlite:////absolute/path/to/dataset.sqlite` and local cost models; label every timing as simulated.
- **Inputs to check:** local model configs or GGUF headers for both models, and WS-22's run logs for a memory cross-check. If any is missing, report it and stop; don't download.

## Log

- 2026-11-11: Added.
