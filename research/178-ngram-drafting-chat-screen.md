# Exact n-gram drafting in a one-contributor chat keeps the tokens and speeds edits up to 24%, but costs 54–69% on everything else

| | |
|---|---|
| **Conclusion** | Failed gate (local screen): edits 0.76× and 0.90× plain, other prompts 1.54–1.69×; tokens identical (measured, S5-nano on 680M) |
| **Status** | executed |
| **ID** | DO-86 (report 178) |
| **Category** | browser |
| **Where** | this machine (S5-nano on the Radeon 680M, one native contributor); the remote Qwen3-Coder arm was not run, as the frozen gate required |
| **Priority** | high |
| **Date** | Added 2026-11-14; finished 2026-11-25 |
| **Path** | `YGARATY_DRAFT=3,4` (new): DO-25's exact n-gram drafting in a served one-stage, one-slot chat session, against the default path (one-stage local loop with the resident burst) |
| **Result** | Correctness passes: all five prompts' 128 tokens identical across all six runs. Speed fails: median ms per token, plain against draft: Python edit 53.0 → 40.2 (0.76×), JavaScript edit 52.7 → 47.6 (0.90×), new code 52.2 → 85.5 (1.64×), Portuguese prose 52.3 → 80.7 (1.54×), English prose 51.9 → 87.5 (1.69×). The gate needed edits ≤ 0.7× and nothing else more than 5% slower |
| **Evidence** | measured-local: `runs/experiments/DO-86/local-screen/` (per run, streamed per-token timing and text), `runs/experiments/DO-86/local-try/` (functional check) |

## Question

Exact drafting cut edit and code time by 61–73% with identical tokens (experiments 111, 112) on fixed prompts, but
production chats never enable it: `qwen35_session` refuses `--draft` with `--chat` or `--batch`, and prepared
sessions set no `draft`. Can drafting run in a one-contributor chat with chat slots and the local loop, and how
much does it cut ms per token on code and edit prompts?

Needs DO-85 deployed first (otherwise the report round trip, not the GPU, sets the pace).

## Prediction

Code and edit prompts: about 2–3× fewer ms per token; new prose: unchanged (cautious drafting sends none);
tokens identical.

## Gate (local screen, frozen 2026-11-25, before the timed runs)

S5-nano (Qwen3.5 family; report 175's full bundle) served on this machine (Radeon 680M) with one native contributor, one chat slot, the default one-stage local loop. Arms: **plain** (today's path, resident burst) and **draft** (`YGARATY_DRAFT=3,4`: DO-25's exact n-gram drafting, n-gram 3, up to 4 drafts per pass). Prompts (`runs/experiments/DO-86/prompts.json`): two edit prompts (a Python and a JavaScript function returned with one change), one new-code prompt, two prose prompts (Portuguese, English). 128 greedy tokens, streamed (`DO-54/tools/turns.py`: ms per token after the first token). Order plain, draft, draft, plain, plain, draft.

- **Correctness:** every prompt's text identical between arms in every run.
- **Works (local)** if the edit prompts' median ms per token with drafting is at most 0.7 × plain, and no other prompt is more than 5% slower (medians over the three runs). The remote Qwen3-Coder arm follows only if the screen works, with its own budget.

## Protocol

TBD. Local screen first: wire drafting into the chat path (with one active slot, or per slot under batching),
run `qwen35_session local --chat` with `--send-delay`, chat_load-style prompts (code, edit, prose), interleaved
arms, 3 runs each. Remote A100 arm only after the local screen passes and with a budget.

## Log

- 2026-11-14: added (from the DO-85 follow-up discussion).

## Limits

Premise: drafting is not yet supported with chats or batching; the work includes making it so.
- 2026-11-25: Started (user: "DO-86 n-gram drafting", local screen with S5-nano). `YGARATY_DRAFT=N,K` added to prepared sessions (`_legacy/prepared.py`; one stage, one slot, no batching; off by default). Functional check: the Python edit prompt gave identical 64 tokens with and without drafting (4.16 s against 4.70 s for the request). Gate frozen above.
- 2026-11-25: **Screen runs** (order plain, draft, draft, plain, plain, draft; `runs/experiments/M-9/tools/serve_run.sh` with `CLIENT=DO-54/tools/turns.py MAX=128`): per run, plain 51.8–53.1 ms per token on every prompt (spread under 1%); draft 39.7–40.9 (Python edit), 47.3–47.6 (JavaScript edit), 84.6–85.8 (new code), 80.1–81.5 (Portuguese), 87.4–88.4 (English). **Gate: FAIL** (speed); correctness PASS.

## Conclusion

Drafting works in a chat session and keeps the tokens exactly, but as wired today it is a net loss. Turning it on also turns off the resident burst (DO-87: the token stays on the GPU and the next step is submitted before the read-back), and each pass pays a checking cost (several rows, state saved and restored when a draft is rejected). On prose and new code, where few drafts are accepted, that makes every token 54–69% slower; on edits, where long stretches copy the prompt, it gains 10–24%, less than the 30% the gate asked for. To be worth having, drafting needs the resident loop's overlap, and it should only be paid for while drafts are being accepted (for example, switch it on after the output starts copying the prompt, off otherwise). That is a new experiment, not a rescue of this one.

## Limits

- One small model on one integrated GPU; the 30B model on an A100 has a different balance of compute per row and per pass.
- n = 3, k = 4 only; adaptive settings (DO-25's `adapt`) not tried.
- Five prompts, 128 tokens each.

## Reproduce

`YGARATY_DRAFT=3,4 CLIENT=DO-54/tools/turns.py MAX=128 PROMPTS=runs/experiments/DO-86/prompts.json NAMES=edit_py,edit_js,code_new,prose_pt,prose_en bash runs/experiments/M-9/tools/serve_run.sh runs/experiments/M-9/bundles/full PORT OUT` (without `YGARATY_DRAFT` for plain).
