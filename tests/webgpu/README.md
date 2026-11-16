# Public verification backlog

## 2026-11-15 — mandatory worker count and RAM summary

`launch` and `bootstrap` require `--num-workers` (aliases `--workers` and
`--contributors`). The installer refuses a launch without it before installing.
Prepared plans must match the count; local preparation creates that many layer
stages or rejects explicit splits with a different count. Launch prints context,
slots, per-worker and aggregate configured GPU allocation budgets in GiB and GB.
These are capacity budgets, not fabricated peak-memory estimates or host RAM
requirements; browser/driver overhead and device acceptance remain separate.

122 public Python tests pass, covering omission, mismatched splits and summary
units/values as well as existing model, download, tunnel and serving behavior.
Product examples now specify the worker count explicitly.


## 2026-11-15 — two-contributor public Spaces examples

Examples now use published prepared bundles: S5-Nano with a 3 GiB per-stage
weight budget and Qwen3.5-9B with 4 GiB, both two contributors, one slot and
4,096 context tokens. `--contributors` rejects mismatching prepared plans;
weight budgets are checked from the manifest before artifact downloads and
again on cache reuse. Per-model default workspaces separate model artifacts
and credentials. The example wrapper uses `launch` for automatic sharing.

Live anonymous probes now pass: both manifests returned HTTP 200 and all 461
S5 objects / 469 Qwen objects returned HEAD 200, with no declared size mismatch.
The public `prepare_session` metadata/fresh-plan checks pass for both real
manifests. [Probe evidence](evidence/public-spaces-2026-11-15.json) pins the
manifest digests, splits and measured declared weights. No model weights were
downloaded and no inference or GPU allocation was run in these checks.

119 public Python tests pass, including preset selection, conservative runtime
defaults, model-isolated workspaces and rejection of stage-count/weight-budget
mismatches before weight downloads. Historical Qwen allocation evidence uses
16/16 stages on larger GPUs; the current public manifest uses 17/15. Acceptance
of these exact bundles on actual 3/4 GiB devices remains a runtime gate.


## 2026-11-15 — automatic public sharing

Launch now defaults to an anonymous localhost.run HTTPS tunnel, started only
once Uvicorn has bound the coordinator socket. `--local`, `--public-url` and
`--prepare-only` bypass tunnel creation. The SSH child uses no user identities
or global SSH configuration, stores host trust in the workspace, and is stopped
on server shutdown, startup failure, timeout or tunnel loss. Domain rotation
stops the run with an instruction to restart and share the new link.

Verified: 110 Python tests and five installer tests pass, including 14 new
fake-process/lifecycle tests for parsing real registration events, SSH options,
timeout, startup cancellation, process termination/kill, domain rotation,
connection loss, bind failure, local mode and shutdown during tunnel startup.
Focused Ruff, shell syntax and Git whitespace checks pass.

Two live checks used only temporary synthetic data: an HTTPS tunnel forwarded
authentication and two SSE events separated by 2.0 seconds without buffering;
the actual ASGI launch flow then served model discovery, streaming completions
and contributor status through a public tunnel using the fake engine. Both
closed their SSH subprocesses and local servers. No real model or GPU was run.
The provider's free tier has temporary hostnames and a transfer-speed limit;
large-model distribution performance through it has not been validated.


## 2026-11-15 — advanced launch and product documentation

`ygaraty launch` composes checkpoint/bundle preparation, scoped keys and the
public host. It prints contributor/private-chat links, app settings and readiness
changes. The browser chat polls readiness and the authenticated `/v1/models`
route exposes the configured model. The installer now starts this launch flow.
User guides live in `docs/`; internal notes stay beside their source or tests.

Verified offline: 86 Python tests, 106 JavaScript browser/fake-device tests and
five installer tests. New coverage includes launch argument propagation,
invalid settings before writes, custom model geometry, bundle/key reuse,
fragment-only scoped links, model discovery authorization, readiness changes,
monitor shutdown, browser credential handling and streamed conversations.
Focused Ruff checks, JavaScript/shell syntax and Git whitespace checks pass.
No TypeScript or generated inference modules changed. No real model, GPU,
external HTTPS endpoint or browser UI was exercised in this change.


## 2026-11-15 — self-contained WebGPU bootstrap example

Changes: `ygaraty bootstrap` composes the existing local inspection, bundle
builder, validation, credential initialization and serving APIs. The example
wrapper installs only public packages. Profiles cover historically measured
Qwen3.5-9B and partially validated Qwen3-Coder-30B-A3B. Existing compiler, plan,
kernel, buffer/state and runtime implementations are unchanged.

Completed offline:

- 73 public Python tests, including nine new bootstrap tests: dry-run isolation,
  prepare/launch composition, key preservation, repeat validation, recipe changes,
  wrong-family rejection, incomplete workspace refusal, preparation/validation
  failure and rejection of a replaced valid bundle. Numerical preparation and
  serving are mocked.
- Existing fresh family plans, serialization/admission and fake execution checks
  pass within that suite. The operation-schema regeneration check passes.
- 103 browser fake-device tests pass; no GPU or browser contributor is launched.
- One shell-wrapper test uses a fake executable and confirms argument/working
  directory preservation. It does not install packages or start a service.
- Four public runtime wheels build offline; their isolated tests also pass all
  73 Python and 103 browser cases, with private imports blocked.
- Focused Ruff lint/format, shell syntax and whitespace checks pass.

TypeScript regeneration was not rerun: this checkout has no local `tsc` or Node
build dependencies. No TypeScript or generated browser module changed. Existing
JavaScript tests and Python schema comparison passed. Install the pinned build
tools on a permitted host before a future browser-source change.

Still pending on an authorized host: prepare a real matching checkpoint, launch
the extracted public host, admit all four Qwen3.5 contributors, verify generated
tokens/state behavior, test restart/cancellation/disconnection, and compare
latency and peak memory with the measured baseline. Qwen3-Coder retains its
separate partial-validation limits. Offline tests cannot establish these results.
No live inference, model training, coordinator or remote job was run here.

Reproduce with installed public/test dependencies:

```sh
python -m unittest discover -s tests/webgpu -v
python tools/webgpu/inference_schema.py --check
node --test tests/webgpu/browser/*.test.mjs
python tools/check_wheels.py /absolute/path/to/public-wheels
```

See the [quickstart](../../../docs/quickstart.md),
[contributor guide](../../../docs/contributing-gpu.md) and
[API examples](../../../docs/api.md) for the supported launch workflow.

## Calculated worker memory (2026-11-15)

The latest launch path supersedes the fixed 3/4 GiB preset budgets above.
Both public models select Spaces manifests for 1, 2, 3 or 4 workers. Estimates
are derived from the exact bundle tensor inventory and `Stage.load` allocations,
plus recurrent snapshots for every slot. Recommendations add 20% + 256 MiB
per worker for transient allocations and driver/browser overhead. They are not
measured GPU peaks or host RAM estimates. Whole-GiB runtime caps round up the
largest recommendation. The estimator rejects repack/draft configurations.

`test_memory.py` runs the real browser loader against a fake GPU allocator and
checks 1–4 worker layouts, context changes, slots and tied embedding reuse.
`evidence/worker-memory-2026-11-15.json` records public manifest SHA-256 values
and exact matching persistent allocations for all eight actual published plans.
The comparison does not download weights, execute inference or certify small
GPUs. Device acceptance and real-model end-to-end checks remain separate gates.

The download tests verify memory admission happens before fetching artifacts
and is repeated when reusing a verified cache. Launch tests cover all eight URL
selections and separate cache/key directories by model and worker count.

## Public WebRTC/TURN launch integration

`launch` and `serve` select auto/relay/webrtc/turn routes. Auto uses WebRTC when
TURN host and secret are configured; explicit WebRTC modes fail before model
preparation when missing settings. Secrets are factory dependencies, not bundle
or session-contract fields. Public chat stays on HTTP/SSE; browser chat-direct
fixtures are disabled while contributor-to-contributor links use WebRTC.

The ASGI tests exercise real protocol signaling on a fake engine, per-browser
join credentials, ICE authorization, expired/replaced assignments, private
candidate filtering, route capabilities and both streaming/nonstreaming chat.
Browser tests exercise instance credentials on ICE/signaling fetches and
background heartbeats. Existing inference browser tests remain unchanged.

No live TURN allocation, NAT traversal or GPU inference was run for this wiring.
Deployment acceptance must test two remote browsers with direct and relay-only
ICE, contributor replacement, link failure/fallback and the OpenAI API.
