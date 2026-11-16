# Contributor launchers

`headless.sh` is the packaged Chromium helper; `tools/headless-contributor.sh`
delegates to it. Python `contributors.py` supervises process groups for the CLI.
The native runtime entry point and browser host are `browser/native_contributor.mjs`
and `native_environment.mjs`. They import the existing contributor module locally,
so admission, instance credentials, weight checks and inference stay shared.

`native_gpu.mjs` extracts Deno/wgpu and Dawn initialization from the DO-68 runner
archived at `mess-ledger/cloud/native-only-followup-2026-11-15/ygaraty_cloud/webapp/native_runner.mjs`.
That archive is provenance, not a runtime dependency. The public `native_runner.mjs`
remains the benchmark entry point; it now reports failures with a nonzero exit.
No separate Bun worker was present in the archive. Bun support uses the Dawn
binding through the same initializer and remains experimental.

Tests cover invitations, shared admission, instance-authenticated leave, failures,
command construction and process cleanup. Real local compute probes passed on
Deno 2.9.5, Bun 1.4.2 and Node with webgpu 0.6.2. All three also passed shared float16 admission against a localhost fixture,
including worker credentials and leave cleanup. Full live model inference and
multi-machine transport performance are not validated by those probes.
