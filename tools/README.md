# Maintainer tools

These tools verify and package the public distributions. User documentation lives
in [docs/](../docs/README.md). The public code must never import the private cloud
application. Original export manifests are preserved in the external historical
archive; product changes are tracked by Git.

## Offline verification

Use existing local tools; these commands start no sockets. Wheel checks also run tiny local numerical
training examples:

```bash
# From the repository root:
python -m unittest discover -s tests/contracts -v
python tools/webgpu/check_offline.py
# After building the public wheels locally:
python tools/check_wheels.py /absolute/path/to/wheels
```

Root tests check Tape finite-difference gradients, DiLoCo recovery, and
Python/JavaScript training parity. Wheel checks import supported public research
helpers with private imports blocked. Optional Torch and GPU checks are skipped
when their dependencies or opt-in hardware flag are absent.

Fresh family plans, operation schema, generated TypeScript outputs, fake GPU
traces, scoped admission, streaming, cancellation, control capacity and package
isolation are checked. Source/trace equality does not prove retained GPU speed.
Numerical/state parity and latency/peak-memory comparisons remain required on an
authorized host before production use. The code retains the repository's MIT
license; no publication or deployment has occurred.

Tokenization drift from upstream 7380f08 has been reconciled in the public
prompt/conversation/protocol paths. Public admission uses the batch API in its
bounded executor, retaining an unfinished encoding reservation after caller
cancellation. Offline tests cover dispatch and event-loop progress with synthetic
tokens; real throughput and numerical acceptance remain pending.

## Build the public distributions

```sh
python -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir dist ./research .
python tools/check_wheels.py dist
```

This requires existing local build tools. Browser regeneration requires the
versions pinned in `package-lock.json`; install those beforehand
on a permitted host. No cloud dependency is required. Real GPU acceptance is
still pending; these are offline-verified source exports, not deployed releases.

## Headless contributors

`tools/headless-contributor.sh URL [COUNT]` runs COUNT headless Chromium
contributors on this machine's GPU for the share link printed by
`ygaraty launch`. They accept automatically (`autostart=1`), each with its own
profile under `~/.cache/ygaraty/contributors/`. Profiles keep the browser's
verified weight cache, and each contributor takes the first profile no running
browser holds, so several runs can share a machine. Ctrl-C stops every browser
it started; if one exits on its own, the script reports it and stops the rest.
Use it to test a launch without opening contributor tabs by hand.

```sh
tools/headless-contributor.sh 'http://127.0.0.1:8765/#key=…' 2
```

## Installer checks

Run `python tools/test_installer.py` for isolated installer checks with fake Git,
Python and pip executables. No packages, models or servers are downloaded or run.

## Native contributors

`deno-contributor.sh` / `wgpu-contributor.sh` use Deno's built-in WebGPU.
`node-contributor.sh` and experimental `bun-contributor.sh` use the public Dawn
binding. Each delegates to `ygaraty contribute` with its runtime selected; pass
`--num-workers N` for multiple processes or `--check` for a GPU compute probe.
See [contributor setup](../docs/contributing-gpu.md#contributors-from-command-line).
