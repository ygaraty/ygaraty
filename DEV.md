# Developing Ygaraty

How to run and test the latest code from a checkout. User installation is in
the [README](README.md); maintainer packaging checks are in
[tools/README.md](tools/README.md).

## Run the latest code from a checkout

Use an editable install in the checkout. Code changes, including browser
modules, take effect on the next launch without reinstalling:

```sh
cd ygaraty                       # the repository root
python3 -m venv .venv
.venv/bin/pip install -e '.[server,prepare,test]'
export PATH="$PWD/.venv/bin:$PATH"   # add to ~/.bashrc to keep it
```

To update, `git pull`. Re-run the `pip install` line when a `pyproject.toml`
changes or modules move: the `ygaraty`
command's entry point is written at install time, so after a move it fails
with `ModuleNotFoundError` until you reinstall. Restart any running `ygaraty
launch` after pulling too: it imports some modules lazily (the terminal chat
when the model becomes ready) and fails if they moved underneath it.

The runtime is one distribution, with implementations under `ygaraty/`.
Do not also put another installation's `bin` directory (for
example `~/.local/share/ygaraty/venv/bin`) on `PATH`, or `ygaraty` may resolve
to an older copy.

### Why not the curl installer?

`curl -fsSL https://raw.githubusercontent.com/ygaraty/ygaraty/main/install.sh | bash`
is the user path, but it does not suit development:

- While the repository is private, the raw URL returns 404, and the installer's
  own Git fetch fails the same way.
- Reruns reuse the existing installation (the `.installed` marker in
  `~/.local/share/ygaraty`) and never upgrade it.

To test the installer itself from the checkout, install into a new directory:

```sh
YGARATY_SOURCE_DIR="$PWD" YGARATY_INSTALL_DIR="$HOME/.local/share/ygaraty-test" \
  bash ./install.sh --install-only
```

That copy is not editable; remove the directory to discard it.

## Launch for local testing

Links are local by default (`http://127.0.0.1:8765/…`); no tunnel is used.
Contributors on this machine or network connect over WebRTC without TURN and
fall back to the coordinator relay.

```sh
ygaraty launch --model qwen3.5-0.8b --num-workers 2
```

- `qwen3.5-0.8b` is the fastest model to test with: about 0.37 GB per worker,
  loaded in about a minute from Spaces.
- `--lan` shares the links on your network as `https://192.168.x.y:8765/…`
  with a self-signed certificate (each device accepts the warning once;
  `tools/headless-contributor.sh` accepts it automatically for private
  addresses). Plain `http://` LAN links would load without WebGPU.
- `--public` shares the links through a Cloudflare quick tunnel (needs
  `cloudflared` on PATH, or `YGARATY_CLOUDFLARED=/path/to/cloudflared`). Links
  go through Cloudflare, so workers use the relay, not WebRTC.
- `--request-log` prints HTTP requests and session events. By default the
  terminal shows the contributor link and one live row per worker.
- `--serve-weights` downloads all weights to this machine and serves them,
  instead of contributors fetching them from Spaces.
- `--port` and `--workspace` keep a second launch apart from the first.

When the model is ready, launch opens a `>` chat prompt. Each answer is
followed by a line with time to first token, tokens and tokens/s, and the route
with WebRTC link state. `/net` shows link details, `/reset` starts a new
conversation, `/quit` stops the server, and Ctrl-D closes only the prompt.

### Headless contributors

`tools/headless-contributor.sh` runs contributors as headless Chromium on this
machine's GPU and accepts automatically:

```sh
tools/headless-contributor.sh 'http://127.0.0.1:8765/#key=…' 2
```

Each contributor gets its own profile under `~/.cache/ygaraty/contributors/`,
which keeps its verified weight cache, so restarting is quick. Several runs can
share the machine. Progress is printed every 10 seconds; Ctrl-C stops every
browser the script started. Set `CHROMIUM` to choose another browser.

## Tests

From the repository root, with the editable install above:

```sh
.venv/bin/pip install httpx ruff==0.14.0 pytest   # once
.venv/bin/python -m unittest discover -s tests/webgpu
node --test tests/webgpu/browser/*.test.mjs
.venv/bin/python -m pytest tests/inference_support
.venv/bin/python tools/test_installer.py
.venv/bin/ruff check ygaraty/backends/webgpu
```

## Browser modules

The `.mjs` files in `ygaraty/backends/webgpu/browser/` are generated
from the `.mts` sources next to them. Edit the `.mts` file, then rebuild and
check that the committed output matches:

```sh
npm install --no-save --no-package-lock   # once; versions are pinned in package.json
npm run build                             # tsc
npm run check                             # generated files match their sources
```

Commit both the `.mts` source and the generated `.mjs`.
