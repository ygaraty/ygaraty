# Install, invite, chat

## 1. Choose a public Spaces model

Use Linux or macOS with Python 3.12+ (`venv`/`pip`), Git and curl.
Public sharing (`--public`) also requires `cloudflared`; local launch does not.
The examples download prepared models from our public DigitalOcean Spaces
storage. No checkpoint directory, cloud credentials or numerical preparation is
needed. The server keeps only metadata (about 26 MB); each contributor's
browser downloads its share of the weights directly from Spaces.

`ygaraty models list` shows the supported models, their public bundles and
validation status, and `ygaraty models info NAME` the GPU memory and download
each worker needs (see [command line](commands.md)).
Choose 1, 2, 3 or 4 workers for any model. The examples below use two.
Before downloading weights, launch calculates and prints GPU RAM per worker
from the matching Spaces bundle. With the default 16,384-token context and one
chat slot, recommendations are about 0.88 GB each for Qwen3.5-0.8B, 2.65 GB
each for S5-Nano or 4.12 GB each for Qwen3.5-9B, including headroom. See [all configurations](public-bundles.md).

## 2. Install and launch

Qwen3.5-0.8B (smallest and fastest):

```sh
curl -fsSL https://raw.githubusercontent.com/ygaraty/ygaraty/v0.1.0a1/install.sh | bash -s -- --num-workers 2 --model qwen3.5-0.8b --open-chat
```

S5-Nano:

```sh
curl -fsSL https://raw.githubusercontent.com/ygaraty/ygaraty/v0.1.0a1/install.sh | bash -s -- --num-workers 2 --model s5-nano --open-chat
```

Qwen3.5-9B:

```sh
curl -fsSL https://raw.githubusercontent.com/ygaraty/ygaraty/v0.1.0a1/install.sh | bash -s -- --num-workers 2 --model qwen3.5-9b --open-chat
```

Run one at a time. The installer uses the versioned preview `v0.1.0a1`. From this
checkout, use `YGARATY_SOURCE_DIR="$PWD" bash ./install.sh --num-workers 2 --model s5-nano`.

The installation is in `~/.local/share/ygaraty/venv`. The installer keeps separate
caches, keys and state in `~/.local/share/ygaraty/workspaces/MODEL/N-workers/`. Reruns
validate and reuse cached artifacts. Ctrl-C stops the server and public tunnel.

The installer prints an `export PATH=...` line for this terminal. Add that line
to `~/.zshrc` (Zsh) or `~/.bashrc` (Bash) for future terminals. Bash login shells
must source `~/.bashrc` from `~/.bash_profile`.

To upgrade without deleting models or session files, see [Update Ygaraty](#update-ygaraty).

After installation, use the CLI directly:

```sh
~/.local/share/ygaraty/venv/bin/ygaraty launch --num-workers 2 --model s5-nano --open-chat
```

Direct CLI runs default to `runs/MODEL/N-workers/` in the current directory. To reuse an
installer-created cache, add
`--workspace "$HOME/.local/share/ygaraty/workspaces/s5-nano/2-workers"`.

## 3. Invite your contributors

The terminal shows one link to share, what a contributor needs, and a live row
per worker:

```text
  Ygaraty  qwen3.5-0.8b · 1 worker · 16,384-token context · 1 chat at a time

  Open this link in Chrome or Edge on a computer with 1.22 GB of free GPU memory:

    http://127.0.0.1:8765/#key=…

  One person (or this computer) opening it is enough.
  Shared on this computer only. Other computers: --lan (your network) or --public (internet).

  Workers  1 of 1 joined  0:21
    A  downloading ████████░░░░░░░░░░░░  41%  0.21 GB of 0.52 GB  11.7 MB/s
```

Share only that link. Opening it and clicking **Accept and contribute** is all a
contributor does; the page downloads its part of the model and checks the GPU.
Keep the tabs open; tabs on the same GPU share its memory. When every worker is
ready, the chat prompt opens and the web chat link and API details are printed.

Links work on this machine only by default. `--lan` shares them on your local
network (HTTPS with a self-signed certificate) and `--public` on the internet
through a Cloudflare quick tunnel. See
[sharing details](advanced.md#share-over-the-internet).

## 4. Chat or connect an app

Open the private chat link. It supplies the owner key and shows **Waiting for
contributors** until the model is ready. A line under the title shows speed,
tokens and each worker's state; the printed session page shows more.
`--open-chat` opens it on the server's
desktop; on a headless server, open the printed link on your own computer.

The **Connect another app** section shows the API base URL and model. Use the
owner key as the app's API key. See [app setup](api.md).

## Installation options

`--install-only` installs code without downloading models or starting a server.
`PYTHON` selects a Python executable; `YGARATY_INSTALL_DIR` selects an absolute
installation path; `YGARATY_WORKSPACE_ROOT` selects the parent of per-model
workspaces. Set these on the `bash` side of a curl pipeline.

`YGARATY_REF` selects a public Git branch, tag or commit on first install. For a
pinned install, use the same commit in the script URL and that variable. Reruns
reuse the installed version; for an upgrade, install into a new directory and
pass the existing `--workspace`. See [public bundles](public-bundles.md) for
custom model URLs and [model preparation](model-bundles.md) for local checkpoints.

## Optional WebRTC

With `YGARATY_TURN_HOST` and `YGARATY_TURN_SECRET` configured for your TURN
server, the same launch command automatically enables WebRTC between workers.
Use `--route turn` for TURN-only peer links, or `--route relay` for coordinator
relay. Chat and app URLs stay the same. See [TURN setup](turn.md).


(update-ygaraty)=
## Update Ygaraty

Stop your running server with Ctrl-C, then run:

```sh
ygaraty update
```

This installs latest `main` in a fresh environment, checks it, then switches the
existing command path to it. Workspaces, downloaded models, keys and checkpoints
are left untouched, including workspaces on an external drive. Failed downloads,
package installation or health checks leave the previous environment active.
Previous environments are retained under the installation's `.versions/` folder.

When a model preset moves to a new public bundle version, launch downloads and
validates it automatically in a separate cache inside the same workspace. Older
model caches, keys and session state are retained. Failed downloads do not
replace a working cache. Repeated launches reuse the verified version; selecting
an older bundle URL reuses its cache. This also applies to `--serve-weights`.
Contributors reconnect to the selected bundle and may need to fetch its weights.


Restart with the same launch command and `--workspace`, then reconnect your
contributors. This preserves files, not active connections or in-memory chats.
Training checkpoint compatibility across versions is not guaranteed; finish an
experiment on its original revision or use the same revision to resume it.

To select a particular revision:

```sh
ygaraty update --ref main
```

Replace `main` with a branch, tag or commit that includes the update command.
The updater manages installer-created environments; it does not rewrite a source
checkout or an arbitrary pip environment. It installs the server and preparation
dependencies; reinstall any additional optional packages you use afterward.

If your older installation does not recognize `update`, bootstrap it with:

```sh
curl -fsSL https://raw.githubusercontent.com/ygaraty/ygaraty/main/install.sh | bash -s -- --update
```

For a nondefault installation, pass `YGARATY_INSTALL_DIR` to that command's
`bash` process. Updating does not change PATH or move your workspaces.
