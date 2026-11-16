# Install, invite, chat

## 1. Choose a public Spaces model

Use Linux or macOS with Python 3.12+ (`venv`/`pip`), Git, curl and OpenSSH.
The examples download prepared models from our public DigitalOcean Spaces
storage. No checkpoint directory, cloud credentials or numerical preparation is
needed. The server keeps only metadata (about 26 MB); each contributor's
browser downloads its share of the weights directly from Spaces.

Choose 1, 2, 3 or 4 workers for any model. The examples below use two.
Before downloading weights, launch calculates and prints GPU RAM per worker
from the matching Spaces bundle. With the default 4,096-token context and one
chat slot, recommendations are about 0.78 GB each for Qwen3.5-0.8B, 2.37 GB
each for S5-Nano or 3.85 GB each for Qwen3.5-9B, including headroom. See [all configurations](public-bundles.md).

## 2. Install and launch

Qwen3.5-0.8B (smallest and fastest):

```sh
curl -fsSL https://raw.githubusercontent.com/ygaraty/ygaraty/main/install.sh | bash -s -- --num-workers 2 --model qwen3.5-0.8b --open-chat
```

S5-Nano:

```sh
curl -fsSL https://raw.githubusercontent.com/ygaraty/ygaraty/main/install.sh | bash -s -- --num-workers 2 --model s5-nano --open-chat
```

Qwen3.5-9B:

```sh
curl -fsSL https://raw.githubusercontent.com/ygaraty/ygaraty/main/install.sh | bash -s -- --num-workers 2 --model qwen3.5-9b --open-chat
```

Run one at a time. The installer uses the code published to `main`. From this
checkout, use `YGARATY_SOURCE_DIR="$PWD" bash ./install.sh --num-workers 2 --model s5-nano`.

The installation is in `~/.local/share/ygaraty/venv`. The installer keeps separate
caches, keys and state in `~/.local/share/ygaraty/workspaces/MODEL/N-workers/`. Reruns
validate and reuse cached artifacts. Ctrl-C stops the server and public tunnel.

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
  Ygaraty  qwen3.5-0.8b · 1 worker · 4,096-token context · 1 chat at a time

  Open this link in Chrome or Edge on a computer with 1.02 GB of free GPU memory:

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
contributors** until the model is ready. `--open-chat` opens it on the server's
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
