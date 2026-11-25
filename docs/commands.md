# Command line

The `ygaraty` command has two ways to run a model, plus helpers to choose a
model and to check a running server.

| Command | What it does |
| --- | --- |
| `ygaraty models list` | The supported models: family, layers, public bundles (1–4 workers) and validation status. |
| `ygaraty models info NAME` | For 1–4 workers: layers, download and free GPU memory each worker needs, context and output limit. Reads the public manifests only. |
| `ygaraty launch` | Everything for a session in one command: model, keys, server, sharing, a live console and a chat prompt. |
| `ygaraty serve` | Only the server: no console, sharing or chat prompt. For services and scripts. |
| `ygaraty status [URL]` | A running server's state, workers and statistics. |
| `ygaraty contribute URL` | Run a contributor from the command line instead of a browser tab. |
| `ygaraty model …` | Inspect a checkpoint, prepare and validate bundles ([model bundles](model-bundles.md)). |
| `ygaraty init` | Create an owner and a contributor key in a directory. |

## Launch or serve?

Both start the same server for the same model and answer the same chat page,
session page and API. They differ in what they do around it.

`launch` is for a person at a terminal. It prepares the model (from a supported
model name, a public bundle, a local bundle or a checkpoint), creates keys,
can share the links on your network (`--lan`) or the internet (`--public`,
through a Cloudflare tunnel), shows the link to share and one live row per
worker while they load, and then opens a chat prompt with a status line pinned
to the top of the terminal. Ctrl-C stops it.

`serve` is for running the server under a service manager (systemd, a
container) or a script. It downloads a supported model or reads a prepared
bundle, prints its links once, then logs HTTP requests. It never opens a
tunnel, a browser or a prompt, and never prepares a checkpoint (use
`ygaraty model prepare` or `launch --prepare-only` first). Put it behind your
own HTTPS proxy and pass that address as `--public-url`.

| | `launch` | `serve` |
| --- | --- | --- |
| Model | `--model NAME`, `--bundle`, `--bundle-url` or `--checkpoint` | `--model NAME`, `--bundle` or `--bundle-url` |
| Workers | `--num-workers N` (required) | `--num-workers N`; with `--bundle`, the bundle's stage count |
| Workspace | `--workspace`, default `runs/MODEL/N-workers` | the same |
| Keys | created in the workspace | the same; `--keys DIR` to use your own |
| Sharing | `--lan`, `--public`, `--public-url` | `--public-url` (your own proxy) |
| Output | links, live worker rows, then a chat prompt | links once, then request logs |
| Preparing a checkpoint | yes (`--checkpoint`, `--prepare-only`) | no |

Options with the same name mean the same thing in both: `--model`,
`--num-workers`, `--workspace`, `--context`, `--memory-gb`, `--slots`,
`--host`, `--port`, `--route`, `--turn-host`, `--turn-secret-file`,
`--public-url`, `--open-status`, `--config` and `--save-config`.

```sh
# A person trying a model with two GPUs on the same network:
ygaraty launch --model s5-nano --num-workers 2 --lan

# The same model as a service, behind your own HTTPS proxy:
ygaraty serve --model s5-nano --num-workers 2 --host 127.0.0.1 --port 8765 \
  --public-url https://ai.example.com
```

## Choose a model

```sh
ygaraty models list            # supported models
ygaraty models list --all      # also experimental ones (local checkpoints)
ygaraty models info s5-nano    # needs per worker for 1–4 workers
ygaraty models info qwen3.5-9b --context 8192 --slots 2
```

`launch` and `serve` need `--model`; without it they list the supported models
and stop. With `--bundle`, `--bundle-url` or `--checkpoint`, `--model` is the
name apps use in API requests.

## Session page and status

Every server has a session page at `/`. It shows:

- the model and whether the session is ready;
- each worker: waiting, downloading (with %), checking the GPU or ready, and
  whether it is still connected;
- how workers are linked: the relay, or WebRTC direct or through TURN;
- statistics: tokens, speed, time to first token, chats and answers.

It needs a key by default. The links `launch` and `serve` print carry one: the
contributor link (`/#key=…`) and the session page link (`/#owner=…`, which also
shows the contributor link to copy). Start with `--open-status` to show the
status to anyone, without a key. It never shows keys or addresses.

The web chat and the terminal chat show a one-line summary. To check a
running server from a terminal:

```sh
ygaraty status                                   # http://127.0.0.1:8765
ygaraty status --workspace runs/s5-nano/2-workers
ygaraty status https://ai.example.com --key-file owner.key --json
ygaraty status https://192.168.1.20:8765 --workspace W --insecure   # --lan
```

`status` takes the key from `--key`, `--key-file`, `--workspace` or
`YGARATY_KEY`.

## Save settings and restart

`--save-config` writes the command's settings to `WORKSPACE/ygaraty.toml`.
Afterwards the same session starts with:

```sh
ygaraty launch --workspace runs/s5-nano/2-workers
# or, in that directory:
ygaraty launch
```

`launch` and `serve` read `--config FILE`, else `WORKSPACE/ygaraty.toml` when
`--workspace` is given, else `./ygaraty.toml`. Flags override the file. Keys are
option names without the leading dashes:

```toml
model = "s5-nano"
num-workers = 2
slots = 1
route = "relay"
lan = true
open-status = false
```

Unknown keys and invalid values are errors. Relative paths are relative to the
file's directory.

## Output limit

Public models answer with up to 4,096 tokens, in a 16,384-token context.
`ygaraty models info NAME` shows both. A request's `max_tokens` can ask for
fewer tokens, not more. Long answers take time: 4,096 tokens took about
5 minutes at 14 tokens per second on a laptop GPU.

For your own bundles, set the limit when preparing:

```sh
ygaraty model prepare ... --context 16384 --generate 4096
```

## Exit codes and JSON

| Code | Meaning |
| --- | --- |
| 0 | Success; for `status`, the session is ready |
| 1 | The command failed (invalid setting, missing file, server unreachable, key refused) |
| 2 | Invalid command line (unknown option, missing value) |
| 3 | `status` only: the server answers but the session is not ready |

`models list`, `models info`, `status`, `launch --prepare-only` and the
`model` subcommands print JSON with `--json` (the `model` subcommands always
do). With `--json`, errors are printed as `{"error": "..."}` on standard output
as well as on standard error, and progress lines go to standard error.
`contribute` exits with its contributor process's code.

`ygaraty bootstrap` is deprecated: use `ygaraty launch --checkpoint DIR
--model NAME --num-workers N`, with `--prepare-only` to prepare without
starting.
