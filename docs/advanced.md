# Advanced mode

Use `ygaraty launch --num-workers 2` to configure the model and start a foreground server in one
command. If the installer’s CLI is not on PATH, replace `ygaraty` below with
`~/.local/share/ygaraty/venv/bin/ygaraty`.

```sh
ygaraty launch --num-workers 2 --model s5-nano \
  --context 4096 --slots 1 --open-chat

ygaraty launch --num-workers 2 --model qwen3.5-9b \
  --context 4096 --slots 1 --open-chat
```

The command downloads the prepared public Spaces bundle and creates scoped keys,
starts the coordinator and an automatic HTTPS tunnel, prints
links and app settings, and reports when contributors make the session ready.
While it waits, chat remains available but sending is disabled; API completion
requests receive HTTP 503. Losing contributors makes the model unavailable
until the backend reports readiness again.

## Launch settings

| Flag | Meaning and default |
| --- | --- |
| `--checkpoint PATH` | Optional local preparation from GGUF/config/tokenizer; otherwise use a public preset. |
| `--bundle-url URL` | Override the preset with a public HTTPS prepared bundle. |
| `--num-workers N` | **Required.** Number of GPU workers, 1–26. Public presets support 1, 2, 3 or 4. Prepared bundles must have exactly this stage count; local splits must match it. `--workers` and `--contributors` are aliases. |
| `--bundle PATH` | Already-prepared `bundle.json`; skips checkpoint preparation. |
| `--model NAME` / `--model-id NAME` | API model ID, default `qwen3.5-9b`. Known names select presets; other names inspect the checkpoint and default to a single stage with an 8 GiB budget. |
| `--gguf NAME` | Filename inside the checkpoint, default `model.gguf`; first shard for split GGUF. |
| `--splits LIST` | Increasing layer boundaries from zero to the model's full layer count; at most 26 stages. Checkpoint preparation only. |
| `--context N` | Context tokens, default 4096 for public presets and preparation; existing capacity for explicit bundles. |
| `--generate N` | Prepared maximum output tokens, default 128; must be positive and smaller than context. Checkpoint preparation only. |
| `--memory-gb N` | Optional runtime budget override in GiB; must cover the calculated recommendation. Does not change the memory estimate. |
| `--slots N` | Concurrent conversation slots, 1–16; default 1. More slots need more memory. |
| `--workspace PATH` | Model, keys and state directory; CLI default `runs/MODEL/N-workers`, installer default `~/.local/share/ygaraty/workspaces/MODEL/N-workers`. |
| `--host ADDRESS` | Local listen address, default `127.0.0.1`. Automatic sharing works with this loopback default. |
| `--port N` | Listen port, default 8765. |
| `--route auto\|relay\|webrtc\|turn` | Default `auto`: WebRTC when TURN is configured or the links are local (no `--public`/`--public-url`), otherwise coordinator relay. On a local network browsers connect over their local addresses without TURN, and any pair whose link does not open within 30 s uses the relay. With `--public`, `webrtc` needs TURN. `turn` forces relay-only ICE and always needs TURN. |
| `--turn-host HOST` | Override `YGARATY_TURN_HOST`; hostname or IPv4 address. |
| `--turn-secret-file PATH` | Read the shared coturn secret from a protected file instead of `YGARATY_TURN_SECRET`. |
| `--lan [ADDRESS]` | Share on the local network: links use this machine's LAN address (or ADDRESS) over HTTPS, and the server listens on all interfaces. Browsers enable WebGPU only on secure pages, so a self-signed certificate is made in the workspace (`.lan/`, reused); each device accepts its warning once. Allow inbound TCP on the port in this machine's firewall. Contributors on the network use WebRTC without TURN. Cannot be combined with `--public` or `--public-url`. |
| `--public` | Share the links on the internet through a free [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/) (needs [`cloudflared`](https://github.com/cloudflare/cloudflared)). Off by default: without it no tunnel service is contacted and links work on this machine only. |
| `--local` | Run locally only. This is the default; the flag is kept for compatibility. |
| `--public-url URL` | Use an existing HTTPS endpoint instead of an automatic tunnel. No path, query, fragment or embedded credentials. HTTP is allowed for localhost. Cannot be combined with `--local`. |
| `--open-chat` | Open private chat in the server machine's default browser. Omit on headless hosts. |
| `--prepare-only` | Prepare and validate the model and keys, then exit. Computes numerical references. |
| `--no-terminal-chat` | Do not open the `>` chat prompt in the launch terminal when the model becomes ready. It is skipped automatically when input is not a terminal. After each answer a line shows time to first token, tokens and decode rate (tokens/s after the first token), and the route with WebRTC link state. `/net` shows each link's details, `/reset` starts a new conversation, `/quit` stops the server like Ctrl-C, and Ctrl-D closes only the prompt while the server keeps running. |
| `--request-log` | Also print HTTP requests and contributor session events. By default the terminal shows the contributor link and one live row per worker until the model is ready; events always go to `state/<session>/session.log`. |
| `--serve-weights` | Public bundles only. Download all weights to this machine and serve them to contributors. By default the server fetches only metadata (about 26 MB) and each contributor's browser downloads its weights directly from public storage, verifying every piece's SHA-256. |

Both public models support 1–4 workers with one slot and 4,096 context tokens
by default. Worker count selects the matching Spaces manifest; its split plan
and 256-token generation cap are preserved. See [memory estimates](public-bundles.md).

Optional `--checkpoint` preparation has separate profiles: S5-Nano uses two
16-layer stages, Qwen uses four eight-layer stages, and Qwen3-Coder-30B uses one
48-layer stage. Custom names inspect checkpoint geometry. Use explicit `--splits`
and `--memory-gb` when preparing a different layout.

Generation is currently greedy (`temperature=0`, one completion). Sampling,
tools, images, embeddings and the Responses API are not implemented. Per-request
`max_tokens` can reduce the output budget. These limitations apply to both
chat and connected apps.

## Share over the internet

With `--public`, launch starts `cloudflared tunnel --url http://127.0.0.1:PORT`
and prints its temporary `https://….trycloudflare.com` address once the address
answers. No Cloudflare account, domain, inbound firewall rule or proxy setup is
required. Install `cloudflared` from
[its releases](https://github.com/cloudflare/cloudflared#installing-cloudflared)
(`brew install cloudflared`, `sudo pacman -S cloudflared`, or the .deb/.rpm), or
point `YGARATY_CLOUDFLARED` at the binary. The local coordinator stays bound to
loopback.

The same generated origin serves contribution, private chat and the `/v1` API.
Ctrl-C closes the tunnel and server. If the tunnel disconnects or its domain
changes, launch stops with a clear error; rerun it and share the new link.
It does not silently switch contributors to a different address.

Quick tunnels are meant for testing: the address changes every launch and
Cloudflare gives no uptime guarantee. Weights do not go through the tunnel
(contributors download them from public storage); pages, model traffic and API
requests do, so workers use the coordinator relay. Measured with Qwen3.5-0.8B
and two workers: 6.5 tokens/s against 54 on the local network
([TESTED.md](https://github.com/ygaraty/ygaraty/blob/main/TESTED.md)). Cloudflare terminates HTTPS and forwards requests,
including API credentials; your model server and keys stay on your machine.
Without `--public`, no traffic goes through a tunnel service.

For an existing stable endpoint, pass `--public-url https://ai.example.com` to
skip automatic tunneling. This advanced override expects your endpoint already
to forward to the coordinator; it is optional, not part of normal setup.
`--prepare-only` does not create a tunnel or start a server.

Links put access keys in the URL fragment, which browsers do not send in HTTP
requests. Anyone with the contributor link can use its contributor credential;
anyone with the private chat link can use the owner API. The chat page removes
the fragment from its address bar after reading it and keeps the key in memory.

## Restart and reuse

Repeat the same checkpoint command to validate and reuse its bundle and keys.
Changing preparation settings requires a new workspace; existing model data is
never silently replaced. Host, port, public URL and slots can change on restart.
A restart creates a new session epoch; chat history is not persisted.

To restart without the original checkpoint:

```sh
ygaraty launch --num-workers 2 --bundle /srv/ygaraty/session/model/bundle.json \
  --workspace /srv/ygaraty/session --model qwen3.5-9b --memory-gb 4
```

Bundle splits and output limits are already baked in. Set the model ID and
runtime memory/slot settings explicitly when resuming from a bundle. To manage
keys, bundle preparation and serving separately, see [model bundles](model-bundles.md).

## Memory summary

Worker count is mandatory. After reading the manifest and before downloading
weights, launch prints weights, estimated GPU allocations and recommended RAM
for each worker, plus the total. The calculation includes KV/recurrent state,
scratch buffers and resident snapshots; the recommendation adds 20% + 256 MiB
headroom per worker. Context and chat slots affect the result. Local checkpoint
runs calculate memory after preparing their bundle.

The estimate is not a measured GPU peak or server system-RAM requirement.
Device self-tests must still pass. `--memory-gb` optionally sets a higher runtime
budget; it does not substitute a fixed number for the calculated requirement.

## WebRTC from the command line

Set TURN host/secret once, then keep using the same launch command. Or select
`--route webrtc` explicitly; `--route turn` keeps peer links on TURN only.
The chat page and OpenAI-compatible API still use HTTPS. Contributors perform
GPU inference in their browsers. See [TURN setup and route choices](turn.md).
