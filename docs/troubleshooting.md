# Troubleshooting

| Problem | What to do |
| --- | --- |
| `Python 3.12+ required` | Install Python 3.12 or newer with venv/pip, or set `PYTHON` to that executable. |
| Installation interrupted | Inspect the incomplete install directory. Choose a new absolute `YGARATY_INSTALL_DIR` to retry; the installer refuses to overwrite it. |
| `ygaraty` not found | Use `~/.local/share/ygaraty/venv/bin/ygaraty`, or add that bin directory to PATH. |
| Workspace is incomplete or belongs to another recipe | Keep it for inspection and select a new `--workspace`. Changing model preparation settings does not overwrite existing data. |
| Checkpoint does not match the profile | Check that GGUF, config and tokenizer belong to the intended model. Set the correct `--model`; custom names use inspected geometry. |
| Chat waits for contributors | Open the contributor link on each required device and accept contribution. Wait for loading and self-tests. The default model needs four stages. |
| Shared link only works on the server | Links are local by default. Add `--lan` for your network or `--public` for the internet. |
| Public link cannot be created | Install [`cloudflared`](https://github.com/cloudflare/cloudflared#installing-cloudflared) (or set `YGARATY_CLOUDFLARED`) and allow outbound HTTPS. Retry launch, or use `--lan` on a local network. |
| Public link expired or tunnel disconnected | Rerun launch and share the new link. Prepared models and keys are reused. |
| Chat through the public link is slow | Quick-tunnel traffic uses the relay. For sustained use, supply your own stable endpoint with `--public-url`, or share with `--lan`. |
| WebGPU is unavailable | Use a WebGPU-capable browser/device and a secure context: localhost or HTTPS. |
| API gives 401 | Use the contents of `keys/owner.key`, not a filename, contributor key, or full private chat URL. |
| API gives 400 | Use text Chat Completions, temperature 0 and only the supported fields. See [API](api.md). |
| API gives 503 | Check contributor readiness and keep their tabs open. Also inspect server output for worker or execution failures. |
| Browser does not open on the server | Open the printed private chat link manually on your computer. `--open-chat` acts on the server's desktop. |
| GPU runs out of memory | Use smaller stages, a smaller context or fewer slots in a new workspace. Weight budget is not a total peak-memory estimate. |

The self-hosted coordinator is experimental. Offline tests cover orchestration
and fake engines/devices; real-model startup, generated text, browser behavior,
latency and memory of the extracted product still need end-to-end validation.
