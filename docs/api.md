# Connect apps and use the API

## Start a server

After [installing Ygaraty](quickstart.md), launch a session:

```sh
ygaraty launch --model qwen3.5-9b --num-workers 2 \
  --workspace runs/api-session --public --open-chat
```

Launch prints the RAM required per worker, a contributor link, your private chat
link and the OpenAI-compatible URL ending in `/v1`. Public sharing requires
`cloudflared` ([setup](advanced.md)); omit `--public` for local-only access.
Open the contributor link yourself or share it with others, and wait for both
workers to load their weights. Keep the server and contributors running.

In a second terminal, from the same directory, configure your connection:

```sh
# Replace this with the /v1 URL printed by launch.
export OPENAI_BASE_URL="https://random-words.trycloudflare.com/v1"
export OPENAI_API_KEY="$(cat runs/api-session/keys/owner.key)"
```

Keep the owner key and private chat link private; share only the contributor link.

## Connect an app

In an app that supports a custom OpenAI-compatible Chat Completions provider, set:

| Setting | Value |
| --- | --- |
| Base URL | The printed URL ending in `/v1`, e.g. `https://random-words.trycloudflare.com/v1` |
| API key | Contents of your workspace's `keys/owner.key` |
| Model | The name passed to `--model`, default `qwen3.5-9b` |
| Temperature | `0` |

With `--public`, the tunnel supplies this public HTTPS address; no proxy setup
is needed. The address changes on a fresh launch, so update your app then.
Without it, use the printed `http://127.0.0.1:8765/v1` from this machine.

Some apps ask for the full chat URL instead: use
`https://ai.example.com/v1/chat/completions`. Apps that require tools, images,
Responses, nonzero temperature or other unsupported fields will not work with
this version. Browser-only apps on another origin need their own same-origin
proxy; the server does not enable cross-origin access.

The private chat page shows these connection settings under **Connect another
app**. Contributor keys cannot call the chat API. Owner keys cannot contribute.

## Discover the model

```sh
curl "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"
```

`GET /v1/models` returns the one configured model, even while it waits for
contributors. It requires the owner key.

## Send a message

```sh
curl -N "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5-9b","messages":[{"role":"user","content":"Hello!"}],"temperature":0,"max_tokens":64,"stream":true}'
```

Supported fields are `model`, text `messages` (`role`/`content`), `stream`,
`max_tokens`, `temperature: 0`, `n: 1` and optional `conversation_id`.
Unknown fields are rejected. Streaming uses SSE completion chunks and `[DONE]`;
`stream: false` returns a single completion. Every request needs the full message
history. To continue a conversation, also reuse `conversation_id`; omit it to
start a new conversation. Requests for the same conversation run one at a time.

## Readiness and cancellation

`GET /v1/session` with the owner bearer key returns session state and effective
capacity. Wait for `state: "ready"` before sending. The server returns 503 for
completion requests while contributors are unavailable, 429 when full, and 401
for a missing or incorrect owner key. Invalid or unsupported inputs return 400.

Streaming responses include `X-Request-ID`. Cancel a request with
`POST /v1/requests/{request_id}/cancel`, using the same owner key. Disconnecting
a stream also requests cancellation. Cancellation does not guarantee immediate
GPU termination; an uncooperative worker may make the session unavailable.

This is a text-only Chat Completions subset, not the entire OpenAI API. There is
no durable request recovery, account management or persistent chat history.
