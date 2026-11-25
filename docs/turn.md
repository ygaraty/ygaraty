# Host a TURN server

`ygaraty launch` supports WebRTC from the command line. Contributors still use
their browsers for GPU inference; the server command selects how those browsers
exchange intermediate model data. Chat and OpenAI-compatible clients keep using
the same HTTPS chat page and `/v1` endpoint.

## Launch with WebRTC

Configure your running coturn server once. Set `YGARATY_TURN_HOST` and securely
supply `YGARATY_TURN_SECRET` in the server environment. The secret must match
coturn's `static-auth-secret`. Then use the usual command:

```sh
ygaraty launch --model s5-nano --num-workers 2 --open-chat
```

The default `--route auto` selects WebRTC when TURN is configured, otherwise the
coordinator relay. A partially configured TURN environment fails with a setup
message rather than silently changing the intended route.

You can also select the route and supply the secret through a protected file:

```sh
ygaraty launch --model s5-nano --num-workers 2 --open-chat \
  --route webrtc --turn-host turn.example.org \
  --turn-secret-file /etc/ygaraty/turn.secret
```

Use the same flags with `--model qwen3.5-9b` and any supported worker count.
Do not put the shared secret directly in command arguments or share links.
`--turn-secret-file` reads the secret only, not the full coturn configuration.

| Route | Behavior |
| --- | --- |
| `auto` (default) | WebRTC with configured TURN; coordinator relay otherwise |
| `webrtc` | Direct WebRTC when possible, TURN when needed; requires TURN settings |
| `turn` | WebRTC with relay-only ICE candidates through TURN; requires TURN settings |
| `relay` | Coordinator relay; ignores TURN settings |

Both WebRTC modes retain coordinator fallback if peer links cannot be established
or fail. The contributor page displays actual link status, including fallback.
`--route turn` prevents direct peer candidates; it does not remove the coordinator
from the session. Direct WebRTC can reveal network addresses to fellow
contributors; the contribution page explains the chosen policy before consent.
A single worker has no inter-worker link, so WebRTC adds no inter-worker benefit.

TURN is separate from the automatic HTTPS share link. The normal launch command
still generates that link; an HTTP proxy or the launch SSH tunnel cannot provide
TURN's UDP relay port range. Launch connects to your configured TURN service;
it does not provision a public TURN server or open firewall ports.

## Deploy coturn

Use a Linux host with a public address, coturn installed, and a DNS name such as
`turn.example.org`. Obtain a TLS certificate for that name through your existing
certificate management process.

1. Copy [turnserver.conf.example](https://github.com/ygaraty/ygaraty/blob/main/deploy/turn/turnserver.conf.example) to
   `/etc/turnserver.conf`. Keep it readable only by the service account (0600).
2. Replace the example public IP, realm, server name and certificate paths.
   Generate a shared secret with `openssl rand -hex 32` and replace `CHANGE-ME`.
   Keep the actual configuration and secret outside source control.
3. Give the coturn service account access to the certificate/key and log directory.
   If the coordinator is on another host, add its address to `denied-peer-ip`.
   Retain the private/loopback address exclusions and quotas. Do not deny the
   TURN host's own public address: two relayed peers must be able to reach each
   other's allocations. Keep unrelated services on that host bound to loopback.
4. Allow inbound 3478/UDP, 3478/TCP, 5349/TCP and 49160–49260/UDP in both the
   host and provider firewall. Configure any external-IP/NAT mapping for the host.
5. Start or restart your coturn service with this configuration. Check its logs
   for successful listeners and certificate loading. Reload/restart as required
   by your certificate renewal process.

The example uses expiring credentials, no static user accounts, a two-hour
credential lifetime by default, and allocation/bandwidth quotas. Adjust quotas
for your deployment. A listening service alone does not demonstrate connectivity:
verify a real authenticated allocation and data transfer between remote peers,
including forced TURN relay and TLS transport where required.

## TURN settings

The public Python helper reads these server-side environment variables:

| Variable | Meaning |
| --- | --- |
| `YGARATY_TURN_HOST` | TURN hostname or IPv4 address; no scheme or port |
| `YGARATY_TURN_SECRET` | Same secret as coturn's `static-auth-secret`; 32+ characters |
| `YGARATY_TURN_TTL` | Credential lifetime in seconds; default 7200, range 60–43200 |
| `YGARATY_TURN_PORT` | UDP/TCP port; default 3478 |
| `YGARATY_TURN_TLS_PORT` | TLS port; default 5349; empty disables TLS URLs |

`ygaraty.backends.webgpu.turn.TurnConfig.from_env()` returns the configured issuer, or
`None` when the host or secret is absent. After authenticating a contributor and
checking their session/slot assignment, a host can call
`config.credentials(session_id, slot)` to obtain expiring ICE credentials.
The shared secret stays on the server. Returned credentials belong only to the
authorized contributor; do not publish them in logs, bundles or share links.

The public host issues ICE credentials only after contributor authentication,
per-browser authorization and a current stage-ownership check. Leaving or losing
the stage blocks new credentials; previously issued credentials expire according
to their TTL. Responses are marked `no-store`. WebRTC signaling is restricted to
neighbouring stages and retains the engine's size and rate limits.

Automated checks cover route selection, authentication, signaling and chat/API
compatibility with fake engines. A real coturn allocation, NAT traversal and
real-model multi-device inference still require deployment validation.
