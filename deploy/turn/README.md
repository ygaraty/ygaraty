# TURN server example

[turnserver.conf.example](turnserver.conf.example) is a coturn configuration
example for WebRTC contributor connections. It belongs to the public networking
infrastructure and is separate from the Go content-addressed block relay.

The file contains placeholders, not deployment credentials. Review its address,
certificate, peer restrictions, quotas and shared-secret settings for the target
host before use. The application issuing short-lived TURN credentials must share
the server's authentication secret. The `YGARATY_TURN_*` names in its comments
configure the public launcher: its default `--route auto` selects WebRTC when
both the host and secret are present.

This example does not install, start or configure a service. TURN remains optional
infrastructure for applications that implement WebRTC routing.

## Public deployment and credential helper

User-facing instructions live in [docs/turn.md](../../docs/turn.md).

The public credential issuer is `ygaraty.backends.webgpu.turn.TurnConfig`. It uses only
Python's standard library, preserves the deployed credential format, and has no
Django, cloud database or provider dependency. Its caller must authenticate the
contributor and verify session/slot ownership before issuing credentials.

The public factory supports relay, direct WebRTC and relay-only TURN routes.
TURN config is passed as a server-side dependency, never serialized in the
session contract. Chat/API ingress remains HTTP; only contributor links use
WebRTC. The host issues per-browser credentials at join and requires them for
worker-specific operations on WebRTC sessions. ICE/signaling also verify current
stage ownership. Signaling long polls use data transport capacity rather than
occupying the reserved control pool.

Tests: `python -m unittest discover -s tests/webgpu -p 'test_turn.py'`.
They verify signatures, expiry, environment settings, scope and secret redaction;
they do not launch coturn or establish a live WebRTC allocation.
