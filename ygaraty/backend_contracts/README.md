# Ygaraty backend contracts

Distribution: `ygaraty.backend_contracts`. Import: `ygaraty.backend_contracts`.
Python 3.12+, standard library only, MIT. Version `0.1.0a1` is an initial
development contract, not a stable release or an inference implementation.

This package defines the boundary between an inference backend and its host.
The host may be a free self-host application or a private commercial platform.
The backend never needs an ORM model, billing service or provider credential.

## API map

| Module | Responsibility |
| --- | --- |
| `models` | Immutable model/configuration identity, session specification, readiness and effective capacities |
| `access` | Trusted host-issued grant scopes, expiry and policy ceilings |
| `chat` | Text requests, errors, event payloads and event sequence validation |
| `lifecycle` | Revocation scope and cancellation receipts |
| `protocols` | Backend factory/session/handle and injected host ports |
| `registry` | Explicit factory registration and selection |
| `contributors` | The contributor port: join, heartbeat, backend calls, leave, roster |
| `testing` | Conformance checks every backend runs against its own factory |
| `persistence` | Durable metadata records, terminal transitions and storage interfaces |

Import types from their owning modules. The package root exports
`BackendRegistry` and `CONTRACT_MAJOR`. Runtime value objects are frozen
dataclasses with immutable collections and strict validation. Protocols describe
implementation obligations; Python does not enforce those obligations at runtime.
`RuntimeServices` holds live injected ports and is not a persistable value object.

## Session and configuration identity

`SessionSpec` binds an opaque session ID and runtime epoch to `ModelIdentity`
and `BackendConfiguration`. Model identity includes checkpoint, tokenizer and
template digests plus precision. Digests are opaque identifiers at this layer;
artifact adapters must validate their algorithms, syntax and contents. Display
names alone must not select model bytes.

Configuration is a tagged, versioned immutable byte envelope, limited to 1 MiB.
It contains no secrets or executable Python. Its factory owns the schema decoder
and must reject unknown fields, unsupported versions and invalid bounds before
acquiring resources. The envelope alone is **not** validated backend configuration.
Typed WebGPU options belong in the future WebGPU package; private options belong
in their private implementation. No generic JSON/pickle object loader is supplied.

`EffectiveCapabilities` reports actual loaded bounds and supported history/routes.
It cannot grant permission. Effective admission limits are the intersection of
model/device capacity, backend policy and the host grant. A ready snapshot must
include capabilities; `open()` alone does not imply readiness. Snapshot revisions
are runtime revisions, separate from the storage record's optimistic-lock revision.

## Composition and lifecycle

The host explicitly registers approved factory instances in `BackendRegistry`.
Registration calls pure `describe()` and rejects duplicate IDs or unsupported
contract majors. It does not discover plugins, import backend names from user
input, initialize engines or select a fallback implementation.

`await registry.open(spec, services)` calls the selected factory's validation
before opening. Factories must also validate when opened directly and unwind any
partially acquired resources on failure. Opening may prepare resources; it never
rents infrastructure.

The backend exposes `snapshot()`, `admit(request, grant)`, `revoke(scope)` and
`close()`. Expected lifecycle is preparing → ready ↔ unavailable → draining →
closed, with failed for unrecoverable runtime errors. Implementations enforce
transitions; `SessionSnapshot` is a validated observation, not a state machine.

The host injects three narrow ports through `RuntimeServices`: `Clock`,
`AccessPolicy` and `RequestLedger`. Clocks separate wall-time expiry from monotonic
deadlines. Artifact resolution and telemetry ports will be designed with the first
real adapter, without adding generic service locators to this boundary.

## Contributors

Most backends run their model work on contributors: rented GPUs or volunteer
browsers that join a session. The host (Ygaraty's own server, or Kaa) owns
transport, authentication and slot reservations. The backend owns everything
inside a contributor call. A session that uses contributors also implements
`contributors()`, returning a `ContributorPort`; `contributor_port(session)`
returns it, or `None` for a self-contained backend.

| Step | Port method | Host route (Kaa) |
| --- | --- | --- |
| Take a reserved slot | `join(JoinTicket)` → `JoinReply` | `POST /api/join` |
| Stay alive, report loading | `heartbeat(id, payload)` → bool | `POST /api/heartbeat` |
| Backend operation (ready, claim work, send a result, ...) | `call(id, name, payload)` | `POST /api/NAME`, or `GET /api/recv` |
| Leave | `leave(id, reason)` | `POST /api/leave` |
| Show and persist slots | `roster()` → `RosterEntry` tuple | session status |
| Drop silent contributors | `expire()` | every few seconds |

Payloads are JSON-object bytes of at most 1 MiB; the host never decodes or edits
them. `operations()` lists a backend's call names, so a host needs one set of
routes for every backend. Rejections raise `ContributorError` with a fault:
`invalid` (400), `unsupported` (403), `not_ready` (409) or `gone` (410: join
again). The port is synchronous because hosts call it from request threads; each
method returns within a few seconds.

## Conformance

Each backend runs the shared checks against its own factory, with fakes:

```python
from unittest import IsolatedAsyncioTestCase
from ygaraty.backend_contracts.testing import BackendConformance, assert_stream

class MyBackendConformance(BackendConformance, IsolatedAsyncioTestCase):
    def make_factory(self): ...
    def make_spec(self): ...
    def make_services(self): ...
```

They check discovery without resources, strict configuration validation,
snapshots, idempotent close and the contributor port's rules. Pass chat streams in
a backend's own tests through `assert_stream`, which requires one valid,
terminated event sequence. WebGPU (this repository) and Kaa's vLLM backend run it.

## Authentication and admission

A session ID is an identifier, **not a credential**. The host authenticates a
bearer key or scoped ticket, checks membership and constructs `AccessGrant` from
trusted state. Never deserialize an HTTP body into a trusted grant.

`grant.require(...)` checks session/epoch, action and expiry using an explicit
clock value. It cannot prove authenticity or check revocation. `AccessPolicy`
must additionally validate current membership/key/grant versions and fail closed
when unavailable. Revalidate before queued dispatch and periodically during
running work; use `revoke()` to actively stop matching work. Revocation filters
intersect; an epoch-only scope targets all its work.

Admission must validate readiness, grants, model identity, supported capabilities,
token budget and conversation ownership. Reserve bounded capacity, persist an
unfinished request record, then return a handle. Roll back reservations on error.
The HTTP adapter must await admission before emitting streaming headers.
Queues must have finite capacity and deadlines; a queued handle is not proof that
execution has started. Backend implementations own fairness and state isolation by
`(session, epoch, principal, conversation)`.

## Requests and streaming

`ChatRequest` is text-only with explicit `full`, `append` or `reset` history mode.
Conversation IDs contain 1–64 ASCII letters, digits, underscores or hyphens.
Requests contain at most 128 nonempty messages, each at most 100,000 characters;
the final message is from the user. Empty messages are allowed only as an empty
tuple for reset. These are in-process envelope bounds, not a replacement for HTTP
byte limits or model-specific token budgets. The initial surface supports only
the existing greedy, single-output behavior; sampling/tool/multimodal options
are not implied. Existing HTTP normalization/clamping needs compatibility adapters.

A handle has one event consumer and monotonically numbered events starting at
zero: accepted, zero or more text deltas, then exactly one of completed, cancelled
or failed. Completed includes finish reason and token usage. Failed carries a
stable code and a host-safe message; implementations must sanitize it. Cancelled
includes a receipt distinguishing requested, acknowledged and timed-out stopping.
Sequence validation is explicit:

```python
from ygaraty.backend_contracts.chat import EventSequence

async def consume(handle, emit):
    sequence = EventSequence(handle.request_id)
    try:
        async for event in handle.events():
            sequence.accept(event)
            await emit(event)
        sequence.finish()
    finally:
        await handle.aclose()
```

`EventSequence` detects wrong identity, gaps, duplicate accepted/terminal events
and truncated streams. It does not deliver, buffer, persist or replay events.
Network disconnects may hide a terminal outcome; there is no exactly-once network
delivery or automatic generation retry. Direct browser transports will need an
adapter for their out-of-band text path; they are not implemented by this iterator.

Cancellation and close are idempotent control operations. Cancellation requested
or timed out is not proof that compute stopped. Retain resource accounting until
cleanup or quarantine uncertain state. Closing the response must close its handle.
Backpressure must bound event count and bytes without blocking cancellation behind
a full output queue. Preserve native async execution and WebGPU's bounded producer
bridge when implementing adapters.

## Persistence and the Coopy boundary

`SessionStore` owns session specifications/revisions and bounded request records.
`RequestLedger` is its narrower backend-facing admission/finish interface. The
protocol methods specify atomic duplicate rejection, current-epoch admission,
compare-and-swap close/restart, and idempotent terminal writes. Storage failures
propagate. Required admission writes must succeed before dispatch; failed terminal
writes must not prevent compute cleanup and require host reconciliation.

`RequestRecord.finish()` implements an immutable terminal transition. Repeating
the same outcome is idempotent; changing a terminal outcome raises `StoreConflict`.
The store must apply this under its own atomic mutation boundary. No concrete
store, crash durability or concurrent-writer correctness is implemented here.

On restart the host must atomically rotate to a never-used epoch and mark unfinished
old-epoch records interrupted **before** opening fresh runtime resources. Old
credentials and delayed outcomes cannot resurrect that work. Restored configuration
does not restore GPU state, KV caches, active streams or readiness.

The planned Coopy adapter belongs to the public self-host app, not this package:

- Pin a reviewed Coopy version and declare the serializer/schema migration policy.
- Use one writer with deterministic state mutations; do I/O and clock/ID generation
  outside journaled commands. Make durability acknowledgement explicit.
- Journal session/admission/terminal transitions, not every output token, heartbeat
  or tensor. Bound retention and checkpoint size; keep weights in artifact storage.
- Keep plaintext credentials, live handles and GPU buffers out of snapshots.
- Verify crash recovery and epoch reconciliation before claiming durable sessions.

Chat history, credential storage, catalog editing, retention administration and
usage/billing ledgers are outside this initial storage API. They must not be
smuggled into backend configuration or generic extension dictionaries.

## Versioning and current limits

`CONTRACT_MAJOR = 1` negotiates this Python interface; the package remains pre-release.
Configuration schemas, future JSON/HTTP/worker envelopes and inference plan formats
have separate versions. This package provides no wire codec and does not change
existing HTTP/SSE, browser worker bytes or compiled plans.

Implemented and offline-tested: value validation/immutability, grant scope/expiry,
event sequencing, terminal-record transitions, registry selection/validation and
imports without private or optional dependencies. Protocol compliance of real
engines and durable stores still needs authorized-host validation. The
`ygaraty.backends.webgpu` module now implements preparation, factory/session/request
adapters and a single-session HTTP host, verified with fake engines offline.
See the [user guide](../../docs/README.md). Coopy durability, a private native
adapter and actual end-to-end inference validation remain pending.

From the repository root:

```bash
python -m unittest discover -s tests/contracts -v
ruff check .
ruff format --check .
python -m pip wheel --no-index --no-deps --no-build-isolation --wheel-dir dist .
```

Tests need no third-party packages. Offline builds require local setuptools 77+
and packaging 24.2+. Backend extraction must freshly compile affected inference
plans, regenerate applicable outputs and verify state/buffer compatibility.
Numerical parity and latency/memory comparisons remain authorized-host gates.

These contracts are implemented here and ship in the single `ygaraty` wheel.
They import only the standard library; importing them does not load the
compiler or inference backend.
