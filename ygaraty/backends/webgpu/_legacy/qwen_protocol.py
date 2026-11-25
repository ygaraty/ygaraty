"""Shared Qwen/WebGPU HTTP protocol, token routing and chat state.

No model execution, model preparation, compiler, experiment launcher or sockets.
Prepared web deployments and source-distribution jobs share these methods.
"""

from ygaraty.inference_support.http import reply, stream

import json
import os
import queue
import secrets
import threading
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import numpy as np

STATIC = Path(__file__).resolve().parent.parent / "browser"
PAGE = "qwen35_contributor.mjs"
MODULES = {
    "chat_slots.mjs",
    "contributor_network.mjs",
    "contributor_loading.mjs",
    "spaces_download.mjs",
    PAGE,
    "qwen35_gpu.mjs",
    "dense_gpu.mjs",
    "kolibri_gpu.mjs",
    "inference_plan.mjs",
    "inference_schema.mjs",
    "inference_contracts.mjs",
    "inference_layout.mjs",
    "speculative.mjs",
    "qwen35_peer.mjs",
    "webgpu_kernels.mjs",
}
STAGE_NAMES = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Kolibri-1 plans use up to about 10 stages
)
# The matrix-vector kernel's rows per thread: 2 for one token, 4 for up to 8 tokens (one group). Same outputs
# bit for bit as one row per thread (experiment 104); 1.98x batched throughput on the 680M (108), 1.86x on two
# RTX 3060s (110), single tokens no slower. None: one row per thread (`--matvec off`, YGARATY_MATVEC=off).
MATVEC_DEFAULT = dict(single=2, rows=4, tokens=8)
# Experiments 121, 125, 127: the coalesced one-token kernel (16 threads per row, 4 per item; 81-99% of the 680M's
# copy roof, Qwen3.5-9B 23% faster there and about 20% on an A100, reference-exact within 2.5e-5), batches through
# its batched form (each row the lone conversation's outputs bit for bit; 2.2x at 2 conversations on an A100), and
# from 32 conversations the GEMM with 64x32 tiles (1.34x at 32 on the 680M). YGARATY_KERNELS=off: the kernels above.
KERNELS_DEFAULT = dict(
    matvec_c=dict(tpr=16, lanes=4),
    matvec_b=8,
    batch_gemm={"from": 32, "tile": dict(GR=64, GT=32, RR=4, RT=8)},
    q8_fast_up_to=8,
)  # experiment 132: Q8_0 vectorized for decode-sized calls (never slower)


def kernel_defaults(family="qwen35"):
    """Experiment 136 (DO-75): for Qwen3.5 also group one-token products of the same input into one dispatch
    (experiment 135; tokens identical, 11.5% shorter A100 step). YGARATY_GROUP_MATMUL=off: without it."""
    import os

    if os.environ.get("YGARATY_KERNELS") == "off":
        return {}
    out = dict(KERNELS_DEFAULT)
    if family == "qwen35" and os.environ.get("YGARATY_GROUP_MATMUL") != "off":
        out["group_matmul"] = True
    # Experiment 144 (DO-83): `repack: "auto"` (each contributor measures whether the exact repack pays) failed its
    # gate (the A100 measurement is too short to be stable), so it is not a default; a fixture may still set it.
    return out


DONE_BYTES = (
    4 << 20
)  # a contributor's final report (`done`): about 20,000 calls with digests
SELFTEST_TOLERANCE = (
    1e-5  # max |err| / max |ref|; the kernels are within 6e-7 on the 680M
)
# (tensor, rows (first, count), tokens[, kernel]): every matmul kernel (1 and 3 tokens
# matrix-vector, 9 tiled, 21 and 64 register-tiled, Q8_0 generic), each
# K-quant, unaligned Q6_K blocks and row counts that are not tile multiples.
SELFTEST_CASES = (
    ("blk.0.attn_qkv.weight", (0, 256), 1),  # Q5_K
    ("blk.0.attn_qkv.weight", (0, 256), 64),
    ("blk.0.attn_gate.weight", (0, 200), 3),  # Q4_K
    ("blk.0.attn_gate.weight", (0, 200), 21),
    ("blk.0.ffn_down.weight", (0, 130), 1),  # Q6_K, 12,288 columns
    ("blk.0.ffn_down.weight", (0, 130), 9),
    ("blk.0.ffn_down.weight", (0, 130), 64),
    ("blk.0.ssm_alpha.weight", (0, 32), 3),  # Q8_0
    ("output.weight", (1000, 1001), 5),  # Q6_K
    (
        "blk.0.ffn_down.weight",
        (0, 130),
        5,
        "matvec",
    ),  # DO-38's batched decode: matrix-vector, two token groups
    (
        "blk.0.attn_qkv.weight",
        (0, 256),
        8,
        "matvec",
    ),  # DO-25 checking passes: 8 tokens (one group of the
    (
        "blk.0.ffn_down.weight",
        (0, 130),
        9,
        "matvec",
    ),  # rows-per-thread kernel, WS-16) and 9 (two groups)
)


from .chat_contract import (
    QWEN_CHAT as QWEN_CHAT,
    CHAT_SEPARATORS as CHAT_SEPARATORS,
    CHAT_ENDS as CHAT_ENDS,
    chat_format,
    chat_message,
    encode_ids,
    chat_prompt as chat_prompt,
    validate_conversation_id,
)


def dense_selftest_cases(model):
    """The dense and kolibri1 families' self-test: every block type the checkpoint's layers use (its first tensor of that
    type), each through the matrix-vector (1, 3 tokens), tiled (21) and register-tiled (64) kernels, then the
    output head and DO-38's batched matrix-vector path."""
    first = {}
    layer_names = sorted(
        (
            n
            for n in model.tensors
            if n.startswith("blk.") and model.tensors[n].kind != "F32"
        ),
        key=lambda n: (int(n.split(".")[1]), n),
    )
    for name in layer_names:
        first.setdefault(model[name].kind, name)
    cases = []
    for name in list(first.values())[:3]:
        rows = (0, min(256, model[name].shape[0]))
        cases += [
            (name, rows, 1),
            (name, rows, 3),
            (name, rows, 21),
            (name, rows, 64),
        ]
    vocab = model["output.weight"].shape[0]
    first_row = min(
        1000, max(0, vocab - 1001)
    )  # rows 1000-2000 (not a multiple of 16), or what a small vocab has
    return tuple(
        cases
        + [
            ("output.weight", (first_row, min(1001, vocab - first_row)), 5),
            (
                layer_names[0],
                (0, min(130, model[layer_names[0]].shape[0])),
                5,
                "matvec",
            ),
        ]
    )


# Signaling limits (user_webrtc.md, section 4).
SIGNAL_QUEUE = 256  # waiting signals per contributor
SIGNAL_RATE, SIGNAL_BURST = (
    20.0,
    40.0,
)  # signals per second per contributor, token bucket
SDP_BYTES, CANDIDATE_BYTES = 16 * 1024, 512
ROUTE_BYTES = 8 * 1024
CANDIDATE_TYPES = {"host", "srflx", "prflx", "relay"}
LINK_STATES = {
    "new",
    "connecting",
    "connected",
    "disconnected",
    "failed",
    "closed",
}


class SignalRejected(Exception):
    """A signaling message refused, with the HTTP status for the page."""

    def __init__(self, status, text):
        super().__init__(text)
        self.status = status


class SendBusy(SignalRejected):
    """WS-51: the next stage's mailbox stayed full; the contributor sends the same numbered call again."""

    def __init__(self):
        super().__init__(429, "next stage busy; retry this send")


def neighbours(stage, stages):
    """The stages a stage links to directly: the next, the previous, and (last, first) for tokens."""
    return {(stage + 1) % stages, (stage - 1) % stages} - {stage}


def relay_only(candidate):
    """A private session's candidate line: TURN relay candidates only, and without the related address
    (`raddr`, the browser's own public address as the TURN server saw it), so peers learn no contributor's
    address. None: drop it."""
    words = candidate.split()
    try:
        if words[words.index("typ") + 1] != "relay":
            return None
    except (ValueError, IndexError):
        return None
    for key, blank in (("raddr", "0.0.0.0"), ("rport", "0")):
        if key in words and words.index(key) + 1 < len(words):
            words[words.index(key) + 1] = blank
    return " ".join(words)


def private_sdp(sdp):
    """An SDP with only relay candidates, their related addresses blanked (see `relay_only`)."""
    lines = []
    for line in sdp.split("\r\n"):
        if line.startswith("a=candidate:"):
            kept = relay_only(line[2:])
            if kept is None:
                continue
            line = "a=" + kept
        lines.append(line)
    return "\r\n".join(lines)


def clean_signal(data, sender, to, private=False):
    """A signaling message re-built from its allowed fields only: an SDP offer (from the lower stage of the
    pair) or answer, or one ICE candidate. `private`: relay candidates only, without related addresses."""
    if not isinstance(data, dict) or len(data) != 1:
        raise SignalRejected(400, "a signal is one sdp or one candidate")
    if "sdp" in data:
        sdp = data["sdp"]
        if (
            not isinstance(sdp, dict)
            or set(sdp) != {"type", "sdp"}
            or sdp["type"] not in ("offer", "answer")
            or not isinstance(sdp["sdp"], str)
            or len(sdp["sdp"]) > SDP_BYTES
        ):
            raise SignalRejected(400, "malformed sdp")
        if (sdp["type"] == "offer") != (sender < to):
            raise SignalRejected(
                403, "the lower stage of a pair makes the offer"
            )
        return {
            "sdp": {
                "type": sdp["type"],
                "sdp": private_sdp(sdp["sdp"]) if private else sdp["sdp"],
            }
        }
    if "candidate" in data:
        c = data["candidate"]
        if (
            not isinstance(c, dict)
            or not set(c)
            <= {"candidate", "sdpMid", "sdpMLineIndex", "usernameFragment"}
            or not isinstance(c.get("candidate"), str)
            or len(c["candidate"]) > CANDIDATE_BYTES
            or not isinstance(c.get("sdpMid"), (str, type(None)))
            or len(c.get("sdpMid") or "") > 32
            or not (
                c.get("sdpMLineIndex") is None
                or (
                    type(c["sdpMLineIndex"]) is int
                    and 0 <= c["sdpMLineIndex"] < 16
                )
            )
            or not isinstance(c.get("usernameFragment"), (str, type(None)))
            or len(c.get("usernameFragment") or "") > 256
        ):
            raise SignalRejected(400, "malformed candidate")
        out = {
            k: c[k]
            for k in (
                "candidate",
                "sdpMid",
                "sdpMLineIndex",
                "usernameFragment",
            )
            if k in c
        }
        if (
            private and out["candidate"]
        ):  # an empty candidate ends gathering and passes as it is
            out["candidate"] = relay_only(out["candidate"])
            if out["candidate"] is None:
                raise SignalRejected(
                    403, "a private session takes TURN relay candidates only"
                )
        return {"candidate": out}
    raise SignalRejected(400, "a signal is one sdp or one candidate")


WAKE_SIGNAL = object()  # wakes a signaling long poll; filtered, never sent


def clean_links(links, stages, chat=False):
    """A stage's WebRTC link stats, kept to the documented fields: candidate types, protocol, round trip and
    byte and message counts, never addresses or free text. `chat`: the stage's link to the chat browser
    (peer -1), not its links to other stages."""
    if not isinstance(links, list) or len(links) > 2:
        raise ValueError("route: at most two links")
    out = []
    for link in links:
        if not isinstance(link, dict):
            raise ValueError("route: a link is an object")
        entry = {}
        if type(link.get("peer")) is int and (
            link["peer"] == -1 if chat else 0 <= link["peer"] < stages
        ):
            entry["peer"] = link["peer"]
        for key in ("open", "rejected"):
            if isinstance(link.get(key), bool):
                entry[key] = link[key]
        for key, allowed in (
            ("state", LINK_STATES),
            ("local", CANDIDATE_TYPES),
            ("remote", CANDIDATE_TYPES),
            ("protocol", {"udp", "tcp"}),
            ("relayProtocol", {"udp", "tcp", "tls"}),
        ):
            if link.get(key) in allowed:
                entry[key] = link[key]
        if link.get("dropped") in (
            "late",
            "stalled",
            "disconnected",
            "failed",
            "other",
        ):
            entry["dropped"] = link["dropped"]
        for key in (
            "bytesSent",
            "bytesReceived",
            "messagesSent",
            "messagesReceived",
            "relinks",
            "openedMs",
        ):
            if type(link.get(key)) is int and 0 <= link[key] < 2**53:
                entry[key] = link[key]
        if (
            isinstance(link.get("rttMs"), (int, float))
            and not isinstance(link["rttMs"], bool)
            and 0 <= link["rttMs"] < 1e6
        ):
            entry["rttMs"] = round(float(link["rttMs"]), 2)
        if "statsError" in link:
            entry["statsError"] = True
        out.append(entry)
    return out


def clean_route(route, stages):
    """The route stats in a `done` report: {atStart, atEnd, hiddenBytesSent, chat}, cleaned like /api/route.
    `chat` is the direct chat link of a first or last stage (DO-51), the only link of a one-stage session."""
    if route is None:
        return None
    if not isinstance(route, dict):
        return {"invalid": True}
    try:
        out = {
            k: clean_links(route[k], stages)
            for k in ("atStart", "atEnd")
            if route.get(k) is not None
        }
        if route.get("chat") is not None:
            out["chat"] = clean_links(route["chat"], stages, chat=True)
    except ValueError:
        return {"invalid": True}
    if (
        type(route.get("hiddenBytesSent")) is int
        and route["hiddenBytesSent"] >= 0
    ):
        out["hiddenBytesSent"] = route["hiddenBytesSent"]
    return out


# Per-stage backlog, including metadata. Reject rather than block coordinator
# threads when a receiver stalls. Retried receive batches are bounded by this too.
class BoundedMailbox(queue.Queue):
    def __init__(self, max_bytes=32 * 1024 * 1024, max_items=256):
        super().__init__(maxsize=max_items)
        self.max_bytes = max_bytes
        self.pending_bytes = 0
        self.peak_bytes = 0  # WS-51 evidence: the most this mailbox held
        self.waits = 0  # sends that had to wait for room

    @staticmethod
    def item_size(item):
        return (
            len(item[2])
            + len(json.dumps(item[1]).encode())
            + len(item[0])
            + 64
        )

    def room(self, size, items=1):
        """Whether `items` more items of `size` bytes in all fit. An empty mailbox takes anything, so one call
        larger than the bound still passes (alone). Call with `not_full` held."""
        if self._qsize() == 0:
            return True
        return (
            self._qsize() + items <= self.maxsize
            and self.pending_bytes + size <= self.max_bytes
        )

    def wait_room(self, size, items=1, timeout=5.0):
        """WS-51 backpressure: wait until the receiver has read enough for `items` items of `size` bytes. False
        after `timeout` s; the sender is then told to retry the same numbered send."""
        deadline = time.monotonic() + timeout
        with self.not_full:
            if not self.room(size, items):
                self.waits += 1
            while not self.room(size, items):
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self.not_full.wait(left)
            return True

    def put(self, item, block=True, timeout=None):
        size = self.item_size(item)
        with self.not_full:
            if not self.room(size):
                raise SignalRejected(
                    429,
                    "Stage mailbox capacity reached; reconnect the contributor",
                )
            self._put((size, item))
            self.pending_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.pending_bytes)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def _get(self):
        size, item = super()._get()
        self.pending_bytes -= size
        return item


def mailbox_bytes(stage):
    """A stage's relay mailbox: room for a whole-context prefill of the largest hand-offs it can receive (every
    call of max_tokens until max_positions, plus slack), at least 32 MiB. DO-95: Inkling 2's hand-offs are several
    vectors per token on a float32 wire (64 tokens x 4 x 2304 x 4 bytes = 2.4 MB per call), so a long prompt's
    prefill from a faster upstream stage overran the fixed 32 MiB (and then a 64-call bound at 8k positions). The
    relay has no backpressure: an upstream stage may finish its whole prefill before the next stage reads any."""
    try:
        fixture, manifest = stage["fixture"], stage["manifest"]
        calls = -(-fixture.get("max_positions", 4096) // fixture.get("max_tokens", 64)) + 8
        plan = manifest.get("plan") or {}
        width = (manifest.get("config") or {}).get("hidden", 0) * max(1, plan.get("input_vectors") or 1)
        call = fixture.get("max_tokens", 64) * width * (4 if fixture.get("wire") == "f32" else 2)
    except (AttributeError, KeyError, TypeError):
        call = 0
    window = mailbox_calls()
    if window and call:  # WS-51: senders wait for room (backpressure), so a few calls are enough
        return window * call
    return max(32 * 1024 * 1024, calls * call)


def recv_wait_seconds():
    """How long a contributor's `recv` waits for a message before an empty answer (default 20 s). WS-54: a short
    wait (YGARATY_RECV_WAIT_MS, e.g. 100) answers every ~100 ms, so the connection that carries hidden states to the
    next stage never idles past TCP's retransmission timeout, and a hand-off does not restart in slow start."""
    value = os.environ.get("YGARATY_RECV_WAIT_MS")
    if not value:
        return 20.0
    return min(20.0, max(0.02, float(value) / 1000))


def mailbox_calls():
    """WS-51: the relay mailbox holds this many of the largest hand-off calls, and a sender waits (or is told to
    retry) while it is full. 0 restores DO-95's whole-context mailboxes with no waiting. YGARATY_MAILBOX_CALLS."""
    import os

    value = int(os.environ.get("YGARATY_MAILBOX_CALLS", "4"))
    if value < 0:
        raise ValueError("YGARATY_MAILBOX_CALLS must be 0 or more")
    return value


class SessionProtocol:
    def initialize_protocol(self, args, delayer=None):
        self.lock = threading.RLock()
        self.contributors = {}
        self.contributor_roster()
        self.owner = [None] * len(self.stages)
        self.inbox = [
            BoundedMailbox(max_bytes=mailbox_bytes(stage)) for stage in self.stages
        ]
        self.conv_tokens = [[] for _ in range(self.conversations)]
        self.conv_times = [[] for _ in range(self.conversations)]
        self.conv_done = set()
        self.tokens, self.token_times = (
            self.conv_tokens[0],
            self.conv_times[0],
        )  # conversation 0, as before DO-38
        self.delayer = delayer
        self.hop_bytes = [0] * len(self.stages)
        self.hosts = {}
        self.state, self.go_time, self.error = "waiting", None, None
        self.finished = threading.Event()
        self.started = time.time()
        self.log_lock = threading.Lock()
        self.log = (self.out / "session.log").open("a")
        # chat mode: one conversation kept in the contributors' caches
        self.im_end = self.tokenizer.token_to_id("<|im_end|>")
        self.stop_ids = set(self.eos) | (
            {self.im_end} if self.im_end is not None else set()
        )
        self.chat_lock = threading.Lock()
        self.turn, self.turns, self.turn_ids = None, [], []
        self.position, self.last_token = 0, None
        # Chat recovery: the tokens the stages have run (each completed turn's prompt and answer, except the
        # answer's last token, which starts the next prompt); after a stage is replaced every stage starts
        # empty and the next prompt carries this history (`replay`).
        self.history, self.replay, self.restarts, self.epoch = [], False, 0, 0
        # Retries: the last batch each stage's mailbox handed out (sequence number, items), sent again when
        # the browser says (ack) it did not get it.
        self.handed = [None] * len(self.stages)
        # DO-40 signaling: SDP offers, answers and ICE candidates, one bounded queue per contributor (a
        # replacement starts empty), from its neighbouring stages only.
        self.signal_boxes, self.signal_buckets = {}, {}
        # The first stage's report of the tokens it got straight from the last stage (`fed`), checked against
        # the tokens the last stage reported to us; chat turns are numbered so a late report is ignored.
        self.turn_id, self.fed, self.finished_turn, self.diverged = (
            0,
            {},
            None,
            False,
        )
        self.concurrent_chats = None
        if (
            args.chat
            and self.conversations > 1
            and not getattr(args, "production_chat", False)
        ):
            from .chat_slots import ChatSlots

            self.concurrent_chats = ChatSlots(self, self.conversations)

    def say(self, line):
        stamp = f"{time.time() - self.started:9.3f}"
        if getattr(self, "console", True):
            print(f"[{stamp}] {line}", flush=True)
        with getattr(self, "log_lock", threading.Lock()):
            if not self.log.closed:
                self.log.write(f"{stamp} {line}\n")
                self.log.flush()

    # --- protocol ---------------------------------------------------------

    def contributor_roster(self):
        from .contributor_roster import ContributorRoster

        # Lazy construction also supports lightweight offline protocol fixtures.
        with self.lock:
            if not hasattr(self, "_contributor_roster"):
                self._contributor_roster = ContributorRoster(
                    [
                        dict(
                            stage=STAGE_NAMES[k],
                            layers=stage.get("layers"),
                            pieces=stage.get("pieces", []),
                        )
                        for k, stage in enumerate(self.stages)
                    ],
                    database=(
                        getattr(self.args, "contributor_db", None)
                        or self.out / "contributors.sqlite"
                    )
                    if hasattr(self, "out")
                    else None,
                )
            return self._contributor_roster

    def expire_contributors(self):
        for record in self.contributor_roster().expire():
            self.leave(record["id"], "heartbeat expired")

    def leave(self, cid, reason="left session"):
        self.contributor_roster().retire(cid, reason)
        with self.lock:
            c = self.contributors.get(cid, {})
            stage = c.get("stage")
            # Reuse preparation slots only before the session has started.
            if self.state == "waiting":
                if stage is not None and self.owner[stage] == cid:
                    self.owner[stage] = None
                c.pop("blobs", None)
                return
        if stage is not None:
            self.fail(cid, reason)

    def join(self, body, peer, *, slot=None):
        if (
            not isinstance(body, dict)
            or len(json.dumps(body).encode()) > 32768
        ):
            raise SignalRejected(413, "Contributor metadata exceeds 32 KiB")
        self.expire_contributors()
        cid = secrets.token_hex(8)
        roster = self.contributor_roster()
        with self.lock:
            prefer = slot if slot is not None else body.get("prefer")
            reservation = roster.reserve(
                cid,
                prefer if type(prefer) is int else None,
                strict=slot is not None,
            )
            if self.state != "waiting":
                roster.retire(cid, "session is not accepting contributors")
                reservation["state"] = "former"
        if reservation["state"] not in {"preparing", "active"}:
            return dict(
                id=cid,
                accepted=False,
                reason="All required contributor slots are occupied",
                roster=roster.snapshot(),
            )
        with self.lock:
            self.contributors[cid] = dict(
                body,
                id=cid,
                peer=peer,
                joined=time.time() - self.started,
                stage=None,
                blobs={},
            )
        a = body.get("adapter", {})
        self.say(
            f"join {cid} from {peer}: {a.get('vendor')} {a.get('architecture')} {a.get('description') or ''}".rstrip()
        )
        return dict(
            id=cid,
            accepted=True,
            selftest=self.selftest,
            assignment=reservation["assignment"],
            roster=roster.snapshot(),
        )

    def store_blob(self, cid, name, data):
        sizes = {f"y{i}": ref.size * 4 for i, ref in enumerate(self.expected)}
        sizes.update(
            bits=self.f16_want.size * 2,
            wire=self.f16_want.size * 2,
            back=self.f16_want.size * 4,
            unwire=self.f16_want.size * 4,
        )
        if name not in sizes or len(data) != sizes[name]:
            raise SignalRejected(
                400, "Unknown self-test blob or incorrect byte length"
            )
        if sum(sizes.values()) > 64 * 1024 * 1024:
            raise SignalRejected(
                413, "Self-test exceeds the 64 MiB contributor budget"
            )
        roster = self.contributor_roster()
        with self.lock, roster.lock:
            c = self.contributors[cid]
            record = roster.records.get(cid, {})
            if (
                self.state != "waiting"
                or c.get("stage") is not None
                or "blobs" not in c
                or record.get("state") != "preparing"
            ):
                raise SignalRejected(
                    409, "Contributor is not accepting self-test uploads"
                )
            # Known names replace previous bytes; duplicate requests cannot grow memory.
            c["blobs"][name] = data
            roster.touch(cid)

    def check(self, cid):
        self.expire_contributors()
        roster = self.contributor_roster()
        reservation = roster.reserve(cid)
        if reservation["state"] not in {"preparing", "active"}:
            return dict(
                ok=False,
                reason="Contributor slot is no longer available",
                assignment=None,
            )
        roster.touch(cid)
        with self.lock:
            c = self.contributors[cid]
            blobs = c.pop("blobs", {})
        results, ok, reason = [], True, ""
        for i, ref in enumerate(self.expected):
            got = np.frombuffer(blobs.get(f"y{i}", b""), dtype=np.float32)
            if got.size != ref.size:
                ok, reason = (
                    False,
                    f"kernel case {i}: {got.size} outputs, expected {ref.size}",
                )
                results.append(dict(case=i, error=reason))
                continue
            rel = float(
                np.abs(got.reshape(ref.shape) - ref).max() / np.abs(ref).max()
            )
            known = (getattr(self, "selftest", None) or {}).get("cases", [])
            case = (
                known[i]
                if i < len(known) and isinstance(known[i], dict)
                else {}
            )
            label = case.get("name") or (
                SELFTEST_CASES[i][0]
                if i < len(SELFTEST_CASES)
                else f"case {i}"
            )
            tokens = case.get("tokens", ref.shape[0])
            results.append(
                dict(case=i, tensor=label, tokens=tokens, max_rel=rel)
            )
            if not rel <= SELFTEST_TOLERANCE:
                ok, reason = (
                    False,
                    f"kernel case {i} ({label}, {tokens} tokens): error {rel:.2e} of the largest output",
                )
        want = self.f16_want.view(np.uint16)
        nan = np.isnan(self.f16_want)
        f16 = {}
        for name in ("bits", "wire"):
            got = np.frombuffer(blobs.get(name, b""), dtype=np.uint16)
            same = got.size == want.size and bool(
                np.all(
                    (got == want)
                    | (nan & ((got & 0x7C00) == 0x7C00) & ((got & 0x3FF) != 0))
                )
            )
            f16[name] = same
            if not same:
                ok, reason = (
                    False,
                    f"float16 conversion ({'cache' if name == 'bits' else 'wire'}) differs from round to nearest even",
                )
        exact = self.f16_want.astype(np.float32)
        for name in ("back", "unwire"):
            got = np.frombuffer(blobs.get(name, b""), dtype=np.float32)
            same = got.size == exact.size and bool(
                np.all(
                    (got.view(np.uint32) == exact.view(np.uint32))
                    | (np.isnan(got) & np.isnan(exact))
                )
            )
            f16[name] = same
            if not same:
                ok, reason = (
                    False,
                    f"float16 to float32 ({'cache' if name == 'back' else 'wire'}) is not exact",
                )
        c["selftest"] = dict(ok=ok, reason=reason, kernels=results, f16=f16)
        worst = max((r.get("max_rel", 1.0) for r in results), default=None)
        assignment = None
        if ok:
            with self.lock, roster.lock:
                roster.touch(cid)
                free = (
                    [reservation["slot"]]
                    if self.owner[reservation["slot"]] is None
                    else []
                )
                if free and self.state == "waiting":
                    k = free[0]
                    self.owner[k] = cid
                    c["stage"] = k
                    self.signal_boxes[cid] = queue.Queue(SIGNAL_QUEUE)
                    s = self.stages[k]
                    assignment = dict(
                        stage=k,
                        manifest=s["manifest"],
                        fixture=s["fixture"],
                        delivery="spaces"
                        if getattr(self, "storage", None)
                        else "local",
                        hashes={n: self.hashes[n] for n in s["names"]},
                        assignment=reservation["assignment"],
                        file_bytes={n: self.files[n][2] for n in s["names"]},
                        total_bytes=sum(self.files[n][2] for n in s["names"]),
                    )
        if not ok or assignment is None:
            roster.retire(cid, reason or "no assignment available")
        verdict = "passed" if ok else f"FAILED ({reason})"
        where = (
            f", stage {STAGE_NAMES[c['stage']]}"
            if assignment
            else (", not needed (stages taken)" if ok else "")
        )
        self.say(
            f"self-test {cid}: {verdict}; worst kernel {worst:.1e}{where}"
        )
        return dict(ok=ok, reason=reason, assignment=assignment)

    def ready(self, cid, body):
        self.expire_contributors()
        roster = self.contributor_roster()
        with self.lock, roster.lock:
            c = self.contributors[cid]
            if (
                c["stage"] is None
                or self.owner[c["stage"]] != cid
                or self.state != "waiting"
            ):
                raise ValueError("No available assigned stage")
            roster.activate(cid)
            c["ready"] = dict(body, at=time.time() - self.started)
        self.say(
            f"ready {cid} (stage {STAGE_NAMES[c['stage']]}): load {body['loadMs'] / 1e3:.1f} s, "
            f"downloaded {body['downloadedBytes'] / 1e9:.2f} GB, cached {body['cachedBytes'] / 1e9:.2f} GB, "
            f"allocated {body['allocated'] / 1e9:.2f} GB"
            + (
                f"; repack {body['repack'].get('repack')}"
                if isinstance(body.get("repack"), dict)
                else ""
            )
        )
        with self.lock:
            if self.state == "waiting" and all(
                o is not None and "ready" in self.contributors[o]
                for o in self.owner
            ):
                self.state, self.go_time = "running", time.time()
                go = True
            else:
                go = False
        if go:
            self.say(
                f"all {len(self.stages)} stages ready: go ({len(self.prompt)} prompt tokens, up to {self.args.generate} generated)"
            )
            fresh = [
                k
                for k, o in enumerate(self.owner)
                if not self.contributors[o].get("started")
            ]
            for k in fresh:  # after a restart, only the new browsers start
                self.contributors[self.owner[k]]["started"] = True
                self.inbox[k].put(("go", dict(epoch=self.epoch), b""))
            if (
                self.args.route == "webrtc"
            ):  # the running neighbours of a new browser link to it again
                for k, o in enumerate(self.owner):
                    if k not in fresh:
                        for new in sorted(
                            neighbours(k, len(self.stages)) & set(fresh)
                        ):
                            self.inbox[k].put(
                                (
                                    "relink",
                                    dict(epoch=self.epoch, stage=new),
                                    b"",
                                )
                            )

    def deliver(self, k, item):
        if getattr(self, "delayer", None):
            self.delayer.put(self.inbox[k], item)
        else:
            self.inbox[k].put(item)

    def send(self, cid, kind, meta, body, n=None):
        c = self.contributors[cid]
        if (
            c.get("stage") is None
            or int(meta.get("epoch", 0) if isinstance(meta, dict) else 0)
            < self.epoch
        ):
            return  # sent before the stages were reset for a replaced stage: stale
        if n is not None:  # a retried send that arrived the first time
            with self.lock:
                if n <= c.get("sent", -1):
                    return
        self.contributor_roster().touch(cid)
        k = c["stage"]
        route = self.stages[k]["fixture"]["plan_route"]
        if (
            kind in ("hidden", "hiddens")
            and mailbox_calls()
            and not getattr(self, "delayer", None)
        ):
            # WS-51 backpressure: hold the sender until the next stage has read enough. Its send chain is
            # serial, so it computes ahead by at most what its own process keeps; the relay keeps a few calls.
            items = len(meta["items"]) if kind == "hiddens" else 1
            size = len(body) + 512 * items
            if not self.inbox[route["next"]].wait_room(size, items):
                self.backpressure = getattr(self, "backpressure", 0) + 1
                raise SendBusy()
        if n is not None:  # taken: a retry of the same number is dropped from here on
            with self.lock:
                if n <= c.get("sent", -1):
                    return
                c["sent"] = n
        if kind in ("hidden", "hiddens") and route["output"] != "hidden":
            raise ValueError(
                "Plan does not allow hidden output from this stage"
            )
        if kind == "hidden":
            self.hop_bytes[k] += len(body)
            self.deliver(route["next"], ("hidden", meta, body))
            return
        if (
            kind == "hiddens"
        ):  # one decode position for each of several conversations
            self.hop_bytes[k] += len(body)
            size = len(body) // len(meta["items"])
            for i, item in enumerate(meta["items"]):
                item = dict(
                    item,
                    epoch=meta.get("epoch", 0),
                    chat_lease=item.get("chat_lease", meta.get("chat_lease")),
                    chat_turn=item.get("chat_turn", meta.get("chat_turn")),
                )
                if (
                    "pseq" in meta
                ):  # a relayed copy of a direct-link message: the receiver drops it if it has it
                    item = dict(item, pseq=meta["pseq"], psub=i)
                self.deliver(
                    route["next"],
                    ("hidden", item, body[i * size : (i + 1) * size]),
                )
            return
        if (
            kind == "fed"
        ):  # the first stage: tokens it received straight from the last stage
            if k != 0 or self.args.route != "webrtc":
                raise ValueError(
                    "only the first stage of a direct session reports fed tokens"
                )
            return self.check_fed(meta)
        if kind not in ("token", "tokens") or route["output"] != "token":
            raise ValueError(f"stage {k} cannot send {kind!r}")
        items = (
            meta["items"]
            if kind == "tokens"
            else [dict(token=meta["token"], conv=meta.get("conv", 0))]
        )
        if meta.get("resend"):
            # Tokens the last stage sent on its direct link (and we recorded as `direct`) that the link lost:
            # only forwarded to the first stage, which drops any it already got and applies its stop rule.
            if self.args.route != "webrtc" or not isinstance(
                meta.get("pseq"), int
            ):
                raise ValueError(
                    "a resend needs a direct session and its sequence number"
                )
            with self.lock:
                for i, item in enumerate(items):
                    packet = dict(meta, **item)
                    pool = getattr(self, "concurrent_chats", None)
                    if pool is not None and not pool.accepts(packet):
                        continue
                    if getattr(
                        self, "concurrent_chat", None
                    ) and not self.concurrent_chat.current(packet):
                        continue
                    forwarded = dict(
                        token=item["token"],
                        conv=item.get("conv", 0),
                        pseq=meta["pseq"],
                        psub=i,
                    )
                    if pool is not None:
                        forwarded.update(
                            epoch=self.epoch,
                            chat_epoch=packet["chat_epoch"],
                            chat_turn=packet["chat_turn"],
                        )
                    if "generation" in packet:
                        forwarded.update(
                            generation=packet["generation"], epoch=self.epoch
                        )
                    self.deliver(0, ("token", forwarded, b""))
            return
        if (
            meta.get("direct")
            and self.args.route != "webrtc"
            and len(self.stages) != 1
        ):
            raise ValueError(
                "only a direct session or a one-stage session (DO-61) reports tokens it fed itself"
            )
        # A context switch can advance the epoch after the admission check above.
        # Recheck while holding the same lock as cache reset before touching a turn.
        with self.lock if self.args.chat else nullcontext():
            if int(meta.get("epoch", 0)) < self.epoch:
                return
            for item in (
                items
            ):  # `direct`: the last stage also sent tokens to the first
                if getattr(self, "concurrent_chat", None):
                    self.concurrent_chat.token(
                        dict(meta, **item),
                        item["token"],
                        forward=not meta.get("direct"),
                    )
                    continue
                pool = getattr(self, "concurrent_chats", None)
                if pool is not None and not pool.accepts(dict(meta, **item)):
                    continue
                self.on_token(
                    item.get("conv", 0),
                    item["token"],
                    forward=not meta.get("direct"),
                )
                if meta.get("direct"):
                    self.compare_fed(item.get("conv", 0))

    def recorded_tokens(self, conv):
        if self.args.chat:  # the turn in progress, or the one that just ended (its last report may come late)
            return (
                self.turn["tokens"]
                if self.turn is not None
                else self.finished_turn
            )
        return self.conv_tokens[conv]

    def check_fed(self, meta):
        if getattr(self, "concurrent_chat", None):
            return self.concurrent_chat.fed(meta)
        pool = getattr(self, "concurrent_chats", None)
        if pool is not None:
            if pool.accepts(meta):
                return pool.channel(meta.get("conv", 0)).check_fed(meta)
            return
        tokens, conv = meta.get("items"), meta.get("conv", 0)
        if (
            not isinstance(tokens, list)
            or len(tokens) > 4096
            or any(type(t) is not int for t in tokens)
            or type(conv) is not int
            or not 0 <= conv < self.conversations
        ):
            raise ValueError("malformed fed report")
        if self.args.chat and meta.get("turn") != self.turn_id:
            return  # a report from an earlier turn
        with self.lock:
            pending = self.fed.setdefault(conv, [])
            if len(pending) + len(tokens) > self.args.generate:
                raise SignalRejected(
                    429, "Token report exceeds generation budget"
                )
            pending.extend(tokens)
        self.compare_fed(conv)

    def compare_fed(self, conv):
        """The first stage must have received exactly the tokens the last stage reported to us; a last stage
        that sends one answer to the user and another to the pipeline stops the turn."""
        pool = getattr(self, "concurrent_chats", None)
        if pool is not None:
            channel = pool.channel(conv)
            if channel is not None:
                return channel.compare_fed(conv)
            return
        fed, recorded = self.fed.get(conv), self.recorded_tokens(conv)
        if not fed or recorded is None:
            return
        n = min(len(fed), len(recorded))
        if fed[:n] == recorded[:n]:
            return
        self.say(
            f"conversation {conv}: the first stage got different tokens from the ones the last stage reported"
        )
        self.fed[conv] = []
        if self.args.chat:
            if (
                self.turn is not None
            ):  # chat() resets the stages when the turn ends with this error
                self.turn["events"].put(
                    dict(
                        error="contributors disagreed about the answer; the conversation was reset"
                    )
                )
            else:
                self.diverged = True  # the stages' caches differ from the history: rebuilt before the next turn
        else:
            with self.lock:
                self.state, self.error = (
                    "failed",
                    "contributors disagreed about the generated tokens",
                )
            self.finished.set()

    def signal(self, cid, body):
        """DO-40: a WebRTC signaling message (SDP or ICE candidate) from a contributor to a neighbouring stage,
        checked and re-built (user_webrtc.md, section 4), queued for the contributor holding that stage.
        {delivered: False} when no browser holds that stage yet (the signal is dropped)."""
        if self.args.route != "webrtc":
            raise SignalRejected(403, "this session does not use direct links")
        c = self.contributors[cid]
        stage = c.get("stage")
        if (
            stage is None
            or not isinstance(body, dict)
            or type(body.get("to")) is not int
        ):
            raise SignalRejected(400, "no stage to signal")
        to = body["to"]
        if to not in neighbours(stage, len(self.stages)):
            raise SignalRejected(403, "signals go to neighbouring stages only")
        data = clean_signal(
            body.get("data"),
            stage,
            to,
            private=getattr(self.args, "ice_policy", "all") == "relay",
        )
        now = time.monotonic()
        with self.lock:
            tokens, at = self.signal_buckets.get(cid, (SIGNAL_BURST, now))
            tokens = min(SIGNAL_BURST, tokens + (now - at) * SIGNAL_RATE)
            if tokens < 1:
                self.signal_buckets[cid] = (tokens, now)
                raise SignalRejected(429, "too many signals")
            self.signal_buckets[cid] = (tokens - 1, now)
            target = self.owner[to]
            box = self.signal_boxes.get(target) if target is not None else None
            if (
                box is None
            ):  # nobody holds that stage yet: the sender offers again after go (reoffer)
                return dict(delivered=False)
        self.contributor_roster().touch(cid)
        try:
            box.put_nowait(dict(sender=stage, epoch=self.epoch, data=data))
        except queue.Full:
            raise SignalRejected(
                429, "the other stage is not reading its signals"
            ) from None
        return dict(delivered=True)

    def signals_for(self, cid, wait=20):
        """The signaling messages waiting for this contributor (long-poll, [] after `wait` s)."""
        c = self.contributors[cid]
        if c.get("stage") is None:
            raise ValueError("no stage")
        if getattr(self, "state", None) == "cancelled":
            return []  # stopped: never hold shutdown with a fresh long poll
        with self.lock:
            box = self.signal_boxes.setdefault(cid, queue.Queue(SIGNAL_QUEUE))
        try:
            items = [box.get(timeout=wait)]
        except queue.Empty:
            return []
        while True:
            try:
                items.append(box.get_nowait())
            except queue.Empty:
                return [item for item in items if item is not WAKE_SIGNAL]

    def wake_signal_waiters(self):
        """Release signaling long polls at shutdown (the marker is never sent)."""
        boxes = getattr(self, "signal_boxes", None)
        if not boxes:
            return  # no WebRTC signaling (or a partly built session)
        with self.lock:
            boxes = list(boxes.values())
        for box in boxes:
            try:
                box.put_nowait(WAKE_SIGNAL)
            except queue.Full:
                pass  # a full box does not block its reader

    def note_route(self, cid, body):
        """A stage's WebRTC link stats (candidate types, round trip, bytes), checked and without addresses."""
        c = self.contributors[cid]
        if c.get("stage") is None or not isinstance(body, dict):
            raise ValueError("no stage")
        links = clean_links(body.get("links"), len(self.stages))
        chat = body.get("chat")
        if chat is not None:
            if (
                not isinstance(chat, list)
                or len(chat) > 1
                or (chat and c["stage"] not in (0, len(self.stages) - 1))
            ):
                raise ValueError("route: chat is one endpoint link")
            # The same whitelist; the chat peer is -1, outside the worker stage range.
            if any(
                not isinstance(item, dict) or item.get("peer") != -1
                for item in chat
            ):
                raise ValueError("route: invalid chat peer")
            c["chat_route"] = clean_links(chat, len(self.stages), chat=True)
        else:
            c["chat_route"] = (
                None  # Older clients did not report chat connectivity.
            )
        previous = c.get("route")
        c["route"] = links
        c["route_at"] = time.time()

        def signature(items):
            return [
                (
                    item.get("peer"),
                    item.get("open"),
                    item.get("local"),
                    item.get("remote"),
                    item.get("dropped"),
                )
                for item in (items or [])
            ]

        if previous is not None and signature(previous) == signature(links):
            return
        self.say(
            f"route {cid} (stage {STAGE_NAMES[c['stage']]}): "
            + ", ".join(
                f"{STAGE_NAMES[l['peer']] if 'peer' in l else '?'} {'open' if l.get('open') else 'closed'}"
                f" {l.get('local', '?')}-{l.get('remote', '?')} {l.get('protocol', '')}"
                + (f" {l['rttMs']:.0f} ms" if "rttMs" in l else "")
                + (
                    f", opened after {l['openedMs'] / 1e3:.1f} s"
                    if "openedMs" in l
                    else ""
                )
                + (f", dropped ({l['dropped']})" if "dropped" in l else "")
                for l in links
            )
        )

    def on_token(self, conv, token, forward=True):
        if self.args.chat:
            pool = getattr(self, "concurrent_chats", None)
            channel = pool.channel(conv) if pool is not None else self
            if channel is not None:
                return channel.chat_token(token, forward)
            return
        now = time.time()
        with self.lock:
            if conv in self.conv_done:
                return  # DO-25: the rest of a checked group after the conversation's last token
            self.conv_tokens[conv].append(token)
            self.conv_times[conv].append(now - self.go_time)
            n = len(self.conv_tokens[conv])
            total = sum(map(len, self.conv_tokens))
        if conv == 0:
            text = self.tokenizer.decode([token]).replace("\n", "\\n")
            if n == 1:
                self.say(
                    f"first token after {now - self.go_time:.2f} s: {token} {text!r}"
                )
            elif n % 16 == 0 or n == self.args.generate:
                gaps = np.diff(self.token_times)
                self.say(
                    f"token {n}: {token} {text!r}; median {np.median(gaps) * 1e3:.0f} ms per token"
                    + (
                        f" (all conversations: {total} tokens)"
                        if self.conversations > 1
                        else ""
                    )
                )
        # The first stage's rule (its fixture's stop_ids, <|im_end|> included): a direct session's first stage
        # stops a conversation there on its own, so the record must end it too (DO-48: they hung otherwise).
        if n >= self.args.generate or token in getattr(
            self, "stop_ids", self.eos
        ):
            with self.lock:
                self.conv_done.add(conv)
                last = len(self.conv_done) == self.conversations
                if last:
                    self.state = "stopping"
            if last:
                for box in self.inbox:
                    box.put(("stop", {}, b""))
        elif forward:
            self.deliver(0, ("token", dict(token=token, conv=conv), b""))

    # --- chat -------------------------------------------------------------

    def chat_tokens(self, message, first):
        """The chat template's tokens for one user message, after the previous answer's last token, which
        was generated but not yet run through the stages. ChatML with the bundle's chat format (`first`: what
        a conversation starts with, `generation`: the assistant's opening; see `chat_format`), Qwen3.5's with
        thinking off when the bundle has none."""
        fmt = chat_format(getattr(self, "chat_style", None))
        text = chat_message("user", message, fmt) + fmt["generation"]
        if first:
            return encode_ids(self.tokenizer, fmt["first"] + text)
        ids = [self.last_token]
        if fmt.get("style") == "tagged":  # DO-95: an assistant turn closes with its end token (Inkling 2's EOS)
            close = self.tokenizer.token_to_id(fmt["assistant_end"])
            if self.last_token != close:
                ids.append(close)
            return ids + encode_ids(self.tokenizer, text)
        if self.last_token != self.im_end:  # the answer was cut at --generate
            ids.append(self.im_end)
        return ids + encode_ids(self.tokenizer, fmt["end"] + text)

    def chat(self, message, reset=False, cancel_event=None, **kwargs):
        if getattr(self, "concurrent_chat", None):
            yield from self.concurrent_chat.chat(
                message, reset, cancel_event, **kwargs
            )
            return
        pool = getattr(self, "concurrent_chats", None)
        if pool is not None:
            yield from pool.chat(message, reset, cancel_event, **kwargs)
        else:
            yield from self._chat_serial(
                message, reset, cancel_event, **kwargs
            )

    def _chat_serial(
        self,
        message,
        reset=False,
        cancel_event=None,
        *,
        prompt_tokens=None,
        max_tokens=None,
        direct_lease=None,
        conversation_id=None,
    ):
        """One turn: yields events (dicts) as the answer's tokens arrive."""
        with self.chat_lock:
            # One worker KV cache, bounded independent logical histories. Selection and
            # generation share the same lock, so queued requests cannot swap live state.
            if cancel_event is not None and cancel_event.is_set():
                return
            selected = validate_conversation_id(
                conversation_id
                if conversation_id is not None
                else ("api" if prompt_tokens is not None else "legacy-browser")
            )
            if not hasattr(self, "chat_histories"):
                self.chat_histories = {}
                self.active_conversation = "legacy-browser"

            def snapshot():
                return dict(
                    history=list(self.history),
                    last_token=getattr(self, "last_token", None),
                    finished_turn=getattr(self, "finished_turn", None),
                )

            if selected != self.active_conversation:
                self.chat_histories[self.active_conversation] = snapshot()
                if (
                    selected not in self.chat_histories
                    and len(self.chat_histories) >= 16
                ):
                    yield dict(
                        error="This session has reached its 16-conversation limit. Create a new session."
                    )
                    return
                saved = self.chat_histories.get(
                    selected,
                    dict(history=[], last_token=None, finished_turn=None),
                )
                self.history = list(saved["history"])
                self.last_token = saved["last_token"]
                self.finished_turn = saved["finished_turn"]
                self.active_conversation = selected
                self.drop_stage_state()  # advances epoch; old workers' packets are discarded
                self.diverged = False
            limit = self.args.generate if max_tokens is None else max_tokens
            try:
                if cancel_event is not None and cancel_event.is_set():
                    return
                if self.state != "running":
                    yield dict(
                        error=f"the contributors are not ready (state {self.state})"
                    )
                    return
                if self.diverged:  # a finished turn's tokens disagreed: the stages start again from the history
                    self.diverged = False
                    self.drop_stage_state()
                if reset and (self.position or self.history or self.replay):
                    self.reset_conversation()
                    yield dict(reset=True)
                if not message and prompt_tokens is None:
                    yield dict(done=True, position=self.position)
                    return
                tokens = (
                    list(prompt_tokens)
                    if prompt_tokens is not None
                    else self.chat_tokens(
                        message, self.position == 0 and not self.history
                    )
                )
                replayed = self.replay
                if replayed:  # every stage restarted empty: the conversation so far comes first
                    tokens = self.history + tokens
                    self.replay = False
                if self.position + len(tokens) + limit > self.capacity:
                    self.reset_conversation()
                    yield dict(
                        reset=True,
                        reason=f"the conversation reached the capacity of {self.capacity} positions",
                    )
                    tokens = (
                        list(prompt_tokens)
                        if prompt_tokens is not None
                        else self.chat_tokens(message, True)
                    )
                    if len(tokens) + limit > self.capacity:
                        yield dict(
                            error=f"the message is {len(tokens)} tokens; at most {self.capacity - limit}"
                        )
                        return
                events = queue.Queue()
                start = time.time()
                start_clock = time.monotonic()
                self.turn = dict(
                    limit=limit,
                    events=events,
                    tokens=[],
                    times=[],
                    start=start,
                    prompt=len(tokens),
                    position=self.position,
                    prompt_ids=list(tokens),
                    history_before=[] if replayed else list(self.history),
                )
                self.position += len(tokens)
                self.say(
                    f"chat turn {len(self.turns) + 1}: {len(tokens)} prompt tokens at position {self.turn['position']}"
                )
                with self.lock:
                    self.turn_id += 1
                    self.fed = {}
                # `limit` and `turn`: in a direct session the first stage stops the turn itself, by this rule
                prompt_packet = (
                    "prompt",
                    dict(limit=limit, turn=self.turn_id),
                    np.asarray(tokens, dtype=np.uint32).tobytes(),
                )
                direct = getattr(self, "chat_direct", None)
                invitation = (
                    direct.prepare(
                        direct_lease,
                        tokens,
                        self.turn_id,
                        self.epoch,
                        limit,
                        conversation_id=selected,
                    )
                    if direct
                    else None
                )
                if invitation:
                    yield invitation
                else:
                    self.inbox[0].put(prompt_packet)
                shown = ""
                last_event = time.monotonic()
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    if invitation and direct.relay_due():
                        self.inbox[0].put(prompt_packet)
                    try:
                        event = events.get(timeout=0.25)
                    except queue.Empty:
                        if (
                            time.monotonic() - last_event
                            < self.args.turn_timeout
                        ):
                            continue
                        yield dict(
                            error=f"no token for {self.args.turn_timeout} s"
                        )
                        return
                    last_event = time.monotonic()
                    if event is None:
                        break
                    if "error" in event:
                        yield event
                        return
                    text = self.tokenizer.decode(
                        [
                            t
                            for t in self.turn["tokens"]
                            if t not in self.stop_ids
                        ]
                    )
                    if (
                        not text.endswith("\ufffd") and text != shown
                    ):  # wait for whole UTF-8 characters
                        yield dict(text=text[len(shown) :])
                        shown = text
                t = self.turn
                gaps = np.diff(t["times"])
                record = dict(
                    prompt_tokens=t["prompt"],
                    start_position=t["position"],
                    generated=len(t["tokens"]),
                    total_s=max(0, time.monotonic() - start_clock),
                    tokens_per_s=(len(t["times"]) - 1)
                    / (t["times"][-1] - t["times"][0])
                    if len(t["times"]) > 1 and t["times"][-1] > t["times"][0]
                    else None,
                    prompt_transport=(direct.prompt_transport() or "unknown")
                    if invitation
                    else "coordinator",
                    first_token_s=t["times"][0] - start
                    if t["times"]
                    else None,
                    ms_per_token=float(np.median(gaps) * 1e3)
                    if gaps.size
                    else None,
                    stopped=(
                        "end"
                        if t["tokens"] and t["tokens"][-1] in self.stop_ids
                        else "length"
                    ),
                )
                self.turns.append(record)
                if getattr(self.args, "logit_digests", False) or getattr(
                    self.args, "record_turn_ids", False
                ):  # WS-27: diagnostic runs only
                    self.turn_ids.append(
                        dict(
                            prompt_ids=list(t["prompt_ids"]),
                            tokens=list(t["tokens"]),
                        )
                    )
                self.history = (
                    t["history_before"] + t["prompt_ids"] + t["tokens"][:-1]
                )
                self.finished_turn = t["tokens"]
                self.turn = None
                self.say(
                    f"chat turn {len(self.turns)}: {record['generated']} tokens, first after "
                    f"{record['first_token_s'] or 0:.2f} s, {record['ms_per_token'] or 0:.0f} ms per token ({record['stopped']})"
                )
                yield dict(
                    record, done=True, position=self.position, full_text=shown
                )
            finally:
                if getattr(self, "chat_direct", None):
                    self.chat_direct.finish()
                if self.turn is not None and self.state == "running":
                    # An interrupted stream leaves partial KV state on the stages.
                    # Keep completed history, discard stale messages, and replay on
                    # the next turn using the same epoch/reset protocol as recovery.
                    self.drop_stage_state()
                self.turn = None
                self.chat_histories[selected] = snapshot()

    def drop_stage_state(self):
        """Every stage clears its caches (a reset of a new epoch: what was sent before it is dropped), and the
        next turn brings the conversation so far (`history`) through the stages again."""
        with self.lock:
            self.epoch += 1
            self.turn = None
            for box in self.inbox:
                while True:
                    try:
                        box.get_nowait()
                    except queue.Empty:
                        break
                box.put(("reset", dict(epoch=self.epoch), b""))
            self.handed = [None] * len(self.inbox)
            self.position, self.replay = 0, bool(self.history)

    def chat_token(self, token, forward=True):
        t = self.turn
        if t is None:
            return
        if len(t["tokens"]) >= t.get("limit", self.args.generate):
            return  # late/hostile reports cannot grow a completed turn
        t["tokens"].append(token)
        t["times"].append(time.time())
        if token in self.stop_ids or len(t["tokens"]) >= t.get(
            "limit", self.args.generate
        ):
            self.last_token = token
            t["events"].put(dict(token=token))
            t["events"].put(None)
        else:
            self.position += 1
            if forward:  # in a direct or one-stage session (DO-61) the first stage already has it
                self.deliver(
                    0, ("token", dict(token=token), b"")
                )  # --relay-delay holds it like a hand-off
            t["events"].put(dict(token=token))

    def reset_conversation(self):
        self.epoch += 1
        for box in self.inbox:
            box.put(("reset", dict(epoch=self.epoch), b""))
        self.position, self.last_token = 0, None
        self.history, self.replay = [], False
        self.say("chat: conversation reset")

    def done(self, cid, body):
        self.contributor_roster().retire(cid, "finished")
        c = self.contributors[cid]
        c.pop("blobs", None)
        if c.get("stage") is None:
            return
        if isinstance(body, dict) and "route" in body:
            body = dict(
                body, route=clean_route(body["route"], len(self.stages))
            )
        c["done"] = body
        self.say(
            f"done {cid} (stage {STAGE_NAMES[c['stage']]}): {len(body['calls'])} calls"
        )
        with self.lock:
            if all(
                o is not None and "done" in self.contributors[o]
                for o in self.owner
            ):
                self.state = "finished"
                self.finished.set()

    def end_chat(self):
        """End a chat session gracefully: every stage gets `stop` and reports `done` (its calls, timings and, with
        --logit-digests, every token's logits digest; WS-27), then the session is `finished`. False when it is not
        a running chat with no turn in progress (a turn's end, or a failure, owns the stages then)."""
        with self.lock:
            if (
                not self.args.chat
                or self.state != "running"
                or self.turn is not None
                or (
                    getattr(self, "concurrent_chat", None)
                    and self.concurrent_chat.active
                )
                or (
                    getattr(self, "concurrent_chats", None) is not None
                    and self.concurrent_chats.active()
                )
            ):
                return False
            self.state = "stopping"
        for box in self.inbox:
            box.put(("stop", {}, b""))
        self.say("chat: session ending; stages asked to stop")
        return True

    def restart(self, stage, text):
        """A chat lost a stage (its browser left or failed): instead of ending, the session waits for a
        browser to take that stage. The other stages keep their browsers and weights; they clear their
        caches (a `reset` of a new epoch: anything they sent before it is dropped), and once the new browser
        is ready the next message brings the conversation so far (`history`) through every stage."""
        with self.lock:
            self.state = "waiting"
            if getattr(self, "concurrent_chat", None):
                self.concurrent_chat.abort(
                    "A contributor disconnected. Retry after reconnection."
                )
            self.restarts += 1
            self.epoch += 1
            lost = self.owner[stage]
            if lost:
                self.contributors[lost]["stage"] = None
            self.owner[stage] = None
            for k, box in enumerate(self.inbox):
                while True:  # messages of the lost turn
                    try:
                        box.get_nowait()
                    except queue.Empty:
                        break
                if k != stage:
                    box.put(("reset", dict(epoch=self.epoch), b""))
                    if self.args.route == "webrtc" and k in neighbours(
                        stage, len(self.stages)
                    ):
                        box.put(
                            (
                                "unlink",
                                dict(epoch=self.epoch, stage=stage),
                                b"",
                            )
                        )  # its link to the lost stage
            self.handed[stage] = None
            self.state, self.go_time = "waiting", None
            if getattr(self, "concurrent_chats", None) is not None:
                self.concurrent_chats.interrupt(
                    f"stage {STAGE_NAMES[stage]} left; retry after its replacement"
                )
            self.position, self.replay = 0, bool(self.history)
            self.say(
                f"stage {STAGE_NAMES[stage]} lost ({text[:200]}); waiting for a browser to take it "
                f"(restart {self.restarts}; {len(self.history)} tokens of conversation kept)"
            )
        if self.turn is not None:
            self.turn["events"].put(
                dict(
                    error=f"stage {STAGE_NAMES[stage]} left; the conversation resumes when "
                    "a browser takes its place"
                )
            )

    def fail(self, cid, text):
        self.contributor_roster().retire(cid, text[:200])
        self.say(f"error from {cid}: {text[:2000]}")
        with self.lock:
            contributor = self.contributors.get(cid, {})
            contributor.pop("blobs", None)
            stage = contributor.get("stage")
            if stage is None:
                return  # a browser that never got a stage does not stop the session
            if self.state == "waiting":
                if self.owner[stage] == cid:
                    self.owner[stage] = None
                return  # a failed preparation frees its slot for another visitor
            if self.state in {"finished", "failed"}:
                return
            if self.args.chat and self.state == "running":
                recover = True
            else:
                recover = False
                self.state, self.error = (
                    "failed",
                    f"{cid} (stage {STAGE_NAMES[stage]}): {text}",
                )
        if recover:
            return self.restart(stage, text)
        if self.turn is not None:
            self.turn["events"].put(
                dict(error=f"stage {STAGE_NAMES[stage]} stopped: {text[:300]}")
            )
        for record in self.contributor_roster().snapshot()["contributors"]:
            self.contributor_roster().retire(record["id"], "session failed")
        self.finished.set()

    def heartbeat(self, cid):
        """A contributor's heartbeat. One retired for missing heartbeats that is in fact alive (a background tab
        whose timers the browser slowed) gets its stage back if the session is still waiting and the stage is
        free. False: its reservation has ended for good."""
        roster = self.contributor_roster()
        try:
            roster.touch(cid)
            return True
        except ValueError:
            pass
        with self.lock:
            c = self.contributors.get(cid, {})
            stage = c.get("stage")
            if (
                self.state != "waiting"
                or stage is None
                or self.owner[stage] not in (None, cid)
            ):
                return False
            if not roster.revive(
                cid, "active" if "ready" in c else "preparing"
            ):
                return False
            self.owner[stage] = cid
        self.say(
            f"{cid} is back (its heartbeats were late): stage {STAGE_NAMES[stage]} kept"
        )
        return True

    def note_progress(self, cid, data):
        """A heartbeat may carry the browser's loading progress ({"progress": {phase, loaded, total,
        cached}}), shown on the coordinator page."""
        try:
            p = json.loads(data or b"{}").get("progress")
        except (ValueError, AttributeError):
            return
        if not isinstance(p, dict):
            return
        clean = {
            k: p[k]
            for k in ("loaded", "total", "cached")
            if isinstance(p.get(k), (int, float))
        }
        if isinstance(p.get("phase"), str):
            clean["phase"] = p["phase"][:40]
        clean["at"] = time.time()
        self.contributors[cid]["progress"] = clean

    def status(self):
        self.expire_contributors()
        roster = self.contributor_roster().snapshot()
        for record in roster[
            "contributors"
        ]:  # loading progress the browsers reported
            progress = self.contributors.get(record["id"], {}).get("progress")
            if progress:
                record["progress"] = dict(
                    progress, age_s=round(time.time() - progress["at"], 1)
                )
        active_ids = {
            record["id"]
            for record in roster["contributors"]
            if record["state"] == "active"
        }
        with self.lock:
            return dict(
                chat_direct=self.args.route == "webrtc"
                and bool(self.args.chat)
                and getattr(self, "concurrent_chats", None) is None,
                roster=roster,
                state=self.state,
                error=self.error,
                tokens=sum(map(len, self.conv_tokens)),
                generate=self.args.generate,
                chat=bool(self.args.chat),
                position=self.position,
                capacity=self.capacity,
                concurrent_limit=getattr(
                    getattr(self, "concurrent_chat", None), "slots", 1
                ),
                active_conversations=len(self.concurrent_chat.active)
                if getattr(self, "concurrent_chat", None)
                else int(self.turn is not None),
                chat_slots=self.conversations if self.args.chat else 0,
                active_chats=len(self.concurrent_chat.active)
                if getattr(self, "concurrent_chat", None)
                else (
                    sum(
                        c.busy for c in self.concurrent_chats.channels.values()
                    )
                    if getattr(self, "concurrent_chats", None)
                    else int(self.turn is not None)
                ),
                turns=sum(
                    len(c.turns)
                    for c in self.concurrent_chats.channels.values()
                )
                if getattr(self, "concurrent_chats", None)
                else len(self.turns),
                # WS-51: each stage's relay mailbox (bound, most held, sends that waited) and retries told
                mailboxes=[
                    dict(bound=box.max_bytes, peak=box.peak_bytes, waits=box.waits)
                    for box in self.inbox
                ],
                send_retries=getattr(self, "backpressure", 0),
                # network: relay, or direct links (private: TURN only); links: each stage's last cleaned
                # report (candidate types, round trip, how it ended), never addresses
                network="relay"
                if getattr(self.args, "route", "relay") != "webrtc"
                else (
                    "private"
                    if getattr(self.args, "ice_policy", "all") == "relay"
                    else "direct"
                ),
                stages=[
                    dict(
                        stage=STAGE_NAMES[k],
                        contributor=o,
                        ready=o in active_ids
                        and "ready" in self.contributors[o]
                        if o
                        else False,
                        done="done" in self.contributors[o] if o else False,
                        links=self.contributors[o].get("route")
                        if o
                        and isinstance(self.contributors[o].get("route"), list)
                        else None,
                    )
                    for k, o in enumerate(self.owner)
                ],
                contributors=len(self.contributors),
            )

    # --- report -----------------------------------------------------------

    def conversation_report(self):
        out = []
        for c, (tokens, times) in enumerate(
            zip(self.conv_tokens, self.conv_times)
        ):
            gaps = np.diff(times) if len(times) > 1 else np.array([])
            out.append(
                dict(
                    conv=self.args.first_conversation + c,
                    prompt_tokens=len(self.prompts[c]),
                    tokens=tokens,
                    token_times_s=times,
                    first_token_s=times[0] if times else None,
                    inter_token_ms_median=float(np.median(gaps) * 1e3)
                    if gaps.size
                    else None,
                    inter_token_ms_p95=float(np.percentile(gaps, 95) * 1e3)
                    if gaps.size
                    else None,
                )
            )
        return out

    def throughput(self):
        """Total tokens/s: `steady`, from the moment every conversation has
        its first token to the moment the first one ends (tokens arriving in
        that window over its length); `overall`, all tokens from go to the
        last one."""
        times = [t for t in self.conv_times if t]
        if len(times) < self.conversations or not times:
            return None
        start, end = max(t[0] for t in times), min(t[-1] for t in times)
        inside = sum(1 for t in times for x in t if start < x <= end)
        last = max(t[-1] for t in times)
        return dict(
            steady_tokens_per_s=inside / (end - start)
            if end > start
            else None,
            window_s=end - start,
            window_tokens=inside,
            overall_tokens_per_s=sum(map(len, times)) / last if last else None,
            total_tokens=sum(map(len, times)),
            conversations=self.conversations,
        )

    def report(self):
        gaps = (
            np.diff(self.token_times)
            if len(self.token_times) > 1
            else np.array([])
        )
        contributors = []
        for c in self.contributors.values():
            entry = {k: v for k, v in c.items() if k not in ("blobs",)}
            if c.get("stage") is not None:
                entry["stage_name"] = STAGE_NAMES[c["stage"]]
                entry["layers"] = self.stages[c["stage"]]["layers"]
                calls = (c.get("done") or {}).get("calls", [])
                prefill = [
                    x for x in calls if x["tokens"] > 1
                ]  # decode steps are single positions
                steps = [x for x in calls if x["tokens"] == 1]
                if calls:
                    entry["summary"] = dict(
                        prefill_compute_ms=sum(
                            x["computeMs"] for x in prefill
                        ),
                        prefill_wait_ms=sum(x["waitMs"] for x in prefill),
                        step_compute_ms_median=float(
                            np.median([x["computeMs"] for x in steps])
                        )
                        if steps
                        else None,
                        step_wait_ms_median=float(
                            np.median([x["waitMs"] for x in steps])
                        )
                        if steps
                        else None,
                        steps=len(steps),
                        batched_tokens=sum(x.get("batch", 1) for x in steps),
                    )
            contributors.append(entry)
        report = dict(
            mode="session",
            state=self.state,
            error=self.error,
            splits=self.splits,
            kv=self.args.kv,
            wire="f16",
            chunk=self.args.chunk,
            prompt_tokens=len(self.prompt),
            generate=self.args.generate,
            capacity=self.capacity,
            prompt_source=self.prompt_source,
            route=(
                "webrtc data channels between contributors (coordinator: signaling, token record)"
                if self.args.route == "webrtc"
                else "through the coordinator (HTTP long-poll mailboxes)"
            ),
            ice_servers=[
                {k: v for k, v in x.items() if k == "urls"} for x in self.ice
            ],
            ice_policy=self.args.ice_policy,  # no TURN credentials
            selftest_tolerance=SELFTEST_TOLERANCE,
            time_to_first_token_s=self.token_times[0]
            if self.token_times
            else None,
            token_times_s=self.token_times,
            tokens=self.tokens,
            text=self.tokenizer.decode(self.tokens),
            inter_token_ms_median=float(np.median(gaps) * 1e3)
            if gaps.size
            else None,
            inter_token_ms_p95=float(np.percentile(gaps, 95) * 1e3)
            if gaps.size
            else None,
            tokens_per_s=float(1 / np.mean(gaps)) if gaps.size else None,
            hop_bytes=self.hop_bytes[:-1],
            contributors=contributors,
            hosts=self.hosts,
            chat_turns=self.turns,
            **({"chat_turn_ids": self.turn_ids} if self.turn_ids else {}),
            relay_delay_ms=self.args.relay_delay,
            batch=bool(self.args.batch),
            conversations=self.conversation_report(),
            throughput=self.throughput(),
        )
        self.write_report(report)
        return report

    def write_report(self, report):
        (self.out / "report.json").write_text(json.dumps(report, indent=2))






def authorized(h, session, query):
    key = h.headers.get("X-Key") or query.get("key", [""])[0]
    return secrets.compare_digest(key, session.key)


def respond(h, *args, **kwargs):
    """`h.reply` where the handler has one (this module's server, test doubles), else `reply`."""
    method = getattr(h, "reply", None)
    return (
        method(*args, **kwargs)
        if callable(method)
        else reply(h, *args, **kwargs)
    )


def session_get(h, session, path, query):
    """A GET on one session; `path` is relative to the session (the general coordinator serves a session
    under /sessions/<job>/, this module's own server at /)."""
    if path == "/":
        return respond(h, 200, JOIN_PAGE, "text/html; charset=utf-8")
    if path == "/coordinator":
        return respond(
            h,
            200,
            (STATIC / "session_coordinator.html").read_text(),
            "text/html; charset=utf-8",
        )
    if path == "/chat":
        return respond(h, 200, CHAT_PAGE, "text/html; charset=utf-8")
    if path == "/favicon.png":
        return respond(
            h, 200, (STATIC / "favicon.png").read_bytes(), "image/png"
        )
    if path in (
        "/brand.css",
        "/training.css",
        "/session_chat.mjs",
        "/session_coordinator.mjs",
    ):
        return respond(
            h,
            200,
            (STATIC / path[1:]).read_bytes(),
            "text/css" if path.endswith(".css") else "text/javascript",
        )
    if path[1:] in MODULES:
        return respond(
            h, 200, (STATIC / path[1:]).read_bytes(), "text/javascript"
        )
    if not authorized(h, session, query):
        return respond(h, 403, {"error": "missing or wrong session key"})
    if path.startswith("/f/"):
        name = path[3:]
        if name in session.test_files:
            return respond(
                h, 200, session.test_files[name], "application/octet-stream"
            )
        if name in session.files:
            if getattr(session, "storage", None):
                return respond(
                    h,
                    409,
                    {
                        "error": "Use the scoped download URL endpoint for Spaces pieces"
                    },
                )
            source, offset, length = session.files[name]
            h.send_response(200)
            h.send_header("Content-Type", "application/octet-stream")
            h.send_header("Content-Length", str(length))
            h.end_headers()
            with open(source, "rb") as handle:
                handle.seek(offset)
                left = length
                while left and (block := handle.read(min(1 << 20, left))):
                    h.wfile.write(block)
                    left -= len(block)
            return None
        return respond(h, 404, {"error": name})
    if path.startswith("/api/recv/"):
        if getattr(session, "state", None) == "cancelled":
            # A stopped session: answer at once so shutdown is not held up
            # by a fresh 20 s long poll.
            return respond(h, 410, {"error": "session stopped"})
        cid = path.rsplit("/", 1)[1]
        c = session.contributors.get(cid)
        if c is None or c.get("stage") is None:
            return respond(h, 404, {"error": "no stage"})
        k = c["stage"]
        ack = int(query["ack"][0]) if "ack" in query else None
        handed = session.handed[k]
        if ack is not None and handed is not None and ack < handed[0]:
            # The browser did not get the last batch (its request failed after we answered): send it again.
            seq, items = handed
            meta = [[kind, m, len(d)] for kind, m, d in items]
            return respond(
                h,
                200,
                b"".join(d for _, _, d in items),
                "application/octet-stream",
                (
                    ("X-Kind", "bundle"),
                    ("X-Meta", json.dumps(meta)),
                    ("X-Seq", str(seq)),
                ),
            )
        box = session.inbox[k]
        try:
            first = box.get(timeout=recv_wait_seconds())
        except queue.Empty:
            return respond(h, 204)
        if "all" not in query:
            kind, meta, data = first
            return respond(
                h,
                200,
                data,
                "application/octet-stream",
                (("X-Kind", kind), ("X-Meta", json.dumps(meta))),
            )
        items = [first]
        batch_bytes = BoundedMailbox.item_size(first)
        # Snapshot the item count and cap bytes: a concurrent sender must not
        # keep this drain alive forever or grow one response without a bound.
        for _ in range(min(box.qsize(), 255)):
            if batch_bytes >= 32 * 1024 * 1024:
                break
            try:
                item = box.get_nowait()
            except queue.Empty:
                break
            items.append(item)
            batch_bytes += BoundedMailbox.item_size(item)
        seq = (handed[0] if handed else 0) + 1
        if ack is not None:
            session.handed[k] = (seq, items)
        meta = [[kind, m, len(d)] for kind, m, d in items]
        return respond(
            h,
            200,
            b"".join(d for _, _, d in items),
            "application/octet-stream",
            (
                ("X-Kind", "bundle"),
                ("X-Meta", json.dumps(meta)),
                ("X-Seq", str(seq)),
            ),
        )
    if path.startswith("/api/signal/"):
        cid = path.rsplit("/", 1)[1]
        if (
            cid not in session.contributors
            or session.contributors[cid].get("stage") is None
        ):
            return respond(h, 404, {"error": "no stage"})
        return respond(h, 200, session.signals_for(cid))
    if path == "/api/status":
        return respond(h, 200, session.status())
    if path.startswith("/api/download/"):
        # Public piece URL; the browser checks size and SHA-256 itself.
        name = unquote(path.removeprefix("/api/download/"))
        try:
            return respond(h, 200, session.download(name))
        except (AttributeError, ValueError):
            return respond(h, 404, {"error": "no remote piece"})
    return respond(h, 404, {"error": path})


def session_post(h, session, path, query, data):
    """A POST on one session (`path` relative to it, `data` the raw body)."""
    if not authorized(h, session, query):
        return respond(h, 403, {"error": "missing or wrong session key"})
    parts = path.split("/")
    try:
        if (
            len(parts) > 2
            and parts[2] in {"join", "ready", "host", "heartbeat", "error"}
            and len(data) > 32768
        ):
            raise SignalRejected(413, "Contributor metadata exceeds 32 KiB")
        # `done` comes once per contributor with every call (and, with --logit-digests, every token's digest): a
        # 2048-token run or a long chat is far above 32 KiB, and was refused, losing the report (WS-27).
        if len(parts) > 2 and parts[2] == "done" and len(data) > DONE_BYTES:
            raise SignalRejected(
                413, f"Contributor report exceeds {DONE_BYTES >> 20} MiB"
            )
        if path == "/api/join":
            return respond(
                h, 200, session.join(json.loads(data), h.client_address[0])
            )
        if path == "/api/chat":
            request = json.loads(data or b"{}")
            return stream(
                h,
                session,
                session.chat(
                    str(request.get("message", "")).strip(),
                    bool(request.get("reset")),
                    cancel_event=getattr(h, "cancelled", None),
                    direct_lease=request.get("direct_lease"),
                    conversation_id=validate_conversation_id(
                        request.get("conversation_id", "legacy-browser")
                    ),
                ),
            )
        if path.startswith("/api/host/"):
            session.hosts[parts[3]] = json.loads(data)
            return respond(h, 200, {})
        if len(parts) < 3 or parts[1] != "api":
            return respond(h, 404, {"error": path})
        cid = parts[3] if len(parts) > 3 else ""
        if parts[2] == "error":
            session.fail(cid, data.decode(errors="replace"))
            return respond(h, 200, {})
        if cid not in session.contributors:
            return respond(h, 404, {"error": "unknown contributor"})
        if parts[2] == "heartbeat":
            session.expire_contributors()
            if not session.heartbeat(cid):
                return respond(
                    h,
                    410,
                    {
                        "error": "reservation ended: this browser's slot was given away; reload to rejoin"
                    },
                )
            session.note_progress(cid, data)
            return respond(
                h, 200, {"roster": session.contributor_roster().snapshot()}
            )
        if parts[2] == "leave":
            session.leave(cid)
            return respond(h, 200, {})
        if parts[2] == "blob":
            session.store_blob(cid, parts[4], data)
            return respond(h, 200, {})
        if parts[2] == "selftest":
            return respond(h, 200, session.check(cid))
        if parts[2] == "ready":
            session.ready(cid, json.loads(data))
            return respond(h, 200, {})
        if parts[2] == "send":
            # DO-85 (local harness only): `--send-delay MS` emulates the contributor's round trip on its sends,
            # half before the coordinator sees the message, half before the reply.
            delay = (getattr(session.args, "send_delay", 0) or 0) / 2e3
            if delay:
                time.sleep(delay)
            session.send(
                cid,
                query["kind"][0],
                json.loads(query.get("meta", ["{}"])[0]),
                data,
                int(query["n"][0]) if "n" in query else None,
            )
            if delay:
                time.sleep(delay)
            return respond(h, 200, {})
        if parts[2] == "done":
            session.done(cid, json.loads(data))
            return respond(h, 200, {})
        if parts[2] == "signal":
            if len(data) > SDP_BYTES + 1024:
                return respond(h, 413, {"error": "signal too large"})
            return respond(h, 200, session.signal(cid, json.loads(data)) or {})
        if (
            parts[2] == "route"
        ):  # DO-40: how each WebRTC link connected (candidate types, round trip)
            if len(data) > ROUTE_BYTES:
                return respond(h, 413, {"error": "route stats too large"})
            session.note_route(cid, json.loads(data))
            return respond(h, 200, {})
    except SendBusy as error:
        return respond(
            h, error.status, {"error": str(error), "retry": True}, extra=(("Retry-After", "0"),)
        )
    except SignalRejected as error:
        return respond(h, error.status, {"error": str(error)})
    except Exception as error:  # report to the page and the log, keep serving
        session.say(f"error handling {path}: {error!r}")
        return respond(h, 500, {"error": repr(error)})
    return respond(h, 404, {"error": path})


def handler(session):
    """A server for one session at / (tests and lightweight fixtures; `serve` runs the general coordinator)."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def reply(self, code, body=b"", mime="application/json", extra=()):
            return reply(self, code, body, mime, extra)

        def do_GET(self):
            url = urlparse(self.path)
            return session_get(self, session, url.path, parse_qs(url.query))

        def do_POST(self):
            url = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            return session_post(
                self,
                session,
                url.path,
                parse_qs(url.query),
                self.rfile.read(n) if n else b"",
            )

    return Handler


JOIN_PAGE = (STATIC / "session_contributor.html").read_text()
CHAT_PAGE = (STATIC / "session_chat.html").read_text()
