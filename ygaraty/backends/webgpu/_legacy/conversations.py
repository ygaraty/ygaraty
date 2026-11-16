"""Configurable isolated, interleaved chat slots over shared model weights.

A slot generation fences late packets without resetting other conversations.
Each turn replays its own text history into an empty slot; no state is shared.
"""

import queue
import threading
import time
import numpy as np
from ..buffers import TokenMailbox


class Conversations:
    def __init__(self, engine, slots=8):
        self.engine, self.slots = engine, slots
        self.lock = threading.RLock()
        self.active, self.histories = {}, {}
        self.generation = 0

    def current(self, meta):
        t = self.active.get(meta.get("conv", 0))
        return t if t and meta.get("generation") == t["generation"] else None

    def abort(self, reason):
        with self.lock:
            for t in self.active.values():
                if getattr(self.engine, "throughput", None):
                    self.engine.throughput.finish(t["slot"])
                t["events"].put({"error": reason})

    def compare(self, t):
        n = min(len(t["fed"]), len(t["tokens"]))
        if t["fed"][:n] != t["tokens"][:n]:
            t["events"].put(
                {
                    "error": "Contributors disagreed about the answer; retry this conversation."
                }
            )

    def fed(self, meta):
        with self.lock:
            t = self.current(meta)
            if not t:
                return
            items = meta.get("items")
            if (
                not isinstance(items, list)
                or any(type(x) is not int for x in items)
                or len(t["fed"]) + len(items) > t["limit"]
            ):
                raise ValueError("Invalid token receipt report")
            t["fed"].extend(items)
            self.compare(t)

    def token(self, meta, token, forward):
        e = self.engine
        with self.lock:
            t = self.current(meta)
            if not t or len(t["tokens"]) >= t["limit"] or t["complete"]:
                return
            if getattr(e, "throughput", None):
                e.throughput.token(t["slot"])
            t["tokens"].append(token)
            t["times"].append(time.time())
            self.compare(t)
            t["events"].put({"token": token})
            if token in e.stop_ids or len(t["tokens"]) >= t["limit"]:
                t["complete"] = True
                if getattr(e, "throughput", None):
                    e.throughput.finish(t["slot"])
                t["events"].put(None)
            elif forward:
                e.deliver(
                    0,
                    (
                        "token",
                        dict(
                            token=token,
                            conv=t["slot"],
                            generation=t["generation"],
                            epoch=e.epoch,
                        ),
                        b"",
                    ),
                )

    def chat(
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
        from .qwen_protocol import validate_conversation_id, chat_prompt
        from .chat_contract import encode_ids

        e = self.engine
        selected = validate_conversation_id(
            conversation_id or "legacy-browser"
        )
        if not message and prompt_tokens is None:
            with self.lock:
                busy = any(
                    x["conversation"] == selected for x in self.active.values()
                )
                if reset and not busy:
                    self.histories.pop(selected, None)
            yield (
                {
                    "error": "This conversation already has a response in progress."
                }
                if busy
                else {"done": True, "position": 0}
            )
            return
        t = None
        direct = getattr(e, "chat_direct", None)
        invitation = None
        with self.lock:
            if e.state != "running":
                failure = "The contributors are not ready."
            elif any(
                x["conversation"] == selected for x in self.active.values()
            ):
                failure = (
                    "This conversation already has a response in progress."
                )
            elif len(self.active) >= self.slots:
                failure = "All conversation slots are busy. Retry when a response finishes."
            else:
                failure = None
                history = (
                    []
                    if reset or prompt_tokens is not None
                    else list(self.histories.get(selected, []))
                )
                messages = history + [("user", message)]
                tokens = (
                    list(prompt_tokens)
                    if prompt_tokens is not None
                    else encode_ids(
                        e.tokenizer, chat_prompt(messages, e.chat_style)
                    )
                )
                remaining = e.capacity - len(tokens)
                limit = (
                    remaining
                    if max_tokens is None
                    else min(max_tokens, remaining)
                )
                if limit <= 0:
                    failure = "The conversation fills the context window. Start a new conversation or shorten the messages."
                else:
                    slot = next(
                        i for i in range(self.slots) if i not in self.active
                    )
                    self.generation += 1
                    t = dict(
                        slot=slot,
                        conversation=selected,
                        generation=self.generation,
                        limit=limit,
                        tokens=[],
                        times=[],
                        fed=[],
                        events=TokenMailbox(),
                        complete=False,
                    )
                    self.active[slot] = t
                    if getattr(e, "throughput", None):
                        e.throughput.start(slot)
                    packet_meta = dict(
                        conv=slot,
                        generation=self.generation,
                        epoch=e.epoch,
                        turn=self.generation,
                        limit=limit,
                    )
                    for box in e.inbox:
                        box.put(("reset", packet_meta, b""))
        if failure:
            code = (
                "session_busy"
                if "eight" in failure
                else "conversation_busy"
                if "in progress" in failure
                else "inference_error"
            )
            yield {
                "error": failure,
                "code": code,
                "status": 429
                if code == "session_busy"
                else 409
                if code == "conversation_busy"
                else 400,
            }
            return
        try:
            if cancel_event is not None and cancel_event.is_set():
                return
            start, clock = time.time(), time.monotonic()
            packet = (
                "prompt",
                packet_meta,
                np.asarray(tokens, dtype=np.uint32).tobytes(),
            )
            if direct:
                invitation = direct.prepare(
                    direct_lease,
                    tokens,
                    t["generation"],
                    e.epoch,
                    limit,
                    conversation_id=selected,
                    conv=t["slot"],
                    generation=t["generation"],
                )
            if invitation:
                yield invitation
            else:
                e.inbox[0].put(packet)
            shown, last = "", time.monotonic()
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return
                if invitation and direct.relay_due():
                    e.inbox[0].put(packet)
                try:
                    event = t["events"].get(timeout=0.25)
                except queue.Empty:
                    if time.monotonic() - last >= e.args.turn_timeout:
                        yield {
                            "error": f"No token for {e.args.turn_timeout} s"
                        }
                        return
                    continue
                last = time.monotonic()
                if event is None:
                    break
                if "error" in event:
                    yield event
                    return
                text = e.tokenizer.decode(
                    [x for x in t["tokens"] if x not in e.stop_ids]
                )
                if not text.endswith("\ufffd") and text != shown:
                    yield {"text": text[len(shown) :]}
                    shown = text
            times = t["times"]
            record = dict(
                prompt_tokens=len(tokens),
                start_position=0,
                generated=len(t["tokens"]),
                total_s=time.monotonic() - clock,
                first_token_s=times[0] - start if times else None,
                tokens_per_s=(len(times) - 1) / (times[-1] - times[0])
                if len(times) > 1 and times[-1] > times[0]
                else None,
                ms_per_token=float(np.median(np.diff(times)) * 1000)
                if len(times) > 1
                else None,
                stopped="end" if t["tokens"][-1] in e.stop_ids else "length",
                prompt_transport=(direct.prompt_transport() or "unknown")
                if invitation
                else "coordinator",
            )
            with self.lock:
                if prompt_tokens is None:
                    self.histories.pop(selected, None)
                    self.histories[selected] = messages + [
                        ("assistant", shown)
                    ]
                    while len(self.histories) > 16:
                        self.histories.pop(next(iter(self.histories)))
                e.turns.append(record)
            yield dict(
                record,
                done=True,
                position=len(tokens) + len(t["tokens"]),
                full_text=shown,
            )
        finally:
            if invitation:
                direct.finish()
            with self.lock:
                # Fence outstanding direct packets before a slot is reused.
                self.generation += 1
                for box in e.inbox:
                    box.put(
                        (
                            "reset",
                            dict(
                                conv=t["slot"],
                                generation=self.generation,
                                epoch=e.epoch,
                            ),
                            b"",
                        )
                    )
                self.active.pop(t["slot"], None)
                if getattr(e, "throughput", None):
                    e.throughput.finish(t["slot"])
