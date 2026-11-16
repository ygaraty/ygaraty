"""Bounded independent chat histories sharing one model's GPU cache slots."""

import threading

from .qwen_protocol import SessionProtocol, validate_conversation_id


class SlotMailbox:
    def __init__(self, channel, stage):
        self.channel, self.stage = channel, stage

    def put(self, packet):
        kind, meta, body = packet
        c = self.channel
        c.parent.inbox[self.stage].put(
            (
                kind,
                dict(
                    meta,
                    conv=c.slot,
                    epoch=c.parent.epoch,
                    chat_epoch=c.generation,
                    chat_turn=c.turn_id,
                ),
                body,
            )
        )


class ChatChannel(SessionProtocol):
    """Reuse turn streaming while keeping every mutable chat field private."""

    concurrent_chats = None
    chat_direct = None  # The legacy direct-browser grant has only one owner.

    def __init__(self, parent, identity):
        self.parent = parent
        self.slot = None
        self.generation = 0
        self.chat_lock = threading.Lock()
        self.active_conversation = identity
        self.chat_histories = {}
        self.turn, self.turns, self.turn_ids = None, [], []
        self.position, self.last_token = 0, None
        self.history, self.replay, self.diverged = [], False, False
        self.turn_id, self.finished_turn, self.fed = 0, None, {}
        self.inbox = [SlotMailbox(self, k) for k in range(len(parent.inbox))]
        self.busy = False

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def say(self, line):
        self.parent.say(line)

    def deliver(self, stage, packet):
        self.inbox[stage].put(packet)

    def drop_stage_state(self):
        with self.parent.lock:
            pool = self.parent.concurrent_chats
            pool.generations[self.slot] += 1
            self.generation = pool.generations[self.slot]
            self.worker_epoch = self.parent.epoch
            self.turn = None
            self.position, self.replay = 0, bool(self.history)
            self.fed = {}
            for box in self.inbox:
                box.put(("reset", {}, b""))

    def reset_conversation(self):
        self.history, self.last_token = [], None
        self.drop_stage_state()


class ChatSlots:
    def __init__(self, parent, count):
        self.parent = parent
        self.channels = {}
        self.slots = [None] * count
        self.generations = [0] * count

    def channel(self, slot):
        if type(slot) is not int or not 0 <= slot < len(self.slots):
            return None
        return self.slots[slot]

    def accepts(self, meta):
        channel = self.channel(meta.get("conv", 0))
        return (
            channel is not None
            and meta.get("chat_epoch") == channel.generation
            and meta.get("chat_turn") == channel.turn_id
        )

    def chat(self, message, reset=False, cancel_event=None, **kwargs):
        identity = validate_conversation_id(
            kwargs.get("conversation_id")
            or (
                "api"
                if kwargs.get("prompt_tokens") is not None
                else "legacy-browser"
            )
        )
        with self.parent.lock:
            channel, error = self.acquire(identity, cancel_event)
        if error:
            yield error
        if channel is None:
            return
        try:
            yield from channel._chat_serial(
                message,
                reset,
                cancel_event,
                **dict(kwargs, conversation_id=identity),
            )
        finally:
            with self.parent.lock:
                channel.busy = False

    def acquire(self, identity, cancel_event):
        if cancel_event is not None and cancel_event.is_set():
            return None, None
        channel = self.channels.get(identity)
        if channel is None:
            if len(self.channels) >= 16:
                return None, {
                    "error": "This session reached its 16-conversation limit."
                }
            channel = ChatChannel(self.parent, identity)
        if channel.busy:
            return None, {
                "error": "This conversation already has an answer in progress."
            }
        if channel.slot is None:
            slot = next(
                (i for i, c in enumerate(self.slots) if c is None), None
            )
            if slot is None:
                slot = next(
                    (i for i, c in enumerate(self.slots) if not c.busy), None
                )
            if slot is None:
                return None, {
                    "error": "All chat slots are busy. Retry after completion."
                }
            previous = self.slots[slot]
            if previous is not None:
                previous.slot = None
                previous.position, previous.replay = 0, bool(previous.history)
            self.channels[identity] = channel
            self.slots[slot] = channel
            channel.slot = slot
            channel.drop_stage_state()
        elif channel.worker_epoch != self.parent.epoch:
            channel.drop_stage_state()
        channel.busy = True
        return channel, None

    def interrupt(self, reason):
        """Global worker replacement resets all slots; wake every stream."""
        for channel in self.channels.values():
            if channel.turn is not None:
                channel.turn["events"].put({"error": reason})
            channel.position, channel.replay = 0, bool(channel.history)
            channel.fed = {}

    def active(self):
        return any(c.busy for c in self.channels.values())
