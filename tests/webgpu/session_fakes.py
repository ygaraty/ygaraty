"""In-process metadata/fake-stream fixtures; no model or service startup."""

import threading
from types import SimpleNamespace

from test_preparation import inputs
from ygaraty.backend_contracts.chat import ChatRequest, Message, Role
from ygaraty.backend_contracts.models import HistoryMode
from ygaraty.backend_contracts.protocols import RuntimeServices

from ygaraty.backends.webgpu.configuration import WebGPUConfig
from ygaraty.backends.webgpu.hosting import LocalAccess, MemoryLedger, SystemClock
from ygaraty.backends.webgpu.session import WebGPUSession

OWNER = "o" * 40
CONTRIBUTOR = "c" * 40


class FakeEngine:
    def __init__(self):
        self.state, self.capacity = "running", 128
        self.chat_style = None
        self.tokenizer = SimpleNamespace(
            encode=lambda text: SimpleNamespace(ids=[1, 2])
        )
        self.tokenizer.encode_batch = lambda texts: [
            self.tokenizer.encode(t) for t in texts
        ]
        self.concurrent_chat = SimpleNamespace(
            lock=threading.RLock(), histories={}
        )
        self.calls = []
        self.mode = "normal"
        self.started, self.finished, self.release = (
            threading.Event() for _ in range(3)
        )
        self.closed = False

    def chat(self, *, cancel_event, **arguments):
        self.calls.append(arguments)
        self.started.set()
        try:
            if self.mode == "blocked":
                cancel_event.wait(2)
                return
            if self.mode == "uncooperative":
                self.release.wait(2)
                return
            if self.mode == "flood":
                for _ in range(2000):
                    if cancel_event.is_set():
                        return
                    yield {"text": "x" * 1000}
            else:
                yield {"text": "hello "}
                yield {"text": "world"}
            yield {
                "done": True,
                "prompt_tokens": 2,
                "generated": 2,
                "stopped": "end",
            }
        finally:
            self.finished.set()

    def close(self):
        self.closed = True

    def status(self):
        return {"state": self.state}

    def join(self, body, peer):
        return {"id": "fake", "accepted": True}


class FailingLedger(MemoryLedger):
    fail_admit = fail_finish = False

    async def admit(self, record):
        if self.fail_admit:
            raise OSError("Unavailable")
        await super().admit(record)

    async def finish(self, key, outcome):
        if self.fail_finish:
            raise OSError("Unavailable")
        return await super().finish(key, outcome)


def setup():
    spec, raw = inputs(conversation_slots=2)
    clock = SystemClock()
    access = LocalAccess(spec, OWNER, CONTRIBUTOR, 2, clock)
    ledger = FailingLedger(spec)
    services = RuntimeServices(clock, access, ledger)
    engine = FakeEngine()
    session = WebGPUSession(
        spec, WebGPUConfig.from_contract(spec.configuration), services, engine
    )
    session.revalidate_interval = 0.01
    return SimpleNamespace(
        spec=spec,
        raw=raw,
        services=services,
        engine=engine,
        access=access,
        ledger=ledger,
        session=session,
    )


def request(identity="r", conversation="chat"):
    return ChatRequest(
        identity,
        conversation,
        "fixture",
        HistoryMode.FULL,
        (Message(Role.USER, "Hello"),),
    )


async def collect(handle):
    return [event async for event in handle.events()]
