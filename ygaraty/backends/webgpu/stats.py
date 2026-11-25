"""Live request statistics for one session: tokens, speed and concurrency.

The session records each chat request it admits (API, web chat and terminal
chat all use the same endpoint): when it started, when its first text left
the engine, and its token counts when it ended. The status page, the chat
pages and the terminal chat read a summary of the same numbers.

Times come from a monotonic clock. Averages cover the most recent requests
(RECENT), so a slow first request after loading stops dominating them.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass

RECENT = 50
WINDOW_SECONDS = 60.0


@dataclass(slots=True)
class _Request:
    started: float
    first: float | None = None


class SessionStats:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.opened = clock()
        self.active: dict[str, _Request] = {}
        self.peak_active = 0
        self.completed = self.failed = self.cancelled = 0
        self.prompt_tokens = self.generated_tokens = 0
        self.ttfts: deque[float] = deque(maxlen=RECENT)
        self.rates: deque[float] = deque(maxlen=RECENT)
        # (end time, tokens) of recent requests, for throughput over a window.
        self.ends: deque[tuple[float, int]] = deque(maxlen=1024)
        self.last: dict | None = None

    def started(self, request_id):
        with self.lock:
            self.active[request_id] = _Request(self.clock())
            self.peak_active = max(self.peak_active, len(self.active))

    def first_text(self, request_id):
        with self.lock:
            record = self.active.get(request_id)
            if record is not None and record.first is None:
                record.first = self.clock()

    def finished(self, request_id, outcome, prompt=0, generated=0):
        """outcome: "completed", "failed" or "cancelled"."""
        now = self.clock()
        with self.lock:
            record = self.active.pop(request_id, None)
            if outcome == "completed":
                self.completed += 1
            elif outcome == "cancelled":
                self.cancelled += 1
            else:
                self.failed += 1
            self.prompt_tokens += max(0, int(prompt or 0))
            self.generated_tokens += max(0, int(generated or 0))
            if generated:
                self.ends.append((now, int(generated)))
            if record is None or outcome != "completed":
                return
            ttft = (
                None if record.first is None else record.first - record.started
            )
            # Decode rate after the first token, so TTFT does not dilute it.
            rate = (
                (generated - 1) / (now - record.first)
                if record.first is not None
                and generated > 1
                and now > record.first
                else None
            )
            if ttft is not None:
                self.ttfts.append(ttft)
            if rate is not None:
                self.rates.append(rate)
            self.last = dict(
                ttft_s=_round(ttft),
                tokens_per_s=_round(rate, 1),
                prompt_tokens=int(prompt or 0),
                generated_tokens=int(generated or 0),
                seconds=_round(now - record.started),
            )

    def summary(self):
        now = self.clock()
        with self.lock:
            recent = [n for at, n in self.ends if now - at <= WINDOW_SECONDS]
            return dict(
                uptime_s=round(now - self.opened, 1),
                active_requests=len(self.active),
                peak_active_requests=self.peak_active,
                requests=dict(
                    completed=self.completed,
                    failed=self.failed,
                    cancelled=self.cancelled,
                ),
                prompt_tokens=self.prompt_tokens,
                generated_tokens=self.generated_tokens,
                total_tokens=self.prompt_tokens + self.generated_tokens,
                ttft_s=dict(
                    average=_round(_mean(self.ttfts)),
                    best=_round(min(self.ttfts, default=None)),
                ),
                tokens_per_s=dict(
                    average=_round(_mean(self.rates), 1),
                    best=_round(max(self.rates, default=None), 1),
                ),
                # All requests' output tokens over the last minute.
                throughput_tokens_per_s=round(sum(recent) / WINDOW_SECONDS, 2),
                last=self.last,
            )


def _mean(values):
    return sum(values) / len(values) if values else None


def _round(value, digits=3):
    return None if value is None else round(value, digits)
