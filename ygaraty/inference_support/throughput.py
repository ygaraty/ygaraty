"""Bounded, aggregate session throughput; never records prompts or token IDs."""
import math
import threading
import time


class Throughput:
    window = 10

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.tokens = 0
        self.seconds = 0.0
        self.peak = 0.0
        self.active = set()
        self.since = None
        self.buckets = {}
        self.closed = False

    def restore(self, saved):
        with self.lock:
            self.tokens = int(saved.get('generated_tokens', 0))
            self.seconds = float(saved.get('active_seconds', 0))
            self.peak = float(saved.get('peak_tokens_per_s', 0))

    def start(self, slot):
        with self.lock:
            if self.closed:
                return
            if not self.active:
                self.since = self.clock()
            self.active.add(slot)

    def _rate(self, now):
        cutoff = math.floor(now) - self.window + 1
        self.buckets = {s: n for s, n in self.buckets.items() if s >= cutoff}
        return sum(self.buckets.values()) / self.window

    def token(self, slot):
        with self.lock:
            if self.closed or slot not in self.active:
                return
            now = self.clock()
            second = math.floor(now)
            self.tokens += 1
            self.buckets[second] = self.buckets.get(second, 0) + 1
            self.peak = max(self.peak, self._rate(now))

    def finish(self, slot):
        with self.lock:
            self.active.discard(slot)
            if not self.active and self.since is not None:
                self.seconds += max(0, self.clock() - self.since)
                self.since = None

    def close(self):
        with self.lock:
            for slot in list(self.active):
                self.finish(slot)
            self.closed = True

    def snapshot(self):
        with self.lock:
            now = self.clock()
            seconds = self.seconds + (max(0, now-self.since) if self.since is not None else 0)
            return dict(version=1, generated_tokens=self.tokens,
                        active_seconds=round(seconds, 6),
                        average_tokens_per_s=round(self.tokens/seconds, 3) if seconds else None,
                        recent_tokens_per_s=round(self._rate(now), 3),
                        peak_tokens_per_s=round(self.peak, 3),
                        active_chats=len(self.active), window_seconds=self.window)
