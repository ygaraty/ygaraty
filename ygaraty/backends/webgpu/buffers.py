"""Bounded token notifications without blocking contributor control traffic."""

import queue
import threading


class TokenMailbox:
    def __init__(self, capacity: int = 512):
        self._queue = queue.Queue(maxsize=capacity)
        self._lock = threading.Lock()
        self._overflow = False

    def put(self, value) -> None:
        with self._lock:
            if self._overflow:
                return
            try:
                self._queue.put_nowait(value)
            except queue.Full:
                self._overflow = True
                # Wake the reader with a failure, never silently omit tokens.
                while True:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
                self._queue.put_nowait({"error": "Response consumer too slow"})

    def get(self, timeout=None):
        return self._queue.get(timeout=timeout)
