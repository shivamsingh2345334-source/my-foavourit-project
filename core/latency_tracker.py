"""
Micro-step latency tracking. Uses time.time() checkpoints and prints every step
to the console in the format:  [LATENCY] STT Groq (chunk 1): 120ms
"""
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Dict, List, Optional


class _Step:
    """Handle yielded by step()/astep() so callers can rename the step after the
    fact (e.g. 'Redis Cache Check' -> 'Redis Cache Hit') or attach extra info."""

    __slots__ = ("name", "extra")

    def __init__(self, name: str):
        self.name = name
        self.extra = ""


class LatencyTracker:
    def __init__(self, request_id: str = "-"):
        self.request_id = request_id
        self.t_created = time.time()
        self.anchor: Optional[float] = None  # moment the user stopped speaking
        self.steps: List[Dict] = []
        self._seen: set = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ anchor
    def set_anchor(self, label: str = "end-of-speech") -> None:
        self.anchor = time.time()
        print(f"[LATENCY] -- anchor set: {label} (request={self.request_id})")

    # ------------------------------------------------------------------ record
    def record(self, name: str, ms: float, extra: str = "") -> None:
        with self._lock:
            self.steps.append({"name": name, "ms": round(ms, 1), "kind": "step"})
        print(f"[LATENCY] {name}: {ms:.0f}ms{extra}")

    @contextmanager
    def step(self, name: str):
        s = _Step(name)
        t0 = time.time()
        try:
            yield s
        except BaseException:
            s.name += " [FAILED]"
            raise
        finally:
            self.record(s.name, (time.time() - t0) * 1000, s.extra)

    @asynccontextmanager
    async def astep(self, name: str):
        s = _Step(name)
        t0 = time.time()
        try:
            yield s
        except BaseException:
            s.name += " [FAILED]"
            raise
        finally:
            self.record(s.name, (time.time() - t0) * 1000, s.extra)

    # --------------------------------------------------------------- milestone
    def milestone(self, name: str, once: bool = False) -> Optional[float]:
        """Cumulative time since end-of-speech (or since turn start if no anchor)."""
        with self._lock:
            if once and name in self._seen:
                return None
            self._seen.add(name)
            base = self.anchor if self.anchor is not None else self.t_created
            ms = (time.time() - base) * 1000
            self.steps.append({"name": name, "ms": round(ms, 1), "kind": "milestone"})
        since = "end-of-speech" if self.anchor is not None else "turn start"
        print(f"[LATENCY] * {name}: {ms:.0f}ms (cumulative since {since})")
        return ms

    # ------------------------------------------------------------------ report
    def report(self) -> List[Dict]:
        with self._lock:
            return list(self.steps)

    def print_summary(self) -> None:
        steps = self.report()
        print("=" * 78)
        print(f"[LATENCY SUMMARY] request={self.request_id}")
        for s in steps:
            mark = "*" if s["kind"] == "milestone" else "-"
            print(f"  {mark} {s['name']:<58} {s['ms']:>8.0f} ms")
        print("=" * 78)


def get_tracker(t: Optional[LatencyTracker]) -> LatencyTracker:
    """Lets infrastructure wrappers be called without a tracker (ad-hoc use)."""
    return t if t is not None else LatencyTracker("adhoc")
