"""Client-side request rate limiting.

Railtracks has no request-rate limiter (it only retries after a 429). Providers
like Radium enforce a per-organization RPM *and* an acceleration cap, so we pace
requests ourselves. One limiter is shared by every model in a `rate_group`.
"""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Async token bucket: `rpm` requests per minute, bursts of at most `burst`.

    Starts with a single token so a run ramps up instead of firing a full
    burst at once (Radium rejects sudden jumps relative to the previous minute).
    """

    def __init__(self, rpm: float, burst: int | None = None) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        self.rate = rpm / 60.0  # tokens per second
        self.capacity = float(burst if burst is not None else max(1, int(self.rate * 2)))
        self._tokens = 1.0
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    async def acquire(self) -> float:
        """Wait for a token. Returns the seconds spent waiting."""
        waited = 0.0
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return waited
                delay = (1.0 - self._tokens) / self.rate
                waited += delay
                await asyncio.sleep(delay)
