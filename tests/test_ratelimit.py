import asyncio
import time

import pytest

from pooria_bench.ratelimit import RateLimiter


def test_rate_limiter_paces_requests():
    async def main():
        limiter = RateLimiter(rpm=1200, burst=1)  # 20/s, no bursting
        t0 = time.monotonic()
        await asyncio.gather(*(limiter.acquire() for _ in range(6)))
        return time.monotonic() - t0

    elapsed = asyncio.run(main())
    # first token is free, the next five take 1/20 s each
    assert elapsed == pytest.approx(0.25, abs=0.08)


def test_rate_limiter_rejects_bad_rpm():
    with pytest.raises(ValueError):
        RateLimiter(rpm=0)
