"""Per-call instrumentation for Railtracks models.

Two pieces, both working around gaps in Railtracks' public telemetry (see GAPS.md):

1. `instrumented(cls)` subclasses an `rt.llm.*` class and overrides its public
   `extract_message_info` classmethod, so the `MessageInfo` on every response also
   carries reasoning tokens, cached tokens and finish_reason. Railtracks' own
   `MessageInfo` drops all three even though litellm returns them.

2. `recording_middleware` is a Railtracks *model middleware* (`rt.wrap_llm`). It
   wraps every model call inside an agent's tool loop, paces it through the
   sample's rate limiter and concurrency cap, times it with a monotonic clock and appends a
   `CallRecord` to the `SampleTrace` of whichever sample is running. The trace is
   found through a ContextVar, which Railtracks propagates into its node tasks,
   so concurrent samples never mix their records.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import railtracks as rt

from .ratelimit import RateLimiter


@dataclass
class CallRecord:
    """One model call made while solving a sample."""

    latency_s: float
    queued_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    finish_reason: str | None = None
    model_name: str | None = None
    provider_cost_usd: float | None = None
    tool_calls: list[str] = field(default_factory=list)
    has_reasoning_text: bool = False
    error: str | None = None


@dataclass
class SampleTrace:
    """Everything recorded while one sample runs."""

    limiter: RateLimiter | None = None
    slots: asyncio.Semaphore | None = None  # caps concurrent calls per rate_group
    calls: list[CallRecord] = field(default_factory=list)


CURRENT_TRACE: ContextVar[SampleTrace | None] = ContextVar("pooria_bench_trace", default=None)


def _get(obj: Any, *path: str) -> Any:
    for name in path:
        if obj is None:
            return None
        obj = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
    return obj


_INSTRUMENTED: dict[type, type] = {}


def instrumented(cls: type) -> type:
    """Return a subclass of a litellm-backed `rt.llm` class that keeps usage extras."""
    if cls in _INSTRUMENTED:
        return _INSTRUMENTED[cls]
    if not hasattr(cls, "extract_message_info"):
        _INSTRUMENTED[cls] = cls  # e.g. EchoLLM: nothing to extract
        return cls

    class _Instrumented(cls):  # type: ignore[misc, valid-type]
        @classmethod
        def extract_message_info(cls_, model_response, latency, requested_model=None):  # noqa: N805
            info = super().extract_message_info(model_response, latency, requested_model)
            usage = _get(model_response, "usage")
            info.reasoning_tokens = _get(usage, "completion_tokens_details", "reasoning_tokens")
            info.cached_tokens = _get(usage, "prompt_tokens_details", "cached_tokens")
            choices = _get(model_response, "choices") or []
            info.finish_reason = _get(choices[0], "finish_reason") if choices else None
            return info

    _Instrumented.__name__ = cls.__name__
    _Instrumented.__qualname__ = cls.__qualname__
    _INSTRUMENTED[cls] = _Instrumented
    return _Instrumented


def _tool_call_names(response: Any) -> list[str]:
    content = _get(response, "message", "content")
    if isinstance(content, list):
        return [str(_get(c, "name")) for c in content if _get(c, "name") is not None]
    return []


@rt.wrap_llm
async def recording_middleware(llm_call, message_history, schema, tools):
    """Model middleware: rate-limit, time and record every model call."""
    trace = CURRENT_TRACE.get()
    if trace is None:  # running outside the benchmark runner
        return await llm_call(message_history, schema, tools)

    wait_start = time.perf_counter()
    async with trace.slots if trace.slots is not None else contextlib.nullcontext():
        if trace.limiter is not None:
            await trace.limiter.acquire()
        queued = time.perf_counter() - wait_start
        start = time.perf_counter()
        try:
            response = await llm_call(message_history, schema, tools)
        except Exception as exc:
            trace.calls.append(
                CallRecord(latency_s=time.perf_counter() - start, queued_s=queued, error=repr(exc))
            )
            raise
        elapsed = time.perf_counter() - start

    info = response.message_info
    trace.calls.append(
        CallRecord(
            latency_s=elapsed,
            queued_s=queued,
            input_tokens=info.input_tokens,
            output_tokens=info.output_tokens,
            reasoning_tokens=getattr(info, "reasoning_tokens", None),
            cached_tokens=getattr(info, "cached_tokens", None),
            finish_reason=getattr(info, "finish_reason", None),
            model_name=info.model_name,
            provider_cost_usd=info.total_cost,
            tool_calls=_tool_call_names(response),
            has_reasoning_text=bool(getattr(response, "reasoning", None)),
        )
    )
    return response
