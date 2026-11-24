"""Result records and JSONL storage."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field


class CallSummary(BaseModel):
    latency_s: float
    queued_s: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    finish_reason: str | None = None
    tool_calls: list[str] = Field(default_factory=list)
    has_reasoning_text: bool = False
    error: str | None = None


class SampleResult(BaseModel):
    run_id: str
    task: str
    model: str
    model_id: str
    sample_id: str
    repeat: int = 0
    category: str | None = None
    output: Any = None
    target: Any = None
    extracted: str | None = None
    score: float | None = None
    correct: bool | None = None
    score_details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    latency_s: float = 0.0
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    provider_cost_usd: float | None = None
    finish_reasons: list[str | None] = Field(default_factory=list)
    calls: list[CallSummary] = Field(default_factory=list)
    started_at: str | None = None

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.model, self.sample_id, self.repeat)


class ResultWriter:
    """Append-only JSONL writer, safe to share between concurrent samples."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    async def write(self, record: BaseModel) -> None:
        line = record.model_dump_json() + "\n"
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)


def read_results(path: Path) -> Iterator[SampleResult]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield SampleResult.model_validate(json.loads(line))


def latest_results(path: Path) -> list[SampleResult]:
    """One record per (model, sample, repeat), keeping the last attempt written."""
    latest: dict[tuple[str, str, int], SampleResult] = {}
    for r in read_results(path):
        latest[r.key] = r
    return list(latest.values())
