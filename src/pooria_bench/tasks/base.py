"""Task abstraction.

A task owns its data, the Railtracks graph that solves one sample, and the
scoring. It never names a model: the runner hands `build()` a `BuildContext`
for each model under test, and the same graph is rebuilt around that model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import railtracks as rt
from pydantic import BaseModel, Field

from ..instrument import recording_middleware


class Sample(BaseModel):
    id: str
    input: Any
    target: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Score(BaseModel):
    value: float
    correct: bool
    extracted: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass
class BuildContext:
    """What a task gets when it builds its graph for one model."""

    llm: Any
    model_name: str
    model_middleware: list[Any] = field(default_factory=lambda: [recording_middleware])

    def agent(self, *, model_middleware: list[Any] | None = None, **kwargs: Any):
        """`rt.agent_node` bound to this model, with the benchmark's recording middleware.

        Use this (not `rt.agent_node` directly) for every agent in a task, so every
        model call is rate-limited and recorded.
        """
        return rt.agent_node(
            llm=self.llm,
            model_middleware=[*self.model_middleware, *(model_middleware or [])],
            **kwargs,
        )


class Task(ABC):
    name: ClassVar[str]
    description: ClassVar[str]

    @abstractmethod
    def load(self, limit: int | None = None) -> list[Sample]:
        """Return the samples, first `limit` only when given."""

    @abstractmethod
    def build(self, ctx: BuildContext):
        """Return the entry-point node (agent or function node) for one model."""

    def make_input(self, sample: Sample) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Positional and keyword arguments for `flow.ainvoke`."""
        return (sample.input,), {}

    def extract_output(self, result: Any) -> Any:
        """Turn a Railtracks result into the value `score()` sees."""
        structured = getattr(result, "structured", None)
        if isinstance(structured, BaseModel):
            return structured.model_dump()
        if hasattr(result, "content"):
            return result.content
        return result

    @abstractmethod
    def score(self, sample: Sample, output: Any) -> Score:
        """Score one output against the sample's target."""
