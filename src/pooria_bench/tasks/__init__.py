"""Task registry."""

from __future__ import annotations

from .aime2025 import AIME2025
from .base import BuildContext, Sample, Score, Task
from .smoke_qa import SmokeQA
from .smoke_tools import SmokeTools

TASKS: dict[str, type[Task]] = {t.name: t for t in (SmokeQA, SmokeTools, AIME2025)}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]()
    except KeyError:
        raise KeyError(f"Unknown task {name!r}. Known: {', '.join(TASKS)}") from None


__all__ = ["BuildContext", "Sample", "Score", "Task", "TASKS", "get_task"]
