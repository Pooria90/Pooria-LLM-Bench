"""AIME (I and II): 30 problems per year with integer answers 0-999.

Data: data/aime<year>.jsonl (built by scripts/build_aime.py from MathArena).
Figures are given to the model as text (TikZ, Asymptote, a LaTeX array or an
ASCII grid); those problems are tagged category="figure" so reports can split
them out.

Prompt: the problem followed by MathArena-style instructions, as the user message,
with no system prompt. Scoring: last `\\boxed{}` integer, exact match.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import scorers
from .base import BuildContext, Sample, Score, Task

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

INSTRUCTIONS = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "The answer is an integer between 0 and 999 inclusive."
)


class AIME(Task):
    year: int

    def load(self, limit: int | None = None) -> list[Sample]:
        samples = []
        with (DATA_DIR / f"aime{self.year}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                samples.append(
                    Sample(
                        id=rec["id"],
                        input=rec["problem"],
                        target=rec["answer"],
                        metadata={
                            "category": "figure" if rec["figure"] else "text",
                            "figure": rec["figure"],
                            "problem_type": rec["problem_type"],
                            "section": rec["section"],
                            "number": rec["number"],
                            "aops_url": rec["aops_url"],
                        },
                    )
                )
        return samples[:limit] if limit else samples

    def build(self, ctx: BuildContext):
        return ctx.agent(name="AIMESolver")

    def make_input(self, sample: Sample) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return (f"{sample.input}\n\n{INSTRUCTIONS}",), {}

    def score(self, sample: Sample, output: Any) -> Score:
        return scorers.boxed_integer_match("" if output is None else str(output), sample.target)


class AIME2025(AIME):
    name = "aime2025"
    year = 2025
    description = "AIME 2025 I+II: 30 problems, integer answers (6 figures given as TikZ/Asymptote/LaTeX text)."


class AIME2026(AIME):
    name = "aime2026"
    year = 2026
    description = "AIME 2026 I+II: 30 problems, integer answers (2 figures given as Asymptote / ASCII-grid text)."
