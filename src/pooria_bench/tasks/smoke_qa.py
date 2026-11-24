"""Tiny closed-form QA set for checking the pipeline end to end.

Not a benchmark: 12 hand-written questions with unambiguous answers, a mix of
arithmetic, instruction following and stable facts. Cheap enough to run on every
model after any harness change.
"""

from __future__ import annotations

from typing import Any

from .. import scorers
from .base import BuildContext, Sample, Score, Task

SYSTEM = (
    "Answer the question. Think as needed, then end your reply with a final line "
    "of the form 'Answer: <answer>' and nothing after it."
)

# (id, category, question, target, kind)
_ITEMS: list[tuple[str, str, str, Any, str]] = [
    ("arith-1", "arithmetic", "What is 17 * 23?", 391, "number"),
    ("arith-2", "arithmetic", "What is 2**20 - 1?", 1048575, "number"),
    ("arith-3", "arithmetic", "A shirt costs $40 and is discounted 15%, then 8% sales tax is added. What is the final price in dollars?", 36.72, "number"),
    ("arith-4", "arithmetic", "How many seconds are there in 3.5 hours?", 12600, "number"),
    ("logic-1", "reasoning", "Alice is older than Bob. Bob is older than Carol. Who is the youngest? Answer with just the name.", "Carol", "exact"),
    ("logic-2", "reasoning", "If today is Wednesday, what day of the week will it be 100 days from today? Answer with just the day.", "Friday", "exact"),
    ("logic-3", "reasoning", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost, in cents?", 5, "number"),
    ("fact-1", "knowledge", "What is the chemical symbol for tungsten?", "W", "exact"),
    ("fact-2", "knowledge", "How many sides does a dodecagon have?", 12, "number"),
    ("fact-3", "knowledge", "What is the capital city of Australia? Answer with just the city.", "Canberra", "exact"),
    ("fmt-1", "instructions", "Write the word 'benchmark' backwards, in lowercase, with no other text.", "kramhcneb", "exact"),
    ("fmt-2", "instructions", "Count the letter 'r' in the word 'strawberry'. Answer with just the number.", 3, "number"),
]


class SmokeQA(Task):
    name = "smoke-qa"
    description = "12 closed-form questions to validate the harness (not a benchmark)."

    def load(self, limit: int | None = None) -> list[Sample]:
        samples = [
            Sample(id=i, input=q, target=t, metadata={"category": c, "kind": k})
            for i, c, q, t, k in _ITEMS
        ]
        return samples[:limit] if limit else samples

    def build(self, ctx: BuildContext):
        return ctx.agent(name="SmokeQA", system_message=SYSTEM)

    def score(self, sample: Sample, output: Any) -> Score:
        if sample.metadata["kind"] == "number":
            return scorers.numeric_match(str(output), sample.target)
        return scorers.exact_match(str(output), sample.target)
