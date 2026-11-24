"""Aggregate results.jsonl into a per-model (optionally per-category) report.

Railtracks' evaluation aggregates are means only; a model comparison needs
confidence intervals, tail latency and cost per correct answer.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .results import SampleResult, latest_results


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


@dataclass
class Row:
    model: str
    group: str | None
    n: int
    errors: int
    correct: int
    format_misses: int
    no_reasoning: int
    accuracy: float | None
    ci_low: float | None
    ci_high: float | None
    p50_latency_s: float | None
    p95_latency_s: float | None
    calls_per_sample: float
    in_tokens_per_sample: float
    out_tokens_per_sample: float
    reasoning_tokens_per_sample: float
    output_tok_per_s: float | None
    truncated: int
    refusals: int
    cost_usd: float | None
    cost_per_correct_usd: float | None


def _format_ok(r: SampleResult) -> bool | None:
    d = r.score_details
    return d.get("format_ok", d.get("answer_line"))  # "answer_line": key used before 2026-11-24


def aggregate(results: list[SampleResult], by_category: bool = False) -> list[Row]:
    groups: dict[tuple[str, str | None], list[SampleResult]] = defaultdict(list)
    for r in results:
        groups[(r.model, r.category if by_category else None)].append(r)

    rows: list[Row] = []
    for (model, group), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        scored = [r for r in rs if r.error is None and r.correct is not None]
        correct = sum(1 for r in scored if r.correct)
        n = len(scored)
        lo, hi = wilson_ci(correct, n) if n else (None, None)
        lat = [r.latency_s for r in scored]
        call_time = sum(c.latency_s for r in rs for c in r.calls)
        out_tok = sum(r.output_tokens for r in rs)
        costs = [r.cost_usd for r in rs if r.cost_usd is not None]
        cost = sum(costs) if costs else None
        k = len(rs)
        rows.append(
            Row(
                model=model,
                group=group,
                n=n,
                errors=sum(1 for r in rs if r.error),
                correct=correct,
                format_misses=sum(
                    1 for r in scored if _format_ok(r) is False and not r.score_details.get("empty_output")
                ),
                no_reasoning=sum(1 for r in scored if r.reasoning_tokens == 0),
                accuracy=correct / n if n else None,
                ci_low=lo,
                ci_high=hi,
                p50_latency_s=percentile(lat, 50),
                p95_latency_s=percentile(lat, 95),
                calls_per_sample=sum(r.n_calls for r in rs) / k,
                in_tokens_per_sample=sum(r.input_tokens for r in rs) / k,
                out_tokens_per_sample=out_tok / k,
                reasoning_tokens_per_sample=sum(r.reasoning_tokens for r in rs) / k,
                output_tok_per_s=out_tok / call_time if call_time > 0 else None,
                truncated=sum(1 for r in rs for f in r.finish_reasons if f == "length"),
                refusals=sum(1 for r in rs for f in r.finish_reasons if f == "content_filter"),
                cost_usd=cost,
                cost_per_correct_usd=(cost / correct) if cost is not None and correct else None,
            )
        )
    return rows


def _fmt(v, spec: str = ".2f", suffix: str = "") -> str:
    return "-" if v is None else f"{v:{spec}}{suffix}"


COLUMNS = [
    ("model", lambda r: r.model),
    ("group", lambda r: r.group or ""),
    ("n", lambda r: str(r.n)),
    ("err", lambda r: str(r.errors)),
    ("accuracy", lambda r: _fmt(r.accuracy and r.accuracy * 100, ".1f", "%")),
    ("no-fmt", lambda r: str(r.format_misses)),
    ("no-think", lambda r: str(r.no_reasoning)),
    ("95% CI", lambda r: "-" if r.ci_low is None else f"{r.ci_low*100:.0f}-{r.ci_high*100:.0f}%"),
    ("p50 s", lambda r: _fmt(r.p50_latency_s)),
    ("p95 s", lambda r: _fmt(r.p95_latency_s)),
    ("calls", lambda r: _fmt(r.calls_per_sample, ".1f")),
    ("in tok", lambda r: _fmt(r.in_tokens_per_sample, ".0f")),
    ("out tok", lambda r: _fmt(r.out_tokens_per_sample, ".0f")),
    ("reason tok", lambda r: _fmt(r.reasoning_tokens_per_sample, ".0f")),
    ("out tok/s", lambda r: _fmt(r.output_tok_per_s, ".0f")),
    ("trunc", lambda r: str(r.truncated)),
    ("refusals", lambda r: str(r.refusals)),
    ("cost $", lambda r: _fmt(r.cost_usd, ".4f")),
    ("$/correct", lambda r: _fmt(r.cost_per_correct_usd, ".5f")),
]


def to_markdown(rows: list[Row], title: str) -> str:
    cols = [c for c in COLUMNS if c[0] != "group" or any(r.group for r in rows)]
    lines = [f"# {title}", "", "| " + " | ".join(c[0] for c in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f(r) for _, f in cols) + " |" for r in rows]
    lines += [
        "",
        "- accuracy excludes errored samples; `err` counts them. CI is a 95% Wilson interval.",
        "- `no-think` counts answers produced with zero reasoning tokens (on hard problems, a sign of recall).",
        "- `no-fmt` counts answers not in the requested format (`Answer:` line or `\\boxed{}`); scored best-effort.",
        "- latency is wall time per sample (whole agent loop, including tool calls and client-side queueing).",
        "- `out tok/s` = output tokens / summed model-call time, including time to first token (not a pure decode rate).",
        "- `trunc` counts model calls that stopped with finish_reason=length (output budget exhausted).",
        "- `refusals` counts model calls that stopped with finish_reason=content_filter (e.g. Anthropic `refusal`).",
        "- cost uses configs/models.yaml pricing when set, otherwise litellm's estimate.",
    ]
    return "\n".join(lines) + "\n"


def build_report(run_dir: Path, by_category: bool = False) -> tuple[list[Row], str]:
    results = latest_results(run_dir / "results.jsonl")
    rows = aggregate(results, by_category=by_category)
    md = to_markdown(rows, f"Report: {run_dir.name}")
    return rows, md

