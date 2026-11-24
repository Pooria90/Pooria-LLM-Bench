"""Experiment runner: task x models x repeats -> results.jsonl.

Railtracks evaluates *past* sessions but has no runner that invokes an agent over
a dataset, so this module is that runner. Each sample runs through a Railtracks
Flow built for its model; usage comes from the recording model middleware.
"""

from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import railtracks as rt
from pydantic import BaseModel

from .instrument import CURRENT_TRACE, SampleTrace
from .models import ModelSpec, build_llm
from .ratelimit import RateLimiter
from .results import CallSummary, ResultWriter, SampleResult, latest_results
from .tasks.base import BuildContext, Sample, Task


@dataclass
class RunConfig:
    limit: int | None = None
    repeats: int = 1
    concurrency: int = 4  # in-flight samples per model
    timeout_s: float = 300.0  # per sample, whole agent loop
    out_dir: Path = Path("runs")
    run_dir: Path | None = None  # set to resume an existing run
    retry_errors: bool = True  # on resume, re-run samples that errored
    save_transcripts: bool = True
    overrides: dict[str, Any] = field(default_factory=dict)  # extra rt.llm params for every model


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def describe_error(exc: BaseException) -> str:
    """One-line error with the root provider cause.

    Railtracks wraps provider failures (e.g. LLMRateLimitError around a RetryError
    around litellm's exception) and its message omits the underlying status/text.
    """
    parts = [f"{type(exc).__name__}: {str(exc).strip().splitlines()[0] if str(exc).strip() else ''}"]
    seen: set[int] = set()
    cur: BaseException | None = exc
    root: BaseException | None = None
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        inner = getattr(cur, "exception_list", None)
        if inner:
            root = inner[-1]
            parts.append(f"attempts={len(inner)}")
        cur = cur.__cause__ or cur.__context__
        if cur is not None:
            root = root or cur
    if root is not None and root is not exc:
        parts.append(f"root={type(root).__name__}: {' '.join(str(root).split())[:400]}")
    return " | ".join(parts)[:2000]


def _summarize(spec: ModelSpec, trace: SampleTrace) -> dict[str, Any]:
    calls = trace.calls
    tot = lambda attr: sum(getattr(c, attr) or 0 for c in calls)  # noqa: E731
    in_tok, out_tok, cached = tot("input_tokens"), tot("output_tokens"), tot("cached_tokens")
    provider_costs = [c.provider_cost_usd for c in calls if c.provider_cost_usd is not None]
    provider_cost = sum(provider_costs) if provider_costs else None
    cost = spec.pricing.cost(in_tok, out_tok, cached) if spec.pricing else provider_cost
    return dict(
        n_calls=len(calls),
        input_tokens=in_tok,
        output_tokens=out_tok,
        reasoning_tokens=tot("reasoning_tokens"),
        cached_tokens=cached,
        cost_usd=cost,
        provider_cost_usd=provider_cost,
        finish_reasons=[c.finish_reason for c in calls],
        calls=[CallSummary(**{k: getattr(c, k) for k in CallSummary.model_fields}) for c in calls],
    )


def _transcript(conn: rt.FlowConnection) -> list[dict[str, Any]]:
    out = []
    try:
        for hist in conn.message_histories():
            out.append(
                {
                    "node": hist.node_name,
                    "messages": [{"role": str(m.role), "content": str(m.content)} for m in hist.message_history],
                }
            )
    except Exception as exc:  # transcripts are best-effort debugging aid
        out.append({"error": repr(exc)})
    return out


async def run_benchmark(
    task: Task,
    specs: list[ModelSpec],
    cfg: RunConfig,
    on_result: Callable[[SampleResult], None] | None = None,
) -> Path:
    samples = task.load(cfg.limit)
    run_dir = cfg.run_dir or cfg.out_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{task.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    results_path = run_dir / "results.jsonl"
    transcripts_path = run_dir / "transcripts.jsonl"

    done = {
        r.key for r in latest_results(results_path) if not (cfg.retry_errors and r.error)
    }

    manifest_path = run_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "task": task.name,
        "models": {s.name: s.model_dump(mode="json") for s in specs},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
        "n_samples": len(samples),
        "railtracks_version": version("railtracks"),
        "litellm_version": version("litellm"),
        "python": platform.python_version(),
        "git_sha": _git_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if manifest_path.exists():  # resuming: keep the original, append the new invocation
        old = json.loads(manifest_path.read_text())
        old.setdefault("resumed", []).append(manifest)
        manifest = old
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    writer = ResultWriter(results_path)
    transcript_writer = ResultWriter(transcripts_path) if cfg.save_transcripts else None
    limiters: dict[str, RateLimiter] = {}
    for s in specs:
        if s.rpm and s.group not in limiters:
            limiters[s.group] = RateLimiter(s.rpm)
    semaphores = {s.name: asyncio.Semaphore(cfg.concurrency) for s in specs}
    flows = {
        s.name: rt.Flow(
            f"{task.name}:{s.name}",
            entry_point=task.build(BuildContext(llm=build_llm(s, **cfg.overrides), model_name=s.name)),
            timeout=cfg.timeout_s,
        )
        for s in specs
    }

    async def run_one(spec: ModelSpec, sample: Sample, repeat: int) -> None:
        async with semaphores[spec.name]:
            trace = SampleTrace(limiter=limiters.get(spec.group))
            token = CURRENT_TRACE.set(trace)
            started = datetime.now(timezone.utc).isoformat()
            output = error = score = None
            conn = flows[spec.name].connect()
            t0 = time.perf_counter()
            try:
                args, kwargs = task.make_input(sample)
                result = await asyncio.wait_for(conn.ainvoke(*args, **kwargs), cfg.timeout_s + 30)
                output = task.extract_output(result)
                score = task.score(sample, output)
            except Exception as exc:
                error = describe_error(exc)
            finally:
                latency = time.perf_counter() - t0
                CURRENT_TRACE.reset(token)

            record = SampleResult(
                run_id=run_id,
                task=task.name,
                model=spec.name,
                model_id=spec.model_id,
                sample_id=sample.id,
                repeat=repeat,
                category=sample.metadata.get("category"),
                output=output,
                target=sample.target,
                extracted=score.extracted if score else None,
                score=score.value if score else None,
                correct=score.correct if score else None,
                score_details=score.details if score else {},
                error=error,
                latency_s=latency,
                started_at=started,
                **_summarize(spec, trace),
            )
            await writer.write(record)
            if transcript_writer is not None:
                await transcript_writer.write(
                    _Transcript(model=spec.name, sample_id=sample.id, repeat=repeat, histories=_transcript(conn))
                )
            if on_result is not None:
                on_result(record)

    # Interleave models so they progress evenly and share the same time window.
    jobs = [
        run_one(spec, sample, rep)
        for sample in samples
        for rep in range(cfg.repeats)
        for spec in specs
        if (spec.name, sample.id, rep) not in done
    ]
    await asyncio.gather(*jobs)
    return run_dir



class _Transcript(BaseModel):
    model: str
    sample_id: str
    repeat: int
    histories: list[dict[str, Any]]


def count_pending(task: Task, specs: list[ModelSpec], cfg: RunConfig) -> int:
    samples = task.load(cfg.limit)
    done: set[tuple[str, str, int]] = set()
    if cfg.run_dir is not None:
        done = {
            r.key
            for r in latest_results(cfg.run_dir / "results.jsonl")
            if not (cfg.retry_errors and r.error)
        }
    return sum(
        1
        for s in samples
        for rep in range(cfg.repeats)
        for spec in specs
        if (spec.name, s.id, rep) not in done
    )



def rescore_run(run_dir: Path, task: Task) -> int:
    """Re-apply the task's current scorer to saved outputs (no model calls).

    Tool usage is rebuilt from the stored per-call records, so scorers that read
    the trace (e.g. smoke-tools) see the same data they saw live. Updated records
    are appended; readers keep the last record per (model, sample, repeat).
    """
    from .instrument import CallRecord

    samples = {s.id: s for s in task.load()}
    path = run_dir / "results.jsonl"
    updated = []
    for r in latest_results(path):
        if r.error is not None or r.sample_id not in samples:
            continue
        trace = SampleTrace(calls=[CallRecord(**c.model_dump()) for c in r.calls])
        token = CURRENT_TRACE.set(trace)
        try:
            score = task.score(samples[r.sample_id], r.output)
        finally:
            CURRENT_TRACE.reset(token)
        updated.append(r.model_copy(update={
            "score": score.value, "correct": score.correct,
            "extracted": score.extracted, "score_details": score.details,
        }))
    with path.open("a", encoding="utf-8") as fh:
        for r in updated:
            fh.write(r.model_dump_json() + "\n")
    return len(updated)
