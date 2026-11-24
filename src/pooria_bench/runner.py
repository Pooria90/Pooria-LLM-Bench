"""Experiment runner: task x models x repeats -> results.jsonl.

Railtracks evaluates *past* sessions but has no runner that invokes an agent over
a dataset, so this module is that runner. Each sample runs through a Railtracks
Flow built for its model; usage comes from the recording model middleware.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
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
    sample_ids: list[str] | None = None  # run only these sample ids
    categories: list[str] | None = None  # run only samples whose metadata category is listed


def select_samples(task: Task, cfg: RunConfig) -> list[Sample]:
    samples = task.load(cfg.limit if not (cfg.sample_ids or cfg.categories) else None)
    if cfg.sample_ids:
        wanted = set(cfg.sample_ids)
        unknown = wanted - {s.id for s in samples}
        if unknown:
            raise KeyError(f"Unknown sample ids for {task.name}: {sorted(unknown)}")
        samples = [s for s in samples if s.id in wanted]
    if cfg.categories:
        samples = [s for s in samples if s.metadata.get("category") in set(cfg.categories)]
    return samples[: cfg.limit] if cfg.limit else samples


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


class _Transcript(BaseModel):
    model: str
    sample_id: str
    repeat: int
    histories: list[dict[str, Any]]


def _write_samples(path: Path, samples: list[Sample]) -> None:
    """Samples as run, merged with any already saved (resumes may add samples)."""
    existing: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing[json.loads(line)["id"]] = line
    for smp in samples:
        existing[smp.id] = smp.model_dump_json()
    path.write_text("".join(line + "\n" for line in existing.values()), encoding="utf-8")


def _message_dict(m: Any) -> dict[str, Any]:
    out = {"role": getattr(m.role, "value", str(m.role)), "content": str(m.content)}
    reasoning = getattr(m, "reasoning_content", None)
    if reasoning:
        out["reasoning"] = reasoning
    return out


def _transcript(conn: rt.FlowConnection) -> list[dict[str, Any]]:
    out = []
    try:
        for hist in conn.message_histories():
            out.append(
                {
                    "node": hist.node_name,
                    "messages": [_message_dict(m) for m in hist.message_history],
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
    samples = select_samples(task, cfg)
    # Railtracks runs every model call synchronously in a worker thread (asyncio.to_thread),
    # so the default executor's size silently caps concurrency, and time spent waiting for
    # a free thread would be counted as model latency. Size it for the planned load.
    workers = max(32, 2 * cfg.concurrency * len(specs) + 8)
    asyncio.get_running_loop().set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pb-call")
    )
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
    _write_samples(run_dir / "samples.jsonl", samples)

    writer = ResultWriter(results_path)
    transcript_writer = ResultWriter(transcripts_path) if cfg.save_transcripts else None
    limiters: dict[str, RateLimiter] = {}
    slots: dict[str, asyncio.Semaphore] = {}
    for s in specs:
        if s.rpm and s.group not in limiters:
            limiters[s.group] = RateLimiter(s.rpm)
        if s.max_concurrent and s.group not in slots:
            slots[s.group] = asyncio.Semaphore(s.max_concurrent)
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
            trace = SampleTrace(limiter=limiters.get(spec.group), slots=slots.get(spec.group))
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
                last = trace.calls[-1] if trace.calls else None
                if last is not None and last.error is None and last.finish_reason in ("length", "content_filter"):
                    # The model call itself succeeded but left no usable text (output budget
                    # exhausted, or a provider content filter). Railtracks raises for that, but
                    # it is a model failure: score it as wrong instead of dropping it as an error.
                    score = task.score(sample, None)
                    score.details["empty_output"] = last.finish_reason
                else:
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


def count_pending(task: Task, specs: list[ModelSpec], cfg: RunConfig) -> int:
    samples = select_samples(task, cfg)
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
