"""Command line: `pooria-bench models | tasks | ping | run | report`."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from .models import build_llm, load_models, resolve_models
from .report import COLUMNS, build_report
from .runner import RunConfig, count_pending, rescore_run, run_benchmark
from .tasks import TASKS, get_task

app = typer.Typer(add_completion=False, no_args_is_help=True, help="LLM benchmark harness built on Railtracks.")
console = Console(width=None if Console().is_terminal else 200)

ModelsOpt = typer.Option(..., "--model", "-m", help="Model name from configs/models.yaml, or tag:<tag>. Repeatable.")
ConfigOpt = typer.Option(None, "--config", help="Path to a models.yaml (default: configs/models.yaml).")


@app.callback()
def _setup() -> None:
    load_dotenv()


@app.command()
def models(config: Optional[Path] = ConfigOpt) -> None:
    """List configured models and whether their API key is set."""
    table = Table("name", "provider", "model_id", "$ in/out per 1M", "rpm", "key", "tags")
    for s in load_models(config).values():
        price = f"{s.pricing.input:g} / {s.pricing.output:g}" if s.pricing else "litellm"
        table.add_row(s.name, s.provider, s.model_id, price, str(s.rpm or ""),
                      "yes" if s.key_available() else "[red]missing[/red]", ",".join(s.tags))
    console.print(table)


@app.command()
def tasks() -> None:
    """List registered tasks."""
    table = Table("task", "samples", "description")
    for name, cls in TASKS.items():
        table.add_row(name, str(len(cls().load())), cls.description)
    console.print(table)


@app.command()
def ping(
    model: list[str] = ModelsOpt,
    prompt: str = typer.Option("What is 17*23? Reply with just the number.", "--prompt"),
    config: Optional[Path] = ConfigOpt,
) -> None:
    """Send one message to each model through Railtracks and print usage."""
    import railtracks as rt

    from .instrument import CURRENT_TRACE, SampleTrace
    from .tasks.base import BuildContext

    async def one(spec):
        trace = SampleTrace()
        CURRENT_TRACE.set(trace)
        agent = BuildContext(llm=build_llm(spec), model_name=spec.name).agent(name="Ping")
        t0 = time.perf_counter()
        try:
            res = await rt.Flow("ping", entry_point=agent).ainvoke(prompt)
            return spec, res.content, None, trace, time.perf_counter() - t0
        except Exception as exc:
            return spec, None, f"{type(exc).__name__}: {exc}", trace, time.perf_counter() - t0

    async def main():
        return await asyncio.gather(*(one(s) for s in resolve_models(model, load_models(config))))

    table = Table("model", "reply", "s", "in", "out", "reasoning", "cached", "finish", "error")
    for spec, content, err, trace, dt in asyncio.run(main()):
        c = trace.calls[-1] if trace.calls else None
        table.add_row(spec.name, (content or "")[:60].replace("\n", " "), f"{dt:.2f}",
                      *(str(getattr(c, a, "")) if c else "" for a in
                        ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens", "finish_reason")),
                      (err or "")[:80])
    console.print(table)


@app.command()
def run(
    task: str = typer.Argument(..., help="Task name (see `pooria-bench tasks`)."),
    model: list[str] = ModelsOpt,
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Only the first N samples."),
    repeats: int = typer.Option(1, "--repeats", "-r", help="Runs per sample (for variance)."),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="In-flight samples per model."),
    timeout: float = typer.Option(300.0, "--timeout", help="Seconds per sample."),
    out: Path = typer.Option(Path("runs"), "--out", help="Parent directory for new runs."),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Existing run dir to continue."),
    param: list[str] = typer.Option([], "--param", "-p", help="Extra LLM param for every model, key=json_value."),
    only: list[str] = typer.Option([], "--only", help="Run only these sample ids (repeatable or comma-separated)."),
    category: list[str] = typer.Option([], "--category", help="Run only samples in this metadata category."),
    config: Optional[Path] = ConfigOpt,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Run a task against one or more models and print the report."""
    t = get_task(task)
    specs = resolve_models(model, load_models(config))
    missing = [s.name for s in specs if not s.key_available()]
    if missing:
        raise typer.BadParameter(f"API key missing for: {', '.join(missing)}")
    overrides = {}
    for p in param:
        k, _, v = p.partition("=")
        try:
            overrides[k] = json.loads(v)
        except json.JSONDecodeError:
            overrides[k] = v
    ids = [i.strip() for chunk in only for i in chunk.split(",") if i.strip()]
    cfg = RunConfig(limit=limit, repeats=repeats, concurrency=concurrency, timeout_s=timeout,
                    out_dir=out, run_dir=resume, overrides=overrides,
                    sample_ids=ids or None, categories=category or None)

    pending = count_pending(t, specs, cfg)
    console.print(f"[bold]{t.name}[/bold]: {pending} sample runs pending across {len(specs)} model(s): "
                  + ", ".join(s.name for s in specs))
    if pending and not yes and not typer.confirm("Proceed?", default=True):
        raise typer.Abort()

    with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=console) as progress:
        bar = progress.add_task(t.name, total=pending)

        def on_result(r):
            progress.advance(bar)
            if r.error:
                progress.console.print(f"[red]{r.model} {r.sample_id}: {r.error[:160]}[/red]")

        run_dir = asyncio.run(run_benchmark(t, specs, cfg, on_result=on_result))

    console.print(f"Results: {run_dir}")
    _print_report(run_dir, by_category=False)


@app.command()
def report(
    run_dir: Path = typer.Argument(..., help="A run directory under runs/."),
    by_category: bool = typer.Option(False, "--by-category", help="Break down by sample category."),
) -> None:
    """Print (and write report.md for) an existing run."""
    _print_report(run_dir, by_category)


@app.command()
def rescore(run_dir: Path = typer.Argument(..., help="A run directory under runs/.")) -> None:
    """Re-score saved outputs with the task's current scorer (no model calls)."""
    task = get_task(json.loads((run_dir / "manifest.json").read_text())["task"])
    n = rescore_run(run_dir, task)
    console.print(f"Re-scored {n} records.")
    _print_report(run_dir, by_category=False)


def _print_report(run_dir: Path, by_category: bool) -> None:
    rows, md = build_report(run_dir, by_category=by_category)
    (run_dir / ("report_by_category.md" if by_category else "report.md")).write_text(md)
    cols = [c for c in COLUMNS if c[0] != "group" or by_category]
    table = Table(*(c[0] for c in cols), title=run_dir.name)
    for r in rows:
        table.add_row(*(f(r) for _, f in cols))
    console.print(table)


def main() -> None:
    app()
