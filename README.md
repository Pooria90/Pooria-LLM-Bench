# pooria-bench

An LLM benchmark harness built on [Railtracks](https://docs.railtracks.org). It was made to
benchmark Radium's models (Hal, Clarke, Tycho) against their claimed comparables, and to
pressure-test Railtracks as a harness framework (see [GAPS.md](GAPS.md)).

Tasks are written once as Railtracks agents, tools and flows. The runner rebuilds each task
around every model under test, so swapping or adding a model is a config change.

## Quickstart

```bash
uv sync
cp .env.example .env   # then fill in the keys you have
uv run pooria-bench models                     # registry + which keys are set
uv run pooria-bench tasks
uv run pooria-bench ping -m tag:radium         # one call per model, prints usage
uv run pooria-bench run smoke-qa -m tag:radium -m tag:radium-comparable
uv run pooria-bench report runs/<run-dir> --by-category
uv run pytest
```

`-m` takes model names from `configs/models.yaml` or `tag:<tag>`. Useful `run` flags:
`--limit N`, `--repeats R`, `--concurrency C` (per model), `--timeout S` (per sample),
`--param key=value` (extra LLM params for every model, e.g. `-p reasoning_effort='"low"'`),
and `--resume runs/<dir>`, which skips finished samples and retries errored ones.
After changing a scorer, run `rescore runs/<dir>` to re-score saved outputs without any model calls.

## Layout

```
configs/models.yaml        model registry: provider, model id, params, pricing, rate limits
src/pooria_bench/
  models.py                registry -> instrumented rt.llm instances
  instrument.py            model middleware: per-call timing, usage, rate limiting
  ratelimit.py             async token bucket shared per rate_group
  runner.py                task x models x repeats -> runs/<id>/results.jsonl (+ resume, rescore)
  results.py               SampleResult schema, JSONL IO
  report.py                accuracy + Wilson CI, p50/p95 latency, tokens, cost/correct
  scorers.py               exact / numeric / multiple-choice matching
  tasks/                   one module per task
tests/                     offline tests (scripted Railtracks model, no network)
notes/                     research notes (Radium, Railtracks assessment)
```

Each run writes `manifest.json` (models, config, versions, git sha), `results.jsonl`
(one line per model x sample x repeat, with per-call usage) and `transcripts.jsonl`.

## Adding a task

Subclass `Task` in `src/pooria_bench/tasks/`, implement `load`, `build` and `score`, and
register it in `tasks/__init__.py`. In `build`, create agents with `ctx.agent(...)`, never
`rt.agent_node(...)` directly. That helper attaches the recording/rate-limit middleware and
binds the model under test.

```python
class MyTask(Task):
    name = "my-task"
    description = "..."

    def load(self, limit=None):
        return [Sample(id="1", input="question", target="42")][:limit]

    def build(self, ctx):
        return ctx.agent(name="Solver", system_message="...", tool_nodes=[my_tool])

    def score(self, sample, output):
        return scorers.exact_match(output, sample.target)
```

## Adding a model

Add an entry to `configs/models.yaml`. Any OpenAI-compatible endpoint works with
`provider: openai_compatible` plus `api_base` and `api_key_env`. Set `pricing` when
litellm doesn't know the model, otherwise cost will be missing.
