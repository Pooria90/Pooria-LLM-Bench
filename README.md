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
uv run python dashboard/serve.py               # browse results at http://127.0.0.1:8765
uv run pytest
```

## AIME (2025, 2026)

```bash
uv run pooria-bench run aime2026 -m tag:aime64 --timeout 3600 -c 5   # 64k budget: Hal, GPT-5.5, Opus 4.7
uv run pooria-bench run aime2025 -m tag:aime-set --timeout 1800 -c 5 # 32k budget: Hal, GPT-5.5, Opus 4.6
```

- **Problems:** all 30 problems per year (`data/aime<year>.jsonl`; see `data/README.md`). Figures are
  given as text (TikZ, Asymptote, a LaTeX array or an ASCII grid) and tagged `category=figure`
  (`report --by-category` splits them out).
- **Model sets:** each is Radium Hal against GPT-5.5 and a Claude Opus model, all at
  `reasoning_effort=high` where available.
  - `aime64`: 65,536-token output budget, with Opus 4.7.
  - `aime-set`: 32,768 tokens (Hal's documented maximum), with Opus 4.6.
- **Prompt:** MathArena's: the problem, then "reason step by step… `\boxed{}`… an integer between 0
  and 999", with no system prompt.
- **Scoring:** the last `\boxed{}` integer. A reply that is only an integer is accepted but flagged as
  a format miss.
- **Running out of budget** counts as wrong (column `trunc`, ✂ in the dashboard).
- **Memorization check:** `no-think` counts answers produced with zero reasoning tokens. On AIME
  2025, Opus 4.7 answered hard problems instantly with no reasoning, so the 2025 set uses Opus 4.6.

`-m` takes model names from `configs/models.yaml` or `tag:<tag>`. Useful `run` flags:
`--limit N`, `--repeats R`, `--concurrency C` (per model), `--timeout S` (per sample),
`--param key=value` (extra LLM params for every model, e.g. `-p reasoning_effort='"low"'`),
`--only I-01,II-15` / `--category figure` (pick samples), and `--resume runs/<dir>`, which skips
finished samples and retries errored ones.
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
dashboard/                 standalone results browser (reads run folders only; see dashboard/README.md)
data/, scripts/            datasets and the scripts that build them
tests/                     offline tests (scripted Railtracks model, no network)
notes/                     research notes (Radium, Railtracks assessment)
```

Each run writes these files:

- `manifest.json`: models, config, versions, git sha
- `samples.jsonl`: the samples as run
- `results.jsonl`: one line per model × sample × repeat, with per-call usage
- `transcripts.jsonl`: the messages, including any reasoning the provider returned

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
