# Results dashboard

A local page for browsing benchmark runs one problem at a time: the problem (with
LaTeX rendered), the correct answer, and each model's answer, verdict, tokens,
time, cost, reasoning and exact prompt.

```bash
uv run python dashboard/serve.py          # then open http://127.0.0.1:8765
```

Options: `--port 9000`, `--runs path/to/runs`, `--host 0.0.0.0` (to reach it from
another machine; it has no authentication, so only do that on a trusted network).

**Using it**

- Pick a run at the top. In-progress runs turn on auto-refresh (every 15 s).
- Click a row in the problem table, then step with ← / → (or j / k). `r` reloads.
- Filters: models disagree, no model correct, out of budget / errors, answered
  with no reasoning, category (`text` / `figure`) and problem type.
- Each answer card can show the provider's reasoning, the exact prompt and the raw answer text.

**Verdicts**

- ✓ correct
- ✗ wrong answer
- ✂ out of budget: the output limit was hit before an answer
- ⊘ refused: content filter
- ! error: infrastructure failure; excluded from accuracy
- `0r` marks answers produced with zero reasoning tokens. On hard problems, that suggests recall.

## Why it is separate

The dashboard imports nothing from `pooria_bench`. It only reads the run
directory format, so it can be reworked freely:

| File | Contents |
| --- | --- |
| `manifest.json` | task, models (with params), config, versions; `resumed` lists later invocations |
| `samples.jsonl` | one sample per line: `id`, `input`, `target`, `metadata` |
| `results.jsonl` | one result per line; the last line per (model, sample, repeat) wins |
| `transcripts.jsonl` | `model`, `sample_id`, `repeat`, `histories[].messages[]` (`role`, `content`, optional `reasoning`) |

`serve.py` is standard-library Python. The page (`static/`) is plain HTML/CSS/JS,
using KaTeX, marked and DOMPurify from cdnjs, so the browser needs internet access.
