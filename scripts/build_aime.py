"""Build data/aime<year>.jsonl from MathArena's AIME datasets.

Source of truth is MathArena (Hugging Face), which gives every figure as text
(TikZ, Asymptote, a LaTeX array, or an ASCII grid), so each problem can go to a
text-only model. Answers are cross-checked against two independent copies; any
disagreement aborts the build.

Usage:  uv run python scripts/build_aime.py 2025
        uv run python scripts/build_aime.py 2026
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROWS_API = "https://datasets-server.huggingface.co/rows?"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
AOPS = "https://artofproblemsolving.com/wiki/index.php/{year}_AIME_{section}_Problems/Problem_{number}"

# Per year: the MathArena dataset, plus (dataset, config, split, row -> (index 1..30, answer)) checks.
SOURCES = {
    2025: {
        "matharena": ("MathArena/aime_2025", "default", "train"),
        "checks": [
            ("yentinglin/aime_2025", "default", "train", lambda r: (int(r["id"]) + 1, int(r["answer"]))),
            ("math-ai/aime25", "default", "test", lambda r: (int(r["id"]) + 1, int(r["answer"]))),
        ],
    },
    2026: {
        "matharena": ("MathArena/aime_2026", "default", "train"),
        "checks": [
            ("math-ai/aime26", "default", "test", lambda r: (int(r["id"]), int(r["answer"]))),
            ("zhenghaoxu/aime-2026", "default", "validation",
             lambda r: (int(r["metadata"]["extra_info"]["index"]), int(r["label"]))),
        ],
    },
}


def fetch_rows(ds: str, cfg: str, split: str) -> list[dict]:
    query = urllib.parse.urlencode({"dataset": ds, "config": cfg, "split": split, "offset": 0, "length": 100})
    with urllib.request.urlopen(ROWS_API + query, timeout=60) as resp:
        return [r["row"] for r in json.load(resp)["rows"]]


def figure_kind(problem: str) -> str | None:
    if r"\begin{tikzpicture}" in problem:
        return "tikz"
    if r"\begin{asy}" in problem or "[asy]" in problem:
        return "asy"
    if r"\begin{array}" in problem and r"\hline" in problem:
        return "latex-array"
    if len(re.findall(r"(?m)^\|.*\|[ \t]*$", problem)) >= 3:
        return "ascii-grid"
    return None


def build(year: int) -> list[dict]:
    src = SOURCES[year]
    rows = sorted(fetch_rows(*src["matharena"]), key=lambda r: int(r["problem_idx"]))
    assert len(rows) == 30, f"expected 30 problems, got {len(rows)}"
    checks = []
    for ds, cfg, split, key in src["checks"]:
        answers = dict(key(r) for r in fetch_rows(ds, cfg, split))
        assert len(answers) == 30, f"{ds}: expected 30 answers"
        checks.append((ds, answers))

    records = []
    for row in rows:
        idx, answer = int(row["problem_idx"]), int(row["answer"])
        for ds, answers in checks:
            if answers[idx] != answer:
                raise SystemExit(f"Answer mismatch for problem {idx}: MathArena={answer}, {ds}={answers[idx]}")
        section, number = ("I", idx) if idx <= 15 else ("II", idx - 15)
        records.append(
            {
                "id": f"{section}-{number:02d}",
                "section": section,
                "number": number,
                "problem": row["problem"],
                "answer": answer,
                "problem_type": list(row.get("problem_type") or []),
                "figure": figure_kind(row["problem"]),
                "source": src["matharena"][0],
                "aops_url": AOPS.format(year=year, section=section, number=number),
            }
        )
    return records


def main() -> int:
    years = [int(a) for a in sys.argv[1:]] or sorted(SOURCES)
    for year in years:
        records = build(year)
        out = DATA_DIR / f"aime{year}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        figs = [f"{r['id']}:{r['figure']}" for r in records if r["figure"]]
        print(f"Wrote {len(records)} problems to {out} (figures as text: {', '.join(figs) or 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
