"""Build data/aime2025.jsonl from MathArena's AIME 2025 dataset.

Source of truth: MathArena/aime_2025 (Hugging Face), which already encodes the six
figure problems as text (TikZ, Asymptote, or a LaTeX array), so every problem can
be given to a text-only model. Answers are cross-checked against two independent
copies (yentinglin/aime_2025 and math-ai/aime25); any disagreement aborts the build.

Usage:  uv run python scripts/build_aime2025.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

ROWS_API = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset=0&length=100"
OUT = Path(__file__).resolve().parents[1] / "data" / "aime2025.jsonl"


def fetch_rows(ds: str, cfg: str, split: str) -> list[dict]:
    with urllib.request.urlopen(ROWS_API.format(ds=ds, cfg=cfg, split=split), timeout=60) as resp:
        return [r["row"] for r in json.load(resp)["rows"]]


def figure_kind(problem: str) -> str | None:
    if r"\begin{tikzpicture}" in problem:
        return "tikz"
    if r"\begin{asy}" in problem:
        return "asy"
    if r"\begin{array}" in problem and r"\hline" in problem:
        return "latex-array"
    return None


def main() -> int:
    matharena = fetch_rows("MathArena/aime_2025", "default", "train")
    yentinglin = fetch_rows("yentinglin/aime_2025", "default", "train")
    mathai = fetch_rows("math-ai/aime25", "default", "test")
    assert len(matharena) == len(yentinglin) == len(mathai) == 30, "expected 30 problems in every source"

    records = []
    for i, row in enumerate(sorted(matharena, key=lambda r: int(r["problem_idx"]))):
        idx = int(row["problem_idx"])
        answer = int(row["answer"])
        others = {int(yentinglin[i]["answer"]), int(mathai[i]["answer"])}
        if others != {answer}:
            print(f"Answer mismatch for problem {idx}: MathArena={answer}, others={others}", file=sys.stderr)
            return 1
        section, number = ("I", idx) if idx <= 15 else ("II", idx - 15)
        url = yentinglin[i]["url"]
        assert re.search(rf"2025_AIME_{section}_Problems/Problem_{number}$", url), (idx, url)
        fig = figure_kind(row["problem"])
        records.append(
            {
                "id": f"{section}-{number:02d}",
                "section": section,
                "number": number,
                "problem": row["problem"],
                "answer": answer,
                "problem_type": list(row["problem_type"]),
                "figure": fig,
                "source": "MathArena/aime_2025",
                "aops_url": url,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_fig = sum(1 for r in records if r["figure"])
    print(f"Wrote {len(records)} problems ({n_fig} with figures as text) to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
