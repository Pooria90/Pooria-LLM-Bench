"""Reference-based scoring helpers.

Railtracks' evaluators (LLM judge, tool-use stats, inference stats) have no notion
of a reference answer, so the deterministic scorers live here.
"""

from __future__ import annotations

import math
import re
import string
from typing import Iterable

from .tasks.base import Score

_ANSWER_PATTERNS = [
    r"(?:final\s+answer|answer)\s*(?:is|:)\s*\**\s*(.+)",
]
_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"


def normalize(text: str) -> str:
    """Lowercase, trim, collapse whitespace, strip surrounding punctuation and markup."""
    text = str(text).strip().lower()
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(string.punctuation + " ")


def extract_answer(text: str) -> str:
    """Pull the final answer out of free text.

    Prefers an explicit "Answer: X" / "Final answer: X" line (the last one), and
    otherwise falls back to the last non-empty line.
    """
    text = str(text or "").strip()
    for pattern in _ANSWER_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip().splitlines()[0].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def has_answer_line(text: str) -> bool:
    """Whether the model followed the "Answer: X" output format."""
    return any(re.search(p, str(text or ""), flags=re.IGNORECASE) for p in _ANSWER_PATTERNS)


def extract_number(text: str) -> float | None:
    """Last number in the answer portion of `text` (commas allowed as thousands separators)."""
    candidates = re.findall(_NUMBER, extract_answer(text)) or re.findall(_NUMBER, str(text))
    if not candidates:
        return None
    try:
        return float(candidates[-1].replace(",", ""))
    except ValueError:
        return None


def extract_choice(text: str, choices: Iterable[str] = "ABCD") -> str | None:
    """Extract a multiple-choice letter such as "(B)", "Answer: B" or a bare "B"."""
    letters = "".join(choices)
    text = str(text or "")
    explicit = re.findall(
        rf"(?:answer|option|choice)\s*(?:is|:)?\s*\(?\**([{letters}])\b", text, flags=re.IGNORECASE
    )
    if explicit:
        return explicit[-1].upper()
    bare = re.findall(rf"(?<![A-Za-z])\(?([{letters}])\)?(?![A-Za-z])", extract_answer(text))
    return bare[-1].upper() if bare else None


def exact_match(output: str, target: str | Iterable[str]) -> Score:
    targets = [target] if isinstance(target, str) else list(target)
    pred = extract_answer(output)
    ok = any(normalize(pred) == normalize(t) for t in targets)
    return Score(value=float(ok), correct=ok, extracted=pred, details={"answer_line": has_answer_line(output)})


def numeric_match(output: str, target: float, rel_tol: float = 1e-4, abs_tol: float = 1e-6) -> Score:
    pred = extract_number(output)
    ok = pred is not None and math.isclose(pred, float(target), rel_tol=rel_tol, abs_tol=abs_tol)
    return Score(
        value=float(ok), correct=ok, extracted=None if pred is None else repr(pred),
        details={"answer_line": has_answer_line(output)},
    )


def choice_match(output: str, target: str, choices: Iterable[str] = "ABCD") -> Score:
    pred = extract_choice(output, choices)
    ok = pred is not None and pred == target.upper()
    return Score(value=float(ok), correct=ok, extracted=pred, details={"answer_line": has_answer_line(output)})
