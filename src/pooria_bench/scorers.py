"""Reference-based scoring helpers.

Railtracks' evaluators (LLM judge, tool-use stats, inference stats) have no notion
of a reference answer, so the deterministic scorers live here.

Every scorer reports `details["format_ok"]`: whether the model used the requested
answer format (an "Answer:" line, or `\\boxed{}` for math). Answers in the wrong
format are still scored on a best-effort basis but counted separately in reports.
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from typing import Iterable

from .tasks.base import Score

_ANSWER_PATTERNS = [
    r"(?:final\s+answer|answer)\s*(?:is|:)\s*\**\s*(.+)",
]
_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"


def strip_invisible(text: str) -> str:
    """Remove Unicode format characters (zero-width spaces/joiners, tag characters, BOM).

    Seen in practice: Claude Opus 5.5 occasionally prefixes replies with U+E007F
    (CANCEL TAG), and its refusals can consist entirely of zero-width characters.
    """
    return "".join(ch for ch in str(text or "") if unicodedata.category(ch) != "Cf")


def normalize(text: str) -> str:
    """Lowercase, trim, collapse whitespace, strip surrounding punctuation and markup."""
    text = strip_invisible(text).strip().lower()
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(string.punctuation + " ")


def extract_answer(text: str) -> str:
    """Pull the final answer out of free text.

    Prefers an explicit "Answer: X" / "Final answer: X" line (the last one), and
    otherwise falls back to the last non-empty line.
    """
    text = strip_invisible(text).strip()
    for pattern in _ANSWER_PATTERNS:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip().splitlines()[0].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def has_answer_line(text: str) -> bool:
    """Whether the model followed the "Answer: X" output format."""
    text = strip_invisible(text)
    return any(re.search(p, text, flags=re.IGNORECASE) for p in _ANSWER_PATTERNS)


def extract_number(text: str) -> float | None:
    """Last number in the answer portion of `text` (commas allowed as thousands separators)."""
    text = strip_invisible(text)
    candidates = re.findall(_NUMBER, extract_answer(text)) or re.findall(_NUMBER, text)
    if not candidates:
        return None
    try:
        return float(candidates[-1].replace(",", ""))
    except ValueError:
        return None


def extract_choice(text: str, choices: Iterable[str] = "ABCD") -> str | None:
    """Extract a multiple-choice letter such as "(B)", "Answer: B" or a bare "B"."""
    letters = "".join(choices)
    text = strip_invisible(text)
    explicit = re.findall(
        rf"(?:answer|option|choice)\s*(?:is|:)?\s*\(?\**([{letters}])\b", text, flags=re.IGNORECASE
    )
    if explicit:
        return explicit[-1].upper()
    bare = re.findall(rf"(?<![A-Za-z])\(?([{letters}])\)?(?![A-Za-z])", extract_answer(text))
    return bare[-1].upper() if bare else None


def extract_boxed(text: str) -> str | None:
    """Content of the last `\\boxed{...}` (or `\\fbox{...}`), with nested braces handled."""
    text = strip_invisible(text)
    starts = [m.end() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    for start in reversed(starts):
        depth, i = 1, start
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth == 0:
            return text[start : i - 1].strip()
    return None


def parse_integer(text: str | None) -> int | None:
    """Parse an integer answer written in LaTeX, e.g. `070`, `\\text{259}`, `336^\\circ`, `1{,}024`.

    If the text is an equation (`m+n=293`), the right-hand side of the last `=` is used.
    Anything else that is not a plain integer returns None rather than guessing.
    """
    if text is None:
        return None
    s = strip_invisible(text)
    s = re.sub(r"\\(?:text|textbf|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    if "=" in s:
        s = s.rsplit("=", 1)[1]
    s = re.sub(r"\^\s*\{?\\circ\}?|\\?%|\\?\$|\\[,;!: ]|~|\{,\}", "", s)
    s = s.replace(",", "").strip().rstrip(".").strip()
    return int(s) if re.fullmatch(r"[-+]?\d+", s) else None


def exact_match(output: str, target: str | Iterable[str]) -> Score:
    targets = [target] if isinstance(target, str) else list(target)
    pred = extract_answer(output)
    ok = any(normalize(pred) == normalize(t) for t in targets)
    return Score(value=float(ok), correct=ok, extracted=pred, details={"format_ok": has_answer_line(output)})


def numeric_match(output: str, target: float, rel_tol: float = 1e-4, abs_tol: float = 1e-6) -> Score:
    pred = extract_number(output)
    ok = pred is not None and math.isclose(pred, float(target), rel_tol=rel_tol, abs_tol=abs_tol)
    return Score(
        value=float(ok), correct=ok, extracted=None if pred is None else repr(pred),
        details={"format_ok": has_answer_line(output)},
    )


def choice_match(output: str, target: str, choices: Iterable[str] = "ABCD") -> Score:
    pred = extract_choice(output, choices)
    ok = pred is not None and pred == target.upper()
    return Score(value=float(ok), correct=ok, extracted=pred, details={"format_ok": has_answer_line(output)})


def boxed_integer_match(output: str, target: int) -> Score:
    """Score a `\\boxed{}` integer answer (AIME-style).

    Uses the last `\\boxed{}`. Without one, falls back to an "Answer:" line or a reply
    that is nothing but an integer, and marks the format as not followed.
    """
    boxed = extract_boxed(output)
    bare = strip_invisible(output).strip().strip("*$ ").rstrip(".").strip()
    if boxed is not None:
        raw = boxed
    elif has_answer_line(output):
        raw = extract_answer(output)
    elif re.fullmatch(r"[-+]?\d+", bare):
        raw = bare
    else:
        raw = None
    pred = parse_integer(raw)
    ok = pred is not None and pred == int(target)
    extracted = str(pred) if pred is not None else (raw[:80] if raw else None)
    return Score(value=float(ok), correct=ok, extracted=extracted, details={"format_ok": boxed is not None})
