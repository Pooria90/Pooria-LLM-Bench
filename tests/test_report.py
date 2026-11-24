import pytest

from pooria_bench.report import aggregate, percentile, wilson_ci
from pooria_bench.results import CallSummary, SampleResult


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(77, 80)  # Radium's published Hal number
    assert lo == pytest.approx(0.8955, abs=1e-3)
    assert hi == pytest.approx(0.9872, abs=1e-3)
    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and lo < 1.0


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([], 50) is None


def _r(model, sid, correct, *, error=None, out=10, latency=1.0, cost=0.01, finish="stop"):
    return SampleResult(
        run_id="r", task="t", model=model, model_id=model, sample_id=sid,
        correct=None if error else correct, score=None if error else float(correct), error=error,
        latency_s=latency, n_calls=1, input_tokens=20, output_tokens=out, cost_usd=cost,
        finish_reasons=[finish], calls=[CallSummary(latency_s=latency / 2, output_tokens=out)],
    )


def test_aggregate_excludes_errors_from_accuracy_and_counts_truncation():
    rows = aggregate([
        _r("a", "1", True), _r("a", "2", False, finish="length"), _r("a", "3", True, error="boom"),
        _r("b", "1", True),
    ])
    a = next(r for r in rows if r.model == "a")
    assert (a.n, a.errors, a.correct) == (2, 1, 1)
    assert a.accuracy == 0.5
    assert a.truncated == 1
    assert a.cost_usd == pytest.approx(0.03)
    assert a.cost_per_correct_usd == pytest.approx(0.03)
    assert a.output_tok_per_s == pytest.approx(30 / 1.5)


def test_aggregate_counts_refusals_and_format_misses():
    r1 = _r("a", "1", False, finish="content_filter")
    r2 = _r("a", "2", True)
    r1.score_details = {"format_ok": False}
    r2.score_details = {"answer_line": False}  # legacy key still understood
    (row,) = aggregate([r1, r2])
    assert row.refusals == 1 and row.format_misses == 2
