"""End-to-end runner tests through real Railtracks flows with an offline model."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time

import pytest
from railtracks.llm.message import ToolMessage

from pooria_bench import runner as runner_mod
from pooria_bench.results import latest_results
from pooria_bench.runner import RunConfig, run_benchmark
from pooria_bench.tasks import Score, get_task

from .conftest import ScriptedLLM


def _patch_llm(monkeypatch, llm):
    monkeypatch.setattr(runner_mod, "build_llm", lambda spec, **kw: llm)


def _qa_script(messages, tools):
    question = str(messages[-1].content)
    answers = {"17 * 23": "391", "2**20": "1048575", "dodecagon": "12"}
    for key, val in answers.items():
        if key in question:
            return f"Let me think.\nAnswer: {val}"
    return "Answer: no idea"


def test_smoke_qa_end_to_end_records_usage_and_cost(tmp_path, monkeypatch, scripted_spec):
    _patch_llm(monkeypatch, ScriptedLLM(_qa_script, tokens=(100, 50)))
    task = get_task("smoke-qa")
    run_dir = asyncio.run(run_benchmark(task, [scripted_spec], RunConfig(out_dir=tmp_path)))

    results = {r.sample_id: r for r in latest_results(run_dir / "results.jsonl")}
    assert len(results) == 12
    assert results["arith-1"].correct and results["arith-2"].correct and results["fact-2"].correct
    assert not results["fact-3"].correct
    r = results["arith-1"]
    assert (r.n_calls, r.input_tokens, r.output_tokens) == (1, 100, 50)
    assert r.cost_usd == pytest.approx((100 * 1.0 + 50 * 2.0) / 1e6)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["railtracks_version"] and manifest["n_samples"] == 12
    assert (run_dir / "transcripts.jsonl").exists()


def _tools_script(messages, tools):
    """Look up the order, then its rate, then answer from the tool results."""
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    question = next(str(m.content) for m in messages if str(m.role).endswith("user"))
    order_id = re.search(r"A-\d{4}", question).group(0)
    if not tool_msgs:
        return [("lookup_order", {"order_id": order_id})]
    if len(tool_msgs) == 1:
        order = eval(tool_msgs[0].content.result)  # dict repr from the tool
        return [("get_exchange_rate", {"currency": order["currency"]})]
    order = eval(tool_msgs[0].content.result)
    rate = eval(tool_msgs[1].content.result)["usd_per_unit"]
    return f"Answer: {order['amount'] * rate:.2f}"


def test_smoke_tools_runs_real_tool_loop(tmp_path, monkeypatch, scripted_spec):
    _patch_llm(monkeypatch, ScriptedLLM(_tools_script))
    task = get_task("smoke-tools")
    cfg = RunConfig(out_dir=tmp_path, limit=3)
    run_dir = asyncio.run(run_benchmark(task, [scripted_spec], cfg))

    results = {r.sample_id: r for r in latest_results(run_dir / "results.jsonl")}
    t2 = results["t2"]
    assert t2.error is None, t2.error
    assert t2.correct, (t2.output, t2.target)
    assert t2.n_calls == 3
    assert t2.score_details["tools_used"] == ["get_exchange_rate", "lookup_order"]
    assert t2.score_details["missing_tools"] == []


def test_errors_are_recorded_and_resume_retries_only_failures(tmp_path, monkeypatch, scripted_spec):
    calls = {"n": 0}

    def flaky(messages, tools):
        calls["n"] += 1
        if "17 * 23" in str(messages[-1].content):
            raise RuntimeError("provider exploded")
        return "Answer: 0"

    _patch_llm(monkeypatch, ScriptedLLM(flaky))
    task = get_task("smoke-qa")
    run_dir = asyncio.run(run_benchmark(task, [scripted_spec], RunConfig(out_dir=tmp_path, limit=3)))
    first = {r.sample_id: r for r in latest_results(run_dir / "results.jsonl")}
    assert first["arith-1"].error and "provider exploded" in first["arith-1"].error
    assert first["arith-2"].error is None

    calls["n"] = 0
    _patch_llm(monkeypatch, ScriptedLLM(lambda m, t: "Answer: 391"))
    asyncio.run(run_benchmark(task, [scripted_spec], RunConfig(run_dir=run_dir, limit=3)))
    after = {r.sample_id: r for r in latest_results(run_dir / "results.jsonl")}
    assert after["arith-1"].error is None and after["arith-1"].correct
    # the two samples that succeeded the first time were not re-run
    lines = (run_dir / "results.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4


def test_rescore_applies_current_scorer_without_model_calls(tmp_path, monkeypatch, scripted_spec):
    _patch_llm(monkeypatch, ScriptedLLM(lambda m, t: "Answer: 391"))
    task = get_task("smoke-qa")
    run_dir = asyncio.run(run_benchmark(task, [scripted_spec], RunConfig(out_dir=tmp_path, limit=2)))
    assert {r.sample_id: r.correct for r in latest_results(run_dir / "results.jsonl")} == {
        "arith-1": True, "arith-2": False}

    monkeypatch.setattr(task, "score", lambda s, o: Score(value=1.0, correct=True))
    assert runner_mod.rescore_run(run_dir, task) == 2
    assert all(r.correct for r in latest_results(run_dir / "results.jsonl"))


def test_describe_error_surfaces_root_cause():
    try:
        try:
            raise ValueError("429 Too Many Requests: slow down")
        except ValueError as inner:
            raise RuntimeError("LLM call failed after retries") from inner
    except RuntimeError as exc:
        msg = runner_mod.describe_error(exc)
    assert "RuntimeError" in msg and "root=ValueError: 429 Too Many Requests" in msg


def test_samples_file_and_sample_selection(tmp_path, monkeypatch, scripted_spec):
    _patch_llm(monkeypatch, ScriptedLLM(lambda m, t: r"\boxed{70}"))
    task = get_task("aime2025")
    cfg = RunConfig(out_dir=tmp_path, sample_ids=["I-01", "II-03"])
    run_dir = asyncio.run(run_benchmark(task, [scripted_spec], cfg))
    saved = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text().splitlines()]
    assert [s["id"] for s in saved] == ["I-01", "II-03"]
    assert saved[1]["metadata"]["figure"] == "tikz" and "tikzpicture" in saved[1]["input"]
    results = {r.sample_id: r for r in latest_results(run_dir / "results.jsonl")}
    assert results["I-01"].correct and not results["II-03"].correct
    # a resume that adds an earlier sample keeps dataset order in samples.jsonl
    asyncio.run(run_benchmark(task, [scripted_spec], RunConfig(run_dir=run_dir, sample_ids=["I-01", "I-02", "II-03"])))
    order = [json.loads(line)["id"] for line in (run_dir / "samples.jsonl").read_text().splitlines()]
    assert order == ["I-01", "I-02", "II-03"]
    with pytest.raises(KeyError):
        runner_mod.select_samples(task, RunConfig(sample_ids=["nope"]))
    assert {s.id for s in runner_mod.select_samples(task, RunConfig(categories=["figure"]))} == {
        "I-02", "I-10", "I-11", "II-03", "II-05", "II-06"}


class _SlowLLM(ScriptedLLM):
    """Model that tracks how many calls are in flight at once.

    Railtracks runs the sync `_chat` in a worker thread even for async flows,
    so the overlap is tracked there, under a thread lock.
    """

    def __init__(self):
        super().__init__(lambda m, t: "Answer: 0")
        self.active = self.peak = 0
        self._lock = threading.Lock()

    def _chat(self, messages):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return self._respond(messages)


def test_max_concurrent_caps_calls_across_the_rate_group(tmp_path, monkeypatch, scripted_spec):
    llm = _SlowLLM()
    _patch_llm(monkeypatch, llm)
    a = scripted_spec.model_copy(update={"name": "a", "rate_group": "g", "max_concurrent": 2})
    b = scripted_spec.model_copy(update={"name": "b", "rate_group": "g", "max_concurrent": 2})
    asyncio.run(run_benchmark(get_task("smoke-qa"), [a, b], RunConfig(out_dir=tmp_path, concurrency=6)))
    assert llm.peak == 2


def test_empty_output_from_budget_exhaustion_is_scored_wrong_not_an_error(tmp_path, monkeypatch, scripted_spec):
    class TruncatingLLM(ScriptedLLM):
        def _respond(self, messages, tools=None):
            resp = super()._respond(messages, tools)
            resp.message_info.finish_reason = "length"  # what the instrumented models record
            return resp

    def empty(messages, tools):
        return None  # litellm/Railtracks give None content when the budget runs out

    _patch_llm(monkeypatch, TruncatingLLM(empty))
    task = get_task("aime2025")
    run_dir = asyncio.run(run_benchmark(task, [scripted_spec], RunConfig(out_dir=tmp_path, sample_ids=["I-01"])))
    (r,) = latest_results(run_dir / "results.jsonl")
    assert r.error is None, r.error
    assert r.correct is False
    assert r.finish_reasons == ["length"]
    assert r.score_details.get("empty_output") == "length"  # took the model-failure path
