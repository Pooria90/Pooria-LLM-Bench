> Initial assessment from before the harness existed. The living, evidence-backed list is [GAPS.md](../GAPS.md).

# Railtracks as a benchmark harness — assessment (v1.8.1, 2026-11-24)

Checked against the docs (docs.railtracks.org/llms.txt) and the installed `railtracks==1.8.1` source.

## What works out of the box
- **Model swap is one line.** `rt.llm.OpenAICompatibleProvider("hal-1.0", api_base="https://api.radium.cloud/v1", api_key=...)`
  constructs fine. It goes through litellm with `custom_llm_provider="openai"`, and extra kwargs (`seed`, `extra_body`, ...)
  pass straight through. Claude uses `rt.llm.AnthropicLLM`, OpenAI uses `rt.llm.OpenAILLM`.
- **Agentic tasks** (tool use, multi-turn, coding): `function_node` tools, the `agent_node` loop, `MaxCalls`/`Timeout`
  middleware, a Docker sandbox pattern, structured output via Pydantic `output_schema`.
- **Recording**: every run lands in `.railtracks/data/sessions`. `flow.connect()` exposes per-run message histories,
  and concurrent runs work with one connection each.
- **Retries**: `rt.llm.retries.ExponentialRetry` (Radium asks for a 9s initial backoff).
- **Usage stats**: `LLMInferenceEvaluator` reports input/output tokens, cost and latency per call.
- **Reasoning text** surfaces on `response.reasoning` when the provider returns it through litellm.

## Gaps for benchmarking (candidate issues/PRs)
1. **No experiment/dataset runner.** Evaluations only analyze *past* sessions. The docs say "Agent Experimentation"
   is on the roadmap. Missing: run dataset × models × repeats, a concurrency cap, resume/caching of finished items.
2. **No reference-based scoring.** `AgentDataPoint` has no expected-answer field. The built-in evaluators are LLM judge,
   tool-use stats and inference stats. Missing: exact match, regex/numeric match, MCQ letter extraction,
   JSON-schema validity, code-execution pass@k.
3. **Thin timing metrics.** `MessageInfo` has only total latency, measured with `time.time()` (not monotonic).
   Missing: time to first token, output tokens/s, p50/p95 aggregates. The streaming path takes latency
   at the usage chunk.
4. **Missing usage fields.** No reasoning tokens, cached tokens or `finish_reason` on `MessageInfo` (truncation is only logged).
5. **Cost is `None` for Radium.** `hal-1.0`/`clarke-1.0`/`tycho-1.0` aren't in litellm's price map. Workaround:
   `litellm.register_model(...)`. Better: a per-model pricing override on the provider.
6. **No client-side rate limiting.** No RPM/semaphore limiter. Radium enforces 1,000 RPM plus a 3× acceleration cap.
7. **No statistics or cross-model reports.** Aggregates are means only, with no confidence intervals or per-category breakdown.
8. **Reasoning control is model-dependent.** Hyperparameter validation is skipped for `OpenAICompatibleProvider`
   (by design), and Radium documents no reasoning toggle, so a run must record whether reasoning actually happened.
9. **Small doc/code inconsistencies:**
   - `evaluations/runners/_evaluate.py` mentions an `EvaluationDataset` type that doesn't exist in the package.
   - The Numerical-metrics docs use `from railtracks import evaluation as eval`, but the module is `evaluations`.
   - The JudgeEvaluator custom-prompt example uses `eval.JudgeEvaluator`.
   - The harness overview says plainly: "Not an evaluation harness".

## Proposed setup
- **Tasks** = Railtracks agents/flows plus tools, written once and parameterized by `llm`.
- **Model registry** = one config mapping names → `rt.llm.*` instances (Radium via OpenAICompatibleProvider,
  Claude via AnthropicLLM).
- **Thin runner of our own** (filling gaps 1–7): dataset loader → `flow.connect().ainvoke()` under a semaphore → scorer →
  JSONL results with usage/latency/cost → report with CIs.
- Every workaround is logged as a gap, to become a Railtracks issue/PR.
