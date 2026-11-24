# Railtracks gaps found while building pooria-bench

A living log for pressure-testing Railtracks (v1.8.1) as a benchmark harness.
Each entry says what is missing, the evidence, what this repo does instead, and
what an upstream fix could look like. Candidates for Railtracks issues/PRs.

## What worked well

- **One-line model swap.** `OpenAICompatibleProvider` reached Radium and xAI unchanged, and
  `OpenAILLM`, `AnthropicLLM` and `GeminiLLM` worked for the rest: 14 models, no per-model code.
- **Model middleware (`rt.wrap_llm`)** is the right hook for per-call telemetry and pacing
  (`src/pooria_bench/instrument.py`).
- **ContextVars propagate into node execution**, so per-sample traces stay isolated under concurrency.
- **`flow.connect()` + `message_histories()`** gives full transcripts, including after failures.
- **`function_node` tools + `agent_node` loop** ran multi-step tool use on every provider tested.
- **`EchoLLM`** is easy to subclass into a scripted offline model, so the real Railtracks
  loop is unit-testable (`tests/conftest.py`).

## Gaps

| # | Gap | Evidence | Our workaround | Upstream idea |
|---|-----|----------|----------------|---------------|
| 1 | No experiment runner: evaluations only read *past* sessions | docs "Evaluations > Preface": agent experimentation is future work | `runner.py`: dataset x models x repeats, per-model concurrency, resume | `rt.evaluations.run(dataset, flow_factory, models, ...)` |
| 2 | No reference-based scoring; `AgentDataPoint` has no expected-answer field | `evaluations/point.py` | `scorers.py` (exact / numeric / choice + format-miss detection) | deterministic evaluators plus an `expected_output` field |
| 3 | `MessageInfo` drops reasoning tokens, cached tokens and `finish_reason`, though litellm returns all three | raw Radium response has `completion_tokens_details.reasoning_tokens`, `prompt_tokens_details.cached_tokens`; `MessageInfo` has neither | subclass overriding `extract_message_info` (`instrument.py`) | add the fields to `MessageInfo`; also surface `finish_reason` (now only logged) |
| 4 | Latency is `time.time()` wall time around the whole call; no time-to-first-token | `_litellm_wrapper._invoke` | `perf_counter` in middleware; **TTFT not measured yet** | monotonic clock; TTFT on streamed calls |
| 5 | Cost is `None` for models missing from litellm's price map (all of Radium) | `extract_message_info` -> `completion_cost` fails silently | pricing table in `configs/models.yaml` | `pricing=` on the provider, or document `litellm.register_model` |
| 6 | No client-side rate limiting | only retry-after-429 exists | token bucket per `rate_group`, applied in model middleware | `RateLimit(rpm=...)` prebuilt middleware |
| 7 | `RetryApproach._compute_delay(attempt)` never sees the exception, so it can't honour `Retry-After`; first wait is `base**0` = up to 1 s | Sonnet 4.6 exhausted 4 tries on 429s in the first live run | more tries in config; resume re-runs errored samples | pass the exception to `_compute_delay`; honour `Retry-After`; `min_delay` |
| 8 | Error messages hide the root cause: `LLMRateLimitError` -> `RetryError` -> litellm error, and the text shows none of it | `"Max retries exceeded"` with no status code | `runner.describe_error` walks the cause chain | include last attempt's error in the message |
| 9 | Model middleware is per-agent only; nothing applies it to every agent in a Flow | `agent_node(model_middleware=...)` only | `BuildContext.agent()` helper injects it | Flow- or session-level `model_middleware=` |
| 10 | Token semantics differ by provider and pass through unnormalized | xAI: `completion_tokens`=1 with `reasoning_tokens`=108 (excluded); OpenAI-style providers include reasoning in completion tokens | raw values recorded; **not normalized yet** | normalize, or document per provider |
| 11 | Aggregation is means only: no confidence intervals or tail latency | `LLMInferenceEvaluator` | `report.py` (Wilson CI, p50/p95, $/correct) | add CIs/percentiles to aggregates |
| 12 | Docs vs code | (a) `_evaluate.py` mentions an `EvaluationDataset` type that doesn't exist; (b) Numerical-metrics docs import `railtracks.evaluation as eval` (module is `evaluations`); (c) Judge custom-prompt example uses `eval.JudgeEvaluator`; (d) Structured Output docs say you can pass `output_schema` *and* `tool_nodes` to one `agent_node`, but `agent_node` raises `NodeCreationError` for that | read in source | n/a | doc fixes |
| 13 | Noise: every run warns that no `.railtracks` directory exists unless `railtracks init` was run | `observability/storage.py` UserWarning | ignored in pytest config | warn once, or only on write |

## Model behaviour noticed along the way (not Railtracks issues)

- Radium adds ~105 input tokens of hidden prompt per call (a 20-token prompt bills ~126-130).
- All three Radium models return reasoning, including Tycho, which Radium's site lists as non-reasoning.
- xAI's grok-4.7 bills ~1,150 cached input tokens of hidden system prompt per call.
- On smoke-qa, Clarke never used the required `Answer:` line (12/12 format misses).
- Opus 5.5's ping reply came back as `?391`; worth checking what the leading character is.
