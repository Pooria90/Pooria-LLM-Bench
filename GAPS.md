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
| 14 | Every model call runs synchronously in a worker thread (`asyncio.to_thread(model.chat)`); litellm's async API is never used. The default executor's size (min(32, CPUs+4)) silently caps concurrency, and the wait for a free thread lands inside model-middleware timing | `built_nodes/llm/model_invoker.py`; a test model's `_achat` was never called | runner sizes the default executor to the planned concurrency | use `litellm.acompletion` on the async path, or document the executor limit |
| 15 | When a model spends its whole output budget on reasoning, Railtracks raises `LLMError("The model returned no text content…")` instead of returning the response with `finish_reason=length`; callers can't tell truncation from a real failure | Hal and Opus 4.6 on AIME I-15 at a 32k budget | runner checks the last call's recorded `finish_reason` and scores the sample as wrong, not as an error | a `LLMTruncatedError` carrying usage and `finish_reason`, or return the empty response |
| 16 | Refusals pass as normal replies. Anthropic `stop_reason: refusal` becomes litellm `content_filter`, Railtracks keeps the text (here: only invisible zero-width characters) and drops `stop_details` (category, explanation) | captured Opus 5.5 refusal `req_011ChNLhwoWxJFPYRpZsNMg7`, category `reasoning_extraction`, on "What is 17*23?" | `finish_reason` recorded via #3; report counts refusals; scorers strip invisible characters | flag refusals on the response; keep `stop_details` |
| 17 | `reasoning_effort` is typed `Literal["none","minimal","low","medium","high"]`, but providers now accept more (Anthropic `xhigh`/`max`, OpenAI `xhigh`); values pass through at runtime, but type checkers reject them | `_litellm_wrapper.LiteLLMWrapper.__init__` signature | none needed yet | widen the type, or accept `str` |

Worked well this round: `reasoning_effort="high"` on `AnthropicLLM` reaches Claude as adaptive thinking with
`effort: high` and summarized thinking (litellm's mapping), so the same knob drives OpenAI and Anthropic.

## Model behaviour noticed along the way (not Railtracks issues)

- Radium adds ~105 input tokens of hidden prompt per call (a 20-token prompt bills ~126-130).
- All three Radium models return reasoning, including Tycho, which Radium's site lists as non-reasoning.
- xAI's grok-4.7 bills ~1,150 cached input tokens of hidden system prompt per call.
- On smoke-qa, Clarke never used the required `Answer:` line (12/12 format misses).
- Opus 5.5's `?391`: Anthropic's API sometimes prefixes an otherwise normal reply with U+E007F
  (CANCEL TAG), which terminals draw as `?`. Seen 2 times in ~100 calls; the text litellm returned
  matched the raw API bytes in every captured call (52/52). Opus 5.5 also refused this trivial prompt
  once in 40 calls as `reasoning_extraction`.
- Opus 4.7 on AIME 2025: answered I-01 and I-15 correctly in ~1 s with zero thinking and no `\boxed{}`,
  despite adaptive thinking at effort `high` and a "reason step by step" prompt. That is a strong sign it
  recalls AIME 2025 answers. Opus 4.6 reasons through the same problems.
- Hal reasons 4-7x longer than GPT-5.5 and Opus on the same problem, and ignores `reasoning_effort`.
- This Anthropic account rejects ~5 concurrent connections with 429 "Number of concurrent
  connections has exceeded your rate limit" (now capped with `max_concurrent: 3`).
