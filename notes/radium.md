# Radium LLMs — research notes (gathered 2026-11-24)

Sources: radium.cloud, docs.radium.cloud (quickstart, llms.txt, openai-python), radium.cloud/pricing,
radium.cloud/resources/benchmarks, Railtown press release (2026-11-19). All numbers are vendor-stated unless noted.

## Company
- "Frontier model alternative with drop-in LLM APIs to replace OpenAI and Anthropic." CEO/cofounder: Adam Hendin.
- HQ: San Francisco + second HQ in Canada (press release); a directory says Toronto (unverified).
- Claims 0% data retention; SOC 2 / ISO 27001 / HIPAA / GDPR all "in progress". Data residency: US, Canada (ambiguous table).

## Models
| Model | API ID | Purpose | Context | Max out | $/1M in / out (cached in) | Input | Reasoning | Vendor-claimed comparable |
|---|---|---|---|---|---|---|---|---|
| Hal 1.0 | `hal-1.0` | reasoning, coding, agents, tools | 128K | 32,768 | 2.00 / 10.00 (0.20) | text, images | yes | Claude Opus 4.7 / GPT-5.5 |
| Clarke 1.0 | `clarke-1.0` | RAG, chat, retrieval | 200K | 16,384 | 0.75 / 4.50 (0.075) | text, images | yes | Claude Sonnet 4.6 / GPT-5.4 |
| Tycho 1.0 | `tycho-1.0` | classification, extraction, high volume | 128K | 8,192 | 0.10 / 1.50 (0.02) | text | no | Claude Haiku 4.5 / GPT-5.4 mini |

Hal 2.0 is an invite-only preview (no ID/pricing published).

## API
- OpenAI-compatible: `base_url="https://api.radium.cloud/v1"`, `POST /v1/chat/completions`, header `Authorization: Bearer $RADIUM_API_KEY`.
- Anthropic-compatible: `base_url="https://api.radium.cloud"`, `POST /v1/messages`, header `x-api-key`, supports `cache_control`.
- Also `GET /v1/models`, `POST /v1/messages/count_tokens`.
- Supported: streaming, parallel tool calls, JSON mode + schema. Not supported: Responses API, embeddings, batch, file upload.
- Gotchas: with reasoning on, a thinking block precedes the text (read the last content block); keep `max_tokens >= 512`
  so reasoning doesn't eat the answer. No documented parameter to toggle reasoning.
- Rate limits (per org): 1,000 RPM, burst 33, no TPM limit; 3x acceleration cap over 10s windows.
  Retry 429/500/503 with backoff starting at 9s, honor Retry-After.
- Signup: https://deploy.radium.cloud/sign-up — $25 free credit, no waitlist for the 1.0 models.

## Vendor benchmark (radium.cloud/resources/benchmarks)
80 structured tasks with exact-JSON-match scoring; perf at 4 concurrent requests. Radium ran with thinking OFF, Claude with adaptive thinking.

| Pair | Accuracy | Output tok/s | p50 end-to-end |
|---|---|---|---|
| Hal vs Opus 4.7 | 96.25% vs 93.75% | 499.5 vs 107.7 | 2.31s vs 1.92s |
| Clarke vs Sonnet | 85% vs 100% | 85.6 vs 100.0 | 0.69s vs 2.82s |
| Tycho vs Haiku | 91.25% vs 100% | 376.1 vs 408.3 | 1.49s vs 2.05s |

The weakest category is "policy application" (Clarke 50%, Tycho 65%, Hal 85%). The vendor says the benchmark excludes hard
reasoning, open-ended quality, long context, tool use, multimodal, and safety. Its pricing chart names
different competitor models (Opus 5.5, GPT-6) than the rest of the site.

## Gaps worth testing ourselves
- Independent reproduction of the vendor numbers, and standard benchmarks (MMLU-Pro, GPQA, HumanEval/LiveCodeBench, etc.).
- Tool use / agentic tasks (Hal's main pitch), long context (Clarke's 200K), and reasoning on vs off.
- Real throughput/latency from our region, under concurrency, and cost per correct answer.
