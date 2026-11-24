"""Model registry: `configs/models.yaml` -> instrumented Railtracks LLMs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import railtracks as rt
import yaml
from pydantic import BaseModel, Field

from .instrument import instrumented

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"

Provider = Literal["openai", "anthropic", "gemini", "openai_compatible", "echo"]


class Pricing(BaseModel):
    """USD per 1M tokens."""

    input: float
    output: float
    cached_input: float | None = None

    def cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        cached = min(cached_tokens or 0, input_tokens)
        cached_rate = self.cached_input if self.cached_input is not None else self.input
        return (
            (input_tokens - cached) * self.input
            + cached * cached_rate
            + output_tokens * self.output
        ) / 1_000_000


class RetrySpec(BaseModel):
    max_tries: int = 4
    base: float = 3.0


class ModelSpec(BaseModel):
    name: str
    provider: Provider
    model_id: str
    api_base: str | None = None
    api_key_env: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    pricing: Pricing | None = None
    rate_group: str | None = None
    rpm: float | None = None
    retry: RetrySpec | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def group(self) -> str:
        return self.rate_group or self.name

    def key_available(self) -> bool:
        env = self.api_key_env or _DEFAULT_KEY_ENV.get(self.provider)
        return env is None or bool(os.environ.get(env))


_DEFAULT_KEY_ENV: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai_compatible": None,
    "echo": None,
}


def load_models(path: Path | str | None = None) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    defaults = raw.get("defaults", {}) or {}
    specs: dict[str, ModelSpec] = {}
    for name, entry in (raw.get("models") or {}).items():
        entry = dict(entry)
        if entry.get("provider") != "echo":
            entry["params"] = {**defaults.get("params", {}), **(entry.get("params") or {})}
            entry.setdefault("retry", defaults.get("retry"))
        specs[name] = ModelSpec(name=name, **entry)
    return specs


def resolve_models(names: list[str], registry: dict[str, ModelSpec]) -> list[ModelSpec]:
    """Resolve names; `tag:<t>` selects every model carrying tag `t`."""
    out: list[ModelSpec] = []
    for name in names:
        if name.startswith("tag:"):
            tag = name[4:]
            matched = [s for s in registry.values() if tag in s.tags]
            if not matched:
                raise KeyError(f"No models tagged {tag!r}")
            out.extend(matched)
        elif name in registry:
            out.append(registry[name])
        else:
            raise KeyError(f"Unknown model {name!r}. Known: {', '.join(registry)}")
    seen: set[str] = set()
    return [s for s in out if not (s.name in seen or seen.add(s.name))]


def build_llm(spec: ModelSpec, **overrides: Any):
    """Build the Railtracks LLM for a spec, instrumented to keep usage extras."""
    params = {**spec.params, **overrides}
    if spec.retry is not None:
        params.setdefault(
            "retry_approach",
            rt.llm.retries.ExponentialRetry(max_tries=spec.retry.max_tries, base=spec.retry.base),
        )

    if spec.provider == "echo":
        return rt.llm.EchoLLM()
    if spec.provider == "openai":
        return instrumented(rt.llm.OpenAILLM)(spec.model_id, **params)
    if spec.provider == "anthropic":
        return instrumented(rt.llm.AnthropicLLM)(spec.model_id, **params)
    if spec.provider == "gemini":
        return instrumented(rt.llm.GeminiLLM)(model_name=spec.model_id, **params)
    if spec.provider == "openai_compatible":
        if not spec.api_base or not spec.api_key_env:
            raise ValueError(f"{spec.name}: openai_compatible needs api_base and api_key_env")
        key = os.environ.get(spec.api_key_env)
        if not key:
            raise ValueError(f"{spec.name}: env var {spec.api_key_env} is not set")
        return instrumented(rt.llm.OpenAICompatibleProvider)(
            spec.model_id, api_base=spec.api_base, api_key=key, **params
        )
    raise ValueError(f"Unknown provider {spec.provider!r}")
