"""Shared test fixtures: an offline scripted model and a model spec for it."""

from __future__ import annotations

import itertools
from typing import Callable

import pytest
import railtracks as rt
from railtracks.llm.message import AssistantMessage
from railtracks.llm.response import MessageInfo, Response
from railtracks.llm.content import ToolCall

from pooria_bench.models import ModelSpec

_ids = itertools.count()


class ScriptedLLM(rt.llm.EchoLLM):
    """Offline model whose reply is computed by `script(messages, tools)`.

    `script` returns either a string (final answer) or a list of
    (tool_name, arguments) tuples (a tool-calling turn).
    """

    def __init__(self, script: Callable, tokens: tuple[int, int] = (10, 5)) -> None:
        super().__init__()
        self._script = script
        self._tokens = tokens

    def model_name(self) -> str:
        return "scripted"

    def _respond(self, messages, tools=None) -> Response:
        out = self._script(list(messages), tools)
        if isinstance(out, list):
            content = [ToolCall(identifier=f"call_{next(_ids)}", name=n, arguments=a) for n, a in out]
        else:
            content = out
        info = MessageInfo(input_tokens=self._tokens[0], output_tokens=self._tokens[1], latency=0.0,
                           model_name="scripted", total_cost=None)
        return Response(message=AssistantMessage(content), message_info=info)

    def _chat(self, messages):
        return self._respond(messages)

    def _chat_with_tools(self, messages, tools):
        return self._respond(messages, tools)

    async def _achat(self, messages):
        return self._respond(messages)

    async def _achat_with_tools(self, messages, tools):
        return self._respond(messages, tools)


@pytest.fixture
def scripted_spec() -> ModelSpec:
    return ModelSpec(name="scripted", provider="echo", model_id="scripted",
                     pricing={"input": 1.0, "output": 2.0, "cached_input": 0.1})
