"""A scripted provider for tests: no network, no cost, fully deterministic."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cutpilot.agent.llm.base import LLMRequest, LLMResponse

Responder = Callable[[LLMRequest], str | dict[str, Any] | list[Any]]


@dataclass
class FakeProvider:
    """Answers from a queue or a function, and records every request.

    Queue entries and responder results may be strings or JSON-serialisable
    objects; objects are dumped so tests can write plans as plain dicts.
    """

    queue: list[str | dict[str, Any] | list[Any]] = field(default_factory=list)
    responder: Responder | None = None
    model: str = "fake-model"
    requests: list[LLMRequest] = field(default_factory=list)

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.responder is not None:
            answer = self.responder(request)
        elif self.queue:
            answer = self.queue.pop(0)
        else:
            raise AssertionError("FakeProvider ran out of scripted answers")
        text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return LLMResponse(text=text, model=self.model)

    @property
    def last(self) -> LLMRequest:
        return self.requests[-1]
