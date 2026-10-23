"""The interface every LLM backend implements.

The planner and the grounder depend on this, never on a vendor SDK. Two things
follow from that. Tests run against FakeProvider with no network, no cost and
no rate limit. And the provider can change -- another Gemini model, a local VLM
as the week-11 grounding baseline, or Vertex if the project moves to GCP --
without the agent noticing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Base for every provider failure."""


class LLMUnavailable(LLMError):
    """Every model in the chain was exhausted on retryable errors.

    Distinct from a bad request: this one is worth telling the user to try
    again, and it is what the demo fallback exists for.
    """


class LLMBadRequest(LLMError):
    """The request itself was rejected. Retrying will not help."""


class LLMBadOutput(LLMError):
    """The model answered, but not in the shape that was asked for."""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    system: str | None = None
    images: tuple[bytes, ...] = ()
    """JPEG bytes. Order is preserved and is part of the cache key."""

    schema: dict[str, Any] | None = None
    """JSON Schema for structured output. None means free text."""

    thinking_level: str | None = None
    temperature: float = 0.0


@dataclass
class LLMResponse:
    text: str
    model: str
    """The model that actually answered -- may be a fallback."""

    cached: bool = False
    latency_ms: float = 0.0
    attempts: int = 1
    usage: dict[str, int | None] = field(default_factory=dict)


class LLMProvider(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse: ...


def generate_structured(
    provider: LLMProvider,
    model_type: type[T],
    prompt: str,
    *,
    system: str | None = None,
    images: tuple[bytes, ...] = (),
    thinking_level: str | None = None,
) -> tuple[T, LLMResponse]:
    """Ask for output matching `model_type` and validate it.

    Schema-valid is not the same as meaningful -- the first live probe returned
    a perfectly valid plan that put "the other people" in an include list -- so
    callers still normalise what comes back. This only guarantees the shape.
    """
    response = provider.generate(
        LLMRequest(
            prompt=prompt,
            system=system,
            images=images,
            schema=model_type.model_json_schema(),
            thinking_level=thinking_level,
        )
    )
    try:
        return model_type.model_validate_json(response.text), response
    except ValidationError as exc:
        raise LLMBadOutput(f"{model_type.__name__} did not validate: {exc}", response.text) from exc
