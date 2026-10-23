"""LLM provider abstraction (Gemini today, others behind the same interface)."""

from __future__ import annotations

from cutpilot.agent.llm.base import (
    LLMBadOutput,
    LLMBadRequest,
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUnavailable,
    generate_structured,
)
from cutpilot.agent.llm.fake import FakeProvider
from cutpilot.core.cache import ResponseCache
from cutpilot.core.config import Settings, get_settings


def provider_from_settings(settings: Settings | None = None) -> LLMProvider:
    """The configured Gemini provider: primary model, fallbacks, cache."""
    from cutpilot.agent.llm.gemini import GeminiProvider

    s = settings or get_settings()
    models = [s.gemini_model, *[m for m in s.gemini_fallback_models if m != s.gemini_model]]
    return GeminiProvider(
        s.gemini_api_key,
        models,
        cache=ResponseCache(s.cache_dir, enabled=s.llm_cache_enabled),
        timeout_seconds=s.llm_timeout_seconds,
        max_attempts=s.llm_max_attempts,
        seed=s.llm_seed,
    )


__all__ = [
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMError",
    "LLMUnavailable",
    "LLMBadRequest",
    "LLMBadOutput",
    "FakeProvider",
    "generate_structured",
    "provider_from_settings",
]
