"""Gemini through google-genai, with the failure handling a live demo needs.

Retry and model fallback are one policy implemented here rather than split
between the SDK's built-in retry and an outer loop. The first probe against the
free tier returned 503 on three of five calls and took 235 seconds on one of
the two that succeeded, so the policy is not hypothetical:

    per model    up to `max_attempts` on 429 / 5xx / timeout, exponential backoff
    across models  fall through the chain when a model stays unavailable
    never        retry a 400 -- the request is wrong, not the service

Automatic function calling is switched off explicitly. The agent never lets the
model call tools; it emits a plan the user approves. The SDK also warns when AFC
is left on with generate_content.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from cutpilot.agent.llm.base import (
    LLMBadRequest,
    LLMRequest,
    LLMResponse,
    LLMUnavailable,
)
from cutpilot.core.cache import ResponseCache

RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})


class GeminiProvider:
    def __init__(
        self,
        api_key: str,
        models: Sequence[str],
        *,
        cache: ResponseCache | None = None,
        timeout_seconds: float = 45.0,
        max_attempts: int = 3,
        seed: int | None = 7,
        backoff_base: float = 1.5,
        sleep: Callable[[float], None] = time.sleep,
        client: object | None = None,
    ) -> None:
        if not models:
            raise ValueError("at least one model is required")
        self.models = list(models)
        self.cache = cache
        self.max_attempts = max(1, max_attempts)
        self.seed = seed
        self.backoff_base = backoff_base
        self._sleep = sleep

        if client is not None:
            self._client = client
        else:
            from google import genai
            from google.genai import types

            if not api_key:
                raise LLMBadRequest("CUTPILOT_GEMINI_API_KEY is not set")
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=int(timeout_seconds * 1000),
                    # One attempt at the transport layer; the policy lives here.
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )

    # ------------------------------------------------------------------ cache

    def _cache_key(self, request: LLMRequest) -> str:
        return ResponseCache.make_key(
            # Keyed on the primary: a fallback's answer is still the answer to
            # this request, and replaying it is what reproducibility needs.
            model=self.models[0],
            prompt=request.prompt,
            images=list(request.images),
            system=request.system,
            schema=request.schema,
            thinking_level=request.thinking_level,
            temperature=request.temperature,
            seed=self.seed,
        )

    # ---------------------------------------------------------------- request

    def _config(self, request: LLMRequest) -> object:
        from google.genai import types

        config: dict[str, object] = {
            "temperature": request.temperature,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if self.seed is not None:
            config["seed"] = self.seed
        if request.system:
            config["system_instruction"] = request.system
        if request.schema is not None:
            config["response_mime_type"] = "application/json"
            # response_json_schema accepts $defs/$ref; the older response_schema
            # takes an OpenAPI subset that rejected nested models in the past.
            config["response_json_schema"] = request.schema
        if request.thinking_level:
            config["thinking_config"] = types.ThinkingConfig(thinking_level=request.thinking_level)
        return types.GenerateContentConfig(**config)

    def _contents(self, request: LLMRequest) -> list[object]:
        from google.genai import types

        parts: list[object] = [
            types.Part.from_bytes(data=image, mime_type="image/jpeg") for image in request.images
        ]
        parts.append(request.prompt)
        return parts

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        code = getattr(exc, "code", None)
        return code if isinstance(code, int) else None

    @classmethod
    def _is_retryable(cls, exc: Exception) -> bool:
        code = cls._status_code(exc)
        if code is not None:
            return code in RETRYABLE_CODES
        # Transport-level timeouts arrive without a status code.
        name = type(exc).__name__.lower()
        return "timeout" in name or "connect" in name

    def generate(self, request: LLMRequest) -> LLMResponse:
        key = self._cache_key(request)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return LLMResponse(
                text=hit["text"], model=hit["model"], cached=True, usage=hit.get("usage", {})
            )

        config = self._config(request)
        contents = self._contents(request)
        attempts = 0
        failures: list[str] = []
        started = time.perf_counter()

        for model in self.models:
            for attempt in range(self.max_attempts):
                attempts += 1
                try:
                    raw = self._client.models.generate_content(
                        model=model, contents=contents, config=config
                    )
                except Exception as exc:  # noqa: BLE001 - classified below
                    code = self._status_code(exc)
                    if not self._is_retryable(exc):
                        raise LLMBadRequest(f"{model}: {exc}") from exc
                    failures.append(f"{model}#{attempt + 1}: {code or type(exc).__name__}")
                    if attempt + 1 < self.max_attempts:
                        self._sleep(self.backoff_base**attempt)
                    continue

                usage_meta = getattr(raw, "usage_metadata", None)
                usage = {
                    "prompt": getattr(usage_meta, "prompt_token_count", None),
                    "output": getattr(usage_meta, "candidates_token_count", None),
                    "thinking": getattr(usage_meta, "thoughts_token_count", None),
                }
                text = raw.text or ""
                response = LLMResponse(
                    text=text,
                    model=model,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    attempts=attempts,
                    usage=usage,
                )
                if self.cache is not None:
                    self.cache.set(
                        key,
                        {"text": text, "model": model, "usage": usage},
                        meta={"attempts": attempts, "failures": failures},
                    )
                return response

        raise LLMUnavailable(
            f"all models exhausted after {attempts} attempts: " + ", ".join(failures)
        )
