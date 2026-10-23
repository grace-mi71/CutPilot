"""Retry, fallback and caching in the Gemini provider -- against a stub client,
so the policy is tested without the network it exists to survive."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from cutpilot.agent.llm import (
    FakeProvider,
    LLMBadOutput,
    LLMBadRequest,
    LLMRequest,
    LLMUnavailable,
    generate_structured,
)
from cutpilot.agent.llm.gemini import GeminiProvider
from cutpilot.core.cache import ResponseCache


class StubAPIError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


class StubClient:
    """Plays back a script of outcomes per model: an int raises that status."""

    def __init__(self, script: dict[str, list[int | str]]) -> None:
        self.script = {model: list(outcomes) for model, outcomes in script.items()}
        self.calls: list[str] = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, *, model, contents, config):
        self.calls.append(model)
        outcome = self.script[model].pop(0)
        if isinstance(outcome, int):
            raise StubAPIError(outcome)
        return SimpleNamespace(
            text=outcome,
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=5, thoughts_token_count=0
            ),
        )


def _provider(script, *, models=("primary", "fallback"), attempts=3, cache=None):
    client = StubClient(script)
    provider = GeminiProvider(
        "unused",
        list(models),
        client=client,
        max_attempts=attempts,
        cache=cache,
        sleep=lambda _seconds: None,
    )
    return provider, client


REQUEST = LLMRequest(prompt="hello")


class TestRetryPolicy:
    def test_first_success_is_returned(self):
        provider, client = _provider({"primary": ["ok"]})
        response = provider.generate(REQUEST)
        assert (response.text, response.model, response.attempts) == ("ok", "primary", 1)

    def test_overload_is_retried_on_the_same_model(self):
        provider, client = _provider({"primary": [503, 503, "ok"]})
        response = provider.generate(REQUEST)
        assert response.model == "primary"
        assert client.calls == ["primary"] * 3

    def test_exhausted_primary_falls_through_to_the_fallback(self):
        """The failure observed live: the primary stays at 503."""
        provider, client = _provider({"primary": [503, 503, 503], "fallback": ["ok"]})
        response = provider.generate(REQUEST)
        assert response.model == "fallback"
        assert response.attempts == 4

    def test_rate_limit_counts_as_retryable(self):
        provider, _ = _provider({"primary": [429, "ok"]})
        assert provider.generate(REQUEST).text == "ok"

    def test_bad_request_is_never_retried(self):
        """A 400 -- e.g. a thinking level the model rejects -- is a bug in the
        request. Retrying it wastes the quota and hides the cause."""
        provider, client = _provider({"primary": [400, "ok"], "fallback": ["ok"]})
        with pytest.raises(LLMBadRequest):
            provider.generate(REQUEST)
        assert client.calls == ["primary"]

    def test_everything_exhausted_raises_unavailable(self):
        provider, _ = _provider({"primary": [503] * 3, "fallback": [503] * 3})
        with pytest.raises(LLMUnavailable, match="6 attempts"):
            provider.generate(REQUEST)

    def test_backoff_grows_between_attempts(self):
        waits: list[float] = []
        client = StubClient({"primary": [503, 503, "ok"]})
        GeminiProvider(
            "unused", ["primary"], client=client, sleep=waits.append, backoff_base=2.0
        ).generate(REQUEST)
        assert waits == [1.0, 2.0]


class TestCache:
    def test_second_identical_request_never_reaches_the_model(self, tmp_path):
        cache = ResponseCache(tmp_path)
        provider, client = _provider({"primary": ["first", "second"]}, cache=cache)
        provider.generate(REQUEST)
        replay = provider.generate(REQUEST)
        assert replay.cached
        assert replay.text == "first"
        assert client.calls == ["primary"]

    def test_fallback_answers_replay_under_the_same_key(self, tmp_path):
        cache = ResponseCache(tmp_path)
        provider, _ = _provider({"primary": [503] * 3, "fallback": ["fb"]}, cache=cache)
        provider.generate(REQUEST)
        replay = provider.generate(REQUEST)
        assert (replay.cached, replay.model, replay.text) == (True, "fallback", "fb")

    def test_thinking_level_is_part_of_the_key(self, tmp_path):
        cache = ResponseCache(tmp_path)
        provider, client = _provider({"primary": ["a", "b"]}, cache=cache)
        provider.generate(LLMRequest(prompt="x", thinking_level="low"))
        provider.generate(LLMRequest(prompt="x", thinking_level="high"))
        assert len(client.calls) == 2


class Answer(BaseModel):
    value: int


class TestStructured:
    def test_valid_json_is_parsed(self):
        result, _ = generate_structured(FakeProvider(queue=[{"value": 3}]), Answer, "q")
        assert result.value == 3

    def test_schema_is_sent_with_the_request(self):
        fake = FakeProvider(queue=[{"value": 3}])
        generate_structured(fake, Answer, "q")
        assert fake.last.schema == Answer.model_json_schema()

    def test_invalid_shape_raises_with_the_raw_text(self):
        with pytest.raises(LLMBadOutput) as info:
            generate_structured(FakeProvider(queue=['{"value": "nope"}']), Answer, "q")
        assert "nope" in info.value.raw
