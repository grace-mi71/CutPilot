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
    LLMQuotaExhausted,
    LLMRequest,
    LLMUnavailable,
    generate_structured,
)
from cutpilot.agent.llm.gemini import GeminiProvider, retry_delay_of
from cutpilot.core.cache import ResponseCache


class StubAPIError(Exception):
    def __init__(self, code: int, retry: str | None = None) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code
        self.details = (
            {"error": {"details": [{"@type": "RetryInfo", "retryDelay": retry}]}} if retry else None
        )


DAILY_QUOTA = StubAPIError(429, retry="73502s")
"""What the free tier actually returned on 2026-10-23: 20 requests per model
per day, reset in about 20 hours."""
BURST = StubAPIError(429, retry="4s")


class StubClient:
    """Plays back a script of outcomes per model: an int raises that status."""

    def __init__(self, script: dict[str, list[int | str]]) -> None:
        self.script = {model: list(outcomes) for model, outcomes in script.items()}
        self.calls: list[str] = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, *, model, contents, config):
        self.calls.append(model)
        outcome = self.script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
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


class TestQuota:
    def test_daily_quota_skips_straight_to_the_fallback(self):
        """Retrying a model whose quota resets in 20 hours wastes attempts."""
        provider, client = _provider({"primary": [DAILY_QUOTA, "never"], "fallback": ["ok"]})
        response = provider.generate(REQUEST)
        assert response.model == "fallback"
        assert client.calls == ["primary", "fallback"]

    def test_every_model_out_of_quota_says_when_it_resets(self):
        provider, _ = _provider({"primary": [DAILY_QUOTA], "fallback": [DAILY_QUOTA]})
        with pytest.raises(LLMQuotaExhausted) as info:
            provider.generate(REQUEST)
        assert info.value.resets_in_hours == pytest.approx(20.4, abs=0.1)

    def test_a_short_burst_limit_waits_as_told_and_retries(self):
        waits: list[float] = []
        client = StubClient({"primary": [BURST, "ok"]})
        provider = GeminiProvider("unused", ["primary"], client=client, sleep=waits.append)
        assert provider.generate(REQUEST).text == "ok"
        assert waits == [4.0]

    def test_quota_and_overload_mixed_is_still_unavailable(self):
        provider, _ = _provider({"primary": [DAILY_QUOTA], "fallback": [503, 503, 503]})
        with pytest.raises(LLMUnavailable) as info:
            provider.generate(REQUEST)
        assert not isinstance(info.value, LLMQuotaExhausted)


class TestRetryDelay:
    def test_reads_retry_info(self):
        assert retry_delay_of(DAILY_QUOTA) == 73502.0

    def test_falls_back_to_the_message(self):
        assert retry_delay_of(Exception("Please retry in 20h25m2.85s.")) == pytest.approx(
            20 * 3600 + 25 * 60 + 2.85
        )

    def test_none_when_absent(self):
        assert retry_delay_of(Exception("boom")) is None


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
