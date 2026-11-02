"""The spending cap: estimated from token counts, kept on disk, enforced before a call."""

from __future__ import annotations

import pytest

from cutpilot.agent.llm import BudgetGuard, LLMBudgetExceeded, LLMRequest, LLMResponse


class Priced:
    """Answers with fixed token counts, optionally from cache."""

    model = "priced"

    def __init__(self, prompt=1_000_000, output=100_000, thinking=100_000, cached=False):
        self.usage = {"prompt": prompt, "output": output, "thinking": thinking}
        self.cached = cached
        self.calls = 0

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(text="{}", model=self.model, cached=self.cached, usage=self.usage)


def guard(inner, tmp_path, budget=10.0) -> BudgetGuard:
    return BudgetGuard(
        inner, ledger=tmp_path / "spend.json", budget_usd=budget,
        input_per_million=1.5, output_per_million=7.5,
    )  # fmt: skip


def test_cost_counts_thinking_as_output(tmp_path):
    g = guard(Priced(), tmp_path)
    g.generate(LLMRequest(prompt="x"))
    # 1M in at 1.5 + 0.2M out (100k output + 100k thinking) at 7.5
    assert g.spent == pytest.approx(1.5 + 1.5)
    assert g.summary()["calls"] == 1


def test_stops_once_the_budget_is_spent(tmp_path):
    inner = Priced()
    g = guard(inner, tmp_path, budget=5.0)
    g.generate(LLMRequest(prompt="x"))  # 3.0
    g.generate(LLMRequest(prompt="x"))  # 6.0 -- allowed, the cap is checked before a call
    with pytest.raises(LLMBudgetExceeded):
        g.generate(LLMRequest(prompt="x"))
    assert inner.calls == 2


def test_survives_a_restart(tmp_path):
    guard(Priced(), tmp_path).generate(LLMRequest(prompt="x"))
    assert guard(Priced(), tmp_path).spent == pytest.approx(3.0)


def test_cache_hits_are_free(tmp_path):
    g = guard(Priced(cached=True), tmp_path)
    g.generate(LLMRequest(prompt="x"))
    assert g.spent == 0.0


def test_missing_counts_are_guessed_high(tmp_path):
    g = guard(Priced(prompt=None, output=None, thinking=None), tmp_path)
    g.generate(LLMRequest(prompt="x" * 100, images=(b"jpg",)))
    assert g.spent > 0.01  # 2000 guessed output tokens alone
