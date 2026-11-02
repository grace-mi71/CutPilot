"""A hard stop on what the model may cost, enforced in code.

Billing alerts on Google's side arrive after the money is spent, and a test
loop that re-runs grounding on every frame can spend a lot quickly. So every
paid call goes through this guard: the cost of each answer is estimated from
the token counts it reports, written to a ledger on disk (so it survives
restarts), and once the total reaches the budget no further call is made.

The estimate deliberately errs high. Prices are settings rather than constants
because they change -- check them against the provider's pricing page -- and a
missing token count is charged as a generous guess instead of as zero.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from cutpilot.agent.llm.base import LLMError, LLMProvider, LLMRequest, LLMResponse

GUESS_TOKENS_PER_IMAGE = 1500
GUESS_OUTPUT_TOKENS = 2000


class LLMBudgetExceeded(LLMError):
    def __init__(self, spent: float, budget: float) -> None:
        super().__init__(f"LLM budget exhausted: ${spent:.4f} of ${budget:.2f}")
        self.spent = spent
        self.budget = budget


class BudgetGuard:
    def __init__(
        self,
        inner: LLMProvider,
        *,
        ledger: Path,
        budget_usd: float,
        input_per_million: float,
        output_per_million: float,
    ) -> None:
        self.inner = inner
        self.ledger = ledger
        self.budget_usd = budget_usd
        self.input_per_million = input_per_million
        self.output_per_million = output_per_million
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        models = getattr(self.inner, "models", None)
        return models[0] if models else getattr(self.inner, "model", type(self.inner).__name__)

    # ------------------------------------------------------------- ledger

    def _read(self) -> dict:
        if not self.ledger.exists():
            return {"usd": 0.0, "calls": 0, "prompt_tokens": 0, "output_tokens": 0}
        return json.loads(self.ledger.read_text(encoding="utf-8"))

    @property
    def spent(self) -> float:
        with self._lock:
            return float(self._read()["usd"])

    def cost_of(self, request: LLMRequest, response: LLMResponse) -> tuple[float, int, int]:
        usage = response.usage or {}
        prompt = usage.get("prompt")
        if prompt is None:
            prompt = len(request.prompt) + len(request.system or "")
            prompt += GUESS_TOKENS_PER_IMAGE * len(request.images)
        output = usage.get("output")
        thinking = usage.get("thinking") or 0
        # Thinking tokens are billed as output.
        out = GUESS_OUTPUT_TOKENS if output is None else output + thinking
        usd = prompt * self.input_per_million / 1e6 + out * self.output_per_million / 1e6
        return usd, prompt, out

    # ------------------------------------------------------------ calling

    def generate(self, request: LLMRequest) -> LLMResponse:
        with self._lock:
            spent = float(self._read()["usd"])
        if spent >= self.budget_usd:
            raise LLMBudgetExceeded(spent, self.budget_usd)

        response = self.inner.generate(request)
        if response.cached:
            return response

        usd, prompt, out = self.cost_of(request, response)
        with self._lock:
            state = self._read()
            state["usd"] = float(state["usd"]) + usd
            state["calls"] = int(state["calls"]) + 1
            state["prompt_tokens"] = int(state["prompt_tokens"]) + prompt
            state["output_tokens"] = int(state["output_tokens"]) + out
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.ledger.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
            tmp.replace(self.ledger)
        return response

    def summary(self) -> dict:
        with self._lock:
            state = self._read()
        return {**state, "budget_usd": self.budget_usd}
