"""A keyword planner that stands in for the model when there is none.

For building and demoing the interface without an API key or quota. It is not
a baseline and is never used in evaluation: it reads a handful of keywords and
gets anything subtle wrong. What it does do is drive every path the frontend
has to handle -- a plan, a correction, a clarification, and an open choice:

    planner     op / kind / times / ids from keywords; a sentence without an
                op word edits the current plan instead of replacing it
    grounding   always finds nobody, so every described target becomes a
                "which one?" over everyone on screen -- the click path
"""

from __future__ import annotations

import json
import re

from cutpilot.agent.llm.base import LLMRequest, LLMResponse
from cutpilot.agent.prompts import PLANNER_SYSTEM

OP_WORDS = (
    ("REFRAME", ("세로", "쇼츠", "숏츠", "9:16", "16:9", "리프레임", "비율")),
    ("BLUR", ("블러", "흐리")),
    ("MOSAIC", ("모자이크", "가려", "가리", "안 보이게")),
)
KIND_WORDS = (("face", ("얼굴",)), ("license_plate", ("번호판",)))
TRACK_ID = re.compile(r"@?((?:person|face|license_plate)_\d+)")
NUMBER = r"(\d+(?:\.\d+)?)"
SPARE_WORDS = ("빼고", "제외", "말고")

CLARIFY = (
    "오프라인 모드라 간단한 표현만 알아듣습니다. "
    "예: '가방 멘 사람 모자이크', '5초부터 10초까지 얼굴 블러', '세로로 바꿔줘'."
)


def _section(prompt: str, title: str) -> str:
    match = re.search(rf"## {title}[^\n]*\n(.*?)(?=\n## |\Z)", prompt, re.S)
    return match.group(1).strip() if match else ""


def _first(text: str, table) -> str | None:
    return next((value for value, words in table if any(w in text for w in words)), None)


def _times(text: str, ui: str) -> dict | None:
    if "선택" in text and (m := re.search(rf"선택 구간: {NUMBER}s ~ {NUMBER}s", ui)):
        return {"start": float(m.group(1)), "end": float(m.group(2))}
    if m := re.search(rf"{NUMBER}\s*초?\s*(?:~|-|부터|에서)\s*{NUMBER}\s*초", text):
        return {"start": float(m.group(1)), "end": float(m.group(2))}
    found: dict = {}
    if m := re.search(rf"{NUMBER}\s*초\s*(?:부터|이후)", text):
        found["start"] = float(m.group(1))
    if m := re.search(rf"{NUMBER}\s*초\s*까지", text):
        found["end"] = float(m.group(1))
    if found:
        return {"start": found.get("start"), "end": found.get("end")}
    return None


def _described(text: str) -> str | None:
    """'가방 멘 사람 모자이크' -> '가방 멘 사람'. Time words are dropped."""
    words = [w for w in text.split() if not re.search(r"\d", w)]
    for i, word in enumerate(words):
        if word.startswith("사람"):
            phrase = " ".join(words[max(0, i - 3) : i] + ["사람"])
            return phrase
    return None


def plan_from_keywords(prompt: str) -> dict:
    text = _section(prompt, "사용자 요청")
    ui = _section(prompt, "UI 상태")
    current_raw = _section(prompt, "현재 계획")
    current = json.loads(current_raw) if current_raw else []

    op = _first(text, OP_WORDS)
    kind = _first(text, KIND_WORDS)
    times = _times(text, ui)
    ids = TRACK_ID.findall(text)
    spare = any(w in text for w in SPARE_WORDS)
    phrase = None if ids else _described(text)
    names_target = bool(ids or phrase)

    if current and not names_target:
        # A correction: keep the plan, change only what was said.
        operations = []
        for old in current:
            new_op = op if op and (op == "REFRAME") == (old["op"] == "REFRAME") else old["op"]
            operations.append(
                {
                    "op": new_op,
                    "selector": {
                        "kind": kind or old["selector"]["kind"],
                        "include_phrases": old["selector"]["include"],
                        "exclude_phrases": old["selector"]["exclude"],
                    },
                    "time_range": times or old["time_range"],
                    "aspect_ratio": old.get("aspect_ratio"),
                    "strength": old.get("strength"),
                }
            )
        return {"operations": operations}

    if op is None:
        return {"operations": [], "clarification": CLARIFY}

    targets = ids or ([phrase] if phrase else [])
    include, exclude = ([], targets) if spare and op != "REFRAME" else (targets, [])
    return {
        "operations": [
            {
                "op": op,
                "selector": {
                    "kind": "person" if op == "REFRAME" else kind or "person",
                    "include_phrases": include,
                    "exclude_phrases": exclude,
                },
                "time_range": times or {},
                "aspect_ratio": ("16:9" if "16:9" in text else "9:16") if op == "REFRAME" else None,
            }
        ]
    }


class OfflineProvider:
    model = "offline-keywords"

    def generate(self, request: LLMRequest) -> LLMResponse:
        if request.system == PLANNER_SYSTEM:
            answer = plan_from_keywords(request.prompt)
        else:
            answer = {"objects": []}
        return LLMResponse(text=json.dumps(answer, ensure_ascii=False), model=self.model)
