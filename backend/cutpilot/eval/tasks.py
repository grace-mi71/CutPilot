"""Evaluation tasks: an instruction, what the user was looking at, and the edit they meant.

Ground truth is written against a frozen tracking result (eval/<set>/tracks.json),
so track ids stay valid whatever YOLO version runs later. Targets are track ids,
or "all" for a whole-kind edit; times are seconds, None for the clip's start or end.

Levels follow the proposal:

    L1  direct      ids, whole kind, explicit times -- no grounding needed
    L2  described   one person picked out by appearance
    L3  ambiguous   several people, exclusions, or a description that fits more
                    than one person while the user meant one
    L4  composite   several steps, or a reference by time ("who was there from the start")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from cutpilot.schemas.plan import OpKind, TimeRange
from cutpilot.schemas.session import UIContext
from cutpilot.schemas.track import ObjectKind

Level = Literal["L1", "L2", "L3", "L4"]


class ExpectedOp(BaseModel):
    op: OpKind
    kind: ObjectKind = "person"
    targets: list[str] | Literal["all"]
    exclude: list[str] = Field(default_factory=list)
    start: float | None = None
    end: float | None = None
    aspect_ratio: str | None = None

    @property
    def time_range(self) -> TimeRange:
        return TimeRange(start=self.start, end=self.end)


class Task(BaseModel):
    id: str
    level: Level
    text: str
    playhead: float | None = None
    selection: TimeRange | None = None
    expected: list[ExpectedOp]
    note: str = ""

    @property
    def ui(self) -> UIContext:
        return UIContext(playhead=self.playhead, selection=self.selection)


class TaskSet(BaseModel):
    name: str
    source: str
    """Video file, relative to the data directory."""
    labeled_by: str
    tasks: list[Task]

    @classmethod
    def load(cls, path: Path) -> TaskSet:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
