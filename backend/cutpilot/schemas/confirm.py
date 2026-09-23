"""What the user sees before anything is rendered.

The confirmation step is the project's contribution in data form. Two rules
shape it:

1. Ask only where asking buys something. An unambiguous instruction gets a
   summary and an approve button; a vague reference gets a highlight and a
   question; a tie between candidates gets a choice. `ConfirmItem.status`
   carries that decision per field.

2. Surface what only shows up after rendering. A tracking gap, a target that
   the reframe crops away, a selector that matches nothing -- all of these are
   cheap to detect from boxes and expensive to discover in the output. They
   become warnings here instead.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cutpilot.schemas.plan import EditPlan, TimeRange
from cutpilot.schemas.track import Box

ConfirmField = Literal["selector", "region", "time_range", "op"]

ConfirmStatus = Literal["ok", "confirm", "choose"]
"""ok      no ambiguity -- show it, do not ask
   confirm the agent picked one, show it and ask
   choose  several candidates fit; do not guess"""

WarningKind = Literal["track_gap", "out_of_frame", "empty_selection"]

OverlayRole = Literal["target", "candidate", "excluded", "reframe_subject"]
"""Drives colour on the preview: what gets edited, what the user is being asked
to choose between, what was deliberately spared, and who the crop follows."""


class Overlay(BaseModel):
    """One box drawn on the keyframe."""

    track_id: str
    box: Box
    role: OverlayRole
    label: str | None = None
    operation_index: int | None = None


class ConfirmItem(BaseModel):
    """One field of one operation, and whether the user needs to weigh in."""

    operation_index: int
    field: ConfirmField
    label: str
    status: ConfirmStatus = "ok"
    candidates: list[str] = Field(default_factory=list)
    """Track ids to choose between. Only meaningful when status is "choose"."""

    @property
    def needs_input(self) -> bool:
        return self.status != "ok"


class PlanWarning(BaseModel):
    """Something the user should know before paying for a render."""

    kind: WarningKind
    message: str
    operation_index: int | None = None
    track_id: str | None = None
    time_range: TimeRange | None = None

    @property
    def is_privacy_risk(self) -> bool:
        """A gap in a blur target means an uncovered face in the output."""
        return self.kind == "track_gap"


class Confirmation(BaseModel):
    """The full confirmation view for one turn."""

    plan: EditPlan
    keyframe_t: float
    overlays: list[Overlay] = Field(default_factory=list)
    items: list[ConfirmItem] = Field(default_factory=list)
    warnings: list[PlanWarning] = Field(default_factory=list)

    @property
    def needs_input(self) -> bool:
        """False means: show the summary and let them approve straight away."""
        return any(item.needs_input for item in self.items)

    @property
    def blocking_items(self) -> list[ConfirmItem]:
        """Choices the agent refuses to make on the user's behalf."""
        return [item for item in self.items if item.status == "choose"]

    def items_for(self, operation_index: int) -> list[ConfirmItem]:
        return [item for item in self.items if item.operation_index == operation_index]
