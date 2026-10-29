"""Request and response bodies -- the contract the frontend codes against.

Domain models (Track, Confirmation, EditPlan) go out as they are; these only
wrap them with what a browser needs on top: URLs instead of disk paths, and
whether the approve button should be live.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cutpilot.schemas.confirm import Confirmation
from cutpilot.schemas.plan import EditPlan
from cutpilot.schemas.session import UIContext
from cutpilot.schemas.track import Box, ObjectKind, Track

# ------------------------------------------------------------------ videos


class VideoOut(BaseModel):
    id: str
    duration: float
    fps: float
    width: int
    height: int
    proxy_url: str
    """Play this in the preview, not the original: same timing, far lighter."""
    sprite_url: str
    sprite_interval: float
    sprite_columns: int
    tracks: list[Track]
    """Presence segments per object -- the timeline lanes."""


class TrackBox(BaseModel):
    track_id: str
    kind: ObjectKind
    box: Box


class BoxesOut(BaseModel):
    """Every object on screen at one moment, for drawing clickable boxes."""

    t: float
    boxes: list[TrackBox]


# ---------------------------------------------------------------- sessions


class CreateSession(BaseModel):
    video_id: str


class SendMessage(BaseModel):
    text: str = Field(min_length=1)
    ui: UIContext | None = None


class Choose(BaseModel):
    track_ids: list[str] = Field(min_length=1)
    side: Literal["include", "exclude"] = "include"


class Toggle(BaseModel):
    track_id: str


class SetTimeRange(BaseModel):
    start: float | None = None
    end: float | None = None


class Approve(BaseModel):
    preview: bool = True
    """True renders the proxy in about a second; False renders at source size."""


class ReplyOut(BaseModel):
    status: Literal["plan", "clarify", "error"]
    text: str
    notes: list[str] = Field(default_factory=list)
    confirmation: Confirmation | None = None
    can_approve: bool = False
    blockers: list[str] = Field(default_factory=list)
    """Why approval is refused right now, in words the user can act on."""


class TurnOut(BaseModel):
    role: str
    text: str


class RenderOut(BaseModel):
    url: str
    preview: bool
    width: int
    height: int
    frames: int


class SessionOut(BaseModel):
    id: str
    video_id: str
    provider: str
    """'offline-keywords' when no model is attached -- worth showing in the UI."""
    confirmation: Confirmation | None = None
    can_approve: bool = False
    blockers: list[str] = Field(default_factory=list)
    approved: EditPlan | None = None
    history: list[TurnOut] = Field(default_factory=list)
    render: RenderOut | None = None


class ApproveOut(BaseModel):
    plan: EditPlan
    render: RenderOut
