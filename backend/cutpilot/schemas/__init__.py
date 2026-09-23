"""Pydantic domain models shared across layers.

These are the contract between backend and the (deferred) timeline UI:
    VideoAsset  uploaded video + proxy + thumbnail sprite + metadata
    Track       one tracked object with its presence segments on the timeline
    EditPlan    declarative, user-editable plan produced by the agent

Coordinates are normalized 0.0-1.0 throughout; see schemas.track.
"""

from cutpilot.schemas.confirm import (
    Confirmation,
    ConfirmField,
    ConfirmItem,
    ConfirmStatus,
    Overlay,
    OverlayRole,
    PlanWarning,
    WarningKind,
)
from cutpilot.schemas.plan import (
    EditPlan,
    EditPlanDraft,
    Operation,
    OperationDraft,
    OpKind,
    Region,
    Selector,
    SelectorDraft,
    TimeRange,
)
from cutpilot.schemas.track import (
    Box,
    ObjectKind,
    Segment,
    TimedBox,
    Track,
    TrackBoxes,
)
from cutpilot.schemas.video import VideoAsset

__all__ = [
    # video
    "VideoAsset",
    # track
    "Box",
    "TimedBox",
    "Segment",
    "Track",
    "TrackBoxes",
    "ObjectKind",
    # plan
    "TimeRange",
    "Selector",
    "Operation",
    "EditPlan",
    "SelectorDraft",
    "OperationDraft",
    "EditPlanDraft",
    "OpKind",
    "Region",
    # confirm
    "Overlay",
    "OverlayRole",
    "ConfirmItem",
    "ConfirmField",
    "ConfirmStatus",
    "PlanWarning",
    "WarningKind",
    "Confirmation",
]
