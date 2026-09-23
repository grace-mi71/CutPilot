"""Turning a selector into the tracks and boxes an edit actually lands on.

Two translations live here, and both exist for the same reason: what the user
can *describe* and what the edit must *cover* are often different objects.

    "the man in the black cap -- just his face"

The cap is only visible on the person box, so grounding has to run against
people. The mosaic has to land on a face. Something has to walk from one to the
other, and that something is the parent/child link built by link_children().
"""

from __future__ import annotations

from collections import Counter

from cutpilot.schemas.plan import Region, Selector
from cutpilot.schemas.track import Box, ObjectKind, Track, TrackBoxes

# A face sits almost wholly inside its person's box, so containment -- not IoU --
# is the right test: IoU between a face and a whole person is tiny even when the
# face unambiguously belongs to them.
CONTAINMENT_THRESHOLD = 0.6

# A child must agree with the same parent on most frames it was seen in. One
# frame where two people cross should not reassign a face.
AGREEMENT_RATIO = 0.5

CHILD_OF: dict[ObjectKind, ObjectKind] = {"face": "person"}
"""Which kind a child attaches to. Plates stay unattached in the MVP -- nothing
asks for "that car's plate" yet, and vehicles are not tracked as objects."""


def link_children(
    tracks: list[Track],
    boxes: dict[str, TrackBoxes],
    *,
    containment_threshold: float = CONTAINMENT_THRESHOLD,
    agreement_ratio: float = AGREEMENT_RATIO,
) -> list[Track]:
    """Fill in `parent_id` by geometry, returning updated copies.

    For each frame a child is seen in, find the parent-kind box that contains it
    best; the parent winning a majority of those frames gets the link.
    """
    by_kind: dict[ObjectKind, list[Track]] = {}
    for track in tracks:
        by_kind.setdefault(track.kind, []).append(track)

    resolved: list[Track] = []
    for track in tracks:
        parent_kind = CHILD_OF.get(track.kind)
        candidates = by_kind.get(parent_kind, []) if parent_kind else []
        if not candidates or track.id not in boxes:
            resolved.append(track.model_copy())
            continue

        votes: Counter[str] = Counter()
        samples = 0
        for child_box in boxes[track.id].boxes:
            best_id, best_score = None, containment_threshold
            for parent in candidates:
                parent_box = boxes.get(parent.id)
                if parent_box is None:
                    continue
                box = parent_box.at(child_box.t)
                if box is None:
                    continue
                score = Box(
                    x=child_box.x, y=child_box.y, w=child_box.w, h=child_box.h
                ).containment_in(box)
                if score > best_score:
                    best_id, best_score = parent.id, score
            samples += 1
            if best_id is not None:
                votes[best_id] += 1

        parent_id = None
        if votes and samples:
            winner, count = votes.most_common(1)[0]
            if count / samples >= agreement_ratio:
                parent_id = winner

        resolved.append(track.model_copy(update={"parent_id": parent_id}))

    return resolved


def translate_to_kind(track_id: str, kind: ObjectKind, tracks: list[Track]) -> list[str]:
    """Map an id onto the given kind, walking the parent link either way.

    A person asked for as a face yields that person's face tracks; a face asked
    for as a person yields its owner. An id already of the right kind passes
    straight through. An unknown id yields nothing rather than raising -- a
    selector that resolves to nothing becomes an `empty_selection` warning the
    user can see, which beats a stack trace.
    """
    index = {t.id: t for t in tracks}
    track = index.get(track_id)
    if track is None:
        return []
    if track.kind == kind:
        return [track.id]

    children = [t.id for t in tracks if t.kind == kind and t.parent_id == track.id]
    if children:
        return children

    if track.parent_id and (parent := index.get(track.parent_id)) and parent.kind == kind:
        return [parent.id]

    return []


def resolve_selector(selector: Selector, tracks: list[Track]) -> list[Track]:
    """The tracks a selector picks out, after kind translation.

    Empty `include` means the whole kind, which is how "cover every face" and
    "cover everyone except him" are expressed -- the second is the first with an
    exclusion, not a separate mode.
    """
    pool = [t for t in tracks if t.kind == selector.kind]

    if selector.include:
        wanted: set[str] = set()
        for raw_id in selector.include:
            wanted.update(translate_to_kind(raw_id, selector.kind, tracks))
        pool = [t for t in pool if t.id in wanted]

    if selector.exclude:
        unwanted: set[str] = set()
        for raw_id in selector.exclude:
            unwanted.update(translate_to_kind(raw_id, selector.kind, tracks))
        pool = [t for t in pool if t.id not in unwanted]

    return pool


def resolve_region_box(
    track: Track,
    region: Region,
    t: float,
    tracks: list[Track],
    boxes: dict[str, TrackBoxes],
) -> Box | None:
    """The box an edit covers at time `t`.

    The MVP only implements "bbox" -- the selected object's own box. Sub-object
    regions land here as extra branches when they arrive; nothing else in the
    pipeline has to change.
    """
    if region == "bbox":
        source = boxes.get(track.id)
        return source.at(t) if source else None

    if region == "face":
        for face_id in translate_to_kind(track.id, "face", tracks):
            if (source := boxes.get(face_id)) and (box := source.at(t)):
                return box
        return None

    if region == "body":  # pragma: no cover - reserved
        raise NotImplementedError("region='body' is reserved for a later stage")

    return None
