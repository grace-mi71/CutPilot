"""Offline keyword-chat sessions using synthetic tracks; no analysis or rendering."""
from __future__ import annotations

import threading
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from cutpilot.agent.llm.offline import OfflineProvider
from cutpilot.agent.session import EditSession, approval_blockers
from cutpilot.api.models import ReplyOut, SendMessage
from cutpilot.schemas.track import Segment, TimedBox, Track, TrackBoxes
from cutpilot.schemas.video import VideoAsset


def create_offline_demo_router() -> APIRouter:
    router = APIRouter(prefix="/demo/offline-sessions", tags=["offline chat"])
    sessions: dict[str, tuple[EditSession, threading.Lock]] = {}
    sessions_lock = threading.Lock()

    @router.post("", status_code=201)
    def create_demo_session() -> dict[str, str]:
        # These are explicitly fictional people, not detections from an upload.
        tracks = [
            Track(id=f"person_{i:02}", kind="person", segments=[Segment(start=0, end=30)])
            for i in (1, 2)
        ]
        boxes = {
            track.id: TrackBoxes(
                track_id=track.id,
                boxes=[
                    TimedBox(t=step / 2, x=0.1 if i == 0 else 0.55, y=0.1, w=0.3, h=0.8)
                    for step in range(61)
                ],
            )
            for i, track in enumerate(tracks)
        }
        asset = VideoAsset(
            id="offline-demo", source_path="", proxy_path="", sprite_path="",
            duration=30, fps=10, width=640, height=360,
        )
        session = EditSession(
            OfflineProvider(), asset=asset, tracks=tracks, boxes=boxes,
            frame_loader=lambda _t: b"",
        )
        session_id = uuid4().hex
        with sessions_lock:
            sessions[session_id] = (session, threading.Lock())
        return {"id": session_id, "mode": "offline-demo"}

    @router.post("/{session_id}/messages", response_model=ReplyOut)
    def send_demo_message(session_id: str, body: SendMessage) -> ReplyOut:
        with sessions_lock:
            entry = sessions.get(session_id)
        if entry is None:
            raise HTTPException(404, "오프라인 대화가 종료되었습니다. 다시 시도해 주세요.")
        session, lock = entry
        with lock:
            reply = session.say(body.text)
            confirmation = reply.confirmation
            return ReplyOut(
                status=reply.status,
                text=reply.text.replace("괜찮으면 승인해 주세요.", "편집 계획을 확인해 주세요."),
                notes=reply.notes,
                confirmation=confirmation, can_approve=False,
                blockers=approval_blockers(confirmation) if confirmation else [],
            )

    return router
