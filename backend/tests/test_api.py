"""The HTTP surface, upload to render, over a small synthetic clip.

Analysis is stubbed with the three-person scene from the grounding tests --
YOLO is not what is under test -- but probing, rendering and file serving are
real, so the clip has to be a real video.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cutpilot.agent.llm import OfflineProvider
from cutpilot.api.app import create_app
from cutpilot.core.config import Settings
from cutpilot.schemas import Track, VideoAsset
from cutpilot.vision.detect import TrackingResult
from cutpilot.vision.resolve import link_children
from tests.test_grounding import DURATION, GEOMETRY, _boxes, _presence

FPS = 10.0


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("clip") / "scene.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (192, 108))
    rng = np.random.default_rng(0)
    for _ in range(int(DURATION * FPS)):
        writer.write(rng.integers(0, 255, (108, 192, 3), dtype=np.uint8))
    writer.release()
    return path


def scene_analyzer(output_dir: Path):
    def analyze(source: Path, video_id: str):
        work = output_dir / video_id
        work.mkdir(parents=True, exist_ok=True)
        proxy = work / "proxy.mp4"
        shutil.copy(source, proxy)
        sprite = work / "sprite.jpg"
        cv2.imwrite(str(sprite), np.zeros((9, 16, 3), np.uint8))

        boxes = {tid: _boxes(tid) for tid in GEOMETRY}
        tracks = [
            Track(
                id=tid, kind="face" if tid.startswith("face") else "person",
                segments=_presence(tid),
            )
            for tid in GEOMETRY
        ]  # fmt: skip
        asset = VideoAsset(
            id=video_id, source_path=str(source), proxy_path=str(proxy),
            sprite_path=str(sprite), duration=DURATION, fps=FPS, width=192, height=108,
        )  # fmt: skip
        return asset, TrackingResult(tracks=link_children(tracks, boxes), boxes=boxes)

    return analyze


@pytest.fixture
def client(tmp_path) -> TestClient:
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )  # fmt: skip
    app = create_app(
        settings, analyze=scene_analyzer(settings.output_dir), provider_factory=OfflineProvider
    )
    return TestClient(app)


@pytest.fixture
def video(client, clip) -> dict:
    with clip.open("rb") as f:
        response = client.post("/videos", files={"file": ("scene.mp4", f, "video/mp4")})
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
def session(client, video) -> str:
    response = client.post("/sessions", json={"video_id": video["id"]})
    assert response.status_code == 201
    return response.json()["id"]


def say(client, session, text, **ui):
    response = client.post(f"/sessions/{session}/messages", json={"text": text, "ui": ui or None})
    assert response.status_code == 200, response.text
    return response.json()


class TestVideos:
    def test_upload_returns_the_timeline(self, client, video):
        assert video["duration"] == DURATION
        assert {t["id"] for t in video["tracks"]} == set(GEOMETRY)
        assert client.get(video["proxy_url"]).status_code == 200

    def test_video_survives_a_restart(self, client, video, tmp_path):
        settings = Settings(
            _env_file=None, data_dir=tmp_path / "data", output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
        )  # fmt: skip
        fresh = TestClient(create_app(settings, analyze=None, provider_factory=OfflineProvider))
        assert fresh.get(f"/videos/{video['id']}").json()["id"] == video["id"]

    def test_boxes_at_a_moment(self, client, video):
        boxes = client.get(f"/videos/{video['id']}/boxes", params={"t": 5.0}).json()["boxes"]
        assert "person_03" not in {b["track_id"] for b in boxes}  # absent 4-6s
        assert "person_01" in {b["track_id"] for b in boxes}

    def test_non_video_is_rejected(self, client):
        response = client.post("/videos", files={"file": ("a.mp4", b"not a video", "video/mp4")})
        assert response.status_code == 400

    def test_unknown_video(self, client):
        assert client.get("/videos/nope").status_code == 404


class TestConversation:
    def test_described_target_asks_which_one(self, client, session):
        reply = say(client, session, "가방 멘 사람 모자이크", playhead=1.0)
        assert reply["status"] == "plan"
        assert not reply["can_approve"]
        assert reply["blockers"]
        assert reply["confirmation"]["items"][1]["status"] == "choose"
        assert reply["confirmation"]["overlays"]

    def test_choose_then_approve_renders(self, client, session):
        say(client, session, "가방 멘 사람 모자이크", playhead=1.0)
        reply = client.post(
            f"/sessions/{session}/operations/0/choose", json={"track_ids": ["person_02"]}
        ).json()
        assert reply["can_approve"]

        approved = client.post(f"/sessions/{session}/approve", json={"preview": True})
        assert approved.status_code == 200, approved.text
        render = approved.json()["render"]
        assert render["frames"] == int(DURATION * FPS)
        assert client.get(render["url"]).status_code == 200
        assert client.get(f"/sessions/{session}").json()["render"]["url"] == render["url"]

    def test_spoken_correction_keeps_the_rest(self, client, session):
        say(client, session, "person_02 블러")
        reply = say(client, session, "3초부터 6초까지")
        op = reply["confirmation"]["plan"]["operations"][0]
        assert op["op"] == "BLUR"
        assert op["selector"]["include"] == ["person_02"]
        assert (op["time_range"]["start"], op["time_range"]["end"]) == (3.0, 6.0)

    def test_clicks(self, client, session):
        say(client, session, "사람 전부 모자이크")
        base = f"/sessions/{session}/operations/0"
        reply = client.post(f"{base}/toggle", json={"track_id": "person_01"}).json()
        assert reply["confirmation"]["plan"]["operations"][0]["selector"]["exclude"] == [
            "person_01"
        ]
        reply = client.put(f"{base}/time_range", json={"start": 1, "end": 4}).json()
        assert reply["confirmation"]["plan"]["operations"][0]["time_range"]["end"] == 4.0
        reply = client.delete(base).json()
        assert reply["confirmation"]["plan"]["operations"] == []

    def test_out_of_scope_asks_back(self, client, session):
        assert say(client, session, "자막 넣어줘")["status"] == "clarify"

    def test_history_is_exposed(self, client, session):
        say(client, session, "person_02 블러")
        history = client.get(f"/sessions/{session}").json()["history"]
        assert history[0] == {"role": "user", "text": "person_02 블러"}


class TestErrors:
    def test_approval_refused_with_reasons(self, client, session):
        say(client, session, "가방 멘 사람 모자이크")
        response = client.post(f"/sessions/{session}/approve")
        assert response.status_code == 409
        assert response.json()["detail"]["reasons"]

    def test_bad_click_is_400(self, client, session):
        say(client, session, "person_02 블러")
        response = client.post(
            f"/sessions/{session}/operations/0/toggle", json={"track_id": "person_99"}
        )
        assert response.status_code == 400

    def test_missing_operation_is_404(self, client, session):
        say(client, session, "person_02 블러")
        assert client.delete(f"/sessions/{session}/operations/5").status_code == 404

    def test_unknown_session_is_404(self, client):
        assert client.get("/sessions/nope").status_code == 404
