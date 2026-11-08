from fastapi.testclient import TestClient

from cutpilot.api.app import create_app
from cutpilot.core.config import Settings


def test_offline_demo_is_separate_from_video_and_supports_corrections(tmp_path):
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
    )
    client = TestClient(create_app(settings))
    created = client.post("/demo/offline-sessions")
    assert created.status_code == 201
    assert created.json()["mode"] == "offline-demo"
    session_id = created.json()["id"]
    url = f"/demo/offline-sessions/{session_id}/messages"

    reply = client.post(url, json={"text": "person_02 블러"}).json()
    op = reply["confirmation"]["plan"]["operations"][0]
    assert (op["op"], op["selector"]["include"]) == ("BLUR", ["person_02"])
    assert reply["can_approve"] is False

    reply = client.post(url, json={"text": "3초부터 6초까지"}).json()
    assert reply["confirmation"]["plan"]["operations"][0]["time_range"] == {
        "start": 3.0, "end": 6.0,
    }
    assert client.post("/demo/offline-sessions/unknown/messages", json={"text": "블러"}).status_code == 404


def test_offline_demo_described_person_and_unsupported_request(tmp_path):
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data", output_dir=tmp_path / "output",
        cache_dir=tmp_path / "cache",
    )
    client = TestClient(create_app(settings))
    session_id = client.post("/demo/offline-sessions").json()["id"]
    url = f"/demo/offline-sessions/{session_id}/messages"
    reply = client.post(url, json={"text": "가방 멘 사람 모자이크"}).json()
    assert reply["status"] == "plan"
    assert reply["blockers"]
    other_id = client.post("/demo/offline-sessions").json()["id"]
    reply = client.post(f"/demo/offline-sessions/{other_id}/messages", json={"text": "자막 넣어줘"}).json()
    assert reply["status"] == "clarify"
