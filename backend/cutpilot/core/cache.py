"""On-disk cache for LLM responses.

Its first job is reproducibility, not speed. Week 11 measures the same
instruction set under three interaction conditions; if the model answers
differently on the second run, the comparison measures nothing. Cached replies
keep a benchmark reproducible, keep re-runs off the rate limit, and make it
possible to replay a failure case exactly while debugging.

The key covers everything that could change an answer -- model, prompt, images,
and generation settings -- so a pinned model plus a warm cache gives the same
plan every time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ResponseCache:
    """Content-addressed JSON cache. One file per response."""

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self.directory = directory
        self.enabled = enabled
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(
        *,
        model: str,
        prompt: str,
        images: list[bytes] | None = None,
        **params: Any,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(model.encode())
        digest.update(b"\x00")
        digest.update(prompt.encode())
        for image in images or []:
            digest.update(b"\x00")
            digest.update(hashlib.sha256(image).digest())
        if params:
            digest.update(b"\x00")
            digest.update(json.dumps(params, sort_keys=True, default=str).encode())
        return digest.hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        except (json.JSONDecodeError, KeyError, OSError):
            # A truncated write from an interrupted run should be a miss, not a
            # crash. The next call overwrites it.
            return None

    def set(self, key: str, response: Any, *, meta: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        payload = {"response": response, "meta": meta or {}}
        tmp = self._path(key).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path(key))

    def clear(self) -> int:
        removed = 0
        for path in self.directory.glob("*.json"):
            path.unlink()
            removed += 1
        return removed

    def __len__(self) -> int:
        return sum(1 for _ in self.directory.glob("*.json")) if self.enabled else 0
