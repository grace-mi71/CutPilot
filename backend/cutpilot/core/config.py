"""Typed settings, loaded from the environment or a local .env.

Two values here are load-bearing beyond convenience:

    gemini_model    pinned, not "latest". Week 11 compares three interaction
                    conditions against each other; if the model shifts under
                    the measurements the comparison means nothing.
    device          every inference path takes a CPU fallback, so a broken CUDA
                    install stops inference without stopping development.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env", PROJECT_ROOT / ".env"),
        env_prefix="CUTPILOT_",
        extra="ignore",
    )

    # ---- LLM -------------------------------------------------------------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    """Pinned deliberately. Bump it as a decision, not as a side effect."""

    llm_cache_enabled: bool = True
    llm_timeout_seconds: float = 60.0

    # ---- Inference -------------------------------------------------------
    device: Literal["auto", "cuda", "cpu"] = "auto"
    person_model: str = "yolo11n.pt"
    face_model: str = "yolo11n-face.pt"
    plate_model: str = "yolo11n-plate.pt"
    detect_confidence: float = 0.35

    # ---- Ingest ----------------------------------------------------------
    proxy_height: int = 480
    sprite_interval: float = 1.0
    sprite_thumb_width: int = 160
    sprite_columns: int = 20
    max_video_seconds: float = 180.0
    """Uploads are capped while the pipeline is young; long clips turn a slow
    render into a stalled demo."""

    # ---- Editing ---------------------------------------------------------
    default_blur_strength: int = 25
    default_mosaic_strength: int = 20
    reframe_smoothing_window: float = 1.0
    """Seconds of trajectory averaged to keep the crop from jittering."""

    # ---- Paths -----------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    output_dir: Path = Field(default=PROJECT_ROOT / "outputs")
    weights_dir: Path = Field(default=PROJECT_ROOT / "weights")
    cache_dir: Path = Field(default=PROJECT_ROOT / "outputs" / ".llm_cache")

    @field_validator("data_dir", "output_dir", "weights_dir", "cache_dir", mode="after")
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @property
    def has_llm(self) -> bool:
        return bool(self.gemini_api_key)

    def resolved_device(self) -> str:
        """Resolve "auto" to cuda only if a CUDA kernel can actually run.

        torch.cuda.is_available() returns True on a card whose architecture the
        build has no kernels for, which fails much later and much less clearly.
        """
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        if not torch.cuda.is_available():
            return "cpu"
        major, minor = torch.cuda.get_device_capability()
        return "cuda" if f"sm_{major}{minor}" in torch.cuda.get_arch_list() else "cpu"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.output_dir, self.weights_dir, self.cache_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
