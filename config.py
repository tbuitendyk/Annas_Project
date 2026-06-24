"""
config.py — CleanStream configuration with validation.
Reads from cleanstream.toml, falls back to safe defaults.
"""

import os
import toml
from pydantic import BaseModel, Field
from pathlib import Path

CONFIG_PATH = Path("cleanstream.toml")


class CaptureConfig(BaseModel):
    monitor_index: int = Field(0, description="Which monitor to capture (0 = primary)")
    window_title: str = Field("", description="Capture a specific window by title (empty = full monitor)")
    fps: int = Field(24, ge=1, le=60)
    frame_width: int = Field(1920)
    frame_height: int = Field(1080)
    audio_device_in: str = Field("", description="Audio input device name (empty = system default loopback)")
    audio_device_out: str = Field("", description="Audio output device name (empty = system default)")
    audio_samplerate: int = Field(44100)
    audio_channels: int = Field(2)


class BufferConfig(BaseModel):
    delay_seconds: float = Field(4.0, ge=1.0, le=10.0, description="A/V delay buffer in seconds")
    max_seconds: float = Field(600.0, description="Maximum ring buffer size (seconds). 600 = 10 min pause buffer")


class OutputConfig(BaseModel):
    virtual_camera_device: str = Field("", description="Virtual cam device (empty = auto-detect)")
    preview_enabled: bool = Field(True, description="Show preview window")
    preview_scale: float = Field(0.4, description="Preview window scale factor")


class FilterConfig(BaseModel):
    enabled: bool = Field(False, description="AI filtering enabled (False = pure pass-through)")
    audio_enabled: bool = Field(False)
    video_enabled: bool = Field(False)


class AppConfig(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)


def load_config() -> AppConfig:
    if CONFIG_PATH.exists():
        raw = toml.load(CONFIG_PATH)
        return AppConfig(**raw)
    return AppConfig()


def save_config(cfg: AppConfig) -> None:
    with open(CONFIG_PATH, "w") as f:
        toml.dump(cfg.model_dump(), f)
