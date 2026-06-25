"""
config.py — CleanStream configuration with validation.
Reads from cleanstream.toml, falls back to safe defaults.
"""

import os
import toml
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

CONFIG_PATH = Path("cleanstream.toml")


# ── Content filtering taxonomy ──────────────────────────────────────────────
#
# The user assigns each content category a severity in the GUI; the severity
# determines the action applied to flagged frames/sections:
#
#   off      → category ignored
#   mild     → detect-and-localize: blur just the offending region
#   moderate → whole-frame blur
#   severe   → skip the section (seamlessly if it fits the skip buffer, else
#              cut to black + mute until the scene ends)
#
# Categories are user-facing concepts; the vision detector maps its output to
# these keys. Kept here (not in config) so the GUI and detector share one list.

SEVERITY_LEVELS = ("off", "mild", "moderate", "severe")

SEVERITY_ACTIONS = {
    "off":      None,
    "mild":     "blur_region",
    "moderate": "blur_frame",
    "severe":   "skip",
}

CONTENT_CATEGORIES = [
    {"key": "nudity",            "label": "Nudity / explicit",
     "desc": "Exposed breasts, genitals or buttocks",            "default": "severe"},
    {"key": "sexual_content",    "label": "Sexual content",
     "desc": "Sex acts, simulated or explicit",                  "default": "severe"},
    {"key": "bedroom_suggestive","label": "Suggestive / bedroom",
     "desc": "Partial undress, suggestive posing, bedroom scenes","default": "moderate"},
    {"key": "kissing_intimacy",  "label": "Kissing / intimacy",
     "desc": "Passionate kissing, making out",                   "default": "mild"},
    {"key": "violence",          "label": "Violence",
     "desc": "Fighting, weapons, assault",                       "default": "moderate"},
    {"key": "gore_injury",       "label": "Gore / graphic injury",
     "desc": "Blood, wounds, mutilation",                        "default": "severe"},
]


def default_categories() -> dict:
    """Fresh {category_key: default_severity} map from the registry."""
    return {c["key"]: c["default"] for c in CONTENT_CATEGORIES}



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


class ContentFilterConfig(BaseModel):
    """Video content filtering: per-category severities + detection settings."""
    enabled: bool = Field(False, description="Master switch for video content filtering")
    device: str = Field("auto", description="Inference device: auto | gpu | cpu")
    sample_hz: float = Field(6.0, ge=1.0, le=30.0,
                             description="Frames per second analyzed by the detector")
    skip_buffer_seconds: float = Field(
        60.0, ge=0.0, le=600.0,
        description="Lookahead for seamless skipping of 'severe' scenes; "
                    "beyond this length a scene is blacked out + muted instead")
    categories: dict[str, str] = Field(default_factory=default_categories,
                                       description="category_key -> severity level")

    @field_validator("categories")
    @classmethod
    def _coerce_categories(cls, v):
        # Start from registry defaults so new categories appear automatically,
        # and drop unknown keys / invalid severities from older config files.
        cleaned = default_categories()
        for key, sev in (v or {}).items():
            if key in cleaned and sev in SEVERITY_LEVELS:
                cleaned[key] = sev
        return cleaned

    @field_validator("device")
    @classmethod
    def _coerce_device(cls, v):
        return v if v in ("auto", "gpu", "cpu") else "auto"


class AppConfig(BaseModel):
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    content: ContentFilterConfig = Field(default_factory=ContentFilterConfig)


def load_config() -> AppConfig:
    if CONFIG_PATH.exists():
        raw = toml.load(CONFIG_PATH)
        return AppConfig(**raw)
    return AppConfig()


def save_config(cfg: AppConfig) -> None:
    with open(CONFIG_PATH, "w") as f:
        toml.dump(cfg.model_dump(), f)
