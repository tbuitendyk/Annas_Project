"""
output/processor.py — Applies actions to frames and audio chunks
before they're sent to the virtual output.

In pass-through mode (filter disabled), frames flow through untouched.
When AI flags are present, this applies the appropriate effect:

Video actions:
  pass   → frame unchanged
  blur   → Gaussian blur over flagged regions (or whole frame)
  black  → replace frame with solid black

Audio actions:
  pass   → chunk unchanged
  mute   → replace samples with silence
  bleep  → replace flagged sample ranges with a 1kHz sine tone
"""

import numpy as np
import cv2
from loguru import logger

from buffer.ring_buffer import VideoFrame, AudioChunk


# ── Video processing ────────────────────────────────────────────────────────

def process_video_frame(vf: VideoFrame) -> np.ndarray:
    """
    Apply the action stamped on a VideoFrame and return the output frame.
    """
    if vf.action == "pass":
        return vf.frame

    # "skip" frames are seamlessly dropped by the output thread when the skip
    # buffer can cover the scene; if one reaches here (buffer drained, or the
    # seamless path is unavailable) it falls back to solid black.
    if vf.action in ("black", "skip"):
        return np.zeros_like(vf.frame)

    if vf.action == "blur":
        out = vf.frame.copy()
        if vf.blur_regions:
            # Blur only specified regions
            for (x, y, w, h) in vf.blur_regions:
                roi = out[y:y+h, x:x+w]
                blurred = cv2.GaussianBlur(roi, (51, 51), 0)
                out[y:y+h, x:x+w] = blurred
        else:
            # Blur entire frame
            out = cv2.GaussianBlur(out, (51, 51), 0)
        return out

    logger.warning(f"Unknown video action: {vf.action!r}, passing through")
    return vf.frame


# ── Audio processing ────────────────────────────────────────────────────────

def process_audio_chunk(chunk: AudioChunk) -> np.ndarray:
    """
    Apply the action stamped on an AudioChunk and return output samples.
    """
    if chunk.action == "pass":
        return chunk.samples

    samples = chunk.samples.copy()

    if chunk.action == "mute":
        return np.zeros_like(samples)

    if chunk.action == "bleep":
        if chunk.bleep_ranges:
            for (start, end) in chunk.bleep_ranges:
                start = max(0, start)
                end = min(len(samples), end)
                bleep = _generate_bleep(end - start, chunk.samplerate, samples.shape[1] if samples.ndim > 1 else 1)
                samples[start:end] = bleep
        else:
            # Bleep the entire chunk
            bleep = _generate_bleep(len(samples), chunk.samplerate, samples.shape[1] if samples.ndim > 1 else 1)
            samples = bleep
        return samples

    logger.warning(f"Unknown audio action: {chunk.action!r}, passing through")
    return chunk.samples


def _generate_bleep(n_samples: int, samplerate: int, channels: int, freq: float = 1000.0) -> np.ndarray:
    """Generate a 1kHz sine wave bleep tone."""
    t = np.linspace(0, n_samples / samplerate, n_samples, endpoint=False)
    tone = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    if channels > 1:
        tone = np.stack([tone] * channels, axis=1)
    return tone
