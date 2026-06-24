"""
buffer/ring_buffer.py — Thread-safe ring buffer for video frames and audio chunks.

Holds up to `max_seconds` of content. The consumer always reads
frames that are exactly `delay_seconds` behind the producer,
giving the AI pipeline time to analyze and flag segments before
they're released to the output.
"""

import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class VideoFrame:
    frame: np.ndarray       # HxWx3 uint8 BGR
    timestamp: float        # monotonic time of capture
    action: str = "pass"    # pass | blur | black — set by AI pipeline
    blur_regions: list = field(default_factory=list)  # [(x,y,w,h), ...]


@dataclass
class AudioChunk:
    samples: np.ndarray     # float32 samples, shape (N, channels)
    timestamp: float
    samplerate: int
    action: str = "pass"    # pass | mute | bleep
    bleep_ranges: list = field(default_factory=list)  # [(start_sample, end_sample), ...]


class VideoRingBuffer:
    """
    Thread-safe video frame buffer with configurable delay.
    Producer calls push(), consumer calls pop_ready() which only
    returns frames whose timestamp is >= delay_seconds old.
    """

    def __init__(self, delay_seconds: float = 4.0, max_seconds: float = 10.0, fps: int = 30):
        self.delay_seconds = delay_seconds
        self._max_frames = int(max_seconds * fps)
        self._buf: deque[VideoFrame] = deque(maxlen=self._max_frames)
        self._lock = threading.Lock()
        self._pushed = 0
        self._dropped = 0

    def push(self, frame: np.ndarray) -> None:
        vf = VideoFrame(frame=frame, timestamp=time.monotonic())
        with self._lock:
            if len(self._buf) == self._buf.maxlen:
                self._dropped += 1
            self._buf.append(vf)
            self._pushed += 1

    def pop_ready(self) -> VideoFrame | None:
        """Return oldest frame if it's older than delay_seconds, else None."""
        now = time.monotonic()
        with self._lock:
            if not self._buf:
                return None
            oldest = self._buf[0]
            if now - oldest.timestamp >= self.delay_seconds:
                return self._buf.popleft()
        return None

    def peek_unprocessed(self, window_seconds: float = 2.0) -> list[VideoFrame]:
        """
        Return frames in the newest `window_seconds` of the buffer
        for the AI pipeline to analyze. Does NOT remove them.
        """
        now = time.monotonic()
        with self._lock:
            return [
                f for f in self._buf
                if now - f.timestamp <= window_seconds
            ]

    def update_actions(self, updates: dict[float, tuple[str, list]]) -> None:
        """
        AI pipeline calls this to stamp frames with actions before they're released.
        updates = { timestamp: (action, blur_regions) }
        """
        with self._lock:
            for frame in self._buf:
                if frame.timestamp in updates:
                    action, regions = updates[frame.timestamp]
                    frame.action = action
                    frame.blur_regions = regions

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "buffered_frames": len(self._buf),
                "pushed": self._pushed,
                "dropped": self._dropped,
            }


class AudioRingBuffer:
    """
    Thread-safe audio chunk buffer with configurable delay.
    Chunks are variable-length numpy arrays.
    """

    def __init__(self, delay_seconds: float = 4.0, max_seconds: float = 10.0, samplerate: int = 44100):
        self.delay_seconds = delay_seconds
        self._samplerate = samplerate
        self._max_samples = int(max_seconds * samplerate)
        self._buf: deque[AudioChunk] = deque()
        self._total_samples = 0
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray, samplerate: int) -> None:
        chunk = AudioChunk(
            samples=samples.copy(),
            timestamp=time.monotonic(),
            samplerate=samplerate,
        )
        with self._lock:
            self._buf.append(chunk)
            self._total_samples += len(samples)
            # Trim buffer if it exceeds max
            while self._total_samples > self._max_samples and self._buf:
                dropped = self._buf.popleft()
                self._total_samples -= len(dropped.samples)

    def pop_ready(self) -> AudioChunk | None:
        now = time.monotonic()
        with self._lock:
            if not self._buf:
                return None
            oldest = self._buf[0]
            if now - oldest.timestamp >= self.delay_seconds:
                self._total_samples -= len(oldest.samples)
                return self._buf.popleft()
        return None

    def peek_seconds(self, seconds: float = 3.0) -> list[AudioChunk]:
        """Return newest `seconds` worth of chunks for AI analysis."""
        now = time.monotonic()
        with self._lock:
            return [c for c in self._buf if now - c.timestamp <= seconds]

    def peek_window(self, min_age: float, max_age: float) -> list[AudioChunk]:
        """
        Return chunks whose age is between min_age and max_age seconds.
        Used by AI pipeline to analyze audio that's still safely in the buffer.
        e.g. peek_window(1.0, 4.0) returns chunks from 1-4 seconds ago.
        """
        now = time.monotonic()
        with self._lock:
            return [
                c for c in self._buf
                if min_age <= (now - c.timestamp) <= max_age
            ]

    def update_actions(self, updates: dict[float, tuple[str, list]]) -> None:
        with self._lock:
            for chunk in self._buf:
                if chunk.timestamp in updates:
                    action, ranges = updates[chunk.timestamp]
                    chunk.action = action
                    chunk.bleep_ranges = ranges

    @property
    def buffered_seconds(self) -> float:
        with self._lock:
            return self._total_samples / self._samplerate
