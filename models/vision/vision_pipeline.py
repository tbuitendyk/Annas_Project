"""
models/vision/vision_pipeline.py — Real-time video content analysis.

Mirrors AudioAnalysisPipeline: a background thread that samples the newest
buffered frames at a fixed rate, runs the hybrid ContentDetector, decides one
action per frame using the user's per-category severities, and stamps the ring
buffer so the output thread applies the effect on release.

Subsampling: only ~sample_hz frames/second are actually run through the model;
each verdict is held across the gap by stamping every captured frame in that
window. Objectionable content persists for seconds, so this is visually
lossless while cutting inference cost several-fold.

Action vocabulary stamped onto VideoFrame.action:
    "blur"  (+ blur_regions)  — mild (localized) or moderate (whole frame)
    "skip"                    — severe; output skips/blacks the section
Frames the detector clears are left as "pass" (never un-stamped here, so once
flagged a frame stays flagged until it's consumed).
"""

import time
import threading
from loguru import logger

from buffer.ring_buffer import VideoRingBuffer
from models.vision.detector import ContentDetector, decide_frame_action


class VideoAnalysisPipeline(threading.Thread):
    def __init__(self, video_buf: VideoRingBuffer, content_cfg):
        super().__init__(daemon=True, name="VideoAnalysis")
        self.buf = video_buf
        self.cfg = content_cfg                 # config.ContentFilterConfig
        self._stop_event = threading.Event()
        self._detector = None
        self._ready = False
        self._frames_analyzed = 0
        self._detections = 0

    # Decision action -> (VideoFrame.action, regions)
    @staticmethod
    def _to_frame_action(action: str, regions: list) -> tuple[str, list]:
        if action == "blur_region":
            return "blur", regions
        if action == "blur_frame":
            return "blur", []
        if action == "skip":
            return "skip", []
        return "pass", []

    def run(self):
        try:
            self._detector = ContentDetector(device=self.cfg.device)
            self._detector.load()
        except Exception as e:
            logger.error(f"VideoAnalysis: detector init failed: {e}")
            return
        self._ready = self._detector.ready
        if not self._ready:
            logger.error("VideoAnalysis: no detector backend — video filtering disabled")
            return

        sample_hz = max(1.0, float(self.cfg.sample_hz))
        interval = 1.0 / sample_hz
        logger.info(f"VideoAnalysis: running at {sample_hz:.0f} Hz on {self._detector.device}")

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            # Newest slice of buffered frames; analyze the most recent one and
            # hold its verdict across the whole slice (the subsample gap).
            window = self.buf.peek_unprocessed(window_seconds=interval * 2.0)
            if not window:
                time.sleep(interval)
                continue

            newest = window[-1]
            try:
                hits = self._detector.detect(newest.frame)
            except Exception as e:
                logger.debug(f"VideoAnalysis: detect error: {e}")
                hits = []

            action, regions = decide_frame_action(hits, self.cfg.categories)
            self._frames_analyzed += 1

            if action != "pass":
                self._detections += 1
                vf_action, vf_regions = self._to_frame_action(action, regions)
                updates = {f.timestamp: (vf_action, vf_regions) for f in window}
                self.buf.update_actions(updates)
                logger.debug(f"VideoAnalysis: {action} over {len(window)} frames "
                             f"(regions={len(vf_regions)})")

            elapsed = time.monotonic() - t0
            sleep = interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def stop(self):
        self._stop_event.set()

    @property
    def stats(self) -> dict:
        return {
            "ready": self._ready,
            "device": self._detector.device if self._detector else "—",
            "frames_analyzed": self._frames_analyzed,
            "detections": self._detections,
        }
