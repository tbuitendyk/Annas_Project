"""
models/audio/audio_pipeline.py — Real-time audio analysis pipeline.

Uses independent 5-second chunks fed to Whisper separately, which
prevents Whisper from skipping sections it deems "not speech" when
analyzing longer clips with background music or sound effects.
"""

import time
import threading
import numpy as np
from pathlib import Path
from loguru import logger

from buffer.ring_buffer import AudioRingBuffer
from models.audio.classifier import ProfanityClassifier

CHUNK_SECONDS  = 5.0   # size of each independent Whisper chunk
STEP_SECONDS   = 2.5   # advance window by this much each pass (50% overlap)
MUTE_MARGIN    = 0.35  # silence padding around flagged words (seconds)


class AudioAnalysisPipeline(threading.Thread):
    """
    Analyzes buffered audio using independent overlapping 5-second chunks.

    Each chunk is fed to Whisper independently so it can't skip sections
    that contain background music. Overlapping windows ensure no word
    falls in a gap between chunks.
    """

    def __init__(self, audio_buf: AudioRingBuffer,
                 model_size: str = "small",
                 log_path: Path = Path("flagged_audio.jsonl")):
        super().__init__(daemon=True, name="AudioAnalysis")
        self.buf = audio_buf
        self.model_size = model_size
        self.log_path = log_path
        self._stop_event = threading.Event()
        self._whisper = None
        self._classifier = ProfanityClassifier()
        self._detections = 0
        self._chunks_analyzed = 0
        self._ready = False
        # Window position: age of the start of current analysis window
        self._window_start_age = None
        # Track which timestamps we've already muted to avoid double-muting
        self._muted_ranges: list[tuple[float, float]] = []

    def _load_whisper(self):
        try:
            import whisper
            logger.info(f"AudioAnalysis: loading Whisper '{self.model_size}'...")
            self._whisper = whisper.load_model(self.model_size)
            logger.info("AudioAnalysis: Whisper ready")
            self._ready = True
        except ImportError:
            logger.error("Whisper not installed. Run: pip install openai-whisper")
        except Exception as e:
            logger.error(f"AudioAnalysis: failed to load Whisper: {e}")

    def run(self):
        self._load_whisper()
        if not self._ready:
            return

        logger.info("AudioAnalysis: pipeline running (chunked mode)")

        # Start window at a safe position in the buffer
        safety_tail = 0.8
        self._window_start_age = self.buf.delay_seconds - safety_tail - CHUNK_SECONDS

        while not self._stop_event.is_set():
            chunks = self._get_window_chunks()

            if not chunks:
                time.sleep(0.1)
                continue

            samplerate = chunks[0].samplerate
            samples = np.concatenate([c.samples for c in chunks], axis=0)

            # Convert to mono float32
            mono = samples.mean(axis=1).astype(np.float32) if samples.ndim > 1 \
                   else samples.astype(np.float32)

            # Resample to 16kHz
            if samplerate != 16000:
                mono = self._resample(mono, samplerate, 16000)

            # Transcribe this chunk independently
            try:
                result = self._whisper.transcribe(
                    mono,
                    language="en",                      # skip language detection overhead
                    word_timestamps=True,
                    fp16=False,
                    verbose=False,
                    condition_on_previous_text=False,   # each chunk is independent
                    no_speech_threshold=0.1,            # aggressive — catch speech over music
                    temperature=(0.0, 0.2, 0.4),        # retry if low confidence
                    compression_ratio_threshold=3.0,
                )
            except Exception as e:
                logger.warning(f"Whisper error: {e}")
                time.sleep(0.5)
                self._advance_window()
                continue

            self._chunks_analyzed += 1
            text = result.get("text", "").strip()

            # Skip hallucinations (repetitive output)
            if self._is_hallucination(result):
                logger.debug(f"AudioAnalysis: hallucination detected, skipping")
                self._advance_window()
                continue

            if text:
                severity, hits = self._classifier.classify(text)
                logger.debug(f"AudioAnalysis: [{severity}] {text!r}")

                if severity != "clean" and hits:
                    logger.info(f"AudioAnalysis: FLAGGED [{severity}] {text!r} hits={hits}")
                    self._detections += 1
                    mute_ranges = self._get_mute_ranges(result, hits)
                    if mute_ranges:
                        self._stamp_chunks(chunks, mute_ranges, samplerate)
                    self._log_detection(text, severity, hits)

            self._advance_window()

    def _get_window_chunks(self) -> list:
        """Return chunks in the current analysis window."""
        if self._window_start_age is None:
            return []
        min_age = self._window_start_age
        max_age = self._window_start_age + CHUNK_SECONDS
        chunks = self.buf.peek_window(min_age, max_age)
        if not chunks:
            return []
        total = sum(len(c.samples) / c.samplerate for c in chunks)
        if total < CHUNK_SECONDS * 0.4:
            return []
        return chunks

    def _advance_window(self):
        """Slide window forward by STEP_SECONDS."""
        if self._window_start_age is None:
            return
        self._window_start_age -= STEP_SECONDS
        safety_head = 0.5
        safety_tail = 0.8
        min_age = safety_head
        max_age = self.buf.delay_seconds - safety_tail - CHUNK_SECONDS
        self._window_start_age = max(min_age, min(self._window_start_age, max_age))

    def _is_hallucination(self, result: dict) -> bool:
        """Detect repetitive hallucination output."""
        words = [
            w["word"].strip().lower()
            for seg in result.get("segments", [])
            for w in seg.get("words", [])
        ]
        if len(words) < 6:
            return False
        from collections import Counter
        top_word, top_count = Counter(words).most_common(1)[0]
        return top_count / len(words) > 0.55

    def _stamp_chunks(self, chunks: list, mute_ranges: list, samplerate: int):
        """Stamp ring buffer chunks with mute actions."""
        updates = {}
        chunk_start = 0
        for chunk in chunks:
            chunk_len = len(chunk.samples)
            chunk_ranges = []
            for (start_s, end_s) in mute_ranges:
                start_sample = int(start_s * samplerate)
                end_sample   = int(end_s   * samplerate)
                local_start  = max(0, start_sample - chunk_start)
                local_end    = min(chunk_len, end_sample - chunk_start)
                if local_start < local_end:
                    chunk_ranges.append((local_start, local_end))
            if chunk_ranges:
                updates[chunk.timestamp] = ("mute", chunk_ranges)
            chunk_start += chunk_len
        if updates:
            self.buf.update_actions(updates)

    def _get_mute_ranges(self, whisper_result: dict,
                         hits: list) -> list[tuple[float, float]]:
        """Extract per-word mute ranges from Whisper timestamps."""
        import re
        hit_words = {re.sub(r"[^\w]", "", w.lower()) for w, _ in hits}
        ranges = []
        try:
            for segment in whisper_result.get("segments", []):
                for word_info in segment.get("words", []):
                    word = re.sub(r"[^\w]", "", word_info.get("word", "").lower())
                    if word in hit_words:
                        start = max(0.0, word_info["start"] - MUTE_MARGIN)
                        end   = word_info["end"] + MUTE_MARGIN
                        ranges.append((start, end))
                        logger.debug(f"  mute: {start:.2f}s → {end:.2f}s ({word!r})")
        except Exception as e:
            logger.warning(f"Word timestamp extraction failed: {e}")
            ranges = [(CHUNK_SECONDS/2 - 1.0, CHUNK_SECONDS/2 + 1.0)]
        return ranges

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        try:
            from scipy.signal import resample_poly
            from math import gcd
            d = gcd(orig_sr, target_sr)
            return resample_poly(audio, target_sr//d, orig_sr//d).astype(np.float32)
        except ImportError:
            ratio = target_sr / orig_sr
            idx = np.linspace(0, len(audio)-1, int(len(audio)*ratio))
            return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)

    def _log_detection(self, text: str, severity: str, hits: list):
        import json
        entry = {"timestamp": time.time(), "text": text,
                 "severity": severity, "hits": hits, "reviewed": False}
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Could not write detection log: {e}")

    def stop(self):
        self._stop_event.set()

    def reload_classifier(self):
        self._classifier.reload()
        logger.info("AudioAnalysis: classifier reloaded")

    @property
    def stats(self) -> dict:
        return {
            "ready": self._ready,
            "chunks_analyzed": self._chunks_analyzed,
            "detections": self._detections,
            "model": self.model_size,
        }
