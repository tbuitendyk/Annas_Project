"""
pipeline.py — CleanStream pipeline orchestrator.

Adds:
  - pause() / resume() with optional OS media key injection
  - window_title support for capture source selection
  - shared _pause_event threading.Event passed to all threads
"""

import threading
from loguru import logger

from config import AppConfig
from buffer.ring_buffer import VideoRingBuffer, AudioRingBuffer
from capture.capture import VideoCapture, AudioCapture
from output.output import VideoOutput, AudioOutput

DEFAULT_AUDIO_SR = 48000


class Pipeline:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._threads: list[threading.Thread] = []
        self._video_capture: VideoCapture | None = None
        self._audio_capture: AudioCapture | None = None
        self._video_output: VideoOutput | None = None
        self._audio_output: AudioOutput | None = None
        self._video_buf: VideoRingBuffer | None = None
        self._audio_buf: AudioRingBuffer | None = None
        self._audio_analysis = None
        self._frames_out = 0

        # Shared pause state — set = paused
        self._pause_event = threading.Event()

        # Screen rect (left, top, w, h) of a window to keep out of the capture
        # (the View window), or None. The UI publishes this each frame and the
        # capture thread reads it. Single-attribute read/write is atomic in
        # CPython, so no lock is needed for one tuple/None value.
        self._exclusion_rect = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self, preview_callback=None):
        logger.info("Pipeline starting...")
        c = self.cfg.capture
        b = self.cfg.buffer

        self._video_buf = VideoRingBuffer(
            delay_seconds=b.delay_seconds,
            max_seconds=b.max_seconds,
            fps=c.fps,
        )
        self._audio_buf = AudioRingBuffer(
            delay_seconds=b.delay_seconds,
            max_seconds=b.max_seconds,
            samplerate=DEFAULT_AUDIO_SR,
        )

        self._video_capture = VideoCapture(
            c, self._video_buf, self._pause_event,
            exclusion_provider=lambda: self._exclusion_rect,
        )
        self._audio_capture = AudioCapture(c, self._audio_buf, self._pause_event)
        self._video_output  = VideoOutput(
            c, self.cfg.output, self._video_buf,
            preview_callback=self._wrap_preview(preview_callback),
            pause_event=self._pause_event,
        )
        self._audio_output = AudioOutput(c, self._audio_buf, self._pause_event)

        self._threads = [
            self._video_capture,
            self._audio_capture,
            self._video_output,
            self._audio_output,
        ]

        for t in self._threads:
            t.start()
            logger.info(f"  started {t.name}")

        logger.info("Pipeline running")

    def _wrap_preview(self, cb):
        if cb is None:
            return None
        def wrapper(frame):
            self._frames_out += 1
            cb(frame)
        return wrapper

    def stop(self):
        # Resume first so threads don't block on pause check
        self._pause_event.clear()
        logger.info("Pipeline stopping...")
        if self._audio_analysis is not None:
            self._audio_analysis.stop()
        for t in [self._video_capture, self._audio_capture,
                  self._video_output, self._audio_output]:
            if t is not None:
                t.stop()
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads.clear()
        logger.info("Pipeline stopped")

    # ── Pause / Resume ─────────────────────────────────────────────────────

    @property
    def paused(self) -> bool:
        return self._pause_event.is_set()

    def pause(self):
        """
        Pause output and capture. Tries to also pause the source app
        via OS media key (Windows: VK_MEDIA_PLAY_PAUSE). If media key
        injection fails or is not available, pauses output only.
        """
        if self._pause_event.is_set():
            return
        self._pause_event.set()
        logger.info("Pipeline paused")
        self._send_media_key_pause()

    def resume(self):
        """Resume output and capture, and send play key to source app."""
        if not self._pause_event.is_set():
            return
        self._pause_event.clear()
        logger.info("Pipeline resumed")
        self._send_media_key_play()

    def toggle_pause(self):
        if self.paused:
            self.resume()
        else:
            self.pause()

    def _send_media_key_pause(self):
        """Send OS media pause key to try to pause the source app."""
        self._send_media_key(0xB3)  # VK_MEDIA_PLAY_PAUSE

    def _send_media_key_play(self):
        """Send OS media play key to resume the source app."""
        self._send_media_key(0xB3)  # Same key toggles play/pause

    def _send_media_key(self, vk: int):
        """
        Inject a virtual key press using Windows SendInput.
        Works for most media players and browser tabs. Fails silently
        on Linux or if pywin32/ctypes is unavailable.
        """
        try:
            import ctypes
            import ctypes.wintypes

            KEYEVENTF_KEYUP = 0x0002
            INPUT_KEYBOARD  = 1

            class KEYBDINPUT(ctypes.Structure):
                _fields_ = [
                    ("wVk",         ctypes.wintypes.WORD),
                    ("wScan",       ctypes.wintypes.WORD),
                    ("dwFlags",     ctypes.wintypes.DWORD),
                    ("time",        ctypes.wintypes.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
                ]

            class INPUT(ctypes.Structure):
                class _INPUT(ctypes.Union):
                    _fields_ = [("ki", KEYBDINPUT)]
                _anonymous_ = ("_input",)
                _fields_ = [("type", ctypes.wintypes.DWORD), ("_input", _INPUT)]

            def make_input(vk, flags):
                i = INPUT()
                i.type = INPUT_KEYBOARD
                i.ki.wVk = vk
                i.ki.dwFlags = flags
                return i

            inputs = (INPUT * 2)(
                make_input(vk, 0),
                make_input(vk, KEYEVENTF_KEYUP),
            )
            ctypes.windll.user32.SendInput(2, inputs, ctypes.sizeof(INPUT))
            logger.debug(f"Media key sent: 0x{vk:02X}")
        except Exception as e:
            logger.debug(f"Media key injection unavailable: {e}")

    # ── Window title (capture source) ─────────────────────────────────────

    def set_capture_window(self, title: str):
        """
        Switch capture source to a specific window by title substring.
        Pass "" to revert to full monitor capture.
        Takes effect next time pipeline is started.
        """
        self.cfg.capture.window_title = title
        display = repr(title) if title else "'full monitor'"
        logger.info(f"Capture source set to: {display}")

    # ── Capture exclusion (View window) ────────────────────────────────────

    def set_exclusion_rect(self, rect: tuple[int, int, int, int] | None):
        """
        Publish the screen rect (left, top, width, height) of a window that
        should be blanked out of mss captures, or None to clear it. Used by
        the Linux/mss fallback for keeping the View window from capturing
        itself; on Windows display affinity handles exclusion and this is
        harmless. Safe to call from the UI thread.
        """
        self._exclusion_rect = rect

    # ── Stats ──────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        audio_ai = {}
        if self._audio_analysis is not None:
            audio_ai = self._audio_analysis.stats
        return {
            "video_frames":  self._video_buf.stats["buffered_frames"] if self._video_buf else 0,
            "audio_seconds": self._audio_buf.buffered_seconds if self._audio_buf else 0.0,
            "frames_out":    self._frames_out,
            "audio_ai":      audio_ai,
            "paused":        self.paused,
            "source_blank":  self._video_capture.source_blank if self._video_capture else False,
        }

    # ── AI hooks ───────────────────────────────────────────────────────────

    def attach_audio_model(self, whisper_model_size: str = "small"):
        from models.audio.audio_pipeline import AudioAnalysisPipeline
        if self._audio_buf is None:
            logger.error("Pipeline not started — call start() first")
            return
        self._audio_analysis = AudioAnalysisPipeline(
            self._audio_buf,
            model_size=whisper_model_size,
        )
        self._audio_analysis.start()
        self._threads.append(self._audio_analysis)
        logger.info(f"Audio model attached: Whisper {whisper_model_size} + classifier")

    def attach_vision_model(self, model):
        logger.info(f"Vision model attached: {model}")
