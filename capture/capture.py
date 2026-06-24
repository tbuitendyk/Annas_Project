"""
capture/capture.py — Screen and audio capture threads.

VideoCapture: captures either a full monitor or a specific window by title.
              Uses dxcam on Windows, mss elsewhere.
AudioCapture: captures system audio via sounddevice loopback.

Both respect a shared _pause_event — when paused, capture stops pushing
to the buffer (no data accumulates) while the output holds its last frame.
"""

import sys
import time
import threading
import numpy as np
import cv2
from loguru import logger

from buffer.ring_buffer import VideoRingBuffer, AudioRingBuffer
from config import CaptureConfig


# ── Window discovery helpers ───────────────────────────────────────────────

def list_capturable_windows() -> list[dict]:
    """
    Return a list of visible windows that can be captured.
    Each entry: {title, hwnd (Windows only)}
    Works on Windows via win32gui, falls back to empty list on Linux.
    """
    windows = []
    if sys.platform == "win32":
        try:
            import win32gui
            def _cb(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd).strip()
                    if title:
                        windows.append({"title": title, "hwnd": hwnd})
            win32gui.EnumWindows(_cb, None)
        except ImportError:
            logger.warning("pywin32 not installed — window list unavailable. "
                           "Run: pip install pywin32")
    return windows


def get_window_rect(title: str) -> tuple[int, int, int, int] | None:
    """
    Return (left, top, width, height) of the first window whose title
    contains `title` (case-insensitive). Returns None if not found.
    Windows only — returns None on Linux.
    """
    if sys.platform != "win32":
        return None
    try:
        import win32gui
        result = None
        def _cb(hwnd, _):
            nonlocal result
            if result:
                return
            if win32gui.IsWindowVisible(hwnd):
                t = win32gui.GetWindowText(hwnd)
                if title.lower() in t.lower():
                    rect = win32gui.GetWindowRect(hwnd)
                    l, t2, r, b = rect
                    if r - l > 10 and b - t2 > 10:
                        result = (l, t2, r - l, b - t2)
        win32gui.EnumWindows(_cb, None)
        return result
    except ImportError:
        return None


# ── Capture exclusion (keep our own windows out of the captured image) ──────

# WDA_EXCLUDEFROMCAPTURE (Windows 10 2004+): the window is still drawn on the
# physical monitor but is omitted from screen-capture APIs — including the
# DXGI Desktop Duplication path that dxcam uses, and Windows Graphics Capture.
# This is what lets CleanStream's View window sit on top of the media on the
# SAME monitor without the capture loop grabbing its own (already-cleaned)
# output, which would otherwise produce an infinite-mirror feedback loop.
WDA_NONE               = 0x00000000
WDA_MONITOR            = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011


def exclude_from_capture(hwnd: int) -> bool:
    """
    Ask Windows to keep window `hwnd` out of screen captures while still
    drawing it normally on the physical display. Returns True on success.

    No-op (returns False) on non-Windows platforms. On Linux there is no
    per-window capture-exclusion API for mss, so callers fall back to
    region-subtraction (see VideoCapture's exclusion_provider).
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        import ctypes.wintypes
        user32 = ctypes.windll.user32

        # HWNDs are pointer-sized — pin the signatures so 64-bit handles
        # aren't silently truncated to a 32-bit int by ctypes' defaults.
        user32.GetAncestor.restype = ctypes.wintypes.HWND
        user32.GetAncestor.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.UINT]
        user32.SetWindowDisplayAffinity.restype = ctypes.wintypes.BOOL
        user32.SetWindowDisplayAffinity.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.DWORD]

        # Display affinity can only be set on a top-level window. Tk's
        # winfo_id() may hand back a child HWND, so walk up to the root.
        GA_ROOT = 2
        root = user32.GetAncestor(hwnd, GA_ROOT) or hwnd

        if user32.SetWindowDisplayAffinity(root, WDA_EXCLUDEFROMCAPTURE):
            logger.info("Capture exclusion enabled (WDA_EXCLUDEFROMCAPTURE)")
            return True

        # Pre-2004 Windows: WDA_MONITOR at least stops the mirror (the window
        # shows as black in captures instead of being omitted cleanly).
        if user32.SetWindowDisplayAffinity(root, WDA_MONITOR):
            logger.warning("Capture exclusion: WDA_EXCLUDEFROMCAPTURE unsupported, "
                           "using WDA_MONITOR fallback (captures see black here)")
            return True

        logger.warning("Capture exclusion failed: SetWindowDisplayAffinity returned 0")
        return False
    except Exception as e:
        logger.warning(f"Capture exclusion unavailable: {e}")
        return False


# ── Video capture ──────────────────────────────────────────────────────────

class VideoCapture(threading.Thread):
    """
    Captures screen frames and pushes them to the video ring buffer.

    Capture source (set before starting):
      window_title = ""     → full monitor capture (default)
      window_title = "Brave"→ capture only that window by title substring

    Pause behaviour: when paused, capture loop idles — no frames pushed,
    buffer does not grow.
    """

    def __init__(self, cfg: CaptureConfig, video_buf: VideoRingBuffer,
                 pause_event: threading.Event | None = None,
                 exclusion_provider=None):
        super().__init__(daemon=True, name="VideoCapture")
        self.cfg = cfg
        self.buf = video_buf
        self._pause_event = pause_event  # set = paused
        self._stop_event = threading.Event()
        self._frame_count = 0
        self._cam = None
        # Returns the screen rect (left, top, w, h) of a window to blank out
        # of mss captures, or None. Used on Linux where there's no OS-level
        # capture-exclusion API — on Windows WDA_EXCLUDEFROMCAPTURE handles it
        # and this stays a no-op. Must be callable from this thread without
        # touching Tk objects (the UI publishes a plain tuple each frame).
        self._exclusion_provider = exclusion_provider

    def run(self):
        window = self.cfg.window_title.strip()

        if window:
            logger.info(f"VideoCapture: window mode — looking for {window!r}")
            self._run_window(window)
        else:
            if sys.platform == "win32":
                if self._init_dxcam():
                    self._run_dxcam()
                    return
            self._run_mss()

    # ── dxcam (full monitor, Windows) ──────────────────────────────────────

    def _init_dxcam(self) -> bool:
        try:
            import dxcam
            self._cam = dxcam.create(output_color="BGR")
            self._cam.start(target_fps=self.cfg.fps)
            logger.info("VideoCapture: using dxcam (DirectX fast path)")
            return True
        except Exception as e:
            logger.warning(f"dxcam unavailable, falling back to mss: {e}")
            return False

    def _run_dxcam(self):
        while not self._stop_event.is_set():
            if self._pause_event and self._pause_event.is_set():
                time.sleep(0.05)
                continue
            frame = self._cam.get_latest_frame()
            if frame is not None:
                frame = self._resize(frame)
                self.buf.push(frame)
                self._frame_count += 1
            else:
                time.sleep(0.001)

    # ── mss (full monitor, cross-platform) ────────────────────────────────

    def _run_mss(self):
        import mss
        target_interval = 1.0 / self.cfg.fps
        with mss.MSS() as sct:
            monitor = sct.monitors[self.cfg.monitor_index + 1]
            logger.info(f"VideoCapture: mss monitor {self.cfg.monitor_index} — {monitor}")
            while not self._stop_event.is_set():
                if self._pause_event and self._pause_event.is_set():
                    time.sleep(0.05)
                    continue
                t0 = time.monotonic()
                img = sct.grab(monitor)
                frame = np.ascontiguousarray(np.array(img)[:, :, :3])
                frame = self._mask_exclusion(frame, monitor["left"], monitor["top"])
                frame = self._resize(frame)
                self.buf.push(frame)
                self._frame_count += 1
                elapsed = time.monotonic() - t0
                sleep = target_interval - elapsed
                if sleep > 0:
                    time.sleep(sleep)

    # ── Window capture ────────────────────────────────────────────────────

    def _run_window(self, title: str):
        """
        Capture a specific window by tracking its on-screen rectangle.

        Both dxcam (Windows) and mss (Linux) read the *composited* desktop,
        so a region grab only crops to the window's bounds — it does NOT hide
        other windows stacked on top of it. Keeping CleanStream's own View
        window out of the capture is handled separately: by display affinity
        on Windows (capture.exclude_from_capture) and by region-subtraction on
        Linux (the exclusion_provider mask).
        """
        if sys.platform == "win32":
            self._run_window_dxcam(title)
        else:
            self._run_window_mss(title)

    def _run_window_dxcam(self, title: str):
        """
        Windows: capture the target window's region via dxcam, following it
        as it moves or resizes. (Self-capture of the View window is prevented
        by WDA_EXCLUDEFROMCAPTURE, not by region cropping.)
        """
        try:
            import dxcam
        except ImportError:
            logger.warning("dxcam not available for window capture, falling back to mss")
            self._run_window_mss(title)
            return

        target_interval = 1.0 / self.cfg.fps
        cam = None
        warned_missing = False

        while not self._stop_event.is_set():
            if self._pause_event and self._pause_event.is_set():
                time.sleep(0.05)
                continue

            t0 = time.monotonic()
            rect = get_window_rect(title)

            if rect is None:
                if not warned_missing:
                    logger.warning(f"VideoCapture: window {title!r} not found — "
                                   "waiting...")
                    warned_missing = True
                if cam is not None:
                    try:
                        cam.stop()
                    except Exception:
                        pass
                    cam = None
                time.sleep(0.5)
                continue

            warned_missing = False
            l, t, w, h = rect
            region = (l, t, l + w, t + h)  # dxcam uses (left, top, right, bottom)

            # Recreate camera if region changed (window moved/resized)
            if cam is None:
                try:
                    cam = dxcam.create(output_color="BGR", region=region)
                    cam.start(target_fps=self.cfg.fps)
                    logger.info(f"VideoCapture: dxcam window region {region}")
                except Exception as e:
                    logger.warning(f"VideoCapture: dxcam region failed: {e}")
                    cam = None
                    time.sleep(0.1)
                    continue

            try:
                frame = cam.get_latest_frame()
                if frame is not None:
                    frame = self._resize(frame)
                    self.buf.push(frame)
                    self._frame_count += 1
            except Exception as e:
                logger.warning(f"VideoCapture: dxcam frame error: {e}")
                try:
                    cam.stop()
                except Exception:
                    pass
                cam = None

            elapsed = time.monotonic() - t0
            sleep = target_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

        if cam is not None:
            try:
                cam.stop()
            except Exception:
                pass

    def _run_window_mss(self, title: str):
        """
        Linux fallback: mss region grab. mss reads the composited display, so
        windows stacked on top WILL appear in the capture — the View window is
        kept out via region-subtraction (_mask_exclusion). For best results
        keep any other overlapping windows off the captured area.
        """
        import mss
        target_interval = 1.0 / self.cfg.fps
        warned_missing = False

        with mss.MSS() as sct:
            while not self._stop_event.is_set():
                if self._pause_event and self._pause_event.is_set():
                    time.sleep(0.05)
                    continue

                t0 = time.monotonic()
                rect = get_window_rect(title)

                if rect is None:
                    if not warned_missing:
                        logger.warning(f"VideoCapture: window {title!r} not found")
                        warned_missing = True
                    monitor = sct.monitors[self.cfg.monitor_index + 1]
                else:
                    warned_missing = False
                    l, t, w, h = rect
                    monitor = {"left": l, "top": t, "width": w, "height": h}

                try:
                    img = sct.grab(monitor)
                    frame = np.ascontiguousarray(np.array(img)[:, :, :3])
                    frame = self._mask_exclusion(frame, monitor["left"], monitor["top"])
                    frame = self._resize(frame)
                    self.buf.push(frame)
                    self._frame_count += 1
                except Exception as e:
                    logger.warning(f"VideoCapture: grab failed: {e}")

                elapsed = time.monotonic() - t0
                sleep = target_interval - elapsed
                if sleep > 0:
                    time.sleep(sleep)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _mask_exclusion(self, frame: np.ndarray, origin_x: int, origin_y: int) -> np.ndarray:
        """
        Blank (paint black) the excluded window's rectangle within a raw
        grabbed frame, given the frame's top-left in screen coordinates.

        This is the Linux/mss fallback for keeping the View window out of the
        capture: mss reads the composited desktop, so wherever the View window
        overlaps the captured area its pixels are already our own output. We
        can't recover the media behind it there, but blanking the region stops
        the infinite-mirror feedback. On Windows this is a no-op because the
        provider is left unset (WDA_EXCLUDEFROMCAPTURE removes the window for
        us before compositing).
        """
        if self._exclusion_provider is None:
            return frame
        rect = self._exclusion_provider()
        if not rect:
            return frame
        rl, rt, rw, rh = rect
        fh, fw = frame.shape[:2]
        x0 = max(0, rl - origin_x)
        y0 = max(0, rt - origin_y)
        x1 = min(fw, rl + rw - origin_x)
        y1 = min(fh, rt + rh - origin_y)
        if x1 > x0 and y1 > y0:
            frame[y0:y1, x0:x1] = 0
        return frame

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = self.cfg.frame_height, self.cfg.frame_width
        fh, fw = frame.shape[:2]
        if fh == h and fw == w:
            return frame
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

    def stop(self):
        self._stop_event.set()
        if self._cam is not None:
            try:
                self._cam.stop()
            except Exception:
                pass


# ── Audio capture ──────────────────────────────────────────────────────────

class AudioCapture(threading.Thread):
    """
    Captures audio from a loopback / virtual cable device.
    When paused, the callback still fires but discards samples so the
    audio device stream stays alive and no gap/pop occurs on resume.
    """

    CHUNK_DURATION = 0.1

    @staticmethod
    def _detect_cable_device(sd) -> int | None:
        devices = sd.query_devices()
        cable_48k, cable_any = [], []
        for i, d in enumerate(devices):
            if d["max_input_channels"] < 1:
                continue
            name = d["name"].lower()
            if "cable output" in name and "vb-audio" in name:
                if int(d["default_samplerate"]) == 48000:
                    cable_48k.append((i, d["name"]))
                else:
                    cable_any.append((i, d["name"]))
        if cable_48k:
            idx, name = cable_48k[0]
            logger.info(f"AudioCapture: auto-detected CABLE Output (48kHz WASAPI): [{idx}] {name!r}")
            return idx
        if cable_any:
            idx, name = cable_any[0]
            logger.info(f"AudioCapture: auto-detected CABLE Output: [{idx}] {name!r}")
            return idx
        logger.warning("AudioCapture: no VB-Audio CABLE Output found — using system default input")
        return None

    def __init__(self, cfg: CaptureConfig, audio_buf: AudioRingBuffer,
                 pause_event: threading.Event | None = None):
        super().__init__(daemon=True, name="AudioCapture")
        self.cfg = cfg
        self.buf = audio_buf
        self._pause_event = pause_event
        self._stop_event = threading.Event()
        self._chunk_count = 0

    def run(self):
        import sounddevice as sd

        raw = self.cfg.audio_device_in.strip()
        if raw.isdigit():
            device = int(raw)
        elif raw:
            device = raw
        else:
            device = self._detect_cable_device(sd)

        device_info = sd.query_devices(device)
        samplerate = int(device_info["default_samplerate"])
        channels = min(int(device_info["max_input_channels"]), self.cfg.audio_channels)
        blocksize = int(samplerate * self.CHUNK_DURATION)

        logger.info(f"AudioCapture: device={device!r} name={device_info['name']!r} "
                    f"native_sr={samplerate} ch={channels}")

        def callback(indata: np.ndarray, frames: int, time_info, status):
            if status:
                logger.warning(f"AudioCapture status: {status}")
            # When paused, discard audio so buffer doesn't grow
            if self._pause_event and self._pause_event.is_set():
                return
            self.buf.push(indata.copy(), samplerate)
            self._chunk_count += 1

        try:
            with sd.InputStream(
                device=device,
                channels=channels,
                samplerate=samplerate,
                blocksize=blocksize,
                dtype="float32",
                callback=callback,
            ):
                logger.info("AudioCapture: stream open, capturing...")
                self._stop_event.wait()
        except Exception as e:
            logger.error(f"AudioCapture failed: {e}")
            logger.info("Tip: check SETUP.md for virtual audio cable instructions")

    def stop(self):
        self._stop_event.set()

    @staticmethod
    def list_devices() -> list[dict]:
        import sounddevice as sd
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "inputs": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
