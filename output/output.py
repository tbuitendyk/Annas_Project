"""
output/output.py — Virtual camera and audio output threads.
"""

import time
import queue
import threading
import numpy as np
import cv2
from loguru import logger

from buffer.ring_buffer import VideoRingBuffer, AudioRingBuffer
from output.processor import process_video_frame, process_audio_chunk
from config import CaptureConfig, OutputConfig

OUT_W, OUT_H = 1280, 720  # Output at 720p — 56% less data than 1080p


class VideoOutput(threading.Thread):
    def __init__(self, cfg_capture: CaptureConfig, cfg_output: OutputConfig,
                 video_buf: VideoRingBuffer, preview_callback=None,
                 pause_event: threading.Event | None = None):
        super().__init__(daemon=True, name="VideoOutput")
        self.cfg_c = cfg_capture
        self.cfg_o = cfg_output
        self.buf = video_buf
        self._preview_callback = preview_callback
        self._pause_event = pause_event
        self._stop_event = threading.Event()
        self._frame_count = 0
        self.last_frame: np.ndarray | None = None
        self._preview_q: queue.Queue = queue.Queue(maxsize=2)
        self._preview_thread = threading.Thread(
            target=self._preview_worker, daemon=True, name="PreviewWorker"
        )

    def _preview_worker(self):
        while True:
            try:
                frame = self._preview_q.get(timeout=1.0)
                if frame is None:
                    break
                if self._preview_callback:
                    self._preview_callback(frame)
            except queue.Empty:
                continue

    def _dispatch_preview(self, frame: np.ndarray):
        try:
            self._preview_q.put_nowait(frame)
        except queue.Full:
            pass

    def run(self):
        try:
            import pyvirtualcam
        except ImportError:
            logger.error("pyvirtualcam not installed. Run: pip install pyvirtualcam")
            return

        # Boost thread priority on Windows
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), 1)
            logger.info("VideoOutput: thread priority boosted")
        except Exception:
            pass

        self._preview_thread.start()

        fps = self.cfg_c.fps
        target_interval = 1.0 / fps
        logger.info(f"VideoOutput: opening virtual camera {OUT_W}x{OUT_H}@{fps}fps")

        try:
            with pyvirtualcam.Camera(width=OUT_W, height=OUT_H, fps=fps,
                                     fmt=pyvirtualcam.PixelFormat.BGR) as cam:
                logger.info(f"VideoOutput: virtual camera active — {cam.device}")
                black = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)

                while not self._stop_event.is_set():
                    t0 = time.monotonic()

                    # When paused: hold last frame, don't drain buffer
                    if self._pause_event and self._pause_event.is_set():
                        cam.send(self.last_frame if self.last_frame is not None else black)
                        elapsed = time.monotonic() - t0
                        sleep = target_interval - elapsed
                        if sleep > 0:
                            time.sleep(sleep)
                        continue

                    vf = self.buf.pop_ready()

                    if vf is not None:
                        out_frame = process_video_frame(vf)
                        # Downscale to 720p
                        if out_frame.shape[0] != OUT_H or out_frame.shape[1] != OUT_W:
                            out_frame = cv2.resize(out_frame, (OUT_W, OUT_H),
                                                   interpolation=cv2.INTER_LINEAR)
                        self.last_frame = out_frame
                        cam.send(out_frame)
                        self._frame_count += 1
                        self._dispatch_preview(out_frame)
                    else:
                        # Send last good frame to avoid flashing black
                        cam.send(self.last_frame if self.last_frame is not None else black)

                    elapsed = time.monotonic() - t0
                    sleep = target_interval - elapsed
                    if sleep > 0:
                        time.sleep(sleep)

        except Exception as e:
            logger.error(f"VideoOutput error: {e}")
            logger.info("Tip: make sure OBS or v4l2loopback virtual camera driver is installed (see SETUP.md)")
        finally:
            self._preview_q.put(None)

    def stop(self):
        self._stop_event.set()



class AudioOutput(threading.Thread):
    def __init__(self, cfg: CaptureConfig, audio_buf: AudioRingBuffer,
                 pause_event: threading.Event | None = None):
        super().__init__(daemon=True, name="AudioOutput")
        self.cfg = cfg
        self.buf = audio_buf
        self._pause_event = pause_event
        self._stop_event = threading.Event()
        self._chunk_count = 0

    @staticmethod
    def _get_output_device() -> tuple:
        """
        Auto-detect the current default output device and its native
        sample rate and channel count.
        Always uses the system default so switching earbuds/speakers
        in Windows Sound Settings is picked up automatically.
        """
        import sounddevice as sd
        device_index = sd.default.device[1]  # [0]=input, [1]=output
        info = sd.query_devices(device_index)
        samplerate = int(info["default_samplerate"])
        channels = min(int(info["max_output_channels"]), 2)  # cap at stereo
        return device_index, samplerate, channels

    def run(self):
        import sounddevice as sd
        from scipy.signal import resample_poly
        from math import gcd

        # Query default device fresh each time the pipeline starts
        device, samplerate, channels = self._get_output_device()
        logger.info(f"AudioOutput: device={device} name={sd.query_devices(device)['name']!r} "
                    f"sr={samplerate} ch={channels}")

        try:
            with sd.OutputStream(
                device=device,
                channels=channels,
                samplerate=samplerate,
                dtype="float32",
            ) as stream:
                logger.info("AudioOutput: stream open, playing...")
                while not self._stop_event.is_set():
                    # When paused: write silence, don't drain buffer
                    if self._pause_event and self._pause_event.is_set():
                        silence = np.zeros((int(samplerate * 0.02), channels), dtype=np.float32)
                        stream.write(silence)
                        continue

                    chunk = self.buf.pop_ready()
                    if chunk is not None:
                        out_samples = process_audio_chunk(chunk)

                        # Resample if capture rate differs from output device rate
                        if chunk.samplerate != samplerate:
                            divisor = gcd(chunk.samplerate, samplerate)
                            up = samplerate // divisor
                            down = chunk.samplerate // divisor
                            if out_samples.ndim == 1:
                                out_samples = resample_poly(out_samples, up, down).astype(np.float32)
                            else:
                                out_samples = np.stack(
                                    [resample_poly(out_samples[:, c], up, down)
                                     for c in range(out_samples.shape[1])], axis=1
                                ).astype(np.float32)

                        # Match channel count to output device
                        if out_samples.ndim == 1:
                            out_samples = np.stack([out_samples] * channels, axis=1)
                        elif out_samples.shape[1] < channels:
                            out_samples = np.stack([out_samples[:, 0]] * channels, axis=1)
                        elif out_samples.shape[1] > channels:
                            out_samples = out_samples[:, :channels]

                        stream.write(out_samples)
                        self._chunk_count += 1
                    else:
                        silence = np.zeros((int(samplerate * 0.005), channels), dtype=np.float32)
                        stream.write(silence)
        except Exception as e:
            print(f"AUDIO OUTPUT ERROR: {e}", flush=True)
            logger.error(f"AudioOutput error: {e}")
            logger.info("Tip: check SETUP.md for audio device setup instructions")

    def stop(self):
        self._stop_event.set()
