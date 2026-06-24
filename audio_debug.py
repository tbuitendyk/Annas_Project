"""
audio_debug.py — Drop this in your cleanstream folder and run it
WHILE CleanStream is running to see exactly what the audio buffer is doing.

py -3.13 audio_debug.py
"""
import time
import sys
sys.path.insert(0, ".")

from config import load_config
from buffer.ring_buffer import AudioRingBuffer
from capture.capture import AudioCapture

cfg = load_config()

# Create a fresh buffer with 0 delay so we can hear immediately
buf = AudioRingBuffer(delay_seconds=0.1, max_seconds=5.0, samplerate=cfg.capture.audio_samplerate)
capture = AudioCapture(cfg.capture, buf)
capture.start()

print("Listening for 3 seconds...")
time.sleep(3)

chunks = buf.peek_seconds(3.0)
print(f"Chunks captured: {len(chunks)}")
if chunks:
    import numpy as np
    all_samples = np.concatenate([c.samples for c in chunks])
    rms = float(np.sqrt(np.mean(all_samples**2)))
    print(f"RMS level: {rms:.6f} (0 = silence, >0.001 = audio present)")
    print(f"Total samples: {len(all_samples)}")
    print(f"Duration: {len(all_samples)/cfg.capture.audio_samplerate:.2f}s")

    if rms < 0.0001:
        print("\nRESULT: Buffer is receiving SILENCE — browser not sending audio to CABLE Input")
    else:
        print("\nRESULT: Audio IS present in buffer — problem is in output routing")

    # Try playing it directly
    print("\nAttempting to play captured audio directly to default output...")
    import sounddevice as sd
    idx = sd.default.device[1]
    info = sd.query_devices(idx)
    sr = int(info["default_samplerate"])
    ch = min(int(info["max_output_channels"]), 2)
    samples = all_samples[:sr*2]  # play 2 seconds
    if samples.ndim == 1:
        samples = np.stack([samples]*ch, axis=1)
    sd.play(samples, samplerate=sr, device=idx)
    sd.wait()
    print("Done — did you hear anything?")
else:
    print("No chunks captured — AudioCapture is not receiving any audio")

capture.stop()
