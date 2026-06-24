"""
audio_debug2.py — Tests Whisper with chunked approach to avoid skipping segments.
py -3.13 audio_debug2.py
"""
import sys
import time
import numpy as np
import sounddevice as sd
sys.path.insert(0, ".")

# Auto-detect CABLE Output
devices = sd.query_devices()
cable_device = None
for i, d in enumerate(devices):
    if "cable output" in d["name"].lower() and d["max_input_channels"] > 0:
        if int(d["default_samplerate"]) == 48000:
            cable_device = i
            break
if cable_device is None:
    for i, d in enumerate(devices):
        if "cable output" in d["name"].lower() and d["max_input_channels"] > 0:
            cable_device = i
            break

info = sd.query_devices(cable_device)
samplerate = int(info["default_samplerate"])
channels = min(int(info["max_input_channels"]), 2)
print(f"Recording from: [{cable_device}] {info['name']} @ {samplerate}Hz")
print("Recording 30 seconds...")

duration = 30
recording = sd.rec(int(duration * samplerate), samplerate=samplerate,
                   channels=channels, dtype="float32", device=cable_device)
for i in range(duration, 0, -1):
    print(f"  {i}s remaining...", end="\r")
    time.sleep(1)
sd.wait()
print("\nRecording done.")

# Convert to mono 16kHz
mono = recording.mean(axis=1) if recording.ndim > 1 else recording.flatten()
mono = mono.astype(np.float32)
from scipy.signal import resample_poly
from math import gcd
divisor = gcd(samplerate, 16000)
mono_16k = resample_poly(mono, 16000//divisor, samplerate//divisor).astype(np.float32)

import whisper
print("\nLoading Whisper 'small' model...")
model = whisper.load_model("small")

# Approach 1: Full clip, aggressive settings
print("\n--- APPROACH 1: Full clip, aggressive no_speech settings ---")
result1 = model.transcribe(
    mono_16k,
    language="en",              # force English — avoids language detection overhead
    word_timestamps=True,
    fp16=False,
    verbose=False,
    condition_on_previous_text=False,
    no_speech_threshold=0.1,   # very aggressive — transcribe almost everything
    temperature=(0.0, 0.2, 0.4, 0.6),  # retry with higher temp if low confidence
    compression_ratio_threshold=3.0,
)
print(f"TEXT: {result1['text']}")
print(f"Segments transcribed: {len(result1.get('segments', []))}")

# Approach 2: Feed in 5-second chunks independently
print("\n--- APPROACH 2: Chunked (5s each, independent) ---")
chunk_size = 5 * 16000  # 5 seconds at 16kHz
all_text = []
all_words = []
for i in range(0, len(mono_16k), chunk_size):
    chunk = mono_16k[i:i+chunk_size]
    if len(chunk) < 16000:  # skip chunks under 1s
        continue
    offset = i / 16000
    r = model.transcribe(
        chunk,
        language="en",
        word_timestamps=True,
        fp16=False,
        verbose=False,
        condition_on_previous_text=False,
        no_speech_threshold=0.1,
        temperature=(0.0, 0.2, 0.4),
    )
    text = r.get("text", "").strip()
    if text:
        all_text.append(text)
        print(f"  [{offset:.1f}s-{offset+5:.1f}s]: {text!r}")
        for seg in r.get("segments", []):
            for w in seg.get("words", []):
                all_words.append({
                    "word": w["word"],
                    "start": w["start"] + offset,
                    "end": w["end"] + offset,
                })

full_text = " ".join(all_text)
print(f"\nFULL TEXT (chunked): {full_text}")
print("\nALL WORDS WITH TIMESTAMPS:")
for w in all_words:
    print(f"  {w['start']:5.2f}s - {w['end']:5.2f}s : {w['word']!r}")

print("\nCLASSIFIER:")
from models.audio.classifier import ProfanityClassifier
clf = ProfanityClassifier()
severity, hits = clf.classify(full_text)
print(f"Severity: {severity}, Hits: {hits}")
