"""
Quick diagnostic for the video content-detection backends.

Run from the project root:   python check_vision.py

It imports each dependency directly (so you see the REAL error if one is
missing or broken), reports GPU availability, then actually loads the hybrid
ContentDetector and prints which backends came up.
"""

import sys

print("Python:", sys.version.split()[0], "(", sys.executable, ")\n")

print("Imports:")
for mod in ["numpy", "cv2", "PIL", "torch", "open_clip", "nudenet"]:
    try:
        m = __import__(mod)
        print(f"  [ok]   {mod:12s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  [FAIL] {mod:12s} {type(e).__name__}: {e}")

try:
    import torch
    print("\n  torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  GPU:", torch.cuda.get_device_name(0))
except Exception:
    pass

print("\nLoading ContentDetector (this downloads weights on first run)…")
try:
    from models.vision.detector import ContentDetector
    d = ContentDetector(device="auto")
    d.load()
    print(f"\nResult: ready={d.ready}  device={d.device}  "
          f"NudeNet={'yes' if d._nude is not None else 'NO'}  "
          f"CLIP={'yes' if d._clip is not None else 'NO'}")
    if not d.ready:
        print("\n-> No backend loaded. The warnings above this line (and any "
              "[FAIL] imports) show why.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"\nContentDetector failed to construct: {type(e).__name__}: {e}")
