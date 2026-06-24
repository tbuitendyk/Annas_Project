"""
main.py — CleanStream entry point.

Run with: python main.py

On first launch, walks you through device selection.
After that, reads cleanstream.toml for settings.
"""

import sys
from loguru import logger
from config import load_config, save_config
from pipeline import Pipeline
from ui.control_panel import ControlPanel


def first_run_setup(cfg):
    """Simple CLI setup wizard for first launch."""
    from capture.capture import AudioCapture
    import mss

    print("\n=== CleanStream — First Run Setup ===\n")

    # Monitor selection
    with mss.mss() as sct:
        monitors = sct.monitors[1:]  # skip combined monitor at index 0
        print("Available monitors:")
        for i, m in enumerate(monitors):
            print(f"  [{i}] {m['width']}x{m['height']} at ({m['left']},{m['top']})")
    choice = input(f"\nWhich monitor to capture? [0]: ").strip() or "0"
    cfg.capture.monitor_index = int(choice)

    # Audio device selection
    devices = AudioCapture.list_devices()
    if devices:
        print("\nAvailable audio input devices:")
        for d in devices:
            print(f"  [{d['index']}] {d['name']}")
        print("\nEnter the name (or part of it) of your virtual audio cable:")
        print("  Windows: 'CABLE Output'")
        print("  Linux:   'cleanstream_in.monitor' or similar")
        name = input("Device name (leave blank for system default): ").strip()
        cfg.capture.audio_device_in = name
    else:
        print("No audio input devices found — audio will be disabled")

    # Buffer delay
    delay = input("\nBuffer delay in seconds [4.0]: ").strip() or "4.0"
    cfg.buffer.delay_seconds = float(delay)

    save_config(cfg)
    print("\nSetup saved to cleanstream.toml — you can edit this file anytime.\n")


def main():
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}", level="DEBUG")
    try:
        logger.add("cleanstream.log", rotation="10 MB", retention="7 days", level="DEBUG")
    except Exception as e:
        print(f"Warning: could not open log file: {e}")

    cfg = load_config()

    # Run setup wizard if no config exists
    from pathlib import Path
    if not Path("cleanstream.toml").exists():
        first_run_setup(cfg)

    pipeline = Pipeline(cfg)
    panel = ControlPanel(pipeline)

    logger.info("CleanStream ready — pass-through skeleton v0.1")
    logger.info("Next: attach audio model (Phase 2) and vision model (Phase 3)")

    panel.run()


if __name__ == "__main__":
    main()
