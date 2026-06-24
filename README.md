# CleanStream

A real-time AI-powered content filter for movies and streaming video. Sits between your media source and your screen, intercepting video and audio to detect and filter profanity and nudity before it reaches the viewer.

## How it works

CleanStream captures your screen and system audio, holds them in a 4-second ring buffer (like broadcast TV delay), runs AI analysis on the buffered content, and outputs the filtered stream to a virtual camera and speakers. The viewer watches the virtual camera output — they never see the raw feed.

```
Media source → Screen capture → Ring buffer (4s) → AI analysis → Virtual camera → Screen
Browser/DVD  → Audio loopback → Ring buffer (4s) → AI analysis → Virtual speakers → Ears
```

## Features (current — Phase 1)

- ✅ Real-time screen capture via DirectX (Windows) or mss (Linux)
- ✅ Audio capture via virtual audio cable
- ✅ 4-second A/V ring buffer with precise delay
- ✅ Virtual camera output (OBS Virtual Camera / v4l2loopback)
- ✅ Pass-through mode (no AI filtering yet — pipeline only)
- ✅ Control panel UI with live preview and stats
- ✅ View Mode window you can place on the *same* monitor as the media —
     it's excluded from the screen capture (Windows display affinity, or a
     region mask elsewhere) so it never captures its own output
- ✅ Frame action system (blur / black / mute / bleep) ready for AI

## Roadmap

- 🔲 Phase 2: Whisper STT + custom text classifier for profanity detection
- 🔲 Phase 3: MobileNetV2 fine-tuned for nudity detection
- 🔲 Phase 4: Severity router (mild/moderate/severe) + family profiles
- 🔲 Phase 5: Labeling UI for model training and correction

## Requirements

- Python 3.10+
- Windows or Linux
- OBS Studio (for virtual camera driver)
- VB-Audio Virtual Cable (Windows) or PipeWire null sink (Linux)

## Installation

```bash
pip install -r requirements.txt
```

See [SETUP.md](SETUP.md) for virtual camera and audio driver setup.

## Usage

```bash
python main.py
```

On first run, a setup wizard will ask for your monitor and audio device. Settings are saved to `cleanstream.toml`.

### Watching on the same monitor as the media (View Mode)

Open **View Mode** and drag the window onto the monitor your media is on. Two
things make this work without breaking playback:

- The View window is **excluded from the screen capture** (Windows display
  affinity, or a region mask elsewhere), so it never captures its own output.
- The window is held **very slightly translucent** (254/255), so Chromium
  browsers don't count it as an occluder and keep playing the video. Without
  this, YouTube and similar players freeze the picture (audio keeps going) the
  moment their window is fully covered.

If a browser or player *still* pauses its video when covered, disable its
window-occlusion detection. The **Troubleshooting tab** in the app has a
copy button for each browser; the same settings are:

| Browser            | Setting                                                                 |
|--------------------|-------------------------------------------------------------------------|
| Brave              | `brave://flags/#calculate-native-win-occlusion` → Disabled → Relaunch   |
| Chrome             | `chrome://flags/#calculate-native-win-occlusion` → Disabled → Relaunch  |
| Edge               | `edge://flags/#calculate-native-win-occlusion` → Disabled → Restart     |
| Opera / Vivaldi    | `opera://flags/#…` / `vivaldi://flags/#…` (same flag)                    |
| Firefox            | `about:config` → `widget.windows.window_occlusion_tracking.enabled` = `false` |

Cross-platform fallback — relaunch any Chromium browser with these flags
(works on Windows, macOS and Linux):

```bash
chrome --disable-features=CalculateNativeWinOcclusion --disable-backgrounding-occluded-windows
```

## Project structure

```
cleanstream/
├── buffer/         # Ring buffer for video frames and audio chunks
├── capture/        # Screen and audio capture threads
├── output/         # Virtual camera and speaker output threads
├── ui/             # Tkinter control panel
├── config.py       # Configuration schema and loader
├── pipeline.py     # Pipeline orchestrator
└── main.py         # Entry point
```
