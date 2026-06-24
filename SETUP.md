# CleanStream — Setup Guide

## Python
Requires Python 3.10+. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Windows setup

### 1. Virtual camera (OBS Virtual Camera)
- Install OBS Studio: https://obsproject.com
- Launch OBS at least once so it registers the virtual camera driver
- You do NOT need to use OBS — just install it for the driver
- Verify: in any video call app, you should see "OBS Virtual Camera" as a camera option

### 2. Virtual audio cable (VB-Audio)
- Download and install: https://vb-audio.com/Cable/
- After install, you'll have two new devices:
  - "CABLE Input" — a virtual speaker (apps play audio here)
  - "CABLE Output" — a virtual mic (your app reads from here)
- In your browser or media player: set audio output to "CABLE Input"
- CleanStream reads from "CABLE Output", processes, and plays to your real speakers

### 3. Screen capture permissions
- No special permissions needed on Windows for screen capture
- For DRM content (Netflix): use Firefox and set in about:config:
    media.wmf.disable-d3d11-for-DXVA2 = true

---

## Linux setup

### 1. Virtual camera (v4l2loopback)
```bash
# Install
sudo apt install v4l2loopback-dkms v4l2loopback-utils   # Debian/Ubuntu
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="CleanStream" exclusive_caps=1

# Make permanent (load on boot)
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf
echo 'options v4l2loopback devices=1 video_nr=10 card_label="CleanStream" exclusive_caps=1' \
  | sudo tee /etc/modprobe.d/v4l2loopback.conf
```

### 2. Virtual audio sink (PipeWire/PulseAudio)
```bash
# PulseAudio (most distros)
pactl load-module module-null-sink sink_name=cleanstream_in sink_properties=device.description=CleanStream_Input
pactl load-module module-null-sink sink_name=cleanstream_out sink_properties=device.description=CleanStream_Output

# Set your browser/player to output to "CleanStream_Input"
# CleanStream reads from "CleanStream_Input.monitor" and writes to "CleanStream_Output"
```

### 3. Screen capture
```bash
# PipeWire screen capture (recommended - works with Wayland and X11)
sudo apt install pipewire pipewire-pulse xdg-desktop-portal-gnome
```

---

## Running CleanStream

```bash
python main.py
```

On first run, a setup wizard will guide you through selecting:
- Which screen/window to capture
- Which audio device to capture from
- Buffer delay (default: 4 seconds)
