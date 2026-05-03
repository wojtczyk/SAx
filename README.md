# SAX Mission Station

Phase 1 Python prototype for a Mac-based edge mission station:

- Live camera/video ingest
- YOLO object detection
- Local event timeline
- Speech-to-text operator notes
- Offline SITREP generation from recent detections and notes
- Mission profiles for Search And Rescue, Detect, Track, Map, and Protect workflows
- Simulated drone control and Assisted Search mission logging
- Typed and voice command parsing for drone simulation, notes, and SITREP generation
- Markdown mission report export at `mission_station/exports/YYYY-MM-DD_HHMM_sax_report.md`
- A path to swap webcam input for Parrot AR.Drone video

## Runtime

Use Python 3.11 or 3.12. The default `python3` on this machine is currently 3.14, which is ahead of the versions most CV/ML packages support reliably.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you do not have Python 3.12 installed, install it first with Homebrew:

```bash
brew install python@3.12
```

## Run

```bash
streamlit run mission_station/app.py
```

## AR.Drone Probe

Connect the Mac to the AR.Drone Wi-Fi network, then run:

```bash
python scripts/drone_probe.py
```

If the video probe succeeds, use this source in the app with **Server video source** mode:

```text
tcp://192.168.1.1:5555
```

## macOS Camera Notes

The app has two input modes:

- **Browser snapshot**: Uses Streamlit's browser camera widget. This is the easiest way to test YOLO on a Mac because camera permission is granted to the browser.
- **Server video source**: Uses OpenCV from the Python process. Use this for video files, network streams, and AR.Drone experiments.

If `Server video source` with `0` shows:

```text
Could not open video source: 0
```

macOS probably denied camera access to the Python process. For webcam testing, switch to `Browser snapshot` mode. If you need live webcam capture through OpenCV, run Streamlit from Terminal or iTerm and grant camera permission to that app:

```bash
cd /Users/martin/Projects/hack/sax
source .venv/bin/activate
streamlit run mission_station/app.py
```

Then check **System Settings -> Privacy & Security -> Camera** and make sure the terminal app you launched from is allowed.

The app starts with your Mac webcam. Once the core loop works, connect to the AR.Drone Wi-Fi and try the drone stream source:

```text
tcp://192.168.1.1:5555
```

AR.Drone 2.0 streams video on TCP port `5555`. Some setups need a PaVE-to-H.264 unwrap step before OpenCV can decode it; the app keeps the source configurable so we can iterate without changing the UI.

## Hackathon Story

The prototype turns raw drone/camera video and operator speech into structured, local events. It is designed for edge use first, with the Heltec LoRa module kept as a later degraded-comms alert transport.
