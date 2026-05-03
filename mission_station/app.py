from __future__ import annotations

from collections import Counter, defaultdict
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from sax_station.detector import Detection, YoloDetector
from sax_station.drone import DroneState, assisted_search_sequence, command
from sax_station.events import EventStore
from sax_station.profiles import DEFAULT_PROFILE_NAME, MISSION_PROFILES, MissionProfile, get_profile
from sax_station.sitrep import generate_sitrep
from sax_station.speech import SpeechRecorder, TranscriptionUnavailable


APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def init_state() -> None:
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("last_logged_at", {})
    st.session_state.setdefault("camera_source", "0")
    st.session_state.setdefault("latest_sitrep", "")
    st.session_state.setdefault("mission_profile", DEFAULT_PROFILE_NAME)
    st.session_state.setdefault("drone_state", DroneState.DISARMED.value)


def parse_source(raw: str) -> int | str:
    value = raw.strip()
    return int(value) if value.isdigit() else value


def should_log(
    detection: Detection,
    profile: MissionProfile,
    cooldown_seconds: float = 3.0,
) -> bool:
    key = f"{profile.code}:{detection.label}"
    now = time.time()
    last_logged_at = st.session_state.last_logged_at.get(key, 0.0)
    if now - last_logged_at < cooldown_seconds:
        return False
    st.session_state.last_logged_at[key] = now
    return True


def render_timeline(store: EventStore) -> None:
    st.subheader("Mission Timeline")
    events = store.recent(limit=30)
    if not events:
        st.caption("No events yet.")
        return

    for event in events:
        st.markdown(
            f"**{event['created_at']}** · `{event['kind']}` · "
            f"{event['summary']}"
        )


def render_sitrep(store: EventStore, profile: MissionProfile) -> None:
    st.subheader("SITREP")
    event_limit = st.slider("Events to summarize", 5, 100, 30, 5)
    if st.button("Generate SITREP", width="stretch"):
        st.session_state.latest_sitrep = generate_sitrep(
            store.recent(limit=event_limit),
            profile,
        )

    if st.session_state.latest_sitrep:
        st.text_area(
            "Current SITREP",
            value=st.session_state.latest_sitrep,
            height=220,
        )
    else:
        st.caption("Generate a SITREP after capturing detections or notes.")


def render_drone_controls(store: EventStore) -> None:
    st.subheader("Drone Control")
    current_state = DroneState(st.session_state.drone_state)
    st.metric("Simulation state", current_state.value)

    def run_command(action: str) -> None:
        result = command(DroneState(st.session_state.drone_state), action)
        st.session_state.drone_state = result.state.value
        store.add(
            result.event_kind,
            result.summary,
            {
                "simulated": True,
                "action": action,
                "state": result.state.value,
            },
        )
        st.rerun()

    cols = st.columns(2)
    if cols[0].button("Arm", width="stretch"):
        run_command("arm")
    if cols[1].button("Takeoff", width="stretch"):
        run_command("takeoff")

    cols = st.columns(2)
    if cols[0].button("Scan", width="stretch"):
        run_command("scan")
    if cols[1].button("Pause", width="stretch"):
        run_command("pause")

    cols = st.columns(2)
    if cols[0].button("Land", width="stretch"):
        run_command("land")
    if cols[1].button("Emergency Land", width="stretch"):
        run_command("emergency_land")

    if st.button("Run Assisted Search", width="stretch"):
        for result in assisted_search_sequence():
            st.session_state.drone_state = result.state.value
            store.add(
                result.event_kind,
                result.summary,
                {
                    "simulated": True,
                    "action": "assisted_search",
                    "state": result.state.value,
                },
            )
        st.rerun()

    st.caption("Simulation only. Real AR.Drone commands will be wired after video/control tests.")


def log_detections(
    store: EventStore,
    detections: list[Detection],
    profile: MissionProfile,
) -> None:
    for detection in detections:
        if should_log(detection, profile):
            is_priority = profile.is_priority(detection.label)
            summary_prefix = "Priority" if is_priority else "Observed"
            store.add(
                "object_detected",
                f"{summary_prefix} {detection.label} detected ({detection.confidence:.0%})",
                {
                    **detection.as_metadata(),
                    "mission_profile": profile.name,
                    "mission_code": profile.code,
                    "priority": is_priority,
                },
            )


def render_current_objects(
    slot,
    detections: list[Detection],
    profile: MissionProfile,
) -> None:
    with slot.container():
        st.subheader("Current Objects")
        if not detections:
            st.caption("No objects above threshold.")
            return

        counts: Counter[str] = Counter()
        max_confidence: dict[str, float] = defaultdict(float)
        confidences: dict[str, list[float]] = defaultdict(list)

        for detection in detections:
            counts[detection.label] += 1
            max_confidence[detection.label] = max(
                max_confidence[detection.label],
                detection.confidence,
            )
            confidences[detection.label].append(detection.confidence)

        for label, count in counts.most_common():
            priority_marker = "priority" if profile.is_priority(label) else "observed"
            confidence_list = ", ".join(
                f"{confidence:.0%}"
                for confidence in sorted(confidences[label], reverse=True)[:5]
            )
            st.markdown(f"**{label}**")
            st.write(
                f"{count} detection{'s' if count != 1 else ''} · "
                f"best {max_confidence[label]:.0%} · {priority_marker}"
            )
            st.caption(f"confidences: {confidence_list}")


def run_browser_snapshot_mode(
    store: EventStore,
    detector: YoloDetector,
    model_name: str,
    confidence: float,
    profile: MissionProfile,
    video_col,
    detections_slot,
) -> None:
    with video_col:
        st.info("macOS grants camera access to the browser for this mode.")
        result_slot = st.empty()
        photo = st.camera_input("Capture sensor frame")

    if photo is None:
        render_current_objects(detections_slot, [], profile)
        return

    detector.load(model_name)
    bytes_data = photo.getvalue()
    image_array = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    detections = detector.detect(frame, confidence)
    log_detections(store, detections, profile)
    annotated = detector.draw(frame, detections)

    with result_slot.container():
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption="Annotated detections",
            channels="RGB",
            width="stretch",
        )

    render_current_objects(detections_slot, detections, profile)


def open_capture(source: str) -> cv2.VideoCapture:
    parsed_source = parse_source(source)
    if isinstance(parsed_source, int):
        capture = cv2.VideoCapture(parsed_source, cv2.CAP_AVFOUNDATION)
        if capture.isOpened():
            return capture
        capture.release()

    return cv2.VideoCapture(parsed_source)


def main() -> None:
    st.set_page_config(page_title="SAx", layout="wide")
    init_state()

    store = EventStore(DATA_DIR / "events.sqlite3")
    detector = YoloDetector()
    recorder = SpeechRecorder(DATA_DIR)

    st.title("SAx")

    with st.sidebar:
        st.header("Mission")
        profile_name = st.selectbox(
            "Profile",
            list(MISSION_PROFILES.keys()),
            index=list(MISSION_PROFILES.keys()).index(
                st.session_state.mission_profile
                if st.session_state.mission_profile in MISSION_PROFILES
                else DEFAULT_PROFILE_NAME
            ),
        )
        st.session_state.mission_profile = profile_name
        profile = get_profile(profile_name)
        st.caption(profile.description)
        st.caption(f"Priority labels: {', '.join(sorted(profile.priority_labels))}")

        st.header("Input")
        input_mode = st.radio(
            "Mode",
            ["Browser snapshot", "Server video source"],
            help=(
                "Browser snapshot uses browser camera permission. Server video source "
                "uses OpenCV and is better for drone/video streams."
            ),
        )
        source = st.text_input(
            "Video source",
            value=st.session_state.camera_source,
            help="Use 0 for webcam, a video path, or tcp://192.168.1.1:5555 for AR.Drone experiments.",
        )
        st.session_state.camera_source = source

        model_name = st.selectbox(
            "YOLO model",
            ["yolo11n.pt", "yolo11s.pt", "yolov8n.pt"],
            index=0,
        )
        confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
        frame_stride = st.slider("Detect every N frames", 1, 10, 3, 1)

        st.header("Controls")
        cols = st.columns(2)
        if cols[0].button("Start", width="stretch"):
            st.session_state.running = True
        if cols[1].button("Stop", width="stretch"):
            st.session_state.running = False

        if st.button("Clear timeline", width="stretch"):
            store.clear()
            st.session_state.last_logged_at = {}
            st.session_state.latest_sitrep = ""
            st.rerun()

    video_col, intel_col = st.columns([2, 1])
    with video_col:
        st.subheader("Sensor Feed")
        frame_slot = st.empty()
    detections_slot = intel_col.empty()

    with intel_col:
        render_drone_controls(store)

        st.subheader("Operator Notes")
        typed_note = st.text_area("Manual note", placeholder="Possible movement near the entrance...")
        if st.button("Add note", width="stretch", disabled=not typed_note.strip()):
            store.add("operator_note", typed_note.strip(), {})
            st.rerun()

        record_seconds = st.slider("Record seconds", 2, 15, 5, 1)
        if st.button("Record voice note", width="stretch"):
            try:
                audio_path = recorder.record(record_seconds)
                transcript = recorder.transcribe(audio_path)
                store.add("voice_note", transcript, {"audio_path": str(audio_path)})
                st.success(transcript)
            except TranscriptionUnavailable as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Could not record/transcribe audio: {exc}")

        render_sitrep(store, profile)
        render_timeline(store)

    if input_mode == "Browser snapshot":
        run_browser_snapshot_mode(
            store,
            detector,
            model_name,
            confidence,
            profile,
            video_col,
            detections_slot,
        )
        return

    if not st.session_state.running:
        frame_slot.info("Press Start to open the video source.")
        return

    detector.load(model_name)
    capture = open_capture(source)
    if not capture.isOpened():
        frame_slot.error(
            f"Could not open video source: {source}\n\n"
            "On macOS, OpenCV needs camera permission for the app that launched Python. "
            "For webcam testing, use Browser snapshot mode. For the AR.Drone, keep using "
            "Server video source with tcp://192.168.1.1:5555."
        )
        st.session_state.running = False
        return

    frame_index = 0
    latest_detections: list[Detection] = []

    while st.session_state.running:
        ok, frame = capture.read()
        if not ok:
            frame_slot.warning("No frame received from video source.")
            break

        frame_index += 1
        if frame_index % frame_stride == 0:
            latest_detections = detector.detect(frame, confidence)
            log_detections(store, latest_detections, profile)

        annotated = detector.draw(frame, latest_detections)
        frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        render_current_objects(detections_slot, latest_detections, profile)

        time.sleep(0.03)

    capture.release()
    st.session_state.running = False


if __name__ == "__main__":
    main()
