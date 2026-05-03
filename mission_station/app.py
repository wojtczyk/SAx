from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from sax_station.ardrone import (
    CTRL_STATE_NAMES,
    DEFAULT_DRONE_HOST,
    probe_drone,
    read_navdata_snapshot,
    send_real_drone_command,
    video_source_url,
)
from sax_station.commands import ParsedCommand, parse_command
from sax_station.detector import Detection, YoloDetector
from sax_station.drone import DroneState, assisted_search_sequence, command
from sax_station.events import EventStore
from sax_station.exporter import export_mission_report
from sax_station.profiles import DEFAULT_PROFILE_NAME, MISSION_PROFILES, MissionProfile, get_profile
from sax_station.sitrep import generate_sitrep
from sax_station.speech import SpeechRecorder, TranscriptionUnavailable


APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
EXPORT_DIR = APP_ROOT / "exports"
DATA_DIR.mkdir(exist_ok=True)


def apply_compact_layout() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 3.1rem !important;
            padding-bottom: 1rem !important;
        }

        [data-testid="stSidebarContent"] {
            padding-top: 1rem !important;
        }

        h1 {
            margin-top: 0;
            margin-bottom: 0.4rem;
            line-height: 1;
        }

        h2, h3 {
            margin-top: 0.35rem;
            margin-bottom: 0.45rem;
            line-height: 1.1;
        }

        .sax-title {
            font-size: 1.65rem;
            font-weight: 750;
            line-height: 1;
            margin: 0 0 0.3rem 0;
        }

        .sax-telemetry-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem 1rem;
            margin: 0.75rem 0 0.75rem 0;
        }

        .sax-telemetry-label {
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }

        .sax-telemetry-value {
            font-size: 1.65rem;
            font-weight: 650;
            line-height: 1.05;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .sax-telemetry-wide {
            grid-column: 1 / -1;
        }

        .sax-telemetry-detail {
            grid-column: 1 / -1;
            font-size: 0.8rem;
            opacity: 0.72;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("last_logged_at", {})
    st.session_state.setdefault("camera_source", "0")
    st.session_state.setdefault("latest_sitrep", "")
    st.session_state.setdefault("mission_profile", DEFAULT_PROFILE_NAME)
    st.session_state.setdefault("drone_state", DroneState.DISARMED.value)
    st.session_state.setdefault("drone_host", DEFAULT_DRONE_HOST)
    st.session_state.setdefault("drone_command_mode", "Simulation")
    st.session_state.setdefault("last_navdata_auto_at", 0.0)


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


def render_mission_export(
    store: EventStore,
    profile: MissionProfile,
    video_source: str,
    model_name: str,
    confidence: float,
) -> None:
    st.subheader("Mission Export")
    if st.button("Export Mission Report", width="stretch"):
        path = export_mission_report(
            EXPORT_DIR,
            store.recent(limit=200),
            profile,
            st.session_state.drone_state,
            video_source,
            model_name,
            confidence,
            st.session_state.latest_sitrep,
        )
        st.session_state.latest_export = str(path)

    latest_export = st.session_state.get("latest_export")
    if latest_export:
        st.caption(f"Saved: {latest_export}")


def current_navdata_snapshot() -> dict | None:
    result = st.session_state.get("drone_probe")
    if not result:
        return None
    return result.get("navdata_snapshot") or result.get("battery")


def store_navdata_snapshot(
    snapshot,
    replace_failed: bool = False,
) -> None:
    result = st.session_state.get("drone_probe")
    if not result:
        return

    existing = current_navdata_snapshot()
    if snapshot.ok or replace_failed or not existing or not existing.get("ok"):
        st.session_state.drone_probe = {
            **result,
            "battery": snapshot.__dict__,
            "navdata_snapshot": snapshot.__dict__,
        }


def is_drone_video_source(source: str, drone_host: str) -> bool:
    value = source.strip()
    return value == video_source_url(drone_host) or (
        value.startswith("tcp://") and drone_host in value and ":5555" in value
    )


def navdata_metric_html(label: str, value: str, wide: bool = False) -> str:
    class_name = "sax-telemetry-item sax-telemetry-wide" if wide else "sax-telemetry-item"
    return (
        f'<div class="{class_name}">'
        f'<div class="sax-telemetry-label">{escape(label)}</div>'
        f'<div class="sax-telemetry-value">{escape(value)}</div>'
        "</div>"
    )


def render_navdata_snapshot(slot, snapshot: dict | None) -> None:
    if not snapshot:
        slot.empty()
        return

    if not snapshot.get("ok"):
        detail = snapshot.get("detail", "navdata unavailable")
        slot.markdown(
            (
                '<div class="sax-telemetry-grid">'
                + navdata_metric_html("Battery", "unknown")
                + navdata_metric_html("Altitude", "unknown")
                + f'<div class="sax-telemetry-detail">Navdata detail: {escape(str(detail))}</div>'
                + "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    battery = snapshot.get("battery_percent")
    battery_value = f"{battery}%" if battery is not None else "unknown"

    altitude_cm = snapshot.get("altitude_cm")
    altitude_value = f"{altitude_cm / 100:.2f} m" if altitude_cm is not None else "unknown"

    ctrl_state = snapshot.get("ctrl_state")
    state_code = ctrl_state >> 16 if ctrl_state is not None else None
    flying_state = (
        CTRL_STATE_NAMES.get(state_code, f"Unknown ({state_code})")
        if state_code is not None
        else "Unknown"
    )

    yaw = snapshot.get("psi_mdeg")
    yaw_value = f"{yaw / 1000:.1f} deg" if yaw is not None else "unknown"

    velocity = (
        snapshot.get("vx"),
        snapshot.get("vy"),
        snapshot.get("vz"),
    )
    if all(value is not None for value in velocity):
        velocity_value = (
            f"{velocity[0] / 1000:.2f}, "
            f"{velocity[1] / 1000:.2f}, "
            f"{velocity[2] / 1000:.2f} m/s"
        )
    else:
        velocity_value = "unknown"

    slot.markdown(
        (
            '<div class="sax-telemetry-grid">'
            + navdata_metric_html("Battery", battery_value)
            + navdata_metric_html("Altitude", altitude_value)
            + navdata_metric_html("Flying State", flying_state)
            + navdata_metric_html("Yaw", yaw_value)
            + navdata_metric_html("Velocity", velocity_value, wide=True)
            + "</div>"
        ),
        unsafe_allow_html=True,
    )


def maybe_auto_refresh_navdata(slot, source: str, drone_host: str) -> None:
    if not is_drone_video_source(source, drone_host):
        return

    now = time.monotonic()
    if now - st.session_state.last_navdata_auto_at < 1.0:
        return
    st.session_state.last_navdata_auto_at = now

    snapshot = read_navdata_snapshot(
        drone_host,
        timeout_seconds=0.15,
        initialize=False,
    )
    store_navdata_snapshot(snapshot)
    render_navdata_snapshot(slot, current_navdata_snapshot())


def execute_drone_action(
    store: EventStore,
    action: str,
    command_mode: str,
    drone_host: str,
) -> None:
    if command_mode == "Real AR.Drone":
        result = send_real_drone_command(drone_host, action)
        if action == "land" and result.ok:
            st.session_state.drone_state = DroneState.DISARMED.value
        if action == "emergency_land" and result.ok:
            st.session_state.drone_state = DroneState.EMERGENCY.value
        store.add(
            result.event_kind,
            result.summary,
            {
                "real": True,
                "ok": result.ok,
                "action": result.action,
                "host": drone_host,
                "detail": result.detail,
            },
        )
        return

    if action == "assisted_search":
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
        return

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


def execute_operator_command(
    store: EventStore,
    profile: MissionProfile,
    parsed: ParsedCommand,
    source: str,
    command_mode: str,
    drone_host: str,
) -> None:
    if parsed.intent == "empty":
        return

    if parsed.intent in {
        "arm",
        "takeoff",
        "scan",
        "pause",
        "land",
        "flat_trim",
        "emergency_land",
        "assisted_search",
    }:
        execute_drone_action(store, parsed.intent, command_mode, drone_host)
        store.add(
            "operator_command",
            f"{source} command executed: {parsed.intent.replace('_', ' ')}",
            {"source": source, "intent": parsed.intent},
        )
        return

    if parsed.intent == "sitrep":
        st.session_state.latest_sitrep = generate_sitrep(store.recent(limit=30), profile)
        store.add(
            "operator_command",
            f"{source} command executed: generate sitrep",
            {"source": source, "intent": parsed.intent},
        )
        return

    if parsed.intent == "note":
        if parsed.value:
            store.add(
                "operator_note",
                parsed.value,
                {"source": source, "intent": parsed.intent},
            )
        return

    store.add(
        "operator_command",
        f"{source} command not recognized: {parsed.value}",
        {"source": source, "intent": parsed.intent, "raw": parsed.value},
    )


def render_drone_controls(
    store: EventStore,
    profile: MissionProfile,
    drone_host: str,
) -> None:
    st.subheader("Drone Control")
    current_state = DroneState(st.session_state.drone_state)
    command_mode = st.radio(
        "Command mode",
        ["Simulation", "Real AR.Drone"],
        key="drone_command_mode",
        horizontal=True,
    )
    real_mode = command_mode == "Real AR.Drone"
    st.metric("Simulation state", current_state.value)
    if real_mode:
        st.warning("Real mode only enables Flat Trim, Land, and Emergency Stop.")

    def run_command(action: str) -> None:
        execute_drone_action(store, action, command_mode, drone_host)
        st.rerun()

    cols = st.columns(2)
    if cols[0].button("Arm", width="stretch", disabled=real_mode):
        run_command("arm")
    if cols[1].button("Takeoff", width="stretch", disabled=real_mode):
        run_command("takeoff")

    cols = st.columns(2)
    if cols[0].button("Scan", width="stretch", disabled=real_mode):
        run_command("scan")
    if cols[1].button("Pause", width="stretch", disabled=real_mode):
        run_command("pause")

    if st.button("Flat Trim", width="stretch"):
        run_command("flat_trim")

    cols = st.columns(2)
    if cols[0].button("Land", width="stretch"):
        run_command("land")
    if cols[1].button("Emergency Stop", width="stretch"):
        run_command("emergency_land")

    if st.button("Run Assisted Search", width="stretch", disabled=real_mode):
        execute_drone_action(store, "assisted_search", command_mode, drone_host)
        st.rerun()

    with st.form("operator_command_form", clear_on_submit=True):
        command_text = st.text_input(
            "Command",
            placeholder="arm, takeoff, scan, sitrep, note possible survivor...",
        )
        submitted = st.form_submit_button("Run Command", width="stretch")

    if submitted and command_text.strip():
        execute_operator_command(
            store,
            profile,
            parse_command(command_text),
            source="typed",
            command_mode=command_mode,
            drone_host=drone_host,
        )
        st.rerun()

    st.caption("Real takeoff and movement stay disabled until safe command tests pass.")


def render_drone_diagnostics(store: EventStore):
    st.subheader("AR.Drone Link")
    drone_host = st.text_input("Drone host", key="drone_host")
    st.caption(f"Video source: `{video_source_url(drone_host)}`")
    telemetry_slot = st.empty()

    if st.button("Probe AR.Drone", width="stretch"):
        result = probe_drone(drone_host)
        st.session_state.drone_probe = {
            "host": result["host"],
            "local_address": result["local_address"],
            "likely_on_drone_network": result["likely_on_drone_network"],
            "video_source": result["video_source"],
            "video": result["video"].__dict__,
            "navdata": result["navdata"].__dict__,
            "control": result["control"].__dict__,
            "battery": result["battery"].__dict__,
            "navdata_snapshot": result["battery"].__dict__,
            "ready_for_video": result["ready_for_video"],
            "ready_for_control": result["ready_for_control"],
        }
        store.add(
            "drone_probe",
            (
                "AR.Drone probe: "
                f"video={'ready' if result['ready_for_video'] else 'not ready'}, "
                f"control={'ready' if result['ready_for_control'] else 'not ready'}"
            ),
            st.session_state.drone_probe,
        )
        if result["ready_for_video"]:
            st.session_state.camera_source = result["video_source"]
        st.rerun()

    result = st.session_state.get("drone_probe")
    if not result:
        return telemetry_slot

    render_navdata_snapshot(telemetry_slot, current_navdata_snapshot())

    if st.button("Read Navdata", width="stretch"):
        snapshot = read_navdata_snapshot(drone_host, initialize=True)
        store_navdata_snapshot(snapshot, replace_failed=True)
        store.add(
            "drone_navdata",
            (
                f"AR.Drone navdata: battery {snapshot.battery_percent}%, altitude {snapshot.altitude_cm} cm"
                if snapshot.ok
                else f"AR.Drone navdata unavailable: {snapshot.detail}"
            ),
            snapshot.__dict__,
        )
        st.rerun()

    if st.session_state.running and is_drone_video_source(
        st.session_state.camera_source,
        drone_host,
    ):
        st.caption("Telemetry auto-refreshes while the drone video stream is running.")

    st.write(f"Local route address: `{result['local_address']}`")
    if not result["likely_on_drone_network"]:
        st.warning("Mac does not appear to be on the AR.Drone Wi-Fi network.")

    for key in ["video", "navdata", "control"]:
        probe = result[key]
        status = "OK" if probe["ok"] else "FAIL"
        st.write(f"{status} · {probe['name']} · {probe['detail']}")

    if result["ready_for_video"]:
        st.success("Video port is open. Switch to Server video source and press Start.")
    else:
        st.warning("Connect Mac Wi-Fi to the AR.Drone network, then probe again.")

    return telemetry_slot


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
    lines = ["### Current Objects"]
    if not detections:
        lines.append("No objects above threshold.")
        slot.markdown("\n\n".join(lines))
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
        lines.extend(
            [
                f"**{label}**",
                (
                    f"{count} detection{'s' if count != 1 else ''} · "
                    f"best {max_confidence[label]:.0%} · {priority_marker}"
                ),
                f"confidences: {confidence_list}",
            ]
        )

    slot.markdown("\n\n".join(lines))


def run_browser_snapshot_mode(
    store: EventStore,
    detector: YoloDetector,
    model_name: str,
    confidence: float,
    profile: MissionProfile,
    frame_slot,
    detections_slot,
) -> None:
    with frame_slot.container():
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
    apply_compact_layout()
    init_state()

    store = EventStore(DATA_DIR / "events.sqlite3")
    detector = YoloDetector()
    recorder = SpeechRecorder(DATA_DIR)

    st.markdown('<div class="sax-title">SAx</div>', unsafe_allow_html=True)

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
        detections_slot = st.empty()

    with intel_col:
        telemetry_slot = render_drone_diagnostics(store)
        render_drone_controls(store, profile, st.session_state.drone_host)

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
                execute_operator_command(
                    store,
                    profile,
                    parse_command(transcript),
                    source="voice",
                    command_mode=st.session_state.drone_command_mode,
                    drone_host=st.session_state.drone_host,
                )
                st.success(transcript)
            except TranscriptionUnavailable as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Could not record/transcribe audio: {exc}")

        render_sitrep(store, profile)
        render_mission_export(store, profile, source, model_name, confidence)
        render_timeline(store)

    if input_mode == "Browser snapshot":
        run_browser_snapshot_mode(
            store,
            detector,
            model_name,
            confidence,
            profile,
            frame_slot,
            detections_slot,
        )
        return

    if not st.session_state.running:
        frame_slot.info("Press Start to open the video source.")
        render_current_objects(detections_slot, [], profile)
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
        maybe_auto_refresh_navdata(telemetry_slot, source, st.session_state.drone_host)

        time.sleep(0.03)

    capture.release()
    st.session_state.running = False


if __name__ == "__main__":
    main()
