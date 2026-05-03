from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from sax_station.ardrone import (
    ARDroneATClient,
    CTRL_STATE_NAMES,
    DEFAULT_DRONE_HOST,
    DRONE_STATE_EMERGENCY_MASK,
    DRONE_STATE_FLYING_MASK,
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

        h2 {
            font-size: 1.25rem !important;
        }

        h3 {
            font-size: 1.05rem !important;
        }

        .sax-title {
            font-size: 1.45rem;
            font-weight: 750;
            line-height: 1;
            margin: 0 0 0.3rem 0;
        }

        .sax-telemetry-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.45rem 0.75rem;
            margin: 0.55rem 0 0.55rem 0;
        }

        .sax-telemetry-label {
            font-size: 0.72rem;
            font-weight: 700;
            margin-bottom: 0.08rem;
            opacity: 0.78;
        }

        .sax-telemetry-value {
            font-size: 1.02rem;
            font-weight: 650;
            line-height: 1.15;
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

        .sax-status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin: 0.05rem 0 0.55rem 0;
        }

        .sax-status-pill {
            border: 1px solid rgba(128, 128, 128, 0.26);
            border-radius: 6px;
            background: rgba(128, 128, 128, 0.12);
            padding: 0.22rem 0.45rem;
            font-size: 0.78rem;
            line-height: 1.1;
        }

        .sax-status-pill strong {
            font-weight: 750;
        }

        .sax-detections {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.84rem;
            margin-top: 0.15rem;
        }

        .sax-detections th,
        .sax-detections td {
            border-bottom: 1px solid rgba(128, 128, 128, 0.22);
            padding: 0.25rem 0.35rem 0.25rem 0;
            text-align: left;
            vertical-align: middle;
        }

        .sax-detections th {
            font-size: 0.72rem;
            opacity: 0.72;
            font-weight: 750;
            text-transform: uppercase;
        }

        .sax-empty {
            font-size: 0.84rem;
            opacity: 0.72;
            margin-top: 0.15rem;
        }

        .sax-control-state {
            font-size: 0.82rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0;
        }

        .sax-control-heading {
            font-size: 1.05rem;
            font-weight: 750;
            margin: 0 !important;
            line-height: 1.1;
        }

        .sax-control-status {
            display: flex;
            align-items: baseline;
            gap: 0.45rem;
            margin: 0.25rem 0 0.45rem 0;
            font-size: 0.78rem;
            opacity: 0.82;
        }

        .sax-compact-warning {
            border-left: 3px solid #f2cc60;
            border-radius: 4px;
            background: rgba(242, 204, 96, 0.16);
            padding: 0.3rem 0.5rem;
            margin: 0.28rem 0 0.35rem 0;
            font-size: 0.78rem;
            line-height: 1.2;
        }

        .sax-compact-error {
            border-left: 3px solid #ff6b6b;
            border-radius: 4px;
            background: rgba(255, 107, 107, 0.16);
            padding: 0.3rem 0.5rem;
            margin: 0.28rem 0 0.35rem 0;
            font-size: 0.78rem;
            line-height: 1.2;
        }

        .sax-compact-hint {
            font-size: 0.78rem;
            opacity: 0.72;
            margin: 0.18rem 0;
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
    st.session_state.setdefault("drone_video_capture", None)
    st.session_state.setdefault("drone_video_source", "")
    st.session_state.setdefault("drone_video_frame_index", 0)
    st.session_state.setdefault("drone_video_missed_frames", 0)
    st.session_state.setdefault("last_drone_panel_refresh_at", 0.0)
    st.session_state.setdefault("last_hover_keepalive_at", 0.0)
    st.session_state.setdefault("last_hover_keepalive_error", "")
    st.session_state.setdefault("latest_detections", [])
    st.session_state.setdefault("webcam_photo_bytes", None)
    st.session_state.setdefault("webcam_capture_key", 0)
    if st.session_state.drone_command_mode not in {"Simulation", "Drone"}:
        st.session_state.drone_command_mode = "Drone"


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


def render_timeline(store: EventStore, show_title: bool = True) -> None:
    if show_title:
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


def render_sitrep(
    store: EventStore,
    profile: MissionProfile,
    show_title: bool = True,
) -> None:
    if show_title:
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
    show_title: bool = True,
) -> None:
    if show_title:
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


def navdata_readings(snapshot: dict | None) -> dict[str, str]:
    readings = {
        "battery": "unknown",
        "altitude": "unknown",
        "state": "unknown",
        "yaw": "unknown",
        "velocity": "unknown",
    }
    if not snapshot or not snapshot.get("ok"):
        return readings

    battery = snapshot.get("battery_percent")
    if battery is not None:
        readings["battery"] = f"{battery}%"

    altitude_cm = snapshot.get("altitude_cm")
    if altitude_cm is not None:
        readings["altitude"] = f"{altitude_cm / 100:.2f} m"

    drone_state = snapshot.get("drone_state")
    if drone_state is not None:
        if drone_state & DRONE_STATE_EMERGENCY_MASK:
            readings["state"] = "Emergency"
        elif drone_state & DRONE_STATE_FLYING_MASK:
            readings["state"] = "Flying"
        else:
            readings["state"] = "Landed"

    ctrl_state = snapshot.get("ctrl_state")
    state_code = ctrl_state >> 16 if ctrl_state is not None else None
    if state_code is not None and readings["state"] == "unknown":
        readings["state"] = CTRL_STATE_NAMES.get(state_code, f"Unknown ({state_code})")

    yaw = snapshot.get("psi_mdeg")
    if yaw is not None:
        readings["yaw"] = f"{yaw / 1000:.1f} deg"

    velocity = (
        snapshot.get("vx"),
        snapshot.get("vy"),
        snapshot.get("vz"),
    )
    if all(value is not None for value in velocity):
        readings["velocity"] = (
            f"{velocity[0] / 1000:.2f}, "
            f"{velocity[1] / 1000:.2f}, "
            f"{velocity[2] / 1000:.2f} m/s"
        )

    return readings


def drone_reports_emergency(snapshot: dict | None) -> bool:
    if not snapshot or not snapshot.get("ok"):
        return False
    drone_state = snapshot.get("drone_state")
    return bool(drone_state is not None and drone_state & DRONE_STATE_EMERGENCY_MASK)


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

    readings = navdata_readings(snapshot)

    slot.markdown(
        (
            '<div class="sax-telemetry-grid">'
            + navdata_metric_html("Battery", readings["battery"])
            + navdata_metric_html("Altitude", readings["altitude"])
            + navdata_metric_html("State", readings["state"])
            + navdata_metric_html("Yaw", readings["yaw"])
            + navdata_metric_html("Velocity", readings["velocity"], wide=True)
            + "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_status_strip(
    slot,
    source: str,
    detections: list[Detection],
) -> None:
    result = st.session_state.get("drone_probe")
    drone_status = "connected" if result and result.get("ready_for_video") else "not linked"
    video_status = "live" if st.session_state.running else "standby"
    readings = navdata_readings(current_navdata_snapshot())
    object_count = len(detections)
    object_status = f"{object_count} object{'s' if object_count != 1 else ''}"

    items = [
        ("Drone", drone_status),
        ("Battery", readings["battery"]),
        ("Altitude", readings["altitude"]),
        ("Video", video_status),
        ("Objects", object_status),
    ]
    pill_html = "".join(
        (
            '<span class="sax-status-pill">'
            f"<strong>{escape(label)}</strong> {escape(value)}"
            "</span>"
        )
        for label, value in items
    )
    slot.markdown(
        f'<div class="sax-status-strip">{pill_html}</div>',
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


@st.fragment(run_every=0.25)
def flight_keepalive_fragment(drone_host: str) -> None:
    if st.session_state.drone_command_mode != "Drone":
        return
    if st.session_state.drone_state not in {
        DroneState.AIRBORNE.value,
        DroneState.PAUSED.value,
    }:
        return
    if drone_reports_emergency(current_navdata_snapshot()):
        return

    now = time.monotonic()
    if now - st.session_state.last_hover_keepalive_at < 0.2:
        return

    st.session_state.last_hover_keepalive_at = now
    try:
        ARDroneATClient(drone_host).hover(repeat=1, interval_seconds=0.0)
        st.session_state.last_hover_keepalive_error = ""
    except OSError as exc:
        st.session_state.last_hover_keepalive_error = str(exc)


def execute_drone_action(
    store: EventStore,
    action: str,
    command_mode: str,
    drone_host: str,
) -> None:
    if command_mode == "Drone":
        result = send_real_drone_command(drone_host, action)
        snapshot_after = read_navdata_snapshot(
            drone_host,
            timeout_seconds=0.5,
            initialize=False,
        )
        store_navdata_snapshot(snapshot_after)
        if action == "takeoff" and result.ok:
            st.session_state.drone_state = DroneState.AIRBORNE.value
        if action == "pause" and result.ok:
            st.session_state.drone_state = DroneState.PAUSED.value
        if action == "land" and result.ok:
            st.session_state.drone_state = DroneState.DISARMED.value
        if action == "emergency_land" and result.ok:
            st.session_state.drone_state = DroneState.EMERGENCY.value
        if action == "reset_emergency" and result.ok:
            st.session_state.drone_state = DroneState.DISARMED.value
        store.add(
            result.event_kind,
            result.summary,
            {
                "real": True,
                "ok": result.ok,
                "action": result.action,
                "host": drone_host,
                "detail": result.detail,
                "navdata_after": snapshot_after.__dict__,
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
        "reset_emergency",
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
    current_state = DroneState(st.session_state.drone_state)
    header_cols = st.columns([1.15, 2.35], vertical_alignment="center")
    with header_cols[0]:
        st.markdown(
            '<h3 class="sax-control-heading">Drone Control</h3>',
            unsafe_allow_html=True,
        )
    with header_cols[1]:
        command_mode = st.radio(
            "Command mode",
            ["Simulation", "Drone"],
            key="drone_command_mode",
            horizontal=True,
            label_visibility="collapsed",
        )

    real_mode = command_mode == "Drone"
    probe = st.session_state.get("drone_probe") or {}
    control_ready = bool(probe.get("ready_for_control"))
    snapshot = current_navdata_snapshot()
    battery_percent = (
        snapshot.get("battery_percent")
        if snapshot and snapshot.get("ok")
        else None
    )
    battery_ok = battery_percent is not None and battery_percent >= 20
    emergency_active = drone_reports_emergency(snapshot)

    st.markdown(
        (
            '<div class="sax-control-status">'
            '<span>State</span>'
            f'<span class="sax-control-state">{escape(current_state.value)}</span>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    if real_mode:
        st.markdown(
            '<div class="sax-compact-warning">Drone mode can spin motors. Keep clear.</div>',
            unsafe_allow_html=True,
        )
        guard_hints = []
        if not control_ready:
            guard_hints.append("Probe Drone before takeoff or hover.")
        if emergency_active:
            st.markdown(
                '<div class="sax-compact-error">Drone reports emergency state. Clear Emergency before takeoff.</div>',
                unsafe_allow_html=True,
            )
        if battery_percent is None:
            guard_hints.append("Live battery telemetry required.")
        elif not battery_ok:
            guard_hints.append("Battery must be at least 20%.")
        if guard_hints:
            st.markdown(
                f'<div class="sax-compact-hint">{" ".join(guard_hints)}</div>',
                unsafe_allow_html=True,
            )
        keepalive_error = st.session_state.get("last_hover_keepalive_error")
        if keepalive_error:
            st.markdown(
                f'<div class="sax-compact-error">Hover keepalive failed: {escape(keepalive_error)}</div>',
                unsafe_allow_html=True,
            )

    def run_command(action: str) -> None:
        execute_drone_action(store, action, command_mode, drone_host)
        st.rerun()

    if real_mode:
        takeoff_disabled = (
            not control_ready
            or not battery_ok
            or emergency_active
        )
    else:
        takeoff_disabled = False
    hover_disabled = real_mode and not control_ready
    trim_disabled = real_mode and not control_ready
    land_disabled = real_mode and not control_ready

    cols = st.columns(2)
    if cols[0].button(
        "Arm / Flat Trim",
        width="stretch",
        disabled=trim_disabled,
    ):
        run_command("flat_trim" if real_mode else "arm")
    if cols[1].button("Takeoff", width="stretch", disabled=takeoff_disabled):
        run_command("takeoff")

    cols = st.columns(2)
    if cols[0].button("Scan", width="stretch", disabled=real_mode):
        run_command("scan")
    pause_label = "Hover" if real_mode else "Pause"
    if cols[1].button(pause_label, width="stretch", disabled=hover_disabled):
        run_command("pause")

    if real_mode and st.button(
        "Clear Emergency",
        width="stretch",
        disabled=not control_ready or not emergency_active,
    ):
        run_command("reset_emergency")

    cols = st.columns(2)
    if cols[0].button("Land", width="stretch", disabled=land_disabled):
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

    st.caption("Movement and scan remain disabled for real drone mode.")


def render_drone_diagnostics(store: EventStore):
    with st.expander("Drone Link", expanded=True):
        drone_host = st.text_input("Drone host", key="drone_host")
        st.caption(f"Video source: `{video_source_url(drone_host)}`")

        if st.button("Probe Drone", width="stretch"):
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
                    "Drone probe: "
                    f"video={'ready' if result['ready_for_video'] else 'not ready'}, "
                    f"control={'ready' if result['ready_for_control'] else 'not ready'}"
                ),
                st.session_state.drone_probe,
            )
            if result["ready_for_video"]:
                st.session_state.camera_source = result["video_source"]
            st.rerun()

        telemetry_slot = st.empty()
        result = st.session_state.get("drone_probe")
        if not result:
            return telemetry_slot

        st.write(f"Local route address: `{result['local_address']}`")
        if not result["likely_on_drone_network"]:
            st.warning("Mac does not appear to be on the AR.Drone Wi-Fi network.")

        for key in ["video", "navdata", "control"]:
            probe = result[key]
            status = "OK" if probe["ok"] else "FAIL"
            st.write(f"{status} · {probe['name']} · {probe['detail']}")

        if result["ready_for_video"]:
            st.success("Video port is open. Switch to Drone video and press Start.")
        else:
            st.warning("Connect Mac Wi-Fi to the AR.Drone network, then probe again.")

        render_navdata_snapshot(telemetry_slot, current_navdata_snapshot())

        if st.session_state.running and is_drone_video_source(
            st.session_state.camera_source,
            drone_host,
        ):
            st.caption("Telemetry auto-refreshes while the drone video stream is running.")

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
    if not detections:
        slot.markdown(
            """
            <h3>Current Objects</h3>
            <div class="sax-empty">No objects above threshold.</div>
            """,
            unsafe_allow_html=True,
        )
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

    rows = []
    for label, count in counts.most_common():
        priority_marker = "priority" if profile.is_priority(label) else "observed"
        confidence_list = ", ".join(
            f"{confidence:.0%}"
            for confidence in sorted(confidences[label], reverse=True)[:5]
        )
        rows.append(
            "<tr>"
            f"<td><strong>{escape(label)}</strong></td>"
            f"<td>{count}</td>"
            f"<td>{max_confidence[label]:.0%}</td>"
            f"<td>{escape(priority_marker)}</td>"
            f"<td>{escape(confidence_list)}</td>"
            "</tr>"
        )

    slot.markdown(
        (
            "<h3>Current Objects</h3>"
            '<table class="sax-detections">'
            "<thead><tr>"
            "<th>Object</th><th>Count</th><th>Best</th><th>Type</th><th>Confidences</th>"
            "</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        ),
        unsafe_allow_html=True,
    )


def run_webcam_mode(
    store: EventStore,
    detector: YoloDetector,
    model_name: str,
    confidence: float,
    profile: MissionProfile,
    status_slot,
    frame_slot,
    detections_slot,
) -> None:
    render_status_strip(status_slot, st.session_state.camera_source, [])

    if st.session_state.webcam_photo_bytes is None:
        with frame_slot.container():
            photo = st.camera_input(
                "Capture sensor frame",
                key=f"webcam_capture_{st.session_state.webcam_capture_key}",
            )

        if photo is not None:
            st.session_state.webcam_photo_bytes = photo.getvalue()
            st.rerun()

        render_current_objects(detections_slot, [], profile)
        return

    detector.load(model_name)
    image_array = np.frombuffer(st.session_state.webcam_photo_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    detections = detector.detect(frame, confidence)
    log_detections(store, detections, profile)
    annotated = detector.draw(frame, detections)

    with frame_slot.container():
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption="Annotated detections",
            channels="RGB",
            width="stretch",
        )
        if st.button("Clear image", width="stretch"):
            st.session_state.webcam_photo_bytes = None
            st.session_state.webcam_capture_key += 1
            st.rerun()

    render_status_strip(status_slot, st.session_state.camera_source, detections)
    render_current_objects(detections_slot, detections, profile)


def open_capture(source: str) -> cv2.VideoCapture:
    parsed_source = parse_source(source)
    if isinstance(parsed_source, int):
        capture = cv2.VideoCapture(parsed_source, cv2.CAP_AVFOUNDATION)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
        capture.release()

    capture = cv2.VideoCapture(parsed_source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def render_video_frame(slot, frame) -> None:
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 82],
    )
    if ok:
        slot.image(encoded.tobytes(), width="stretch")
    else:
        slot.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            output_format="JPEG",
            width="stretch",
        )


def release_drone_video_capture() -> None:
    capture = st.session_state.get("drone_video_capture")
    if capture is not None:
        capture.release()
    st.session_state.drone_video_capture = None
    st.session_state.drone_video_source = ""
    st.session_state.drone_video_frame_index = 0
    st.session_state.drone_video_missed_frames = 0


def drone_video_capture_for(source: str) -> cv2.VideoCapture:
    capture = st.session_state.get("drone_video_capture")
    if capture is not None and st.session_state.drone_video_source == source:
        return capture

    release_drone_video_capture()
    capture = open_capture(source)
    st.session_state.drone_video_capture = capture
    st.session_state.drone_video_source = source
    st.session_state.drone_video_frame_index = 0
    st.session_state.drone_video_missed_frames = 0
    st.session_state.latest_detections = []
    return capture


@st.fragment(run_every=0.12)
def render_drone_video_fragment(
    store: EventStore,
    detector: YoloDetector,
    source: str,
    model_name: str,
    confidence: float,
    frame_stride: int,
    profile: MissionProfile,
    status_slot,
    frame_slot,
    detections_slot,
    telemetry_slot,
) -> None:
    if not st.session_state.running:
        release_drone_video_capture()
        render_status_strip(status_slot, source, [])
        frame_slot.info("Press Start to open the video source.")
        render_current_objects(detections_slot, [], profile)
        return

    detector.load(model_name)
    capture = drone_video_capture_for(source)
    if not capture.isOpened():
        frame_slot.error(
            f"Could not open video source: {source}\n\n"
            "On macOS, OpenCV needs camera permission for the app that launched Python. "
            "For webcam testing, use Webcam mode. For the AR.Drone, keep using "
            "Drone video with tcp://192.168.1.1:5555."
        )
        st.session_state.running = False
        release_drone_video_capture()
        return

    ok, frame = capture.read()
    if not ok:
        st.session_state.drone_video_missed_frames += 1
        if st.session_state.drone_video_missed_frames >= 12:
            frame_slot.warning("No frame received from video source.")
        render_status_strip(
            status_slot,
            source,
            st.session_state.latest_detections,
        )
        return

    st.session_state.drone_video_missed_frames = 0
    st.session_state.drone_video_frame_index += 1
    if st.session_state.drone_video_frame_index % frame_stride == 0:
        st.session_state.latest_detections = detector.detect(frame, confidence)
        log_detections(store, st.session_state.latest_detections, profile)

    latest_detections = st.session_state.latest_detections
    annotated = detector.draw(frame, latest_detections)
    render_video_frame(frame_slot, annotated)
    maybe_auto_refresh_navdata(telemetry_slot, source, st.session_state.drone_host)

    now = time.monotonic()
    if now - st.session_state.last_drone_panel_refresh_at >= 0.35:
        st.session_state.last_drone_panel_refresh_at = now
        render_status_strip(status_slot, source, latest_detections)
        render_current_objects(detections_slot, latest_detections, profile)


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

        st.header("Controls")
        cols = st.columns(2)
        if cols[0].button("Start", width="stretch"):
            st.session_state.running = True
        if cols[1].button("Stop", width="stretch"):
            st.session_state.running = False
            release_drone_video_capture()

        if st.button("Clear timeline", width="stretch"):
            store.clear()
            st.session_state.last_logged_at = {}
            st.session_state.latest_sitrep = ""
            st.rerun()

        st.header("Input")
        input_mode = st.radio(
            "Mode",
            ["Webcam", "Drone Video"],
            help=(
                "Webcam uses browser camera permission. Drone Video "
                "uses OpenCV and is better for drone/video streams."
            ),
            label_visibility="collapsed",
        )
        with st.expander("Details", expanded=False):
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

    video_col, intel_col = st.columns([2, 1])
    with video_col:
        st.subheader("Sensor Feed")
        status_slot = st.empty()
        frame_slot = st.empty()
        detections_slot = st.empty()

    with intel_col:
        telemetry_slot = render_drone_diagnostics(store)
        render_drone_controls(store, profile, st.session_state.drone_host)
        flight_keepalive_fragment(st.session_state.drone_host)

        with st.expander("Operator Notes", expanded=False):
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

        with st.expander("SITREP", expanded=False):
            render_sitrep(store, profile, show_title=False)

        with st.expander("Mission Export", expanded=False):
            render_mission_export(
                store,
                profile,
                source,
                model_name,
                confidence,
                show_title=False,
            )

        with st.expander("Mission Timeline", expanded=False):
            render_timeline(store, show_title=False)

    if input_mode == "Webcam":
        run_webcam_mode(
            store,
            detector,
            model_name,
            confidence,
            profile,
            status_slot,
            frame_slot,
            detections_slot,
        )
        return

    if not st.session_state.running:
        release_drone_video_capture()

    render_drone_video_fragment(
        store,
        detector,
        source,
        model_name,
        confidence,
        frame_stride,
        profile,
        status_slot,
        frame_slot,
        detections_slot,
        telemetry_slot,
    )


if __name__ == "__main__":
    main()
