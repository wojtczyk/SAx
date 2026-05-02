from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from .profiles import MissionProfile


VEHICLE_LABELS = {
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
}


def _format_label_summary(object_events: list[dict[str, Any]]) -> str:
    counts: Counter[str] = Counter()
    max_confidence: dict[str, float] = defaultdict(float)

    for event in object_events:
        metadata = event.get("metadata", {})
        label = str(metadata.get("label") or event["summary"].split(" detected")[0])
        confidence = float(metadata.get("confidence") or 0.0)
        counts[label] += 1
        max_confidence[label] = max(max_confidence[label], confidence)

    parts = []
    for label, count in counts.most_common():
        confidence = max_confidence[label]
        confidence_text = f", max {confidence:.0%}" if confidence else ""
        parts.append(f"{label} x{count}{confidence_text}")
    return "; ".join(parts)


def _format_notes(note_events: list[dict[str, Any]], limit: int = 3) -> str:
    if not note_events:
        return "No operator notes recorded."

    recent_notes = note_events[-limit:]
    return " ".join(f"{event['created_at']}: {event['summary']}" for event in recent_notes)


def _priority_events(
    object_events: list[dict[str, Any]],
    profile: MissionProfile,
) -> list[dict[str, Any]]:
    return [
        event
        for event in object_events
        if profile.is_priority(str(event.get("metadata", {}).get("label", "")))
    ]


def _assessment(
    object_events: list[dict[str, Any]],
    note_events: list[dict[str, Any]],
    profile: MissionProfile,
) -> str:
    labels = [
        str(event.get("metadata", {}).get("label", "")).lower()
        for event in object_events
    ]
    priority_events = _priority_events(object_events, profile)
    has_person = "person" in labels
    has_vehicle = any(label in VEHICLE_LABELS for label in labels)

    if priority_events:
        return (
            f"{profile.name} priority activity detected. Focus areas: "
            f"{profile.assessment_focus}."
        )
    if has_person and has_vehicle:
        return "Human and vehicle activity detected in the current observation window."
    if has_person:
        return "Human activity detected. Maintain observation and collect another confirmation if movement persists."
    if has_vehicle:
        return "Vehicle-like activity detected. Preserve timestamps and continue monitoring for direction of travel."
    if note_events:
        return "Operator notes recorded, but no priority object classes were detected in the selected window."
    return "No notable activity recorded in the selected window."


def _recommendation(
    object_events: list[dict[str, Any]],
    profile: MissionProfile,
) -> str:
    if _priority_events(object_events, profile):
        return profile.recommendation
    return "Continue scanning and add an operator note if visual context changes."


def generate_sitrep(events: list[dict[str, Any]], profile: MissionProfile) -> str:
    if not events:
        return (
            f"SITREP generated {datetime.now().strftime('%H:%M:%S')}\n"
            f"Mission profile: {profile.name} ({profile.code})\n"
            "Window: no timeline events available.\n"
            "Activity: no detections or operator notes recorded.\n"
            "Assessment: no notable activity recorded yet.\n"
            "Recommended action: collect a snapshot or add an operator note."
        )

    if all("id" in event for event in events):
        chronological_events = sorted(events, key=lambda event: int(event["id"]))
    else:
        chronological_events = sorted(events, key=lambda event: str(event["created_at"]))
    object_events = [
        event for event in chronological_events if event["kind"] == "object_detected"
    ]
    note_events = [
        event
        for event in chronological_events
        if event["kind"] in {"operator_note", "voice_note"}
    ]

    first_seen = chronological_events[0]["created_at"]
    last_seen = chronological_events[-1]["created_at"]
    label_summary = (
        _format_label_summary(object_events)
        if object_events
        else "No object detections recorded."
    )
    priority_count = len(_priority_events(object_events, profile))

    return "\n".join(
        [
            f"SITREP generated {datetime.now().strftime('%H:%M:%S')}",
            f"Mission profile: {profile.name} ({profile.code})",
            f"Window: {first_seen} to {last_seen}",
            (
                f"Activity: {len(object_events)} object events; "
                f"{priority_count} {profile.detection_term} events. {label_summary}"
            ),
            f"Operator notes: {_format_notes(note_events)}",
            f"Assessment: {_assessment(object_events, note_events, profile)}",
            f"Recommended action: {_recommendation(object_events, profile)}",
        ]
    )
