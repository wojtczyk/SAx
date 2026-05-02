from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any


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


def _assessment(object_events: list[dict[str, Any]], note_events: list[dict[str, Any]]) -> str:
    labels = [
        str(event.get("metadata", {}).get("label", "")).lower()
        for event in object_events
    ]
    has_person = "person" in labels
    has_vehicle = any(label in VEHICLE_LABELS for label in labels)

    if has_person and has_vehicle:
        return "Human and vehicle activity detected in the current observation window."
    if has_person:
        return "Human activity detected. Maintain observation and collect another confirmation if movement persists."
    if has_vehicle:
        return "Vehicle-like activity detected. Preserve timestamps and continue monitoring for direction of travel."
    if note_events:
        return "Operator notes recorded, but no priority object classes were detected in the selected window."
    return "No notable activity recorded in the selected window."


def _recommendation(object_events: list[dict[str, Any]]) -> str:
    priority_labels = {"person", *VEHICLE_LABELS}
    labels = {
        str(event.get("metadata", {}).get("label", "")).lower()
        for event in object_events
    }
    if labels & priority_labels:
        return "Keep the sensor on the area of interest, capture one more frame, and add an operator note with location context."
    return "Continue scanning and add an operator note if visual context changes."


def generate_sitrep(events: list[dict[str, Any]]) -> str:
    if not events:
        return (
            f"SITREP generated {datetime.now().strftime('%H:%M:%S')}\n"
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

    return "\n".join(
        [
            f"SITREP generated {datetime.now().strftime('%H:%M:%S')}",
            f"Window: {first_seen} to {last_seen}",
            f"Activity: {len(object_events)} object events. {label_summary}",
            f"Operator notes: {_format_notes(note_events)}",
            f"Assessment: {_assessment(object_events, note_events)}",
            f"Recommended action: {_recommendation(object_events)}",
        ]
    )
