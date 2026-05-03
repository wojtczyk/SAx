from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .profiles import MissionProfile


def _filename(now: datetime) -> str:
    return f"{now.strftime('%Y-%m-%d_%H%M')}_sax_report.md"


def _event_lines(events: list[dict[str, Any]]) -> list[str]:
    if not events:
        return ["No events recorded."]

    lines = []
    for event in reversed(events):
        lines.append(f"- {event['created_at']} | {event['kind']} | {event['summary']}")
    return lines


def _event_counts(events: list[dict[str, Any]]) -> list[str]:
    counts = Counter(event["kind"] for event in events)
    if not counts:
        return ["No events recorded."]
    return [f"- {kind}: {count}" for kind, count in counts.most_common()]


def export_mission_report(
    export_dir: Path,
    events: list[dict[str, Any]],
    profile: MissionProfile,
    drone_state: str,
    video_source: str,
    model_name: str,
    confidence: float,
    latest_sitrep: str,
    now: datetime | None = None,
) -> Path:
    timestamp = now or datetime.now()
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / _filename(timestamp)

    sitrep = latest_sitrep.strip() or "No SITREP generated."
    content = "\n".join(
        [
            "# SAx Mission Report",
            "",
            f"Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Mission profile: {profile.name} ({profile.code})",
            f"Drone simulation state: {drone_state}",
            f"Video source: {video_source}",
            f"YOLO model: {model_name}",
            f"Confidence threshold: {confidence:.2f}",
            "",
            "## SITREP",
            "",
            "```text",
            sitrep,
            "```",
            "",
            "## Event Counts",
            "",
            *_event_counts(events),
            "",
            "## Timeline",
            "",
            *_event_lines(events),
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path
