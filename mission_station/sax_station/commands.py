from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedCommand:
    intent: str
    value: str = ""


DRONE_INTENTS = {
    "arm",
    "takeoff",
    "scan",
    "pause",
    "land",
    "flat_trim",
    "reset_emergency",
    "emergency_land",
    "assisted_search",
}


def parse_command(raw: str) -> ParsedCommand:
    text = " ".join(raw.lower().strip().split())
    if not text:
        return ParsedCommand("empty")

    if text.startswith("note "):
        return ParsedCommand("note", raw.strip()[5:].strip())
    if text.startswith("add note "):
        return ParsedCommand("note", raw.strip()[9:].strip())
    if text.startswith("mark "):
        return ParsedCommand("note", raw.strip()[5:].strip())

    if text in {"arm", "arm drone"}:
        return ParsedCommand("arm")
    if text in {"takeoff", "take off", "launch", "launch drone"}:
        return ParsedCommand("takeoff")
    if text in {"scan", "start scan", "begin scan", "start search"}:
        return ParsedCommand("scan")
    if text in {"pause", "hold", "hover"}:
        return ParsedCommand("pause")
    if text in {"flat trim", "flat_trim", "trim", "calibrate"}:
        return ParsedCommand("flat_trim")
    if text in {"clear emergency", "reset emergency", "reset drone", "clear error"}:
        return ParsedCommand("reset_emergency")
    if text in {"land", "land drone"}:
        return ParsedCommand("land")
    if text in {"emergency land", "emergency landing", "emergency stop", "abort"}:
        return ParsedCommand("emergency_land")
    if text in {"assisted search", "run assisted search", "start assisted search"}:
        return ParsedCommand("assisted_search")
    if text in {"sitrep", "generate sitrep", "create sitrep"}:
        return ParsedCommand("sitrep")

    return ParsedCommand("unknown", raw.strip())
