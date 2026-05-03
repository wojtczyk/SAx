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
    "move_forward",
    "move_back",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "yaw_left",
    "yaw_right",
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
    if text in {"forward", "move forward", "nudge forward", "front"}:
        return ParsedCommand("move_forward")
    if text in {"back", "backward", "move back", "move backward", "nudge back"}:
        return ParsedCommand("move_back")
    if text in {"left", "strafe left", "move left", "nudge left"}:
        return ParsedCommand("move_left")
    if text in {"right", "strafe right", "move right", "nudge right"}:
        return ParsedCommand("move_right")
    if text in {"up", "ascend", "move up", "nudge up"}:
        return ParsedCommand("move_up")
    if text in {"down", "descend", "move down", "nudge down"}:
        return ParsedCommand("move_down")
    if text in {"yaw left", "turn left", "rotate left", "spin left"}:
        return ParsedCommand("yaw_left")
    if text in {"yaw right", "turn right", "rotate right", "spin right"}:
        return ParsedCommand("yaw_right")
    if text in {"flat trim", "flat_trim", "trim", "calibrate"}:
        return ParsedCommand("flat_trim")
    if text in {"clear emergency", "reset emergency", "reset drone", "clear error"}:
        return ParsedCommand("reset_emergency")
    if text in {"land", "land drone"}:
        return ParsedCommand("land")
    if text in {"emergency land", "emergency landing", "emergency stop", "abort"}:
        return ParsedCommand("emergency_land")
    if text in {
        "assisted search",
        "run assisted search",
        "start assisted search",
        "autonomous search",
        "run autonomous search",
        "start autonomous search",
    }:
        return ParsedCommand("assisted_search")
    if text in {"sitrep", "generate sitrep", "create sitrep"}:
        return ParsedCommand("sitrep")

    return ParsedCommand("unknown", raw.strip())
