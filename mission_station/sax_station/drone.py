from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DroneState(StrEnum):
    DISARMED = "DISARMED"
    ARMED = "ARMED"
    AIRBORNE = "AIRBORNE"
    SCANNING = "SCANNING"
    PAUSED = "PAUSED"
    LANDING = "LANDING"
    EMERGENCY = "EMERGENCY"


@dataclass(frozen=True)
class DroneCommandResult:
    state: DroneState
    summary: str
    event_kind: str = "drone_command"


def command(current_state: DroneState, action: str) -> DroneCommandResult:
    if action == "arm":
        if current_state in {DroneState.AIRBORNE, DroneState.SCANNING}:
            return DroneCommandResult(current_state, "Arm ignored: drone is already airborne.")
        return DroneCommandResult(DroneState.ARMED, "Drone simulation armed.")

    if action == "takeoff":
        if current_state != DroneState.ARMED:
            return DroneCommandResult(current_state, "Takeoff blocked: arm the drone simulation first.")
        return DroneCommandResult(DroneState.AIRBORNE, "Simulated takeoff and hover.")

    if action == "scan":
        if current_state not in {DroneState.AIRBORNE, DroneState.PAUSED}:
            return DroneCommandResult(current_state, "Scan blocked: drone simulation must be airborne.")
        return DroneCommandResult(DroneState.SCANNING, "Simulated assisted search scan started.")

    if action == "pause":
        if current_state not in {DroneState.AIRBORNE, DroneState.SCANNING}:
            return DroneCommandResult(current_state, "Pause ignored: drone simulation is not airborne.")
        return DroneCommandResult(DroneState.PAUSED, "Simulated drone paused in hover.")

    if action == "land":
        if current_state in {DroneState.DISARMED, DroneState.ARMED}:
            return DroneCommandResult(DroneState.DISARMED, "Land ignored: drone simulation is not airborne.")
        return DroneCommandResult(DroneState.DISARMED, "Simulated landing complete.")

    if action == "emergency_land":
        return DroneCommandResult(
            DroneState.EMERGENCY,
            "Emergency land triggered in simulation.",
            event_kind="drone_emergency",
        )

    return DroneCommandResult(current_state, f"Unknown simulated drone action: {action}.")


def assisted_search_sequence() -> list[DroneCommandResult]:
    return [
        DroneCommandResult(DroneState.ARMED, "Assisted Search: arm simulation."),
        DroneCommandResult(DroneState.AIRBORNE, "Assisted Search: takeoff and hover."),
        DroneCommandResult(DroneState.SCANNING, "Assisted Search: rotate scan left."),
        DroneCommandResult(DroneState.SCANNING, "Assisted Search: rotate scan right."),
        DroneCommandResult(DroneState.SCANNING, "Assisted Search: capture and analyze sensor frame."),
        DroneCommandResult(DroneState.DISARMED, "Assisted Search: land and disarm simulation."),
    ]
