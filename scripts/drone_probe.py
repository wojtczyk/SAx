from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mission_station.sax_station.ardrone import DEFAULT_DRONE_HOST, probe_drone


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a Parrot AR.Drone network endpoint.")
    parser.add_argument("--host", default=DEFAULT_DRONE_HOST)
    args = parser.parse_args()

    result = probe_drone(args.host)
    print(f"AR.Drone host: {result['host']}")
    print(f"Local address used: {result['local_address']}")
    print(f"Likely on drone network: {result['likely_on_drone_network']}")
    print(f"Video source: {result['video_source']}")

    for key in ["video", "navdata", "control"]:
        probe = result[key]
        status = "ok" if probe.ok else "failed"
        print(f"{probe.name}: {status} ({probe.detail})")

    battery = result["battery"]
    battery_text = f"{battery.battery_percent}%" if battery.ok else f"unavailable ({battery.detail})"
    print(f"Battery: {battery_text}")
    altitude_text = (
        f"{battery.altitude_cm} cm ({battery.altitude_cm / 100:.2f} m)"
        if battery.ok and battery.altitude_cm is not None
        else f"unavailable ({battery.detail})"
    )
    print(f"Altitude: {altitude_text}")
    print(f"Flying state: {battery.flying_state_name}")
    heading_text = f"{battery.yaw_degrees:.1f} deg" if battery.yaw_degrees is not None else "unavailable"
    print(f"Yaw/heading: {heading_text}")
    velocity = battery.velocity_mps
    velocity_text = (
        f"vx={velocity[0]:.2f} m/s, vy={velocity[1]:.2f} m/s, vz={velocity[2]:.2f} m/s"
        if velocity is not None
        else "unavailable"
    )
    print(f"Velocity: {velocity_text}")
    print(f"Ready for video: {result['ready_for_video']}")
    print(f"Ready for control: {result['ready_for_control']}")


if __name__ == "__main__":
    main()
