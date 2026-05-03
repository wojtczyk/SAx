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

    print(f"Ready for video: {result['ready_for_video']}")
    print(f"Ready for control: {result['ready_for_control']}")


if __name__ == "__main__":
    main()
