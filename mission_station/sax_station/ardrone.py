from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
import threading
import time
from typing import Any


DEFAULT_DRONE_HOST = "192.168.1.1"
VIDEO_PORT = 5555
NAVDATA_PORT = 5554
CONTROL_PORT = 5556
NAVDATA_HEADER = 0x55667788
NAVDATA_DEMO_TAG = 0
NUDGE_FORWARD_SPEED = 0.08
NUDGE_BACK_SPEED = 0.12
NUDGE_ROLL_SPEED = 0.14
NUDGE_VERTICAL_SPEED = 0.18
NUDGE_YAW_SPEED = 0.28
NUDGE_REPEAT = 9
NUDGE_INTERVAL_SECONDS = 0.03
SCAN_STEPS = ("yaw_left", "yaw_right", "yaw_right", "yaw_left")
ASSISTED_SEARCH_STEPS = (
    "yaw_left",
    "yaw_right",
    "move_forward",
    "yaw_left",
    "yaw_right",
    "move_back",
)
REF_LAND = 290717696
REF_TAKEOFF = 290718208
REF_EMERGENCY = 290717952
DRONE_STATE_FLYING_MASK = 1 << 0
DRONE_STATE_EMERGENCY_MASK = 1 << 31
CTRL_STATE_NAMES = {
    0: "Unknown",
    1: "Inited",
    2: "Landed",
    3: "Flying",
    4: "Hovering",
    5: "Test",
    6: "Taking off",
    7: "Flying",
    8: "Landing",
    9: "Looping",
}
_SEQUENCE_LOCK = threading.Lock()
_NEXT_SEQUENCE = max(1, int(time.monotonic() * 1000))


def next_at_sequence() -> int:
    global _NEXT_SEQUENCE
    with _SEQUENCE_LOCK:
        sequence = _NEXT_SEQUENCE
        _NEXT_SEQUENCE += 1
        return sequence


def float_arg(value: float) -> int:
    return struct.unpack("<i", struct.pack("<f", float(value)))[0]


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class RealDroneCommandResult:
    ok: bool
    summary: str
    action: str
    detail: str
    event_kind: str = "drone_real_command"


@dataclass(frozen=True)
class NavdataSnapshot:
    ok: bool
    battery_percent: int | None
    ctrl_state: int | None
    sequence: int | None
    drone_state: int | None
    detail: str
    altitude_cm: int | None = None
    theta_mdeg: float | None = None
    phi_mdeg: float | None = None
    psi_mdeg: float | None = None
    vx: float | None = None
    vy: float | None = None
    vz: float | None = None

    @property
    def flying_state_code(self) -> int | None:
        if self.ctrl_state is None:
            return None
        return self.ctrl_state >> 16

    @property
    def flying_state_name(self) -> str:
        code = self.flying_state_code
        if code is None:
            return "Unknown"
        return CTRL_STATE_NAMES.get(code, f"Unknown ({code})")

    @property
    def yaw_degrees(self) -> float | None:
        if self.psi_mdeg is None:
            return None
        return self.psi_mdeg / 1000

    @property
    def velocity_mps(self) -> tuple[float, float, float] | None:
        if self.vx is None or self.vy is None or self.vz is None:
            return None
        return (self.vx / 1000, self.vy / 1000, self.vz / 1000)


class ARDroneATClient:
    def __init__(
        self,
        host: str = DEFAULT_DRONE_HOST,
        local_port: int = CONTROL_PORT,
        sequence_start: int | None = None,
    ) -> None:
        self.host = host
        self.local_port = local_port
        self.sequence = sequence_start

    def _next_sequence(self) -> int:
        if self.sequence is None:
            return next_at_sequence()

        sequence = self.sequence
        self.sequence += 1
        return sequence

    @staticmethod
    def _format_arg(value: int | str) -> str:
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)

    def _command(self, name: str, *args: int | str) -> str:
        values = ",".join(
            self._format_arg(value)
            for value in (self._next_sequence(), *args)
        )
        return f"AT*{name}={values}\r"

    def _send(self, commands: list[str]) -> None:
        payload = "".join(commands).encode("ascii")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(1.0)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            sock.bind(("", self.local_port))
            sock.sendto(payload, (self.host, CONTROL_PORT))

    def init_navdata(self, repeat: int = 3, interval_seconds: float = 0.03) -> None:
        for _ in range(repeat):
            self._send(
                [
                    self._command("CONFIG", "general:navdata_demo", "TRUE"),
                    self._command("CTRL", 5, 0),
                    self._command("COMWDG"),
                ]
            )
            time.sleep(interval_seconds)

    def flat_trim(self) -> None:
        self._send([self._command("FTRIM")])

    def takeoff(self, repeat: int = 60, interval_seconds: float = 0.03) -> None:
        self.init_navdata(repeat=1)
        self.flat_trim()
        for _ in range(repeat):
            self._send(
                [
                    self._command("REF", REF_TAKEOFF),
                    self._command("COMWDG"),
                ]
            )
            time.sleep(interval_seconds)
        self.hover(repeat=20)

    def hover(self, repeat: int = 10, interval_seconds: float = 0.03) -> None:
        for _ in range(repeat):
            self._send(
                [
                    self._command("PCMD", 0, 0, 0, 0, 0),
                    self._command("COMWDG"),
                ]
            )
            time.sleep(interval_seconds)

    def move(
        self,
        *,
        roll: float = 0.0,
        pitch: float = 0.0,
        gaz: float = 0.0,
        yaw: float = 0.0,
        repeat: int = NUDGE_REPEAT,
        interval_seconds: float = NUDGE_INTERVAL_SECONDS,
        hover_after: bool = True,
    ) -> None:
        for _ in range(repeat):
            self._send(
                [
                    self._command(
                        "PCMD",
                        1,
                        float_arg(roll),
                        float_arg(pitch),
                        float_arg(gaz),
                        float_arg(yaw),
                    ),
                    self._command("COMWDG"),
                ]
            )
            time.sleep(interval_seconds)

        if hover_after:
            self.hover(repeat=4, interval_seconds=interval_seconds)

    def nudge(self, action: str) -> None:
        if action == "move_forward":
            self.move(pitch=-NUDGE_FORWARD_SPEED)
            return
        if action == "move_back":
            self.move(pitch=NUDGE_BACK_SPEED)
            return
        if action == "move_left":
            self.move(roll=-NUDGE_ROLL_SPEED)
            return
        if action == "move_right":
            self.move(roll=NUDGE_ROLL_SPEED)
            return
        if action == "move_up":
            self.move(gaz=NUDGE_VERTICAL_SPEED)
            return
        if action == "move_down":
            self.move(gaz=-NUDGE_VERTICAL_SPEED)
            return
        if action == "yaw_left":
            self.move(yaw=-NUDGE_YAW_SPEED)
            return
        if action == "yaw_right":
            self.move(yaw=NUDGE_YAW_SPEED)
            return

        raise ValueError(f"unknown nudge action: {action}")

    def run_nudge_sequence(
        self,
        steps: tuple[str, ...],
        pause_seconds: float = 0.12,
    ) -> None:
        for step in steps:
            self.nudge(step)
            time.sleep(pause_seconds)
        self.hover(repeat=6)

    def scan(self) -> None:
        self.run_nudge_sequence(SCAN_STEPS)

    def assisted_search(self) -> None:
        self.run_nudge_sequence(ASSISTED_SEARCH_STEPS)

    def land(self, repeat: int = 30, interval_seconds: float = 0.03) -> None:
        for _ in range(repeat):
            self._send(
                [
                    self._command("REF", REF_LAND),
                    self._command("COMWDG"),
                ]
            )
            time.sleep(interval_seconds)

    def emergency_stop(self) -> None:
        self._send(
            [
                self._command("REF", REF_LAND),
                self._command("REF", REF_EMERGENCY),
                self._command("REF", REF_LAND),
            ]
        )

    def reset_emergency(self, repeat: int = 8, interval_seconds: float = 0.03) -> None:
        for _ in range(repeat):
            self._send(
                [
                    self._command("REF", REF_EMERGENCY),
                    self._command("COMWDG"),
                ]
            )
            time.sleep(interval_seconds)


def local_address_for(host: str, port: int = CONTROL_PORT) -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((host, port))
            return str(sock.getsockname()[0])
    except OSError:
        return None


def video_source_url(host: str = DEFAULT_DRONE_HOST) -> str:
    return f"tcp://{host}:{VIDEO_PORT}"


def probe_tcp_port(host: str, port: int, timeout_seconds: float = 1.5) -> ProbeResult:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return ProbeResult(f"TCP {host}:{port}", True, "open")
    except OSError as exc:
        return ProbeResult(f"TCP {host}:{port}", False, str(exc))


def probe_udp_socket(host: str, port: int, timeout_seconds: float = 1.5) -> ProbeResult:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(timeout_seconds)
            sock.connect((host, port))
            return ProbeResult(f"UDP {host}:{port}", True, "socket ready")
    except OSError as exc:
        return ProbeResult(f"UDP {host}:{port}", False, str(exc))


def parse_navdata_packet(packet: bytes) -> NavdataSnapshot:
    if len(packet) < 16:
        return NavdataSnapshot(False, None, None, None, None, "packet too short")

    header, drone_state, sequence, _vision = struct.unpack_from("<IIII", packet, 0)
    if header != NAVDATA_HEADER:
        return NavdataSnapshot(
            False,
            None,
            None,
            sequence,
            drone_state,
            f"unexpected navdata header: 0x{header:08x}",
        )

    offset = 16
    while offset + 4 <= len(packet):
        tag, size = struct.unpack_from("<HH", packet, offset)
        if size < 4:
            return NavdataSnapshot(False, None, None, sequence, drone_state, "invalid option size")
        if offset + size > len(packet):
            return NavdataSnapshot(False, None, None, sequence, drone_state, "truncated option")

        if tag == NAVDATA_DEMO_TAG:
            if size < 40:
                return NavdataSnapshot(False, None, None, sequence, drone_state, "demo option too small")
            (
                ctrl_state,
                battery_percent,
                theta,
                phi,
                psi,
                altitude_cm,
                vx,
                vy,
                vz,
            ) = struct.unpack_from("<IIfffifff", packet, offset + 4)
            return NavdataSnapshot(
                True,
                int(battery_percent),
                int(ctrl_state),
                int(sequence),
                int(drone_state),
                "demo navdata parsed",
                int(altitude_cm),
                float(theta),
                float(phi),
                float(psi),
                float(vx),
                float(vy),
                float(vz),
            )

        offset += size

    return NavdataSnapshot(False, None, None, sequence, drone_state, "demo option not found")


def read_navdata_snapshot(
    host: str = DEFAULT_DRONE_HOST,
    timeout_seconds: float = 2.0,
    initialize: bool = True,
) -> NavdataSnapshot:
    deadline = time.monotonic() + timeout_seconds
    last_snapshot: NavdataSnapshot | None = None
    init_detail: str | None = None

    if initialize:
        try:
            ARDroneATClient(host).init_navdata()
        except OSError as exc:
            init_detail = f"navdata init failed: {exc}"

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(min(0.3, max(0.05, timeout_seconds)))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            try:
                sock.bind(("", NAVDATA_PORT))
            except OSError:
                sock.bind(("", 0))

            sock.sendto(struct.pack("<I", 1), (host, NAVDATA_PORT))

            while time.monotonic() < deadline:
                try:
                    packet, _address = sock.recvfrom(65535)
                except (TimeoutError, socket.timeout):
                    sock.sendto(struct.pack("<I", 1), (host, NAVDATA_PORT))
                    continue

                snapshot = parse_navdata_packet(packet)
                last_snapshot = snapshot
                if snapshot.ok:
                    return snapshot
    except OSError as exc:
        detail = str(exc)
        if init_detail:
            detail = f"{detail}; {init_detail}"
        return NavdataSnapshot(False, None, None, None, None, detail)

    if last_snapshot:
        if init_detail and not last_snapshot.ok:
            return NavdataSnapshot(
                last_snapshot.ok,
                last_snapshot.battery_percent,
                last_snapshot.ctrl_state,
                last_snapshot.sequence,
                last_snapshot.drone_state,
                f"{last_snapshot.detail}; {init_detail}",
                last_snapshot.altitude_cm,
                last_snapshot.theta_mdeg,
                last_snapshot.phi_mdeg,
                last_snapshot.psi_mdeg,
                last_snapshot.vx,
                last_snapshot.vy,
                last_snapshot.vz,
            )
        return last_snapshot

    detail = "no navdata received"
    if init_detail:
        detail = f"{detail}; {init_detail}"
    return NavdataSnapshot(False, None, None, None, None, detail)


def probe_drone(host: str = DEFAULT_DRONE_HOST) -> dict[str, Any]:
    local_address = local_address_for(host)
    likely_on_drone_network = bool(local_address and local_address.startswith("192.168.1."))
    video = probe_tcp_port(host, VIDEO_PORT)
    navdata = probe_udp_socket(host, NAVDATA_PORT)
    control = probe_udp_socket(host, CONTROL_PORT)
    snapshot = (
        read_navdata_snapshot(host)
        if likely_on_drone_network
        else NavdataSnapshot(False, None, None, None, None, "not on AR.Drone network")
    )
    return {
        "host": host,
        "local_address": local_address,
        "likely_on_drone_network": likely_on_drone_network,
        "video_source": video_source_url(host),
        "video": video,
        "navdata": navdata,
        "control": control,
        "battery": snapshot,
        "ready_for_video": video.ok,
        "ready_for_control": likely_on_drone_network and navdata.ok and control.ok,
    }


def send_real_drone_command(
    host: str,
    action: str,
) -> RealDroneCommandResult:
    nudge_actions = {
        "move_forward",
        "move_back",
        "move_left",
        "move_right",
        "move_up",
        "move_down",
        "yaw_left",
        "yaw_right",
    }
    if action not in {
        "flat_trim",
        "takeoff",
        "pause",
        "scan",
        "assisted_search",
        "land",
        "emergency_land",
        "reset_emergency",
        *nudge_actions,
    }:
        return RealDroneCommandResult(
            False,
            f"Drone command blocked: {action.replace('_', ' ')} is not enabled yet.",
            action,
            "Only flat trim, clear emergency, takeoff, hover, scan, autonomous search, nudge movement, land, and emergency stop are enabled in real mode.",
            event_kind="drone_real_command_blocked",
        )

    try:
        client = ARDroneATClient(host)
        if action == "flat_trim":
            client.flat_trim()
            return RealDroneCommandResult(
                True,
                "Drone flat trim sent.",
                action,
                "AT*FTRIM sent on UDP 5556. Drone should be stationary and on level ground.",
            )
        if action == "takeoff":
            client.takeoff()
            return RealDroneCommandResult(
                True,
                "Drone takeoff command sent.",
                action,
                "AT*REF takeoff was repeated on UDP 5556, followed by zero-motion hover commands.",
            )
        if action == "pause":
            client.hover()
            return RealDroneCommandResult(
                True,
                "Drone hover command sent.",
                action,
                "AT*PCMD zero-motion hover command repeated on UDP 5556.",
            )
        if action in nudge_actions:
            client.nudge(action)
            return RealDroneCommandResult(
                True,
                f"Drone {action.replace('_', ' ')} nudge sent.",
                action,
                "Short AT*PCMD movement pulse sent on UDP 5556, followed by zero-motion hover.",
            )
        if action == "scan":
            client.scan()
            return RealDroneCommandResult(
                True,
                "Drone scan pattern sent.",
                action,
                "Stationary yaw sweep sent as short AT*PCMD pulses, followed by zero-motion hover.",
            )
        if action == "assisted_search":
            client.assisted_search()
            return RealDroneCommandResult(
                True,
                "Drone autonomous search pattern sent.",
                action,
                "Conservative scan-forward-scan-back pattern sent as short AT*PCMD pulses, followed by zero-motion hover.",
            )
        if action == "land":
            client.land()
            return RealDroneCommandResult(
                True,
                "Drone land command sent.",
                action,
                "AT*REF land command repeated on UDP 5556.",
            )
        if action == "reset_emergency":
            client.reset_emergency()
            return RealDroneCommandResult(
                True,
                "Drone emergency state clear command sent.",
                action,
                "AT*REF emergency bit was repeated on UDP 5556 to clear the emergency latch.",
            )

        client.emergency_stop()
        return RealDroneCommandResult(
            True,
            "Drone emergency stop sent.",
            action,
            "AT*REF emergency sequence sent. This cuts motors and may cause a crash if airborne.",
            event_kind="drone_real_emergency",
        )
    except OSError as exc:
        return RealDroneCommandResult(
            False,
            f"Drone command failed: {action.replace('_', ' ')}.",
            action,
            str(exc),
            event_kind="drone_real_command_failed",
        )
