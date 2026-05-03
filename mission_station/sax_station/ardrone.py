from __future__ import annotations

from dataclasses import dataclass
import socket
from typing import Any


DEFAULT_DRONE_HOST = "192.168.1.1"
VIDEO_PORT = 5555
NAVDATA_PORT = 5554
CONTROL_PORT = 5556


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    detail: str


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


def probe_drone(host: str = DEFAULT_DRONE_HOST) -> dict[str, Any]:
    local_address = local_address_for(host)
    likely_on_drone_network = bool(local_address and local_address.startswith("192.168.1."))
    video = probe_tcp_port(host, VIDEO_PORT)
    navdata = probe_udp_socket(host, NAVDATA_PORT)
    control = probe_udp_socket(host, CONTROL_PORT)
    return {
        "host": host,
        "local_address": local_address,
        "likely_on_drone_network": likely_on_drone_network,
        "video_source": video_source_url(host),
        "video": video,
        "navdata": navdata,
        "control": control,
        "ready_for_video": video.ok,
        "ready_for_control": likely_on_drone_network and navdata.ok and control.ok,
    }
