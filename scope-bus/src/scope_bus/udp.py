"""UDP sender and receiver using multicast for fan-out to multiple receivers."""

from __future__ import annotations

import logging
import socket
import struct
import time

logger = logging.getLogger(__name__)

# Link-local multicast group — TTL=1 keeps traffic on the local machine only.
# All scope-bus senders and receivers use this group; the port is the channel.
_MULTICAST_GROUP = "239.255.42.99"
_MULTICAST_TTL = 1
_REBIND_DELAY = 3.0  # seconds to wait after port change before rebinding


class UDPSender:
    """Non-blocking UDP multicast sender with debounced port updates."""

    def __init__(self, port: int = 9400) -> None:
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, _MULTICAST_TTL)
        self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        self._pending_port: int | None = None
        self._port_changed_at: float = 0.0
        print(f"[UDP-TX] ready → {_MULTICAST_GROUP}:{port}", flush=True)

    @property
    def port(self) -> int:
        return self._port

    def send(self, message: "str | dict") -> None:
        """Broadcast a message to all receivers on this port.

        Accepts a plain string or a dict (serialised as JSON).
        """
        import json
        if isinstance(message, dict):
            payload = json.dumps(message).encode("utf-8")
        else:
            payload = str(message).encode("utf-8")
        try:
            self._sock.sendto(payload, (_MULTICAST_GROUP, self._port))
        except Exception:
            logger.exception("UDP send failed")

    def update_port(self, new_port: int) -> None:
        """Update channel port (debounced — applies after 3s of stable value)."""
        if new_port == self._port:
            self._pending_port = None
            return
        now = time.monotonic()
        if new_port != self._pending_port:
            self._pending_port = new_port
            self._port_changed_at = now
            return
        if (now - self._port_changed_at) >= _REBIND_DELAY:
            print(f"[UDP-TX] port changed {self._port} → {new_port}", flush=True)
            self._port = new_port
            self._pending_port = None

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


class UDPReceiver:
    """Non-blocking UDP multicast receiver with debounced port rebinding.

    Multiple receivers can listen on the same port — all receive every message
    sent to that port via the shared multicast group.
    """

    def __init__(self, port: int = 9400) -> None:
        self._port = port
        self._sock = self._bind(port)
        self._pending_port: int | None = None
        self._port_changed_at: float = 0.0

    @property
    def port(self) -> int:
        return self._port

    def _bind(self, port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", port))
        mreq = struct.pack("4sL", socket.inet_aton(_MULTICAST_GROUP), socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.setblocking(False)
        print(f"[UDP-RX] joined {_MULTICAST_GROUP}:{port}", flush=True)
        return sock

    def poll(self) -> "str | dict | None":
        """Drain the socket and return the latest message, or None.

        If the message is valid JSON, returns the parsed object (dict).
        Otherwise returns the raw string.
        """
        import json
        latest: "str | dict | None" = None
        while True:
            try:
                data, _ = self._sock.recvfrom(65535)
                text = data.decode("utf-8", errors="replace")
                try:
                    latest = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    latest = text
            except BlockingIOError:
                break
        return latest

    def update_port(self, new_port: int) -> None:
        """Rejoin on a new port (debounced — applies after 3s of stable value)."""
        if new_port == self._port:
            self._pending_port = None
            return
        now = time.monotonic()
        if new_port != self._pending_port:
            self._pending_port = new_port
            self._port_changed_at = now
            return
        if (now - self._port_changed_at) >= _REBIND_DELAY:
            print(f"[UDP-RX] port changed {self._port} → {new_port}, rejoining...", flush=True)
            try:
                self._sock.close()
            except Exception:
                pass
            self._port = new_port
            self._sock = self._bind(new_port)
            self._pending_port = None

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass
