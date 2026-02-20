"""MIDI receiver — reads CC and note messages in a background thread."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

_REBIND_DELAY = 3.0

try:
    import mido
    _MIDO_AVAILABLE = True
except ImportError:
    _MIDO_AVAILABLE = False


class MIDIReceiver:
    """Non-blocking MIDI input that tracks CC values and note states.

    Runs a background thread that continuously reads MIDI messages.
    CC values are normalized to 0.0–1.0. Note states are True (held) / False (released).
    """

    def __init__(self, port_name: str | None = None) -> None:
        if not _MIDO_AVAILABLE:
            raise ImportError(
                "mido is required for MIDI support. Install with: pip install mido python-rtmidi"
            )

        self._cc: dict[int, float] = {}        # cc_number → 0.0–1.0
        self._notes: dict[int, bool] = {}       # note_number → held?
        self._lock = threading.Lock()
        self._port_name = port_name
        self._port: mido.ports.BaseInput | None = None
        self._running = True
        self._pending_port: str | None = None
        self._port_changed_at: float = 0.0

        self._open_port(port_name)
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _open_port(self, port_name: str | None) -> None:
        """Open a MIDI input port."""
        try:
            if self._port is not None:
                self._port.close()
        except Exception:
            pass

        try:
            if port_name and port_name != "auto":
                self._port = mido.open_input(port_name)
            else:
                self._port = mido.open_input()  # first available
            actual_name = getattr(self._port, "name", port_name or "default")
            print(f"[MIDI-RX] opened: {actual_name}", flush=True)
        except Exception:
            logger.exception("Failed to open MIDI port: %s", port_name)
            self._port = None

    def _read_loop(self) -> None:
        """Background thread: read MIDI messages and update state."""
        while self._running:
            if self._port is None:
                time.sleep(0.5)
                continue
            try:
                for msg in self._port.iter_pending():
                    self._handle_message(msg)
            except Exception:
                logger.exception("MIDI read error")
            time.sleep(0.005)  # ~200Hz poll rate

    def _handle_message(self, msg: mido.Message) -> None:
        with self._lock:
            if msg.type == "control_change":
                self._cc[msg.control] = msg.value / 127.0
            elif msg.type == "note_on":
                self._notes[msg.note] = msg.velocity > 0
            elif msg.type == "note_off":
                self._notes[msg.note] = False

    def get_cc(self, cc: int, default: float = 0.0) -> float:
        """Get the current value of a CC channel (0.0–1.0)."""
        with self._lock:
            return self._cc.get(cc, default)

    def get_note(self, note: int) -> bool:
        """Check if a note is currently held."""
        with self._lock:
            return self._notes.get(note, False)

    def get_all_cc(self) -> dict[int, float]:
        """Get a snapshot of all CC values."""
        with self._lock:
            return dict(self._cc)

    def get_all_notes(self) -> dict[int, bool]:
        """Get a snapshot of all note states."""
        with self._lock:
            return dict(self._notes)

    def update_port(self, port_name: str) -> None:
        """Switch to a different MIDI device (debounced — applies after 3s)."""
        current = self._port_name or "auto"
        if port_name == current:
            self._pending_port = None
            return
        now = time.monotonic()
        if port_name != self._pending_port:
            self._pending_port = port_name
            self._port_changed_at = now
            return
        if (now - self._port_changed_at) >= _REBIND_DELAY:
            print(f"[MIDI-RX] switching port: {current} → {port_name}", flush=True)
            self._port_name = port_name
            self._open_port(port_name)
            self._pending_port = None

    @staticmethod
    def list_ports() -> list[str]:
        """List available MIDI input port names."""
        if not _MIDO_AVAILABLE:
            return []
        try:
            return mido.get_input_names()
        except Exception:
            return []

    def close(self) -> None:
        self._running = False
        try:
            if self._port is not None:
                self._port.close()
        except Exception:
            pass
