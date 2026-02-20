"""Shared base for Ollama API clients."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import httpx

logger = logging.getLogger(__name__)


class _OllamaBase:
    """Shared state and behaviour for Ollama VLM and LLM clients."""

    def __init__(self, url: str = "http://localhost:11434", model: str = "") -> None:
        self._url = url
        self._model = model
        self._client = httpx.Client(base_url=url, timeout=60.0)
        self._last_response: str = ""
        self._last_send_time: float = 0.0
        self._pending: bool = False
        self._lock = threading.Lock()
        self._callback: Callable[[str], None] | None = None

    def should_send(self, interval: float) -> bool:
        """Return True if enough time has elapsed since the last query."""
        if self._pending:
            return False
        return (time.monotonic() - self._last_send_time) >= interval

    def get_last_response(self) -> str:
        """Return the most recent response text."""
        with self._lock:
            return self._last_response

    @property
    def is_pending(self) -> bool:
        return self._pending
