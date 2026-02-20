"""Ollama Vision Language Model client — sends frames, gets text descriptions."""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from typing import Callable

import httpx
import torch
from scope_bus import tensor_to_pil

logger = logging.getLogger(__name__)


class OllamaVLM:
    """Async Ollama VLM client that queries a vision model in a background thread.

    Usage::

        vlm = OllamaVLM(url="http://localhost:11434", model="llava:7b")

        # In your pipeline's __call__:
        if vlm.should_send(interval=3.0):
            vlm.query_async(frame_tensor, prompt="Describe this image.")

        text = vlm.get_last_response()
    """

    def __init__(self, url: str = "http://localhost:11434", model: str = "llava:7b") -> None:
        self._url = url
        self._model = model
        self._client = httpx.Client(base_url=url, timeout=60.0)
        self._last_response: str = ""
        self._last_send_time: float = 0.0
        self._pending: bool = False
        self._lock = threading.Lock()
        self._callback: Callable[[str], None] | None = None

    def should_send(self, interval: float) -> bool:
        """Check if enough time has elapsed since the last query."""
        if self._pending:
            return False
        return (time.monotonic() - self._last_send_time) >= interval

    def query_async(
        self,
        frame: torch.Tensor,
        prompt: str,
        callback: Callable[[str], None] | None = None,
    ) -> None:
        """Send a frame to the VLM in a background thread.

        Args:
            frame: A single (H, W, C) float32 tensor in [0, 1].
            prompt: The text prompt to send alongside the image.
            callback: Optional function called with the response text.
        """
        self._last_send_time = time.monotonic()
        self._pending = True
        self._callback = callback
        image = tensor_to_pil(frame)
        threading.Thread(target=self._query, args=(image, prompt), daemon=True).start()

    def _query(self, image, prompt: str) -> None:
        try:
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            response = self._client.post(
                "/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False,
                },
            )
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            logger.info("VLM response: %s", result)
            print(f"[VLM] {result}", flush=True)
            with self._lock:
                self._last_response = result
            if self._callback:
                self._callback(result)
        except Exception:
            logger.exception("Failed to query Ollama VLM")
        finally:
            self._pending = False

    def get_last_response(self) -> str:
        """Return the most recent VLM response text."""
        with self._lock:
            return self._last_response

    @property
    def is_pending(self) -> bool:
        return self._pending
