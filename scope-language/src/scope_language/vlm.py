"""Ollama Vision Language Model client — sends frames, gets text descriptions."""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from typing import Callable

import torch
from scope_bus import tensor_to_pil

from ._base import _OllamaBase

logger = logging.getLogger(__name__)


class OllamaVLM(_OllamaBase):
    """Async Ollama VLM client that queries a vision model in a background thread.

    Usage::

        vlm = OllamaVLM(url="http://localhost:11434", model="llava:7b")

        # In your pipeline's __call__:
        if vlm.should_send(interval=3.0):
            vlm.query_async(frame_tensor, prompt="Describe this image.")

        text = vlm.get_last_response()
    """

    def __init__(self, url: str = "http://localhost:11434", model: str = "llava:7b") -> None:
        super().__init__(url=url, model=model)

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
