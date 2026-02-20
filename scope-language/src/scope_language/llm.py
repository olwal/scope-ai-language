"""Ollama Language Model client — sends text prompts, gets text responses."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from ._base import _OllamaBase

logger = logging.getLogger(__name__)


class OllamaLLM(_OllamaBase):
    """Async Ollama LLM client that queries a text model in a background thread.

    Usage::

        llm = OllamaLLM(url="http://localhost:11434", model="llama3.2:3b")

        # In your pipeline's __call__:
        if llm.should_send(interval=3.0):
            llm.query_async(prompt="Rewrite this: a cat on a couch")

        text = llm.get_last_response()
    """

    def __init__(self, url: str = "http://localhost:11434", model: str = "llama3.2:3b") -> None:
        super().__init__(url=url, model=model)

    def query_async(
        self,
        prompt: str,
        system: str = "",
        callback: Callable[[str], None] | None = None,
    ) -> None:
        """Send a text prompt to the LLM in a background thread.

        Args:
            prompt: The user prompt to send.
            system: Optional system prompt for context/personality.
            callback: Optional function called with the response text.
        """
        self._last_send_time = time.monotonic()
        self._pending = True
        self._callback = callback
        threading.Thread(target=self._query, args=(prompt, system), daemon=True).start()

    def _query(self, prompt: str, system: str) -> None:
        try:
            payload: dict = {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
            }
            if system:
                payload["system"] = system
            response = self._client.post("/api/generate", json=payload)
            response.raise_for_status()
            result = response.json().get("response", "").strip()
            logger.info("LLM response: %s", result)
            print(f"[LLM] {result}", flush=True)
            with self._lock:
                self._last_response = result
            if self._callback:
                self._callback(result)
        except Exception:
            logger.exception("Failed to query Ollama LLM")
        finally:
            self._pending = False
