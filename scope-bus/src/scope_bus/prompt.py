"""Prompt injection helper — deduplicates and injects text prompts."""

from __future__ import annotations


class PromptInjector:
    """Injects text prompts into pipeline output dicts, only when the text changes."""

    def __init__(self) -> None:
        self._last_injected: str = ""

    def inject_if_new(self, output: dict, text: str, weight: float = 100.0) -> None:
        """Append a prompt to ``output["prompts"]`` only when *text* changes.

        Existing upstream prompts are preserved. Deduplication is based on the
        text value — if the same text arrives again, nothing is added.
        """
        if text and text != self._last_injected:
            self._last_injected = text
            existing = list(output.get("prompts", []))
            existing.append({"text": text, "weight": weight})
            output["prompts"] = existing
