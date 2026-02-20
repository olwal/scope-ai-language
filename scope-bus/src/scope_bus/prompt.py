"""Prompt injection helper — deduplicates and injects text prompts."""

from __future__ import annotations


class PromptInjector:
    """Injects text prompts into pipeline output dicts, only when the text changes."""

    def __init__(self) -> None:
        self._last_injected: str = ""

    def inject_if_new(
        self,
        output: dict,
        text: str,
        weight: float = 100.0,
        transition_steps: int = 0,
        interpolation_method: str = "slerp",
    ) -> None:
        """Inject a prompt into *output* only when *text* changes.

        If *transition_steps* > 0, sets ``output["transition"]`` for a smooth
        temporal blend from the current prompt to the new one. Otherwise appends
        directly to ``output["prompts"]``.

        Existing upstream prompts are preserved in both cases.
        Deduplication is based on the text value — repeated identical text is
        ignored.
        """
        if not text or text == self._last_injected:
            return

        self._last_injected = text
        prompt_entry = {"text": text, "weight": weight}

        if transition_steps > 0:
            output["transition"] = {
                "target_prompts": [prompt_entry],
                "num_steps": transition_steps,
                "temporal_interpolation_method": interpolation_method,
            }
        else:
            existing = list(output.get("prompts", []))
            existing.append(prompt_entry)
            output["prompts"] = existing
