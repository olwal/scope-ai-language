from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements
from scope_bus import PromptInjector, UDPSender, apply_overlay_from_kwargs, normalize_input
from scope_language import OllamaLLM

from .schema import LLMOllamaConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class LLMOllamaPipeline(Pipeline):
    """Preprocessor: sends text to Ollama LLM → injects response as prompt + overlay."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return LLMOllamaConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._llm = OllamaLLM(
            url=kwargs.get("ollama_url", "http://localhost:11434"),
            model=kwargs.get("ollama_model", "llama3.2:3b"),
        )
        self._udp = UDPSender(port=kwargs.get("udp_port", 9400))
        self._prompt = PromptInjector()

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("LLMOllamaPipeline requires video input")

        frames = normalize_input(video, self.device)

        self._udp.update_port(kwargs.get("udp_port", self._udp.port))

        # Send to LLM if interval has elapsed
        interval = kwargs.get("send_interval", 5.0)
        if self._llm.should_send(interval):
            input_text = kwargs.get("input_prompt", "a dreamy landscape")
            system = kwargs.get("system_prompt", "")
            self._llm.query_async(
                prompt=input_text,
                system=system,
                callback=self._on_llm_response if kwargs.get("udp_enabled", False) else None,
            )

        response_text = self._llm.get_last_response()

        if kwargs.get("overlay_enabled", True) and response_text:
            frames = apply_overlay_from_kwargs(frames, response_text, kwargs)

        output = {"video": frames.clamp(0, 1)}

        if kwargs.get("inject_prompt", True):
            self._prompt.inject_if_new(
                output, response_text,
                weight=kwargs.get("prompt_weight", 100.0),
                transition_steps=kwargs.get("transition_steps", 0),
                interpolation_method=kwargs.get("interpolation_method", "slerp"),
            )

        return output

    def _on_llm_response(self, text: str) -> None:
        """Broadcast LLM response via UDP."""
        self._udp.send(text)
        print(f"[LLM] UDP sent ({len(text)} bytes) → port {self._udp.port}", flush=True)
