from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements
from scope_bus import PromptInjector, UDPReceiver, UDPSender, normalize_input, render_text_overlay
from scope_language import OllamaVLM

from .schema import VLMOllamaConfig, VLMOllamaPostConfig, VLMOllamaPreConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_overlay(frames: torch.Tensor, text: str, kwargs: dict) -> torch.Tensor:
    return render_text_overlay(
        frames,
        text=text,
        font_family=kwargs.get("font_family", "arial"),
        font_size=kwargs.get("font_size", 24),
        font_color=(
            kwargs.get("font_color_r", 1.0),
            kwargs.get("font_color_g", 1.0),
            kwargs.get("font_color_b", 1.0),
        ),
        opacity=kwargs.get("text_opacity", 1.0),
        position=kwargs.get("text_position", "bottom-left"),
        word_wrap=kwargs.get("word_wrap", True),
        bg_opacity=kwargs.get("bg_opacity", 0.5),
    )


# ---------------------------------------------------------------------------
# Standalone pipeline: VLM + overlay in one place
# ---------------------------------------------------------------------------

class VLMOllamaPipeline(Pipeline):
    """Standalone: queries Ollama VLM, overlays response, optionally injects prompt."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return VLMOllamaConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._vlm = OllamaVLM(
            url=kwargs.get("ollama_url", "http://localhost:11434"),
            model=kwargs.get("ollama_model", "llava:7b"),
        )
        self._prompt = PromptInjector()

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("VLMOllamaPipeline requires video input")

        frames = normalize_input(video, self.device)

        interval = kwargs.get("send_interval", 3.0)
        if self._vlm.should_send(interval):
            self._vlm.query_async(
                frames[0],
                prompt=kwargs.get("vlm_prompt", "Describe what you see in this image in one sentence."),
            )

        response_text = self._vlm.get_last_response()

        if kwargs.get("overlay_enabled", True) and response_text:
            frames = _apply_overlay(frames, response_text, kwargs)

        output = {"video": frames.clamp(0, 1)}
        if kwargs.get("inject_prompt", True):
            self._prompt.inject_if_new(output, response_text, kwargs.get("prompt_weight", 100.0))
        return output


# ---------------------------------------------------------------------------
# Preprocessor: VLM + prompt injection + UDP broadcast (no overlay)
# ---------------------------------------------------------------------------

class VLMOllamaPrePipeline(Pipeline):
    """Preprocessor: raw camera → Ollama VLM → inject prompt + UDP broadcast."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return VLMOllamaPreConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._vlm = OllamaVLM(
            url=kwargs.get("ollama_url", "http://localhost:11434"),
            model=kwargs.get("ollama_model", "llava:7b"),
        )
        self._udp = UDPSender(port=kwargs.get("udp_port", 9400))
        self._prompt = PromptInjector()

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("VLMOllamaPrePipeline requires video input")

        self._udp.update_port(kwargs.get("udp_port", self._udp.port))

        frames = normalize_input(video, self.device)

        interval = kwargs.get("send_interval", 3.0)
        if self._vlm.should_send(interval):
            self._vlm.query_async(
                frames[0],
                prompt=kwargs.get("vlm_prompt", "Describe what you see in this image in one sentence."),
                callback=self._on_vlm_response,
            )

        response_text = self._vlm.get_last_response()

        output = {"video": frames.clamp(0, 1)}
        if kwargs.get("inject_prompt", True):
            self._prompt.inject_if_new(output, response_text, kwargs.get("prompt_weight", 100.0))
        return output

    def _on_vlm_response(self, text: str) -> None:
        self._udp.send(text)
        print(f"[VLM-PRE] UDP sent ({len(text)} bytes) → port {self._udp.port}", flush=True)


# ---------------------------------------------------------------------------
# Postprocessor: receives text via UDP + overlays on AI-processed output
# ---------------------------------------------------------------------------

class VLMOllamaPostPipeline(Pipeline):
    """Postprocessor: receives VLM text via UDP and overlays on AI-processed video."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return VLMOllamaPostConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._last_text: str = ""
        self._udp = UDPReceiver(port=kwargs.get("udp_port", 9400))

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("VLMOllamaPostPipeline requires video input")

        self._udp.update_port(kwargs.get("udp_port", self._udp.port))

        frames = normalize_input(video, self.device)

        # Check for new UDP message
        msg = self._udp.poll()
        if msg is not None:
            self._last_text = msg

        if kwargs.get("overlay_enabled", True) and self._last_text:
            frames = _apply_overlay(frames, self._last_text, kwargs)

        return {"video": frames.clamp(0, 1)}

    def __del__(self):
        self._udp.close()
