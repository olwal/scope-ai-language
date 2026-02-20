from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements
from scope_bus import UDPReceiver, normalize_input, render_text_overlay

from .schema import UDPPromptConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

_OWN_KEYS = {"udp_port", "prompt_weight", "overlay_enabled", "font_size", "text_opacity", "bg_opacity"}


class UDPPromptPipeline(Pipeline):
    """Receives text via UDP and injects it as a prompt for downstream pipelines."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return UDPPromptConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._last_text: str = ""
        self._udp = UDPReceiver(port=kwargs.get("udp_port", 9400))

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("UDPPromptPipeline requires video input")

        frames = normalize_input(video, self.device)

        msg = self._udp.poll()
        if msg is not None:
            self._last_text = msg

        # Overlay current text on video if enabled
        if kwargs.get("overlay_enabled", True) and self._last_text:
            frames = render_text_overlay(
                frames,
                text=self._last_text,
                font_family="arial",
                font_size=kwargs.get("font_size", 20),
                font_color=(1.0, 1.0, 0.4),  # yellow tint to distinguish from VLM overlay
                opacity=kwargs.get("text_opacity", 1.0),
                position="top-left",
                word_wrap=True,
                bg_opacity=kwargs.get("bg_opacity", 0.5),
            )

        # Forward all non-own kwargs
        output: dict = {"video": frames.clamp(0, 1)}
        for key, value in kwargs.items():
            if key != "video" and key not in _OWN_KEYS:
                output[key] = value

        # Inject UDP text as prompt (appended to any existing prompts)
        if self._last_text:
            weight = kwargs.get("prompt_weight", 100.0)
            existing = list(output.get("prompts", []))
            existing.append({"text": self._last_text, "weight": weight})
            output["prompts"] = existing

        return output

    def __del__(self):
        self._udp.close()
