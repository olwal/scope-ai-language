"""OSC Prompt pipeline — receives OSC /prompt messages and injects them as diffusion prompts."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import torch
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from scope.core.pipelines.interface import Pipeline, Requirements
from scope_bus import PromptInjector, normalize_input, render_text_overlay

from .schema import OSCPromptConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

# Keys consumed by this plugin — not forwarded downstream
_OWN_KEYS = {"osc_port", "prompt_weight", "transition_steps", "interpolation_method", "overlay_enabled", "font_size", "text_opacity", "bg_opacity"}


class OSCPromptPipeline(Pipeline):
    """Listens for OSC /prompt messages and injects the text as a diffusion prompt.

    Compatible with TouchDesigner, Ableton Live, Max/MSP, and any tool that sends OSC.

    Send to:  /prompt <string>
    Example:  /prompt "a misty forest at dawn, painterly"
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return OSCPromptConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._last_text: str = ""
        self._lock = threading.Lock()
        self._prompt = PromptInjector()

        # Start OSC server in a background daemon thread (load-time port only)
        osc_port = kwargs.get("osc_port", 9000)
        dispatcher = Dispatcher()
        dispatcher.map("/prompt", self._on_prompt)
        self._server = ThreadingOSCUDPServer(("0.0.0.0", osc_port), dispatcher)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"[OSC-PROMPT] Listening on port {osc_port} — send to /prompt <string>", flush=True)

    def _on_prompt(self, address: str, *args) -> None:
        """Called by the OSC server thread when a /prompt message arrives."""
        text = str(args[0]).strip() if args else ""
        if text:
            with self._lock:
                self._last_text = text
            print(f"[OSC-PROMPT] Received: {text[:80]}", flush=True)

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("OSCPromptPipeline requires video input")

        frames = normalize_input(video, self.device)

        with self._lock:
            text = self._last_text

        # Overlay current text on video if enabled (yellow tint)
        if kwargs.get("overlay_enabled", True) and text:
            frames = render_text_overlay(
                frames,
                text=text,
                font_family="arial",
                font_size=kwargs.get("font_size", 20),
                font_color=(1.0, 1.0, 0.4),  # yellow — matches scope-udp-prompt
                opacity=kwargs.get("text_opacity", 1.0),
                position="top-left",
                word_wrap=True,
                bg_opacity=kwargs.get("bg_opacity", 0.5),
            )

        # Forward all non-own kwargs downstream
        output: dict = {"video": frames.clamp(0, 1)}
        for key, value in kwargs.items():
            if key != "video" and key not in _OWN_KEYS:
                output[key] = value

        # Inject OSC text as prompt (with optional transition)
        if text:
            self._prompt.inject_if_new(
                output, text,
                weight=kwargs.get("prompt_weight", 100.0),
                transition_steps=kwargs.get("transition_steps", 0),
                interpolation_method=kwargs.get("interpolation_method", "slerp"),
            )

        return output

    def __del__(self):
        if hasattr(self, "_server"):
            self._server.shutdown()
