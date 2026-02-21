from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements
from scope_bus import PromptInjector, UDPReceiver, UDPSender, apply_overlay_from_kwargs, normalize_input, render_text_overlay
from scope_language import OllamaVLM

from .schema import VLMOllamaConfig, VLMOllamaPostConfig, VLMOllamaPreConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

_DEFAULT_PROMPT = "Describe what you see in this image in one sentence."


def _settle_prompt(new_prompt: str, pending: str, active: str, changed_at: float, settle_time: float) -> tuple[str, str, float]:
    """Debounce prompt changes. Returns (active_prompt, pending_prompt, changed_at)."""
    now = time.monotonic()
    if new_prompt != pending:
        return active, new_prompt, now
    if settle_time == 0.0 or (now - changed_at) >= settle_time:
        return pending, pending, changed_at
    return active, pending, changed_at


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
        self._active_prompt: str = kwargs.get("vlm_prompt", _DEFAULT_PROMPT)
        self._pending_prompt: str = self._active_prompt
        self._prompt_changed_at: float = 0.0
        self._last_queried_prompt: str = ""

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("VLMOllamaPipeline requires video input")

        frames = normalize_input(video, self.device)

        # Debounce prompt — wait until typing settles before sending to VLM
        self._active_prompt, self._pending_prompt, self._prompt_changed_at = _settle_prompt(
            kwargs.get("vlm_prompt", _DEFAULT_PROMPT),
            self._pending_prompt, self._active_prompt, self._prompt_changed_at,
            kwargs.get("prompt_settle_time", 1.0),
        )

        # Fire immediately when prompt changed (as soon as VLM is free), else use interval
        interval = kwargs.get("send_interval", 3.0)
        prompt_needs_send = self._active_prompt != self._last_queried_prompt
        if not self._vlm.is_pending and (prompt_needs_send or self._vlm.should_send(interval)):
            self._last_queried_prompt = self._active_prompt
            self._vlm.query_async(frames[0], prompt=self._active_prompt)

        response_text = self._vlm.get_last_response()

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
        self._udp = UDPSender(port=kwargs.get("udp_port", 9500))
        self._prompt = PromptInjector()
        self._last_sent_prompt: str = ""  # captured at query time for the UDP callback
        self._active_prompt: str = kwargs.get("vlm_prompt", _DEFAULT_PROMPT)
        self._pending_prompt: str = self._active_prompt
        self._prompt_changed_at: float = 0.0
        self._last_queried_prompt: str = ""

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("VLMOllamaPrePipeline requires video input")

        self._udp.update_port(kwargs.get("udp_port", self._udp.port))

        frames = normalize_input(video, self.device)

        # Debounce prompt — wait until typing settles before sending to VLM
        self._active_prompt, self._pending_prompt, self._prompt_changed_at = _settle_prompt(
            kwargs.get("vlm_prompt", _DEFAULT_PROMPT),
            self._pending_prompt, self._active_prompt, self._prompt_changed_at,
            kwargs.get("prompt_settle_time", 1.0),
        )

        # Fire immediately when prompt changed (as soon as VLM is free), else use interval
        interval = kwargs.get("send_interval", 3.0)
        prompt_needs_send = self._active_prompt != self._last_queried_prompt
        if not self._vlm.is_pending and (prompt_needs_send or self._vlm.should_send(interval)):
            self._last_queried_prompt = self._active_prompt
            self._last_sent_prompt = self._active_prompt
            self._vlm.query_async(
                frames[0],
                prompt=self._last_sent_prompt,
                callback=self._on_vlm_response,
            )

        response_text = self._vlm.get_last_response()

        output = {"video": frames.clamp(0, 1)}
        if kwargs.get("inject_prompt", True):
            self._prompt.inject_if_new(
                output, response_text,
                weight=kwargs.get("prompt_weight", 100.0),
                transition_steps=kwargs.get("transition_steps", 0),
                interpolation_method=kwargs.get("interpolation_method", "slerp"),
            )
        return output

    def _on_vlm_response(self, text: str) -> None:
        # Send both the question and answer so the postprocessor can display both
        self._udp.send({"prompt": self._last_sent_prompt, "response": text})
        print(f"[VLM-PRE] UDP sent ({len(text)} chars) → port {self._udp.port}", flush=True)


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
        self._last_prompt: str = ""
        self._last_response: str = ""
        self._udp = UDPReceiver(port=kwargs.get("udp_port", 9500))

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("VLMOllamaPostPipeline requires video input")

        self._udp.update_port(kwargs.get("udp_port", self._udp.port))

        frames = normalize_input(video, self.device)

        # Check for new UDP message — may be a JSON dict {prompt, response} or plain string
        msg = self._udp.poll()
        if isinstance(msg, dict):
            self._last_prompt = msg.get("prompt", "")
            self._last_response = msg.get("response", "")
        elif isinstance(msg, str):
            self._last_response = msg

        if kwargs.get("overlay_enabled", True):
            # Prompt question at the top (dimmer, smaller)
            if self._last_prompt:
                frames = render_text_overlay(
                    frames,
                    text=self._last_prompt,
                    font_family=kwargs.get("font_family", "arial"),
                    font_size=max(8, kwargs.get("font_size", 24) - 6),
                    font_color=(0.7, 0.85, 1.0),  # light blue — visually distinct from response
                    opacity=kwargs.get("text_opacity", 1.0) * 0.75,
                    position="top-left",
                    word_wrap=True,
                    bg_opacity=kwargs.get("bg_opacity", 0.5),
                )
            # VLM response at the bottom (full size, full opacity)
            if self._last_response:
                frames = apply_overlay_from_kwargs(frames, self._last_response, kwargs)

        return {"video": frames.clamp(0, 1)}

    def __del__(self):
        self._udp.close()
