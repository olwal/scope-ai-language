from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements
from scope_bus import UDPReceiver, normalize_input

from .overlay import render_debug_overlay
from .schema import TextLogConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

# Keys that are part of the monitor's own config — exclude from "all keys" display
_OWN_KEYS = {
    "overlay_enabled", "print_to_stdout", "log_interval", "show_video_shape",
    "show_prompts", "show_all_keys", "font_size", "text_opacity",
    "text_position", "bg_opacity", "max_value_length",
    "udp_enabled", "udp_port",
}


class TextLogPipeline(Pipeline):
    """Debug postprocessor: logs and overlays all kwargs flowing through the pipeline."""

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return TextLogConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._last_print_time: float = 0.0
        self._last_udp_message: str = ""

        # UDP receiver
        self._udp: UDPReceiver | None = None
        if kwargs.get("udp_enabled", True):
            try:
                self._udp = UDPReceiver(port=kwargs.get("udp_port", 9400))
            except Exception as e:
                print(f"[TEXT-LOG] UDP bind failed: {e}", flush=True)

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("TextLogPipeline requires video input")

        frames = normalize_input(video, self.device)

        # Drain UDP messages, keep latest
        if self._udp is not None:
            msg = self._udp.poll()
            if msg is not None:
                self._last_udp_message = msg

        # Read own config params
        overlay_enabled = kwargs.get("overlay_enabled", True)
        print_to_stdout = kwargs.get("print_to_stdout", True)
        log_interval = kwargs.get("log_interval", 1.0)
        show_video_shape = kwargs.get("show_video_shape", True)
        show_prompts = kwargs.get("show_prompts", True)
        show_all_keys = kwargs.get("show_all_keys", True)
        font_size = kwargs.get("font_size", 16)
        text_opacity = kwargs.get("text_opacity", 1.0)
        text_position = kwargs.get("text_position", "top-left")
        bg_opacity = kwargs.get("bg_opacity", 0.7)
        max_value_length = kwargs.get("max_value_length", 80)

        # Build debug info lines
        lines = _build_debug_lines(
            kwargs, frames, show_video_shape, show_prompts, show_all_keys, max_value_length,
        )

        # Add UDP message if present
        if self._last_udp_message:
            lines.insert(0, f"UDP: {self._last_udp_message}")

        # Print to stdout (throttled)
        now = time.monotonic()
        if print_to_stdout and (now - self._last_print_time) >= log_interval:
            self._last_print_time = now
            header = "=" * 60
            print(f"\n{header}", flush=False)
            print(f"[TEXT-LOG] kwargs snapshot @ {time.strftime('%H:%M:%S')}", flush=False)
            print(header, flush=False)
            for line in lines:
                print(f"  {line}", flush=False)
            print(header, flush=True)

        # Overlay on video
        if overlay_enabled and lines:
            frames = render_debug_overlay(
                frames,
                lines=lines,
                font_size=font_size,
                opacity=text_opacity,
                position=text_position,
                bg_opacity=bg_opacity,
            )

        # Pass through video + forward ALL non-video kwargs so downstream still gets them
        output = {"video": frames.clamp(0, 1)}

        for key, value in kwargs.items():
            if key != "video" and key not in _OWN_KEYS:
                output[key] = value

        return output


def _build_debug_lines(
    kwargs: dict,
    frames: torch.Tensor,
    show_video_shape: bool,
    show_prompts: bool,
    show_all_keys: bool,
    max_len: int,
) -> list[str]:
    """Build a list of debug info lines from kwargs."""
    lines: list[str] = []

    if show_video_shape:
        lines.append(f"video: shape={list(frames.shape)} dtype={frames.dtype}")

    if show_prompts:
        prompts = kwargs.get("prompts")
        if prompts:
            lines.append(f"prompts: ({len(prompts)} entries)")
            for i, p in enumerate(prompts):
                if isinstance(p, dict):
                    text = p.get("text", "")
                    weight = p.get("weight", "?")
                    display = text[:max_len] + "..." if len(text) > max_len else text
                    lines.append(f"  [{i}] w={weight} | {display}")
                else:
                    display = str(p)[:max_len]
                    lines.append(f"  [{i}] {display}")
        else:
            lines.append("prompts: (none)")

    if show_all_keys:
        skip = {"video", "prompts"} | _OWN_KEYS
        other_keys = sorted(k for k in kwargs if k not in skip)
        if other_keys:
            lines.append(f"--- other keys ({len(other_keys)}) ---")
            for key in other_keys:
                val = kwargs[key]
                val_str = _format_value(val, max_len)
                lines.append(f"{key}: {val_str}")

    return lines


def _format_value(value: object, max_len: int) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor shape={list(value.shape)} dtype={value.dtype}"
    if isinstance(value, (list, tuple)) and len(value) > 5:
        return f"{type(value).__name__}[{len(value)}] = {str(value[:3])[:max_len]}..."
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s
