"""Scope plugin registration for scope-bus.

Registers a passthrough preprocessor so the library is visible and
uninstallable in the Scope UI. It does nothing except pass video through.
"""

from __future__ import annotations

import torch

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults
from scope.core.pipelines.interface import Pipeline, Requirements

from .frames import normalize_input


class ScopeBusConfig(BasePipelineConfig):
    pipeline_id = "scope-bus-library"
    pipeline_name = "scope-bus · Library"
    pipeline_description = (
        "Shared transport and utility library used by other plugins. "
        "Passthrough — safe to ignore. Install/uninstall to manage the dependency."
    )
    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}


class ScopeBusPipeline(Pipeline):
    """Passthrough pipeline — exists only to make scope-bus visible in the Scope UI."""

    @classmethod
    def get_config_class(cls) -> type[BasePipelineConfig]:
        return ScopeBusConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def prepare(self, **kwargs) -> Requirements:
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("scope-bus library pipeline requires video input")
        frames = normalize_input(video, self.device)
        return {"video": frames.clamp(0, 1)}
