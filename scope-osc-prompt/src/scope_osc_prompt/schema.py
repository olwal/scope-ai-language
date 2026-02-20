from typing import Literal

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    UsageType,
    ui_field_config,
)


class OSCPromptConfig(BasePipelineConfig):
    """Receives OSC messages on /prompt and injects them as diffusion prompts."""

    pipeline_id = "osc-prompt"
    pipeline_name = "OSC Prompt"
    pipeline_description = (
        "Receives OSC messages on /prompt and injects the text as a diffusion prompt. "
        "Compatible with TouchDesigner, Ableton, Max/MSP, and any OSC-capable tool."
    )

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.PREPROCESSOR]

    osc_port: int = Field(
        default=9000,
        ge=1024,
        le=65535,
        description="UDP port to listen for OSC messages on (load-time — restart required to change)",
        json_schema_extra=ui_field_config(order=10, label="OSC Port", is_load_param=True),
    )

    prompt_weight: float = Field(
        default=100.0,
        ge=0.0,
        le=200.0,
        description="Weight of the injected prompt relative to other prompts in the chain",
        json_schema_extra=ui_field_config(order=20, label="Prompt Weight"),
    )

    transition_steps: int = Field(
        default=0, ge=0, le=30,
        description="Frames to blend from the current prompt to the new one (0 = instant)",
        json_schema_extra=ui_field_config(order=22, label="Transition Steps"),
    )

    interpolation_method: Literal["slerp", "linear"] = Field(
        default="slerp",
        description="Blending method for prompt transitions: slerp (smooth) or linear",
        json_schema_extra=ui_field_config(order=24, label="Interpolation"),
    )

    overlay_enabled: bool = Field(
        default=True,
        description="Show the received OSC text as an overlay on the video (for monitoring)",
        json_schema_extra=ui_field_config(order=30, label="Show Overlay"),
    )

    font_size: int = Field(
        default=20,
        ge=8,
        le=64,
        description="Font size of the overlay text",
        json_schema_extra=ui_field_config(order=40, label="Font Size"),
    )

    text_opacity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Opacity of the overlay text",
        json_schema_extra=ui_field_config(order=50, label="Text Opacity"),
    )

    bg_opacity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Opacity of the background box behind the overlay text",
        json_schema_extra=ui_field_config(order=60, label="BG Opacity"),
    )
