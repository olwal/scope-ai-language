from typing import Literal

from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, UsageType, ui_field_config


class UDPPromptConfig(BasePipelineConfig):
    """Receives text via UDP and injects it as a prompt for downstream pipelines."""

    pipeline_id = "udp-prompt"
    pipeline_name = "UDP Prompt"
    pipeline_description = "Receives text via UDP and injects it as a prompt for downstream pipelines"

    supports_prompts = False

    modes = {"video": ModeDefaults(default=True)}

    usage = [UsageType.PREPROCESSOR]

    udp_port: int = Field(
        default=9400,
        ge=1024,
        le=65535,
        description="UDP channel port to listen on (must match the sender's port)",
        json_schema_extra=ui_field_config(
            order=1, label="UDP Port", is_load_param=True, category="configuration"
        ),
    )

    prompt_weight: float = Field(
        default=100.0,
        ge=0.0,
        le=200.0,
        description="Weight of the injected prompt",
        json_schema_extra=ui_field_config(order=2, label="Prompt Weight"),
    )

    transition_steps: int = Field(
        default=0, ge=0, le=30,
        description="Frames to blend from the current prompt to the new one (0 = instant)",
        json_schema_extra=ui_field_config(order=3, label="Transition Steps"),
    )

    interpolation_method: Literal["slerp", "linear"] = Field(
        default="slerp",
        description="Blending method for prompt transitions: slerp (smooth) or linear",
        json_schema_extra=ui_field_config(order=4, label="Interpolation"),
    )

    overlay_enabled: bool = Field(
        default=True,
        description="Overlay the current UDP text on the video for monitoring",
        json_schema_extra=ui_field_config(order=10, label="Show Overlay"),
    )

    font_size: int = Field(
        default=20,
        ge=8,
        le=72,
        description="Font size for the overlay text",
        json_schema_extra=ui_field_config(order=11, label="Font Size"),
    )

    text_opacity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overlay text opacity",
        json_schema_extra=ui_field_config(order=12, label="Opacity"),
    )

    bg_opacity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Background box opacity behind overlay text",
        json_schema_extra=ui_field_config(order=13, label="BG Opacity"),
    )
