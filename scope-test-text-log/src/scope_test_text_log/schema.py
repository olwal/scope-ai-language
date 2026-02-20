from enum import Enum

from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, UsageType, ui_field_config


class TextPosition(str, Enum):
    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"


class TextLogConfig(BasePipelineConfig):
    """Configuration for the Text Log postprocessor."""

    pipeline_id = "text-log"
    pipeline_name = "Text Log"
    pipeline_description = "Debug tool: logs all kwargs (prompts, custom fields, etc.) and overlays them on video"

    supports_prompts = False

    modes = {"video": ModeDefaults(default=True)}

    usage = [UsageType.POSTPROCESSOR]

    # --- UDP Receiver (load-time) ---

    udp_enabled: bool = Field(
        default=True,
        description="Listen for incoming UDP messages from other plugins",
        json_schema_extra=ui_field_config(
            order=0, label="UDP Enabled", is_load_param=True, category="configuration"
        ),
    )

    udp_port: int = Field(
        default=9400,
        ge=1024,
        le=65535,
        description="UDP channel port to listen on (must match the sender's port)",
        json_schema_extra=ui_field_config(
            order=0, label="UDP Port", is_load_param=True, category="configuration"
        ),
    )

    # --- Display ---

    overlay_enabled: bool = Field(
        default=True,
        description="Overlay the debug info on the output video",
        json_schema_extra=ui_field_config(order=1, label="Overlay"),
    )

    print_to_stdout: bool = Field(
        default=True,
        description="Print all kwargs to stdout on each frame",
        json_schema_extra=ui_field_config(order=2, label="Print to stdout"),
    )

    log_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Minimum seconds between stdout prints (avoids flooding)",
        json_schema_extra=ui_field_config(order=3, label="Log Interval"),
    )

    show_video_shape: bool = Field(
        default=True,
        description="Show the video tensor shape and dtype in the overlay",
        json_schema_extra=ui_field_config(order=4, label="Show Video Shape"),
    )

    show_prompts: bool = Field(
        default=True,
        description="Show prompts if present in kwargs",
        json_schema_extra=ui_field_config(order=5, label="Show Prompts"),
    )

    show_all_keys: bool = Field(
        default=True,
        description="Show all non-video kwargs keys and values",
        json_schema_extra=ui_field_config(order=6, label="Show All Keys"),
    )

    # --- Text Appearance ---

    font_size: int = Field(
        default=16,
        ge=8,
        le=72,
        description="Font size in pixels",
        json_schema_extra=ui_field_config(order=20, label="Font Size"),
    )

    text_opacity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Text transparency",
        json_schema_extra=ui_field_config(order=21, label="Opacity"),
    )

    text_position: TextPosition = Field(
        default=TextPosition.TOP_LEFT,
        description="Where to place the debug text",
        json_schema_extra=ui_field_config(order=22, label="Position"),
    )

    bg_opacity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Background box opacity behind text",
        json_schema_extra=ui_field_config(order=23, label="BG Opacity"),
    )

    max_value_length: int = Field(
        default=80,
        ge=20,
        le=500,
        description="Truncate long values to this many characters",
        json_schema_extra=ui_field_config(order=24, label="Max Value Len"),
    )
