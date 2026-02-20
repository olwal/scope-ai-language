from typing import Literal

from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, UsageType, ui_field_config
from scope_bus import OverlayMixin


class LLMOllamaConfig(OverlayMixin):
    """Preprocessor: transforms text through an Ollama LLM and injects as prompt."""

    pipeline_id = "llm-ollama"
    pipeline_name = "LLM Ollama"
    pipeline_description = "Sends text to an Ollama LLM, injects the response as a prompt, and optionally overlays it"

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.PREPROCESSOR]

    # --- Ollama Connection (load-time) ---

    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the Ollama API server",
        json_schema_extra=ui_field_config(
            order=1, label="Ollama URL", is_load_param=True, category="configuration"
        ),
    )

    ollama_model: str = Field(
        default="llama3.2:3b",
        description="Ollama text model to use (e.g. llama3.2:3b, mistral, phi3)",
        json_schema_extra=ui_field_config(
            order=2, label="Model", is_load_param=True, category="configuration"
        ),
    )

    # --- LLM Prompting ---

    system_prompt: str = Field(
        default="Rewrite the following as a vivid, cinematic scene description in one sentence.",
        description="System prompt that sets the LLM's behavior and style",
        json_schema_extra=ui_field_config(order=10, label="System Prompt"),
    )

    input_prompt: str = Field(
        default="a dreamy landscape",
        description="Text input to send to the LLM (used as-is, or combined with upstream prompts)",
        json_schema_extra=ui_field_config(order=11, label="Input Text"),
    )

    send_interval: float = Field(
        default=5.0, ge=0.5, le=60.0,
        description="Seconds between sending requests to the LLM",
        json_schema_extra=ui_field_config(order=12, label="Send Interval"),
    )

    # --- Output ---

    inject_prompt: bool = Field(
        default=True,
        description="Inject the LLM response as a prompt for downstream pipelines",
        json_schema_extra=ui_field_config(order=20, label="Inject Prompt"),
    )

    prompt_weight: float = Field(
        default=100.0, ge=0.0, le=200.0,
        description="Weight of the injected prompt",
        json_schema_extra=ui_field_config(order=21, label="Prompt Weight"),
    )

    transition_steps: int = Field(
        default=0, ge=0, le=30,
        description="Frames to blend from the current prompt to the new one (0 = instant)",
        json_schema_extra=ui_field_config(order=22, label="Transition Steps"),
    )

    interpolation_method: Literal["slerp", "linear"] = Field(
        default="slerp",
        description="Blending method for prompt transitions: slerp (smooth) or linear",
        json_schema_extra=ui_field_config(order=23, label="Interpolation"),
    )

    # --- UDP ---

    udp_enabled: bool = Field(
        default=False,
        description="Broadcast LLM response via UDP",
        json_schema_extra=ui_field_config(order=30, label="UDP Enabled"),
    )

    udp_port: int = Field(
        default=9400, ge=1024, le=65535,
        description="UDP port to broadcast LLM responses to",
        json_schema_extra=ui_field_config(order=31, label="UDP Port"),
    )

    # OverlayMixin provides: overlay_enabled, font_family, font_size, font_color_r/g/b,
    #                        text_opacity, text_position, word_wrap, bg_opacity (orders 15, 20-28)
    # Override font_color_b default to give LLM responses a warm tone
    font_color_b: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Text color — blue channel",
        json_schema_extra=ui_field_config(order=24, label="Color B"),
    )
