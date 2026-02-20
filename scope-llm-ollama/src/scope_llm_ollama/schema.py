from enum import Enum

from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, UsageType, ui_field_config


class FontFamily(str, Enum):
    ARIAL = "arial"
    COURIER = "courier"
    TIMES = "times"
    HELVETICA = "helvetica"


class TextPosition(str, Enum):
    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"


class LLMOllamaConfig(BasePipelineConfig):
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

    # --- Overlay ---

    overlay_enabled: bool = Field(
        default=True,
        description="Overlay the LLM response text on the output video",
        json_schema_extra=ui_field_config(order=40, label="Overlay Text"),
    )

    font_family: FontFamily = Field(
        default=FontFamily.ARIAL,
        description="Typeface for the overlay text",
        json_schema_extra=ui_field_config(order=41, label="Font"),
    )

    font_size: int = Field(
        default=24, ge=8, le=120,
        description="Font size in pixels",
        json_schema_extra=ui_field_config(order=42, label="Font Size"),
    )

    font_color_r: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Text color — red channel",
        json_schema_extra=ui_field_config(order=43, label="Color R"),
    )

    font_color_g: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Text color — green channel",
        json_schema_extra=ui_field_config(order=44, label="Color G"),
    )

    font_color_b: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Text color — blue channel",
        json_schema_extra=ui_field_config(order=45, label="Color B"),
    )

    text_opacity: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Text transparency",
        json_schema_extra=ui_field_config(order=46, label="Opacity"),
    )

    text_position: TextPosition = Field(
        default=TextPosition.BOTTOM_LEFT,
        description="Text placement",
        json_schema_extra=ui_field_config(order=47, label="Position"),
    )

    word_wrap: bool = Field(
        default=True,
        description="Wrap long text to fit within the frame width",
        json_schema_extra=ui_field_config(order=48, label="Word Wrap"),
    )

    bg_opacity: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Background box transparency",
        json_schema_extra=ui_field_config(order=49, label="BG Opacity"),
    )
