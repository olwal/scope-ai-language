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


# ---------------------------------------------------------------------------
# Text overlay fields (shared by standalone + postprocessor)
# ---------------------------------------------------------------------------

class _OverlayMixin(BasePipelineConfig):
    """Overlay text appearance fields. Mixed into configs that render text."""

    overlay_enabled: bool = Field(
        default=True,
        description="Overlay the VLM response text on the output video",
        json_schema_extra=ui_field_config(order=15, label="Overlay Text"),
    )

    font_family: FontFamily = Field(
        default=FontFamily.ARIAL, description="Typeface for the overlay text",
        json_schema_extra=ui_field_config(order=20, label="Font"),
    )

    font_size: int = Field(
        default=24, ge=8, le=120, description="Font size in pixels",
        json_schema_extra=ui_field_config(order=21, label="Font Size"),
    )

    font_color_r: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Text color — red channel",
        json_schema_extra=ui_field_config(order=22, label="Color R"),
    )

    font_color_g: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Text color — green channel",
        json_schema_extra=ui_field_config(order=23, label="Color G"),
    )

    font_color_b: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Text color — blue channel",
        json_schema_extra=ui_field_config(order=24, label="Color B"),
    )

    text_opacity: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Text transparency",
        json_schema_extra=ui_field_config(order=25, label="Opacity"),
    )

    text_position: TextPosition = Field(
        default=TextPosition.BOTTOM_LEFT, description="Text placement",
        json_schema_extra=ui_field_config(order=26, label="Position"),
    )

    word_wrap: bool = Field(
        default=True, description="Wrap long text to fit within the frame width",
        json_schema_extra=ui_field_config(order=27, label="Word Wrap"),
    )

    bg_opacity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Background box transparency",
        json_schema_extra=ui_field_config(order=28, label="BG Opacity"),
    )


# ---------------------------------------------------------------------------
# Standalone: queries VLM + overlays in one pipeline
# ---------------------------------------------------------------------------

class VLMOllamaConfig(_OverlayMixin):
    """Standalone pipeline: queries Ollama VLM, overlays text, injects prompt."""

    pipeline_id = "vlm-ollama"
    pipeline_name = "VLM Ollama"
    pipeline_description = "Queries an Ollama VLM, overlays the response, and optionally injects it as a prompt"

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}

    # --- Ollama Connection (load-time) ---

    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the Ollama API server",
        json_schema_extra=ui_field_config(
            order=1, label="Ollama URL", is_load_param=True, category="configuration"
        ),
    )

    ollama_model: str = Field(
        default="llava:7b",
        description="Ollama vision model to use (e.g. llava:7b, llama3.2-vision, moondream)",
        json_schema_extra=ui_field_config(
            order=2, label="Model", is_load_param=True, category="configuration"
        ),
    )

    # --- VLM Prompt ---

    vlm_prompt: str = Field(
        default="Describe what you see in this image in one sentence.",
        description="Prompt sent to the VLM along with each frame",
        json_schema_extra=ui_field_config(order=10, label="Prompt"),
    )

    send_interval: float = Field(
        default=3.0, ge=0.5, le=30.0,
        description="Seconds between sending frames to the VLM",
        json_schema_extra=ui_field_config(order=11, label="Send Interval"),
    )

    # --- Output ---

    inject_prompt: bool = Field(
        default=True,
        description="Inject the VLM response as a prompt for downstream pipelines",
        json_schema_extra=ui_field_config(order=16, label="Inject Prompt"),
    )

    prompt_weight: float = Field(
        default=100.0, ge=0.0, le=200.0,
        description="Weight of the injected prompt",
        json_schema_extra=ui_field_config(order=17, label="Prompt Weight"),
    )


# ---------------------------------------------------------------------------
# Preprocessor: queries VLM + sends text via UDP (no overlay)
# ---------------------------------------------------------------------------

class VLMOllamaPreConfig(BasePipelineConfig):
    """Preprocessor: captures raw camera → Ollama VLM → injects prompt + UDP broadcast."""

    pipeline_id = "vlm-ollama-pre"
    pipeline_name = "VLM Ollama (Pre)"
    pipeline_description = "Preprocessor: sends camera frames to Ollama VLM, injects prompt, broadcasts text via UDP"

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
        default="llava:7b",
        description="Ollama vision model to use",
        json_schema_extra=ui_field_config(
            order=2, label="Model", is_load_param=True, category="configuration"
        ),
    )

    udp_port: int = Field(
        default=9400,
        ge=1024,
        le=65535,
        description="UDP channel port to broadcast VLM responses on (must match the postprocessor)",
        json_schema_extra=ui_field_config(order=3, label="UDP Port"),
    )

    # --- VLM Prompt ---

    vlm_prompt: str = Field(
        default="Describe what you see in this image in one sentence.",
        description="Prompt sent to the VLM along with each frame",
        json_schema_extra=ui_field_config(order=10, label="Prompt"),
    )

    send_interval: float = Field(
        default=3.0, ge=0.5, le=30.0,
        description="Seconds between sending frames to the VLM",
        json_schema_extra=ui_field_config(order=11, label="Send Interval"),
    )

    # --- Output ---

    inject_prompt: bool = Field(
        default=True,
        description="Inject the VLM response as a prompt for downstream pipelines",
        json_schema_extra=ui_field_config(order=16, label="Inject Prompt"),
    )

    prompt_weight: float = Field(
        default=100.0, ge=0.0, le=200.0,
        description="Weight of the injected prompt",
        json_schema_extra=ui_field_config(order=17, label="Prompt Weight"),
    )


# ---------------------------------------------------------------------------
# Postprocessor: receives text via UDP + overlays on AI-processed output
# ---------------------------------------------------------------------------

class VLMOllamaPostConfig(_OverlayMixin):
    """Postprocessor: receives VLM text via UDP and overlays on AI-processed video."""

    pipeline_id = "vlm-ollama-post"
    pipeline_name = "VLM Ollama (Post)"
    pipeline_description = "Postprocessor: receives VLM text via UDP and overlays on AI-processed video"

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.POSTPROCESSOR]

    udp_port: int = Field(
        default=9400,
        ge=1024,
        le=65535,
        description="UDP channel port to listen on (must match the preprocessor's UDP Port)",
        json_schema_extra=ui_field_config(order=1, label="UDP Port"),
    )
