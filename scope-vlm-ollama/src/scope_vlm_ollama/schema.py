from typing import Literal

from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, UsageType, ui_field_config
from scope_bus import OverlayMixin


# ---------------------------------------------------------------------------
# Shared VLM query fields (connection + prompt + output)
# ---------------------------------------------------------------------------

class _VLMQueryMixin(BasePipelineConfig):
    """Shared Ollama connection, prompt, and output fields for VLM pipelines."""

    # --- Ollama Connection (load-time) ---

    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the Ollama API server",
        json_schema_extra=ui_field_config(
            order=1, label="Ollama URL", is_load_param=True, category="configuration"
        ),
    )

    ollama_model: str = Field(
        # default="llava:7b",
        default="qwen3-vl:2b",
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

    transition_steps: int = Field(
        default=0, ge=0, le=30,
        description="Frames to blend from the current prompt to the new one (0 = instant)",
        json_schema_extra=ui_field_config(order=18, label="Transition Steps"),
    )

    interpolation_method: Literal["slerp", "linear"] = Field(
        default="slerp",
        description="Blending method for prompt transitions: slerp (smooth) or linear",
        json_schema_extra=ui_field_config(order=19, label="Interpolation"),
    )


# ---------------------------------------------------------------------------
# Standalone: queries VLM + overlays in one pipeline
# ---------------------------------------------------------------------------

class VLMOllamaConfig(OverlayMixin, _VLMQueryMixin):
    """Standalone pipeline: queries Ollama VLM, overlays text, injects prompt."""

    pipeline_id = "vlm-ollama"
    pipeline_name = "VLM Ollama"
    pipeline_description = "Queries an Ollama VLM, overlays the response, and optionally injects it as a prompt"

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}


# ---------------------------------------------------------------------------
# Preprocessor: queries VLM + sends text via UDP (no overlay)
# ---------------------------------------------------------------------------

class VLMOllamaPreConfig(_VLMQueryMixin):
    """Preprocessor: captures raw camera → Ollama VLM → injects prompt + UDP broadcast."""

    pipeline_id = "vlm-ollama-pre"
    pipeline_name = "VLM Ollama (Pre)"
    pipeline_description = "Preprocessor: sends camera frames to Ollama VLM, injects prompt, broadcasts text via UDP"

    supports_prompts = False
    modes = {"video": ModeDefaults(default=True)}
    usage = [UsageType.PREPROCESSOR]

    udp_port: int = Field(
        default=9400,
        ge=1024,
        le=65535,
        description="UDP channel port to broadcast VLM responses on (must match the postprocessor)",
        json_schema_extra=ui_field_config(order=3, label="UDP Port"),
    )


# ---------------------------------------------------------------------------
# Postprocessor: receives text via UDP + overlays on AI-processed output
# ---------------------------------------------------------------------------

class VLMOllamaPostConfig(OverlayMixin):
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
