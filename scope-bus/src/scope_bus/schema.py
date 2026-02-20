"""Shared schema helpers — mixins and enums for Scope plugin configs."""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ui_field_config


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


class OverlayMixin(BasePipelineConfig):
    """Overlay text appearance fields. Mixed into configs that render text."""

    overlay_enabled: bool = Field(
        default=True,
        description="Overlay text on the output video",
        json_schema_extra=ui_field_config(order=15, label="Overlay Text"),
    )

    font_family: FontFamily = Field(
        default=FontFamily.ARIAL,
        description="Typeface for the overlay text",
        json_schema_extra=ui_field_config(order=20, label="Font"),
    )

    font_size: int = Field(
        default=24, ge=8, le=120,
        description="Font size in pixels",
        json_schema_extra=ui_field_config(order=21, label="Font Size"),
    )

    font_color_r: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Text color — red channel",
        json_schema_extra=ui_field_config(order=22, label="Color R"),
    )

    font_color_g: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Text color — green channel",
        json_schema_extra=ui_field_config(order=23, label="Color G"),
    )

    font_color_b: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Text color — blue channel",
        json_schema_extra=ui_field_config(order=24, label="Color B"),
    )

    text_opacity: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Text transparency",
        json_schema_extra=ui_field_config(order=25, label="Opacity"),
    )

    text_position: TextPosition = Field(
        default=TextPosition.BOTTOM_LEFT,
        description="Text placement",
        json_schema_extra=ui_field_config(order=26, label="Position"),
    )

    word_wrap: bool = Field(
        default=True,
        description="Wrap long text to fit within the frame width",
        json_schema_extra=ui_field_config(order=27, label="Word Wrap"),
    )

    bg_opacity: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Background box transparency",
        json_schema_extra=ui_field_config(order=28, label="BG Opacity"),
    )
