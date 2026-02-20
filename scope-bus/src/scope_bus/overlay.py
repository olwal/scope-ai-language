"""Text overlay rendering — draws text onto video frame tensors using PIL."""

from __future__ import annotations

import textwrap

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# Map font family names to common system font files
_FONT_MAP = {
    "arial": ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"],
    "courier": ["cour.ttf", "Courier.ttf", "DejaVuSansMono.ttf"],
    "times": ["times.ttf", "Times.ttf", "DejaVuSerif.ttf"],
    "helvetica": ["helvetica.ttf", "Helvetica.ttf", "DejaVuSans.ttf"],
}


def _load_font(family: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a system font, fall back to default."""
    candidates = _FONT_MAP.get(family, _FONT_MAP["arial"])
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_text_overlay(
    frames: torch.Tensor,
    text: str,
    font_family: str = "arial",
    font_size: int = 24,
    font_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    opacity: float = 1.0,
    position: str = "bottom-left",
    word_wrap: bool = True,
    bg_opacity: float = 0.5,
) -> torch.Tensor:
    """Render text onto each frame in a (T, H, W, C) tensor.

    All inputs are in [0, 1] float range. Returns modified tensor.
    """
    if not text:
        return frames

    T, H, W, C = frames.shape
    device = frames.device

    font = _load_font(font_family, font_size)
    margin = 10

    # Word wrap
    if word_wrap:
        max_chars = max(1, (W - 2 * margin) // max(1, font_size // 2))
        lines = textwrap.wrap(text, width=max_chars)
    else:
        lines = [text]

    if not lines:
        return frames

    # Measure text bounds
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [bb[3] - bb[1] for bb in line_bboxes]
    line_widths = [bb[2] - bb[0] for bb in line_bboxes]
    total_height = sum(line_heights) + (len(lines) - 1) * 4
    max_width = max(line_widths) if line_widths else 0

    # Create overlay image
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Compute text block position
    if "top" in position:
        y_start = margin
    else:
        y_start = H - margin - total_height

    if "center" in position:
        x_start = (W - max_width) // 2
    else:
        x_start = margin

    # Draw background box
    if bg_opacity > 0:
        pad = 6
        bg_alpha = int(bg_opacity * 255)
        draw.rectangle(
            [
                x_start - pad,
                y_start - pad,
                x_start + max_width + pad,
                y_start + total_height + pad,
            ],
            fill=(0, 0, 0, bg_alpha),
        )

    # Draw text lines
    r = int(font_color[0] * 255)
    g = int(font_color[1] * 255)
    b = int(font_color[2] * 255)
    a = int(opacity * 255)

    y = y_start
    for i, line in enumerate(lines):
        if "center" in position:
            lx = (W - line_widths[i]) // 2
        else:
            lx = x_start
        draw.text((lx, y), line, font=font, fill=(r, g, b, a))
        y += line_heights[i] + 4

    # Convert overlay to tensor and composite
    overlay_tensor = torch.from_numpy(
        np.array(overlay, dtype="float32")
    ).to(device) / 255.0

    rgb = overlay_tensor[:, :, :3]
    alpha = overlay_tensor[:, :, 3:4]

    result = frames * (1.0 - alpha) + rgb * alpha
    return result


def apply_overlay_from_kwargs(
    frames: torch.Tensor,
    text: str,
    kwargs: dict,
) -> torch.Tensor:
    """Render a text overlay using standard overlay parameters from a kwargs dict.

    Reads: font_family, font_size, font_color_r/g/b, text_opacity,
           text_position, word_wrap, bg_opacity.
    """
    return render_text_overlay(
        frames,
        text=text,
        font_family=kwargs.get("font_family", "arial"),
        font_size=kwargs.get("font_size", 24),
        font_color=(
            kwargs.get("font_color_r", 1.0),
            kwargs.get("font_color_g", 1.0),
            kwargs.get("font_color_b", 1.0),
        ),
        opacity=kwargs.get("text_opacity", 1.0),
        position=kwargs.get("text_position", "bottom-left"),
        word_wrap=kwargs.get("word_wrap", True),
        bg_opacity=kwargs.get("bg_opacity", 0.5),
    )
