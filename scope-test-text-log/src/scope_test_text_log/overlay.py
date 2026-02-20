"""Debug overlay rendering — draws monospaced debug text onto video frame tensors."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _load_mono_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a monospace font for debug display."""
    for name in ["consola.ttf", "Consolas.ttf", "cour.ttf", "DejaVuSansMono.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_debug_overlay(
    frames: torch.Tensor,
    lines: list[str],
    font_size: int = 16,
    opacity: float = 1.0,
    position: str = "top-left",
    bg_opacity: float = 0.7,
) -> torch.Tensor:
    """Render debug lines onto each frame in a (T, H, W, C) tensor."""
    if not lines:
        return frames

    T, H, W, C = frames.shape
    device = frames.device

    font = _load_mono_font(font_size)
    margin = 8
    line_spacing = 3

    # Measure text
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    line_bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    line_heights = [bb[3] - bb[1] for bb in line_bboxes]
    line_widths = [bb[2] - bb[0] for bb in line_bboxes]
    total_height = sum(line_heights) + (len(lines) - 1) * line_spacing
    max_width = max(line_widths) if line_widths else 0

    # Clamp to frame size
    max_width = min(max_width, W - 2 * margin)

    # Create overlay
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Position
    if "top" in position:
        y_start = margin
    else:
        y_start = H - margin - total_height

    if "center" in position:
        x_start = (W - max_width) // 2
    else:
        x_start = margin

    # Background
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

    # Draw lines
    a = int(opacity * 255)
    # Use green for a "terminal" look
    color = (0, 255, 100, a)

    y = y_start
    for i, line in enumerate(lines):
        # Highlight section headers
        if line.startswith("---"):
            draw.text((x_start, y), line, font=font, fill=(180, 180, 180, a))
        elif line.startswith("  ["):
            # Prompt entries — slightly dimmer
            draw.text((x_start, y), line, font=font, fill=(100, 220, 255, a))
        else:
            draw.text((x_start, y), line, font=font, fill=color)
        y += line_heights[i] + line_spacing

    # Composite onto frames
    overlay_tensor = torch.from_numpy(
        np.array(overlay, dtype="float32")
    ).to(device) / 255.0

    rgb = overlay_tensor[:, :, :3]
    alpha = overlay_tensor[:, :, 3:4]

    result = frames * (1.0 - alpha) + rgb * alpha
    return result
