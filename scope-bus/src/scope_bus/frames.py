"""Frame normalization and conversion utilities."""

from __future__ import annotations

import torch
from PIL import Image


def normalize_input(video: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Stack Scope video input into a normalized (T, H, W, C) float32 tensor.

    Input: list of tensors each (1, H, W, C) in [0, 255].
    Output: (T, H, W, C) float32 in [0, 1] on the given device.
    """
    frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)
    return frames.to(device=device, dtype=torch.float32) / 255.0


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    """Convert a single (H, W, C) float32 [0,1] tensor to a PIL RGB image."""
    arr = (frame.clamp(0, 1) * 255).byte().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")
