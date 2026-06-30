# scope-vlm-ollama

Queries an Ollama vision model (VLM) on live video and turns the response into a diffusion prompt. Available as three variants:

| Pipeline ID | Role | Description |
|---|---|---|
| `vlm-ollama` | Main | Query VLM + overlay response + inject prompt |
| `vlm-ollama-pre` | Preprocessor | Query VLM + inject prompt + broadcast via UDP |
| `vlm-ollama-post` | Postprocessor | Receive UDP text + overlay on AI output |

**Typical split chain:** `[VLM Pre] → [AI Model] → [VLM Post]` — the Pre queries the raw camera feed; the Post overlays the description on the AI-processed output.

## Settings

| Setting | Variant | Default | Notes |
|---|---|---|---|
| `ollama_url` | all | `http://localhost:11434` | Load-time connection config |
| `ollama_model` | all | `qwen3-vl:2b` | Vision model (e.g. `llava:7b`, `llama3.2-vision`, `moondream`) |
| `vlm_prompt` | all | "Describe what you see…" | Question sent to the VLM with each frame |
| `send_interval` | all | `3.0` (0.5–30) | Seconds between VLM queries (VLM is slow; 3–10s typical) |
| `prompt_settle_time` | all | `1.0` (0–5) | Debounce: wait this long after the prompt stops changing before using it (avoids mid-typing queries) |
| `inject_prompt` | all | `true` | Use the VLM response as a downstream diffusion prompt |
| `prompt_weight` | all | `100.0` (0–200) | Weight of the injected prompt |
| `transition_steps` | all | `0` (0–30) | Frames to blend from current to new prompt (0 = instant) |
| `interpolation_method` | all | `slerp` | `slerp` (smooth) or `linear` blend for transitions |
| `udp_port` | pre / post | `9400` | Channel for Pre→Post communication (must match on both) |

The Main and Post variants also expose the standard overlay appearance fields (font, color, position, opacity, word-wrap) from `OverlayMixin` — see [scope-bus](../scope-bus/README.md).
