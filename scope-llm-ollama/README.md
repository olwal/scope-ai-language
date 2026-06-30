# scope-llm-ollama

Preprocessor that sends text to an Ollama LLM and injects the rewritten response as a diffusion prompt. Use it to transform a short observation into an elaborate scene description, style directive, or creative prompt. Works well chained before any image-generation model.

## Settings

| Setting | Default | Notes |
|---|---|---|
| `ollama_url` | `http://localhost:11434` | Load-time connection config |
| `ollama_model` | `llama3.2:3b` | Text model (e.g. `llama3.2:3b`, `mistral`, `phi3`) |
| `system_prompt` | "Rewrite the following as a vivid, cinematic scene…" | LLM personality / rewriting instruction |
| `input_prompt` | `a dreamy landscape` | Text fed to the LLM each interval |
| `send_interval` | `5.0` (0.5–60) | Seconds between LLM queries |
| `inject_prompt` | `true` | Send the LLM response downstream as a diffusion prompt |
| `prompt_weight` | `100.0` (0–200) | Weight of the injected prompt |
| `transition_steps` | `0` (0–30) | Frames to blend from current to new prompt (0 = instant) |
| `interpolation_method` | `slerp` | `slerp` (smooth) or `linear` blend for transitions |
| `udp_enabled` | `false` | Optionally broadcast the LLM response to other plugins |
| `udp_port` | `9400` | UDP channel to broadcast on |

Also exposes the standard overlay appearance fields from `OverlayMixin` (defaults to a warm text tone) — see [scope-bus](../scope-bus/README.md).
