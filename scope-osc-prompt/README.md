# scope-osc-prompt

Preprocessor that receives OSC `/prompt` messages and injects the text as a diffusion prompt. Integrate Scope with TouchDesigner, Ableton Live, Max/MSP, or any OSC-capable tool — send a string to `/prompt` on the configured port and it becomes the active prompt.

## Settings

| Setting | Default | Notes |
|---|---|---|
| `osc_port` | `9000` | UDP port to listen for OSC messages (load-time) |
| `prompt_weight` | `100.0` (0–200) | Weight of the injected prompt |
| `transition_steps` | `0` (0–30) | Frames to blend from current to new prompt (0 = instant) |
| `interpolation_method` | `slerp` | `slerp` (smooth) or `linear` blend for transitions |
| `overlay_enabled` | `true` | Show received text on video for monitoring |
| `font_size` / `text_opacity` / `bg_opacity` | `20` / `1.0` / `0.5` | Overlay appearance |

## Sending from TouchDesigner / Python

```python
from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 9000)
client.send_message("/prompt", "a misty forest at dawn, painterly")
```
