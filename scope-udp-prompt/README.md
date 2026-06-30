# scope-udp-prompt

Preprocessor that receives text via UDP multicast and injects it as a diffusion prompt. Bridges any external application — a Python script, a custom controller, or any tool that can send UDP packets — into Scope's prompt chain.

## Settings

| Setting | Default | Notes |
|---|---|---|
| `udp_port` | `9400` | Channel to listen on (load-time) |
| `prompt_weight` | `100.0` (0–200) | Weight of the injected prompt |
| `transition_steps` | `0` (0–30) | Frames to blend from current to new prompt (0 = instant) |
| `interpolation_method` | `slerp` | `slerp` (smooth) or `linear` blend for transitions |
| `overlay_enabled` | `true` | Show received text on video for monitoring |
| `font_size` / `text_opacity` / `bg_opacity` | `20` / `1.0` / `0.5` | Overlay appearance |

Messages are sent over UDP multicast on `239.255.42.99`; the port acts as the channel.

## Sending from Python

```python
import socket

MULTICAST_GROUP = "239.255.42.99"
PORT = 9400

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
sock.sendto("a moonlit forest, painterly".encode(), (MULTICAST_GROUP, PORT))
```
