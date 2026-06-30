# scope-test-text-log

Debug postprocessor (`text-log`) that overlays all pipeline kwargs on the video and prints them to stdout — video shape, prompts, UDP messages, and any extra kwargs flowing through the chain. Drop it at the end of any chain to inspect exactly what's flowing between stages.

## Settings

| Setting | Default | Notes |
|---|---|---|
| `udp_enabled` | `true` | Also listen for incoming UDP messages from other plugins (load-time) |
| `udp_port` | `9400` | UDP channel to listen on |

Plus standard overlay appearance fields (font, size, position, opacity).
