# scope-bus

Shared transport, overlay, and frame utilities. **All other plugins depend on this.** Install it first.

```python
from scope_bus import (
    UDPSender,                 # send text/dict via UDP multicast
    UDPReceiver,               # receive text/dict via UDP multicast
    render_text_overlay,       # draw text onto (T, H, W, C) tensors
    apply_overlay_from_kwargs, # render_text_overlay reading from pipeline kwargs dict
    normalize_input,           # list[Tensor] → (T, H, W, C) float32 [0,1]
    tensor_to_pil,             # (H, W, C) tensor → PIL Image
    PromptInjector,            # dedup-inject prompts to output dict
    OverlayMixin,              # Pydantic mixin: overlay appearance fields for schemas
    FontFamily,                # Enum: arial | courier | times | helvetica
    TextPosition,              # Enum: top-left | top-center | bottom-left | bottom-center
)
```

## The UDP bus

Plugins communicate at runtime using **UDP multicast** on `239.255.42.99`. The port number acts as a **channel** — sender and receiver must use the same port. Multiple receivers on the same port all receive every message (fan-out, no additional routing).

## UDPSender

Multicast sender with debounced port changes. Accepts strings or dicts (serialised as JSON):

```python
sender = UDPSender(port=9400)
sender.send("a sunset over mountains")             # plain text
sender.send({"prompt": "...", "response": "..."})  # JSON dict
sender.update_port(9401)  # debounced 3s — call every frame, applies after stable
```

## UDPReceiver

Multicast receiver, non-blocking poll. Auto-parses JSON:

```python
receiver = UDPReceiver(port=9400)
msg = receiver.poll()  # str, dict (if JSON), or None
```

## render_text_overlay

Composites text onto video frames:

```python
frames = render_text_overlay(
    frames,
    text="VLM response here",
    font_family="arial",        # arial | courier | times | helvetica
    font_size=24,
    font_color=(1.0, 1.0, 1.0), # RGB [0,1]
    opacity=1.0,
    position="bottom-left",     # top-left | top-center | bottom-left | bottom-center
    word_wrap=True,
    bg_opacity=0.5,
)
```

## PromptInjector

Injects prompts only when text changes. Supports instant or smooth transitions:

```python
injector = PromptInjector()

# Instant change (default)
injector.inject_if_new(output, text="a cat on a couch", weight=100.0)
# output["prompts"] is set only when text differs from last call

# Smooth temporal blend (uses Scope's transition API)
injector.inject_if_new(output, text="a stormy sea", weight=100.0,
                       transition_steps=10, interpolation_method="slerp")
# output["transition"] is set with target_prompts + num_steps
```

## normalize_input

Converts Scope's raw video list to a usable tensor:

```python
frames = normalize_input(video, device)
# video: list of (1, H, W, C) uint8 tensors from Scope
# returns: (T, H, W, C) float32 on device, values in [0, 1]
```
