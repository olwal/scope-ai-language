# AI Language Plugins for Daydream Scope

Generative AI plugins for language-driven, real-time video inference and generation. Ollama VLM/LLM pipelines and UDP prompt routing, built on shared libraries for communication and AI services (scope-bus, scope-language).

---

## Architecture Overview

```
scope-bus          ← shared transport + rendering library
scope-language     ← Ollama VLM/LLM clients (depends on scope-bus)

scope-vlm-ollama   ← vision language model pipeline (depends on scope-language)
scope-llm-ollama   ← text language model pipeline (depends on scope-language)
scope-udp-prompt   ← receive UDP text → inject as prompt (depends on scope-bus)

scope-test-text-generator  ← test: rolling counter sender
scope-test-text-log        ← test: debug overlay postprocessor
```

### Pipeline Types

Scope supports three pipeline roles. Each plugin declares its role in `schema.py`:

| Role | `usage =` | Runs | Typical use |
|---|---|---|---|
| **Main** | *(omit)* | In the AI model slot | Full processing pipelines |
| **Preprocessor** | `[UsageType.PREPROCESSOR]` | Before the AI model | Prompt injection, signal routing |
| **Postprocessor** | `[UsageType.POSTPROCESSOR]` | After the AI model | Overlays, logging, routing |

### The UDP Bus

Plugins communicate at runtime using **UDP multicast** on `239.255.42.99`. The port number acts as a **channel** — sender and receiver must use the same port. Multiple receivers on the same port all receive every message (fan-out).

```
[VLM Pre]──UDP:9400──►[VLM Post]   (overlay on AI output)
                  └──►[UDP Prompt] (forward VLM text as prompt)
                  └──►[Text Log]   (debug display)
```

---

## Shared Libraries

### scope-bus

Transport, rendering, and frame utilities. All other plugins depend on this.

```python
from scope_bus import (
    UDPSender,          # send text via UDP multicast
    UDPReceiver,        # receive text via UDP multicast
    render_text_overlay,# draw text onto (T, H, W, C) tensors
    normalize_input,    # list[Tensor] → (T, H, W, C) float32 [0,1]
    tensor_to_pil,      # (H, W, C) tensor → PIL Image
    PromptInjector,     # dedup-append prompts to output dict
)
```

**UDPSender** — multicast sender with debounced port changes:
```python
sender = UDPSender(port=9400)
sender.send("a sunset over mountains")
sender.update_port(9401)  # debounced 3s — call every frame, applies after stable
```

**UDPReceiver** — multicast receiver, non-blocking poll:
```python
receiver = UDPReceiver(port=9400)
msg = receiver.poll()  # returns latest message or None
```

**render_text_overlay** — composites text onto video frames:
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

**PromptInjector** — appends prompts only when text changes:
```python
injector = PromptInjector()
injector.inject_if_new(output, text="a cat on a couch", weight=100.0)
# output["prompts"] is created/appended only when text differs from last call
```

**normalize_input** — converts Scope's raw video list to a usable tensor:
```python
frames = normalize_input(video, device)
# video: list of (1, H, W, C) uint8 tensors from Scope
# returns: (T, H, W, C) float32 on device, values in [0, 1]
```

---

### scope-language

Async Ollama clients for vision and text models.

```python
from scope_language import OllamaVLM, OllamaLLM
```

**OllamaVLM** — sends video frames to a vision model in a background thread:
```python
vlm = OllamaVLM(url="http://localhost:11434", model="llava:7b")

# In __call__ (runs every frame):
if vlm.should_send(interval=3.0):       # time-throttled
    vlm.query_async(
        frames[0],                       # single (H, W, C) tensor
        prompt="Describe what you see",
        callback=lambda text: sender.send(text),  # optional
    )
description = vlm.get_last_response()   # returns last completed response
```

**OllamaLLM** — text-to-text, same async pattern:
```python
llm = OllamaLLM(url="http://localhost:11434", model="llama3.2:3b")

if llm.should_send(interval=5.0):
    llm.query_async(
        prompt="a foggy forest",
        system="Rewrite as a cinematic scene description in one sentence.",
    )
response = llm.get_last_response()
```

---

## Plugins

### scope-vlm-ollama

Queries an Ollama vision model on live video. Available as three variants:

| Pipeline | Role | Description |
|---|---|---|
| **VLM Ollama** | Main | Query VLM + overlay response + inject prompt |
| **VLM Ollama (Pre)** | Preprocessor | Query VLM + inject prompt + broadcast via UDP |
| **VLM Ollama (Post)** | Postprocessor | Receive UDP text + overlay on AI output |

**Typical chain:** `[VLM Pre] → [AI Model] → [VLM Post]`

The Pre queries the raw camera; the Post overlays the description on the AI-processed output.

**Key settings:**
- `ollama_url` / `ollama_model` — load-time connection config
- `vlm_prompt` — question sent to the VLM with each frame
- `send_interval` — seconds between VLM queries (VLM is slow; 3–10s typical)
- `inject_prompt` / `prompt_weight` — whether to use the VLM response as a diffusion prompt
- `udp_port` — channel for Pre→Post communication

---

### scope-llm-ollama

Sends text to an Ollama LLM and injects the response as a diffusion prompt.

**Role:** Preprocessor

**Use case:** Transform a simple input phrase into an elaborate scene description, style directive, or creative prompt. Works well chained before any image generation model.

**Key settings:**
- `system_prompt` — LLM personality / rewriting instruction
- `input_prompt` — the text fed to the LLM each interval
- `send_interval` — query frequency
- `inject_prompt` — send LLM response downstream as a diffusion prompt
- `udp_enabled` / `udp_port` — optionally broadcast LLM response to other plugins

---

### scope-udp-prompt

Receives text via UDP and injects it as a diffusion prompt.

**Role:** Preprocessor

**Use case:** Bridge any external application into Scope's prompt chain. Send prompts from a Python script, a custom controller, OSC bridge, or any other tool that can send UDP packets.

**Key settings:**
- `udp_port` — channel to listen on (load-time)
- `prompt_weight` — weight of injected prompt
- `overlay_enabled` — show received text on video (yellow, top-left) for monitoring

**Sending from Python:**
```python
import socket, struct

MULTICAST_GROUP = "239.255.42.99"
PORT = 9400

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
sock.sendto("a moonlit forest, painterly".encode(), (MULTICAST_GROUP, PORT))
```

---

### scope-test-text-generator

Test plugin that sends a rolling frame counter via UDP. Use this to verify that a receiver (Text Log, UDP Prompt) is working before connecting a real sender.

**Role:** Main pipeline + Pre/Post variant
**Output:** `frame #N | ts=HH:MM:SS` every 2 seconds

---

### scope-test-text-log

Debug postprocessor that overlays all pipeline kwargs on the video and prints them to stdout. Shows video shape, prompts, UDP messages, and any extra kwargs flowing through the chain.

**Role:** Postprocessor
**Use case:** Drop this at the end of any chain to inspect exactly what's flowing between stages.

---

## Installation Order

Dependencies must be installed before the plugins that use them:


Via Scope UI (installs into the correct venv):
1. scope-bus
2. scope-language
3. scope-vlm-ollama, scope-llm-ollama, scope-udp-prompt
4. scope-test-text-generator, scope-test-text-log



After installing `scope-bus` and `scope-language`, they appear in the Scope UI pipeline list as passthrough pipelines. This confirms installation and allows uninstalling via the UI.

---

## Key Daydream Scope Concepts

| Concept | Documentation |
|---|---|
| Pipeline interface (`Pipeline`, `Requirements`) | `scope/src/scope/core/pipelines/interface.py` |
| Schema base class (`BasePipelineConfig`, `ui_field_config`) | `scope/src/scope/core/pipelines/base_schema.py` |
| Plugin registration (`hookimpl`, `register_pipelines`) | `scope/src/scope/core/plugins/hookspecs.py` |
| Preprocessor → main pipeline parameter forwarding | `scope/src/scope/server/pipeline_processor.py` |
| Prompt format (`{"text": str, "weight": float}`) | Consumed by the main diffusion pipeline |
