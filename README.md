# AI Language Plugins for Daydream Scope

Real-time AI plugins that close the loop between **seeing** and **generating** — the system watches live video, reasons about what it sees, and continuously steers the AI image generation based on that understanding.

A vision language model (VLM) observes the raw camera feed and produces semantic descriptions: the mood of a crowd, the species of an animal, the weather in a landscape, the emotional tone of a scene. Those descriptions feed directly into a large language model (LLM), which rewrites them as rich diffusion prompts, which in turn shape what the AI generates — frame by frame, in real time.

**Example:** Point the camera at a cat. Ask the VLM *"what are the natural predators of what you see?"* — it answers *"eagles, foxes, coyotes"*. That response becomes the live diffusion prompt. The AI no longer renders a cat; it renders whatever is hunting it, morphing dynamically as the VLM's answers evolve.

The generation doesn't follow a fixed script — it follows the scene. Prompt state changes smoothly via temporal interpolation between semantic states rather than cutting abruptly between them. Multiple plugins can run in parallel, chained, or controlled from external tools (OSC, UDP) for live performance and installation contexts.

Built on [Ollama](https://ollama.com) for local/remote VLM and LLM inference, with shared libraries for transport, rendering, and prompt routing (scope-bus, scope-language).

https://github.com/user-attachments/assets/a8fc647c-5379-4b51-960e-5ce784035219

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

The Pre queries the raw camera feed; the Post overlays the description on the AI-processed output.

**Key settings:**
- `ollama_url` / `ollama_model` — load-time connection config
- `vlm_prompt` — question sent to the VLM with each frame
- `send_interval` — seconds between VLM queries (VLM is slow; 3–10s typical)
- `inject_prompt` / `prompt_weight` — whether to use the VLM response as a diffusion prompt
- `transition_steps` — frames to blend from current to new prompt (0 = instant)
- `udp_port` — channel for Pre→Post communication (Pre/Post only)

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

**Use case:** Bridge any external application into Scope's prompt chain. Send prompts from a Python script, a custom controller, or any other tool that can send UDP packets.

**Key settings:**
- `udp_port` — channel to listen on (load-time)
- `prompt_weight` — weight of injected prompt
- `transition_steps` — frames to blend from current to new prompt (0 = instant)
- `overlay_enabled` — show received text on video (yellow, top-left) for monitoring

**Sending from Python:**
```python
import socket

MULTICAST_GROUP = "239.255.42.99"
PORT = 9400

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
sock.sendto("a moonlit forest, painterly".encode(), (MULTICAST_GROUP, PORT))
```

---

### scope-osc-prompt

Receives OSC `/prompt` messages and injects the text as a diffusion prompt.

**Role:** Preprocessor

**Use case:** Integrate Scope with TouchDesigner, Ableton Live, Max/MSP, or any other tool that sends OSC. Send a string to `/prompt` on the configured port and it becomes the active diffusion prompt.

**Key settings:**
- `osc_port` — UDP port to listen for OSC messages (load-time, default 9000)
- `prompt_weight` — weight of injected prompt
- `transition_steps` — frames to blend from current to new prompt (0 = instant)
- `overlay_enabled` — show received text on video (yellow, top-left) for monitoring

**Sending from TouchDesigner / Python:**
```python
from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 9000)
client.send_message("/prompt", "a misty forest at dawn, painterly")
```

---

### scope-test-text-log

Debug postprocessor that overlays all pipeline kwargs on the video and prints them to stdout. Shows video shape, prompts, UDP messages, and any extra kwargs flowing through the chain.

**Role:** Postprocessor
**Use case:** Drop this at the end of any chain to inspect exactly what's flowing between stages.

---

## Installation

Dependencies must be installed before the plugins that use them. Via the Scope UI (installs into the correct venv):

1. `scope-bus`
2. `scope-language`
3. `scope-vlm-ollama`, `scope-llm-ollama`, `scope-udp-prompt`, `scope-osc-prompt`
4. `scope-test-text-log`

After installing `scope-bus` and `scope-language`, they appear in the Scope UI pipeline list as passthrough pipelines — confirming installation and allowing uninstall via the UI.

---

## RunPod Deployment

### Daydream Scope Pod

Tested configuration for running Daydream Scope on RunPod:

| Setting | Value |
|---|---|
| GPU | RTX PRO 6000 (1×) |
| vCPU / Memory | 16 vCPU / 188 GB |
| Container disk | 20 GB |
| Network volume | 80 GB (`daydream_scope`, mounted at `/workspace`) |
| On-demand price | ~$1.69/hr compute + $0.003/hr storage |
| Base template | [daydream-scope](https://console.runpod.io/hub/template/daydream-scope?id=aca8mw9ivw) (`aca8mw9ivw`) |

The network volume at `/workspace` persists across pod restarts — use it for model weights and checkpoints.

### Split-Instance Architecture (Recommended)

Ollama VLM/LLM queries are slow (1–5s each) and run in background threads, so Ollama doesn't need to share the GPU with diffusion inference. Running Ollama on a separate, cheaper pod frees the main GPU for full-speed diffusion:

```
[Scope pod: RTX PRO 6000]          [Ollama pod: CPU-only or cheapest GPU]
  StreamDiffusion / other models       ollama serve
  scope-vlm-ollama (pre/post)   ──►   http://<ollama-pod-ip>:11434
  scope-llm-ollama
```

In each VLM/LLM plugin, set `ollama_url` (load-time) to the Ollama pod's public IP:
```
http://213.x.x.x:11434
```

RunPod exposes port 11434 via the pod's public IP when you add it under **Expose TCP Ports** in the pod settings.

### Ollama Pod Setup

For the Ollama-only pod, use the cheapest CPU pod (or any GPU pod). Paste this as the **Container Start Command** when creating the pod or template:

```sh
curl -fsSL https://raw.githubusercontent.com/olwal/scope-ai-language/main/scripts/setup-ollama-pod.sh | sh
```

This installs Ollama, pulls `qwen3-vl:2b`, and starts the server bound to `0.0.0.0:11434`. `OLLAMA_HOST=0.0.0.0` is required so Ollama is reachable via RunPod's TCP port forwarding — without it, Ollama only listens on `127.0.0.1`.

To use a different model, set the `OLLAMA_MODEL` environment variable on the pod before running the script, or add it inline:
```sh
OLLAMA_MODEL=llava:7b curl -fsSL https://raw.githubusercontent.com/olwal/scope-ai-language/main/scripts/setup-ollama-pod.sh | sh
```

The model is downloaded on first boot — subsequent restarts skip the pull if the model is cached on a network volume.

**Recommended model:** `qwen3-vl:2b` — fast, small, capable vision model. For higher quality at the cost of speed: `llava:7b` or `llava:13b`.

### Creating a RunPod Template

To save this as a reusable template in the RunPod console:

1. Go to **Manage → Templates → New Template**
2. Set **Container Image** to any base image with CUDA or a plain Ubuntu image (e.g. `runpod/base:0.4.0-cuda11.8.0`)
3. Under **Container Start Command**, paste the Ollama install script above
4. Under **Expose TCP Ports**, add `11434` (Ollama API)
5. Set **Container Disk** to 5–10 GB (Ollama binary + small model overhead if no volume)
6. Optionally attach a **Network Volume** at `/root/.ollama` to cache pulled models across restarts
7. Save as private template — it will appear in your pod creation flow

For the network volume approach, change the pull line to check first:
```sh
ollama pull qwen3-vl:2b 2>/dev/null || true
```
So re-pulling an already-cached model is a no-op.

---

## Architecture

```
scope-bus          ← shared transport + rendering library
scope-language     ← Ollama VLM/LLM clients (depends on scope-bus)

scope-vlm-ollama   ← vision language model pipeline (depends on scope-language)
scope-llm-ollama   ← text language model pipeline (depends on scope-language)
scope-udp-prompt   ← receive UDP text → inject as prompt (depends on scope-bus)
scope-osc-prompt   ← receive OSC /prompt → inject as prompt (depends on scope-bus)

scope-test-text-log  ← debug: overlay postprocessor
```

### Pipeline Types

Scope supports three pipeline roles, declared in each plugin's `schema.py`:

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

**UDPSender** — multicast sender with debounced port changes. Accepts strings or dicts (serialised as JSON):
```python
sender = UDPSender(port=9400)
sender.send("a sunset over mountains")          # plain text
sender.send({"prompt": "...", "response": "..."})  # JSON dict
sender.update_port(9401)  # debounced 3s — call every frame, applies after stable
```

**UDPReceiver** — multicast receiver, non-blocking poll. Auto-parses JSON:
```python
receiver = UDPReceiver(port=9400)
msg = receiver.poll()  # str, dict (if JSON), or None
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

**PromptInjector** — injects prompts only when text changes. Supports instant or smooth transitions:
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

## Key Daydream Scope Concepts

| Concept | Documentation |
|---|---|
| Pipeline interface (`Pipeline`, `Requirements`) | `scope/src/scope/core/pipelines/interface.py` |
| Schema base class (`BasePipelineConfig`, `ui_field_config`) | `scope/src/scope/core/pipelines/base_schema.py` |
| Plugin registration (`hookimpl`, `register_pipelines`) | `scope/src/scope/core/plugins/hookspecs.py` |
| Preprocessor → main pipeline parameter forwarding | `scope/src/scope/server/pipeline_processor.py` |
| Prompt format (`{"text": str, "weight": float}`) | Consumed by the main diffusion pipeline |
