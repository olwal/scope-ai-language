-+= Update 26 Feb 2026 | AI Language wins top prize in Daydream Scope Plugin contest! =+-

<img width="1200" height="454" alt="ai_language_winner_3" src="https://github.com/user-attachments/assets/63cd1fa2-9b99-4f9a-a778-446ac87c4d38" />

# AI Language Plugins for Daydream Scope

Real-time AI plugins that close the loop between **seeing** and **generating**. The system watches a video stream, reasons about what it sees in real-time, and continuously steers the AI image generation based on that understanding.

A vision language model (VLM) produces semantic descriptions: the mood of a crowd, the species of an animal, the weather in a landscape, the emotional tone of a scene. Those descriptions can optionally feed into a second preprocessor with large language model (LLM), which can rewrite them as rich diffusion prompts, which helps shape what the AI generates, frame by frame, in real time.

https://github.com/user-attachments/assets/aede2dd5-18de-4b4f-824d-652cc8868d8b

## Advanced semantic reasoning about content

**Example:** Point the camera at a cat. Ask the VLM *"what are the natural predators of what you see in three words?"*. It answers *"eagles, foxes, coyotes"*. That response becomes the live diffusion prompt. The AI no longer renders a cat; it renders whatever is hunting it, morphing dynamically as the VLM's answers evolve with each new inference.

The generation doesn't follow a fixed script. It follows the scene. Prompt state changes smoothly via temporal interpolation rather than cutting abruptly between semantic states. Multiple plugins can run in parallel, chained, or driven from external tools (OSC, UDP) for live performance and installation contexts.

<img width="300" height="299" alt="image" src="https://github.com/user-attachments/assets/d309e260-dd95-4156-b311-b4ebd692d3ca" />

<img width="300" height="302" alt="image" src="https://github.com/user-attachments/assets/4a3d702d-2261-4298-b8d6-188ded3d1e9e" />

## Live completion to steer streaming video generation

**Example** By drawing live into the feed (e.g., using local Spout streaming), the VLM can be used to drive a visual auto-complete. While initial strokes are ambigious, with more detail, the VLM inference starts to converge and provide increasingly accurate interpretations. Those are directly fed into the live video generation, serving as both a live autocomplete, but also as a means to create animated drawings.

https://github.com/user-attachments/assets/7465733a-3bcc-40a2-8b04-6236c3188233

## How it works

The plugins slot into Daydream Scope's preprocessor / postprocessor pipeline architecture. A typical split chain:

```
Camera → [VLM Pre] ──────────────────────► [AI Model] → [VLM Post] → Output
               │ UDP multicast 239.255.42.99       ▲
               └──► [UDP Prompt] ─── prompts ──────┘
```

Semantic responses are broadcast over **UDP multicast** so any number of downstream plugins receive them simultaneously — fan-out with no additional routing. The port number acts as a channel: any plugin listening on the same port gets every message.

Prompt transitions use **temporal interpolation** (slerp or linear) to blend smoothly between semantic states over a configurable number of frames, rather than snapping abruptly when the VLM's description changes.

Built on [Ollama](https://ollama.com) for local or remote VLM/LLM inference. Shared libraries handle all transport, frame conversion, text rendering, and prompt injection ([scope-bus](scope-bus/README.md), [scope-language](scope-language/README.md)), so each plugin stays focused on its single role in the chain.

<details>
<summary>More examples</summary>

https://github.com/user-attachments/assets/971cefb0-7f5e-4ff1-b9be-901abef0007f

https://github.com/user-attachments/assets/6420a48f-e79a-4bd5-b27e-d157d0630fea

https://github.com/user-attachments/assets/64e358d1-bd78-4d66-8aae-76654bd9ca1d

https://github.com/user-attachments/assets/2c34cb2e-01bb-4911-81c7-f61c89f56e3e

https://github.com/user-attachments/assets/a8fc647c-5379-4b51-960e-5ce784035219

</details>

---

## Plugins

Each plugin documents its own settings and usage in its folder:

| Plugin | Role | What it does |
|---|---|---|
| [**scope-vlm-ollama**](scope-vlm-ollama/README.md) | Main / Pre / Post | Queries an Ollama vision model on live video; overlays the response and injects it as a prompt (with UDP pre→post support) |
| [**scope-llm-ollama**](scope-llm-ollama/README.md) | Preprocessor | Rewrites text through an Ollama LLM and injects the result as a diffusion prompt |
| [**scope-udp-prompt**](scope-udp-prompt/README.md) | Preprocessor | Receives text via UDP multicast and injects it as a prompt — bridges any external tool |
| [**scope-osc-prompt**](scope-osc-prompt/README.md) | Preprocessor | Receives OSC `/prompt` messages — bridges TouchDesigner, Ableton, Max/MSP, etc. |
| [**scope-test-text-log**](scope-test-text-log/README.md) | Postprocessor | Debug monitor: overlays and logs all kwargs flowing through the chain |

**Shared libraries:** [scope-bus](scope-bus/README.md) (transport + overlay + frame utils) and [scope-language](scope-language/README.md) (async Ollama VLM/LLM clients).

---

## Installation

Dependencies must be installed before the plugins that use them. Via the Scope UI (installs into the correct venv):

1. `scope-bus`
2. `scope-language`
3. `scope-vlm-ollama`, `scope-llm-ollama`, `scope-udp-prompt`, `scope-osc-prompt`
4. `scope-test-text-log`

After installing `scope-bus` and `scope-language`, they appear in the Scope UI pipeline list as passthrough pipelines — confirming installation and allowing uninstall via the UI.

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

Scope supports three pipeline roles, declared in each plugin's `schema.py`:

| Role | `usage =` | Runs | Typical use |
|---|---|---|---|
| **Main** | *(omit)* | In the AI model slot | Full processing pipelines |
| **Preprocessor** | `[UsageType.PREPROCESSOR]` | Before the AI model | Prompt injection, signal routing |
| **Postprocessor** | `[UsageType.POSTPROCESSOR]` | After the AI model | Overlays, logging, routing |

Plugins communicate at runtime over **UDP multicast** on `239.255.42.99`. The port acts as a channel — every receiver on the same port gets every message (fan-out):

```
[VLM Pre]──UDP:9400──►[VLM Post]   (overlay on AI output)
                  └──►[UDP Prompt] (forward VLM text as prompt)
                  └──►[Text Log]   (debug display)
```

---

<details>
<summary><strong>RunPod Deployment</strong></summary>

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

In each VLM/LLM plugin, set `ollama_url` (load-time) to the Ollama pod's public IP (e.g. `http://213.x.x.x:11434`). RunPod exposes port 11434 via the pod's public IP when you add it under **Expose TCP Ports** in the pod settings.

### Ollama Pod Setup

For the Ollama-only pod, use the cheapest CPU pod (or any GPU pod). Paste this as the **Container Start Command** when creating the pod or template:

```sh
curl -fsSL https://raw.githubusercontent.com/olwal/scope-ai-language/main/scripts/setup-ollama-pod.sh | sh
```

This installs Ollama, pulls `qwen3-vl:2b`, and starts the server bound to `0.0.0.0:11434`. `OLLAMA_HOST=0.0.0.0` is required so Ollama is reachable via RunPod's TCP port forwarding — without it, Ollama only listens on `127.0.0.1`.

To use a different model, set `OLLAMA_MODEL` before running the script:

```sh
OLLAMA_MODEL=llava:7b curl -fsSL https://raw.githubusercontent.com/olwal/scope-ai-language/main/scripts/setup-ollama-pod.sh | sh
```

**Recommended model:** `qwen3-vl:2b` — fast, small, capable vision model. For higher quality at the cost of speed: `llava:7b` or `llava:13b`. Attach a network volume at `/root/.ollama` to cache pulled models across restarts.

</details>

<details>
<summary><strong>Key Daydream Scope Concepts</strong></summary>

| Concept | Documentation |
|---|---|
| Pipeline interface (`Pipeline`, `Requirements`) | `scope/src/scope/core/pipelines/interface.py` |
| Schema base class (`BasePipelineConfig`, `ui_field_config`) | `scope/src/scope/core/pipelines/base_schema.py` |
| Plugin registration (`hookimpl`, `register_pipelines`) | `scope/src/scope/core/plugins/hookspecs.py` |
| Preprocessor → main pipeline parameter forwarding | `scope/src/scope/server/pipeline_processor.py` |
| Prompt format (`{"text": str, "weight": float}`) | Consumed by the main diffusion pipeline |

</details>
