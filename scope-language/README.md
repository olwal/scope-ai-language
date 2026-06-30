# scope-language

Async Ollama clients for vision and text models. Depends on [scope-bus](../scope-bus/README.md).

```python
from scope_language import OllamaVLM, OllamaLLM
```

## OllamaVLM

Sends video frames to a vision model in a background thread (so slow inference never blocks the pipeline):

```python
vlm = OllamaVLM(url="http://localhost:11434", model="qwen3-vl:2b")

# In __call__ (runs every frame):
if vlm.should_send(interval=3.0):       # time-throttled
    vlm.query_async(
        frames[0],                       # single (H, W, C) tensor
        prompt="Describe what you see",
        callback=lambda text: sender.send(text),  # optional
    )
description = vlm.get_last_response()   # returns last completed response
```

## OllamaLLM

Text-to-text, same async pattern:

```python
llm = OllamaLLM(url="http://localhost:11434", model="llama3.2:3b")

if llm.should_send(interval=5.0):
    llm.query_async(
        prompt="a foggy forest",
        system="Rewrite as a cinematic scene description in one sentence.",
    )
response = llm.get_last_response()
```
