"""scope-language — Ollama VLM and LLM clients for Scope plugins."""

from scope.core.plugins.hookspecs import hookimpl

from .llm import OllamaLLM
from .vlm import OllamaVLM

__all__ = ["OllamaVLM", "OllamaLLM"]


@hookimpl
def register_pipelines(register):
    from ._plugin import ScopeLanguagePipeline
    register(ScopeLanguagePipeline)
