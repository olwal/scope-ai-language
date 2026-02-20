"""scope-bus — shared transport, overlay, and utility library for Scope plugins."""

from scope.core.plugins.hookspecs import hookimpl

from .frames import normalize_input, tensor_to_pil
from .overlay import render_text_overlay
from .prompt import PromptInjector
from .udp import UDPReceiver, UDPSender


@hookimpl
def register_pipelines(register):
    from ._plugin import ScopeBusPipeline
    register(ScopeBusPipeline)

__all__ = [
    "UDPSender",
    "UDPReceiver",
    "render_text_overlay",
    "normalize_input",
    "tensor_to_pil",
    "PromptInjector",
]
