"""LLM clients for Vision-Language Models."""

from .base import BaseModelClient, ModelAction
from .gemini import GeminiClient
from .local import LocalVLLMClient
from .openai import ChatGPTClient

# Factory function
def create_client(provider: str, config):
    """Create LLM client based on provider name."""
    clients = {
        "gemini": GeminiClient,
        "chatgpt": ChatGPTClient,
        "openai": ChatGPTClient,
        "local_vllm": LocalVLLMClient,
        "local": LocalVLLMClient,
    }
    client_class = clients.get(provider.lower())
    if not client_class:
        raise ValueError(f"Unknown provider: {provider}")
    return client_class(config)

__all__ = [
    "BaseModelClient",
    "ModelAction",
    "GeminiClient",
    "LocalVLLMClient",
    "ChatGPTClient",
    "create_client",
]
