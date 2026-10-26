from switchyard.providers.base import Provider, ProviderError
from switchyard.providers.mock import MockProvider


def build_providers() -> dict[str, Provider]:
    """Instantiate every provider adapter. Real ones are lazy about credentials:
    they only fail (and trigger fallback) when actually called without a key."""
    from switchyard.providers.anthropic import AnthropicProvider
    from switchyard.providers.openai import OpenAIProvider

    return {
        "mock": MockProvider(),
        "openai": OpenAIProvider(),
        "anthropic": AnthropicProvider(),
    }


__all__ = ["MockProvider", "Provider", "ProviderError", "build_providers"]
