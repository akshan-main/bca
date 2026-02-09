"""Factory for creating LLM providers from configuration."""

from __future__ import annotations

from cegraph.config import LLMConfig
from cegraph.llm.base import LLMProvider


def create_provider(config: LLMConfig) -> LLMProvider:
    """Create an LLM provider from configuration.

    Args:
        config: LLM configuration with provider, model, etc.

    Returns:
        An initialized LLM provider.

    Raises:
        ValueError: If the provider is unknown.
        ProviderNotAvailableError: If the provider's SDK is not installed.
    """
    provider = config.provider.lower()

    if provider == "openai" or provider == "local":
        from cegraph.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    elif provider == "anthropic":
        from cegraph.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Supported providers: openai, anthropic, local"
        )
