"""Custom exceptions for CeGraph."""


class CeGraphError(Exception):
    """Base exception for all CeGraph errors."""


class ConfigError(CeGraphError):
    """Configuration-related errors."""


class ParserError(CeGraphError):
    """Code parsing errors."""


class GraphError(CeGraphError):
    """Knowledge graph errors."""


class LLMError(CeGraphError):
    """LLM provider errors."""


class ToolError(CeGraphError):
    """Agent tool execution errors."""


class IndexError(CeGraphError):
    """Indexing errors."""


class ProviderNotAvailableError(LLMError):
    """Raised when an LLM provider's SDK is not installed."""

    def __init__(self, provider: str, package: str):
        super().__init__(
            f"Provider '{provider}' requires the '{package}' package. "
            f"Install it with: pip install cegraph[{provider}]"
        )
