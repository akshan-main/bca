"""Base LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in the conversation."""

    role: str  # "system", "user", "assistant", "tool"
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str = ""  # For tool result messages
    name: str = ""  # Tool name for tool results


class ToolCall(BaseModel):
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result from executing a tool."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


class ToolDefinition(BaseModel):
    """Definition of a tool the LLM can call."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response from the LLM."""

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    system_fingerprint: str = ""  # OpenAI cluster fingerprint for reproducibility auditing

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> LLMResponse:
        """Send a completion request to the LLM."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream a completion response."""
        ...
