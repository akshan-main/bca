"""Anthropic Claude LLM provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from cegraph.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic's Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(model, api_key, base_url)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                from cegraph.exceptions import ProviderNotAvailableError
                raise ProviderNotAvailableError("anthropic", "anthropic")

            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    def _format_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """Convert our Message format to Anthropic's format.

        Returns (system_prompt, messages_list).
        """
        system = ""
        result = []

        for msg in messages:
            if msg.role == "system":
                system = msg.content
                continue

            if msg.role == "tool":
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.tool_calls:
                content: list[dict] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                result.append({"role": "assistant", "content": content})
            else:
                result.append({"role": msg.role, "content": msg.content})

        return system, result

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert tool definitions to Anthropic's format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        seed: int | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        system, formatted_msgs = self._format_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        response = await client.messages.create(**kwargs)

        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "",
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        system, formatted_msgs = self._format_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
