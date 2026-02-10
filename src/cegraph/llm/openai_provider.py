"""OpenAI LLM provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from cegraph.llm.base import (
    LLMProvider,
    LLMResponse,
    Message,
    ToolCall,
    ToolDefinition,
)


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI and OpenAI-compatible APIs (Ollama, vLLM, etc.)."""

    def __init__(
        self, model: str = "gpt-4o", api_key: str | None = None, base_url: str | None = None
    ) -> None:
        super().__init__(model, api_key, base_url)
        self._client = None
        self._async_client = None

    def _get_client(self):
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                from cegraph.exceptions import ProviderNotAvailableError
                raise ProviderNotAvailableError("openai", "openai")

            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._async_client = AsyncOpenAI(**kwargs)
        return self._async_client

    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """Convert our Message format to OpenAI's format."""
        result = []
        for msg in messages:
            if msg.role == "tool":
                result.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
            elif msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    })
                result.append({
                    "role": msg.role,
                    "content": msg.content or None,
                    "tool_calls": tool_calls,
                })
            else:
                result.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        return result

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        """Convert tool definitions to OpenAI's format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = self._format_tools(tools)

        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
