"""Tool registry for agent tools."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from cegraph.llm.base import ToolDefinition


class ToolRegistry:
    """Registry that manages available agent tools.

    Tools are functions decorated with @tool that the LLM agent can call.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}
        self._definitions: dict[str, ToolDefinition] = {}

    def register(self, func: Callable, definition: ToolDefinition) -> None:
        """Register a tool function with its definition."""
        self._tools[definition.name] = func
        self._definitions[definition.name] = definition

    def get(self, name: str) -> Callable | None:
        """Get a tool function by name."""
        return self._tools.get(name)

    def get_definition(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._definitions.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_definitions(self) -> list[ToolDefinition]:
        """Get all tool definitions (for passing to LLM)."""
        return list(self._definitions.values())

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name with the given arguments.

        Returns the result as a string.
        """
        func = self._tools.get(name)
        if func is None:
            return f"Error: Unknown tool '{name}'"

        try:
            if inspect.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)
            return str(result) if result is not None else "Done."
        except Exception as e:
            return f"Error executing tool '{name}': {e}"


# Decorator for defining tools
def tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
):
    """Decorator to mark a function as an agent tool.

    Usage:
        @tool("search_code", "Search for code matching a query", {...})
        def search_code(query: str, file_pattern: str = "") -> str:
            ...
    """

    def decorator(func: Callable) -> Callable:
        func._tool_definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or _infer_parameters(func),
        )
        return func

    return decorator


def _infer_parameters(func: Callable) -> dict[str, Any]:
    """Infer JSON Schema parameters from function signature."""
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = param.annotation
        json_type = "string"  # default

        if annotation != inspect.Parameter.empty:
            json_type = type_map.get(annotation, "string")

        properties[param_name] = {"type": json_type}

        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        else:
            if param.default is not None:
                properties[param_name]["default"] = param.default

    schema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema
