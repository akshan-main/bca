"""ReAct-style agent loop for CeGraph.

Implements the core agentic pattern:
1. Think: Reason about the task and what information is needed
2. Act: Call tools to gather info or make changes
3. Observe: Process tool results
4. Repeat until task is complete
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from cegraph.agent.prompts import get_system_prompt
from cegraph.llm.base import LLMProvider, LLMResponse, Message, ToolCall, ToolResult
from cegraph.tools.registry import ToolRegistry


@dataclass
class AgentStep:
    """A single step in the agent loop."""

    iteration: int
    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    response: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Final result from the agent loop."""

    answer: str
    steps: list[AgentStep]
    total_iterations: int
    total_tokens: int
    success: bool = True
    error: str = ""


class AgentLoop:
    """ReAct agent loop that iteratively uses tools to complete tasks.

    The agent:
    1. Receives a task from the user
    2. Reasons about what to do next
    3. Calls tools to gather information or make changes
    4. Processes tool results
    5. Repeats until it has enough info to give a final answer
    6. Presents the answer/changes for user approval
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        project_name: str = "",
        max_iterations: int = 15,
        on_step: Callable[[AgentStep], None] | None = None,
        on_approval_needed: Callable[[str], bool] | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.project_name = project_name
        self.max_iterations = max_iterations
        self.on_step = on_step
        self.on_approval_needed = on_approval_needed

    async def run(self, task: str, context: str = "") -> AgentResult:
        """Run the agent loop for a given task.

        Args:
            task: The user's request/task.
            context: Additional context (e.g., file content, error messages).

        Returns:
            AgentResult with the final answer and step history.
        """
        messages: list[Message] = [
            Message(role="system", content=get_system_prompt(self.project_name)),
        ]

        # Add context if provided
        user_content = task
        if context:
            user_content = f"{task}\n\nContext:\n{context}"
        messages.append(Message(role="user", content=user_content))

        steps: list[AgentStep] = []
        total_tokens = 0

        for iteration in range(self.max_iterations):
            step = AgentStep(iteration=iteration + 1)

            try:
                response = await self.llm.complete(
                    messages=messages,
                    tools=self.tools.get_definitions(),
                    temperature=0.0,
                )
            except Exception as e:
                return AgentResult(
                    answer="",
                    steps=steps,
                    total_iterations=iteration + 1,
                    total_tokens=total_tokens,
                    success=False,
                    error=f"LLM error: {e}",
                )

            step.usage = response.usage
            total_tokens += sum(response.usage.values())

            if response.has_tool_calls:
                # Agent wants to use tools
                step.thought = response.content
                step.tool_calls = response.tool_calls

                # Add assistant message with tool calls
                messages.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                # Execute each tool call
                for tc in response.tool_calls:
                    result = await self.tools.execute(tc.name, tc.arguments)
                    tool_result = ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content=result,
                    )
                    step.tool_results.append(tool_result)

                    # Add tool result message
                    messages.append(
                        Message(
                            role="tool",
                            content=result,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )

                # Notify step callback
                if self.on_step:
                    self.on_step(step)

                steps.append(step)

            else:
                # Agent is done - final answer
                step.response = response.content

                if self.on_step:
                    self.on_step(step)

                steps.append(step)

                return AgentResult(
                    answer=response.content,
                    steps=steps,
                    total_iterations=iteration + 1,
                    total_tokens=total_tokens,
                    success=True,
                )

        # Max iterations reached
        return AgentResult(
            answer="I reached the maximum number of iterations. Here's what I've found so far based on my analysis.",
            steps=steps,
            total_iterations=self.max_iterations,
            total_tokens=total_tokens,
            success=False,
            error="Max iterations reached",
        )

    async def ask(self, question: str) -> str:
        """Simple Q&A mode - ask a question about the codebase.

        Returns just the answer string.
        """
        result = await self.run(question)
        return result.answer
