"""Built-in tool definitions and implementations for the CeGraph agent."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import networkx as nx

from cegraph.graph.query import GraphQuery
from cegraph.llm.base import ToolDefinition
from cegraph.search.hybrid import HybridSearch
from cegraph.tools.registry import ToolRegistry


class CeGraphTools:
    """Collection of built-in tools for the CeGraph agent.

    Each tool interacts with the knowledge graph, search engine,
    and/or the filesystem to provide the agent with accurate information.
    """

    def __init__(
        self,
        root: Path,
        graph: nx.DiGraph,
        query: GraphQuery,
        search: HybridSearch,
    ) -> None:
        self.root = root
        self.graph = graph
        self.query = query
        self.search = search

    def search_code(self, query: str, file_pattern: str = "", max_results: int = 10) -> str:
        """Search for code in the repository matching a query."""
        results = self.search.search(query, file_pattern, max_results=max_results)
        if not results:
            return f"No results found for '{query}'"

        output = []
        for r in results:
            header = f"**{r.file_path}:{r.line_number}**"
            if r.symbol_name:
                header += f" (in `{r.symbol_name}`)"
            output.append(header)

            if r.context_before:
                for line in r.context_before:
                    output.append(f"  {line}")
            output.append(f"→ {r.line_content}")
            if r.context_after:
                for line in r.context_after:
                    output.append(f"  {line}")
            output.append("")

        return "\n".join(output)

    def search_symbols(self, query: str, kind: str = "", max_results: int = 10) -> str:
        """Search for symbol definitions (functions, classes, etc.)."""
        results = self.search.search_symbols(query, kind, max_results)
        if not results:
            return f"No symbols found matching '{query}'"

        output = []
        for r in results:
            output.append(
                f"- **{r['qualified_name']}** ({r['kind']}) at {r['file_path']}:{r['line']}"
            )
            if r.get("signature"):
                output.append(f"  `{r['signature']}`")
        return "\n".join(output)

    def who_calls(self, symbol_name: str, max_depth: int = 2) -> str:
        """Find all callers of a function/method."""
        results = self.query.who_calls(symbol_name, max_depth=max_depth)
        if not results:
            return f"No callers found for '{symbol_name}'"

        output = [f"Callers of `{symbol_name}`:"]
        for r in results:
            indent = "  " * r["depth"]
            output.append(
                f"{indent}← **{r['name']}** ({r['kind']}) at {r['file_path']}:{r['line']}"
            )
        return "\n".join(output)

    def what_calls(self, symbol_name: str) -> str:
        """Find all symbols called by a function/method."""
        results = self.query.what_calls(symbol_name)
        if not results:
            return f"No callees found for '{symbol_name}'"

        output = [f"`{symbol_name}` calls:"]
        for r in results:
            output.append(
                f"→ **{r['name']}** ({r['kind']}) at {r['file_path']}:{r['line']}"
            )
        return "\n".join(output)

    def impact_of(self, symbol_name: str) -> str:
        """Analyze the impact of changing a symbol."""
        result = self.query.impact_of(symbol_name)
        if not result.get("found"):
            return f"Symbol '{symbol_name}' not found in the knowledge graph"

        output = [f"Impact analysis for `{symbol_name}`:"]
        output.append(f"Risk score: {result['risk_score']:.1%}")
        output.append(f"Direct callers: {len(result['direct_callers'])}")
        output.append(f"Transitive callers: {len(result['transitive_callers'])}")
        output.append(f"Affected files ({len(result['affected_files'])}):")
        for f in result["affected_files"]:
            output.append(f"  - {f}")
        return "\n".join(output)

    def read_file(self, file_path: str, start_line: int = 0, end_line: int = 0) -> str:
        """Read a file from the repository."""
        full_path = self.root / file_path
        if not full_path.exists():
            return f"File not found: {file_path}"
        if not full_path.is_file():
            return f"Not a file: {file_path}"

        # Security: ensure we're within the project root
        try:
            full_path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return f"Access denied: {file_path} is outside the project root"

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Error reading file: {e}"

        lines = content.splitlines()
        if start_line or end_line:
            start = max(0, start_line - 1)
            end = end_line if end_line else len(lines)
            lines = lines[start:end]
            # Add line numbers
            numbered = [
                f"{i + start + 1:4d} | {line}" for i, line in enumerate(lines)
            ]
            return "\n".join(numbered)

        # If file is too long, truncate with message
        if len(lines) > 200:
            numbered = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines[:200])]
            numbered.append(f"\n... ({len(lines) - 200} more lines)")
            return "\n".join(numbered)

        numbered = [f"{i + 1:4d} | {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)

    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file (creates or overwrites)."""
        full_path = self.root / file_path
        try:
            full_path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return f"Access denied: {file_path} is outside the project root"

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {file_path}"

    def edit_file(
        self, file_path: str, old_text: str, new_text: str
    ) -> str:
        """Replace specific text in a file (targeted edit, not full rewrite)."""
        full_path = self.root / file_path
        if not full_path.exists():
            return f"File not found: {file_path}"

        try:
            full_path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return f"Access denied: {file_path} is outside the project root"

        content = full_path.read_text(encoding="utf-8", errors="replace")
        if old_text not in content:
            return f"Text to replace not found in {file_path}"

        count = content.count(old_text)
        if count > 1:
            return f"Text to replace found {count} times. Please provide more context to make it unique."

        new_content = content.replace(old_text, new_text, 1)
        full_path.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {file_path}"

    def list_files(self, path: str = "", pattern: str = "") -> str:
        """List files in the repository."""
        target = self.root / path if path else self.root
        if not target.exists():
            return f"Path not found: {path}"

        try:
            target.resolve().relative_to(self.root.resolve())
        except ValueError:
            return f"Access denied: {path} is outside the project root"

        files = []
        if pattern:
            import fnmatch

            for p in sorted(target.rglob("*")):
                if p.is_file() and fnmatch.fnmatch(p.name, pattern):
                    files.append(str(p.relative_to(self.root)))
        else:
            for p in sorted(target.iterdir()):
                prefix = "d " if p.is_dir() else "f "
                files.append(prefix + str(p.relative_to(self.root)))

        return "\n".join(files) if files else "No files found"

    def get_structure(self, path: str = "") -> str:
        """Get the project structure with symbol counts."""
        structure = self.query.get_structure(path)
        return json.dumps(structure, indent=2) if structure else "No structure data"

    def get_context(self, symbol_name: str) -> str:
        """Get full context for a symbol including its source code and relationships."""
        symbol_ids = self.query.find_symbol(symbol_name)
        if not symbol_ids:
            return f"Symbol '{symbol_name}' not found"

        output = []
        for sid in symbol_ids[:3]:  # Limit to 3 matches
            info = self.query.get_symbol_info(sid)
            if not info:
                continue

            output.append(f"## {info.qualified_name} ({info.kind})")
            output.append(f"**File:** {info.file_path}:{info.line_start}-{info.line_end}")
            if info.signature:
                output.append(f"**Signature:** `{info.signature}`")
            if info.docstring:
                output.append(f"**Docstring:** {info.docstring[:200]}")

            # Show source code
            file_content = self.read_file(
                info.file_path,
                start_line=info.line_start,
                end_line=info.line_end,
            )
            output.append(f"\n```\n{file_content}\n```")

            # Show relationships
            if info.callers:
                output.append(f"\n**Called by:** {', '.join(c.split('::')[-1] for c in info.callers[:5])}")
            if info.callees:
                output.append(f"**Calls:** {', '.join(c.split('::')[-1] for c in info.callees[:5])}")
            if info.children:
                output.append(f"**Contains:** {', '.join(c.split('::')[-1] for c in info.children[:5])}")

            output.append("")

        return "\n".join(output)

    def run_command(self, command: str) -> str:
        """Run a shell command in the project root (for tests, lint, etc.)."""
        import shlex

        # Parse into argv — this prevents injection via shell metacharacters
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return f"Invalid command syntax: {e}"

        if not argv:
            return "Empty command"

        # Security: only allow specific executables (check first token only)
        allowed_executables = {
            "python", "python3", "pytest", "pip", "pip3",
            "npm", "node", "npx", "yarn", "pnpm",
            "go", "cargo", "make", "gradle", "mvn",
            "ruff", "black", "mypy", "flake8", "eslint", "prettier",
            "git", "ls", "cat", "head", "tail", "wc", "find", "grep",
        }
        # For git, only allow safe read-only subcommands
        _git_safe_subcommands = {"status", "diff", "log", "show", "branch", "tag"}

        exe = argv[0].lower()
        if exe not in allowed_executables:
            return f"Command not allowed for safety. Allowed executables: {', '.join(sorted(allowed_executables))}"

        if exe == "git":
            subcommand = argv[1].lower() if len(argv) > 1 else ""
            if subcommand not in _git_safe_subcommands:
                return f"Only read-only git subcommands are allowed: {', '.join(sorted(_git_safe_subcommands))}"

        try:
            result = subprocess.run(
                argv,
                shell=False,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            # Truncate very long output
            if len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Command timed out after 60 seconds"
        except Exception as e:
            return f"Error running command: {e}"


# Tool definitions for the LLM

_TOOL_DEFINITIONS = [
    ToolDefinition(
        name="search_code",
        description="Search for code in the repository matching a text query. Returns matching lines with context and enclosing symbol information.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "file_pattern": {"type": "string", "description": "Glob pattern to filter files (e.g., '*.py')", "default": ""},
                "max_results": {"type": "integer", "description": "Maximum number of results", "default": 10},
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="search_symbols",
        description="Search for symbol definitions (functions, classes, methods, etc.) by name. More precise than search_code for finding definitions.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Symbol name to search for"},
                "kind": {"type": "string", "description": "Filter by symbol kind: function, class, method, variable", "default": ""},
                "max_results": {"type": "integer", "description": "Maximum results", "default": 10},
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="who_calls",
        description="Find all callers of a function or method. Answers: 'Who calls this function?'",
        parameters={
            "type": "object",
            "properties": {
                "symbol_name": {"type": "string", "description": "Name of the function/method"},
                "max_depth": {"type": "integer", "description": "How many levels of callers to traverse", "default": 2},
            },
            "required": ["symbol_name"],
        },
    ),
    ToolDefinition(
        name="what_calls",
        description="Find all functions/methods called by a given function. Answers: 'What does this function call?'",
        parameters={
            "type": "object",
            "properties": {
                "symbol_name": {"type": "string", "description": "Name of the function/method"},
            },
            "required": ["symbol_name"],
        },
    ),
    ToolDefinition(
        name="impact_of",
        description="Analyze the blast radius of changing a symbol. Shows direct callers, transitive callers, affected files, and risk score.",
        parameters={
            "type": "object",
            "properties": {
                "symbol_name": {"type": "string", "description": "Name of the symbol to analyze"},
            },
            "required": ["symbol_name"],
        },
    ),
    ToolDefinition(
        name="read_file",
        description="Read a file from the repository. Can read specific line ranges.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative path to the file"},
                "start_line": {"type": "integer", "description": "Start line number (1-indexed)", "default": 0},
                "end_line": {"type": "integer", "description": "End line number", "default": 0},
            },
            "required": ["file_path"],
        },
    ),
    ToolDefinition(
        name="edit_file",
        description="Make a targeted edit to a file by replacing specific text. Use this for precise changes instead of rewriting entire files.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative path to the file"},
                "old_text": {"type": "string", "description": "The exact text to find and replace"},
                "new_text": {"type": "string", "description": "The replacement text"},
            },
            "required": ["file_path", "old_text", "new_text"],
        },
    ),
    ToolDefinition(
        name="write_file",
        description="Write content to a new file. Use edit_file for modifying existing files.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Relative path for the new file"},
                "content": {"type": "string", "description": "The file content to write"},
            },
            "required": ["file_path", "content"],
        },
    ),
    ToolDefinition(
        name="list_files",
        description="List files and directories in the repository.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Subdirectory to list (empty for root)", "default": ""},
                "pattern": {"type": "string", "description": "Filter by filename pattern (e.g., '*.py')", "default": ""},
            },
        },
    ),
    ToolDefinition(
        name="get_context",
        description="Get comprehensive context for a symbol: source code, relationships (callers, callees), and metadata. Use this before making changes to understand the full picture.",
        parameters={
            "type": "object",
            "properties": {
                "symbol_name": {"type": "string", "description": "Name of the symbol"},
            },
            "required": ["symbol_name"],
        },
    ),
    ToolDefinition(
        name="get_structure",
        description="Get the project directory structure with symbol counts per file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Subdirectory to show (empty for full project)", "default": ""},
            },
        },
    ),
    ToolDefinition(
        name="run_command",
        description="Run a shell command in the project root. Use for running tests, linters, and build commands. Limited to safe commands only.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run"},
            },
            "required": ["command"],
        },
    ),
]


def get_tool_definitions() -> list[ToolDefinition]:
    """Get all built-in tool definitions."""
    return _TOOL_DEFINITIONS.copy()


def get_all_tools(
    root: Path,
    graph: nx.DiGraph,
    query: GraphQuery,
    search: HybridSearch,
) -> ToolRegistry:
    """Create a ToolRegistry populated with all built-in tools."""
    tools = CeGraphTools(root, graph, query, search)
    registry = ToolRegistry()

    # Map tool definitions to their implementations
    impl_map = {
        "search_code": tools.search_code,
        "search_symbols": tools.search_symbols,
        "who_calls": tools.who_calls,
        "what_calls": tools.what_calls,
        "impact_of": tools.impact_of,
        "read_file": tools.read_file,
        "edit_file": tools.edit_file,
        "write_file": tools.write_file,
        "list_files": tools.list_files,
        "get_context": tools.get_context,
        "get_structure": tools.get_structure,
        "run_command": tools.run_command,
    }

    for defn in _TOOL_DEFINITIONS:
        func = impl_map.get(defn.name)
        if func:
            registry.register(func, defn)

    return registry
