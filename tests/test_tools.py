"""Tests for the agent tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from cegraph.graph.builder import GraphBuilder
from cegraph.graph.query import GraphQuery
from cegraph.search.hybrid import HybridSearch
from cegraph.tools.definitions import CeGraphTools, get_all_tools, get_tool_definitions


class TestCeGraphTools:
    def _build_tools(self, tmp_project: Path) -> CeGraphTools:
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)
        query = GraphQuery(graph)
        search = HybridSearch(tmp_project, graph)
        return CeGraphTools(tmp_project, graph, query, search)

    def test_search_code(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.search_code("calculate_total")
        assert "calculate_total" in result
        assert "No results" not in result

    def test_search_symbols(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.search_symbols("User")
        assert "User" in result
        assert "class" in result

    def test_who_calls(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.who_calls("helper_function")
        assert "main" in result.lower() or "Callers" in result

    def test_what_calls(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.what_calls("main")
        assert len(result) > 0

    def test_impact_of(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.impact_of("calculate_total")
        assert "Risk score" in result or "Impact" in result

    def test_read_file(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.read_file("main.py")
        assert "def main" in result

    def test_read_file_with_lines(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.read_file("main.py", start_line=1, end_line=5)
        assert "1" in result  # Line numbers
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) <= 5

    def test_read_file_not_found(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.read_file("nonexistent.py")
        assert "not found" in result.lower()

    def test_read_file_outside_root(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.read_file("../../etc/passwd")
        assert "denied" in result.lower() or "not found" in result.lower()

    def test_edit_file(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)

        # Read original
        original = tools.read_file("utils.py")
        assert "TAX_RATE = 0.08" in original

        # Edit
        result = tools.edit_file("utils.py", "TAX_RATE = 0.08", "TAX_RATE = 0.10")
        assert "Successfully" in result

        # Verify edit
        updated = tools.read_file("utils.py")
        assert "TAX_RATE = 0.10" in updated

    def test_edit_file_text_not_found(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.edit_file("utils.py", "NONEXISTENT_TEXT", "replacement")
        assert "not found" in result.lower()

    def test_write_file(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.write_file("new_file.py", "# New file\nprint('hello')\n")
        assert "Successfully" in result

        content = tools.read_file("new_file.py")
        assert "New file" in content

    def test_list_files(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.list_files()
        assert "main.py" in result

    def test_list_files_pattern(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.list_files(pattern="*.py")
        assert "main.py" in result

    def test_get_context(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.get_context("User")
        assert "User" in result
        assert "class" in result.lower()

    def test_run_command_safe(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.run_command("python --version")
        assert "Python" in result

    def test_run_command_blocked(self, tmp_project: Path):
        tools = self._build_tools(tmp_project)
        result = tools.run_command("rm -rf /")
        assert "not allowed" in result.lower()

    def test_run_command_injection_semicolon(self, tmp_project: Path):
        """Regression: semicolons must not allow chaining arbitrary commands."""
        tools = self._build_tools(tmp_project)
        result = tools.run_command("git status; echo INJECTED")
        # shlex.split produces ["git", "status;", "echo", "INJECTED"]
        # but the subprocess runs without shell=True so "status;" is
        # treated as a literal git arg and git will error or ignore it.
        assert "INJECTED" not in result

    def test_run_command_injection_pipe(self, tmp_project: Path):
        """Regression: pipes must not be interpreted."""
        tools = self._build_tools(tmp_project)
        result = tools.run_command("ls | cat /etc/passwd")
        # shlex gives ["ls", "|", "cat", "/etc/passwd"] — with shell=False
        # ls just treats "|" as a literal filename, no piping occurs.
        assert "root:" not in result

    def test_run_command_git_destructive_blocked(self, tmp_project: Path):
        """Regression: git push/reset/etc. must be blocked."""
        tools = self._build_tools(tmp_project)
        result = tools.run_command("git push --force")
        assert "read-only" in result.lower() or "not allowed" in result.lower()

    def test_run_command_git_safe_subcommands(self, tmp_project: Path):
        """git status/diff/log should still work."""
        tools = self._build_tools(tmp_project)
        # Just check it doesn't reject the command (may fail on non-git dirs)
        result = tools.run_command("git status")
        assert "not allowed" not in result.lower()


class TestToolRegistry:
    def test_get_tool_definitions(self):
        defs = get_tool_definitions()
        assert len(defs) > 0

        names = [d.name for d in defs]
        assert "search_code" in names
        assert "who_calls" in names
        assert "impact_of" in names
        assert "read_file" in names
        assert "edit_file" in names
        assert "run_command" in names

    def test_all_tools_have_descriptions(self):
        for defn in get_tool_definitions():
            assert defn.description, f"Tool {defn.name} has no description"
            assert defn.parameters, f"Tool {defn.name} has no parameters"

    def test_get_all_tools_creates_registry(self, tmp_project: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)
        query = GraphQuery(graph)
        search = HybridSearch(tmp_project, graph)
        registry = get_all_tools(tmp_project, graph, query, search)

        assert len(registry.list_tools()) > 0
        assert registry.get("search_code") is not None
        assert registry.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_registry_execute(self, tmp_project: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)
        query = GraphQuery(graph)
        search = HybridSearch(tmp_project, graph)
        registry = get_all_tools(tmp_project, graph, query, search)

        result = await registry.execute("search_code", {"query": "main"})
        assert "main" in result

    @pytest.mark.asyncio
    async def test_registry_execute_unknown_tool(self, tmp_project: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)
        query = GraphQuery(graph)
        search = HybridSearch(tmp_project, graph)
        registry = get_all_tools(tmp_project, graph, query, search)

        result = await registry.execute("nonexistent_tool", {})
        assert "Unknown tool" in result
