"""Tests for the MCP (Model Context Protocol) server."""

from __future__ import annotations

from pathlib import Path

import pytest

from cegraph.graph.builder import GraphBuilder
from cegraph.graph.store import GraphStore
from cegraph.mcp.server import MCPServer


@pytest.fixture
def mcp_server(tmp_project: Path, tmp_path: Path):
    """Create an MCP server with a built graph."""
    builder = GraphBuilder()
    graph = builder.build_from_directory(tmp_project)

    db_path = tmp_path / "test.db"
    store = GraphStore(db_path)
    store.save(graph, metadata={"stats": builder.get_stats()})
    store.close()

    # Create .cegraph dir in the project
    cs_dir = tmp_project / ".cegraph"
    cs_dir.mkdir(exist_ok=True)
    # Copy DB to the expected location
    import shutil
    shutil.copy(db_path, cs_dir / "graph.db")

    server = MCPServer(root=tmp_project)
    return server


class TestMCPProtocol:
    def test_initialize(self, mcp_server: MCPServer):
        """Test the initialize handshake."""
        result = mcp_server._dispatch("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"},
        })
        assert result["protocolVersion"] == "2024-11-05"
        assert "capabilities" in result
        assert "serverInfo" in result
        assert result["serverInfo"]["name"] == "cegraph"

    def test_tools_list(self, mcp_server: MCPServer):
        """Test listing available tools."""
        result = mcp_server._dispatch("tools/list", {})
        tools = result["tools"]
        assert len(tools) > 0

        tool_names = [t["name"] for t in tools]
        assert "cag_assemble" in tool_names
        assert "search_code" in tool_names
        assert "who_calls" in tool_names
        assert "impact_of" in tool_names
        assert "get_structure" in tool_names

    def test_tool_schema(self, mcp_server: MCPServer):
        """Test that tool schemas are valid."""
        result = mcp_server._dispatch("tools/list", {})
        for tool in result["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_ping(self, mcp_server: MCPServer):
        """Test ping response."""
        result = mcp_server._dispatch("ping", {})
        assert result == {}


class TestMCPTools:
    def test_cag_assemble(self, mcp_server: MCPServer):
        """Test the cag_assemble tool."""
        result = mcp_server._dispatch("tools/call", {
            "name": "cag_assemble",
            "arguments": {
                "task": "fix the main function",
                "token_budget": 4000,
            },
        })
        assert not result.get("isError")
        content = result["content"][0]["text"]
        assert "CAG" in content or "Context" in content

    def test_search_code(self, mcp_server: MCPServer):
        """Test the search_code tool."""
        result = mcp_server._dispatch("tools/call", {
            "name": "search_code",
            "arguments": {"query": "main"},
        })
        assert not result.get("isError")
        content = result["content"][0]["text"]
        assert "main" in content

    def test_who_calls(self, mcp_server: MCPServer):
        """Test the who_calls tool."""
        result = mcp_server._dispatch("tools/call", {
            "name": "who_calls",
            "arguments": {"symbol": "helper_function"},
        })
        assert not result.get("isError")

    def test_impact_of(self, mcp_server: MCPServer):
        """Test the impact_of tool."""
        result = mcp_server._dispatch("tools/call", {
            "name": "impact_of",
            "arguments": {"symbol": "calculate_total"},
        })
        assert not result.get("isError")
        content = result["content"][0]["text"]
        assert "Impact" in content or "calculate_total" in content

    def test_get_structure(self, mcp_server: MCPServer):
        """Test the get_structure tool."""
        result = mcp_server._dispatch("tools/call", {
            "name": "get_structure",
            "arguments": {},
        })
        assert not result.get("isError")
        content = result["content"][0]["text"]
        assert "main.py" in content or "structure" in content.lower()

    def test_find_related(self, mcp_server: MCPServer):
        """Test the find_related tool."""
        result = mcp_server._dispatch("tools/call", {
            "name": "find_related",
            "arguments": {"symbol": "main"},
        })
        assert not result.get("isError")

    def test_unknown_tool(self, mcp_server: MCPServer):
        """Test calling an unknown tool."""
        result = mcp_server._dispatch("tools/call", {
            "name": "nonexistent_tool",
            "arguments": {},
        })
        assert result.get("isError")

    def test_search_no_results(self, mcp_server: MCPServer):
        """Test search with no results."""
        result = mcp_server._dispatch("tools/call", {
            "name": "search_code",
            "arguments": {"query": "zzz_nonexistent_symbol_zzz"},
        })
        assert not result.get("isError")
        content = result["content"][0]["text"]
        assert "No symbols found" in content


class TestMCPConfigGen:
    def test_generate_claude_config(self):
        config = MCPServer.generate_claude_config("/tmp/project")
        assert "cegraph" in config
        assert config["cegraph"]["command"] == "cegraph"

    def test_generate_cursor_config(self):
        config = MCPServer.generate_cursor_config("/tmp/project")
        assert "mcpServers" in config
        assert "cegraph" in config["mcpServers"]


class TestMCPSecurity:
    def test_resources_read_path_traversal(self, mcp_server: MCPServer):
        """Regression: reading files outside the project root must be denied."""
        result = mcp_server._dispatch("resources/read", {
            "uri": "file://../../../../etc/hosts",
        })
        text = result["contents"][0].get("text", "")
        assert "Access denied" in text
        assert "localhost" not in text

    def test_resources_read_valid_file(self, mcp_server: MCPServer):
        """Reading a file inside the project root should work."""
        result = mcp_server._dispatch("resources/read", {
            "uri": "file://main.py",
        })
        text = result["contents"][0].get("text", "")
        assert "Access denied" not in text


class TestMCPMessageHandling:
    def test_handle_notification(self, mcp_server: MCPServer):
        """Test notification handling (no response)."""
        result = mcp_server._handle_message({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        assert result is None  # Notifications don't get responses

    def test_handle_request(self, mcp_server: MCPServer):
        """Test request handling (has id, gets response)."""
        result = mcp_server._handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "ping",
            "params": {},
        })
        assert result is not None
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1

    def test_handle_unknown_method(self, mcp_server: MCPServer):
        """Test error handling for unknown methods."""
        result = mcp_server._handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown/method",
            "params": {},
        })
        assert "error" in result
