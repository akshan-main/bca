"""MCP (Model Context Protocol) Server for CeGraph.

Exposes CeGraph's tools to any MCP-compatible client:
  - Claude Code
  - Cursor
  - Windsurf
  - Any custom MCP client

Usage:
    cegraph serve              # Start the MCP server
    cegraph serve --transport stdio  # Explicit stdio transport
"""

from cegraph.mcp.server import MCPServer

__all__ = ["MCPServer"]
