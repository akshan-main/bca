"""MCP Server — expose CeGraph tools via the Model Context Protocol.

Implements the MCP protocol (JSON-RPC 2.0 over stdio) from scratch.
No external MCP SDK dependency — just pure protocol implementation.

This is the bridge that makes CeGraph's knowledge graph available to:
  - Claude Code (via ~/.claude/mcp_servers.json)
  - Cursor (via .cursor/mcp.json)
  - Any MCP-compatible AI coding tool

Protocol reference: https://modelcontextprotocol.io/specification
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from cegraph.config import GRAPH_DB_FILE, find_project_root, get_cegraph_dir

logger = logging.getLogger("cegraph.mcp")


class MCPServer:
    """Model Context Protocol server for CeGraph.

    Exposes the knowledge graph and context assembler as MCP tools,
    allowing AI assistants to query your codebase.
    """

    PROTOCOL_VERSION = "2024-11-05"
    SERVER_NAME = "cegraph"
    SERVER_VERSION = "0.1.0"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or find_project_root() or Path.cwd()
        self._graph = None
        self._store = None
        self._query = None
        self._assembler = None
        self._tools = self._define_tools()

    def _ensure_graph(self):
        """Lazy-load the knowledge graph."""
        if self._graph is not None:
            return

        from cegraph.graph.query import GraphQuery
        from cegraph.graph.store import GraphStore

        db_path = get_cegraph_dir(self.root) / GRAPH_DB_FILE
        if not db_path.exists():
            raise RuntimeError(
                f"No CeGraph index found at {db_path}. "
                "Run 'cegraph init' first."
            )

        self._store = GraphStore(db_path)
        self._graph = self._store.load()
        if self._graph is None:
            raise RuntimeError("Failed to load knowledge graph")
        self._query = GraphQuery(self._graph, self._store)

    def _ensure_cag(self):
        """Lazy-load the context assembler."""
        if self._assembler is not None:
            return

        self._ensure_graph()
        from cegraph.context.engine import ContextAssembler
        self._assembler = ContextAssembler(self.root, self._graph, self._query)

    def _define_tools(self) -> list[dict]:
        """Define the MCP tools we expose."""
        return [
            {
                "name": "cag_assemble",
                "description": (
                    "Assemble budgeted code context for a task using CAG "
                    "(Context Assembly Generation). Given a task description, "
                    "returns relevant code symbols from the knowledge graph -- "
                    "dependency-ordered, relevance-scored, within a token budget. "
                    "Use this BEFORE reading files to get scoped context."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Natural language description of what you're doing",
                        },
                        "token_budget": {
                            "type": "integer",
                            "description": "Maximum tokens to include (default: 8000)",
                            "default": 8000,
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["precise", "smart", "thorough"],
                            "description": "How aggressively to expand context (default: smart)",
                            "default": "smart",
                        },
                        "focus_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional files to prioritize",
                        },
                    },
                    "required": ["task"],
                },
            },
            {
                "name": "search_code",
                "description": (
                    "Search for code symbols (functions, classes, methods) in the "
                    "knowledge graph. Returns matching symbols with their locations."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (symbol name or keyword)",
                        },
                        "kind": {
                            "type": "string",
                            "description": "Filter by kind: function, class, method",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "who_calls",
                "description": (
                    "Find all callers of a function or method. "
                    "Traverses the call graph to show what code depends on a symbol."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Name of the function/method",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Max depth of caller traversal (default: 3)",
                            "default": 3,
                        },
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "impact_of",
                "description": (
                    "Analyze the blast radius of changing a symbol. "
                    "Returns risk score, affected files, and dependency chains."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Name of the symbol to analyze",
                        },
                    },
                    "required": ["symbol"],
                },
            },
            {
                "name": "get_structure",
                "description": (
                    "Get the overall codebase structure — files and their key symbols."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "find_related",
                "description": (
                    "Find symbols related to a given symbol through the knowledge graph. "
                    "Shows callers, callees, siblings, and other connections."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Name of the symbol",
                        },
                    },
                    "required": ["symbol"],
                },
            },
        ]

    # =========================================================================
    # Tool Implementations
    # =========================================================================

    def _handle_tool_call(self, name: str, arguments: dict) -> Any:
        """Execute a tool and return the result."""
        if name == "cag_assemble":
            return self._tool_cag_assemble(arguments)
        elif name == "search_code":
            return self._tool_search_code(arguments)
        elif name == "who_calls":
            return self._tool_who_calls(arguments)
        elif name == "impact_of":
            return self._tool_impact_of(arguments)
        elif name == "get_structure":
            return self._tool_get_structure(arguments)
        elif name == "find_related":
            return self._tool_find_related(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    def _tool_cag_assemble(self, args: dict) -> str:
        """CAG: Assemble budgeted context for a task."""
        self._ensure_cag()
        from cegraph.context.models import ContextStrategy

        strategy_map = {
            "precise": ContextStrategy.PRECISE,
            "smart": ContextStrategy.SMART,
            "thorough": ContextStrategy.THOROUGH,
        }

        package = self._assembler.assemble(
            task=args["task"],
            token_budget=args.get("token_budget", 8000),
            strategy=strategy_map.get(args.get("strategy", "smart"), ContextStrategy.SMART),
            focus_files=args.get("focus_files"),
        )

        # Return the rendered context + summary
        summary = package.summary()
        rendered = package.render()
        return f"{summary}\n\n---\n\n{rendered}"

    def _tool_search_code(self, args: dict) -> str:
        self._ensure_graph()
        results = self._query.find_symbol(args["query"])
        if not results:
            return f"No symbols found matching '{args['query']}'"

        lines = []
        for sid in results[:20]:
            data = self._graph.nodes.get(sid, {})
            name = data.get("name", sid)
            kind = data.get("kind", "")
            fp = data.get("file_path", "")
            line = data.get("line_start", 0)
            if args.get("kind") and kind != args["kind"]:
                continue
            lines.append(f"  {name} ({kind}) at {fp}:{line}")

        return "\n".join(lines) if lines else f"No symbols found matching '{args['query']}'"

    def _tool_who_calls(self, args: dict) -> str:
        self._ensure_graph()
        callers = self._query.who_calls(
            args["symbol"], max_depth=args.get("max_depth", 3)
        )
        if not callers:
            return f"No callers found for '{args['symbol']}'"

        lines = [f"Callers of '{args['symbol']}':"]
        for c in callers:
            indent = "  " * c.get("depth", 1)
            lines.append(f"{indent}{c['name']} ({c['kind']}) at {c['file_path']}:{c['line']}")
        return "\n".join(lines)

    def _tool_impact_of(self, args: dict) -> str:
        self._ensure_graph()
        impact = self._query.impact_of(args["symbol"])
        if not impact.get("found"):
            return f"Symbol '{args['symbol']}' not found in the knowledge graph"

        risk = impact.get("risk_score", 0)
        risk_label = "LOW" if risk < 0.2 else "MEDIUM" if risk < 0.5 else "HIGH"

        lines = [
            f"Impact Analysis for '{args['symbol']}':",
            f"  Risk: {risk_label} ({risk:.1%})",
            f"  Direct callers: {len(impact.get('direct_callers', []))}",
            f"  Transitive callers: {len(impact.get('transitive_callers', []))}",
            f"  Affected files: {len(impact.get('affected_files', []))}",
        ]

        files = impact.get("affected_files", [])
        if files:
            lines.append("\n  Affected files:")
            for f in files:
                lines.append(f"    - {f}")

        return "\n".join(lines)

    def _tool_get_structure(self, _args: dict) -> str:
        self._ensure_graph()
        structure = self._query.get_structure()

        lines = ["Codebase structure:"]
        self._render_structure(structure, lines, indent=1)
        return "\n".join(lines)

    def _render_structure(self, node: dict, lines: list[str], indent: int) -> None:
        """Recursively render the structure tree."""
        prefix = "  " * indent
        for key, value in sorted(node.items()):
            if key.startswith("_"):
                continue
            if isinstance(value, dict):
                syms = value.get("_symbols", 0)
                lang = value.get("_language", "")
                # Check if it's a file (has _language/_symbols) or a directory
                if "_language" in value or "_symbols" in value:
                    lang_str = f" [{lang}]" if lang else ""
                    lines.append(f"{prefix}{key}{lang_str} ({syms} symbols)")
                else:
                    lines.append(f"\n{prefix}{key}/")
                    self._render_structure(value, lines, indent + 1)

    def _tool_find_related(self, args: dict) -> str:
        self._ensure_graph()
        related = self._query.find_related(args["symbol"])
        if not related:
            return f"No related symbols found for '{args['symbol']}'"

        lines = [f"Related to '{args['symbol']}':"]
        for r in related:
            lines.append(f"  {r['name']} ({r['kind']}) — {r.get('relation', 'related')}")
        return "\n".join(lines)

    # =========================================================================
    # MCP Protocol Implementation (JSON-RPC 2.0 over stdio)
    # =========================================================================

    async def run_stdio(self) -> None:
        """Run the MCP server over stdio (the standard transport)."""
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin.buffer)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout.buffer
        )
        writer = asyncio.StreamWriter(
            writer_transport, writer_protocol,
            None, asyncio.get_event_loop(),
        )

        logger.info("CeGraph MCP server started (stdio transport)")

        while True:
            try:
                message = await self._read_message(reader)
                if message is None:
                    break
                response = self._handle_message(message)
                if response is not None:
                    await self._write_message(writer, response)
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                break

        logger.info("MCP server shutting down")

    async def _read_message(self, reader: asyncio.StreamReader) -> dict | None:
        """Read a JSON-RPC message with Content-Length header."""
        # Read headers
        content_length = 0
        while True:
            line = await reader.readline()
            if not line:
                return None
            line = line.decode("utf-8").strip()
            if not line:
                break  # End of headers
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":")[1].strip())

        if content_length == 0:
            return None

        # Read body
        body = await reader.readexactly(content_length)
        return json.loads(body.decode("utf-8"))

    async def _write_message(self, writer: asyncio.StreamWriter, message: dict) -> None:
        """Write a JSON-RPC response with Content-Length header."""
        body = json.dumps(message).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode()
        writer.write(header + body)
        await writer.drain()

    def _handle_message(self, message: dict) -> dict | None:
        """Route a JSON-RPC message to the appropriate handler."""
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params", {})

        # Notifications (no id) don't get responses
        if msg_id is None:
            self._handle_notification(method, params)
            return None

        try:
            result = self._dispatch(method, params)
            return {"jsonrpc": "2.0", "id": msg_id, "result": result}
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": str(e)},
            }

    def _handle_notification(self, method: str, params: dict) -> None:
        """Handle a notification (no response needed)."""
        if method == "notifications/initialized":
            logger.info("Client initialized")
        elif method == "notifications/cancelled":
            logger.info(f"Request cancelled: {params.get('requestId')}")

    def _dispatch(self, method: str, params: dict) -> Any:
        """Dispatch a JSON-RPC method to its handler."""
        if method == "initialize":
            return self._rpc_initialize(params)
        elif method == "tools/list":
            return self._rpc_tools_list(params)
        elif method == "tools/call":
            return self._rpc_tools_call(params)
        elif method == "resources/list":
            return self._rpc_resources_list(params)
        elif method == "resources/read":
            return self._rpc_resources_read(params)
        elif method == "ping":
            return {}
        else:
            raise ValueError(f"Unknown method: {method}")

    def _rpc_initialize(self, params: dict) -> dict:
        """Handle the initialize handshake."""
        return {
            "protocolVersion": self.PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
            "serverInfo": {
                "name": self.SERVER_NAME,
                "version": self.SERVER_VERSION,
            },
        }

    def _rpc_tools_list(self, params: dict) -> dict:
        """List available tools."""
        return {"tools": self._tools}

    def _rpc_tools_call(self, params: dict) -> dict:
        """Call a tool and return the result."""
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            result = self._handle_tool_call(name, arguments)
            return {
                "content": [{"type": "text", "text": result}],
                "isError": False,
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error: {e}"}],
                "isError": True,
            }

    def _rpc_resources_list(self, params: dict) -> dict:
        """List available resources (codebase files)."""
        resources = []
        try:
            self._ensure_graph()
            for node_id, data in self._graph.nodes(data=True):
                if data.get("type") == "file":
                    fp = data.get("path", "")
                    resources.append({
                        "uri": f"file://{fp}",
                        "name": fp,
                        "mimeType": "text/plain",
                    })
        except Exception:
            pass

        return {"resources": resources}

    def _rpc_resources_read(self, params: dict) -> dict:
        """Read a resource by URI."""
        uri = params.get("uri", "")
        # Strip file:// prefix
        fp = uri.replace("file://", "")
        full_path = self.root / fp

        # Security: prevent path traversal outside project root
        try:
            full_path.resolve().relative_to(self.root.resolve())
        except ValueError:
            msg = f"Access denied: {fp} is outside the project root"
            return {"contents": [{"uri": uri, "text": msg}]}

        if not full_path.exists():
            return {"contents": [{"uri": uri, "text": f"File not found: {fp}"}]}

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": content}]}
        except Exception as e:
            return {"contents": [{"uri": uri, "text": f"Error reading {fp}: {e}"}]}

    # =========================================================================
    # MCP Config Generators
    # =========================================================================

    @staticmethod
    def generate_claude_config(project_path: str | None = None) -> dict:
        """Generate MCP config for Claude Code (~/.claude/mcp_servers.json)."""
        return {
            "cegraph": {
                "command": "cegraph",
                "args": ["serve", "--transport", "stdio"],
                "cwd": project_path or ".",
            }
        }

    @staticmethod
    def generate_cursor_config(project_path: str | None = None) -> dict:
        """Generate MCP config for Cursor (.cursor/mcp.json)."""
        return {
            "mcpServers": {
                "cegraph": {
                    "command": "cegraph",
                    "args": ["serve", "--transport", "stdio"],
                    "cwd": project_path or ".",
                }
            }
        }
