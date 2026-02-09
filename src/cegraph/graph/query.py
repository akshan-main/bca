"""High-level query interface for the knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from cegraph.graph.store import GraphStore


@dataclass
class SymbolInfo:
    """Rich information about a symbol."""

    id: str
    name: str
    qualified_name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    signature: str
    docstring: str = ""
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    parent: str = ""


class GraphQuery:
    """Query engine for the code knowledge graph.

    Provides high-level methods for common queries: who calls what,
    impact analysis, related symbols, etc.
    """

    def __init__(self, graph: nx.DiGraph, store: GraphStore | None = None) -> None:
        self.graph = graph
        self.store = store
        self._name_index: dict[str, list[str]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Build a name->node_id lookup index."""
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "symbol":
                continue
            name = data.get("name", "")
            qname = data.get("qualified_name", "")
            if name:
                self._name_index.setdefault(name, []).append(node_id)
            if qname and qname != name:
                self._name_index.setdefault(qname, []).append(node_id)

    def find_symbol(self, name: str) -> list[str]:
        """Find symbol node IDs matching a name (exact or qualified)."""
        results = self._name_index.get(name, [])
        if not results:
            # Try partial match
            for key, ids in self._name_index.items():
                if name.lower() in key.lower():
                    results.extend(ids)
        return list(set(results))

    def get_symbol_info(self, symbol_id: str) -> SymbolInfo | None:
        """Get detailed information about a symbol."""
        if not self.graph.has_node(symbol_id):
            return None

        data = self.graph.nodes[symbol_id]
        if data.get("type") != "symbol":
            return None

        # Get callers (reverse edges with kind=calls)
        callers = []
        for pred in self.graph.predecessors(symbol_id):
            edge_data = self.graph.edges[pred, symbol_id]
            if edge_data.get("kind") == "calls":
                callers.append(pred)

        # Get callees (forward edges with kind=calls)
        callees = []
        for succ in self.graph.successors(symbol_id):
            edge_data = self.graph.edges[symbol_id, succ]
            if edge_data.get("kind") == "calls":
                callees.append(succ)

        # Get children (contains edges)
        children = []
        for succ in self.graph.successors(symbol_id):
            edge_data = self.graph.edges[symbol_id, succ]
            if edge_data.get("kind") == "contains":
                children.append(succ)

        # Get parent
        parent = ""
        for pred in self.graph.predecessors(symbol_id):
            edge_data = self.graph.edges[pred, symbol_id]
            if edge_data.get("kind") == "contains":
                pred_data = self.graph.nodes.get(pred, {})
                if pred_data.get("type") == "symbol":
                    parent = pred
                    break

        return SymbolInfo(
            id=symbol_id,
            name=data.get("name", ""),
            qualified_name=data.get("qualified_name", ""),
            kind=data.get("kind", ""),
            file_path=data.get("file_path", ""),
            line_start=data.get("line_start", 0),
            line_end=data.get("line_end", 0),
            signature=data.get("signature", ""),
            docstring=data.get("docstring", ""),
            callers=callers,
            callees=callees,
            children=children,
            parent=parent,
        )

    def who_calls(self, name: str, max_depth: int = 1) -> list[dict]:
        """Find all callers of a symbol, optionally going N levels deep.

        Returns list of {symbol_id, name, file_path, line, depth}
        """
        symbol_ids = self.find_symbol(name)
        if not symbol_ids:
            return []

        results = []
        visited = set()

        def _traverse(node_id: str, depth: int) -> None:
            if depth > max_depth or node_id in visited:
                return
            visited.add(node_id)

            for pred in self.graph.predecessors(node_id):
                edge_data = self.graph.edges[pred, node_id]
                if edge_data.get("kind") != "calls":
                    continue
                pred_data = self.graph.nodes.get(pred, {})
                if pred_data.get("type") != "symbol":
                    continue

                results.append({
                    "symbol_id": pred,
                    "name": pred_data.get("qualified_name", pred_data.get("name", "")),
                    "kind": pred_data.get("kind", ""),
                    "file_path": pred_data.get("file_path", ""),
                    "line": pred_data.get("line_start", 0),
                    "depth": depth,
                })
                _traverse(pred, depth + 1)

        for sid in symbol_ids:
            _traverse(sid, 1)

        return results

    def what_calls(self, name: str) -> list[dict]:
        """Find all symbols called by the given symbol."""
        symbol_ids = self.find_symbol(name)
        results = []

        for sid in symbol_ids:
            for succ in self.graph.successors(sid):
                edge_data = self.graph.edges[sid, succ]
                if edge_data.get("kind") != "calls":
                    continue
                succ_data = self.graph.nodes.get(succ, {})
                if succ_data.get("type") != "symbol":
                    continue
                results.append({
                    "symbol_id": succ,
                    "name": succ_data.get("qualified_name", succ_data.get("name", "")),
                    "kind": succ_data.get("kind", ""),
                    "file_path": succ_data.get("file_path", ""),
                    "line": succ_data.get("line_start", 0),
                })

        return results

    def impact_of(self, name: str, max_depth: int = 3) -> dict:
        """Analyze the impact of changing a symbol.

        Returns a dict with:
        - direct_callers: immediate callers
        - transitive_callers: all callers up to max_depth
        - affected_files: set of files that could be affected
        - risk_score: rough risk assessment (0-1)
        """
        symbol_ids = self.find_symbol(name)
        if not symbol_ids:
            return {
                "symbol": name,
                "found": False,
                "direct_callers": [],
                "transitive_callers": [],
                "affected_files": [],
                "risk_score": 0.0,
            }

        direct = self.who_calls(name, max_depth=1)
        transitive = self.who_calls(name, max_depth=max_depth)

        affected_files = set()
        for item in transitive:
            if item.get("file_path"):
                affected_files.add(item["file_path"])
        # Also include the symbol's own file
        for sid in symbol_ids:
            data = self.graph.nodes.get(sid, {})
            if data.get("file_path"):
                affected_files.add(data["file_path"])

        # Risk score based on impact breadth
        total_files = sum(1 for _, d in self.graph.nodes(data=True) if d.get("type") == "file")
        risk_score = min(len(affected_files) / max(total_files, 1), 1.0)

        return {
            "symbol": name,
            "found": True,
            "direct_callers": direct,
            "transitive_callers": transitive,
            "affected_files": sorted(affected_files),
            "risk_score": round(risk_score, 3),
        }

    def get_file_symbols(self, file_path: str) -> list[dict]:
        """Get all symbols defined in a file."""
        file_node = f"file::{file_path}"
        if not self.graph.has_node(file_node):
            return []

        symbols = []
        for succ in self.graph.successors(file_node):
            data = self.graph.nodes.get(succ, {})
            if data.get("type") == "symbol":
                symbols.append({
                    "id": succ,
                    **{k: v for k, v in data.items() if k != "type"},
                })

        return sorted(symbols, key=lambda s: s.get("line_start", 0))

    def get_structure(self, path_prefix: str = "") -> dict:
        """Get the directory/file structure with symbol counts."""
        structure: dict = {}

        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "file":
                continue
            file_path = data.get("path", "")
            if path_prefix and not file_path.startswith(path_prefix):
                continue

            parts = file_path.split("/")
            current = structure
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = {
                "_language": data.get("language", ""),
                "_symbols": data.get("symbol_count", 0),
            }

        return structure

    def find_related(self, name: str, max_hops: int = 2) -> list[dict]:
        """Find symbols related to the given one within N hops."""
        symbol_ids = self.find_symbol(name)
        if not symbol_ids:
            return []

        related = set()
        visited = set()

        def _bfs(start: str, hops: int) -> None:
            if hops > max_hops or start in visited:
                return
            visited.add(start)

            # Forward edges
            for succ in self.graph.successors(start):
                succ_data = self.graph.nodes.get(succ, {})
                if succ_data.get("type") == "symbol":
                    related.add(succ)
                    _bfs(succ, hops + 1)

            # Backward edges
            for pred in self.graph.predecessors(start):
                pred_data = self.graph.nodes.get(pred, {})
                if pred_data.get("type") == "symbol":
                    related.add(pred)
                    _bfs(pred, hops + 1)

        for sid in symbol_ids:
            _bfs(sid, 0)

        # Remove the original symbols
        related -= set(symbol_ids)

        results = []
        for node_id in related:
            data = self.graph.nodes.get(node_id, {})
            results.append({
                "symbol_id": node_id,
                "name": data.get("qualified_name", data.get("name", "")),
                "kind": data.get("kind", ""),
                "file_path": data.get("file_path", ""),
                "line": data.get("line_start", 0),
            })

        return results
