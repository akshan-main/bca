"""Build a knowledge graph from parsed code symbols."""

from __future__ import annotations

import hashlib
from pathlib import Path

import networkx as nx

from cegraph.config import IndexerConfig, ProjectConfig
from cegraph.parser.core import collect_files, parse_directory, parse_files
from cegraph.parser.models import FileSymbols, Relationship


class GraphBuilder:
    """Builds and maintains a code knowledge graph.

    The graph has two types of nodes:
    - File nodes: represent source files
    - Symbol nodes: represent code symbols (functions, classes, etc.)

    Edges represent relationships (calls, imports, inherits, contains, etc.)
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._file_hashes: dict[str, str] = {}
        self._unresolved: list[Relationship] = []

    def build_from_directory(
        self,
        root: str | Path,
        config: ProjectConfig | None = None,
        progress_callback: callable | None = None,
    ) -> nx.DiGraph:
        """Build the full knowledge graph from a directory.

        Args:
            root: Root directory to index.
            config: Project configuration.
            progress_callback: Optional callback(file_path, current, total).

        Returns:
            The constructed NetworkX directed graph.
        """
        root = Path(root).resolve()
        indexer_config = config.indexer if config else IndexerConfig()

        # Reset state so reusing a builder doesn't accumulate stale data
        self.graph = nx.DiGraph()
        self._file_hashes = {}
        self._unresolved = []

        # Parse all files
        all_parsed = parse_directory(root, indexer_config, progress_callback)

        # Build graph from parsed results
        for file_symbols in all_parsed:
            self._add_file(file_symbols, root)

        # Resolve cross-file references
        self._resolve_references()

        return self.graph

    def incremental_build(
        self,
        root: str | Path,
        graph: nx.DiGraph,
        old_hashes: dict[str, str],
        config: ProjectConfig | None = None,
        progress_callback: callable | None = None,
    ) -> tuple[nx.DiGraph, list[str]]:
        """Rebuild only changed files, reusing the existing graph.

        Args:
            root: Project root directory.
            graph: Previously built graph to update.
            old_hashes: File path -> SHA256 hash from previous build.
            config: Project configuration.
            progress_callback: Optional callback.

        Returns:
            (updated_graph, list_of_changed_file_paths)
        """
        root = Path(root).resolve()
        indexer_config = config.indexer if config else IndexerConfig()

        self.graph = graph
        self._file_hashes = dict(old_hashes)
        self._unresolved = []

        # Discover current files and compute hashes
        current_files = collect_files(root, indexer_config)
        current_rel = {}
        for f in current_files:
            rel = str(f.relative_to(root))
            try:
                content = f.read_bytes()
                current_rel[rel] = hashlib.sha256(content).hexdigest()[:16]
            except OSError:
                continue

        # Diff: find added, changed, deleted
        old_set = set(old_hashes.keys())
        new_set = set(current_rel.keys())
        deleted = old_set - new_set
        added = new_set - old_set
        changed = {
            f for f in old_set & new_set
            if old_hashes[f] != current_rel[f]
        }
        dirty = added | changed

        if not dirty and not deleted:
            return self.graph, []

        # Remove nodes/edges for deleted and changed files
        for fp in deleted | changed:
            self._remove_file_from_graph(fp)
            self._file_hashes.pop(fp, None)

        # Parse only dirty files
        if dirty:
            parsed = parse_files(root, sorted(dirty), indexer_config,
                                 progress_callback)
            for fs in parsed:
                self._add_file(fs, root)

        # Update hashes
        for fp in dirty:
            if fp in current_rel:
                self._file_hashes[fp] = current_rel[fp]

        # Re-resolve all unresolved references
        self._resolve_references()

        return self.graph, sorted(deleted | dirty)

    def _remove_file_from_graph(self, file_path: str) -> None:
        """Remove a file and all its symbols from the in-memory graph."""
        file_node = f"file::{file_path}"
        if not self.graph.has_node(file_node):
            return
        # Collect symbol nodes belonging to this file
        to_remove = [file_node]
        for succ in list(self.graph.successors(file_node)):
            data = self.graph.nodes.get(succ, {})
            if data.get("type") == "symbol":
                to_remove.append(succ)
        for node in to_remove:
            self.graph.remove_node(node)  # also removes all edges

    def _add_file(self, fs: FileSymbols, root: Path) -> None:
        """Add a file and its symbols to the graph."""
        file_path = fs.file_path

        # Add file node
        self.graph.add_node(
            f"file::{file_path}",
            type="file",
            path=file_path,
            language=fs.language,
            symbol_count=len(fs.symbols),
            import_count=len(fs.imports),
        )

        # Compute file hash for change detection
        try:
            full_path = root / file_path
            content = full_path.read_bytes()
            self._file_hashes[file_path] = hashlib.sha256(content).hexdigest()[:16]
        except OSError:
            pass

        # Add symbol nodes
        for symbol in fs.symbols:
            attrs = {
                "type": "symbol",
                "name": symbol.name,
                "qualified_name": symbol.qualified_name,
                "kind": symbol.kind.value,
                "file_path": symbol.file_path,
                "line_start": symbol.line_start,
                "line_end": symbol.line_end,
                "signature": symbol.signature,
                "docstring": symbol.docstring,
            }
            self.graph.add_node(symbol.id, **attrs)

            # Link symbol to its file
            self.graph.add_edge(
                f"file::{file_path}",
                symbol.id,
                kind="contains",
            )

        # Add relationships
        for rel in fs.relationships:
            if rel.resolved or self._try_resolve(rel):
                self.graph.add_edge(
                    rel.source,
                    rel.target,
                    kind=rel.kind.value,
                    file_path=rel.file_path,
                    line=rel.line,
                )
            else:
                self._unresolved.append(rel)

    def _try_resolve(self, rel: Relationship) -> bool:
        """Try to resolve a relationship's target to an existing node."""
        target = rel.target

        # Direct match
        if self.graph.has_node(target):
            return True

        # Try finding by name across all files
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "symbol":
                continue
            if data.get("name") == target or data.get("qualified_name") == target:
                rel.target = node_id
                rel.resolved = True
                return True

        return False

    def _resolve_references(self) -> None:
        """Try to resolve all unresolved references after the full graph is built."""
        still_unresolved = []

        # Build a lookup index: name -> [node_ids]
        name_index: dict[str, list[str]] = {}
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "symbol":
                continue
            name = data.get("name", "")
            qname = data.get("qualified_name", "")
            if name:
                name_index.setdefault(name, []).append(node_id)
            if qname and qname != name:
                name_index.setdefault(qname, []).append(node_id)

        for rel in self._unresolved:
            target = rel.target
            # Try exact match by name
            candidates = name_index.get(target, [])

            # Try dotted parts (e.g., "module.func" -> "func")
            if not candidates and "." in target:
                parts = target.split(".")
                candidates = name_index.get(parts[-1], [])

            if candidates:
                # Pick the best candidate (same file first, then any)
                best = None
                for c in candidates:
                    c_data = self.graph.nodes[c]
                    if c_data.get("file_path") == rel.file_path:
                        best = c
                        break
                if best is None:
                    best = candidates[0]

                self.graph.add_edge(
                    rel.source,
                    best,
                    kind=rel.kind.value,
                    file_path=rel.file_path,
                    line=rel.line,
                )
            else:
                still_unresolved.append(rel)

        self._unresolved = still_unresolved

    def get_file_mtimes(self, root: Path) -> dict[str, float]:
        """Collect current mtimes for all tracked files."""
        mtimes = {}
        for rel_path in self._file_hashes:
            try:
                mtimes[rel_path] = (root / rel_path).stat().st_mtime
            except OSError:
                pass
        return mtimes

    def get_dir_mtimes(self, root: Path) -> dict[str, float]:
        """Collect mtimes for directories containing tracked files."""
        dirs: set[str] = set()
        for fp in self._file_hashes:
            p = Path(fp).parent
            while str(p) != ".":
                dirs.add(str(p))
                p = p.parent
            dirs.add(".")
        mtimes = {}
        for d in dirs:
            try:
                mtimes[d] = (root / d).stat().st_mtime
            except OSError:
                pass
        return mtimes

    def get_stats(self) -> dict:
        """Get graph statistics."""
        node_types: dict[str, int] = {}
        edge_types: dict[str, int] = {}

        for _, data in self.graph.nodes(data=True):
            kind = data.get("kind", data.get("type", "unknown"))
            node_types[kind] = node_types.get(kind, 0) + 1

        for _, _, data in self.graph.edges(data=True):
            kind = data.get("kind", "unknown")
            edge_types[kind] = edge_types.get(kind, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "files": node_types.get("file", 0),
            "functions": node_types.get("function", 0) + node_types.get("method", 0),
            "classes": node_types.get("class", 0),
            "unresolved_refs": len(self._unresolved),
        }
