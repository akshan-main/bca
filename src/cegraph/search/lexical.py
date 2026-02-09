"""Lexical (text-based) code search with ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx


@dataclass
class SearchResult:
    """A single search result."""

    file_path: str
    line_number: int
    line_content: str
    score: float
    context_before: list[str] = field(default_factory=list)
    context_after: list[str] = field(default_factory=list)
    symbol_name: str = ""
    symbol_kind: str = ""


class LexicalSearch:
    """Text-based code search with TF-IDF-like ranking.

    Searches through actual source files with context lines,
    plus leverages the knowledge graph for symbol-aware results.
    """

    def __init__(self, root: Path, graph: nx.DiGraph | None = None) -> None:
        self.root = root
        self.graph = graph

    def search(
        self,
        query: str,
        file_pattern: str = "",
        max_results: int = 20,
        context_lines: int = 2,
        regex: bool = False,
    ) -> list[SearchResult]:
        """Search for code matching the query.

        Args:
            query: Search query (text or regex pattern).
            file_pattern: Glob pattern to filter files (e.g., "*.py").
            max_results: Maximum results to return.
            context_lines: Number of context lines before/after match.
            regex: If True, treat query as regex pattern.

        Returns:
            Ranked list of search results.
        """
        results: list[SearchResult] = []

        if regex:
            try:
                pattern = re.compile(query, re.IGNORECASE)
            except re.error:
                return []
        else:
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        # Get files from the graph or scan directory
        files = self._get_searchable_files(file_pattern)

        for file_path in files:
            full_path = self.root / file_path
            if not full_path.exists():
                continue

            try:
                lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            for i, line in enumerate(lines):
                if pattern.search(line):
                    # Get context
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)

                    result = SearchResult(
                        file_path=file_path,
                        line_number=i + 1,
                        line_content=line,
                        score=self._score_match(query, line, file_path),
                        context_before=lines[start:i],
                        context_after=lines[i + 1 : end],
                    )

                    # Enrich with symbol info from graph
                    if self.graph:
                        self._enrich_with_symbol(result, file_path, i + 1)

                    results.append(result)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_results]

    def search_symbols(
        self, query: str, kind: str = "", max_results: int = 20
    ) -> list[dict]:
        """Search through symbol definitions in the knowledge graph."""
        if not self.graph:
            return []

        results = []
        query_lower = query.lower()

        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "symbol":
                continue

            name = data.get("name", "")
            qname = data.get("qualified_name", "")
            sig = data.get("signature", "")
            doc = data.get("docstring", "")
            sym_kind = data.get("kind", "")

            if kind and sym_kind != kind:
                continue

            # Score based on match quality
            score = 0.0
            if query_lower == name.lower():
                score = 1.0
            elif query_lower in name.lower():
                score = 0.8
            elif query_lower in qname.lower():
                score = 0.6
            elif query_lower in sig.lower():
                score = 0.4
            elif query_lower in doc.lower():
                score = 0.3
            else:
                continue

            results.append({
                "id": node_id,
                "name": name,
                "qualified_name": qname,
                "kind": sym_kind,
                "file_path": data.get("file_path", ""),
                "line": data.get("line_start", 0),
                "signature": sig,
                "score": score,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:max_results]

    def _get_searchable_files(self, file_pattern: str) -> list[str]:
        """Get list of files to search."""
        files = []
        if self.graph:
            for node_id, data in self.graph.nodes(data=True):
                if data.get("type") == "file":
                    path = data.get("path", "")
                    if file_pattern:
                        from fnmatch import fnmatch

                        if not fnmatch(path, file_pattern):
                            continue
                    files.append(path)
        else:
            # Scan directory
            import os

            for dirpath, _, filenames in os.walk(self.root):
                for filename in filenames:
                    rel_path = str(Path(dirpath, filename).relative_to(self.root))
                    if file_pattern:
                        from fnmatch import fnmatch

                        if not fnmatch(rel_path, file_pattern):
                            continue
                    files.append(rel_path)
        return sorted(files)

    def _score_match(self, query: str, line: str, file_path: str) -> float:
        """Score a match based on quality heuristics."""
        score = 1.0

        # Exact case match bonus
        if query in line:
            score += 0.5

        # Definition bonus (line starts with def, class, function, etc.)
        stripped = line.strip()
        if any(
            stripped.startswith(kw)
            for kw in (
                "def ", "class ", "function ", "const ",
                "let ", "var ", "fn ", "func ", "pub ",
            )
        ):
            score += 1.0

        # Shorter lines are usually more relevant
        if len(stripped) < 80:
            score += 0.3

        # Test files penalty
        if "test" in file_path.lower():
            score -= 0.3

        return score

    def _enrich_with_symbol(
        self, result: SearchResult, file_path: str, line_num: int
    ) -> None:
        """Try to find the enclosing symbol for a search result."""
        if not self.graph:
            return

        best_symbol = None
        best_range = float("inf")

        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "symbol":
                continue
            if data.get("file_path") != file_path:
                continue

            start = data.get("line_start", 0)
            end = data.get("line_end", 0)
            if start <= line_num <= end:
                range_size = end - start
                if range_size < best_range:
                    best_range = range_size
                    best_symbol = data

        if best_symbol:
            result.symbol_name = best_symbol.get("qualified_name", best_symbol.get("name", ""))
            result.symbol_kind = best_symbol.get("kind", "")
