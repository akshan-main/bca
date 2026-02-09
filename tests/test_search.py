"""Tests for the search module."""

from __future__ import annotations

from pathlib import Path

import pytest

from cegraph.graph.builder import GraphBuilder
from cegraph.search.lexical import LexicalSearch
from cegraph.search.hybrid import HybridSearch


class TestLexicalSearch:
    def _build_search(self, tmp_project: Path) -> LexicalSearch:
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)
        return LexicalSearch(tmp_project, graph)

    def test_search_basic(self, tmp_project: Path):
        search = self._build_search(tmp_project)
        results = search.search("calculate_total")
        assert len(results) > 0
        assert any("calculate_total" in r.line_content for r in results)

    def test_search_with_file_pattern(self, tmp_project: Path):
        search = self._build_search(tmp_project)
        results = search.search("def ", file_pattern="*.py")
        assert len(results) > 0
        assert all(r.file_path.endswith(".py") for r in results)

    def test_search_context_lines(self, tmp_project: Path):
        search = self._build_search(tmp_project)
        results = search.search("calculate_total", context_lines=3)
        if results:
            # Should have context
            assert len(results[0].context_before) > 0 or len(results[0].context_after) > 0

    def test_search_regex(self, tmp_project: Path):
        search = self._build_search(tmp_project)
        results = search.search(r"def \w+\(self", regex=True)
        assert len(results) > 0

    def test_search_no_results(self, tmp_project: Path):
        search = self._build_search(tmp_project)
        results = search.search("xyznonexistent123")
        assert len(results) == 0

    def test_search_symbols(self, tmp_project: Path):
        search = self._build_search(tmp_project)
        results = search.search_symbols("User")
        assert len(results) > 0
        assert any(r["name"] == "User" for r in results)

    def test_search_symbols_by_kind(self, tmp_project: Path):
        search = self._build_search(tmp_project)
        results = search.search_symbols("", kind="function")
        assert len(results) > 0
        assert all(r["kind"] == "function" for r in results)

    def test_symbol_enrichment(self, tmp_project: Path):
        search = self._build_search(tmp_project)
        results = search.search("TAX_RATE")
        # Should find the constant and know it's in utils.py
        assert len(results) > 0
        assert any("utils" in r.file_path for r in results)


class TestHybridSearch:
    def test_search_delegates_to_lexical(self, tmp_project: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)
        search = HybridSearch(tmp_project, graph)

        results = search.search("calculate_total")
        assert len(results) > 0

    def test_search_symbols(self, tmp_project: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)
        search = HybridSearch(tmp_project, graph)

        results = search.search_symbols("Order")
        assert len(results) > 0
