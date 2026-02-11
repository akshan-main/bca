"""Tests for the search module."""

from __future__ import annotations

from pathlib import Path


from cegraph.graph.builder import GraphBuilder
from cegraph.search.classifier import QueryClassifier
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


class TestQueryClassifier:
    def _build_classifier(self, tmp_project: Path) -> QueryClassifier:
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)
        return QueryClassifier(graph)

    def test_exact_query(self, tmp_project: Path):
        """Queries with specific identifiers should classify as exact."""
        classifier = self._build_classifier(tmp_project)
        result = classifier.classify("Fix the bug in calculate_total function")
        assert result.query_type == "exact"
        assert result.recommended_method == "bca"

    def test_vague_query(self, tmp_project: Path):
        """Natural language queries without identifiers should classify as vague."""
        classifier = self._build_classifier(tmp_project)
        result = classifier.classify("fix the login flow and improve error handling")
        assert result.query_type == "vague"
        assert result.recommended_method == "bm25"

    def test_structural_query(self, tmp_project: Path):
        """Queries about call relationships should classify as structural."""
        classifier = self._build_classifier(tmp_project)
        result = classifier.classify("who calls helper_function")
        assert result.query_type == "structural"
        assert result.recommended_method == "graph"

    def test_structural_impact(self, tmp_project: Path):
        """Impact queries should classify as structural."""
        classifier = self._build_classifier(tmp_project)
        result = classifier.classify("what is the impact of changing calculate_total")
        assert result.query_type == "structural"

    def test_classification_has_features(self, tmp_project: Path):
        """Classification should include debug features."""
        classifier = self._build_classifier(tmp_project)
        result = classifier.classify("Fix the User class")
        assert "entity_density" in result.features
        assert "symbol_hit_ratio" in result.features
        assert "idf_weighted_score" in result.features
        assert "structural_score" in result.features

    def test_confidence_range(self, tmp_project: Path):
        """Confidence should be in [0, 1]."""
        classifier = self._build_classifier(tmp_project)
        for query in [
            "fix calculate_total",
            "improve the app",
            "who calls main",
        ]:
            result = classifier.classify(query)
            assert 0.0 <= result.confidence <= 1.0


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
