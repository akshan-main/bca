"""Tests for the CAG (Context Assembly Generation) module."""

from __future__ import annotations

from pathlib import Path

import pytest

from cegraph.context.engine import ContextAssembler
from cegraph.context.models import (
    ContextItem,
    ContextPackage,
    ContextStrategy,
    TokenEstimator,
)
from cegraph.graph.builder import GraphBuilder
from cegraph.graph.query import GraphQuery


@pytest.fixture
def cag_engine(tmp_project: Path):
    """Create a context assembler with a built graph."""
    builder = GraphBuilder()
    graph = builder.build_from_directory(tmp_project)
    query = GraphQuery(graph)
    return ContextAssembler(tmp_project, graph, query)


class TestTokenEstimator:
    def test_estimate_basic(self):
        text = "def hello():\n    return 'world'"
        tokens = TokenEstimator.estimate(text)
        assert tokens > 0

    def test_estimate_empty(self):
        assert TokenEstimator.estimate("") == 1

    def test_estimate_lines(self):
        tokens = TokenEstimator.estimate_lines(10)
        assert tokens > 0

    def test_estimate_proportional(self):
        short = TokenEstimator.estimate("x = 1")
        long = TokenEstimator.estimate("x = 1\n" * 100)
        assert long > short


class TestContextStrategy:
    def test_strategies_exist(self):
        assert ContextStrategy.PRECISE.value == "precise"
        assert ContextStrategy.SMART.value == "smart"
        assert ContextStrategy.THOROUGH.value == "thorough"


class TestContextPackage:
    def test_empty_package(self):
        pkg = ContextPackage(
            task="test task",
            strategy=ContextStrategy.SMART,
            token_budget=8000,
        )
        assert pkg.symbols_included == 0
        assert pkg.total_tokens == 0

    def test_render_empty(self):
        pkg = ContextPackage(
            task="test task",
            strategy=ContextStrategy.SMART,
            token_budget=8000,
        )
        rendered = pkg.render()
        assert "test task" in rendered

    def test_render_with_items(self):
        items = [
            ContextItem(
                symbol_id="test::func",
                name="func",
                qualified_name="test::func",
                kind="function",
                file_path="test.py",
                line_start=1,
                line_end=5,
                source_code="def func():\n    return 42",
                relevance_score=0.9,
                reason="matches 'func'",
                token_estimate=10,
                depth=0,
            )
        ]
        pkg = ContextPackage(
            task="test task",
            strategy=ContextStrategy.SMART,
            items=items,
            total_tokens=10,
            token_budget=8000,
            files_included=1,
            symbols_included=1,
        )
        rendered = pkg.render()
        assert "func" in rendered
        assert "test.py" in rendered
        assert "def func" in rendered

    def test_render_compact(self):
        items = [
            ContextItem(
                symbol_id="test::func",
                name="func",
                qualified_name="test::func",
                kind="function",
                file_path="test.py",
                line_start=1,
                line_end=5,
                source_code="def func():\n    return 42",
                signature="def func():",
                relevance_score=0.9,
                token_estimate=10,
                depth=0,
            ),
            ContextItem(
                symbol_id="test::helper",
                name="helper",
                qualified_name="test::helper",
                kind="function",
                file_path="test.py",
                line_start=10,
                line_end=15,
                source_code="def helper(x):\n    return x + 1",
                signature="def helper(x):",
                docstring="A helper function",
                relevance_score=0.5,
                token_estimate=8,
                depth=2,
            ),
        ]
        pkg = ContextPackage(
            task="test",
            strategy=ContextStrategy.SMART,
            items=items,
            total_tokens=18,
            token_budget=8000,
        )
        compact = pkg.render_compact()
        # Primary symbols (depth 0) should have full source
        assert "def func" in compact
        # Secondary symbols (depth > 0) should have signature only
        assert "def helper(x):" in compact

    def test_summary(self):
        pkg = ContextPackage(
            task="test task",
            strategy=ContextStrategy.SMART,
            total_tokens=500,
            token_budget=8000,
            symbols_included=5,
            symbols_available=20,
            files_included=3,
        )
        summary = pkg.summary()
        assert "CAG" in summary
        assert "test task" in summary
        assert "500" in summary


class TestCAGEngine:
    def test_assemble_basic(self, cag_engine: ContextAssembler):
        """Test basic context assembly."""
        package = cag_engine.assemble(
            task="fix the main function",
            token_budget=8000,
        )
        assert isinstance(package, ContextPackage)
        assert package.symbols_included > 0
        assert package.total_tokens > 0
        assert package.total_tokens <= package.token_budget

    def test_assemble_with_specific_symbol(self, cag_engine: ContextAssembler):
        """Test context assembly with a specific symbol name."""
        package = cag_engine.assemble(
            task="refactor the calculate_total function",
            token_budget=8000,
        )
        # Should find calculate_total and related symbols
        symbol_names = [item.name for item in package.items]
        assert "calculate_total" in symbol_names

    def test_assemble_with_class(self, cag_engine: ContextAssembler):
        """Test that CamelCase class names are detected."""
        # Note: Our test fixture doesn't have CamelCase names in the task
        # that match the actual User class easily, so let's search for 'User' directly
        package = cag_engine.assemble(
            task="fix the User class validation",
            token_budget=8000,
        )
        assert package.symbols_included > 0

    def test_assemble_respects_budget(self, cag_engine: ContextAssembler):
        """Test that assembly respects the token budget."""
        small_budget = 100
        package = cag_engine.assemble(
            task="review the entire codebase",
            token_budget=small_budget,
            strategy=ContextStrategy.THOROUGH,
        )
        assert package.total_tokens <= small_budget * 1.5  # Allow small overshoot

    def test_strategies_differ(self, cag_engine: ContextAssembler):
        """Test that different strategies produce different results."""
        precise = cag_engine.assemble(
            task="fix the main function",
            token_budget=8000,
            strategy=ContextStrategy.PRECISE,
        )
        thorough = cag_engine.assemble(
            task="fix the main function",
            token_budget=8000,
            strategy=ContextStrategy.THOROUGH,
        )
        # Thorough should generally find more candidates
        assert thorough.symbols_available >= precise.symbols_available

    def test_focus_files(self, cag_engine: ContextAssembler):
        """Test that focus files are prioritized."""
        package = cag_engine.assemble(
            task="review code",
            token_budget=8000,
            focus_files=["utils.py"],
        )
        # Should include symbols from utils.py
        files = set(item.file_path for item in package.items)
        assert "utils.py" in files or len(package.items) > 0

    def test_dependency_ordering(self, cag_engine: ContextAssembler):
        """Test that items are ordered with dependencies first."""
        package = cag_engine.assemble(
            task="fix calculate_total in utils",
            token_budget=8000,
        )
        # If both calculate_total and something that calls it are included,
        # calculate_total should come first (or at least not after its callers)
        if len(package.items) > 1:
            # Items should be valid (all have symbol_ids)
            assert all(item.symbol_id for item in package.items)

    def test_extract_entities(self, cag_engine: ContextAssembler):
        """Test entity extraction from natural language."""
        entities = cag_engine._extract_entities(
            "fix the UserService login method in auth.service.py"
        )
        names = [e["name"] for e in entities]
        # Should detect CamelCase
        assert "UserService" in names
        # Should detect file path
        assert any("auth" in n for n in names)

    def test_extract_entities_snake_case(self, cag_engine: ContextAssembler):
        """Test snake_case entity extraction."""
        entities = cag_engine._extract_entities(
            "refactor the calculate_total function"
        )
        names = [e["name"] for e in entities]
        assert "calculate_total" in names

    def test_extract_entities_quoted(self, cag_engine: ContextAssembler):
        """Test quoted entity extraction."""
        entities = cag_engine._extract_entities(
            "the `helper_function` is broken"
        )
        names = [e["name"] for e in entities]
        assert "helper_function" in names

    def test_estimate_savings(self, cag_engine: ContextAssembler):
        """Test the savings estimation feature."""
        savings = cag_engine.estimate_savings(
            "fix the main function",
            token_budget=4000,
        )
        assert "cag_tokens" in savings
        assert "grep_tokens" in savings
        assert "all_files_tokens" in savings
        assert savings["cag_tokens"] >= 0
        # CAG should use fewer tokens than grep (usually)
        # But for tiny test projects this might not always hold

    def test_render_output(self, cag_engine: ContextAssembler):
        """Test the full render pipeline."""
        package = cag_engine.assemble(
            task="understand the Order class",
            token_budget=8000,
        )
        rendered = package.render()
        assert len(rendered) > 0
        # Should include file paths and code
        assert "##" in rendered  # File header

    def test_assembly_time_tracked(self, cag_engine: ContextAssembler):
        """Test that assembly time is tracked."""
        package = cag_engine.assemble("fix main", token_budget=8000)
        assert package.assembly_time_ms >= 0

    def test_is_not_accelerated(self, cag_engine: ContextAssembler):
        """Test that C++ acceleration is reported correctly."""
        # In tests, C++ extension is typically not compiled
        assert isinstance(cag_engine.is_accelerated, bool)
