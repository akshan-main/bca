"""Tests for the knowledge graph module."""

from __future__ import annotations

from pathlib import Path


from cegraph.graph.builder import GraphBuilder
from cegraph.graph.query import GraphQuery
from cegraph.graph.store import GraphStore


class TestGraphBuilder:
    def test_build_from_directory(self, tmp_project: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_stats(self, tmp_project: Path):
        builder = GraphBuilder()
        builder.build_from_directory(tmp_project)
        stats = builder.get_stats()

        assert stats["files"] > 0
        assert stats["functions"] > 0
        assert stats["classes"] > 0
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0

    def test_file_nodes_present(self, tmp_project: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)

        file_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") == "file"
        ]
        file_paths = [graph.nodes[n].get("path") for n in file_nodes]
        assert "main.py" in file_paths
        assert "utils.py" in file_paths
        assert "models.py" in file_paths

    def test_symbol_nodes_have_attributes(self, tmp_project: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)

        for node_id, data in graph.nodes(data=True):
            if data.get("type") == "symbol":
                assert "name" in data
                assert "kind" in data
                assert "file_path" in data
                assert "line_start" in data

    def test_call_edges_exist(self, tmp_project: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)

        call_edges = [
            (u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "calls"
        ]
        assert len(call_edges) > 0

    def test_contains_edges_exist(self, tmp_project: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)

        contains_edges = [
            (u, v) for u, v, d in graph.edges(data=True) if d.get("kind") == "contains"
        ]
        assert len(contains_edges) > 0


class TestGraphBuilderReuse:
    def test_rebuild_resets_state(self, tmp_path: Path):
        """Regression: reusing a builder must not accumulate stale nodes."""
        # First project
        p1 = tmp_path / "proj1"
        p1.mkdir()
        (p1 / "a.py").write_text("def a_func():\n    pass\n")

        builder = GraphBuilder()
        g1 = builder.build_from_directory(p1)
        _ = {d["path"] for _, d in g1.nodes(data=True) if d.get("type") == "file"}

        # Second project with different files
        p2 = tmp_path / "proj2"
        p2.mkdir()
        (p2 / "b.py").write_text("def b_func():\n    pass\n")

        g2 = builder.build_from_directory(p2)
        g2_files = {d["path"] for _, d in g2.nodes(data=True) if d.get("type") == "file"}

        # g2 should NOT contain files from g1
        assert "a.py" not in g2_files
        assert "b.py" in g2_files


class TestGraphStore:
    def test_save_and_load(self, tmp_project: Path, tmp_path: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)

        db_path = tmp_path / "test.db"
        store = GraphStore(db_path)
        store.save(graph, metadata={"test": True})

        loaded = store.load()
        assert loaded is not None
        assert loaded.number_of_nodes() == graph.number_of_nodes()
        assert loaded.number_of_edges() == graph.number_of_edges()
        store.close()

    def test_search_symbols(self, tmp_project: Path, tmp_path: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)

        db_path = tmp_path / "test.db"
        store = GraphStore(db_path)
        store.save(graph)

        results = store.search_symbols(query="main")
        assert len(results) > 0
        assert any(r["name"] == "main" for r in results)
        store.close()

    def test_search_by_kind(self, tmp_project: Path, tmp_path: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)

        db_path = tmp_path / "test.db"
        store = GraphStore(db_path)
        store.save(graph)

        results = store.search_symbols(kind="class")
        assert len(results) > 0
        assert all(r["kind"] == "class" for r in results)
        store.close()

    def test_metadata(self, tmp_project: Path, tmp_path: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)

        db_path = tmp_path / "test.db"
        store = GraphStore(db_path)
        store.save(graph, metadata={"version": "1.0"})

        assert store.get_metadata("version") == "1.0"
        store.close()


class TestGraphQuery:
    def _build_query(self, tmp_project: Path) -> GraphQuery:
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)
        return GraphQuery(graph)

    def test_find_symbol(self, tmp_project: Path):
        query = self._build_query(tmp_project)
        results = query.find_symbol("main")
        assert len(results) > 0

    def test_find_symbol_partial(self, tmp_project: Path):
        query = self._build_query(tmp_project)
        results = query.find_symbol("helper")
        assert len(results) > 0

    def test_who_calls(self, tmp_project: Path):
        query = self._build_query(tmp_project)
        callers = query.who_calls("helper_function")
        # main() calls helper_function()
        caller_names = [c["name"] for c in callers]
        assert any("main" in name for name in caller_names)

    def test_what_calls(self, tmp_project: Path):
        query = self._build_query(tmp_project)
        callees = query.what_calls("main")
        # main() should call several functions
        assert len(callees) > 0

    def test_impact_of(self, tmp_project: Path):
        query = self._build_query(tmp_project)
        impact = query.impact_of("calculate_total")
        assert impact["found"] is True
        assert len(impact["affected_files"]) > 0
        assert impact["risk_score"] >= 0

    def test_impact_not_found(self, tmp_project: Path):
        query = self._build_query(tmp_project)
        impact = query.impact_of("nonexistent_function")
        assert impact["found"] is False

    def test_get_file_symbols(self, tmp_project: Path):
        query = self._build_query(tmp_project)
        symbols = query.get_file_symbols("main.py")
        assert len(symbols) > 0
        names = [s["name"] for s in symbols]
        assert "main" in names

    def test_get_structure(self, tmp_project: Path):
        query = self._build_query(tmp_project)
        structure = query.get_structure()
        assert "main.py" in structure or len(structure) > 0

    def test_find_related(self, tmp_project: Path):
        query = self._build_query(tmp_project)
        related = query.find_related("calculate_total")
        assert len(related) > 0

    def test_get_symbol_info(self, tmp_project: Path):
        query = self._build_query(tmp_project)
        symbol_ids = query.find_symbol("User")
        assert len(symbol_ids) > 0
        # Find the class definition (not import)
        class_info = None
        for sid in symbol_ids:
            info = query.get_symbol_info(sid)
            if info and info.kind == "class":
                class_info = info
                break
        assert class_info is not None
        assert class_info.name == "User"
        assert class_info.kind == "class"
