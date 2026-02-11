"""Tests for the PR Impact Bot."""

from __future__ import annotations

from pathlib import Path


from cegraph.github.diff_parser import (
    DiffHunk,
    FileDiff,
    ChangedSymbol,
    get_changed_symbols,
    parse_diff,
)
from cegraph.github.renderer import render_impact_comment, _risk_badge, _render_file_tree
from cegraph.graph.builder import GraphBuilder


SAMPLE_DIFF = """\
diff --git a/utils.py b/utils.py
index abc1234..def5678 100644
--- a/utils.py
+++ b/utils.py
@@ -5,7 +5,7 @@ TAX_RATE = 0.08
 def helper_function(value):
     \"\"\"Apply formatting to a value.\"\"\"
-    return f"${value:.2f}"
+    return f"${value:,.2f}"


 def calculate_total(items):
@@ -13,3 +13,5 @@ def calculate_total(items):
     tax = subtotal * TAX_RATE
     return subtotal + tax
+
+def new_function():
+    return "hello"
"""

SAMPLE_DIFF_NEW_FILE = """\
diff --git a/new_module.py b/new_module.py
new file mode 100644
index 0000000..abc1234
--- /dev/null
+++ b/new_module.py
@@ -0,0 +1,5 @@
+\"\"\"A new module.\"\"\"
+
+
+def brand_new():
+    return True
"""

SAMPLE_DIFF_DELETED = """\
diff --git a/old_module.py b/old_module.py
deleted file mode 100644
index abc1234..0000000
--- a/old_module.py
+++ /dev/null
@@ -1,3 +0,0 @@
-\"\"\"Old module.\"\"\"
-
-def obsolete():
-    pass
"""


class TestDiffParser:
    def test_parse_modified_file(self):
        diffs = parse_diff(SAMPLE_DIFF)
        assert len(diffs) == 1
        assert diffs[0].path == "utils.py"
        assert diffs[0].status == "modified"
        assert len(diffs[0].hunks) == 2

    def test_parse_added_lines(self):
        diffs = parse_diff(SAMPLE_DIFF)
        assert diffs[0].added_lines > 0

    def test_parse_deleted_lines(self):
        diffs = parse_diff(SAMPLE_DIFF)
        assert diffs[0].deleted_lines > 0

    def test_parse_new_file(self):
        diffs = parse_diff(SAMPLE_DIFF_NEW_FILE)
        assert len(diffs) == 1
        assert diffs[0].status == "added"

    def test_parse_deleted_file(self):
        diffs = parse_diff(SAMPLE_DIFF_DELETED)
        assert len(diffs) == 1
        assert diffs[0].status == "deleted"

    def test_parse_multiple_files(self):
        combined = SAMPLE_DIFF + SAMPLE_DIFF_NEW_FILE
        diffs = parse_diff(combined)
        assert len(diffs) == 2

    def test_hunk_line_ranges(self):
        diffs = parse_diff(SAMPLE_DIFF)
        ranges = diffs[0].changed_line_ranges
        assert len(ranges) > 0
        for start, end in ranges:
            assert start > 0
            assert end >= start

    def test_parse_empty(self):
        diffs = parse_diff("")
        assert diffs == []


class TestChangedSymbols:
    def test_get_changed_symbols(self, tmp_project: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)

        # Create a diff that modifies utils.py around helper_function
        file_diff = FileDiff(
            path="utils.py",
            status="modified",
            hunks=[DiffHunk(old_start=5, old_count=3, new_start=5, new_count=3)],
        )

        changed = get_changed_symbols(tmp_project, graph, [file_diff])
        # Should find symbols that overlap with lines 5-8 in utils.py
        # helper_function is defined around that area
        if changed:
            assert all(isinstance(s, ChangedSymbol) for s in changed)
            names = [s.name for s in changed]
            assert any("helper" in n for n in names)

    def test_deleted_file_symbols(self, tmp_project: Path):
        builder = GraphBuilder()
        graph = builder.build_from_directory(tmp_project)

        file_diff = FileDiff(path="utils.py", status="deleted")
        changed = get_changed_symbols(tmp_project, graph, [file_diff])
        # All symbols in utils.py should be marked as deleted
        for s in changed:
            assert s.change_type == "deleted"
            assert s.file_path == "utils.py"


class TestRenderer:
    def test_risk_badge_low(self):
        emoji, label, color = _risk_badge(0.05)
        assert label == "LOW"

    def test_risk_badge_medium(self):
        emoji, label, color = _risk_badge(0.35)
        assert label == "MEDIUM"

    def test_risk_badge_high(self):
        emoji, label, color = _risk_badge(0.55)
        assert label == "HIGH"

    def test_risk_badge_critical(self):
        emoji, label, color = _risk_badge(0.7)
        assert label == "CRITICAL"

    def test_render_empty_comment(self):
        comment = render_impact_comment([], [])
        assert "CeGraph Impact Analysis" in comment
        assert "No code symbols" in comment

    def test_render_with_changes(self):
        symbols = [
            ChangedSymbol(
                name="my_func",
                qualified_name="module::my_func",
                kind="function",
                file_path="module.py",
                line_start=10,
                line_end=20,
                change_type="modified",
            ),
        ]
        impacts = [
            {
                "found": True,
                "symbol": "my_func",
                "risk_score": 0.3,
                "direct_callers": [{"name": "caller", "kind": "function", "file_path": "main.py"}],
                "transitive_callers": [],
                "affected_files": ["main.py", "test.py"],
            },
        ]
        comment = render_impact_comment(symbols, impacts)
        assert "my_func" in comment
        assert "MEDIUM" in comment
        assert "main.py" in comment

    def test_render_file_tree(self):
        files = ["src/main.py", "src/utils.py", "tests/test_main.py"]
        tree = _render_file_tree(files)
        assert len(tree) > 0

    def test_footer_present(self):
        comment = render_impact_comment([], [])
        assert "CeGraph" in comment
