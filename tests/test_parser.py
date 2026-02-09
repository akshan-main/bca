"""Tests for the code parser module."""

from __future__ import annotations

import pytest

from cegraph.parser.models import SymbolKind, RelKind, detect_language
from cegraph.parser.python_parser import parse_python_file
from cegraph.parser.core import parse_file


class TestLanguageDetection:
    def test_python(self):
        assert detect_language("main.py") == "python"
        assert detect_language("types.pyi") == "python"

    def test_javascript(self):
        assert detect_language("app.js") == "javascript"
        assert detect_language("component.jsx") == "javascript"

    def test_typescript(self):
        assert detect_language("app.ts") == "typescript"
        assert detect_language("component.tsx") == "typescript"

    def test_go(self):
        assert detect_language("main.go") == "go"

    def test_rust(self):
        assert detect_language("main.rs") == "rust"

    def test_unknown(self):
        assert detect_language("readme.md") is None
        assert detect_language("data.json") is None
        # Languages without tree-sitter grammars are not supported
        assert detect_language("main.rb") is None
        assert detect_language("main.php") is None


class TestPythonParser:
    def test_parse_functions(self, sample_python_source: str):
        result = parse_python_file("sample.py", sample_python_source)
        assert result.language == "python"
        assert len(result.errors) == 0

        # Check functions are found
        func_names = [
            s.name for s in result.symbols if s.kind == SymbolKind.FUNCTION
        ]
        assert "create_processor" in func_names
        assert "run_pipeline" in func_names

    def test_parse_classes(self, sample_python_source: str):
        result = parse_python_file("sample.py", sample_python_source)

        class_names = [
            s.name for s in result.symbols if s.kind == SymbolKind.CLASS
        ]
        assert "BaseProcessor" in class_names
        assert "AdvancedProcessor" in class_names

    def test_parse_methods(self, sample_python_source: str):
        result = parse_python_file("sample.py", sample_python_source)

        methods = [s for s in result.symbols if s.kind == SymbolKind.METHOD]
        method_names = [m.name for m in methods]
        assert "__init__" in method_names
        assert "process" in method_names
        assert "_transform" in method_names
        assert "batch_process" in method_names

    def test_parse_imports(self, sample_python_source: str):
        result = parse_python_file("sample.py", sample_python_source)
        assert "os" in result.imports
        assert "typing.List" in result.imports
        assert "pathlib.Path" in result.imports

    def test_parse_constants(self, sample_python_source: str):
        result = parse_python_file("sample.py", sample_python_source)
        constants = [
            s for s in result.symbols if s.kind == SymbolKind.CONSTANT
        ]
        assert any(c.name == "CONSTANT_VALUE" for c in constants)

    def test_parse_inheritance(self, sample_python_source: str):
        result = parse_python_file("sample.py", sample_python_source)
        inherits = [
            r for r in result.relationships if r.kind == RelKind.INHERITS
        ]
        assert any("AdvancedProcessor" in r.source and "BaseProcessor" in r.target for r in inherits)

    def test_parse_calls(self, sample_python_source: str):
        result = parse_python_file("sample.py", sample_python_source)
        calls = [r for r in result.relationships if r.kind == RelKind.CALLS]
        # run_pipeline calls create_processor
        assert any(
            "run_pipeline" in r.source and "create_processor" in r.target
            for r in calls
        )

    def test_parse_docstrings(self, sample_python_source: str):
        result = parse_python_file("sample.py", sample_python_source)
        funcs = {s.name: s for s in result.symbols if s.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD)}
        assert "Factory function" in funcs["create_processor"].docstring
        assert "Transform a single item" in funcs["_transform"].docstring

    def test_parse_contains_relationships(self, sample_python_source: str):
        result = parse_python_file("sample.py", sample_python_source)
        contains = [r for r in result.relationships if r.kind == RelKind.CONTAINS]
        # BaseProcessor should contain process, _transform, __init__
        assert any("BaseProcessor" in r.source and "process" in r.target for r in contains)

    def test_syntax_error_handling(self):
        result = parse_python_file("bad.py", "def broken(:\n  pass")
        assert len(result.errors) > 0

    def test_empty_file(self):
        result = parse_python_file("empty.py", "")
        assert result.language == "python"
        assert len(result.symbols) == 0


class TestJavaScriptParser:
    def test_parse_classes(self, sample_js_source: str):
        result = parse_file("app.js", sample_js_source)
        assert result is not None
        class_names = [s.name for s in result.symbols if s.kind == SymbolKind.CLASS]
        assert "UserService" in class_names

    def test_parse_functions(self, sample_js_source: str):
        result = parse_file("app.js", sample_js_source)
        assert result is not None
        func_names = [s.name for s in result.symbols if s.kind == SymbolKind.FUNCTION]
        assert "formatName" in func_names

    def test_parse_methods(self, sample_js_source: str):
        result = parse_file("app.js", sample_js_source)
        assert result is not None
        method_names = [s.name for s in result.symbols if s.kind == SymbolKind.METHOD]
        assert "getUser" in method_names
        assert "createUser" in method_names

    def test_parse_imports(self, sample_js_source: str):
        result = parse_file("app.js", sample_js_source)
        assert result is not None
        # tree-sitter captures import statements as raw text
        assert any("react" in imp for imp in result.imports)

    def test_typescript_detection(self):
        result = parse_file("app.ts", "const x: number = 1;")
        assert result is not None
        assert result.language == "typescript"


class TestCoreParser:
    def test_auto_detect_python(self, sample_python_source: str):
        result = parse_file("sample.py", sample_python_source)
        assert result is not None
        assert result.language == "python"

    def test_auto_detect_javascript(self, sample_js_source: str):
        result = parse_file("app.js", sample_js_source)
        assert result is not None
        assert result.language == "javascript"

    def test_unsupported_file(self):
        result = parse_file("readme.md", "# Hello")
        assert result is None
