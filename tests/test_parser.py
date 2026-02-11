"""Tests for the code parser module."""

from __future__ import annotations


from cegraph.parser.models import SymbolKind, RelKind, detect_language
from cegraph.parser.python_parser import parse_python_file
from cegraph.parser.core import parse_file


class TestLanguageDetection:
    def test_python(self):
        assert detect_language("main.py") == "python"
        assert detect_language("types.pyi") == "python"

    def test_unknown(self):
        assert detect_language("readme.md") is None
        assert detect_language("data.json") is None
        assert detect_language("app.js") is None
        assert detect_language("app.ts") is None
        assert detect_language("main.go") is None
        assert detect_language("main.rs") is None


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


class TestAnnotatedAssignments:
    def test_annotated_variable(self):
        source = '''
name: str = "hello"
count: int = 42
'''
        result = parse_python_file("ann.py", source)
        vars = [s for s in result.symbols if s.kind == SymbolKind.VARIABLE]
        var_names = [v.name for v in vars]
        assert "name" in var_names
        assert "count" in var_names

    def test_annotated_constant(self):
        source = 'MAX_SIZE: int = 100\n'
        result = parse_python_file("ann.py", source)
        consts = [s for s in result.symbols if s.kind == SymbolKind.CONSTANT]
        assert any(c.name == "MAX_SIZE" for c in consts)

    def test_annotated_class_field(self):
        source = '''
class User:
    name: str
    age: int = 0
    email: str = ""
'''
        result = parse_python_file("ann.py", source)
        vars = [s for s in result.symbols if s.kind == SymbolKind.VARIABLE]
        var_names = [v.name for v in vars]
        assert "name" in var_names
        assert "age" in var_names
        assert "email" in var_names

    def test_type_of_relationship(self):
        source = 'model: Agent = None\n'
        result = parse_python_file("ann.py", source)
        type_rels = [r for r in result.relationships if r.kind == RelKind.TYPE_OF]
        assert any("model" in r.source and r.target == "Agent" for r in type_rels)

    def test_pydantic_style_model(self):
        source = '''
class Config:
    provider: str = "anthropic"
    model: str = "claude"
    max_tokens: int = 4096
    temperature: float = 0.0
'''
        result = parse_python_file("ann.py", source)
        vars = [s for s in result.symbols if s.kind == SymbolKind.VARIABLE]
        var_names = [v.name for v in vars]
        assert "provider" in var_names
        assert "model" in var_names
        assert "max_tokens" in var_names
        assert "temperature" in var_names


class TestCoreParser:
    def test_auto_detect_python(self, sample_python_source: str):
        result = parse_file("sample.py", sample_python_source)
        assert result is not None
        assert result.language == "python"

    def test_unsupported_javascript(self):
        result = parse_file("app.js", "function foo() {}")
        assert result is None

    def test_unsupported_file(self):
        result = parse_file("readme.md", "# Hello")
        assert result is None
