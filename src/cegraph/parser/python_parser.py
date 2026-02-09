"""Python-specific parser using the built-in ast module. Always available, no extra deps."""

from __future__ import annotations

import ast
from pathlib import Path

from cegraph.parser.models import (
    FileSymbols,
    Relationship,
    RelKind,
    Symbol,
    SymbolKind,
)


def parse_python_file(file_path: str, source: str | None = None) -> FileSymbols:
    """Parse a Python file and extract symbols and relationships."""
    if source is None:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")

    result = FileSymbols(file_path=file_path, language="python")

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        result.errors.append(f"SyntaxError: {e}")
        return result

    lines = source.splitlines()
    _extract_from_module(tree, file_path, lines, result)
    return result


def _get_docstring(node: ast.AST) -> str:
    """Extract docstring from a node if present."""
    if (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, (ast.Constant,))
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value.strip()
    return ""


def _get_decorators(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Extract decorator names."""
    decorators = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            decorators.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            decorators.append(ast.dump(dec))
        elif isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                decorators.append(dec.func.id)
            elif isinstance(dec.func, ast.Attribute):
                decorators.append(ast.dump(dec.func))
    return decorators


def _get_function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]
) -> str:
    """Extract the function signature from source lines."""
    start = node.lineno - 1
    sig_lines = []
    for i in range(start, min(start + 10, len(lines))):
        line = lines[i]
        sig_lines.append(line.strip())
        if ":" in line:
            # Check if we've reached the colon that ends the signature
            text = "".join(sig_lines)
            if text.count("(") <= text.count(")"):
                break
    sig = " ".join(sig_lines)
    # Trim to just the def ... : part
    if ":" in sig:
        sig = sig[: sig.rindex(":") + 1]
    return sig


def _extract_from_module(
    tree: ast.Module,
    file_path: str,
    lines: list[str],
    result: FileSymbols,
    parent_name: str = "",
    parent_id: str = "",
) -> None:
    """Recursively extract symbols from an AST module/class body."""

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _extract_import(node, file_path, result)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _extract_function(node, file_path, lines, result, parent_name, parent_id)

        elif isinstance(node, ast.ClassDef):
            _extract_class(node, file_path, lines, result, parent_name, parent_id)

        elif isinstance(node, ast.Assign):
            _extract_assignment(node, file_path, result, parent_name, parent_id)


def _extract_import(
    node: ast.Import | ast.ImportFrom, file_path: str, result: FileSymbols
) -> None:
    """Extract import statements."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name
            result.imports.append(alias.name)
            result.symbols.append(
                Symbol(
                    name=name,
                    kind=SymbolKind.IMPORT,
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                )
            )
            result.relationships.append(
                Relationship(
                    source=f"{file_path}::{name}",
                    target=alias.name,
                    kind=RelKind.IMPORTS,
                    file_path=file_path,
                    line=node.lineno,
                )
            )
    elif isinstance(node, ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            full_import = f"{module}.{alias.name}" if module else alias.name
            result.imports.append(full_import)
            result.symbols.append(
                Symbol(
                    name=name,
                    kind=SymbolKind.IMPORT,
                    file_path=file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                )
            )
            result.relationships.append(
                Relationship(
                    source=f"{file_path}::{name}",
                    target=full_import,
                    kind=RelKind.IMPORTS,
                    file_path=file_path,
                    line=node.lineno,
                )
            )


def _extract_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
    lines: list[str],
    result: FileSymbols,
    parent_name: str,
    parent_id: str,
) -> None:
    """Extract function/method definition."""
    qualified = f"{parent_name}.{node.name}" if parent_name else node.name
    kind = SymbolKind.METHOD if parent_name else SymbolKind.FUNCTION
    sig = _get_function_signature(node, lines)

    symbol = Symbol(
        name=node.name,
        qualified_name=qualified,
        kind=kind,
        file_path=file_path,
        line_start=node.lineno,
        line_end=node.end_lineno or node.lineno,
        signature=sig,
        docstring=_get_docstring(node),
        decorators=_get_decorators(node),
        parent=parent_id,
    )
    result.symbols.append(symbol)

    # Add contains relationship
    if parent_id:
        result.relationships.append(
            Relationship(
                source=parent_id,
                target=symbol.id,
                kind=RelKind.CONTAINS,
                file_path=file_path,
                line=node.lineno,
            )
        )

    # Extract calls within the function body
    _extract_calls(node, file_path, symbol.id, result)


def _extract_class(
    node: ast.ClassDef,
    file_path: str,
    lines: list[str],
    result: FileSymbols,
    parent_name: str,
    parent_id: str,
) -> None:
    """Extract class definition and its members."""
    qualified = f"{parent_name}.{node.name}" if parent_name else node.name

    symbol = Symbol(
        name=node.name,
        qualified_name=qualified,
        kind=SymbolKind.CLASS,
        file_path=file_path,
        line_start=node.lineno,
        line_end=node.end_lineno or node.lineno,
        signature=f"class {node.name}",
        docstring=_get_docstring(node),
        decorators=_get_decorators(node),
        parent=parent_id,
    )
    result.symbols.append(symbol)

    # Add contains relationship
    if parent_id:
        result.relationships.append(
            Relationship(
                source=parent_id,
                target=symbol.id,
                kind=RelKind.CONTAINS,
                file_path=file_path,
                line=node.lineno,
            )
        )

    # Extract inheritance
    for base in node.bases:
        base_name = _node_to_name(base)
        if base_name:
            result.relationships.append(
                Relationship(
                    source=symbol.id,
                    target=base_name,
                    kind=RelKind.INHERITS,
                    file_path=file_path,
                    line=node.lineno,
                )
            )

    # Recurse into class body
    _extract_from_module(node, file_path, lines, result, qualified, symbol.id)


def _extract_assignment(
    node: ast.Assign,
    file_path: str,
    result: FileSymbols,
    parent_name: str,
    parent_id: str,
) -> None:
    """Extract variable/constant assignments at module or class level."""
    for target in node.targets:
        name = _node_to_name(target)
        if not name:
            continue
        qualified = f"{parent_name}.{name}" if parent_name else name
        kind = SymbolKind.CONSTANT if name.isupper() else SymbolKind.VARIABLE

        symbol = Symbol(
            name=name,
            qualified_name=qualified,
            kind=kind,
            file_path=file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            parent=parent_id,
        )
        result.symbols.append(symbol)


def _extract_calls(
    node: ast.AST, file_path: str, caller_id: str, result: FileSymbols
) -> None:
    """Walk a function body and extract all function calls."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            callee = _node_to_name(child.func)
            if callee:
                result.relationships.append(
                    Relationship(
                        source=caller_id,
                        target=callee,
                        kind=RelKind.CALLS,
                        file_path=file_path,
                        line=child.lineno if hasattr(child, "lineno") else 0,
                    )
                )


def _node_to_name(node: ast.AST) -> str:
    """Convert an AST node to a dotted name string."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        parent = _node_to_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    elif isinstance(node, ast.Subscript):
        return _node_to_name(node.value)
    return ""
