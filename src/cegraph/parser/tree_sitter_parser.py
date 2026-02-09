"""Tree-sitter based parser for accurate multi-language parsing."""

from __future__ import annotations

from pathlib import Path

from cegraph.parser.models import (
    FileSymbols,
    RelKind,
    Relationship,
    Symbol,
    SymbolKind,
)

# Tree-sitter language module mapping
# These grammars are required dependencies (installed with pip install cegraph)
_TS_LANGUAGE_MODULES = {
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "java": "tree_sitter_java",
}

# Node types that represent symbol definitions per language
_SYMBOL_NODE_TYPES = {
    "javascript": {
        "function_declaration": SymbolKind.FUNCTION,
        "class_declaration": SymbolKind.CLASS,
        "method_definition": SymbolKind.METHOD,
        "arrow_function": SymbolKind.FUNCTION,
        "lexical_declaration": None,
    },
    "typescript": {
        "function_declaration": SymbolKind.FUNCTION,
        "class_declaration": SymbolKind.CLASS,
        "method_definition": SymbolKind.METHOD,
        "interface_declaration": SymbolKind.INTERFACE,
        "type_alias_declaration": SymbolKind.TYPE_ALIAS,
        "enum_declaration": SymbolKind.ENUM,
        "arrow_function": SymbolKind.FUNCTION,
    },
    "go": {
        "function_declaration": SymbolKind.FUNCTION,
        "method_declaration": SymbolKind.METHOD,
        "type_declaration": SymbolKind.CLASS,
    },
    "rust": {
        "function_item": SymbolKind.FUNCTION,
        "struct_item": SymbolKind.CLASS,
        "enum_item": SymbolKind.ENUM,
        "trait_item": SymbolKind.INTERFACE,
        "impl_item": SymbolKind.CLASS,
    },
    "java": {
        "class_declaration": SymbolKind.CLASS,
        "interface_declaration": SymbolKind.INTERFACE,
        "method_declaration": SymbolKind.METHOD,
        "enum_declaration": SymbolKind.ENUM,
    },
}

# Call expression node types per language
_CALL_NODE_TYPES = {
    "javascript": ["call_expression"],
    "typescript": ["call_expression"],
    "go": ["call_expression"],
    "rust": ["call_expression", "macro_invocation"],
    "java": ["method_invocation"],
}


def is_available(language: str | None = None) -> bool:
    """Check if tree-sitter and the required language grammar are available."""
    try:
        import tree_sitter  # noqa: F401
    except ImportError:
        return False

    if language is None:
        return True

    module_name = _TS_LANGUAGE_MODULES.get(language)
    if not module_name:
        return False

    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _get_language(lang: str):
    """Get a tree-sitter Language object for the given language."""
    from tree_sitter import Language

    module_name = _TS_LANGUAGE_MODULES.get(lang)
    if not module_name:
        raise ValueError(f"No tree-sitter grammar for language: {lang}")

    module = __import__(module_name)
    return Language(module.language())


def parse_tree_sitter_file(
    file_path: str, language: str, source: str | None = None
) -> FileSymbols:
    """Parse a file using tree-sitter for accurate AST extraction."""
    from tree_sitter import Parser

    if source is None:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")

    result = FileSymbols(file_path=file_path, language=language)
    source_bytes = source.encode("utf-8")

    try:
        ts_language = _get_language(language)
        parser = Parser(ts_language)
        tree = parser.parse(source_bytes)
    except Exception as e:
        result.errors.append(f"tree-sitter parse error: {e}")
        return result

    _walk_tree(tree.root_node, file_path, language, source_bytes, result)
    return result


def _walk_tree(
    node,
    file_path: str,
    language: str,
    source: bytes,
    result: FileSymbols,
    parent_name: str = "",
    parent_id: str = "",
) -> None:
    """Recursively walk the tree-sitter AST and extract symbols."""
    symbol_types = _SYMBOL_NODE_TYPES.get(language, {})

    for child in node.children:
        node_type = child.type

        if node_type in symbol_types:
            kind = symbol_types[node_type]
            if kind is None:
                # Unwrap (e.g., decorated_definition)
                _walk_tree(child, file_path, language, source, result, parent_name, parent_id)
                continue

            name = _extract_name(child, language)
            if not name:
                continue

            qualified = f"{parent_name}.{name}" if parent_name else name
            if parent_name and kind == SymbolKind.FUNCTION:
                kind = SymbolKind.METHOD

            sig_start = child.start_byte
            sig_end = min(sig_start + 200, child.end_byte)
            sig_text = source[sig_start:sig_end].decode("utf-8", errors="replace")
            # Trim signature to first line or first brace
            sig_text = sig_text.split("\n")[0].strip()

            symbol = Symbol(
                name=name,
                qualified_name=qualified,
                kind=kind,
                file_path=file_path,
                line_start=child.start_point[0] + 1,
                line_end=child.end_point[0] + 1,
                column_start=child.start_point[1],
                column_end=child.end_point[1],
                signature=sig_text,
                parent=parent_id,
            )
            result.symbols.append(symbol)

            if parent_id:
                result.relationships.append(
                    Relationship(
                        source=parent_id,
                        target=symbol.id,
                        kind=RelKind.CONTAINS,
                        file_path=file_path,
                        line=child.start_point[0] + 1,
                    )
                )

            # Extract calls within this symbol
            call_types = _CALL_NODE_TYPES.get(language, [])
            _extract_ts_calls(child, file_path, language, source, symbol.id, call_types, result)

            # Recurse for nested definitions (e.g., methods in classes)
            _walk_tree(
                child, file_path, language, source, result, qualified, symbol.id
            )

        elif node_type in ("import_statement", "import_from_statement", "import_declaration"):
            _extract_ts_import(child, file_path, source, result)
        else:
            # Continue walking
            _walk_tree(child, file_path, language, source, result, parent_name, parent_id)


def _extract_name(node, language: str) -> str:
    """Extract the name of a symbol from its tree-sitter node."""
    # Look for a name/identifier child
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier", "property_identifier"):
            return child.text.decode("utf-8")
    # For some languages, try named children
    name_child = node.child_by_field_name("name")
    if name_child:
        return name_child.text.decode("utf-8")
    return ""


def _extract_ts_calls(
    node, file_path: str, language: str, source: bytes,
    caller_id: str, call_types: list[str], result: FileSymbols
) -> None:
    """Extract function calls from a tree-sitter node."""
    if node.type in call_types:
        # Get the function being called
        func_node = node.child_by_field_name("function")
        if func_node is None and node.children:
            func_node = node.children[0]
        if func_node:
            callee = func_node.text.decode("utf-8")
            # Clean up multiline callees
            callee = callee.split("(")[0].strip()
            if callee and len(callee) < 100:
                result.relationships.append(
                    Relationship(
                        source=caller_id,
                        target=callee,
                        kind=RelKind.CALLS,
                        file_path=file_path,
                        line=node.start_point[0] + 1,
                    )
                )
        return

    for child in node.children:
        _extract_ts_calls(child, file_path, language, source, caller_id, call_types, result)


def _extract_ts_import(node, file_path: str, source: bytes, result: FileSymbols) -> None:
    """Extract import information from a tree-sitter node."""
    text = node.text.decode("utf-8")
    result.imports.append(text)
