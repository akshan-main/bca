"""Data models for parsed code symbols and relationships."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SymbolKind(str, Enum):
    """Types of code symbols."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    INTERFACE = "interface"
    ENUM = "enum"
    TYPE_ALIAS = "type_alias"


class RelKind(str, Enum):
    """Types of relationships between symbols."""

    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    USES = "uses"
    CONTAINS = "contains"
    OVERRIDES = "overrides"
    DECORATES = "decorates"
    TYPE_OF = "type_of"


class Symbol(BaseModel):
    """A code symbol (function, class, method, variable, etc.)."""

    id: str = ""  # auto-generated: "file_path::name"
    name: str
    qualified_name: str = ""  # e.g., "MyClass.my_method"
    kind: SymbolKind
    file_path: str
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0
    signature: str = ""  # e.g., "def my_method(self, x: int) -> str"
    docstring: str = ""
    decorators: list[str] = Field(default_factory=list)
    parent: str = ""  # parent symbol ID (e.g., class for a method)

    def model_post_init(self, __context: object) -> None:
        if not self.id:
            self.id = f"{self.file_path}::{self.qualified_name or self.name}"
        if not self.qualified_name:
            self.qualified_name = self.name


class Relationship(BaseModel):
    """A relationship between two symbols."""

    source: str  # symbol ID
    target: str  # symbol ID or unresolved name
    kind: RelKind
    file_path: str
    line: int
    resolved: bool = False  # whether target is a resolved symbol ID


class FileSymbols(BaseModel):
    """All symbols and relationships extracted from a single file."""

    file_path: str
    language: str
    symbols: list[Symbol] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)  # raw import strings
    errors: list[str] = Field(default_factory=list)


# Language detection by file extension
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
}


def detect_language(file_path: str) -> str | None:
    """Detect programming language from file extension."""
    from pathlib import Path

    ext = Path(file_path).suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext)
