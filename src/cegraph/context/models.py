"""Data models for budgeted context assembly."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ContextStrategy(str, Enum):
    """Strategy for assembling context."""

    PRECISE = "precise"  # Only directly relevant symbols
    SMART = "smart"  # Graph-expanded with relevance scoring (default)
    THOROUGH = "thorough"  # Deep expansion, all related code


class ContextItem(BaseModel):
    """A single item in the assembled context."""

    symbol_id: str
    name: str
    qualified_name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    source_code: str
    signature: str = ""
    docstring: str = ""
    relevance_score: float = 0.0
    reason: str = ""  # Why this was included
    token_estimate: int = 0
    depth: int = 0  # Distance from seed symbols
    is_dependency: bool = False  # Included via dependency closure
    is_skeleton: bool = False  # Skeleton mode: signature+docstring only (budget fallback)


class ContextPackage(BaseModel):
    """The complete assembled context package ready for LLM consumption."""

    task: str
    strategy: ContextStrategy
    items: list[ContextItem] = Field(default_factory=list)
    seed_symbols: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    token_budget: int = 0
    files_included: int = 0
    symbols_included: int = 0
    symbols_available: int = 0  # Total candidates before budget cut
    budget_used_pct: float = 0.0
    assembly_time_ms: float = 0.0
    # Debug/metrics info (populated by assembler, not used in rendering)
    entities_extracted: int = 0
    entities_mapped: int = 0  # How many entities resolved to graph symbols
    closure_added_symbols: int = 0
    closure_added_tokens: int = 0
    frontier_visited: int = 0  # BFS expansion candidates before scoring

    def render(self, include_line_numbers: bool = True, include_metadata: bool = True) -> str:
        """Render the context package as a string for LLM consumption.

        This is the key output - a carefully structured text that gives the LLM
        exactly what it needs to understand the relevant code.
        """
        sections: list[str] = []

        if include_metadata:
            sections.append(f"# Codebase Context for: {self.task}")
            sections.append(
                f"# {self.symbols_included} symbols from {self.files_included} files "
                f"(~{self.total_tokens:,} tokens, {self.budget_used_pct:.0f}% of budget)"
            )
            sections.append("")

        # Group items by file
        by_file: dict[str, list[ContextItem]] = {}
        for item in self.items:
            by_file.setdefault(item.file_path, []).append(item)

        for file_path, items in sorted(by_file.items()):
            sections.append(f"## {file_path}")
            if include_metadata:
                reasons = set(i.reason for i in items if i.reason)
                if reasons:
                    sections.append(f"# Included because: {'; '.join(reasons)}")
            sections.append("")

            # Sort items by line number
            items.sort(key=lambda x: x.line_start)

            for item in items:
                if include_metadata:
                    sections.append(
                        f"# [{item.kind}] {item.qualified_name} "
                        f"(relevance: {item.relevance_score:.2f}, depth: {item.depth})"
                    )

                if include_line_numbers:
                    lines = item.source_code.splitlines()
                    for i, line in enumerate(lines):
                        sections.append(f"{item.line_start + i:4d} | {line}")
                else:
                    sections.append(item.source_code)

                sections.append("")

        return "\n".join(sections)

    def render_compact(self) -> str:
        """Render a compact version - signatures + docstrings only for secondary symbols."""
        sections: list[str] = []
        sections.append(f"# Context for: {self.task}")
        sections.append("")

        by_file: dict[str, list[ContextItem]] = {}
        for item in self.items:
            by_file.setdefault(item.file_path, []).append(item)

        for file_path, items in sorted(by_file.items()):
            sections.append(f"## {file_path}")
            items.sort(key=lambda x: x.line_start)

            for item in items:
                if item.depth == 0:
                    # Primary symbols: full source
                    sections.append(item.source_code)
                else:
                    # Secondary: signature + docstring only
                    sections.append(item.signature)
                    if item.docstring:
                        doc_preview = item.docstring[:150]
                        if len(item.docstring) > 150:
                            doc_preview += "..."
                        sections.append(f'    """{doc_preview}"""')
                sections.append("")

        return "\n".join(sections)

    def summary(self) -> str:
        """Human-readable summary of what's in the context."""
        lines = [
            f"CAG Context Package for: {self.task}",
            f"Strategy: {self.strategy.value}",
            f"Tokens: {self.total_tokens:,} / {self.token_budget:,} ({self.budget_used_pct:.0f}%)",
            f"Symbols: {self.symbols_included} included, {self.symbols_available} candidates",
            f"Files: {self.files_included}",
            f"Assembly time: {self.assembly_time_ms:.1f}ms",
            "",
            "Included symbols:",
        ]
        for item in self.items:
            marker = ">" if item.depth == 0 else " " * item.depth + "·"
            lines.append(
                f"  {marker} {item.qualified_name} ({item.kind}) "
                f"[{item.file_path}:{item.line_start}] "
                f"score={item.relevance_score:.2f} ~{item.token_estimate}tok"
            )
            if item.reason:
                lines.append(f"    reason: {item.reason}")

        return "\n".join(lines)


class TokenEstimator:
    """Estimate token counts for code."""

    # Rough heuristic: 1 token ≈ 4 characters for code
    CHARS_PER_TOKEN = 4.0

    @classmethod
    def estimate(cls, text: str) -> int:
        """Estimate token count for a string."""
        return max(1, int(len(text) / cls.CHARS_PER_TOKEN))

    @classmethod
    def estimate_lines(cls, line_count: int, avg_line_length: int = 40) -> int:
        """Estimate tokens for a given number of lines."""
        return max(1, int(line_count * avg_line_length / cls.CHARS_PER_TOKEN))
