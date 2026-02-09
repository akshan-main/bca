"""Core parser orchestration - selects the best parser for each file."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from cegraph.config import IndexerConfig
from cegraph.parser.models import FileSymbols, detect_language


def parse_file(file_path: str, source: str | None = None) -> FileSymbols | None:
    """Parse a single file, auto-detecting language and selecting the best parser.

    Returns None if the file's language is not supported.

    - Python: uses stdlib ast (zero deps, high accuracy)
    - JS/TS, Go, Rust, Java: uses tree-sitter (required dep, full AST)
    """
    language = detect_language(file_path)
    if not language:
        return None

    # Python uses stdlib AST — better than tree-sitter for Python
    if language == "python":
        from cegraph.parser.python_parser import parse_python_file

        return parse_python_file(file_path, source)

    # Everything else uses tree-sitter
    from cegraph.parser.tree_sitter_parser import is_available, parse_tree_sitter_file

    if is_available(language):
        return parse_tree_sitter_file(file_path, language, source)

    # Language detected but no tree-sitter grammar installed
    return None


def parse_directory(
    root: str | Path,
    config: IndexerConfig | None = None,
    progress_callback: callable | None = None,
) -> list[FileSymbols]:
    """Parse all supported source files in a directory tree.

    Args:
        root: Root directory to scan.
        config: Indexer configuration for exclusion patterns.
        progress_callback: Optional callback(file_path, current, total) for progress.

    Returns:
        List of FileSymbols for each parsed file.
    """
    root = Path(root).resolve()
    if config is None:
        config = IndexerConfig()

    # Collect files to parse
    files = _collect_files(root, config)

    results = []
    total = len(files)
    for i, file_path in enumerate(files):
        if progress_callback:
            progress_callback(str(file_path), i + 1, total)

        try:
            rel_path = str(file_path.relative_to(root))
            source = file_path.read_text(encoding="utf-8", errors="replace")
            parsed = parse_file(rel_path, source)
            if parsed:
                results.append(parsed)
        except Exception:
            # Skip files that can't be parsed
            continue

    return results


def collect_files(root: str | Path, config: IndexerConfig | None = None) -> list[Path]:
    """Public API: collect all parseable files in a directory."""
    root = Path(root).resolve()
    if config is None:
        config = IndexerConfig()
    return _collect_files(root, config)


def parse_files(
    root: str | Path,
    file_paths: list[str],
    config: IndexerConfig | None = None,
    progress_callback: callable | None = None,
) -> list[FileSymbols]:
    """Parse a specific set of files (by relative path).

    Like parse_directory but only processes the listed files.
    """
    root = Path(root).resolve()
    if config is None:
        config = IndexerConfig()

    results = []
    total = len(file_paths)
    for i, rel_path in enumerate(file_paths):
        if progress_callback:
            progress_callback(rel_path, i + 1, total)
        full_path = root / rel_path
        if not full_path.exists():
            continue
        try:
            source = full_path.read_text(encoding="utf-8", errors="replace")
            parsed = parse_file(rel_path, source)
            if parsed:
                results.append(parsed)
        except Exception:
            continue
    return results


def _collect_files(root: Path, config: IndexerConfig) -> list[Path]:
    """Collect all parseable files, respecting exclusion patterns."""
    files = []
    max_size = config.max_file_size_kb * 1024

    # Read .gitignore if available
    gitignore_patterns = _read_gitignore(root)
    all_exclude = config.exclude_patterns + gitignore_patterns

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)

        # Filter out excluded directories
        dirnames[:] = [
            d
            for d in dirnames
            if not _should_exclude(os.path.join(rel_dir, d) if rel_dir != "." else d, all_exclude)
        ]

        for filename in filenames:
            rel_path = (
                os.path.join(rel_dir, filename) if rel_dir != "." else filename
            )

            # Check exclusion patterns
            if _should_exclude(rel_path, all_exclude):
                continue

            # Check language support
            lang = detect_language(filename)
            if lang is None:
                continue

            # Filter by configured languages (empty list = all)
            if config.languages and lang not in config.languages:
                continue

            full_path = Path(dirpath) / filename

            # Check file size
            try:
                if full_path.stat().st_size > max_size:
                    continue
            except OSError:
                continue

            files.append(full_path)

    return sorted(files)


def _should_exclude(path: str, patterns: list[str]) -> bool:
    """Check if a path matches any exclusion pattern."""
    path_parts = Path(path).parts
    for pattern in patterns:
        # Check against full path
        if fnmatch.fnmatch(path, pattern):
            return True
        # Check against any path component
        for part in path_parts:
            if fnmatch.fnmatch(part, pattern):
                return True
    return False


def _read_gitignore(root: Path) -> list[str]:
    """Read .gitignore patterns from the project root."""
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        return []

    patterns = []
    try:
        for line in gitignore.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                # Normalize the pattern
                if line.endswith("/"):
                    line = line[:-1]
                patterns.append(line)
    except OSError:
        pass
    return patterns
