"""Git diff parser — extract changed symbols from unified diffs.

Parses the output of `git diff` to identify which symbols (functions, classes,
methods) were modified, added, or deleted. This is the input layer for the
PR Impact Bot.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DiffHunk:
    """A single hunk from a unified diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FileDiff:
    """Changes to a single file."""
    path: str
    status: str  # 'added', 'modified', 'deleted', 'renamed'
    old_path: str | None = None  # For renames
    hunks: list[DiffHunk] = field(default_factory=list)
    added_lines: int = 0
    deleted_lines: int = 0

    @property
    def changed_line_ranges(self) -> list[tuple[int, int]]:
        """Get ranges of changed lines in the new file."""
        ranges = []
        for hunk in self.hunks:
            ranges.append((hunk.new_start, hunk.new_start + hunk.new_count))
        return ranges


@dataclass
class ChangedSymbol:
    """A symbol that was affected by the diff."""
    name: str
    qualified_name: str
    kind: str
    file_path: str
    line_start: int
    line_end: int
    change_type: str  # 'modified', 'added', 'deleted'
    lines_changed: int = 0


def parse_diff(diff_text: str) -> list[FileDiff]:
    """Parse unified diff text into structured FileDiff objects."""
    files: list[FileDiff] = []
    current_file: FileDiff | None = None
    current_hunk: DiffHunk | None = None

    for line in diff_text.splitlines():
        # New file header
        if line.startswith("diff --git"):
            if current_file:
                files.append(current_file)
            # Extract paths
            parts = line.split(" b/")
            path = parts[-1] if len(parts) > 1 else ""
            current_file = FileDiff(path=path, status="modified")
            current_hunk = None
            continue

        if current_file is None:
            continue

        # File status markers
        if line.startswith("new file"):
            current_file.status = "added"
        elif line.startswith("deleted file"):
            current_file.status = "deleted"
        elif line.startswith("rename from"):
            current_file.old_path = line.split("rename from ")[-1]
            current_file.status = "renamed"
        elif line.startswith("--- a/"):
            pass  # Old file path, already captured
        elif line.startswith("+++ b/"):
            current_file.path = line[6:]
        elif line.startswith("@@"):
            # Hunk header: @@ -old_start,old_count +new_start,new_count @@
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                current_hunk = DiffHunk(
                    old_start=int(match.group(1)),
                    old_count=int(match.group(2) or "1"),
                    new_start=int(match.group(3)),
                    new_count=int(match.group(4) or "1"),
                )
                current_file.hunks.append(current_hunk)
        elif current_hunk is not None:
            current_hunk.lines.append(line)
            if line.startswith("+") and not line.startswith("+++"):
                current_file.added_lines += 1
            elif line.startswith("-") and not line.startswith("---"):
                current_file.deleted_lines += 1

    if current_file:
        files.append(current_file)

    return files


def get_changed_symbols(
    root: Path, graph, file_diffs: list[FileDiff]
) -> list[ChangedSymbol]:
    """Map diff hunks to symbols using the knowledge graph.

    For each changed line range, find which symbols in the graph
    overlap with those lines.
    """
    changed: list[ChangedSymbol] = []
    seen = set()

    for fd in file_diffs:
        if fd.status == "deleted":
            # All symbols in deleted file are affected
            for node_id, data in graph.nodes(data=True):
                if data.get("type") == "symbol" and data.get("file_path") == fd.path:
                    key = (data.get("name"), fd.path)
                    if key not in seen:
                        changed.append(ChangedSymbol(
                            name=data.get("name", ""),
                            qualified_name=data.get("qualified_name", ""),
                            kind=data.get("kind", ""),
                            file_path=fd.path,
                            line_start=data.get("line_start", 0),
                            line_end=data.get("line_end", 0),
                            change_type="deleted",
                        ))
                        seen.add(key)
            continue

        # For modified/added files, find overlapping symbols
        ranges = fd.changed_line_ranges
        for node_id, data in graph.nodes(data=True):
            if data.get("type") != "symbol":
                continue
            if data.get("file_path") != fd.path:
                continue

            sym_start = data.get("line_start", 0)
            sym_end = data.get("line_end", 0)

            # Check if any hunk overlaps with this symbol
            for r_start, r_end in ranges:
                if sym_start <= r_end and sym_end >= r_start:
                    key = (data.get("name"), fd.path)
                    if key not in seen:
                        overlap = min(sym_end, r_end) - max(sym_start, r_start)
                        change_type = "added" if fd.status == "added" else "modified"
                        changed.append(ChangedSymbol(
                            name=data.get("name", ""),
                            qualified_name=data.get("qualified_name", ""),
                            kind=data.get("kind", ""),
                            file_path=fd.path,
                            line_start=sym_start,
                            line_end=sym_end,
                            change_type=change_type,
                            lines_changed=max(0, overlap),
                        ))
                        seen.add(key)
                    break

    return changed


def get_git_diff(root: Path, base: str = "main") -> str:
    """Get the git diff between the current branch and base."""
    try:
        result = subprocess.run(
            ["git", "diff", f"{base}...HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
        # Fallback: diff against base directly
        result = subprocess.run(
            ["git", "diff", base],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def get_pr_diff(root: Path) -> str:
    """Get the diff for the current PR (GitHub Actions context)."""
    import os
    base_ref = os.environ.get("GITHUB_BASE_REF", "main")
    return get_git_diff(root, base=f"origin/{base_ref}")
