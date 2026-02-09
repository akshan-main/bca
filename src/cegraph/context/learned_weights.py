"""Learn edge-type weights from git commit co-change history.

Analyzes which symbols change together in commits.  If two symbols connected by
an edge of type k frequently co-change, that edge type gets a higher weight.

This replaces the hand-tuned _EDGE_WEIGHTS dict with empirically derived values.
"""

from __future__ import annotations

import subprocess
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx


def learn_edge_weights(
    root: Path,
    graph: nx.DiGraph,
    max_commits: int = 500,
    min_samples: int = 5,
) -> dict[str, float]:
    """Learn edge-type weights from git co-change frequency.

    For each edge type k, compute:
        w(k) = P(co-change | edge of type k)

    where co-change means both endpoints were modified in the same commit.

    Falls back to default weights if git history is unavailable or insufficient.

    Args:
        root: Repository root path.
        graph: Code knowledge graph.
        max_commits: Maximum number of commits to analyze.
        min_samples: Minimum edge samples per type to trust the learned weight.

    Returns:
        Dict mapping edge type to learned weight in [0, 1].
    """
    from cegraph.context.engine import _EDGE_WEIGHTS

    try:
        changed_files_per_commit = _get_commit_file_changes(root, max_commits)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return dict(_EDGE_WEIGHTS)

    if len(changed_files_per_commit) < 10:
        return dict(_EDGE_WEIGHTS)

    # Build file -> symbols index
    file_to_symbols: dict[str, set[str]] = defaultdict(set)
    for node_id, data in graph.nodes(data=True):
        if data.get("type") == "symbol":
            fp = data.get("file_path", "")
            if fp:
                file_to_symbols[fp].add(node_id)

    # For each commit, find which symbols co-changed
    co_changed_pairs: set[tuple[str, str]] = set()
    for files in changed_files_per_commit:
        commit_symbols: set[str] = set()
        for f in files:
            commit_symbols |= file_to_symbols.get(f, set())

        # All pairs of co-changed symbols
        sym_list = sorted(commit_symbols)
        for i in range(len(sym_list)):
            for j in range(i + 1, min(i + 50, len(sym_list))):
                co_changed_pairs.add((sym_list[i], sym_list[j]))

    # Count co-changes per edge type
    edge_type_total: Counter[str] = Counter()
    edge_type_cochanged: Counter[str] = Counter()

    for u, v, data in graph.edges(data=True):
        kind = data.get("kind", "")
        if not kind:
            continue
        edge_type_total[kind] += 1
        if (u, v) in co_changed_pairs or (v, u) in co_changed_pairs:
            edge_type_cochanged[kind] += 1

    # Compute learned weights
    learned: dict[str, float] = {}
    for kind in _EDGE_WEIGHTS:
        total = edge_type_total.get(kind, 0)
        cochanged = edge_type_cochanged.get(kind, 0)

        if total >= min_samples:
            raw = cochanged / total
            # Clamp to [0.1, 1.0] and smooth toward prior
            prior = _EDGE_WEIGHTS[kind]
            alpha = min(total / 50, 1.0)  # confidence ramp
            learned[kind] = alpha * raw + (1 - alpha) * prior
            learned[kind] = max(0.1, min(1.0, learned[kind]))
        else:
            learned[kind] = _EDGE_WEIGHTS[kind]

    return learned


def _get_commit_file_changes(
    root: Path, max_commits: int
) -> list[set[str]]:
    """Extract per-commit changed file sets from git log."""
    result = subprocess.run(
        ["git", "log", f"--max-count={max_commits}", "--name-only", "--format=%H"],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=30,
    )
    if result.returncode != 0:
        return []

    commits: list[set[str]] = []
    current_files: set[str] = set()

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            if current_files:
                commits.append(current_files)
                current_files = set()
            continue
        if re.match(r"^[0-9a-f]{40}$", line):
            if current_files:
                commits.append(current_files)
            current_files = set()
        else:
            current_files.add(line)

    if current_files:
        commits.append(current_files)

    return commits
