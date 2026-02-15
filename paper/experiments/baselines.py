"""Baseline comparison runner for BCA evaluation.

Implements the baseline retrieval methods described in the evaluation plan
and compares them against BCA on recall and token usage.

Baselines:
  1. Full file (grep): all files containing query terms
  2. BM25 (lexical): symbol-level BM25 scoring, greedy packing
  3. Unweighted BFS: graph BFS without edge weights or closure
  4. BCA (ours): full pipeline

Usage:
    python -m paper.experiments.baselines --repo /path/to/repo --task "fix the bug"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cegraph.context.engine import ContextAssembler
from cegraph.context.models import ContextStrategy, TokenEstimator
from cegraph.graph.builder import GraphBuilder
from cegraph.graph.query import GraphQuery


@dataclass
class BaselineResult:
    """Result of a single baseline run."""

    method: str
    task: str
    budget: int
    tokens_used: int
    symbols_selected: int
    files_included: int
    assembly_time_ms: float
    selected_symbols: list[str] = field(default_factory=list)
    recall: float | None = None
    retrieval_scores: list[float] = field(default_factory=list)  # Raw scores for Router B features


# ---------------------------------------------------------------------------
# Baseline 1: Full file (grep match)
# ---------------------------------------------------------------------------

def baseline_grep(
    repo_path: Path,
    task: str,
    budget: int,
    graph,
) -> BaselineResult:
    """Select all files containing any task keyword, truncate to budget."""
    start = time.time()

    keywords = set(re.findall(r"\b([A-Za-z_]\w{2,})\b", task))
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "have",
        "fix", "bug", "add", "class", "function", "method", "file",
    }
    keywords -= stop

    matched_files: list[tuple[str, str]] = []
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "file":
            continue
        fp = data.get("path", "")
        full_path = repo_path / fp
        if not full_path.exists():
            continue
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if any(kw.lower() in content.lower() for kw in keywords):
            matched_files.append((fp, content))

    # Pack files until budget
    total_tokens = 0
    selected_content: list[str] = []
    files_used = 0
    for fp, content in matched_files:
        tokens = TokenEstimator.estimate(content)
        if total_tokens + tokens > budget:
            remaining = budget - total_tokens
            if remaining > 50:
                chars = int(remaining * TokenEstimator.CHARS_PER_TOKEN)
                selected_content.append(content[:chars])
                total_tokens += remaining
                files_used += 1
            break
        selected_content.append(content)
        total_tokens += tokens
        files_used += 1

    elapsed = (time.time() - start) * 1000

    return BaselineResult(
        method="grep",
        task=task,
        budget=budget,
        tokens_used=total_tokens,
        symbols_selected=0,
        files_included=files_used,
        assembly_time_ms=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# Baseline 2: BM25 (symbol-level lexical)
# ---------------------------------------------------------------------------

def baseline_bm25(
    repo_path: Path,
    task: str,
    budget: int,
    graph,
) -> BaselineResult:
    """BM25 scoring over symbols, greedy packing by score."""
    start = time.time()

    query_terms = re.findall(r"\b([A-Za-z_]\w{2,})\b", task.lower())
    query_tf = Counter(query_terms)

    # Collect symbol documents
    symbols: list[dict] = []
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "symbol":
            continue
        name = data.get("name", "")
        qname = data.get("qualified_name", "")
        doc = data.get("docstring", "")
        text = f"{name} {qname} {doc}".lower()
        symbols.append({"id": node_id, "text": text, "data": data})

    n = len(symbols)
    if n == 0:
        return BaselineResult(
            method="bm25", task=task, budget=budget,
            tokens_used=0, symbols_selected=0, files_included=0,
            assembly_time_ms=0,
        )

    # IDF
    doc_freq: Counter[str] = Counter()
    for sym in symbols:
        terms_in_doc = set(re.findall(r"\b\w+\b", sym["text"]))
        for t in terms_in_doc:
            doc_freq[t] += 1

    k1 = 1.5
    b = 0.75
    avg_dl = sum(len(s["text"]) for s in symbols) / n

    # Score each symbol
    scored: list[tuple[float, dict]] = []
    for sym in symbols:
        dl = len(sym["text"])
        score = 0.0
        for term, qtf in query_tf.items():
            tf = sym["text"].count(term)
            df = doc_freq.get(term, 0)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
            score += idf * tf_norm * qtf
        if score > 0:
            scored.append((score, sym))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Capture top-50 positive scores for Router B confidence features (capped to avoid bloat)
    all_scores = [s for s, _ in scored if s > 0][:50]

    # Greedy packing
    selected: list[str] = []
    total_tokens = 0
    files = set()
    for score, sym in scored:
        line_start = sym["data"].get("line_start", 0)
        line_end = sym["data"].get("line_end", 0)
        line_count = max(1, line_end - line_start + 1)
        cost = TokenEstimator.estimate_lines(line_count)
        if total_tokens + cost > budget:
            continue
        selected.append(sym["data"].get("qualified_name", sym["id"]))
        total_tokens += cost
        fp = sym["data"].get("file_path", "")
        if fp:
            files.add(fp)

    elapsed = (time.time() - start) * 1000

    return BaselineResult(
        method="bm25",
        task=task,
        budget=budget,
        tokens_used=total_tokens,
        symbols_selected=len(selected),
        files_included=len(files),
        assembly_time_ms=round(elapsed, 1),
        selected_symbols=selected,
        retrieval_scores=all_scores,
    )


# ---------------------------------------------------------------------------
# Baseline 3: Unweighted BFS (no edge weights, no closure, no submodular)
# ---------------------------------------------------------------------------

def baseline_unweighted_bfs(
    repo_path: Path,
    task: str,
    budget: int,
    graph,
    query_engine: GraphQuery,
) -> BaselineResult:
    """Unweighted BFS from seeds, greedy packing without closure."""
    from cegraph.context.engine import AblationConfig

    start = time.time()
    ablation = AblationConfig(
        dependency_closure=False,
        submodular_coverage=False,
        centrality_scoring=False,
        file_proximity=False,
        kind_weights=False,
        dependency_ordering=False,
    )
    assembler = ContextAssembler(repo_path, graph, query_engine, ablation=ablation)
    package = assembler.assemble(task=task, token_budget=budget, strategy=ContextStrategy.SMART)
    elapsed = (time.time() - start) * 1000

    return BaselineResult(
        method="unweighted_bfs",
        task=task,
        budget=budget,
        tokens_used=package.total_tokens,
        symbols_selected=package.symbols_included,
        files_included=package.files_included,
        assembly_time_ms=round(elapsed, 1),
        selected_symbols=[
            item.qualified_name or item.name for item in package.items
        ],
    )


# ---------------------------------------------------------------------------
# Baseline 4: Keyword map (aider-style structural summary + relevant files)
# ---------------------------------------------------------------------------

def baseline_keyword_map(
    repo_path: Path,
    task: str,
    budget: int,
    graph,
) -> BaselineResult:
    """Structural summary (file tree + signatures) plus relevant file content.

    Mimics aider's repo-map approach: generate a compact structural overview
    of the repo, then append full content of files matching the query.
    """
    start = time.time()

    # Phase 1: Build structural map (file tree + function/class signatures)
    map_lines: list[str] = []
    files_by_path: dict[str, list[dict]] = {}

    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "symbol":
            continue
        fp = data.get("file_path", "")
        if not fp:
            continue
        files_by_path.setdefault(fp, []).append(data)

    for fp in sorted(files_by_path):
        map_lines.append(fp)
        symbols = sorted(files_by_path[fp], key=lambda d: d.get("line_start", 0))
        for sym in symbols:
            kind = sym.get("kind", "")
            name = sym.get("name", "")
            sig = sym.get("signature", "")
            if kind in ("class", "function", "method"):
                indent = "    " if kind == "method" else "  "
                display = sig if sig else f"{kind} {name}"
                map_lines.append(f"{indent}{display}")

    struct_map = "\n".join(map_lines)
    map_tokens = TokenEstimator.estimate(struct_map)

    # Truncate map if it exceeds budget
    if map_tokens > budget:
        chars_allowed = int(budget * TokenEstimator.CHARS_PER_TOKEN)
        struct_map = struct_map[:chars_allowed]
        map_tokens = budget

    # Phase 2: With remaining budget, add relevant file content
    remaining = budget - map_tokens
    keywords = set(re.findall(r"\b([A-Za-z_]\w{2,})\b", task))
    stop = {"the", "and", "for", "that", "this", "with", "from", "fix", "bug", "add"}
    keywords -= stop

    selected_symbols: list[str] = []
    files_used = set()
    content_tokens = 0

    if remaining > 0:
        for fp in sorted(files_by_path):
            full_path = repo_path / fp
            if not full_path.exists():
                continue
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            if not any(kw.lower() in content.lower() for kw in keywords):
                continue

            file_tokens = TokenEstimator.estimate(content)
            if content_tokens + file_tokens > remaining:
                continue

            content_tokens += file_tokens
            files_used.add(fp)
            for sym in files_by_path.get(fp, []):
                qn = sym.get("qualified_name", sym.get("name", ""))
                if qn:
                    selected_symbols.append(qn)

    elapsed = (time.time() - start) * 1000

    return BaselineResult(
        method="keyword_map",
        task=task,
        budget=budget,
        tokens_used=map_tokens + content_tokens,
        symbols_selected=len(selected_symbols),
        files_included=len(files_used),
        assembly_time_ms=round(elapsed, 1),
        selected_symbols=selected_symbols,
    )


# ---------------------------------------------------------------------------
# BCA (ours)
# ---------------------------------------------------------------------------

def run_bca(
    repo_path: Path,
    task: str,
    budget: int,
    graph,
    query_engine: GraphQuery,
) -> BaselineResult:
    """Full BCA pipeline."""
    start = time.time()
    assembler = ContextAssembler(repo_path, graph, query_engine)
    package = assembler.assemble(task=task, token_budget=budget, strategy=ContextStrategy.SMART)
    elapsed = (time.time() - start) * 1000

    return BaselineResult(
        method="bca",
        task=task,
        budget=budget,
        tokens_used=package.total_tokens,
        symbols_selected=package.symbols_included,
        files_included=package.files_included,
        assembly_time_ms=round(elapsed, 1),
        selected_symbols=[
            item.qualified_name or item.name for item in package.items
        ],
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_comparison(
    repo_path: Path,
    task: str,
    budgets: list[int],
    ground_truth_symbols: list[str] | None = None,
) -> list[BaselineResult]:
    """Run all baselines for a single task across budgets."""
    builder = GraphBuilder()
    graph = builder.build_from_directory(repo_path)
    query = GraphQuery(graph)

    results: list[BaselineResult] = []

    for budget in budgets:
        methods = [
            baseline_grep(repo_path, task, budget, graph),
            baseline_bm25(repo_path, task, budget, graph),
            baseline_keyword_map(repo_path, task, budget, graph),
            baseline_unweighted_bfs(repo_path, task, budget, graph, query),
            run_bca(repo_path, task, budget, graph, query),
        ]

        for r in methods:
            # Compute recall if ground truth provided
            if ground_truth_symbols and r.selected_symbols:
                hits = sum(
                    1 for gt in ground_truth_symbols
                    if any(gt in s or s in gt for s in r.selected_symbols)
                )
                r.recall = hits / len(ground_truth_symbols)
            results.append(r)

    return results


def format_comparison_table(results: list[BaselineResult]) -> str:
    """Format comparison results as a table."""
    lines = []

    budgets = sorted(set(r.budget for r in results))

    for budget in budgets:
        lines.append(f"\n{'='*72}")
        lines.append(f"Budget: {budget} tokens")
        lines.append(f"{'='*72}")
        lines.append(
            f"{'Method':<20} {'Syms':>5} {'Files':>5} {'Tokens':>7} "
            f"{'Time':>8} {'Recall':>7}"
        )
        lines.append("-" * 72)

        budget_results = [r for r in results if r.budget == budget]
        for r in budget_results:
            recall_str = f"{r.recall:.3f}" if r.recall is not None else "n/a"
            lines.append(
                f"{r.method:<20} {r.symbols_selected:>5} "
                f"{r.files_included:>5} {r.tokens_used:>7} "
                f"{r.assembly_time_ms:>7.1f}ms {recall_str:>7}"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="BCA baseline comparison")
    parser.add_argument("--repo", required=True, help="Path to repository")
    parser.add_argument("--task", help="Single task description")
    parser.add_argument("--tasks-file", help="JSONL file with tasks")
    parser.add_argument(
        "--budgets", default="1000,2000,4000,8000",
        help="Comma-separated budget values",
    )
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--ground-truth", help="Comma-separated ground-truth symbol names")
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    budgets = [int(b) for b in args.budgets.split(",")]
    gt_symbols = args.ground_truth.split(",") if args.ground_truth else None

    tasks: list[dict] = []
    if args.task:
        tasks.append({"task": args.task, "ground_truth": gt_symbols})
    elif args.tasks_file:
        with open(args.tasks_file) as f:
            for line in f:
                tasks.append(json.loads(line))
    else:
        parser.error("Provide --task or --tasks-file")

    all_results: list[BaselineResult] = []
    for t in tasks:
        task_str = t["task"]
        gt = t.get("ground_truth", gt_symbols)
        print(f"\nTask: {task_str}")
        results = run_comparison(repo_path, task_str, budgets, gt)
        all_results.extend(results)
        print(format_comparison_table(results))

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
