"""End-to-end benchmark harness for BCA evaluation.

Takes coding tasks, assembles context with each method, calls an LLM,
applies the generated patch, runs tests, and records pass/fail.

This is the line between "implementation" and "paper": it produces the
pass@1 vs token-budget plots that constitute evidence.

Usage:
    # Full paper run (245 tasks, all methods, both query types):
    python -m paper.experiments.benchmark \
        --tasks-file paper/experiments/eval_tasks_full.jsonl \
        --budgets 1000,4000,8000,10000 \
        --query-types exact,vague \
        --provider openai \
        --model gpt-4o-mini-2024-07-18 \
        --output-dir paper/results/

    # Quick sanity check (2 tasks, 2 budgets, 5 methods):
    python -m paper.experiments.benchmark \
        --tasks-file paper/experiments/eval_tasks_trial.jsonl \
        --quick \
        --output-dir paper/results/trial/

Task JSONL format:
    {
        "task_id": "unique-id",
        "repo_path": "/path/to/repo",
        "repo_url": "https://github.com/...",  (optional, for cloning)
        "commit": "abc123",                     (optional, checkout before eval)
        "description": "Fix the bug in ...",
        "vague_description": "Something seems off...",  (optional, user-reported)
        "test_cmd": "python -m pytest tests/test_foo.py -x",
        "setup_cmd": "pip install -e .",        (optional)
        "in_place": true,                       (optional, for editable installs)
        "timeout": 60                           (optional, test timeout in seconds)
    }
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

# Load .env file from project root
_env_file = Path(__file__).resolve().parents[2] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from cegraph.config import LLMConfig
from cegraph.context.engine import AblationConfig, ContextAssembler
from cegraph.context.models import ContextStrategy, TokenEstimator
from cegraph.graph.builder import GraphBuilder
from cegraph.graph.query import GraphQuery
from cegraph.llm.base import LLMResponse, Message
from cegraph.llm.factory import create_provider
from paper.experiments.baselines import (
    baseline_bm25,
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EvalTask:
    task_id: str
    repo_path: str
    description: str
    test_cmd: str
    repo_url: str = ""
    commit: str = ""
    setup_cmd: str = ""
    vague_description: str = ""
    dev_report_description: str = ""  # test failure + traceback (no line numbers)
    in_place: bool = False  # True for editable installs (e.g. pydantic-ai)
    timeout: int = 60
    mutation: dict = field(default_factory=dict)  # {file, original, mutated, line_num}
    category: str = ""  # subsystem category (e.g. "auth", "models", "ssrf")
    mutation_type: str = ""  # mutation kind (e.g. "comparison_swap", "constant_mutation")
    source: str = ""  # task origin: "handcrafted" or "discovered"


@dataclass
class EvalResult:
    task_id: str
    method: str
    budget: int
    query_type: str  # "exact", "dev_report", or "vague"
    tokens_used: int
    symbols_selected: int
    files_included: int
    assembly_time_ms: float
    llm_time_ms: float
    llm_input_tokens: int
    llm_output_tokens: int
    tests_passed: bool
    test_output: str
    patch: str
    error: str = ""
    test_time_ms: float = 0  # Time for apply + test
    # Failure diagnosis
    failure_mode: str = ""  # pass, no_patch, patch_apply_fail, test_fail, regression, timeout, syntax_error, llm_error, assembly_error
    # Retrieval quality
    target_file_hit: bool = False  # Did context include the mutated file?
    target_symbol_hit: bool = False  # Did context include the mutated function/class?
    context_patch_overlap: float = 0.0  # Fraction of context file paths referenced in patch
    # Patch quality
    patch_files_changed: int = 0
    patch_lines_changed: int = 0
    edit_distance_lines: int = -1  # Distance from edited line to mutated line (-1 = unknown)
    # Entity & seed metrics (logged for every method, not just BCA)
    entity_count_extracted: int = 0  # Entities found in query text
    entity_count_mapped: int = 0  # Entities that resolved to graph symbols
    query_identifier_density: float = 0.0  # Fraction of query tokens that are identifiers
    seed_symbol_keys: list[str] = field(default_factory=list)  # Graph symbol IDs of seeds
    mutation_symbol_key: str = ""  # Graph symbol ID containing the mutation
    # Graph distance: seed symbols to mutation symbol
    min_hops_seed_to_mutation: int = -1  # -1 = unknown/unreachable
    median_hops_seed_to_mutation: float = -1.0
    # BCA-specific debug (None/0 for non-BCA methods)
    bca_closure_added_symbols: int = 0
    bca_closure_added_tokens: int = 0
    bca_frontier_visited: int = 0
    # Context symbols (for post-hoc dependency coverage analysis)
    context_symbol_keys: list[str] = field(default_factory=list)
    # Code metadata (per-task constants, logged per-attempt for self-contained analysis)
    mutation_symbol_lines: int = 0  # Line span of the mutated function/class
    mutation_symbol_kind: str = ""  # function, method, class, constant, etc.
    mutation_file_symbols: int = 0  # Number of symbols in the mutated file
    graph_node_count: int = 0  # Total graph nodes (normalizes across repos)
    # Router B confidence features (retrieval scores, computed pre-LLM).
    # WARNING: Raw scores are NOT comparable across methods (BM25, TF-IDF, BCA use
    # different scales). Use scale-free features for cross-method routing.
    # --- Scale-free features (safe for cross-method comparison) ---
    retrieval_top1_top2_gap: float = 0.0  # Score gap: rank-1 minus rank-2 (scale-dependent but gap is relative)
    retrieval_softmax_entropy: float = 0.0  # Entropy of softmax(top-K scores) in bits. Low = confident, high = scattered.
    retrieval_softmax_tau: float = 0.0  # Temperature used for softmax (median(|top-K scores|)). Logged for auditability.
    retrieval_effective_candidates: float = 0.0  # exp(entropy) = perplexity. "How many plausible choices?"
    retrieval_top5_ratio: float = 0.0  # mean(top5) / top1. Close to 1.0 = flat ranking, close to 0 = clear winner.
    retrieval_within95_count: int = 0  # How many symbols score >= 95% of top1. Measures ranking sharpness.
    retrieval_scored_symbols: int = 0  # Number of symbols with score > 0
    # --- Raw score features (per-method only, NOT for cross-method comparison) ---
    retrieval_top1_score: float = 0.0  # Highest retrieval score (scale varies by method)
    retrieval_top5_mean_score: float = 0.0  # Mean of top-5 scores (scale varies)
    # --- Coverage confidence features (scale-free, method-agnostic) ---
    retrieval_budget_utilization: float = 0.0  # tokens_used / budget (already in other fields but useful as feature)
    retrieval_file_concentration: float = 0.0  # Fraction of top-K symbols in the same file as top-1. High = focused.
    # Slicing dimensions (ensure every result is self-contained for analysis)
    repo_name: str = ""  # Repository name (e.g. "pydantic-ai", "httpx")
    category: str = ""  # Subsystem category (e.g. "auth", "ssrf", "client")
    mutation_type: str = ""  # Mutation kind (e.g. "comparison_swap", "handcrafted")
    source: str = ""  # Task origin: "handcrafted" or "discovered"


# ---------------------------------------------------------------------------
# Context assembly methods
# ---------------------------------------------------------------------------

# Module-level variable to capture the last ContextPackage from BCA methods.
# Read by _run_single_eval to extract debug info without changing method signatures.
_last_context_package = None

# Module-level list to capture retrieval scores from the most recent assembly call.
# Populated by scored methods (BM25, TF-IDF, embedding, BCA); empty for unscored
# methods (grep, no_retrieval, naive_random, keyword_map, target_file).
# Read by _run_single_eval to compute Router B confidence features.
_last_retrieval_scores: list[float] = []


def _compute_retrieval_confidence(scores: list[float]) -> dict[str, float | int]:
    """Compute Router B confidence features from raw retrieval scores.

    Uses softmax probabilities for entropy (not raw normalization) so the
    distribution is well-defined even with negative or unbounded scores.
    All features except raw scores are scale-free or scale-invariant.
    """
    import math

    empty = {
        "retrieval_top1_score": 0.0,
        "retrieval_top1_top2_gap": 0.0,
        "retrieval_softmax_entropy": 0.0,
        "retrieval_softmax_tau": 0.0,
        "retrieval_effective_candidates": 0.0,
        "retrieval_top5_ratio": 0.0,
        "retrieval_within95_count": 0,
        "retrieval_top5_mean_score": 0.0,
        "retrieval_scored_symbols": 0,
    }
    if not scores:
        return empty

    sorted_scores = sorted(scores, reverse=True)
    top1 = sorted_scores[0]
    top2 = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    top5 = sorted_scores[:5]
    top_k = sorted_scores[:20]  # Use top-20 for entropy

    # --- Softmax entropy (proper probability distribution) ---
    # Temperature tau = median(|scores|) to avoid degenerate distributions.
    # If tau is tiny, fall back to tau=1.0.
    abs_scores = [abs(s) for s in top_k]
    abs_sorted = sorted(abs_scores)
    tau = abs_sorted[len(abs_sorted) // 2] if abs_sorted else 1.0
    if tau < 1e-9:
        tau = 1.0

    # Softmax with numerical stability (subtract max before exp)
    max_s = max(top_k)
    exps = [math.exp((s - max_s) / tau) for s in top_k]
    exp_sum = sum(exps)
    entropy = 0.0
    if exp_sum > 0:
        for e in exps:
            p = e / exp_sum
            if p > 1e-12:
                entropy -= p * math.log2(p)

    effective_candidates = 2.0 ** entropy  # Perplexity: "how many plausible choices"

    # --- Scale-free ratio features ---
    top5_ratio = (sum(top5) / len(top5)) / top1 if top1 > 1e-12 else 0.0
    threshold_95 = top1 * 0.95
    within95 = sum(1 for s in sorted_scores if s >= threshold_95)

    return {
        "retrieval_top1_score": round(top1, 6),
        "retrieval_top1_top2_gap": round(top1 - top2, 6),
        "retrieval_softmax_entropy": round(entropy, 4),
        "retrieval_softmax_tau": round(tau, 6),
        "retrieval_effective_candidates": round(effective_candidates, 2),
        "retrieval_top5_ratio": round(top5_ratio, 4),
        "retrieval_within95_count": within95,
        "retrieval_top5_mean_score": round(sum(top5) / len(top5), 6),
        "retrieval_scored_symbols": len(scores),
    }


def _compute_file_concentration(context: str) -> float:
    """Fraction of context symbols that share a file with the most-frequent file.

    High concentration = retrieval is focused on one file.
    Low concentration = retrieval is scattered across many files.
    Scale-free: always in [0, 1] regardless of method.
    """
    # Extract file headers from context (format: "# filepath:start-end")
    file_refs = re.findall(r"^# ([a-zA-Z0-9_/.]+\.\w+)", context, re.MULTILINE)
    if not file_refs:
        return 0.0
    from collections import Counter as _Counter
    counts = _Counter(file_refs)
    most_common_count = counts.most_common(1)[0][1]
    return round(most_common_count / len(file_refs), 4)


def _extract_bca_scores(package) -> list[float]:
    """Extract relevance scores from a BCA ContextPackage for Router B features.

    Returns top-50 positive scores (sorted descending). Capped to avoid
    storing hundreds of scores per attempt in artifacts.
    """
    scores = sorted(
        [item.relevance_score for item in package.items if item.relevance_score > 0],
        reverse=True,
    )
    return scores[:50]


def assemble_bca(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """BCA context assembly. Returns (context_str, tokens, syms, files, time_ms)."""
    global _last_context_package, _last_retrieval_scores
    start = time.time()
    assembler = ContextAssembler(repo_path, graph, query)
    package = assembler.assemble(task=task, token_budget=budget, strategy=ContextStrategy.SMART)
    elapsed = (time.time() - start) * 1000
    _last_context_package = package
    _last_retrieval_scores = _extract_bca_scores(package)
    # Use lean rendering: no metadata annotations or line-number prefixes.
    # This matches the format of other methods (just file headers + source code)
    # and avoids wasting budget tokens on annotations the LLM doesn't need.
    return (
        package.render(include_metadata=False, include_line_numbers=False),
        package.total_tokens,
        package.symbols_included,
        package.files_included,
        round(elapsed, 1),
    )


def assemble_bca_d1(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """BCA with PRECISE strategy (depth=1, min_score=0.3). Shallow expansion."""
    global _last_context_package, _last_retrieval_scores
    start = time.time()
    assembler = ContextAssembler(repo_path, graph, query)
    package = assembler.assemble(task=task, token_budget=budget, strategy=ContextStrategy.PRECISE)
    elapsed = (time.time() - start) * 1000
    _last_context_package = package
    _last_retrieval_scores = _extract_bca_scores(package)
    return (
        package.render(include_metadata=False, include_line_numbers=False),
        package.total_tokens,
        package.symbols_included,
        package.files_included,
        round(elapsed, 1),
    )


def assemble_bca_d5(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """BCA with THOROUGH strategy (depth=5, min_score=0.05). Deep expansion."""
    global _last_context_package, _last_retrieval_scores
    start = time.time()
    assembler = ContextAssembler(repo_path, graph, query)
    package = assembler.assemble(task=task, token_budget=budget, strategy=ContextStrategy.THOROUGH)
    elapsed = (time.time() - start) * 1000
    _last_context_package = package
    _last_retrieval_scores = _extract_bca_scores(package)
    return (
        package.render(include_metadata=False, include_line_numbers=False),
        package.total_tokens,
        package.symbols_included,
        package.files_included,
        round(elapsed, 1),
    )


def assemble_bca_no_closure(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """BCA without dependency closure."""
    global _last_context_package, _last_retrieval_scores
    start = time.time()
    ablation = AblationConfig(dependency_closure=False)
    assembler = ContextAssembler(repo_path, graph, query, ablation=ablation)
    package = assembler.assemble(task=task, token_budget=budget, strategy=ContextStrategy.SMART)
    elapsed = (time.time() - start) * 1000
    _last_context_package = package
    _last_retrieval_scores = _extract_bca_scores(package)
    return (
        package.render(include_metadata=False, include_line_numbers=False),
        package.total_tokens,
        package.symbols_included,
        package.files_included,
        round(elapsed, 1),
    )


def assemble_bm25(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """BM25 symbol retrieval + greedy packing."""
    global _last_retrieval_scores
    start = time.time()
    result = baseline_bm25(repo_path, task, budget, graph)
    _last_retrieval_scores = result.retrieval_scores
    # BM25 gives us symbol names; we need to render their source
    content_parts = []
    for sym_qname in result.selected_symbols:
        for node_id, data in graph.nodes(data=True):
            if data.get("type") != "symbol":
                continue
            qn = data.get("qualified_name", "")
            if qn != sym_qname:
                continue
            fp = data.get("file_path", "")
            line_start = data.get("line_start", 0)
            line_end = data.get("line_end", 0)
            if fp and line_start and line_end:
                full_path = repo_path / fp
                if full_path.exists():
                    try:
                        lines = full_path.read_text(
                            encoding="utf-8", errors="replace"
                        ).splitlines()
                        source = "\n".join(
                            lines[max(0, line_start - 1):line_end]
                        )
                        content_parts.append(f"# {fp}:{line_start}-{line_end}\n{source}")
                    except OSError:
                        pass
            break

    context = "\n\n".join(content_parts)
    elapsed = (time.time() - start) * 1000
    return (
        context,
        result.tokens_used,
        result.symbols_selected,
        result.files_included,
        round(elapsed, 1),
    )


def assemble_grep(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """Full-file grep baseline."""
    global _last_retrieval_scores
    _last_retrieval_scores = []  # Grep has no per-symbol scoring
    start = time.time()
    keywords = set(re.findall(r"\b([A-Za-z_]\w{2,})\b", task))
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "have",
        "fix", "bug", "add", "class", "function", "method", "file",
    }
    keywords -= stop

    matched_content = []
    total_tokens = 0
    files_used = 0

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
            file_tokens = TokenEstimator.estimate(content)
            if total_tokens + file_tokens > budget:
                remaining = budget - total_tokens
                if remaining > 50:
                    chars = int(remaining * TokenEstimator.CHARS_PER_TOKEN)
                    matched_content.append(f"# {fp}\n{content[:chars]}")
                    total_tokens += remaining
                    files_used += 1
                break
            matched_content.append(f"# {fp}\n{content}")
            total_tokens += file_tokens
            files_used += 1

    context = "\n\n".join(matched_content)
    elapsed = (time.time() - start) * 1000
    return (context, total_tokens, 0, files_used, round(elapsed, 1))


def assemble_keyword_map(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """Repo map (compact file tree + relevant symbol source code).

    Mimics aider's approach: provide structural overview for navigation,
    then fill remaining budget with actual source code of relevant symbols.
    The LLM needs real source code to produce valid SEARCH/REPLACE blocks.
    """
    global _last_retrieval_scores
    _last_retrieval_scores = []  # Repo map has no per-symbol scoring
    start = time.time()

    # Phase 1: Build compact file tree (just paths, ~1 token per line)
    files_by_path: dict[str, list[dict]] = {}
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "symbol":
            continue
        fp = data.get("file_path", "")
        if fp:
            files_by_path.setdefault(fp, []).append(data)

    tree_lines = ["# Repository structure"]
    for fp in sorted(files_by_path):
        tree_lines.append(fp)

    file_tree = "\n".join(tree_lines)
    tree_tokens = TokenEstimator.estimate(file_tree)
    if tree_tokens > budget // 4:
        # Cap tree at 25% of budget
        chars_allowed = int((budget // 4) * TokenEstimator.CHARS_PER_TOKEN)
        file_tree = file_tree[:chars_allowed]
        tree_tokens = budget // 4

    # Phase 2: Fill remaining budget with actual source code of relevant symbols
    remaining = budget - tree_tokens
    keywords = set(re.findall(r"\b([A-Za-z_]\w{2,})\b", task))
    stop = {"the", "and", "for", "that", "this", "with", "from", "fix", "bug", "add"}
    keywords -= stop

    # Score symbols by keyword relevance
    scored_symbols: list[tuple[int, dict]] = []
    for fp, syms in files_by_path.items():
        for sym in syms:
            name = sym.get("name", "").lower()
            qname = sym.get("qualified_name", "").lower()
            doc = sym.get("docstring", "").lower()
            sig = sym.get("signature", "").lower()
            text = f"{name} {qname} {doc} {sig}"
            hits = sum(1 for kw in keywords if kw.lower() in text)
            if hits > 0:
                scored_symbols.append((hits, sym))

    scored_symbols.sort(key=lambda x: x[0], reverse=True)

    # Greedy pack source code from highest-relevance symbols
    source_parts = []
    content_tokens = 0
    files_used = set()
    syms_selected = 0

    for _score, sym in scored_symbols:
        fp = sym.get("file_path", "")
        line_start = sym.get("line_start", 0)
        line_end = sym.get("line_end", 0)
        if not fp or not line_start or not line_end:
            continue
        line_count = max(1, line_end - line_start + 1)
        cost = TokenEstimator.estimate_lines(line_count)
        if content_tokens + cost > remaining:
            continue
        full_path = repo_path / fp
        if not full_path.exists():
            continue
        try:
            lines = full_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            source = "\n".join(lines[max(0, line_start - 1):line_end])
            source_parts.append(f"# {fp}:{line_start}-{line_end}\n{source}")
            content_tokens += cost
            syms_selected += 1
            files_used.add(fp)
        except OSError:
            continue

    context_parts = [file_tree]
    if source_parts:
        context_parts.append("\n# Relevant source code\n")
        context_parts.append("\n\n".join(source_parts))
    context = "\n\n".join(context_parts)

    elapsed = (time.time() - start) * 1000
    return (
        context,
        tree_tokens + content_tokens,
        syms_selected,
        len(files_used),
        round(elapsed, 1),
    )


def assemble_vector(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """Vector retrieval baseline (TF-IDF + cosine similarity).

    Uses TF-IDF for zero-dependency reproducibility. For dense embeddings,
    install sentence-transformers and set BENCHMARK_USE_DENSE=1.
    """
    global _last_retrieval_scores
    start = time.time()

    # Collect symbol documents
    symbols = []
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "symbol":
            continue
        name = data.get("name", "")
        qname = data.get("qualified_name", "")
        doc = data.get("docstring", "")
        sig = data.get("signature", "")
        text = f"{name} {qname} {doc} {sig}"
        symbols.append({"id": node_id, "text": text, "data": data})

    if not symbols:
        _last_retrieval_scores = []
        elapsed = (time.time() - start) * 1000
        return ("", 0, 0, 0, round(elapsed, 1))

    use_dense = os.environ.get("BENCHMARK_USE_DENSE", "")

    if use_dense:
        scores = _vector_score_dense(task, symbols)
    else:
        scores = _vector_score_tfidf(task, symbols)

    # Capture positive scores for Router B confidence features
    _last_retrieval_scores = sorted([s for s in scores if s > 0], reverse=True)[:50]

    # Sort by score descending, greedy pack
    scored = sorted(zip(scores, symbols), key=lambda x: x[0], reverse=True)

    selected_parts = []
    tokens_used = 0
    files = set()
    syms_selected = 0

    for score, sym in scored:
        if score <= 0:
            continue
        data = sym["data"]
        fp = data.get("file_path", "")
        line_start = data.get("line_start", 0)
        line_end = data.get("line_end", 0)
        line_count = max(1, line_end - line_start + 1)
        cost = TokenEstimator.estimate_lines(line_count)
        if tokens_used + cost > budget:
            continue

        # Load source
        if fp and line_start and line_end:
            full_path = repo_path / fp
            if full_path.exists():
                try:
                    lines = full_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    source = "\n".join(lines[max(0, line_start - 1):line_end])
                    selected_parts.append(f"# {fp}:{line_start}-{line_end}\n{source}")
                    tokens_used += cost
                    syms_selected += 1
                    if fp:
                        files.add(fp)
                except OSError:
                    pass

    context = "\n\n".join(selected_parts)
    elapsed = (time.time() - start) * 1000
    return (context, tokens_used, syms_selected, len(files), round(elapsed, 1))


def _vector_score_tfidf(query: str, symbols: list[dict]) -> list[float]:
    """TF-IDF cosine similarity scoring."""
    import math
    from collections import Counter

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    query_tokens = tokenize(query)
    doc_tokens = [tokenize(s["text"]) for s in symbols]

    # Build vocabulary
    all_docs = doc_tokens + [query_tokens]
    doc_freq: Counter[str] = Counter()
    for doc in all_docs:
        for token in set(doc):
            doc_freq[token] += 1

    n = len(all_docs)

    def tfidf_vector(tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {}
        for term, count in tf.items():
            idf = math.log((n + 1) / (doc_freq.get(term, 0) + 1)) + 1
            vec[term] = count * idf
        return vec

    def cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    query_vec = tfidf_vector(query_tokens)
    return [cosine_sim(query_vec, tfidf_vector(dt)) for dt in doc_tokens]


def _vector_score_dense(query: str, symbols: list[dict]) -> list[float]:
    """Dense embedding scoring using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers not installed, falling back to TF-IDF")
        return _vector_score_tfidf(query, symbols)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [s["text"] for s in symbols]
    query_emb = model.encode([query], normalize_embeddings=True)
    doc_embs = model.encode(texts, normalize_embeddings=True, batch_size=64)
    scores = (doc_embs @ query_emb.T).flatten()
    return scores.tolist()


def assemble_embedding(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """OpenAI embedding baseline (text-embedding-3-small)."""
    global _last_retrieval_scores
    start = time.time()

    # Collect symbol documents
    symbols = []
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "symbol":
            continue
        name = data.get("name", "")
        qname = data.get("qualified_name", "")
        doc = data.get("docstring", "")
        sig = data.get("signature", "")
        text = f"{name} {qname} {doc} {sig}".strip()
        if text:
            symbols.append({"id": node_id, "text": text, "data": data})

    if not symbols:
        _last_retrieval_scores = []
        elapsed = (time.time() - start) * 1000
        return ("", 0, 0, 0, round(elapsed, 1))

    scores = _embedding_score_openai(task, symbols, cache_key=str(repo_path))

    # Capture positive scores for Router B confidence features
    _last_retrieval_scores = sorted([s for s in scores if s > 0], reverse=True)[:50]

    # Sort by score descending, greedy pack
    scored = sorted(zip(scores, symbols), key=lambda x: x[0], reverse=True)

    selected_parts = []
    tokens_used = 0
    files = set()
    syms_selected = 0

    for score, sym in scored:
        if score <= 0:
            continue
        data = sym["data"]
        fp = data.get("file_path", "")
        line_start = data.get("line_start", 0)
        line_end = data.get("line_end", 0)
        line_count = max(1, line_end - line_start + 1)
        cost = TokenEstimator.estimate_lines(line_count)
        if tokens_used + cost > budget:
            continue

        if fp and line_start and line_end:
            full_path = repo_path / fp
            if full_path.exists():
                try:
                    lines = full_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    source = "\n".join(lines[max(0, line_start - 1):line_end])
                    selected_parts.append(f"# {fp}:{line_start}-{line_end}\n{source}")
                    tokens_used += cost
                    syms_selected += 1
                    if fp:
                        files.add(fp)
                except OSError:
                    pass

    context = "\n\n".join(selected_parts)
    elapsed = (time.time() - start) * 1000
    return (context, tokens_used, syms_selected, len(files), round(elapsed, 1))


# Corpus embedding cache: repo_path → (symbol_ids, numpy matrix)
_corpus_embedding_cache: dict[str, tuple[list[str], object]] = {}


def _embedding_score_openai(
    query_text: str, symbols: list[dict], *, cache_key: str = "",
) -> list[float]:
    """Score symbols using OpenAI text-embedding-3-small.

    When *cache_key* is provided (typically str(repo_path)), corpus embeddings
    are computed once and reused for subsequent queries against the same repo.
    Only the query embedding is re-computed each call.  The cache stores
    symbol IDs alongside embeddings and verifies alignment on every hit;
    a mismatch triggers a fresh embedding.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("  WARNING: OPENAI_API_KEY not set, falling back to TF-IDF for embedding")
        return _vector_score_tfidf(query_text, symbols)

    try:
        import numpy as np
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        current_ids = [s["id"] for s in symbols]

        # --- Corpus embeddings (cached per repo, verified by symbol IDs) ---
        cache_hit = False
        if cache_key and cache_key in _corpus_embedding_cache:
            cached_ids, corpus_matrix = _corpus_embedding_cache[cache_key]
            if cached_ids == current_ids:
                cache_hit = True
            else:
                print("    [embedding] Symbol order changed — re-embedding corpus")

        if not cache_hit:
            texts = [s["text"] for s in symbols]
            raw_embeddings: list[list[float]] = []
            batch_size = 512
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                )
                raw_embeddings.extend([d.embedding for d in resp.data])
            # Store as numpy matrix (n_symbols × dim)
            corpus_matrix = np.array(raw_embeddings, dtype=np.float32)
            if cache_key:
                _corpus_embedding_cache[cache_key] = (current_ids, corpus_matrix)
                print(
                    f"    [embedding] Cached {corpus_matrix.shape[0]} corpus "
                    f"embeddings ({corpus_matrix.shape[1]}d)"
                )

        # --- Query embedding (always fresh) ---
        q_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query_text],
        )
        q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)

        # --- Cosine similarity via dot product ---
        # OpenAI embeddings are L2-normalized, so dot product = cosine sim
        scores = (corpus_matrix @ q_vec).tolist()
        return scores

    except Exception as e:
        print(f"  WARNING: OpenAI embedding failed ({e}), falling back to TF-IDF")
        return _vector_score_tfidf(query_text, symbols)


def assemble_no_retrieval(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """No-retrieval baseline: LLM gets only the bug description, no code context.

    Proves retrieval matters at all. If this scores well, retrieval adds
    little value. If near zero, retrieval is carrying the run.
    """
    global _last_retrieval_scores
    _last_retrieval_scores = []
    start = time.time()
    context = "(No code context provided. Fix the bug based on the description alone.)"
    tokens = TokenEstimator.estimate(context)
    elapsed = (time.time() - start) * 1000
    return (context, tokens, 0, 0, round(elapsed, 1))


def assemble_naive_random(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
    *, _task_id: str = "", _query_type: str = "",
) -> tuple[str, int, int, int, float]:
    """Random context baseline: random symbols packed up to budget.

    Proves the win is "right context," not "any context." A method
    that can't beat random symbol selection isn't doing useful retrieval.

    Uses the same symbol universe (same graph indexing, same token estimator,
    same budget rule) as other symbol-based methods. Seed is derived from
    (task_id, budget, query_type) for exact reproducibility.
    """
    global _last_retrieval_scores
    _last_retrieval_scores = []  # Random has no scoring
    start = time.time()
    # Deterministic seed from (task_id, budget, query_type) — uses hashlib, not hash(),
    # because hash() is randomized per-process unless PYTHONHASHSEED=0
    import hashlib
    seed_str = f"{_task_id}:{budget}:{_query_type}" if _task_id else task
    rng = random.Random(int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32))

    # Collect all symbols with valid source ranges
    symbols = []
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "symbol":
            continue
        fp = data.get("file_path", "")
        line_start = data.get("line_start", 0)
        line_end = data.get("line_end", 0)
        if fp and line_start and line_end:
            symbols.append(data)

    rng.shuffle(symbols)

    # Greedy pack random symbols up to budget
    parts = []
    tokens_used = 0
    files: set[str] = set()
    syms = 0

    for data in symbols:
        fp = data.get("file_path", "")
        line_start = data.get("line_start", 0)
        line_end = data.get("line_end", 0)
        line_count = max(1, line_end - line_start + 1)
        cost = TokenEstimator.estimate_lines(line_count)
        if tokens_used + cost > budget:
            continue
        full_path = repo_path / fp
        if not full_path.exists():
            continue
        try:
            lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
            source = "\n".join(lines[max(0, line_start - 1):line_end])
            parts.append(f"# {fp}:{line_start}-{line_end}\n{source}")
            tokens_used += cost
            syms += 1
            files.add(fp)
        except OSError:
            continue

    context = "\n\n".join(parts)
    elapsed = (time.time() - start) * 1000
    return (context, tokens_used, syms, len(files), round(elapsed, 1))


def assemble_target_file(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
    mutation: dict | None = None,
) -> tuple[str, int, int, int, float]:
    """Privileged ceiling: LLM gets the exact file containing the bug.

    NOT a retrieval method — this is an upper bound probe that separates
    "retrieval failure" from "LLM cannot repair even with the right file."
    Budget caps how much of the file to include.
    """
    global _last_retrieval_scores
    _last_retrieval_scores = []  # Ceiling method, no scoring
    start = time.time()

    if not mutation or "file" not in mutation:
        elapsed = (time.time() - start) * 1000
        return ("", 0, 0, 0, round(elapsed, 1))

    fp = mutation["file"]
    full_path = repo_path / fp
    if not full_path.exists():
        elapsed = (time.time() - start) * 1000
        return ("", 0, 0, 0, round(elapsed, 1))

    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        elapsed = (time.time() - start) * 1000
        return ("", 0, 0, 0, round(elapsed, 1))

    file_tokens = TokenEstimator.estimate(content)
    if file_tokens > budget:
        # Truncate to budget
        chars = int(budget * TokenEstimator.CHARS_PER_TOKEN)
        content = content[:chars]
        file_tokens = budget

    context = f"# {fp}\n{content}"
    elapsed = (time.time() - start) * 1000
    return (context, file_tokens, 0, 1, round(elapsed, 1))


def assemble_bca_no_scoring(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """BCA with scoring disabled — graph traversal + closure only."""
    global _last_context_package, _last_retrieval_scores
    start = time.time()
    ablation = AblationConfig(
        centrality_scoring=False,
        file_proximity=False,
        kind_weights=False,
        submodular_coverage=False,
    )
    assembler = ContextAssembler(repo_path, graph, query, ablation=ablation)
    package = assembler.assemble(task=task, token_budget=budget, strategy=ContextStrategy.SMART)
    elapsed = (time.time() - start) * 1000
    _last_context_package = package
    _last_retrieval_scores = _extract_bca_scores(package)
    return (
        package.render(include_metadata=False, include_line_numbers=False),
        package.total_tokens,
        package.symbols_included,
        package.files_included,
        round(elapsed, 1),
    )


# All assembly methods.
# Note: target_file has extra signature (mutation=) — handled specially in _run_single_eval.
METHODS: dict[str, callable] = {
    "no_retrieval": assemble_no_retrieval,
    "naive_random": assemble_naive_random,
    "grep": assemble_grep,
    "bm25": assemble_bm25,
    "keyword_map": assemble_keyword_map,
    "vector": assemble_vector,
    "embedding": assemble_embedding,
    "bca_d1": assemble_bca_d1,
    "bca": assemble_bca,
    "bca_d5": assemble_bca_d5,
    "bca_no_closure": assemble_bca_no_closure,
    "bca_no_scoring": assemble_bca_no_scoring,
    "target_file": assemble_target_file,
}


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a coding assistant. You will be given source code context "
    "and a bug description.\n"
    "Your job is to produce a SEARCH/REPLACE edit that fixes the bug.\n\n"
    "Output format — one or more blocks like this:\n\n"
    "```\n"
    "FILE: path/to/file.py\n"
    "SEARCH:\n"
    "<exact lines from the buggy file>\n"
    "REPLACE:\n"
    "<corrected lines>\n"
    "```\n\n"
    "Rules:\n"
    "- The FILE path must be the EXACT path shown in the context\n"
    "  (e.g. 'src/cegraph/foo.py', not 'cegraph/foo.py').\n"
    "- The SEARCH block must contain the EXACT lines from the file\n"
    "  (copy-paste, including indentation).\n"
    "- Do NOT include line numbers or prefixes in SEARCH/REPLACE.\n"
    "- Only change the minimum lines needed to fix the bug.\n"
    "- You may output multiple FILE/SEARCH/REPLACE blocks if the fix\n"
    "  spans multiple files.\n"
    "- Output ONLY the edit blocks, no explanation before or after."
)


def build_prompt(context: str, task: str) -> str:
    """Build the user prompt from assembled context and task."""
    return f"""## Source Code Context

{context}

## Bug Report

{task}

## Fix

Produce SEARCH/REPLACE blocks to fix this bug."""


_RETRY_DELAYS = [5, 8, 15, 30, 60]  # deterministic backoff: 5s, 8s, 15s, 30s, 60s, then fail

# Model version tracking — populated on first LLM call for reproducibility logging.
# Stores the model ID and any fingerprint returned by the provider.
_model_version_info: dict[str, str] = {}


async def call_llm(
    provider, context: str, task: str,
) -> tuple[str, float, int, int]:
    """Call the LLM with retry on rate limits.

    Returns (response_text, time_ms, input_tokens, output_tokens).
    Retries with fixed delays on 429/rate-limit errors, then skips.
    """
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=build_prompt(context, task)),
    ]

    last_error = None
    max_attempts = len(_RETRY_DELAYS) + 1
    for attempt in range(max_attempts):
        if attempt > 0:
            wait = _RETRY_DELAYS[attempt - 1]
            print(f"    retry {attempt}/{len(_RETRY_DELAYS)} after {wait}s...")
            await asyncio.sleep(wait)

        try:
            start = time.time()
            response: LLMResponse = await provider.complete(
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
                seed=42,
            )
            elapsed = (time.time() - start) * 1000

            input_tokens = (
                response.usage.get("prompt_tokens", 0)
                or response.usage.get("input_tokens", 0)
            )
            output_tokens = (
                response.usage.get("completion_tokens", 0)
                or response.usage.get("output_tokens", 0)
            )

            # Capture model version info on first successful call
            if not _model_version_info:
                if response.system_fingerprint:
                    _model_version_info["system_fingerprint"] = response.system_fingerprint
                # Log it once
                if _model_version_info:
                    print(f"  Model version: {_model_version_info}")

            return (response.content, round(elapsed, 1), input_tokens, output_tokens)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                last_error = e
                continue
            raise  # Non-rate-limit errors propagate immediately

    raise last_error  # All retries exhausted


# ---------------------------------------------------------------------------
# Patch application and testing
# ---------------------------------------------------------------------------

@dataclass
class SearchReplaceEdit:
    """A single search/replace edit."""
    file_path: str
    search: str
    replace: str


def _strip_line_prefixes(text: str) -> str:
    """Strip line-number prefixes like '  13 | ' that the LLM copies from context."""
    lines = text.splitlines()
    # Check if most lines have the pattern: optional_spaces digits space pipe space
    prefix_re = re.compile(r"^\s*\d+\s*\|\s?")
    has_prefix = sum(1 for ln in lines if prefix_re.match(ln) or ln.strip() == "")
    if has_prefix >= len(lines) * 0.5 and len(lines) > 0:
        stripped = []
        for ln in lines:
            m = prefix_re.match(ln)
            if m:
                stripped.append(ln[m.end():])
            else:
                stripped.append(ln)
        return "\n".join(stripped)
    return text


_FILE_LINE_RE = re.compile(
    r"^([a-zA-Z0-9_./-]+\.\w+)(?::(\d+)(?:-(\d+))?)?$"
)


def _parse_file_marker(line: str) -> str | None:
    """Try to parse a file path from various LLM output formats.

    Handles:
      FILE: path/to/file.py
      path/to/file.py:10-20
      path/to/file.py
    """
    stripped = line.strip()
    if stripped.startswith("FILE:"):
        return stripped[5:].strip()
    m = _FILE_LINE_RE.match(stripped)
    if m and "/" in m.group(1):
        return m.group(1)
    return None


def extract_edits(llm_output: str) -> list[SearchReplaceEdit]:
    """Extract SEARCH/REPLACE edit blocks from LLM output."""
    edits: list[SearchReplaceEdit] = []

    # Strip code fences if present
    text = llm_output
    if "```" in text:
        parts = text.split("```")
        text = "\n".join(parts[1::2]) if len(parts) > 1 else text

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for FILE: marker or path:line pattern
        file_path = _parse_file_marker(line)
        if file_path:
            search_lines: list[str] = []
            replace_lines: list[str] = []
            i += 1

            # Find SEARCH: (or treat lines before REPLACE: as search)
            has_search = False
            while i < len(lines):
                s = lines[i].strip()
                if s.startswith("SEARCH:"):
                    has_search = True
                    i += 1
                    break
                if s.startswith("REPLACE:"):
                    break
                i += 1

            # Collect search content until REPLACE:
            if has_search:
                while i < len(lines) and not lines[i].strip().startswith("REPLACE:"):
                    search_lines.append(lines[i])
                    i += 1
            else:
                # No SEARCH: marker — use lines before REPLACE: as search
                # (some LLMs output path\ncode\nREPLACE:\ncode)
                j = i
                while j < len(lines) and not lines[j].strip().startswith("REPLACE:"):
                    search_lines.append(lines[j])
                    j += 1
                i = j

            if i < len(lines) and lines[i].strip().startswith("REPLACE:"):
                i += 1  # skip REPLACE: line

            # Collect replace content until next file marker or end
            while i < len(lines):
                stripped = lines[i].strip()
                if _parse_file_marker(stripped) or stripped == "```":
                    break
                replace_lines.append(lines[i])
                i += 1

            search_text = "\n".join(search_lines)
            replace_text = "\n".join(replace_lines)

            search_text = _strip_line_prefixes(search_text)
            replace_text = _strip_line_prefixes(replace_text)

            search_text = search_text.rstrip("\n")
            replace_text = replace_text.rstrip("\n")

            if search_text.strip():
                edits.append(SearchReplaceEdit(
                    file_path=file_path,
                    search=search_text,
                    replace=replace_text,
                ))
        else:
            i += 1

    return edits


def extract_patch(llm_output: str) -> str:
    """Extract patch info from LLM output. Returns raw text for logging."""
    edits = extract_edits(llm_output)
    if not edits:
        return ""
    parts = []
    for e in edits:
        parts.append(f"FILE: {e.file_path}")
        parts.append(f"SEARCH:\n{e.search}")
        parts.append(f"REPLACE:\n{e.replace}")
        parts.append("")
    return "\n".join(parts)


def _resolve_file_path(work_dir: Path, fp: str) -> Path | None:
    """Try to find a file when the LLM outputs a wrong path.

    Common issue: LLM writes 'cegraph/foo.py' instead of 'src/cegraph/foo.py'.
    """
    # Try adding common prefixes
    for prefix in ("src/", "lib/", "pkg/"):
        candidate = work_dir / prefix / fp
        if candidate.exists():
            return candidate

    # Try matching by filename anywhere in the tree
    name = Path(fp).name
    matches = list(work_dir.rglob(name))
    # Filter to matches whose path ends with the given fp
    for m in matches:
        try:
            rel = m.relative_to(work_dir)
            if str(rel).endswith(fp):
                return m
        except ValueError:
            continue
    return None


def _apply_edits_to_dir(
    work_dir: Path, edits: list[SearchReplaceEdit],
) -> tuple[int, str]:
    """Apply search/replace edits to files in work_dir.

    Returns (applied_count, error_msg). error_msg is empty on success.
    """
    applied = 0
    for edit in edits:
        fp = edit.file_path
        for prefix in ("a/", "b/"):
            if fp.startswith(prefix):
                fp = fp[len(prefix):]

        target = work_dir / fp
        if not target.exists():
            resolved = _resolve_file_path(work_dir, fp)
            if resolved:
                target = resolved
            else:
                return 0, f"file not found: {fp}"

        content = target.read_text()
        if edit.search in content:
            content = content.replace(edit.search, edit.replace, 1)
            target.write_text(content)
            applied += 1
        else:
            search_norm = "\n".join(s.rstrip() for s in edit.search.splitlines())
            content_norm = "\n".join(s.rstrip() for s in content.splitlines())
            if search_norm in content_norm:
                content = content_norm.replace(search_norm, edit.replace, 1)
                target.write_text(content)
                applied += 1
            else:
                return 0, f"search text not found in {fp}"

    return applied, ""


def apply_and_test(
    repo_path: Path,
    llm_output: str,
    test_cmd: str,
    timeout: int = 60,
    in_place: bool = False,
) -> tuple[bool, str]:
    """Apply LLM edits and run tests. Supports two modes:

    - in_place=False (default): copy repo to tmpdir, apply edits, test there.
      Good for PYTHONPATH-based projects (like CeGraph).
    - in_place=True: apply edits directly to repo_path, run tests, then revert.
      Required for editable installs (pip install -e) where the installed package
      IS the source directory (e.g. pydantic-ai).
    """
    edits = extract_edits(llm_output)
    if not edits:
        return False, "no edits extracted"

    if in_place:
        return _apply_and_test_in_place(repo_path, edits, test_cmd, timeout)
    else:
        return _apply_and_test_copy(repo_path, edits, test_cmd, timeout)


def _apply_and_test_in_place(
    repo_path: Path,
    edits: list[SearchReplaceEdit],
    test_cmd: str,
    timeout: int,
) -> tuple[bool, str]:
    """Apply edits in-place, run tests, then revert all changes.

    Includes byte-identical revert verification: after reverting, we hash
    the file contents and compare against pre-edit hashes to guarantee
    the repo is restored to its exact prior state.
    """
    # Save originals for revert — store both content and SHA-256 hash
    originals: dict[Path, str] = {}
    original_hashes: dict[Path, str] = {}
    for edit in edits:
        fp = edit.file_path
        for prefix in ("a/", "b/"):
            if fp.startswith(prefix):
                fp = fp[len(prefix):]
        target = repo_path / fp
        if not target.exists():
            resolved = _resolve_file_path(repo_path, fp)
            if resolved:
                target = resolved
            else:
                return False, f"file not found: {fp}"
        if target not in originals:
            raw_bytes = target.read_bytes()
            originals[target] = raw_bytes.decode("utf-8")
            original_hashes[target] = hashlib.sha256(raw_bytes).hexdigest()

    try:
        applied, err = _apply_edits_to_dir(repo_path, edits)
        if err:
            return False, err
        if applied == 0:
            return False, "no edits applied"

        # Run targeted test
        try:
            test_result = subprocess.run(
                test_cmd.split(),
                capture_output=True,
                text=True,
                cwd=repo_path,
                timeout=timeout,
            )
            passed = test_result.returncode == 0
            output = test_result.stdout[-2000:] + "\n" + test_result.stderr[-2000:]
            if not passed:
                return False, output.strip()
            return True, output.strip()
        except subprocess.TimeoutExpired:
            return False, "test timeout"
        except Exception as e:
            return False, f"test error: {e}"
    finally:
        # Always revert, then verify byte-identical restoration
        for target, content in originals.items():
            target.write_text(content)

        # Verify byte-identical revert
        for target, expected_hash in original_hashes.items():
            actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                print(
                    f"  REVERT ERROR: {target.name} hash mismatch! "
                    f"expected={expected_hash[:12]}, got={actual_hash[:12]}"
                )
                # Force-write from saved content to guarantee restoration
                target.write_text(originals[target])
                final_hash = hashlib.sha256(target.read_bytes()).hexdigest()
                if final_hash != expected_hash:
                    raise RuntimeError(
                        f"FATAL: Cannot restore {target} to original state. "
                        f"This is a data integrity violation."
                    )


def _apply_and_test_copy(
    repo_path: Path,
    edits: list[SearchReplaceEdit],
    test_cmd: str,
    timeout: int,
) -> tuple[bool, str]:
    """Copy repo to tmpdir, apply edits, run tests."""
    with tempfile.TemporaryDirectory(prefix="bca_eval_") as tmpdir:
        work_dir = Path(tmpdir) / "repo"
        shutil.copytree(
            repo_path, work_dir,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", ".cegraph",
                "node_modules", ".venv", "venv",
            ),
        )

        applied, err = _apply_edits_to_dir(work_dir, edits)
        if err:
            return False, err
        if applied == 0:
            return False, "no edits applied"

        # Run targeted test
        env = {**os.environ, "PYTHONPATH": str(work_dir / "src")}
        try:
            test_result = subprocess.run(
                test_cmd.split(),
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=timeout,
                env=env,
            )
            passed = test_result.returncode == 0
            output = test_result.stdout[-2000:] + "\n" + test_result.stderr[-2000:]

            if not passed:
                return False, output.strip()

            # Regression check: run full test suite
            regression_result = subprocess.run(
                ["python", "-m", "pytest", "tests/", "-x", "-q"],
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=timeout * 2,
                env=env,
            )
            if regression_result.returncode != 0:
                output += "\n[REGRESSION] Full test suite failed:\n"
                output += regression_result.stdout[-1000:] + "\n" + regression_result.stderr[-500:]
                return False, output.strip()

            return True, output.strip()
        except subprocess.TimeoutExpired:
            return False, "test timeout"
        except Exception as e:
            return False, f"test error: {e}"


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def _apply_mutation(repo_path: Path, mutation: dict) -> tuple[str, str] | None:
    """Apply a mutation to the repo. Returns (original_content, sha256_hash) for restoration.

    Uses line_num anchor when available for precise targeting.
    Falls back to first-match replacement with ambiguity warning.
    """
    if not mutation:
        return None
    file_path = repo_path / mutation["file"]
    if not file_path.exists():
        return None
    raw_bytes = file_path.read_bytes()
    original = raw_bytes.decode("utf-8")
    original_hash = hashlib.sha256(raw_bytes).hexdigest()
    mut_original = mutation["original"]
    mut_replacement = mutation["mutated"]

    # Line-anchored replacement (preferred)
    line_num = mutation.get("line_num")
    if line_num and "\n" not in mut_original:
        lines = original.splitlines()
        line_idx = line_num - 1
        if 0 <= line_idx < len(lines) and mut_original.strip() in lines[line_idx]:
            lines[line_idx] = lines[line_idx].replace(mut_original, mut_replacement, 1)
            mutated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
            file_path.write_text(mutated)
            return (original, original_hash)

    # Fallback: first-match with ambiguity warning
    occurrences = original.count(mut_original)
    if occurrences > 1:
        print(f"  WARNING: mutation snippet appears {occurrences} times in {mutation['file']}")
    if occurrences == 0:
        return None  # mutation string not found

    mutated = original.replace(mut_original, mut_replacement, 1)
    file_path.write_text(mutated)
    return (original, original_hash)


def _restore_mutation(repo_path: Path, mutation: dict, original: str, expected_hash: str) -> None:
    """Restore original content after mutation, with byte-identical verification."""
    file_path = repo_path / mutation["file"]
    file_path.write_text(original)

    # Verify byte-identical restoration
    actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"FATAL: Mutation revert failed for {mutation['file']}. "
            f"expected={expected_hash[:12]}, got={actual_hash[:12]}. "
            f"Repo state may be corrupted."
        )


# ---------------------------------------------------------------------------
# Pre-launch metric helpers
# ---------------------------------------------------------------------------

def _extract_entities_standalone(task_text: str) -> list[dict]:
    """Extract code entities from query text (mirrors ContextAssembler._extract_entities).

    Returns list of {"name": ..., "type": ..., "confidence": ...}.
    Used to log entity metrics for ALL methods, not just BCA.
    """
    entities: list[dict] = []
    seen: set[str] = set()

    # CamelCase
    for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", task_text):
        name = m.group(1)
        if name not in seen:
            entities.append({"name": name, "type": "class", "confidence": 0.9})
            seen.add(name)

    # snake_case
    for m in re.finditer(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", task_text):
        name = m.group(1)
        skip = {"the_", "in_the", "to_the", "of_the", "and_the", "is_a", "has_a"}
        if name not in seen and not any(name.startswith(s) for s in skip):
            entities.append({"name": name, "type": "function", "confidence": 0.8})
            seen.add(name)

    # Dotted paths
    for m in re.finditer(r"\b(\w+\.\w+(?:\.\w+)*)\b", task_text):
        name = m.group(1)
        if name not in seen and not name[0].isdigit():
            entities.append({"name": name, "type": "path", "confidence": 0.85})
            seen.add(name)

    # File paths
    for m in re.finditer(r"\b([\w/]+\.(?:py|js|ts|go|rs|java))\b", task_text):
        name = m.group(1)
        if name not in seen:
            entities.append({"name": name, "type": "file", "confidence": 0.95})
            seen.add(name)

    # Quoted strings
    for m in re.finditer(r"[`'\"](\w+(?:\.\w+)*)[`'\"]", task_text):
        name = m.group(1)
        if name not in seen and len(name) > 1:
            entities.append({"name": name, "type": "quoted", "confidence": 0.9})
            seen.add(name)

    return entities


def _resolve_entities_to_seeds(entities: list[dict], graph, query: GraphQuery) -> list[str]:
    """Resolve extracted entities to graph symbol IDs. Returns list of symbol IDs."""
    seed_ids: list[str] = []
    seen: set[str] = set()

    # Build a basename→file_node index for fuzzy file matching
    file_basename_map: dict[str, list[str]] = {}
    for nid in graph.nodes:
        nid_str = str(nid)
        if nid_str.startswith("file::"):
            basename = Path(nid_str.removeprefix("file::")).name
            file_basename_map.setdefault(basename, []).append(nid_str)

    def _try_file_lookup(name: str) -> None:
        """Try to resolve a name as a file reference."""
        # Exact match
        file_node = f"file::{name}"
        if graph.has_node(file_node):
            for succ in graph.successors(file_node):
                data = graph.nodes.get(succ, {})
                if data.get("type") == "symbol" and succ not in seen:
                    seed_ids.append(succ)
                    seen.add(succ)
            return
        # Basename match (e.g. "_decoders.py" → "httpx/_decoders.py")
        basename = Path(name).name
        for fn in file_basename_map.get(basename, []):
            if graph.has_node(fn):
                for succ in graph.successors(fn):
                    data = graph.nodes.get(succ, {})
                    if data.get("type") == "symbol" and succ not in seen:
                        seed_ids.append(succ)
                        seen.add(succ)

    _file_suffixes = {".py", ".js", ".ts", ".go", ".rs", ".java"}

    for entity in entities:
        name = entity["name"]
        is_file_like = any(name.endswith(ext) for ext in _file_suffixes)

        if entity["type"] == "file" or is_file_like:
            _try_file_lookup(name)
        else:
            matches = query.find_symbol(name)
            for sid in matches:
                if sid in seen:
                    continue
                data = graph.nodes.get(sid, {})
                if data.get("type") != "symbol":
                    continue
                seed_ids.append(sid)
                seen.add(sid)

    return seed_ids


def _find_mutation_symbol(graph, mutation: dict) -> dict:
    """Find the graph symbol containing the mutated line.

    Returns dict with keys: key, lines, kind, file_symbols.
    All empty/0 if not found.
    """
    empty = {"key": "", "lines": 0, "kind": "", "file_symbols": 0}
    if not mutation or "file" not in mutation or "line_num" not in mutation:
        return empty

    mut_file = mutation["file"]
    mut_line = mutation["line_num"]
    best_id = ""
    best_span = float("inf")
    best_kind = ""
    file_symbols = 0

    for nid, ndata in graph.nodes(data=True):
        if ndata.get("kind") == "file" or ndata.get("type") == "file":
            continue
        npath = ndata.get("file_path", "") or ndata.get("path", "")
        if not npath and "::" in str(nid):
            npath = str(nid).split("::")[0]
        if not npath:
            continue
        if not (npath.endswith(mut_file) or mut_file.endswith(npath)):
            continue

        # Count all symbols in this file
        file_symbols += 1

        nstart = ndata.get("line_start", 0) or ndata.get("start_line", 0)
        nend = ndata.get("line_end", 0) or ndata.get("end_line", 0)
        if nstart and nend and nstart <= mut_line <= nend:
            span = nend - nstart
            if span < best_span:
                best_span = span
                best_id = str(nid)
                best_kind = ndata.get("kind", "")

    if not best_id:
        return {"key": "", "lines": 0, "kind": "", "file_symbols": file_symbols}

    return {
        "key": best_id,
        "lines": best_span + 1,  # inclusive line count
        "kind": best_kind,
        "file_symbols": file_symbols,
    }


def _compute_graph_hops(graph, seed_ids: list[str], target_id: str) -> tuple[int, float]:
    """Compute min and median shortest-path hops from seed symbols to target symbol.

    Uses undirected BFS (ignoring edge direction) since we care about structural distance.
    Returns (min_hops, median_hops). Both are -1 if unreachable or no seeds.
    """
    if not seed_ids or not target_id:
        return -1, -1.0

    undirected = graph.to_undirected()
    distances: list[int] = []

    for sid in seed_ids:
        if not undirected.has_node(sid) or not undirected.has_node(target_id):
            continue
        try:
            d = nx.shortest_path_length(undirected, sid, target_id)
            distances.append(d)
        except nx.NetworkXNoPath:
            continue

    if not distances:
        return -1, -1.0

    distances.sort()
    min_hops = distances[0]
    n = len(distances)
    if n % 2 == 1:
        median_hops = float(distances[n // 2])
    else:
        median_hops = (distances[n // 2 - 1] + distances[n // 2]) / 2.0

    return min_hops, median_hops


def _compute_query_identifier_density(task_text: str) -> float:
    """Fraction of query tokens that look like code identifiers (CamelCase, snake_case, dotted)."""
    tokens = task_text.split()
    if not tokens:
        return 0.0
    identifier_pattern = re.compile(
        r"^[A-Z][a-z]+(?:[A-Z][a-z]+)+$"  # CamelCase
        r"|^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$"  # snake_case
        r"|^\w+\.\w+(?:\.\w+)*$"  # dotted path
        r"|^[\w/]+\.(?:py|js|ts|go|rs|java)$"  # file path
    )
    id_count = sum(1 for t in tokens if identifier_pattern.match(t.strip("\"'`.,;:()")))
    return round(id_count / len(tokens), 4)


def _extract_context_symbol_keys(context: str, graph) -> list[str]:
    """Extract symbol IDs present in context text.

    Checks only symbols from files whose headers appear in the context,
    and uses qualified_name (not just short name) to avoid false positives
    from common names like 'Any' or 'BaseModel'.
    """
    symbol_keys: list[str] = []
    seen: set[str] = set()

    # Extract file paths from context headers (# file.py or ## file.py)
    ctx_files = set(re.findall(r"(?:^|\n)##?\s+([\w/._-]+\.(?:py|js|ts|go|rs|java))", context))
    if not ctx_files:
        return []

    for nid, ndata in graph.nodes(data=True):
        if ndata.get("kind") == "file" or ndata.get("type") == "file":
            continue
        # Skip imports — they're noisy and not meaningful context symbols
        if ndata.get("kind") == "import":
            continue

        npath = ndata.get("file_path", "") or ndata.get("path", "")
        if not npath and "::" in str(nid):
            npath = str(nid).split("::")[0]
        if not npath:
            continue

        # Check if this symbol's file is in context
        file_in_ctx = False
        for cf in ctx_files:
            if npath.endswith(cf) or cf.endswith(npath):
                file_in_ctx = True
                break
        if not file_in_ctx:
            continue

        # Use qualified_name to avoid false positives from short common names
        qname = ndata.get("qualified_name", "")
        name = ndata.get("name", "")
        nid_str = str(nid)

        # A symbol is "in context" if its defining signature/name appears in the text.
        # For classes/functions, check the qualified name or "def name" / "class name" pattern
        if qname and len(qname) > 3:
            # Check qualified name or "def qname" or "class qname"
            if qname in context and nid_str not in seen:
                symbol_keys.append(nid_str)
                seen.add(nid_str)
                continue
        if name and len(name) > 3:
            # Check "def name(" or "class name" pattern to reduce false positives
            if (f"def {name}" in context or f"class {name}" in context) and nid_str not in seen:
                symbol_keys.append(nid_str)
                seen.add(nid_str)

    return symbol_keys


async def _run_single_eval(
    task: EvalTask,
    method_name: str,
    budget: int,
    query_type: str,
    description: str,
    repo_path: Path,
    graph,
    query: GraphQuery,
    provider,
    output_dir: Path,
    run_idx: int,
    total_runs: int,
) -> EvalResult:
    """Run a single evaluation: assemble → LLM → patch → test."""
    print(f"\n  [{run_idx}/{total_runs}] {method_name} @ B={budget} ({query_type})")

    method_fn = METHODS[method_name]

    # 1. Assemble context from buggy code
    try:
        # target_file needs the mutation dict to know which file to read
        if method_name == "target_file":
            context, tokens_used, syms, files, asm_time = method_fn(
                repo_path, description, budget, graph, query,
                mutation=task.mutation,
            )
        elif method_name == "naive_random":
            context, tokens_used, syms, files, asm_time = method_fn(
                repo_path, description, budget, graph, query,
                _task_id=task.task_id, _query_type=query_type,
            )
        else:
            context, tokens_used, syms, files, asm_time = method_fn(
                repo_path, description, budget, graph, query,
            )
    except Exception as e:
        err_type = type(e).__name__
        print(f"    assembly error ({err_type}): {e}")
        return EvalResult(
            task_id=task.task_id, method=method_name, budget=budget,
            query_type=query_type,
            tokens_used=0, symbols_selected=0, files_included=0,
            assembly_time_ms=0, llm_time_ms=0,
            llm_input_tokens=0, llm_output_tokens=0,
            tests_passed=False, test_output="", patch="",
            error=f"{err_type}: {e}",
            failure_mode="assembly_error",
            repo_name=Path(task.repo_path).name if task.repo_path else "",
        )

    print(
        f"    context: {tokens_used} tokens, {syms} symbols, "
        f"{files} files ({asm_time}ms)"
    )

    # 2. Call LLM (with throttle to avoid rate limits)
    await asyncio.sleep(1.0)  # 1s between calls
    llm_start = time.time()
    try:
        llm_response, llm_time, in_tok, out_tok = await call_llm(
            provider, context, description,
        )
    except Exception as e:
        err_type = type(e).__name__
        llm_elapsed = round((time.time() - llm_start) * 1000, 1)
        print(f"    LLM error ({err_type}): {e} [{llm_elapsed}ms]")
        return EvalResult(
            task_id=task.task_id, method=method_name, budget=budget,
            query_type=query_type,
            tokens_used=tokens_used, symbols_selected=syms,
            files_included=files, assembly_time_ms=asm_time,
            llm_time_ms=llm_elapsed, llm_input_tokens=0, llm_output_tokens=0,
            tests_passed=False, test_output="", patch="",
            error=f"{err_type}: {e}",
            failure_mode="llm_error",
            repo_name=Path(task.repo_path).name if task.repo_path else "",
        )

    print(f"    LLM: {in_tok} in, {out_tok} out ({llm_time}ms)")

    # 3. Extract edits and compute patch quality metrics
    edits = extract_edits(llm_response)
    patch = extract_patch(llm_response)
    patch_files = len(set(e.file_path for e in edits))
    patch_lines = sum(
        max(len(e.search.splitlines()), len(e.replace.splitlines()))
        for e in edits
    )
    print(f"    edits: {len(edits)} blocks, {patch_files} files, {patch_lines} lines")

    # 4. Check retrieval quality metrics
    target_hit = False
    target_sym_hit = False
    ctx_patch_overlap = 0.0
    if task.mutation and "file" in task.mutation:
        mut_file = task.mutation["file"]
        target_hit = mut_file in context or Path(mut_file).name in context
        # Target-symbol recall: check if the mutated function appears in context
        mut_line = task.mutation.get("line_num")
        if mut_line and graph is not None:
            try:
                # Find symbols in the mutation file that contain the mutated line
                for nid, ndata in graph.nodes(data=True):
                    if ndata.get("kind") == "file":
                        continue
                    npath = ndata.get("file_path", "") or ndata.get("path", "")
                    if not npath:
                        # Try extracting from node ID
                        if "::" in str(nid):
                            npath = str(nid).split("::")[0]
                    if not npath:
                        continue
                    if not (npath.endswith(mut_file) or mut_file.endswith(npath)):
                        continue
                    nstart = ndata.get("line_start", 0) or ndata.get("start_line", 0)
                    nend = ndata.get("line_end", 0) or ndata.get("end_line", 0)
                    if nstart <= mut_line <= nend:
                        sym_name = ndata.get("name", "") or ndata.get("qualified_name", "")
                        if sym_name and sym_name in context:
                            target_sym_hit = True
                            break
            except Exception:
                pass

    # Context-patch overlap: what fraction of context file paths appear in the patch?
    if edits and context:
        ctx_files = set(re.findall(r"# ([a-zA-Z0-9_/.]+\.\w+)", context))
        patch_files_set = set(e.file_path for e in edits)
        if ctx_files:
            ctx_patch_overlap = len(ctx_files & patch_files_set) / len(ctx_files)

    # 5. Apply edits to buggy code and test
    test_start = time.time()
    passed, test_output = apply_and_test(
        repo_path, llm_response, task.test_cmd, task.timeout,
        in_place=task.in_place,
    )
    test_time = (time.time() - test_start) * 1000

    # 6. Classify failure mode
    failure_mode = ""
    if passed:
        failure_mode = "pass"
    elif not edits:
        failure_mode = "no_patch"
    elif "file not found" in test_output or "no edits applied" in test_output:
        failure_mode = "patch_apply_fail"
    elif "search text not found" in test_output:
        failure_mode = "patch_apply_fail"
    elif "test timeout" in test_output:
        failure_mode = "timeout"
    elif "[REGRESSION]" in test_output:
        failure_mode = "regression"
    elif "SyntaxError" in test_output or "ImportError" in test_output:
        failure_mode = "syntax_error"
    else:
        failure_mode = "test_fail"

    # 7. Edit locality: distance from edited line to mutated line
    edit_dist = -1
    if task.mutation and "line_num" in task.mutation and edits:
        mut_line = task.mutation["line_num"]
        mut_file = task.mutation["file"]
        for e in edits:
            if mut_file in e.file_path or Path(mut_file).name in e.file_path:
                # Find line number of the edit in the original file
                target = repo_path / e.file_path
                if not target.exists():
                    resolved = _resolve_file_path(repo_path, e.file_path)
                    if resolved:
                        target = resolved
                if target.exists():
                    try:
                        content = target.read_text()
                        idx = content.find(e.search[:50])  # first 50 chars
                        if idx >= 0:
                            edit_line = content[:idx].count("\n") + 1
                            edit_dist = abs(edit_line - mut_line)
                            break
                    except OSError:
                        pass

    # 8. Entity extraction & seed resolution (ALL methods, for metrics)
    entities = _extract_entities_standalone(description)
    entity_count_extracted = len(entities)
    seed_keys = _resolve_entities_to_seeds(entities, graph, query) if graph is not None else []
    entity_count_mapped = len(seed_keys)
    qid_density = _compute_query_identifier_density(description)

    # 9. Mutation symbol key (reuse for graph distance)
    mut_info = _find_mutation_symbol(graph, task.mutation) if graph is not None else {"key": "", "lines": 0, "kind": "", "file_symbols": 0}
    mut_sym_key = mut_info["key"]

    # 10. Graph distance: seed symbols → mutation symbol
    min_hops, median_hops = _compute_graph_hops(graph, seed_keys, mut_sym_key) if graph is not None else (-1, -1.0)

    # 11. Context symbol keys (for post-hoc dependency coverage)
    ctx_sym_keys = _extract_context_symbol_keys(context, graph) if graph is not None else []

    # 12. Router B confidence features from retrieval scores
    global _last_retrieval_scores
    retrieval_conf = _compute_retrieval_confidence(_last_retrieval_scores)
    _last_retrieval_scores = []  # Reset for next call

    # 13. BCA-specific debug from _last_context_package
    global _last_context_package
    bca_closure_syms = 0
    bca_closure_toks = 0
    bca_frontier = 0
    if method_name in ("bca", "bca_d1", "bca_d5", "bca_no_closure", "bca_no_scoring") and _last_context_package is not None:
        bca_closure_syms = getattr(_last_context_package, "closure_added_symbols", 0)
        bca_closure_toks = getattr(_last_context_package, "closure_added_tokens", 0)
        bca_frontier = getattr(_last_context_package, "frontier_visited", 0)
        _last_context_package = None  # Reset for next call

    status = "PASS" if passed else f"FAIL({failure_mode})"
    print(
        f"    result: {status}"
        + (f"  hit={target_hit} sym_hit={target_sym_hit} dist={edit_dist}" if task.mutation else "")
        + f"  entities={entity_count_extracted}/{entity_count_mapped}"
        + (f" hops={min_hops}" if min_hops >= 0 else "")
    )

    result = EvalResult(
        task_id=task.task_id,
        method=method_name,
        budget=budget,
        query_type=query_type,
        tokens_used=tokens_used,
        symbols_selected=syms,
        files_included=files,
        assembly_time_ms=asm_time,
        llm_time_ms=llm_time,
        llm_input_tokens=in_tok,
        llm_output_tokens=out_tok,
        tests_passed=passed,
        test_output=test_output[-500:],
        patch=patch[:2000],
        test_time_ms=round(test_time, 1),
        failure_mode=failure_mode,
        target_file_hit=target_hit,
        target_symbol_hit=target_sym_hit,
        context_patch_overlap=round(ctx_patch_overlap, 3),
        patch_files_changed=patch_files,
        patch_lines_changed=patch_lines,
        edit_distance_lines=edit_dist,
        # Pre-launch metrics
        entity_count_extracted=entity_count_extracted,
        entity_count_mapped=entity_count_mapped,
        query_identifier_density=qid_density,
        seed_symbol_keys=seed_keys,
        mutation_symbol_key=mut_sym_key,
        min_hops_seed_to_mutation=min_hops,
        median_hops_seed_to_mutation=median_hops,
        bca_closure_added_symbols=bca_closure_syms,
        bca_closure_added_tokens=bca_closure_toks,
        bca_frontier_visited=bca_frontier,
        context_symbol_keys=ctx_sym_keys,
        # Code metadata
        mutation_symbol_lines=mut_info["lines"],
        mutation_symbol_kind=mut_info["kind"],
        mutation_file_symbols=mut_info["file_symbols"],
        graph_node_count=graph.number_of_nodes() if graph is not None else 0,
        # Router B confidence features (scale-free + raw)
        retrieval_top1_score=retrieval_conf["retrieval_top1_score"],
        retrieval_top1_top2_gap=retrieval_conf["retrieval_top1_top2_gap"],
        retrieval_softmax_entropy=retrieval_conf["retrieval_softmax_entropy"],
        retrieval_softmax_tau=retrieval_conf["retrieval_softmax_tau"],
        retrieval_effective_candidates=retrieval_conf["retrieval_effective_candidates"],
        retrieval_top5_ratio=retrieval_conf["retrieval_top5_ratio"],
        retrieval_within95_count=retrieval_conf["retrieval_within95_count"],
        retrieval_top5_mean_score=retrieval_conf["retrieval_top5_mean_score"],
        retrieval_scored_symbols=retrieval_conf["retrieval_scored_symbols"],
        # Coverage confidence (method-agnostic, scale-free)
        retrieval_budget_utilization=round(tokens_used / budget, 4) if budget > 0 else 0.0,
        retrieval_file_concentration=_compute_file_concentration(context),
        repo_name=Path(task.repo_path).name if task.repo_path else "",
        category=task.category,
        mutation_type=task.mutation_type or "handcrafted",
        source=task.source or "handcrafted",
    )

    # Save per-run artifact
    artifact_dir = output_dir / task.task_id / method_name / str(budget) / query_type
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "context.txt").write_text(context)
    (artifact_dir / "llm_response.txt").write_text(llm_response)
    (artifact_dir / "patch.diff").write_text(patch)
    (artifact_dir / "test_output.txt").write_text(test_output)
    (artifact_dir / "result.json").write_text(
        json.dumps(asdict(result), indent=2)
    )

    return result


async def run_benchmark(
    tasks: list[EvalTask],
    budgets: list[int],
    methods: list[str],
    llm_config: LLMConfig,
    output_dir: Path,
    query_types: list[str] | None = None,
) -> list[EvalResult]:
    """Run the full benchmark: mutate → context → LLM → patch → test → result.

    For each task with a mutation:
      1. Apply mutation to create buggy code
      2. Assemble context from buggy code (using pre-built graph)
      3. Call LLM to generate a fix
      4. Apply the fix and run tests
      5. Restore original code

    The code graph is built once per unique repo (not per task) since
    single-line mutations don't change the graph structure — functions
    and classes still exist at the same line ranges. Context assembly
    reads source from disk at eval time, so it picks up the mutated
    content without needing a graph rebuild.

    Args:
        query_types: List of query types to run. Default: ["exact"].
            Use ["exact", "vague", "dev_report"] for all three tiers.
    """
    if query_types is None:
        query_types = ["exact"]

    provider = create_provider(llm_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[EvalResult] = []
    total_runs = len(tasks) * len(budgets) * len(methods) * len(query_types)
    run_idx = 0

    # Run setup_cmd once per unique repo (installs deps for test execution).
    # Cached by repo path so it only runs once even with 100+ tasks per repo.
    _setup_done: set[str] = set()
    for task in tasks:
        if not task.setup_cmd or not task.repo_path:
            continue
        rp = str(Path(task.repo_path).resolve())
        if rp in _setup_done:
            continue
        _setup_done.add(rp)
        print(f"\n  Running setup for {rp}...")
        print(f"    cmd: {task.setup_cmd}")
        try:
            setup_result = subprocess.run(
                task.setup_cmd, shell=True, capture_output=True, text=True,
                timeout=300, cwd=rp,
            )
            if setup_result.returncode != 0:
                print(f"  FATAL: setup_cmd failed (rc={setup_result.returncode})")
                print(f"    stderr: {setup_result.stderr[-500:]}")
                raise RuntimeError(
                    f"Setup failed for {rp}: {setup_result.stderr[-200:]}"
                )
            print(f"    setup OK ({rp})")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Setup timed out after 300s for {rp}")

    # --- Preflight: verify test infrastructure works on clean code ---
    # Runs one task's test_cmd per repo on un-mutated code. If the test infra
    # is broken (missing deps, conftest errors), we fail fast before burning
    # any LLM credits.
    _preflight_done: set[str] = set()
    for task in tasks:
        if not task.repo_path or not task.test_cmd:
            continue
        rp = str(Path(task.repo_path).resolve())
        if rp in _preflight_done:
            continue
        _preflight_done.add(rp)
        print(f"\n  Preflight test for {rp}...")
        print(f"    cmd: {task.test_cmd}")
        try:
            preflight = subprocess.run(
                task.test_cmd.split(),
                capture_output=True, text=True,
                timeout=task.timeout + 30,
                cwd=rp,
            )
            if preflight.returncode != 0:
                stderr_tail = preflight.stderr[-800:] if preflight.stderr else ""
                stdout_tail = preflight.stdout[-400:] if preflight.stdout else ""
                print(f"  FATAL: Preflight test failed (rc={preflight.returncode})")
                print(f"    stderr: {stderr_tail}")
                print(f"    stdout: {stdout_tail}")
                raise RuntimeError(
                    f"Preflight test failed for {rp}. Test infrastructure is broken. "
                    f"Fix before running benchmark to avoid wasting LLM credits.\n"
                    f"  cmd: {task.test_cmd}\n  stderr: {stderr_tail[-200:]}"
                )
            print("    preflight OK — tests pass on clean code")
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Preflight test timed out (>{task.timeout + 30}s) for {rp}. "
                f"Test infrastructure may be broken or test is too slow.\n"
                f"  cmd: {task.test_cmd}"
            )

    # Build graph once per unique repo path (not per task).
    # Single-line mutations don't change graph structure; source is loaded
    # from disk at assembly time, so the mutated content is picked up.
    graph_cache: dict[str, tuple] = {}  # repo_path_str -> (graph, query)
    graph_build_times: dict[str, float] = {}  # repo_path_str -> build_time_s

    for task in tasks:
        repo_path = Path(task.repo_path).resolve()
        repo_key = str(repo_path)

        if repo_key not in graph_cache:
            print(f"\n  Building graph for {repo_path}...")
            build_start = time.time()
            builder = GraphBuilder()
            graph = builder.build_from_directory(repo_path)
            query_obj = GraphQuery(graph)
            graph_cache[repo_key] = (graph, query_obj)
            build_time = time.time() - build_start
            graph_build_times[repo_key] = build_time
            n_nodes = graph.number_of_nodes()
            print(f"  Graph built: {n_nodes} nodes in {build_time:.1f}s (cached for all tasks)")

        graph, query = graph_cache[repo_key]

        print(f"\n{'='*60}")
        print(f"Task: {task.task_id}")
        print(f"Repo: {repo_path}")
        print(f"Description: {task.description[:80]}")
        if task.vague_description and "vague" in query_types:
            print(f"Vague: {task.vague_description[:80]}")
        print(f"{'='*60}")

        # Apply mutation to create buggy code
        mutation_backup = None  # (original_content, original_hash)
        if task.mutation:
            mutation_backup = _apply_mutation(repo_path, task.mutation)
            if mutation_backup is None:
                print("  WARNING: mutation could not be applied, skipping task")
                continue
            print(f"  Mutation applied: {task.mutation['file']}")

        try:
            for qt in query_types:
                if qt == "vague":
                    if not task.vague_description:
                        print(f"  SKIP vague: no vague_description for {task.task_id}")
                        continue
                    description = task.vague_description
                elif qt == "dev_report":
                    if not task.dev_report_description:
                        print(f"  SKIP dev_report: no dev_report_description for {task.task_id}")
                        continue
                    description = task.dev_report_description
                else:
                    description = task.description

                # Methods whose output doesn't depend on budget (context is fixed).
                # Run once at min budget; duplicate results for other budgets to keep
                # the grid complete without wasting LLM calls + test time.
                _BUDGET_INDEPENDENT = {"no_retrieval"}

                for budget in budgets:
                    for method_name in methods:
                        # Skip redundant budget runs for budget-independent methods
                        if method_name in _BUDGET_INDEPENDENT and budget != budgets[0]:
                            # Reuse result from first budget
                            src = next(
                                (r for r in all_results
                                 if r.task_id == task.task_id and r.method == method_name
                                 and r.query_type == qt and r.budget == budgets[0]),
                                None,
                            )
                            if src is not None:
                                from dataclasses import replace as _dc_replace
                                dup = _dc_replace(src, budget=budget)
                                all_results.append(dup)
                                run_idx += 1
                                print(f"  [{run_idx}/{total_runs}] {task.task_id} {method_name} B={budget} {qt} (reused from B={budgets[0]})")
                                continue

                        run_idx += 1
                        result = await _run_single_eval(
                            task=task,
                            method_name=method_name,
                            budget=budget,
                            query_type=qt,
                            description=description,
                            repo_path=repo_path,
                            graph=graph,
                            query=query,
                            provider=provider,
                            output_dir=output_dir,
                            run_idx=run_idx,
                            total_runs=total_runs,
                        )
                        all_results.append(result)

        finally:
            # Always restore original code with byte-identical verification
            if mutation_backup is not None:
                orig_content, orig_hash = mutation_backup
                _restore_mutation(repo_path, task.mutation, orig_content, orig_hash)
                print(f"\n  Mutation restored: {task.mutation['file']} (hash verified)")

    return all_results, graph_build_times


def compute_oracle(results: list[EvalResult], budgets: list[int]) -> dict:
    """Compute oracle upper bound: best method per-task at each budget.

    Oracle picks the best method for each (task, budget, query_type) triple
    after seeing all results — this is the ceiling any router could approach.
    """
    query_types = sorted(set(r.query_type for r in results))
    oracle = {}

    for qt in query_types:
        qt_results = [r for r in results if r.query_type == qt]
        for budget in budgets:
            b_results = [r for r in qt_results if r.budget == budget]
            tasks = sorted(set(r.task_id for r in b_results))
            passes = 0
            for tid in tasks:
                task_results = [r for r in b_results if r.task_id == tid]
                if any(r.tests_passed for r in task_results):
                    passes += 1
            oracle_rate = passes / len(tasks) if tasks else 0
            oracle[(qt, budget)] = oracle_rate

    return oracle


def bootstrap_paired_ci(
    results: list[EvalResult],
    budgets: list[int],
    n_bootstrap: int = 10000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[tuple[str, str, int, str], tuple[float, float, float]]:
    """Compute paired bootstrap CIs for pass@1 differences between methods.

    For each pair of methods at each (budget, query_type), we:
      1. Build per-task paired differences: d_i = pass_A(task_i) - pass_B(task_i)
      2. Resample tasks B times with replacement
      3. Compute mean difference for each bootstrap sample
      4. Return (point_estimate, ci_lo, ci_hi) at the given alpha level

    A CI that excludes 0 indicates a statistically significant difference.

    Returns:
        Dict mapping (method_a, method_b, budget, query_type) -> (mean_diff, ci_lo, ci_hi)
    """
    rng = random.Random(seed)

    # Index results by (task_id, method, budget, query_type)
    result_map: dict[tuple[str, str, int, str], bool] = {}
    for r in results:
        result_map[(r.task_id, r.method, r.budget, r.query_type)] = r.tests_passed

    query_types = sorted(set(r.query_type for r in results))
    methods = sorted(set(r.method for r in results))
    task_ids = sorted(set(r.task_id for r in results))

    cis: dict[tuple[str, str, int, str], tuple[float, float, float]] = {}

    for qt in query_types:
        for budget in budgets:
            # Get tasks that have results for this (budget, query_type)
            tasks_here = [
                tid for tid in task_ids
                if any(
                    (tid, m, budget, qt) in result_map
                    for m in methods
                )
            ]
            if len(tasks_here) < 2:
                continue

            for i, method_a in enumerate(methods):
                for method_b in methods[i + 1:]:
                    # Build paired differences
                    diffs = []
                    for tid in tasks_here:
                        pass_a = 1 if result_map.get((tid, method_a, budget, qt), False) else 0
                        pass_b = 1 if result_map.get((tid, method_b, budget, qt), False) else 0
                        diffs.append(pass_a - pass_b)

                    if not diffs:
                        continue

                    n = len(diffs)
                    point_est = sum(diffs) / n

                    # Bootstrap
                    boot_means = []
                    for _ in range(n_bootstrap):
                        sample = [diffs[rng.randint(0, n - 1)] for _ in range(n)]
                        boot_means.append(sum(sample) / n)

                    boot_means.sort()
                    lo_idx = int(n_bootstrap * (alpha / 2))
                    hi_idx = int(n_bootstrap * (1 - alpha / 2)) - 1
                    ci_lo = boot_means[max(0, lo_idx)]
                    ci_hi = boot_means[min(len(boot_means) - 1, hi_idx)]

                    cis[(method_a, method_b, budget, qt)] = (point_est, ci_lo, ci_hi)

    return cis


def format_bootstrap_analysis(
    results: list[EvalResult],
    budgets: list[int],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> str:
    """Format paired bootstrap CI analysis as readable text.

    Only shows pairs where the CI excludes zero (significant differences)
    plus a summary of all BCA comparisons.
    Excludes ceiling methods (target_file).
    """
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    cis = bootstrap_paired_ci(grid_results, budgets, n_bootstrap=n_bootstrap, seed=seed)
    if not cis:
        return ""

    query_types = sorted(set(r.query_type for r in results))
    lines = []

    for qt in query_types:
        qt_label = qt.upper()
        lines.append(f"\n{'='*70}")
        lines.append(f"Paired Bootstrap CIs (pass@1 difference) [{qt_label} queries]")
        lines.append(f"  n_bootstrap={n_bootstrap}, seed={seed}, alpha=0.05")
        lines.append(f"{'='*70}")

        # Group by budget
        for budget in budgets:
            lines.append(f"\n  Budget = {budget}")
            lines.append(f"  {'Method A':<18} {'Method B':<18} {'Diff':>6} {'95% CI':>16} {'Sig':>5}")
            lines.append(f"  {'-'*65}")

            relevant = {
                (ma, mb): (d, lo, hi)
                for (ma, mb, b, q), (d, lo, hi) in cis.items()
                if b == budget and q == qt
            }

            # Sort by absolute difference descending
            for (ma, mb), (diff, lo, hi) in sorted(
                relevant.items(), key=lambda x: abs(x[1][0]), reverse=True
            ):
                sig = "*" if lo > 0 or hi < 0 else ""
                lines.append(
                    f"  {ma:<18} {mb:<18} {diff:>+.3f} [{lo:>+.3f}, {hi:>+.3f}] {sig:>5}"
                )

        # Summary: how many significant pairs per budget
        lines.append("\n  Summary: significant pairs (CI excludes 0)")
        for budget in budgets:
            relevant = {
                k: v for k, v in cis.items()
                if k[2] == budget and k[3] == qt
            }
            n_sig = sum(1 for (_, lo, hi) in relevant.values() if lo > 0 or hi < 0)
            lines.append(f"    B={budget}: {n_sig}/{len(relevant)} pairs significant")

    return "\n".join(lines)


def format_results(results: list[EvalResult], budgets: list[int]) -> str:
    """Format results as pass@1 table, grouped by query_type.

    Excludes ceiling methods (target_file) — those are reported separately.
    """
    # Filter out ceiling methods from the main table
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    query_types = sorted(set(r.query_type for r in grid_results))
    methods = sorted(set(r.method for r in grid_results))
    lines = []

    oracle = compute_oracle(grid_results, budgets)

    for qt in query_types:
        qt_results = [r for r in grid_results if r.query_type == qt]
        qt_label = qt.upper()

        lines.append(f"\n{'='*70}")
        lines.append(f"Pass@1 by Method and Budget [{qt_label} queries]")
        lines.append(f"{'='*70}")

        header = f"{'Method':<20}" + "".join(f"  B={b:>5}" for b in budgets) + "    Avg"
        lines.append(header)
        lines.append("-" * 70)

        for m in methods:
            row = f"{m:<20}"
            method_passes = []
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    pass_rate = sum(1 for r in runs if r.tests_passed) / len(runs)
                    row += f"  {pass_rate:>5.2f}"
                    method_passes.append(pass_rate)
                else:
                    row += f"  {'n/a':>5}"
            if method_passes:
                row += f"  {sum(method_passes)/len(method_passes):>5.2f}"
            lines.append(row)

        # Oracle row
        row = f"{'oracle':<20}"
        oracle_vals = []
        for b in budgets:
            rate = oracle.get((qt, b))
            if rate is not None:
                row += f"  {rate:>5.2f}"
                oracle_vals.append(rate)
            else:
                row += f"  {'n/a':>5}"
        if oracle_vals:
            row += f"  {sum(oracle_vals)/len(oracle_vals):>5.2f}"
        lines.append(row)

        lines.append("")
        lines.append(f"{'='*70}")
        lines.append(f"Mean Tokens Used / Budget [{qt_label} queries]")
        lines.append(f"{'='*70}")
        lines.append(header)
        lines.append("-" * 70)

        for m in methods:
            row = f"{m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    mean_pct = sum(r.tokens_used / r.budget * 100 for r in runs) / len(runs)
                    row += f"  {mean_pct:>4.0f}%"
                else:
                    row += f"  {'n/a':>5}"
            lines.append(row)

        lines.append("")
        lines.append(f"{'='*70}")
        lines.append(f"Mean LLM Tokens (input + output) [{qt_label} queries]")
        lines.append(f"{'='*70}")
        lines.append(header)
        lines.append("-" * 70)

        for m in methods:
            row = f"{m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    mean_total = sum(
                        r.llm_input_tokens + r.llm_output_tokens for r in runs
                    ) / len(runs)
                    row += f"  {mean_total:>5.0f}"
                else:
                    row += f"  {'n/a':>5}"
            lines.append(row)

    return "\n".join(lines)


def format_per_repo_results(
    results: list[EvalResult],
    tasks: list[EvalTask],
    budgets: list[int],
) -> str:
    """Format pass@1 tables broken down by repository.

    Derives repo name from task.repo_path (uses directory name).
    Produces a separate pass@1 table for each repo, enabling cross-repo comparison.
    Excludes ceiling methods (target_file).
    """
    from pathlib import Path as _P

    # Build task_id -> repo_name mapping
    task_repo: dict[str, str] = {}
    for t in tasks:
        repo_name = _P(t.repo_path).name if t.repo_path else "unknown"
        task_repo[t.task_id] = repo_name

    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    repos = sorted(set(task_repo.get(r.task_id, "unknown") for r in grid_results))

    if len(repos) < 2:
        return ""  # Skip if single repo — main table already covers it

    query_types = sorted(set(r.query_type for r in grid_results))
    methods = sorted(set(r.method for r in grid_results))
    lines = []

    for repo in repos:
        repo_task_ids = {tid for tid, rn in task_repo.items() if rn == repo}
        repo_results = [r for r in grid_results if r.task_id in repo_task_ids]
        n_tasks = len(set(r.task_id for r in repo_results))

        for qt in query_types:
            qt_results = [r for r in repo_results if r.query_type == qt]
            qt_label = qt.upper()

            lines.append(f"\n{'='*70}")
            lines.append(f"Pass@1 — {repo} (n={n_tasks} tasks) [{qt_label} queries]")
            lines.append(f"{'='*70}")

            header = f"{'Method':<20}" + "".join(f"  B={b:>5}" for b in budgets) + "    Avg"
            lines.append(header)
            lines.append("-" * 70)

            for m in methods:
                row = f"{m:<20}"
                method_rates = []
                for b in budgets:
                    runs = [r for r in qt_results if r.method == m and r.budget == b]
                    if runs:
                        rate = sum(1 for r in runs if r.tests_passed) / len(runs)
                        row += f"  {rate:>5.2f}"
                        method_rates.append(rate)
                    else:
                        row += f"  {'n/a':>5}"
                if method_rates:
                    row += f"  {sum(method_rates)/len(method_rates):>5.2f}"
                lines.append(row)

    return "\n".join(lines)


def _bootstrap_single_ci(
    pass_values: list[int], n_bootstrap: int = 5000, seed: int = 42, alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap CI for a single pass@1 rate. Returns (ci_lo, ci_hi)."""
    if len(pass_values) < 2:
        rate = sum(pass_values) / len(pass_values) if pass_values else 0
        return (rate, rate)
    rng = random.Random(seed)
    n = len(pass_values)
    boot = sorted(
        sum(pass_values[rng.randint(0, n - 1)] for _ in range(n)) / n
        for _ in range(n_bootstrap)
    )
    lo_idx = int(n_bootstrap * (alpha / 2))
    hi_idx = int(n_bootstrap * (1 - alpha / 2)) - 1
    return (boot[max(0, lo_idx)], boot[min(len(boot) - 1, hi_idx)])


def format_results_with_ci(results: list[EvalResult], budgets: list[int]) -> str:
    """Format pass@1 table with per-cell 95% bootstrap CIs.

    Excludes ceiling methods (target_file) — those are reported separately.
    """
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    query_types = sorted(set(r.query_type for r in grid_results))
    methods = sorted(set(r.method for r in grid_results))
    oracle = compute_oracle(grid_results, budgets)
    lines = []

    for qt in query_types:
        qt_results = [r for r in grid_results if r.query_type == qt]
        qt_label = qt.upper()

        lines.append(f"\n{'='*90}")
        lines.append(f"Pass@1 with 95% Bootstrap CI [{qt_label} queries]")
        lines.append(f"{'='*90}")

        header = f"{'Method':<20}" + "".join(f"  {'B=' + str(b):>14}" for b in budgets) + "      Avg"
        lines.append(header)
        lines.append("-" * 90)

        for m in methods:
            row = f"{m:<20}"
            method_rates = []
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    passes = [1 if r.tests_passed else 0 for r in runs]
                    rate = sum(passes) / len(passes)
                    ci_lo, ci_hi = _bootstrap_single_ci(passes)
                    row += f"  {rate:.2f}[{ci_lo:.2f},{ci_hi:.2f}]"
                    method_rates.append(rate)
                else:
                    row += f"  {'n/a':>14}"
            if method_rates:
                avg = sum(method_rates) / len(method_rates)
                row += f"  {avg:>8.2f}"
            lines.append(row)

        # Oracle row
        row = f"{'oracle':<20}"
        ovals = []
        for b in budgets:
            rate = oracle.get((qt, b))
            if rate is not None:
                row += f"  {rate:>14.2f}"
                ovals.append(rate)
            else:
                row += f"  {'n/a':>14}"
        if ovals:
            row += f"  {sum(ovals)/len(ovals):>8.2f}"
        lines.append(row)

    return "\n".join(lines)


def format_ceiling_probe(results: list[EvalResult], budgets: list[int]) -> str:
    """Format target_file results as a separate ceiling probe table.

    target_file uses privileged information (the exact mutation file) and is
    NOT a retrieval method. Report it separately to keep the main table clean.
    Interpretation: "If file identification were solved, this is the repair ceiling."
    """
    tf_results = [r for r in results if r.method == "target_file"]
    if not tf_results:
        return ""

    query_types = sorted(set(r.query_type for r in tf_results))
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append("Ceiling Probe: target_file (privileged file identification)")
    lines.append("  Interpretation: upper bound on repair rate if retrieval were perfect.")
    lines.append(f"{'='*70}")

    for qt in query_types:
        qt_label = qt.upper()
        qt_results = [r for r in tf_results if r.query_type == qt]

        lines.append(f"\n  [{qt_label} queries]")
        lines.append(f"  {'Budget':<10} {'Pass@1':>8} {'N':>5} {'Med. Tokens':>12}")
        lines.append(f"  {'-'*40}")

        for b in budgets:
            b_results = [r for r in qt_results if r.budget == b]
            if not b_results:
                continue
            passes = sum(1 for r in b_results if r.tests_passed)
            rate = passes / len(b_results)
            tokens = sorted(r.tokens_used for r in b_results)
            median_tok = tokens[len(tokens) // 2] if tokens else 0
            lines.append(f"  B={b:<7} {rate:>8.3f} {len(b_results):>5} {median_tok:>12}")

    return "\n".join(lines)


# Sentinel set for methods excluded from the main comparison grid.
# These use privileged information and are reported in separate tables.
_CEILING_METHODS = {"target_file"}


def compute_router_loo(
    results: list[EvalResult],
    tasks: list[EvalTask],
    budgets: list[int],
) -> dict:
    """Compute router pass@1 via leave-one-out cross-validation.

    For each (budget, query_type), trains a simple majority-vote router:
      - For each held-out task i, find the method with the highest pass rate
        on all other tasks at this (budget, query_type).
      - Apply that method's result to task i.

    This is the simplest possible "pick the best method" router.
    Excludes ceiling methods (target_file) from the routing candidates.
    Returns dict mapping (query_type, budget) -> router_pass_rate.
    """
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    query_types = sorted(set(r.query_type for r in grid_results))
    all_methods = sorted(set(r.method for r in grid_results))
    task_ids = sorted(set(r.task_id for r in grid_results))

    # Index: (task_id, method, budget, query_type) -> passed
    idx: dict[tuple[str, str, int, str], bool] = {}
    for r in grid_results:
        idx[(r.task_id, r.method, r.budget, r.query_type)] = r.tests_passed

    router_results = {}

    for qt in query_types:
        for budget in budgets:
            tasks_here = [
                tid for tid in task_ids
                if any((tid, m, budget, qt) in idx for m in all_methods)
            ]
            if len(tasks_here) < 2:
                continue

            correct = 0
            for held_out in tasks_here:
                # Train: pick best method on all other tasks
                best_method = None
                best_rate = -1
                for m in all_methods:
                    passes = sum(
                        1 for tid in tasks_here
                        if tid != held_out and idx.get((tid, m, budget, qt), False)
                    )
                    rate = passes / (len(tasks_here) - 1)
                    if rate > best_rate:
                        best_rate = rate
                        best_method = m
                # Apply to held-out task
                if best_method and idx.get((held_out, best_method, budget, qt), False):
                    correct += 1

            router_results[(qt, budget)] = correct / len(tasks_here) if tasks_here else 0

    return router_results


def format_decomposition(
    results: list[EvalResult],
    tasks: list[EvalTask],
    budgets: list[int],
) -> str:
    """Format pass@1 decomposed by mutation type and by category.

    Shows which mutation families and subsystems each method handles well,
    supporting the "conditional advantage" claim.
    Excludes ceiling methods (target_file) — those are reported separately.
    """
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    lines = []

    query_types = sorted(set(r.query_type for r in grid_results))
    methods = sorted(set(r.method for r in grid_results))

    # Build task lookup for metadata
    _task_lookup: dict[str, EvalTask] = {t.task_id: t for t in tasks}

    def extract_mutation_type(task_id: str) -> str:
        t = _task_lookup.get(task_id)
        if t and t.mutation_type:
            return t.mutation_type
        # Fallback: infer from task_id pattern (d-{stem}-L{line}-{mutation_type})
        if task_id.startswith("d-"):
            parts = task_id.split("-")
            if len(parts) >= 4:
                return parts[-1]
        return "handcrafted"

    def extract_category(task_id: str, tasks_list: list[EvalTask]) -> str:
        t = _task_lookup.get(task_id)
        if t and t.category:
            return t.category
        # Fallback: derive from file path
        for t2 in tasks_list:
            if t2.task_id == task_id and t2.mutation:
                from pathlib import Path as _P
                fname = _P(t2.mutation.get("file", "")).stem.lstrip("_")
                return fname or "other"
        return "unknown"

    # Use budget endpoints only (1K and max) to keep table readable
    show_budgets = [budgets[0], budgets[-1]] if len(budgets) > 1 else budgets

    for qt in query_types:
        qt_results = [r for r in grid_results if r.query_type == qt]
        qt_label = qt.upper()

        # --- By mutation type ---
        lines.append(f"\n{'='*80}")
        lines.append(f"Pass@1 by Mutation Type [{qt_label} queries]")
        lines.append(f"{'='*80}")

        mt_tasks: dict[str, list[str]] = {}
        for r in qt_results:
            mt = extract_mutation_type(r.task_id)
            mt_tasks.setdefault(mt, [])
            if r.task_id not in mt_tasks[mt]:
                mt_tasks[mt].append(r.task_id)

        for b in show_budgets:
            lines.append(f"\n  Budget = {b}")
            header = f"  {'Mut. Type':<22}" + "".join(f"  {m[:12]:>12}" for m in methods) + "    N"
            lines.append(header)
            lines.append(f"  {'-'*len(header)}")

            for mt in sorted(mt_tasks.keys()):
                tids = mt_tasks[mt]
                row = f"  {mt:<22}"
                for m in methods:
                    passes = sum(
                        1 for tid in tids
                        for r in qt_results
                        if r.task_id == tid and r.method == m and r.budget == b and r.tests_passed
                    )
                    total = sum(
                        1 for tid in tids
                        for r in qt_results
                        if r.task_id == tid and r.method == m and r.budget == b
                    )
                    rate = passes / total if total else 0
                    row += f"  {rate:>12.2f}"
                row += f"  {len(tids):>4}"
                lines.append(row)

        # --- By category ---
        lines.append(f"\n{'='*80}")
        lines.append(f"Pass@1 by Category [{qt_label} queries]")
        lines.append(f"{'='*80}")

        cat_tasks: dict[str, list[str]] = {}
        for r in qt_results:
            cat = extract_category(r.task_id, tasks)
            cat_tasks.setdefault(cat, [])
            if r.task_id not in cat_tasks[cat]:
                cat_tasks[cat].append(r.task_id)

        for b in show_budgets:
            lines.append(f"\n  Budget = {b}")
            header = f"  {'Category':<22}" + "".join(f"  {m[:12]:>12}" for m in methods) + "    N"
            lines.append(header)
            lines.append(f"  {'-'*len(header)}")

            for cat in sorted(cat_tasks.keys()):
                tids = cat_tasks[cat]
                if len(tids) < 2:
                    continue  # Skip categories with <2 tasks
                row = f"  {cat:<22}"
                for m in methods:
                    passes = sum(
                        1 for tid in tids
                        for r in qt_results
                        if r.task_id == tid and r.method == m and r.budget == b and r.tests_passed
                    )
                    total = sum(
                        1 for tid in tids
                        for r in qt_results
                        if r.task_id == tid and r.method == m and r.budget == b
                    )
                    rate = passes / total if total else 0
                    row += f"  {rate:>12.2f}"
                row += f"  {len(tids):>4}"
                lines.append(row)

    return "\n".join(lines)


def format_conditional_bins(results: list[EvalResult], budgets: list[int]) -> str:
    """Format pass@1 stratified by identifier density and hop distance.

    These conditional bins answer "when does BCA win?" by slicing results
    along two mechanistic axes:
      - Identifier density: do code identifiers appear in the query? (0 vs >0)
      - Hop distance: how far is the mutation from the nearest seed? (0, 1-2, 3+)
    Excludes ceiling methods (target_file).
    """
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    lines = []

    query_types = sorted(set(r.query_type for r in grid_results))
    methods = sorted(set(r.method for r in grid_results))
    show_budgets = [budgets[0], budgets[-1]] if len(budgets) > 1 else budgets

    # --- By identifier density ---
    def density_bin(density: float) -> str:
        if density == 0.0:
            return "zero (no identifiers)"
        return "positive (has identifiers)"

    for qt in query_types:
        qt_results = [r for r in grid_results if r.query_type == qt]
        qt_label = qt.upper()

        lines.append(f"\n{'='*80}")
        lines.append(f"Pass@1 by Identifier Density [{qt_label} queries]")
        lines.append(f"{'='*80}")

        # Group task_ids by density bin (use first result per task for the density)
        task_density: dict[str, str] = {}
        for r in qt_results:
            if r.task_id not in task_density:
                task_density[r.task_id] = density_bin(r.query_identifier_density)

        bin_tasks: dict[str, list[str]] = {}
        for tid, db in task_density.items():
            bin_tasks.setdefault(db, [])
            if tid not in bin_tasks[db]:
                bin_tasks[db].append(tid)

        for b in show_budgets:
            lines.append(f"\n  Budget = {b}")
            header = f"  {'Identifier Density':<30}" + "".join(f"  {m[:12]:>12}" for m in methods) + "    N"
            lines.append(header)
            lines.append(f"  {'-'*len(header)}")

            for db in sorted(bin_tasks.keys()):
                tids = bin_tasks[db]
                row = f"  {db:<30}"
                for m in methods:
                    passes = sum(
                        1 for tid in tids
                        for r in qt_results
                        if r.task_id == tid and r.method == m and r.budget == b and r.tests_passed
                    )
                    total = sum(
                        1 for tid in tids
                        for r in qt_results
                        if r.task_id == tid and r.method == m and r.budget == b
                    )
                    rate = passes / total if total else 0
                    row += f"  {rate:>12.2f}"
                row += f"  {len(tids):>4}"
                lines.append(row)

        # --- By hop distance ---
        lines.append(f"\n{'='*80}")
        lines.append(f"Pass@1 by Hop Distance (seed → mutation) [{qt_label} queries]")
        lines.append(f"{'='*80}")

        def hop_bin(hops: int) -> str:
            if hops < 0:
                return "unreachable / unknown"
            if hops == 0:
                return "0 hops (direct hit)"
            if hops <= 2:
                return "1-2 hops (close)"
            return "3+ hops (distant)"

        # Group task_ids by hop bin (use first result per task for hops)
        task_hops: dict[str, str] = {}
        for r in qt_results:
            if r.task_id not in task_hops:
                task_hops[r.task_id] = hop_bin(r.min_hops_seed_to_mutation)

        hbin_tasks: dict[str, list[str]] = {}
        for tid, hb in task_hops.items():
            hbin_tasks.setdefault(hb, [])
            if tid not in hbin_tasks[hb]:
                hbin_tasks[hb].append(tid)

        for b in show_budgets:
            lines.append(f"\n  Budget = {b}")
            header = f"  {'Hop Distance':<30}" + "".join(f"  {m[:12]:>12}" for m in methods) + "    N"
            lines.append(header)
            lines.append(f"  {'-'*len(header)}")

            for hb in sorted(hbin_tasks.keys()):
                tids = hbin_tasks[hb]
                row = f"  {hb:<30}"
                for m in methods:
                    passes = sum(
                        1 for tid in tids
                        for r in qt_results
                        if r.task_id == tid and r.method == m and r.budget == b and r.tests_passed
                    )
                    total = sum(
                        1 for tid in tids
                        for r in qt_results
                        if r.task_id == tid and r.method == m and r.budget == b
                    )
                    rate = passes / total if total else 0
                    row += f"  {rate:>12.2f}"
                row += f"  {len(tids):>4}"
                lines.append(row)

        # --- By mutation symbol size ---
        lines.append(f"\n{'='*80}")
        lines.append(f"Pass@1 by Mutation Size (function/class lines) [{qt_label} queries]")
        lines.append(f"{'='*80}")

        def size_bin(lines_count: int) -> str:
            if lines_count <= 0:
                return "unknown"
            if lines_count < 5:
                return "<5 lines (tiny)"
            if lines_count < 20:
                return "5-19 lines (small)"
            if lines_count < 50:
                return "20-49 lines (medium)"
            if lines_count < 100:
                return "50-99 lines (large)"
            return "100+ lines (very large)"

        task_size: dict[str, str] = {}
        for r in qt_results:
            if r.task_id not in task_size:
                task_size[r.task_id] = size_bin(r.mutation_symbol_lines)

        sbin_tasks: dict[str, list[str]] = {}
        for tid, sb in task_size.items():
            sbin_tasks.setdefault(sb, [])
            if tid not in sbin_tasks[sb]:
                sbin_tasks[sb].append(tid)

        for b in show_budgets:
            lines.append(f"\n  Budget = {b}")
            header = f"  {'Mutation Size':<30}" + "".join(f"  {m[:12]:>12}" for m in methods) + "    N"
            lines.append(header)
            lines.append(f"  {'-'*len(header)}")

            for sb in sorted(sbin_tasks.keys()):
                tids = sbin_tasks[sb]
                if len(tids) < 2:
                    continue  # Skip bins with <2 tasks
                row = f"  {sb:<30}"
                for m in methods:
                    passes = sum(
                        1 for tid in tids
                        for r in qt_results
                        if r.task_id == tid and r.method == m and r.budget == b and r.tests_passed
                    )
                    total = sum(
                        1 for tid in tids
                        for r in qt_results
                        if r.task_id == tid and r.method == m and r.budget == b
                    )
                    rate = passes / total if total else 0
                    row += f"  {rate:>12.2f}"
                row += f"  {len(tids):>4}"
                lines.append(row)

    return "\n".join(lines)


def format_failure_diagnosis(results: list[EvalResult], budgets: list[int]) -> str:
    """Format failure mode breakdown per method and budget.

    Shows WHY methods fail: no_patch, patch_apply_fail, syntax_error,
    test_fail, regression, timeout. Enables claims like "BCA improves
    pass@1 because it reduces wrong-file edits and patch failures."
    Excludes ceiling methods.
    """
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    query_types = sorted(set(r.query_type for r in grid_results))
    methods = sorted(set(r.method for r in grid_results))
    failure_modes = ["pass", "no_patch", "patch_apply_fail", "syntax_error", "test_fail", "regression", "timeout", "llm_error", "assembly_error"]
    lines = []

    for qt in query_types:
        qt_results = [r for r in grid_results if r.query_type == qt]
        qt_label = qt.upper()

        lines.append(f"\n{'='*100}")
        lines.append(f"Failure Mode Breakdown [{qt_label} queries]")
        lines.append(f"{'='*100}")

        for b in budgets:
            b_results = [r for r in qt_results if r.budget == b]
            if not b_results:
                continue
            lines.append(f"\n  Budget = {b}")
            header = f"  {'Method':<20}" + "".join(f"  {fm[:12]:>8}" for fm in failure_modes) + "    N"
            lines.append(header)
            lines.append(f"  {'-'*len(header)}")

            for m in methods:
                m_results = [r for r in b_results if r.method == m]
                if not m_results:
                    continue
                n = len(m_results)
                row = f"  {m:<20}"
                for fm in failure_modes:
                    count = sum(1 for r in m_results if r.failure_mode == fm)
                    pct = count / n * 100 if n else 0
                    row += f"  {pct:>7.0f}%"
                row += f"  {n:>4}"
                lines.append(row)

    return "\n".join(lines)


def format_retrieval_metrics(results: list[EvalResult], budgets: list[int]) -> str:
    """Format retrieval quality metrics per method and budget.

    Includes target file hit rate and budget utilization.
    Excludes ceiling methods.
    """
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    query_types = sorted(set(r.query_type for r in grid_results))
    methods = sorted(set(r.method for r in grid_results))
    lines = []

    for qt in query_types:
        qt_results = [r for r in grid_results if r.query_type == qt]
        qt_label = qt.upper()

        lines.append(f"\n{'='*90}")
        lines.append(f"Retrieval Quality Metrics [{qt_label} queries]")
        lines.append(f"{'='*90}")

        header = f"  {'Method':<20}" + "".join(f"  B={b:>10}" for b in budgets)
        lines.append("\n  --- Target File Hit Rate (%) ---")
        lines.append(header)
        lines.append(f"  {'-'*len(header)}")

        for m in methods:
            row = f"  {m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    hits = sum(1 for r in runs if r.target_file_hit)
                    pct = hits / len(runs) * 100
                    row += f"  {pct:>9.0f}%"
                else:
                    row += f"  {'n/a':>10}"
            lines.append(row)

        lines.append("\n  --- Target Symbol Hit Rate (%) ---")
        lines.append(header)
        lines.append(f"  {'-'*len(header)}")

        for m in methods:
            row = f"  {m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    hits = sum(1 for r in runs if r.target_symbol_hit)
                    pct = hits / len(runs) * 100
                    row += f"  {pct:>9.0f}%"
                else:
                    row += f"  {'n/a':>10}"
            lines.append(row)

        lines.append("\n  --- Budget Utilization (% of budget used) ---")
        lines.append(header)
        lines.append(f"  {'-'*len(header)}")

        for m in methods:
            row = f"  {m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    mean_util = sum(r.tokens_used / r.budget * 100 for r in runs if r.budget > 0) / len(runs)
                    row += f"  {mean_util:>9.0f}%"
                else:
                    row += f"  {'n/a':>10}"
            lines.append(row)

        lines.append("\n  --- Mean Total LLM Tokens (context + prompt + completion) ---")
        lines.append(header)
        lines.append(f"  {'-'*len(header)}")

        for m in methods:
            row = f"  {m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    mean_total = sum(r.llm_input_tokens + r.llm_output_tokens for r in runs) / len(runs)
                    row += f"  {mean_total:>10.0f}"
                else:
                    row += f"  {'n/a':>10}"
            lines.append(row)

    return "\n".join(lines)


def format_patch_quality(results: list[EvalResult], budgets: list[int]) -> str:
    """Format patch quality metrics per method and budget.

    Shows files changed and lines changed — smaller, more local edits
    are better. Excludes ceiling methods.
    """
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    query_types = sorted(set(r.query_type for r in grid_results))
    methods = sorted(set(r.method for r in grid_results))
    lines = []

    for qt in query_types:
        qt_results = [r for r in grid_results if r.query_type == qt]
        qt_label = qt.upper()

        lines.append(f"\n{'='*90}")
        lines.append(f"Patch Quality Metrics [{qt_label} queries]")
        lines.append(f"{'='*90}")

        header = f"  {'Method':<20}" + "".join(f"  B={b:>10}" for b in budgets)

        lines.append("\n  --- Mean Files Changed (per attempt) ---")
        lines.append(header)
        lines.append(f"  {'-'*len(header)}")

        for m in methods:
            row = f"  {m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    mean_files = sum(r.patch_files_changed for r in runs) / len(runs)
                    row += f"  {mean_files:>10.1f}"
                else:
                    row += f"  {'n/a':>10}"
            lines.append(row)

        lines.append("\n  --- Mean Lines Changed (per attempt) ---")
        lines.append(header)
        lines.append(f"  {'-'*len(header)}")

        for m in methods:
            row = f"  {m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    mean_lines = sum(r.patch_lines_changed for r in runs) / len(runs)
                    row += f"  {mean_lines:>10.1f}"
                else:
                    row += f"  {'n/a':>10}"
            lines.append(row)

        # Passes only — patch quality of successful repairs
        lines.append("\n  --- Mean Lines Changed (PASSING attempts only) ---")
        lines.append(header)
        lines.append(f"  {'-'*len(header)}")

        for m in methods:
            row = f"  {m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b and r.tests_passed]
                if runs:
                    mean_lines = sum(r.patch_lines_changed for r in runs) / len(runs)
                    row += f"  {mean_lines:>10.1f}"
                else:
                    row += f"  {'---':>10}"
            lines.append(row)

    return "\n".join(lines)


def format_latency_cost(results: list[EvalResult], budgets: list[int], graph_build_time: float = 0) -> str:
    """Format latency breakdown and cost analysis per method.

    Shows assembly time, LLM time, test time, and amortized graph build cost.
    Excludes ceiling methods.
    """
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    query_types = sorted(set(r.query_type for r in grid_results))
    methods = sorted(set(r.method for r in grid_results))
    n_tasks = len(set(r.task_id for r in grid_results))
    lines = []

    for qt in query_types:
        qt_results = [r for r in grid_results if r.query_type == qt]
        qt_label = qt.upper()

        lines.append(f"\n{'='*90}")
        lines.append(f"Latency & Cost Breakdown [{qt_label} queries]")
        lines.append(f"{'='*90}")

        header = f"  {'Method':<20}" + "".join(f"  B={b:>10}" for b in budgets)

        lines.append("\n  --- Mean Assembly Time (ms) ---")
        lines.append(header)
        lines.append(f"  {'-'*len(header)}")
        for m in methods:
            row = f"  {m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    mean_t = sum(r.assembly_time_ms for r in runs) / len(runs)
                    row += f"  {mean_t:>10.0f}"
                else:
                    row += f"  {'n/a':>10}"
            lines.append(row)

        lines.append("\n  --- Mean LLM Time (ms) ---")
        lines.append(header)
        lines.append(f"  {'-'*len(header)}")
        for m in methods:
            row = f"  {m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    mean_t = sum(r.llm_time_ms for r in runs) / len(runs)
                    row += f"  {mean_t:>10.0f}"
                else:
                    row += f"  {'n/a':>10}"
            lines.append(row)

        lines.append("\n  --- Mean Test Time (ms) ---")
        lines.append(header)
        lines.append(f"  {'-'*len(header)}")
        for m in methods:
            row = f"  {m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    mean_t = sum(r.test_time_ms for r in runs) / len(runs)
                    row += f"  {mean_t:>10.0f}"
                else:
                    row += f"  {'n/a':>10}"
            lines.append(row)

    # Amortized indexing cost
    if graph_build_time > 0:
        lines.append(f"\n  Graph build time: {graph_build_time:.1f}s")
        lines.append(f"  Amortized per task: {graph_build_time / n_tasks:.2f}s ({n_tasks} tasks)")
        lines.append("  Note: graph-based methods (bca*) require indexing; lexical methods (grep, bm25) do not.")

    return "\n".join(lines)


def format_edit_locality(results: list[EvalResult], budgets: list[int]) -> str:
    """Format edit locality: distance from edited line to mutated line.

    Smaller = more precise fix. A method that edits line 50 for a bug on line 52
    is more precise than one that rewrites lines 1-100.
    Excludes ceiling methods.
    """
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    query_types = sorted(set(r.query_type for r in grid_results))
    methods = sorted(set(r.method for r in grid_results))
    lines = []

    for qt in query_types:
        qt_results = [r for r in grid_results if r.query_type == qt]
        qt_label = qt.upper()

        lines.append(f"\n{'='*90}")
        lines.append(f"Edit Locality [{qt_label} queries]")
        lines.append("  (Mean distance in lines from edit to mutation. Lower = more precise.)")
        lines.append(f"{'='*90}")

        header = f"  {'Method':<20}" + "".join(f"  B={b:>10}" for b in budgets)

        lines.append("\n  --- Mean Edit Distance (lines, all attempts with known distance) ---")
        lines.append(header)
        lines.append(f"  {'-'*len(header)}")
        for m in methods:
            row = f"  {m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b and r.edit_distance_lines >= 0]
                if runs:
                    mean_d = sum(r.edit_distance_lines for r in runs) / len(runs)
                    row += f"  {mean_d:>10.1f}"
                else:
                    row += f"  {'---':>10}"
            lines.append(row)

        lines.append("\n  --- Context-Patch Overlap (fraction of context files referenced in patch) ---")
        lines.append(header)
        lines.append(f"  {'-'*len(header)}")
        for m in methods:
            row = f"  {m:<20}"
            for b in budgets:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    mean_o = sum(r.context_patch_overlap for r in runs) / len(runs)
                    row += f"  {mean_o:>10.2f}"
                else:
                    row += f"  {'n/a':>10}"
            lines.append(row)

    return "\n".join(lines)


def _collect_run_metadata(args, tasks, budgets, methods, results, query_types):
    """Collect metadata for reproducibility.

    Logs everything needed to reproduce the run: repo commit, tasks hash,
    model ID + fingerprint, seeds, method config.
    """
    import datetime
    import platform

    git_hash = "unknown"
    try:
        git_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if git_result.returncode == 0:
            git_hash = git_result.stdout.strip()
    except Exception:
        pass

    # Hash the tasks file for reproducibility verification
    tasks_hash = "unknown"
    try:
        tasks_path = Path(args.tasks_file)
        if tasks_path.exists():
            tasks_hash = hashlib.sha256(tasks_path.read_bytes()).hexdigest()
    except Exception:
        pass

    # Get target repo commits (one per unique repo)
    target_repo_commits = {}
    seen_repos = set()
    for t in tasks:
        rp = Path(t.repo_path).resolve() if t.repo_path else None
        if rp and str(rp) not in seen_repos:
            seen_repos.add(str(rp))
            try:
                repo_git = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=5,
                    cwd=rp,
                )
                if repo_git.returncode == 0:
                    target_repo_commits[rp.name] = repo_git.stdout.strip()
            except Exception:
                target_repo_commits[rp.name] = "unknown"
    target_repo_commit = target_repo_commits.get(
        next(iter(target_repo_commits), ""), "unknown"
    )

    total_pass = sum(1 for r in results if r.tests_passed)
    total_runs = len(results)

    # Hash the experiment spec for protocol integrity verification
    spec_hash = "unknown"
    spec_path = Path(__file__).parent / "experiment_spec.json"
    try:
        if spec_path.exists():
            spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    except Exception:
        pass

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_hash,
        "target_repo_commit": target_repo_commit,
        "experiment_spec_sha256": spec_hash,
        "python_version": sys.version,
        "platform": platform.platform(),
        "llm_provider": args.provider,
        "llm_model": args.model,
        "budgets": budgets,
        "methods": methods,
        "query_types": query_types,
        "ceiling_methods": sorted(_CEILING_METHODS),
        "num_tasks": len(tasks),
        "total_runs": total_runs,
        "total_pass": total_pass,
        "pass_rate": round(total_pass / total_runs, 4) if total_runs else 0,
        "tasks_file": args.tasks_file,
        "tasks_file_sha256": tasks_hash,
        "bootstrap_seed": 42,
        "bootstrap_n": 10000,
        "naive_random_seed_scheme": "(task_id, budget, query_type) hash",
        "llm_params": {
            "temperature": 0.0,
            "max_tokens": 4096,
            "seed": 42,
            "top_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
        "model_version_info": dict(_model_version_info),
        "retry_delays_seconds": _RETRY_DELAYS,
        "target_repo_commits": target_repo_commits,
        "bca_strategy_configs": {
            "bca_d1": {"strategy": "PRECISE", "max_depth": 1, "min_score": 0.3},
            "bca": {"strategy": "SMART", "max_depth": 3, "min_score": 0.1},
            "bca_d5": {"strategy": "THOROUGH", "max_depth": 5, "min_score": 0.05},
            "bca_no_closure": {"strategy": "SMART", "ablation": "no_closure"},
            "bca_no_scoring": {"strategy": "SMART", "ablation": "no_scoring"},
        },
        "repos": list(target_repo_commits.keys()),
    }


def main():
    parser = argparse.ArgumentParser(description="BCA end-to-end benchmark")
    parser.add_argument("--tasks-file", required=True, help="JSONL file with eval tasks")
    parser.add_argument(
        "--budgets", default="1000,4000,8000,10000",
        help="Comma-separated budget values",
    )
    parser.add_argument(
        "--methods",
        default="no_retrieval,naive_random,grep,bm25,vector,embedding,keyword_map,bca_d1,bca,bca_d5,bca_no_closure,bca_no_scoring,target_file",
        help="Comma-separated method names (default: all 13 methods for paper run)",
    )
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument(
        "--model", default="gpt-4o-mini-2024-07-18",
        help="LLM model. Default: gpt-4o-mini-2024-07-18 (arXiv). "
             "Conference upgrade: gpt-4o-2024-08-06",
    )
    parser.add_argument("--output-dir", default="paper/results", help="Output directory")
    parser.add_argument(
        "--query-types", default="exact,vague",
        help="Comma-separated query types (default: exact,vague for dual-query paper mode)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: 2 tasks, 2 budgets, 5 methods (~16 LLM calls)",
    )
    parser.add_argument(
        "--repos-dir", default=None,
        help="Base directory containing target repositories",
    )
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    query_types = [qt.strip() for qt in args.query_types.split(",")]

    if args.quick:
        budgets = [1000, 4000]
        methods = ["no_retrieval", "naive_random", "grep", "bca", "target_file"]
        query_types = ["exact"]

    llm_config = LLMConfig(provider=args.provider, model=args.model)
    if not llm_config.api_key:
        parser.error(
            "No API key found. Set the appropriate environment variable "
            "(e.g., ANTHROPIC_API_KEY or OPENAI_API_KEY)."
        )

    with open(args.tasks_file) as f:
        eval_fields = {f.name for f in EvalTask.__dataclass_fields__.values()}
        tasks = []
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Strip extra fields not in EvalTask
            filtered = {k: v for k, v in data.items() if k in eval_fields}
            tasks.append(EvalTask(**filtered))

    if args.repos_dir:
        repos_base = Path(args.repos_dir).resolve()
        for t in tasks:
            if t.repo_path and not Path(t.repo_path).is_absolute():
                t.repo_path = str(repos_base / t.repo_path)

    if args.quick:
        tasks = tasks[:2]

    # --- Verify repo commits match pinned expectations ---
    _verified_repos: set[str] = set()
    for t in tasks:
        if not t.commit or not t.repo_path:
            continue
        rp = str(Path(t.repo_path).resolve())
        if rp in _verified_repos:
            continue
        _verified_repos.add(rp)
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5, cwd=rp,
            )
            actual = head.stdout.strip() if head.returncode == 0 else "unknown"
            if not t.commit.startswith(actual[:len(t.commit)]) and actual != "unknown":
                print(f"ERROR: Repo {rp} is at {actual[:12]} but tasks expect {t.commit[:12]}")
                print(f"  Run: cd {rp} && git checkout {t.commit}")
                sys.exit(1)
            else:
                print(f"  Repo {Path(rp).name}: commit {actual[:12]} OK")
        except Exception as e:
            print(f"  WARNING: Could not verify commit for {rp}: {e}")

    # Verify and print experiment spec
    spec_path = Path(__file__).parent / "experiment_spec.json"
    if spec_path.exists():
        spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()[:16]
        print(f"\n  Experiment spec: {spec_path.name} (sha256:{spec_hash})")
    else:
        print("\n  WARNING: experiment_spec.json not found — protocol not frozen")

    total_runs = len(tasks) * len(budgets) * len(methods) * len(query_types)
    print(f"Benchmark: {len(tasks)} tasks, {len(budgets)} budgets, {len(methods)} methods, {len(query_types)} query types")
    print(f"Total runs: {total_runs}")
    print(f"LLM: {args.provider}/{args.model}")
    print(f"Methods: {methods}")
    print(f"Budgets: {budgets}")
    print(f"Query types: {query_types}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save pre-run config immediately (before any LLM calls).
    # If the run crashes, this file still records what was attempted.
    run_config = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tasks_file": str(Path(args.tasks_file).resolve()),
        "task_count": len(tasks),
        "task_ids": [t.task_id for t in tasks],
        "budgets": budgets,
        "methods": methods,
        "query_types": query_types,
        "provider": args.provider,
        "model": args.model,
        "output_dir": str(output_dir.resolve()),
        "experiment_spec_sha256": spec_hash if spec_path.exists() else None,
        "repo_commits": {
            str(Path(t.repo_path).name): t.commit
            for t in tasks if t.commit and t.repo_path
        },
        "total_runs": total_runs,
        "quick_mode": args.quick,
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"\n  Pre-run config saved to {output_dir / 'run_config.json'}")

    results, graph_build_times = asyncio.run(run_benchmark(tasks, budgets, methods, llm_config, output_dir, query_types))
    total_graph_build_time = sum(graph_build_times.values())

    # Save aggregate results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Save reproducibility metadata
    metadata = _collect_run_metadata(args, tasks, budgets, methods, results, query_types)
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Table 1: Pass@1 (basic) ---
    summary = format_results(results, budgets)
    print(summary)
    (output_dir / "summary.txt").write_text(summary)

    # --- Table 1a: Pass@1 per-repo breakdown ---
    per_repo = format_per_repo_results(results, tasks, budgets)
    if per_repo:
        print(per_repo)
        (output_dir / "per_repo_results.txt").write_text(per_repo)

    # --- Table 1b: Pass@1 with per-cell bootstrap CIs ---
    ci_summary = format_results_with_ci(results, budgets)
    print(ci_summary)
    (output_dir / "summary_with_ci.txt").write_text(ci_summary)

    # --- Ceiling probe: target_file (separate from main grid) ---
    ceiling_text = format_ceiling_probe(results, budgets)
    if ceiling_text:
        print(ceiling_text)
        (output_dir / "ceiling_probe.txt").write_text(ceiling_text)

    # --- Router (LOO-CV) — excludes ceiling methods ---
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    oracle = compute_oracle(grid_results, budgets)
    router = compute_router_loo(results, tasks, budgets)
    query_types_seen = sorted(set(r.query_type for r in grid_results))
    methods_seen = sorted(set(r.method for r in grid_results))

    router_lines = ["\n" + "=" * 70, "Oracle vs Router vs Best Single Method", "=" * 70]
    for qt in query_types_seen:
        qt_label = qt.upper()
        router_lines.append(f"\n  [{qt_label} queries]")
        router_lines.append(f"  {'Budget':<10} {'Best Single':>12} {'Router(LOO)':>12} {'Oracle':>10} {'Best Method':<20}")
        router_lines.append(f"  {'-'*65}")
        qt_results = [r for r in grid_results if r.query_type == qt]
        for b in budgets:
            # Find best single method at this (budget, qt)
            best_m, best_rate = "", 0
            for m in methods_seen:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    rate = sum(1 for r in runs if r.tests_passed) / len(runs)
                    if rate > best_rate:
                        best_rate, best_m = rate, m
            r_rate = router.get((qt, b), 0)
            o_rate = oracle.get((qt, b), 0)
            router_lines.append(
                f"  B={b:<7} {best_rate:>12.3f} {r_rate:>12.3f} {o_rate:>10.3f} {best_m:<20}"
            )
    router_text = "\n".join(router_lines)
    print(router_text)
    (output_dir / "router_analysis.txt").write_text(router_text)

    # --- Decomposition tables (mutation type × method, category × method) ---
    decomp = format_decomposition(results, tasks, budgets)
    print(decomp)
    (output_dir / "decomposition.txt").write_text(decomp)

    # --- Conditional bins (identifier density, hop distance, mutation size) ---
    cond_bins = format_conditional_bins(results, budgets)
    print(cond_bins)
    (output_dir / "conditional_bins.txt").write_text(cond_bins)

    # --- Paired bootstrap CI analysis (appendix) ---
    bootstrap_text = format_bootstrap_analysis(results, budgets)
    if bootstrap_text:
        print(bootstrap_text)
        (output_dir / "bootstrap_analysis.txt").write_text(bootstrap_text)

        # Save raw bootstrap CIs as JSON for downstream use (excludes ceiling methods)
        cis = bootstrap_paired_ci(grid_results, budgets)
        ci_json = {
            f"{ma}_vs_{mb}_B{b}_{qt}": {
                "method_a": ma, "method_b": mb,
                "budget": b, "query_type": qt,
                "mean_diff": round(d, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "significant": lo > 0 or hi < 0,
            }
            for (ma, mb, b, qt), (d, lo, hi) in cis.items()
        }
        with open(output_dir / "bootstrap_cis.json", "w") as f:
            json.dump(ci_json, f, indent=2)

    # --- Failure mode breakdown ---
    diagnosis = format_failure_diagnosis(results, budgets)
    if diagnosis:
        print(diagnosis)
        (output_dir / "failure_diagnosis.txt").write_text(diagnosis)

    # --- Retrieval quality metrics ---
    retrieval = format_retrieval_metrics(results, budgets)
    if retrieval:
        print(retrieval)
        (output_dir / "retrieval_metrics.txt").write_text(retrieval)

    # --- Patch quality metrics ---
    patch_q = format_patch_quality(results, budgets)
    if patch_q:
        print(patch_q)
        (output_dir / "patch_quality.txt").write_text(patch_q)

    # --- Latency & cost breakdown ---
    latency = format_latency_cost(results, budgets, graph_build_time=total_graph_build_time)
    if latency:
        print(latency)
        (output_dir / "latency_cost.txt").write_text(latency)

    # --- Edit locality & context-patch overlap ---
    locality = format_edit_locality(results, budgets)
    if locality:
        print(locality)
        (output_dir / "edit_locality.txt").write_text(locality)

    print(f"\nResults saved to {output_dir}/")
    print(f"Per-run artifacts in {output_dir}/<task_id>/<method>/<budget>/")
    print("\nOutput files:")
    print("  summary.txt              - Pass@1 table (main grid, no ceiling)")
    print("  per_repo_results.txt     - Pass@1 broken down by repository")
    print("  summary_with_ci.txt      - Pass@1 with 95% bootstrap CIs")
    print("  ceiling_probe.txt        - target_file ceiling probe")
    print("  router_analysis.txt      - Oracle vs Router vs Best Single")
    print("  decomposition.txt        - Pass@1 by mutation type and category")
    print("  conditional_bins.txt     - Pass@1 by identifier density, hops, mutation size")
    print("  failure_diagnosis.txt    - Failure mode breakdown")
    print("  retrieval_metrics.txt    - Target file hit rate, budget utilization")
    print("  patch_quality.txt        - Patch size and locality")
    print("  latency_cost.txt         - Latency breakdown + amortized indexing")
    print("  edit_locality.txt        - Edit distance + context-patch overlap")
    print("  bootstrap_analysis.txt   - Paired bootstrap CI details")
    print("  run_metadata.json        - Full reproducibility metadata")


if __name__ == "__main__":
    main()
