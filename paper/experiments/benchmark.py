"""End-to-end benchmark harness for BCA evaluation.

Takes coding tasks, assembles context with each method, calls an LLM,
applies the generated patch, runs tests, and records pass/fail.

This is the line between "implementation" and "paper": it produces the
pass@1 vs token-budget plots that constitute evidence.

Usage:
    python -m paper.experiments.benchmark \
        --tasks-file paper/experiments/eval_tasks.jsonl \
        --budgets 1000,2000,4000,8000 \
        --provider anthropic \
        --model claude-sonnet-4-5-20250929 \
        --output-dir paper/results/

Task JSONL format:
    {
        "task_id": "unique-id",
        "repo_path": "/path/to/repo",
        "repo_url": "https://github.com/...",  (optional, for cloning)
        "commit": "abc123",                     (optional, checkout before eval)
        "description": "Fix the bug in ...",
        "test_cmd": "python -m pytest tests/test_foo.py -x",
        "setup_cmd": "pip install -e .",        (optional)
        "timeout": 60                           (optional, test timeout in seconds)
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
    baseline_grep,
    baseline_repo_map,
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
    timeout: int = 60
    mutation: dict = field(default_factory=dict)  # {file, original, mutated}


@dataclass
class EvalResult:
    task_id: str
    method: str
    budget: int
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


# ---------------------------------------------------------------------------
# Context assembly methods
# ---------------------------------------------------------------------------

def assemble_bca(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """BCA context assembly. Returns (context_str, tokens, syms, files, time_ms)."""
    start = time.time()
    assembler = ContextAssembler(repo_path, graph, query)
    package = assembler.assemble(task=task, token_budget=budget, strategy=ContextStrategy.SMART)
    elapsed = (time.time() - start) * 1000
    return (
        package.render(),
        package.total_tokens,
        package.symbols_included,
        package.files_included,
        round(elapsed, 1),
    )


def assemble_bca_no_closure(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """BCA without dependency closure."""
    start = time.time()
    ablation = AblationConfig(dependency_closure=False)
    assembler = ContextAssembler(repo_path, graph, query, ablation=ablation)
    package = assembler.assemble(task=task, token_budget=budget, strategy=ContextStrategy.SMART)
    elapsed = (time.time() - start) * 1000
    return (
        package.render(),
        package.total_tokens,
        package.symbols_included,
        package.files_included,
        round(elapsed, 1),
    )


def assemble_bm25(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """BM25 symbol retrieval + greedy packing."""
    start = time.time()
    result = baseline_bm25(repo_path, task, budget, graph)
    # BM25 gives us symbol names; we need to render their source
    content_parts = []
    tokens_used = 0
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


def assemble_repo_map(
    repo_path: Path, task: str, budget: int, graph, query: GraphQuery,
) -> tuple[str, int, int, int, float]:
    """Repo map (structural summary + relevant file content)."""
    start = time.time()

    # Build structural map
    files_by_path: dict[str, list[dict]] = {}
    for node_id, data in graph.nodes(data=True):
        if data.get("type") != "symbol":
            continue
        fp = data.get("file_path", "")
        if fp:
            files_by_path.setdefault(fp, []).append(data)

    map_lines = []
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
    if map_tokens > budget:
        chars_allowed = int(budget * TokenEstimator.CHARS_PER_TOKEN)
        struct_map = struct_map[:chars_allowed]
        map_tokens = budget

    # Fill remaining budget with relevant file content
    remaining = budget - map_tokens
    keywords = set(re.findall(r"\b([A-Za-z_]\w{2,})\b", task))
    stop = {"the", "and", "for", "that", "this", "with", "from", "fix", "bug", "add"}
    keywords -= stop

    file_content = []
    content_tokens = 0
    files_used = set()
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
            ft = TokenEstimator.estimate(content)
            if content_tokens + ft > remaining:
                continue
            file_content.append(f"# {fp}\n{content}")
            content_tokens += ft
            files_used.add(fp)

    context = struct_map + "\n\n" + "\n\n".join(file_content) if file_content else struct_map
    elapsed = (time.time() - start) * 1000
    return (
        context,
        map_tokens + content_tokens,
        0,
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
        elapsed = (time.time() - start) * 1000
        return ("", 0, 0, 0, round(elapsed, 1))

    use_dense = os.environ.get("BENCHMARK_USE_DENSE", "")

    if use_dense:
        scores = _vector_score_dense(task, symbols)
    else:
        scores = _vector_score_tfidf(task, symbols)

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
        import numpy as np
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
        elapsed = (time.time() - start) * 1000
        return ("", 0, 0, 0, round(elapsed, 1))

    scores = _embedding_score_openai(task, symbols)

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


def _embedding_score_openai(query_text: str, symbols: list[dict]) -> list[float]:
    """Score symbols using OpenAI text-embedding-3-small."""
    import math

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("  WARNING: OPENAI_API_KEY not set, falling back to TF-IDF for embedding")
        return _vector_score_tfidf(query_text, symbols)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        texts = [s["text"] for s in symbols]
        # Batch embed (API limit ~2048 inputs per call)
        all_embeddings = []
        batch_size = 512
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in resp.data])

        # Embed query
        q_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query_text],
        )
        q_emb = q_resp.data[0].embedding

        # Cosine similarity
        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        return [cosine(q_emb, emb) for emb in all_embeddings]

    except Exception as e:
        print(f"  WARNING: OpenAI embedding failed ({e}), falling back to TF-IDF")
        return _vector_score_tfidf(query_text, symbols)


# All assembly methods
METHODS: dict[str, callable] = {
    "grep": assemble_grep,
    "bm25": assemble_bm25,
    "repo_map": assemble_repo_map,
    "vector": assemble_vector,
    "embedding": assemble_embedding,
    "bca": assemble_bca,
    "bca_no_closure": assemble_bca_no_closure,
}


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a coding assistant. You will be given source code context and a bug description.
Your job is to produce a SEARCH/REPLACE edit that fixes the bug.

Output format — one or more blocks like this:

```
FILE: path/to/file.py
SEARCH:
<exact lines from the buggy file>
REPLACE:
<corrected lines>
```

Rules:
- The SEARCH block must contain the EXACT lines from the file (copy-paste, including indentation).
- Only change the minimum lines needed to fix the bug.
- You may output multiple FILE/SEARCH/REPLACE blocks if the fix spans multiple files.
- Output ONLY the edit blocks, no explanation before or after."""


def build_prompt(context: str, task: str) -> str:
    """Build the user prompt from assembled context and task."""
    return f"""## Source Code Context

{context}

## Bug Report

{task}

## Fix

Produce SEARCH/REPLACE blocks to fix this bug."""


async def call_llm(
    provider, context: str, task: str,
) -> tuple[str, float, int, int]:
    """Call the LLM and return (response_text, time_ms, input_tokens, output_tokens)."""
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=build_prompt(context, task)),
    ]

    start = time.time()
    response: LLMResponse = await provider.complete(
        messages=messages,
        temperature=0.0,
        max_tokens=4096,
    )
    elapsed = (time.time() - start) * 1000

    input_tokens = response.usage.get("prompt_tokens", 0) or response.usage.get("input_tokens", 0)
    output_tokens = response.usage.get("completion_tokens", 0) or response.usage.get("output_tokens", 0)

    return (response.content, round(elapsed, 1), input_tokens, output_tokens)


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
    has_prefix = sum(1 for l in lines if prefix_re.match(l) or l.strip() == "")
    if has_prefix >= len(lines) * 0.5 and len(lines) > 0:
        stripped = []
        for l in lines:
            m = prefix_re.match(l)
            if m:
                stripped.append(l[m.end():])
            else:
                stripped.append(l)
        return "\n".join(stripped)
    return text


def extract_edits(llm_output: str) -> list[SearchReplaceEdit]:
    """Extract SEARCH/REPLACE edit blocks from LLM output."""
    edits: list[SearchReplaceEdit] = []

    # Strip code fences if present
    text = llm_output
    if "```" in text:
        # Remove outer code fences
        parts = text.split("```")
        # Take content between fences
        text = "\n".join(parts[1::2]) if len(parts) > 1 else text

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for FILE: marker
        if line.startswith("FILE:"):
            file_path = line[5:].strip()
            search_lines: list[str] = []
            replace_lines: list[str] = []
            i += 1

            # Find SEARCH:
            while i < len(lines) and not lines[i].strip().startswith("SEARCH:"):
                i += 1
            i += 1  # skip SEARCH: line

            # Collect search content until REPLACE:
            while i < len(lines) and not lines[i].strip().startswith("REPLACE:"):
                search_lines.append(lines[i])
                i += 1
            i += 1  # skip REPLACE: line

            # Collect replace content until next FILE: or end or ```
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped.startswith("FILE:") or stripped == "```":
                    break
                replace_lines.append(lines[i])
                i += 1

            search_text = "\n".join(search_lines)
            replace_text = "\n".join(replace_lines)

            # Strip line-number prefixes the LLM copies from context
            # Pattern: optional spaces, digits, space, pipe, space (e.g. "  13 | ")
            search_text = _strip_line_prefixes(search_text)
            replace_text = _strip_line_prefixes(replace_text)

            # Strip trailing empty lines
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


def apply_and_test(
    repo_path: Path,
    llm_output: str,
    test_cmd: str,
    timeout: int = 60,
) -> tuple[bool, str]:
    """Copy repo to tmpdir, apply search/replace edits, run tests."""
    edits = extract_edits(llm_output)
    if not edits:
        return False, "no edits extracted"

    with tempfile.TemporaryDirectory(prefix="bca_eval_") as tmpdir:
        work_dir = Path(tmpdir) / "repo"
        shutil.copytree(
            repo_path, work_dir,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", ".cegraph",
                "node_modules", ".venv", "venv",
            ),
        )

        # Apply search/replace edits
        applied = 0
        for edit in edits:
            # Normalize path — strip leading a/ or b/
            fp = edit.file_path
            for prefix in ("a/", "b/"):
                if fp.startswith(prefix):
                    fp = fp[len(prefix):]

            target = work_dir / fp
            if not target.exists():
                return False, f"file not found: {fp}"

            content = target.read_text()
            if edit.search in content:
                content = content.replace(edit.search, edit.replace, 1)
                target.write_text(content)
                applied += 1
            else:
                # Try with normalized whitespace (strip trailing spaces per line)
                search_norm = "\n".join(l.rstrip() for l in edit.search.splitlines())
                content_norm = "\n".join(l.rstrip() for l in content.splitlines())
                if search_norm in content_norm:
                    content = content_norm.replace(search_norm, edit.replace, 1)
                    target.write_text(content)
                    applied += 1
                else:
                    return False, f"search text not found in {fp}"

        if applied == 0:
            return False, "no edits applied"

        # Run tests
        try:
            test_result = subprocess.run(
                test_cmd.split(),
                capture_output=True,
                text=True,
                cwd=work_dir,
                timeout=timeout,
                env={**os.environ, "PYTHONPATH": str(work_dir / "src")},
            )
            passed = test_result.returncode == 0
            output = test_result.stdout[-2000:] + "\n" + test_result.stderr[-2000:]
            return passed, output.strip()
        except subprocess.TimeoutExpired:
            return False, "test timeout"
        except Exception as e:
            return False, f"test error: {e}"


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def _apply_mutation(repo_path: Path, mutation: dict) -> str | None:
    """Apply a mutation to the repo. Returns original content for restoration."""
    if not mutation:
        return None
    file_path = repo_path / mutation["file"]
    if not file_path.exists():
        return None
    original = file_path.read_text()
    mutated = original.replace(mutation["original"], mutation["mutated"], 1)
    if mutated == original:
        return None  # mutation string not found
    file_path.write_text(mutated)
    return original


def _restore_mutation(repo_path: Path, mutation: dict, original: str) -> None:
    """Restore original content after mutation."""
    file_path = repo_path / mutation["file"]
    file_path.write_text(original)


async def run_benchmark(
    tasks: list[EvalTask],
    budgets: list[int],
    methods: list[str],
    llm_config: LLMConfig,
    output_dir: Path,
) -> list[EvalResult]:
    """Run the full benchmark: mutate → context → LLM → patch → test → result.

    For each task with a mutation:
      1. Apply mutation to create buggy code
      2. Build graph from buggy code
      3. Assemble context from buggy code using each method
      4. Send context + bug description to LLM
      5. Apply LLM's patch to a temp copy of the buggy code
      6. Run tests — pass means the LLM fixed the bug
      7. Restore original code
    """
    provider = create_provider(llm_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[EvalResult] = []
    total_runs = len(tasks) * len(budgets) * len(methods)
    run_idx = 0

    for task in tasks:
        repo_path = Path(task.repo_path).resolve()

        print(f"\n{'='*60}")
        print(f"Task: {task.task_id}")
        print(f"Repo: {repo_path}")
        print(f"Description: {task.description[:80]}")
        print(f"{'='*60}")

        # Apply mutation to create buggy code
        original_content = None
        if task.mutation:
            original_content = _apply_mutation(repo_path, task.mutation)
            if original_content is None:
                print("  WARNING: mutation could not be applied, skipping task")
                continue
            print(f"  Mutation applied: {task.mutation['file']}")

        try:
            # Build graph from (possibly mutated) code
            builder = GraphBuilder()
            graph = builder.build_from_directory(repo_path)
            query = GraphQuery(graph)

            for budget in budgets:
                for method_name in methods:
                    run_idx += 1
                    print(f"\n  [{run_idx}/{total_runs}] {method_name} @ B={budget}")

                    method_fn = METHODS[method_name]

                    # 1. Assemble context from buggy code
                    try:
                        context, tokens_used, syms, files, asm_time = method_fn(
                            repo_path, task.description, budget, graph, query,
                        )
                    except Exception as e:
                        print(f"    assembly error: {e}")
                        all_results.append(EvalResult(
                            task_id=task.task_id, method=method_name, budget=budget,
                            tokens_used=0, symbols_selected=0, files_included=0,
                            assembly_time_ms=0, llm_time_ms=0,
                            llm_input_tokens=0, llm_output_tokens=0,
                            tests_passed=False, test_output="", patch="",
                            error=str(e),
                        ))
                        continue

                    print(f"    context: {tokens_used} tokens, {syms} symbols, {files} files ({asm_time}ms)")

                    # 2. Call LLM
                    try:
                        llm_response, llm_time, in_tok, out_tok = await call_llm(
                            provider, context, task.description,
                        )
                    except Exception as e:
                        print(f"    LLM error: {e}")
                        all_results.append(EvalResult(
                            task_id=task.task_id, method=method_name, budget=budget,
                            tokens_used=tokens_used, symbols_selected=syms,
                            files_included=files, assembly_time_ms=asm_time,
                            llm_time_ms=0, llm_input_tokens=0, llm_output_tokens=0,
                            tests_passed=False, test_output="", patch="",
                            error=str(e),
                        ))
                        continue

                    print(f"    LLM: {in_tok} in, {out_tok} out ({llm_time}ms)")

                    # 3. Extract edits
                    edits = extract_edits(llm_response)
                    patch = extract_patch(llm_response)
                    print(f"    edits: {len(edits)} blocks, {len(patch)} chars")

                    # 4. Apply edits to buggy code and test
                    passed, test_output = apply_and_test(
                        repo_path, llm_response, task.test_cmd, task.timeout,
                    )
                    status = "PASS" if passed else "FAIL"
                    print(f"    result: {status}")

                    result = EvalResult(
                        task_id=task.task_id,
                        method=method_name,
                        budget=budget,
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
                    )
                    all_results.append(result)

                    # Save per-run artifact
                    artifact_dir = output_dir / task.task_id / method_name / str(budget)
                    artifact_dir.mkdir(parents=True, exist_ok=True)
                    (artifact_dir / "context.txt").write_text(context[:50000])
                    (artifact_dir / "llm_response.txt").write_text(llm_response)
                    (artifact_dir / "patch.diff").write_text(patch)
                    (artifact_dir / "test_output.txt").write_text(test_output)
                    (artifact_dir / "result.json").write_text(
                        json.dumps(asdict(result), indent=2)
                    )
        finally:
            # Always restore original code
            if original_content is not None:
                _restore_mutation(repo_path, task.mutation, original_content)
                print(f"\n  Mutation restored: {task.mutation['file']}")

    return all_results


def format_results(results: list[EvalResult], budgets: list[int]) -> str:
    """Format results as pass@1 table."""
    methods = sorted(set(r.method for r in results))
    lines = []

    lines.append(f"\n{'='*70}")
    lines.append("Pass@1 by Method and Budget")
    lines.append(f"{'='*70}")

    header = f"{'Method':<20}" + "".join(f"  B={b:>5}" for b in budgets) + "    Avg"
    lines.append(header)
    lines.append("-" * 70)

    for m in methods:
        row = f"{m:<20}"
        method_passes = []
        for b in budgets:
            runs = [r for r in results if r.method == m and r.budget == b]
            if runs:
                pass_rate = sum(1 for r in runs if r.tests_passed) / len(runs)
                row += f"  {pass_rate:>5.2f}"
                method_passes.append(pass_rate)
            else:
                row += f"  {'n/a':>5}"
        if method_passes:
            row += f"  {sum(method_passes)/len(method_passes):>5.2f}"
        lines.append(row)

    lines.append("")
    lines.append(f"{'='*70}")
    lines.append("Mean Tokens Used / Budget")
    lines.append(f"{'='*70}")
    lines.append(header)
    lines.append("-" * 70)

    for m in methods:
        row = f"{m:<20}"
        for b in budgets:
            runs = [r for r in results if r.method == m and r.budget == b]
            if runs:
                mean_pct = sum(r.tokens_used / r.budget * 100 for r in runs) / len(runs)
                row += f"  {mean_pct:>4.0f}%"
            else:
                row += f"  {'n/a':>5}"
        lines.append(row)

    lines.append("")
    lines.append(f"{'='*70}")
    lines.append("Mean LLM Tokens (input + output)")
    lines.append(f"{'='*70}")
    lines.append(header)
    lines.append("-" * 70)

    for m in methods:
        row = f"{m:<20}"
        for b in budgets:
            runs = [r for r in results if r.method == m and r.budget == b]
            if runs:
                mean_total = sum(
                    r.llm_input_tokens + r.llm_output_tokens for r in runs
                ) / len(runs)
                row += f"  {mean_total:>5.0f}"
            else:
                row += f"  {'n/a':>5}"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="BCA end-to-end benchmark")
    parser.add_argument("--tasks-file", required=True, help="JSONL file with eval tasks")
    parser.add_argument(
        "--budgets", default="1000,2000,4000,8000",
        help="Comma-separated budget values",
    )
    parser.add_argument(
        "--methods",
        default="grep,bm25,vector,embedding,repo_map,bca,bca_no_closure",
        help="Comma-separated method names",
    )
    parser.add_argument("--provider", default="openai", help="LLM provider")
    parser.add_argument("--model", default="gpt-4o", help="LLM model")
    parser.add_argument("--output-dir", default="paper/results", help="Output directory")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test: 2 tasks, 2 budgets, 3 methods (6 LLM calls)",
    )
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]

    if args.quick:
        budgets = [2000, 4000]
        methods = ["grep", "bm25", "bca"]

    llm_config = LLMConfig(provider=args.provider, model=args.model)
    if not llm_config.api_key:
        parser.error(
            f"No API key found. Set the appropriate environment variable "
            f"(e.g., ANTHROPIC_API_KEY or OPENAI_API_KEY)."
        )

    with open(args.tasks_file) as f:
        tasks = [EvalTask(**json.loads(line)) for line in f if line.strip()]

    if args.quick:
        tasks = tasks[:2]

    print(f"Benchmark: {len(tasks)} tasks, {len(budgets)} budgets, {len(methods)} methods")
    print(f"Total runs: {len(tasks) * len(budgets) * len(methods)}")
    print(f"LLM: {args.provider}/{args.model}")
    print(f"Methods: {methods}")
    print(f"Budgets: {budgets}")

    output_dir = Path(args.output_dir)
    results = asyncio.run(run_benchmark(tasks, budgets, methods, llm_config, output_dir))

    # Save aggregate results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    summary = format_results(results, budgets)
    print(summary)
    (output_dir / "summary.txt").write_text(summary)

    print(f"\nResults saved to {output_dir}/")
    print(f"Per-run artifacts in {output_dir}/<task_id>/<method>/<budget>/")


if __name__ == "__main__":
    main()
