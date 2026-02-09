"""Generate evaluation tasks by introducing known bugs into the codebase.

Each task:
  1. Has a specific bug introduced via a patch (the "mutation")
  2. Has a test command that FAILS with the bug and PASSES without it
  3. Has a natural-language description of the bug symptom

The benchmark harness gives each method the buggy code + description,
asks the LLM to produce a fix, and checks if the fix makes the tests pass.

Usage:
    python -m paper.experiments.make_tasks --repo /path/to/repo --output paper/experiments/eval_tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import shutil
from pathlib import Path

# Each mutation is:
#   file: relative path to the file to mutate
#   original: exact string to find
#   mutated: replacement string (introduces the bug)
#   test_cmd: test command that catches this bug
#   description: what a developer would report as a bug symptom

MUTATIONS = [
    # --- Verified detectable (tests catch these) ---
    {
        "task_id": "find-symbol-partial-match",
        "file": "src/cegraph/graph/query.py",
        "original": "if name.lower() in key.lower():",
        "mutated": "if name.lower() == key.lower():",
        "test_cmd": "python -m pytest tests/test_graph.py::TestGraphQuery::test_find_symbol_partial -x",
        "description": "GraphQuery.find_symbol no longer finds symbols by partial name match — only exact matches work.",
    },
    {
        "task_id": "who-calls-wrong-edge",
        "file": "src/cegraph/graph/query.py",
        "original": 'if edge_data.get("kind") != "calls":\n                    continue',
        "mutated": 'if edge_data.get("kind") != "contains":\n                    continue',
        "test_cmd": "python -m pytest tests/test_graph.py::TestGraphQuery::test_who_calls -x",
        "description": "GraphQuery.who_calls returns containment relationships instead of call relationships — asking who calls a function returns its parent class instead.",
    },
    {
        "task_id": "builder-no-reset",
        "file": "src/cegraph/graph/builder.py",
        "original": "        # Reset state so reusing a builder doesn't accumulate stale data\n        self.graph = nx.DiGraph()\n        self._file_hashes = {}\n        self._unresolved = []",
        "mutated": "        # Reset state so reusing a builder doesn't accumulate stale data\n        # self.graph = nx.DiGraph()\n        # self._file_hashes = {}\n        # self._unresolved = []",
        "test_cmd": "python -m pytest tests/test_graph.py::TestGraphBuilderReuse -x",
        "description": "GraphBuilder accumulates stale nodes and edges when build_from_directory is called multiple times — the second build has duplicate symbols from the first.",
    },
    {
        "task_id": "file-symbols-wrong-node",
        "file": "src/cegraph/graph/query.py",
        "original": 'file_node = f"file::{file_path}"',
        "mutated": 'file_node = f"{file_path}"',
        "test_cmd": "python -m pytest tests/test_graph.py::TestGraphQuery::test_get_file_symbols -x",
        "description": "GraphQuery.get_file_symbols always returns an empty list — it cannot find any file nodes in the graph.",
    },
    {
        "task_id": "context-strategy-enum",
        "file": "src/cegraph/context/models.py",
        "original": 'SMART = "smart"',
        "mutated": 'SMART = "balanced"',
        "test_cmd": "python -m pytest tests/test_context.py::TestContextStrategy -x",
        "description": 'ContextStrategy.SMART has the wrong string value — code that checks strategy == "smart" fails silently.',
    },
    # --- Additional mutations targeting tight assertions ---
    {
        "task_id": "detect-language-python",
        "file": "src/cegraph/parser/models.py",
        "original": '".py": "python",',
        "mutated": '".py": "python3",',
        "test_cmd": "python -m pytest tests/test_parser.py::TestLanguageDetection -x",
        "description": 'detect_language returns "python3" instead of "python" for .py files, causing the parser to fail to select the correct Python parser.',
    },
    {
        "task_id": "config-default-provider",
        "file": "src/cegraph/config.py",
        "original": 'provider: str = "anthropic"',
        "mutated": 'provider: str = "openai"',
        "test_cmd": "python -m pytest tests/test_config.py::TestConfig::test_default_config -x",
        "description": "Default LLM provider is set to openai instead of anthropic, breaking configurations that rely on the anthropic default.",
    },
    {
        "task_id": "config-max-iterations",
        "file": "src/cegraph/config.py",
        "original": "max_iterations: int = 15",
        "mutated": "max_iterations: int = 150",
        "test_cmd": "python -m pytest tests/test_config.py::TestConfig::test_default_config -x",
        "description": "Default agent max_iterations is set to 150 instead of 15, allowing runaway agent loops that waste API credits.",
    },
    {
        "task_id": "mcp-server-name",
        "file": "src/cegraph/mcp/server.py",
        "original": '"name": "cegraph"',
        "mutated": '"name": "codesight"',
        "test_cmd": "python -m pytest tests/test_mcp.py -x",
        "description": "MCP server reports its name as 'codesight' instead of 'cegraph', breaking MCP client discovery.",
    },
    {
        "task_id": "entity-extract-camelcase",
        "file": "src/cegraph/context/engine.py",
        "original": 'r"([A-Z][a-z]+(?:[A-Z][a-z]+)+)"',
        "mutated": 'r"([A-Z][a-z]+)"',
        "test_cmd": "python -m pytest tests/test_context.py::TestCAGEngine::test_extract_entities -x",
        "description": "Entity extraction regex matches single capitalized words instead of CamelCase identifiers, flooding the seed set with common words.",
    },
    {
        "task_id": "search-empty-query",
        "file": "src/cegraph/search/hybrid.py",
        "original": "if not query or not query.strip():",
        "mutated": "if not query:",
        "test_cmd": "python -m pytest tests/test_search.py -x",
        "description": "Search does not handle whitespace-only queries — passing '   ' to search_symbols causes an unfiltered full-table scan.",
    },
    {
        "task_id": "graph-structure-path-prefix",
        "file": "src/cegraph/graph/query.py",
        "original": "if path_prefix and not file_path.startswith(path_prefix):",
        "mutated": "if path_prefix and file_path.startswith(path_prefix):",
        "test_cmd": "python -m pytest tests/test_graph.py::TestGraphQuery::test_get_structure -x",
        "description": "get_structure excludes files that match the path prefix instead of including them — the filter logic is inverted.",
    },
]


def verify_mutations(repo_path: Path) -> list[dict]:
    """Verify each mutation is detectable by its test command."""
    valid_tasks = []

    for mutation in MUTATIONS:
        task_id = mutation["task_id"]
        file_path = repo_path / mutation["file"]

        if not file_path.exists():
            print(f"  SKIP {task_id}: file not found: {mutation['file']}")
            continue

        content = file_path.read_text()
        if mutation["original"] not in content:
            print(f"  SKIP {task_id}: original string not found in {mutation['file']}")
            continue

        # Verify the test passes on clean code
        clean_result = subprocess.run(
            mutation["test_cmd"].split(),
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=30,
        )
        if clean_result.returncode != 0:
            print(f"  SKIP {task_id}: test already fails on clean code")
            continue

        # Apply mutation, verify test fails
        mutated_content = content.replace(mutation["original"], mutation["mutated"], 1)
        file_path.write_text(mutated_content)

        try:
            mutated_result = subprocess.run(
                mutation["test_cmd"].split(),
                capture_output=True,
                text=True,
                cwd=repo_path,
                timeout=30,
            )
            if mutated_result.returncode == 0:
                print(f"  SKIP {task_id}: test still passes with mutation (not detectable)")
                continue

            print(f"  OK   {task_id}: mutation detected by test")
            valid_tasks.append(mutation)
        finally:
            # Restore original
            file_path.write_text(content)

    return valid_tasks


def write_eval_tasks(
    repo_path: Path,
    mutations: list[dict],
    output_file: Path,
) -> None:
    """Write evaluation tasks as JSONL."""
    with open(output_file, "w") as f:
        for mutation in mutations:
            task = {
                "task_id": mutation["task_id"],
                "repo_path": str(repo_path.resolve()),
                "description": mutation["description"],
                "test_cmd": mutation["test_cmd"],
                "mutation": {
                    "file": mutation["file"],
                    "original": mutation["original"],
                    "mutated": mutation["mutated"],
                },
            }
            f.write(json.dumps(task) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate eval tasks from mutations")
    parser.add_argument("--repo", default=".", help="Path to repository")
    parser.add_argument(
        "--output",
        default="paper/experiments/eval_tasks.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Verify each mutation is detectable (default: True)",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    print(f"Repository: {repo_path}")
    print(f"Mutations defined: {len(MUTATIONS)}")

    if args.verify:
        print("\nVerifying mutations...")
        valid = verify_mutations(repo_path)
    else:
        valid = MUTATIONS

    print(f"\nValid tasks: {len(valid)}/{len(MUTATIONS)}")

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_eval_tasks(repo_path, valid, output_file)
    print(f"Written to {output_file}")


if __name__ == "__main__":
    main()
