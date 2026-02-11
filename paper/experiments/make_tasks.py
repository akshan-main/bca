"""Generate evaluation tasks by introducing known bugs into the codebase.

Each task:
  1. Has a specific bug introduced via a patch (the "mutation")
  2. Has a test command that FAILS with the bug and PASSES without it
  3. Has a natural-language description of the bug symptom
  4. Has a line_num anchor for precise mutation application

The benchmark harness gives each method the buggy code + description,
asks the LLM to produce a fix, and checks if the fix makes the tests pass.

Usage:
    python -m paper.experiments.make_tasks --repo /path/to/repo --output paper/experiments/eval_tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

# Each mutation is:
#   file: relative path to the file to mutate
#   line_num: 1-based line number where the original string starts
#   original: exact string to find (must match the line at line_num)
#   mutated: replacement string (introduces the bug)
#   test_cmd: test command that catches this bug
#   description: what a developer would report as a bug symptom

MUTATIONS = [
    {
        "task_id": "find-symbol-partial-match",
        "file": "src/cegraph/graph/query.py",
        "line_num": 62,
        "original": "if name.lower() in key.lower():",
        "mutated": "if name.lower() == key.lower():",
        "test_cmd": "python -m pytest tests/test_graph.py::TestGraphQuery::test_find_symbol_partial -x",
        "description": "GraphQuery.find_symbol no longer finds symbols by partial name match — only exact matches work.",
        "vague_description": "Searching for symbols only works when I type the full exact name. Substring searches return nothing.",
    },
    {
        "task_id": "who-calls-wrong-edge",
        "file": "src/cegraph/graph/query.py",
        "line_num": 141,
        "original": 'if edge_data.get("kind") != "calls":',
        "mutated": 'if edge_data.get("kind") != "contains":',
        "test_cmd": "python -m pytest tests/test_graph.py::TestGraphQuery::test_who_calls -x",
        "description": "GraphQuery.who_calls returns containment relationships instead of call relationships — asking who calls a function returns its parent class instead.",
        "vague_description": "When I ask 'who calls this function', the results show parent classes instead of actual callers. The call graph seems wrong.",
    },
    {
        "task_id": "builder-no-reset",
        "file": "src/cegraph/graph/builder.py",
        "line_num": 49,
        "original": "        # Reset state so reusing a builder doesn't accumulate stale data\n        self.graph = nx.DiGraph()\n        self._file_hashes = {}\n        self._unresolved = []",
        "mutated": "        # Reset state so reusing a builder doesn't accumulate stale data\n        # self.graph = nx.DiGraph()\n        # self._file_hashes = {}\n        # self._unresolved = []",
        "test_cmd": "python -m pytest tests/test_graph.py::TestGraphBuilderReuse -x",
        "description": "GraphBuilder accumulates stale nodes and edges when build_from_directory is called multiple times — the second build has duplicate symbols from the first.",
        "vague_description": "Re-indexing the project creates duplicate entries. The graph keeps growing with stale data from previous runs.",
    },
    {
        "task_id": "file-symbols-wrong-node",
        "file": "src/cegraph/graph/query.py",
        "line_num": 233,
        "original": 'file_node = f"file::{file_path}"',
        "mutated": 'file_node = f"{file_path}"',
        "test_cmd": "python -m pytest tests/test_graph.py::TestGraphQuery::test_get_file_symbols -x",
        "description": "GraphQuery.get_file_symbols always returns an empty list — it cannot find any file nodes in the graph.",
        "vague_description": "Listing symbols for a specific file always returns empty, even though the file is definitely indexed.",
    },
    {
        "task_id": "context-strategy-enum",
        "file": "src/cegraph/context/models.py",
        "line_num": 14,
        "original": 'SMART = "smart"',
        "mutated": 'SMART = "balanced"',
        "test_cmd": "python -m pytest tests/test_context.py::TestContextStrategy -x",
        "description": 'ContextStrategy.SMART has the wrong string value — code that checks strategy == "smart" fails silently.',
        "vague_description": "The default context assembly strategy seems to be silently ignored. It behaves the same regardless of which strategy I pick.",
    },
    {
        "task_id": "detect-language-python",
        "file": "src/cegraph/parser/models.py",
        "line_num": 87,
        "original": '".py": "python",',
        "mutated": '".py": "python3",',
        "test_cmd": "python -m pytest tests/test_parser.py::TestLanguageDetection -x",
        "description": 'detect_language returns "python3" instead of "python" for .py files, causing the parser to fail to select the correct Python parser.',
        "vague_description": "Python files are not being parsed at all. The indexer skips every .py file as if it doesn't recognize the language.",
    },
    {
        "task_id": "config-default-provider",
        "file": "src/cegraph/config.py",
        "line_num": 21,
        "original": 'provider: str = "anthropic"',
        "mutated": 'provider: str = "openai"',
        "test_cmd": "python -m pytest tests/test_config.py::TestConfig::test_default_config -x",
        "description": "Default LLM provider is set to openai instead of anthropic, breaking configurations that rely on the anthropic default.",
        "vague_description": "Fresh installs default to the wrong LLM provider. Users have to manually override the config even though the docs say it works out of the box.",
    },
    {
        "task_id": "config-max-iterations",
        "file": "src/cegraph/config.py",
        "line_num": 44,
        "original": "max_iterations: int = 15",
        "mutated": "max_iterations: int = 150",
        "test_cmd": "python -m pytest tests/test_config.py::TestConfig::test_default_config -x",
        "description": "Default agent max_iterations is set to 150 instead of 15, allowing runaway agent loops that waste API credits.",
        "vague_description": "The agent runs for way too long and burns through API credits. It seems like there's no reasonable iteration cap by default.",
    },
    {
        "task_id": "mcp-server-name",
        "file": "src/cegraph/mcp/server.py",
        "line_num": 36,
        "original": 'SERVER_NAME = "cegraph"',
        "mutated": 'SERVER_NAME = "codesight"',
        "test_cmd": "python -m pytest tests/test_mcp.py -x",
        "description": "MCP server reports its name as 'codesight' instead of 'cegraph', breaking MCP client discovery.",
        "vague_description": "MCP clients can't find the server. The server seems to advertise itself under the wrong name.",
    },
    {
        "task_id": "entity-extract-camelcase",
        "file": "src/cegraph/context/engine.py",
        "line_num": 323,
        "original": r'r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b"',
        "mutated": r'r"\b([A-Z][a-z]+)\b"',
        "test_cmd": "python -m pytest tests/test_context.py::TestCAGEngine::test_extract_entities -x",
        "description": "Entity extraction regex matches single capitalized words instead of CamelCase identifiers, flooding the seed set with common words.",
        "vague_description": "Context assembly returns too many irrelevant symbols. It seems to pick up random common words as seeds instead of actual code identifiers.",
    },
    {
        "task_id": "graph-structure-path-prefix",
        "file": "src/cegraph/graph/query.py",
        "line_num": 256,
        "original": "if path_prefix and not file_path.startswith(path_prefix):",
        "mutated": "if path_prefix and file_path.startswith(path_prefix):",
        "test_cmd": "python -m pytest tests/test_graph.py::TestGraphQuery::test_get_structure -x",
        "description": "get_structure excludes files that match the path prefix instead of including them — the filter logic is inverted.",
        "vague_description": "Filtering the codebase structure by directory path returns the opposite of what it should — files outside the directory appear instead of files inside.",
    },
]


def preflight_check(repo_path: Path) -> list[str]:
    """Verify mutation integrity before running anything.

    Checks:
      1. Unique task IDs
      2. Target files exist
      3. Original string appears exactly once in target file (or at line_num)
      4. line_num anchor matches the original string

    Returns list of error messages (empty = all good).
    """
    errors = []
    seen_ids = set()

    for mutation in MUTATIONS:
        task_id = mutation["task_id"]

        # Unique task IDs
        if task_id in seen_ids:
            errors.append(f"{task_id}: duplicate task_id")
        seen_ids.add(task_id)

        # File exists
        file_path = repo_path / mutation["file"]
        if not file_path.exists():
            errors.append(f"{task_id}: file not found: {mutation['file']}")
            continue

        content = file_path.read_text()
        lines = content.splitlines()
        original = mutation["original"]
        first_line = original.split("\n")[0]

        # Check line_num anchor
        line_num = mutation.get("line_num")
        if line_num:
            line_idx = line_num - 1
            if line_idx < 0 or line_idx >= len(lines):
                errors.append(f"{task_id}: line_num {line_num} out of range (file has {len(lines)} lines)")
            elif first_line.strip() not in lines[line_idx]:
                errors.append(
                    f"{task_id}: line_num {line_num} doesn't match original. "
                    f"Expected '{first_line.strip()}', got '{lines[line_idx].strip()}'"
                )

        # Check uniqueness (for single-line mutations)
        if "\n" not in original:
            occurrences = content.count(original)
            if occurrences == 0:
                errors.append(f"{task_id}: original string not found in {mutation['file']}")
            elif occurrences > 1 and not line_num:
                errors.append(
                    f"{task_id}: original appears {occurrences} times in {mutation['file']} "
                    f"and no line_num anchor — ambiguous"
                )

    return errors


def verify_mutations(repo_path: Path) -> tuple[list[dict], list[str]]:
    """Verify each mutation is detectable by its test command.

    Returns (valid_tasks, validation_errors).
    """
    valid_tasks = []
    validation_errors = []

    for mutation in MUTATIONS:
        task_id = mutation["task_id"]
        file_path = repo_path / mutation["file"]

        if not file_path.exists():
            msg = f"SKIP {task_id}: file not found: {mutation['file']}"
            print(f"  {msg}")
            validation_errors.append(msg)
            continue

        content = file_path.read_text()
        original = mutation["original"]

        if original not in content:
            msg = f"SKIP {task_id}: original string not found in {mutation['file']}"
            print(f"  {msg}")
            validation_errors.append(msg)
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
            msg = f"SKIP {task_id}: test already fails on clean code"
            print(f"  {msg}")
            validation_errors.append(msg)
            continue

        # Apply mutation using line-anchored replacement
        mutated_content = _apply_mutation_to_content(content, mutation)
        if mutated_content is None:
            msg = f"SKIP {task_id}: could not apply mutation"
            print(f"  {msg}")
            validation_errors.append(msg)
            continue

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
                msg = f"SKIP {task_id}: test still passes with mutation (not detectable)"
                print(f"  {msg}")
                validation_errors.append(msg)
                continue

            print(f"  OK   {task_id}: mutation detected by test")
            valid_tasks.append(mutation)
        finally:
            # Restore original
            file_path.write_text(content)

    return valid_tasks, validation_errors


def _apply_mutation_to_content(content: str, mutation: dict) -> str | None:
    """Apply a mutation to file content, using line_num anchor if available."""
    original = mutation["original"]
    replacement = mutation["mutated"]
    line_num = mutation.get("line_num")

    if line_num and "\n" not in original:
        # Single-line, line-anchored replacement
        lines = content.splitlines()
        line_idx = line_num - 1
        if 0 <= line_idx < len(lines) and original.strip() in lines[line_idx]:
            lines[line_idx] = lines[line_idx].replace(original, replacement, 1)
            return "\n".join(lines) + ("\n" if content.endswith("\n") else "")

    # Fallback: first-match replacement
    if original in content:
        return content.replace(original, replacement, 1)

    return None


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
                "vague_description": mutation.get("vague_description", ""),
                "test_cmd": mutation["test_cmd"],
                "mutation": {
                    "file": mutation["file"],
                    "original": mutation["original"],
                    "mutated": mutation["mutated"],
                    "line_num": mutation.get("line_num"),
                },
            }
            f.write(json.dumps(task) + "\n")


def write_report(
    repo_path: Path,
    valid: list[dict],
    validation_errors: list[str],
    preflight_errors: list[str],
    output_file: Path,
) -> None:
    """Write discovery/validation report JSON."""
    report = {
        "repo_path": str(repo_path.resolve()),
        "mutations_defined": len(MUTATIONS),
        "mutations_valid": len(valid),
        "valid_task_ids": [m["task_id"] for m in valid],
        "preflight_errors": preflight_errors,
        "validation_errors": validation_errors,
        "curation": "manual — 11 hand-crafted mutations targeting known code patterns",
    }
    report_file = output_file.parent / "eval_tasks_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {report_file}")


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
    parser.add_argument(
        "--preflight-only", action="store_true",
        help="Only run preflight checks, don't verify against tests",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    print(f"Repository: {repo_path}")
    print(f"Mutations defined: {len(MUTATIONS)}")

    # Preflight checks
    print("\nPreflight checks...")
    preflight_errors = preflight_check(repo_path)
    if preflight_errors:
        for err in preflight_errors:
            print(f"  FAIL: {err}")
        print(f"\n{len(preflight_errors)} preflight error(s)")
        if args.preflight_only:
            return
    else:
        print("  All preflight checks passed")

    if args.preflight_only:
        return

    # Verify mutations
    validation_errors = []
    if args.verify:
        print("\nVerifying mutations...")
        valid, validation_errors = verify_mutations(repo_path)
    else:
        valid = list(MUTATIONS)

    print(f"\nValid tasks: {len(valid)}/{len(MUTATIONS)}")

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_eval_tasks(repo_path, valid, output_file)
    print(f"Written to {output_file}")

    write_report(repo_path, valid, validation_errors, preflight_errors, output_file)


if __name__ == "__main__":
    main()
