#!/usr/bin/env python3
"""Capture tracebacks from mutated code and generate dev_report_description.

For each task in eval_tasks_full.jsonl:
  1. Apply the mutation (introduce the bug)
  2. Run test_cmd on the buggy code
  3. Capture the full pytest output (stdout + stderr)
  4. Extract: failing test name, traceback, assertion error
  5. Generate a dev_report_description — what a developer would see
  6. Revert the mutation (byte-identical)

The dev_report_description is a "developer report" tier:
  - Contains: failing test name, error type, assertion message, traceback
  - Does NOT contain: line numbers, file paths with line refs, mutation type/operator
  - Simulates: a developer running `pytest` and pasting the failure output

Usage:
    python -m paper.experiments.capture_tracebacks \
        --tasks paper/experiments/eval_tasks_full.jsonl \
        --output paper/experiments/eval_tasks_full.jsonl

    # Dry run (don't write output):
    python -m paper.experiments.capture_tracebacks \
        --tasks paper/experiments/eval_tasks_full.jsonl --dry-run

    # Single task:
    python -m paper.experiments.capture_tracebacks \
        --tasks paper/experiments/eval_tasks_full.jsonl \
        --task-id usage-total-tokens-math --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Mutation apply / revert (reused from benchmark.py, byte-identical)
# ---------------------------------------------------------------------------

def _apply_mutation(repo_path: Path, mutation: dict) -> tuple[str, str] | None:
    """Apply a mutation to the repo. Returns (original_content, sha256_hash)."""
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

    line_num = mutation.get("line_num")
    if line_num and "\n" not in mut_original:
        lines = original.splitlines()
        line_idx = line_num - 1
        if 0 <= line_idx < len(lines) and mut_original.strip() in lines[line_idx]:
            lines[line_idx] = lines[line_idx].replace(mut_original, mut_replacement, 1)
            mutated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
            file_path.write_text(mutated)
            return (original, original_hash)

    occurrences = original.count(mut_original)
    if occurrences == 0:
        return None
    mutated = original.replace(mut_original, mut_replacement, 1)
    file_path.write_text(mutated)
    return (original, original_hash)


def _restore_mutation(repo_path: Path, mutation: dict, original: str, expected_hash: str) -> None:
    """Restore original content after mutation, with byte-identical verification."""
    file_path = repo_path / mutation["file"]
    file_path.write_text(original)
    actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"FATAL: Mutation revert failed for {mutation['file']}. "
            f"Expected hash {expected_hash[:12]}, got {actual_hash[:12]}"
        )


# ---------------------------------------------------------------------------
# Test runner — captures FULL output (not truncated)
# ---------------------------------------------------------------------------

def run_tests_full(test_cmd: str, repo_path: Path, timeout: int = 120) -> tuple[int, str]:
    """Run test command and return (returncode, full_output)."""
    try:
        result = subprocess.run(
            test_cmd.split(),
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except OSError as e:
        return -2, str(e)


# ---------------------------------------------------------------------------
# Traceback extraction and description generation
# ---------------------------------------------------------------------------

# Regex to strip file-path line numbers from tracebacks
# Matches: File "/path/to/file.py", line 42, in func_name
_LINE_REF_RE = re.compile(
    r'File "([^"]+)", line (\d+), in (.+)',
)

# Regex to strip raw line numbers from pytest short summary
_PYTEST_LINE_RE = re.compile(r':(\d+):')


def _extract_failing_tests(output: str) -> list[str]:
    """Extract failing test names from pytest output.

    Looks for patterns like:
      FAILED tests/test_foo.py::test_bar - AssertionError: ...
      FAILED tests/test_foo.py::TestClass::test_method
      ___ test_name ___  (underscored header in FAILURES section)
    """
    tests = []
    for line in output.splitlines():
        stripped = line.strip()
        # Strategy 1: pytest short test summary: FAILED tests/...::test_name
        if stripped.startswith("FAILED "):
            rest = stripped[7:].strip()
            if " - " in rest:
                test_path = rest.split(" - ", 1)[0].strip()
            else:
                test_path = rest.strip()
            tests.append(test_path)

    # Strategy 2: If no FAILED lines, extract from underscored test headers
    # e.g.: ___ test_digest_auth_with_401_nonce_counting ___
    if not tests:
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("_") and stripped.endswith("_"):
                # Extract test name from between underscores
                inner = stripped.strip("_ ").strip()
                if inner and not any(kw in inner for kw in ["FAILURES", "ERRORS"]):
                    tests.append(inner)

    return tests


def _extract_traceback_block(output: str) -> str:
    """Extract the relevant traceback from pytest output.

    Captures from the first 'FAILURES' or 'ERROR' section to the next section
    or end of output.
    """
    lines = output.splitlines()
    in_failure = False
    tb_lines: list[str] = []

    for i, line in enumerate(lines):
        # Start capturing at FAILURES section or error section
        if "FAILURES" in line and "=" in line:
            in_failure = True
            continue
        # Stop at next section header (short test summary, warnings, etc.)
        if in_failure and line.strip().startswith("=") and any(
            kw in line for kw in ["short test summary", "warnings summary",
                                   "passed", "failed", "error"]
        ):
            break
        if in_failure:
            tb_lines.append(line)

    if not tb_lines:
        # Fallback: look for "Traceback (most recent call last):" blocks
        for i, line in enumerate(lines):
            if "Traceback (most recent call last):" in line:
                in_failure = True
            if in_failure:
                tb_lines.append(line)
                # End after the error line (not indented, not "File")
                if (len(tb_lines) > 2 and not line.startswith(" ")
                    and not line.startswith("Traceback")):
                    break

    return "\n".join(tb_lines)


def _extract_error_type_and_message(output: str) -> tuple[str, str]:
    """Extract the error type and message from pytest output.

    Returns (error_type, error_message).
    Handles:
      - Standard: `AssertionError: assert x == y`
      - Pytest E lines: `E       AssertionError: assert (1 + 1) == 1`
      - Pytest Failed: `E       Failed: DID NOT RAISE <class '...'>`
      - Short summary: `FAILED test_foo.py::test - AssertionError: ...`
    """
    lines = output.splitlines()

    # Strategy 1: Look for "E" prefix lines (pytest error detail lines)
    e_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("E "):
            e_lines.append(stripped[2:].strip())

    if e_lines:
        first_e = e_lines[0]
        # Check for "ErrorType: message" pattern
        match = re.match(r'([\w.]+(?:Error|Exception|Warning)):\s*(.*)', first_e)
        if match:
            return match.group(1), match.group(2)
        # Check for "Failed: DID NOT RAISE" pattern
        match = re.match(r'Failed:\s*(.*)', first_e)
        if match:
            return "Failed", match.group(1)
        # Fallback: first E-line is the error message
        return "AssertionError", first_e

    # Strategy 2: Look in short test summary line
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("FAILED ") and " - " in stripped:
            _, msg = stripped.split(" - ", 1)
            match = re.match(r'([\w.]+(?:Error|Exception|Warning)):\s*(.*)', msg)
            if match:
                return match.group(1), match.group(2)
            return "", msg

    # Strategy 3: Look for bare exception lines
    for line in reversed(lines):
        stripped = line.strip()
        match = re.match(r'([\w.]+(?:Error|Exception|Warning)):\s*(.*)', stripped)
        if match:
            return match.group(1), match.group(2)

    return "", ""


def _sanitize_traceback(raw_tb: str) -> str:
    """Strip line numbers and absolute paths from traceback.

    Keeps: file names, function names, error type, assertion message.
    Strips: line numbers, absolute paths (shows relative), internal pytest frames.
    """
    sanitized_lines = []
    for line in raw_tb.splitlines():
        # Replace: File "/abs/path/to/file.py", line 42, in func_name
        # With:    File "file.py", in func_name
        match = _LINE_REF_RE.search(line)
        if match:
            filepath, _lineno, func = match.groups()
            # Keep only the filename (or relative from repo)
            short_path = Path(filepath).name
            indent = line[:len(line) - len(line.lstrip())]
            line = f'{indent}File "{short_path}", in {func}'

        # Strip line numbers from pytest short summary lines
        line = _PYTEST_LINE_RE.sub(':', line)

        # Strip absolute paths (e.g. from os.environ repr, conda paths)
        line = re.sub(r'/Users/[^\s,\'\"}\]]+', '<...>', line)
        line = re.sub(r'/home/[^\s,\'\"}\]]+', '<...>', line)

        # Skip lines that are just long environ() dumps
        if line.strip().startswith("self = environ(") and len(line) > 200:
            line = "    self = environ({...})"

        sanitized_lines.append(line)

    return "\n".join(sanitized_lines)


def generate_dev_report(
    task_id: str,
    test_cmd: str,
    failing_tests: list[str],
    raw_traceback: str,
    error_type: str,
    error_message: str,
) -> str:
    """Generate a dev_report_description from captured test failure data.

    Format: What a developer would write in a bug report after seeing test failures.
    Contains: test name, error type, assertion message, sanitized traceback.
    Does NOT contain: line numbers, mutation type, operator changes.
    """
    # Sanitize traceback (strip line numbers, absolute paths)
    clean_tb = _sanitize_traceback(raw_traceback)

    # Truncate traceback to keep descriptions reasonable
    tb_lines = clean_tb.splitlines()
    if len(tb_lines) > 25:
        tb_lines = tb_lines[:25] + ["    ..."]
    clean_tb = "\n".join(tb_lines)

    # Build the dev report description
    parts = []

    # Failing test name (stripped of line numbers)
    if failing_tests:
        test_names = ", ".join(failing_tests[:3])  # cap at 3
        parts.append(f"Test failure: {test_names}")
    else:
        parts.append(f"Test failure when running: {test_cmd}")

    # Error type and message
    if error_type and error_message:
        parts.append(f"Error: {error_type}: {error_message}")
    elif error_type:
        parts.append(f"Error: {error_type}")

    # Sanitized traceback
    if clean_tb.strip():
        parts.append(f"Traceback:\n{clean_tb}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

def capture_all(
    tasks_file: Path,
    output_file: Path | None = None,
    task_filter: str | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> list[dict]:
    """Capture tracebacks for all tasks and generate dev_report_description."""

    with open(tasks_file) as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(tasks)} tasks from {tasks_file}")

    if task_filter:
        tasks = [t for t in tasks if t["task_id"] == task_filter]
        print(f"Filtered to {len(tasks)} tasks matching '{task_filter}'")

    # Group by repo to minimize setup overhead
    by_repo: dict[str, list[dict]] = {}
    for t in tasks:
        by_repo.setdefault(t["repo_path"], []).append(t)

    results: list[dict] = []
    total = len(tasks)
    done = 0
    failed_capture = 0

    for repo_path_str, repo_tasks in by_repo.items():
        repo_path = Path(repo_path_str)
        print(f"\n{'='*70}")
        print(f"Repo: {repo_path.name} ({len(repo_tasks)} tasks)")
        print(f"{'='*70}")

        if not repo_path.exists():
            print(f"  ERROR: repo path does not exist: {repo_path}")
            for t in repo_tasks:
                t["dev_report_description"] = ""
                results.append(t)
                done += 1
            continue

        for t in repo_tasks:
            done += 1
            task_id = t["task_id"]
            mutation = t.get("mutation", {})
            test_cmd = t["test_cmd"]

            print(f"\n  [{done}/{total}] {task_id}")

            if not mutation:
                print(f"    SKIP: no mutation data")
                t["dev_report_description"] = ""
                results.append(t)
                continue

            # Apply mutation
            backup = _apply_mutation(repo_path, mutation)
            if backup is None:
                print(f"    ERROR: could not apply mutation to {mutation.get('file', '?')}")
                t["dev_report_description"] = ""
                results.append(t)
                failed_capture += 1
                continue

            original_content, original_hash = backup

            try:
                # Run tests on buggy code
                rc, full_output = run_tests_full(test_cmd, repo_path, timeout=t.get("timeout", 60))

                if rc == 0:
                    # Mutation survived — tests pass on buggy code (shouldn't happen for killed mutations)
                    print(f"    WARNING: tests pass on buggy code (rc=0), mutation not killed?")
                    t["dev_report_description"] = ""
                    t["_capture_status"] = "tests_pass_on_buggy"
                    results.append(t)
                    continue

                if rc == -1:
                    print(f"    WARNING: test timed out")
                    t["dev_report_description"] = f"Test failure when running: {test_cmd}\n\nError: Test execution timed out."
                    t["_capture_status"] = "timeout"
                    results.append(t)
                    continue

                # Extract failure data
                failing_tests = _extract_failing_tests(full_output)
                raw_traceback = _extract_traceback_block(full_output)
                error_type, error_message = _extract_error_type_and_message(full_output)

                # Generate dev report description
                dev_report = generate_dev_report(
                    task_id=task_id,
                    test_cmd=test_cmd,
                    failing_tests=failing_tests,
                    raw_traceback=raw_traceback,
                    error_type=error_type,
                    error_message=error_message,
                )

                t["dev_report_description"] = dev_report
                t["_capture_status"] = "ok"

                if verbose:
                    print(f"    Failing tests: {failing_tests}")
                    print(f"    Error: {error_type}: {error_message}")
                    print(f"    Traceback lines: {len(raw_traceback.splitlines())}")
                    print(f"    Dev report ({len(dev_report)} chars):")
                    for line in dev_report.splitlines()[:5]:
                        print(f"      {line}")
                    if len(dev_report.splitlines()) > 5:
                        print(f"      ... ({len(dev_report.splitlines()) - 5} more lines)")
                else:
                    # Compact summary
                    test_str = failing_tests[0] if failing_tests else "?"
                    err_str = f"{error_type}: {error_message[:60]}" if error_type else "unknown error"
                    print(f"    OK: {test_str} | {err_str} | {len(dev_report)} chars")

                results.append(t)

            finally:
                # Always revert
                _restore_mutation(repo_path, mutation, original_content, original_hash)
                if verbose:
                    print(f"    Reverted: {mutation['file']}")

    # Summary
    ok_count = sum(1 for t in results if t.get("_capture_status") == "ok")
    timeout_count = sum(1 for t in results if t.get("_capture_status") == "timeout")
    pass_count = sum(1 for t in results if t.get("_capture_status") == "tests_pass_on_buggy")
    empty_count = sum(1 for t in results if not t.get("dev_report_description"))

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total tasks:        {total}")
    print(f"  Captured OK:        {ok_count}")
    print(f"  Timeout:            {timeout_count}")
    print(f"  Tests pass (bug?):  {pass_count}")
    print(f"  Failed to capture:  {failed_capture}")
    print(f"  Empty description:  {empty_count}")

    # Clean up internal status field before writing
    for t in results:
        t.pop("_capture_status", None)

    # Write output
    if not dry_run and output_file:
        with open(output_file, "w") as f:
            for t in results:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(results)} tasks to {output_file}")
    elif dry_run:
        print(f"\nDry run — no output written")

    return results


def main():
    parser = argparse.ArgumentParser(description="Capture tracebacks from mutated code")
    parser.add_argument("--tasks", required=True, help="Path to eval_tasks_full.jsonl")
    parser.add_argument("--output", help="Output JSONL path (default: overwrite input)")
    parser.add_argument("--task-id", help="Run for a single task only")
    parser.add_argument("--dry-run", action="store_true", help="Don't write output")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    tasks_file = Path(args.tasks)
    output_file = Path(args.output) if args.output else tasks_file

    capture_all(
        tasks_file=tasks_file,
        output_file=output_file,
        task_filter=args.task_id,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
