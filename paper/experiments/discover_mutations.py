"""Automatically discover testable single-line mutations in a Python codebase.

Scans source files for mutation candidates, applies common mutation patterns,
tests each mutation against the project's test suite, and outputs results as JSON.

Designed for pydantic-ai but works on any Python project with a conventional
test layout (tests/test_{module}.py).

Usage:
    python discover_mutations.py /path/to/pydantic-ai --max-mutations-per-file 20 --output discovered.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SRC_DIR = "pydantic_ai_slim/pydantic_ai"
TESTS_DIR = "tests"
TEST_TIMEOUT = 60  # seconds per test run

# Patterns we skip when scanning for mutation candidates
SKIP_LINE_PATTERNS = [
    re.compile(r"^\s*#"),           # comments
    re.compile(r"^\s*$"),           # blank lines
    re.compile(r'^\s*"""'),         # docstring open/close
    re.compile(r"^\s*'''"),         # docstring open/close
    re.compile(r"^\s*import\s"),    # imports
    re.compile(r"^\s*from\s"),      # from imports
    re.compile(r"^\s*class\s"),     # class definitions
    re.compile(r"^\s*@"),           # decorators
    re.compile(r"^\s*def\s"),       # function definitions (we mutate bodies, not signatures)
    re.compile(r"^\s*pass\s*$"),    # pass statements
    re.compile(r"^\s*\.\.\."),      # ellipsis
    re.compile(r'^\s*raise\s+NotImplementedError'),  # stubs
]

# Patterns for trivial self-assignments in __init__
INIT_SELF_ASSIGN = re.compile(r"^\s*self\.\w+\s*=\s*\w+\s*$")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    """Log a message to stderr."""
    print(msg, file=sys.stderr, flush=True)


def log_progress(current: int, total: int, msg: str) -> None:
    """Log a progress message to stderr."""
    print(f"  [{current}/{total}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MutationCandidate:
    file: str               # relative path from repo root
    line_num: int           # 1-based
    original: str           # original line content (stripped of leading/trailing whitespace)
    mutated: str            # mutated line content (same stripping)
    mutation_type: str      # category name
    test_cmd: str           # pytest command to run
    killed: Optional[bool] = None  # True if tests fail with mutation, None if untested

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line_num": self.line_num,
            "original": self.original,
            "mutated": self.mutated,
            "mutation_type": self.mutation_type,
            "test_cmd": self.test_cmd,
            "killed": self.killed,
        }


@dataclass
class FunctionSpan:
    """Tracks a function/method body region in a file."""
    name: str
    start_line: int   # 1-based, first line of body
    end_line: int      # 1-based, last line of body
    in_init: bool      # True if this is __init__


# ---------------------------------------------------------------------------
# AST analysis: find function body ranges
# ---------------------------------------------------------------------------

def find_function_spans(source: str) -> list[FunctionSpan]:
    """Parse source and return line ranges of all function/method bodies."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.body:
                continue
            body_start = node.body[0].lineno
            body_end = _last_line(node)
            spans.append(FunctionSpan(
                name=node.name,
                start_line=body_start,
                end_line=body_end,
                in_init=(node.name == "__init__"),
            ))
    return spans


def _last_line(node: ast.AST) -> int:
    """Recursively find the last line number in an AST subtree."""
    last = getattr(node, "end_lineno", None) or getattr(node, "lineno", 0)
    for child in ast.iter_child_nodes(node):
        child_last = _last_line(child)
        if child_last > last:
            last = child_last
    return last


def line_in_function_body(line_num: int, spans: list[FunctionSpan]) -> Optional[FunctionSpan]:
    """Check if a 1-based line number falls inside any function body."""
    for span in spans:
        if span.start_line <= line_num <= span.end_line:
            return span
    return None


# ---------------------------------------------------------------------------
# Mutation patterns
# ---------------------------------------------------------------------------

@dataclass
class MutationPattern:
    """A single mutation rule: find a pattern in a line and replace it."""
    name: str
    regex: re.Pattern
    replacement: str
    priority: int = 0  # higher = more interesting, preferred

    def apply(self, line: str) -> Optional[str]:
        """Apply this mutation to a line. Returns mutated line or None."""
        match = self.regex.search(line)
        if match:
            result = line[:match.start()] + self.regex.sub(self.replacement, line[match.start():], count=1)
            if result != line:
                return result
        return None


@dataclass
class CallableMutationPattern:
    """A mutation rule with a callable replacement function."""
    name: str
    priority: int = 0

    def apply(self, line: str) -> Optional[str]:
        raise NotImplementedError


class ComparisonSwapPattern(CallableMutationPattern):
    """Swap comparison operators."""
    SWAPS = [
        ("==", "!="),
        ("!=", "=="),
        (">=", ">"),
        (">", ">="),
        ("<=", "<"),
        ("<", "<="),
        (">", "<"),
        ("<", ">"),
    ]

    def __init__(self):
        super().__init__(name="comparison_swap", priority=8)

    def apply(self, line: str) -> Optional[str]:
        stripped = line.lstrip()

        # Don't mutate string literals or comments
        if stripped.startswith("#") or stripped.startswith(("'", '"')):
            return None

        for old, new in self.SWAPS:
            # Use word-boundary-aware replacement to avoid mangling strings
            # Look for operator surrounded by spaces or adjacent to identifiers
            pattern = re.compile(
                r'(?<!=)(?<!<)(?<!>)(?<!!)' + re.escape(old) + r'(?!=)(?!<)(?!>)'
            )
            match = pattern.search(line)
            if match:
                # Make sure we're not inside a string literal
                before = line[:match.start()]
                if _in_string(before):
                    continue
                result = line[:match.start()] + new + line[match.end():]
                if result != line:
                    return result
        return None


class BooleanFlipPattern(CallableMutationPattern):
    """Flip `and` <-> `or` boolean operators."""

    def __init__(self):
        super().__init__(name="boolean_flip", priority=7)

    def apply(self, line: str) -> Optional[str]:
        # and -> or
        result = re.sub(r'\band\b', 'or', line, count=1)
        if result != line and not _in_string(line[:line.find(' and ')]):
            return result
        # or -> and
        result = re.sub(r'\bor\b', 'and', line, count=1)
        if result != line and not _in_string(line[:line.find(' or ')]):
            return result
        return None


class ConditionInversionPattern(CallableMutationPattern):
    """Invert conditions: `not x` -> `x`, add `not` before conditions."""

    def __init__(self):
        super().__init__(name="condition_inversion", priority=6)

    def apply(self, line: str) -> Optional[str]:
        # Remove `not ` prefix from conditions
        match = re.search(r'\bnot\s+(\w)', line)
        if match:
            before = line[:match.start()]
            if not _in_string(before):
                result = line[:match.start()] + line[match.start() + 4:]
                if result.strip() and result != line:
                    return result

        # Add `not` to isinstance checks
        match = re.search(r'(?<!\bnot\s)(isinstance\()', line)
        if match:
            before = line[:match.start()]
            if not _in_string(before):
                result = line[:match.start()] + "not " + line[match.start():]
                if result != line:
                    return result

        return None


class ArithmeticSwapPattern(CallableMutationPattern):
    """Swap arithmetic operators: + <-> -, * <-> /."""

    def __init__(self):
        super().__init__(name="arithmetic_swap", priority=9)

    def apply(self, line: str) -> Optional[str]:
        stripped = line.lstrip()

        # Skip decorator lines, imports, string-only lines
        if stripped.startswith(("@", "import ", "from ", "#", "'", '"')):
            return None

        # + -> - (but not in string concatenation or +=)
        match = re.search(r'(?<!\+)\+(?!\+|=)', line)
        if match:
            before = line[:match.start()]
            if not _in_string(before) and not before.rstrip().endswith(("'", '"')):
                # Make sure it looks like arithmetic (has digits or identifiers on both sides)
                left = line[:match.start()].rstrip()
                right = line[match.end():].lstrip()
                if (left and right and
                    (left[-1].isalnum() or left[-1] in ')_') and
                    (right[0].isalnum() or right[0] in '(_')):
                    result = line[:match.start()] + "-" + line[match.end():]
                    return result

        # - -> + (but not negative signs or -=)
        match = re.search(r'(?<![a-zA-Z_0-9(,=\[])\s*-\s*(?!=)', line)
        if match:
            # More conservative: look for ` - ` pattern (binary minus)
            match2 = re.search(r'(\w)\s+-\s+(\w)', line)
            if match2:
                before = line[:match2.start()]
                if not _in_string(before):
                    pos = line.index('-', match2.start())
                    result = line[:pos] + '+' + line[pos + 1:]
                    return result

        # * -> / (but not ** or *args)
        match = re.search(r'(?<!\*)\*(?!\*|=)', line)
        if match:
            before = line[:match.start()]
            if not _in_string(before):
                left = line[:match.start()].rstrip()
                right = line[match.end():].lstrip()
                if (left and right and
                    (left[-1].isalnum() or left[-1] in ')_') and
                    (right[0].isalnum() or right[0] in '(_')):
                    result = line[:match.start()] + "/" + line[match.end():]
                    return result

        return None


class ValueSwapPattern(CallableMutationPattern):
    """Swap True/False, 0/1."""

    def __init__(self):
        super().__init__(name="value_swap", priority=5)

    def apply(self, line: str) -> Optional[str]:
        # True -> False
        match = re.search(r'\bTrue\b', line)
        if match:
            before = line[:match.start()]
            if not _in_string(before):
                result = line[:match.start()] + "False" + line[match.end():]
                return result

        # False -> True
        match = re.search(r'\bFalse\b', line)
        if match:
            before = line[:match.start()]
            if not _in_string(before):
                result = line[:match.start()] + "True" + line[match.end():]
                return result

        return None


class NoneCheckPattern(CallableMutationPattern):
    """Swap `is not None` <-> `is None`."""

    def __init__(self):
        super().__init__(name="none_check_swap", priority=9)

    def apply(self, line: str) -> Optional[str]:
        # is not None -> is None
        match = re.search(r'\bis\s+not\s+None\b', line)
        if match:
            before = line[:match.start()]
            if not _in_string(before):
                result = line[:match.start()] + "is None" + line[match.end():]
                return result

        # is None -> is not None
        match = re.search(r'\bis\s+None\b', line)
        if match:
            before = line[:match.start()]
            if not _in_string(before):
                result = line[:match.start()] + "is not None" + line[match.end():]
                return result

        return None


class MembershipTestPattern(CallableMutationPattern):
    """Swap `not in` <-> `in`."""

    def __init__(self):
        super().__init__(name="membership_swap", priority=7)

    def apply(self, line: str) -> Optional[str]:
        # not in -> in
        match = re.search(r'\bnot\s+in\b', line)
        if match:
            before = line[:match.start()]
            if not _in_string(before):
                result = line[:match.start()] + "in" + line[match.end():]
                return result

        # in -> not in (only if it looks like a membership test)
        match = re.search(r'(\w)\s+in\s+(\w)', line)
        if match:
            before = line[:match.start()]
            if not _in_string(before):
                # Find the 'in' keyword position
                in_pos = line.index(' in ', match.start())
                result = line[:in_pos] + " not in" + line[in_pos + 3:]
                return result

        return None


class ConstantMutationPattern(CallableMutationPattern):
    """Mutate integer literals: n -> n+1, n -> n-1, n -> n*10, n -> n//10."""

    def __init__(self):
        super().__init__(name="constant_mutation", priority=4)

    def apply(self, line: str) -> Optional[str]:
        stripped = line.lstrip()

        # Skip lines that are just assignments to version strings, etc.
        if "version" in stripped.lower() or "__" in stripped:
            return None

        # Find integer literals (not in strings, not part of identifiers)
        for match in re.finditer(r'(?<![a-zA-Z_.])\b(\d+)\b(?![a-zA-Z_])', line):
            before = line[:match.start()]
            if _in_string(before):
                continue

            value = int(match.group(1))

            # Skip 0 and 1 -- use value swap for those
            # Focus on meaningful constants
            if value == 0:
                result = line[:match.start()] + "1" + line[match.end():]
                return result
            elif value == 1:
                result = line[:match.start()] + "0" + line[match.end():]
                return result
            elif value > 1:
                # n -> n + 1 for small constants, n -> n * 10 for larger ones
                new_value = value + 1
                result = line[:match.start()] + str(new_value) + line[match.end():]
                return result

        return None


class ReturnValuePattern(CallableMutationPattern):
    """Swap return values: return X -> return not X, return [] -> return [None]."""

    def __init__(self):
        super().__init__(name="return_value_swap", priority=3)

    def apply(self, line: str) -> Optional[str]:
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]

        # return True -> return False and vice versa
        if re.match(r'return\s+True\s*$', stripped):
            return indent + "return False"
        if re.match(r'return\s+False\s*$', stripped):
            return indent + "return True"

        # return [] -> return [None]
        if re.match(r'return\s+\[\]\s*$', stripped):
            return indent + "return [None]"

        # return {} -> return {"_": None}
        if re.match(r'return\s+\{\}\s*$', stripped):
            return indent + 'return {"_": None}'

        # return 0 -> return 1
        if re.match(r'return\s+0\s*$', stripped):
            return indent + "return 1"

        # return 1 -> return 0
        if re.match(r'return\s+1\s*$', stripped):
            return indent + "return 0"

        return None


# ---------------------------------------------------------------------------
# Helper: rough string detection
# ---------------------------------------------------------------------------

def _in_string(text: str) -> bool:
    """Rough heuristic: check if the end of `text` is inside a string literal.

    Counts unescaped quotes. Not perfect, but good enough for mutation filtering.
    """
    single_count = 0
    double_count = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\':
            i += 2
            continue
        if ch == "'" and double_count % 2 == 0:
            single_count += 1
        elif ch == '"' and single_count % 2 == 0:
            double_count += 1
        i += 1
    return (single_count % 2 != 0) or (double_count % 2 != 0)


# ---------------------------------------------------------------------------
# All mutation patterns, ordered by priority (highest first)
# ---------------------------------------------------------------------------

ALL_PATTERNS: list[CallableMutationPattern] = sorted([
    ComparisonSwapPattern(),
    BooleanFlipPattern(),
    ConditionInversionPattern(),
    ArithmeticSwapPattern(),
    ValueSwapPattern(),
    NoneCheckPattern(),
    MembershipTestPattern(),
    ConstantMutationPattern(),
    ReturnValuePattern(),
], key=lambda p: -p.priority)


# ---------------------------------------------------------------------------
# Test file discovery
# ---------------------------------------------------------------------------

def find_test_file(source_file: str, repo_path: Path, tests_dir_name: str = TESTS_DIR) -> Optional[str]:
    """Heuristic: given a source file path, find the corresponding test file.

    Strategy:
      1. Extract module name from source file
      2. Look for tests/test_{module}.py
      3. Look for tests/test_{parent}_{module}.py
      4. Fallback: search for test files that import the module
    """
    src_path = Path(source_file)
    module_name = src_path.stem  # e.g., "usage" from "usage.py"

    if module_name.startswith("_"):
        clean_name = module_name.lstrip("_")
    else:
        clean_name = module_name

    tests_dir = repo_path / tests_dir_name
    if not tests_dir.is_dir():
        return None

    # Strategy 1: Direct match tests/test_{module}.py
    candidates = [
        f"test_{clean_name}.py",
        f"test_{module_name}.py",
    ]

    # Strategy 2: Include parent directory in test name
    parent = src_path.parent.name
    if parent and parent not in (".",):
        candidates.append(f"test_{parent}_{clean_name}.py")

    # Strategy 3: Common naming variations
    # e.g., "usage.py" -> "test_usage_limits.py", "_ssrf.py" -> "test_ssrf.py"
    candidates.append(f"test_{clean_name}s.py")  # plural

    # Check flat first, then nested subdirectories
    for candidate in candidates:
        test_path = tests_dir / candidate
        if test_path.exists():
            return str(test_path.relative_to(repo_path))

    # Strategy 3b: Search recursively in test subdirectories
    for candidate in candidates:
        for test_path in tests_dir.rglob(candidate):
            return str(test_path.relative_to(repo_path))

    # Strategy 4: Search for any test file containing the module name (recursive)
    for test_file in sorted(tests_dir.rglob("test_*.py")):
        if clean_name in test_file.stem:
            return str(test_file.relative_to(repo_path))

    # Strategy 5: Check imports in test files (slower, last resort, recursive)
    module_import_pattern = re.compile(
        rf'\b{re.escape(clean_name)}\b'
    )
    for test_file in sorted(tests_dir.rglob("test_*.py")):
        try:
            content = test_file.read_text(errors="replace")
            # Only look at first 50 lines (imports section)
            header = "\n".join(content.splitlines()[:50])
            if module_import_pattern.search(header):
                return str(test_file.relative_to(repo_path))
        except OSError:
            continue

    return None


# ---------------------------------------------------------------------------
# Source file scanning
# ---------------------------------------------------------------------------

def should_skip_line(line: str, line_num: int, spans: list[FunctionSpan]) -> bool:
    """Determine if a line should be skipped for mutation."""
    stripped = line.strip()

    # Empty or trivial
    if not stripped or len(stripped) < 3:
        return True

    # Pattern-based skip
    for pattern in SKIP_LINE_PATTERNS:
        if pattern.match(line):
            return True

    # Must be inside a function body
    span = line_in_function_body(line_num, spans)
    if span is None:
        return True

    # Skip trivial self assignments in __init__
    if span.in_init and INIT_SELF_ASSIGN.match(line):
        return True

    # Skip lines that are only string literals (docstrings inside functions)
    if stripped.startswith(("'", '"')) and stripped.endswith(("'", '"')):
        return True
    if stripped.startswith(('"""', "'''")):
        return True
    if stripped.endswith(('"""', "'''")):
        return True

    # Skip type annotations only (no executable code)
    if re.match(r'^\s*\w+\s*:\s*\w', stripped) and '=' not in stripped:
        return True

    # Skip `else:`, `try:`, `except ...:`, `finally:`
    if re.match(r'^\s*(else|try|finally)\s*:\s*$', line):
        return True
    if re.match(r'^\s*except\b', line):
        return True

    return False


def generate_mutations_for_line(
    line: str, line_num: int, spans: list[FunctionSpan]
) -> Optional[tuple[str, str, str]]:
    """Generate the best single mutation for a line.

    Returns (mutated_line_stripped, mutation_type, original_stripped) or None.
    """
    stripped = line.strip()

    # Try each pattern in priority order, return the first successful one
    for pattern in ALL_PATTERNS:
        result = pattern.apply(line)
        if result is not None:
            result_stripped = result.strip()
            # Sanity: skip if mutation is whitespace-only or identical
            if result_stripped == stripped:
                continue
            # Skip if the mutation only differs in whitespace
            if result_stripped.replace(" ", "") == stripped.replace(" ", ""):
                continue
            return (result_stripped, pattern.name, stripped)

    return None


def scan_file(
    file_path: Path,
    repo_path: Path,
    max_mutations: int,
    tests_dir_name: str = TESTS_DIR,
    python_cmd: str = "python",
) -> list[MutationCandidate]:
    """Scan a single source file for mutation candidates."""
    try:
        source = file_path.read_text(errors="replace")
    except OSError as e:
        log(f"  Warning: cannot read {file_path}: {e}")
        return []

    lines = source.splitlines()
    spans = find_function_spans(source)

    if not spans:
        return []

    rel_path = str(file_path.relative_to(repo_path))
    test_file = find_test_file(rel_path, repo_path, tests_dir_name)

    if test_file is None:
        log(f"  No test file found for {rel_path}, skipping")
        return []

    test_cmd = f"{python_cmd} -m pytest {test_file} -x"

    candidates = []
    for i, line in enumerate(lines):
        line_num = i + 1

        if should_skip_line(line, line_num, spans):
            continue

        mutation = generate_mutations_for_line(line, line_num, spans)
        if mutation is None:
            continue

        mutated_stripped, mutation_type, original_stripped = mutation

        candidates.append(MutationCandidate(
            file=rel_path,
            line_num=line_num,
            original=original_stripped,
            mutated=mutated_stripped,
            mutation_type=mutation_type,
            test_cmd=test_cmd,
        ))

        if len(candidates) >= max_mutations:
            break

    return candidates


def discover_all_mutations(
    repo_path: Path,
    src_dir: str,
    max_per_file: int,
    tests_dir_name: str = TESTS_DIR,
    python_cmd: str = "python",
) -> list[MutationCandidate]:
    """Scan all Python files in src_dir for mutation candidates."""
    source_dir = repo_path / src_dir
    if not source_dir.is_dir():
        log(f"Error: source directory not found: {source_dir}")
        return []

    py_files = sorted(source_dir.rglob("*.py"))
    log(f"Found {len(py_files)} Python source files in {src_dir}")

    all_candidates = []
    for file_path in py_files:
        # Skip __pycache__, etc.
        if "__pycache__" in str(file_path):
            continue

        candidates = scan_file(file_path, repo_path, max_per_file, tests_dir_name, python_cmd)
        if candidates:
            log(f"  {file_path.relative_to(repo_path)}: {len(candidates)} candidates")
            all_candidates.extend(candidates)

    return all_candidates


# ---------------------------------------------------------------------------
# Mutation testing
# ---------------------------------------------------------------------------

def apply_mutation(file_path: Path, line_num: int, original: str, mutated: str) -> Optional[str]:
    """Apply a mutation to a file. Returns original content for revert, or None on failure."""
    try:
        content = file_path.read_text()
    except OSError:
        return None

    lines = content.splitlines()
    idx = line_num - 1

    if idx < 0 or idx >= len(lines):
        return None

    current_line = lines[idx]
    if original not in current_line:
        return None

    # Preserve indentation: replace only the stripped content
    lines[idx] = current_line.replace(original, mutated, 1)

    new_content = "\n".join(lines)
    if content.endswith("\n"):
        new_content += "\n"

    try:
        file_path.write_text(new_content)
    except OSError:
        return None

    return content


def revert_file(file_path: Path, original_content: str) -> bool:
    """Revert a file to its original content."""
    try:
        file_path.write_text(original_content)
        return True
    except OSError:
        return False


def run_tests(test_cmd: str, repo_path: Path, timeout: int = 60) -> tuple[int, str]:
    """Run a test command and return (returncode, output_snippet)."""
    try:
        result = subprocess.run(
            test_cmd.split(),
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        # Combine stdout and stderr, truncate to last 500 chars for logging
        output = (result.stdout + result.stderr)[-500:]
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except OSError as e:
        return -2, str(e)


def test_mutations(
    candidates: list[MutationCandidate],
    repo_path: Path,
    skip_clean_check: bool = False,
    timeout: int = 60,
) -> list[MutationCandidate]:
    """Test each mutation candidate by applying it and running tests.

    Returns the list of candidates with `killed` field populated.
    """
    # Group candidates by test_cmd to avoid re-running clean checks
    clean_cache: dict[str, bool] = {}  # test_cmd -> passes_clean
    tested = []
    total = len(candidates)

    for i, candidate in enumerate(candidates):
        file_path = repo_path / candidate.file
        log_progress(i + 1, total, f"{candidate.file}:{candidate.line_num} ({candidate.mutation_type})")

        # Check if clean tests pass (cached per test_cmd)
        if candidate.test_cmd not in clean_cache:
            if skip_clean_check:
                clean_cache[candidate.test_cmd] = True
            else:
                log(f"    Checking clean tests: {candidate.test_cmd}")
                returncode, output = run_tests(candidate.test_cmd, repo_path, timeout=timeout)
                clean_passes = (returncode == 0)
                clean_cache[candidate.test_cmd] = clean_passes
                if not clean_passes:
                    log(f"    SKIP: tests fail on clean code (rc={returncode})")

        if not clean_cache[candidate.test_cmd]:
            log(f"    SKIP: tests fail on clean code for {candidate.test_cmd}")
            candidate.killed = None
            tested.append(candidate)
            continue

        # Apply mutation
        original_content = apply_mutation(
            file_path, candidate.line_num, candidate.original, candidate.mutated
        )
        if original_content is None:
            log(f"    SKIP: could not apply mutation")
            candidate.killed = None
            tested.append(candidate)
            continue

        try:
            # Run tests with mutation applied
            returncode, output = run_tests(candidate.test_cmd, repo_path, timeout=timeout)

            if returncode != 0:
                candidate.killed = True
                log(f"    KILLED (rc={returncode})")
            else:
                candidate.killed = False
                log(f"    SURVIVED -- mutation not detected")
        except Exception as e:
            log(f"    ERROR: {e}")
            candidate.killed = None
        finally:
            # Always revert
            if not revert_file(file_path, original_content):
                log(f"    CRITICAL: failed to revert {candidate.file}! Manual fix needed.")
                sys.exit(1)

        tested.append(candidate)

    return tested


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Discover testable single-line mutations in a Python codebase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python discover_mutations.py /path/to/pydantic-ai --max-mutations-per-file 20 --output discovered.json
  python discover_mutations.py /path/to/pydantic-ai --discover-only --output candidates.json
  python discover_mutations.py /path/to/pydantic-ai --src-dir src/ --tests-dir tests/
        """,
    )
    parser.add_argument(
        "repo",
        help="Path to the repository root",
    )
    parser.add_argument(
        "--src-dir",
        default=SRC_DIR,
        help=f"Source directory relative to repo root (default: {SRC_DIR})",
    )
    parser.add_argument(
        "--tests-dir",
        default=TESTS_DIR,
        help=f"Tests directory relative to repo root (default: {TESTS_DIR})",
    )
    parser.add_argument(
        "--max-mutations-per-file",
        type=int,
        default=20,
        help="Maximum mutations to generate per file (default: 20)",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=0,
        help="Maximum total mutations to test (0 = no limit, default: 0)",
    )
    parser.add_argument(
        "--output",
        default="discovered.json",
        help="Output JSON file (default: discovered.json)",
    )
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only discover candidates, skip mutation testing",
    )
    parser.add_argument(
        "--skip-clean-check",
        action="store_true",
        help="Skip verifying that tests pass on clean code (faster, less safe)",
    )
    parser.add_argument(
        "--killed-only",
        action="store_true",
        help="Only include killed mutations in output",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TEST_TIMEOUT,
        help=f"Timeout per test run in seconds (default: {TEST_TIMEOUT})",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable to use for running tests (default: python)",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    if not repo_path.is_dir():
        log(f"Error: {repo_path} is not a directory")
        sys.exit(1)

    log(f"Repository: {repo_path}")
    log(f"Source dir: {args.src_dir}")
    log(f"Tests dir:  {args.tests_dir}")
    log("")

    # Phase 1: Discovery
    log("=== Phase 1: Discovering mutation candidates ===")
    start = time.time()
    candidates = discover_all_mutations(repo_path, args.src_dir, args.max_mutations_per_file, args.tests_dir, args.python)
    elapsed = time.time() - start
    log(f"\nDiscovered {len(candidates)} mutation candidates in {elapsed:.1f}s")

    if not candidates:
        log("No mutation candidates found. Check --src-dir and that test files exist.")
        sys.exit(0)

    # Apply max-total limit
    if args.max_total > 0 and len(candidates) > args.max_total:
        log(f"Limiting to {args.max_total} candidates (from {len(candidates)})")
        candidates = candidates[:args.max_total]

    # Phase 2: Testing
    if args.discover_only:
        log("\n=== Skipping mutation testing (--discover-only) ===")
        results = candidates
    else:
        log(f"\n=== Phase 2: Testing {len(candidates)} mutations ===")
        start = time.time()
        results = test_mutations(candidates, repo_path, skip_clean_check=args.skip_clean_check, timeout=args.timeout)
        elapsed = time.time() - start

        killed = sum(1 for r in results if r.killed is True)
        survived = sum(1 for r in results if r.killed is False)
        skipped = sum(1 for r in results if r.killed is None)
        log(f"\nMutation testing complete in {elapsed:.1f}s")
        log(f"  Killed:   {killed}")
        log(f"  Survived: {survived}")
        log(f"  Skipped:  {skipped}")
        if killed + survived > 0:
            score = killed / (killed + survived)
            log(f"  Kill rate: {score:.1%}")

    # Filter if requested
    if args.killed_only:
        results = [r for r in results if r.killed is True]
        log(f"\nFiltered to {len(results)} killed mutations")

    # Output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = [r.to_dict() for r in results]
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    log(f"\nResults written to {output_path}")

    # Also print summary to stdout
    print(json.dumps({
        "total_candidates": len(candidates),
        "tested": len(results),
        "killed": sum(1 for r in results if r.killed is True),
        "survived": sum(1 for r in results if r.killed is False),
        "skipped": sum(1 for r in results if r.killed is None),
        "output_file": str(output_path),
    }, indent=2))


if __name__ == "__main__":
    main()
