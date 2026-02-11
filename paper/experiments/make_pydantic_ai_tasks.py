"""Generate eval_tasks.jsonl for pydantic-ai from discovered mutations.

Takes the output of discover_mutations.py (killed mutations JSON) and the
hand-crafted mutations in pydantic_ai_tasks.py, merges them, and produces
a benchmark-ready JSONL file with full reproducibility metadata.

Usage:
    python -m paper.experiments.make_pydantic_ai_tasks \
        --repo /path/to/pydantic-ai \
        --discovered paper/experiments/killed_mutations.json \
        --output paper/experiments/eval_tasks_pydantic_ai.jsonl \
        --seed 42

Reproducibility:
    - Fixed random seed for deterministic selection
    - All configuration parameters recorded in discovery_report.json
    - Full candidate list with outcomes in the report
    - Selection algorithm is round-robin with per-file and per-category caps
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

# Import hand-crafted mutations (these have both exact and vague descriptions)
from pydantic_ai_tasks import MUTATIONS as HANDCRAFTED

# -------------------------------------------------------------------------
# Constants — these define the selection policy and are recorded in the report
# -------------------------------------------------------------------------

SELECTION_CONFIG = {
    "max_per_file": 15,           # Liberal cap — use most available mutations
    "max_per_category": 25,       # Liberal cap — let natural distribution emerge
    "max_per_mutation_type": 30,  # Prevent boolean-style mutations from dominating
    "min_per_category": 1,        # Try to include at least 1 from each category with kills
    "shuffle_within_category": True,  # Shuffle within category before selection
}

# Pydantic-ai repo metadata — frozen for reproducibility
REPO_METADATA = {
    "repo_url": "https://github.com/pydantic/pydantic-ai",
    "commit": "69a578a1e1012e0241d15321e45ab978962ed0d7",
    "commit_short": "69a578a",
    "commit_message": "Add multi-run aggregation support (repeat parameter) to pydantic-evals (#4253)",
    "source_dir": "pydantic_ai_slim/pydantic_ai",
    "tests_dir": "tests",
    "install_cmd": "pip install -e pydantic_graph && pip install -e 'pydantic_ai_slim[test]'",
}

# -------------------------------------------------------------------------
# Category assignment
# -------------------------------------------------------------------------

# Explicit file→category mapping. More precise than substring matching.
FILE_CATEGORY_MAP = {
    "usage.py": "usage",
    "_ssrf.py": "ssrf",
    "exceptions.py": "exceptions",
    "settings.py": "settings",
    "messages.py": "messages",
    "result.py": "result",
    "concurrency.py": "concurrency",
    "retries.py": "retries",
    "direct.py": "direct_api",
    "_utils.py": "utils",
    "_json_schema.py": "json_schema",
    "_parts_manager.py": "parts_manager",
    "_output.py": "output",
    "_thinking_part.py": "thinking",
    "tools.py": "tools",
    "builtin_tools.py": "builtin_tools",
    "format_prompt.py": "formatting",
    "run.py": "run",
    "mcp.py": "mcp",
}


def categorize_mutation(file_path: str) -> str:
    """Assign a category to a mutation based on its file path."""
    fname = Path(file_path).name
    if fname in FILE_CATEGORY_MAP:
        return FILE_CATEGORY_MAP[fname]
    # Fallback: use parent directory
    parts = Path(file_path).parts
    if "models" in parts:
        return "models"
    if "agent" in parts:
        return "agent"
    if "toolsets" in parts:
        return "toolsets"
    if "ui" in parts:
        return "ui"
    if "embeddings" in parts:
        return "embeddings"
    return "other"


# -------------------------------------------------------------------------
# Loading and deduplication
# -------------------------------------------------------------------------

def load_discovered(discovered_file: Path, repo_path: Path | None = None) -> list[dict]:
    """Load killed mutations, filtering out those that produce invalid syntax.

    If repo_path is provided, each mutation is applied in-memory and ast.parse'd.
    Mutations that fail to parse are excluded (they would crash the benchmark).
    """
    import ast

    with open(discovered_file) as f:
        data = json.load(f)
    killed = [m for m in data if m.get("killed") is True]

    if repo_path is None:
        return killed

    valid = []
    skipped = 0
    for m in killed:
        fp = repo_path / m["file"]
        if not fp.exists():
            valid.append(m)  # Can't validate, keep it
            continue
        try:
            source = fp.read_text(encoding="utf-8")
            lines = source.splitlines()
            line_idx = m["line_num"] - 1
            if 0 <= line_idx < len(lines):
                mutated_line = lines[line_idx].replace(m["original"], m["mutated"], 1)
                lines[line_idx] = mutated_line
                ast.parse("\n".join(lines))
            valid.append(m)
        except SyntaxError:
            skipped += 1

    if skipped:
        print(f"  Filtered {skipped} syntax-invalid mutations ({len(valid)} remaining)")
    return valid


# Full discovery stats from mutation_discovery.log (1454 candidates tested)
DISCOVERY_STATS = {
    "total_candidates": 1454,
    "killed": 199,
    "survived": 770,
    "skipped": 485,
    "kill_rate": 0.205,
    "test_time_seconds": 1202,
}


def deduplicate(handcrafted: list[dict], discovered: list[dict]) -> list[dict]:
    """Remove discovered mutations that overlap with hand-crafted ones."""
    hc_keys = {(m["file"], m["line_num"]) for m in handcrafted}
    return [m for m in discovered if (m["file"], m["line_num"]) not in hc_keys]


# -------------------------------------------------------------------------
# Deterministic diverse selection
# -------------------------------------------------------------------------

def select_diverse(
    discovered: list[dict],
    target_count: int,
    seed: int,
    config: dict,
) -> tuple[list[dict], dict]:
    """Select a diverse subset of discovered mutations.

    Uses a fixed seed for reproducibility. Selection is round-robin across
    categories, with per-file and per-category caps.

    Returns (selected, selection_stats).
    """
    rng = random.Random(seed)

    max_per_file = config["max_per_file"]
    max_per_category = config["max_per_category"]
    max_per_mutation_type = config.get("max_per_mutation_type")

    # Group by category
    by_category: dict[str, list[dict]] = {}
    for m in discovered:
        cat = categorize_mutation(m["file"])
        by_category.setdefault(cat, []).append(m)

    # Shuffle within each category for unbiased selection
    if config.get("shuffle_within_category", True):
        for cat in by_category:
            rng.shuffle(by_category[cat])

    selected: list[dict] = []
    file_counts: dict[str, int] = {}
    category_counts: dict[str, int] = Counter()
    mutation_type_counts: dict[str, int] = Counter()

    # Round-robin across categories (sorted for determinism)
    remaining = {cat: list(muts) for cat, muts in by_category.items()}
    categories = sorted(remaining.keys())

    while len(selected) < target_count and any(remaining.values()):
        made_progress = False
        for cat in categories:
            if not remaining[cat]:
                continue
            if len(selected) >= target_count:
                break
            if category_counts[cat] >= max_per_category:
                continue

            # Pick next eligible mutation from this category
            for i, m in enumerate(remaining[cat]):
                file_key = m["file"]
                if file_counts.get(file_key, 0) >= max_per_file:
                    continue
                # Enforce per-mutation-type cap to prevent one mutation family
                # (e.g. boolean-style: membership_swap + none_check_swap)
                # from dominating the benchmark
                mt = m.get("mutation_type", "unknown")
                if max_per_mutation_type and mutation_type_counts[mt] >= max_per_mutation_type:
                    continue

                selected.append(m)
                file_counts[file_key] = file_counts.get(file_key, 0) + 1
                category_counts[cat] += 1
                mutation_type_counts[mt] += 1
                remaining[cat].pop(i)
                made_progress = True
                break

        if not made_progress:
            break  # No more eligible mutations in any category

    stats = {
        "target_count": target_count,
        "actual_count": len(selected),
        "seed": seed,
        "categories_used": dict(category_counts),
        "files_used": dict(file_counts),
        "categories_available": {cat: len(muts) for cat, muts in by_category.items()},
        "categories_exhausted": [cat for cat in categories if not remaining.get(cat)],
        "mutation_type_counts": dict(mutation_type_counts),
    }

    return selected, stats


# -------------------------------------------------------------------------
# Task ID and description generation
# -------------------------------------------------------------------------

def generate_task_id(mutation: dict) -> str:
    """Generate a unique, readable task ID from a discovered mutation."""
    file_stem = Path(mutation["file"]).stem.lstrip("_")
    return f"d-{file_stem}-L{mutation['line_num']}-{mutation['mutation_type']}"


def generate_exact_description(mutation: dict) -> str:
    """Generate an exact (developer-level) description for a discovered mutation.

    Rules for exact descriptions:
    - Reference the specific function/method and file
    - Describe what the code does wrong (not what the correct behavior is)
    - Use technical language appropriate for a developer reading the source
    """
    mt = mutation["mutation_type"]
    orig = mutation["original"]
    mutated = mutation["mutated"]
    file_short = Path(mutation["file"]).name
    line = mutation["line_num"]

    if mt == "comparison_swap":
        return (
            f"In {file_short}:{line}, the comparison operator was changed: "
            f"'{orig}' became '{mutated}'. This alters the boundary condition."
        )
    elif mt == "none_check_swap":
        return (
            f"In {file_short}:{line}, a None check was inverted: "
            f"'{orig}' became '{mutated}'. This reverses when the value is considered present vs absent."
        )
    elif mt == "boolean_flip":
        return (
            f"In {file_short}:{line}, a boolean operator was swapped: "
            f"'{orig}' became '{mutated}'. This changes the logical condition."
        )
    elif mt == "membership_swap":
        return (
            f"In {file_short}:{line}, a membership test was inverted: "
            f"'{orig}' became '{mutated}'. This reverses which values pass the check."
        )
    elif mt == "condition_inversion":
        return (
            f"In {file_short}:{line}, a condition was inverted: "
            f"'{orig}' became '{mutated}'. The branch now triggers in the opposite case."
        )
    elif mt == "arithmetic_swap":
        return (
            f"In {file_short}:{line}, an arithmetic operator was changed: "
            f"'{orig}' became '{mutated}'. This produces wrong numeric results."
        )
    elif mt == "value_swap":
        return (
            f"In {file_short}:{line}, a value was swapped: "
            f"'{orig}' became '{mutated}'."
        )
    elif mt == "constant_mutation":
        return (
            f"In {file_short}:{line}, a constant was changed: "
            f"'{orig}' became '{mutated}'. This shifts a boundary or default."
        )
    else:
        return (
            f"In {file_short}:{line}, the code was changed from "
            f"'{orig}' to '{mutated}' ({mt})."
        )


def generate_vague_description(mutation: dict) -> str:
    """Generate a vague (user-reported symptom) description for a discovered mutation.

    IMPORTANT: Templates are category-agnostic — they describe observable symptoms
    keyed ONLY by mutation type, never by subsystem/category. This avoids leaking
    privileged knowledge about which subsystem is broken into the vague query.

    A real user would say "a boundary check seems wrong" — not "the SSRF boundary
    check seems wrong." The retrieval system must figure out which subsystem to look
    at from the symptom alone.

    Multiple templates per mutation type provide variety. Template selection is
    deterministic (hash of file+line) so results are reproducible.

    Template rules:
    - No file paths, line numbers, function names, or class names
    - No subsystem/category references (no "SSRF", "usage", "JSON schema", etc.)
    - Describe observable behavior only: what fails, what's unexpected
    - Use impersonal voice ("Something seems wrong...", "The system...")
    - Be plausible but imprecise — as a user filing a bug report would write
    """
    mt = mutation["mutation_type"]

    # Deterministic template selection (hashlib — immune to PYTHONHASHSEED randomization)
    import hashlib
    selector = int(hashlib.sha256(f"{mutation['file']}:{mutation['line_num']}".encode()).hexdigest(), 16) % 3

    # Symptom templates by mutation type only — no category knowledge
    templates = {
        "comparison_swap": [
            "A threshold or limit check seems to trigger at the wrong value. The boundary is off by one or uses the wrong comparison direction.",
            "Something that should be caught at an exact limit slips through, or triggers too early. The comparison logic seems wrong.",
            "A numeric check is behaving unexpectedly at the boundary. Values exactly at the threshold are handled incorrectly.",
        ],
        "none_check_swap": [
            "A feature that should be disabled when not configured is always active, or vice versa. The None/not-None check seems inverted.",
            "Setting a value to None should disable something, but it has no effect. Or leaving it unset activates behavior that shouldn't be there.",
            "Optional configuration is being ignored. When I explicitly set a value, the system behaves as if it's not set, and when I don't set it, the system acts like it is.",
        ],
        "boolean_flip": [
            "A logical condition evaluates to the opposite of what it should. The system does the wrong thing when a condition is true vs false.",
            "An 'or' that should be 'and' (or vice versa) is causing wrong behavior. The boolean logic combines conditions incorrectly.",
            "A feature flag or boolean check seems inverted. Enabling something disables it, or the logic short-circuits in the wrong direction.",
        ],
        "membership_swap": [
            "An inclusion check is backwards. Items that should be accepted are rejected, and items that should be rejected pass through.",
            "A set membership or list containment test is inverted. The code includes what it should exclude and excludes what it should include.",
            "A filter or validation check accepts the wrong set of values. Things that should match don't, and things that shouldn't match do.",
        ],
        "condition_inversion": [
            "A conditional branch is inverted. The code takes the 'if' path when it should take the 'else' path, and vice versa.",
            "A guard clause or check does the opposite of what it should. The happy path and error path are swapped.",
            "Logic seems backwards. An operation that should run only in certain cases runs in the opposite cases instead.",
        ],
        "arithmetic_swap": [
            "A numerical calculation gives wrong output. The math is off — values are larger or smaller than expected by a consistent amount.",
            "An accumulator or counter is going in the wrong direction. Values that should increase are decreasing, or the formula uses the wrong operator.",
            "Computed values are consistently wrong. The computation seems to use subtraction where it should add, or vice versa.",
        ],
        "constant_mutation": [
            "A default value or configuration constant seems wrong. The system uses a different threshold or limit than what's documented.",
            "A hardcoded value is slightly off. Behavior at specific boundaries doesn't match expectations — like an off-by-one in a limit.",
            "A numeric constant or default parameter has the wrong value. The system behaves as if configured with a different setting than intended.",
        ],
        "value_swap": [
            "Two values appear to be swapped. The system uses value A where it should use value B, and value B where it should use value A.",
            "An assignment or return value is wrong — it returns the first argument where it should return the second, or similar.",
            "Two related values are mixed up. The system applies the wrong one of two options depending on context.",
        ],
    }

    if mt in templates:
        options = templates[mt]
        return options[selector % len(options)]

    # Fallback for unknown mutation types
    return "Something isn't working correctly. Behavior doesn't match what the documentation describes."


# -------------------------------------------------------------------------
# JSONL output with full metadata
# -------------------------------------------------------------------------

def write_tasks(
    repo_path: Path,
    handcrafted: list[dict],
    discovered: list[dict],
    output_file: Path,
) -> None:
    """Write merged eval tasks as JSONL with full reproducibility metadata."""
    with open(output_file, "w") as f:
        # Hand-crafted first (have vague descriptions)
        for m in handcrafted:
            task = {
                "task_id": m["task_id"],
                "repo_path": str(repo_path.resolve()),
                "repo_url": REPO_METADATA["repo_url"],
                "commit": REPO_METADATA["commit"],
                "setup_cmd": REPO_METADATA["install_cmd"],
                "description": m["description"],
                "vague_description": m.get("vague_description", ""),
                "test_cmd": m["test_cmd"],
                "in_place": True,
                "timeout": 60,
                "mutation": {
                    "file": m["file"],
                    "original": m["original"],
                    "mutated": m["mutated"],
                    "line_num": m.get("line_num"),
                },
                "source": "handcrafted",
                "category": categorize_mutation(m["file"]),
            }
            f.write(json.dumps(task) + "\n")

        # Discovered mutations
        for m in discovered:
            task_id = generate_task_id(m)
            task = {
                "task_id": task_id,
                "repo_path": str(repo_path.resolve()),
                "repo_url": REPO_METADATA["repo_url"],
                "commit": REPO_METADATA["commit"],
                "setup_cmd": REPO_METADATA["install_cmd"],
                "description": generate_exact_description(m),
                "vague_description": generate_vague_description(m),
                "test_cmd": m["test_cmd"],
                "in_place": True,
                "timeout": 60,
                "mutation": {
                    "file": m["file"],
                    "original": m["original"],
                    "mutated": m["mutated"],
                    "line_num": m["line_num"],
                },
                "source": "discovered",
                "category": categorize_mutation(m["file"]),
                "mutation_type": m["mutation_type"],
            }
            f.write(json.dumps(task) + "\n")


def write_discovery_report(
    repo_path: Path,
    handcrafted: list[dict],
    all_discovered: list[dict],
    deduped: list[dict],
    selected: list[dict],
    selection_stats: dict,
    seed: int,
    target: int,
    output_dir: Path,
) -> None:
    """Write a full discovery report for auditability."""
    # Compute file hash of the discovered mutations JSON for integrity
    discovered_file = output_dir / "killed_mutations.json"
    file_hash = ""
    if discovered_file.exists():
        file_hash = hashlib.sha256(discovered_file.read_bytes()).hexdigest()[:16]

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo": REPO_METADATA,
        "selection_config": SELECTION_CONFIG,
        "seed": seed,
        "target_total": target,
        "handcrafted_count": len(handcrafted),
        "discovery": {
            "total_killed": len(all_discovered),
            "after_dedup": len(deduped),
            "selected": len(selected),
            "killed_mutations_sha256": file_hash,
        },
        "selection_stats": selection_stats,
        "final_counts": {
            "total_tasks": len(handcrafted) + len(selected),
            "handcrafted": len(handcrafted),
            "discovered": len(selected),
        },
        "category_distribution": dict(Counter(
            categorize_mutation(m["file"])
            for m in handcrafted + selected
        )),
        "mutation_type_distribution": dict(Counter(
            m.get("mutation_type", "handcrafted")
            for m in selected
        )),
        "file_distribution": dict(Counter(
            m["file"] for m in handcrafted + selected
        )),
        "all_candidates_summary": DISCOVERY_STATS,
    }

    report_file = output_dir / "discovery_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Discovery report written to {report_file}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate pydantic-ai eval tasks from hand-crafted + discovered mutations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Reproducibility:
  The --seed flag ensures deterministic selection. Running with the same
  seed and input data always produces the same output JSONL.

  A discovery_report.json is written alongside the output with full
  metadata: selection config, category/file distributions, and input
  file checksums.
        """,
    )
    parser.add_argument("--repo", required=True, help="Path to pydantic-ai repo")
    parser.add_argument(
        "--discovered", default="",
        help="Path to discovered mutations JSON (from discover_mutations.py)",
    )
    parser.add_argument(
        "--target", type=int, default=200,
        help="Target total number of tasks (default: 200, uses all available)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic selection (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="paper/experiments/eval_tasks_pydantic_ai.jsonl",
        help="Output JSONL file",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()

    # Verify commit matches
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_path, timeout=5,
        )
        actual_commit = result.stdout.strip()
        if actual_commit != REPO_METADATA["commit"]:
            print(f"WARNING: repo is at {actual_commit[:12]}, expected {REPO_METADATA['commit_short']}")
            print(f"  Run: cd {repo_path} && git checkout {REPO_METADATA['commit_short']}")
    except Exception:
        pass

    handcrafted = list(HANDCRAFTED)
    print(f"Hand-crafted mutations: {len(handcrafted)}")

    all_discovered: list[dict] = []
    deduped: list[dict] = []
    selected: list[dict] = []
    selection_stats: dict = {}

    if args.discovered:
        discovered_file = Path(args.discovered)
        if discovered_file.exists():
            all_discovered = load_discovered(discovered_file, repo_path=repo_path)
            print(f"Discovered mutations (killed, syntax-valid): {len(all_discovered)}")

            deduped = deduplicate(handcrafted, all_discovered)
            print(f"After dedup: {len(deduped)}")

            target_discovered = max(0, args.target - len(handcrafted))
            selected, selection_stats = select_diverse(
                deduped, target_discovered, seed=args.seed, config=SELECTION_CONFIG,
            )
            print(f"Selected {len(selected)} diverse discovered mutations (seed={args.seed})")

            # Category breakdown
            cats = Counter(categorize_mutation(m["file"]) for m in selected)
            print(f"Category distribution: {dict(sorted(cats.items()))}")

    total = len(handcrafted) + len(selected)
    print(f"\nTotal tasks: {total} ({len(handcrafted)} hand-crafted + {len(selected)} discovered)")

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_tasks(repo_path, handcrafted, selected, output_file)
    print(f"Tasks written to {output_file}")

    # Write discovery report
    write_discovery_report(
        repo_path, handcrafted, all_discovered, deduped, selected,
        selection_stats, args.seed, args.target, output_file.parent,
    )


if __name__ == "__main__":
    main()
