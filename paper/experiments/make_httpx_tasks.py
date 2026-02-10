"""Generate eval_tasks.jsonl for httpx from discovered mutations.

Takes the output of discover_mutations.py (killed mutations JSON) and produces
a benchmark-ready JSONL file with full reproducibility metadata.

Usage:
    python -m paper.experiments.make_httpx_tasks \
        --repo /path/to/httpx \
        --discovered paper/experiments/httpx_killed.json \
        --output paper/experiments/eval_tasks_httpx.jsonl \
        --seed 42
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

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

SELECTION_CONFIG = {
    "max_per_file": 15,
    "max_per_category": 25,
    "max_per_mutation_type": 30,
    "min_per_category": 1,
    "shuffle_within_category": True,
}

REPO_METADATA = {
    "repo_url": "https://github.com/encode/httpx",
    "commit": "",  # Will be filled at runtime
    "commit_short": "",
    "source_dir": "httpx",
    "tests_dir": "tests",
    "install_cmd": "pip install -e '.[brotli,zstd,cli,http2,socks]' && pip install pytest uvicorn trio trustme chardet",
    "python_cmd": "",  # Will be filled at runtime
}


# -------------------------------------------------------------------------
# Category assignment
# -------------------------------------------------------------------------

FILE_CATEGORY_MAP = {
    "_api.py": "api",
    "_auth.py": "auth",
    "_client.py": "client",
    "_config.py": "config",
    "_content.py": "content",
    "_decoders.py": "decoders",
    "_exceptions.py": "exceptions",
    "_main.py": "cli",
    "_models.py": "models",
    "_multipart.py": "multipart",
    "_status_codes.py": "status_codes",
    "_urlparse.py": "url_parsing",
    "_urls.py": "urls",
    "_utils.py": "utils",
    "_types.py": "types",
}


def categorize_mutation(file_path: str) -> str:
    fname = Path(file_path).name
    if fname in FILE_CATEGORY_MAP:
        return FILE_CATEGORY_MAP[fname]
    parts = Path(file_path).parts
    if "_transports" in parts or "transports" in parts:
        return "transports"
    return "other"


# -------------------------------------------------------------------------
# Loading and selection (reused from make_pydantic_ai_tasks)
# -------------------------------------------------------------------------

def load_discovered(discovered_file: Path) -> list[dict]:
    with open(discovered_file) as f:
        data = json.load(f)
    return [m for m in data if m.get("killed") is True]


def select_diverse(
    discovered: list[dict],
    target_count: int,
    seed: int,
    config: dict,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)

    max_per_file = config["max_per_file"]
    max_per_category = config["max_per_category"]
    max_per_mutation_type = config.get("max_per_mutation_type")

    by_category: dict[str, list[dict]] = {}
    for m in discovered:
        cat = categorize_mutation(m["file"])
        by_category.setdefault(cat, []).append(m)

    if config.get("shuffle_within_category", True):
        for cat in by_category:
            rng.shuffle(by_category[cat])

    selected: list[dict] = []
    file_counts: dict[str, int] = {}
    category_counts: dict[str, int] = Counter()
    mutation_type_counts: dict[str, int] = Counter()

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

            for i, m in enumerate(remaining[cat]):
                file_key = m["file"]
                if file_counts.get(file_key, 0) >= max_per_file:
                    continue
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
            break

    stats = {
        "target_count": target_count,
        "actual_count": len(selected),
        "seed": seed,
        "categories_used": dict(category_counts),
        "files_used": dict(file_counts),
        "categories_available": {cat: len(muts) for cat, muts in by_category.items()},
        "mutation_type_counts": dict(mutation_type_counts),
    }

    return selected, stats


# -------------------------------------------------------------------------
# Description generation
# -------------------------------------------------------------------------

def generate_task_id(mutation: dict) -> str:
    file_stem = Path(mutation["file"]).stem.lstrip("_")
    return f"httpx-{file_stem}-L{mutation['line_num']}-{mutation['mutation_type']}"


def generate_exact_description(mutation: dict) -> str:
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
    mt = mutation["mutation_type"]
    selector = hash(f"{mutation['file']}:{mutation['line_num']}") % 3

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
        "return_value_swap": [
            "A function returns the wrong value. The return statement gives back the opposite of what the caller expects.",
            "A return value is incorrect. The function returns True when it should return False, or an empty result instead of a populated one.",
            "The return value from a function doesn't match its contract. Callers get unexpected results.",
        ],
    }

    if mt in templates:
        options = templates[mt]
        return options[selector % len(options)]

    return "Something isn't working correctly. Behavior doesn't match what the documentation describes."


# -------------------------------------------------------------------------
# JSONL output
# -------------------------------------------------------------------------

def write_tasks(
    repo_path: Path,
    discovered: list[dict],
    output_file: Path,
    python_cmd: str,
) -> None:
    with open(output_file, "w") as f:
        for m in discovered:
            task_id = generate_task_id(m)
            # Rewrite test_cmd to use relative python if it has an absolute path
            test_cmd = m["test_cmd"]
            if python_cmd != "python":
                test_cmd = test_cmd.replace(python_cmd, "python", 1)

            task = {
                "task_id": task_id,
                "repo_path": str(repo_path.resolve()),
                "repo_url": REPO_METADATA["repo_url"],
                "commit": REPO_METADATA["commit"],
                "setup_cmd": REPO_METADATA["install_cmd"],
                "description": generate_exact_description(m),
                "vague_description": generate_vague_description(m),
                "test_cmd": test_cmd,
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
    all_discovered: list[dict],
    selected: list[dict],
    selection_stats: dict,
    seed: int,
    target: int,
    discovered_file: Path,
    output_dir: Path,
) -> None:
    file_hash = ""
    if discovered_file.exists():
        file_hash = hashlib.sha256(discovered_file.read_bytes()).hexdigest()[:16]

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repo": REPO_METADATA,
        "selection_config": SELECTION_CONFIG,
        "seed": seed,
        "target_total": target,
        "discovery": {
            "total_killed": len(all_discovered),
            "selected": len(selected),
            "killed_mutations_sha256": file_hash,
        },
        "selection_stats": selection_stats,
        "final_counts": {
            "total_tasks": len(selected),
        },
        "category_distribution": dict(Counter(
            categorize_mutation(m["file"]) for m in selected
        )),
        "mutation_type_distribution": dict(Counter(
            m.get("mutation_type", "unknown") for m in selected
        )),
        "file_distribution": dict(Counter(
            m["file"] for m in selected
        )),
    }

    report_file = output_dir / "httpx_discovery_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Discovery report written to {report_file}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate httpx eval tasks from discovered mutations",
    )
    parser.add_argument("--repo", required=True, help="Path to httpx repo")
    parser.add_argument(
        "--discovered", required=True,
        help="Path to discovered mutations JSON (killed only)",
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
        "--python", default="python",
        help="Python executable used during discovery (for test_cmd rewriting)",
    )
    parser.add_argument(
        "--output",
        default="paper/experiments/eval_tasks_httpx.jsonl",
        help="Output JSONL file",
    )
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()

    # Get commit info
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=repo_path, timeout=5,
        )
        REPO_METADATA["commit"] = result.stdout.strip()
        REPO_METADATA["commit_short"] = result.stdout.strip()[:12]
    except Exception:
        pass

    REPO_METADATA["python_cmd"] = args.python

    discovered_file = Path(args.discovered)
    all_discovered = load_discovered(discovered_file)
    print(f"Discovered mutations (killed): {len(all_discovered)}")

    selected, selection_stats = select_diverse(
        all_discovered, args.target, seed=args.seed, config=SELECTION_CONFIG,
    )
    print(f"Selected {len(selected)} diverse mutations (seed={args.seed})")

    cats = Counter(categorize_mutation(m["file"]) for m in selected)
    print(f"Category distribution: {dict(sorted(cats.items()))}")

    mts = Counter(m.get("mutation_type", "unknown") for m in selected)
    print(f"Mutation type distribution: {dict(sorted(mts.items()))}")

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_tasks(repo_path, selected, output_file, args.python)
    print(f"\nTasks written to {output_file}")
    print(f"Total tasks: {len(selected)}")

    write_discovery_report(
        repo_path, all_discovered, selected,
        selection_stats, args.seed, args.target,
        discovered_file, output_file.parent,
    )


if __name__ == "__main__":
    main()
