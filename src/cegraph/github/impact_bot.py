"""PR Impact Bot — automatic blast radius analysis for pull requests.

This is the main entry point for the GitHub Action. It:
1. Parses the PR diff to find changed symbols
2. Runs impact analysis on each changed symbol
3. Generates a beautiful markdown comment
4. Posts it to the PR (or outputs it for the Action to post)

Usage:
    # As a CLI command
    cegraph impact-pr --base main --format markdown

    # In a GitHub Action
    cegraph impact-pr --format github-comment
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from cegraph.github.diff_parser import (
    get_changed_symbols,
    get_git_diff,
    get_pr_diff,
    parse_diff,
)
from cegraph.github.renderer import render_impact_comment


def run_impact_analysis(
    root: Path,
    base: str = "main",
    is_pr: bool = False,
) -> dict:
    """Run the full impact analysis pipeline.

    Returns:
        Dict with 'comment' (markdown), 'risk_score', 'changed_symbols', etc.
    """
    # Load the knowledge graph
    from cegraph.config import GRAPH_DB_FILE, get_cegraph_dir
    from cegraph.graph.query import GraphQuery
    from cegraph.graph.store import GraphStore

    db_path = get_cegraph_dir(root) / GRAPH_DB_FILE
    store = GraphStore(db_path)
    graph = store.load()

    if graph is None:
        store.close()
        return {
            "error": "No CeGraph index found. Run 'cegraph init' first.",
            "comment": "## CeGraph Impact Analysis\n\n> No index found. Run `cegraph init` first.",
        }

    query = GraphQuery(graph, store)

    # Get the diff
    if is_pr:
        diff_text = get_pr_diff(root)
    else:
        diff_text = get_git_diff(root, base)

    if not diff_text:
        store.close()
        return {
            "comment": "## CeGraph Impact Analysis\n\n> No changes detected.",
            "risk_score": 0,
            "changed_symbols": [],
        }

    # Parse diff into file changes
    file_diffs = parse_diff(diff_text)

    # Map changes to symbols
    changed_symbols = get_changed_symbols(root, graph, file_diffs)

    # Run impact analysis on each changed symbol (use qualified_name for precision)
    impact_results = []
    for sym in changed_symbols:
        impact = query.impact_of(sym.qualified_name)
        if not impact.get("found"):
            # Fall back to short name if qualified name doesn't match
            impact = query.impact_of(sym.name)
        impact_results.append(impact)

    # Get stats for the comment
    stats = store.get_metadata("stats")

    # Generate the markdown comment
    comment = render_impact_comment(
        changed_symbols=changed_symbols,
        impact_results=impact_results,
        stats=stats,
    )

    # Compute aggregate risk
    max_risk = 0.0
    for impact in impact_results:
        if impact.get("found"):
            max_risk = max(max_risk, impact.get("risk_score", 0))

    store.close()

    return {
        "comment": comment,
        "risk_score": max_risk,
        "changed_symbols": [
            {
                "name": s.name,
                "kind": s.kind,
                "file": s.file_path,
                "change_type": s.change_type,
            }
            for s in changed_symbols
        ],
        "files_affected": len(set(
            f for impact in impact_results
            for f in impact.get("affected_files", [])
            if impact.get("found")
        )),
        "file_diffs": [
            {
                "path": fd.path,
                "status": fd.status,
                "added": fd.added_lines,
                "deleted": fd.deleted_lines,
            }
            for fd in file_diffs
        ],
    }


def post_github_comment(comment: str) -> bool:
    """Post a comment to the current PR using the GitHub API.

    Requires GITHUB_TOKEN and GITHUB_EVENT_PATH environment variables
    (available in GitHub Actions).
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return False

    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path or not Path(event_path).exists():
        return False

    with open(event_path) as f:
        event = json.load(f)

    pr_number = event.get("pull_request", {}).get("number")
    repo = os.environ.get("GITHUB_REPOSITORY", "")

    if not pr_number or not repo:
        return False

    # Find and update existing comment, or create new one
    marker = "<!-- cegraph-impact-bot -->"
    comment_with_marker = f"{marker}\n{comment}"

    try:
        # Check for existing comment
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/issues/{pr_number}/comments",
             "--jq", f'[.[] | select(.body | startswith("{marker}"))][0].id'],
            capture_output=True, text=True, timeout=15,
        )

        existing_id = result.stdout.strip()

        if existing_id and existing_id != "null":
            # Update existing comment
            update = subprocess.run(
                ["gh", "api", "--method", "PATCH",
                 f"repos/{repo}/issues/comments/{existing_id}",
                 "-f", f"body={comment_with_marker}"],
                capture_output=True, timeout=15,
            )
            if update.returncode != 0:
                return False
        else:
            # Create new comment
            create = subprocess.run(
                ["gh", "api", "--method", "POST",
                 f"repos/{repo}/issues/{pr_number}/comments",
                 "-f", f"body={comment_with_marker}"],
                capture_output=True, timeout=15,
            )
            if create.returncode != 0:
                return False

        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
