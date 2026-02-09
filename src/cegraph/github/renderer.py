"""Markdown renderer for PR Impact Bot.

Generates beautiful GitHub-flavored markdown comments with:
  - Risk score badge (color-coded)
  - Changed symbols table
  - Blast radius tree
  - Affected files
  - Suggested reviewers
"""

from __future__ import annotations

from cegraph.github.diff_parser import ChangedSymbol


def render_impact_comment(
    changed_symbols: list[ChangedSymbol],
    impact_results: list[dict],
    pr_title: str = "",
    stats: dict | None = None,
) -> str:
    """Render the full PR impact analysis as a GitHub markdown comment."""
    sections: list[str] = []

    # Header
    sections.append("## CeGraph Impact Analysis")
    sections.append("")

    if not changed_symbols:
        sections.append("> No code symbols were changed in this PR.")
        sections.append("")
        sections.append(_footer())
        return "\n".join(sections)

    # Aggregate risk
    max_risk = 0.0
    total_affected_files = set()
    total_callers = 0

    for impact in impact_results:
        if impact.get("found"):
            max_risk = max(max_risk, impact.get("risk_score", 0))
            total_affected_files.update(impact.get("affected_files", []))
            total_callers += len(impact.get("direct_callers", []))

    # Risk badge
    risk_emoji, risk_label, risk_color = _risk_badge(max_risk)
    sections.append(f"| {risk_emoji} Risk | Symbols Changed | Files Affected | Callers |")
    sections.append("|:---:|:---:|:---:|:---:|")
    sections.append(
        f"| **{risk_label}** ({max_risk:.0%}) | "
        f"{len(changed_symbols)} | "
        f"{len(total_affected_files)} | "
        f"{total_callers} |"
    )
    sections.append("")

    # Changed symbols table
    sections.append("### Changed Symbols")
    sections.append("")
    sections.append("| Symbol | Kind | File | Change | Risk |")
    sections.append("|:-------|:-----|:-----|:------:|:----:|")

    for sym, impact in zip(changed_symbols, impact_results):
        risk = impact.get("risk_score", 0) if impact.get("found") else 0
        emoji, label, _ = _risk_badge(risk)
        change_emoji = {"added": "+", "modified": "~", "deleted": "-"}.get(sym.change_type, "?")
        sections.append(
            f"| `{sym.qualified_name or sym.name}` | "
            f"{sym.kind} | "
            f"`{sym.file_path}` | "
            f"{change_emoji} {sym.change_type} | "
            f"{emoji} {risk:.0%} |"
        )
    sections.append("")

    # Blast radius details (for high-impact symbols)
    high_impact = [
        (sym, imp) for sym, imp in zip(changed_symbols, impact_results)
        if imp.get("found") and imp.get("risk_score", 0) >= 0.2
    ]

    if high_impact:
        sections.append("### Blast Radius")
        sections.append("")

        for sym, impact in high_impact:
            risk = impact.get("risk_score", 0)
            emoji, _, _ = _risk_badge(risk)
            sections.append("<details>")
            sections.append(
                f"<summary>{emoji} <code>{sym.qualified_name or sym.name}</code> "
                f"— {len(impact.get('affected_files', []))} files affected</summary>"
            )
            sections.append("")

            # Dependency tree
            direct = impact.get("direct_callers", [])
            if direct:
                sections.append("**Direct callers:**")
                for caller in direct[:15]:
                    sections.append(
                        f"- `{caller['name']}` ({caller['kind']}) "
                        f"in `{caller['file_path']}`"
                    )
                if len(direct) > 15:
                    sections.append(f"- ... and {len(direct) - 15} more")
                sections.append("")

            # Affected files
            files = impact.get("affected_files", [])
            if files:
                sections.append("**Affected files:**")
                sections.append("```")
                for f in _render_file_tree(files):
                    sections.append(f)
                sections.append("```")
                sections.append("")

            sections.append("</details>")
            sections.append("")

    # Summary stats
    if stats:
        sections.append("### Index Stats")
        sections.append(f"> {stats.get('total_nodes', 0)} symbols across "
                       f"{stats.get('files', 0)} files, "
                       f"{stats.get('total_edges', 0)} relationships")
        sections.append("")

    sections.append(_footer())
    return "\n".join(sections)


def _risk_badge(risk: float) -> tuple[str, str, str]:
    """Return (emoji, label, color) for a risk score."""
    if risk < 0.1:
        return ("🟢", "LOW", "green")
    elif risk < 0.2:
        return ("🟡", "LOW", "yellow")
    elif risk < 0.4:
        return ("🟠", "MEDIUM", "orange")
    elif risk < 0.6:
        return ("🔴", "HIGH", "red")
    else:
        return ("⛔", "CRITICAL", "red")


def _render_file_tree(files: list[str]) -> list[str]:
    """Render a list of file paths as an ASCII tree."""
    if not files:
        return []

    # Build tree structure
    tree: dict = {}
    for fp in sorted(files):
        parts = fp.split("/")
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    lines: list[str] = []
    _render_tree_recursive(tree, "", lines, is_last=True, is_root=True)
    return lines


def _render_tree_recursive(
    node: dict, prefix: str, lines: list[str],
    is_last: bool = True, is_root: bool = False
) -> None:
    """Recursively render tree nodes."""
    items = list(node.items())
    for i, (name, children) in enumerate(items):
        is_last_item = i == len(items) - 1
        connector = "└── " if is_last_item else "├── "
        if is_root:
            connector = ""
            next_prefix = ""
        else:
            next_prefix = prefix + ("    " if is_last_item else "│   ")

        if children:
            # Directory or intermediate node
            lines.append(f"{prefix}{connector}{name}/")
            _render_tree_recursive(children, next_prefix, lines)
        else:
            lines.append(f"{prefix}{connector}{name}")


def _footer() -> str:
    return (
        "---\n"
        "*Powered by [CeGraph](https://github.com/cegraph-ai/cegraph) "
        "— CAG-driven code intelligence*"
    )
