#!/usr/bin/env python3
"""Generate all paper figures from benchmark results.

Usage:
    python -m paper.experiments.generate_figures --run-dir paper/results/run3

Produces PNG figures in paper/results/run3/figures/
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")

CEILING_METHODS = {"target_file"}

QT_LABELS = {
    "exact": "Dev-localized (exact)",
    "dev_report": "Dev-report (traceback)",
    "vague": "Vague (symptom-only)",
}

# Method display config
METHOD_COLORS = {
    "no_retrieval": "#888888",
    "bm25": "#2196F3",
    "vector": "#4CAF50",
    "embedding": "#00BCD4",
    "keyword_map": "#FF9800",
    "bca_d1": "#E91E63",
    "bca": "#9C27B0",
    "bca_d5": "#673AB7",
    "bca_no_closure": "#F44336",
    "bca_no_scoring": "#795548",
    "target_file": "#000000",
}

METHOD_LABELS = {
    "no_retrieval": "No Retrieval",
    "bm25": "BM25",
    "vector": "TF-IDF Vector",
    "embedding": "Dense Embedding",
    "keyword_map": "Keyword Map",
    "bca_d1": "BCA (d=1)",
    "bca": "BCA (d=3)",
    "bca_d5": "BCA (d=5)",
    "bca_no_closure": "BCA no-closure",
    "bca_no_scoring": "BCA no-scoring",
    "target_file": "Target File (ceiling)",
}

METHOD_ORDER = [
    "no_retrieval", "bm25", "vector", "embedding", "keyword_map",
    "bca_d1", "bca", "bca_d5", "bca_no_closure", "bca_no_scoring",
    "target_file",
]


def load_results(run_dir: Path) -> list[dict]:
    with open(run_dir / "results.json") as f:
        return json.load(f)


def pass_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("tests_passed")) / len(results)


# ─── Figure 1: Pass@1 vs Budget ──────────────────────────────────────────

def fig_pass_vs_budget(results, budgets, fig_dir):
    """Line chart: Pass@1 vs token budget, faceted by query type."""
    query_types = sorted(set(r["query_type"] for r in results))
    n_qt = len(query_types)
    fig, axes = plt.subplots(1, n_qt, figsize=(7 * n_qt, 5), sharey=True)
    if n_qt == 1:
        axes = [axes]

    for ax, qt in zip(axes, query_types):
        qt_results = [r for r in results if r["query_type"] == qt]
        methods_present = sorted(set(r["method"] for r in qt_results),
                                  key=lambda m: METHOD_ORDER.index(m)
                                  if m in METHOD_ORDER else 99)

        for m in methods_present:
            rates = []
            for b in budgets:
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                rates.append(pass_rate(runs))

            style = "--" if m in CEILING_METHODS else "-"
            marker = "s" if m in CEILING_METHODS else "o"
            ax.plot(budgets, rates, style, marker=marker, markersize=4,
                    color=METHOD_COLORS.get(m, "#666"),
                    label=METHOD_LABELS.get(m, m), linewidth=1.5)

        title = QT_LABELS.get(qt, qt.upper())
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Token Budget", fontsize=11)
        ax.set_xscale("linear")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k"))
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("Pass@1" if ax is axes[0] else "", fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle("Pass@1 by Method and Token Budget (N=245 tasks)", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_pass_vs_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1_pass_vs_budget.png")


# ─── Figure 2: Failure Mode Breakdown ────────────────────────────────────

def fig_failure_modes(results, budgets, fig_dir):
    """Stacked bar chart of failure modes at B=10000."""
    target_budget = max(budgets)

    failure_order = ["pass", "test_fail", "patch_apply_fail", "syntax_error",
                     "regression", "timeout", "no_patch"]
    failure_colors = {
        "pass": "#4CAF50",
        "test_fail": "#F44336",
        "patch_apply_fail": "#FF9800",
        "syntax_error": "#9C27B0",
        "regression": "#795548",
        "timeout": "#607D8B",
        "no_patch": "#BDBDBD",
    }

    query_types = sorted(set(r["query_type"] for r in results))
    n_qt = len(query_types)
    fig, axes = plt.subplots(1, n_qt, figsize=(7 * n_qt, 5))
    if n_qt == 1:
        axes = [axes]

    for ax, qt in zip(axes, query_types):
        qt_results = [r for r in results if r["query_type"] == qt
                      and r["method"] not in CEILING_METHODS
                      and r["budget"] == target_budget]
        methods = sorted(set(r["method"] for r in qt_results),
                          key=lambda m: METHOD_ORDER.index(m)
                          if m in METHOD_ORDER else 99)

        # Compute failure mode fractions
        data = {}
        for m in methods:
            runs = [r for r in qt_results if r["method"] == m]
            total = len(runs)
            if total == 0:
                continue
            counts = defaultdict(int)
            for r in runs:
                fm = r.get("failure_mode", "unknown")
                counts[fm] += 1
            data[m] = {fm: counts.get(fm, 0) / total for fm in failure_order}

        x_pos = range(len(methods))
        bottom = [0.0] * len(methods)

        for fm in failure_order:
            heights = [data.get(m, {}).get(fm, 0) for m in methods]
            ax.bar(x_pos, heights, bottom=bottom, label=fm,
                   color=failure_colors.get(fm, "#999"), width=0.7)
            bottom = [b + h for b, h in zip(bottom, heights)]

        title = QT_LABELS.get(qt, qt.upper())
        ax.set_title(f"{title} (B={target_budget})",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Fraction" if ax is axes[0] else "")
        ax.set_ylim(0, 1.05)

    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.suptitle(f"Failure Mode Breakdown at B={target_budget} (N=245)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_failure_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2_failure_modes.png")


# ─── Figure 3: Retrieval vs Outcome Scatter ──────────────────────────────

def fig_retrieval_vs_outcome(results, budgets, fig_dir):
    """Scatter: target file hit rate vs pass@1 per method×budget."""
    grid_results = [r for r in results if r["method"] not in CEILING_METHODS]
    methods = sorted(set(r["method"] for r in grid_results))

    query_types = sorted(set(r["query_type"] for r in grid_results))
    n_qt = len(query_types)
    fig, axes = plt.subplots(1, n_qt, figsize=(6 * n_qt, 5))
    if n_qt == 1:
        axes = [axes]

    for ax, qt in zip(axes, query_types):
        qt_results = [r for r in grid_results if r["query_type"] == qt]

        for m in methods:
            xs, ys, sizes = [], [], []
            for b in budgets:
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                if not runs:
                    continue
                file_hit = sum(1 for r in runs if r.get("target_file_hit")) / len(runs)
                p1 = pass_rate(runs)
                xs.append(file_hit)
                ys.append(p1)
                sizes.append(b / 200)  # Scale marker size by budget

            ax.scatter(xs, ys, c=METHOD_COLORS.get(m, "#666"), s=sizes,
                       label=METHOD_LABELS.get(m, m), alpha=0.7, edgecolors="white",
                       linewidth=0.5)

        # Diagonal reference
        ax.plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=1)

        title = QT_LABELS.get(qt, qt.upper())
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Target File Hit Rate", fontsize=11)
        ax.set_ylabel("Pass@1" if ax is axes[0] else "", fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle("Retrieval Quality vs Repair Success (marker size ~ budget)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_retrieval_vs_outcome.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_retrieval_vs_outcome.png")


# ─── Figure 4: Per-Repo Comparison ──────────────────────────────────────

def fig_per_repo(results, budgets, fig_dir):
    """Grouped bar chart: pass@1 by method, grouped by repo, at B=10000."""
    target_budget = max(budgets)
    grid_results = [r for r in results if r["method"] not in CEILING_METHODS
                    and r["budget"] == target_budget]
    repos = sorted(set(r.get("repo_name", "unknown") for r in grid_results))
    methods = sorted(set(r["method"] for r in grid_results),
                      key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)

    query_types = sorted(set(r["query_type"] for r in grid_results))
    n_qt = len(query_types)
    fig, axes = plt.subplots(1, n_qt, figsize=(7 * n_qt, 5))
    if n_qt == 1:
        axes = [axes]

    for ax, qt in zip(axes, query_types):
        qt_results = [r for r in grid_results if r["query_type"] == qt]

        x = range(len(methods))
        width = 0.35

        for i, repo in enumerate(repos):
            rates = []
            for m in methods:
                runs = [r for r in qt_results if r["method"] == m
                        and r.get("repo_name") == repo]
                rates.append(pass_rate(runs))
            offset = (i - len(repos)/2 + 0.5) * width
            ax.bar([xi + offset for xi in x], rates, width,
                   label=repo, alpha=0.8)

        title = QT_LABELS.get(qt, qt.upper())
        ax.set_title(f"{title} (B={target_budget})",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Pass@1" if ax is axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Pass@1 by Repository at B={target_budget}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_per_repo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4_per_repo.png")


# ─── Figure 5: Ceiling Probe ─────────────────────────────────────────────

def fig_ceiling(results, budgets, fig_dir):
    """Bar chart: target_file ceiling vs best method vs oracle."""
    query_types = sorted(set(r["query_type"] for r in results))
    n_qt = len(query_types)
    fig, axes = plt.subplots(1, n_qt, figsize=(6 * n_qt, 5))
    if n_qt == 1:
        axes = [axes]

    for ax, qt in zip(axes, query_types):
        qt_results = [r for r in results if r["query_type"] == qt]
        grid_qt = [r for r in qt_results if r["method"] not in CEILING_METHODS]

        ceiling_rates = []
        best_rates = []
        oracle_rates = []

        for b in budgets:
            # Ceiling (target_file)
            ceil_runs = [r for r in qt_results if r["method"] == "target_file"
                         and r["budget"] == b]
            ceiling_rates.append(pass_rate(ceil_runs))

            # Best single method
            methods = set(r["method"] for r in grid_qt)
            best = 0
            for m in methods:
                runs = [r for r in grid_qt if r["method"] == m and r["budget"] == b]
                best = max(best, pass_rate(runs))
            best_rates.append(best)

            # Oracle
            by_task = defaultdict(list)
            for r in grid_qt:
                if r["budget"] == b:
                    by_task[r["task_id"]].append(r)
            oracle_pass = sum(1 for trs in by_task.values()
                              if any(r.get("tests_passed") for r in trs))
            oracle_rates.append(oracle_pass / len(by_task) if by_task else 0)

        x = range(len(budgets))
        width = 0.25
        ax.bar([xi - width for xi in x], best_rates, width, label="Best Single Method",
               color="#2196F3", alpha=0.8)
        ax.bar(x, oracle_rates, width, label="Oracle (any method)",
               color="#4CAF50", alpha=0.8)
        ax.bar([xi + width for xi in x], ceiling_rates, width, label="Target File (ceiling)",
               color="#FF9800", alpha=0.8)

        title = QT_LABELS.get(qt, qt.upper())
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b//1000}k" for b in budgets])
        ax.set_xlabel("Token Budget")
        ax.set_ylabel("Pass@1" if ax is axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Best Method vs Oracle vs Ceiling",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_ceiling_probe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig5_ceiling_probe.png")


# ─── Figure 6: Ablation (BCA variants) ──────────────────────────────────

def fig_ablation(results, budgets, fig_dir):
    """Line chart: BCA depth/ablation variants."""
    bca_methods = ["bca_d1", "bca", "bca_d5", "bca_no_closure", "bca_no_scoring"]

    query_types = sorted(set(r["query_type"] for r in results))
    n_qt = len(query_types)
    fig, axes = plt.subplots(1, n_qt, figsize=(6 * n_qt, 5))
    if n_qt == 1:
        axes = [axes]

    for ax, qt in zip(axes, query_types):
        qt_results = [r for r in results if r["query_type"] == qt]

        for m in bca_methods:
            rates = []
            for b in budgets:
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                rates.append(pass_rate(runs))
            ax.plot(budgets, rates, "-o", markersize=5,
                    color=METHOD_COLORS.get(m, "#666"),
                    label=METHOD_LABELS.get(m, m), linewidth=1.5)

        title = QT_LABELS.get(qt, qt.upper())
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Token Budget", fontsize=11)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k"))
        ax.set_ylabel("Pass@1" if ax is axes[0] else "", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("BCA Ablation: Depth and Component Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig6_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig6_ablation.png")


# ─── Figure 7: Heatmap — Method × Mutation Type ─────────────────────────

def fig_mutation_heatmap(results, budgets, fig_dir):
    """Heatmap: pass@1 by method × mutation_type at B=10000."""
    target_budget = max(budgets)

    query_types = sorted(set(r["query_type"] for r in results))
    n_qt = len(query_types)
    fig, axes = plt.subplots(1, n_qt, figsize=(7 * n_qt, 6))
    if n_qt == 1:
        axes = [axes]

    for ax, qt in zip(axes, query_types):
        qt_results = [r for r in results if r["query_type"] == qt
                      and r["budget"] == target_budget
                      and r["method"] not in CEILING_METHODS]

        methods = sorted(set(r["method"] for r in qt_results),
                          key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
        mut_types = sorted(set(r.get("mutation_type", "unknown") for r in qt_results))

        matrix = []
        for m in methods:
            row = []
            for mt in mut_types:
                runs = [r for r in qt_results if r["method"] == m
                        and r.get("mutation_type") == mt]
                row.append(pass_rate(runs) if runs else 0)
            matrix.append(row)

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(mut_types)))
        ax.set_xticklabels(mut_types, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=8)
        title = QT_LABELS.get(qt, qt.upper())
        ax.set_title(f"{title} (B={target_budget})", fontsize=13, fontweight="bold")

        # Add value annotations
        for i in range(len(methods)):
            for j in range(len(mut_types)):
                val = matrix[i][j]
                color = "white" if val < 0.3 or val > 0.7 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=6, color=color)

    fig.colorbar(im, ax=axes, label="Pass@1", shrink=0.8)
    fig.suptitle("Pass@1 by Method × Mutation Type",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig7_mutation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig7_mutation_heatmap.png")


# ─── Figure 8: Entity Density Effect ────────────────────────────────────

def fig_entity_density(results, budgets, fig_dir):
    """Bar chart: pass@1 for tasks with/without identifiers."""
    target_budget = max(budgets)

    fig, ax = plt.subplots(figsize=(10, 5))

    grid_results = [r for r in results if r["method"] not in CEILING_METHODS
                    and r["budget"] == target_budget]
    methods = sorted(set(r["method"] for r in grid_results),
                      key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)

    x = range(len(methods))

    query_types = sorted(set(r["query_type"] for r in grid_results))
    n_qt = len(query_types)
    width = 0.8 / max(n_qt, 1)

    for i, qt in enumerate(query_types):
        label = QT_LABELS.get(qt, qt)
        qt_results = [r for r in grid_results if r["query_type"] == qt]
        rates = [pass_rate([r for r in qt_results if r["method"] == m]) for m in methods]
        offset = (i - n_qt / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], rates, width, label=label, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Pass@1", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(f"Effect of Query Specificity on Pass@1 (B={target_budget}, N=245)",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(fig_dir / "fig8_entity_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig8_entity_density.png")


def main():
    if not HAS_MPL:
        print("ERROR: matplotlib required. pip install matplotlib")
        return

    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    results = load_results(run_dir)
    budgets = sorted(set(r["budget"] for r in results))

    print(f"Loaded {len(results)} results, generating figures...")

    fig_pass_vs_budget(results, budgets, fig_dir)
    fig_failure_modes(results, budgets, fig_dir)
    fig_retrieval_vs_outcome(results, budgets, fig_dir)
    fig_per_repo(results, budgets, fig_dir)
    fig_ceiling(results, budgets, fig_dir)
    fig_ablation(results, budgets, fig_dir)
    fig_mutation_heatmap(results, budgets, fig_dir)
    fig_entity_density(results, budgets, fig_dir)

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
