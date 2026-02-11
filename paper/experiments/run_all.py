"""Run all experiments and produce aggregate tables for the paper.

Usage:
    python -m paper.experiments.run_all --repo /path/to/repo
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from paper.experiments.ablation import AblationResult, run_ablation
from paper.experiments.baselines import BaselineResult, run_comparison


def aggregate_baselines(results: list[BaselineResult]) -> str:
    """Produce per-method, per-budget aggregate table."""
    # Group by (method, budget)
    groups: dict[tuple[str, int], list[BaselineResult]] = defaultdict(list)
    for r in results:
        groups[(r.method, r.budget)].append(r)

    methods = ["grep", "bm25", "repo_map", "unweighted_bfs", "bca"]
    budgets = sorted(set(r.budget for r in results))

    lines = []

    # Table 1: Mean recall by method and budget
    lines.append("=" * 70)
    lines.append("Table 1: Mean Recall (across tasks)")
    lines.append("=" * 70)
    header = f"{'Method':<20}" + "".join(f"  B={b:>5}" for b in budgets)
    lines.append(header)
    lines.append("-" * 70)
    for m in methods:
        row = f"{m:<20}"
        for b in budgets:
            rs = groups.get((m, b), [])
            recalls = [r.recall for r in rs if r.recall is not None]
            if recalls:
                row += f"  {statistics.mean(recalls):>6.3f}"
            else:
                row += f"  {'n/a':>6}"
        lines.append(row)

    lines.append("")

    # Table 2: Mean tokens used / budget adherence
    lines.append("=" * 70)
    lines.append("Table 2: Mean Budget Usage (tokens / budget)")
    lines.append("=" * 70)
    lines.append(header)
    lines.append("-" * 70)
    for m in methods:
        row = f"{m:<20}"
        for b in budgets:
            rs = groups.get((m, b), [])
            if rs:
                mean_pct = statistics.mean(r.tokens_used / r.budget * 100 for r in rs)
                row += f"  {mean_pct:>5.1f}%"
            else:
                row += f"  {'n/a':>6}"
        lines.append(row)

    lines.append("")

    # Table 3: Budget violations (exceeds budget)
    lines.append("=" * 70)
    lines.append("Table 3: Budget Violations (tokens > budget)")
    lines.append("=" * 70)
    lines.append(header)
    lines.append("-" * 70)
    for m in methods:
        row = f"{m:<20}"
        for b in budgets:
            rs = groups.get((m, b), [])
            if rs:
                violations = sum(1 for r in rs if r.tokens_used > r.budget)
                row += f"  {violations:>4}/{len(rs):<1}"
            else:
                row += f"  {'n/a':>6}"
        lines.append(row)

    lines.append("")

    # Table 4: Mean symbols per 1k tokens (efficiency)
    lines.append("=" * 70)
    lines.append("Table 4: Symbols per 1000 Tokens (efficiency)")
    lines.append("=" * 70)
    lines.append(header)
    lines.append("-" * 70)
    for m in methods:
        row = f"{m:<20}"
        for b in budgets:
            rs = groups.get((m, b), [])
            if rs:
                efficiencies = []
                for r in rs:
                    if r.tokens_used > 0:
                        efficiencies.append(r.symbols_selected / r.tokens_used * 1000)
                if efficiencies:
                    row += f"  {statistics.mean(efficiencies):>6.1f}"
                else:
                    row += f"  {'n/a':>6}"
            else:
                row += f"  {'n/a':>6}"
        lines.append(row)

    lines.append("")

    # Table 5: Mean assembly time
    lines.append("=" * 70)
    lines.append("Table 5: Mean Assembly Time (ms)")
    lines.append("=" * 70)
    lines.append(header)
    lines.append("-" * 70)
    for m in methods:
        row = f"{m:<20}"
        for b in budgets:
            rs = groups.get((m, b), [])
            if rs:
                mean_t = statistics.mean(r.assembly_time_ms for r in rs)
                row += f"  {mean_t:>5.1f}m"
            else:
                row += f"  {'n/a':>6}"
        lines.append(row)

    return "\n".join(lines)


def aggregate_ablation(results: list[AblationResult]) -> str:
    """Produce per-config, per-budget aggregate table."""
    groups: dict[tuple[str, int], list[AblationResult]] = defaultdict(list)
    for r in results:
        groups[(r.config_name, r.budget)].append(r)

    configs = [
        "full", "-dependency_closure", "-submodular_coverage",
        "-centrality_scoring", "-file_proximity", "-kind_weights",
        "-dependency_ordering", "base_bfs_only", "+pagerank", "+learned_weights",
    ]
    budgets = sorted(set(r.budget for r in results))

    lines = []

    lines.append("=" * 70)
    lines.append("Table 6: Ablation - Mean Recall")
    lines.append("=" * 70)
    header = f"{'Config':<25}" + "".join(f"  B={b:>5}" for b in budgets)
    lines.append(header)
    lines.append("-" * 70)
    for c in configs:
        row = f"{c:<25}"
        for b in budgets:
            rs = groups.get((c, b), [])
            recalls = [r.recall for r in rs if r.recall is not None]
            if recalls:
                row += f"  {statistics.mean(recalls):>6.3f}"
            else:
                row += f"  {'n/a':>6}"
        lines.append(row)

    lines.append("")

    lines.append("=" * 70)
    lines.append("Table 7: Ablation - Mean Budget Usage (%)")
    lines.append("=" * 70)
    lines.append(header)
    lines.append("-" * 70)
    for c in configs:
        row = f"{c:<25}"
        for b in budgets:
            rs = groups.get((c, b), [])
            if rs:
                mean_pct = statistics.mean(r.budget_used_pct for r in rs)
                row += f"  {mean_pct:>5.1f}%"
            else:
                row += f"  {'n/a':>6}"
        lines.append(row)

    lines.append("")

    lines.append("=" * 70)
    lines.append("Table 8: Ablation - Budget Violations")
    lines.append("=" * 70)
    lines.append(header)
    lines.append("-" * 70)
    for c in configs:
        row = f"{c:<25}"
        for b in budgets:
            rs = groups.get((c, b), [])
            if rs:
                violations = sum(1 for r in rs if r.tokens_used > r.budget)
                row += f"  {violations:>4}/{len(rs):<1}"
            else:
                row += f"  {'n/a':>6}"
        lines.append(row)

    return "\n".join(lines)


def generate_latex_baseline_table(results: list[BaselineResult]) -> str:
    """Generate LaTeX table for baselines."""
    groups: dict[tuple[str, int], list[BaselineResult]] = defaultdict(list)
    for r in results:
        groups[(r.method, r.budget)].append(r)

    methods = [
        ("grep", "Full-file grep"),
        ("bm25", "BM25"),
        ("repo_map", "Repo map"),
        ("unweighted_bfs", "Unweighted BFS"),
        ("bca", "BCA (ours)"),
    ]
    budgets = sorted(set(r.budget for r in results))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline comparison on \textsc{CeGraph} self-evaluation (12 tasks, mean across tasks).}",
        r"\label{tab:baselines}",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(budgets) * 2 + "}",
        r"\toprule",
    ]

    # Header
    header = r"& " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{$B={b}$}}" for b in budgets
    ) + r" \\"
    lines.append(header)

    subheader = r"\textbf{Method}"
    for _ in budgets:
        subheader += r" & \textbf{Recall} & \textbf{Use\%}"
    subheader += r" \\"
    lines.append(r"\cmidrule(lr){2-" + str(1 + len(budgets) * 2) + "}")
    lines.append(subheader)
    lines.append(r"\midrule")

    for method_key, method_name in methods:
        row = method_name
        for b in budgets:
            rs = groups.get((method_key, b), [])
            recalls = [r.recall for r in rs if r.recall is not None]
            mean_recall = statistics.mean(recalls) if recalls else 0
            mean_use = statistics.mean(r.tokens_used / r.budget * 100 for r in rs) if rs else 0
            violations = sum(1 for r in rs if r.tokens_used > r.budget)
            use_str = f"{mean_use:.0f}\\%"
            if violations > 0:
                use_str = f"\\textcolor{{red}}{{{mean_use:.0f}\\%}}"
            row += f" & {mean_recall:.2f} & {use_str}"
        row += r" \\"
        if method_key == "bca":
            row = r"\textbf{" + method_name + "}"
            for b in budgets:
                rs = groups.get((method_key, b), [])
                recalls = [r.recall for r in rs if r.recall is not None]
                mean_recall = statistics.mean(recalls) if recalls else 0
                mean_use = statistics.mean(r.tokens_used / r.budget * 100 for r in rs) if rs else 0
                row += f" & \\textbf{{{mean_recall:.2f}}} & {mean_use:.0f}\\%"
            row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_latex_ablation_table(results: list[AblationResult]) -> str:
    """Generate LaTeX table for ablation."""
    groups: dict[tuple[str, int], list[AblationResult]] = defaultdict(list)
    for r in results:
        groups[(r.config_name, r.budget)].append(r)

    configs = [
        ("full", "Full BCA"),
        ("-dependency_closure", "$-$Dep. closure"),
        ("-submodular_coverage", "$-$Submod. coverage"),
        ("-centrality_scoring", "$-$Centrality"),
        ("-file_proximity", "$-$File proximity"),
        ("-kind_weights", "$-$Kind weights"),
        ("-dependency_ordering", "$-$Dep. ordering"),
        ("base_bfs_only", "Base BFS only"),
        ("+pagerank", "$+$PageRank"),
        ("+learned_weights", "$+$Learned weights"),
    ]
    budgets = sorted(set(r.budget for r in results))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: mean recall and budget usage across 12 tasks.}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{l" + "r" * len(budgets) * 2 + "}",
        r"\toprule",
    ]

    header = r"& " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{$B={b}$}}" for b in budgets
    ) + r" \\"
    lines.append(header)
    subheader = r"\textbf{Config}"
    for _ in budgets:
        subheader += r" & \textbf{Recall} & \textbf{Use\%}"
    subheader += r" \\"
    lines.append(r"\cmidrule(lr){2-" + str(1 + len(budgets) * 2) + "}")
    lines.append(subheader)
    lines.append(r"\midrule")

    for config_key, config_name in configs:
        row = config_name
        for b in budgets:
            rs = groups.get((config_key, b), [])
            recalls = [r.recall for r in rs if r.recall is not None]
            mean_recall = statistics.mean(recalls) if recalls else 0
            mean_use = statistics.mean(r.budget_used_pct for r in rs) if rs else 0
            violations = sum(1 for r in rs if r.tokens_used > r.budget)
            use_str = f"{mean_use:.0f}\\%"
            if violations > 0:
                use_str = f"\\textcolor{{red}}{{{mean_use:.0f}\\%}}"
            row += f" & {mean_recall:.2f} & {use_str}"
        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run all paper experiments")
    parser.add_argument("--repo", required=True, help="Path to repository")
    parser.add_argument(
        "--tasks-file",
        default="paper/experiments/tasks.jsonl",
        help="JSONL file with tasks",
    )
    parser.add_argument(
        "--budgets", default="1000,2000,4000,8000",
        help="Comma-separated budget values",
    )
    parser.add_argument("--output-dir", default="paper/experiments", help="Output directory")
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    budgets = [int(b) for b in args.budgets.split(",")]
    output_dir = Path(args.output_dir)

    with open(args.tasks_file) as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    print(f"Running experiments: {len(tasks)} tasks, budgets={budgets}")
    print(f"Repository: {repo_path}")
    print()

    # Run baselines
    print("--- Baseline Comparison ---")
    all_baseline: list[BaselineResult] = []
    for t in tasks:
        task_str = t["task"]
        gt = t.get("ground_truth")
        results = run_comparison(repo_path, task_str, budgets, gt)
        all_baseline.extend(results)
        recalls = {r.method: r.recall for r in results if r.budget == budgets[-1]}
        print(f"  {task_str[:60]:60s} recall@{budgets[-1]}: {recalls}")

    print()
    print(aggregate_baselines(all_baseline))

    # Run ablation
    print("\n\n--- Ablation Study ---")
    all_ablation: list[AblationResult] = []
    for t in tasks:
        task_str = t["task"]
        gt = t.get("ground_truth")
        results = run_ablation(repo_path, task_str, budgets, gt)
        all_ablation.extend(results)

    print(aggregate_ablation(all_ablation))

    # Save raw results
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump([asdict(r) for r in all_baseline], f, indent=2)

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump([asdict(r) for r in all_ablation], f, indent=2)

    # Generate LaTeX tables
    latex_baseline = generate_latex_baseline_table(all_baseline)
    latex_ablation = generate_latex_ablation_table(all_ablation)

    with open(output_dir / "tables.tex", "w") as f:
        f.write("% Auto-generated from paper.experiments.run_all\n")
        f.write("% Do not edit manually.\n\n")
        f.write(latex_baseline)
        f.write("\n\n")
        f.write(latex_ablation)

    print(f"\nLaTeX tables written to {output_dir / 'tables.tex'}")
    print(f"Raw results written to {output_dir}")


if __name__ == "__main__":
    main()
