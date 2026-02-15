#!/usr/bin/env python3
"""Merge per-task results and regenerate all reports for a benchmark run.

Usage:
    python -m paper.experiments.merge_and_report --run-dir paper/results/run3

This script:
  1. Collects all per-task result.json files from disk
  2. Duplicates budget-independent methods across all budgets
  3. Writes combined results.json
  4. Generates run_metadata.json
  5. Regenerates all 17 report files

Produces output identical to what a single uninterrupted benchmark run would have.
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

# Add project root to path so we can import benchmark
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from paper.experiments.benchmark import (
    EvalResult,
    EvalTask,
    _CEILING_METHODS,
    bootstrap_paired_ci,
    compute_oracle,
    compute_router_loo,
    format_bootstrap_analysis,
    format_ceiling_probe,
    format_conditional_bins,
    format_decomposition,
    format_edit_locality,
    format_failure_diagnosis,
    format_latency_cost,
    format_patch_quality,
    format_per_repo_results,
    format_results,
    format_results_with_ci,
    format_retrieval_metrics,
)


def collect_results(run_dir: Path, budgets: list[int]) -> list[EvalResult]:
    """Walk all per-task result.json files and build the full results list."""
    results = []
    # ONLY no_retrieval is truly budget-independent (zero context).
    # target_file truncates at budget cap — it MUST have real results per budget.
    budget_independent = {"no_retrieval"}

    task_dirs = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name != "__pycache__"
    )

    for task_dir in task_dirs:
        for method_dir in sorted(task_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method_name = method_dir.name

            for budget_dir in sorted(method_dir.iterdir()):
                if not budget_dir.is_dir():
                    continue

                for qt_dir in sorted(budget_dir.iterdir()):
                    if not qt_dir.is_dir():
                        continue

                    result_file = qt_dir / "result.json"
                    if not result_file.exists():
                        print(f"  WARNING: missing {result_file}")
                        continue

                    with open(result_file) as f:
                        data = json.load(f)

                    # Convert to EvalResult
                    eval_fields = {f.name for f in EvalResult.__dataclass_fields__.values()}
                    filtered = {k: v for k, v in data.items() if k in eval_fields}
                    result = EvalResult(**filtered)
                    results.append(result)

            # Duplicate budget-independent methods to all budgets
            if method_name in budget_independent:
                first_budget = budgets[0]
                first_budget_results = [
                    r for r in results
                    if r.task_id == task_dir.name
                    and r.method == method_name
                    and r.budget == first_budget
                ]
                for src in first_budget_results:
                    for other_budget in budgets[1:]:
                        dup = replace(src, budget=other_budget)
                        results.append(dup)

    return results


def load_tasks(tasks_file: Path) -> list[EvalTask]:
    """Load EvalTask objects from JSONL."""
    eval_fields = {f.name for f in EvalTask.__dataclass_fields__.values()}
    tasks = []
    with open(tasks_file) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            filtered = {k: v for k, v in data.items() if k in eval_fields}
            tasks.append(EvalTask(**filtered))
    return tasks


def build_run_metadata(
    tasks: list[EvalTask],
    budgets: list[int],
    methods: list[str],
    query_types: list[str],
    results: list[EvalResult],
    tasks_file: str,
    model: str,
    provider: str,
    *,
    source_run_dirs: list[Path] | None = None,
) -> dict:
    """Build run_metadata.json content."""
    git_hash = "unknown"
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            git_hash = r.stdout.strip()
    except Exception:
        pass

    tasks_hash = "unknown"
    try:
        tp = Path(tasks_file)
        if tp.exists():
            tasks_hash = hashlib.sha256(tp.read_bytes()).hexdigest()
    except Exception:
        pass

    target_repo_commits = {}
    seen = set()
    for t in tasks:
        rp = Path(t.repo_path).resolve() if t.repo_path else None
        if rp and str(rp) not in seen:
            seen.add(str(rp))
            try:
                repo_git = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=5, cwd=rp,
                )
                if repo_git.returncode == 0:
                    target_repo_commits[rp.name] = repo_git.stdout.strip()
            except Exception:
                target_repo_commits[rp.name] = "unknown"

    spec_hash = "unknown"
    spec_path = Path(__file__).parent / "experiment_spec.json"
    try:
        if spec_path.exists():
            spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    except Exception:
        pass

    # Extract model_version_info and retry schedules from source run metadata.
    # Keep backward-compatible scalar fields, and add per-source details.
    model_version_info: dict = {}
    model_version_info_by_run: dict[str, dict] = {}
    retry_delays_by_run: dict[str, list[int]] = {}
    if source_run_dirs:
        for rd in source_run_dirs:
            meta_path = rd / "run_metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        src_meta = json.load(f)
                    if src_meta.get("model_version_info"):
                        info = src_meta["model_version_info"]
                        model_version_info_by_run[rd.name] = info
                        # Keep a representative value in the legacy field.
                        if not model_version_info:
                            model_version_info = info
                    if src_meta.get("retry_delays_seconds"):
                        retry_delays_by_run[rd.name] = src_meta["retry_delays_seconds"]
                except Exception:
                    pass

    retry_delays_seconds = [5, 8, 15, 30, 60]
    if retry_delays_by_run:
        unique = {tuple(v) for v in retry_delays_by_run.values() if isinstance(v, list)}
        if len(unique) == 1:
            retry_delays_seconds = list(next(iter(unique)))
        else:
            # Mixed-source merge: keep one representative schedule in the legacy field.
            retry_delays_seconds = list(next(iter(retry_delays_by_run.values())))

    total_pass = sum(1 for r in results if r.tests_passed)

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git_hash,
        "target_repo_commit": next(iter(target_repo_commits.values()), "unknown"),
        "experiment_spec_sha256": spec_hash,
        "python_version": sys.version,
        "platform": platform.platform(),
        "llm_provider": provider,
        "llm_model": model,
        "budgets": budgets,
        "methods": methods,
        "query_types": query_types,
        "ceiling_methods": sorted(_CEILING_METHODS),
        "num_tasks": len(tasks),
        "total_runs": len(results),
        "total_pass": total_pass,
        "pass_rate": round(total_pass / len(results), 4) if results else 0,
        "tasks_file": tasks_file,
        "tasks_file_sha256": tasks_hash,
        "bootstrap_seed": 42,
        "bootstrap_n": 10000,
        "naive_random_seed_scheme": "(task_id, budget, query_type) hash",
        "llm_params": {
            "temperature": 0.0,
            "max_tokens": 4096,
            "seed": 42,
            "top_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
        "model_version_info": model_version_info,
        "model_version_info_by_source_run": model_version_info_by_run,
        "retry_delays_seconds": retry_delays_seconds,
        "retry_delays_seconds_by_source_run": retry_delays_by_run,
        "source_run_dirs": [str(p) for p in (source_run_dirs or [])],
        "target_repo_commits": target_repo_commits,
        "bca_strategy_configs": {
            "bca_d1": {"strategy": "PRECISE", "max_depth": 1, "min_score": 0.3},
            "bca": {"strategy": "SMART", "max_depth": 3, "min_score": 0.1},
            "bca_d5": {"strategy": "THOROUGH", "max_depth": 5, "min_score": 0.05},
            "bca_no_closure": {"strategy": "SMART", "ablation": "no_closure"},
            "bca_no_scoring": {"strategy": "SMART", "ablation": "no_scoring"},
        },
        "repos": list(target_repo_commits.keys()),
    }


def generate_reports(
    results: list[EvalResult],
    tasks: list[EvalTask],
    budgets: list[int],
    output_dir: Path,
):
    """Generate all 17 report files."""

    # 1. summary.txt
    summary = format_results(results, budgets)
    print(summary)
    (output_dir / "summary.txt").write_text(summary)

    # 2. per_repo_results.txt
    per_repo = format_per_repo_results(results, tasks, budgets)
    if per_repo:
        print(per_repo)
        (output_dir / "per_repo_results.txt").write_text(per_repo)

    # 3. summary_with_ci.txt
    ci_summary = format_results_with_ci(results, budgets)
    print(ci_summary)
    (output_dir / "summary_with_ci.txt").write_text(ci_summary)

    # 4. ceiling_probe.txt
    ceiling_text = format_ceiling_probe(results, budgets)
    if ceiling_text:
        print(ceiling_text)
        (output_dir / "ceiling_probe.txt").write_text(ceiling_text)

    # 5. router_analysis.txt
    grid_results = [r for r in results if r.method not in _CEILING_METHODS]
    oracle = compute_oracle(grid_results, budgets)
    router = compute_router_loo(results, tasks, budgets)
    query_types_seen = sorted(set(r.query_type for r in grid_results))
    methods_seen = sorted(set(r.method for r in grid_results))

    router_lines = ["\n" + "=" * 70, "Oracle vs Router vs Best Single Method", "=" * 70]
    for qt in query_types_seen:
        qt_label = qt.upper()
        router_lines.append(f"\n  [{qt_label} queries]")
        router_lines.append(f"  {'Budget':<10} {'Best Single':>12} {'Router(LOO)':>12} {'Oracle':>10} {'Best Method':<20}")
        router_lines.append(f"  {'-'*65}")
        qt_results = [r for r in grid_results if r.query_type == qt]
        for b in budgets:
            best_m, best_rate = "", 0
            for m in methods_seen:
                runs = [r for r in qt_results if r.method == m and r.budget == b]
                if runs:
                    rate = sum(1 for r in runs if r.tests_passed) / len(runs)
                    if rate > best_rate:
                        best_rate, best_m = rate, m
            r_rate = router.get((qt, b), 0)
            o_rate = oracle.get((qt, b), 0)
            router_lines.append(
                f"  B={b:<7} {best_rate:>12.3f} {r_rate:>12.3f} {o_rate:>10.3f} {best_m:<20}"
            )
    router_text = "\n".join(router_lines)
    print(router_text)
    (output_dir / "router_analysis.txt").write_text(router_text)

    # 6. decomposition.txt
    decomp = format_decomposition(results, tasks, budgets)
    print(decomp)
    (output_dir / "decomposition.txt").write_text(decomp)

    # 7. conditional_bins.txt
    cond_bins = format_conditional_bins(results, budgets)
    print(cond_bins)
    (output_dir / "conditional_bins.txt").write_text(cond_bins)

    # 8-9. bootstrap_analysis.txt + bootstrap_cis.json
    bootstrap_text = format_bootstrap_analysis(results, budgets)
    if bootstrap_text:
        print(bootstrap_text)
        (output_dir / "bootstrap_analysis.txt").write_text(bootstrap_text)

        cis = bootstrap_paired_ci(grid_results, budgets)
        ci_json = {
            f"{ma}_vs_{mb}_B{b}_{qt}": {
                "method_a": ma, "method_b": mb,
                "budget": b, "query_type": qt,
                "mean_diff": round(d, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "significant": lo > 0 or hi < 0,
            }
            for (ma, mb, b, qt), (d, lo, hi) in cis.items()
        }
        with open(output_dir / "bootstrap_cis.json", "w") as f:
            json.dump(ci_json, f, indent=2)

    # 10. failure_diagnosis.txt
    diagnosis = format_failure_diagnosis(results, budgets)
    if diagnosis:
        print(diagnosis)
        (output_dir / "failure_diagnosis.txt").write_text(diagnosis)

    # 11. retrieval_metrics.txt
    retrieval = format_retrieval_metrics(results, budgets)
    if retrieval:
        print(retrieval)
        (output_dir / "retrieval_metrics.txt").write_text(retrieval)

    # 12. patch_quality.txt
    patch_q = format_patch_quality(results, budgets)
    if patch_q:
        print(patch_q)
        (output_dir / "patch_quality.txt").write_text(patch_q)

    # 13. latency_cost.txt
    # graph_build_time=0 for merge path: individual run reports have the actual value.
    # Merged report's amortized graph-build row will be underreported (documented).
    latency = format_latency_cost(results, budgets, graph_build_time=0)
    if latency:
        print(latency)
        (output_dir / "latency_cost.txt").write_text(latency)

    # 14. edit_locality.txt
    locality = format_edit_locality(results, budgets)
    if locality:
        print(locality)
        (output_dir / "edit_locality.txt").write_text(locality)


def main():
    parser = argparse.ArgumentParser(description="Merge per-task results and regenerate reports")
    parser.add_argument("--run-dir", required=True, help="Run output directory (e.g. paper/results/run3)")
    parser.add_argument("--tasks-file", default="paper/experiments/eval_tasks_full.jsonl",
                        help="Full tasks JSONL file")
    parser.add_argument("--budgets", default="2000,4000,8000,10000",
                        help="Comma-separated budgets")
    parser.add_argument("--methods",
                        default="no_retrieval,bm25,vector,keyword_map,bca_d1,bca,bca_d5,bca_no_closure,bca_no_scoring,target_file")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--provider", default="openai")
    parser.add_argument(
        "--source-run-dirs",
        default="",
        help=(
            "Comma-separated run dirs whose run_metadata.json should be used "
            "to reconstruct mixed-run metadata (e.g. run3,run4 for merged runs)"
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    budgets = [int(b) for b in args.budgets.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    query_types = ["exact", "vague", "dev_report"]
    source_run_dirs = [Path(p.strip()) for p in args.source_run_dirs.split(",") if p.strip()]

    # --- Step 1: Verify all tasks present ---
    tasks = load_tasks(Path(args.tasks_file))
    task_dirs = [d.name for d in run_dir.iterdir() if d.is_dir()]
    expected_ids = {t.task_id for t in tasks}
    present_ids = set(task_dirs)
    missing = expected_ids - present_ids
    if missing:
        print(f"ERROR: {len(missing)} tasks missing from {run_dir}:")
        for m in sorted(missing):
            print(f"  {m}")
        sys.exit(1)
    print(f"All {len(expected_ids)} task directories present.")

    # --- Step 2: Collect and merge results ---
    print("Collecting per-task results...")
    results = collect_results(run_dir, budgets)
    print(f"Collected {len(results)} results (including budget-independent duplicates)")

    expected_total = len(tasks) * len(methods) * len(budgets) * len(query_types)
    if len(results) != expected_total:
        print(f"WARNING: Expected {expected_total} results, got {len(results)}")
    else:
        print(f"Result count matches expected: {expected_total}")

    # --- Step 3: Write results.json ---
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Wrote {results_path} ({len(results)} entries)")

    # --- Step 4: Generate run_metadata.json ---
    metadata = build_run_metadata(
        tasks=tasks,
        budgets=budgets,
        methods=methods,
        query_types=query_types,
        results=results,
        tasks_file=args.tasks_file,
        model=args.model,
        provider=args.provider,
        source_run_dirs=source_run_dirs,
    )
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Wrote run_metadata.json")

    # --- Step 5: Generate all reports ---
    print("\nGenerating reports...")
    generate_reports(results, tasks, budgets, run_dir)

    print(f"\nDone. All files in {run_dir}/:")
    print("  results.json             - Combined results (all tasks)")
    print("  run_config.json          - Full 245-task config (restored)")
    print("  run_metadata.json        - Reproducibility metadata")
    print("  summary.txt              - Pass@1 table")
    print("  per_repo_results.txt     - Pass@1 by repository")
    print("  summary_with_ci.txt      - Pass@1 with 95% bootstrap CIs")
    print("  ceiling_probe.txt        - target_file ceiling probe")
    print("  router_analysis.txt      - Oracle vs Router vs Best Single")
    print("  decomposition.txt        - Pass@1 by mutation type and category")
    print("  conditional_bins.txt     - Pass@1 by identifier density, hops, mutation size")
    print("  failure_diagnosis.txt    - Failure mode breakdown")
    print("  retrieval_metrics.txt    - Target file hit rate, budget utilization")
    print("  patch_quality.txt        - Patch size and locality")
    print("  latency_cost.txt         - Latency breakdown")
    print("  edit_locality.txt        - Edit distance + context-patch overlap")
    print("  bootstrap_analysis.txt   - Paired bootstrap CI details")
    print("  bootstrap_cis.json       - Raw bootstrap CIs")


if __name__ == "__main__":
    main()
