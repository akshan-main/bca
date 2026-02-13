#!/usr/bin/env python3
"""Post-hoc analyses on benchmark results. No reruns needed.

Usage:
    python -m paper.experiments.posthoc_analysis --run-dir paper/results/run3

Produces:
  - cost_analysis.txt          Cost per solved task by method
  - router_logistic.txt        Logistic regression router results
  - retrieval_outcome.txt      Retrieval-outcome decoupling analysis
  - context_redundancy.txt     Context symbol coverage and efficiency
  - khop_coverage.txt          k-hop conditional analysis
  - posthoc_summary.txt        Combined summary of all post-hoc analyses
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# gpt-4o-mini-2024-07-18 pricing (per 1M tokens, as of 2024)
INPUT_PRICE_PER_M = 0.15   # $0.15 per 1M input tokens
OUTPUT_PRICE_PER_M = 0.60  # $0.60 per 1M output tokens

CEILING_METHODS = {"target_file"}


def load_results(run_dir: Path) -> list[dict]:
    with open(run_dir / "results.json") as f:
        return json.load(f)


# ─── 1. Cost per Solved Task ───────────────────────────────────────────────

def cost_analysis(results: list[dict], budgets: list[int]) -> str:
    lines = [
        "",
        "=" * 90,
        "Cost Analysis: Token Usage & Cost per Solved Task",
        "  Model: gpt-4o-mini-2024-07-18",
        f"  Pricing: ${INPUT_PRICE_PER_M}/M input, ${OUTPUT_PRICE_PER_M}/M output",
        "=" * 90,
    ]

    query_types = sorted(set(r["query_type"] for r in results
                             if r["method"] not in CEILING_METHODS))
    for qt in query_types:
        qt_results = [r for r in results if r["query_type"] == qt
                      and r["method"] not in CEILING_METHODS]
        methods = sorted(set(r["method"] for r in qt_results))

        # --- Cost per attempt ---
        lines.append(f"\n  --- Mean Cost per Attempt (USD) [{qt.upper()} queries] ---")
        lines.append(f"  {'Method':<22} " + "  ".join(f"B={b:>7}" for b in budgets))
        lines.append(f"  {'-' * (22 + 12 * len(budgets))}")

        for m in methods:
            vals = []
            for b in budgets:
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                if not runs:
                    vals.append("---")
                    continue
                costs = []
                for r in runs:
                    inp = r.get("llm_input_tokens", 0) or 0
                    out = r.get("llm_output_tokens", 0) or 0
                    cost = inp * INPUT_PRICE_PER_M / 1e6 + out * OUTPUT_PRICE_PER_M / 1e6
                    costs.append(cost)
                mean_cost = sum(costs) / len(costs)
                vals.append(f"${mean_cost:.5f}")
            lines.append(f"  {m:<22} " + "  ".join(f"{v:>9}" for v in vals))

        # --- Cost per solved task ---
        lines.append(f"\n  --- Cost per Solved Task (USD, passing only) [{qt.upper()} queries] ---")
        lines.append(f"  {'Method':<22} " + "  ".join(f"B={b:>7}" for b in budgets)
                      + "    Solves")
        lines.append(f"  {'-' * (22 + 12 * len(budgets) + 10)}")

        for m in methods:
            vals = []
            total_solves = 0
            for b in budgets:
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                passing = [r for r in runs if r.get("tests_passed")]
                total_solves += len(passing)
                if not passing:
                    vals.append("---")
                    continue
                costs = []
                for r in passing:
                    inp = r.get("llm_input_tokens", 0) or 0
                    out = r.get("llm_output_tokens", 0) or 0
                    cost = inp * INPUT_PRICE_PER_M / 1e6 + out * OUTPUT_PRICE_PER_M / 1e6
                    costs.append(cost)
                mean_cost = sum(costs) / len(costs)
                vals.append(f"${mean_cost:.5f}")
            lines.append(f"  {m:<22} " + "  ".join(f"{v:>9}" for v in vals)
                          + f"    {total_solves:>4}")

        # --- Total benchmark cost ---
        lines.append(f"\n  --- Total Benchmark Cost [{qt.upper()} queries] ---")
        total_cost = 0
        for r in qt_results:
            inp = r.get("llm_input_tokens", 0) or 0
            out = r.get("llm_output_tokens", 0) or 0
            total_cost += inp * INPUT_PRICE_PER_M / 1e6 + out * OUTPUT_PRICE_PER_M / 1e6
        lines.append(f"  Total: ${total_cost:.2f} across {len(qt_results)} attempts")

        # Per-method totals
        for m in methods:
            m_results = [r for r in qt_results if r["method"] == m]
            m_cost = sum(
                (r.get("llm_input_tokens", 0) or 0) * INPUT_PRICE_PER_M / 1e6
                + (r.get("llm_output_tokens", 0) or 0) * OUTPUT_PRICE_PER_M / 1e6
                for r in m_results
            )
            lines.append(f"    {m:<20}: ${m_cost:.2f} ({len(m_results)} attempts)")

    return "\n".join(lines)


# ─── 2. Logistic Regression Router ────────────────────────────────────────

def logistic_router(results: list[dict], budgets: list[int]) -> str:
    """Train a logistic regression router using sklearn if available."""
    lines = [
        "",
        "=" * 90,
        "Post-Hoc Logistic Regression Router",
        "=" * 90,
    ]

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        lines.append("\n  sklearn not available. Install with: pip install scikit-learn")
        lines.append("  Falling back to feature importance analysis only.\n")
        return _feature_importance_fallback(results, budgets, lines)

    grid_results = [r for r in results if r["method"] not in CEILING_METHODS]
    methods = sorted(set(r["method"] for r in grid_results))
    query_types = sorted(set(r["query_type"] for r in grid_results))

    for qt in query_types:
        qt_results = [r for r in grid_results if r["query_type"] == qt]
        lines.append(f"\n  [{qt.upper()} queries]")

        for b in budgets:
            b_results = [r for r in qt_results if r["budget"] == b]

            # Group by task_id, find which method(s) passed
            by_task = defaultdict(dict)
            for r in b_results:
                by_task[r["task_id"]][r["method"]] = r

            # Build training data: for each task, features → best method
            X, y, task_ids = [], [], []
            for tid, method_results in by_task.items():
                # Find methods that passed
                passing = [m for m, r in method_results.items() if r.get("tests_passed")]
                if not passing:
                    continue  # Skip tasks where nothing passed

                # Use first method alphabetically as target (deterministic)
                best = sorted(passing)[0]

                # Features from the first available result (task-level, same for all methods)
                sample = next(iter(method_results.values()))
                features = _extract_features(sample)
                if features is None:
                    continue

                X.append(features)
                y.append(best)
                task_ids.append(tid)

            if len(set(y)) < 2:
                lines.append(f"    B={b}: Too few classes ({len(set(y))}) for logistic regression")
                continue

            X = np.array(X, dtype=float)
            y_arr = np.array(y)

            # Handle NaN/inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # LOO-CV
            correct = 0
            oracle_correct = 0
            best_single_correct = 0

            # Find best single method
            method_wins = defaultdict(int)
            for label in y_arr:
                method_wins[label] += 1
            best_single = max(method_wins, key=method_wins.get)

            for i in range(len(X)):
                X_train = np.delete(X, i, axis=0)
                y_train = np.delete(y_arr, i, axis=0)
                X_test = X[i:i+1]
                y_test = y_arr[i]

                if len(set(y_train)) < 2:
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
                clf.fit(X_train_s, y_train)
                pred = clf.predict(X_test_s)[0]

                if pred == y_test:
                    correct += 1
                oracle_correct += 1  # Oracle always picks correctly
                if best_single == y_test:
                    best_single_correct += 1

            n = len(X)
            if n > 0:
                router_acc = correct / n
                oracle_acc = oracle_correct / n
                best_single_acc = best_single_correct / n
                gap_closed = ((router_acc - best_single_acc)
                              / (oracle_acc - best_single_acc) * 100
                              if oracle_acc > best_single_acc else 0)

                lines.append(f"    B={b}: n={n} solvable tasks")
                lines.append(f"      Best single: {best_single} ({best_single_acc:.1%})")
                lines.append(f"      Router LOO:  {router_acc:.1%}")
                lines.append(f"      Oracle:      {oracle_acc:.1%}")
                lines.append(f"      Gap closed:  {gap_closed:.1f}%")

                # Fit on full data to get feature importances
                scaler = StandardScaler()
                X_s = scaler.fit_transform(X)
                clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
                clf.fit(X_s, y_arr)

                feature_names = _feature_names()
                lines.append(f"      Top features (by max |coef| across classes):")
                max_coefs = np.max(np.abs(clf.coef_), axis=0)
                top_idx = np.argsort(max_coefs)[::-1][:5]
                for idx in top_idx:
                    lines.append(f"        {feature_names[idx]:<35} |coef|={max_coefs[idx]:.3f}")

    return "\n".join(lines)


def _feature_names():
    return [
        "entity_count_mapped",
        "query_identifier_density",
        "budget_log",
        "retrieval_softmax_entropy",
        "retrieval_effective_candidates",
        "retrieval_top1_score",
        "retrieval_budget_utilization",
        "retrieval_file_concentration",
        "mutation_symbol_lines_log",
        "mutation_file_symbols",
        "graph_node_count_log",
    ]


def _extract_features(r: dict) -> list[float] | None:
    """Extract feature vector from a result dict."""
    try:
        return [
            r.get("entity_count_mapped", 0) or 0,
            r.get("query_identifier_density", 0) or 0,
            math.log2(max(r.get("budget", 2000), 1)),
            r.get("retrieval_softmax_entropy", 0) or 0,
            r.get("retrieval_effective_candidates", 0) or 0,
            r.get("retrieval_top1_score", 0) or 0,
            r.get("retrieval_budget_utilization", 0) or 0,
            r.get("retrieval_file_concentration", 0) or 0,
            math.log2(max(r.get("mutation_symbol_lines", 1) or 1, 1)),
            r.get("mutation_file_symbols", 0) or 0,
            math.log2(max(r.get("graph_node_count", 1) or 1, 1)),
        ]
    except (TypeError, ValueError):
        return None


def _feature_importance_fallback(results, budgets, lines):
    """When sklearn is unavailable, compute simple win-rate statistics."""
    grid_results = [r for r in results if r["method"] not in CEILING_METHODS]
    methods = sorted(set(r["method"] for r in grid_results))
    query_types = sorted(set(r["query_type"] for r in grid_results))

    for qt in query_types:
        qt_results = [r for r in grid_results if r["query_type"] == qt]
        lines.append(f"\n  [{qt.upper()} queries]")
        lines.append(f"  Method win rates (tasks where method is sole solver):")

        for b in budgets:
            b_results = [r for r in qt_results if r["budget"] == b]
            by_task = defaultdict(dict)
            for r in b_results:
                by_task[r["task_id"]][r["method"]] = r

            sole_wins = defaultdict(int)
            any_wins = defaultdict(int)
            for tid, mr in by_task.items():
                passing = [m for m, r in mr.items() if r.get("tests_passed")]
                for m in passing:
                    any_wins[m] += 1
                if len(passing) == 1:
                    sole_wins[passing[0]] += 1

            lines.append(f"    B={b}:")
            for m in methods:
                lines.append(f"      {m:<20} sole={sole_wins.get(m, 0):>3}  "
                              f"any={any_wins.get(m, 0):>3}")

    return "\n".join(lines)


# ─── 3. Retrieval-Outcome Decoupling ──────────────────────────────────────

def retrieval_outcome_decoupling(results: list[dict], budgets: list[int]) -> str:
    lines = [
        "",
        "=" * 90,
        "Retrieval-Outcome Decoupling Analysis",
        "  (Does finding the right file/symbol guarantee a correct fix?)",
        "=" * 90,
    ]

    grid_results = [r for r in results if r["method"] not in CEILING_METHODS]
    methods = sorted(set(r["method"] for r in grid_results))
    query_types = sorted(set(r["query_type"] for r in grid_results))

    for qt in query_types:
        qt_results = [r for r in grid_results if r["query_type"] == qt]
        lines.append(f"\n  [{qt.upper()} queries]")

        # File hit → pass conversion rate
        lines.append(f"\n  --- File Hit → Pass Conversion Rate ---")
        lines.append(f"  (Of attempts that found the target file, what % actually passed?)")
        lines.append(f"  {'Method':<22} " + "  ".join(f"B={b:>7}" for b in budgets))
        lines.append(f"  {'-' * (22 + 12 * len(budgets))}")

        for m in methods:
            vals = []
            for b in budgets:
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                file_hits = [r for r in runs if r.get("target_file_hit")]
                if not file_hits:
                    vals.append("---")
                    continue
                passes = sum(1 for r in file_hits if r.get("tests_passed"))
                rate = passes / len(file_hits)
                vals.append(f"{rate:.0%}")
            lines.append(f"  {m:<22} " + "  ".join(f"{v:>9}" for v in vals))

        # Symbol hit → pass conversion rate
        lines.append(f"\n  --- Symbol Hit → Pass Conversion Rate ---")
        lines.append(f"  (Of attempts that found the target symbol, what % actually passed?)")
        lines.append(f"  {'Method':<22} " + "  ".join(f"B={b:>7}" for b in budgets))
        lines.append(f"  {'-' * (22 + 12 * len(budgets))}")

        for m in methods:
            vals = []
            for b in budgets:
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                sym_hits = [r for r in runs if r.get("target_symbol_hit")]
                if not sym_hits:
                    vals.append("---")
                    continue
                passes = sum(1 for r in sym_hits if r.get("tests_passed"))
                rate = passes / len(sym_hits)
                vals.append(f"{rate:.0%}")
            lines.append(f"  {m:<22} " + "  ".join(f"{v:>9}" for v in vals))

        # Pass WITHOUT file hit (hallucination success)
        lines.append(f"\n  --- Passes WITHOUT Target File in Context ---")
        lines.append(f"  (LLM solved without seeing the right file — memorization/hallucination)")
        lines.append(f"  {'Method':<22} " + "  ".join(f"B={b:>7}" for b in budgets))
        lines.append(f"  {'-' * (22 + 12 * len(budgets))}")

        for m in methods:
            vals = []
            for b in budgets:
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                no_file_pass = [r for r in runs
                                if r.get("tests_passed") and not r.get("target_file_hit")]
                total_pass = sum(1 for r in runs if r.get("tests_passed"))
                if total_pass == 0:
                    vals.append("---")
                    continue
                rate = len(no_file_pass) / total_pass if total_pass > 0 else 0
                vals.append(f"{len(no_file_pass)}/{total_pass}")
            lines.append(f"  {m:<22} " + "  ".join(f"{v:>9}" for v in vals))

    return "\n".join(lines)


# ─── 4. Context Redundancy / Efficiency ───────────────────────────────────

def context_redundancy(results: list[dict], budgets: list[int]) -> str:
    lines = [
        "",
        "=" * 90,
        "Context Efficiency Analysis",
        "  (How efficiently does each method use its token budget?)",
        "=" * 90,
    ]

    grid_results = [r for r in results if r["method"] not in CEILING_METHODS]
    methods = sorted(set(r["method"] for r in grid_results))
    query_types = sorted(set(r["query_type"] for r in grid_results))

    for qt in query_types:
        qt_results = [r for r in grid_results if r["query_type"] == qt]

        # Symbols per 1000 tokens (symbol density)
        lines.append(f"\n  --- Symbols per 1000 Tokens [{qt.upper()} queries] ---")
        lines.append(f"  {'Method':<22} " + "  ".join(f"B={b:>7}" for b in budgets))
        lines.append(f"  {'-' * (22 + 12 * len(budgets))}")

        for m in methods:
            vals = []
            for b in budgets:
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                if not runs:
                    vals.append("---")
                    continue
                densities = []
                for r in runs:
                    tokens = r.get("tokens_used", 0) or 0
                    symbols = r.get("symbols_selected", 0) or 0
                    if tokens > 0:
                        densities.append(symbols / tokens * 1000)
                if densities:
                    vals.append(f"{sum(densities)/len(densities):.1f}")
                else:
                    vals.append("---")
            lines.append(f"  {m:<22} " + "  ".join(f"{v:>9}" for v in vals))

        # Files per 1000 tokens (file diversity)
        lines.append(f"\n  --- Files per 1000 Tokens [{qt.upper()} queries] ---")
        lines.append(f"  {'Method':<22} " + "  ".join(f"B={b:>7}" for b in budgets))
        lines.append(f"  {'-' * (22 + 12 * len(budgets))}")

        for m in methods:
            vals = []
            for b in budgets:
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                if not runs:
                    vals.append("---")
                    continue
                densities = []
                for r in runs:
                    tokens = r.get("tokens_used", 0) or 0
                    files = r.get("files_included", 0) or 0
                    if tokens > 0:
                        densities.append(files / tokens * 1000)
                if densities:
                    vals.append(f"{sum(densities)/len(densities):.1f}")
                else:
                    vals.append("---")
            lines.append(f"  {m:<22} " + "  ".join(f"{v:>9}" for v in vals))

        # Tokens per symbol (granularity)
        lines.append(f"\n  --- Tokens per Symbol (context granularity) [{qt.upper()} queries] ---")
        lines.append(f"  {'Method':<22} " + "  ".join(f"B={b:>7}" for b in budgets))
        lines.append(f"  {'-' * (22 + 12 * len(budgets))}")

        for m in methods:
            vals = []
            for b in budgets:
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                if not runs:
                    vals.append("---")
                    continue
                ratios = []
                for r in runs:
                    tokens = r.get("tokens_used", 0) or 0
                    symbols = r.get("symbols_selected", 0) or 0
                    if symbols > 0:
                        ratios.append(tokens / symbols)
                if ratios:
                    vals.append(f"{sum(ratios)/len(ratios):.1f}")
                else:
                    vals.append("---")
            lines.append(f"  {m:<22} " + "  ".join(f"{v:>9}" for v in vals))

        # Retrieval confidence (entropy-based)
        lines.append(f"\n  --- Retrieval Confidence (lower entropy = more confident) [{qt.upper()}] ---")
        lines.append(f"  {'Method':<22} " + "  ".join(f"B={b:>7}" for b in budgets))
        lines.append(f"  {'-' * (22 + 12 * len(budgets))}")

        for m in methods:
            vals = []
            for b in budgets:
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                if not runs:
                    vals.append("---")
                    continue
                entropies = [r.get("retrieval_softmax_entropy", 0) or 0 for r in runs]
                if entropies:
                    vals.append(f"{sum(entropies)/len(entropies):.2f}")
                else:
                    vals.append("---")
            lines.append(f"  {m:<22} " + "  ".join(f"{v:>9}" for v in vals))

    return "\n".join(lines)


# ─── 5. k-Hop / Seed-to-Mutation Analysis ─────────────────────────────────

def khop_analysis(results: list[dict], budgets: list[int]) -> str:
    lines = [
        "",
        "=" * 90,
        "Seed-to-Mutation Distance Analysis (k-hop)",
        "  (How far is the mutation from the closest seed symbol?)",
        "=" * 90,
    ]

    grid_results = [r for r in results if r["method"] not in CEILING_METHODS]
    methods = sorted(set(r["method"] for r in grid_results))
    query_types = sorted(set(r["query_type"] for r in grid_results))

    for qt in query_types:
        qt_results = [r for r in grid_results if r["query_type"] == qt]

        # Distribution of min_hops
        lines.append(f"\n  --- min_hops Distribution [{qt.upper()} queries] ---")
        lines.append(f"  (BCA methods only; -1 = no path found)")

        bca_methods = [m for m in methods if m.startswith("bca")]
        for m in bca_methods:
            m_results = [r for r in qt_results if r["method"] == m]
            hops = [r.get("min_hops_seed_to_mutation", -1) for r in m_results]
            hop_dist = defaultdict(int)
            for h in hops:
                if h == -1:
                    hop_dist["unreachable"] += 1
                elif h == 0:
                    hop_dist["0 (exact)"] += 1
                elif h <= 2:
                    hop_dist["1-2 (near)"] += 1
                elif h <= 5:
                    hop_dist["3-5 (mid)"] += 1
                else:
                    hop_dist["6+ (far)"] += 1
            total = len(hops)
            lines.append(f"    {m}:")
            for bucket in ["0 (exact)", "1-2 (near)", "3-5 (mid)", "6+ (far)", "unreachable"]:
                count = hop_dist.get(bucket, 0)
                pct = count / total * 100 if total > 0 else 0
                lines.append(f"      {bucket:<20}: {count:>4} ({pct:>5.1f}%)")

        # Pass@1 by min_hops (BCA only, most interesting)
        lines.append(f"\n  --- Pass@1 by min_hops (BCA-SMART, d=3) [{qt.upper()} queries] ---")
        lines.append(f"  {'Hop Bucket':<22} " + "  ".join(f"B={b:>7}" for b in budgets)
                      + "       N")
        lines.append(f"  {'-' * (22 + 12 * len(budgets) + 8)}")

        bca_results = [r for r in qt_results if r["method"] == "bca"]

        def _hop_bucket(h):
            if h == -1:
                return "unreachable"
            elif h == 0:
                return "0 (exact)"
            elif h <= 2:
                return "1-2 (near)"
            elif h <= 5:
                return "3-5 (mid)"
            else:
                return "6+ (far)"

        for bucket_name in ["0 (exact)", "1-2 (near)", "3-5 (mid)", "6+ (far)", "unreachable"]:
            vals = []
            bucket_n = 0
            for b in budgets:
                runs = [r for r in bca_results if r["budget"] == b
                        and _hop_bucket(r.get("min_hops_seed_to_mutation", -1)) == bucket_name]
                if not runs:
                    vals.append("---")
                    continue
                bucket_n = len(runs)
                rate = sum(1 for r in runs if r.get("tests_passed")) / len(runs)
                vals.append(f"{rate:.2f}")
            lines.append(f"  {bucket_name:<22} " + "  ".join(f"{v:>9}" for v in vals)
                          + f"    {bucket_n:>4}")

        # Closure contribution analysis
        lines.append(f"\n  --- Closure Budget Consumption [{qt.upper()} queries] ---")
        lines.append(f"  {'Method':<22} {'Mean Syms':>10} {'Mean Toks':>10} "
                      f"{'% Budget':>10} {'Frontier':>10}")
        lines.append(f"  {'-' * 65}")

        for m in bca_methods:
            m_results = [r for r in qt_results if r["method"] == m]
            if not m_results:
                continue
            cls_syms = [r.get("bca_closure_added_symbols", 0) or 0 for r in m_results]
            cls_toks = [r.get("bca_closure_added_tokens", 0) or 0 for r in m_results]
            budgets_used = [r.get("tokens_used", 1) or 1 for r in m_results]
            frontier = [r.get("bca_frontier_visited", 0) or 0 for r in m_results]

            pct_budget = [t / b * 100 if b > 0 else 0
                          for t, b in zip(cls_toks, budgets_used)]

            lines.append(
                f"  {m:<22} "
                f"{sum(cls_syms)/len(cls_syms):>10.1f} "
                f"{sum(cls_toks)/len(cls_toks):>10.1f} "
                f"{sum(pct_budget)/len(pct_budget):>9.1f}% "
                f"{sum(frontier)/len(frontier):>10.0f}"
            )

    return "\n".join(lines)


# ─── 6. Method Uniqueness Analysis ────────────────────────────────────────

def method_uniqueness(results: list[dict], budgets: list[int]) -> str:
    """Which tasks can ONLY be solved by a specific method?"""
    lines = [
        "",
        "=" * 90,
        "Method Uniqueness: Sole-Solver Analysis",
        "  (Tasks solvable by exactly one method — the router's value proposition)",
        "=" * 90,
    ]

    grid_results = [r for r in results if r["method"] not in CEILING_METHODS]
    methods = sorted(set(r["method"] for r in grid_results))
    query_types = sorted(set(r["query_type"] for r in grid_results))

    for qt in query_types:
        qt_results = [r for r in grid_results if r["query_type"] == qt]

        for b in budgets:
            b_results = [r for r in qt_results if r["budget"] == b]
            by_task = defaultdict(dict)
            for r in b_results:
                by_task[r["task_id"]][r["method"]] = r

            sole_solver = defaultdict(list)  # method → [task_ids]
            multi_solver = 0
            unsolvable = 0
            total = len(by_task)

            for tid, mr in by_task.items():
                passing = [m for m, r in mr.items() if r.get("tests_passed")]
                if len(passing) == 0:
                    unsolvable += 1
                elif len(passing) == 1:
                    sole_solver[passing[0]].append(tid)
                else:
                    multi_solver += 1

            solvable = total - unsolvable
            lines.append(f"\n  [{qt.upper()}, B={b}]  "
                          f"Solvable: {solvable}/{total}  "
                          f"Multi-solver: {multi_solver}  "
                          f"Sole-solver: {sum(len(v) for v in sole_solver.values())}")

            for m in methods:
                tasks = sole_solver.get(m, [])
                if tasks:
                    lines.append(f"    {m:<22}: {len(tasks)} unique solves")

    return "\n".join(lines)


# ─── 7. Per-Repo Method Rankings ──────────────────────────────────────────

def per_repo_rankings(results: list[dict], budgets: list[int]) -> str:
    lines = [
        "",
        "=" * 90,
        "Per-Repository Method Rankings",
        "  (Does the best method differ between repos?)",
        "=" * 90,
    ]

    grid_results = [r for r in results if r["method"] not in CEILING_METHODS]
    methods = sorted(set(r["method"] for r in grid_results))
    repos = sorted(set(r.get("repo_name", "unknown") for r in grid_results))
    query_types = sorted(set(r["query_type"] for r in grid_results))

    for qt in query_types:
        lines.append(f"\n  [{qt.upper()} queries]")
        qt_results = [r for r in grid_results if r["query_type"] == qt]

        for b in budgets:
            lines.append(f"\n  B={b}:")
            lines.append(f"    {'Method':<22}" + "".join(f"{repo:>15}" for repo in repos)
                          + f"{'Overall':>12}")
            lines.append(f"    {'-' * (22 + 15 * len(repos) + 12)}")

            for m in methods:
                vals = []
                for repo in repos:
                    runs = [r for r in qt_results if r["method"] == m
                            and r["budget"] == b and r.get("repo_name") == repo]
                    if runs:
                        rate = sum(1 for r in runs if r.get("tests_passed")) / len(runs)
                        vals.append(f"{rate:.2f}")
                    else:
                        vals.append("---")

                # Overall
                runs = [r for r in qt_results if r["method"] == m and r["budget"] == b]
                overall = sum(1 for r in runs if r.get("tests_passed")) / len(runs) if runs else 0

                lines.append(f"    {m:<22}" + "".join(f"{v:>15}" for v in vals)
                              + f"{overall:>12.2f}")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Post-hoc analyses on benchmark results")
    parser.add_argument("--run-dir", required=True, help="Run output directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results = load_results(run_dir)
    budgets = sorted(set(r["budget"] for r in results))

    print(f"Loaded {len(results)} results, budgets={budgets}")

    # 1. Cost analysis
    print("\n--- Computing cost analysis ---")
    cost_text = cost_analysis(results, budgets)
    (run_dir / "cost_analysis.txt").write_text(cost_text)
    print(cost_text)

    # 2. Logistic regression router
    print("\n--- Training logistic regression router ---")
    router_text = logistic_router(results, budgets)
    (run_dir / "router_logistic.txt").write_text(router_text)
    print(router_text)

    # 3. Retrieval-outcome decoupling
    print("\n--- Computing retrieval-outcome decoupling ---")
    decoup_text = retrieval_outcome_decoupling(results, budgets)
    (run_dir / "retrieval_outcome.txt").write_text(decoup_text)
    print(decoup_text)

    # 4. Context redundancy
    print("\n--- Computing context efficiency ---")
    redundancy_text = context_redundancy(results, budgets)
    (run_dir / "context_redundancy.txt").write_text(redundancy_text)
    print(redundancy_text)

    # 5. k-hop analysis
    print("\n--- Computing k-hop analysis ---")
    khop_text = khop_analysis(results, budgets)
    (run_dir / "khop_coverage.txt").write_text(khop_text)
    print(khop_text)

    # 6. Method uniqueness
    print("\n--- Computing method uniqueness ---")
    unique_text = method_uniqueness(results, budgets)
    (run_dir / "method_uniqueness.txt").write_text(unique_text)
    print(unique_text)

    # 7. Per-repo rankings
    print("\n--- Computing per-repo rankings ---")
    repo_text = per_repo_rankings(results, budgets)
    (run_dir / "per_repo_rankings.txt").write_text(repo_text)
    print(repo_text)

    # Combined summary
    combined = "\n\n".join([
        "POST-HOC ANALYSIS RESULTS",
        f"Run: {run_dir}",
        f"Results: {len(results)}",
        cost_text,
        router_text,
        decoup_text,
        redundancy_text,
        khop_text,
        unique_text,
        repo_text,
    ])
    (run_dir / "posthoc_summary.txt").write_text(combined)

    print(f"\n\nDone. New files in {run_dir}/:")
    print("  cost_analysis.txt        - Token usage & cost per solved task")
    print("  router_logistic.txt      - Logistic regression router results")
    print("  retrieval_outcome.txt    - Retrieval-outcome decoupling")
    print("  context_redundancy.txt   - Context efficiency metrics")
    print("  khop_coverage.txt        - Seed-to-mutation distance analysis")
    print("  method_uniqueness.txt    - Sole-solver analysis")
    print("  per_repo_rankings.txt    - Per-repository method rankings")
    print("  posthoc_summary.txt      - Combined summary")


if __name__ == "__main__":
    main()
