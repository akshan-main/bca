#!/usr/bin/env python3
"""Router A/B analysis: pre-retrieval vs dry-run routing.

Router A ("Choose-first"): Uses only query and repo features available
    BEFORE running any retrieval method. No retrieval metrics, no mutation
    features (deployment leakage). Candidates: all 10 non-ceiling methods
    including no_retrieval.

Router B ("Dry-run then choose"): Runs all retrieval methods locally
    (no LLM call), computes confidence metrics per method, then picks
    the best. Candidates: 9 retrieval methods (no_retrieval excluded).
    Features: Router A features + per-method {top1_score, budget_util, entropy}.

Both use LOO-CV logistic regression. Evaluation: router picks one method,
success = that method is in the task's passing set. Compared against
majority-vote baseline, random baseline, and oracle upper bound.

Feature leakage rules (from ChatGPT Router A/B framework):
  NEVER use as router features:
    - target_file_hit, target_symbol_hit (requires knowing mutation)
    - min_hops_*, median_hops_* (requires knowing mutation)
    - mutation_symbol_lines, mutation_file_symbols (router doesn't know mutation site)
    - tests_passed, failure_mode, patch, test_output (post-LLM)
  These are for EXPLANATORY analysis only, not routing.

Usage:
    python -m paper.experiments.router_ab --run-dir paper/results/run3
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

CEILING_METHODS = {"target_file"}

# Methods that do retrieval (Router B candidates)
RETRIEVAL_METHODS = sorted([
    "bca", "bca_d1", "bca_d5", "bca_no_closure", "bca_no_scoring",
    "bm25", "embedding", "keyword_map", "vector",
])

# All non-ceiling methods (Router A candidates)
ALL_METHODS = sorted(["no_retrieval"] + RETRIEVAL_METHODS)


def load_results(run_dir: Path) -> list[dict]:
    with open(run_dir / "results.json") as f:
        return json.load(f)


def _safe_log2(x):
    return math.log2(max(x, 1)) if x and x > 0 else 0.0


def _safe_val(r, key, default=0.0):
    v = r.get(key, default)
    return float(v) if v is not None else default


# ─── Feature Extraction ──────────────────────────────────────────────

ROUTER_A_FEATURES = [
    "entity_count_mapped",
    "query_identifier_density",
    "graph_node_count_log",
]


def extract_router_a_features(sample: dict) -> list[float]:
    """Pre-retrieval features only. Method-independent, no leakage."""
    return [
        _safe_val(sample, "entity_count_mapped"),
        _safe_val(sample, "query_identifier_density"),
        _safe_log2(_safe_val(sample, "graph_node_count", 1)),
    ]


def router_b_feature_names():
    names = list(ROUTER_A_FEATURES)
    for m in RETRIEVAL_METHODS:
        for metric in ["top1_score", "budget_util", "entropy"]:
            names.append(f"{m}_{metric}")
    return names


def extract_router_b_features(task_methods: dict[str, dict]) -> list[float]:
    """Router A features + per-method retrieval confidence.

    For each retrieval method, includes that method's top1_score,
    budget_utilization, and softmax_entropy. If a method's result
    is missing for this task, features default to 0.
    """
    sample = next(iter(task_methods.values()))
    features = extract_router_a_features(sample)

    for method in RETRIEVAL_METHODS:
        r = task_methods.get(method, {})
        features.extend([
            _safe_val(r, "retrieval_top1_score"),
            _safe_val(r, "retrieval_budget_utilization"),
            _safe_val(r, "retrieval_softmax_entropy"),
        ])

    return features


# ─── LOO-CV Router ───────────────────────────────────────────────────

def loo_cv_router(X_raw, passing_sets, candidate_methods, strategy="smart"):
    """LOO-CV logistic regression router.

    For each held-out task:
      - Train on remaining tasks
      - Predict a method
      - Success if predicted method is in the task's passing set

    Label strategies:
      "safest": label = passing method with highest overall pass rate.
                Converges to majority vote (baseline check).
      "smart":  label = best_single if it passes, else rarest passer.
                Teaches model: use default, deviate only when needed.

    Returns dict with accuracies, gap closed, feature importances.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = len(X_raw)
    X = np.array(X_raw, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    cand_set = set(candidate_methods)

    # Method win rates (how often each method passes across all tasks)
    method_wins = defaultdict(int)
    for ps in passing_sets:
        for m in ps:
            if m in cand_set:
                method_wins[m] += 1

    # Best single method (majority vote)
    best_method = max(candidate_methods,
                      key=lambda m: sum(1 for ps in passing_sets if m in ps))

    # Assign training labels based on strategy
    y = []
    for ps in passing_sets:
        # Sort candidates for deterministic tie-breaking (set iteration
        # order depends on PYTHONHASHSEED; sorting removes that source
        # of non-reproducibility).
        candidates = sorted(m for m in ps if m in cand_set)
        if not candidates:
            y.append(sorted(ps)[0])
        elif strategy == "safest":
            # Always pick the method with highest overall pass rate
            y.append(max(candidates, key=lambda m: method_wins.get(m, 0)))
        elif strategy == "smart":
            # Use best_single if it passes, else pick rarest passer
            if best_method in candidates:
                y.append(best_method)
            else:
                y.append(min(candidates, key=lambda m: method_wins.get(m, 0)))
        else:
            y.append(min(candidates, key=lambda m: method_wins.get(m, 0)))
    y = np.array(y)

    # Best single method baseline (already computed above for labels)
    best_single_correct = sum(1 for ps in passing_sets if best_method in ps)
    best_single_acc = best_single_correct / n

    # Random baseline: E[success] = mean(|passing ∩ candidates| / |candidates|)
    random_accs = []
    for ps in passing_sets:
        overlap = len(ps & cand_set)
        random_accs.append(overlap / len(candidate_methods))
    random_acc = sum(random_accs) / len(random_accs)

    if len(set(y)) < 2:
        # Degenerate: all labels same → router = majority vote
        pred_method = y[0]
        router_correct = sum(1 for ps in passing_sets if pred_method in ps)
        return {
            "router_acc": router_correct / n,
            "best_single_acc": best_single_acc,
            "best_single_method": best_method,
            "random_acc": random_acc,
            "oracle_acc": 1.0,
            "gap_closed": 0.0,
            "n": n,
        }

    # LOO-CV
    router_correct = 0
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i + 1]

        if len(set(y_train)) < 2:
            from collections import Counter
            pred = Counter(y_train).most_common(1)[0][0]
        else:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            clf.fit(X_train_s, y_train)
            pred = clf.predict(X_test_s)[0]

        if pred in passing_sets[i]:
            router_correct += 1

    router_acc = router_correct / n
    oracle_acc = 1.0
    gap = oracle_acc - best_single_acc
    gap_closed = (router_acc - best_single_acc) / gap * 100 if gap > 0 else 0.0

    # Feature importances from full-data fit
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_s, y)
    if clf.coef_.ndim > 1:
        fi = np.max(np.abs(clf.coef_), axis=0)
    else:
        fi = np.abs(clf.coef_[0])

    return {
        "router_acc": router_acc,
        "best_single_acc": best_single_acc,
        "best_single_method": best_method,
        "random_acc": random_acc,
        "oracle_acc": oracle_acc,
        "gap_closed": gap_closed,
        "n": n,
        "feature_importances": fi,
        "classes": list(clf.classes_),
    }


# ─── Router A Analysis ───────────────────────────────────────────────

def _build_task_data(results, budgets, qt, candidate_set, extract_fn):
    """Build (X, passing_sets) for a given query type and budget."""
    grid_results = [r for r in results if r["method"] not in CEILING_METHODS]
    qt_results = [r for r in grid_results if r["query_type"] == qt]
    slices = {}

    for b in budgets:
        b_results = [r for r in qt_results if r["budget"] == b]
        by_task = defaultdict(dict)
        for r in b_results:
            if r["method"] in candidate_set:
                by_task[r["task_id"]][r["method"]] = r

        X, passing_sets = [], []
        for tid, mr in sorted(by_task.items()):
            passing = set(m for m, r in mr.items()
                          if r.get("tests_passed") and m in candidate_set)
            if not passing:
                continue
            X.append(extract_fn(mr))
            passing_sets.append(passing)
        slices[b] = (X, passing_sets)
    return slices


def _extract_a_from_task(mr):
    """Extract Router A features from any method's result for this task."""
    return extract_router_a_features(next(iter(mr.values())))


def router_a_analysis(results: list[dict], budgets: list[int]) -> str:
    lines = [
        "",
        "=" * 90,
        "Router A: Choose-First (Pre-Retrieval Features Only)",
        "  Decision based on query + repo features BEFORE running any retrieval.",
        "  Features: entity_count_mapped, query_identifier_density, graph_node_count_log",
        "  Candidates: " + ", ".join(ALL_METHODS),
        "  Evaluation: LOO-CV, success = predicted method is in passing set",
        "  Two strategies: 'smart' (default→deviate) and 'safest' (always safest)",
        "=" * 90,
    ]

    try:
        import numpy as np
    except ImportError:
        lines.append("\n  numpy/sklearn not available. pip install scikit-learn numpy")
        return "\n".join(lines)

    cand_set = set(ALL_METHODS)
    query_types = sorted(set(r["query_type"] for r in results
                             if r["method"] not in CEILING_METHODS))
    for qt in query_types:
        slices = _build_task_data(results, budgets, qt, cand_set, _extract_a_from_task)
        lines.append(f"\n  [{qt.upper()} queries]")

        for b in budgets:
            X, passing_sets = slices[b]
            if len(X) < 5:
                lines.append(f"    B={b}: Too few solvable tasks ({len(X)})")
                continue

            res_smart = loo_cv_router(X, passing_sets, ALL_METHODS, strategy="smart")
            res_safe = loo_cv_router(X, passing_sets, ALL_METHODS, strategy="safest")

            lines.append(f"    B={b}: n={res_smart['n']} solvable tasks")
            lines.append(f"      Majority vote: {res_smart['best_single_method']} "
                         f"({res_smart['best_single_acc']:.1%})")
            lines.append(f"      Router A (smart):  {res_smart['router_acc']:.1%}  "
                         f"gap={res_smart['gap_closed']:+.1f}%")
            lines.append(f"      Router A (safest): {res_safe['router_acc']:.1%}  "
                         f"gap={res_safe['gap_closed']:+.1f}%")
            lines.append(f"      Random:            {res_smart['random_acc']:.1%}")
            lines.append(f"      Oracle:            {res_smart['oracle_acc']:.1%}")

            if "feature_importances" in res_smart:
                fi = res_smart["feature_importances"]
                top_idx = sorted(range(len(fi)), key=lambda i: fi[i], reverse=True)
                lines.append(f"      Top features (smart):")
                for idx in top_idx[:min(5, len(ROUTER_A_FEATURES))]:
                    if idx < len(ROUTER_A_FEATURES):
                        lines.append(f"        {ROUTER_A_FEATURES[idx]:<35} "
                                     f"|coef|={fi[idx]:.3f}")

    return "\n".join(lines)


# ─── Router B Analysis ───────────────────────────────────────────────

def router_b_analysis(results: list[dict], budgets: list[int]) -> str:
    rb_features = router_b_feature_names()

    lines = [
        "",
        "=" * 90,
        "Router B: Dry-Run Then Choose (Retrieval Confidence Features)",
        "  Runs all retrieval methods locally (no LLM call), examines their",
        "  confidence metrics, picks the best method, then calls LLM once.",
        "  Features: Router A + per-method {top1_score, budget_util, entropy}",
        f"  Total features: {len(rb_features)} "
        f"(3 query + {len(RETRIEVAL_METHODS)} methods x 3 metrics)",
        "  Candidates: " + ", ".join(RETRIEVAL_METHODS),
        "=" * 90,
    ]

    try:
        import numpy as np
    except ImportError:
        lines.append("\n  numpy/sklearn not available. pip install scikit-learn numpy")
        return "\n".join(lines)

    cand_set = set(RETRIEVAL_METHODS)
    query_types = sorted(set(r["query_type"] for r in results
                             if r["method"] not in CEILING_METHODS))
    for qt in query_types:
        slices = _build_task_data(results, budgets, qt, cand_set,
                                  extract_router_b_features)
        lines.append(f"\n  [{qt.upper()} queries]")

        for b in budgets:
            X, passing_sets = slices[b]
            if len(X) < 5:
                lines.append(f"    B={b}: Too few solvable tasks ({len(X)})")
                continue

            res_smart = loo_cv_router(X, passing_sets, RETRIEVAL_METHODS, "smart")
            res_safe = loo_cv_router(X, passing_sets, RETRIEVAL_METHODS, "safest")

            lines.append(f"    B={b}: n={res_smart['n']} solvable tasks")
            lines.append(f"      Majority vote: {res_smart['best_single_method']} "
                         f"({res_smart['best_single_acc']:.1%})")
            lines.append(f"      Router B (smart):  {res_smart['router_acc']:.1%}  "
                         f"gap={res_smart['gap_closed']:+.1f}%")
            lines.append(f"      Router B (safest): {res_safe['router_acc']:.1%}  "
                         f"gap={res_safe['gap_closed']:+.1f}%")
            lines.append(f"      Random:            {res_smart['random_acc']:.1%}")
            lines.append(f"      Oracle:            {res_smart['oracle_acc']:.1%}")

            if "feature_importances" in res_smart:
                fi = res_smart["feature_importances"]
                top_idx = sorted(range(len(fi)), key=lambda i: fi[i],
                                 reverse=True)[:7]
                lines.append(f"      Top features (smart, by max |coef|):")
                for idx in top_idx:
                    if idx < len(rb_features):
                        lines.append(f"        {rb_features[idx]:<35} "
                                     f"|coef|={fi[idx]:.3f}")

    return "\n".join(lines)


# ─── Comparison Table ────────────────────────────────────────────────

def comparison_table(results: list[dict], budgets: list[int]) -> str:
    lines = [
        "",
        "=" * 90,
        "Router Comparison: A vs B vs Majority-Vote vs Random vs Oracle",
        f"  Router A candidates: {len(ALL_METHODS)} methods (incl. no_retrieval)",
        f"  Router B candidates: {len(RETRIEVAL_METHODS)} retrieval methods (excl. no_retrieval)",
        "=" * 90,
    ]

    try:
        import numpy as np
    except ImportError:
        lines.append("\n  numpy/sklearn not available.")
        return "\n".join(lines)

    cand_a = set(ALL_METHODS)
    cand_b = set(RETRIEVAL_METHODS)
    query_types = sorted(set(r["query_type"] for r in results
                             if r["method"] not in CEILING_METHODS))

    for qt in query_types:
        slices_a = _build_task_data(results, budgets, qt, cand_a, _extract_a_from_task)
        slices_b = _build_task_data(results, budgets, qt, cand_b,
                                    extract_router_b_features)

        lines.append(f"\n  [{qt.upper()} queries]")
        lines.append(
            f"  {'Budget':<10} {'Maj(A)':>8} {'Rnd(A)':>8} {'RtrA':>8} {'Orc(A)':>8} {'N(A)':>6} "
            f"{'Maj(B)':>8} {'Rnd(B)':>8} {'RtrB':>8} {'Orc(B)':>8} {'N(B)':>6}"
        )
        lines.append(f"  {'-' * 98}")

        for b in budgets:
            X_a, ps_a = slices_a[b]
            X_b, ps_b = slices_b[b]

            ra_acc = mv_a = rnd_a = orc_a = "---"
            rb_acc = mv_b = rnd_b = orc_b = "---"
            na_str = nb_str = "---"

            if len(X_a) >= 5:
                ra = loo_cv_router(X_a, ps_a, ALL_METHODS, "smart")
                ra_acc = f"{ra['router_acc']:.1%}"
                mv_a = f"{ra['best_single_acc']:.1%}"
                rnd_a = f"{ra['random_acc']:.1%}"
                orc_a = f"{ra['oracle_acc']:.1%}"
                na_str = str(ra["n"])

            if len(X_b) >= 5:
                rb = loo_cv_router(X_b, ps_b, RETRIEVAL_METHODS, "smart")
                rb_acc = f"{rb['router_acc']:.1%}"
                mv_b = f"{rb['best_single_acc']:.1%}"
                rnd_b = f"{rb['random_acc']:.1%}"
                orc_b = f"{rb['oracle_acc']:.1%}"
                nb_str = str(rb["n"])

            lines.append(
                f"  B={b:<7} {mv_a:>8} {rnd_a:>8} {ra_acc:>8} {orc_a:>8} {na_str:>6} "
                f"{mv_b:>8} {rnd_b:>8} {rb_acc:>8} {orc_b:>8} {nb_str:>6}"
            )

        lines.append(
            "\n  NOTE: A and B use different candidate sets, so solvable-task pools may differ."
            "\n  Compare RouterA against A baselines (Maj/Rnd/Orc A), and RouterB against B baselines."
        )

    return "\n".join(lines)


# ─── Feature Leakage Audit ───────────────────────────────────────────

def leakage_audit() -> str:
    return """
==========================================================================================
Feature Validity Audit
  (Which features can each router legitimately use in a real deployment?)
==========================================================================================

  Router A (pre-retrieval, choose-first):
    VALID:
      entity_count_extracted      — parsed from query text
      entity_count_mapped         — looked up in graph (cheap, local)
      query_identifier_density    — computed from query text
      graph_node_count            — repo-level constant (known at init)
      budget                      — given by user/system

    INVALID (not available before retrieval):
      retrieval_*                 — requires running a retrieval method
      tokens_used, symbols_selected, files_included — assembly output
      assembly_time_ms            — assembly output
      bca_*                       — BCA-specific assembly output

    INVALID (leakage — requires knowing mutation location):
      mutation_symbol_lines       — size of mutated function
      mutation_file_symbols       — symbols in mutated file
      target_file_hit             — requires knowing target file
      target_symbol_hit           — requires knowing target symbol
      min_hops_seed_to_mutation   — requires knowing mutation
      median_hops_seed_to_mutation
      edit_distance_lines         — post-hoc evaluation metric

  Router B (dry-run retrieval, then choose):
    VALID (all Router A features PLUS per-method):
      retrieval_top1_score        — confidence of top retrieval result
      retrieval_budget_utilization — fraction of budget used
      retrieval_softmax_entropy   — score distribution spread
      retrieval_effective_candidates — number of competitive candidates
      retrieval_top1_top2_gap     — gap between #1 and #2 scores
      tokens_used                 — context size produced
      symbols_selected            — number of symbols selected
      files_included              — number of files in context
      bca_frontier_visited        — BCA traversal breadth
      bca_closure_added_*         — BCA closure overhead

      NOTE: These are per-method. Router B has one value per candidate method.

    STILL INVALID (leakage):
      target_file_hit, target_symbol_hit, mutation_*, min_hops_*,
      edit_distance_*, tests_passed, failure_mode, patch, test_output
"""


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Router A/B analysis")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results = load_results(run_dir)
    budgets = sorted(set(r["budget"] for r in results))

    hash_seed = os.environ.get("PYTHONHASHSEED", "unset")
    print(f"Loaded {len(results)} results, budgets={budgets}")
    print(f"PYTHONHASHSEED={hash_seed}")

    # Feature leakage audit
    audit = leakage_audit()
    print(audit)

    # Router A
    print("\n--- Router A (pre-retrieval) ---")
    ra_text = router_a_analysis(results, budgets)
    print(ra_text)

    # Router B
    print("\n--- Router B (dry-run retrieval) ---")
    rb_text = router_b_analysis(results, budgets)
    print(rb_text)

    # Comparison table
    print("\n--- Comparison ---")
    comp_text = comparison_table(results, budgets)
    print(comp_text)

    # Write output (include hash seed header for reproducibility)
    header = (f"PYTHONHASHSEED={hash_seed}\n"
              f"Loaded {len(results)} results, budgets={budgets}\n")
    full_output = header + "\n\n".join([audit, ra_text, rb_text, comp_text])
    out_path = run_dir / "router_ab.txt"
    out_path.write_text(full_output)

    print(f"\n\nOutput written to: {out_path}")


if __name__ == "__main__":
    main()
