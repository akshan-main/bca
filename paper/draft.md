# CeGraph: Budgeted Context Assembly via Code Knowledge Graphs

## Abstract

Context assembly under a token budget is a decision problem: which code fragments should an LLM
see to fix a bug, given finite prompt space? We present a reproducible benchmark isolating context
selection from code generation, using mutation-based fault injection across two Python repositories
(245 tasks) and three query tiers of decreasing information content: dev-localized (file + line +
operator), dev-report (traceback + test name), and symptom-only (vague description). We evaluate
nine retrieval methods—including BCA (Budgeted Context Assembly), a graph-guided approach using
code knowledge graphs—across four token budgets (2k–10k tokens) for a total of 29,400 attempts,
all using a single pinned model snapshot (gpt-4o-mini-2024-07-18).

Our central finding is a monotonic three-tier gradient: every method's pass@1 drops from exact
(48–86%) through dev-report (2–25%) to vague (0–7%), with no exceptions. This gradient manifests
in failure modes (localization errors → comprehension errors), retrieval-outcome decoupling (file
hit → pass conversion: 72–86% exact, 19–25% dev-report, 2–12% vague), and task solvability
(97% oracle on exact vs 40% on dev-report vs 7% on vague).

No single method dominates: no_retrieval wins on exact (0.86), vector search leads on dev-report
(0.23), and BM25 leads on vague (0.05). BCA catches up at high budget on dev-report, matching
vector at B=10k (0.24 vs 0.23). A lightweight router using retrieval confidence features closes
18.2% of the oracle gap on dev-report without additional LLM calls.

We release the full harness, all 29,400 per-attempt artifacts, and mechanistic analyses (hop
distance, failure taxonomy, edit locality, patch quality) to support future work on conditional
context assembly.

---

## 1. Introduction

Large language models increasingly serve as automated coding assistants, generating patches for
bug reports and feature requests. A critical bottleneck is *context selection*: which code
fragments should the model see? Token budgets force a selection problem—whole-file dumping
wastes budget on irrelevant code, while keyword retrieval may miss structural dependencies.

The community treats this as a single-method retrieval competition ("our RAG beats your RAG"),
evaluated on a single query style. We argue this framing obscures the real structure of the
problem. Context assembly under budget is a *conditional* decision: the right retrieval method
depends on how much the query reveals about the bug's location.

We make three contributions:

1. **A reproducible mutation-based benchmark** with three query tiers (dev-localized, dev-report,
   vague), two repositories, pinned model snapshots, byte-identical revert, and full per-attempt
   artifact logging. The benchmark isolates context selection from code generation by using
   single-shot repair with a fixed model.

2. **Mechanistic analysis** explaining *why* methods succeed or fail: graph hop distance from
   query seeds to mutation site, failure mode taxonomy (localization vs comprehension errors),
   retrieval-outcome decoupling, edit locality, and patch quality metrics.

3. **A router framework** demonstrating that lightweight per-query method selection using
   retrieval confidence features closes 18.2% of the oracle gap on the realistic dev-report
   tier, without additional LLM calls.

---

## 2. Related Work

**Code retrieval for LLMs.** Repository-level code generation benchmarks (RepoBench, CrossCodeEval,
SWE-bench) evaluate end-to-end performance but conflate retrieval quality with generation quality.
Our benchmark isolates context selection by fixing the generation model and varying only the
retrieved context.

**Retrieval-augmented generation.** RAG systems for code (GitHub Copilot, Cursor, Aider) use
keyword search, embedding similarity, or repository maps. These are typically evaluated on
pass@k without mechanistic analysis of *why* retrieval succeeds or fails.

**Knowledge graphs for code.** Static analysis graphs (call graphs, data flow graphs) have been
used for code understanding but rarely evaluated as retrieval mechanisms under token budgets.
BCA (Budgeted Context Assembly) uses a code knowledge graph for weighted BFS expansion from
query-derived seed symbols, with relevance scoring and greedy budget packing.

**Mutation testing.** Mutation testing is a standard technique for evaluating test suite quality.
We repurpose it as a controlled fault injection method: each mutation creates a bug with a known
test oracle, enabling automated evaluation of repair success.

---

## 3. Benchmark Design

### 3.1 Repositories

| Property | pydantic-ai | httpx |
|----------|-------------|-------|
| Stars | 14.8k | 13k+ |
| Source files | 186 | ~100 |
| Lines of code | 53k | ~15k |
| Killed mutations | 199/1454 (14%) | 210/388 (54%) |
| Tasks (after filtering) | 128 | 117 |
| Pinned commit | `69a578a` | `ae1b9f6` |

Total: **245 tasks** across both repositories.

### 3.2 Mutation Types

We use 9 mutation operators: none_check_swap (86 tasks), condition_inversion (64),
boolean_flip (57), comparison_swap (47), value_swap (28), constant_mutation (29),
handcrafted (14), arithmetic_swap (13), and 4 others. Syntax-invalid mutations and
mutations without a failing test oracle are filtered.

### 3.3 Three Query Tiers

Each task generates three query descriptions of decreasing information content:

1. **Dev-localized (exact)**: File name, line number, and operator change.
   Example: *"In _client.py:172, the comparison operator was changed: '>' became '>='"*

2. **Dev-report**: Failing test name, sanitized traceback, and error message. No line numbers,
   no file paths in the query (file paths are stripped from tracebacks).
   Example: *"Test test_retry_logic fails with AssertionError: expected 3 retries but got 0"*

3. **Vague (symptom-only)**: Natural language symptom description with no code identifiers.
   Example: *"The retry mechanism doesn't work correctly when the server returns errors"*

### 3.4 Methods

We evaluate 9 retrieval methods plus a privileged ceiling:

| Method | Description |
|--------|-------------|
| **no_retrieval** | No context; LLM sees only the query |
| **bm25** | BM25 keyword search over code symbols |
| **vector** | TF-IDF cosine similarity search |
| **repo_map** | Function/class signatures (tree-sitter) |
| **bca** (d=3) | Graph-guided BFS, depth 3, with scoring + closure |
| **bca_d1** (d=1) | BCA with depth 1 (shallow) |
| **bca_d5** (d=5) | BCA with depth 5 (deep) |
| **bca_no_closure** | BCA without dependency closure |
| **bca_no_scoring** | BCA without relevance scoring |
| **target_file** | Ceiling: privileged access to the mutated file |

### 3.5 Evaluation Protocol

- Model: `gpt-4o-mini-2024-07-18` (pinned snapshot)
- Single-shot generation (pass@1), temperature=0, seed=42
- 4 token budgets: 2000, 4000, 8000, 10000
- Total: 245 tasks × 10 methods × 4 budgets × 3 query types = **29,400 attempts**
- Success criterion: the task's oracle test passes on the patched code
- Byte-identical git revert between tasks
- `system_fingerprint` logged per response

---

## 4. BCA: Budgeted Context Assembly

BCA assembles context through 8 phases:

1. **Entity extraction**: Parse query for code identifiers (function names, class names, module paths)
2. **Seed finding**: Resolve extracted entities to symbols in the code knowledge graph
3. **BFS expansion**: Weighted breadth-first search from seeds to depth *d*, following call/import edges
4. **Relevance scoring**: Score each visited symbol by query similarity and graph centrality
5. **Dependency closure**: Add import dependencies of selected symbols
6. **Greedy packing**: Select highest-scoring symbols that fit within the token budget
7. **Source loading**: Read actual source code for selected symbols
8. **Dependency ordering**: Sort context in topological order for the LLM

The code knowledge graph is built from Python AST parsing, capturing function definitions,
class definitions, imports, and call relationships. Graph construction takes ~2 seconds for
pydantic-ai (186 files, 53k lines).

---

## 5. Results

### 5.1 Main Results

**Table 1: Pass@1 by method and query tier (B=10000)**

| Method | Exact | Dev-Report | Vague |
|--------|-------|------------|-------|
| no_retrieval | **0.86** | 0.02 | 0.00 |
| bm25 | 0.76 | 0.20 | **0.05** |
| vector | 0.79 | 0.23 | 0.03 |
| repo_map | 0.75 | 0.19 | 0.03 |
| bca (d=3) | 0.69 | 0.23 | 0.02 |
| bca_d1 | 0.69 | **0.24** | 0.01 |
| bca_no_closure | 0.72 | **0.24** | 0.02 |
| Oracle | 0.98 | 0.42 | 0.06 |
| Ceiling (target_file) | 0.92 | 0.48 | 0.30 |

### 5.2 Three-Tier Gradient

The monotonic gradient exact > dev-report > vague holds for every method at every budget, with
no exceptions. Key observations:

1. **Exact trivializes the task**: no_retrieval achieves 86%. The query leaks the bug location,
   making retrieval unnecessary. Retrieval methods add noise that reduces performance.

2. **Dev-report is the meaningful evaluation tier**: Methods score 10–25%. The traceback provides
   enough signal for retrieval but not enough for the LLM to fix without context. no_retrieval
   drops to 2%.

3. **Vague breaks the model**: All methods score 0–7%. The ceiling probe (target_file) reaches
   only 30%, confirming the bottleneck is model comprehension, not retrieval quality.

4. **BCA catches up at high budget**: At B=2000, BCA (0.11) trails vector (0.21) by 10pp on
   dev-report. At B=10000, bca_no_closure (0.24) matches vector (0.23). Graph-guided retrieval
   needs budget to include both seed symbols and their dependency neighborhoods.

### 5.3 Ceiling Analysis

| Query Tier | Best Method | Ceiling (target_file) | Oracle | Gap |
|-----------|------------|----------------------|--------|-----|
| Exact | 0.86 (no_retrieval) | 0.92 | 0.98 | 6pp |
| Dev-report | 0.24 (bca_d1, bca_no_closure) | 0.48 | 0.42 | 18pp |
| Vague | 0.05 (bm25) | 0.30 | 0.06 | 1pp |

The dev-report ceiling (0.48) reveals that even with the *perfect file*, gpt-4o-mini fixes only
48% of tasks from a traceback. This means 52% of tasks exceed the model's single-shot capability
regardless of retrieval quality.

---

## 6. Analysis

### 6.1 Failure Mode Gradient

**Table 2: Failure mode distribution at B=10000 (representative methods)**

| Mode | Exact (vector) | Dev-Report (vector) | Vague (bm25) |
|------|---------------|--------------------|-----------|
| pass | 79% | 23% | 5% |
| patch_apply_fail | 18% | 16% | 20% |
| test_fail | 2% | 54% | 70% |
| syntax_error | 0% | 2% | 4% |

The failure mode shifts with query tier:
- **Exact**: Failures are localization errors (patch_apply_fail) — the LLM knows the fix but
  targets the wrong location.
- **Dev-report**: Blended — both localization errors (16%) and comprehension errors (54%).
- **Vague**: Failures are comprehension errors (test_fail) — the LLM generates a syntactically
  valid patch that doesn't fix the bug.

This gradient demonstrates that failure modes shift continuously with query information content.
Benchmarks using only one query type capture only one failure regime.

### 6.2 Retrieval-Outcome Decoupling

Finding the right file does not guarantee a correct fix. File hit → pass conversion rates
at B=10000:
- Exact: 72–86% (retrieval ≈ repair)
- Dev-report: 19–25% (75–81% of correct retrievals still fail)
- Vague: 2–12% (retrieval barely helps)

On exact queries, no_retrieval solves 211/211 tasks *without any context* — all passes are pure
memorization/reasoning from the description. On dev-report, 0–1 passes occur without the target
file in context, confirming that code context is essential.

### 6.3 Hop Distance Predicts Task Difficulty

For BCA methods on dev-report, the graph distance from the nearest seed symbol to the mutation
site strongly predicts pass@1:
- Hop 0 (direct hit): 35% pass@1 at B=10000 (N=135)
- Hop 1–2 (near): 7% pass@1 (N=89)
- Hop 3–5 (mid): 19% pass@1 (N=21)

The 5× drop from hop-0 to hop 1–2 is the clearest evidence that graph distance from seed to
mutation predicts repair difficulty.

### 6.4 BCA Ablation

| Ablation | Dev-Report Avg | Effect |
|----------|---------------|--------|
| bca (d=3, full) | 0.16 | baseline |
| bca_no_closure | **0.18** | +2pp (closure hurts slightly) |
| bca_d1 (d=1) | **0.17** | shallower = slightly better |
| bca_d5 (d=5) | 0.15 | deeper = worse |
| bca_no_scoring | 0.11 | −5pp (scoring is essential) |

Scoring is the critical BCA component (removing it drops 5pp). Closure is negligible (<1%
budget overhead for d≥3). Depth 1 slightly outperforms depth 3 on dev-report — the default
depth may be too deep, including irrelevant distant symbols.

### 6.5 Edit Locality and Patch Quality

Edit distance from patch to mutation: exact 0.5–11 lines, dev-report 47–143 lines,
vague 50–164 lines. BCA produces the highest context-patch overlap (0.17–0.19 on dev-report)
— the LLM references BCA-provided context more, even when it fails. Passing patches are
smaller (1.2–2.5 lines) than failing patches (1.9–6.4 lines); over-editing correlates with
failure.

---

## 7. Conditional Routing

### 7.1 Router Design

We evaluate two deployment-valid routers:

**Router A (pre-retrieval)**: Uses 3 query-level features (entity count, identifier density,
graph node count) to select among 9 methods including no_retrieval. Zero additional cost.

**Router B (dry-run retrieval)**: Runs all 8 retrieval methods locally, examines their confidence
metrics (top-1 score, entropy, budget utilization), then calls the LLM once with the selected
method's context. 27 features total.

Both use leave-one-out cross-validated logistic regression on solvable tasks only.

### 7.2 Router Results

**Table 3: Router comparison on dev-report (the routing-relevant tier)**

| Budget | Majority Vote | Router A | Router B | Oracle | N(A) | N(B) |
|--------|--------------|----------|----------|--------|------|------|
| B=2000 | 60.5% | 65.1% (+11.8%) | **68.2% (+18.2%)** | 100% | 86 | 85 |
| B=4000 | 58.3% | 58.3% | **61.5% (+7.5%)** | 100% | 96 | 96 |
| B=8000 | 57.5% | 60.4% | 59.4% | 100% | 106 | 106 |
| B=10000 | 58.8% | 61.8% | 60.8% | 100% | 102 | 102 |

N(A) and N(B) differ because Router A includes no_retrieval as a candidate (9 methods),
while Router B excludes it (8 methods), yielding slightly different solvable-task sets.

Dev-report is the only tier where routing adds value. On exact, no_retrieval dominates (88.7%)
and methods are too correlated (230/238 multi-solver). On vague, too few tasks are solvable
(15–20) for the logistic model to learn.

### 7.3 Why Dev-Report Enables Routing

1. **Sufficient method divergence**: 21–29 sole-solver tasks per budget
2. **No dominant method**: vector/bm25/repo_map split wins roughly evenly
3. **Informative retrieval features**: BCA scoring entropy and BM25 top-1 confidence predict
   which method "got it right" before the LLM call
4. **Assembly is cheap**: Running all 8 methods takes ~1 second (27–266ms each), while a single
   LLM call takes 1.5–7 seconds. Router B adds <15% latency overhead.

---

## 8. Discussion

### What We Learn About Context Assembly

Context assembly is not a universal retrieval problem — it is conditional on query information
content. When the query leaks the bug location (exact), retrieval is unnecessary. When the query
provides structural clues (dev-report), retrieval quality matters and method choice matters. When
the query is purely symptomatic (vague), model comprehension is the bottleneck, not retrieval.

This has practical implications: real developer queries span this spectrum. A system that routes
queries to the appropriate retrieval strategy — or bypasses retrieval entirely when the query is
self-sufficient — will outperform any single retrieval method.

### BCA's Position

BCA is not universally superior to BM25 or vector search. Its advantage appears at high budget
on the dev-report tier, where graph-guided expansion from traceback-derived seeds includes
relevant dependency context. At low budget or on exact/vague queries, simpler methods suffice.
The ablation shows that relevance scoring is BCA's essential component, not depth or closure.

### Limitations

- **Python-only**: Two Python repositories. No evidence for generalization to other languages.
- **Mutation proxy**: Single-line mutations approximate real bugs but miss multi-file bugs,
  algorithmic errors, and integration issues.
- **Single model**: gpt-4o-mini results. Stronger models may change the relative rankings.
- **Two repos**: This is a detailed case study, not a general claim. Conference-scale evaluation
  requires 5+ repositories.
- **Exact descriptions are too informative**: no_retrieval's 86% on exact shows this tier
  primarily evaluates the LLM's ability to follow instructions, not retrieval quality.

---

## 9. Conclusion

We present a mutation-based benchmark for evaluating context assembly under token budgets,
with three query tiers that create a controlled information spectrum. Our 29,400-attempt study
reveals that no single retrieval method dominates: the right method depends on query information
content. BCA's graph-guided approach catches up with keyword methods at high budget when
structural clues are available (dev-report tier), while BM25 and vector search perform better
at low budget or when queries contain direct keyword matches.

The three-tier gradient — and its manifestation in failure modes, retrieval-outcome decoupling,
and routing potential — demonstrates that context assembly benchmarks should evaluate across
multiple query tiers. Evaluating on a single query style misses the conditional structure that
determines when each method adds value.

We release the complete benchmark, all per-attempt artifacts, and mechanistic analyses to
support future work on conditional context assembly and routing.

---

## Reproducibility

- Model: `gpt-4o-mini-2024-07-18` (pinned snapshot, not alias)
- Temperature=0, seed=42, top_p=1
- Repositories: pydantic-ai `69a578a`, httpx `ae1b9f6`
- Retry backoff: run3 (exact+vague) [2, 5, 10, 30, 60]s; run4 (dev_report) [5, 8, 15, 30, 60]s
- Byte-identical git revert between tasks
- `system_fingerprint` logged per API response
- Bootstrap CIs: n=10000, seed=42
- Total cost: $22.71 (gpt-4o-mini pricing: $0.15/M input, $0.60/M output)
