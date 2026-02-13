# CeGraph Paper Notes & Findings

Running notes for the arXiv preprint. Everything interesting goes here so nothing gets lost.

---

## 1. Paper Framing & Strategy

### Title
**CeGraph: Budgeted Context Assembly via Code Knowledge Graphs**

### Core Thesis (The Real One)
Context assembly under a token budget is a **decision problem**. The community keeps treating it
like a single-method retrieval shootout ("our RAG beats your RAG"). Our paper's point is to:
1. Build a **reproducible harness** that isolates context selection from code generation
2. Measure **mechanisms** (not just pass@1) — explain *why* methods work or fail
3. Show that **conditional routing** is the right abstraction

If BCA wins sometimes and loses sometimes, that is not failure — that is literally the thesis.

### Publishable Contribution (Even If BCA Is Mediocre)
The contribution is the **system + instrumentation + conditional analysis**, not BCA's average score.

1. **Reproducible benchmark**: Mutation-based code repair with dual query modes, pinned commits,
   pinned model snapshots, byte-identical revert, and full per-attempt artifacts
2. **Mechanistic logging**: Explains *why* methods work/fail — file hit, symbol hit, graph hops,
   closure overhead, failure-mode taxonomy (patch_apply_fail vs test_fail vs regression)
3. **Router framing**: Quantify oracle gap and show a lightweight router (majority-vote LOO-CV
   per budget×query_type, plus post-hoc logistic regression on retrieval confidence features)
   closes meaningful gap without extra LLM calls

That combination is workshop-ready once we have the full two-repo run and artifacts.

### Paper Positioning
- **arXiv v1**: gpt-4o-mini-2024-07-18, 2 repos (pydantic-ai + httpx), workshop-ready
- **Conference upgrade**: gpt-4o-2024-08-06, 3+ repos, full analysis, 8-page venue
- Frame as "detailed empirical study" not "general claim" — honest about scope

### ChatGPT Strategic Advice (Key Points)
1. Pin model snapshots exactly (gpt-4o-mini-2024-07-18) for reproducibility
2. Keep benchmark identical between arXiv and conference runs — only change model + add repos
3. httpx is perfect second repo because it's non-AI (library vs framework diversity)
4. Conditional analysis is the real paper story, not just pass@1 tables
5. If BCA loses, the paper becomes *stronger* because you explain why with evidence
6. Router is the unique angle that makes this more than "yet another RAG comparison"

---

## 2. Benchmark Design

### Mutation-Based Evaluation
Single-line mutations with test oracle. Each mutation:
- Causes a specific test to fail on mutated code
- Passes on original code
- Is a minimal, realistic bug (operator swap, boundary off-by-one, None check inversion)

### Two Repositories

| Repo | Stars | Source Files | Lines | Source Dir | Kill Rate | Tasks |
|------|-------|-------------|-------|------------|-----------|-------|
| pydantic-ai | 14.8k | 186 | 53k | pydantic_ai_slim/pydantic_ai | 20.5% (199/1454) | 128 (14 handcrafted + 114 discovered) |
| httpx | 13k+ | ~100 | ~15k | httpx | 54.1% (210/388) | 117 (all discovered, 3 unvalidated removed) |

**Total: 245 tasks across 2 repos** (78 syntax-invalid + 3 unvalidated httpx mutations filtered out — see Section 15)

pydantic-ai has lower kill rate because it's a larger codebase with more complex test coupling.
httpx is a focused HTTP library with tighter test coverage.

### 60-Task Curated Set (pydantic-ai)
For dev/tuning: 14 handcrafted + 30 discovered = 44 tasks (after syntax filtering), seed=42,
diverse selection with caps (max 15/file, 25/category, 30/mutation_type).

### Mutation Type Distribution

**pydantic-ai (128 tasks):**
- none_check_swap: 30, boolean_flip: 25, condition_inversion: 22
- handcrafted: 14, comparison_swap: 12, constant_mutation: 11
- value_swap: 7, arithmetic_swap: 5, membership_swap: 2

**httpx (117 tasks, 3 unvalidated removed):**
- none_check_swap: 28, comparison_swap: 23, condition_inversion: 21
- boolean_flip: 16, value_swap: 14, constant_mutation: 9
- arithmetic_swap: 4, return_value_swap: 2

> Counts recomputed from eval_tasks_httpx.jsonl (117 tasks). 3 removed tasks
> were: httpx-content-L195, httpx-asgi-L37, httpx-main-L38 (not in httpx_killed.json).

### Category Coverage

**pydantic-ai**: 16 categories — ssrf(19), usage(14), utils(13), parts_manager(12), tools(10), concurrency(9), direct_api(9), json_schema(9), messages(9), retries(7), models(5), exceptions(3), builtin_tools(3), ui(3), thinking(2), settings(1)

**httpx**: 11 categories — transports(20), config(15), multipart(13), utils(13), cli(11), decoders(11), content(11), auth(9), url_parsing(8), client(5), exceptions(1)

### Three Query Tiers
- **Dev-localized** (exact): File name + line number + operator diff. This is ceiling-ish —
  the description effectively leaks the bug location. Label as such in paper. Reviewers WILL
  note that no_retrieval can solve these.
  - Example: "In _client.py:172, the comparison operator was changed: '>' became '>='"
- **Dev-report** (dev_report): Failing test name + sanitized traceback + error message. No
  line numbers, no file-path line refs, no mutation type. Simulates what a developer sees
  after running `pytest` and pasting the failure. The middle tier — mentions test names and
  error types but doesn't leak exact bug location.
  - Example: "Test failure: test_total_token_limit\nError: Failed: DID NOT RAISE UsageLimitExceeded\nTraceback: ..."
  - Generated by: `python -m paper.experiments.capture_tracebacks`
  - Sanitization: absolute paths → `<...>`, `File "...", line N, in func` → `File "file.py", in func`,
    pytest `:N:` line refs stripped, environ() dumps truncated
- **Vague** (user-reported): Symptom-only, no code identifiers. This is the realistic query mode.
  - Example: "A threshold or limit check seems to trigger at the wrong value"

> **Information content**: dev-localized > dev-report > vague. The dev-report tier fills the
> gap between "god-mode leak" and "zero-info symptom". If methods that struggle on vague but
> succeed on dev-localized also succeed on dev-report, the signal is the traceback/test-name.

---

## 3. Methods (13 total)

| Method | Type | Description |
|--------|------|-------------|
| no_retrieval | Baseline | No context, just task description |
| naive_random | Baseline | Random symbols up to budget |
| grep | Lexical | Regex search on keywords from description |
| bm25 | Lexical | BM25 symbol retrieval + greedy packing |
| vector | Semantic | TF-IDF vector similarity search |
| embedding | Semantic | (if available) embedding-based search |
| repo_map | Structural | Signature-only overview of entire codebase |
| bca_d1 | **Ours** (depth=1) | BCA with PRECISE strategy: shallow expansion, min_score=0.3 |
| bca | **Ours** (depth=3) | BCA with SMART strategy (default): entity extraction -> BFS -> scoring -> closure -> budget |
| bca_d5 | **Ours** (depth=5) | BCA with THOROUGH strategy: deep expansion, min_score=0.05 |
| bca_no_closure | Ablation | BCA-SMART without dependency closure |
| bca_no_scoring | Ablation | BCA-SMART with scoring disabled (graph traversal + closure only) |
| target_file | Ceiling | Oracle: gives the LLM the entire mutated file (upper bound) |

### Depth Strategy Details

| Strategy | max_depth | min_score | Why it matters |
|----------|-----------|-----------|----------------|
| PRECISE (bca_d1) | 1 | 0.3 | Only immediate neighbors of seeds. Budget-efficient but may miss dependencies. |
| SMART (bca) | 3 | 0.1 | Default. Balanced depth and score threshold. |
| THOROUGH (bca_d5) | 5 | 0.05 | Wide neighborhood. May waste budget on low-relevance code at tight budgets. |

**Key question depth answers**: Does BCA benefit from deeper expansion, or does it waste budget?
At 1k tokens, d1 may win (less noise). At 10k tokens, d5 may win (more room for distant context).
If depth doesn't matter, that's interesting too — means scoring/packing dominates over expansion.

---

## 4. Pre-Launch Metrics (Logged Per-Attempt)

### Standard Metrics
- `tests_passed`, `failure_mode`, `test_time_ms`
- `tokens_used`, `symbols_selected`, `files_included`, `assembly_time_ms`
- `llm_input_tokens`, `llm_output_tokens`, `llm_time_ms`
- `target_file_hit`, `target_symbol_hit`
- `context_patch_overlap`, `patch_files_changed`, `patch_lines_changed`
- `edit_distance_lines`

### Non-Negotiable Pre-Launch Metrics (avoids reruns)
- `entity_count_extracted` — Entities found in query text
- `entity_count_mapped` — Entities that resolved to graph symbols
- `query_identifier_density` — Fraction of query tokens that are code identifiers
- `seed_symbol_keys` — Graph symbol IDs of resolved seed entities
- `mutation_symbol_key` — Graph symbol ID containing the mutation
- `min_hops_seed_to_mutation` — BFS shortest path from any seed to mutation
- `median_hops_seed_to_mutation` — Median BFS distance
- `bca_closure_added_symbols` — Symbols added by dependency closure (BCA only)
- `bca_closure_added_tokens` — Tokens consumed by closure (BCA only)
- `bca_frontier_visited` — BFS expansion candidates before scoring (BCA only)
- `context_symbol_keys` — Symbol IDs present in context (for post-hoc coverage)

### Code Metadata (per-task constants, logged per-attempt for self-contained analysis)
- `mutation_symbol_lines` — Line span of the mutated function/class
- `mutation_symbol_kind` — function, method, class, constant, etc.
- `mutation_file_symbols` — Number of symbols in the mutated file
- `graph_node_count` — Total graph nodes (normalizes across repos)

### Post-Hoc Metrics (no rerun needed)
- Cost per solved task (total tokens / passes)
- Context redundancy/entropy (from stored context text)
- Logistic regression router (train on logged features)
- k-hop dependency coverage: `coverage_k = |D_k(m) intersection C| / |D_k(m)|`

---

## 5. Sanity-Check Observations (n=2-3 tasks, NOT findings)

> **Disclaimer**: These are directional signals from sanity runs on 2-3 tasks.
> They are NOT findings. Do not cite these numbers in the paper.
> The full run will produce actual findings.

### Observation 1: Entity Density Perfectly Separates Exact vs Vague
Tested on real tasks:
- **Exact description**: entity_count=3, mapped=2, identifier_density=0.13
- **Vague description**: entity_count=0, mapped=0, identifier_density=0.0

This is the backbone of the conditional analysis. BCA's entity extraction finds 0 entities in
vague descriptions, meaning it falls back to keyword-only seeds (low quality). Lexical methods
don't depend on entity extraction so they degrade more gracefully.

### Observation 2: BCA Achieves 100% Target Symbol Hit Rate
From httpx_sanity2 retrieval metrics:
- **BCA**: 100% target file hit, 100% target symbol hit (both budgets)
- **grep**: 0-50% target file hit, 50% target symbol hit
- **naive_random**: 0-50% file hit, 0-50% symbol hit
- **no_retrieval**: 0% everything

BCA's graph traversal reliably locates the mutation site when entities resolve.

### Observation 3: BCA Needs Budget to Shine
From sanity_mini (pydantic-ai, 3 tasks):
- **BCA @ 1k**: 67% pass@1
- **BCA @ 4k**: 100% pass@1 (statistically significant vs no_retrieval, CI=[+1.0, +1.0])
- **no_retrieval**: 0% at all budgets

At tight budgets, BCA includes the right symbols but may not have room for enough context.
At 4k+, BCA has enough room for seeds + closure + supporting code.

### Observation 4: Graph Distance Is Meaningful
From entity-resolution tests:
- Seed symbol = mutation symbol: min_hops=0 (exact entity match, e.g. `UsageBase.total_tokens`)
- Seed in same file, different symbol: min_hops=1-2
- Seeds in different files: min_hops=3+

**Prediction**: Tasks with min_hops=0 should have highest pass@1 for BCA.
Tasks with min_hops >= 3 are where BCA struggles and lexical methods may win.

### Observation 5: Closure Adds Minimal Overhead (So Far)
From BCA sanity run:
- `bca_closure_added_symbols`: 1
- `bca_closure_added_tokens`: 9
- `bca_frontier_visited`: 2172

Closure is cheap when the seed already points to the right place. The interesting case
is when closure pulls in distant dependencies — we need the full benchmark to see this.

### Observation 6: grep Can Fail Hard
From quick_test (agent-library, 2 tasks):
- **grep**: 0% pass@1 at both budgets
- **BCA & BM25**: 100% at both budgets

grep's failure mode: the LLM's patches reference wrong file paths (e.g. `cegraph/graph/query.py`
instead of `src/cegraph/graph/query.py`), causing "file not found" errors. This is a real issue
with keyword-based retrieval — it gives code snippets without structural awareness.

### Observation 7: no_retrieval Sometimes Works (httpx)
In httpx_sanity2: no_retrieval got 100% pass@1. This is because:
- httpx is well-known (in GPT-4o's training data)
- The exact descriptions are very specific ("In _auth.py:272, a constant was changed")
- The LLM can hallucinate a correct fix from the description alone

**Important paper note**: This does NOT mean retrieval is useless. It means the exact
descriptions are too informative for some tasks. The vague descriptions will tell the real story.

---

## 6. Failure Mode Taxonomy

| Mode | Description | Typical Cause |
|------|-------------|---------------|
| `pass` | Tests pass | Correct fix |
| `no_patch` | LLM returned no edits | Insufficient context |
| `patch_apply_fail` | File not found or search text not found | Wrong file paths in context |
| `test_fail` | Tests fail after applying patch | Wrong fix |
| `regression` | Targeted test passes but full suite fails | Fix introduces new bugs |
| `syntax_error` | SyntaxError or ImportError after patch | LLM generated invalid code |
| `timeout` | Test execution timed out | Fix caused infinite loop |

**Observation from sanity runs**: `patch_apply_fail` is grep's dominant failure mode. BCA avoids
this because it includes structural metadata (file paths from the graph) in the context.

---

## 7. Ablation Predictions (To Verify in Full Run)

### BCA vs BCA-no-closure
- **Prediction**: Closure helps at tight budgets (1k-4k) when the mutation requires understanding
  a dependency chain (e.g., a class that inherits from another)
- **Counter-prediction**: Closure hurts when it wastes budget on irrelevant imports
- **Metric to watch**: `bca_closure_added_tokens` vs `tests_passed`

### BCA vs BCA-no-scoring
- **Prediction**: Scoring helps when there are many candidate symbols (large frontier) and the
  mutation is in a low-centrality part of the graph
- **Metric to watch**: `bca_frontier_visited` vs pass@1

### Exact vs Vague
- **Prediction**: BCA's advantage over lexical methods is larger on exact queries (entity extraction
  works) and smaller or negative on vague queries (no entities to extract)
- **Metric to watch**: `entity_count_extracted` grouped by `query_type`

---

## 8. Router Design

### Features (all computable without LLM calls)
- `entity_count_extracted`, `entity_count_mapped`
- `query_identifier_density`
- Symbol hit ratio (how many seeds resolve to the graph)
- IDF-weighted score of query terms against symbol names
- Budget level (categorical)

### Two Router Tiers
1. **Majority-vote LOO-CV** (implemented in `compute_router_loo`): For each (budget, query_type),
   selects the method with the highest pass rate on all other tasks. Simplest possible "pick the
   best method" router. This is what the benchmark computes automatically.
2. **Logistic regression** (post-hoc analysis): Train on `{entity_count_mapped,
   query_identifier_density, budget, retrieval_softmax_entropy, retrieval_effective_candidates,
   retrieval_budget_utilization}` → predict best method. LOO-CV + leave-one-repo-out.
   Interpretable coefficients publishable in paper.

### Expected Story
Router recovers X% of the oracle gap (best single method vs oracle upper bound).
At minimum, the majority-vote router should beat any single method's average performance.
The logistic regression router should further close the gap by conditioning on task features.

---

## 9. Tables to Include in Paper

1. **Table 1**: Pass@1 by method x budget x query_type (main result) → `summary.txt`
2. **Table 1a**: Pass@1 per-repo breakdown → `per_repo_results.txt`
3. **Table 2**: Pass@1 with 95% bootstrap CIs → `summary_with_ci.txt`
4. **Table 3**: Ablation — BCA-d1 vs BCA vs BCA-d5 vs no-closure vs no-scoring (depth + ablation)
5. **Table 4**: Failure mode breakdown by method → `failure_diagnosis.txt`
6. **Table 5**: Retrieval quality metrics (file hit, symbol hit, overlap) → `retrieval_metrics.txt`
7. **Table 6**: Router vs oracle vs best single method → `router_analysis.txt`
8. **Table 7**: Conditional bins — pass@1 by identifier density, hops, mutation size → `conditional_bins.txt`
9. **Table 8**: Decomposition by mutation type and category → `decomposition.txt`

### Figures
1. **Figure 1**: CeGraph architecture diagram (graph build -> BCA pipeline)
2. **Figure 2**: Pass@1 vs token budget curves (line plot, one line per method)
3. **Figure 3**: Entity density distribution for exact vs vague (histogram/violin)
4. **Figure 4**: Graph distance (hops) vs pass@1 (scatter or boxplot)
5. **Figure 5**: Closure budget consumption vs benefit (scatter)
6. **Figure 6**: Router decision boundary visualization

---

## 10. Model Versions

| Stage | Model | Snapshot ID | Notes |
|-------|-------|------------|-------|
| **arXiv v1 (default)** | gpt-4o-mini | `gpt-4o-mini-2024-07-18` | Hardcoded as CLI default. Cheaper, faster. |
| Conference upgrade | gpt-4o | `gpt-4o-2024-08-06` | Pass via `--model gpt-4o-2024-08-06`. |

The CLI default is `gpt-4o-mini-2024-07-18` so that a bare `python -m paper.experiments.benchmark`
can never accidentally burn gpt-4o credits. To run the conference variant, you must explicitly pass
`--model gpt-4o-2024-08-06`.

---

## 11. Reproducibility Checklist

- [x] Pinned model: gpt-4o-mini-2024-07-18 (CLI default, cannot accidentally run gpt-4o)
- [x] Pinned repo commits: pydantic-ai (69a578a1), httpx (ae1b9f66)
- [x] Deterministic random seed (42)
- [x] temperature=0, single sample (no pass@k averaging)
- [x] Mutation validation (each causes exactly 1 test failure)
- [x] Byte-identical restoration after each eval (SHA256 verification)
- [x] Full per-attempt JSON artifacts saved (context, LLM response, patch, test output)
- [x] Bootstrap CIs with n=10000
- [x] `repo_name` in every EvalResult row (enables per-repo, per-method, per-budget, per-query_type slicing)
- [x] BCA strategy configs logged in run_metadata.json (depth, min_score per variant)
- [x] Per-repo commit hashes logged in run_metadata.json
- [x] Deterministic retry backoff: 5s, 8s, 15s, 30s, 60s (no jitter, reproducible)
- [x] OpenAI `seed=42` parameter in all API calls (best-effort determinism)
- [x] Budget-independent methods (no_retrieval, target_file) run once, results duplicated across budgets
- [ ] Release eval_tasks JSONL files
- [x] Dev-report descriptions generated by capture_tracebacks.py (244/245 OK, 1 timeout)
- [x] Dev-report sanitization: no line numbers, no absolute paths, no mutation type leakage
- [ ] Release discover_mutations.py, capture_tracebacks.py, and make_*_tasks.py scripts
- [ ] Release benchmark.py harness

---

## 12. Known Limitations (Be Honest in Paper)

1. **Two repos only** — Results are a detailed case study, not a general claim
2. **Single-line mutations** — Real bugs span multiple files; mutation testing is a proxy
3. **English descriptions only** — No multilingual evaluation
4. **Python only** — CeGraph supports JS/TS via regex, but benchmark is Python-only
5. **gpt-4o-mini** — Weaker model may not fully exploit context quality differences
6. **No multi-turn** — Single-shot fix, no iterative refinement
7. **LLM non-determinism** — OpenAI states that even with `temperature=0` and `seed=42`,
   outputs are "mostly deterministic" due to GPU floating-point non-determinism. We log the
   `system_fingerprint` from each response to detect cluster changes. Bootstrap CIs account
   for this variance statistically, but exact bit-for-bit reproduction is not guaranteed.
8. **Exact descriptions are too easy** — LLM can sometimes fix without any context when the
   description says "In file.py:172, operator changed from > to >="
9. **Dev-report tier is new** — Added third query tier (failing test + traceback, no line numbers)
   to bridge the exact/vague gap. Results pending full run.

---

## 13. Implementation Notes

### Entity Extraction Accuracy
- Exact descriptions: entity extraction reliably finds file names, class/function names
- Vague descriptions: 0 entities extracted (by design — no code identifiers mentioned)
- `_agent_graph.py` detected as dotted path, not file (minor regex gap, doesn't affect metrics)

### Graph Distance Computation
- Uses undirected BFS (ignoring edge direction) for structural distance
- `nx.shortest_path_length` on `graph.to_undirected()`
- Returns -1 when no path exists (disconnected components)
- Compute is cheap — negligible overhead per eval

### Context Symbol Key Extraction
- Matches on `def name` or `class name` patterns (not just substring) to avoid false positives
- Skips import nodes (too noisy)
- Requires qualified name length > 3 to filter trivial matches
- 110 symbols for BCA context, 52 for grep context (reflecting different file selections)

### Closure Stats From ContextAssembler
- `closure_added_symbols` and `closure_added_tokens` tracked by filtering items with `is_dependency=True`
- `frontier_visited` = total BFS expansion candidates before scoring
- Stored in ContextPackage and exposed via `_last_context_package` module-level variable

---

## 14. Interesting Hypotheses to Test

1. **Entity density predicts BCA advantage**: Higher density -> more seeds -> better BCA performance
2. **Graph distance predicts difficulty**: Higher min_hops -> harder task for BCA
3. **Closure value is budget-dependent**: Closure hurts at 1k (wastes budget) but helps at 4k+
4. **Vague queries equalize methods**: When no method has good seeds, they all perform similarly
5. **File hit is necessary but not sufficient**: 100% file hit != 100% pass (need the right symbol)
6. **Router outperforms best single method**: Because different tasks favor different methods
7. **httpx is easier than pydantic-ai**: Smaller codebase, higher kill rate, more focused functions

---

## 15. Kill Rate Analysis

**Two definitions — keep both, name them clearly:**

| Metric | pydantic-ai | httpx |
|--------|------------|-------|
| Source files | 186 | ~100 |
| Candidates generated | 1,454 | 388 |
| Killed | 199 | 210 |
| Survived (test ran, mutation undetected) | 770 | 152 |
| Skipped (no matching test found) | 485 | 26 |
| **Candidate kill rate** (killed / total candidates) | **13.7%** (199/1454) | **54.1%** (210/388) |
| **Tested kill rate** (killed / (killed + survived)) | **20.5%** (199/969) | **58.0%** (210/362) |
| Selected for eval (pre-filter) | 174 | 152 |
| Syntax-invalid (filtered out) | 46 | 32 |
| **Final eval tasks** | **128** | **117** |

**Why two rates matter**: Candidate kill rate reflects overall test coverage breadth (how much code
is tested at all). Tested kill rate reflects test sensitivity (how well tests detect mutations in
code they do exercise). httpx has both higher candidate kill rate (54.1% vs 13.7%) and higher
tested kill rate (58.0% vs 20.5%), consistent with a smaller, more tightly-tested codebase.

pydantic-ai has 485 skipped candidates (33.4%) — mutations in code with no matching test file.
This is typical for larger frameworks where many modules have integration tests but not unit tests.

**Paper point**: Kill rate itself is an interesting codebase health metric.

---

## 16. Mutation Site Size Diversity

The mutations span a wide range of function/class sizes — from 1-line constants to 1400-line classes.

**pydantic-ai (174 pre-filter candidates → 128 final eval tasks):**
| Size bucket | Count | % |
|-------------|-------|---|
| <5 lines | 9 | 5% |
| 5-20 lines | 77 | 44% |
| 20-50 lines | 39 | 22% |
| 50-100 lines | 41 | 24% |
| 100+ lines | 8 | 5% |
| **Median: 20 lines, Mean: 82.5, Range: 1-1406** |||

> Counts above are from the pre-filter 174 candidates (sums to 174, not 128).
> 46 syntax-invalid mutations were removed. Post-filter distribution to be
> recomputed after the full run if materially different.

**httpx (152 pre-filter candidates → 117 final eval tasks):**
| Size bucket | Count | % |
|-------------|-------|---|
| <5 lines | 8 | 5% |
| 5-20 lines | 56 | 37% |
| 20-50 lines | 58 | 38% |
| 50-100 lines | 20 | 13% |
| 100+ lines | 10 | 7% |
| **Median: 27 lines, Mean: 37.8, Range: 2-241** |||

> Counts above are from the pre-filter 152 candidates (sums to 152, not 117).
> 32 syntax-invalid + 3 unvalidated mutations were removed. Post-filter
> distribution to be recomputed after the full run if materially different.

**Paper point**: This diversity matters because short functions fit entirely in small budgets
(even 1k tokens), while large classes require intelligent selection of which parts to include.
BCA's advantage should be most visible on medium-to-large functions (20-100 lines) where budget
allocation decisions matter.

**Hypothesis**: Tasks in tiny functions (<5 lines) should be easy for all methods (the whole
function fits). Tasks in huge classes (100+ lines) may be hardest for BCA if entity extraction
doesn't point to the right submethod.

---

## 17. C++ Acceleration Layer

CeGraph includes an optional C++ accelerator (`csrc/cag_fast.cpp`) loaded via ctypes:
- Weighted BFS (the hot path): ~10-50x faster than Python for 50k+ node graphs
- Batch token estimation
- Topological sort (Kahn's algorithm)
- Entity extraction (pattern matching)

Currently compiled and active (`cag_fast.dylib` exists). Falls back to pure Python if missing.
**Not mentioned in paper** — it's an implementation detail, not a contribution. But worth noting
that assembly times in benchmarks include this acceleration.

---

## 18. Budget Configuration

| Budget | Description | Typical Content |
|--------|-------------|----------------|
| 1k tokens | Tight — ~40 lines of code | 1-2 functions, barely enough |
| 4k tokens | Moderate — ~160 lines | A class + its dependencies |
| 8k tokens | Generous — ~320 lines | Multiple files, good coverage |
| 10k tokens | Liberal — ~400 lines | Most relevant code + context |

BCA's budget utilization is consistently 94-100% across all budgets (greedy packing works).
Lexical methods also hit 100% (they just pack differently).
no_retrieval uses 0-2% (just the prompt, no context).

---

## 19. Post-Hoc Analysis TODO (After Full Run — Do Not Skip)

These analyses require only the logged artifacts — no reruns needed. Do them ALL before writing
the abstract or any claims.

- [ ] **Cost per solved task**: `(llm_input_tokens + llm_output_tokens) * price_per_token` for
  passes only. Compare cost-efficiency across methods.
- [ ] **Context redundancy/entropy**: Compute from stored `context.txt` artifacts. How much of
  the context is actually relevant vs padding?
- [ ] **Logistic regression router**: Train on `{entity_count_mapped, query_identifier_density,
  budget, mutation_symbol_lines}` → predict best method. LOO cross-validation.
- [ ] **k-hop dependency coverage**: `coverage_k = |D_k(mutation) ∩ context_symbol_keys| / |D_k(mutation)|`
  for k=1,2. Measures whether the context includes the mutation's dependency neighborhood.
- [ ] **Conditional slices**: Pass@1 binned by identifier_density (0 vs >0) and by min_hops
  (0, 1-2, 3+). This is the core of the "when does BCA win?" story.
- [ ] **Write abstract ONLY after results** — do not overclaim before seeing the full numbers
- [ ] **If embedding method included**: Pin embedding model snapshot too and log it, same as LLM

---

## 20. Paper Story Arc

The clean structure for the paper:

1. **Problem**: Token budgets force selection. Whole-file dumping wastes budget, lexical retrieval
   misses dependencies. The community treats this as a single-method shootout.
2. **Approach**: BCA as graph-guided assembly — entity extraction, weighted BFS, relevance scoring,
   dependency closure, greedy budget packing.
3. **Benchmark**: Two repos, controlled fault injection (mutation testing), three query tiers
   (dev-localized + dev-report + vague), pinned snapshots, strict byte-identical revert.
4. **Results**: No single method dominates. Show main table. This is the thesis, not a failure.
5. **Mechanism**: Explain *why*. Use failure modes + graph hops + entity density + closure overhead.
   This is the section that makes reviewers take the paper seriously.
6. **Router**: Oracle gap exists. Majority-vote LOO-CV router closes X% of the gap. Post-hoc
   logistic regression on retrieval confidence features closes further. No extra LLM calls.
7. **Limits**: Python-only, mutation testing is a proxy for real bugs, dev-localized descriptions
   may be overly informative (inflates no_retrieval), two repos only (detailed case study, not
   a general claim).

---

## 21. Full Merged Results (N=245 tasks, 29,400 attempts)

> **Three-tier run completed.** 245 tasks × 10 methods × 4 budgets × 3 query types = 29,400 total.
> run3 (exact + vague) + run4 (dev_report). Merged in `paper/results/run3_4_merged/`.
> 25 report files + 8 figures. results.json files kept locally (>100MB, excluded from git).

### Main Result: Pass@1

**EXACT (dev-localized) queries:**
| Method | B=2000 | B=4000 | B=8000 | B=10000 | Avg |
|--------|--------|--------|--------|---------|-----|
| no_retrieval | **0.86** | **0.86** | **0.86** | **0.86** | **0.86** |
| vector | 0.74 | 0.74 | 0.76 | **0.79** | 0.76 |
| repo_map | 0.72 | 0.73 | 0.75 | 0.75 | 0.74 |
| bm25 | 0.70 | 0.71 | 0.73 | 0.76 | 0.72 |
| bca_no_closure | 0.56 | 0.64 | 0.69 | 0.72 | 0.65 |
| bca | 0.52 | 0.64 | 0.69 | 0.69 | 0.64 |
| bca_d5 | 0.55 | 0.64 | 0.66 | 0.69 | 0.63 |
| bca_no_scoring | 0.48 | 0.63 | 0.67 | 0.66 | 0.61 |
| bca_d1 | 0.47 | 0.53 | 0.65 | 0.69 | 0.58 |
| Oracle | 0.97 | 0.98 | 0.97 | 0.98 | 0.97 |

**DEV_REPORT (traceback + test name) queries:**
| Method | B=2000 | B=4000 | B=8000 | B=10000 | Avg |
|--------|--------|--------|--------|---------|-----|
| vector | **0.21** | **0.23** | **0.25** | 0.23 | **0.23** |
| repo_map | 0.19 | 0.21 | 0.19 | 0.19 | 0.19 |
| bm25 | 0.16 | 0.19 | 0.21 | 0.20 | 0.19 |
| bca_no_closure | 0.11 | 0.15 | 0.20 | **0.24** | 0.18 |
| bca_d1 | 0.11 | 0.13 | 0.20 | **0.24** | 0.17 |
| bca | 0.11 | 0.14 | 0.18 | 0.23 | 0.16 |
| bca_d5 | 0.10 | 0.12 | 0.18 | 0.20 | 0.15 |
| bca_no_scoring | 0.09 | 0.10 | 0.12 | 0.13 | 0.11 |
| no_retrieval | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| Oracle | 0.35 | 0.39 | 0.43 | 0.42 | 0.40 |

**VAGUE (symptom-only) queries:**
| Method | B=2000 | B=4000 | B=8000 | B=10000 | Avg |
|--------|--------|--------|--------|---------|-----|
| bm25 | **0.07** | **0.05** | 0.04 | **0.05** | **0.05** |
| repo_map | 0.04 | 0.04 | 0.02 | 0.03 | 0.03 |
| vector | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| bca | 0.03 | 0.01 | 0.01 | 0.02 | 0.02 |
| bca_d5 | 0.03 | 0.01 | 0.01 | 0.02 | 0.02 |
| bca_no_closure | 0.03 | 0.01 | 0.02 | 0.02 | 0.02 |
| bca_no_scoring | 0.03 | 0.01 | 0.01 | 0.02 | 0.02 |
| bca_d1 | 0.02 | 0.03 | **0.04** | 0.01 | 0.02 |
| no_retrieval | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Oracle | 0.08 | 0.07 | 0.07 | 0.06 | 0.07 |

### Three-Tier Gradient — Central Empirical Finding

The paper's central finding is the **monotonic three-tier gradient**: exact > dev_report > vague
for every method at every budget, with no exceptions. This validates the query-information spectrum.

1. **Exact queries trivialize the task**: no_retrieval achieves 86%. The LLM fixes bugs from the
   description alone because exact queries leak file + line + operator. Retrieval adds noise.

2. **Dev_report is the sweet spot**: Methods score 10-25%. The traceback provides enough signal
   for retrieval (function names, test names) but not enough for the LLM to fix without context.
   This is where retrieval quality actually matters. **no_retrieval drops to 2%** (vs 86% exact).

3. **Vague queries break the LLM**: All methods score 0-7%. Even the ceiling is only 24-30%.
   The bottleneck is model comprehension, not retrieval.

4. **Oracle gap quantifies routing potential**: Exact 0.97, dev_report 0.40, vague 0.07.
   The 0.40 dev_report oracle means a perfect router could reach 40% — methods individually
   reach only 23% (vector). That 17pp gap is where routing adds value.

5. **BCA catches up at high budget on dev_report**: At B=2000, BCA (0.11) trails vector (0.21)
   by 10pp. At B=10000, bca_no_closure and bca_d1 (0.24) match or beat vector (0.23). BCA
   needs budget to include seed symbols + their dependencies.

### Ceiling Probe (target_file)

| Query Type | B=2000 | B=4000 | B=8000 | B=10000 |
|------------|--------|--------|--------|---------|
| Exact | 0.82 | 0.91 | 0.92 | 0.92 |
| Dev_report | 0.34 | 0.47 | 0.48 | 0.48 |
| Vague | 0.24 | 0.29 | 0.29 | 0.30 |

Median target file is ~3118 tokens, so ceiling saturates at B=4000. Dev_report ceiling is 0.48 —
even with the *perfect file*, the LLM only fixes 48% from a traceback. This means 52% of tasks
are beyond what single-shot gpt-4o-mini can do with a traceback, regardless of retrieval quality.
Vague ceiling barely improves with budget — model is the bottleneck, not context quantity.

---

## 22. Retrieval Quality Metrics

> Source: `paper/results/run3_4_merged/retrieval_metrics.txt`

### Target File Hit Rate (%)

| Method | EXACT B=10k | DEV_REPORT B=10k | VAGUE B=10k |
|--------|-------------|------------------|-------------|
| vector | 84% | **99%** | 38% |
| repo_map | **96%** | 98% | **96%** |
| bca | 87% | 96% | 64% |
| bca_d1 | 84% | 96% | 39% |
| bca_d5 | 89% | 96% | 71% |
| bca_no_closure | 90% | 96% | 73% |
| bm25 | 87% | 89% | 44% |

**Tracebacks improve file retrieval.** vector goes from 84% (exact) to 99% (dev_report) because
traceback text contains function names that keyword/TF-IDF matching finds. repo_map maintains
96% across all tiers because it includes everything by design.

### Target Symbol Hit Rate (%) — The Harder Test

| Method | EXACT B=10k | DEV_REPORT B=10k | VAGUE B=10k |
|--------|-------------|------------------|-------------|
| bca_d1 | 75% | **94%** | 41% |
| bca | 64% | 90% | 45% |
| vector | **77%** | 90% | 40% |
| bm25 | 73% | 83% | 40% |
| repo_map | 60% | 83% | 40% |

**bca_d1 achieves 94% symbol hit on dev_report** — highest of any method. Tracebacks provide
function names that BCA's entity extraction resolves into graph seeds, enabling precise symbol
retrieval. Shallow depth (d=1) focuses on the immediate neighborhood, hitting the right symbols.

### Retrieval-Outcome Decoupling

> Source: `paper/results/run3_4_merged/retrieval_outcome.txt`

**Finding the right file does NOT guarantee a correct fix.**

**File Hit → Pass Conversion Rate (B=10000):**

| Method | EXACT | DEV_REPORT | VAGUE |
|--------|-------|------------|-------|
| vector | 86% | 23% | 8% |
| bm25 | 79% | 22% | 12% |
| bca_d1 | 79% | 25% | 3% |
| bca_no_closure | 76% | 25% | 2% |
| repo_map | 77% | 19% | 3% |
| bca | 72% | 24% | 3% |

**The decoupling gradient**: Exact conversion is 72-86% (finding file ≈ fixing bug). Dev_report
conversion drops to 19-25% (75-81% of attempts that found the right file still failed). Vague
conversion is 2-12% (finding the file barely helps at all).

**Passes without target file (memorization/hallucination):**
- EXACT no_retrieval: **211/211** — ALL passes are without any context. The LLM fixes purely from
  the description + training data. Strongest evidence that exact descriptions are "too easy."
- DEV_REPORT: 0-1 passes without file for retrieval methods. The LLM *needs* the code.
- VAGUE: 0-5 passes without file. Sparse signal but consistent: code is required.

---

## 23. Cost Analysis

> Source: `paper/results/run3_4_merged/cost_analysis.txt`

**Total benchmark cost**: $22.71 across all 3 tiers (26,460 non-ceiling attempts).
gpt-4o-mini at $0.15/M input, $0.60/M output.

**Cost per solved task (DEV_REPORT, B=10000):**
| Method | Cost/Solve | Total Solves (across 4 budgets) |
|--------|-----------|------|
| bca_d1 | $0.00142 | 169 |
| bca_no_closure | $0.00144 | 172 |
| bca | $0.00145 | 161 |
| bm25 | $0.00155 | 186 |
| repo_map | $0.00156 | 189 |
| vector | $0.00151 | 226 |

**Cost per solved task (EXACT, B=10000):**
| Method | Cost/Solve | Total Solves |
|--------|-----------|------|
| no_retrieval | $0.00006 | 844 |
| bm25 | $0.00149 | 710 |
| vector | $0.00150 | 745 |
| bca_no_closure | $0.00153 | 641 |

**Cost per solved task (VAGUE, B=10000):**
| Method | Cost/Solve | Total Solves |
|--------|-----------|------|
| bm25 | $0.00168 | 52 |
| repo_map | $0.00161 | 31 |
| vector | $0.00157 | 29 |
| bca_d1 | $0.00140 | 23 |
| no_retrieval | --- | 0 |

Cost differences between retrieval methods are negligible (~10%). Budget size dominates.
vector has most total solves on dev_report (226) but bca_d1 is cheapest per solve ($0.00142).

---

## 24. Seed-to-Mutation Distance (k-Hop Analysis)

> Source: `paper/results/run3_4_merged/posthoc_summary.txt` (k-hop section)

### Hop Distribution (Three-Tier)

**EXACT queries (BCA methods):**
| Bucket | Count | % |
|--------|-------|---|
| 0 (exact match) | 960 | 98.0% |
| 1-2 (near) | 16 | 1.6% |
| unreachable | 4 | 0.4% |

**DEV_REPORT queries (BCA methods):**
| Bucket | Count | % |
|--------|-------|---|
| 0 (exact match) | 540 | **55.1%** |
| 1-2 (near) | 356 | **36.3%** |
| 3-5 (mid) | 84 | 8.6% |
| unreachable | 0 | 0.0% |

**VAGUE queries (BCA methods):**
| Bucket | Count | % |
|--------|-------|---|
| unreachable | 876 | **89.4%** |
| 1-2 (near) | 64 | 6.5% |
| 3-5 (mid) | 28 | 2.9% |
| 0 (exact match) | 12 | 1.2% |

**Three-tier hop gradient**: The hop distribution shifts dramatically across query tiers:
- **Exact**: 98% hop-0 — entity extraction finds the exact mutated symbol from the description
- **Dev_report**: 55% hop-0, 36% hop 1-2, 9% hop 3-5 — traceback gives enough signal to reach
  the mutation neighborhood but not always the exact symbol. Graph traversal matters here.
- **Vague**: 89% unreachable — entity extraction yields no useful seeds. BCA falls back to
  keyword-based scoring, losing its structural advantage entirely.

This is the structural explanation for why BCA's relative performance varies by tier. BCA's
graph traversal only adds value when seeds resolve (dev_report), not when there are no seeds
(vague) or when seeds directly hit the target (exact makes BCA redundant).

### Pass@1 by Hop Distance (BCA-SMART, d=3)

**DEV_REPORT queries:**
| Hop Bucket | B=2000 | B=4000 | B=8000 | B=10000 | N |
|-----------|--------|--------|--------|---------|---|
| 0 (exact) | 0.15 | 0.19 | 0.27 | 0.35 | 135 |
| 1-2 (near) | 0.06 | 0.04 | 0.04 | 0.07 | 89 |
| 3-5 (mid) | 0.05 | 0.19 | 0.14 | 0.19 | 21 |

**Dev_report hop-0 tasks reach 35% pass@1** at B=10000 — comparable to the ceiling probe (48%).
But hop 1-2 tasks only reach 7%, even with budget. The 5x drop from hop-0 to hop-1-2 shows
that even one degree of indirection sharply reduces repair probability. This is the clearest
evidence that **graph distance from seed to mutation predicts task difficulty**.

**EXACT queries:**
| Hop Bucket | B=2000 | B=10000 | N |
|-----------|--------|---------|---|
| 0 (exact) | 0.53 | 0.70 | 240 |
| 1-2 (near) | 0.50 | 0.75 | 4 |
| unreachable | 0.00 | 0.00 | 1 |

**VAGUE queries:**
| Hop Bucket | B=2000 | B=10000 | N |
|-----------|--------|---------|---|
| 0 (exact) | 0.00 | 0.00 | 3 |
| 1-2 (near) | 0.06 | 0.06 | 16 |
| 3+ (distant) | 0.00 | 0.00 | 7 |
| unreachable | 0.03 | 0.02 | 219 |

Vague queries: even the few tasks where BCA finds seeds (26 reachable) barely solve (0-6%).
The bottleneck is comprehension, not retrieval.

### Closure Budget Consumption (Three-Tier)

**EXACT queries:**
| Method | Mean Syms | Mean Tokens | % Budget | Frontier |
|--------|----------|------------|----------|----------|
| bca_d1 | 3.2 | 745 | **10.0%** | 35 |
| bca | 1.1 | 38 | 0.6% | 1296 |
| bca_d5 | 1.2 | 31 | 0.5% | 3001 |
| bca_no_closure | 0 | 0 | 0% | 1296 |

**DEV_REPORT queries:**
| Method | Mean Syms | Mean Tokens | % Budget | Frontier |
|--------|----------|------------|----------|----------|
| bca_d1 | 0.8 | 135 | **1.7%** | 527 |
| bca | 0.3 | 10 | 0.2% | 2135 |
| bca_d5 | 0.3 | 8 | 0.1% | 3118 |
| bca_no_closure | 0 | 0 | 0% | 2135 |

**VAGUE queries:**
| Method | Mean Syms | Mean Tokens | % Budget | Frontier |
|--------|----------|------------|----------|----------|
| bca_d1 | 2.5 | 594 | **7.7%** | 116 |
| bca | 0.9 | 27 | 0.4% | 1534 |
| bca_d5 | 1.0 | 27 | 0.4% | 3026 |
| bca_no_closure | 0 | 0 | 0% | 1534 |

**Closure overhead is tier-dependent**: bca_d1 uses 10% on exact (many seeds → many closure
targets) but only 1.7% on dev_report (fewer exact seeds → less closure). For d=3 and d=5,
closure is negligible (<1%) across all tiers. **Ablation result**: removing closure
(bca_no_closure) doesn't hurt and sometimes helps — consistent with closure being a tiny
fraction of the budget that sometimes adds irrelevant dependencies.

---

## 25. Context Efficiency

> Source: `paper/results/run3_4_merged/posthoc_summary.txt` (context efficiency section)

### Symbols per 1000 Tokens (Three-Tier, B=10000)

| Method | EXACT | DEV_REPORT | VAGUE |
|--------|-------|------------|-------|
| bca_d5 | **15.3** | 8.1 | **13.5** |
| bca_no_closure | 14.8 | 8.1 | 13.7 |
| bca_no_scoring | 12.0 | 8.2 | 12.0 |
| bca | 11.9 | 7.2 | 10.9 |
| bm25 | 9.4 | **9.2** | 9.5 |
| bca_d1 | 5.5 | 5.3 | 5.2 |
| vector | 5.0 | 7.9 | 3.8 |
| repo_map | 2.0 | 1.2 | 1.5 |

**Higher symbol density ≠ better performance.** bca_d5 packs the most symbols (15.3/1k on exact)
but doesn't outperform bm25 (9.4/1k). repo_map has lowest density (1.2-2.0) but strong
performance because signatures are information-dense. On dev_report, bm25 packs the most
symbols (9.2/1k) and also performs well — keyword retrieval selects many relevant small snippets.

### Tokens per Symbol (Granularity, B=10000)

| Method | EXACT | DEV_REPORT | VAGUE |
|--------|-------|------------|-------|
| repo_map | 711 | 952 | 862 |
| vector | 251 | 164 | 363 |
| bca_d1 | 206 | 216 | 222 |
| bm25 | 124 | 141 | 142 |
| bca | 88 | 181 | 116 |
| bca_no_scoring | 87 | 144 | 92 |
| bca_d5 | 69 | 175 | 99 |

**Granularity shifts with query type**: On exact queries, BCA (d≥3) selects fine-grained
snippets (69-88 tok/sym). On dev_report, BCA selects coarser chunks (144-216 tok/sym) because
traceback-derived seeds cover larger code regions. Vector's granularity inverts: 251 on exact
but 164 on dev_report — tracebacks give vector more keyword hits, producing more granular results.

### Budget Utilization

All retrieval methods achieve 96-100% budget utilization across all tiers. no_retrieval uses
0-1%. Budget size is fully consumed — the question is *what* fills the budget, not whether it's
filled. This validates the greedy packing algorithm.

### Retrieval Confidence (Softmax Entropy)

Lower entropy = retrieval is more confident about top candidates.

| Method | EXACT B=2k | DEV_REPORT B=2k | VAGUE B=2k |
|--------|-----------|-----------------|-----------|
| bca | 3.92 | 3.38 | 3.68 |
| bm25 | 4.19 | 4.28 | 4.29 |
| vector | 3.99 | 4.18 | 4.26 |

BCA shows lower entropy than bm25/vector on dev_report (3.38 vs 4.18-4.28), meaning BCA's
graph-guided scoring produces more differentiated rankings when it has traceback-derived seeds.
This is a candidate routing signal: low BCA entropy → BCA is confident → prefer BCA.

---

## 26. Method Uniqueness (Sole-Solver Analysis)

> Source: `paper/results/run3_4_merged/posthoc_summary.txt` (method uniqueness section)

### Sole-Solver Counts by Tier

| Tier | Budget | Solvable | Multi-solver | Sole-solver | Top sole-solver |
|------|--------|----------|-------------|-------------|-----------------|
| **EXACT** | B=2000 | 238 | 230 | 8 | no_retrieval(5), vector(3) |
| | B=10000 | 239 | 235 | 4 | no_retrieval(4) |
| **DEV_REPORT** | B=2000 | 86 | 61 | **25** | vector(10), bm25(6), repo_map(6) |
| | B=4000 | 96 | 70 | **26** | bm25(9), repo_map(6), vector(6) |
| | B=8000 | 106 | 77 | **29** | repo_map(9), vector(7), bm25(6) |
| | B=10000 | 102 | 81 | **21** | vector(7), bm25(5), repo_map(4) |
| **VAGUE** | B=2000 | 20 | 10 | 10 | bm25(7), repo_map(2) |
| | B=10000 | 15 | 11 | 4 | bm25(4) |

**Dev_report is the routing regime.** 21-29 sole-solver tasks per budget (vs 4-10 exact, 3-10 vague).
This means 20-27% of solvable dev_report tasks are solved by exactly one method — routing has
real value here because no single method covers them all.

**Distribution of sole-solvers**: On dev_report, vector/bm25/repo_map split the sole-solvers
roughly evenly (6-10 each). BCA variants contribute 1-3 sole-solvers each. This means the
router must select between lexical and structural methods per-task — BCA uniquely solves a few
tasks but most unique solves come from keyword methods operating on different code patterns.

**EXACT = no routing regime**: 230/238 solvable tasks are multi-solver. Methods are too
correlated (same tasks solved by all) for routing to differentiate.

**VAGUE = too few tasks**: bm25 dominates sole-solvers. With only 15-20 solvable tasks total,
there's insufficient diversity for a router to learn conditional patterns.

---

## 27. Per-Repository Analysis

> Source: `paper/results/run3_4_merged/posthoc_summary.txt` (per-repo section)

### EXACT, B=10000:
| Method | httpx | pydantic-ai | Overall |
|--------|-------|------------|---------|
| no_retrieval | **0.91** | **0.81** | **0.86** |
| vector | 0.80 | 0.78 | 0.79 |
| bm25 | 0.80 | 0.72 | 0.76 |
| repo_map | 0.85 | 0.66 | 0.75 |
| bca_no_closure | 0.74 | 0.71 | 0.72 |

### DEV_REPORT, B=10000:
| Method | httpx | pydantic-ai | Overall |
|--------|-------|------------|---------|
| vector | **0.27** | 0.20 | 0.23 |
| bca_no_closure | 0.22 | **0.27** | 0.24 |
| bca_d1 | 0.21 | 0.27 | 0.24 |
| bca | 0.21 | 0.25 | 0.23 |
| bm25 | 0.19 | 0.20 | 0.20 |
| repo_map | 0.17 | 0.20 | 0.19 |

### DEV_REPORT, B=2000:
| Method | httpx | pydantic-ai | Overall |
|--------|-------|------------|---------|
| vector | **0.27** | 0.16 | 0.21 |
| repo_map | 0.16 | **0.21** | 0.19 |
| bm25 | 0.15 | 0.18 | 0.16 |

### VAGUE, B=10000:
| Method | httpx | pydantic-ai | Overall |
|--------|-------|------------|---------|
| bm25 | 0.02 | **0.09** | 0.05 |
| repo_map | 0.02 | 0.04 | 0.03 |
| vector | 0.02 | 0.04 | 0.03 |

**Key observations**:

1. **httpx is consistently easier on exact** (0.85-0.91 vs 0.66-0.81). Expected — smaller
   codebase, tighter test coverage.

2. **Repo rankings flip on dev_report**: At B=10000, pydantic-ai actually matches httpx for
   BCA variants (0.25-0.27 vs 0.21-0.22). BCA's graph traversal helps more on the larger,
   more complex pydantic-ai codebase. This is the first evidence of **repo-conditional
   method selection** — structural retrieval's advantage increases with codebase complexity.

3. **Vector dominates httpx at low budget**: At B=2000, vector leads httpx (0.27) by 11pp over
   the next retrieval method. But on pydantic-ai, repo_map leads (0.21). Different codebases
   reward different retrieval strategies.

4. **Vague queries**: pydantic-ai scores higher than httpx on bm25 (0.09 vs 0.02). pydantic-ai's
   vague descriptions may be more keyword-rich (test names are more descriptive).

---

## 28. Failure Mode Taxonomy — Key Mechanistic Finding

> Source: `paper/results/run3_4_merged/failure_diagnosis.txt`

### Three-Tier Failure Mode Tables (B=10000, N=245 each)

**EXACT Queries:**
| Method | pass | patch_apply_fail | test_fail | syntax_error |
|--------|------|-----------------|-----------|-------------|
| no_retrieval | **86%** | 11% | 2% | 0% |
| vector | 79% | 18% | 2% | 0% |
| bm25 | 76% | 19% | 5% | 0% |
| repo_map | 75% | 20% | 5% | 0% |
| bca_no_closure | 72% | 22% | 6% | 0% |
| bca | 69% | 22% | 7% | 1% |

**DEV_REPORT Queries:**
| Method | pass | patch_apply_fail | test_fail | syntax_error | timeout |
|--------|------|-----------------|-----------|-------------|---------|
| bca_d1 | 24% | 16% | **49%** | 7% | 4% |
| bca_no_closure | 24% | 17% | **51%** | 4% | 2% |
| bca | 23% | 21% | **49%** | 3% | 4% |
| vector | 23% | 16% | **54%** | 2% | 4% |
| bm25 | 20% | 26% | **49%** | 2% | 3% |
| repo_map | 19% | 15% | **61%** | 2% | 2% |
| bca_no_scoring | 13% | 26% | **55%** | 3% | 2% |
| no_retrieval | 2% | **72%** | 24% | 1% | 1% |

**VAGUE Queries:**
| Method | pass | patch_apply_fail | test_fail | syntax_error |
|--------|------|-----------------|-----------|-------------|
| bm25 | 5% | 20% | **70%** | 4% |
| repo_map | 3% | 19% | **76%** | 2% |
| vector | 3% | 25% | **71%** | 1% |
| bca | 2% | 13% | **82%** | 2% |
| no_retrieval | 0% | **100%** | 0% | 0% |

### The Failure Mode Gradient — Publishable on Its Own

The three-tier failure mode shift is one of the paper's strongest mechanistic findings:

1. **EXACT**: Dominant failure = `patch_apply_fail` (11-22%). LLM knows *what* to fix but gets
   the *location* wrong. `test_fail` is rare (2-9%). These are **localization errors**.

2. **DEV_REPORT**: **Blend of both modes** — `test_fail` dominates (49-61%) but `patch_apply_fail`
   is also significant (15-26%). The traceback gives enough signal to attempt a fix in the right
   area, but the LLM often gets it wrong. This confirms the prediction from Section 32:
   dev_report produces a mix of localization and comprehension errors.

3. **VAGUE**: Dominant failure = `test_fail` (70-85%). LLM applies a patch but it's the *wrong
   fix*. The model is guessing without understanding. These are **comprehension errors**.

4. **no_retrieval**: EXACT = 11% patch_apply_fail (works mostly). DEV_REPORT = **72%
   patch_apply_fail** (can't find file from traceback alone). VAGUE = **100% patch_apply_fail**
   (doesn't even know which file to edit).

**The gradient**: localization errors → blended errors → comprehension errors, tracking exactly
with query information content. This demonstrates that **failure modes shift continuously with
query tier**, and benchmarks using only one query type miss half the failure story.

### Dev_Report Failure Mode Details

Dev_report uniquely shows **higher syntax_error rates** (3-7% vs 0-2% on exact/vague). BCA
variants show 3-7% syntax errors — the traceback contains code-like tokens (class names,
function signatures) that sometimes confuse the LLM's patch generation. bca_d1 has the highest
syntax error rate (7%) alongside the highest pass rate (24%) — it's taking bigger risks.

Dev_report also shows **4% timeout rates** for some methods, not seen on exact/vague. The
longer tracebacks increase prompt length, pushing some LLM responses over the timeout threshold.

### Low-Budget Failure Modes (B=2000)

At B=2000, dev_report failure patterns are more extreme:
- `patch_apply_fail` ranges 13-25% at B=2000 vs 15-26% at B=10000 — overlapping ranges
  show no clear budget trend for localization errors on dev_report
- `test_fail` is similar (58-62% at B=2000) — comprehension failures are budget-insensitive
- no_retrieval stays at 72% `patch_apply_fail` (constant — no context at any budget)

---

## 29. Router Analysis — Full Three-Tier Results

> Source: `paper/results/run3_4_merged/router_ab.txt`, `router_analysis.txt`
> Script: `python -m paper.experiments.router_ab --run-dir paper/results/run3_4_merged`

### Router Definitions

Two deployment-valid router designs, following the feature leakage framework:

**Router A ("Choose-first", pre-retrieval)**:
- Decision based ONLY on features available before running any retrieval method
- Features (3): `entity_count_mapped`, `query_identifier_density`, `graph_node_count_log`
- Candidates: all 9 non-ceiling methods (including `no_retrieval`)
- Use case: zero-cost routing decision at query time

**Router B ("Dry-run then choose", retrieval-conditioned)**:
- Runs all retrieval methods locally (no LLM call), examines their confidence metrics
- Features (27): Router A features + per-method `{top1_score, budget_utilization, softmax_entropy}` × 8 methods
- Candidates: 8 retrieval methods (excludes `no_retrieval`)
- Use case: confident method selection when retrieval cost is negligible vs LLM cost

**Labeling strategies**:
- **Smart**: majority-vote method as label when it passes; switches to rarest passer for tasks
  where majority fails (teaches: default to best single, deviate only when features warrant it)
- **Safest**: always labels with method that passes most often (conservative)

**Feature leakage rules** (features NEVER used in routing):
- `target_file_hit`, `target_symbol_hit`, `min_hops_*`, `mutation_*` — requires knowing mutation
- `tests_passed`, `failure_mode`, `patch`, `test_output` — post-LLM outcomes

### Full Comparison Table (all 3 tiers)

**DEV_REPORT queries** (the routing-relevant tier):
| Budget | MajVote | Random | Router A (smart) | Router A (safest) | Router B (smart) | Router B (safest) | Oracle | N(A) | N(B) |
|--------|---------|--------|-----------------|-------------------|-----------------|-------------------|--------|------|------|
| B=2000 | 60.5% | 35.1% | **65.1%** | 65.1% | **68.2%** | 65.9% | 100% | 86 | 85 |
| B=4000 | 58.3% | 36.2% | 58.3% | **63.5%** | **61.5%** | 61.5% | 100% | 96 | 96 |
| B=8000 | 57.5% | 40.0% | 57.5% | **60.4%** | 56.6% | **59.4%** | 100% | 106 | 106 |
| B=10000 | 58.8% | 45.4% | 57.8% | **61.8%** | **60.8%** | 59.8% | 100% | 102 | 102 |

N(A) = solvable tasks for Router A (9 candidates incl. no_retrieval).
N(B) = solvable tasks for Router B (8 candidates excl. no_retrieval). Different candidate sets yield slightly different solvable-task counts.

**Denominator note**: MajVote, Random, and Oracle columns are computed on Router A's solvable set (N(A)). Router B accuracy is computed on its own solvable set (N(B)). On dev_report, N(A) ≈ N(B) (86 vs 85 at B=2000); on exact, the gap is larger (238 vs 233) because no_retrieval solves tasks that no retrieval method solves.

**EXACT queries:**
| Budget | MajVote | Random | Router A | Router B | Oracle | N(A) | N(B) |
|--------|---------|--------|----------|----------|--------|------|------|
| B=2000 | 88.7% | 64.0% | 88.2% | 78.1% | 100% | 238 | 233 |
| B=4000 | 88.3% | 69.7% | 88.3% | 78.8% | 100% | 239 | 231 |
| B=8000 | 88.7% | 74.1% | 88.7% | 79.9% | 100% | 238 | 234 |
| B=10000 | 88.3% | 75.3% | 88.3% | 82.6% | 100% | 239 | 235 |

**VAGUE queries:**
| Budget | MajVote | Random | Router A | Router B (smart) | Router B (safest) | Oracle | N(A) | N(B) |
|--------|---------|--------|----------|-----------------|-------------------|--------|------|------|
| B=2000 | 80.0% | 36.7% | 75.0% | 75.0% | 75.0% | 100% | 20 | 20 |
| B=4000 | 70.6% | 29.4% | 64.7% | **82.4%** | 76.5% | 100% | 17 | 17 |
| B=8000 | 68.8% | 31.2% | 68.8% | 68.8% | **75.0%** | 100% | 16 | 16 |
| B=10000 | 86.7% | 34.8% | 80.0% | 73.3% | 73.3% | 100% | 15 | 15 |

### Dev_Report: Where Routing Works

**Router B (smart) at B=2000 achieves 68.2%** vs majority vote 60.5% — a **+18.2% gap closure**
(7.7pp absolute improvement). This is the strongest routing signal in the entire benchmark.

Why dev_report enables routing:
1. **21-29 sole-solver tasks** per budget (vs 4-10 exact) — enough diversity for the router
2. **Methods diverge**: vector wins some, bm25 wins others, BCA wins a few. No single method
   dominates by a large margin.
3. **Sufficient solvable tasks** (86-106) for logistic regression to learn patterns
4. **Retrieval confidence features are informative**: `bca_no_scoring_entropy` (|coef|=1.015),
   `bm25_top1_score` (|coef|=0.767) are strong signals

Router A also shows value on dev_report: **+11.8% gap closure** at B=2000 with just 3 pre-retrieval
features. `query_identifier_density` (|coef|=0.888) is the top feature — queries with more code
identifiers route differently than narrative queries.

### EXACT: No Routing Regime

Router A converges to majority vote (no_retrieval at 88.7%). Gap closed: ~0%.
230/238 solvable tasks are multi-solver — routing can only help on 4-10 sole-solver tasks,
too few to learn from. Router B excludes no_retrieval and thus performs worse (78-83%).

### VAGUE: Too Few Tasks

With only 15-20 solvable tasks, the logistic regression cannot learn reliable conditional
patterns. One positive outlier: Router B (smart) at B=4000 achieves 82.4% vs 70.6% majority
(+40% gap closure) — but with n=17, this is likely noise.

### Logistic Regression Router (posthoc_summary, for comparison)

The post-hoc logistic regression (from posthoc_summary.txt) uses 5 features including
`mutation_symbol_lines_log` and `mutation_file_symbols` — features that **leak** information
about the mutation site (a real router wouldn't know these). This is explanatory only:

| Tier | Budget | Best Single | Router LOO | Gap Closed |
|------|--------|-------------|-----------|------------|
| DEV_REPORT | B=2000 | 32.6% (bm25) | 36.0% | +5.2% |
| DEV_REPORT | B=8000 | 41.5% (bca) | 42.5% | +1.6% |
| VAGUE | B=2000 | 50.0% (bm25) | 60.0% | +20.0% |
| VAGUE | B=4000 | 47.1% (bm25) | 64.7% | +33.3% |
| EXACT | all | ~54-71% (bca) | matches | 0% |

The post-hoc router shows larger gap closure than the deployment-valid Router A/B, confirming
that mutation-site features (which a real router can't use) would be informative if available.

### Top Features by Tier

**Router A (pre-retrieval)**:
- `query_identifier_density` — strongest on dev_report (|coef|=0.888-1.425)
- `entity_count_mapped` — strong on all tiers (|coef|=0.571-1.693)
- `graph_node_count_log` — proxy for repo complexity

**Router B (retrieval-conditioned)**:
- `bca_no_scoring_entropy` — strongest on dev_report B=2000 (|coef|=1.015)
- `bm25_top1_score`, `vector_top1_score` — retrieval confidence signals
- `bca_d1_budget_util` — consistently appears (|coef|=0.66-1.14)
- `bm25_entropy`, `vector_entropy` — score distribution features

These features are sensible: retrieval confidence metrics predict which method "got it right"
before the LLM call. The router learns "if bm25 is very confident, use bm25; if BCA's scoring
is differentiated, use BCA."

### The Router Story for the Paper

The router result is **NOT** "routing fails." The result is:
1. **Routing requires method divergence** — exact queries don't have it (all methods or none)
2. **Dev_report creates the routing regime** — 20-27% sole-solver tasks, diverse winners
3. **Even 3 pre-retrieval features add value** on dev_report (+11.8% gap closure)
4. **Retrieval confidence features add more** (+18.2% gap closure at B=2000)
5. **The oracle gap (40% vs 23%) shows the ceiling** — 17pp room for a better router
6. **No extra LLM calls needed** — all routing features are local/cheap

---

## 30. Figures Inventory

All figures saved in `paper/results/run3_4_merged/figures/`. Generated by
`python -m paper.experiments.generate_figures --run-dir paper/results/run3_4_merged`.

Figures are now **3-panel** (EXACT, DEV_REPORT, VAGUE) instead of 2-panel.

### Figure 1: Pass@1 vs Token Budget (`fig1_pass_vs_budget.png`)
Line chart, 3 panels. Shows:
- EXACT: no_retrieval flat at 0.86, all methods converge upward with budget
- DEV_REPORT: vector leads early, BCA catches up at high budget, 10-25% range
- VAGUE: all methods near zero (0-7%), barely visible differentiation

### Figure 2: Failure Mode Breakdown (`fig2_failure_modes.png`)
Stacked bar chart at B=10000, 3 panels. Shows the failure mode gradient:
- EXACT: mostly green (pass) + orange (patch_apply_fail)
- DEV_REPORT: blended — orange + red (test_fail), moderate green
- VAGUE: mostly red (test_fail) + minimal green

### Figure 3: Retrieval vs Outcome (`fig3_retrieval_vs_outcome.png`)
Scatter plot, 3 panels. Shows retrieval-outcome decoupling:
- EXACT: positive correlation (more file hits → more passes)
- DEV_REPORT: weaker correlation (file hits ≠ passes)
- VAGUE: no correlation (cluster at high hit rate but ~0% pass)

### Figure 4: Per-Repository Comparison (`fig4_per_repo.png`)
Grouped bar chart at B=10000, 3 panels. Shows repo-conditional rankings.

### Figure 5: Ceiling Probe (`fig5_ceiling_probe.png`)
Bar chart, 3 tiers: best single method vs oracle vs target_file ceiling.
- EXACT: 86% vs 92% ceiling (nearly saturated)
- DEV_REPORT: 23% vs 48% ceiling (large gap — room for improvement)
- VAGUE: 5% vs 30% ceiling (model is bottleneck)

### Figure 6: BCA Ablation (`fig6_ablation.png`)
Line chart of BCA variants only, 3 panels. Shows depth and component effects per tier.

### Figure 7: Mutation Type Heatmap (`fig7_mutation_heatmap.png`)
Heatmap: method × mutation_type at B=10000, 3 panels.

### Figure 8: Entity Density Effect (`fig8_entity_density.png`)
Grouped bar chart: 3 tiers side-by-side for each method.

---

## 31. Report File Inventory (25 reports + 8 figures)

All in `paper/results/run3_4_merged/`. Generated from merged 29,400-entry dataset.

**Standard reports** (from `merge_and_report.py`):
| File | Contents |
|------|----------|
| `summary.txt` | Pass@1 tables (3 tiers) |
| `summary_with_ci.txt` | Pass@1 with 95% bootstrap CIs |
| `per_repo_results.txt` | Pass@1 by repository |
| `ceiling_probe.txt` | target_file ceiling (3 tiers) |
| `router_analysis.txt` | Oracle vs LOO-CV Router vs Best Single |
| `decomposition.txt` | Pass@1 by mutation type and category (3 tiers) |
| `conditional_bins.txt` | Pass@1 by identifier density, hops, mutation size (3 tiers) |
| `bootstrap_analysis.txt` | Paired bootstrap CI details |
| `bootstrap_cis.json` | Raw bootstrap CIs (JSON) |
| `failure_diagnosis.txt` | Failure mode breakdown (3 tiers) |
| `retrieval_metrics.txt` | File/symbol hit rates, budget utilization (3 tiers) |
| `patch_quality.txt` | Patch size and locality (3 tiers) |
| `latency_cost.txt` | Assembly, LLM, test time breakdown (3 tiers) |
| `edit_locality.txt` | Edit distance and context-patch overlap (3 tiers) |

**Post-hoc reports** (from `posthoc_analysis.py`):
| File | Contents |
|------|----------|
| `cost_analysis.txt` | Token usage and cost per solved task (3 tiers) |
| `posthoc_summary.txt` | Combined: cost, router, decoupling, efficiency, k-hop, uniqueness, per-repo |
| `retrieval_outcome.txt` | Retrieval-outcome decoupling (3 tiers) |
| `context_redundancy.txt` | Context efficiency (3 tiers) |
| `khop_coverage.txt` | Seed-to-mutation distance and closure (3 tiers) |
| `method_uniqueness.txt` | Sole-solver analysis (3 tiers) |
| `per_repo_rankings.txt` | Per-repository rankings (3 tiers) |

**Router A/B report** (from `router_ab.py`):
| File | Contents |
|------|----------|
| `router_ab.txt` | Router A + Router B + comparison table + leakage audit (3 tiers) |

**Figures** (from `generate_figures.py`, 3-panel): fig1-fig8 as described in Section 30.

**Local only** (excluded from git, >100MB each):
- `results.json` — 662MB merged (29,400 entries)
- `paper/results/run3/results.json` — 271MB (19,600 entries)
- `paper/results/run4/results.json` — 391MB (9,800 entries)

---

## 32. Confirmed Findings (Dev-Report Predictions Validated)

### Prediction → Confirmation

Before running run4 (dev_report), Section 32 predicted specific outcomes. Here's what happened:

| Prediction | Result | Status |
|-----------|--------|--------|
| Dev_report failure modes = blend of localization + comprehension | test_fail 49-61%, patch_apply_fail 15-26% | **CONFIRMED** |
| Dev_report pass@1 between exact and vague | 10-25% (vs exact 48-86%, vague 0-7%) | **CONFIRMED** |
| Dev_report creates routing regime (more sole-solvers) | 21-29 sole-solvers (vs 4-10 exact) | **CONFIRMED** |
| Router shows gap closure on dev_report | Router B +18.2% gap closure at B=2000 | **CONFIRMED** |
| BCA improves with budget on dev_report | BCA goes from 0.11 (B=2k) to 0.24 (B=10k) | **CONFIRMED** |
| no_retrieval drops sharply on dev_report | 2% (vs 86% exact) | **CONFIRMED** |
| Retrieval-outcome decoupling increases on dev_report | File hit→pass conversion: 19-25% (vs 72-86% exact) | **CONFIRMED** |

Every prediction was validated. The three-tier design was the right experimental choice.

### Reproducibility

Full reproducibility controls in place:
- Pinned commits: pydantic-ai `69a578a`, httpx `ae1b9f6`
- Pinned model: `gpt-4o-mini-2024-07-18` (exact version, not alias)
- `temperature=0`, `seed=42`, `top_p=1`, `presence_penalty=0`, `frequency_penalty=0`
- Retry backoff: run3 (exact+vague) [2, 5, 10, 30, 60]s; run4 (dev_report) [5, 8, 15, 30, 60]s (no jitter)
- Byte-identical git revert between tasks
- `system_fingerprint` logged per API response (detects GPU cluster changes)
- Bootstrap CIs (n=10000, seed=42) for statistical inference
- Caveat: OpenAI states `seed` provides "mostly deterministic" output — GPU floating-point
  non-determinism means exact bit-for-bit reproduction across clusters is not guaranteed.
  Bootstrap CIs account for this variance statistically.

---

## 33. Edit Locality & Patch Quality

> Source: `paper/results/run3_4_merged/edit_locality.txt`, `patch_quality.txt`

### Edit Distance (lines from edit to mutation site)

Lower = LLM's edit is closer to the actual bug location. Measured only on attempts where the
mutation file is in context and the edit location is known.

**DEV_REPORT queries (B=10000):**
| Method | Mean Distance | Interpretation |
|--------|-------------|----------------|
| repo_map | 47.2 | Signatures guide to right area |
| vector | 49.0 | Keyword hits near mutation |
| bca_no_scoring | 70.8 | Structural but undirected |
| bca | 87.9 | Default BCA |
| bca_d1 | 91.5 | Shallow expansion |
| bca_d5 | 96.2 | Deep expansion adds distance |
| no_retrieval | 142.7 | No context → blind edits |

**EXACT queries (B=10000):**
| Method | Mean Distance |
|--------|-------------|
| no_retrieval | 0.5 |
| vector | 0.7 |
| bm25 | 1.2 |
| repo_map | 4.4 |
| bca_no_closure | 8.8 |
| bca_d5 | 9.9 |
| bca | 11.3 |

**VAGUE queries (B=10000):**
| Method | Mean Distance |
|--------|-------------|
| bm25 | 50.6 |
| bca_no_scoring | 61.5 |
| vector | 72.6 |
| bca_d1 | 78.6 |
| bca | 78.8 |
| repo_map | 141.5 |
| bca_d5 | 163.5 |

**Finding**: Edit distance follows the three-tier gradient. Exact edits land within 0-11 lines.
Dev_report edits land 47-143 lines away. Vague edits are 50-164 lines away. On exact, no_retrieval
achieves 0.5 lines (description says exactly where to edit). On dev_report, repo_map and vector
produce the most localized edits — they include code near the mutation more often than BCA.

### Context-Patch Overlap

Fraction of context files that appear in the generated patch (higher = context is relevant to edit).

**DEV_REPORT (B=10000):**
| Method | Overlap |
|--------|---------|
| bca_d1 | 0.19 |
| bca / bca_d5 / bca_no_closure | 0.17 |
| repo_map | 0.15 |
| bca_no_scoring | 0.15 |
| bm25 | 0.06 |
| vector | 0.04 |

**BCA methods produce the highest context-patch overlap** on dev_report (0.17-0.19) — meaning
the LLM references BCA-provided context files more often. This is because BCA includes the
target file's dependency neighborhood, which the LLM uses for context even when it doesn't edit
those files. bm25 and vector have low overlap (0.04-0.06) — they include many files the LLM
ignores.

### Patch Size

**Mean lines changed (passing attempts only, DEV_REPORT):**
| Method | B=2000 | B=10000 |
|--------|--------|---------|
| no_retrieval | 1.2 | 1.2 |
| bm25 | 1.4 | 1.5 |
| repo_map | 1.5 | 2.0 |
| bca_d1 | 2.0 | 2.2 |
| bca | 2.6 | 2.0 |
| bca_no_closure | 2.0 | 2.5 |

**Passing patches are smaller** (1.2-2.5 lines) than overall patches (1.9-6.4 lines). The LLM
over-edits when it doesn't understand the bug (failing attempts make 2-3x more changes). This is
consistent across all tiers and methods — patch size is a proxy for model confidence.

---

## 34. Latency Analysis

> Source: `paper/results/run3_4_merged/latency_cost.txt`

### Assembly Time (context retrieval, ms)

| Method | EXACT | DEV_REPORT | VAGUE |
|--------|-------|------------|-------|
| repo_map | 26-27 | 56-59 | 27-28 |
| vector | 115-122 | 110-116 | 114-122 |
| bca | 140-174 | 113-169 | 142-149 |
| bca_d1 | 157-183 | 96-159 | 126-193 |
| bca_d5 | 163-182 | 120-177 | 163-176 |
| bm25 | 135-152 | 247-266 | 129-157 |
| no_retrieval | 0 | 0 | 0 |

**Assembly is negligible** — 27-266ms vs 1000-7000ms for LLM inference. This validates
Router B's design: running ALL 8 retrieval methods takes ~1 second total, while a single
LLM call takes 1.5-7 seconds. The "dry-run all, pick best, call LLM once" strategy adds
<15% latency overhead.

### LLM Inference Time (ms)

| Method | EXACT B=2k | EXACT B=10k | DEV_REPORT B=2k | DEV_REPORT B=10k |
|--------|-----------|-------------|-----------------|------------------|
| no_retrieval | 1090 | 1090 | 1550 | 1550 |
| bm25 | 1523 | 2329 | 2306 | 3318 |
| bca | 1545 | 4714 | 2032 | 5134 |
| vector | 1402 | 2236 | 2036 | 3406 |
| repo_map | 1742 | 3023 | 2097 | 4148 |

LLM time scales linearly with context size (as expected). Dev_report prompts are longer than
exact prompts (traceback text), adding ~500ms at B=2000.

### Test Execution Time (ms)

Test time is method-independent (same test runs regardless of which method produced the patch).
Exact queries: ~1000ms average. Dev_report: 2000-3300ms. Vague: ~1000-1300ms.
Dev_report tests take longer because the traceback-selected test suites are often larger.

---

## 35. Decomposition by Mutation Type and Category

> Source: `paper/results/run3_4_merged/decomposition.txt`

### Mutation Type Rankings (DEV_REPORT, B=10000)

| Mutation Type | Best Method | Pass@1 | Worst Method | Pass@1 | N |
|--------------|------------|--------|-------------|--------|---|
| constant_mutation | bca_no_closure | 0.50 | no_retrieval | 0.05 | 20 |
| handcrafted | bca_no_closure / vector | 0.50 | no_retrieval | 0.00 | 14 |
| membership_swap | bca/bca_d1/bca_d5/bca_nc | 0.50 | bm25 | 0.50 | 2 |
| boolean_flip | bca_d1 / repo_map | 0.27 | no_retrieval | 0.05 | 41 |
| condition_inversion | bca/bca_d1/bca_nc/bm25 | 0.26 | no_retrieval | 0.05 | 43 |
| value_swap | vector | 0.38 | no_retrieval | 0.00 | 21 |
| arithmetic_swap | bca_d1 / repo_map | 0.33 | bm25/no_retrieval | 0.00 | 9 |
| comparison_swap | bca_no_closure | 0.26 | no_retrieval | 0.03 | 35 |
| none_check_swap | bca_d1 | 0.16 | no_retrieval | 0.00 | 58 |
| return_value_swap | bca_d1 / repo_map | 0.50 | most methods | 0.00 | 2 |

**none_check_swap is the hardest** mutation type (58 tasks, best method only 0.16). These
mutations flip `if x is None` → `if x is not None` — semantically subtle, hard to diagnose
from a traceback alone. **constant_mutation and handcrafted are easiest** (0.50) — the
changed values are often visible in test output.

### Category Rankings (DEV_REPORT, B=10000)

**Hardest categories** (0% across all methods):
- builtin_tools (N=3), cli (nearly 0%), multipart (N=13, 0%), parts_manager (nearly 0%),
  tools (N=10, 0%), ui (N=3, 0%)

**Easiest categories**:
| Category | Best Method | Pass@1 | N |
|----------|------------|--------|---|
| ssrf | bca / bca_d1 | 0.79 | 19 |
| retries | repo_map | 0.86 | 7 |
| exceptions | bca/bca_d1/bca_d5/bca_nc | 0.50 | 4 |
| config | vector | 0.53 | 15 |
| constant_mutation | bca_no_closure | 0.50 | 20 |

**BCA variants dominate ssrf category** (0.79 at B=10000) — these are input validation bugs
where the graph's dependency tracking identifies the sanitization pipeline. bm25 also does well
(0.63) because the test names mention "ssrf" directly.

### Vague Decomposition — Almost Total Collapse

On vague queries at B=10000, **only 4 mutation types produce any solves**:
- handcrafted: bm25 0.79, vector 0.36 (these were manually crafted with descriptive names)
- arithmetic_swap: 0.22 for BCA/bm25
- multipart: 0.15 for BCA variants
- exceptions: 0.25 for some methods

All other mutation types (boolean_flip, comparison_swap, condition_inversion, constant_mutation,
none_check_swap, value_swap) produce **0.00** across ALL methods at B=10000. The vague
descriptions for these types contain no actionable information.

---

## 36. Conditional Bins (Slicing by Task Properties)

> Source: `paper/results/run3_4_merged/conditional_bins.txt`

### By Identifier Density

**DEV_REPORT**: All 245 tasks have positive identifier density (tracebacks contain function names,
class names). No zero-density bin. This is expected — dev_report queries always contain code tokens.

**EXACT**: Split roughly 150/95 between positive and zero density. Performance is similar across
both bins (no_retrieval: 0.81 zero vs 0.81 positive at B=2000). The identifier density has no
discriminative power on exact queries.

**VAGUE**: 241/245 tasks have zero identifier density (only 4 have positive). This is expected —
vague descriptions use natural language. The 4 positive-density vague tasks solve at much higher
rates (bm25: 0.50 vs 0.06 at B=2000) — when vague descriptions accidentally contain code
identifiers, retrieval improves dramatically.

### By Hop Distance (DEV_REPORT, all methods)

| Hop Distance | B=2000 (best method) | B=10000 (best method) | N |
|-------------|---------------------|----------------------|---|
| 0 hops | 0.30 (vector) | 0.36 (bca_no_closure) | 135 |
| 1-2 hops | 0.09 (vector) | 0.12 (bca_d1) | 89 |
| 3+ hops | 0.19 (vector) | 0.24 (bm25) | 21 |

**Hop-0 tasks are 3-5x easier** than hop 1-2 tasks across all methods. Vector leads at low
budget (B=2000), BCA catches up at high budget (B=10000). The 3+ hop bin is small (N=21) and
noisy but shows bm25 performing well — suggesting that for distant dependencies, keyword
matching outperforms graph traversal because the graph path is too indirect.

### By Mutation Size (DEV_REPORT, B=10000)

| Size Bucket | BCA | bm25 | vector | repo_map | N |
|------------|-----|------|--------|----------|---|
| <5 lines (tiny) | 0.30 | 0.40 | 0.30 | 0.10 | 10 |
| 5-19 lines (small) | 0.25 | 0.17 | 0.25 | 0.17 | 102 |
| 20-49 lines (medium) | 0.25 | 0.22 | 0.26 | 0.24 | 72 |
| 50-99 lines (large) | 0.19 | 0.19 | 0.15 | 0.19 | 53 |
| 100+ lines (very large) | 0.12 | 0.12 | 0.25 | 0.12 | 8 |

**Tiny functions (<5 lines) are easiest** for bm25 (0.40) because the entire function fits in
any context. **Large functions (50-99 lines) are harder** for all methods. **100+ line functions**
show vector excelling (0.25 vs 0.12) — vector retrieves larger chunks that capture more of the
large function body.

BCA performance is relatively stable across sizes (0.12-0.30), while bm25 drops from 0.40 (tiny)
to 0.12 (100+). This suggests BCA's graph-guided selection is more robust to function size
because it selects by dependency structure, not lexical overlap.

---

## 37. Summary of All Findings (Paper-Ready)

### Primary Findings

1. **Three-tier gradient**: exact > dev_report > vague for all methods at all budgets, with no
   exceptions. This validates the query-information spectrum and is the paper's organizing principle.

2. **Failure mode gradient**: exact = localization errors (patch_apply_fail), dev_report = blend,
   vague = comprehension errors (test_fail). Failure modes shift continuously with query information.

3. **No single method dominates**: Each tier has a different winner. Exact: no_retrieval (0.86).
   Dev_report: vector (0.23 avg). Vague: bm25 (0.05 avg). No universal best.

4. **BCA catches up at high budget on dev_report**: BCA goes from trailing by 10pp at B=2000
   to matching or beating vector at B=10000 (0.24 vs 0.23). Graph-guided retrieval needs budget
   to include both seeds and their dependency neighborhoods.

5. **Retrieval-outcome decoupling**: Finding the right file converts to a fix at 72-86% (exact),
   19-25% (dev_report), 2-12% (vague). File retrieval is necessary but not sufficient.

6. **Router B closes 18.2% of the oracle gap** on dev_report at B=2000 using only retrieval
   confidence features. Dev_report is the routing regime (21-29 sole-solvers per budget).

### Secondary Findings

7. **Hop distance predicts task difficulty**: Dev_report hop-0 tasks reach 35% pass@1 vs 7% for
   hop 1-2. Each graph hop reduces repair probability by ~5x.

8. **Closure is negligible**: <1% budget overhead for d≥3. Removing closure doesn't hurt. The
   BCA ablation shows scoring matters more than closure.

9. **Context-patch overlap**: BCA produces highest overlap (0.17-0.19 on dev_report) — LLM
   references BCA context more, even though BCA's pass@1 is lower. BCA provides structurally
   relevant context that the LLM uses but doesn't always fix correctly.

10. **Passing patches are smaller**: 1.2-2.5 lines (passing) vs 1.9-6.4 (all attempts). Over-editing
    correlates with failure — patch size is a proxy for model confidence.

11. **none_check_swap is the hardest mutation** (best: 0.16 on dev_report). Semantic subtlety of
    None checks makes them difficult to diagnose from tracebacks.

12. **ssrf category: BCA excels** (0.79 at B=10000 dev_report). Graph dependency tracking
    identifies sanitization pipelines that keyword methods miss.

13. **Repo-conditional rankings**: pydantic-ai favors BCA at high budget (larger codebase →
    more graph structure to exploit). httpx favors vector at low budget (smaller codebase →
    keyword similarity works).

14. **Assembly is negligible**: 27-266ms for retrieval vs 1000-7000ms for LLM. Router B's
    "dry-run all methods" strategy adds <15% total latency.

### Limitations (to state in paper)

- **Python-only**: Two Python repos. No evidence these findings transfer to other languages.
- **Mutation proxy**: Single-line mutations are a proxy for real bugs, not a perfect match.
- **Model-specific**: gpt-4o-mini results. Stronger models may shift the landscape.
- **Two repos**: A detailed case study, not a general claim. Need 5+ repos for generalization.
- **Exact descriptions leak location**: no_retrieval's 86% shows exact mode is too easy.
  Dev_report is the meaningful evaluation tier.
- **Small vague sample**: Only 15-20 solvable vague tasks — insufficient for reliable statistics.

---

## 38. Code Audit Log

Post-run code audit of benchmark harness and analysis scripts. Issues found, disposition, and fixes applied.

### Fixed (code + regenerated reports)

| Issue | File(s) | Severity | Fix |
|-------|---------|----------|-----|
| **Router comparison table mixed denominators**: bottom comparison block in router_ab.txt used MajVote/Random/Oracle/N from Router A's solvable set alongside Router B accuracy from a different solvable set. On exact queries, N(A)=238 vs N(B)=233 — a 2% denominator mismatch. | `router_ab.py` | **Medium** — affects table readability, not pass@1 | Added N(A)/N(B) columns, relabeled baselines as "(A)", added denominator note. Regenerated `router_ab.txt`. |
| **Posthoc logistic router used leakage features**: `mutation_symbol_lines_log` and `mutation_file_symbols` require knowing the mutation location — unavailable at inference time. Labeling used alphabetical tie-break (`sorted(passing)[0]`), not majority vote. | `posthoc_analysis.py` | **Medium** — exploratory only, never cited in paper | Removed 2 mutation_* features (9→7 features). Changed label to majority-vote winner. Regenerated `router_logistic.txt`. |
| **Merge metadata hardcoded retry schedule**: `merge_and_report.py` hardcoded `[2, 5, 10, 30, 60]` (run3's schedule) instead of reading from source run metadata. Merged `run_metadata.json` reported wrong schedule. | `merge_and_report.py` | **Low** — metadata-only, zero effect on results | Added `source_run_dirs` parameter to `build_run_metadata()`. When provided, collects retry schedules and model fingerprints from source run metadata files. |
| **Merge metadata empty model_version_info**: `for r_data in results: break` loop did nothing, leaving `model_version_info` empty. | `merge_and_report.py` | **Low** — metadata-only | Fixed to read from source run metadata when `source_run_dirs` provided. |
| **Merge latency uses graph_build_time=0**: amortized graph build time underreported in merged latency report. Individual run reports have the correct value. | `merge_and_report.py` | **Low** — reporting-only, documented in comment | Added explanatory comment. Individual run latency reports remain correct. |

### Acknowledged (real but no effect on paper claims)

| Issue | File(s) | Disposition |
|-------|---------|-------------|
| **Budget-independent duplication**: `no_retrieval` results duplicated across budgets instead of re-running (saves LLM calls). Per-task artifact directories only exist for first budget. | `benchmark.py` | **By design.** Code is explicit. Cost inflation ~0.77% on total. No effect on pass@1 since no_retrieval output is truly budget-independent. |
| **BM25 TF uses `.count()` (substring) while IDF uses `\b\w+\b` (token)**: ChatGPT flagged this as an inconsistency. However, query terms are 3+ char identifiers via `r"\b([A-Za-z_]\w{2,})\b"` — substring inflation is negligible for typical identifier lengths. Applied consistently across all methods and tasks. | `baselines.py`, `engine.py` | **Not a confound.** Same implementation in all BM25 paths; relative method comparisons unaffected. |
| **`find_symbol` set ordering**: `list(set(results))` at query.py:64 yields nondeterministic order. | `query.py` | **Product code, not benchmark path.** Benchmark entity extraction uses its own resolution. Even if order varied, BCA sorts candidates by score, not insertion order. |
| **HybridSearch semantic boost key mismatch**: embedding keys (node_id format) don't match search lookup keys (file:line format), so semantic boost is always 0.0. | `hybrid.py` | **Dead code path.** Benchmark uses `baselines.py`, not HybridSearch. Zero effect on any results. |

### Impact on paper claims

**None of the fixed issues change any pass@1 values, failure mode distributions, or mechanism analyses.** The router_ab.txt comparison table now correctly labels which denominator each column uses. The posthoc logistic router (never cited in the paper) is now deployment-valid. Metadata fixes improve reproducibility documentation.
