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

## 21. Full Run3 Results (N=245 tasks, 19,600 attempts)

> **Run completed.** 245 tasks × 10 methods × 4 budgets × 2 query types = 19,600 total attempts.
> All results stored in `paper/results/run3/results.json`. Reports regenerated from per-task artifacts.
> Reports: 17 standard + 8 post-hoc analysis files + 8 figures.

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

### Two-Regime Finding

This is the paper's central empirical finding:

1. **Exact queries trivialize the task**: no_retrieval achieves 86% — the LLM can fix bugs from
   the description alone because exact queries leak file name + line number + operator diff.
   Retrieval *hurts* because it adds noise (BCA at 52-69% vs no_retrieval at 86%).

2. **Vague queries break the LLM**: All methods score 0-7%. The **ceiling** (target_file) is only
   24-30% — even giving the LLM the entire correct file, it can't fix the bug from a symptom-only
   description. This means the bottleneck is the *model*, not the *retrieval*.

3. **Oracle gap**: Exact oracle is 0.97 (near-perfect with the right method), vague oracle is only
   0.07 (no method combination can save you). This quantifies the ceiling on routing benefit.

**Methodological insight**: Most code repair benchmarks (SWE-bench, etc.) have queries somewhere
in the middle but never explicitly measure the spectrum. Our dual-query design exposes this blind
spot: results that look great on dev-localized queries may be meaningless for realistic bug reports.

### Ceiling Probe (target_file)

| Query Type | B=2000 | B=4000 | B=8000 | B=10000 |
|------------|--------|--------|--------|---------|
| Exact | 0.82 | 0.91 | 0.92 | 0.92 |
| Vague | 0.24 | 0.29 | 0.29 | 0.30 |

Median file is ~3118 tokens, so performance saturates at B=4000 for exact. Vague ceiling
barely improves with budget — confirming the model is the bottleneck, not the context quantity.

---

## 22. Retrieval-Outcome Decoupling

> Source: `paper/results/run3/retrieval_outcome.txt`

**Key finding**: Finding the right file does NOT guarantee a correct fix.

**File Hit → Pass Conversion Rate (EXACT, B=10000):**
| Method | Conversion |
|--------|-----------|
| vector | 86% |
| bm25 | 79% |
| bca_d1 | 79% |
| repo_map | 77% |
| bca_no_closure | 76% |
| bca | 72% |

**File Hit → Pass Conversion Rate (VAGUE, B=10000):**
| Method | Conversion |
|--------|-----------|
| bm25 | 12% |
| vector | 8% |
| repo_map | 3% |
| bca | 3% |

**The vague decoupling is dramatic**: repo_map achieves 96% target file hit rate on vague queries
but only 3% pass@1 — a 96→3% conversion. Having the file is necessary but grossly insufficient
when the query doesn't tell the LLM what to fix.

**Hallucination passes**: no_retrieval achieves 211/211 exact passes *without any file in context*.
The LLM fixes bugs purely from the description + memorized training data. This is the strongest
evidence that exact descriptions are "too easy" and not a valid test of retrieval quality.

---

## 23. Cost Analysis

> Source: `paper/results/run3/cost_analysis.txt`

**Total benchmark cost**: $15.05 (exact + vague) for 17,640 non-ceiling attempts.
This is remarkably cheap — gpt-4o-mini at $0.15/M input, $0.60/M output.

**Cost per attempt**: $0.0004-$0.0016 depending on budget (no_retrieval: $0.00006).

**Cost per solved task (EXACT, B=10000):**
| Method | Cost/Solve | Total Solves |
|--------|-----------|-------------|
| no_retrieval | $0.00006 | 844 |
| bm25 | $0.00149 | 710 |
| vector | $0.00150 | 745 |
| bca_no_closure | $0.00153 | 641 |
| bca | $0.00153 | 625 |

**Cost per solved task (VAGUE, B=10000):**
| Method | Cost/Solve | Total Solves |
|--------|-----------|-------------|
| bm25 | $0.00168 | 52 |
| repo_map | $0.00161 | 31 |
| vector | $0.00157 | 29 |
| bca_d1 | $0.00140 | 23 |

**Paper point**: Cost differences between retrieval methods are negligible (~10%). The dominant
cost factor is budget size, not method choice. The conference upgrade to gpt-4o will be ~20x
more expensive per attempt but may unlock the vague regime.

---

## 24. Seed-to-Mutation Distance (k-Hop Analysis)

> Source: `paper/results/run3/khop_coverage.txt`

### Hop Distribution

**EXACT queries (BCA methods):**
| Bucket | Count | % |
|--------|-------|---|
| 0 (exact match) | 960 | 98.0% |
| 1-2 (near) | 16 | 1.6% |
| unreachable | 4 | 0.4% |

**VAGUE queries (BCA methods):**
| Bucket | Count | % |
|--------|-------|---|
| unreachable | 876 | **89.4%** |
| 1-2 (near) | 64 | 6.5% |
| 3-5 (mid) | 28 | 2.9% |
| 0 (exact match) | 12 | 1.2% |

**This explains why BCA struggles on vague queries**: 89.4% of vague tasks have no reachable path
from seeds to the mutation. Entity extraction finds 0 entities → 0 seeds → no graph traversal.
BCA degrades to pure keyword-based retrieval scoring, losing its structural advantage.

### Closure Budget Consumption

| Method | Mean Syms Added | Mean Tokens Added | % Budget | Frontier |
|--------|----------------|-------------------|----------|----------|
| bca_d1 | 3.2 | 745 | **10.0%** | 35 |
| bca | 1.1 | 38 | 0.6% | 1296 |
| bca_d5 | 1.2 | 31 | 0.5% | 3001 |
| bca_no_closure | 0 | 0 | 0% | 1296 |
| bca_no_scoring | 0.7 | 24 | 0.4% | 1296 |

**Surprising**: bca_d1 (shallow depth) has the *highest* closure overhead at 10% of budget. This
is because d=1 visits fewer frontier nodes (35 vs 1296) but the closure adds more per-symbol
because the selected symbols are fewer and closer, triggering more dependency chains.

For d=3 and d=5, closure is negligible (<1% of budget) — the expansion already includes enough
context that closure adds little. **Ablation result**: removing closure (bca_no_closure) doesn't
hurt performance, consistent with closure being a tiny fraction of the budget.

---

## 25. Context Efficiency

> Source: `paper/results/run3/context_redundancy.txt`

### Symbols per 1000 Tokens (Higher = More Symbols Packed)

**EXACT, B=10000:**
| Method | Sym/1000tok | Interpretation |
|--------|------------|----------------|
| bca_d5 | 15.3 | Most symbols packed — many small snippets |
| bca_no_closure | 14.8 | Similar to d5 without closure |
| bca_no_scoring | 12.0 | Moderate packing |
| bca | 11.9 | Default BCA |
| bm25 | 9.4 | Fewer, larger chunks |
| bca_d1 | 5.5 | Fewest symbols — large contiguous blocks |
| vector | 5.0 | Large chunks per symbol |
| repo_map | 2.0 | Signatures only — very sparse |

**Paper point**: Higher symbol density ≠ better performance. bca_d5 packs the most symbols but
doesn't outperform bm25 which packs fewer but more relevant ones. repo_map has the lowest density
(signatures only) but strong performance because signatures are highly information-dense.

### Tokens per Symbol (Granularity)

| Method | Tok/Sym (exact) | Tok/Sym (vague) |
|--------|----------------|-----------------|
| repo_map | 711 | 862 |
| vector | 251 | 363 |
| bca_d1 | 206 | 222 |
| bm25 | 124 | 142 |
| bca | 88 | 116 |
| bca_no_scoring | 87 | 92 |
| bca_d5 | 69 | 99 |

BCA at higher depths selects finer-grained context (smaller symbols). repo_map and vector select
coarser chunks. The optimal granularity depends on the task — this is another axis for the router.

---

## 26. Method Uniqueness (Sole-Solver Analysis)

> Source: `paper/results/run3/method_uniqueness.txt`

**EXACT queries**: Nearly all solvable tasks are multi-solver (230-235 out of 238-239). Only
4-10 tasks per budget have a sole solver, and those are almost all no_retrieval. This means
the router has very little to work with — most tasks are solved by all methods or none.

**VAGUE queries**: More differentiation. At B=2000:
- 20/245 solvable, 10 multi-solver, **10 sole-solver**
- bm25: 7 unique solves
- repo_map: 2 unique solves
- bca_no_scoring: 1 unique solve

**Paper point**: The router's value proposition is clearest on vague queries where methods
have non-overlapping strengths. On exact queries, methods are too correlated (same tasks solved)
for routing to help. This is consistent with the logistic router showing 0% gap closure on exact.

---

## 27. Per-Repository Analysis

> Source: `paper/results/run3/per_repo_rankings.txt`

**EXACT, B=10000:**
| Method | httpx | pydantic-ai | Overall |
|--------|-------|------------|---------|
| no_retrieval | 0.91 | 0.81 | 0.86 |
| bm25 | 0.80 | 0.72 | 0.76 |
| vector | 0.80 | 0.78 | 0.79 |
| repo_map | 0.85 | 0.66 | 0.75 |
| bca_no_closure | 0.74 | 0.71 | 0.72 |
| bca | 0.71 | 0.68 | 0.69 |

**Key observation**: httpx is consistently easier than pydantic-ai across all methods. This is
expected — httpx is smaller (~100 files vs 186), more focused, and has higher kill rate (54% vs
14%). repo_map benefits most from httpx (0.85 vs 0.66) because the entire httpx codebase fits
more easily into signature-based summaries.

**VAGUE queries**: Both repos are near-zero. pydantic-ai has slightly higher vague pass@1 for
bm25 (0.08-0.09 vs 0.02-0.05) — possibly because pydantic-ai's vague descriptions are slightly
more keyword-rich (test names are more descriptive).

---

## 28. Failure Mode Taxonomy (Full Run)

> Source: `paper/results/run3/failure_diagnosis.txt`

### EXACT Queries (B=10000, N=245)
| Method | pass | patch_apply_fail | test_fail | syntax_error |
|--------|------|-----------------|-----------|-------------|
| no_retrieval | **86%** | 11% | 2% | 0% |
| vector | 79% | 18% | 2% | 1% |
| bm25 | 76% | 19% | 5% | 0% |
| repo_map | 75% | 20% | 5% | 0% |
| bca_no_closure | 72% | 22% | 6% | 0% |
| bca | 69% | 22% | 9% | 0% |

### VAGUE Queries (B=10000, N=245)
| Method | pass | patch_apply_fail | test_fail | syntax_error |
|--------|------|-----------------|-----------|-------------|
| bm25 | 5% | 20% | 70% | 4% |
| repo_map | 3% | 19% | 76% | 2% |
| vector | 3% | 25% | 71% | 1% |
| bca | 2% | 13% | 82% | 2% |
| no_retrieval | 0% | **100%** | 0% | 0% |

**The failure mode shift is the story**:
- **EXACT**: Dominant failure is `patch_apply_fail` (11-22%) — the LLM generates a fix but
  references the wrong location. Actual `test_fail` is rare (2-9%).
- **VAGUE**: Dominant failure is `test_fail` (67-85%) — the LLM applies a patch but it's the
  *wrong fix*. The model is guessing, and guessing wrong.
- **no_retrieval on vague** = 100% `patch_apply_fail` — without context, the LLM doesn't even
  know which file to edit.

This taxonomy reveals that exact and vague failures are fundamentally different problems:
exact failures are *localization* errors, vague failures are *comprehension* errors.

---

## 29. Router Analysis

> Source: `paper/results/run3/router_ab.txt`, `router_analysis.txt`
> Script: `python -m paper.experiments.router_ab --run-dir paper/results/run3`

### Router Definitions

Two deployment-valid router designs, following the feature leakage framework:

**Router A ("Choose-first", pre-retrieval)**:
- Decision based ONLY on features available before running any retrieval method
- Features (3): `entity_count_mapped`, `query_identifier_density`, `graph_node_count_log`
- NOT allowed: any `retrieval_*` metric, `tokens_used`, `assembly_time_ms`, `mutation_*`
- Candidates: all 9 non-ceiling methods (including `no_retrieval`)
- Use case: zero-cost routing decision at query time

**Router B ("Dry-run then choose", retrieval-conditioned)**:
- Runs all retrieval methods locally (no LLM call), examines their confidence metrics, picks the best, then calls LLM once
- Features (27): Router A features + per-method `{top1_score, budget_utilization, softmax_entropy}` for each of 8 retrieval methods
- Still deployment-valid: retrieval is local and cheap, decision is pre-LLM
- Candidates: 8 retrieval methods (excludes `no_retrieval` — no retrieval metrics to compare)
- Use case: confident method selection when retrieval cost is negligible vs LLM cost

**Feature leakage rules** (features NEVER used in routing):
- `target_file_hit`, `target_symbol_hit` — requires knowing mutation location
- `min_hops_seed_to_mutation`, `median_hops_*` — requires knowing mutation
- `mutation_symbol_lines`, `mutation_file_symbols` — router doesn't know mutation site
- `tests_passed`, `failure_mode`, `patch`, `test_output` — post-LLM outcomes
- These are for **explanatory analysis** only, not routing.

### Implementation

Both routers use LOO-CV logistic regression (sklearn) with StandardScaler.
Training label strategy: "smart" — uses majority-vote method as label when it passes,
switches to rarest passer only for tasks where majority fails (teaches model: default to
best single, deviate only when features warrant it).
Evaluation: success = predicted method is in the task's passing set (any correct pick counts).

### Results: Comparison Table

```
[EXACT queries]
Budget   MajVote  Random  Router A  Router B  Oracle    N
------   -------  ------  --------  --------  ------  ---
B=2000    88.7%   64.0%    88.2%     78.1%   100.0%  238
B=4000    88.3%   69.7%    88.3%     78.8%   100.0%  239
B=8000    88.7%   74.1%    88.7%     79.9%   100.0%  238
B=10000   88.3%   75.3%    88.3%     82.6%   100.0%  239

[VAGUE queries]
Budget   MajVote  Random  Router A  Router B  Oracle    N
------   -------  ------  --------  --------  ------  ---
B=2000    80.0%   36.7%    75.0%     75.0%   100.0%   20
B=4000    70.6%   29.4%    64.7%     70.6%   100.0%   17
B=8000    68.8%   31.2%    68.8%     68.8%   100.0%   16
B=10000   86.7%   34.8%    80.0%     73.3%   100.0%   15
```

### Interpretation

**Router A (EXACT)**: Converges to majority vote (no_retrieval at 88.7%). The model learns
that no_retrieval dominates exact queries and there are only 4-10 sole-solver tasks per budget
that 3 features cannot reliably identify. **Gap closed: ~0%.**

**Router A (VAGUE)**: Slightly below majority vote. With only 15-20 solvable tasks, there is
insufficient data for the model to learn conditional patterns. bm25 dominates.

**Router B (EXACT)**: Lower than Router A because Router B excludes no_retrieval (the
dominant exact method). Among retrieval methods only, vector leads (78-83%). Router matches
this majority vote. Gap closed: 0%.

**Router B (VAGUE)**: Matches or slightly trails majority vote (bm25). One positive signal:
`safest` strategy at B=8000 achieves 75% vs 68.8% majority (+20% gap closure) — but with
only 16 tasks this is noise, not a reliable finding.

### Why Routing Fails Here (honest assessment)

The two-regime structure of this benchmark eliminates the conditions routing needs:

1. **EXACT**: no_retrieval solves 86% of tasks. The 11% oracle gap is spread across tasks
   where no_retrieval fails, but these tasks have no distinguishing features detectable
   from query metadata alone.

2. **VAGUE**: Only 6-8% of tasks are solvable by ANY method. With 15-20 solvable tasks,
   the sample is too small for a logistic regression to learn reliable conditional patterns.

3. **Method overlap**: 230/238 solvable exact tasks are multi-solver (any method works).
   Only 4-10 are sole-solver. Routing can only improve on sole-solver tasks, but they're
   too rare to learn patterns from.

**The router story is NOT that routing fails in general** — it's that the two-regime gap
leaves no room for conditional routing. A middle query tier (dev-report with failing
test + traceback, no line numbers) would create the regime where methods actually diverge
and routing adds value. This is the primary conference upgrade path.

### Top Features (for paper discussion)

Across all conditions, the most informative features are:
- `entity_count_mapped` — how many query entities resolve to graph symbols (strongest Router A signal)
- `graph_node_count_log` — proxy for repo complexity
- `query_identifier_density` — fraction of query tokens that look like code identifiers
- Router B: `bm25_top1_score`, `bm25_entropy`, `vector_top1_score` — retrieval confidence signals

These are sensible and would likely become predictive in a regime where methods actually diverge.

---

## 30. Figures Inventory

All figures saved in `paper/results/run3/figures/`. Generated by
`python -m paper.experiments.generate_figures --run-dir paper/results/run3`.

### Figure 1: Pass@1 vs Token Budget (`fig1_pass_vs_budget.png`)
Line chart with one line per method, two panels (EXACT, VAGUE). Shows:
- EXACT: no_retrieval flat at 0.86, all methods converge upward with budget
- VAGUE: all methods near zero, barely visible differentiation

### Figure 2: Failure Mode Breakdown (`fig2_failure_modes.png`)
Stacked bar chart at B=10000. Shows the failure mode shift:
- EXACT: mostly green (pass) + orange (patch_apply_fail)
- VAGUE: mostly red (test_fail) + no green

### Figure 3: Retrieval vs Outcome (`fig3_retrieval_vs_outcome.png`)
Scatter plot of target file hit rate vs pass@1 per method×budget. Shows:
- EXACT: positive correlation (more retrieval → more passes)
- VAGUE: cluster near origin (high retrieval → no passes), with repo_map outlier at high hit rate

### Figure 4: Per-Repository Comparison (`fig4_per_repo.png`)
Grouped bar chart at B=10000. Shows httpx consistently easier than pydantic-ai.

### Figure 5: Ceiling Probe (`fig5_ceiling_probe.png`)
Bar chart: best single method vs oracle vs target_file ceiling. Shows:
- EXACT: methods nearly reach ceiling (86% vs 92%)
- VAGUE: huge gap between methods (5%) and ceiling (30%)

### Figure 6: BCA Ablation (`fig6_ablation.png`)
Line chart of BCA variants only. Shows depth and component ablation effects.

### Figure 7: Mutation Type Heatmap (`fig7_mutation_heatmap.png`)
Heatmap: method × mutation_type at B=10000. Shows which mutation types are easier/harder
for each method.

### Figure 8: Entity Density Effect (`fig8_entity_density.png`)
Grouped bar chart: exact vs vague side-by-side for each method. Visualizes the two-regime gap.

---

## 31. Post-Hoc Report File Inventory

Standard reports (regenerated by `merge_and_report.py`):
- `summary.txt` — Pass@1 tables
- `summary_with_ci.txt` — Pass@1 with 95% bootstrap CIs
- `per_repo_results.txt` — Pass@1 by repository
- `ceiling_probe.txt` — target_file ceiling analysis
- `router_analysis.txt` — Oracle vs LOO-CV Router vs Best Single
- `decomposition.txt` — Pass@1 by mutation type and category
- `conditional_bins.txt` — Pass@1 by identifier density, hops, mutation size
- `bootstrap_analysis.txt` — Paired bootstrap CI details
- `bootstrap_cis.json` — Raw bootstrap CIs (JSON)
- `failure_diagnosis.txt` — Failure mode breakdown
- `retrieval_metrics.txt` — Target file/symbol hit rates, budget utilization
- `patch_quality.txt` — Patch size and locality
- `latency_cost.txt` — Assembly, LLM, and test time breakdown
- `edit_locality.txt` — Edit distance and context-patch overlap

Post-hoc reports (generated by `posthoc_analysis.py`):
- `cost_analysis.txt` — Token usage and cost per solved task
- `router_logistic.txt` — Logistic regression router results (superseded by router_ab.txt)

Router A/B report (generated by `router_ab.py`):
- `router_ab.txt` — Router A (pre-retrieval) + Router B (dry-run) + comparison table + leakage audit
- `retrieval_outcome.txt` — Retrieval-outcome decoupling analysis
- `context_redundancy.txt` — Context efficiency (symbols/tokens, granularity, entropy)
- `khop_coverage.txt` — Seed-to-mutation distance and closure analysis
- `method_uniqueness.txt` — Sole-solver analysis
- `per_repo_rankings.txt` — Per-repository method rankings
- `posthoc_summary.txt` — Combined summary of all post-hoc analyses

Figures (generated by `generate_figures.py`):
- `figures/fig1_pass_vs_budget.png`
- `figures/fig2_failure_modes.png`
- `figures/fig3_retrieval_vs_outcome.png`
- `figures/fig4_per_repo.png`
- `figures/fig5_ceiling_probe.png`
- `figures/fig6_ablation.png`
- `figures/fig7_mutation_heatmap.png`
- `figures/fig8_entity_density.png`

---

## 32. Dev-Report Tier Readiness

### Infrastructure Audit (completed)

All reporting scripts now dynamically extract query types from data — no hardcoded `["exact", "vague"]`
loops remain. When run4 (dev_report) results land, every report and figure will include the third tier
automatically.

| Script | Status | What was fixed |
|--------|--------|----------------|
| benchmark.py | OK (already dynamic) | 2 stale comments updated |
| merge_and_report.py | OK (already dynamic) | All 13 report functions use `sorted(set(r["query_type"]))` |
| posthoc_analysis.py | Fixed | 8 functions: cost, router, decoupling, efficiency, khop, uniqueness, per-repo |
| router_ab.py | Fixed | 3 functions: router_a, router_b, comparison_table |
| generate_figures.py | Fixed | All 8 figures: dynamic N-panel subplots, `QT_LABELS` dict for display names |

### Failure Mode Shift — Key Mechanistic Finding

The failure mode shift across query tiers is one of the strongest mechanistic findings:

- **Exact** (dev-localized): Dominant failure = `patch_apply_fail` (11-22%). LLM knows *what* to fix
  but gets the *location* wrong. These are **localization errors**.
- **Vague** (symptom-only): Dominant failure = `test_fail` (67-85%). LLM applies a patch but it's the
  *wrong fix*. These are **comprehension errors**.
- **no_retrieval + vague** = 100% `patch_apply_fail` — without context, LLM doesn't even know
  which file to edit.

**Dev-report prediction**: Should show a *blend* of both failure modes. If the traceback gives enough
signal to find the right file but not enough to understand the fix, we'd expect more `test_fail` than
exact but fewer than vague. If this gradient appears, it confirms the three-tier spectrum is real and
that failure modes shift continuously with query information content.

This gradient is a publishable finding on its own — it demonstrates that context assembly quality
manifests differently depending on query tier, and that benchmarks using only one query type miss
half the failure story.

### Reproducibility

Full reproducibility controls in place:
- Pinned commits: pydantic-ai `69a578a`, httpx `ae1b9f6`
- Pinned model: `gpt-4o-mini-2024-07-18` (exact version, not alias)
- `temperature=0`, `seed=42`, `top_p=1`, `presence_penalty=0`, `frequency_penalty=0`
- Deterministic retry backoff: 5s, 8s, 15s, 30s, 60s (no jitter)
- Byte-identical git revert between tasks
- `system_fingerprint` logged per API response (detects GPU cluster changes)
- Bootstrap CIs (n=10000, seed=42) for statistical inference
- Caveat: OpenAI states `seed` provides "mostly deterministic" output — GPU floating-point
  non-determinism means exact bit-for-bit reproduction across clusters is not guaranteed.
  Bootstrap CIs account for this variance statistically.
