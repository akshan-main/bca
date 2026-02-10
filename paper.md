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
3. **Router framing**: Quantify oracle gap and show a lightweight router (logistic regression on
   graph features) closes meaningful gap without extra LLM calls

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
| pydantic-ai | 14.8k | 186 | 53k | pydantic_ai_slim/pydantic_ai | 20.5% (199/1454) | 174 (14 handcrafted + 160 discovered) |
| httpx | 13k+ | ~100 | ~15k | httpx | 54.1% (210/388) | 152 (all discovered) |

**Total: 326 tasks across 2 repos**

pydantic-ai has lower kill rate because it's a larger codebase with more complex test coupling.
httpx is a focused HTTP library with tighter test coverage.

### 60-Task Curated Set (pydantic-ai)
For dev/tuning: 14 handcrafted + 46 discovered = 60 tasks, seed=42, diverse selection with
caps (max 15/file, 25/category, 30/mutation_type).

### Mutation Type Distribution

**pydantic-ai (174 tasks):**
- none_check_swap: ~45, membership_swap: ~43, comparison_swap: ~36
- boolean_flip: ~27, condition_inversion: ~23, constant_mutation: ~12
- arithmetic_swap: ~6, value_swap: ~7

**httpx (152 tasks):**
- membership_swap: 30, none_check_swap: 28, comparison_swap: 26
- condition_inversion: 21, boolean_flip: 16, value_swap: 16
- constant_mutation: 9, arithmetic_swap: 4, return_value_swap: 2

### Category Coverage

**pydantic-ai**: 20 categories — ssrf(20), usage(16), messages(15), utils(14), json_schema(13), models(12), agent(10), ...

**httpx**: 12 categories — transports(25), config(15), content(15), multipart(15), utils(15), cli(14), decoders(14), auth(11), models(10), url_parsing(9), client(8), exceptions(1)

### Dual Query Modes
- **Dev-localized** (exact): File name + line number + operator diff. This is ceiling-ish —
  the description effectively leaks the bug location. Label as such in paper. Reviewers WILL
  note that no_retrieval can solve these.
  - Example: "In _client.py:172, the comparison operator was changed: '>' became '>='"
- **Vague** (user-reported): Symptom-only, no code identifiers. This is the realistic query mode.
  - Example: "A threshold or limit check seems to trigger at the wrong value"

> **Note**: Consider adding a third "Dev-report" mode later (mentions function/class but no
> line numbers) to fill the gap between god-mode and zero-info.

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

### Training
- Leave-one-out cross-validation over tasks (prevents leakage)
- Logistic regression (interpretable, publishable coefficients)
- Labels: which method passed for each task/budget/query_type combination

### Expected Story
Router recovers X% of the oracle gap (best single method vs oracle upper bound).
At minimum, router should beat any single method's average performance.

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
- [ ] Release discover_mutations.py and make_*_tasks.py scripts
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
| Selected for eval | 174 | 152 |

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

**pydantic-ai (174 tasks):**
| Size bucket | Count | % |
|-------------|-------|---|
| <5 lines | 9 | 5% |
| 5-20 lines | 77 | 44% |
| 20-50 lines | 39 | 22% |
| 50-100 lines | 41 | 24% |
| 100+ lines | 8 | 5% |
| **Median: 20 lines, Mean: 82.5, Range: 1-1406** |||

**httpx (152 tasks):**
| Size bucket | Count | % |
|-------------|-------|---|
| <5 lines | 8 | 5% |
| 5-20 lines | 56 | 37% |
| 20-50 lines | 58 | 38% |
| 50-100 lines | 20 | 13% |
| 100+ lines | 10 | 7% |
| **Median: 27 lines, Mean: 37.8, Range: 2-241** |||

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
3. **Benchmark**: Two repos, controlled fault injection (mutation testing), dual query modes
   (dev-localized + vague), pinned snapshots, strict byte-identical revert.
4. **Results**: No single method dominates. Show main table. This is the thesis, not a failure.
5. **Mechanism**: Explain *why*. Use failure modes + graph hops + entity density + closure overhead.
   This is the section that makes reviewers take the paper seriously.
6. **Router**: Oracle gap exists. Simple logistic regression router closes X% of the gap without
   extra LLM calls. Show router vs best-single-method vs oracle.
7. **Limits**: Python-only, mutation testing is a proxy for real bugs, dev-localized descriptions
   may be overly informative (inflates no_retrieval), two repos only (detailed case study, not
   a general claim).
