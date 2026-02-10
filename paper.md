# CeGraph Paper Notes & Findings

Running notes for the arXiv preprint. Everything interesting goes here so nothing gets lost.

---

## 1. Paper Framing & Strategy

### Title
**CeGraph: Budgeted Context Assembly via Code Knowledge Graphs**

### Core Thesis
There is a "missing middle layer" between naive retrieval (grep/BM25) and whole-file dumping.
BCA formulates context selection as a knapsack-style optimization: entity extraction -> seed
resolution -> weighted BFS -> relevance scoring -> dependency closure -> greedy budget packing.

### Unique Angle: The Router Story
No single retrieval method dominates. The paper's publishable contribution is showing *when and why*
each method wins, backed by causal-ish evidence from logged metrics. A lightweight query classifier
(logistic regression on graph features) routes to the best method per-task and approaches the oracle.

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
- **Exact**: Developer-level descriptions mentioning file names, line numbers, operators
  - Example: "In _client.py:172, the comparison operator was changed: '>' became '>='"
- **Vague**: User-reported symptom descriptions with no internal references
  - Example: "A threshold or limit check seems to trigger at the wrong value"

---

## 3. Methods (11 total)

| Method | Type | Description |
|--------|------|-------------|
| no_retrieval | Baseline | No context, just task description |
| naive_random | Baseline | Random symbols up to budget |
| grep | Lexical | Regex search on keywords from description |
| bm25 | Lexical | BM25 symbol retrieval + greedy packing |
| vector | Semantic | TF-IDF vector similarity search |
| embedding | Semantic | (if available) embedding-based search |
| repo_map | Structural | Signature-only overview of entire codebase |
| bca | **Ours** | Full BCA: entity extraction -> BFS -> scoring -> closure -> budget |
| bca_no_closure | Ablation | BCA without dependency closure |
| bca_no_scoring | Ablation | BCA with scoring disabled (graph traversal + closure only) |
| target_file | Ceiling | Oracle: gives the LLM the entire mutated file (upper bound) |

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

### Post-Hoc Metrics (no rerun needed)
- Cost per solved task (total tokens / passes)
- Context redundancy/entropy (from stored context text)
- Logistic regression router (train on logged features)
- k-hop dependency coverage: `coverage_k = |D_k(m) intersection C| / |D_k(m)|`

---

## 5. Key Findings (from sanity runs)

### Finding 1: Entity Density Perfectly Separates Exact vs Vague
Tested on real tasks:
- **Exact description**: entity_count=3, mapped=2, identifier_density=0.13
- **Vague description**: entity_count=0, mapped=0, identifier_density=0.0

This is the backbone of the conditional analysis. BCA's entity extraction finds 0 entities in
vague descriptions, meaning it falls back to keyword-only seeds (low quality). Lexical methods
don't depend on entity extraction so they degrade more gracefully.

### Finding 2: BCA Achieves 100% Target Symbol Hit Rate
From httpx_sanity2 retrieval metrics:
- **BCA**: 100% target file hit, 100% target symbol hit (both budgets)
- **grep**: 0-50% target file hit, 50% target symbol hit
- **naive_random**: 0-50% file hit, 0-50% symbol hit
- **no_retrieval**: 0% everything

BCA's graph traversal reliably locates the mutation site when entities resolve.

### Finding 3: BCA Needs Budget to Shine
From sanity_mini (pydantic-ai, 3 tasks):
- **BCA @ 1k**: 67% pass@1
- **BCA @ 4k**: 100% pass@1 (statistically significant vs no_retrieval, CI=[+1.0, +1.0])
- **no_retrieval**: 0% at all budgets

At tight budgets, BCA includes the right symbols but may not have room for enough context.
At 4k+, BCA has enough room for seeds + closure + supporting code.

### Finding 4: Graph Distance Is Meaningful
From entity-resolution tests:
- Seed symbol = mutation symbol: min_hops=0 (exact entity match, e.g. `UsageBase.total_tokens`)
- Seed in same file, different symbol: min_hops=1-2
- Seeds in different files: min_hops=3+

**Prediction**: Tasks with min_hops=0 should have highest pass@1 for BCA.
Tasks with min_hops >= 3 are where BCA struggles and lexical methods may win.

### Finding 5: Closure Adds Minimal Overhead (So Far)
From BCA sanity run:
- `bca_closure_added_symbols`: 1
- `bca_closure_added_tokens`: 9
- `bca_frontier_visited`: 2172

Closure is cheap when the seed already points to the right place. The interesting case
is when closure pulls in distant dependencies — we need the full benchmark to see this.

### Finding 6: grep Can Fail Hard
From quick_test (agent-library, 2 tasks):
- **grep**: 0% pass@1 at both budgets
- **BCA & BM25**: 100% at both budgets

grep's failure mode: the LLM's patches reference wrong file paths (e.g. `cegraph/graph/query.py`
instead of `src/cegraph/graph/query.py`), causing "file not found" errors. This is a real issue
with keyword-based retrieval — it gives code snippets without structural awareness.

### Finding 7: no_retrieval Sometimes Works (httpx)
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

**Key insight**: `patch_apply_fail` is grep's dominant failure mode. BCA avoids this because
it includes structural metadata (file paths from the graph) in the context.

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

1. **Table 1**: Pass@1 by method x budget x query_type (main result)
2. **Table 2**: Pass@1 with 95% bootstrap CIs
3. **Table 3**: Ablation (BCA vs no-closure vs no-scoring)
4. **Table 4**: Failure mode breakdown by method
5. **Table 5**: Retrieval quality metrics (file hit, symbol hit, overlap)
6. **Table 6**: Router vs oracle vs best single method
7. **Table 7**: Conditional analysis — pass@1 stratified by entity density or graph distance

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
7. **Exact descriptions are too easy** — LLM can sometimes fix without any context when the
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

| Metric | pydantic-ai | httpx |
|--------|------------|-------|
| Source files | 186 | ~100 |
| Candidates generated | 1,454 | 388 |
| Killed | 199 (13.7%) | 210 (54.1%) |
| Kill rate | 20.5% | 54.1% |
| Selected for eval | 174 | 152 |

httpx's higher kill rate suggests tighter test coverage. pydantic-ai has many mutations in
untested code paths (33.4% skipped = no matching test found, 52.9% survived = test didn't catch it).

**Paper point**: Kill rate itself is an interesting codebase health metric.

---

## 16. Budget Configuration

| Budget | Description | Typical Content |
|--------|-------------|----------------|
| 1k tokens | Tight — ~40 lines of code | 1-2 functions, barely enough |
| 4k tokens | Moderate — ~160 lines | A class + its dependencies |
| 8k tokens | Generous — ~320 lines | Multiple files, good coverage |
| 10k tokens | Liberal — ~400 lines | Most relevant code + context |

BCA's budget utilization is consistently 94-100% across all budgets (greedy packing works).
Lexical methods also hit 100% (they just pack differently).
no_retrieval uses 0-2% (just the prompt, no context).
