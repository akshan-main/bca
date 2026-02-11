# CeGraph: Budgeted Context Assembly via Code Knowledge Graphs

## Abstract

AI coding assistants waste tokens. Given a bug report or feature request,
current tools retrieve context by keyword matching (grep, BM25) or by
dumping entire files — neither approach respects the structure of code.
The result: bloated prompts that exceed token budgets, miss critical
dependencies, or include irrelevant symbols.

We present **CeGraph**, an open-source system that builds a knowledge graph
from a codebase and uses it to assemble *dependency-closed, budget-aware*
context for LLM coding tasks. CeGraph's core algorithm, **Budgeted Context
Assembly (BCA)**, formulates context selection as a knapsack-style
optimization problem: given a natural-language task and a token budget,
BCA extracts entity seeds, performs weighted BFS over the code graph, scores
candidates by relevance and centrality, enforces dependency closure
(definitions before usage), and greedily selects symbols within budget using
a value-density heuristic inspired by submodular optimization.

We evaluate on two actively maintained open-source repositories unrelated
to CeGraph: **pydantic-ai** (14.8k stars, 186 source files, 53k lines) and
**httpx** (13k+ stars, ~100 source files, ~15k lines). Our benchmark uses
245 controlled single-line mutations (128 pydantic-ai, 117 httpx) targeting
core subsystems across both repos. Each mutation is validated: it causes a
specific test to fail on the mutated code and pass on the original. We
provide each retrieval method with the bug description and a token budget,
prompt gpt-4o-mini (temperature=0, seed=42, single sample) to generate a
SEARCH/REPLACE edit, and measure pass@1 against the project's test suite.
We test both *dev-localized* descriptions (referencing file and line) and
*vague* descriptions (user-reported symptoms only) to evaluate robustness
to query quality.

We compare 12 retrieval methods (no-retrieval, random, grep, BM25, TF-IDF
vector, embedding, repo-map, BCA at depth 1/3/5, BCA-without-closure,
BCA-without-scoring) plus a target-file ceiling across 4 token budgets
(1k–10k). Key findings:

1. **Method performance is conditional on query type.** BCA excels on tasks
   requiring structural understanding (call chains, dependency closure),
   BM25 wins on surface-level keyword tasks, and grep degrades sharply
   when the bug description is vague rather than exact.

2. **A simple method router** (majority-vote, leave-one-out cross-validation)
   that selects the best-performing method per (budget, query-type) combination
   approaches the oracle upper bound (best method per-task) without extra LLM
   calls. A post-hoc logistic regression router using retrieval confidence
   features (softmax entropy, effective candidates, budget utilization) is
   evaluated as a feature-conditioned upgrade.

3. **Dependency closure matters.** BCA with closure outperforms
   BCA-without-closure at tight budgets (1k–4k tokens), confirming that
   structural completeness is worth the token cost.

These results are from two repositories and should be interpreted as a
detailed empirical study rather than a general claim. We release all code,
mutations, and benchmark harness to enable replication and extension to
additional codebases.

CeGraph is pip-installable, requires no API keys for indexing, and integrates
as an MCP server with Claude Code, Cursor, and other AI coding tools.

**Keywords:** code retrieval, context assembly, knowledge graphs, LLM coding
assistants, budgeted optimization, MCP
