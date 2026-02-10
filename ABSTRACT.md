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

We evaluate on **pydantic-ai** (14.8k stars, 186 source files, 53k lines),
an actively maintained open-source project unrelated to CeGraph. Our
benchmark uses 14 controlled single-line mutations targeting core subsystems
(usage limits, SSRF protection, exception handling, settings merge). Each
mutation is validated: it causes a specific test to fail on the mutated code
and pass on the original. We provide each retrieval method with the bug
description and a token budget, prompt GPT-4o (temperature=0, single sample)
to generate a unified diff, and measure pass@1 against the project's test
suite. We test both *exact* descriptions (developer-level, referencing
internals) and *vague* descriptions (user-reported symptoms only) to evaluate
robustness to query quality.

We compare 7 retrieval methods (grep, BM25, TF-IDF vector, embedding,
repo-map, BCA, BCA-without-closure) across 4 token budgets (1k–10k). On
pydantic-ai, we find:

1. **Method performance is conditional on query type.** BCA excels on tasks
   requiring structural understanding (call chains, dependency closure),
   BM25 wins on surface-level keyword tasks, and grep degrades sharply
   when the bug description is vague rather than exact.

2. **A lightweight query classifier** that routes to the best method
   per-query — using only features derived from the knowledge graph (symbol
   hit ratio, IDF-weighted scores, entity density) — approaches the oracle
   upper bound (best method per-task) while adding negligible overhead.
   The router is trained with leave-one-out cross-validation over tasks
   to prevent leakage.

3. **Dependency closure matters.** BCA with closure outperforms
   BCA-without-closure at tight budgets (1k–4k tokens), confirming that
   structural completeness is worth the token cost.

These results are from a single (large, real-world) repository and should
be interpreted as a detailed case study rather than a general claim. We
release all code, mutations, and benchmark harness to enable replication
and extension to additional codebases.

CeGraph is pip-installable, requires no API keys for indexing, and integrates
as an MCP server with Claude Code, Cursor, and other AI coding tools.

**Keywords:** code retrieval, context assembly, knowledge graphs, LLM coding
assistants, budgeted optimization, MCP
