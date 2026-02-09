<p align="center">
  <h1 align="center">CeGraph</h1>
  <p align="center"><strong>Knowledge graph that makes AI coding tools smarter</strong></p>
  <p align="center">
    <a href="#cag">CAG Engine</a> |
    <a href="#mcp-server">MCP Server</a> |
    <a href="#pr-impact-bot">PR Impact Bot</a> |
    <a href="#installation">Install</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/cegraph?color=blue" alt="PyPI">
  <img src="https://img.shields.io/pypi/pyversions/cegraph" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://github.com/cegraph-ai/cegraph/actions/workflows/ci.yml/badge.svg" alt="CI">
</p>

---

CeGraph builds a **knowledge graph** of your codebase and uses it to give AI coding tools exactly the right context — not entire files, not keyword matches, but the precise set of symbols needed for any task.

```
$ cegraph context "refactor AgentLoop" --budget 4000

CAG Context Package
  Strategy: smart
  Engine: C++ accelerated
  Tokens: 3,960 / 4,000 (99%)
  Symbols: 12 (from 965 candidates)
  Files: 5
  Time: 0.4ms
```

**80% fewer tokens** than sending all files. Your AI assistant gets better answers, faster, cheaper.

---

## How it works with your AI tools

CeGraph doesn't replace Claude Code, Cursor, or Codex — it makes them **better**:

```bash
# 1. Index your codebase (once)
cegraph init

# 2a. Use as MCP server → Claude Code / Cursor get graph-powered tools
cegraph serve

# 2b. Use in CI → every PR gets automatic blast radius analysis
#     (drop the GitHub Action in your repo)

# 2c. Use from CLI → get budgeted context for any task
cegraph context "fix the auth bug" --budget 4000
```

---

## CAG — Context Assembly Generation <a name="cag"></a>

Given a task and a token budget, CAG selects a **dependency-closed, relevance-scored set of code symbols** for an LLM's context window.

```
Task: "refactor AgentLoop"
         │
         ▼
  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
  │   Extract    │────▶│  Find Seeds  │────▶│  Weighted   │
  │  Entities    │     │  in Graph    │     │    BFS      │
  │              │     │              │     │  (C++ fast) │
  └─────────────┘     └──────────────┘     └──────┬──────┘
                                                   │
  ┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
  │  Dependency  │◀────│   Budget     │◀────│   Score &   │
  │   Order      │     │  Selection   │     │    Rank     │
  └──────┬──────┘     └──────────────┘     └─────────────┘
         │
         ▼
   ContextPackage (ready for LLM)
```

1. **Extract entities** from natural language (CamelCase, snake_case, file paths)
2. **Find seed symbols** in the knowledge graph (965 nodes, 2,402 edges for this repo)
3. **Weighted BFS** — relevance decays along edges (calls: 0.85, imports: 0.5, inherits: 0.9)
4. **Score** by centrality, file proximity, kind importance
5. **Select** within token budget (greedy knapsack)
6. **Order** by dependency (topological sort — definitions before usage)

### Benchmark (run on CeGraph's own codebase)

```
$ cegraph benchmark

CeGraph Benchmark
  Project: cegraph
  Source files: 57 (354 KB)
  graph.db: 346 KB (98% of source)
  Nodes: 965  Edges: 2402

CAG Context Assembly (budget: 4,000 tokens)

  Task                                      symbols   tokens  budget%     time
  ──────────────────────────────────────── ──────── ──────── ──────── ────────
  refactor AgentStep                             22   4,000     100%    1.4ms
  refactor AgentResult                           22   4,000     100%    0.5ms
  refactor AgentLoop                             12   3,960      99%    0.4ms

Token Savings
  CAG:         4,000 tokens (22 symbols, 9 files)
  all files:  20,299 tokens
  Savings:     80%

  Engine: C++ accelerated
```

### C++ acceleration

The BFS hot path is implemented in C++17 (`csrc/cag_fast.cpp`) for 10-50x speedup on large codebases. Falls back to pure Python transparently.

```bash
make -C csrc/   # optional, requires g++ or clang++
```

### Python API

```python
from cegraph.context.engine import ContextAssembler
from cegraph.graph.builder import GraphBuilder
from cegraph.graph.query import GraphQuery

builder = GraphBuilder()
graph = builder.build_from_directory(".")
query = GraphQuery(graph)

assembler = ContextAssembler(Path("."), graph, query)
package = assembler.assemble("fix the login bug", token_budget=4000)

print(package.render())   # Full context, ready for LLM
print(package.summary())  # Quick stats
```

---

## MCP Server <a name="mcp-server"></a>

Expose CeGraph as an [MCP](https://modelcontextprotocol.io/) server so Claude Code, Cursor, Codex, or any MCP client can query your knowledge graph:

```bash
# Generate config for Claude Code
cegraph serve --generate-config claude >> ~/.claude/mcp_servers.json

# Generate config for Cursor
cegraph serve --generate-config cursor >> .cursor/mcp.json
```

Now your AI assistant has access to these tools:

| Tool | What it does |
|:-----|:-------------|
| `cag_assemble` | Assemble budgeted context for a task |
| `search_code` | Search symbols in the knowledge graph |
| `who_calls` | Find all callers of a function |
| `impact_of` | Analyze blast radius of changes |
| `get_structure` | Overview of codebase structure |
| `find_related` | Find related symbols |

The MCP server implements the full JSON-RPC 2.0 protocol over stdio — no external SDK dependency.

---

## PR Impact Bot <a name="pr-impact-bot"></a>

Drop this GitHub Action in your repo — every PR gets automatic blast radius analysis:

```yaml
# .github/workflows/impact.yml
name: CeGraph Impact Analysis
on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  impact:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install cegraph
      - run: cegraph init
      - run: cegraph impact-pr --format github-comment > comment.md
      - uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const body = fs.readFileSync('comment.md', 'utf8');
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: body,
            });
```

The bot generates PR comments like:

```
## CeGraph Impact Analysis

| Risk | Symbols Changed | Files Affected | Callers |
|:---:|:---:|:---:|:---:|
| MEDIUM (38%) | 3 | 7 | 12 |

### Blast Radius
<details>
<summary>UserService.authenticate — 5 files affected</summary>

**Direct callers:**
- login_handler (function) in routes/auth.py
- refresh_token (function) in routes/auth.py
...
</details>
```

---

## Installation <a name="installation"></a>

```bash
pip install cegraph
```

Optional extras:

```bash
pip install cegraph[map]       # Interactive terminal code map (Textual TUI)
pip install cegraph[parsing]   # Tree-sitter for accurate multi-language parsing
pip install cegraph[all]       # Everything
```

## Quick Start

```bash
cegraph init                                    # Index your codebase
cegraph context "fix the login bug" --budget 4000  # Assemble budgeted context
cegraph impact calculate_total                   # Analyze blast radius
cegraph search UserService                       # Search the knowledge graph
cegraph serve                                    # Start MCP server
cegraph benchmark                                # Prove token savings on your repo
```

## Supported Languages

| Language | Parser | Accuracy |
|:---------|:-------|:---------|
| Python | AST (stdlib) | Full |
| JavaScript/TypeScript | tree-sitter | Full |
| Go | tree-sitter | Full |
| Rust | tree-sitter | Full |
| Java | tree-sitter | Full |

## CLI Reference

| Command | Description |
|:--------|:------------|
| `cegraph init` | Index codebase, build knowledge graph |
| `cegraph context TASK` | CAG: assemble budgeted context |
| `cegraph search QUERY` | Search symbols and code |
| `cegraph who-calls SYMBOL` | Find callers of a function |
| `cegraph impact SYMBOL` | Analyze blast radius |
| `cegraph impact-pr` | PR impact analysis (for CI) |
| `cegraph serve` | Start MCP server |
| `cegraph benchmark` | Run token savings benchmark |
| `cegraph map` | Interactive code map (requires `[map]`) |
| `cegraph ask QUESTION` | Ask about the codebase (LLM) |
| `cegraph agent TASK` | Run agentic coding task (LLM) |
| `cegraph config show` | Show configuration |

## Architecture

```
cegraph/
├── context/            # CAG engine (the core innovation)
│   ├── engine.py       # Weighted BFS, scoring, budget selection
│   ├── models.py       # ContextPackage, ContextItem, TokenEstimator
│   └── _native.py      # ctypes bridge to C++ core
├── mcp/                # MCP server (JSON-RPC 2.0 over stdio)
│   └── server.py       # Full protocol implementation
├── github/             # PR Impact Bot
│   ├── impact_bot.py   # Analysis pipeline
│   ├── diff_parser.py  # Git diff → changed symbols
│   └── renderer.py     # Markdown blast radius rendering
├── parser/             # Multi-language code parsing
│   ├── python_parser.py    # Python AST (stdlib, zero deps)
│   ├── javascript_parser.py # JS/TS regex-based
│   ├── generic_parser.py   # Go, Rust, Java, Ruby, C/C++
│   └── tree_sitter_parser.py # Optional tree-sitter
├── graph/              # Knowledge graph (NetworkX + SQLite)
│   ├── builder.py      # Graph construction
│   ├── store.py        # SQLite persistence (normalized, < source size)
│   └── query.py        # who_calls, impact_of, find_related
├── search/             # Lexical + hybrid code search
├── map/                # Interactive Code Map (Textual TUI)
├── llm/                # LLM provider abstraction
├── agent/              # ReAct agent loop
├── tools/              # Tool registry + definitions
├── ui/                 # Rich terminal output
└── cli.py              # Click CLI

csrc/
└── cag_fast.cpp        # C++17 acceleration core

.github/workflows/
├── ci.yml              # Test matrix + lint
└── impact.yml          # PR Impact Bot
```

## Requirements

- Python 3.10+
- C++17 compiler (optional, for acceleration)

## License

MIT
