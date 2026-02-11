# CeGraph - Implementation Plan

## Vision
A pip-installable AI coding assistant that builds a knowledge graph of your codebase, giving LLMs the context they need to make accurate, scoped changes instead of blind guesses.

## The Problem
AI coding assistants today operate blind:
- No understanding of codebase structure, dependencies, or conventions
- Overwrite entire files instead of targeted changes
- Get stuck in error-fix loops
- Produce plausible-looking but subtly wrong code
- Can't predict the blast radius of changes

## The Solution
**CeGraph** = Knowledge Graph + Agentic AI + Human-in-the-Loop

```
┌─────────────────────────────────────────────────┐
│                   Developer                      │
│              (Human-in-the-Loop)                 │
└────────────────────┬────────────────────────────┘
                     │ asks / approves
                     ▼
┌─────────────────────────────────────────────────┐
│              CeGraph Agent Loop                │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Planner  │→ │ Executor │→ │  Verifier    │  │
│  │(decompose│  │(run tools│  │(test, diff,  │  │
│  │ tasks)   │  │ & LLM)   │  │ lint check)  │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
└────────────────────┬────────────────────────────┘
                     │ queries
                     ▼
┌─────────────────────────────────────────────────┐
│              Tool Layer (MCP-like)               │
│  search_code │ who_calls │ impact_of │ read_file│
│  write_file  │ run_tests │ get_context│ ...     │
└────────────────────┬────────────────────────────┘
                     │ reads from
                     ▼
┌─────────────────────────────────────────────────┐
│           Knowledge Graph + Search Index         │
│  ┌───────────────┐  ┌────────────────────────┐  │
│  │  Code Graph   │  │   Search Index         │  │
│  │  (networkx +  │  │   (lexical + optional  │  │
│  │   SQLite)     │  │    semantic embeddings)│  │
│  └───────────────┘  └────────────────────────┘  │
└────────────────────┬────────────────────────────┘
                     │ built from
                     ▼
┌─────────────────────────────────────────────────┐
│              Code Parser Layer                    │
│  Python AST (stdlib)      │ .gitignore aware     │
└─────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Foundation
1. **Project scaffolding** - pyproject.toml, package structure, CLI skeleton
2. **Configuration** - `.cegraph/` directory, config file, gitignore patterns
3. **Code parser** - Extract symbols (functions, classes, imports, calls) from source files
4. **Knowledge graph** - Build and persist the code graph with NetworkX + SQLite

### Phase 2: Intelligence
5. **Search engine** - Lexical search + optional semantic search
6. **Tool layer** - Implement agent tools (search_code, who_calls, impact_of, etc.)
7. **LLM providers** - OpenAI and Anthropic with unified interface

### Phase 3: Agent
8. **Agent loop** - ReAct-style agent with tool calling
9. **Planning** - Task decomposition and execution planning
10. **Verification** - Test running, diff review, lint checking

### Phase 4: Polish
11. **CLI** - Full Click CLI with Rich output
12. **Tests** - Comprehensive test suite
13. **README** - Beautiful docs, examples, GIFs
14. **Publishing** - PyPI, GitHub Actions, badges

## Data Models

### Symbol (node in the graph)
```python
class Symbol:
    name: str           # "MyClass.my_method"
    kind: SymbolKind    # function, class, method, variable, module
    file_path: str      # relative path
    line_start: int
    line_end: int
    signature: str      # "def my_method(self, x: int) -> str"
    docstring: str?
    summary: str?       # AI-generated summary (optional)
```

### Relationship (edge in the graph)
```python
class Relationship:
    source: str         # symbol ID
    target: str         # symbol ID
    kind: RelKind       # calls, imports, inherits, overrides, uses
    file_path: str
    line: int
```

### File (metadata)
```python
class FileInfo:
    path: str
    language: str
    size: int
    hash: str           # for change detection
    symbols: list[str]  # symbol IDs
```

## Graph Queries (Agent Tools)

| Tool | Description | Graph Query |
|------|-------------|-------------|
| `search_code(query)` | Find code matching query | Lexical/semantic search |
| `who_calls(symbol)` | Who calls this function? | Reverse edges: `calls` |
| `what_calls(symbol)` | What does this function call? | Forward edges: `calls` |
| `impact_of(symbol)` | What breaks if I change this? | Transitive closure of reverse `calls` + `imports` |
| `get_context(symbol)` | Full context for a symbol | Node data + neighbors + file content |
| `find_related(symbol)` | Related symbols | N-hop neighborhood |
| `get_structure(path)` | File/directory structure | Subgraph by path prefix |

## LLM Integration

### Provider Abstraction
```python
class LLMProvider(ABC):
    async def complete(messages, tools) -> LLMResponse
    async def stream(messages, tools) -> AsyncIterator[LLMChunk]
```

### Supported Providers
- **OpenAI**: GPT-4o, GPT-4-turbo, o1, o3 via `openai` SDK
- **Anthropic**: Claude Sonnet/Opus via `anthropic` SDK
- **Local**: Any OpenAI-compatible API (Ollama, vLLM, LM Studio)

### Agent Loop (ReAct Pattern)
```
while not done:
    1. Send context + tools to LLM
    2. LLM returns: thought + action (tool call) OR final answer
    3. If tool call: execute tool, add result to context
    4. If final answer: present to user for approval
    5. If approved: apply changes, run verification
    6. If verification fails: loop back with error context
```

## CLI Commands

```bash
# Setup
cegraph init                    # Index current repo
cegraph init --path /my/project # Index specific path
cegraph status                  # Show graph stats

# Query
cegraph ask "how does the auth system work?"
cegraph search "database connection pooling"
cegraph who-calls MyClass.my_method
cegraph impact MyClass.my_method

# Agent
cegraph agent "fix the bug in user authentication"
cegraph agent "add rate limiting to the API endpoints"

# Maintenance
cegraph reindex                 # Rebuild graph
cegraph config set llm.provider openai
cegraph config set llm.model gpt-4o
```

## Configuration (.cegraph/config.toml)

```toml
[project]
name = "my-project"
languages = ["python"]

[llm]
provider = "anthropic"  # openai, anthropic, local
model = "claude-sonnet-4-5-20250929"
api_key_env = "ANTHROPIC_API_KEY"  # env var name, never store keys
max_tokens = 4096
temperature = 0.0

[agent]
max_iterations = 10
auto_verify = true
require_approval = true

[indexer]
exclude_patterns = ["node_modules", "__pycache__", ".git", "dist", "build"]
max_file_size_kb = 500
```

## Success Metrics (for 1000+ stars)
1. Works out of the box: `pip install cegraph && cegraph init && cegraph ask "how does this work?"`
2. Impressive first impression: Beautiful CLI output, fast indexing, accurate answers
3. Solves a real problem: No more blind AI coding
4. Easy to extend: Plugin system for languages and tools
5. Great documentation: Clear README with GIFs, examples, and comparison charts
