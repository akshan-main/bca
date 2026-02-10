# CeGraph - Project Guidelines

## Project Overview
CeGraph is a pip-installable, repository-aware AI coding assistant that builds a knowledge graph of your codebase and uses it to give LLMs accurate, grounded context for coding tasks.

**Package name:** `cegraph`
**CLI command:** `cegraph`
**Python:** 3.10+

## Architecture

```
src/cegraph/
├── cli.py              # Click CLI (entry point)
├── config.py           # Configuration & .cegraph/ management
├── parser/             # Python code parsing (stdlib AST)
├── graph/              # Knowledge graph construction & querying
├── search/             # Lexical + semantic code search
├── tools/              # Agent tools (search_code, who_calls, impact_of, etc.)
├── agent/              # ReAct agent loop with planning & verification
├── llm/                # LLM provider abstraction (OpenAI, Anthropic)
└── ui/                 # Rich terminal UI
```

## Key Design Principles

1. **Local-first**: Everything runs on the user's machine. No data leaves unless calling an LLM API.
2. **Minimal dependencies**: Core functionality works with just `click`, `rich`, `networkx`, `pydantic`.
3. **Python-focused**: Uses stdlib AST for accurate Python parsing with zero external deps.
4. **Provider agnostic**: Supports OpenAI, Anthropic, and local models via OpenAI-compatible APIs.
5. **Human-in-the-loop**: Always confirm before making changes. The developer is the architect.

## Code Conventions

- Use type hints everywhere
- Use `pydantic` for data models
- Use `click` for CLI, `rich` for terminal output
- Use `networkx` for the in-memory knowledge graph
- Use `sqlite3` for persistent storage (stdlib, no extra deps)
- Use `pathlib.Path` not string paths
- Async where it matters (LLM calls), sync everywhere else
- Tests use `pytest` with fixtures in `conftest.py`

## Dependency Tiers

### Required (installed with `pip install cegraph`)
- click >= 8.0
- rich >= 13.0
- networkx >= 3.0
- pydantic >= 2.0

### Optional - LLM Providers (`pip install cegraph[openai]` or `cegraph[anthropic]`)
- openai >= 1.0
- anthropic >= 0.30

### Optional - Semantic Search (`pip install cegraph[search]`)
- numpy >= 1.24
- sentence-transformers (for local embeddings)

### All extras: `pip install cegraph[all]`

## File Naming

- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Test files: `test_<module>.py`

## Testing

```bash
pytest tests/ -v
pytest tests/test_parser.py -v  # specific module
```

## Common Tasks

- **Index a repo:** `cegraph init`
- **Ask a question:** `cegraph ask "how does auth work?"`
- **Run agent task:** `cegraph agent "add error handling to the API"`
- **Search code:** `cegraph search "database connection"`
- **Show graph stats:** `cegraph status`
- **Show call graph:** `cegraph who-calls function_name`
- **Show impact:** `cegraph impact function_name`

## Error Handling

- Use custom exceptions from `cegraph.exceptions`
- Never swallow errors silently
- Always provide actionable error messages
- Gracefully handle missing optional dependencies with helpful install messages
