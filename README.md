# Budgeted Context Assembly for LLM Code Repair

Benchmark and code for studying context assembly methods under token budgets for LLM-based code repair.

## What this is

Given a bug and a token budget, how should you pick which code to show the LLM? This repo contains a benchmark that evaluates different context assembly methods on that question, using controlled single-line mutations across two Python repositories.

## Repo structure

- `src/cegraph/` — library that builds a knowledge graph from Python source and implements the retrieval/assembly methods
- `paper/experiments/` — benchmark runner, mutation discovery, figure generation, analysis scripts
- `paper/results/` — raw results from benchmark runs
- `paper/` — paper draft (not yet published)
- `tests/` — unit tests for the library
- `csrc/` — optional C++ acceleration for graph traversal

## Setup

```bash
pip install -e ".[dev]"
pip install matplotlib   # for figure generation
pytest tests/ -v
```

## Running the benchmark

### 1. Clone and set up target repos

```bash
# pydantic-ai
git clone https://github.com/pydantic/pydantic-ai.git /path/to/repos/pydantic-ai
cd /path/to/repos/pydantic-ai
git checkout 69a578a1e101
pip install -e pydantic_graph && pip install -e 'pydantic_ai_slim[test]' && pip install inline-snapshot

# httpx
git clone https://github.com/encode/httpx.git /path/to/repos/httpx
cd /path/to/repos/httpx
git checkout ae1b9f66238f
pip install -e '.[brotli,zstd,cli,http2,socks]' && pip install pytest uvicorn trio trustme chardet
```

### 2. Run the benchmark

The mutations are already defined in the task files — no discovery step needed.

```bash
export OPENAI_API_KEY=...

python paper/experiments/benchmark.py \
  --tasks-file paper/experiments/eval_tasks_full.jsonl \
  --repos-dir /path/to/repos \
  --budgets 2000,4000,8000,10000 \
  --methods no_retrieval,bm25,vector,repo_map,bca_d1,bca,bca_d5,bca_no_closure,bca_no_scoring,target_file \
  --query-types exact,vague,dev_report \
  --output-dir paper/results/my_run
```

### 3. Generate figures and tables

```bash
python -m paper.experiments.merge_and_report --run-dir paper/results/my_run
python paper/experiments/generate_figures.py --run-dir paper/results/my_run
python paper/experiments/posthoc_analysis.py --run-dir paper/results/my_run
python paper/experiments/router_ab.py --run-dir paper/results/my_run
```

## License

MIT
