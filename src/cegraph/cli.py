"""Command-line interface for CeGraph."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import click

from cegraph import __version__
from cegraph.config import (
    GRAPH_DB_FILE,
    ProjectConfig,
    find_project_root,
    get_cegraph_dir,
    load_config,
    save_config,
    set_config_value,
)
from cegraph.ui.console import Console

console = Console()


def _get_project_root(path: str | None = None) -> Path:
    """Find the project root or error."""
    if path:
        root = Path(path).resolve()
        if not root.exists():
            console.error(f"Path does not exist: {path}")
            sys.exit(1)
        return root

    root = find_project_root()
    if root is None:
        console.error(
            "No CeGraph project found. Run 'cegraph init' first, "
            "or specify a path with --path."
        )
        sys.exit(1)
    return root


def _load_graph(root: Path):
    """Load the knowledge graph and related objects."""
    from cegraph.graph.store import GraphStore

    db_path = get_cegraph_dir(root) / GRAPH_DB_FILE
    store = GraphStore(db_path)
    graph = store.load()
    if graph is None:
        console.error("No index found. Run 'cegraph init' or 'cegraph reindex' first.")
        sys.exit(1)
    return graph, store


@click.group()
@click.version_option(version=__version__, prog_name="cegraph")
def main():
    """CeGraph - AI that actually understands your entire codebase."""
    pass


@main.command()
@click.option("--path", "-p", default=None, help="Path to the project root.")
@click.option("--provider", default=None, help="LLM provider (openai, anthropic, local).")
@click.option("--model", default=None, help="LLM model name.")
def init(path: str | None, provider: str | None, model: str | None):
    """Initialize CeGraph for a repository. Indexes the codebase and builds the knowledge graph."""
    root = Path(path or ".").resolve()
    if not root.exists():
        console.error(f"Path does not exist: {root}")
        sys.exit(1)

    console.banner()
    console.info(f"Initializing CeGraph for: {root}")

    # Create/load config
    config = load_config(root)
    config.name = root.name
    config.root_path = str(root)

    if provider:
        config.llm.provider = provider
    if model:
        config.llm.model = model

    save_config(root, config)
    console.success("Configuration saved")

    # Build the knowledge graph
    _do_index(root, config)


@main.command()
@click.option("--path", "-p", default=None, help="Path to the project root.")
@click.option("--full", is_flag=True, help="Full rebuild instead of incremental.")
def reindex(path: str | None, full: bool):
    """Rebuild the knowledge graph (incremental by default)."""
    root = _get_project_root(path)
    config = load_config(root)
    if full:
        _do_index(root, config)
    else:
        _do_incremental_index(root, config)


def _do_index(root: Path, config: ProjectConfig):
    """Index the codebase and build the knowledge graph."""
    from cegraph.graph.builder import GraphBuilder
    from cegraph.graph.store import GraphStore

    builder = GraphBuilder()

    console.info("Scanning and parsing source files...")
    start_time = time.time()

    with console.indexing_progress() as progress:
        task = progress.add_task("Indexing...", total=None)
        file_count = 0

        def on_progress(file_path: str, current: int, total: int):
            nonlocal file_count
            file_count = total
            progress.update(
                task, total=total, completed=current,
                description=f"Parsing {file_path}",
            )

        graph = builder.build_from_directory(root, config, on_progress)

    elapsed = time.time() - start_time
    stats = builder.get_stats()

    console.success(f"Indexed {stats.get('files', 0)} files in {elapsed:.1f}s")
    console.show_stats(stats)

    # Persist the graph
    db_path = get_cegraph_dir(root) / GRAPH_DB_FILE
    store = GraphStore(db_path)
    store.save(graph, metadata={"stats": stats, "root": str(root)})
    store.set_metadata("file_hashes", builder._file_hashes)
    store.set_metadata("file_mtimes", builder.get_file_mtimes(root))
    store.set_metadata("dir_mtimes", builder.get_dir_mtimes(root))
    store.set_metadata("built_at", time.time())
    store.close()

    console.success("Knowledge graph saved to .cegraph/")


def _do_incremental_index(root: Path, config: ProjectConfig):
    """Incrementally update the knowledge graph (only re-parse changed files)."""
    from cegraph.graph.builder import GraphBuilder
    from cegraph.graph.store import GraphStore

    db_path = get_cegraph_dir(root) / GRAPH_DB_FILE
    store = GraphStore(db_path)
    graph = store.load()
    old_hashes = store.get_metadata("file_hashes")

    if graph is None or old_hashes is None:
        console.info("No previous index found, doing full rebuild...")
        store.close()
        return _do_index(root, config)

    builder = GraphBuilder()
    start_time = time.time()

    with console.indexing_progress() as progress:
        task = progress.add_task("Checking for changes...", total=None)

        def on_progress(file_path: str, current: int, total: int):
            progress.update(
                task, total=total, completed=current,
                description=f"Re-parsing {file_path}",
            )

        graph, changed = builder.incremental_build(
            root, graph, old_hashes, config, on_progress,
        )

    elapsed = time.time() - start_time

    if not changed:
        console.success("Already up to date (no files changed)")
        store.close()
        return

    stats = builder.get_stats()
    console.success(
        f"Updated {len(changed)} file(s) in {elapsed:.1f}s"
    )
    console.show_stats(stats)

    store.save(graph, metadata={"stats": stats, "root": str(root)})
    store.set_metadata("file_hashes", builder._file_hashes)
    store.set_metadata("file_mtimes", builder.get_file_mtimes(root))
    store.set_metadata("dir_mtimes", builder.get_dir_mtimes(root))
    store.set_metadata("built_at", time.time())
    store.close()

    console.success("Knowledge graph updated")


@main.command()
@click.option("--path", "-p", default=None, help="Path to the project root.")
def status(path: str | None):
    """Show the current index status and graph statistics."""
    root = _get_project_root(path)
    graph, store = _load_graph(root)

    stats = store.get_metadata("stats")
    if stats:
        console.banner()
        console.info(f"Project: {root.name}")
        console.show_stats(stats)
    else:
        from cegraph.graph.builder import GraphBuilder

        b = GraphBuilder()
        b.graph = graph
        console.show_stats(b.get_stats())
    store.close()


@main.command()
@click.argument("query")
@click.option("--path", "-p", default=None, help="Path to the project root.")
@click.option("--kind", "-k", default="", help="Filter by symbol kind.")
def search(query: str, path: str | None, kind: str):
    """Search for code or symbols in the repository."""
    root = _get_project_root(path)
    graph, store = _load_graph(root)

    from cegraph.search.hybrid import HybridSearch

    search_engine = HybridSearch(root, graph)

    # Try symbol search first
    symbol_results = search_engine.search_symbols(query, kind=kind)
    if symbol_results:
        console.info(f"Found {len(symbol_results)} symbol(s) matching '{query}':")
        console.show_search_results(symbol_results)
    else:
        console.info(f"No symbol definitions found for '{query}', searching code...")
        code_results = search_engine.search(query)
        if code_results:
            for r in code_results:
                console.console.print(
                    f"  [cyan]{r.file_path}:{r.line_number}[/cyan] {r.line_content.strip()}"
                )
                if r.symbol_name:
                    console.console.print(f"    [dim]in {r.symbol_name}[/dim]")
        else:
            console.warning(f"No results found for '{query}'")

    store.close()


@main.command("who-calls")
@click.argument("symbol_name")
@click.option("--path", "-p", default=None, help="Path to the project root.")
@click.option("--depth", "-d", default=2, help="Max depth of caller traversal.")
def who_calls(symbol_name: str, path: str | None, depth: int):
    """Find all callers of a function or method."""
    root = _get_project_root(path)
    graph, store = _load_graph(root)

    from cegraph.graph.query import GraphQuery

    query = GraphQuery(graph, store)
    callers = query.who_calls(symbol_name, max_depth=depth)

    if callers:
        console.info(f"Callers of '{symbol_name}':")
        console.show_callers(callers, symbol_name)
    else:
        console.warning(f"No callers found for '{symbol_name}'")

    store.close()


@main.command()
@click.argument("symbol_name")
@click.option("--path", "-p", default=None, help="Path to the project root.")
def impact(symbol_name: str, path: str | None):
    """Analyze the blast radius of changing a symbol."""
    root = _get_project_root(path)
    graph, store = _load_graph(root)

    from cegraph.graph.query import GraphQuery

    query = GraphQuery(graph, store)
    result = query.impact_of(symbol_name)
    console.show_impact(result)

    store.close()


# =========================================================================
# CAG -- Context Assembly Generation
# =========================================================================

@main.command()
@click.argument("task")
@click.option("--path", "-p", default=None, help="Path to the project root.")
@click.option("--budget", "-b", default=8000, type=int, help="Token budget (default: 8000).")
@click.option(
    "--strategy", "-s",
    type=click.Choice(["precise", "smart", "thorough"]),
    default="smart",
    help="Context strategy (default: smart).",
)
@click.option(
    "--compact", is_flag=True,
    help="Use compact rendering (signatures only for deep symbols).",
)
@click.option("--savings", is_flag=True, help="Show token savings comparison.")
@click.option("--focus", "-f", multiple=True, help="Focus files (can specify multiple).")
def context(
    task: str, path: str | None, budget: int, strategy: str,
    compact: bool, savings: bool, focus: tuple[str, ...]
):
    """Assemble budgeted code context for a task using CAG.

    Given a natural language task description, assembles relevant code from the
    knowledge graph -- scored, ranked, and dependency-ordered within a token
    budget.

    Examples:

        cegraph context "fix the login bug in AuthService"

        cegraph context "add pagination to the user list API" --budget 4000

        cegraph context "refactor calculate_total" --strategy thorough --savings
    """
    root = _get_project_root(path)
    graph, store = _load_graph(root)

    from cegraph.context.engine import ContextAssembler
    from cegraph.context.models import ContextStrategy
    from cegraph.graph.query import GraphQuery

    query = GraphQuery(graph, store)
    assembler = ContextAssembler(root, graph, query)

    strategy_map = {
        "precise": ContextStrategy.PRECISE,
        "smart": ContextStrategy.SMART,
        "thorough": ContextStrategy.THOROUGH,
    }

    if savings:
        console.info("Computing token savings comparison...")
        result = assembler.estimate_savings(task, budget)
        console.console.print()
        console.console.print("[bold]CAG Token Savings Analysis[/bold]")
        console.console.print(f"  Task: {task}")
        accel_str = "[green]C++ accelerated[/green]" if result["accelerated"] else "Python"
        console.console.print(f"  Backend: {accel_str}")
        console.console.print()

        if result["cag_symbols"] == 0:
            console.warning(
                "CAG found no matching symbols in the knowledge graph. "
                "Check that the symbol names in your task match actual code in this project."
            )
            console.console.print()

        console.console.print(f"  [bold cyan]CAG:[/bold cyan] {result['cag_tokens']:,} tokens "
                            f"({result['cag_symbols']} symbols, {result['cag_files']} files)")
        console.console.print(f"  [dim]grep:[/dim] {result['grep_tokens']:,} tokens "
                            f"({result['grep_files']} files)")
        console.console.print(f"  [dim]all files:[/dim] {result['all_files_tokens']:,} tokens")
        console.console.print()
        console.console.print(f"  Savings vs grep: [green]{result['savings_vs_grep']}[/green]")
        console.console.print(f"  Savings vs all:  [green]{result['savings_vs_all']}[/green]")
    else:
        package = assembler.assemble(
            task=task,
            token_budget=budget,
            strategy=strategy_map[strategy],
            focus_files=list(focus) if focus else None,
        )

        # Show summary
        console.console.print()
        console.console.print("[bold]CAG Context Package[/bold]")
        console.console.print(f"  Strategy: {strategy}")
        accel = assembler.is_accelerated
        accel_str = "[green]C++ accelerated[/green]" if accel else "Python"
        console.console.print(f"  Backend: {accel_str}")
        console.console.print(
            f"  Tokens: {package.total_tokens:,} / {package.token_budget:,} "
            f"({package.budget_used_pct:.0f}%)"
        )
        console.console.print(
            f"  Symbols: {package.symbols_included} "
            f"(from {package.symbols_available} candidates)"
        )
        console.console.print(f"  Files: {package.files_included}")
        console.console.print(f"  Time: {package.assembly_time_ms:.1f}ms")
        console.console.print()

        if package.symbols_included == 0:
            console.warning(
                "No matching symbols found. The names in your task description "
                "must match actual symbols in this codebase.\n"
                "  Try: cegraph search <name> to find valid symbol names."
            )
            console.console.print()

        # Output the context
        if compact:
            console.console.print(package.render_compact())
        else:
            console.console.print(package.render())

    store.close()


# =========================================================================
# MCP Server
# =========================================================================

@main.command()
@click.option("--path", "-p", default=None, help="Path to the project root.")
@click.option(
    "--transport", "-t",
    type=click.Choice(["stdio"]),
    default="stdio",
    help="Transport protocol (default: stdio).",
)
@click.option("--generate-config", type=click.Choice(["claude", "cursor"]),
              default=None, help="Generate MCP config for a client.")
def serve(path: str | None, transport: str, generate_config: str | None):
    """Start the MCP server for AI tool integration.

    Exposes CeGraph's knowledge graph as MCP tools, allowing Claude Code,
    Cursor, and other MCP clients to query your codebase intelligently.

    Setup for Claude Code:

        cegraph serve --generate-config claude >> ~/.claude/mcp_servers.json

    Setup for Cursor:

        cegraph serve --generate-config cursor >> .cursor/mcp.json

    Then restart your AI tool. It will now have access to:
      - cag_assemble: Get budgeted context for any task
      - search_code: Search symbols in the graph
      - who_calls: Find callers of a function
      - impact_of: Analyze blast radius
      - get_structure: See codebase overview
    """
    if generate_config:
        from cegraph.mcp.server import MCPServer

        root_path = str(Path(path or ".").resolve())
        if generate_config == "claude":
            config = MCPServer.generate_claude_config(root_path)
        else:
            config = MCPServer.generate_cursor_config(root_path)
        click.echo(json.dumps(config, indent=2))
        return

    root = _get_project_root(path)
    from cegraph.mcp.server import MCPServer

    server = MCPServer(root)

    if transport == "stdio":
        asyncio.run(server.run_stdio())


# =========================================================================
# PR Impact Bot
# =========================================================================

@main.command("impact-pr")
@click.option("--path", "-p", default=None, help="Path to the project root.")
@click.option("--base", "-b", default="main", help="Base branch to diff against.")
@click.option(
    "--format", "output_format",
    type=click.Choice(["markdown", "json", "github-comment"]),
    default="markdown",
    help="Output format.",
)
def impact_pr(path: str | None, base: str, output_format: str):
    """Analyze PR impact and generate blast radius report.

    Parses the git diff, maps changes to symbols in the knowledge graph,
    and generates a detailed impact analysis.

    Usage in CI:

        cegraph impact-pr --format github-comment

    Local usage:

        cegraph impact-pr --base main --format markdown
    """
    root = _get_project_root(path)

    import os

    from cegraph.github.impact_bot import post_github_comment, run_impact_analysis
    is_pr = bool(os.environ.get("GITHUB_EVENT_PATH"))

    result = run_impact_analysis(root, base=base, is_pr=is_pr)

    if "error" in result:
        console.error(result["error"])
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(result, indent=2, default=str))
    elif output_format == "github-comment":
        click.echo(result["comment"])
        if is_pr:
            if post_github_comment(result["comment"]):
                console.success("Posted impact comment to PR")
            else:
                console.warning("Could not post comment (missing GITHUB_TOKEN or PR context)")
    else:
        click.echo(result["comment"])


# =========================================================================
# Live Code Map
# =========================================================================

@main.command("map")
@click.option("--path", "-p", default=None, help="Path to the project root.")
def code_map(path: str | None):
    """Launch the interactive Live Code Map TUI.

    Explore your codebase as an interactive graph in the terminal.
    Navigate symbols, see relationships, trace call chains, and
    understand impact -- all without leaving your terminal.

    Requires: pip install cegraph[map]

    Keybindings:
      /  - Search symbols
      i  - Impact analysis for selected symbol
      c  - Show callers
      r  - Refresh
      q  - Quit
    """
    root = _get_project_root(path)
    graph, store = _load_graph(root)

    from cegraph.graph.query import GraphQuery
    from cegraph.map.app import launch_map

    query = GraphQuery(graph, store)
    launch_map(root, graph, query)

    store.close()


# =========================================================================
# Agent & Q&A
# =========================================================================

@main.command()
@click.argument("question")
@click.option("--path", "-p", default=None, help="Path to the project root.")
def ask(question: str, path: str | None):
    """Ask a question about the codebase (uses LLM + knowledge graph)."""
    root = _get_project_root(path)
    config = load_config(root)
    graph, store = _load_graph(root)

    _run_agent(root, config, graph, store, question, agent_mode=False)


@main.command()
@click.argument("task")
@click.option("--path", "-p", default=None, help="Path to the project root.")
@click.option("--auto", is_flag=True, help="Skip approval prompts.")
def agent(task: str, path: str | None, auto: bool):
    """Run an agentic task (coding, debugging, refactoring)."""
    root = _get_project_root(path)
    config = load_config(root)
    graph, store = _load_graph(root)

    _run_agent(root, config, graph, store, task, agent_mode=True, auto_approve=auto)


def _run_agent(
    root: Path,
    config: ProjectConfig,
    graph,
    store,
    task: str,
    agent_mode: bool = True,
    auto_approve: bool = False,
):
    """Run the agent loop."""
    from cegraph.agent.loop import AgentLoop, AgentStep
    from cegraph.graph.query import GraphQuery
    from cegraph.llm.factory import create_provider
    from cegraph.search.hybrid import HybridSearch
    from cegraph.tools.definitions import get_all_tools

    # Check for API key (skip for local providers that don't need one)
    llm_config = config.llm
    if not llm_config.api_key and llm_config.provider not in ("local",):
        provider = llm_config.provider
        env_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(
            provider, f"{provider.upper()}_API_KEY"
        )
        console.error(
            f"No API key found for {provider}. "
            f"Set the {env_var} environment variable or configure it with:\n"
            f"  cegraph config set llm.api_key_env {env_var}"
        )
        store.close()
        sys.exit(1)

    try:
        llm = create_provider(llm_config)
    except Exception as e:
        console.error(str(e))
        store.close()
        sys.exit(1)

    query = GraphQuery(graph, store)
    search_engine = HybridSearch(root, graph)
    tools = get_all_tools(root, graph, query, search_engine)

    def on_step(step: AgentStep):
        console.show_agent_step(step)

    agent_loop = AgentLoop(
        llm=llm,
        tools=tools,
        project_name=config.name,
        max_iterations=config.agent.max_iterations,
        on_step=on_step,
    )

    console.info(f"Running {'agent' if agent_mode else 'Q&A'} for: {task}")
    console.console.print()

    result = asyncio.run(agent_loop.run(task))

    if not result.success:
        if result.error:
            console.error(result.error)

    console.console.print()
    console.info(
        f"Completed in {result.total_iterations} step(s), "
        f"~{result.total_tokens:,} tokens used"
    )

    store.close()


# =========================================================================
# Benchmark
# =========================================================================

@main.command()
@click.option("--path", "-p", default=None, help="Path to the project root.")
def benchmark(path: str | None):
    """Run CAG benchmarks against this codebase.

    Shows concrete proof of token savings: indexes the codebase,
    runs CAG on sample tasks, and compares against grep/naive approaches.
    """
    import os
    root = _get_project_root(path)
    graph, store = _load_graph(root)

    from cegraph.context.engine import ContextAssembler
    from cegraph.graph.query import GraphQuery

    query = GraphQuery(graph, store)
    assembler = ContextAssembler(root, graph, query)

    stats = store.get_metadata("stats")
    db_path = get_cegraph_dir(root) / GRAPH_DB_FILE
    db_size = os.path.getsize(db_path) if db_path.exists() else 0

    # Collect source size
    from cegraph.parser.models import detect_language
    total_src = 0
    file_count = 0
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {
            ".git", ".cegraph", "__pycache__", "node_modules",
            ".venv", "venv", "dist", "build",
        }]
        for f in files:
            if detect_language(f):
                fp = os.path.join(dp, f)
                try:
                    total_src += os.path.getsize(fp)
                    file_count += 1
                except OSError:
                    pass

    console.console.print()
    console.console.print("[bold]CeGraph Benchmark[/bold]")
    console.console.print(f"  Project: {root.name}")
    console.console.print(f"  Source files: {file_count} ({total_src / 1024:.0f} KB)")
    pct = db_size / max(total_src, 1) * 100
    console.console.print(
        f"  graph.db: {db_size / 1024:.0f} KB "
        f"({pct:.0f}% of source)"
    )
    if stats:
        console.console.print(
            f"  Nodes: {stats.get('total_nodes', '?')}  "
            f"Edges: {stats.get('total_edges', '?')}"
        )
    console.console.print()

    # Build realistic tasks from actual class + function names
    classes = store.search_symbols(kind="class", limit=50)
    functions = store.search_symbols(kind="function", limit=50)

    def _is_good_symbol(sym: dict) -> bool:
        fp = sym.get("file_path", "")
        return (
            not fp.startswith("tests/")
            and not fp.startswith("examples/")
            and (fp.endswith(".py") or fp.endswith(".js") or fp.endswith(".ts"))
            and not sym["name"].startswith("_")
        )

    tasks = []
    for cls in classes:
        if _is_good_symbol(cls):
            tasks.append(f"refactor {cls['name']}")
            if len(tasks) >= 3:
                break
    for fn in functions:
        if _is_good_symbol(fn):
            tasks.append(f"fix bug in {fn['name']}")
            if len(tasks) >= 5:
                break

    if not tasks:
        console.warning("No symbols found to benchmark against.")
        store.close()
        return

    budget = 4000

    console.console.print(f"[bold]CAG Context Assembly (budget: {budget:,} tokens)[/bold]")
    console.console.print()
    console.console.print(
        f"  {'Task':<40} {'symbols':>8} {'tokens':>8} {'budget%':>8} {'time':>8}"
    )
    console.console.print(f"  {'─' * 40} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    for task in tasks:
        t0 = time.time()
        pkg = assembler.assemble(task, token_budget=budget)
        ms = (time.time() - t0) * 1000
        console.console.print(
            f"  {task:<40} {pkg.symbols_included:>8} {pkg.total_tokens:>7,} "
            f"{pkg.budget_used_pct:>7.0f}% {ms:>6.1f}ms"
        )

    console.console.print()

    # Token savings comparison
    console.console.print("[bold]Token Savings (first task)[/bold]")
    console.console.print()
    result = assembler.estimate_savings(tasks[0], budget)
    cag_t = result["cag_tokens"]
    grep_t = result["grep_tokens"]
    all_t = result["all_files_tokens"]
    console.console.print(
        f"  CAG:       {cag_t:>7,} tokens "
        f"({result['cag_symbols']} symbols, "
        f"{result['cag_files']} files)"
    )
    console.console.print(f"  grep:      {grep_t:>7,} tokens ({result['grep_files']} files)")
    console.console.print(f"  all files: {all_t:>7,} tokens")
    if grep_t > 0 and cag_t < grep_t:
        console.console.print(f"  [green]Savings vs grep: {result['savings_vs_grep']}[/green]")
    console.console.print(f"  [green]Savings vs all:  {result['savings_vs_all']}[/green]")

    console.console.print()
    accel = "C++ accelerated" if assembler.is_accelerated else "Python"
    console.console.print(f"  Backend: {accel}")
    console.console.print()
    store.close()


# =========================================================================
# Config Management
# =========================================================================

@main.command("config")
@click.argument("action", type=click.Choice(["set", "get", "show"]))
@click.argument("key", required=False)
@click.argument("value", required=False)
@click.option("--path", "-p", default=None, help="Path to the project root.")
def config_cmd(action: str, key: str | None, value: str | None, path: str | None):
    """Manage CeGraph configuration."""
    root = _get_project_root(path)
    config = load_config(root)

    if action == "show":
        console.console.print_json(json.dumps(config.model_dump(), indent=2))
    elif action == "get":
        if not key:
            console.error("Usage: cegraph config get <key>")
            sys.exit(1)
        data = config.model_dump()
        parts = key.split(".")
        for part in parts:
            if isinstance(data, dict) and part in data:
                data = data[part]
            else:
                console.error(f"Unknown key: {key}")
                sys.exit(1)
        console.console.print(f"{key} = {data}")
    elif action == "set":
        if not key or value is None:
            console.error("Usage: cegraph config set <key> <value>")
            sys.exit(1)
        try:
            # Try to parse as JSON for non-string values
            try:
                parsed_value = json.loads(value)
            except json.JSONDecodeError:
                parsed_value = value

            config = set_config_value(config, key, parsed_value)
            save_config(root, config)
            console.success(f"Set {key} = {parsed_value}")
        except KeyError:
            console.error(f"Unknown config key: {key}")
            sys.exit(1)


if __name__ == "__main__":
    main()
