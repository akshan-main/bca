"""Rich-powered console output for CeGraph."""

from __future__ import annotations

from rich.console import Console as RichConsole
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from cegraph.agent.loop import AgentStep


class Console:
    """Beautiful terminal output for CeGraph using Rich."""

    def __init__(self) -> None:
        self.console = RichConsole()

    def banner(self) -> None:
        """Show the CeGraph banner."""
        self.console.print(
            Panel(
                "[bold cyan]CeGraph[/bold cyan] [dim]v0.1.0[/dim]\n"
                "[dim]AI that actually understands your entire codebase[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def success(self, message: str) -> None:
        self.console.print(f"[green]✓[/green] {message}")

    def error(self, message: str) -> None:
        self.console.print(f"[red]✗[/red] {message}")

    def warning(self, message: str) -> None:
        self.console.print(f"[yellow]![/yellow] {message}")

    def info(self, message: str) -> None:
        self.console.print(f"[blue]i[/blue] {message}")

    def markdown(self, text: str) -> None:
        """Render markdown text."""
        self.console.print(Markdown(text))

    def code(self, text: str, language: str = "python") -> None:
        """Render syntax-highlighted code."""
        self.console.print(Syntax(text, language, theme="monokai", line_numbers=True))

    def indexing_progress(self) -> Progress:
        """Create a progress bar for indexing."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
        )

    def show_stats(self, stats: dict) -> None:
        """Display graph statistics in a nice table."""
        table = Table(title="Knowledge Graph Statistics", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Count", justify="right", style="cyan")

        table.add_row("Files", str(stats.get("files", 0)))
        table.add_row("Functions/Methods", str(stats.get("functions", 0)))
        table.add_row("Classes", str(stats.get("classes", 0)))
        table.add_row("Total Nodes", str(stats.get("total_nodes", 0)))
        table.add_row("Total Edges", str(stats.get("total_edges", 0)))
        table.add_row("Unresolved Refs", str(stats.get("unresolved_refs", 0)))

        # Edge type breakdown
        edge_types = stats.get("edge_types", {})
        if edge_types:
            table.add_section()
            for kind, count in sorted(edge_types.items(), key=lambda x: -x[1]):
                table.add_row(f"  {kind} edges", str(count))

        self.console.print(table)

    def show_agent_step(self, step: AgentStep) -> None:
        """Display an agent step with tool calls and results."""
        if step.thought:
            self.console.print(
                Panel(
                    Markdown(step.thought),
                    title=f"[bold]Step {step.iteration}[/bold] - Thinking",
                    border_style="blue",
                )
            )

        for tc in step.tool_calls:
            args_str = ", ".join(f"{k}={repr(v)}" for k, v in tc.arguments.items())
            self.console.print(
                f"  [yellow]→[/yellow] [bold]{tc.name}[/bold]({args_str})"
            )

        for tr in step.tool_results:
            # Truncate long results
            content = tr.content
            if len(content) > 500:
                content = content[:500] + "\n... (truncated)"
            style = "red" if tr.is_error else "dim"
            self.console.print(f"  [{style}]{content}[/{style}]")

        if step.response:
            self.console.print(
                Panel(
                    Markdown(step.response),
                    title="[bold green]Answer[/bold green]",
                    border_style="green",
                )
            )

    def show_search_results(self, results: list[dict]) -> None:
        """Display search results."""
        for r in results:
            self.console.print(
                f"  [bold]{r.get('qualified_name', r.get('name', 'unknown'))}[/bold] "
                f"[dim]({r.get('kind', '')})[/dim] "
                f"at [cyan]{r.get('file_path', '')}:{r.get('line', '')}[/cyan]"
            )
            if r.get("signature"):
                self.console.print(f"    [dim]{r['signature']}[/dim]")

    def show_callers(self, callers: list[dict], symbol_name: str) -> None:
        """Display caller tree."""
        tree = Tree(f"[bold cyan]{symbol_name}[/bold cyan]")
        depth_nodes: dict[int, Tree] = {0: tree}

        for caller in callers:
            depth = caller.get("depth", 1)
            parent = depth_nodes.get(depth - 1, tree)
            node = parent.add(
                f"[bold]{caller['name']}[/bold] [dim]({caller['kind']})[/dim] "
                f"at [cyan]{caller['file_path']}:{caller['line']}[/cyan]"
            )
            depth_nodes[depth] = node

        self.console.print(tree)

    def show_impact(self, impact: dict) -> None:
        """Display impact analysis results."""
        if not impact.get("found"):
            self.error(f"Symbol '{impact.get('symbol', '')}' not found")
            return

        risk = impact.get("risk_score", 0)
        risk_color = "green" if risk < 0.2 else "yellow" if risk < 0.5 else "red"

        self.console.print(
            Panel(
                f"[bold]Symbol:[/bold] {impact.get('symbol', '')}\n"
                f"[bold]Risk Score:[/bold] [{risk_color}]{risk:.1%}[/{risk_color}]\n"
                f"[bold]Direct Callers:[/bold] {len(impact.get('direct_callers', []))}\n"
                f"[bold]Transitive Callers:[/bold] {len(impact.get('transitive_callers', []))}\n"
                f"[bold]Affected Files:[/bold] {len(impact.get('affected_files', []))}",
                title="[bold]Impact Analysis[/bold]",
                border_style=risk_color,
            )
        )

        files = impact.get("affected_files", [])
        if files:
            self.console.print("\n[bold]Affected files:[/bold]")
            for f in files:
                self.console.print(f"  [cyan]{f}[/cyan]")

    def confirm(self, message: str) -> bool:
        """Ask for user confirmation."""
        response = self.console.input(f"\n{message} [y/N] ")
        return response.lower().strip() in ("y", "yes")
