"""Live Code Map — interactive terminal UI for codebase exploration.

A Textual-powered TUI that renders your codebase as an explorable graph.
Navigate symbols, see relationships, trace call chains, and understand
impact — all from the terminal.

Requires: pip install cegraph[map]  (installs textual)
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.reactive import reactive
    from textual.widgets import (
        Footer,
        Header,
        Input,
        RichLog,
        Static,
        Tree,
    )
    from textual.widgets.tree import TreeNode

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False


def check_textual():
    """Check if Textual is available."""
    if not HAS_TEXTUAL:
        print("Live Code Map requires Textual. Install it with:")
        print("  pip install cegraph[map]")
        print("  # or: pip install textual")
        sys.exit(1)


if HAS_TEXTUAL:

    class SymbolPanel(Static):
        """Panel showing details about a selected symbol."""

        def update_symbol(self, data: dict) -> None:
            """Update the panel with symbol information."""
            if not data:
                self.update("[dim]Select a symbol to see details[/dim]")
                return

            name = data.get("name", "")
            kind = data.get("kind", "")
            fp = data.get("file_path", "")
            line = data.get("line_start", 0)
            sig = data.get("signature", "")
            doc = data.get("docstring", "")

            kind_colors = {
                "class": "yellow",
                "function": "cyan",
                "method": "green",
                "interface": "magenta",
                "module": "blue",
            }
            color = kind_colors.get(kind, "white")

            lines = [
                f"[bold {color}]{name}[/bold {color}]  [{color}]{kind}[/{color}]",
                f"[dim]{fp}:{line}[/dim]",
                "",
            ]

            if sig:
                lines.append("[bold]Signature:[/bold]")
                lines.append(f"  [cyan]{sig}[/cyan]")
                lines.append("")

            if doc:
                lines.append("[bold]Docstring:[/bold]")
                for doc_line in doc.split("\n")[:5]:
                    lines.append(f"  [dim]{doc_line}[/dim]")
                lines.append("")

            self.update("\n".join(lines))

    class RelationsPanel(RichLog):
        """Panel showing relationships for the selected symbol."""

        def show_relations(self, relations: dict) -> None:
            """Display callers, callees, and related symbols."""
            self.clear()

            callers = relations.get("callers", [])
            callees = relations.get("callees", [])
            related = relations.get("related", [])

            if callers:
                self.write("[bold red]Called by:[/bold red]")
                for c in callers[:10]:
                    self.write(f"  ← {c['name']} [dim]({c['kind']})[/dim]")
                self.write("")

            if callees:
                self.write("[bold green]Calls:[/bold green]")
                for c in callees[:10]:
                    self.write(f"  → {c['name']} [dim]({c['kind']})[/dim]")
                self.write("")

            if related:
                self.write("[bold blue]Related:[/bold blue]")
                for r in related[:10]:
                    rel = r.get("relation", "")
                    self.write(f"  · {r['name']} [dim]({rel})[/dim]")

            if not callers and not callees and not related:
                self.write("[dim]No relationships found[/dim]")

    class ImpactPanel(Static):
        """Panel showing impact analysis for selected symbol."""

        def show_impact(self, impact: dict) -> None:
            """Display impact analysis results."""
            if not impact.get("found"):
                self.update("[dim]No impact data[/dim]")
                return

            risk = impact.get("risk_score", 0)
            if risk < 0.2:
                risk_style = "green"
                risk_label = "LOW"
            elif risk < 0.5:
                risk_style = "yellow"
                risk_label = "MEDIUM"
            else:
                risk_style = "red"
                risk_label = "HIGH"

            lines = [
                "[bold]Impact Analysis[/bold]",
                f"Risk: [{risk_style}]{risk_label} ({risk:.0%})[/{risk_style}]",
                f"Direct callers: {len(impact.get('direct_callers', []))}",
                f"Transitive: {len(impact.get('transitive_callers', []))}",
                f"Files affected: {len(impact.get('affected_files', []))}",
            ]

            files = impact.get("affected_files", [])
            if files:
                lines.append("")
                lines.append("[bold]Affected files:[/bold]")
                for f in files[:8]:
                    lines.append(f"  [cyan]{f}[/cyan]")
                if len(files) > 8:
                    lines.append(f"  [dim]... +{len(files) - 8} more[/dim]")

            self.update("\n".join(lines))

    class CodeMapApp(App):
        """The Live Code Map TUI application."""

        TITLE = "CeGraph — Live Code Map"

        CSS = """
        #file-tree {
            width: 30%;
            border: solid $accent;
            height: 100%;
        }
        #main-area {
            width: 70%;
        }
        #symbol-detail {
            height: 30%;
            border: solid $accent;
            padding: 1;
        }
        #relations {
            height: 40%;
            border: solid $accent;
        }
        #impact {
            height: 30%;
            border: solid $accent;
            padding: 1;
        }
        #search-bar {
            dock: top;
            height: 3;
            padding: 0 1;
        }
        """

        BINDINGS = [
            Binding("q", "quit", "Quit"),
            Binding("/", "focus_search", "Search"),
            Binding("i", "show_impact", "Impact"),
            Binding("c", "show_callers", "Callers"),
            Binding("r", "refresh", "Refresh"),
        ]

        selected_symbol: reactive[str | None] = reactive(None)

        def __init__(self, root: Path, graph, query, **kwargs):
            super().__init__(**kwargs)
            self.root = root
            self.graph = graph
            self.query = query
            self._file_tree_data = self._build_file_tree()

        def compose(self) -> ComposeResult:
            yield Header()
            yield Input(placeholder="Search symbols... (press /)", id="search-bar")
            with Horizontal():
                yield Tree("Codebase", id="file-tree")
                with Vertical(id="main-area"):
                    yield SymbolPanel(id="symbol-detail")
                    yield RelationsPanel(id="relations")
                    yield ImpactPanel(id="impact")
            yield Footer()

        def on_mount(self) -> None:
            """Build the file tree on mount."""
            tree = self.query_one("#file-tree", Tree)
            tree.root.expand()
            self._populate_tree(tree.root, self._file_tree_data)
            tree.root.expand_all()

        def _build_file_tree(self) -> dict:
            """Build a nested dict of file paths -> symbols."""
            structure: dict = {}
            for node_id, data in self.graph.nodes(data=True):
                if data.get("type") != "file":
                    continue
                fp = data.get("path", "")
                parts = fp.split("/")

                node = structure
                for part in parts[:-1]:
                    node = node.setdefault(part, {"_children": {}})["_children"]

                file_key = parts[-1]
                node[file_key] = {"_file": fp, "_symbols": []}

                # Get symbols for this file
                for succ in self.graph.successors(node_id):
                    succ_data = self.graph.nodes.get(succ, {})
                    if succ_data.get("type") == "symbol":
                        kind = succ_data.get("kind", "")
                        if kind not in ("import", "variable"):
                            node[file_key]["_symbols"].append({
                                "id": succ,
                                "name": succ_data.get("name", ""),
                                "kind": kind,
                            })

            return structure

        def _populate_tree(self, parent: TreeNode, data: dict) -> None:
            """Recursively populate the tree widget."""
            for key, value in sorted(data.items()):
                if key.startswith("_"):
                    continue

                if "_file" in value:
                    # It's a file
                    file_node = parent.add(f"[cyan]{key}[/cyan]", data=value)
                    for sym in value.get("_symbols", []):
                        kind = sym.get("kind", "")
                        kind_icon = {
                            "class": "●",
                            "function": "ƒ",
                            "method": "→",
                        }.get(kind, "·")
                        file_node.add_leaf(
                            f"{kind_icon} {sym['name']}",
                            data={"_symbol_id": sym["id"]},
                        )
                elif "_children" in value:
                    # It's a directory
                    dir_node = parent.add(f"[bold]{key}/[/bold]", data=value)
                    self._populate_tree(dir_node, value["_children"])

        def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
            """Handle symbol selection in the tree."""
            data = event.node.data
            if data is None:
                return

            symbol_id = data.get("_symbol_id")
            if symbol_id:
                self.selected_symbol = symbol_id
                self._show_symbol_details(symbol_id)

        def _show_symbol_details(self, symbol_id: str) -> None:
            """Show full details for a symbol."""
            data = self.graph.nodes.get(symbol_id, {})
            if not data:
                return

            # Update symbol detail panel
            detail = self.query_one("#symbol-detail", SymbolPanel)
            detail.update_symbol(data)

            # Update relations panel
            relations_panel = self.query_one("#relations", RelationsPanel)
            relations = self._get_relations(symbol_id)
            relations_panel.show_relations(relations)

            # Update impact panel
            impact_panel = self.query_one("#impact", ImpactPanel)
            name = data.get("name", "")
            impact = self.query.impact_of(name)
            impact_panel.show_impact(impact)

        def _get_relations(self, symbol_id: str) -> dict:
            """Get all relationships for a symbol."""
            callers = []
            callees = []
            related = []

            # Forward edges (callees)
            for succ in self.graph.successors(symbol_id):
                edge_data = self.graph.edges[symbol_id, succ]
                succ_data = self.graph.nodes.get(succ, {})
                if succ_data.get("type") == "symbol":
                    edge_kind = edge_data.get("kind", "")
                    entry = {
                        "name": succ_data.get("name", ""),
                        "kind": succ_data.get("kind", ""),
                        "relation": edge_kind,
                    }
                    if edge_kind == "calls":
                        callees.append(entry)
                    else:
                        related.append(entry)

            # Backward edges (callers)
            for pred in self.graph.predecessors(symbol_id):
                edge_data = self.graph.edges[pred, symbol_id]
                pred_data = self.graph.nodes.get(pred, {})
                if pred_data.get("type") == "symbol":
                    edge_kind = edge_data.get("kind", "")
                    entry = {
                        "name": pred_data.get("name", ""),
                        "kind": pred_data.get("kind", ""),
                        "relation": edge_kind,
                    }
                    if edge_kind == "calls":
                        callers.append(entry)
                    else:
                        related.append(entry)

            return {"callers": callers, "callees": callees, "related": related}

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            """Handle search input."""
            query_text = event.value.strip()
            if not query_text:
                return

            results = self.query.find_symbol(query_text)
            if results:
                self.selected_symbol = results[0]
                self._show_symbol_details(results[0])

            # Clear search
            search = self.query_one("#search-bar", Input)
            search.value = ""

        def action_focus_search(self) -> None:
            """Focus the search bar."""
            self.query_one("#search-bar", Input).focus()

        def action_show_impact(self) -> None:
            """Show impact for current symbol."""
            if self.selected_symbol:
                self._show_symbol_details(self.selected_symbol)

        def action_show_callers(self) -> None:
            """Show callers for current symbol."""
            if self.selected_symbol:
                self._show_symbol_details(self.selected_symbol)

        def action_refresh(self) -> None:
            """Refresh the view."""
            if self.selected_symbol:
                self._show_symbol_details(self.selected_symbol)


def launch_map(root: Path, graph, query) -> None:
    """Launch the Live Code Map TUI."""
    check_textual()
    app = CodeMapApp(root=root, graph=graph, query=query)
    app.run()
