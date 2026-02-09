#!/usr/bin/env python3
"""Demo: Using CeGraph as a Python library.

This shows how to use CeGraph programmatically, not just as a CLI tool.
"""

from pathlib import Path

from cegraph.graph.builder import GraphBuilder
from cegraph.graph.query import GraphQuery
from cegraph.search.hybrid import HybridSearch


def main():
    # Point at any code directory
    project_root = Path(".")

    # 1. Build the knowledge graph
    print("Building knowledge graph...")
    builder = GraphBuilder()
    graph = builder.build_from_directory(project_root)

    stats = builder.get_stats()
    print(f"  Files: {stats['files']}")
    print(f"  Functions/Methods: {stats['functions']}")
    print(f"  Classes: {stats['classes']}")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")

    # 2. Query the graph
    query = GraphQuery(graph)

    # Find a symbol
    print("\n--- Finding 'GraphBuilder' ---")
    symbol_ids = query.find_symbol("GraphBuilder")
    for sid in symbol_ids:
        info = query.get_symbol_info(sid)
        if info:
            print(f"  {info.qualified_name} ({info.kind}) at {info.file_path}:{info.line_start}")
            print(f"  Callers: {len(info.callers)}")
            print(f"  Callees: {len(info.callees)}")

    # Who calls a function?
    print("\n--- Who calls 'parse_file'? ---")
    callers = query.who_calls("parse_file", max_depth=2)
    for c in callers:
        indent = "  " * c["depth"]
        print(f"{indent}  {c['name']} at {c['file_path']}:{c['line']}")

    # Impact analysis
    print("\n--- Impact of changing 'GraphQuery' ---")
    impact = query.impact_of("GraphQuery")
    print(f"  Risk score: {impact['risk_score']:.1%}")
    print(f"  Direct callers: {len(impact['direct_callers'])}")
    print(f"  Affected files: {len(impact['affected_files'])}")
    for f in impact["affected_files"]:
        print(f"    - {f}")

    # 3. Search code
    search = HybridSearch(project_root, graph)

    print("\n--- Searching for 'knowledge graph' ---")
    results = search.search("knowledge graph", max_results=5)
    for r in results:
        print(f"  {r.file_path}:{r.line_number} - {r.line_content.strip()[:80]}")

    print("\n--- Searching for class definitions ---")
    symbols = search.search_symbols("", kind="class", max_results=10)
    for s in symbols:
        print(f"  {s['qualified_name']} at {s['file_path']}:{s['line']}")


if __name__ == "__main__":
    main()
