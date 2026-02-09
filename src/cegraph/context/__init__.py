"""Budgeted Context Assembly (BCA).

Assembles dependency-ordered, relevance-scored context packages from a code
knowledge graph within a token budget.

Usage:
    from cegraph.context import ContextAssembler, ContextPackage

    assembler = ContextAssembler(root, graph, query)
    package = assembler.assemble("fix the login bug", token_budget=8000)
    print(package.render())
"""

from cegraph.context.engine import ContextAssembler
from cegraph.context.models import ContextPackage, ContextItem, ContextStrategy

__all__ = ["ContextAssembler", "ContextPackage", "ContextItem", "ContextStrategy"]
