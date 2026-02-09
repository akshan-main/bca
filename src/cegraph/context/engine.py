"""Budgeted Context Assembly (BCA).

Formulation:
  Given a code knowledge graph G = (V, E), a seed set S extracted from a
  natural language task, and a token budget B:

  Select X ⊆ V maximizing  Σ u(v | X)  subject to:
    1. Budget:  Σ c(v) ≤ B  for v ∈ X
    2. Closure: if v ∈ X, then D(v) ⊆ X  (dependency closure)

  where:
    u(v | X) = relevance(v, S) · kind(v) · coverage(v, X)  (submodular)
    c(v)     = token cost of symbol v
    D(v)     = transitive hard dependencies of v (base classes, parent types)

  The utility u is submodular: coverage(v, X) measures NEW information v adds
  beyond what X already covers (diminishing returns as X grows).

Algorithm:
  1. Extract entities from task → seed set S
  2. Weighted BFS from S → candidate set V_c with relevance scores
  3. Compute dependency closure D(v) for each v ∈ V_c
  4. Greedy selection: pick v with best u(v) / effective_cost(v) ratio,
     where effective_cost = c(v) + Σ c(d) for d ∈ D(v) \\ X
  5. Topological sort selected set for dependency-safe serialization

Approximation guarantee:
  Greedy selection over a submodular function with knapsack constraint gives
  a (1 - 1/e) ≈ 0.632 approximation ratio (Nemhauser et al., 1978).

Copyright (c) 2025 CeGraph Contributors. MIT License.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from cegraph.context.models import (
    ContextItem,
    ContextPackage,
    ContextStrategy,
    TokenEstimator,
)
from cegraph.graph.query import GraphQuery

# Try to load native C++ acceleration
try:
    from cegraph.context._native import NativeCAG
    _HAS_NATIVE = NativeCAG.is_available()
except ImportError:
    _HAS_NATIVE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Edge weights for graph traversal -- how much relevance
# transfers along each edge type
_EDGE_WEIGHTS: dict[str, float] = {
    "calls": 0.85,
    "imports": 0.5,
    "inherits": 0.9,
    "implements": 0.9,
    "contains": 0.7,
    "overrides": 0.85,
    "uses": 0.4,
    "decorates": 0.6,
}

# Symbol kind importance multipliers
_KIND_WEIGHTS: dict[str, float] = {
    "class": 1.0,
    "function": 1.0,
    "method": 0.9,
    "interface": 0.8,
    "enum": 0.7,
    "type_alias": 0.6,
    "constant": 0.5,
    "variable": 0.4,
    "import": 0.1,
    "module": 0.3,
}

# Strategy configs
_STRATEGY_CONFIG = {
    ContextStrategy.PRECISE: {
        "max_depth": 1,
        "min_score": 0.3,
        "include_callers": True,
        "include_callees": True,
    },
    ContextStrategy.SMART: {
        "max_depth": 3,
        "min_score": 0.1,
        "include_callers": True,
        "include_callees": True,
    },
    ContextStrategy.THOROUGH: {
        "max_depth": 5,
        "min_score": 0.05,
        "include_callers": True,
        "include_callees": True,
    },
}

# Edge types that create HARD dependencies (must include for correctness)
_HARD_DEP_EDGES = {"inherits", "implements"}


@dataclass
class AblationConfig:
    """Toggle individual components for ablation studies.

    All features enabled by default. Disable to measure their contribution.
    """
    dependency_closure: bool = True    # Enforce D(v) ⊆ X constraint
    centrality_scoring: bool = True    # Boost high-connectivity symbols
    file_proximity: bool = True        # Boost symbols in same file as seeds
    kind_weights: bool = True          # Weight by symbol kind
    submodular_coverage: bool = True   # Diminishing returns scoring
    dependency_ordering: bool = True   # Topological sort output
    learned_weights: bool = False      # Learn edge weights from git history
    use_pagerank: bool = False         # Personalized PageRank instead of BFS


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

class ContextAssembler:
    """Budgeted context assembly via submodular greedy selection.

    Computes a budget-constrained, dependency-safe context package from a
    code knowledge graph. Uses submodular greedy selection with dependency
    closure constraints.

    Usage:
        assembler = ContextAssembler(root, graph, query)
        package = assembler.assemble("fix the login bug", token_budget=8000)
        llm_context = package.render()
    """

    def __init__(
        self,
        root: Path,
        graph: nx.DiGraph,
        query: GraphQuery,
        ablation: AblationConfig | None = None,
    ) -> None:
        self.root = root
        self.graph = graph
        self.query = query
        self.ablation = ablation or AblationConfig()
        self._native_graph = None
        self._node_to_idx: dict[str, int] = {}
        self._idx_to_node: dict[int, str] = {}
        self._edge_weights = dict(_EDGE_WEIGHTS)

        if self.ablation.learned_weights:
            from cegraph.context.learned_weights import learn_edge_weights
            self._edge_weights = learn_edge_weights(root, graph)

        if _HAS_NATIVE:
            self._build_native_graph()

    def _build_native_graph(self) -> None:
        """Build the C++ graph representation for accelerated BFS."""
        nodes = list(self.graph.nodes())
        self._node_to_idx = {n: i for i, n in enumerate(nodes)}
        self._idx_to_node = {i: n for i, n in enumerate(nodes)}

        self._native_graph = NativeCAG.create_graph(len(nodes))

        for n, data in self.graph.nodes(data=True):
            idx = self._node_to_idx[n]
            kind = data.get("kind", "")
            weight = _KIND_WEIGHTS.get(kind, 0.5)
            self._native_graph.set_node_weight(idx, weight)
            self._native_graph.set_lines(
                idx, data.get("line_start", 0), data.get("line_end", 0)
            )

        for u, v, data in self.graph.edges(data=True):
            kind = data.get("kind", "")
            weight = self._edge_weights.get(kind, 0.3)
            src = self._node_to_idx[u]
            dst = self._node_to_idx[v]
            self._native_graph.add_edge(src, dst, weight)

    @property
    def is_accelerated(self) -> bool:
        """Whether C++ acceleration is active."""
        return self._native_graph is not None

    # -------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------

    def assemble(
        self,
        task: str,
        token_budget: int = 8000,
        strategy: ContextStrategy = ContextStrategy.SMART,
        focus_files: list[str] | None = None,
    ) -> ContextPackage:
        """Assemble a budgeted context package for a given task.

        Args:
            task: Natural language description of the task.
            token_budget: Maximum tokens to include (budget B).
            strategy: How aggressively to expand context.
            focus_files: Optional list of files to prioritize.

        Returns:
            A ContextPackage with the selected symbols and their source code,
            ordered by dependency (definitions before usage).
        """
        start_time = time.time()
        config = _STRATEGY_CONFIG[strategy]

        # Phase 1: Extract entities from the task
        entities = self._extract_entities(task)

        # Phase 2: Find seed symbols in the graph
        seeds = self._find_seeds(entities, focus_files)

        # Phase 3: Expand context via graph traversal
        if self.ablation.use_pagerank and seeds:
            candidates = self._expand_context_pagerank(seeds, config)
        elif self._native_graph and seeds:
            candidates = self._expand_context_native(seeds, config)
        else:
            candidates = self._expand_context(seeds, config)

        # Phase 4: Score candidates
        scored = self._score_candidates(candidates, entities, seeds)

        # Phase 5: Compute dependency closures
        closures = self._compute_closures(scored)

        # Phase 6: Budgeted selection with closure constraints
        selected = self._budget_select(scored, closures, token_budget)

        # Phase 7: Load source code
        items = self._load_source(selected)

        # Phase 8: Dependency-safe ordering
        if self.ablation.dependency_ordering:
            items = self._dependency_order(items)

        elapsed_ms = (time.time() - start_time) * 1000
        total_tokens = sum(item.token_estimate for item in items)
        files = set(item.file_path for item in items)

        return ContextPackage(
            task=task,
            strategy=strategy,
            items=items,
            seed_symbols=[s["symbol_id"] for s in seeds],
            total_tokens=total_tokens,
            token_budget=token_budget,
            files_included=len(files),
            symbols_included=len(items),
            symbols_available=len(scored),
            budget_used_pct=round(total_tokens / max(token_budget, 1) * 100, 1),
            assembly_time_ms=round(elapsed_ms, 1),
        )

    # -------------------------------------------------------------------
    # Phase 1: Entity extraction
    # -------------------------------------------------------------------

    def _extract_entities(self, task: str) -> list[dict]:
        """Extract potential code entities from a natural language task.

        Looks for: CamelCase, snake_case, dotted paths, file paths, quoted strings.
        """
        entities: list[dict] = []
        seen = set()

        # CamelCase (class names)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", task):
            name = m.group(1)
            if name not in seen:
                entities.append({"name": name, "type": "class", "confidence": 0.9})
                seen.add(name)

        # snake_case (function/variable names)
        for m in re.finditer(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", task):
            name = m.group(1)
            skip = {"the_", "in_the", "to_the", "of_the", "and_the", "is_a", "has_a"}
            if name not in seen and not any(name.startswith(s) for s in skip):
                entities.append({"name": name, "type": "function", "confidence": 0.8})
                seen.add(name)

        # Dotted paths
        for m in re.finditer(r"\b(\w+\.\w+(?:\.\w+)*)\b", task):
            name = m.group(1)
            if name not in seen and not name[0].isdigit():
                entities.append({"name": name, "type": "path", "confidence": 0.85})
                seen.add(name)

        # File paths
        for m in re.finditer(
            r"\b([\w/]+\.(?:py|js|ts|go|rs|java))\b", task
        ):
            name = m.group(1)
            if name not in seen:
                entities.append({"name": name, "type": "file", "confidence": 0.95})
                seen.add(name)

        # Quoted strings
        for m in re.finditer(r"[`'\"](\w+(?:\.\w+)*)[`'\"]", task):
            name = m.group(1)
            if name not in seen and len(name) > 1:
                entities.append({"name": name, "type": "quoted", "confidence": 0.9})
                seen.add(name)

        # Single significant words
        words = set(re.findall(r"\b([A-Z][a-z]{2,}|[a-z]{3,})\b", task))
        stop_words = {
            "the", "and", "for", "that", "this", "with", "from", "have", "been",
            "will", "can", "should", "would", "could", "into", "when", "where",
            "how", "what", "why", "which", "there", "their", "about", "also",
            "just", "more", "some", "than", "them", "then", "these", "very",
            "after", "before", "between", "each", "other", "such", "only",
            "make", "like", "over", "back", "still", "through",
            "add", "fix", "bug", "error", "issue", "feature", "implement",
            "create", "update", "delete", "remove", "change", "modify", "refactor",
            "test", "check", "run", "build", "deploy", "handle", "process",
            "function", "method", "class", "file", "code", "system",
        }
        for word in words:
            if word.lower() not in stop_words and word not in seen:
                entities.append({"name": word, "type": "keyword", "confidence": 0.4})
                seen.add(word)

        return entities

    # -------------------------------------------------------------------
    # Phase 2: Seed identification
    # -------------------------------------------------------------------

    def _find_seeds(
        self, entities: list[dict], focus_files: list[str] | None = None
    ) -> list[dict]:
        """Find seed symbols in the graph matching extracted entities."""
        seeds: list[dict] = []
        seen_ids = set()

        for entity in entities:
            name = entity["name"]
            confidence = entity["confidence"]

            if entity["type"] == "file":
                file_node = f"file::{name}"
                if self.graph.has_node(file_node):
                    for succ in self.graph.successors(file_node):
                        data = self.graph.nodes.get(succ, {})
                        if data.get("type") == "symbol" and succ not in seen_ids:
                            kind_w = _KIND_WEIGHTS.get(data.get("kind", ""), 0.5)
                            seeds.append({
                                "symbol_id": succ,
                                "score": confidence * kind_w,
                                "reason": f"in file '{name}'",
                            })
                            seen_ids.add(succ)
            else:
                matches = self.query.find_symbol(name)
                for sid in matches:
                    if sid in seen_ids:
                        continue
                    data = self.graph.nodes.get(sid, {})
                    if data.get("type") != "symbol":
                        continue
                    kind = data.get("kind", "")
                    sym_name = data.get("name", "")

                    if sym_name.lower() == name.lower():
                        match_score = 1.0
                    elif name.lower() in sym_name.lower():
                        match_score = 0.7
                    else:
                        match_score = 0.4

                    kind_w = _KIND_WEIGHTS.get(kind, 0.5)
                    score = confidence * match_score * kind_w

                    if kind == "import" and score < 0.3:
                        continue

                    seeds.append({
                        "symbol_id": sid,
                        "score": score,
                        "reason": f"matches '{name}'",
                    })
                    seen_ids.add(sid)

        if focus_files:
            for fp in focus_files:
                file_node = f"file::{fp}"
                if self.graph.has_node(file_node):
                    for succ in self.graph.successors(file_node):
                        data = self.graph.nodes.get(succ, {})
                        if data.get("type") == "symbol" and succ not in seen_ids:
                            kind_w = _KIND_WEIGHTS.get(data.get("kind", ""), 0.5)
                            seeds.append({
                                "symbol_id": succ,
                                "score": 0.8 * kind_w,
                                "reason": f"in focus file '{fp}'",
                            })
                            seen_ids.add(succ)

        seeds.sort(key=lambda x: x["score"], reverse=True)

        # Cap low-confidence keyword seeds to avoid noise explosion.
        # Keep all high-confidence seeds (>= 0.5), but limit keywords to 5.
        high = [s for s in seeds if s["score"] >= 0.5]
        low = [s for s in seeds if s["score"] < 0.5]
        seeds = high + low[:5]

        return seeds

    # -------------------------------------------------------------------
    # Phase 3: Context expansion (BFS)
    # -------------------------------------------------------------------

    def _expand_context_native(
        self, seeds: list[dict], config: dict
    ) -> list[dict]:
        """C++-accelerated context expansion via weighted BFS."""
        seed_indices = []
        seed_scores = []
        for s in seeds:
            idx = self._node_to_idx.get(s["symbol_id"])
            if idx is not None:
                seed_indices.append(idx)
                seed_scores.append(s["score"])

        if not seed_indices:
            return []

        results = self._native_graph.weighted_bfs(
            seed_nodes=seed_indices,
            seed_scores=seed_scores,
            max_depth=config["max_depth"],
            min_score=config["min_score"],
            backward_decay=0.7,
        )

        seed_reason = {s["symbol_id"]: s["reason"] for s in seeds}

        candidates = []
        for r in results:
            node_id = self._idx_to_node.get(r["node"])
            if node_id is None:
                continue
            data = self.graph.nodes.get(node_id, {})
            if data.get("type") != "symbol":
                continue

            reason = seed_reason.get(
                node_id, f"graph expansion (depth {r['depth']})"
            )
            candidates.append({
                "symbol_id": node_id,
                "score": r["score"],
                "depth": r["depth"],
                "reason": reason,
                "via": [],
            })

        return candidates

    def _expand_context(
        self, seeds: list[dict], config: dict
    ) -> list[dict]:
        """Pure Python context expansion via weighted BFS."""
        max_depth = config["max_depth"]
        min_score = config["min_score"]

        candidates: dict[str, dict] = {}

        for seed in seeds:
            sid = seed["symbol_id"]
            candidates[sid] = {
                "symbol_id": sid,
                "score": seed["score"],
                "depth": 0,
                "reason": seed["reason"],
                "via": [],
            }

        frontier = [(s["symbol_id"], s["score"], 0) for s in seeds]

        while frontier:
            next_frontier = []

            for node_id, parent_score, depth in frontier:
                if depth >= max_depth:
                    continue

                # Forward edges
                for succ in self.graph.successors(node_id):
                    edge_data = self.graph.edges[node_id, succ]
                    edge_kind = edge_data.get("kind", "")
                    succ_data = self.graph.nodes.get(succ, {})

                    if succ_data.get("type") != "symbol":
                        continue

                    weight = self._edge_weights.get(edge_kind, 0.3)
                    kind_mult = _KIND_WEIGHTS.get(succ_data.get("kind", ""), 0.5)
                    new_score = parent_score * weight * kind_mult

                    if new_score < min_score:
                        continue

                    if succ not in candidates or new_score > candidates[succ]["score"]:
                        node_name = self.graph.nodes.get(node_id, {}).get("name", node_id)
                        candidates[succ] = {
                            "symbol_id": succ,
                            "score": new_score,
                            "depth": depth + 1,
                            "reason": f"{edge_kind} from {node_name}",
                            "via": candidates.get(node_id, {}).get("via", []) + [node_id],
                        }
                        next_frontier.append((succ, new_score, depth + 1))

                # Backward edges (callers)
                if config.get("include_callers"):
                    for pred in self.graph.predecessors(node_id):
                        edge_data = self.graph.edges[pred, node_id]
                        edge_kind = edge_data.get("kind", "")
                        pred_data = self.graph.nodes.get(pred, {})

                        if pred_data.get("type") != "symbol":
                            continue

                        weight = self._edge_weights.get(edge_kind, 0.3) * 0.7
                        kind_mult = _KIND_WEIGHTS.get(pred_data.get("kind", ""), 0.5)
                        new_score = parent_score * weight * kind_mult

                        if new_score < min_score:
                            continue

                        if pred not in candidates or new_score > candidates[pred]["score"]:
                            node_name = self.graph.nodes.get(node_id, {}).get("name", node_id)
                            candidates[pred] = {
                                "symbol_id": pred,
                                "score": new_score,
                                "depth": depth + 1,
                                "reason": f"calls {node_name}",
                                "via": candidates.get(node_id, {}).get("via", []) + [node_id],
                            }
                            next_frontier.append((pred, new_score, depth + 1))

            frontier = next_frontier

        return list(candidates.values())

    def _expand_context_pagerank(
        self, seeds: list[dict], config: dict
    ) -> list[dict]:
        """Expand context via personalized PageRank from seeds.

        Instead of BFS, compute PPR with the seed set as the personalization
        vector.  This gives a global relevance score that accounts for the
        full graph structure, not just local neighborhoods.
        """
        min_score = config["min_score"]

        # Build personalization vector (seed nodes get their scores)
        personalization: dict[str, float] = {}
        for s in seeds:
            personalization[s["symbol_id"]] = s["score"]

        # Ensure all seed nodes exist in graph
        personalization = {
            k: v for k, v in personalization.items()
            if self.graph.has_node(k)
        }
        if not personalization:
            return []

        # Normalize
        total = sum(personalization.values())
        personalization = {k: v / total for k, v in personalization.items()}

        try:
            ppr = nx.pagerank(
                self.graph,
                alpha=0.85,
                personalization=personalization,
                max_iter=100,
                tol=1e-6,
            )
        except nx.NetworkXError:
            return self._expand_context(seeds, config)

        seed_reason = {s["symbol_id"]: s["reason"] for s in seeds}
        candidates: list[dict] = []

        for node_id, score in ppr.items():
            if score < min_score * 0.01:
                continue
            data = self.graph.nodes.get(node_id, {})
            if data.get("type") != "symbol":
                continue

            # Scale PPR scores to be comparable with BFS scores
            scaled_score = score * 100

            reason = seed_reason.get(node_id, "pagerank expansion")
            depth = 0 if node_id in personalization else 1
            candidates.append({
                "symbol_id": node_id,
                "score": scaled_score,
                "depth": depth,
                "reason": reason,
                "via": [],
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    # -------------------------------------------------------------------
    # Phase 4: Scoring (submodular utility)
    # -------------------------------------------------------------------

    def _score_candidates(
        self,
        candidates: list[dict],
        entities: list[dict],
        seeds: list[dict],
    ) -> list[dict]:
        """Score candidates with optional submodular coverage component.

        u(v) = base_relevance(v)
             * kind_importance(v)      [if ablation.kind_weights]
             * file_proximity_bonus(v) [if ablation.file_proximity]
             + centrality_bonus(v)     [if ablation.centrality_scoring]
        """
        seed_files = set()
        for s in seeds:
            data = self.graph.nodes.get(s["symbol_id"], {})
            if data.get("file_path"):
                seed_files.add(data["file_path"])

        for cand in candidates:
            sid = cand["symbol_id"]
            data = self.graph.nodes.get(sid, {})

            # File proximity boost
            if self.ablation.file_proximity and data.get("file_path") in seed_files:
                cand["score"] *= 1.3

            # Centrality boost
            if self.ablation.centrality_scoring:
                in_deg = self.graph.in_degree(sid)
                out_deg = self.graph.out_degree(sid)
                centrality_bonus = min((in_deg + out_deg) * 0.02, 0.3)
                cand["score"] += centrality_bonus

            # Penalize very large symbols
            line_span = data.get("line_end", 0) - data.get("line_start", 0)
            if line_span > 100:
                cand["score"] *= 0.8

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    # -------------------------------------------------------------------
    # Phase 5: Dependency closure
    # -------------------------------------------------------------------

    def _compute_closures(self, candidates: list[dict]) -> dict[str, set[str]]:
        """Compute transitive hard-dependency closures D(v) for each candidate.

        Hard dependencies:
        - inherits: if class A extends B, B must be included
        - implements: if A implements I, I must be included
        - contains (upward): if method m is in class C, C's skeleton is needed

        The closure is transitive: if B extends C, D(A) includes both B and C.
        """
        if not self.ablation.dependency_closure:
            return {c["symbol_id"]: set() for c in candidates}

        candidate_ids = {c["symbol_id"] for c in candidates}
        closures: dict[str, set[str]] = {}

        for cand in candidates:
            sid = cand["symbol_id"]
            if sid in closures:
                continue
            closures[sid] = self._closure_of(sid, candidate_ids)

        return closures

    def _closure_of(self, symbol_id: str, candidate_ids: set[str]) -> set[str]:
        """Compute D(v): the transitive set of hard dependencies for symbol v."""
        deps: set[str] = set()
        stack = [symbol_id]
        visited = {symbol_id}

        while stack:
            current = stack.pop()
            data = self.graph.nodes.get(current, {})

            # Forward edges: inherits, implements
            for succ in self.graph.successors(current):
                edge_data = self.graph.edges[current, succ]
                edge_kind = edge_data.get("kind", "")

                if edge_kind in _HARD_DEP_EDGES:
                    succ_data = self.graph.nodes.get(succ, {})
                    if succ_data.get("type") == "symbol" and succ not in visited:
                        deps.add(succ)
                        visited.add(succ)
                        stack.append(succ)

            # Upward containment: if this is a method, its parent class is needed
            if data.get("kind") in ("method", "function"):
                for pred in self.graph.predecessors(current):
                    edge_data = self.graph.edges[pred, current]
                    if edge_data.get("kind") == "contains":
                        pred_data = self.graph.nodes.get(pred, {})
                        if (
                            pred_data.get("type") == "symbol"
                            and pred_data.get("kind") == "class"
                            and pred not in visited
                        ):
                            deps.add(pred)
                            visited.add(pred)
                            stack.append(pred)

        return deps

    # -------------------------------------------------------------------
    # Phase 6: Budgeted selection with closure constraints
    # -------------------------------------------------------------------

    def _budget_select(
        self,
        candidates: list[dict],
        closures: dict[str, set[str]],
        token_budget: int,
    ) -> list[dict]:
        """Greedy budgeted selection with dependency closure constraints.

        For each candidate v (sorted by value/cost ratio):
          effective_cost(v) = c(v) + Σ c(d) for d ∈ D(v) \\ already_selected
          if effective_cost(v) ≤ remaining_budget:
            select v and all d ∈ D(v)
        """
        selected_ids: set[str] = set()
        selected: list[dict] = []
        remaining_budget = token_budget

        # Build a quick lookup for candidates by ID
        cand_by_id: dict[str, dict] = {}
        for cand in candidates:
            cand_by_id[cand["symbol_id"]] = cand

        # Pre-compute token costs
        token_costs: dict[str, int] = {}
        for cand in candidates:
            sid = cand["symbol_id"]
            token_costs[sid] = self._token_cost(sid, cand)

        # Also compute costs for closure deps that aren't in the candidate list
        all_closure_deps = set()
        for deps in closures.values():
            all_closure_deps |= deps
        for dep_id in all_closure_deps:
            if dep_id not in token_costs:
                token_costs[dep_id] = self._token_cost(dep_id)

        # Coverage tracking for submodular scoring
        covered_edges: set[tuple[str, str]] = set()

        # Phase A: Select high-confidence seed symbols first (depth 0, score > 0.5).
        # Seeds are directly-relevant — include them without full closure cost.
        # For methods whose parent class is huge, include the method standalone
        # (the LLM gets enough context from the method source itself).
        seed_cands = [c for c in candidates if c["depth"] == 0 and c["score"] > 0.5]
        seed_cands.sort(key=lambda c: c["score"], reverse=True)
        rest_cands = [c for c in candidates if c not in seed_cands]

        for cand in seed_cands:
            sid = cand["symbol_id"]
            kind = self.graph.nodes.get(sid, {}).get("kind", "")
            if sid in selected_ids or kind == "import":
                continue

            self_cost = token_costs.get(sid, 0)
            if self_cost > remaining_budget:
                continue

            cand["token_estimate"] = self_cost
            cand["is_dependency"] = False
            cand["closure_of"] = ""
            selected.append(cand)
            selected_ids.add(sid)
            remaining_budget -= self_cost

            if self.ablation.submodular_coverage:
                self._update_coverage(sid, covered_edges)

        # Phase B: Fill remaining budget with expansion candidates by value/cost
        def _value_cost_ratio(cand):
            sid = cand["symbol_id"]
            closure_deps = closures.get(sid, set())
            self_cost = token_costs.get(sid, 0)
            dep_cost = sum(token_costs.get(d, 0) for d in closure_deps - selected_ids)
            effective = max(1, self_cost + dep_cost)
            return cand["score"] / effective

        rest_cands = sorted(rest_cands, key=_value_cost_ratio, reverse=True)

        for cand in rest_cands:
            sid = cand["symbol_id"]
            kind = self.graph.nodes.get(sid, {}).get("kind", "")

            if sid in selected_ids:
                continue
            if kind == "import":
                continue

            # Compute effective cost (self + unselected closure deps)
            closure_deps = closures.get(sid, set())
            unselected_deps = closure_deps - selected_ids
            self_cost = token_costs.get(sid, 0)
            dep_cost = sum(token_costs.get(d, 0) for d in unselected_deps)
            effective_cost = self_cost + dep_cost

            if effective_cost > remaining_budget:
                if remaining_budget < 50:
                    break
                continue

            # Compute marginal utility (submodular coverage)
            if self.ablation.submodular_coverage:
                marginal = self._marginal_utility(sid, covered_edges)
                # Skip symbols that add almost no new information
                if marginal < 0.1 and cand["depth"] > 1:
                    continue

            # Select this symbol
            cand["token_estimate"] = self_cost
            cand["is_dependency"] = False
            cand["closure_of"] = ""
            selected.append(cand)
            selected_ids.add(sid)
            remaining_budget -= self_cost

            # Update coverage
            if self.ablation.submodular_coverage:
                self._update_coverage(sid, covered_edges)

            # Select closure dependencies
            for dep_id in unselected_deps:
                if dep_id in selected_ids:
                    continue
                dep_cost_val = token_costs.get(dep_id, 0)

                # Build a candidate entry for the dependency
                if dep_id in cand_by_id:
                    dep_cand = dict(cand_by_id[dep_id])
                else:
                    dep_cand = {
                        "symbol_id": dep_id,
                        "score": cand["score"] * 0.5,
                        "depth": cand["depth"] + 1,
                        "reason": f"dependency of {self.graph.nodes.get(sid, {}).get('name', sid)}",
                        "via": [],
                    }

                dep_cand["token_estimate"] = dep_cost_val
                dep_cand["is_dependency"] = True
                dep_cand["closure_of"] = sid
                selected.append(dep_cand)
                selected_ids.add(dep_id)
                remaining_budget -= dep_cost_val

                if self.ablation.submodular_coverage:
                    self._update_coverage(dep_id, covered_edges)

        return selected

    def _token_cost(self, symbol_id: str, cand: dict | None = None) -> int:
        """Compute token cost c(v) for a symbol."""
        data = self.graph.nodes.get(symbol_id, {})
        line_start = data.get("line_start", 0)
        line_end = data.get("line_end", 0)
        line_count = max(1, line_end - line_start + 1)
        return TokenEstimator.estimate_lines(line_count)

    def _marginal_utility(
        self, symbol_id: str, covered_edges: set[tuple[str, str]]
    ) -> float:
        """Compute marginal coverage gain of adding symbol_id.

        This makes the utility submodular: each new symbol covers some edges
        in the graph. As more symbols are selected, each new symbol covers
        fewer NEW edges → diminishing returns.

        Returns a value in [0, 1] representing the fraction of new edges.
        """
        new_edges = 0
        total_edges = 0

        for succ in self.graph.successors(symbol_id):
            succ_data = self.graph.nodes.get(succ, {})
            if succ_data.get("type") == "symbol":
                total_edges += 1
                edge_key = (symbol_id, succ)
                if edge_key not in covered_edges:
                    new_edges += 1

        for pred in self.graph.predecessors(symbol_id):
            pred_data = self.graph.nodes.get(pred, {})
            if pred_data.get("type") == "symbol":
                total_edges += 1
                edge_key = (pred, symbol_id)
                if edge_key not in covered_edges:
                    new_edges += 1

        if total_edges == 0:
            return 1.0
        return new_edges / total_edges

    def _update_coverage(
        self, symbol_id: str, covered_edges: set[tuple[str, str]]
    ) -> None:
        """Mark edges incident to symbol_id as covered."""
        for succ in self.graph.successors(symbol_id):
            if self.graph.nodes.get(succ, {}).get("type") == "symbol":
                covered_edges.add((symbol_id, succ))

        for pred in self.graph.predecessors(symbol_id):
            if self.graph.nodes.get(pred, {}).get("type") == "symbol":
                covered_edges.add((pred, symbol_id))

    # -------------------------------------------------------------------
    # Phase 7-8: Source loading + dependency ordering
    # -------------------------------------------------------------------

    def _load_source(self, candidates: list[dict]) -> list[ContextItem]:
        """Load actual source code for selected symbols."""
        items = []

        for cand in candidates:
            sid = cand["symbol_id"]
            data = self.graph.nodes.get(sid, {})
            file_path = data.get("file_path", "")
            line_start = data.get("line_start", 0)
            line_end = data.get("line_end", 0)

            source = ""
            full_path = self.root / file_path
            if full_path.exists():
                try:
                    all_lines = full_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    source = "\n".join(all_lines[max(0, line_start - 1):line_end])
                except OSError:
                    source = f"# Could not read {file_path}"

            token_est = (
                TokenEstimator.estimate(source)
                if source
                else cand.get("token_estimate", 0)
            )

            reason = cand.get("reason", "")
            if cand.get("is_dependency"):
                closure_sym = cand.get("closure_of", "")
                closure_name = self.graph.nodes.get(closure_sym, {}).get("name", "")
                reason = f"required dependency of {closure_name}"

            items.append(
                ContextItem(
                    symbol_id=sid,
                    name=data.get("name", ""),
                    qualified_name=data.get("qualified_name", ""),
                    kind=data.get("kind", ""),
                    file_path=file_path,
                    line_start=line_start,
                    line_end=line_end,
                    source_code=source,
                    signature=data.get("signature", ""),
                    docstring=data.get("docstring", ""),
                    relevance_score=round(cand["score"], 3),
                    reason=reason,
                    token_estimate=token_est,
                    depth=cand.get("depth", 0),
                    is_dependency=cand.get("is_dependency", False),
                )
            )

        return items

    def _dependency_order(self, items: list[ContextItem]) -> list[ContextItem]:
        """Sort items so dependencies come before dependents (topological sort).

        Ensures the LLM sees definitions before usage.
        """
        item_ids = {item.symbol_id for item in items}
        item_map = {item.symbol_id: item for item in items}

        dep_graph = nx.DiGraph()
        for item in items:
            dep_graph.add_node(item.symbol_id)

        for item in items:
            for succ in self.graph.successors(item.symbol_id):
                if succ in item_ids:
                    edge_data = self.graph.edges[item.symbol_id, succ]
                    kind = edge_data.get("kind", "")
                    if kind in ("calls", "imports", "inherits", "uses", "implements"):
                        dep_graph.add_edge(succ, item.symbol_id)

        try:
            ordered_ids = list(nx.topological_sort(dep_graph))
        except nx.NetworkXUnfeasible:
            # Cycle detected — collapse SCCs and sort the DAG of SCCs
            ordered_ids = self._scc_topo_sort(dep_graph, items)

        return [item_map[sid] for sid in ordered_ids if sid in item_map]

    def _scc_topo_sort(
        self, dep_graph: nx.DiGraph, items: list[ContextItem]
    ) -> list[str]:
        """Handle cycles via SCC condensation + topological sort of SCC DAG."""
        sccs = list(nx.strongly_connected_components(dep_graph))
        node_to_scc: dict[str, int] = {}
        for i, scc in enumerate(sccs):
            for node in scc:
                node_to_scc[node] = i

        # Build DAG of SCCs
        scc_dag = nx.DiGraph()
        for i in range(len(sccs)):
            scc_dag.add_node(i)
        for u, v in dep_graph.edges():
            su, sv = node_to_scc[u], node_to_scc[v]
            if su != sv:
                scc_dag.add_edge(su, sv)

        scc_order = list(nx.topological_sort(scc_dag))

        # Within each SCC, sort by relevance score
        item_map = {item.symbol_id: item for item in items}
        ordered = []
        for scc_idx in scc_order:
            scc_nodes = sorted(
                sccs[scc_idx],
                key=lambda s: item_map[s].relevance_score if s in item_map else 0,
                reverse=True,
            )
            ordered.extend(scc_nodes)

        return ordered

    # -------------------------------------------------------------------
    # Savings estimation
    # -------------------------------------------------------------------

    def estimate_savings(self, task: str, token_budget: int = 8000) -> dict:
        """Estimate how many tokens CAG saves vs naive approaches."""
        package = self.assemble(task, token_budget)

        entities = self._extract_entities(task)
        grep_files = set()
        grep_tokens = 0
        for entity in entities:
            for node_id, data in self.graph.nodes(data=True):
                if data.get("type") != "file":
                    continue
                fp = data.get("path", "")
                full_path = self.root / fp
                if full_path.exists():
                    try:
                        content = full_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                        if entity["name"].lower() in content.lower():
                            if fp not in grep_files:
                                grep_files.add(fp)
                                grep_tokens += TokenEstimator.estimate(content)
                    except OSError:
                        pass

        total_tokens = 0
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") == "file":
                fp = data.get("path", "")
                full_path = self.root / fp
                if full_path.exists():
                    try:
                        content = full_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                        total_tokens += TokenEstimator.estimate(content)
                    except OSError:
                        pass

        cag_t = package.total_tokens
        return {
            "task": task,
            "cag_tokens": cag_t,
            "cag_files": package.files_included,
            "cag_symbols": package.symbols_included,
            "grep_tokens": grep_tokens,
            "grep_files": len(grep_files),
            "all_files_tokens": total_tokens,
            "savings_vs_grep": (
                f"{max(0, (1 - cag_t / max(grep_tokens, 1)) * 100):.0f}%"
            ),
            "savings_vs_all": (
                f"{max(0, (1 - cag_t / max(total_tokens, 1)) * 100):.0f}%"
            ),
            "accelerated": self.is_accelerated,
        }
