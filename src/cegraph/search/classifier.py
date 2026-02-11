"""Query classifier for adaptive retrieval routing.

Classifies coding queries into types (EXACT, VAGUE, STRUCTURAL) and
recommends the optimal retrieval strategy.  Uses IDF-weighted symbol hits,
operator intent detection, and entity density — all computed from the
knowledge graph itself (no external models needed).

Instead of using one retrieval method for all queries, classify first,
then dispatch to the optimal strategy.

Classification features:
  1. Symbol hit ratio:  What fraction of query entities match graph symbols?
     High → query names specific code → EXACT
  2. IDF-weighted score:  Matches on rare symbols (low document frequency)
     are stronger signals than matches on common names ("get", "data").
  3. Operator intent:  Structural keywords ("calls", "imports", "extends",
     "inherits", "impact") signal graph-based queries → STRUCTURAL
  4. Entity density:  Ratio of extracted entities to total words.
     Low → mostly natural language → VAGUE

Decision rules:
  - STRUCTURAL if operator intent score > 0.5
  - EXACT if IDF-weighted symbol hit score > threshold AND entity density > 0.3
  - VAGUE otherwise
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx


@dataclass
class QueryClassification:
    """Result of query classification."""

    query_type: str  # "exact", "vague", "structural"
    confidence: float  # 0-1
    recommended_method: str  # "bca", "bm25", "graph"
    features: dict = field(default_factory=dict)


# Structural operator keywords and their weights
_STRUCTURAL_OPERATORS: dict[str, float] = {
    "calls": 1.0,
    "called": 0.9,
    "callers": 1.0,
    "callees": 1.0,
    "imports": 0.8,
    "imported": 0.7,
    "extends": 0.9,
    "inherits": 0.9,
    "implements": 0.9,
    "overrides": 0.8,
    "impact": 1.0,
    "affects": 0.8,
    "affected": 0.8,
    "depends": 0.7,
    "dependency": 0.7,
    "dependencies": 0.7,
    "upstream": 0.6,
    "downstream": 0.6,
    "who": 0.5,  # "who calls X?"
    "what": 0.3,  # "what does X call?"
}


class QueryClassifier:
    """Classifies coding queries into types for retrieval routing.

    Uses the knowledge graph's symbol table as its vocabulary — the same
    graph built by CeGraph's indexer.  No external models or embeddings.

    Usage:
        classifier = QueryClassifier(graph)
        result = classifier.classify("Fix the bug in GraphQuery.who_calls")
        # result.query_type == "exact"
        # result.recommended_method == "bca"
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph
        self._symbol_names: set[str] = set()
        self._name_to_ids: dict[str, list[str]] = {}
        self._idf: dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Pre-compute symbol name index and IDF scores."""
        # Collect all symbol names
        name_freq: Counter[str] = Counter()
        total_symbols = 0

        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "symbol":
                continue
            kind = data.get("kind", "")
            if kind == "import":
                continue

            total_symbols += 1
            name = data.get("name", "").lower()
            qname = data.get("qualified_name", "").lower()

            self._symbol_names.add(name)
            self._name_to_ids.setdefault(name, []).append(node_id)

            # Tokenize name for IDF
            tokens = set(re.findall(r"[a-z]+", name))
            tokens |= set(re.findall(r"[a-z]+", qname))
            for t in tokens:
                name_freq[t] += 1

        # Compute IDF: log(N / df)
        n = max(total_symbols, 1)
        for token, df in name_freq.items():
            self._idf[token] = math.log(n / max(df, 1))

    def classify(self, query: str) -> QueryClassification:
        """Classify a query into type and recommended retrieval method."""
        features = self._extract_features(query)

        # Decision rules
        structural_score = features["structural_score"]
        idf_score = features["idf_weighted_score"]
        entity_density = features["entity_density"]
        symbol_hit_ratio = features["symbol_hit_ratio"]

        entity_count = features["entity_count"]

        if structural_score > 0.5:
            query_type = "structural"
            method = "graph"
            confidence = min(1.0, structural_score)
        elif symbol_hit_ratio >= 0.8 and entity_count >= 1:
            # High symbol match rate — query names specific code symbols
            query_type = "exact"
            method = "bca"
            confidence = min(1.0, symbol_hit_ratio * 0.6 + idf_score * 0.4)
        elif idf_score > 0.3 and entity_density > 0.2:
            query_type = "exact"
            method = "bca"
            confidence = min(1.0, idf_score * 0.7 + symbol_hit_ratio * 0.3)
        else:
            query_type = "vague"
            method = "bm25"
            confidence = min(1.0, 1.0 - entity_density)

        return QueryClassification(
            query_type=query_type,
            confidence=round(confidence, 3),
            recommended_method=method,
            features=features,
        )

    def _extract_features(self, query: str) -> dict:
        """Extract classification features from a query string."""
        query_lower = query.lower()
        words = re.findall(r"\b\w+\b", query_lower)
        total_words = len(words) or 1

        # 1. Entity extraction (lightweight version of BCA Phase 1)
        entities: list[str] = []

        # CamelCase
        for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", query):
            entities.append(m.group(1).lower())

        # snake_case
        for m in re.finditer(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", query):
            name = m.group(1)
            if not any(name.startswith(s) for s in ("the_", "in_the", "to_the")):
                entities.append(name)

        # Dotted paths
        for m in re.finditer(r"\b(\w+\.\w+(?:\.\w+)*)\b", query):
            entities.append(m.group(1).lower())

        # File paths
        for m in re.finditer(r"\b([\w/]+\.(?:py|js|ts|go|rs|java))\b", query):
            entities.append(m.group(1).lower())

        # Quoted strings
        for m in re.finditer(r"[`'\"](\w+(?:\.\w+)*)[`'\"]", query):
            if len(m.group(1)) > 1:
                entities.append(m.group(1).lower())

        entities = list(dict.fromkeys(entities))  # deduplicate, preserve order

        # 2. Symbol hit ratio
        hits = 0
        for ent in entities:
            ent_lower = ent.lower()
            # Check exact and partial matches
            if ent_lower in self._symbol_names:
                hits += 1
            elif any(ent_lower in sn for sn in self._symbol_names):
                hits += 0.5

        symbol_hit_ratio = hits / max(len(entities), 1)

        # 3. IDF-weighted score
        idf_score = 0.0
        query_tokens = set(re.findall(r"[a-z]+", query_lower))
        max_possible_idf = 0.0
        for token in query_tokens:
            idf = self._idf.get(token, 0)
            if idf > 0:
                idf_score += idf
            # Max possible = if all tokens had max IDF
            max_possible_idf += max(self._idf.values()) if self._idf else 1

        idf_score = idf_score / max(max_possible_idf, 1)

        # 4. Structural operator detection
        structural_score = 0.0
        for word in words:
            if word in _STRUCTURAL_OPERATORS:
                structural_score += _STRUCTURAL_OPERATORS[word]

        # Normalize: cap at 1.0
        structural_score = min(1.0, structural_score / 1.5)

        # 5. Entity density
        entity_density = len(entities) / total_words

        return {
            "entities": entities,
            "entity_count": len(entities),
            "total_words": total_words,
            "entity_density": round(entity_density, 3),
            "symbol_hits": hits,
            "symbol_hit_ratio": round(symbol_hit_ratio, 3),
            "idf_weighted_score": round(idf_score, 3),
            "structural_score": round(structural_score, 3),
        }
