"""Hybrid search combining lexical and optional semantic search."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from cegraph.search.lexical import LexicalSearch, SearchResult


class HybridSearch:
    """Combined search engine.

    Uses lexical search as the baseline, and adds semantic search
    when embeddings are available (optional numpy dependency).
    """

    def __init__(self, root: Path, graph: nx.DiGraph | None = None) -> None:
        self.root = root
        self.graph = graph
        self.lexical = LexicalSearch(root, graph)
        self._embeddings: dict[str, list[float]] | None = None

    def search(
        self,
        query: str,
        file_pattern: str = "",
        max_results: int = 20,
        context_lines: int = 2,
    ) -> list[SearchResult]:
        """Run hybrid search combining lexical and semantic results."""
        lexical_results = self.lexical.search(
            query, file_pattern, max_results=max_results * 2, context_lines=context_lines
        )

        # If we have embeddings, boost with semantic similarity
        if self._embeddings:
            semantic_boost = self._semantic_scores(query)
            for result in lexical_results:
                key = f"{result.file_path}:{result.line_number}"
                boost = semantic_boost.get(key, 0.0)
                result.score += boost * 0.5  # Weight semantic at 50% of lexical

            lexical_results.sort(key=lambda r: r.score, reverse=True)

        return lexical_results[:max_results]

    def search_symbols(self, query: str, kind: str = "", max_results: int = 20) -> list[dict]:
        """Search symbols using lexical + optional semantic."""
        return self.lexical.search_symbols(query, kind, max_results)

    def build_embeddings(self, force: bool = False) -> bool:
        """Build semantic embeddings for code in the graph.

        Requires numpy. Returns True if embeddings were built.
        """
        if self._embeddings and not force:
            return True

        try:
            import numpy as np
        except ImportError:
            return False

        if not self.graph:
            return False

        # Build simple bag-of-words embeddings (fast, no ML models needed)
        # For production, you'd want sentence-transformers here
        vocab: dict[str, int] = {}
        documents: list[tuple[str, str]] = []  # (key, text)

        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "symbol":
                continue
            text = " ".join(
                filter(
                    None,
                    [
                        data.get("name", ""),
                        data.get("qualified_name", ""),
                        data.get("signature", ""),
                        data.get("docstring", ""),
                    ],
                )
            )
            # Tokenize
            tokens = _tokenize(text)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
            documents.append((node_id, text))

        if not vocab:
            return False

        # Build TF-IDF-like vectors
        n_docs = len(documents)
        vocab_size = len(vocab)
        doc_freq = np.zeros(vocab_size)

        vectors: dict[str, np.ndarray] = {}
        for key, text in documents:
            tokens = _tokenize(text)
            vec = np.zeros(vocab_size)
            for token in tokens:
                idx = vocab.get(token)
                if idx is not None:
                    vec[idx] += 1
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
                vectors[key] = vec
            # Count document frequency
            for token in set(tokens):
                idx = vocab.get(token)
                if idx is not None:
                    doc_freq[idx] += 1

        # Apply IDF weighting
        idf = np.log(n_docs / (doc_freq + 1))
        for key in vectors:
            vectors[key] *= idf

        self._embeddings = {k: v.tolist() for k, v in vectors.items()}
        self._vocab = vocab
        self._idf = idf.tolist()
        return True

    def _semantic_scores(self, query: str) -> dict[str, float]:
        """Compute semantic similarity scores for a query."""
        if not self._embeddings:
            return {}

        try:
            import numpy as np
        except ImportError:
            return {}

        # Embed the query
        tokens = _tokenize(query)
        vocab_size = len(self._vocab)
        query_vec = np.zeros(vocab_size)
        for token in tokens:
            idx = self._vocab.get(token)
            if idx is not None:
                query_vec[idx] += 1
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec /= norm
        query_vec *= np.array(self._idf)

        # Compute cosine similarity with all embeddings
        scores: dict[str, float] = {}
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return scores

        for key, vec_list in self._embeddings.items():
            vec = np.array(vec_list)
            sim = np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec) + 1e-8)
            if sim > 0.1:
                scores[key] = float(sim)

        return scores


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer that splits on non-alphanumeric and camelCase."""
    import re

    # Split camelCase and snake_case
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace(".", " ")
    tokens = re.findall(r"[a-zA-Z]{2,}", text.lower())
    return tokens
