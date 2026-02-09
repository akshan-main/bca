"""Native C++ acceleration for CAG (Context Assembly Generation).

Provides Python bindings to the C++ core via ctypes.
Falls back gracefully to pure Python if the native library is unavailable.

The native implementation provides 10-50x speedup for:
  - Weighted BFS graph traversal (the hot path)
  - Batch token estimation
  - Topological sort for dependency ordering
  - Entity extraction from text
"""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path

# Attempt to load the native library
_lib = None
_lib_path = None


def _find_library() -> str | None:
    """Search for the compiled C++ library."""
    # Platform-specific extension
    system = platform.system()
    if system == "Darwin":
        ext = "dylib"
    elif system == "Windows":
        ext = "dll"
    else:
        ext = "so"

    name = f"cag_fast.{ext}"

    # Search locations (in priority order)
    search_dirs = [
        Path(__file__).parent,                          # Installed alongside Python code
        Path(__file__).parent.parent.parent.parent / "csrc",  # Development: csrc/
        Path.cwd() / "csrc",                           # Current directory
    ]

    for d in search_dirs:
        candidate = d / name
        if candidate.exists():
            return str(candidate)

    return None


def _load_library():
    """Load the native library, or return None."""
    global _lib, _lib_path

    if _lib is not None:
        return _lib

    path = _find_library()
    if path is None:
        return None

    try:
        lib = ctypes.CDLL(path)

        # Verify it's the right library
        lib.cag_is_available.restype = ctypes.c_int32
        if lib.cag_is_available() != 1:
            return None

        # Set up function signatures
        _setup_signatures(lib)

        _lib = lib
        _lib_path = path
        return lib
    except (OSError, AttributeError):
        return None


def _setup_signatures(lib):
    """Define ctypes function signatures for type safety."""
    # Graph creation
    lib.cag_graph_create.argtypes = [ctypes.c_int32]
    lib.cag_graph_create.restype = ctypes.c_void_p

    lib.cag_graph_add_edge.argtypes = [
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_float,
    ]
    lib.cag_graph_add_edge.restype = None

    lib.cag_graph_set_node_weight.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_float]
    lib.cag_graph_set_node_weight.restype = None

    lib.cag_graph_set_lines.argtypes = [
        ctypes.c_void_p, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32,
    ]
    lib.cag_graph_set_lines.restype = None

    # Weighted BFS
    lib.cag_weighted_bfs.argtypes = [
        ctypes.c_void_p,                                    # graph
        ctypes.POINTER(ctypes.c_int32),                     # seed_nodes
        ctypes.POINTER(ctypes.c_float),                     # seed_scores
        ctypes.c_int32,                                     # num_seeds
        ctypes.c_int32,                                     # max_depth
        ctypes.c_float,                                     # min_score
        ctypes.c_float,                                     # backward_decay
        ctypes.POINTER(ctypes.c_int32),                     # out_nodes
        ctypes.POINTER(ctypes.c_float),                     # out_scores
        ctypes.POINTER(ctypes.c_int32),                     # out_depths
        ctypes.c_int32,                                     # max_results
    ]
    lib.cag_weighted_bfs.restype = ctypes.c_int32

    # Token estimation
    lib.cag_estimate_tokens.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.cag_estimate_tokens.restype = ctypes.c_int32

    # Topological sort
    lib.cag_topological_sort.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.cag_topological_sort.restype = ctypes.c_int32

    # Entity extraction
    lib.cag_extract_entities.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
    ]
    lib.cag_extract_entities.restype = ctypes.c_int32

    # Cleanup
    lib.cag_graph_destroy.argtypes = [ctypes.c_void_p]
    lib.cag_graph_destroy.restype = None

    # Version
    lib.cag_version.argtypes = []
    lib.cag_version.restype = ctypes.c_char_p


class NativeCAG:
    """Python wrapper around the C++ CAG core.

    Usage:
        if NativeCAG.is_available():
            graph = NativeCAG.create_graph(num_nodes)
            graph.add_edge(0, 1, 0.85)
            results = graph.weighted_bfs(seeds, scores, max_depth=3)
            graph.destroy()
    """

    @staticmethod
    def is_available() -> bool:
        """Check if the native C++ library is available."""
        return _load_library() is not None

    @staticmethod
    def version() -> str:
        """Get the native library version."""
        lib = _load_library()
        if lib is None:
            return "unavailable"
        return lib.cag_version().decode("utf-8")

    @staticmethod
    def create_graph(num_nodes: int) -> NativeGraph:
        """Create a new native graph."""
        lib = _load_library()
        if lib is None:
            raise RuntimeError("Native CAG library not available")
        handle = lib.cag_graph_create(num_nodes)
        return NativeGraph(lib, handle, num_nodes)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count using the native implementation."""
        lib = _load_library()
        if lib is None:
            return max(1, len(text) // 4)
        encoded = text.encode("utf-8")
        return lib.cag_estimate_tokens(encoded, len(encoded))

    @staticmethod
    def extract_entities(text: str, max_entities: int = 200) -> list[dict]:
        """Extract code entities from text using native C++."""
        lib = _load_library()
        if lib is None:
            return []

        encoded = text.encode("utf-8")
        starts = (ctypes.c_int32 * max_entities)()
        ends = (ctypes.c_int32 * max_entities)()
        types = (ctypes.c_int32 * max_entities)()

        count = lib.cag_extract_entities(
            encoded, len(encoded), starts, ends, types, max_entities
        )

        type_map = {0: "class", 1: "function", 2: "path", 3: "file", 4: "quoted"}
        results = []
        for i in range(count):
            name = text[starts[i]:ends[i]]
            results.append({
                "name": name,
                "type": type_map.get(types[i], "unknown"),
                "start": starts[i],
                "end": ends[i],
            })

        return results


class NativeGraph:
    """Handle to a native C++ graph."""

    def __init__(self, lib, handle, num_nodes: int):
        self._lib = lib
        self._handle = handle
        self._num_nodes = num_nodes
        self._destroyed = False

    def add_edge(self, src: int, dst: int, weight: float) -> None:
        self._lib.cag_graph_add_edge(self._handle, src, dst, weight)

    def set_node_weight(self, node: int, weight: float) -> None:
        self._lib.cag_graph_set_node_weight(self._handle, node, weight)

    def set_lines(self, node: int, start: int, end: int) -> None:
        self._lib.cag_graph_set_lines(self._handle, node, start, end)

    def weighted_bfs(
        self,
        seed_nodes: list[int],
        seed_scores: list[float],
        max_depth: int = 3,
        min_score: float = 0.1,
        backward_decay: float = 0.7,
        max_results: int = 5000,
    ) -> list[dict]:
        """Run weighted BFS and return results."""
        n = len(seed_nodes)
        c_seeds = (ctypes.c_int32 * n)(*seed_nodes)
        c_scores = (ctypes.c_float * n)(*seed_scores)

        out_nodes = (ctypes.c_int32 * max_results)()
        out_scores = (ctypes.c_float * max_results)()
        out_depths = (ctypes.c_int32 * max_results)()

        count = self._lib.cag_weighted_bfs(
            self._handle,
            c_seeds, c_scores, n,
            max_depth, min_score, backward_decay,
            out_nodes, out_scores, out_depths,
            max_results,
        )

        return [
            {"node": out_nodes[i], "score": out_scores[i], "depth": out_depths[i]}
            for i in range(count)
        ]

    def topological_sort(self, nodes: list[int]) -> list[int]:
        """Topological sort a subset of nodes."""
        n = len(nodes)
        c_nodes = (ctypes.c_int32 * n)(*nodes)
        out_order = (ctypes.c_int32 * n)()

        count = self._lib.cag_topological_sort(self._handle, c_nodes, n, out_order)
        return [out_order[i] for i in range(count)]

    def destroy(self) -> None:
        """Free the native graph."""
        if not self._destroyed:
            self._lib.cag_graph_destroy(self._handle)
            self._destroyed = True

    def __del__(self):
        self.destroy()
