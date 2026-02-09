"""Persistent storage for the knowledge graph using SQLite.

Uses normalized integer IDs and a single unified node table for
minimal storage overhead. Text node IDs (e.g. "file.py::func") are
reconstructed on load from path + qualified_name data.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import networkx as nx


class GraphStore:
    """Persists and loads the knowledge graph using SQLite.

    Storage is compact: file paths are interned (stored once), nodes
    use integer IDs, and text node IDs are reconstructed on load.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._create_tables()
        return self._conn

    def _create_tables(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            -- Intern file paths (stored once, referenced by integer)
            CREATE TABLE IF NOT EXISTS path_map (
                pid INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL
            );

            -- Unified node table (symbols + files)
            -- NULL columns cost 0 bytes in SQLite storage
            CREATE TABLE IF NOT EXISTS nodes (
                nid INTEGER PRIMARY KEY,
                node_type TEXT NOT NULL,         -- 'symbol' or 'file'
                path_id INTEGER NOT NULL REFERENCES path_map(pid),
                -- Symbol-only fields
                name TEXT,
                qualified_name TEXT,
                kind TEXT,                       -- 'function', 'class', etc.
                line_start INTEGER,
                line_end INTEGER,
                signature TEXT,
                docstring TEXT,
                -- File-only fields
                language TEXT,
                symbol_count INTEGER,
                import_count INTEGER
            );

            -- Edges between nodes (integer FKs)
            CREATE TABLE IF NOT EXISTS edges (
                source_nid INTEGER NOT NULL REFERENCES nodes(nid),
                target_nid INTEGER NOT NULL REFERENCES nodes(nid),
                kind TEXT NOT NULL,
                path_id INTEGER REFERENCES path_map(pid),
                line INTEGER,
                PRIMARY KEY (source_nid, target_nid, kind)
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
            CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
            CREATE INDEX IF NOT EXISTS idx_nodes_path ON nodes(path_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_nid);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_nid);
        """)
        conn.commit()

    # ------------------------------------------------------------------
    # Text ID reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _make_text_id(node_type: str, path: str, qualified_name: str = "") -> str:
        """Reconstruct a text node ID from stored fields."""
        if node_type == "file":
            return f"file::{path}"
        return f"{path}::{qualified_name}"

    def _resolve_text_id(self, conn: sqlite3.Connection, text_id: str) -> int | None:
        """Find the nid for a text-based node ID."""
        if text_id.startswith("file::"):
            fp = text_id[6:]  # strip "file::"
            row = conn.execute(
                """SELECT n.nid FROM nodes n
                   JOIN path_map p ON p.pid = n.path_id
                   WHERE n.node_type = 'file' AND p.path = ?""",
                (fp,),
            ).fetchone()
        elif "::" in text_id:
            fp, qname = text_id.split("::", 1)
            row = conn.execute(
                """SELECT n.nid FROM nodes n
                   JOIN path_map p ON p.pid = n.path_id
                   WHERE n.node_type = 'symbol' AND p.path = ? AND n.qualified_name = ?""",
                (fp, qname),
            ).fetchone()
        else:
            # Bare name — search by name or qualified_name
            row = conn.execute(
                "SELECT nid FROM nodes WHERE name = ? OR qualified_name = ?",
                (text_id, text_id),
            ).fetchone()
        return row["nid"] if row else None

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, graph: nx.DiGraph, metadata: dict | None = None) -> None:
        """Save the full graph to compact SQLite tables."""
        conn = self._get_conn()

        # Drop all legacy tables
        for legacy in ("graph_data", "symbols", "relationships", "files", "node_map"):
            conn.execute(f"DROP TABLE IF EXISTS {legacy}")

        # Clear current data
        conn.execute("DELETE FROM nodes")
        conn.execute("DELETE FROM edges")
        conn.execute("DELETE FROM path_map")

        path_cache: dict[str, int] = {}
        text_to_nid: dict[str, int] = {}
        nid_counter = 1

        def intern_path(fp: str) -> int:
            if fp in path_cache:
                return path_cache[fp]
            conn.execute("INSERT OR IGNORE INTO path_map (path) VALUES (?)", (fp,))
            row = conn.execute("SELECT pid FROM path_map WHERE path = ?", (fp,)).fetchone()
            pid = row["pid"]
            path_cache[fp] = pid
            return pid

        # Insert all nodes
        for node_id, data in graph.nodes(data=True):
            ntype = data.get("type", "")
            if ntype == "symbol":
                fp = data.get("file_path", "")
                pid = intern_path(fp) if fp else 0
                conn.execute(
                    """INSERT INTO nodes
                    (nid, node_type, path_id, name, qualified_name, kind,
                     line_start, line_end, signature, docstring)
                    VALUES (?, 'symbol', ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        nid_counter, pid,
                        data.get("name", ""),
                        data.get("qualified_name", ""),
                        data.get("kind", ""),
                        data.get("line_start", 0),
                        data.get("line_end", 0),
                        data.get("signature", ""),
                        data.get("docstring", ""),
                    ),
                )
            elif ntype == "file":
                fp = data.get("path", "")
                pid = intern_path(fp) if fp else 0
                conn.execute(
                    """INSERT INTO nodes
                    (nid, node_type, path_id, language, symbol_count, import_count)
                    VALUES (?, 'file', ?, ?, ?, ?)""",
                    (
                        nid_counter, pid,
                        data.get("language", ""),
                        data.get("symbol_count", 0),
                        data.get("import_count", 0),
                    ),
                )
            else:
                # Unknown node type — store minimally
                conn.execute(
                    "INSERT INTO nodes (nid, node_type, path_id) VALUES (?, ?, 0)",
                    (nid_counter, ntype or "unknown"),
                )

            text_to_nid[node_id] = nid_counter
            nid_counter += 1

        # Insert all edges
        for src, tgt, data in graph.edges(data=True):
            src_nid = text_to_nid.get(src)
            tgt_nid = text_to_nid.get(tgt)
            if src_nid is None or tgt_nid is None:
                continue
            fp = data.get("file_path", "")
            pid = intern_path(fp) if fp else None
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO edges
                    (source_nid, target_nid, kind, path_id, line)
                    VALUES (?, ?, ?, ?, ?)""",
                    (src_nid, tgt_nid, data.get("kind", ""), pid, data.get("line", 0)),
                )
            except sqlite3.IntegrityError:
                pass

        if metadata:
            for key, value in metadata.items():
                conn.execute(
                    "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )

        conn.commit()
        conn.execute("VACUUM")

    def load(self) -> nx.DiGraph | None:
        """Reconstruct the full NetworkX graph from SQLite."""
        conn = self._get_conn()

        # Detect which schema version we have
        has_nodes = False
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM nodes").fetchone()
            has_nodes = row is not None and row["cnt"] > 0
        except sqlite3.OperationalError:
            pass

        if not has_nodes:
            return self._load_legacy()

        # Load path lookup
        pid_to_path: dict[int, str] = {}
        for row in conn.execute("SELECT pid, path FROM path_map").fetchall():
            pid_to_path[row["pid"]] = row["path"]

        # Load nid → text_id mapping
        nid_to_text: dict[int, str] = {}
        graph = nx.DiGraph()

        for row in conn.execute("SELECT * FROM nodes").fetchall():
            nid = row["nid"]
            ntype = row["node_type"]
            fp = pid_to_path.get(row["path_id"], "")

            if ntype == "file":
                text_id = f"file::{fp}"
                nid_to_text[nid] = text_id
                graph.add_node(
                    text_id,
                    type="file",
                    path=fp,
                    language=row["language"] or "",
                    symbol_count=row["symbol_count"] or 0,
                    import_count=row["import_count"] or 0,
                )
            elif ntype == "symbol":
                qname = row["qualified_name"] or ""
                text_id = f"{fp}::{qname}"
                nid_to_text[nid] = text_id
                graph.add_node(
                    text_id,
                    type="symbol",
                    name=row["name"] or "",
                    qualified_name=qname,
                    kind=row["kind"] or "",
                    file_path=fp,
                    line_start=row["line_start"] or 0,
                    line_end=row["line_end"] or 0,
                    signature=row["signature"] or "",
                    docstring=row["docstring"] or "",
                )

        # Load edges
        for row in conn.execute("SELECT * FROM edges").fetchall():
            src = nid_to_text.get(row["source_nid"])
            tgt = nid_to_text.get(row["target_nid"])
            if src and tgt:
                fp = pid_to_path.get(row["path_id"]) if row["path_id"] else ""
                graph.add_edge(
                    src, tgt,
                    kind=row["kind"],
                    file_path=fp or "",
                    line=row["line"] or 0,
                )

        return graph

    def _load_legacy(self) -> nx.DiGraph | None:
        """Fallback: load from any older schema version."""
        conn = self._get_conn()

        # Try normalized schema v1 (node_map + symbols + relationships)
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM node_map").fetchone()
            if row and row["cnt"] > 0:
                return self._load_v1_normalized()
        except sqlite3.OperationalError:
            pass

        # Try text-ID schema (symbols with text id column)
        try:
            row = conn.execute("SELECT COUNT(*) as cnt FROM files").fetchone()
            if row and row["cnt"] > 0:
                test = conn.execute("SELECT id FROM symbols LIMIT 1").fetchone()
                if test is not None:
                    return self._load_text_tables()
        except sqlite3.OperationalError:
            pass

        # Try JSON blob (oldest)
        try:
            row = conn.execute("SELECT data FROM graph_data WHERE id = 1").fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        from networkx.readwrite import json_graph
        return json_graph.node_link_graph(json.loads(row["data"]))

    def _load_v1_normalized(self) -> nx.DiGraph | None:
        """Load from node_map + symbols + relationships schema."""
        conn = self._get_conn()
        nid_to_text = {}
        for row in conn.execute("SELECT nid, text_id FROM node_map").fetchall():
            nid_to_text[row["nid"]] = row["text_id"]

        pid_to_path = {}
        try:
            for row in conn.execute("SELECT pid, path FROM path_map").fetchall():
                pid_to_path[row["pid"]] = row["path"]
        except sqlite3.OperationalError:
            pass

        graph = nx.DiGraph()
        for row in conn.execute("SELECT * FROM files").fetchall():
            text_id = nid_to_text.get(row["nid"], "")
            fp = pid_to_path.get(row["path_id"], "") if "path_id" in row.keys() else ""
            if not fp and text_id.startswith("file::"):
                fp = text_id[6:]
            graph.add_node(
                text_id, type="file", path=fp,
                language=row["language"] or "",
                symbol_count=row["symbol_count"] or 0,
                import_count=row.get("import_count", 0) or 0,
            )

        for row in conn.execute("SELECT * FROM symbols").fetchall():
            text_id = nid_to_text.get(row["nid"], "")
            fp = pid_to_path.get(row["path_id"], "") if "path_id" in row.keys() else ""
            graph.add_node(
                text_id, type="symbol",
                name=row["name"], qualified_name=row["qualified_name"],
                kind=row["kind"], file_path=fp,
                line_start=row["line_start"] or 0, line_end=row["line_end"] or 0,
                signature=row["signature"] or "", docstring=row["docstring"] or "",
            )

        for row in conn.execute("SELECT * FROM relationships").fetchall():
            src = nid_to_text.get(row["source_nid"], "")
            tgt = nid_to_text.get(row["target_nid"], "")
            fp = pid_to_path.get(row["path_id"]) if row["path_id"] else ""
            graph.add_edge(src, tgt, kind=row["kind"], file_path=fp or "", line=row["line"] or 0)

        return graph

    def _load_text_tables(self) -> nx.DiGraph | None:
        """Load from old text-ID table schema."""
        conn = self._get_conn()
        graph = nx.DiGraph()
        for row in conn.execute("SELECT * FROM files").fetchall():
            graph.add_node(
                f"file::{row['path']}", type="file", path=row["path"],
                language=row["language"] or "", symbol_count=row["symbol_count"] or 0,
                import_count=0,
            )
        for row in conn.execute("SELECT * FROM symbols").fetchall():
            graph.add_node(
                row["id"], type="symbol", name=row["name"],
                qualified_name=row["qualified_name"], kind=row["kind"],
                file_path=row["file_path"],
                line_start=row["line_start"] or 0, line_end=row["line_end"] or 0,
                signature=row["signature"] or "", docstring=row["docstring"] or "",
            )
        for row in conn.execute("SELECT * FROM relationships").fetchall():
            graph.add_edge(
                row["source"], row["target"], kind=row["kind"],
                file_path=row["file_path"] or "", line=row["line"] or 0,
            )
        return graph

    # ------------------------------------------------------------------
    # Query methods (external API uses text node IDs)
    # ------------------------------------------------------------------

    def search_symbols(
        self,
        query: str = "",
        kind: str = "",
        file_path: str = "",
        limit: int = 50,
    ) -> list[dict]:
        """Search symbols using the indexed SQLite tables."""
        conn = self._get_conn()
        conditions = ["n.node_type = 'symbol'"]
        params: list = []

        if query:
            conditions.append("(n.name LIKE ? OR n.qualified_name LIKE ? OR n.signature LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like, like])
        if kind:
            conditions.append("n.kind = ?")
            params.append(kind)
        if file_path:
            conditions.append("p.path LIKE ?")
            params.append(f"%{file_path}%")

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"""SELECT (p.path || '::' || n.qualified_name) as id,
                       n.name, n.qualified_name, n.kind,
                       p.path as file_path, n.line_start, n.line_end,
                       n.signature, n.docstring
                FROM nodes n
                JOIN path_map p ON p.pid = n.path_id
                WHERE {where} LIMIT ?""",
            params + [limit],
        ).fetchall()

        return [dict(row) for row in rows]

    def get_callers(self, symbol_id: str) -> list[dict]:
        """Get all symbols that call the given symbol."""
        conn = self._get_conn()
        target_nid = self._resolve_text_id(conn, symbol_id)
        if target_nid is None:
            return []

        rows = conn.execute(
            """SELECT (sp.path || '::' || s.qualified_name) as id,
                      s.name, s.qualified_name, s.kind,
                      sp.path as file_path, s.line_start, s.line_end,
                      s.signature, s.docstring,
                      e.line as call_line, rp.path as call_file
               FROM edges e
               JOIN nodes s ON s.nid = e.source_nid AND s.node_type = 'symbol'
               JOIN path_map sp ON sp.pid = s.path_id
               LEFT JOIN path_map rp ON rp.pid = e.path_id
               WHERE e.target_nid = ? AND e.kind = 'calls'""",
            (target_nid,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_callees(self, symbol_id: str) -> list[dict]:
        """Get all symbols called by the given symbol."""
        conn = self._get_conn()
        source_nid = self._resolve_text_id(conn, symbol_id)
        if source_nid is None:
            return []

        rows = conn.execute(
            """SELECT (sp.path || '::' || s.qualified_name) as id,
                      s.name, s.qualified_name, s.kind,
                      sp.path as file_path, s.line_start, s.line_end,
                      s.signature, s.docstring,
                      e.line as call_line, rp.path as call_file
               FROM edges e
               JOIN nodes s ON s.nid = e.target_nid AND s.node_type = 'symbol'
               JOIN path_map sp ON sp.pid = s.path_id
               LEFT JOIN path_map rp ON rp.pid = e.path_id
               WHERE e.source_nid = ? AND e.kind = 'calls'""",
            (source_nid,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_metadata(self, key: str) -> str | None:
        """Get a metadata value."""
        conn = self._get_conn()
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row["value"])
        return None

    def set_metadata(self, key: str, value) -> None:
        """Set a single metadata value without a full save."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        conn.commit()

    def delete_file_data(self, file_path: str) -> None:
        """Remove a file node, its symbol nodes, and all related edges."""
        conn = self._get_conn()
        pid_row = conn.execute(
            "SELECT pid FROM path_map WHERE path = ?", (file_path,)
        ).fetchone()
        if not pid_row:
            return
        pid = pid_row["pid"]
        nids = [
            r["nid"]
            for r in conn.execute(
                "SELECT nid FROM nodes WHERE path_id = ?", (pid,)
            ).fetchall()
        ]
        if not nids:
            return
        placeholders = ",".join("?" * len(nids))
        conn.execute(
            f"DELETE FROM edges WHERE source_nid IN ({placeholders})"  # noqa: S608
            f" OR target_nid IN ({placeholders})",
            nids + nids,
        )
        conn.execute(
            f"DELETE FROM nodes WHERE nid IN ({placeholders})",  # noqa: S608
            nids,
        )
        conn.execute("DELETE FROM path_map WHERE pid = ?", (pid,))
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
