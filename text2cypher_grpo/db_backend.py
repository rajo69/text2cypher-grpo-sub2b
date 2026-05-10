"""Cypher execution backend abstraction.

Two implementations:
  - KuzuBackend (primary; cross-platform; archived but functional 0.11.3)
  - FalkorDBLiteBackend (Kaggle-only; requires Python ≥ 3.12; schemaless)

See report.md §kuzu-archived and plan.md §7.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

ErrorKind = Literal["parse", "runtime", "timeout", "ddl"] | None


@dataclass
class ExecutionResult:
    success: bool
    rows: list[tuple] | None = None
    error: str | None = None
    error_kind: ErrorKind = None

    def grade(self) -> float:
        """Coarse 0/0.3/0.7 grade used by reward_judge.

        Final 1.0 grade requires comparison to gold result-set, which is the caller's job.
        """
        if not self.success:
            if self.error_kind == "parse":
                return 0.0
            return 0.3  # runtime / ddl / timeout
        return 0.7  # executes; correctness vs gold checked separately


class CypherBackend(ABC):
    @abstractmethod
    def execute(self, schema_ddl: str, query: str, timeout_ms: int = 1000) -> ExecutionResult: ...

    @abstractmethod
    def name(self) -> str: ...


class KuzuBackend(CypherBackend):
    """Each call creates a fresh in-memory kuzu DB, applies the DDL, runs the query.

    Per-call DB creation is OK because text2cypher schemas are small and kuzu's
    in-memory mode is fast (<10ms for typical schemas). Caller may cache by
    schema-id if needed.
    """

    def __init__(self):
        try:
            import kuzu  # noqa: F401
        except ImportError as e:
            raise ImportError("kuzu is not installed. Run: pip install kuzu") from e

    def name(self) -> str:
        return "kuzu"

    def execute(self, schema_ddl: str, query: str, timeout_ms: int = 1000) -> ExecutionResult:
        import kuzu
        try:
            db = kuzu.Database(":memory:")
            conn = kuzu.Connection(db)
        except Exception as e:
            return ExecutionResult(False, error=f"db init: {e}", error_kind="runtime")

        # Apply DDL — split on `;` and run each statement individually so one bad
        # CREATE doesn't sink the rest. Failures here are tracked but non-fatal.
        for stmt in schema_ddl.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            try:
                conn.execute(stmt)
            except Exception as e:
                logger.debug("DDL failed: %s -- %s", stmt[:120], e)
        # Now the query.
        try:
            r = conn.execute(query)
        except Exception as e:
            msg = str(e)
            kind: ErrorKind = "parse" if "Parser" in msg or "syntax" in msg.lower() else "runtime"
            return ExecutionResult(False, error=msg[:300], error_kind=kind)
        # Iterate result. kuzu QueryResult is a generator; collect rows defensively.
        try:
            rows: list[tuple] = []
            for row in r:
                rows.append(tuple(row) if not isinstance(row, tuple) else row)
                if len(rows) > 10000:  # safety: don't blow memory on a runaway query
                    break
            return ExecutionResult(True, rows=rows)
        except Exception as e:
            return ExecutionResult(False, error=f"row iteration: {e}", error_kind="runtime")


class FalkorDBLiteBackend(CypherBackend):
    """Schemaless RedisGraph-style Cypher. Linux/macOS only (Unix sockets).

    Use only if Python ≥ 3.12 and Linux. Otherwise fall back to KuzuBackend.
    """

    def __init__(self):
        if sys.version_info < (3, 12):
            raise RuntimeError(
                "FalkorDBLite requires Python >= 3.12; current=%d.%d. Use KuzuBackend instead."
                % sys.version_info[:2]
            )
        try:
            from redislite.falkordb_client import FalkorDB  # noqa: F401
        except ImportError as e:
            raise ImportError("falkordblite is not installed. pip install falkordblite") from e

    def name(self) -> str:
        return "falkordblite"

    def execute(self, schema_ddl: str, query: str, timeout_ms: int = 1000) -> ExecutionResult:
        # FalkorDB is schemaless, so we ignore schema_ddl. Tradeoff: any query that depends on
        # type-checked properties will *not* fail in FalkorDBLite (it accepts more) → execution
        # accuracy is more lenient than kuzu. Use only for cross-checking.
        from redislite.falkordb_client import FalkorDB
        import tempfile, os
        # Per-call sandbox so concurrent rollouts don't collide
        tmpdir = tempfile.mkdtemp(prefix="falkordb_")
        db_path = os.path.join(tmpdir, "g.db")
        try:
            db = FalkorDB(db_path)
            g = db.select_graph("g")
            try:
                r = g.query(query)
            except Exception as e:
                msg = str(e)
                kind: ErrorKind = "parse" if "syntax" in msg.lower() else "runtime"
                return ExecutionResult(False, error=msg[:300], error_kind=kind)
            return ExecutionResult(True, rows=list(r.result_set or []))
        finally:
            try:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass


def get_backend(name: str = "kuzu") -> CypherBackend:
    name = name.lower().strip()
    if name == "kuzu":
        return KuzuBackend()
    if name in ("falkordb", "falkordblite"):
        return FalkorDBLiteBackend()
    raise ValueError(f"Unknown backend: {name}")
