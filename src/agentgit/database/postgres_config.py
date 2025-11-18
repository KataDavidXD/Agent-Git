"""PostgreSQL connection utilities.

This module integrates PostgreSQL support without changing the existing
SQLite-based repositories. You can use these helpers in new code or
custom repositories while keeping the current SQLite storage working
as-is.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Optional

try:  # psycopg2 is an optional dependency
    import psycopg2
    from psycopg2.extensions import connection as PGConnection
except ImportError:  # pragma: no cover - handled at runtime
    psycopg2 = None  # type: ignore[assignment]
    PGConnection = object  # type: ignore[assignment]


def get_postgres_dsn() -> str:
    """Return the PostgreSQL DSN/URL from environment.

    Uses the ``DATABASE_URL`` environment variable.
    """
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "PostgreSQL DSN not configured. Set DATABASE_URL in the environment."
        )
    return dsn


def get_postgres_connection(dsn: Optional[str] = None) -> PGConnection:
    """Create a new PostgreSQL connection.

    Args:
        dsn: Optional explicit DSN/URL. If omitted, ``get_postgres_dsn()`` is used.

    Returns:
        A live psycopg2 connection object.

    Raises:
        RuntimeError: If psycopg2 is not installed or DSN is missing.
    """
    if psycopg2 is None:  # type: ignore[truthy-function]
        raise RuntimeError(
            "psycopg2 is required for PostgreSQL support. Install `psycopg2-binary` or `psycopg2`."
        )

    return psycopg2.connect(dsn or get_postgres_dsn())


@contextmanager
def postgres_cursor(dsn: Optional[str] = None) -> Iterator["PGCursor"]:
    """Context manager yielding a cursor with automatic commit/close.

    Example::

        from agentgit.database.postgres_config import postgres_cursor

        with postgres_cursor() as cur:
            cur.execute("SELECT 1")
            row = cur.fetchone()

    """
    conn = get_postgres_connection(dsn)
    try:
        with conn:
            with conn.cursor() as cur:  # type: ignore[attr-defined]
                yield cur
    finally:
        conn.close()


# Type alias for better editor support (optional; avoids importing psycopg2 in callers)
try:  # pragma: no cover - typing helper only
    from psycopg2.extensions import cursor as PGCursor  # type: ignore
except Exception:  # pragma: no cover
    class PGCursor:  # type: ignore[override]
        ...
