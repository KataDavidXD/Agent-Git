"""Database configuration for the rollback agent system.

Repositories can switch between SQLite and PostgreSQL via a
single environment variable.
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

from agentgit.database.postgres_config import get_postgres_connection


def get_database_path() -> str:
    """Get the path to the SQLite database.
    
    Returns:
        Path to the database file
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
    )
    os.makedirs(data_dir, exist_ok=True)
    
    # Return path to database file
    return os.path.join(data_dir, "rollback_agent.db")


def get_db_backend() -> str:
    """Return the configured database backend.

    Uses the ``DATABASE`` and ``DATABASE_URL`` environment variables together.
    Only ``DATABASE=postgres`` with a PostgreSQL ``DATABASE_URL`` enables the
    PostgreSQL backend. All other valid combinations resolve to SQLite.
    """
    backend = os.getenv("DATABASE", "sqlite").strip().lower()
    db_url = os.getenv("DATABASE_URL", "").strip()

    if backend not in {"sqlite", "postgres"}:
        raise RuntimeError(
            "DATABASE must be either 'sqlite' or 'postgres' when set."
        )

    # PostgreSQL: require explicit DATABASE=postgres and a postgres URL
    if backend == "postgres":
        if not db_url:
            raise RuntimeError(
                "DATABASE is set to 'postgres' but DATABASE_URL is empty. "
                "Please set DATABASE_URL to a postgresql:// or postgres:// DSN."
            )
        scheme = db_url.lower()
        if not (scheme.startswith("postgresql://") or scheme.startswith("postgres://")):
            raise RuntimeError(
                "DATABASE is 'postgres' but DATABASE_URL is not a PostgreSQL URL. "
                "Expected DATABASE_URL starting with postgresql:// or postgres://."
            )
        return "postgres"

    # SQLite backend (backend == "sqlite")
    if db_url:
        scheme = db_url.lower()
        if scheme.startswith("sqlite://"):
            return "sqlite"
        if scheme.startswith("postgresql://") or scheme.startswith("postgres://"):
            raise RuntimeError(
                "DATABASE_URL is a PostgreSQL URL but DATABASE is not set to 'postgres'. "
                "Either set DATABASE=postgres or change DATABASE_URL to a sqlite:// URL."
            )
        raise RuntimeError(
            "Unsupported DATABASE_URL scheme. Only postgresql://, postgres:// and sqlite:// are supported."
        )

    # Default: SQLite with local file
    return "sqlite"


def is_postgres_backend() -> bool:
    """Whether PostgreSQL is configured as the active backend."""
    return get_db_backend() == "postgres"


@contextmanager
def get_db_connection(db_path: Optional[str] = None):
    """Yield a database connection for the active backend.

    For SQLite, ``DATABASE_URL`` (sqlite URL) or ``db_path`` (or the default database path) is used and
    foreign key support is enabled. For PostgreSQL, a connection is
    created via :func:`get_postgres_connection`.
    """
    if is_postgres_backend():
        conn = get_postgres_connection()
    else:
        db_url = os.getenv("DATABASE_URL")
        if db_url and db_url.lower().startswith("sqlite://"):
            lower_url = db_url.lower()
            if lower_url.startswith("sqlite:///"):
                path = db_url[len("sqlite:///") :]
            else:
                path = db_url.split("://", 1)[1]
        else:
            path = db_path or get_database_path()

        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()