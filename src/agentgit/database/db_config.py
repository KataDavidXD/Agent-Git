"""Database configuration for the rollback agent system using SQLAlchemy ORM."""

import os
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from agentgit.database.models import Base


def _create_sqlite_engine(database_url: str):
    """Create a SQLite engine with foreign key support.
    
    Args:
        database_url: SQLite URL (e.g., 'sqlite:///path/to/db.db')
    
    Returns:
        SQLAlchemy Engine configured for SQLite
    """
    engine = create_engine(
        database_url,
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    
    return engine


def _normalize_db_url(db_path: str) -> tuple[str, bool]:
    """Normalize a db_path to a SQLAlchemy URL.
    
    Args:
        db_path: Either a filesystem path or a SQLAlchemy URL
    
    Returns:
        Tuple of (url, is_sqlite) where:
        - url: SQLAlchemy URL string
        - is_sqlite: True if the URL is for SQLite
    """
    if "://" in db_path:
        # Full URL like postgresql://... or sqlite:///...
        is_sqlite = db_path.lower().startswith("sqlite://")
        return db_path, is_sqlite
    else:
        # Plain filesystem path -> convert to SQLite URL
        abs_path = os.path.abspath(db_path)
        sqlite_url = f"sqlite:///{abs_path}"
        return sqlite_url, True


# Global engine and session factory (singletons)
_engine = None
_SessionLocal = None


def _get_engine():
    """Get or create the global SQLAlchemy engine (singleton).
    
    Uses DATABASE and DATABASE_URL environment variables to configure
    the connection. Defaults to SQLite with data/rollback_agent.db.
    """
    global _engine
    if _engine is None:
        db_type = os.getenv("DATABASE", "sqlite").strip().lower()
        path_or_dsn = get_database_path()
        
        if db_type == "postgres":
            _engine = create_engine(
                path_or_dsn,
                echo=False,
                pool_pre_ping=True,
            )
        else:
            sqlite_url = f"sqlite:///{path_or_dsn}"
            _engine = _create_sqlite_engine(sqlite_url)
    
    return _engine


def _get_session_factory():
    """Get or create the global session factory (singleton).
    
    Returns a sessionmaker bound to the global engine. This is reused
    for all connections to the default database (no custom db_path).
    """
    global _SessionLocal
    if _SessionLocal is None:
        engine = _get_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
    return _SessionLocal


def get_database_path(db_path: Optional[str] = None) -> str:
    """
    Resolve and return the effective database path or connection string.

    For SQLite, the resolution order is:
      1. If the explicit ``db_path`` argument is provided, return it directly.
      2. If the environment variable ``DATABASE_URL`` exists and starts with 
         ``sqlite://``, extract and return its path.
      3. Otherwise, return the default SQLite path ``data/rollback_agent.db`` 
         under the project root.

    For PostgreSQL, if the ``DATABASE`` environment variable is ``postgres``, 
    return the ``DATABASE_URL`` as the connection string.
    """
    # Explicitly specified database path
    if db_path:
        return db_path
    
    db_type = os.getenv("DATABASE", "sqlite").strip().lower()
    db_url = (os.getenv("DATABASE_URL") or "").strip()

    # PostgreSQL connection string via environment/config
    if db_type == "postgres":
        return db_url
    
    # SQLite connection path
    if db_url and db_url.lower().startswith("sqlite://"):
        lower_url = db_url.lower()
        if lower_url.startswith("sqlite:///"):
            return db_url[len("sqlite:///"):]
        return db_url.split("://", 1)[1]

    # Default SQLite path under project root's 'data' directory
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
    )
    os.makedirs(data_dir, exist_ok=True)

    return os.path.join(data_dir, "rollback_agent.db")


@contextmanager
def get_db_connection(db_path: Optional[str] = None):
    """Yield a SQLAlchemy database session.

    Args:
        db_path: Optional custom database path or URL. If provided:
          - Plain filesystem path (no '://') → treated as SQLite file path
          - URL with '://' → treated as SQLAlchemy URL (sqlite:// or postgresql://)
          If not provided, uses global engine configured via DATABASE env var.
    
    Yields:
        SQLAlchemy Session object
    
    Automatically commits on success, rolls back on exception, and closes
    the session in finally block. Disposes custom engines to prevent leaks.
    
    Design:
        - Custom db_path: Creates temporary engine + sessionmaker, disposes after use
        - No db_path: Reuses global engine + sessionmaker (singleton pattern)
    """
    engine_to_dispose = None
    
    # Test Mode
    if db_path:
        # Create a custom engine and sessionmaker for this specific db_path
        url, is_sqlite = _normalize_db_url(db_path)
        
        if is_sqlite:
            engine = _create_sqlite_engine(url)
        else:
            engine = create_engine(url, echo=False, pool_pre_ping=True)
        
        # Create temporary sessionmaker for custom engine
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        engine_to_dispose = engine
    # Production Mode
    else:
        # Reuse global sessionmaker (performance optimization)
        SessionLocal = _get_session_factory()
    
    session = SessionLocal()
    
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
        # Be destroyed when it is in test mode.
        if engine_to_dispose is not None:
            engine_to_dispose.dispose()


def init_db():
    """Initialize database tables defined in agentgit.database.models."""
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)