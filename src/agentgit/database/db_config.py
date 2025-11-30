"""Database configuration for the rollback agent system using SQLAlchemy ORM."""

import os
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from agentgit.database.models import Base


def _create_db_engine(database_url: str, db_type: str = "sqlite"):
    """Create a SQLAlchemy engine for any supported database type.
    
    Unified engine creation with database-specific configurations.
    Extensible design: add new database types by adding elif branches.
    
    Args:
        database_url: Database connection URL
        db_type: Database type ('sqlite', 'postgres', 'mysql', etc.)
    
    Returns:
        Configured SQLAlchemy Engine
    
    Raises:
        ValueError: If database type is not supported
    """
    db_type = db_type.lower()
    
    if db_type == "sqlite":
        # SQLite-specific configuration
        engine = create_engine(
            database_url,
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        
        # Enable foreign key constraints for SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        
        return engine
    
    elif db_type in ("postgres", "postgresql"):
        # PostgreSQL-specific configuration
        return create_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
        )
    
    # Future database support can be added here:
    # elif db_type == "mysql":
    #     return create_engine(
    #         database_url,
    #         echo=False,
    #         pool_recycle=3600,
    #         pool_pre_ping=True,
    #     )
    
    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def _normalize_db_url(db_path: str) -> tuple[str, str]:
    """Normalize a db_path to a SQLAlchemy URL and detect database type.
    
    Args:
        db_path: Either a filesystem path or a SQLAlchemy URL
    
    Returns:
        Tuple of (url, db_type) where:
        - url: SQLAlchemy URL string
        - db_type: Database type ('sqlite', 'postgres', 'mysql', etc.)
    """
    if "://" in db_path:
        # Full URL - detect database type from scheme
        url_lower = db_path.lower()
        if url_lower.startswith("sqlite://"):
            return db_path, "sqlite"
        elif url_lower.startswith(("postgresql://", "postgres://")):
            return db_path, "postgres"
        # elif url_lower.startswith("mysql://"):
        #     return db_path, "mysql"
        else:
            raise ValueError(f"Unsupported database schema in URL: {db_path!r}.")
    else:
        # Plain filesystem path -> treat as SQLite
        abs_path = os.path.abspath(db_path)
        sqlite_url = f"sqlite:///{abs_path}"
        return sqlite_url, "sqlite"


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
        
        # Use unified engine creation
        if db_type == "postgres":
            _engine = _create_db_engine(path_or_dsn, db_type="postgres")
        else:
            sqlite_url = f"sqlite:///{path_or_dsn}"
            _engine = _create_db_engine(sqlite_url, db_type="sqlite")
    
    return _engine


def _get_session_factory(engine=None):
    """Get or create a session factory.
    
    Unified sessionmaker creation for all scenarios:
    - If engine is provided: creates a new sessionmaker for that engine (test mode)
    - If engine is None: returns the global singleton sessionmaker (production mode)
    
    Args:
        engine: Optional SQLAlchemy Engine. If None, uses global engine.
    
    Returns:
        sessionmaker bound to the specified or global engine
    """
    if engine is not None:
        # Test Mode: Create new sessionmaker for provided engine
        return sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
    
    # Production Mode: Return global singleton sessionmaker
    global _SessionLocal
    if _SessionLocal is None:
        global_engine = _get_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=global_engine
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
    
    if db_path:
        # Test Mode: Create temporary engine and sessionmaker for custom db_path
        url, db_type = _normalize_db_url(db_path)
        engine = _create_db_engine(url, db_type)
        SessionLocal = _get_session_factory(engine)
        engine_to_dispose = engine
    else:
        # Production Mode: Reuse global sessionmaker (performance optimization)
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
        # Dispose engine when in test mode
        if engine_to_dispose is not None:
            engine_to_dispose.dispose()


def init_db():
    """Initialize database tables defined in agentgit.database.models."""
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)