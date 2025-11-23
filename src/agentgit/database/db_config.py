"""Database configuration for the rollback agent system using SQLAlchemy ORM."""

import os
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from agentgit.database.models import Base


def _get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        db_type = os.getenv("DATABASE", "sqlite").strip().lower()
        path_or_dsn = get_database_path()
        
        if db_type == "postgres":
            # PostgreSQL connection
            _engine = create_engine(
                path_or_dsn,
                echo=False,
                pool_pre_ping=True,
            )
        else:
            # SQLite connection
            database_url = f"sqlite:///{path_or_dsn}"
            _engine = create_engine(
                database_url,
                echo=False,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
            
            # Enable foreign keys for SQLite
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
    
    return _engine


# Global engine and session factory
_engine = None
_SessionLocal = None


def _get_session_factory():
    """Get or create the session factory."""
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
    Resolve and return the effective database path or connection string according to environment/configuration.

    For SQLite, the resolution order is:
      1. If the explicit ``db_path`` argument is provided, return it directly.
      2. If the environment variable ``DATABASE_URL`` exists and starts with ``sqlite://``, extract and return its path.
      3. Otherwise, return the default SQLite path ``data/rollback_agent.db`` under the project root.

    For PostgreSQL, if the ``DATABASE`` environment variable is ``postgres``, return the ``DATABASE_URL`` as the connection string.
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
            return db_url[len("sqlite:///") :]
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

    When ``db_path`` is provided:
      * If it contains ``"://"``, it is treated as a full SQLAlchemy URL.
      * Otherwise it is treated as a SQLite filesystem path and converted to
        a ``sqlite:///`` URL with foreign key support enabled.

    When ``db_path`` is not provided, the global engine / session factory is
    used, based on ``DATABASE`` / ``DATABASE_URL`` environment variables.
    """
    engine_to_dispose = None
    
    if db_path:
        if "://" in db_path:
            if db_path.lower().startswith("sqlite://"):
                engine = create_engine(
                    db_path,
                    echo=False,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,
                )

                @event.listens_for(engine, "connect")
                def set_sqlite_pragma(dbapi_conn, connection_record):
                    cursor = dbapi_conn.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.close()
            else:
                engine = create_engine(db_path, echo=False, pool_pre_ping=True)
        else:
            abs_path = os.path.abspath(db_path)
            database_url = f"sqlite:///{abs_path}"
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

        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        engine_to_dispose = engine
    else:
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
        if engine_to_dispose is not None:
            engine_to_dispose.dispose()


def init_db():
    """Initialize database tables defined in agentgit.database.models."""
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)