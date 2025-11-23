"""Repository for external session database operations using SQLAlchemy ORM.

Handles CRUD operations for external sessions in the LangGraph rollback agent system.
"""

from typing import Optional, List
from datetime import datetime, timezone

from agentgit.sessions.external_session import ExternalSession
from agentgit.database.db_config import get_database_path, get_db_connection, init_db
from agentgit.database.models import ExternalSession as ExternalSessionModel


class ExternalSessionRepository:
    """Repository for ExternalSession CRUD operations with SQLAlchemy ORM.

    Manages external sessions which are the user-visible conversation containers.
    Each external session can contain multiple internal sessions with branching support.

    Attributes:
        db_path: Path to the database file or connection string.

    Example:
        >>> repo = ExternalSessionRepository()
        >>> session = ExternalSession(user_id=1, session_name="My Chat")
        >>> saved_session = repo.create(session)
        >>> sessions = repo.get_user_sessions(user_id=1)
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the external session repository.

        Args:
            db_path: Path to database. If None, uses configured default.
        """
        self.db_path = db_path or get_database_path()
        self._init_db()

    def _init_db(self):
        """Initialize the external sessions table if it doesn't exist."""
        init_db()

    def create(self, session: ExternalSession) -> ExternalSession:
        """Create a new external session.

        Args:
            session: ExternalSession object to create.

        Returns:
            The created session with id populated.

        Raises:
            IntegrityError: If user_id doesn't exist.
        """
        session_dict = session.to_dict()

        with get_db_connection(self.db_path) as db_session:
            db_external_session = ExternalSessionModel(
                user_id=session.user_id,
                session_name=session.session_name,
                updated_at=session.updated_at,
                is_active=session.is_active,
                data=session_dict,
                session_metadata=session.metadata,
                branch_count=session.branch_count,
                total_checkpoints=session.total_checkpoints,
            )
            # created_at is auto-generated
            db_session.add(db_external_session)
            db_session.flush()
            session.id = db_external_session.id
            # Update session.created_at from database
            if db_external_session.created_at:
                session.created_at = db_external_session.created_at

        return session

    def update(self, session: ExternalSession) -> bool:
        """Update an existing external session.

        Updates all session data including internal session IDs and current session.

        Args:
            session: ExternalSession object with updated data.

        Returns:
            True if update successful, False if session not found.
        """
        if not session.id:
            return False

        session.updated_at = datetime.now(timezone.utc)
        session_dict = session.to_dict()

        with get_db_connection(self.db_path) as db_session:
            db_external_session = db_session.query(ExternalSessionModel).filter_by(id=session.id).first()
            if db_external_session:
                db_external_session.session_name = session.session_name
                db_external_session.updated_at = session.updated_at
                db_external_session.is_active = session.is_active
                db_external_session.data = session_dict
                db_external_session.session_metadata = session.metadata
                db_external_session.branch_count = session.branch_count
                db_external_session.total_checkpoints = session.total_checkpoints
                return True
            return False

    def get_by_id(self, session_id: int) -> Optional[ExternalSession]:
        """Get an external session by ID.

        Args:
            session_id: The ID of the session to retrieve.

        Returns:
            ExternalSession if found, None otherwise.
        """
        with get_db_connection(self.db_path) as db_session:
            db_external_session = db_session.query(ExternalSessionModel).filter_by(id=session_id).first()
            if db_external_session:
                return self._row_to_session(db_external_session)
        return None

    def get_user_sessions(self, user_id: int, active_only: bool = False) -> List[ExternalSession]:
        """Get all sessions for a user.

        Args:
            user_id: The ID of the user.
            active_only: If True, only return active sessions.

        Returns:
            List of ExternalSession objects, ordered by created_at descending.
        """
        with get_db_connection(self.db_path) as db_session:
            query = db_session.query(ExternalSessionModel).filter_by(user_id=user_id)
            if active_only:
                query = query.filter_by(is_active=True)
            db_sessions = query.order_by(ExternalSessionModel.created_at.desc()).all()
            return [self._row_to_session(db_sess) for db_sess in db_sessions]

    def get_by_internal_session(self, langgraph_session_id: str) -> Optional[ExternalSession]:
        """Find the external session containing a specific internal langgraph session.

        Args:
            langgraph_session_id: The langgraph session ID to search for.

        Returns:
            ExternalSession containing the internal session, None if not found.
        """
        with get_db_connection(self.db_path) as db_session:
            # Search in the JSON data field
            # For PostgreSQL JSONB, we could use JSON operators, but for compatibility we search all
            db_sessions = db_session.query(ExternalSessionModel).all()
            
            for db_sess in db_sessions:
                session = self._row_to_session(db_sess)
                if langgraph_session_id in session.internal_session_ids:
                    return session
        return None

    def add_internal_session(self, external_session_id: int, langgraph_session_id: str) -> bool:
        """Add an internal langgraph session to an external session.

        Args:
            external_session_id: The ID of the external session.
            langgraph_session_id: The langgraph session ID to add.

        Returns:
            True if successful, False if external session not found.
        """
        session = self.get_by_id(external_session_id)
        if not session:
            return False

        session.add_internal_session(langgraph_session_id)
        return self.update(session)

    def set_current_internal_session(self, external_session_id: int, langgraph_session_id: str) -> bool:
        """Set the current internal session for an external session.

        Args:
            external_session_id: The ID of the external session.
            langgraph_session_id: The langgraph session ID to set as current.

        Returns:
            True if successful, False if session not found or langgraph_session_id not in list.
        """
        session = self.get_by_id(external_session_id)
        if not session:
            return False

        if session.set_current_internal_session(langgraph_session_id):
            return self.update(session)
        return False

    def deactivate(self, session_id: int) -> bool:
        """Deactivate an external session (soft delete).

        Args:
            session_id: The ID of the session to deactivate.

        Returns:
            True if deactivation successful, False otherwise.
        """
        session = self.get_by_id(session_id)
        if not session:
            return False

        session.is_active = False
        session.updated_at = datetime.now(timezone.utc)
        return self.update(session)

    def delete(self, session_id: int) -> bool:
        """Permanently delete an external session.

        Args:
            session_id: The ID of the session to delete.

        Returns:
            True if deletion successful, False otherwise.

        Note:
            This will cascade delete all internal sessions and checkpoints
            associated with this external session.
        """
        with get_db_connection(self.db_path) as db_session:
            db_external_session = db_session.query(ExternalSessionModel).filter_by(id=session_id).first()
            if db_external_session:
                db_session.delete(db_external_session)
                return True
            return False

    def check_ownership(self, session_id: int, user_id: int) -> bool:
        """Check if a user owns a specific session.

        Args:
            session_id: The ID of the session.
            user_id: The ID of the user.

        Returns:
            True if the user owns the session, False otherwise.
        """
        with get_db_connection(self.db_path) as db_session:
            count = db_session.query(ExternalSessionModel).filter_by(
                id=session_id, user_id=user_id
            ).count()
            return count > 0

    def count_user_sessions(self, user_id: int, active_only: bool = False) -> int:
        """Count the number of sessions a user has.

        Args:
            user_id: The ID of the user.
            active_only: If True, only count active sessions.

        Returns:
            The number of sessions.
        """
        with get_db_connection(self.db_path) as db_session:
            query = db_session.query(ExternalSessionModel).filter_by(user_id=user_id)
            if active_only:
                query = query.filter_by(is_active=True)
            return query.count()

    def _row_to_session(self, db_sess: ExternalSessionModel) -> ExternalSession:
        """Convert a database model to an ExternalSession object.

        Args:
            db_sess: ExternalSessionModel instance from database.
            
        Returns:
            ExternalSession object with all fields including internal session tracking.
        """
        if db_sess.data and isinstance(db_sess.data, dict):
            session_dict = db_sess.data.copy()
        else:
            # Fallback for older records without JSON data
            session_dict = {
                "id": db_sess.id,
                "user_id": db_sess.user_id,
                "session_name": db_sess.session_name,
                "created_at": db_sess.created_at.isoformat() if db_sess.created_at else None,
                "updated_at": db_sess.updated_at.isoformat() if db_sess.updated_at else None,
                "is_active": db_sess.is_active,
                "internal_session_ids": [],
                "current_internal_session_id": None,
                "metadata": {},
                "branch_count": 0,
                "total_checkpoints": 0
            }
        
        # Override with actual database values for new fields
        if db_sess.session_metadata:
            session_dict["metadata"] = db_sess.session_metadata if isinstance(db_sess.session_metadata, dict) else {}
        session_dict["branch_count"] = db_sess.branch_count or 0
        session_dict["total_checkpoints"] = db_sess.total_checkpoints or 0
        
        session = ExternalSession.from_dict(session_dict)
        session.id = db_sess.id  # Ensure ID is set
        
        return session