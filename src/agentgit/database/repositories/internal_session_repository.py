"""Repository for internal session database operations using SQLAlchemy ORM.

Handles CRUD operations for internal langgraph sessions in the rollback agent system.
"""

from typing import Optional, List
from datetime import datetime, timezone

from agentgit.sessions.internal_session import InternalSession
from agentgit.database.db_config import get_database_path, get_db_connection, init_db
from agentgit.database.models import InternalSession as InternalSessionModel


class InternalSessionRepository:
    """Repository for InternalSession CRUD operations with SQLAlchemy ORM.
    
    Manages internal langgraph sessions which are the actual agent sessions
    running within external sessions.
    
    Attributes:
        db_path: Path to the database file or connection string.
    
    Example:
        >>> repo = InternalSessionRepository()
        >>> session = InternalSession(external_session_id=1, langgraph_session_id="langgraph_123")
        >>> saved = repo.create(session)
        >>> sessions = repo.get_by_external_session(1)
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the internal session repository.
        
        Args:
            db_path: Path to database. If None, uses configured default.
        """
        self.db_path = db_path or get_database_path()
        self._init_db()
    
    def _init_db(self):
        """Initialize the internal sessions table if it doesn't exist."""
        init_db()
    
    def create(self, session: InternalSession) -> InternalSession:
        """Create a new internal session.
        
        Args:
            session: InternalSession object to create.
            
        Returns:
            The created session with id populated.
        """
        # Mark other sessions as not current
        if session.is_current:
            self._mark_all_not_current(session.external_session_id)
        
        with get_db_connection(self.db_path) as db_session:
            db_internal_session = InternalSessionModel(
                external_session_id=session.external_session_id,
                langgraph_session_id=session.langgraph_session_id,
                state_data=session.session_state,
                conversation_history=session.conversation_history,
                is_current=session.is_current,
                checkpoint_count=session.checkpoint_count,
                parent_session_id=session.parent_session_id,
                branch_point_checkpoint_id=session.branch_point_checkpoint_id,
                tool_invocation_count=session.tool_invocation_count,
                session_metadata=session.metadata,
            )
            # created_at is auto-generated
            db_session.add(db_internal_session)
            db_session.flush()
            session.id = db_internal_session.id
            # Update session.created_at from database
            if db_internal_session.created_at:
                session.created_at = db_internal_session.created_at
        
        return session
    
    def update(self, session: InternalSession) -> bool:
        """Update an existing internal session.
        
        Updates session state and conversation history.
        
        Args:
            session: InternalSession object with updated data.
            
        Returns:
            True if update successful, False if session not found.
        """
        if not session.id:
            return False
        
        # Mark other sessions as not current if this one is current
        if session.is_current:
            self._mark_all_not_current(session.external_session_id, exclude_id=session.id)
        
        with get_db_connection(self.db_path) as db_session:
            db_internal_session = db_session.query(InternalSessionModel).filter_by(id=session.id).first()
            if db_internal_session:
                db_internal_session.state_data = session.session_state
                db_internal_session.conversation_history = session.conversation_history
                db_internal_session.is_current = session.is_current
                db_internal_session.checkpoint_count = session.checkpoint_count
                db_internal_session.tool_invocation_count = session.tool_invocation_count
                db_internal_session.session_metadata = session.metadata
                return True
            return False
    
    def get_by_id(self, session_id: int) -> Optional[InternalSession]:
        """Get an internal session by ID.
        
        Args:
            session_id: The ID of the session to retrieve.
            
        Returns:
            InternalSession if found, None otherwise.
        """
        with get_db_connection(self.db_path) as db_session:
            db_internal_session = db_session.query(InternalSessionModel).filter_by(id=session_id).first()
            if db_internal_session:
                return self._row_to_session(db_internal_session)
        return None
    
    def get_by_langgraph_session_id(self, langgraph_session_id: str) -> Optional[InternalSession]:
        """Get an internal session by langgraph session ID.
        
        Args:
            langgraph_session_id: The langgraph session ID.
            
        Returns:
            InternalSession if found, None otherwise.
        """
        with get_db_connection(self.db_path) as db_session:
            db_internal_session = db_session.query(InternalSessionModel).filter_by(
                langgraph_session_id=langgraph_session_id
            ).first()
            if db_internal_session:
                return self._row_to_session(db_internal_session)
        return None
    
    def get_by_external_session(self, external_session_id: int) -> List[InternalSession]:
        """Get all internal sessions for an external session.
        
        Args:
            external_session_id: The ID of the external session.
            
        Returns:
            List of InternalSession objects, ordered by created_at descending.
        """
        with get_db_connection(self.db_path) as db_session:
            db_sessions = db_session.query(InternalSessionModel).filter_by(
                external_session_id=external_session_id
            ).order_by(InternalSessionModel.created_at.desc()).all()
            return [self._row_to_session(db_sess) for db_sess in db_sessions]
    
    def get_current_session(self, external_session_id: int) -> Optional[InternalSession]:
        """Get the current internal session for an external session.
        
        Args:
            external_session_id: The ID of the external session.
            
        Returns:
            The current InternalSession if found, None otherwise.
        """
        with get_db_connection(self.db_path) as db_session:
            db_internal_session = db_session.query(InternalSessionModel).filter_by(
                external_session_id=external_session_id,
                is_current=True
            ).first()
            if db_internal_session:
                return self._row_to_session(db_internal_session)
        return None
    
    def set_current_session(self, session_id: int) -> bool:
        """Set an internal session as the current one for its external session.
        
        Args:
            session_id: The ID of the session to set as current.
            
        Returns:
            True if successful, False if session not found.
        """
        session = self.get_by_id(session_id)
        if not session:
            return False
        
        # Mark all others as not current
        self._mark_all_not_current(session.external_session_id)
        
        # Mark this one as current
        with get_db_connection(self.db_path) as db_session:
            db_internal_session = db_session.query(InternalSessionModel).filter_by(id=session_id).first()
            if db_internal_session:
                db_internal_session.is_current = True
                return True
            return False
    
    def delete(self, session_id: int) -> bool:
        """Delete an internal session.
        
        Args:
            session_id: The ID of the session to delete.
            
        Returns:
            True if deletion successful, False otherwise.
        """
        with get_db_connection(self.db_path) as db_session:
            db_internal_session = db_session.query(InternalSessionModel).filter_by(id=session_id).first()
            if db_internal_session:
                db_session.delete(db_internal_session)
                return True
            return False
    
    def count_sessions(self, external_session_id: int) -> int:
        """Count internal sessions for an external session.
        
        Args:
            external_session_id: The ID of the external session.
            
        Returns:
            Number of internal sessions.
        """
        with get_db_connection(self.db_path) as db_session:
            return db_session.query(InternalSessionModel).filter_by(
                external_session_id=external_session_id
            ).count()
    
    def get_branch_sessions(self, parent_session_id: int) -> List[InternalSession]:
        """Get all sessions branched from a parent session.
        
        Args:
            parent_session_id: The ID of the parent session.
            
        Returns:
            List of InternalSession objects branched from the parent.
        """
        with get_db_connection(self.db_path) as db_session:
            db_sessions = db_session.query(InternalSessionModel).filter_by(
                parent_session_id=parent_session_id
            ).order_by(InternalSessionModel.created_at.desc()).all()
            return [self._row_to_session(db_sess) for db_sess in db_sessions]
    
    def get_session_lineage(self, session_id: int) -> List[InternalSession]:
        """Get the lineage of a session (path from root to this session).
        
        Args:
            session_id: The ID of the session.
            
        Returns:
            List of InternalSession objects from root to current session.
        """
        lineage = []
        current_id = session_id
        
        while current_id:
            session = self.get_by_id(current_id)
            if not session:
                break
            lineage.append(session)
            current_id = session.parent_session_id
        
        return list(reversed(lineage))
    
    def update_tool_count(self, session_id: int, increment: int = 1) -> bool:
        """Update the tool invocation count for a session.
        
        Args:
            session_id: The ID of the session.
            increment: Amount to increment by.
            
        Returns:
            True if update successful, False otherwise.
        """
        with get_db_connection(self.db_path) as db_session:
            db_internal_session = db_session.query(InternalSessionModel).filter_by(id=session_id).first()
            if db_internal_session:
                db_internal_session.tool_invocation_count = (db_internal_session.tool_invocation_count or 0) + increment
                return True
            return False
    
    def _mark_all_not_current(self, external_session_id: int, exclude_id: Optional[int] = None):
        """Mark all internal sessions as not current for an external session.
        
        Args:
            external_session_id: The ID of the external session.
            exclude_id: Optional ID to exclude from the update.
        """
        with get_db_connection(self.db_path) as db_session:
            query = db_session.query(InternalSessionModel).filter_by(external_session_id=external_session_id)
            if exclude_id:
                query = query.filter(InternalSessionModel.id != exclude_id)
            query.update({"is_current": False})
    
    def _row_to_session(self, db_sess: InternalSessionModel) -> InternalSession:
        """Convert a database model to an InternalSession object.
        
        Args:
            db_sess: InternalSessionModel instance from database.
            
        Returns:
            InternalSession object.
        """
        session = InternalSession(
            id=db_sess.id,
            external_session_id=db_sess.external_session_id,
            langgraph_session_id=db_sess.langgraph_session_id,
            session_state=db_sess.state_data if isinstance(db_sess.state_data, dict) else {},
            conversation_history=db_sess.conversation_history if isinstance(db_sess.conversation_history, list) else [],
            created_at=db_sess.created_at,
            is_current=db_sess.is_current,
            checkpoint_count=db_sess.checkpoint_count or 0,
            parent_session_id=db_sess.parent_session_id,
            branch_point_checkpoint_id=db_sess.branch_point_checkpoint_id,
            tool_invocation_count=db_sess.tool_invocation_count or 0,
            metadata=db_sess.session_metadata if isinstance(db_sess.session_metadata, dict) else {}
        )
        
        return session