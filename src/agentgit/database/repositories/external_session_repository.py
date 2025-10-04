"""Repository for external session database operations.

Handles CRUD operations for external sessions in the LangGraph rollback agent system.
"""

import sqlite3
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

from agentgit.sessions.external_session import ExternalSession
from agentgit.database.db_config import get_database_path
from agentgit.database.db_config import get_database_path



class ExternalSessionRepository:
    """Repository for ExternalSession CRUD operations with SQLite.
    
    Manages external sessions which are the user-visible conversation containers.
    Each external session can contain multiple internal sessions with branching support.
    
    Attributes:
        db_path: Path to the SQLite database file.
    
    Example:
        >>> repo = ExternalSessionRepository()
        >>> session = ExternalSession(user_id=1, session_name="My Chat")
        >>> saved_session = repo.create(session)
        >>> sessions = repo.get_user_sessions(user_id=1)
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the external session repository.
        
        Args:
            db_path: Path to SQLite database. If None, uses configured default.
        """
        self.db_path = db_path or get_database_path()
        self._init_db()
    
    def _init_db(self):
        """Initialize the external sessions table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS external_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    is_active INTEGER DEFAULT 1,
                    data TEXT,
                    metadata TEXT,
                    branch_count INTEGER DEFAULT 0,
                    total_checkpoints INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            
            # Check for migration needs
            cursor.execute("PRAGMA table_info(external_sessions)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Add new columns if they don't exist
            if 'metadata' not in columns:
                cursor.execute("ALTER TABLE external_sessions ADD COLUMN metadata TEXT")
            
            if 'branch_count' not in columns:
                cursor.execute("ALTER TABLE external_sessions ADD COLUMN branch_count INTEGER DEFAULT 0")
            
            if 'total_checkpoints' not in columns:
                cursor.execute("ALTER TABLE external_sessions ADD COLUMN total_checkpoints INTEGER DEFAULT 0")
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_external_sessions_user 
                ON external_sessions(user_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_external_sessions_active 
                ON external_sessions(user_id, is_active)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def create(self, session: ExternalSession) -> ExternalSession:
        """Create a new external session.
        
        Args:
            session: ExternalSession object to create.
            
        Returns:
            The created session with id populated.
            
        Raises:
            sqlite3.IntegrityError: If user_id doesn't exist.
        """
        if not session.created_at:
            session.created_at = datetime.now()
        
        session_dict = session.to_dict()
        json_data = json.dumps(session_dict)
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                INSERT INTO external_sessions 
                (user_id, session_name, created_at, updated_at, is_active, data, 
                 metadata, branch_count, total_checkpoints)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.user_id,
                session.session_name,
                session.created_at.isoformat(),
                session.updated_at.isoformat() if session.updated_at else None,
                1 if session.is_active else 0,
                json_data,
                json.dumps(session.metadata) if session.metadata else None,
                session.branch_count,
                session.total_checkpoints
            ))
            
            session.id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()
        
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
        
        session.updated_at = datetime.now()
        session_dict = session.to_dict()
        json_data = json.dumps(session_dict)
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                UPDATE external_sessions 
                SET session_name = ?, updated_at = ?, is_active = ?, data = ?,
                    metadata = ?, branch_count = ?, total_checkpoints = ?
                WHERE id = ?
            """, (
                session.session_name,
                session.updated_at.isoformat(),
                1 if session.is_active else 0,
                json_data,
                json.dumps(session.metadata) if session.metadata else None,
                session.branch_count,
                session.total_checkpoints,
                session.id
            ))
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_by_id(self, session_id: int) -> Optional[ExternalSession]:
        """Get an external session by ID.
        
        Args:
            session_id: The ID of the session to retrieve.
            
        Returns:
            ExternalSession if found, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT id, user_id, session_name, created_at, updated_at, 
                       is_active, data, metadata, branch_count, total_checkpoints
                FROM external_sessions
                WHERE id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_session(row)
        finally:
            conn.close()
        
        return None
    
    def get_user_sessions(self, user_id: int, active_only: bool = False) -> List[ExternalSession]:
        """Get all sessions for a user.
        
        Args:
            user_id: The ID of the user.
            active_only: If True, only return active sessions.
            
        Returns:
            List of ExternalSession objects, ordered by created_at descending.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            if active_only:
                cursor.execute("""
                    SELECT id, user_id, session_name, created_at, updated_at, 
                           is_active, data, metadata, branch_count, total_checkpoints
                    FROM external_sessions
                    WHERE user_id = ? AND is_active = 1
                    ORDER BY created_at DESC
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT id, user_id, session_name, created_at, updated_at, 
                           is_active, data, metadata, branch_count, total_checkpoints
                    FROM external_sessions
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                """, (user_id,))
            
            rows = cursor.fetchall()
            return [self._row_to_session(row) for row in rows]
        finally:
            conn.close()
    
    def get_by_internal_session(self, langgraph_session_id: str) -> Optional[ExternalSession]:
        """Find the external session containing a specific internal langgraph session.
        
        Args:
            langgraph_session_id: The langgraph session ID to search for.
            
        Returns:
            ExternalSession containing the internal session, None if not found.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT id, user_id, session_name, created_at, updated_at, 
                       is_active, data, metadata, branch_count, total_checkpoints
                FROM external_sessions
                WHERE data LIKE ?
            """, (f'%"{langgraph_session_id}"%',))
            
            rows = cursor.fetchall()
            for row in rows:
                session = self._row_to_session(row)
                if langgraph_session_id in session.internal_session_ids:
                    return session
        finally:
            conn.close()
        
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
        # Get the session first to update its data
        session = self.get_by_id(session_id)
        if not session:
            return False
        
        # Update the session object
        session.is_active = False
        session.updated_at = datetime.now()
        
        # Update both the column and the JSON data
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
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                DELETE FROM external_sessions WHERE id = ?
            """, (session_id,))
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def check_ownership(self, session_id: int, user_id: int) -> bool:
        """Check if a user owns a specific session.
        
        Args:
            session_id: The ID of the session.
            user_id: The ID of the user.
            
        Returns:
            True if the user owns the session, False otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT COUNT(*) FROM external_sessions
                WHERE id = ? AND user_id = ?
            """, (session_id, user_id))
            
            count = cursor.fetchone()[0]
            return count > 0
        finally:
            conn.close()
    
    def count_user_sessions(self, user_id: int, active_only: bool = False) -> int:
        """Count the number of sessions a user has.
        
        Args:
            user_id: The ID of the user.
            active_only: If True, only count active sessions.
            
        Returns:
            The number of sessions.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            if active_only:
                cursor.execute("""
                    SELECT COUNT(*) FROM external_sessions
                    WHERE user_id = ? AND is_active = 1
                """, (user_id,))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM external_sessions
                    WHERE user_id = ?
                """, (user_id,))
            
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def _row_to_session(self, row) -> ExternalSession:
        """Convert a database row to an ExternalSession object.
        
        Args:
            row: Tuple containing database fields.
            
        Returns:
            ExternalSession object with all fields including internal session tracking.
        """
        # Handle both old and new row formats
        if len(row) == 7:
            # Old format without new columns
            session_id, user_id, session_name, created_at, updated_at, is_active, json_data = row
            metadata = None
            branch_count = 0
            total_checkpoints = 0
        else:
            # New format with all columns
            session_id, user_id, session_name, created_at, updated_at, is_active, json_data, metadata, branch_count, total_checkpoints = row
        
        if json_data:
            session_dict = json.loads(json_data)
        else:
            # Fallback for older records without JSON data
            session_dict = {
                "id": session_id,
                "user_id": user_id,
                "session_name": session_name,
                "created_at": created_at,
                "updated_at": updated_at,
                "is_active": bool(is_active),
                "internal_session_ids": [],
                "current_internal_session_id": None,
                "metadata": {},
                "branch_count": 0,
                "total_checkpoints": 0
            }
        
        # Override with actual database values for new fields
        if metadata:
            session_dict["metadata"] = json.loads(metadata)
        session_dict["branch_count"] = branch_count or 0
        session_dict["total_checkpoints"] = total_checkpoints or 0
        
        session = ExternalSession.from_dict(session_dict)
        session.id = session_id  # Ensure ID is set
        
        return session