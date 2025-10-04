"""Repository for internal session database operations.

Handles CRUD operations for internal langgraph sessions in the rollback agent system.
"""

import sqlite3
import json
from typing import Optional, List
from datetime import datetime

from agentgit.sessions.internal_session import InternalSession
from agentgit.database.db_config import get_database_path



class InternalSessionRepository:
    """Repository for InternalSession CRUD operations with SQLite.
    
    Manages internal langgraph sessions which are the actual agent sessions
    running within external sessions.
    
    Attributes:
        db_path: Path to the SQLite database file.
    
    Example:
        >>> repo = InternalSessionRepository()
        >>> session = InternalSession(external_session_id=1, langgraph_session_id="langgraph_123")
        >>> saved = repo.create(session)
        >>> sessions = repo.get_by_external_session(1)
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the internal session repository.
        
        Args:
            db_path: Path to SQLite database. If None, uses configured default.
        """
        self.db_path = db_path or get_database_path()
        self._init_db()
    
    def _init_db(self):
        """Initialize the internal sessions table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS internal_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_session_id INTEGER NOT NULL,
                    langgraph_session_id TEXT UNIQUE NOT NULL,
                    state_data TEXT,
                    conversation_history TEXT,
                    created_at TEXT NOT NULL,
                    is_current INTEGER DEFAULT 0,
                    checkpoint_count INTEGER DEFAULT 0,
                    parent_session_id INTEGER,
                    branch_point_checkpoint_id INTEGER,
                    tool_invocation_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    FOREIGN KEY (external_session_id) REFERENCES external_sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (parent_session_id) REFERENCES internal_sessions(id) ON DELETE SET NULL,
                    FOREIGN KEY (branch_point_checkpoint_id) REFERENCES checkpoints(id) ON DELETE SET NULL
                )
            """)
            
            # Check for migration needs
            cursor.execute("PRAGMA table_info(internal_sessions)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Add new columns if they don't exist
            if 'parent_session_id' not in columns:
                cursor.execute("ALTER TABLE internal_sessions ADD COLUMN parent_session_id INTEGER")
            
            if 'branch_point_checkpoint_id' not in columns:
                cursor.execute("ALTER TABLE internal_sessions ADD COLUMN branch_point_checkpoint_id INTEGER")
            
            if 'tool_invocation_count' not in columns:
                cursor.execute("ALTER TABLE internal_sessions ADD COLUMN tool_invocation_count INTEGER DEFAULT 0")
            
            if 'metadata' not in columns:
                cursor.execute("ALTER TABLE internal_sessions ADD COLUMN metadata TEXT")
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_internal_sessions_external 
                ON internal_sessions(external_session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_internal_sessions_langgraph 
                ON internal_sessions(langgraph_session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_internal_sessions_parent 
                ON internal_sessions(parent_session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_internal_sessions_branch 
                ON internal_sessions(branch_point_checkpoint_id)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def create(self, session: InternalSession) -> InternalSession:
        """Create a new internal session.
        
        Args:
            session: InternalSession object to create.
            
        Returns:
            The created session with id populated.
        """
        if not session.created_at:
            session.created_at = datetime.now()
        
        # Mark other sessions as not current
        if session.is_current:
            self._mark_all_not_current(session.external_session_id)
        
        session_dict = session.to_dict()
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                INSERT INTO internal_sessions 
                (external_session_id, langgraph_session_id, state_data, conversation_history, 
                 created_at, is_current, checkpoint_count, parent_session_id,
                 branch_point_checkpoint_id, tool_invocation_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.external_session_id,
                session.langgraph_session_id,
                json.dumps(session.session_state),
                json.dumps(session.conversation_history),
                session.created_at.isoformat(),
                1 if session.is_current else 0,
                session.checkpoint_count,
                session.parent_session_id,
                session.branch_point_checkpoint_id,
                session.tool_invocation_count,
                json.dumps(session.metadata) if session.metadata else None
            ))
            
            session.id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()
        
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
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                UPDATE internal_sessions 
                SET state_data = ?, conversation_history = ?, is_current = ?, 
                    checkpoint_count = ?, tool_invocation_count = ?, metadata = ?
                WHERE id = ?
            """, (
                json.dumps(session.session_state),
                json.dumps(session.conversation_history),
                1 if session.is_current else 0,
                session.checkpoint_count,
                session.tool_invocation_count,
                json.dumps(session.metadata) if session.metadata else None,
                session.id
            ))
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def get_by_id(self, session_id: int) -> Optional[InternalSession]:
        """Get an internal session by ID.
        
        Args:
            session_id: The ID of the session to retrieve.
            
        Returns:
            InternalSession if found, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT id, external_session_id, langgraph_session_id, state_data, 
                       conversation_history, created_at, is_current, checkpoint_count,
                       parent_session_id, branch_point_checkpoint_id, 
                       tool_invocation_count, metadata
                FROM internal_sessions
                WHERE id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_session(row)
        finally:
            conn.close()
        
        return None
    
    def get_by_langgraph_session_id(self, langgraph_session_id: str) -> Optional[InternalSession]:
        """Get an internal session by langgraph session ID.
        
        Args:
            langgraph_session_id: The langgraph session ID.
            
        Returns:
            InternalSession if found, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT id, external_session_id, langgraph_session_id, state_data, 
                       conversation_history, created_at, is_current, checkpoint_count,
                       parent_session_id, branch_point_checkpoint_id, 
                       tool_invocation_count, metadata
                FROM internal_sessions
                WHERE langgraph_session_id = ?
            """, (langgraph_session_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_session(row)
        finally:
            conn.close()
        
        return None
    
    def get_by_external_session(self, external_session_id: int) -> List[InternalSession]:
        """Get all internal sessions for an external session.
        
        Args:
            external_session_id: The ID of the external session.
            
        Returns:
            List of InternalSession objects, ordered by created_at descending.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT id, external_session_id, langgraph_session_id, state_data, 
                       conversation_history, created_at, is_current, checkpoint_count,
                       parent_session_id, branch_point_checkpoint_id, 
                       tool_invocation_count, metadata
                FROM internal_sessions
                WHERE external_session_id = ?
                ORDER BY created_at DESC
            """, (external_session_id,))
            
            rows = cursor.fetchall()
            return [self._row_to_session(row) for row in rows]
        finally:
            conn.close()
    
    def get_current_session(self, external_session_id: int) -> Optional[InternalSession]:
        """Get the current internal session for an external session.
        
        Args:
            external_session_id: The ID of the external session.
            
        Returns:
            The current InternalSession if found, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT id, external_session_id, langgraph_session_id, state_data, 
                       conversation_history, created_at, is_current, checkpoint_count,
                       parent_session_id, branch_point_checkpoint_id, 
                       tool_invocation_count, metadata
                FROM internal_sessions
                WHERE external_session_id = ? AND is_current = 1
                LIMIT 1
            """, (external_session_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_session(row)
        finally:
            conn.close()
        
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
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                UPDATE internal_sessions 
                SET is_current = 1
                WHERE id = ?
            """, (session_id,))
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete(self, session_id: int) -> bool:
        """Delete an internal session.
        
        Args:
            session_id: The ID of the session to delete.
            
        Returns:
            True if deletion successful, False otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                DELETE FROM internal_sessions WHERE id = ?
            """, (session_id,))
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def count_sessions(self, external_session_id: int) -> int:
        """Count internal sessions for an external session.
        
        Args:
            external_session_id: The ID of the external session.
            
        Returns:
            Number of internal sessions.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT COUNT(*) FROM internal_sessions
                WHERE external_session_id = ?
            """, (external_session_id,))
            
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def get_branch_sessions(self, parent_session_id: int) -> List[InternalSession]:
        """Get all sessions branched from a parent session.
        
        Args:
            parent_session_id: The ID of the parent session.
            
        Returns:
            List of InternalSession objects branched from the parent.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT id, external_session_id, langgraph_session_id, state_data, 
                       conversation_history, created_at, is_current, checkpoint_count,
                       parent_session_id, branch_point_checkpoint_id, 
                       tool_invocation_count, metadata
                FROM internal_sessions
                WHERE parent_session_id = ?
                ORDER BY created_at DESC
            """, (parent_session_id,))
            
            rows = cursor.fetchall()
            return [self._row_to_session(row) for row in rows]
        finally:
            conn.close()
    
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
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                UPDATE internal_sessions 
                SET tool_invocation_count = tool_invocation_count + ?
                WHERE id = ?
            """, (increment, session_id))
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def _mark_all_not_current(self, external_session_id: int, exclude_id: Optional[int] = None):
        """Mark all internal sessions as not current for an external session.
        
        Args:
            external_session_id: The ID of the external session.
            exclude_id: Optional ID to exclude from the update.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            if exclude_id:
                cursor.execute("""
                    UPDATE internal_sessions 
                    SET is_current = 0
                    WHERE external_session_id = ? AND id != ?
                """, (external_session_id, exclude_id))
            else:
                cursor.execute("""
                    UPDATE internal_sessions 
                    SET is_current = 0
                    WHERE external_session_id = ?
                """, (external_session_id,))
            
            conn.commit()
        finally:
            conn.close()
    
    def _row_to_session(self, row) -> InternalSession:
        """Convert a database row to an InternalSession object.
        
        Args:
            row: Tuple containing database fields.
            
        Returns:
            InternalSession object.
        """
        # Handle both old and new row formats
        if len(row) == 8:
            # Old format without new columns
            (session_id, external_session_id, langgraph_session_id, state_data, 
             conversation_history, created_at, is_current, checkpoint_count) = row
            parent_session_id = None
            branch_point_checkpoint_id = None
            tool_invocation_count = 0
            metadata = None
        else:
            # New format with all columns
            (session_id, external_session_id, langgraph_session_id, state_data, 
             conversation_history, created_at, is_current, checkpoint_count,
             parent_session_id, branch_point_checkpoint_id, 
             tool_invocation_count, metadata) = row
        
        session = InternalSession(
            id=session_id,
            external_session_id=external_session_id,
            langgraph_session_id=langgraph_session_id,
            session_state=json.loads(state_data) if state_data else {},
            conversation_history=json.loads(conversation_history) if conversation_history else [],
            created_at=datetime.fromisoformat(created_at) if created_at else None,
            is_current=bool(is_current),
            checkpoint_count=checkpoint_count or 0,
            parent_session_id=parent_session_id,
            branch_point_checkpoint_id=branch_point_checkpoint_id,
            tool_invocation_count=tool_invocation_count or 0,
            metadata=json.loads(metadata) if metadata else {}
        )
        
        return session