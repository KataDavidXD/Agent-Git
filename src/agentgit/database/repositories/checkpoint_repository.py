"""Repository for checkpoint database operations.

Handles CRUD operations for checkpoints in the LangGraph rollback agent system.
"""

import sqlite3
import json
from typing import Optional, List, Dict
from datetime import datetime

from agentgit.checkpoints.checkpoint import Checkpoint
from agentgit.database.db_config import get_database_path



class CheckpointRepository:
    """Repository for Checkpoint CRUD operations with SQLite.
    
    Manages checkpoints which capture complete agent state at specific points,
    allowing rollback functionality.
    
    Attributes:
        db_path: Path to the SQLite database file.
    
    Example:
        >>> repo = CheckpointRepository()
        >>> checkpoint = Checkpoint(internal_session_id=1, checkpoint_name="Before action")
        >>> saved = repo.create(checkpoint)
        >>> checkpoints = repo.get_by_internal_session(1)
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the checkpoint repository.
        
        Args:
            db_path: Path to SQLite database. If None, uses configured default.
        """
        self.db_path = db_path or get_database_path()
        self._init_db()
    
    def _init_db(self):
        """Initialize the checkpoints table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    internal_session_id INTEGER NOT NULL,
                    checkpoint_name TEXT,
                    checkpoint_data TEXT NOT NULL,
                    is_auto INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    user_id INTEGER,
                    FOREIGN KEY (internal_session_id) REFERENCES internal_sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
                )
            """)
            
            # Check if we need to add user_id column (for migration)
            cursor.execute("PRAGMA table_info(checkpoints)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'user_id' not in columns:
                cursor.execute("ALTER TABLE checkpoints ADD COLUMN user_id INTEGER")
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_session 
                ON checkpoints(internal_session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_created 
                ON checkpoints(created_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_user 
                ON checkpoints(user_id)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def create(self, checkpoint: Checkpoint) -> Checkpoint:
        """Create a new checkpoint.
        
        Args:
            checkpoint: Checkpoint object to create.
            
        Returns:
            The created checkpoint with id populated.
        """
        if not checkpoint.created_at:
            checkpoint.created_at = datetime.now()
        
        checkpoint_dict = checkpoint.to_dict()
        json_data = json.dumps(checkpoint_dict)
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                INSERT INTO checkpoints 
                (internal_session_id, checkpoint_name, checkpoint_data, is_auto, created_at, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                checkpoint.internal_session_id,
                checkpoint.checkpoint_name,
                json_data,
                1 if checkpoint.is_auto else 0,
                checkpoint.created_at.isoformat(),
                checkpoint.user_id
            ))
            
            checkpoint.id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()
        
        return checkpoint
    
    def get_by_id(self, checkpoint_id: int) -> Optional[Checkpoint]:
        """Get a checkpoint by ID.
        
        Args:
            checkpoint_id: The ID of the checkpoint to retrieve.
            
        Returns:
            Checkpoint if found, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT id, internal_session_id, checkpoint_name, checkpoint_data, 
                       is_auto, created_at
                FROM checkpoints
                WHERE id = ?
            """, (checkpoint_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_checkpoint(row)
        finally:
            conn.close()
        
        return None
    
    def get_by_internal_session(self, internal_session_id: int, 
                               auto_only: bool = False) -> List[Checkpoint]:
        """Get all checkpoints for an internal session.
        
        Args:
            internal_session_id: The ID of the internal session.
            auto_only: If True, only return automatic checkpoints.
            
        Returns:
            List of Checkpoint objects, ordered by created_at descending.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            if auto_only:
                cursor.execute("""
                    SELECT id, internal_session_id, checkpoint_name, checkpoint_data, 
                           is_auto, created_at
                    FROM checkpoints
                    WHERE internal_session_id = ? AND is_auto = 1
                    ORDER BY created_at DESC
                """, (internal_session_id,))
            else:
                cursor.execute("""
                    SELECT id, internal_session_id, checkpoint_name, checkpoint_data, 
                           is_auto, created_at
                    FROM checkpoints
                    WHERE internal_session_id = ?
                    ORDER BY created_at DESC
                """, (internal_session_id,))
            
            rows = cursor.fetchall()
            return [self._row_to_checkpoint(row) for row in rows]
        finally:
            conn.close()
    
    def get_latest_checkpoint(self, internal_session_id: int) -> Optional[Checkpoint]:
        """Get the most recent checkpoint for an internal session.
        
        Args:
            internal_session_id: The ID of the internal session.
            
        Returns:
            The latest Checkpoint if found, None otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT id, internal_session_id, checkpoint_name, checkpoint_data, 
                       is_auto, created_at
                FROM checkpoints
                WHERE internal_session_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (internal_session_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_checkpoint(row)
        finally:
            conn.close()
        
        return None
    
    def delete(self, checkpoint_id: int) -> bool:
        """Delete a checkpoint.
        
        Args:
            checkpoint_id: The ID of the checkpoint to delete.
            
        Returns:
            True if deletion successful, False otherwise.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                DELETE FROM checkpoints WHERE id = ?
            """, (checkpoint_id,))
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete_auto_checkpoints(self, internal_session_id: int, keep_latest: int = 5) -> int:
        """Delete old automatic checkpoints, keeping only the most recent ones.
        
        Args:
            internal_session_id: The ID of the internal session.
            keep_latest: Number of latest auto checkpoints to keep.
            
        Returns:
            Number of checkpoints deleted.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Find IDs of checkpoints to keep
            cursor.execute("""
                SELECT id FROM checkpoints
                WHERE internal_session_id = ? AND is_auto = 1
                ORDER BY created_at DESC
                LIMIT ?
            """, (internal_session_id, keep_latest))
            
            keep_ids = [row[0] for row in cursor.fetchall()]
            
            if keep_ids:
                # Delete auto checkpoints not in the keep list
                placeholders = ','.join('?' * len(keep_ids))
                cursor.execute(f"""
                    DELETE FROM checkpoints
                    WHERE internal_session_id = ? AND is_auto = 1 AND id NOT IN ({placeholders})
                """, [internal_session_id] + keep_ids)
            else:
                # Delete all auto checkpoints if none to keep
                cursor.execute("""
                    DELETE FROM checkpoints
                    WHERE internal_session_id = ? AND is_auto = 1
                """, (internal_session_id,))
            
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
    
    def count_checkpoints(self, internal_session_id: int) -> Dict[str, int]:
        """Count checkpoints for an internal session.
        
        Args:
            internal_session_id: The ID of the internal session.
            
        Returns:
            Dictionary with counts: {'total': n, 'auto': n, 'manual': n}
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN is_auto = 1 THEN 1 ELSE 0 END) as auto,
                    SUM(CASE WHEN is_auto = 0 THEN 1 ELSE 0 END) as manual
                FROM checkpoints
                WHERE internal_session_id = ?
            """, (internal_session_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'total': row[0] or 0,
                    'auto': row[1] or 0,
                    'manual': row[2] or 0
                }
            
            return {'total': 0, 'auto': 0, 'manual': 0}
        finally:
            conn.close()
    
    def get_by_user(self, user_id: int, limit: Optional[int] = None) -> List[Checkpoint]:
        """Get all checkpoints for a specific user.
        
        Args:
            user_id: The ID of the user.
            limit: Optional limit on number of checkpoints to return.
            
        Returns:
            List of Checkpoint objects, ordered by created_at descending.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            query = """
                SELECT id, internal_session_id, checkpoint_name, checkpoint_data, 
                       is_auto, created_at
                FROM checkpoints
                WHERE user_id = ?
                ORDER BY created_at DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (user_id,))
            rows = cursor.fetchall()
            return [self._row_to_checkpoint(row) for row in rows]
        finally:
            conn.close()
    
    def get_checkpoints_with_tools(self, internal_session_id: int) -> List[Checkpoint]:
        """Get checkpoints that have tool invocations.
        
        Args:
            internal_session_id: The ID of the internal session.
            
        Returns:
            List of Checkpoint objects that have tool invocations.
        """
        checkpoints = self.get_by_internal_session(internal_session_id)
        # Filter checkpoints that have tool invocations
        return [cp for cp in checkpoints if cp.has_tool_invocations()]
    
    def update_checkpoint_metadata(self, checkpoint_id: int, metadata: Dict) -> bool:
        """Update the metadata of a checkpoint.
        
        Useful for updating tool track positions or other metadata after creation.
        
        Args:
            checkpoint_id: The ID of the checkpoint to update.
            metadata: New metadata to merge with existing metadata.
            
        Returns:
            True if update successful, False otherwise.
        """
        checkpoint = self.get_by_id(checkpoint_id)
        if not checkpoint:
            return False
        
        # Merge metadata
        checkpoint.metadata.update(metadata)
        
        # Save updated checkpoint
        checkpoint_dict = checkpoint.to_dict()
        json_data = json.dumps(checkpoint_dict)
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("""
                UPDATE checkpoints
                SET checkpoint_data = ?
                WHERE id = ?
            """, (json_data, checkpoint_id))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def search_checkpoints(self, internal_session_id: int, search_term: str) -> List[Checkpoint]:
        """Search checkpoints by name or content.
        
        Args:
            internal_session_id: The ID of the internal session.
            search_term: Term to search for in checkpoint names.
            
        Returns:
            List of matching Checkpoint objects.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute("""
                SELECT id, internal_session_id, checkpoint_name, checkpoint_data, 
                       is_auto, created_at
                FROM checkpoints
                WHERE internal_session_id = ? 
                  AND (checkpoint_name LIKE ? OR checkpoint_data LIKE ?)
                ORDER BY created_at DESC
            """, (internal_session_id, f"%{search_term}%", f"%{search_term}%"))
            
            rows = cursor.fetchall()
            return [self._row_to_checkpoint(row) for row in rows]
        finally:
            conn.close()
    
    def _row_to_checkpoint(self, row) -> Checkpoint:
        """Convert a database row to a Checkpoint object.
        
        Args:
            row: Tuple containing database fields.
            
        Returns:
            Checkpoint object.
        """
        checkpoint_id, internal_session_id, checkpoint_name, json_data, is_auto, created_at = row
        
        checkpoint_dict = json.loads(json_data)
        checkpoint = Checkpoint.from_dict(checkpoint_dict)
        checkpoint.id = checkpoint_id  # Ensure ID is set
        
        return checkpoint