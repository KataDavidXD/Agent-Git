"""Repository for checkpoint database operations using SQLAlchemy ORM.

Handles CRUD operations for checkpoints in the LangGraph rollback agent system.
"""

from typing import Optional, List, Dict
from datetime import datetime, timezone

from sqlalchemy.orm.attributes import flag_modified
from agentgit.checkpoints.checkpoint import Checkpoint
from agentgit.database.db_config import get_database_path, get_db_connection, init_db
from agentgit.database.models import Checkpoint as CheckpointModel


class CheckpointRepository:
    """Repository for Checkpoint CRUD operations with SQLAlchemy ORM.

    Manages checkpoints which capture complete agent state at specific points,
    allowing rollback functionality.

    Attributes:
        db_path: Path to the database file or connection string.

    Example:
        >>> repo = CheckpointRepository()
        >>> checkpoint = Checkpoint(internal_session_id=1, checkpoint_name="Before action")
        >>> saved = repo.create(checkpoint)
        >>> checkpoints = repo.get_by_internal_session(1)
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the checkpoint repository.

        Args:
            db_path: Path to database. If None, uses configured default.
        """
        self.db_path = db_path or get_database_path()
        self._init_db()

    def _init_db(self):
        """Initialize the checkpoints table if it doesn't exist."""
        init_db()

    def create(self, checkpoint: Checkpoint) -> Checkpoint:
        """Create a new checkpoint.

        Args:
            checkpoint: Checkpoint object to create.

        Returns:
            The created checkpoint with id populated.
        """
        checkpoint_dict = checkpoint.to_dict()

        with get_db_connection(self.db_path) as db_session:
            db_checkpoint = CheckpointModel(
                internal_session_id=checkpoint.internal_session_id,
                checkpoint_name=checkpoint.checkpoint_name,
                checkpoint_data=checkpoint_dict,
                is_auto=checkpoint.is_auto,
                user_id=checkpoint.user_id,
            )
            # created_at is auto-generated
            db_session.add(db_checkpoint)
            db_session.flush()
            checkpoint.id = db_checkpoint.id
            # Update checkpoint.created_at from database
            if db_checkpoint.created_at:
                checkpoint.created_at = db_checkpoint.created_at
                # Sync data field with database column to ensure consistency
                db_checkpoint.checkpoint_data['created_at'] = checkpoint.created_at.isoformat()
                flag_modified(db_checkpoint, 'checkpoint_data')

        return checkpoint

    def get_by_id(self, checkpoint_id: int) -> Optional[Checkpoint]:
        """Get a checkpoint by ID.

        Args:
            checkpoint_id: The ID of the checkpoint to retrieve.

        Returns:
            Checkpoint if found, None otherwise.
        """
        with get_db_connection(self.db_path) as db_session:
            db_checkpoint = db_session.query(CheckpointModel).filter_by(id=checkpoint_id).first()
            if db_checkpoint:
                return self._row_to_checkpoint(db_checkpoint)
        return None

    def get_by_internal_session(self, internal_session_id: int, auto_only: bool = False) -> List[Checkpoint]:
        """Get all checkpoints for an internal session.

        Args:
            internal_session_id: The ID of the internal session.
            auto_only: If True, only return automatic checkpoints.

        Returns:
            List of Checkpoint objects, ordered by created_at descending, then id descending.
        """
        with get_db_connection(self.db_path) as db_session:
            query = db_session.query(CheckpointModel).filter_by(internal_session_id=internal_session_id)
            if auto_only:
                query = query.filter_by(is_auto=True)
            db_checkpoints = query.order_by(CheckpointModel.created_at.desc(), CheckpointModel.id.desc()).all()
            return [self._row_to_checkpoint(db_cp) for db_cp in db_checkpoints]

    def get_latest_checkpoint(self, internal_session_id: int) -> Optional[Checkpoint]:
        """Get the most recent checkpoint for an internal session.

        Args:
            internal_session_id: The ID of the internal session.

        Returns:
            The latest Checkpoint if found, None otherwise.
        """
        with get_db_connection(self.db_path) as db_session:
            db_checkpoint = db_session.query(CheckpointModel).filter_by(
                internal_session_id=internal_session_id
            ).order_by(CheckpointModel.created_at.desc(), CheckpointModel.id.desc()).first()
            if db_checkpoint:
                return self._row_to_checkpoint(db_checkpoint)
        return None

    def delete(self, checkpoint_id: int) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: The ID of the checkpoint to delete.

        Returns:
            True if deletion successful, False otherwise.
        """
        with get_db_connection(self.db_path) as db_session:
            db_checkpoint = db_session.query(CheckpointModel).filter_by(id=checkpoint_id).first()
            if db_checkpoint:
                db_session.delete(db_checkpoint)
                return True
            return False

    def delete_auto_checkpoints(self, internal_session_id: int, keep_latest: int = 5) -> int:
        """Delete old automatic checkpoints, keeping only the most recent ones.

        Args:
            internal_session_id: The ID of the internal session.
            keep_latest: Number of latest auto checkpoints to keep.

        Returns:
            Number of checkpoints deleted.
        """
        with get_db_connection(self.db_path) as db_session:
            # Find IDs of checkpoints to keep
            keep_checkpoints = db_session.query(CheckpointModel.id).filter_by(
                internal_session_id=internal_session_id,
                is_auto=True
            ).order_by(CheckpointModel.created_at.desc(), CheckpointModel.id.desc()).limit(keep_latest).all()
            
            keep_ids = [cp.id for cp in keep_checkpoints]
            
            if keep_ids:
                # Delete auto checkpoints not in the keep list
                deleted = db_session.query(CheckpointModel).filter(
                    CheckpointModel.internal_session_id == internal_session_id,
                    CheckpointModel.is_auto == True,
                    CheckpointModel.id.notin_(keep_ids)
                ).delete(synchronize_session=False)
                return deleted
            else:
                # Delete all auto checkpoints if none to keep
                deleted = db_session.query(CheckpointModel).filter_by(
                    internal_session_id=internal_session_id,
                    is_auto=True
                ).delete(synchronize_session=False)
                return deleted

    def count_checkpoints(self, internal_session_id: int) -> Dict[str, int]:
        """Count checkpoints for an internal session.

        Args:
            internal_session_id: The ID of the internal session.

        Returns:
            Dictionary with counts: {'total': n, 'auto': n, 'manual': n}
        """
        with get_db_connection(self.db_path) as db_session:
            total = db_session.query(CheckpointModel).filter_by(
                internal_session_id=internal_session_id
            ).count()
            auto = db_session.query(CheckpointModel).filter_by(
                internal_session_id=internal_session_id,
                is_auto=True
            ).count()
            manual = total - auto
            
            return {
                'total': total,
                'auto': auto,
                'manual': manual
            }

    def get_by_user(self, user_id: int, limit: Optional[int] = None) -> List[Checkpoint]:
        """Get all checkpoints for a specific user.

        Args:
            user_id: The ID of the user.
            limit: Optional limit on number of checkpoints to return.

        Returns:
            List of Checkpoint objects, ordered by created_at descending.
        """
        with get_db_connection(self.db_path) as db_session:
            query = db_session.query(CheckpointModel).filter_by(user_id=user_id).order_by(
                CheckpointModel.created_at.desc(), CheckpointModel.id.desc()
            )
            if limit:
                query = query.limit(limit)
            db_checkpoints = query.all()
            return [self._row_to_checkpoint(db_cp) for db_cp in db_checkpoints]

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

        with get_db_connection(self.db_path) as db_session:
            db_checkpoint = db_session.query(CheckpointModel).filter_by(id=checkpoint_id).first()
            if db_checkpoint:
                db_checkpoint.checkpoint_data = checkpoint_dict
                return True
            return False

    def search_checkpoints(self, internal_session_id: int, search_term: str) -> List[Checkpoint]:
        """Search checkpoints by name or content.

        Args:
            internal_session_id: The ID of the internal session.
            search_term: Term to search for in checkpoint names.

        Returns:
            List of matching Checkpoint objects.
        """
        with get_db_connection(self.db_path) as db_session:
            like_pattern = f"%{search_term}%"
            # For JSON fields, search is database-dependent
            # Simple approach: filter by name and manually filter data
            db_checkpoints = db_session.query(CheckpointModel).filter(
                CheckpointModel.internal_session_id == internal_session_id,
                CheckpointModel.checkpoint_name.like(like_pattern)
            ).order_by(CheckpointModel.created_at.desc(), CheckpointModel.id.desc()).all()
            
            return [self._row_to_checkpoint(db_cp) for db_cp in db_checkpoints]

    def _row_to_checkpoint(self, db_cp: CheckpointModel) -> Checkpoint:
        """Convert a database model to a Checkpoint object.

        Args:
            db_cp: CheckpointModel instance from database.
            
        Returns:
            Checkpoint object.
        """
        checkpoint_dict = db_cp.checkpoint_data if isinstance(db_cp.checkpoint_data, dict) else {}
        checkpoint = Checkpoint.from_dict(checkpoint_dict)
        checkpoint.id = db_cp.id  # Ensure ID is set
        
        return checkpoint