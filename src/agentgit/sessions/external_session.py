"""External session model for the LangGraph rollback agent system.

Represents user-visible sessions that contain multiple internal LangGraph sessions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class ExternalSession:
    """External session model visible to users.
    
    External sessions are the main conversation containers that users interact with.
    Each external session can contain multiple internal sessions created during
    rollback operations, forming a branching timeline structure.
    
    Attributes:
        id: Unique identifier for the session.
        user_id: ID of the user who owns this session.
        session_name: User-friendly name for the session.
        created_at: When the session was created.
        updated_at: When the session was last updated.
        is_active: Whether the session is currently active.
        internal_session_ids: List of internal session IDs (supports LangGraph IDs).
        current_internal_session_id: The currently active internal session ID.
        metadata: Additional session metadata (model config, preferences, etc.).
        branch_count: Number of branches created from rollbacks.
        total_checkpoints: Total number of checkpoints across all internal sessions.
    
    Example:
        >>> session = ExternalSession(
        ...     user_id=1,
        ...     session_name="Project Discussion",
        ...     created_at=datetime.now()
        ... )
        >>> session.add_internal_session("langgraph_session_123")
    """
    
    id: Optional[int] = None
    user_id: int = 0
    session_name: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_active: bool = True
    internal_session_ids: List[str] = field(default_factory=list)
    current_internal_session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    branch_count: int = 0
    total_checkpoints: int = 0
    
    def add_internal_session(self, session_id: str, is_branch: bool = False):
        """Add a new internal session ID to this external session.
        
        Args:
            session_id: The internal session ID to add.
            is_branch: Whether this is a branch from a rollback.
        """
        if session_id not in self.internal_session_ids:
            self.internal_session_ids.append(session_id)
            self.current_internal_session_id = session_id
            if is_branch:
                self.branch_count += 1
            self.updated_at = datetime.now()
    
    def set_current_internal_session(self, session_id: str) -> bool:
        """Set the current active internal session.
        
        Args:
            session_id: The internal session ID to set as current.
            
        Returns:
            True if the session ID exists and was set, False otherwise.
        """
        if session_id in self.internal_session_ids:
            self.current_internal_session_id = session_id
            self.updated_at = datetime.now()
            return True
        return False
    
    def get_branch_info(self) -> Dict[str, Any]:
        """Get information about session branches.
        
        Returns:
            Dictionary with branch statistics.
        """
        return {
            "total_branches": self.branch_count,
            "total_sessions": len(self.internal_session_ids),
            "current_session": self.current_internal_session_id,
            "is_branched": self.branch_count > 0
        }
    
    def update_metadata(self, metadata: Dict[str, Any]):
        """Update session metadata.
        
        Args:
            metadata: Metadata to merge with existing metadata.
        """
        self.metadata.update(metadata)
        self.updated_at = datetime.now()
    
    def increment_checkpoint_count(self):
        """Increment the total checkpoint count."""
        self.total_checkpoints += 1
        self.updated_at = datetime.now()
    
    def get_session_age(self) -> Optional[float]:
        """Get the age of the session in hours.
        
        Returns:
            Age in hours if created_at is set, None otherwise.
        """
        if self.created_at:
            age = datetime.now() - self.created_at
            return age.total_seconds() / 3600
        return None
    
    def to_dict(self) -> dict:
        """Convert session to dictionary representation.
        
        Returns:
            Dictionary with session data.
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_name": self.session_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
            "internal_session_ids": self.internal_session_ids,
            "current_internal_session_id": self.current_internal_session_id,
            "metadata": self.metadata,
            "branch_count": self.branch_count,
            "total_checkpoints": self.total_checkpoints
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExternalSession":
        """Create an ExternalSession from dictionary data.
        
        Args:
            data: Dictionary containing session data.
            
        Returns:
            ExternalSession instance.
        """
        session = cls()
        session.id = data.get("id")
        session.user_id = data.get("user_id", 0)
        session.session_name = data.get("session_name", "")
        session.is_active = data.get("is_active", True)
        session.internal_session_ids = data.get("internal_session_ids", [])
        session.current_internal_session_id = data.get("current_internal_session_id")
        session.metadata = data.get("metadata", {})
        session.branch_count = data.get("branch_count", 0)
        session.total_checkpoints = data.get("total_checkpoints", 0)
        
        if data.get("created_at"):
            session.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            session.updated_at = datetime.fromisoformat(data["updated_at"])
        
        return session