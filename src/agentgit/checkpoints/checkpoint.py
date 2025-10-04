"""Checkpoint model for the LangGraph rollback agent system.

Represents saved states that can be restored with full execution history preservation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any


@dataclass
class Checkpoint:
    """Checkpoint model for saving and restoring LangGraph agent states.
    
    A checkpoint captures the complete state of an internal session at a specific
    point in time, allowing non-destructive rollback to that state by creating 
    a new internal session branch.
    
    Attributes:
        id: Unique identifier for the checkpoint.
        internal_session_id: ID of the internal session this checkpoint was created from.
        checkpoint_name: Optional user-friendly name for the checkpoint.
        session_state: The agent's session state at checkpoint time.
        conversation_history: The conversation history at checkpoint time.
        is_auto: Whether this checkpoint was created automatically.
        created_at: When this checkpoint was created.
        metadata: Additional metadata about the checkpoint.
        user_id: Optional ID of the user who owns this checkpoint.
        tool_invocations: History of tool invocations up to this checkpoint.
    
    Example:
        >>> checkpoint = Checkpoint(
        ...     internal_session_id=1,
        ...     checkpoint_name="Before tool call",
        ...     session_state={"counter": 5},
        ...     is_auto=True
        ... )
    """
    
    id: Optional[int] = None
    internal_session_id: int = 0
    checkpoint_name: Optional[str] = None
    session_state: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    is_auto: bool = False
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[int] = None
    tool_invocations: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert checkpoint to dictionary representation.
        
        Returns:
            Dictionary with checkpoint data.
        """
        return {
            "id": self.id,
            "internal_session_id": self.internal_session_id,
            "checkpoint_name": self.checkpoint_name,
            "session_state": self.session_state,
            "conversation_history": self.conversation_history,
            "is_auto": self.is_auto,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
            "user_id": self.user_id,
            "tool_invocations": self.tool_invocations
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        """Create a Checkpoint from dictionary data.
        
        Args:
            data: Dictionary containing checkpoint data.
            
        Returns:
            Checkpoint instance.
        """
        checkpoint = cls()
        checkpoint.id = data.get("id")
        checkpoint.internal_session_id = data.get("internal_session_id", 0)
        checkpoint.checkpoint_name = data.get("checkpoint_name")
        checkpoint.session_state = data.get("session_state", {})
        checkpoint.conversation_history = data.get("conversation_history", [])
        checkpoint.is_auto = data.get("is_auto", False)
        checkpoint.metadata = data.get("metadata", {})
        checkpoint.user_id = data.get("user_id")
        checkpoint.tool_invocations = data.get("tool_invocations", [])
        
        if data.get("created_at"):
            checkpoint.created_at = datetime.fromisoformat(data["created_at"])
        
        return checkpoint
    
    @classmethod
    def from_internal_session(cls, internal_session, checkpoint_name: Optional[str] = None, 
                            is_auto: bool = False, user_id: Optional[int] = None,
                            tool_invocations: Optional[List[Dict[str, Any]]] = None) -> "Checkpoint":
        """Create a checkpoint from an internal session.
        
        Args:
            internal_session: The internal session to checkpoint.
            checkpoint_name: Optional name for the checkpoint.
            is_auto: Whether this is an automatic checkpoint.
            user_id: Optional user ID who owns this checkpoint.
            tool_invocations: Optional tool invocation history.
            
        Returns:
            New Checkpoint instance with session data.
        """
        # Get the langgraph session ID
        session_id = internal_session.langgraph_session_id
        
        return cls(
            internal_session_id=internal_session.id,
            checkpoint_name=checkpoint_name,
            session_state=internal_session.session_state.copy(),
            conversation_history=internal_session.conversation_history.copy(),
            is_auto=is_auto,
            created_at=datetime.now(),
            user_id=user_id,
            tool_invocations=tool_invocations or [],
            metadata={
                "session_id": session_id,
                "session_type": "langgraph",
                "checkpoint_count": internal_session.checkpoint_count + 1,
                "tool_track_position": 0  # Will be updated by the agent
            }
        )
    
    def get_summary(self) -> str:
        """Get a summary of the checkpoint for display.
        
        Returns:
            Human-readable summary of the checkpoint.
        """
        checkpoint_type = "Auto" if self.is_auto else "Manual"
        created = self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else "unknown"
        messages = len(self.conversation_history)
        tools = len(self.tool_invocations)
        
        summary = f"{self.checkpoint_name or 'Unnamed'} ({checkpoint_type})"
        summary += f"\nCreated: {created}"
        summary += f"\nMessages: {messages}, Tool calls: {tools}"
        
        return summary
    
    def has_tool_invocations(self) -> bool:
        """Check if this checkpoint has any tool invocations.
        
        Returns:
            True if there are tool invocations, False otherwise.
        """
        return len(self.tool_invocations) > 0
    
    def get_tool_track_position(self) -> int:
        """Get the tool track position from metadata.
        
        Returns:
            The tool track position, or 0 if not set.
        """
        return self.metadata.get("tool_track_position", 0)