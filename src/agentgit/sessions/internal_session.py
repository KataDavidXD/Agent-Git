"""Internal session model for the LangGraph rollback agent system.

Represents the actual LangGraph agent sessions within an external session.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
import json


@dataclass
class InternalSession:
    """Internal session model for LangGraph agents.
    
    Represents an actual LangGraph agent session that runs within an external session.
    Stores the session state and conversation history with support for branching timelines.
    
    Attributes:
        id: Unique identifier for the internal session.
        external_session_id: ID of the parent external session.
        langgraph_session_id: The session ID (supports LangGraph IDs).
        session_state: The agent's session state dictionary.
        conversation_history: List of conversation messages.
        created_at: When this internal session was created.
        is_current: Whether this is the current active session.
        checkpoint_count: Number of checkpoints created from this session.
        parent_session_id: ID of parent internal session (for branches).
        branch_point_checkpoint_id: ID of checkpoint this was branched from.
        tool_invocation_count: Total number of tool invocations in this session.
        metadata: Additional session metadata.
    
    Example:
        >>> session = InternalSession(
        ...     external_session_id=1,
        ...     langgraph_session_id="langgraph_abc123",
        ...     session_state={"counter": 0}
        ... )
    """
    
    id: Optional[int] = None
    external_session_id: int = 0
    langgraph_session_id: str = ""  # Renamed for backward compatibility, but supports LangGraph IDs too
    session_state: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: Optional[datetime] = None
    is_current: bool = True
    checkpoint_count: int = 0
    parent_session_id: Optional[int] = None
    branch_point_checkpoint_id: Optional[int] = None
    tool_invocation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs):
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message sender (user, assistant, system).
            content: The message content.
            **kwargs: Additional message metadata.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "turn_number": len([m for m in self.conversation_history if m.get("role") == "user"]) + 1 if role == "user" else None,
            **kwargs
        }
        self.conversation_history.append(message)
    
    def update_state(self, new_state: Dict[str, Any]):
        """Update the session state.
        
        Args:
            new_state: Dictionary with state updates.
        """
        self.session_state.update(new_state)
    
    def increment_tool_count(self, count: int = 1):
        """Increment the tool invocation counter.
        
        Args:
            count: Number to increment by.
        """
        self.tool_invocation_count += count
    
    def is_branch(self) -> bool:
        """Check if this session is a branch from another session.
        
        Returns:
            True if this is a branched session.
        """
        return self.parent_session_id is not None
    
    def get_branch_info(self) -> Dict[str, Any]:
        """Get branch information for this session.
        
        Returns:
            Dictionary with branch details.
        """
        return {
            "is_branch": self.is_branch(),
            "parent_session_id": self.parent_session_id,
            "branch_checkpoint_id": self.branch_point_checkpoint_id,
            "created_from_rollback": self.is_branch()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics.
        
        Returns:
            Dictionary with session statistics.
        """
        user_messages = [m for m in self.conversation_history if m.get("role") == "user"]
        assistant_messages = [m for m in self.conversation_history if m.get("role") == "assistant"]
        
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "checkpoints": self.checkpoint_count,
            "tool_invocations": self.tool_invocation_count,
            "is_active": self.is_current,
            "is_branch": self.is_branch()
        }
    
    def update_metadata(self, metadata: Dict[str, Any]):
        """Update session metadata.
        
        Args:
            metadata: Metadata to merge with existing metadata.
        """
        self.metadata.update(metadata)
    
    def to_dict(self) -> dict:
        """Convert internal session to dictionary representation.
        
        Returns:
            Dictionary with session data.
        """
        return {
            "id": self.id,
            "external_session_id": self.external_session_id,
            "langgraph_session_id": self.langgraph_session_id,
            "session_state": self.session_state,
            "conversation_history": self.conversation_history,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_current": self.is_current,
            "checkpoint_count": self.checkpoint_count,
            "parent_session_id": self.parent_session_id,
            "branch_point_checkpoint_id": self.branch_point_checkpoint_id,
            "tool_invocation_count": self.tool_invocation_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "InternalSession":
        """Create an InternalSession from dictionary data.
        
        Args:
            data: Dictionary containing session data.
            
        Returns:
            InternalSession instance.
        """
        session = cls()
        session.id = data.get("id")
        session.external_session_id = data.get("external_session_id", 0)
        session.langgraph_session_id = data.get("langgraph_session_id", "")
        session.session_state = data.get("session_state", {})
        session.conversation_history = data.get("conversation_history", [])
        session.is_current = data.get("is_current", True)
        session.checkpoint_count = data.get("checkpoint_count", 0)
        session.parent_session_id = data.get("parent_session_id")
        session.branch_point_checkpoint_id = data.get("branch_point_checkpoint_id")
        session.tool_invocation_count = data.get("tool_invocation_count", 0)
        session.metadata = data.get("metadata", {})
        
        if data.get("created_at"):
            session.created_at = datetime.fromisoformat(data["created_at"])
        
        return session
    
    @classmethod
    def create_branch_from_checkpoint(cls, checkpoint, external_session_id: int, 
                                     parent_session_id: int) -> "InternalSession":
        """Create a new internal session branched from a checkpoint.
        
        Args:
            checkpoint: The checkpoint to branch from.
            external_session_id: ID of the external session.
            parent_session_id: ID of the parent internal session.
            
        Returns:
            New InternalSession branched from the checkpoint.
        """
        import uuid
        
        session = cls(
            external_session_id=external_session_id,
            langgraph_session_id=f"langgraph_{uuid.uuid4().hex[:12]}",
            session_state=checkpoint.session_state.copy(),
            conversation_history=checkpoint.conversation_history.copy(),
            created_at=datetime.now(),
            is_current=True,
            parent_session_id=parent_session_id,
            branch_point_checkpoint_id=checkpoint.id,
            metadata={
                "branched_from": checkpoint.checkpoint_name or f"Checkpoint {checkpoint.id}",
                "branch_created_at": datetime.now().isoformat(),
                "session_type": "langgraph"
            }
        )
        
        return session