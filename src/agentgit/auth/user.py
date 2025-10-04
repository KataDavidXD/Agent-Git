"""User authentication and management module with LangGraph integration."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import hashlib
import secrets


@dataclass
class User:
    """User model for authentication and authorization with session management.
    
    This class represents a user in the Rollback Agent System with
    authentication capabilities, admin privileges support, and LangGraph
    session tracking.
    
    Attributes:
        id: Unique identifier for the user in the database.
        username: Unique username for authentication.
        password_hash: SHA256 hashed password for security.
        is_admin: Flag indicating admin privileges (only rootusr by default).
        created_at: Timestamp when the user was created.
        last_login: Timestamp of the user's last login.
        active_sessions: List of active external session IDs for this user.
        preferences: User-specific preferences for agent behavior.
        api_key: Optional API key for programmatic access.
        session_limit: Maximum number of concurrent sessions allowed.
        metadata: Additional user metadata and settings.
    
    Example:
        >>> user = User(username="john_doe")
        >>> user.set_password("secure_password")
        >>> user.verify_password("secure_password")
        True
        >>> user.to_dict()
        {'id': None, 'username': 'john_doe', 'is_admin': False, ...}
    """
    id: Optional[int] = None
    username: str = ""
    password_hash: str = ""
    is_admin: bool = False
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    active_sessions: List[int] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    api_key: Optional[str] = None
    session_limit: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using SHA256.
        
        Args:
            password: Plain text password to hash.
            
        Returns:
            Hexadecimal string representation of the SHA256 hash.
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str) -> bool:
        """Verify a password against the stored hash.
        
        Args:
            password: Plain text password to verify.
            
        Returns:
            True if password matches, False otherwise.
        """
        return self.password_hash == self.hash_password(password)
    
    def set_password(self, password: str):
        """Set a new password for the user.
        
        Args:
            password: New plain text password to set.
        """
        self.password_hash = self.hash_password(password)
    
    def generate_api_key(self) -> str:
        """Generate a new API key for the user.
        
        Returns:
            Generated API key string.
        """
        self.api_key = f"sk-{secrets.token_urlsafe(32)}"
        return self.api_key
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify an API key against the stored key.
        
        Args:
            api_key: API key to verify.
            
        Returns:
            True if key matches, False otherwise.
        """
        return self.api_key == api_key if self.api_key else False
    
    def add_session(self, session_id: int) -> bool:
        """Add a new active session for the user.
        
        Args:
            session_id: External session ID to add.
            
        Returns:
            True if session was added, False if limit exceeded.
        """
        if len(self.active_sessions) >= self.session_limit:
            return False
        
        if session_id not in self.active_sessions:
            self.active_sessions.append(session_id)
        return True
    
    def remove_session(self, session_id: int):
        """Remove a session from active sessions.
        
        Args:
            session_id: External session ID to remove.
        """
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)
    
    def has_session(self, session_id: int) -> bool:
        """Check if user owns a specific session.
        
        Args:
            session_id: External session ID to check.
            
        Returns:
            True if user owns the session.
        """
        return session_id in self.active_sessions
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference value.
        
        Args:
            key: Preference key.
            default: Default value if key not found.
            
        Returns:
            Preference value or default.
        """
        return self.preferences.get(key, default)
    
    def set_preference(self, key: str, value: Any):
        """Set a user preference.
        
        Args:
            key: Preference key.
            value: Preference value.
        """
        self.preferences[key] = value
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Get LangGraph agent configuration based on user preferences.
        
        Returns:
            Dictionary of agent configuration options.
        """
        config = {
            "temperature": self.get_preference("temperature", 0.7),
            "max_tokens": self.get_preference("max_tokens", 2000),
            "model": self.get_preference("model", "gpt-4"),
            "auto_checkpoint": self.get_preference("auto_checkpoint", True),
            "checkpoint_frequency": self.get_preference("checkpoint_frequency", 5),
            "max_checkpoints": self.get_preference("max_checkpoints", 50),
            "enable_tool_rollback": self.get_preference("enable_tool_rollback", True),
        }
        
        # Add any custom tools preferences
        if "custom_tools" in self.preferences:
            config["custom_tools"] = self.preferences["custom_tools"]
        
        return config
    
    def to_dict(self):
        """Convert user object to dictionary.
        
        Returns:
            Dictionary representation of the user (excludes password_hash).
        """
        return {
            "id": self.id,
            "username": self.username,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "active_sessions": self.active_sessions,
            "preferences": self.preferences,
            "api_key": self.api_key,
            "session_limit": self.session_limit,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create a User instance from a dictionary.
        
        Args:
            data: Dictionary containing user data.
            
        Returns:
            User instance populated with the provided data.
        """
        user = cls()
        user.id = data.get("id")
        user.username = data.get("username", "")
        user.password_hash = data.get("password_hash", "")
        user.is_admin = data.get("is_admin", False)
        user.active_sessions = data.get("active_sessions", [])
        user.preferences = data.get("preferences", {})
        user.api_key = data.get("api_key")
        user.session_limit = data.get("session_limit", 5)
        user.metadata = data.get("metadata", {})
        
        if data.get("created_at"):
            user.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("last_login"):
            user.last_login = datetime.fromisoformat(data["last_login"])
        
        return user