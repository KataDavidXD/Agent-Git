"""SQLAlchemy ORM models for the Agent-Git database.

This module defines database models with support for both PostgreSQL and SQLite.
Field types are conditionally defined based on the database backend.
"""

import os
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime,
    ForeignKey, create_engine, Index, func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.types import JSON


# Create declarative base
Base = declarative_base()
AdaptiveJSON = JSON().with_variant(JSONB, 'postgresql')


class User(Base):
    """User model representing system users.
    
    Attributes:
        id: Primary key
        username: Unique username
        password_hash: Hashed password
        is_admin: Admin flag
        created_at: ISO format datetime string
        last_login: ISO format datetime string
        data: JSON string/TEXT containing full user data
        api_key: Optional API key
        session_limit: Maximum concurrent sessions (default 5)
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    last_login = Column(DateTime(timezone=True))
    data = Column(AdaptiveJSON)
    api_key = Column(String(255))
    session_limit = Column(Integer, default=5)
    
    # Relationships
    external_sessions = relationship("ExternalSession", back_populates="user", cascade="all, delete-orphan")
    checkpoints = relationship("Checkpoint", back_populates="user") # TODO: add `cascade="all, delete-orphan"`


class ExternalSession(Base):
    """External session model representing user-visible conversation containers.
    
    Attributes:
        id: Primary key
        user_id: Foreign key to User
        session_name: Session name
        created_at: ISO format datetime string
        updated_at: ISO format datetime string
        is_active: Active flag (1) or inactive (0)
        data: JSON string containing full session data
        metadata: JSON string containing session metadata
        branch_count: Number of branches in this session
        total_checkpoints: Total number of checkpoints
    """
    __tablename__ = "external_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True, nullable=False)
    data = Column(AdaptiveJSON)
    session_metadata = Column("metadata", AdaptiveJSON)
    branch_count = Column(Integer, default=0)
    total_checkpoints = Column(Integer, default=0)
    
    # Relationships
    user = relationship("User", back_populates="external_sessions")
    internal_sessions = relationship("InternalSession", back_populates="external_session", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_external_sessions_user", "user_id"), # TODO: delete?
        Index("idx_external_sessions_active", "user_id", "is_active"),
    )


class InternalSession(Base):
    """Internal session model representing actual LangGraph agent sessions.
    
    Attributes:
        id: Primary key
        external_session_id: Foreign key to ExternalSession
        langgraph_session_id: Unique LangGraph session identifier
        state_data: JSON string containing session state
        conversation_history: JSON string containing conversation
        created_at: ISO format datetime string
        is_current: Whether this is the current active session (1 or 0)
        checkpoint_count: Number of checkpoints in this session
        parent_session_id: Foreign key to parent InternalSession (for branching)
        branch_point_checkpoint_id: Checkpoint ID where branch occurred
        tool_invocation_count: Number of tool invocations
        metadata: JSON string containing session metadata
    """
    __tablename__ = "internal_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    external_session_id = Column(Integer, ForeignKey("external_sessions.id", ondelete="CASCADE"), nullable=False)
    langgraph_session_id = Column(String(255), unique=True, nullable=False)
    state_data = Column(AdaptiveJSON)
    conversation_history = Column(AdaptiveJSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    is_current = Column(Boolean, default=False)
    checkpoint_count = Column(Integer, default=0)
    parent_session_id = Column(Integer, ForeignKey("internal_sessions.id", ondelete="SET NULL"))
    branch_point_checkpoint_id = Column(Integer, ForeignKey("checkpoints.id", ondelete="SET NULL"))
    tool_invocation_count = Column(Integer, default=0)
    session_metadata = Column("metadata", AdaptiveJSON)
    
    # Relationships
    external_session = relationship("ExternalSession", back_populates="internal_sessions")
    checkpoints = relationship("Checkpoint", back_populates="internal_session", foreign_keys="Checkpoint.internal_session_id", cascade="all, delete-orphan")
    parent_session = relationship("InternalSession", remote_side=[id], backref="child_sessions")
    
    # Indexes
    __table_args__ = (
        Index("idx_internal_sessions_external", "external_session_id"),
        Index("idx_internal_sessions_langgraph", "langgraph_session_id"),
        Index("idx_internal_sessions_parent", "parent_session_id"),
        Index("idx_internal_sessions_branch", "branch_point_checkpoint_id"),
    )


class Checkpoint(Base):
    """Checkpoint model representing saved agent states.
    
    Attributes:
        id: Primary key
        internal_session_id: Foreign key to InternalSession
        checkpoint_name: Name/description of checkpoint
        checkpoint_data: JSON string containing complete checkpoint data
        is_auto: Whether checkpoint was auto-created (1) or manual (0)
        created_at: ISO format datetime string
        user_id: Foreign key to User (optional)
    """
    __tablename__ = "checkpoints"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    internal_session_id = Column(Integer, ForeignKey("internal_sessions.id", ondelete="CASCADE"), nullable=False)
    checkpoint_name = Column(String(255))
    checkpoint_data = Column(AdaptiveJSON, nullable=False)
    is_auto = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    
    # Relationships
    internal_session = relationship("InternalSession", back_populates="checkpoints", foreign_keys=[internal_session_id])
    user = relationship("User", back_populates="checkpoints")
    
    # Indexes
    __table_args__ = (
        Index("idx_checkpoints_session", "internal_session_id"),
        Index("idx_checkpoints_created", "created_at"),
        Index("idx_checkpoints_user", "user_id"),
    )
