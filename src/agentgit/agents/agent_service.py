"""Service for managing RollbackAgent instances.

Handles agent creation, rollback operations, and session management.
"""

from typing import Optional, Dict, Any
from datetime import datetime
import os
from langchain_openai import ChatOpenAI

from agentgit.agents.rollback_agent import RollbackAgent
from agentgit.sessions.external_session import ExternalSession
from agentgit.sessions.internal_session import InternalSession
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from agentgit.database.repositories.internal_session_repository import InternalSessionRepository
from agentgit.database.repositories.checkpoint_repository import CheckpointRepository


class AgentService:
    """Service for managing RollbackAgent instances.
    
    Provides high-level operations for creating agents, handling rollbacks,
    and managing the interaction between external and internal sessions.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """Initialize the agent service.
        
        Args:
            model_config: Optional configuration for the AI model.
        """
        self.external_session_repo = ExternalSessionRepository()
        self.internal_session_repo = InternalSessionRepository()
        self.checkpoint_repo = CheckpointRepository()
        
        # Default model configuration
        base_url = os.getenv("BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = self._sanitize_base_url(base_url)
        self.model_config = model_config or {
            "id": "gpt-4o-mini",
            "temperature": 0.7,
            "api_key": api_key,
            "base_url": base_url
        }
        
        self.current_agent: Optional[RollbackAgent] = None

    def _sanitize_base_url(self, raw_url: Optional[str]) -> Optional[str]:
        if not raw_url:
            return None
        url = raw_url.strip().rstrip("/")
        if not url:
            return None
        if not (url.startswith("http://") or url.startswith("https://")):
            url = "https://" + url
        return url
    
    def create_new_agent(
        self,
        external_session_id: int,
        session_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **agent_kwargs
    ) -> RollbackAgent:
        """Create a new RollbackAgent for an external session.
        
        Args:
            external_session_id: ID of the external session.
            session_name: Optional name for the internal session.
            base_url: Optional base URL for the model provider (overrides defaults/env if provided).
            api_key: Optional API key for the model provider (overrides defaults/env if provided).
            **agent_kwargs: Additional arguments for the agent.
            
        Returns:
            The created RollbackAgent instance.
        """
        # Resolve effective model configuration (allow per-call overrides)
        effective_model_config = dict(self.model_config)
        if api_key:
            effective_model_config["api_key"] = api_key
        if base_url:
            effective_model_config["base_url"] = self._sanitize_base_url(base_url)
        
        # Create the model using LangChain's ChatOpenAI
        # Map config keys to LangChain format
        model = ChatOpenAI(
            model=effective_model_config.get("id", "gpt-4o-mini"),
            temperature=effective_model_config.get("temperature", 0.7),
            openai_api_key=effective_model_config.get("api_key"),
            openai_api_base=effective_model_config.get("base_url")
        )
        
        # Create the agent with repositories
        agent = RollbackAgent(
            external_session_id=external_session_id,
            model=model,
            internal_session_repo=self.internal_session_repo,
            checkpoint_repo=self.checkpoint_repo,
            add_history_to_messages=True,
            num_history_runs=5,
            show_tool_calls=True,
            **agent_kwargs
        )
        
        # Set session name if provided
        if session_name and agent.internal_session:
            # Update the internal session name via checkpoint name
            # (Internal sessions don't have names, but we can track it in metadata)
            pass
        
        self.current_agent = agent
        return agent
    
    def resume_agent(
        self,
        external_session_id: int,
        internal_session_id: Optional[int] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Optional[RollbackAgent]:
        """Resume an existing agent session.
        
        Args:
            external_session_id: ID of the external session.
            internal_session_id: Optional specific internal session to resume.
                               If None, uses the current internal session.
            base_url: Optional base URL for the model provider (overrides defaults/env if provided).
            api_key: Optional API key for the model provider (overrides defaults/env if provided).
            
        Returns:
            The resumed RollbackAgent instance, or None if not found.
        """
        # Get the external session
        external_session = self.external_session_repo.get_by_id(external_session_id)
        if not external_session:
            return None
        
        # Get the internal session to resume
        if internal_session_id:
            internal_session = self.internal_session_repo.get_by_id(internal_session_id)
        else:
            internal_session = self.internal_session_repo.get_current_session(external_session_id)
        
        if not internal_session:
            # No existing internal session, create new agent
            return self.create_new_agent(external_session_id)
        
        # Create agent and restore state
        effective_model_config = dict(self.model_config)
        if api_key:
            effective_model_config["api_key"] = api_key
        if base_url:
            effective_model_config["base_url"] = self._sanitize_base_url(base_url)
        
        # Create the model using LangChain's ChatOpenAI
        model = ChatOpenAI(
            model=effective_model_config.get("id", "gpt-4o-mini"),
            temperature=effective_model_config.get("temperature", 0.7),
            openai_api_key=effective_model_config.get("api_key"),
            openai_api_base=effective_model_config.get("base_url")
        )
        
        agent = RollbackAgent(
            external_session_id=external_session_id,
            model=model,
            internal_session_repo=self.internal_session_repo,
            checkpoint_repo=self.checkpoint_repo,
            session_state=internal_session.session_state,
            skip_session_creation=True,  # Don't create a new session, we're resuming
            add_history_to_messages=True,
            num_history_runs=5,
            show_tool_calls=True
        )
        
        # Set the existing internal session (since we skipped creation)
        agent.internal_session = internal_session
        agent.langgraph_session_id = internal_session.langgraph_session_id
        
        # CRITICAL FIX: Restore conversation history for resumed sessions
        # This ensures the agent remembers previous conversations
        if internal_session.conversation_history:
            agent._restored_from_checkpoint = True
            agent._restored_history = internal_session.conversation_history.copy()
        
        self.current_agent = agent
        return agent
    
    def rollback_to_checkpoint(
        self,
        external_session_id: int,
        checkpoint_id: int,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        rollback_tools: bool = True
    ) -> Optional[RollbackAgent]:
        """Create a new agent from a checkpoint (rollback operation).
        
        Args:
            external_session_id: ID of the external session.
            checkpoint_id: ID of the checkpoint to rollback to.
            base_url: Optional base URL for the model provider (overrides defaults/env if provided).
            api_key: Optional API key for the model provider (overrides defaults/env if provided).
            rollback_tools: Whether to rollback tool operations after the checkpoint.
            
        Returns:
            A new RollbackAgent with the checkpoint's state, or None if failed.
        """
        try:
            # If we have a current agent and rollback_tools is enabled, 
            # rollback tool operations after the checkpoint first
            if rollback_tools and self.current_agent:
                checkpoint = self.checkpoint_repo.get_by_id(checkpoint_id)
                if checkpoint and "tool_track_position" in checkpoint.metadata:
                    tool_track_position = checkpoint.metadata["tool_track_position"]
                    print(f"Rolling back tools from position {tool_track_position}...")
                    reverse_results = self.current_agent.rollback_tools_from_track_index(tool_track_position)
                    for rr in reverse_results:
                        if not rr.reversed_successfully:
                            print(f"Warning: Failed to reverse {rr.tool_name}: {rr.error_message}")
            
            effective_model_config = dict(self.model_config)
            if api_key:
                effective_model_config["api_key"] = api_key
            if base_url:
                effective_model_config["base_url"] = self._sanitize_base_url(base_url)
            
            # Create the model using LangChain's ChatOpenAI
            model = ChatOpenAI(
                model=effective_model_config.get("id", "gpt-4o-mini"),
                temperature=effective_model_config.get("temperature", 0.7),
                openai_api_key=effective_model_config.get("api_key"),
                openai_api_base=effective_model_config.get("base_url")
            )
            
            agent = RollbackAgent.from_checkpoint(
                checkpoint_id=checkpoint_id,
                external_session_id=external_session_id,
                model=model,
                checkpoint_repo=self.checkpoint_repo,
                internal_session_repo=self.internal_session_repo,
                add_history_to_messages=True,
                num_history_runs=5,
                show_tool_calls=True
            )
            
            self.current_agent = agent
            return agent
            
        except ValueError as e:
            print(f"Rollback failed: {e}")
            return None
    
    def handle_agent_response(self, agent: RollbackAgent, response: Any) -> bool:
        """Handle agent response and check for rollback requests.
        
        Args:
            agent: The RollbackAgent that generated the response.
            response: The response from the agent.
            
        Returns:
            True if a rollback was requested and should be handled, False otherwise.
        """
        # Check if rollback was requested
        if agent.internal_session and agent.internal_session.session_state.get('rollback_requested'):
            checkpoint_id = agent.internal_session.session_state.get('rollback_checkpoint_id')
            if checkpoint_id:
                # Don't clear the checkpoint_id yet - the caller needs it!
                # Only clear the request flag
                agent.internal_session.session_state['rollback_requested'] = False
                # Keep rollback_checkpoint_id for the caller to use
                agent._save_internal_session()
                
                return True  # Signal that rollback should be performed
        
        return False
    
    def list_internal_sessions(self, external_session_id: int) -> list:
        """List all internal sessions for an external session.
        
        Args:
            external_session_id: ID of the external session.
            
        Returns:
            List of internal sessions.
        """
        return self.internal_session_repo.get_by_external_session(external_session_id)
    
    def list_checkpoints(self, internal_session_id: int) -> list:
        """List all checkpoints for an internal session.
        
        Args:
            internal_session_id: ID of the internal session.
            
        Returns:
            List of checkpoints.
        """
        return self.checkpoint_repo.get_by_internal_session(internal_session_id)
    
    def get_conversation_summary(self, agent: RollbackAgent) -> str:
        """Get a summary of the conversation history.
        
        Args:
            agent: The RollbackAgent instance.
            
        Returns:
            A formatted summary of the conversation.
        """
        history = agent.get_conversation_history()
        
        if not history:
            return "No conversation history yet."
        
        summary = f"Conversation ({len(history)} messages):\n"
        for msg in history[-10:]:  # Show last 10 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            # Truncate long messages
            if len(content) > 100:
                content = content[:97] + "..."
            
            summary += f"\n[{role}] {content}\n"
        
        return summary
    
    def get_active_agent(self, external_session_id: int) -> Optional[RollbackAgent]:
        """Get an active agent for a session if one exists.
        
        Args:
            external_session_id: ID of the external session.
            
        Returns:
            The active RollbackAgent if found, None otherwise.
        """
        return self.active_agents.get(external_session_id)
    
    def cleanup_agent(self, external_session_id: int):
        """Remove an agent from active tracking.
        
        Args:
            external_session_id: ID of the external session.
        """
        if external_session_id in self.active_agents:
            del self.active_agents[external_session_id]
            if self.current_agent and self.current_agent.external_session_id == external_session_id:
                self.current_agent = None
    
    def get_branch_tree(self, external_session_id: int) -> Dict[str, Any]:
        """Get the branch tree structure for an external session.
        
        Args:
            external_session_id: ID of the external session.
            
        Returns:
            Dictionary representing the branch tree structure.
        """
        internal_sessions = self.internal_session_repo.get_by_external_session(external_session_id)
        
        # Build session info map first
        session_info_map = {}
        for session in internal_sessions:
            session_info_map[session.id] = {
                "id": session.id,
                "session_id": session.langgraph_session_id,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "is_current": session.is_current,
                "is_branch": session.is_branch(),
                "checkpoint_count": session.checkpoint_count,
                "tool_invocations": session.tool_invocation_count,
                "children": []
            }
        
        # Build tree structure - first pass: identify roots
        tree = {}
        for session in internal_sessions:
            if not session.parent_session_id:  # Root session
                tree[session.id] = session_info_map[session.id]
        
        # Second pass: add children to parents
        for session in internal_sessions:
            if session.parent_session_id and session.parent_session_id in session_info_map:
                parent_info = session_info_map[session.parent_session_id]
                parent_info["children"].append(session_info_map[session.id])
        
        return tree