"""LangGraph Agent with rollback capabilities.

Implements a checkpoint-based rollback system for LangGraph agents with
time-travel through conversation states and full execution history preservation.
"""

from typing import Optional, Dict, Any, List, Callable, Mapping, Annotated, TypedDict, Sequence
from datetime import datetime
import uuid
import operator
import asyncio
from functools import wraps

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agentgit.sessions.internal_session import InternalSession
from agentgit.checkpoints.checkpoint import Checkpoint
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from agentgit.database.repositories.internal_session_repository import InternalSessionRepository
from agentgit.database.repositories.checkpoint_repository import CheckpointRepository
from agentgit.auth.user import User
from agentgit.core.rollback_protocol import ToolRollbackRegistry, ToolSpec


class AgentState(TypedDict):
    """State definition for the LangGraph agent.
    
    Attributes:
        messages: Conversation history as list of messages
        session_state: Custom session state dictionary
        tool_invocations: History of tool invocations
        rollback_requested: Flag indicating if rollback was requested
        rollback_checkpoint_id: ID of checkpoint to rollback to
        current_turn: Current conversation turn number
        user_id: ID of the user owning this session
        user_preferences: User preferences for agent behavior
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    session_state: Dict[str, Any]
    tool_invocations: List[Dict[str, Any]]
    rollback_requested: bool
    rollback_checkpoint_id: Optional[int]
    current_turn: int
    user_id: Optional[int]
    user_preferences: Dict[str, Any]


class RollbackAgent:
    """LangGraph agent with checkpoint and rollback capabilities.
    
    Implements automatic checkpoint creation, database persistence, and
    non-destructive rollback with branch preservation.
    
    Attributes:
        external_session_id: ID of the external session this agent belongs to
        internal_session: Current internal session being used
        auto_checkpoint: Whether to automatically create checkpoints after tool calls
        graph: The compiled LangGraph workflow
        tool_rollback_registry: Registry for tool rollback operations
    """
    
    def __init__(
        self,
        external_session_id: int,
        model,
        tools: Optional[List[BaseTool]] = None,
        auto_checkpoint: bool = True,
        internal_session_repo: Optional[InternalSessionRepository] = None,
        checkpoint_repo: Optional[CheckpointRepository] = None,
        checkpointer: Optional[MemorySaver] = None,
        skip_session_creation: bool = False,
        reverse_tools: Optional[Mapping[str, Callable[[Mapping[str, Any], Any], Any]]] = None,
        user: Optional[User] = None,
        **kwargs
    ):
        """Initialize the RollbackAgent.
        
        Args:
            external_session_id: ID of the external session
            model: The LangChain model to use (e.g., ChatOpenAI)
            tools: List of tools available to the agent
            auto_checkpoint: Whether to auto-checkpoint after tool calls
            internal_session_repo: Repository for internal session operations
            checkpoint_repo: Repository for checkpoint operations
            checkpointer: LangGraph checkpointer for state persistence
            skip_session_creation: Skip creating a new internal session (for resume/rollback)
            reverse_tools: Mapping of tool names to their reverse functions
            user: Optional User object for user-specific configuration
            **kwargs: Additional configuration options
        """
        self.external_session_id = external_session_id
        self.model = model
        self.user = user
        
        # Apply user preferences if user is provided
        if user:
            agent_config = user.get_agent_config()
            self.auto_checkpoint = agent_config.get("auto_checkpoint", auto_checkpoint)
            # Configure model with user preferences
            if hasattr(model, 'temperature'):
                model.temperature = agent_config.get("temperature", 0.7)
            if hasattr(model, 'max_tokens'):
                model.max_tokens = agent_config.get("max_tokens", 2000)
        else:
            self.auto_checkpoint = auto_checkpoint
            
        self.tools = tools or []
        
        # Generate unique session ID
        self.langgraph_session_id = f"langgraph_{uuid.uuid4().hex[:12]}"
        
        # Initialize repositories
        self.internal_session_repo = internal_session_repo or InternalSessionRepository()
        self.checkpoint_repo = checkpoint_repo or CheckpointRepository()
        self.external_session_repo = ExternalSessionRepository()
        
        # Set up checkpointer (Memory by default)
        if not checkpointer:
            self.checkpointer = MemorySaver()
        else:
            self.checkpointer = checkpointer
        
        # Initialize tool rollback registry
        self.tool_rollback_registry = ToolRollbackRegistry()
        self._reverse_tools_map: Dict[str, Callable] = dict(reverse_tools or {})
        
        # Register tools with reverse handlers
        self._register_reversible_tools()
        
        # Add checkpoint management tools
        self._add_checkpoint_tools()
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
        
        # Create internal session if not skipping
        if not skip_session_creation:
            self.internal_session = self._create_internal_session()
            self._register_with_external_session()
        else:
            self.internal_session = None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self._tool_node)
        workflow.add_node("checkpoint", self._checkpoint_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edges
        workflow.add_conditional_edges(
            "agent",
            self._should_use_tools,
            {
                "tools": "tools",
                "checkpoint": "checkpoint",
                "end": END
            }
        )
        
        # Route from tools to checkpoint if auto_checkpoint is enabled
        # Otherwise go directly back to agent
        if self.auto_checkpoint:
            workflow.add_edge("tools", "checkpoint")
        else:
            workflow.add_edge("tools", "agent")
            
        workflow.add_edge("checkpoint", "agent")
        
        # Compile with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """Agent node that processes messages and decides on actions.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with agent's response
        """
        messages = state["messages"]
        
        # Add system message with user context if available
        if self.user and state.get("user_preferences"):
            prefs = state["user_preferences"]
            if prefs.get("system_prompt"):
                messages = [SystemMessage(content=prefs["system_prompt"])] + list(messages)
        
        # Invoke model with tools
        response = self.model.bind_tools(self.tools).invoke(messages)
        
        # Update turn counter
        current_turn = state.get("current_turn", 0) + 1
        
        return {
            "messages": [response],
            "current_turn": current_turn
        }
    
    def _tool_node(self, state: AgentState) -> Dict[str, Any]:
        """Tool execution node.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with tool results
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # Create a tool node instance for execution
        tool_node = ToolNode(self.tools)
        tool_invocations = state.get("tool_invocations", [])
        
        # Get tool calls from last message
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return {"messages": [], "tool_invocations": tool_invocations}
        
        # Execute the tool node with the current state
        # ToolNode handles the execution internally
        try:
            # ToolNode expects the full state and returns messages
            result = tool_node.invoke(state)
            
            # Track tool invocations for our rollback system
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Track in registry if reversible
                if self.tool_rollback_registry.get_tool(tool_name):
                    self.tool_rollback_registry.record_invocation(
                        tool_name, tool_args, "executed", success=True
                    )
                
                # Record invocation
                tool_invocations.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": "executed",
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Return the messages from ToolNode and our tracked invocations
            return {
                "messages": result.get("messages", []),
                "tool_invocations": tool_invocations
            }
            
        except Exception as e:
            # Handle errors
            error_message = AIMessage(
                content=f"Tool execution error: {str(e)}",
                name="tool_executor"
            )
            
            # Track failures
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                if self.tool_rollback_registry.get_tool(tool_name):
                    self.tool_rollback_registry.record_invocation(
                        tool_name, tool_args, None, success=False, error_message=str(e)
                    )
                
                tool_invocations.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "error": str(e),
                    "success": False,
                    "timestamp": datetime.now().isoformat()
                })
            
            return {
                "messages": [error_message],
                "tool_invocations": tool_invocations
            }
    
    def _checkpoint_node(self, state: AgentState) -> Dict[str, Any]:
        """Node that handles automatic checkpoint creation.
        
        Args:
            state: Current agent state
            
        Returns:
            State (unchanged, checkpoint is a side effect)
        """
        if self.auto_checkpoint and self.internal_session:
            # Get last tool invocation
            tool_invocations = state.get("tool_invocations", [])
            if tool_invocations:
                last_tool = tool_invocations[-1]
                tool_name = last_tool.get("tool", "unknown")
                
                # Skip checkpoint for checkpoint management tools
                if not self._is_checkpoint_tool(tool_name):
                    self._create_auto_checkpoint(f"After {tool_name}")
        
        return {}
    
    def _should_use_tools(self, state: AgentState) -> str:
        """Determine next node based on agent's response.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node name: "tools", "checkpoint", or "end"
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # Check if tools were called
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # Check if we should create checkpoint after tools
            if self.auto_checkpoint:
                # We'll handle checkpoint after tool execution
                return "tools"
            return "tools"
        
        # Check for rollback request
        if state.get("rollback_requested"):
            return "end"
        
        return "end"
    
    def _register_reversible_tools(self):
        """Register tools that have reverse handlers."""
        for tool in self.tools:
            tool_name = tool.name
            if tool_name in self._reverse_tools_map:
                reverse_fn = self._reverse_tools_map[tool_name]
                
                # Wrap forward function
                def _forward_wrapper(args: Mapping[str, Any], _tool=tool):
                    return _tool.invoke(args)
                
                self.tool_rollback_registry.register_tool(
                    ToolSpec(
                        name=tool_name,
                        forward=_forward_wrapper,
                        reverse=reverse_fn
                    )
                )
    
    def _add_checkpoint_tools(self):
        """Add checkpoint management tools to the agent."""
        checkpoint_tools = [
            self._create_tool_wrapper(self.create_checkpoint_tool, "create_checkpoint"),
            self._create_tool_wrapper(self.list_checkpoints_tool, "list_checkpoints"),
            self._create_tool_wrapper(self.rollback_to_checkpoint_tool, "rollback_to_checkpoint"),
            self._create_tool_wrapper(self.delete_checkpoint_tool, "delete_checkpoint"),
            self._create_tool_wrapper(self.get_checkpoint_info_tool, "get_checkpoint_info"),
            self._create_tool_wrapper(self.cleanup_auto_checkpoints_tool, "cleanup_auto_checkpoints")
        ]
        
        self.tools.extend(checkpoint_tools)
    
    def _create_tool_wrapper(self, func: Callable, name: str) -> BaseTool:
        """Create a LangChain tool from a function.
        
        Args:
            func: Function to wrap
            name: Tool name
            
        Returns:
            BaseTool instance
        """
        from langchain_core.tools import Tool
        
        # Use the Tool class directly which has better Pydantic v2 support
        # This avoids the compatibility layer that triggers warnings
        return Tool(
            name=name,
            description=func.__doc__ or f"Tool: {name}",
            func=func
        )
    
    def _create_internal_session(self) -> InternalSession:
        """Create a new internal session for this agent.
        
        Returns:
            The created InternalSession object
        """
        internal_session = InternalSession(
            external_session_id=self.external_session_id,
            langgraph_session_id=self.langgraph_session_id,  # Using langgraph session ID
            session_state={},
            created_at=datetime.now(),
            is_current=True
        )
        
        if self.internal_session_repo:
            internal_session = self.internal_session_repo.create(internal_session)
        
        return internal_session
    
    def _register_with_external_session(self):
        """Register this internal session with the external session."""
        if self.external_session_repo and self.internal_session:
            self.external_session_repo.add_internal_session(
                self.external_session_id,
                self.langgraph_session_id
            )
    
    def run(self, message: str, config: Optional[RunnableConfig] = None) -> Any:
        """Run the agent and save state after completion.
        
        Args:
            message: The user message to process
            config: Optional LangGraph configuration
            
        Returns:
            The agent's response
        """
        # Store message in conversation history
        if self.internal_session:
            self.internal_session.add_message("user", message)
        
        # Track user session if user is provided
        if self.user and self.external_session_id:
            self.user.add_session(self.external_session_id)
        
        # Build messages list with full conversation history
        messages = []
        if self.internal_session and self.internal_session.conversation_history:
            # Convert conversation history to messages (excluding the just-added user message)
            for msg in self.internal_session.conversation_history[:-1]:  # Skip last as we just added it
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
        
        # Add the current message
        messages.append(HumanMessage(content=message))
        
        # Create initial state
        initial_state = AgentState(
            messages=messages,
            session_state=self.internal_session.session_state if self.internal_session else {},
            tool_invocations=[],
            rollback_requested=False,
            rollback_checkpoint_id=None,
            current_turn=self.internal_session.conversation_history.count("user") if self.internal_session else 0,
            user_id=self.user.id if self.user else None,
            user_preferences=self.user.preferences if self.user else {}
        )
        
        # Set up config with thread ID
        if not config:
            config = RunnableConfig(
                configurable={"thread_id": self.langgraph_session_id}
            )
        elif "configurable" not in config:
            config["configurable"] = {"thread_id": self.langgraph_session_id}
        elif "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = self.langgraph_session_id
        
        # Invoke the graph
        result = self.graph.invoke(initial_state, config)
        
        # Extract response
        response_content = self._extract_response_content(result["messages"][-1])
        
        # Store response in conversation history
        if self.internal_session:
            self.internal_session.add_message("assistant", response_content)
            self.internal_session.update_state(result.get("session_state", {}))
            self._save_internal_session()
        
        return response_content
    
    async def arun(self, message: str, config: Optional[RunnableConfig] = None) -> Any:
        """Async version of run method.
        
        Args:
            message: The user message to process
            config: Optional LangGraph configuration
            
        Returns:
            The agent's response
        """
        # Store message in conversation history
        if self.internal_session:
            self.internal_session.add_message("user", message)
        
        # Track user session if user is provided
        if self.user and self.external_session_id:
            self.user.add_session(self.external_session_id)
        
        # Build messages list with full conversation history
        messages = []
        if self.internal_session and self.internal_session.conversation_history:
            # Convert conversation history to messages (excluding the just-added user message)
            for msg in self.internal_session.conversation_history[:-1]:  # Skip last as we just added it
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
        
        # Add the current message
        messages.append(HumanMessage(content=message))
        
        # Create initial state
        initial_state = AgentState(
            messages=messages,
            session_state=self.internal_session.session_state if self.internal_session else {},
            tool_invocations=[],
            rollback_requested=False,
            rollback_checkpoint_id=None,
            current_turn=self.internal_session.conversation_history.count("user") if self.internal_session else 0,
            user_id=self.user.id if self.user else None,
            user_preferences=self.user.preferences if self.user else {}
        )
        
        # Set up config
        if not config:
            config = RunnableConfig(
                configurable={"thread_id": self.langgraph_session_id}
            )
        
        # Invoke the graph asynchronously
        result = await self.graph.ainvoke(initial_state, config)
        
        # Extract and store response
        response_content = self._extract_response_content(result["messages"][-1])
        
        if self.internal_session:
            self.internal_session.add_message("assistant", response_content)
            self.internal_session.update_state(result.get("session_state", {}))
            self._save_internal_session()
        
        return response_content
    
    def _extract_response_content(self, message: BaseMessage) -> str:
        """Extract content from a message.
        
        Args:
            message: The message to extract content from
            
        Returns:
            The extracted content as string
        """
        if hasattr(message, 'content'):
            return message.content
        elif isinstance(message, dict) and 'content' in message:
            return message['content']
        return str(message)
    
    def _is_checkpoint_tool(self, tool_name: Optional[str]) -> bool:
        """Check if a tool is a checkpoint management tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if it's a checkpoint tool, False otherwise
        """
        checkpoint_tool_names = {
            'create_checkpoint',
            'list_checkpoints',
            'rollback_to_checkpoint',
            'delete_checkpoint',
            'get_checkpoint_info',
            'cleanup_auto_checkpoints'
        }
        return tool_name in checkpoint_tool_names
    
    def _create_auto_checkpoint(self, name: str):
        """Create an automatic checkpoint.
        
        Args:
            name: Name for the checkpoint
        """
        if self.checkpoint_repo and self.internal_session and self.internal_session.id:
            checkpoint = Checkpoint.from_internal_session(
                self.internal_session,
                checkpoint_name=name,
                is_auto=True
            )
            
            # Store tool track position
            current_track = self.tool_rollback_registry.get_track()
            checkpoint.metadata["tool_track_position"] = len(current_track)
            
            self.checkpoint_repo.create(checkpoint)
            self.internal_session.checkpoint_count += 1
    
    def _save_internal_session(self):
        """Save the current internal session to database."""
        if self.internal_session_repo and self.internal_session and self.internal_session.id:
            self.internal_session_repo.update(self.internal_session)
    
    # Checkpoint management tools
    def create_checkpoint_tool(self, name: Optional[str] = None) -> str:
        """Create a manual checkpoint of the current conversation state.
        
        Args:
            name: Optional name for the checkpoint
            
        Returns:
            Confirmation message
        """
        if not name:
            name = f"Checkpoint at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        if self.checkpoint_repo and self.internal_session and self.internal_session.id:
            checkpoint = Checkpoint.from_internal_session(
                self.internal_session,
                checkpoint_name=name,
                is_auto=False
            )
            
            # Store tool track position
            current_track = self.tool_rollback_registry.get_track()
            checkpoint.metadata["tool_track_position"] = len(current_track)
            
            saved_checkpoint = self.checkpoint_repo.create(checkpoint)
            self.internal_session.checkpoint_count += 1
            self._save_internal_session()
            
            if saved_checkpoint:
                return f"✓ Checkpoint '{name}' created successfully (ID: {saved_checkpoint.id})"
        
        return "Failed to create checkpoint. Repository or session not available."
    
    def list_checkpoints_tool(self) -> str:
        """List all available checkpoints for the current session.
        
        Returns:
            Formatted list of checkpoints
        """
        if not self.checkpoint_repo or not self.internal_session or not self.internal_session.id:
            return "No active session or checkpoint functionality unavailable."
        
        checkpoints = self.checkpoint_repo.get_by_internal_session(
            self.internal_session.id,
            auto_only=False
        )
        
        if not checkpoints:
            return "No checkpoints found for the current session."
        
        result = "Available checkpoints:\n"
        for cp in checkpoints:
            checkpoint_type = "auto" if cp.is_auto else "manual"
            created = cp.created_at.strftime('%Y-%m-%d %H:%M:%S') if cp.created_at else "unknown"
            name = cp.checkpoint_name or "Unnamed"
            result += f"\n• ID: {cp.id} | {name} | Type: {checkpoint_type} | Created: {created}"
        
        return result
    
    def rollback_to_checkpoint_tool(self, checkpoint_id_or_name) -> str:
        """Request rollback to a specific checkpoint.
        
        Args:
            checkpoint_id_or_name: ID or name of the checkpoint
            
        Returns:
            Status message
        """
        if not self.checkpoint_repo:
            return "Checkpoint functionality is not available."
        
        checkpoint = None
        
        # Try to parse as ID or find by name
        try:
            checkpoint_id = int(checkpoint_id_or_name)
            checkpoint = self.checkpoint_repo.get_by_id(checkpoint_id)
        except (ValueError, TypeError):
            if self.internal_session and self.internal_session.id:
                all_checkpoints = self.checkpoint_repo.get_by_internal_session(
                    self.internal_session.id
                )
                checkpoint_name_lower = str(checkpoint_id_or_name).lower()
                for cp in all_checkpoints:
                    if cp.checkpoint_name and cp.checkpoint_name.lower() == checkpoint_name_lower:
                        checkpoint = cp
                        break
        
        if not checkpoint:
            return f"Checkpoint '{checkpoint_id_or_name}' not found."
        
        # Mark rollback in state (would trigger actual rollback in production)
        if self.internal_session:
            self.internal_session.session_state['rollback_requested'] = True
            self.internal_session.session_state['rollback_checkpoint_id'] = checkpoint.id
        
        return f"Rollback to checkpoint {checkpoint.id} ('{checkpoint.checkpoint_name}') requested."
    
    def delete_checkpoint_tool(self, checkpoint_id: int) -> str:
        """Delete a specific checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to delete
            
        Returns:
            Confirmation message
        """
        if not self.checkpoint_repo:
            return "Checkpoint functionality is not available."
        
        checkpoint = self.checkpoint_repo.get_by_id(checkpoint_id)
        if not checkpoint:
            return f"Checkpoint {checkpoint_id} not found."
        
        if checkpoint.internal_session_id != self.internal_session.id:
            return "You can only delete checkpoints from the current session."
        
        success = self.checkpoint_repo.delete(checkpoint_id)
        
        if success:
            return f"✓ Checkpoint {checkpoint_id} deleted successfully."
        return f"Failed to delete checkpoint {checkpoint_id}."
    
    def get_checkpoint_info_tool(self, checkpoint_id: int) -> str:
        """Get detailed information about a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint
            
        Returns:
            Detailed checkpoint information
        """
        if not self.checkpoint_repo:
            return "Checkpoint functionality is not available."
        
        checkpoint = self.checkpoint_repo.get_by_id(checkpoint_id)
        if not checkpoint:
            return f"Checkpoint {checkpoint_id} not found."
        
        checkpoint_type = "Automatic" if checkpoint.is_auto else "Manual"
        created = checkpoint.created_at.strftime('%Y-%m-%d %H:%M:%S') if checkpoint.created_at else "unknown"
        
        info = f"Checkpoint Details:\n"
        info += f"• ID: {checkpoint.id}\n"
        info += f"• Name: {checkpoint.checkpoint_name or 'Unnamed'}\n"
        info += f"• Type: {checkpoint_type}\n"
        info += f"• Created: {created}\n"
        info += f"• Conversation Length: {len(checkpoint.conversation_history)} messages"
        
        return info
    
    def cleanup_auto_checkpoints_tool(self, keep_latest: int = 5) -> str:
        """Clean up old automatic checkpoints.
        
        Args:
            keep_latest: Number of latest checkpoints to keep
            
        Returns:
            Cleanup summary
        """
        if not self.checkpoint_repo or not self.internal_session or not self.internal_session.id:
            return "No active session or checkpoint functionality unavailable."
        
        deleted_count = self.checkpoint_repo.delete_auto_checkpoints(
            self.internal_session.id,
            keep_latest=keep_latest
        )
        
        if deleted_count > 0:
            return f"✓ Cleaned up {deleted_count} old automatic checkpoints."
        return "No automatic checkpoints to clean up."
    
    # Tool rollback methods
    def rollback_tools(self) -> List[Any]:
        """Run reverse handlers for recorded tools in reverse order.
        
        Returns:
            List of reverse invocation results
        """
        return self.tool_rollback_registry.rollback()
    
    def rollback_tools_from_track_index(self, start_index: int) -> List[Any]:
        """Run reverse handlers from specific track index.
        
        Args:
            start_index: Index to start rollback from
            
        Returns:
            List of reverse invocation results
        """
        from agentgit.core.rollback_protocol import CHECKPOINT_TOOL_NAMES, ReverseInvocationResult
        
        track = self.tool_rollback_registry.get_track()
        results = []
        
        for i in range(len(track) - 1, start_index - 1, -1):
            record = track[i]
            tool_name = record.tool_name
            
            if tool_name in CHECKPOINT_TOOL_NAMES:
                continue
            
            spec = self.tool_rollback_registry.get_tool(tool_name)
            if not spec or spec.reverse is None:
                results.append(
                    ReverseInvocationResult(
                        tool_name=tool_name,
                        reversed_successfully=False,
                        error_message="No reverse handler registered"
                    )
                )
                continue
            
            try:
                spec.reverse(record.args, record.result)
                results.append(
                    ReverseInvocationResult(
                        tool_name=tool_name,
                        reversed_successfully=True
                    )
                )
            except Exception as e:
                results.append(
                    ReverseInvocationResult(
                        tool_name=tool_name,
                        reversed_successfully=False,
                        error_message=str(e)
                    )
                )
        
        return results
    
    def redo_tools(self) -> List[Any]:
        """Re-execute forward handlers for recorded tools.
        
        Returns:
            List of new invocation records
        """
        return self.tool_rollback_registry.redo()
    
    def get_tool_track(self) -> List[Any]:
        """Get current recorded tool invocation track.
        
        Returns:
            List of tool invocation records
        """
        return self.tool_rollback_registry.get_track()
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history for this session.
        
        Returns:
            List of conversation messages
        """
        if self.internal_session:
            return self.internal_session.conversation_history
        return []
    
    def get_session_state(self) -> Dict[str, Any]:
        """Get current session state.
        
        Returns:
            Session state dictionary
        """
        if self.internal_session:
            return self.internal_session.session_state
        return {}
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_id: int,
        external_session_id: int,
        model,
        checkpoint_repo: CheckpointRepository,
        internal_session_repo: InternalSessionRepository,
        **kwargs
    ) -> "RollbackAgent":
        """Create a new agent from a checkpoint (rollback).
        
        Creates a new internal session branched from the checkpoint,
        preserving the forward timeline.
        
        Args:
            checkpoint_id: ID of checkpoint to restore from
            external_session_id: ID of external session
            model: LangChain model to use
            checkpoint_repo: Checkpoint repository
            internal_session_repo: Internal session repository
            **kwargs: Additional agent configuration
            
        Returns:
            New RollbackAgent with checkpoint's state
        """
        # Load the checkpoint
        checkpoint = checkpoint_repo.get_by_id(checkpoint_id)
        if not checkpoint:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        # Create new agent without session
        agent = cls(
            external_session_id=external_session_id,
            model=model,
            internal_session_repo=internal_session_repo,
            checkpoint_repo=checkpoint_repo,
            skip_session_creation=True,
            **kwargs
        )
        
        # Create new internal session as branch from the checkpoint
        branch_session = InternalSession.create_branch_from_checkpoint(
            checkpoint=checkpoint,
            external_session_id=external_session_id,
            parent_session_id=checkpoint.internal_session_id
        )
        
        # Save the branch session to database
        agent.internal_session = internal_session_repo.create(branch_session)
        agent.langgraph_session_id = agent.internal_session.langgraph_session_id
        agent._register_with_external_session()
        
        # Copy checkpoints up to rollback point for snapshot capability
        if checkpoint_repo and agent.internal_session.id:
            original_checkpoints = checkpoint_repo.get_by_internal_session(
                checkpoint.internal_session_id,
                auto_only=False
            )
            
            for cp in original_checkpoints:
                if cp.created_at and checkpoint.created_at and cp.created_at <= checkpoint.created_at:
                    new_checkpoint = Checkpoint(
                        internal_session_id=agent.internal_session.id,
                        checkpoint_name=cp.checkpoint_name,
                        session_state=cp.session_state.copy(),
                        conversation_history=cp.conversation_history.copy(),
                        is_auto=cp.is_auto,
                        created_at=cp.created_at,
                        metadata=cp.metadata.copy()
                    )
                    checkpoint_repo.create(new_checkpoint)
        
        # Restore tool track position if available
        if "tool_track_position" in checkpoint.metadata:
            track_position = checkpoint.metadata["tool_track_position"]
            # Truncate tool track to checkpoint position
            agent.tool_rollback_registry.truncate_track(track_position)
        
        agent._save_internal_session()
        
        return agent