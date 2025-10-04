"""Rollback protocol for tool operations in the LangGraph agent system."""

from typing import Dict, Any, Optional, List, Callable, Mapping
from dataclasses import dataclass


# Checkpoint tool names that should be excluded from rollback
CHECKPOINT_TOOL_NAMES = {
    'create_checkpoint',
    'list_checkpoints', 
    'rollback_to_checkpoint',
    'delete_checkpoint',
    'get_checkpoint_info',
    'cleanup_auto_checkpoints'
}


@dataclass
class ToolSpec:
    """Specification for a reversible tool."""
    name: str
    forward: Callable[[Mapping[str, Any]], Any]
    reverse: Optional[Callable[[Mapping[str, Any], Any], Any]] = None


@dataclass
class ToolInvocationRecord:
    """Record of a tool invocation."""
    tool_name: str
    args: Dict[str, Any]
    result: Any
    success: bool
    error_message: Optional[str] = None


@dataclass
class ReverseInvocationResult:
    """Result of a reverse tool operation."""
    tool_name: str
    reversed_successfully: bool
    error_message: Optional[str] = None


class ToolRollbackRegistry:
    """Registry for managing reversible tool operations."""
    
    def __init__(self):
        """Initialize the registry."""
        self._tools: Dict[str, ToolSpec] = {}
        self._track: List[ToolInvocationRecord] = []
    
    def register_tool(self, spec: ToolSpec):
        """Register a tool with optional reverse handler.
        
        Args:
            spec: Tool specification with forward and optional reverse functions
        """
        self._tools[spec.name] = spec
    
    def get_tool(self, name: str) -> Optional[ToolSpec]:
        """Get a tool specification by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool specification or None if not found
        """
        return self._tools.get(name)
    
    def record_invocation(
        self, 
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Record a tool invocation.
        
        Args:
            tool_name: Name of the tool
            args: Arguments passed to the tool
            result: Result from the tool
            success: Whether the invocation succeeded
            error_message: Optional error message if failed
        """
        record = ToolInvocationRecord(
            tool_name=tool_name,
            args=args,
            result=result,
            success=success,
            error_message=error_message
        )
        self._track.append(record)
    
    def get_track(self) -> List[ToolInvocationRecord]:
        """Get the current tool invocation track.
        
        Returns:
            List of tool invocation records
        """
        return self._track.copy()
    
    def truncate_track(self, position: int):
        """Truncate the track to a specific position.
        
        Args:
            position: Position to truncate to
        """
        self._track = self._track[:position]
    
    def rollback(self) -> List[ReverseInvocationResult]:
        """Rollback all recorded tool invocations.
        
        Returns:
            List of reverse invocation results
        """
        results = []
        
        # Process in reverse order
        for record in reversed(self._track):
            if record.tool_name in CHECKPOINT_TOOL_NAMES:
                continue
                
            spec = self._tools.get(record.tool_name)
            if not spec or not spec.reverse:
                results.append(
                    ReverseInvocationResult(
                        tool_name=record.tool_name,
                        reversed_successfully=False,
                        error_message="No reverse handler registered"
                    )
                )
                continue
            
            try:
                spec.reverse(record.args, record.result)
                results.append(
                    ReverseInvocationResult(
                        tool_name=record.tool_name,
                        reversed_successfully=True
                    )
                )
            except Exception as e:
                results.append(
                    ReverseInvocationResult(
                        tool_name=record.tool_name,
                        reversed_successfully=False,
                        error_message=str(e)
                    )
                )
        
        # Clear track after rollback
        self._track.clear()
        return results
    
    def redo(self) -> List[ToolInvocationRecord]:
        """Re-execute forward handlers for recorded tools.
        
        Returns:
            List of new invocation records
        """
        new_records = []
        old_track = self._track.copy()
        self._track.clear()
        
        for record in old_track:
            spec = self._tools.get(record.tool_name)
            if spec and spec.forward:
                try:
                    result = spec.forward(record.args)
                    self.record_invocation(
                        record.tool_name,
                        record.args,
                        result,
                        success=True
                    )
                    new_records.append(self._track[-1])
                except Exception as e:
                    self.record_invocation(
                        record.tool_name,
                        record.args,
                        None,
                        success=False,
                        error_message=str(e)
                    )
                    new_records.append(self._track[-1])
        
        return new_records