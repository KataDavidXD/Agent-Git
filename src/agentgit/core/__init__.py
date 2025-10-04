"""Core components and protocols for the rollback system."""

from .rollback_protocol import (
    ToolRollbackRegistry,
    ToolSpec,
    ToolInvocationRecord,
    ReverseInvocationResult,
    CHECKPOINT_TOOL_NAMES
)

__all__ = [
    'ToolRollbackRegistry',
    'ToolSpec',
    'ToolInvocationRecord',
    'ReverseInvocationResult',
    'CHECKPOINT_TOOL_NAMES'
]
