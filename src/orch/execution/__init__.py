"""Execution layer for team-of-rivals orchestration"""
from .protocol import RemoteCodeExecutor
from .subprocess_executor import SubprocessExecutor
from .agent_delegator import AgentDelegator
from .router import ExecutionRouter

__all__ = [
    "RemoteCodeExecutor",
    "SubprocessExecutor",
    "AgentDelegator",
    "ExecutionRouter"
]
