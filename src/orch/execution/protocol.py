"""Base interface for code execution (hands vs brains separation)"""
from abc import ABC, abstractmethod
from orch.orchestration.models import ExecutionRequest, ExecutionResult


class RemoteCodeExecutor(ABC):
    """Base interface for code execution (hands vs brains separation)"""

    @abstractmethod
    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute code/task and return summarized result"""
        pass

    @abstractmethod
    def can_handle(self, request: ExecutionRequest) -> bool:
        """Check if this executor can handle the request"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify executor is available and working"""
        pass
