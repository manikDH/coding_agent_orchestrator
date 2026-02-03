"""Base protocol for team-of-rivals agents"""
from abc import ABC, abstractmethod
from orch.orchestration.models import TaskState, AgentMessage, ReviewFeedback


class RoleAgent(ABC):
    """Base protocol for team-of-rivals agents"""

    @property
    @abstractmethod
    def role_name(self) -> str:
        """Agent role: planner, executor, critic, expert"""
        pass

    @property
    @abstractmethod
    def goal(self) -> str:
        """What this agent optimizes for"""
        pass

    @abstractmethod
    async def propose(self, task_state: TaskState) -> AgentMessage:
        """Generate proposal/plan/implementation based on task state"""
        pass

    @abstractmethod
    async def review(self, task_state: TaskState, artifact: dict) -> ReviewFeedback:
        """Review output and provide feedback"""
        pass
