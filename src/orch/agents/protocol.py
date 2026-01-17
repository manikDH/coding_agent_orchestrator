"""Agent adapter protocol - interface for all AI CLI backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator


class OutputFormat(Enum):
    """Output format for agent responses."""

    TEXT = "text"
    JSON = "json"
    STREAM = "stream"


@dataclass
class AgentCapabilities:
    """Declares what an agent backend can do."""

    supports_streaming: bool = False
    supports_sessions: bool = False
    supports_images: bool = False
    supports_files: bool = False
    supports_approval_modes: bool = False
    supports_sandbox: bool = False
    supports_interactive: bool = False
    task_strengths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            "streaming": self.supports_streaming,
            "sessions": self.supports_sessions,
            "images": self.supports_images,
            "files": self.supports_files,
            "approval_modes": self.supports_approval_modes,
            "sandbox": self.supports_sandbox,
            "interactive": self.supports_interactive,
            "strengths": self.task_strengths,
        }


@dataclass
class ExecutionResult:
    """Normalized result from any agent backend."""

    agent_name: str
    success: bool
    content: str
    raw_output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    session_id: str | None = None
    exit_code: int = 0

    @property
    def is_error(self) -> bool:
        """Check if this result represents an error."""
        return not self.success or self.error is not None


class AgentAdapter(ABC):
    """Abstract base class for all AI CLI agent adapters.

    Implement this protocol to add support for a new AI CLI tool.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this agent (e.g., 'gemini', 'codex')."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name (e.g., 'Google Gemini')."""
        ...

    @property
    @abstractmethod
    def executable(self) -> str:
        """Path or name of the CLI executable."""
        ...

    @abstractmethod
    def get_capabilities(self) -> AgentCapabilities:
        """Return capabilities declaration for this agent."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the agent CLI is installed and configured."""
        ...

    @abstractmethod
    def build_command(
        self,
        prompt: str,
        output_format: OutputFormat = OutputFormat.TEXT,
        model: str | None = None,
        session_id: str | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> list[str]:
        """Build the CLI command to execute.

        Args:
            prompt: The prompt to send to the agent.
            output_format: Desired output format.
            model: Optional model override.
            session_id: Optional session ID to resume.
            extra_args: Additional CLI arguments.

        Returns:
            List of command arguments to execute.
        """
        ...

    @abstractmethod
    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> ExecutionResult:
        """Parse CLI output into normalized result.

        Args:
            stdout: Standard output from the CLI.
            stderr: Standard error from the CLI.
            return_code: Exit code from the CLI.

        Returns:
            Normalized ExecutionResult.
        """
        ...

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        output_format: OutputFormat = OutputFormat.TEXT,
        stream: bool = False,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a prompt and return results.

        Args:
            prompt: The prompt to execute.
            output_format: Desired output format.
            stream: Whether to stream output (returns generator if True).
            **kwargs: Additional arguments passed to build_command.

        Returns:
            ExecutionResult with the agent's response.
        """
        ...

    async def stream_execute(
        self,
        prompt: str,
        output_format: OutputFormat = OutputFormat.TEXT,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Execute with streaming output.

        Default implementation raises NotImplementedError.
        Override in subclasses that support streaming.

        Args:
            prompt: The prompt to execute.
            output_format: Desired output format.
            **kwargs: Additional arguments.

        Yields:
            Chunks of output as they arrive.
        """
        raise NotImplementedError(f"{self.name} does not support streaming")
        yield  # Make this a generator

    async def send_feedback(
        self, session_id: str, feedback: str
    ) -> ExecutionResult:
        """Send feedback to an ongoing session.

        Default implementation raises NotImplementedError.
        Override in subclasses that support iterative refinement.

        Args:
            session_id: The session to send feedback to.
            feedback: The feedback message.

        Returns:
            ExecutionResult with the agent's response.
        """
        raise NotImplementedError(f"{self.name} does not support feedback")

    async def interrupt(self, session_id: str, new_direction: str) -> None:
        """Interrupt an ongoing session with new direction.

        Default implementation raises NotImplementedError.
        Override in subclasses that support interruption.

        Args:
            session_id: The session to interrupt.
            new_direction: New instructions for the agent.
        """
        raise NotImplementedError(f"{self.name} does not support interruption")

    def get_session_list(self) -> list[dict[str, Any]]:
        """List available sessions.

        Default implementation returns empty list.
        Override in subclasses that support sessions.

        Returns:
            List of session info dictionaries.
        """
        return []

    def get_config_schema(self) -> dict[str, Any]:
        """Return JSON schema for agent-specific config.

        Override to provide configuration options for this agent.

        Returns:
            JSON schema dictionary.
        """
        return {}
