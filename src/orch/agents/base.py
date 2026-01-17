"""Base implementation for agent adapters with common functionality."""

import asyncio
import shutil
from abc import abstractmethod
from typing import Any

from orch.agents.protocol import (
    AgentAdapter,
    AgentCapabilities,
    ExecutionResult,
    OutputFormat,
)


class BaseAgent(AgentAdapter):
    """Base class with common implementation for agent adapters."""

    _executable_cache: str | None = None

    @property
    def executable(self) -> str:
        """Get the executable path, caching the result."""
        if self._executable_cache is None:
            self._executable_cache = shutil.which(self.cli_name) or self.cli_name
        return self._executable_cache

    @property
    @abstractmethod
    def cli_name(self) -> str:
        """The CLI command name to look up (e.g., 'gemini', 'codex')."""
        ...

    def is_available(self) -> bool:
        """Check if the CLI is available on the system."""
        return shutil.which(self.cli_name) is not None

    async def execute(
        self,
        prompt: str,
        output_format: OutputFormat = OutputFormat.TEXT,
        stream: bool = False,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a prompt using the agent CLI.

        Uses asyncio subprocess for non-blocking execution.
        """
        cmd = self.build_command(
            prompt=prompt,
            output_format=output_format,
            model=kwargs.get("model"),
            session_id=kwargs.get("session_id"),
            extra_args=kwargs.get("extra_args"),
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await proc.communicate()

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            return_code = proc.returncode or 0

            return self.parse_output(stdout, stderr, return_code)

        except FileNotFoundError:
            return ExecutionResult(
                agent_name=self.name,
                success=False,
                content="",
                raw_output=None,
                error=f"CLI '{self.cli_name}' not found. Is it installed?",
                exit_code=127,
            )
        except Exception as e:
            return ExecutionResult(
                agent_name=self.name,
                success=False,
                content="",
                raw_output=None,
                error=f"Execution error: {e}",
                exit_code=1,
            )

    async def stream_execute(
        self,
        prompt: str,
        output_format: OutputFormat = OutputFormat.TEXT,
        **kwargs: Any,
    ):
        """Execute with streaming output.

        Properly handles stderr draining and process cleanup to avoid deadlocks.
        Yields stdout chunks, then yields a final ExecutionResult with any errors.
        """
        cmd = self.build_command(
            prompt=prompt,
            output_format=OutputFormat.STREAM if self.get_capabilities().supports_streaming else output_format,
            model=kwargs.get("model"),
            session_id=kwargs.get("session_id"),
            extra_args=kwargs.get("extra_args"),
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stderr_chunks: list[bytes] = []

        async def drain_stderr() -> None:
            """Drain stderr to prevent buffer deadlock."""
            if proc.stderr:
                async for chunk in proc.stderr:
                    stderr_chunks.append(chunk)

        # Start draining stderr concurrently
        stderr_task = asyncio.create_task(drain_stderr())

        try:
            if proc.stdout:
                async for line in proc.stdout:
                    yield line.decode("utf-8", errors="replace")
        finally:
            # Wait for stderr drain to complete
            await stderr_task

            # Wait for process to exit
            await proc.wait()

            # If there was an error, yield error info
            if proc.returncode and proc.returncode != 0:
                stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")
                yield f"\n[Error: Process exited with code {proc.returncode}]\n"
                if stderr_text:
                    yield f"[stderr: {stderr_text}]\n"

    def get_version(self) -> str | None:
        """Get the CLI version if available."""
        import subprocess

        try:
            result = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} available={self.is_available()}>"
