"""Codex CLI agent adapter."""

import json
import subprocess
from typing import Any

from orch.agents.base import BaseAgent
from orch.agents.protocol import AgentCapabilities, ExecutionResult, OutputFormat


class CodexAgent(BaseAgent):
    """Adapter for OpenAI Codex CLI.

    Codex CLI supports:
    - 'exec' subcommand for non-interactive execution
    - --json for JSONL output
    - resume/fork for session management
    - --model for model selection
    - --ask-for-approval for approval control
    - --sandbox for sandbox modes
    """

    @property
    def name(self) -> str:
        return "codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex"

    @property
    def cli_name(self) -> str:
        return "codex"

    def get_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            supports_streaming=True,
            supports_sessions=True,
            supports_images=True,
            supports_files=True,
            supports_approval_modes=True,
            supports_sandbox=True,
            supports_interactive=True,
            task_strengths=["code", "debugging", "implementation", "refactoring"],
        )

    def build_command(
        self,
        prompt: str,
        output_format: OutputFormat = OutputFormat.TEXT,
        model: str | None = None,
        session_id: str | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> list[str]:
        # Use 'exec' subcommand for non-interactive execution
        cmd = [self.executable, "exec"]

        # Skip git repo check for simple queries
        cmd.append("--skip-git-repo-check")

        if output_format == OutputFormat.JSON:
            cmd.append("--json")

        if model:
            cmd.extend(["--model", model])

        if extra_args:
            # Note: codex exec doesn't support approval modes or sandbox
            # Those are only for interactive mode

            working_dir = extra_args.get("cwd")
            if working_dir:
                cmd.extend(["-C", working_dir])

            # Config overrides via -c flag
            config_overrides = extra_args.get("config")
            if config_overrides:
                for key, value in config_overrides.items():
                    cmd.extend(["-c", f"{key}={value}"])

        # Prompt as positional argument
        cmd.append(prompt)

        return cmd

    def parse_output(
        self, stdout: str, stderr: str, return_code: int
    ) -> ExecutionResult:
        if return_code != 0:
            return ExecutionResult(
                agent_name=self.name,
                success=False,
                content="",
                raw_output={"stdout": stdout, "stderr": stderr},
                metadata={},
                error=stderr or f"Command failed with code {return_code}",
                exit_code=return_code,
            )

        # Codex with --json outputs JSONL events
        content_parts = []
        metadata: dict[str, Any] = {}

        for line in stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    event_type = event.get("type", "")
                    if event_type == "message" or "content" in event:
                        content_parts.append(event.get("content", ""))
                    # Collect metadata from events
                    for key in ("model", "tokens", "latency", "session_id"):
                        if key in event:
                            metadata[key] = event[key]
            except json.JSONDecodeError:
                # Plain text line
                content_parts.append(line)

        return ExecutionResult(
            agent_name=self.name,
            success=True,
            content="\n".join(content_parts) if content_parts else stdout,
            raw_output=stdout,
            metadata=metadata,
            exit_code=0,
        )

    def get_session_list(self) -> list[dict[str, Any]]:
        """List available Codex sessions."""
        # Codex uses 'resume' command with picker, harder to list programmatically
        # For now, return empty - could be enhanced later
        return []

    def build_review_command(
        self,
        base_branch: str | None = None,
        uncommitted: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """Build command for code review."""
        cmd = [self.executable, "review"]

        if uncommitted:
            cmd.append("--uncommitted")
        elif base_branch:
            cmd.extend(["--base", base_branch])

        return cmd

    async def review(
        self,
        base_branch: str | None = None,
        uncommitted: bool = True,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Run code review using Codex."""
        import asyncio

        cmd = self.build_review_command(base_branch, uncommitted, **kwargs)

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

        except Exception as e:
            return ExecutionResult(
                agent_name=self.name,
                success=False,
                content="",
                raw_output=None,
                error=f"Review error: {e}",
                exit_code=1,
            )

    def get_config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Codex model to use",
                    "default": "gpt-5.2-codex",
                },
                "approval_mode": {
                    "type": "string",
                    "enum": ["untrusted", "on-failure", "on-request", "never"],
                    "description": "Approval policy for commands",
                    "default": "on-request",
                },
                "sandbox": {
                    "type": "string",
                    "enum": ["read-only", "workspace-write", "danger-full-access"],
                    "description": "Sandbox mode",
                    "default": "workspace-write",
                },
            },
        }
