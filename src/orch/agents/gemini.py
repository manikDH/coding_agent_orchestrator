"""Gemini CLI agent adapter."""

import json
import subprocess
from typing import Any

from orch.agents.base import BaseAgent
from orch.agents.protocol import AgentCapabilities, ExecutionResult, OutputFormat


class GeminiAgent(BaseAgent):
    """Adapter for Google Gemini CLI.

    Gemini CLI supports:
    - Positional prompts for one-shot queries
    - --output-format for text/json/stream-json
    - --resume for session management
    - --model for model selection
    - --approval-mode for tool approval control
    """

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def display_name(self) -> str:
        return "Google Gemini"

    @property
    def cli_name(self) -> str:
        return "gemini"

    def get_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            supports_streaming=True,
            supports_sessions=True,
            supports_images=False,
            supports_files=True,
            supports_approval_modes=True,
            supports_sandbox=True,
            supports_interactive=True,
            task_strengths=["explanation", "analysis", "general", "research"],
        )

    def build_command(
        self,
        prompt: str,
        output_format: OutputFormat = OutputFormat.TEXT,
        model: str | None = None,
        session_id: str | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> list[str]:
        cmd = [self.executable]

        # Output format mapping
        format_map = {
            OutputFormat.TEXT: "text",
            OutputFormat.JSON: "json",
            OutputFormat.STREAM: "stream-json",
        }
        cmd.extend(["--output-format", format_map[output_format]])

        if model:
            cmd.extend(["--model", model])

        if session_id:
            cmd.extend(["--resume", session_id])

        if extra_args:
            approval_mode = extra_args.get("approval_mode")
            if approval_mode:
                cmd.extend(["--approval-mode", approval_mode])

            sandbox = extra_args.get("sandbox")
            if sandbox:
                cmd.append("--sandbox")

            yolo = extra_args.get("yolo")
            if yolo:
                cmd.append("--yolo")

            include_dirs = extra_args.get("include_directories")
            if include_dirs:
                for dir_path in include_dirs:
                    cmd.extend(["--include-directories", dir_path])

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

        # Try to parse as JSON for metadata extraction
        content = stdout
        metadata: dict[str, Any] = {}

        try:
            data = json.loads(stdout)
            if isinstance(data, dict):
                content = data.get("response", data.get("text", data.get("content", stdout)))
                metadata = {k: v for k, v in data.items() if k not in ("response", "text", "content")}
        except json.JSONDecodeError:
            # Plain text output
            pass

        return ExecutionResult(
            agent_name=self.name,
            success=True,
            content=content,
            raw_output=stdout,
            metadata=metadata,
            exit_code=0,
        )

    def get_session_list(self) -> list[dict[str, Any]]:
        """List available Gemini sessions."""
        try:
            result = subprocess.run(
                [self.executable, "--list-sessions"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            sessions = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    sessions.append({"id": line.strip(), "agent": self.name})
            return sessions
        except Exception:
            return []

    def get_config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Gemini model to use",
                    "default": "gemini-2.0-flash",
                },
                "approval_mode": {
                    "type": "string",
                    "enum": ["default", "auto_edit", "yolo"],
                    "description": "Tool approval mode",
                    "default": "auto_edit",
                },
            },
        }
