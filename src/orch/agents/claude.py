"""Claude Code CLI agent adapter."""

import json
from typing import Any

from orch.agents.base import BaseAgent
from orch.agents.protocol import AgentCapabilities, ExecutionResult, OutputFormat


class ClaudeAgent(BaseAgent):
    """Adapter for Anthropic Claude Code CLI.

    Claude CLI supports:
    - --print for non-interactive prompts
    - --output-format for text/json/stream-json
    - --model for model selection
    - --allowedTools and --dangerously-skip-permissions for tool permissions
    """

    @property
    def name(self) -> str:
        return "claude"

    @property
    def display_name(self) -> str:
        return "Claude Code"

    @property
    def cli_name(self) -> str:
        return "claude"

    def get_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            supports_streaming=True,
            supports_sessions=False,
            supports_images=False,
            supports_files=True,
            supports_approval_modes=False,
            supports_sandbox=False,
            supports_interactive=True,
            task_strengths=["code", "refactoring", "debugging", "architecture"],
        )

    def build_command(
        self,
        prompt: str,
        output_format: OutputFormat = OutputFormat.TEXT,
        model: str | None = None,
        session_id: str | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> list[str]:
        cmd = [self.executable, "--print"]

        format_map = {
            OutputFormat.TEXT: "text",
            OutputFormat.JSON: "json",
            OutputFormat.STREAM: "stream-json",
        }
        cmd.extend(["--output-format", format_map[output_format]])

        # Claude CLI requires --verbose when using stream-json with --print
        if output_format == OutputFormat.STREAM:
            cmd.append("--verbose")

        if model:
            cmd.extend(["--model", model])

        if session_id:
            # Claude Code CLI doesn't expose a stable session flag in non-interactive mode.
            pass

        if extra_args:
            allowed_tools = extra_args.get("allowedTools")
            if allowed_tools is None:
                allowed_tools = extra_args.get("allowed_tools")
            if allowed_tools:
                if isinstance(allowed_tools, (list, tuple, set)):
                    allowed_tools_value = ",".join(str(tool) for tool in allowed_tools)
                else:
                    allowed_tools_value = str(allowed_tools)
                cmd.extend(["--allowedTools", allowed_tools_value])

            skip_permissions = (
                extra_args.get("dangerously_skip_permissions")
                or extra_args.get("dangerously-skip-permissions")
            )
            if skip_permissions:
                cmd.append("--dangerously-skip-permissions")

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

        content = stdout
        metadata: dict[str, Any] = {}

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict):
            content = (
                data.get("response")
                or data.get("content")
                or data.get("text")
                or data.get("completion")
                or stdout
            )
            metadata = {
                key: value
                for key, value in data.items()
                if key not in ("response", "content", "text", "completion")
            }
        elif isinstance(data, list):
            parts: list[str] = []
            for item in data:
                if isinstance(item, dict):
                    part = item.get("content") or item.get("text") or item.get("response")
                    if part:
                        parts.append(str(part))
                else:
                    parts.append(str(item))
            if parts:
                content = "\n".join(parts)
                metadata["items"] = len(data)

        return ExecutionResult(
            agent_name=self.name,
            success=True,
            content=content,
            raw_output=stdout,
            metadata=metadata,
            exit_code=0,
        )

    def get_config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Claude model to use",
                    "default": "sonnet",
                },
                "model_tiers": {
                    "type": "object",
                    "description": "Model tiers for complexity selection",
                    "properties": {
                        "low": {"type": "string", "default": "haiku"},
                        "medium": {"type": "string", "default": "sonnet"},
                        "high": {"type": "string", "default": "opus"},
                    },
                },
            },
        }
