"""OpenCode CLI agent adapter."""

import json
from typing import Any

from orch.agents.base import BaseAgent
from orch.agents.protocol import AgentCapabilities, ExecutionResult, OutputFormat

# Free models available in OpenCode
FREE_MODELS = [
    "opencode/grok-code",  # Grok Fast Code - fast, good for coding tasks
    "opencode/glm-4.7-free",  # GLM 4.7 free tier
    "opencode/minimax-m2.1-free",  # Minimax M2.1 free tier
]

# Default free model - Grok Fast Code is fast and capable
DEFAULT_FREE_MODEL = "opencode/grok-code"


class OpenCodeAgent(BaseAgent):
    """Adapter for OpenCode CLI.

    OpenCode CLI supports:
    - `opencode run [message..]` for non-interactive execution
    - --model provider/model for model selection (restricted to free models)
    - --format for output format (default, json)
    - --continue / --session for session management
    - --agent for agent selection (build, plan, explore, general)
    - --file for file attachments
    - --variant for reasoning effort level

    Free Models Available:
    - opencode/glm-4.7-free: GLM 4.7 free tier - good for general coding tasks
    - opencode/minimax-m2.1-free: Minimax M2.1 free tier - good for analysis

    Agents Available:
    - build: Full permissions for implementation tasks
    - plan: Planning mode with restricted editing
    - explore: Read-only codebase exploration
    - general: General purpose with full tool access
    """

    @property
    def name(self) -> str:
        return "opencode"

    @property
    def display_name(self) -> str:
        return "OpenCode"

    @property
    def cli_name(self) -> str:
        return "opencode"

    def get_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            supports_streaming=False,  # run command doesn't stream
            supports_sessions=True,
            supports_images=False,
            supports_files=True,
            supports_approval_modes=False,
            supports_sandbox=False,
            supports_interactive=True,
            task_strengths=["code", "implementation", "exploration", "planning"],
        )

    def _validate_free_model(self, model: str | None) -> str:
        """Validate and return a free model.

        Args:
            model: Requested model name.

        Returns:
            A valid free model name.

        Raises:
            ValueError: If the requested model is not a free model.
        """
        if model is None:
            return DEFAULT_FREE_MODEL

        # Allow short names without provider prefix
        if "/" not in model:
            model = f"opencode/{model}"

        if model not in FREE_MODELS:
            raise ValueError(
                f"Model '{model}' is not a free model. "
                f"Available free models: {', '.join(FREE_MODELS)}"
            )

        return model

    def build_command(
        self,
        prompt: str,
        output_format: OutputFormat = OutputFormat.TEXT,
        model: str | None = None,
        session_id: str | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> list[str]:
        cmd = [self.executable, "run"]

        # Validate and set model (restricted to free models)
        validated_model = self._validate_free_model(model)
        cmd.extend(["--model", validated_model])

        # Output format mapping
        format_map = {
            OutputFormat.TEXT: "default",
            OutputFormat.JSON: "json",
            OutputFormat.STREAM: "default",  # OpenCode run doesn't support streaming
        }
        cmd.extend(["--format", format_map[output_format]])

        if session_id:
            cmd.extend(["--session", session_id])

        if extra_args:
            # Agent selection (build, plan, explore, general)
            agent = extra_args.get("agent")
            if agent:
                cmd.extend(["--agent", agent])

            # File attachments
            files = extra_args.get("files") or extra_args.get("file")
            if files:
                if isinstance(files, str):
                    files = [files]
                for f in files:
                    cmd.extend(["--file", f])

            # Reasoning variant
            variant = extra_args.get("variant")
            if variant:
                cmd.extend(["--variant", variant])

            # Continue last session
            if extra_args.get("continue"):
                cmd.append("--continue")

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

        content = stdout
        metadata: dict[str, Any] = {}

        # Try to parse as JSON for structured output
        try:
            data = json.loads(stdout)
            if isinstance(data, dict):
                # Extract content from various possible keys
                content = (
                    data.get("response")
                    or data.get("content")
                    or data.get("text")
                    or data.get("message")
                    or stdout
                )
                metadata = {
                    k: v for k, v in data.items()
                    if k not in ("response", "content", "text", "message")
                }
            elif isinstance(data, list):
                # Handle array of events/messages
                parts: list[str] = []
                for item in data:
                    if isinstance(item, dict):
                        part = (
                            item.get("content")
                            or item.get("text")
                            or item.get("message")
                        )
                        if part:
                            parts.append(str(part))
                    else:
                        parts.append(str(item))
                if parts:
                    content = "\n".join(parts)
                    metadata["events"] = len(data)
        except json.JSONDecodeError:
            # Plain text output - use as is
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
        """List available OpenCode sessions."""
        import subprocess

        try:
            result = subprocess.run(
                [self.executable, "session", "list"],
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
                    "description": "OpenCode model to use (free models only)",
                    "default": DEFAULT_FREE_MODEL,
                    "enum": FREE_MODELS,
                },
                "agent": {
                    "type": "string",
                    "enum": ["build", "plan", "explore", "general"],
                    "description": "OpenCode agent to use",
                    "default": "build",
                },
                "variant": {
                    "type": "string",
                    "description": "Reasoning effort variant (e.g., high, max, minimal)",
                },
            },
        }

    @classmethod
    def get_free_models(cls) -> list[str]:
        """Return list of available free models."""
        return FREE_MODELS.copy()
