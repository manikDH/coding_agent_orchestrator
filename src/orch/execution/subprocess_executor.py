"""Execute simple code/commands in isolated subprocess"""
import asyncio
from pathlib import Path
from orch.execution.protocol import RemoteCodeExecutor
from orch.orchestration.models import ExecutionRequest, ExecutionResult


class SubprocessExecutor(RemoteCodeExecutor):
    """Execute simple code/commands in isolated subprocess"""

    HANDLED_TASKS = {
        "run_tests": ["pytest"],
        "lint": ["ruff", "pylint"],
        "type_check": ["mypy"],
        "format_check": ["ruff format --check"],
    }

    def can_handle(self, request: ExecutionRequest) -> bool:
        """Handle if it's a simple command-based task"""
        return request.task_type in self.HANDLED_TASKS

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute subprocess command"""
        cmd = self._build_command(request)

        try:
            # Run with timeout
            timeout = request.constraints.get("timeout", 60)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=request.workspace_context.workspace_root
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            # Summarize result
            summary = self._summarize_result(
                request.task_type,
                stdout.decode(),
                stderr.decode(),
                process.returncode
            )

            return ExecutionResult(
                executor_type="subprocess",
                success=process.returncode == 0,
                summary=summary,
                metrics={"exit_code": process.returncode}
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                executor_type="subprocess",
                success=False,
                summary={"error": "timeout"},
                error="Command timed out"
            )
        except Exception as e:
            return ExecutionResult(
                executor_type="subprocess",
                success=False,
                summary={"error": str(e)},
                error=str(e)
            )

    async def health_check(self) -> bool:
        """Verify executor is working"""
        return True  # Subprocess always available

    def _build_command(self, request: ExecutionRequest) -> list[str]:
        """Build command from request"""
        task_type = request.task_type
        if task_type not in self.HANDLED_TASKS:
            raise ValueError(f"Cannot handle task type: {task_type}")

        # Use first available command for task type
        base_cmd = self.HANDLED_TASKS[task_type][0]
        return base_cmd.split()

    def _summarize_result(
        self,
        task_type: str,
        stdout: str,
        stderr: str,
        exit_code: int
    ) -> dict:
        """Summarize output (don't return full text)"""
        summary = {
            "task_type": task_type,
            "exit_code": exit_code,
            "success": exit_code == 0
        }

        # Add task-specific summary
        if task_type == "run_tests" and "pytest" in stdout:
            summary.update(self._parse_pytest_output(stdout))
        elif task_type == "lint" and stderr:
            summary["issues_found"] = len(stderr.split("\n"))

        # Include error if failed
        if exit_code != 0:
            summary["error_preview"] = stderr[:200] if stderr else stdout[:200]

        return summary

    def _parse_pytest_output(self, output: str) -> dict:
        """Extract pytest summary"""
        # Simple parsing - can be enhanced
        if "passed" in output:
            return {"status": "passed"}
        elif "failed" in output:
            return {"status": "failed"}
        return {"status": "unknown"}
