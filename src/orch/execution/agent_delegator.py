"""Delegate complex tasks to AI CLI agents"""
from orch.execution.protocol import RemoteCodeExecutor
from orch.orchestration.models import ExecutionRequest, ExecutionResult
from orch.agents.registry import AgentRegistry


class AgentDelegator(RemoteCodeExecutor):
    """Delegate complex tasks to AI CLI agents"""

    COMPLEX_TASKS = {
        "implementation", "refactoring", "debugging",
        "analysis", "exploration", "design"
    }

    def __init__(self, agent_registry: AgentRegistry):
        self.agent_registry = agent_registry

    def can_handle(self, request: ExecutionRequest) -> bool:
        """Handle complex implementation/analysis tasks"""
        return request.task_type in self.COMPLEX_TASKS

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Delegate to AI CLI agent"""
        # Select best agent
        agent = self._select_agent(request)

        # Build rich prompt with context
        prompt = self._build_prompt(request)

        # Execute via agent
        agent_result = await agent.execute(prompt, stream=False)

        # Summarize result (context hygiene)
        summary = self._summarize_agent_output(agent_result, request.task_type)

        return ExecutionResult(
            executor_type="agent_cli",
            agent_name=agent.name,
            success=agent_result.success if hasattr(agent_result, 'success') else agent_result.exit_code == 0,
            summary=summary,
            metrics={"agent": agent.name}
        )

    async def health_check(self) -> bool:
        """Check if at least one agent is available"""
        agents = self.agent_registry.list_agents()
        return len(agents) > 0

    def _select_agent(self, request: ExecutionRequest):
        """Select best agent for task"""
        # Use suggestion if provided
        if request.suggested_agent:
            return self.agent_registry.get_agent(request.suggested_agent)

        # Default selection based on task type
        agents = self.agent_registry.list_agents()
        if not agents:
            raise ValueError("No agents available")

        # Prefer codex for implementation
        if request.task_type == "implementation":
            for agent in agents:
                if agent.name == "codex":
                    return agent

        # Return first available
        return agents[0]

    def _build_prompt(self, request: ExecutionRequest) -> str:
        """Build rich prompt with context and suggestions"""
        parts = [
            f"Task: {request.task}",
            f"\nContext:",
            f"- Working directory: {request.workspace_context.workspace_root}",
        ]

        if request.workspace_context.git_branch:
            parts.append(f"- Current branch: {request.workspace_context.git_branch}")

        if request.workspace_context.recent_files:
            files = ", ".join(request.workspace_context.recent_files[:5])
            parts.append(f"- Recent files: {files}")

        # Add suggestions (agent can ignore)
        if request.suggested_approach:
            parts.append(f"\nSuggested approach: {request.suggested_approach}")

        if request.available_skills:
            parts.append("\nYou have these capabilities available:")
            for agent_name, skills in request.available_skills.items():
                if skills:
                    skills_str = ", ".join([s.get('name', '') for s in skills[:3]])
                    parts.append(f"  - {skills_str}")
            parts.append("Feel free to use them if helpful, or take a different approach.")

        return "\n".join(parts)

    def _summarize_agent_output(self, result, task_type: str) -> dict:
        """Extract key info, discard raw output (context hygiene)"""
        summary = {
            "task_type": task_type,
            "status": "success" if (hasattr(result, 'success') and result.success) or (hasattr(result, 'exit_code') and result.exit_code == 0) else "failed",
        }

        # Add task-specific metadata
        if task_type == "implementation":
            content = getattr(result, 'content', '') or getattr(result, 'output', '')
            summary["output_preview"] = content[:200] if content else ""

        return summary
