"""Smart routing logic for determining best agent based on task type."""

import re
from typing import Any

from orch.agents.registry import AgentRegistry
from orch.config.schema import RoutingConfig


class Router:
    """Routes prompts to the best available agent based on task type."""

    def __init__(self, config: RoutingConfig) -> None:
        self.config = config
        self._keyword_patterns: dict[str, re.Pattern[str]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile keyword patterns for faster matching."""
        for task_type, keywords in self.config.keywords.items():
            # Create pattern that matches any of the keywords
            pattern = "|".join(re.escape(kw) for kw in keywords)
            self._keyword_patterns[task_type] = re.compile(pattern, re.IGNORECASE)

    def classify_task(self, prompt: str) -> str:
        """Classify a prompt into a task type.

        Returns:
            Task type string (e.g., 'code', 'explain', 'debug', 'general')
        """
        # Check each task type's keywords
        scores: dict[str, int] = {}

        for task_type, pattern in self._keyword_patterns.items():
            matches = pattern.findall(prompt)
            if matches:
                scores[task_type] = len(matches)

        if scores:
            # Return task type with most keyword matches
            return max(scores, key=scores.get)  # type: ignore

        return "general"

    def get_preferred_agents(self, task_type: str) -> list[str]:
        """Get ordered list of preferred agents for a task type."""
        return self.config.rules.get(task_type, self.config.rules.get("general", []))

    def route(self, prompt: str) -> str:
        """Route a prompt to the best available agent.

        Args:
            prompt: The user's prompt

        Returns:
            Name of the best agent to use
        """
        if not self.config.enabled:
            # Routing disabled, use first available agent
            available = AgentRegistry.get_available()
            if available:
                return available[0].name
            raise ValueError("No agents available")

        # Classify the task
        task_type = self.classify_task(prompt)

        # Get preferred agents for this task type
        preferred = self.get_preferred_agents(task_type)

        # Find first available agent from preferred list
        for agent_name in preferred:
            agent = AgentRegistry.get(agent_name)
            if agent and agent.is_available():
                return agent_name

        # Fall back to any available agent
        available = AgentRegistry.get_available()
        if available:
            return available[0].name

        raise ValueError("No agents available")

    def route_with_explanation(self, prompt: str) -> tuple[str, str]:
        """Route with explanation of why.

        Returns:
            Tuple of (agent_name, explanation)
        """
        task_type = self.classify_task(prompt)
        agent_name = self.route(prompt)

        preferred = self.get_preferred_agents(task_type)

        explanation = (
            f"Task classified as '{task_type}'. "
            f"Preferred agents: {', '.join(preferred)}. "
            f"Selected: {agent_name}"
        )

        return agent_name, explanation

    def get_agents_for_task(self, task_type: str) -> list[str]:
        """Get all available agents suitable for a task type, in preference order."""
        preferred = self.get_preferred_agents(task_type)
        available: list[str] = []

        for agent_name in preferred:
            agent = AgentRegistry.get(agent_name)
            if agent and agent.is_available():
                available.append(agent_name)

        return available


class StrengthBasedRouter(Router):
    """Router that also considers agent capability strengths."""

    def route(self, prompt: str) -> str:
        """Route considering both keywords and agent strengths."""
        task_type = self.classify_task(prompt)

        # Get all available agents
        available = AgentRegistry.get_available()
        if not available:
            raise ValueError("No agents available")

        # Score agents based on strength match
        scores: dict[str, float] = {}

        for agent in available:
            caps = agent.get_capabilities()
            score = 0.0

            # Check if task type is in agent's strengths
            if task_type in caps.task_strengths:
                score += 2.0

            # Check for related strengths
            related_strengths = {
                "code": ["debugging", "implementation", "refactoring"],
                "explain": ["analysis", "research", "general"],
                "debug": ["code", "implementation"],
            }

            for strength in caps.task_strengths:
                if strength in related_strengths.get(task_type, []):
                    score += 1.0

            # Boost from config preference
            preferred = self.get_preferred_agents(task_type)
            if agent.name in preferred:
                position = preferred.index(agent.name)
                score += (len(preferred) - position) * 0.5

            scores[agent.name] = score

        # Return highest scoring agent
        return max(scores, key=scores.get)  # type: ignore
