"""Critic agent role implementations."""
from orch.agents.roles.protocol import RoleAgent
from orch.orchestration.models import TaskState, AgentMessage, ReviewFeedback, Issue


class SecurityCritic(RoleAgent):
    """Security-focused critic with absolute veto power."""

    veto_power = "absolute"

    @property
    def role_name(self) -> str:
        return "security_critic"

    @property
    def goal(self) -> str:
        return "Identify security vulnerabilities and block unsafe changes"

    async def propose(self, task_state: TaskState) -> AgentMessage:
        """Critics do not propose implementation plans."""
        return AgentMessage(role=self.role_name, content="Security critic does not propose")

    async def review(self, task_state: TaskState, artifact: dict) -> ReviewFeedback:
        """Analyze artifacts for security issues and return feedback."""
        raw_issues = artifact.get("security_issues", [])
        issues = [
            Issue(
                category="security",
                severity="critical",
                description=description,
            )
            for description in raw_issues
        ]

        decision = "reject" if issues else "accept"
        severity_score = 0 if issues else 100
        return ReviewFeedback(
            critic_type="security",
            decision=decision,
            issues=issues,
            severity_score=severity_score,
        )


class CorrectnessCritic(RoleAgent):
    """Correctness-focused critic with strong veto power."""

    veto_power = "strong"

    @property
    def role_name(self) -> str:
        return "correctness_critic"

    @property
    def goal(self) -> str:
        return "Identify logic errors, test failures, and edge case gaps"

    async def propose(self, task_state: TaskState) -> AgentMessage:
        """Critics do not propose implementation plans."""
        return AgentMessage(role=self.role_name, content="Correctness critic does not propose")

    async def review(self, task_state: TaskState, artifact: dict) -> ReviewFeedback:
        """Analyze artifacts for correctness issues and return feedback."""
        raw_issues = []
        raw_issues.extend(artifact.get("correctness_issues", []))
        raw_issues.extend(artifact.get("logic_errors", []))
        raw_issues.extend(artifact.get("test_failures", []))

        issues = [
            Issue(
                category="logic_error",
                severity="major",
                description=description,
            )
            for description in raw_issues
        ]

        decision = "reject" if issues else "accept"
        severity_score = 20 if issues else 100
        return ReviewFeedback(
            critic_type="correctness",
            decision=decision,
            issues=issues,
            severity_score=severity_score,
        )
