"""Team-of-Rivals Orchestrator - MVP Implementation"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from orch.config.manager import ConfigManager
from orch.config.schema import get_sessions_dir
from orch.llm.client import LLMClientFactory
from orch.orchestration.checkpoint import Checkpoint, CheckpointManager
from orch.orchestration.complexity import ComplexityAnalyzer
from orch.orchestration.models import AgentMessage


@dataclass
class SessionMetrics:
    """Performance metrics for analysis"""
    executions_count: int = 0
    critique_rounds: int = 0
    revisions_requested: int = 0
    issues_found: dict[str, int] = field(default_factory=dict)
    time_per_phase: dict[str, float] = field(default_factory=dict)
    tokens_used: int = 0


@dataclass
class SessionTrace:
    """Audit log of all agent interactions"""
    messages: list[tuple[str, AgentMessage]] = field(default_factory=list)
    executions: list[tuple[str, dict]] = field(default_factory=list)
    reviews: list[tuple[str, dict]] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)

    def add_message(self, role: str, message: AgentMessage):
        """Add agent message to trace"""
        self.messages.append((role, message))

    def add_execution(self, step: str, result: dict):
        """Add execution result to trace"""
        self.executions.append((step, result))

    def add_review(self, critic: str, feedback: dict):
        """Add critic review to trace"""
        self.reviews.append((critic, feedback))

    def to_markdown(self) -> str:
        """Export trace as markdown"""
        lines = ["# Orchestration Trace\n"]

        if self.messages:
            lines.append("## Agent Messages\n")
            for role, msg in self.messages:
                lines.append(f"### {role}")
                lines.append(f"{msg.content}\n")

        if self.executions:
            lines.append("## Executions\n")
            for step, result in self.executions:
                lines.append(f"- **{step}**: {result.get('status', 'unknown')}")

        if self.reviews:
            lines.append("\n## Reviews\n")
            for critic, feedback in self.reviews:
                lines.append(f"- **{critic}**: {feedback.get('decision', 'unknown')}")

        return "\n".join(lines)


@dataclass
class OrchestrationSession:
    """Tracks state across entire orchestration"""
    id: str
    user_prompt: str
    complexity_level: str
    workspace_root: Path
    started_at: datetime
    completed_at: datetime | None = None

    # Current state
    state: str = "initializing"  # "planning" | "executing" | "critiquing" | "complete" | "failed"
    iteration: int = 0

    # Accumulated data
    trace: SessionTrace = field(default_factory=SessionTrace)
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    checkpoints: list[Checkpoint] = field(default_factory=list)

    # Checkpoint persistence
    checkpoint_dir: Path | None = None

    def to_dict(self) -> dict:
        """Convert to dict for serialization"""
        return {
            "id": self.id,
            "user_prompt": self.user_prompt,
            "complexity_level": self.complexity_level,
            "workspace_root": str(self.workspace_root),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "state": self.state,
            "iteration": self.iteration,
            "metrics": {
                "executions_count": self.metrics.executions_count,
                "critique_rounds": self.metrics.critique_rounds,
                "revisions_requested": self.metrics.revisions_requested,
            }
        }


@dataclass
class OrchestrationResult:
    """Final result from orchestration"""
    session_id: str
    success: bool
    artifact: dict  # The final output
    trace: SessionTrace
    metrics: SessionMetrics
    error: str | None = None


class TeamOrchestrator:
    """MVP Team-of-Rivals Orchestrator"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.session: OrchestrationSession | None = None

    async def orchestrate(
        self, user_prompt: str, options: dict | None = None
    ) -> OrchestrationResult:
        """Main entry point - runs team-of-rivals workflow"""
        options = options or {}

        # Initialize session
        session = self._create_session(user_prompt, options)
        self.session = session

        # Create checkpoint manager
        checkpoint_mgr = CheckpointManager(session.checkpoint_dir)

        try:
            # Checkpoint: init
            await self._checkpoint(checkpoint_mgr, "init")

            # === Complexity Detection ===
            if not options.get("complexity") or options.get("complexity") == "auto":
                session.state = "analyzing_complexity"

                config = ConfigManager.get_config()
                llm_client = LLMClientFactory.create(config)

                analyzer = ComplexityAnalyzer(llm_client, config)
                complexity_result = await analyzer.analyze(
                    user_prompt,
                    None  # workspace_context - TODO: add in future
                )

                session.complexity_level = complexity_result.complexity_level

                await self._checkpoint(
                    checkpoint_mgr,
                    "complexity_detected",
                    complexity=complexity_result.to_dict()
                )
            else:
                # Manual complexity specified
                session.complexity_level = options["complexity"]

            # MVP: Simple workflow without actual agents
            # This demonstrates the structure - full implementation adds real agents

            # Phase 1: Planning (simplified)
            session.state = "planning"
            plan = {"steps": ["Analyze task", "Implement solution", "Test"]}
            await self._checkpoint(checkpoint_mgr, "plan_complete", plan=plan)

            # Phase 2: Execution (simplified)
            session.state = "executing"
            results = {"status": "completed", "output": "MVP implementation"}
            session.metrics.executions_count = 1
            await self._checkpoint(
                checkpoint_mgr, f"execution_{session.iteration}", results=results
            )

            # Phase 3: Critique (simplified)
            session.state = "critiquing"
            critique = {"decision": "accept", "reason": "MVP passes"}
            session.metrics.critique_rounds = 1
            await self._checkpoint(
                checkpoint_mgr, f"critique_{session.iteration}", critique=critique
            )

            # Phase 4: Finalize
            session.state = "complete"
            session.completed_at = datetime.now()
            artifact = {
                "status": "success",
                "plan": plan,
                "results": results,
                "critique": critique
            }
            await self._checkpoint(checkpoint_mgr, "complete", artifact=artifact)

            return OrchestrationResult(
                session_id=session.id,
                success=True,
                artifact=artifact,
                trace=session.trace,
                metrics=session.metrics
            )

        except Exception as e:
            session.state = "failed"
            return OrchestrationResult(
                session_id=session.id,
                success=False,
                artifact={},
                trace=session.trace,
                metrics=session.metrics,
                error=str(e)
            )

    def _create_session(self, prompt: str, options: dict) -> OrchestrationSession:
        """Initialize orchestration session"""
        session_id = str(uuid.uuid4())[:8]

        # Create session directory
        sessions_dir = get_sessions_dir()
        session_dir = sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        return OrchestrationSession(
            id=session_id,
            user_prompt=prompt,
            complexity_level=options.get("complexity", "standard"),
            workspace_root=Path.cwd(),
            started_at=datetime.now(),
            checkpoint_dir=session_dir
        )

    async def _checkpoint(self, mgr: CheckpointManager, phase: str, **data):
        """Save checkpoint"""
        if not self.session:
            return

        checkpoint = Checkpoint(
            session_id=self.session.id,
            phase=phase,
            timestamp=datetime.now(),
            state_snapshot=self.session.to_dict(),
            data=data
        )

        mgr.save_checkpoint(checkpoint)
        self.session.checkpoints.append(checkpoint)
