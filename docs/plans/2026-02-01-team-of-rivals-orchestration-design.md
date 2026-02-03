# Team-of-Rivals Multi-Agent Orchestration Design

**Date:** 2026-02-01
**Status:** Design Complete, Ready for Implementation
**Author:** Design Session with Claude Code

## Executive Summary

This document describes the architecture for transforming `orch` from a simple agent delegation tool into a sophisticated multi-agent orchestration system based on the "team of rivals" concept from the paper "If You Want Coherence, Orchestrate a Team of Rivals: Multi-Agent Models of Organizational Intelligence" (arXiv:2601.14351).

**Key Innovation:** Claude Code becomes a manager that orchestrates specialized AI agents (Planner, Executors, Critics, Experts) with adversarial review, automatic complexity detection, and continuous learning from failures.

**Design Philosophy:**
- Automatic escalation from simple delegation to full team orchestration
- Agent autonomy (suggestions, not commands)
- Context hygiene (summaries, not raw outputs)
- Resilience through checkpointing
- Continuous improvement via failure analytics

## Problem Statement

Single-agent systems (where one LLM handles planning, execution, and self-critique) suffer from:

1. **Context contamination** - Full conversation history dumped into every prompt
2. **Hallucinations** - Errors propagate unchecked
3. **Lack of resilience** - Single failure crashes entire process
4. **Poor auditability** - No clear decision trail

## Solution: Team-of-Rivals Architecture

### High-Level Architecture

```
User Request
    ↓
Claude Code (Manager Role)
    ↓
ComplexityAnalyzer ──→ [Simple] ──→ Router ──→ Single Agent (existing)
    ↓
   [Complex]
    ↓
TeamOrchestrator
    ├── PlannerAgent ──→ Creates structured implementation plan
    ├── ExecutorAgents ──→ ExecutionRouter
    │   ├── SubprocessExecutor (tests, lint, validation)
    │   └── AgentDelegator (delegates to codex/opencode/gemini)
    └── CriticAgents (adversarial review)
        ├── SecurityCritic (absolute veto)
        ├── CorrectnessCritic (strong veto)
        ├── PerformanceCritic (weak veto)
        └── StyleCritic (suggestion only)
    ↓
CriticAggregator (Veto Hierarchy + Weighted Scoring)
    ├── Accept ──→ Return Artifact
    ├── Needs Revision ──→ Refine & Re-execute
    └── Reject ──→ Replan & Re-execute
    ↓
Final Artifact + Trace + Analytics
```

### Core Components

#### 1. Complexity Analyzer (`src/orch/orchestration/analyzer.py`)

**Purpose:** Automatically detect when tasks need team orchestration vs simple delegation

**Detection Signals:**

| Signal Type | Simple Task | Complex Task |
|------------|-------------|--------------|
| Scope | Single file, <50 LOC | Multi-file, architectural changes |
| Keywords | "explain", "what is", "fix typo" | "design", "refactor", "implement system" |
| Security | No auth/crypto/validation | Mentions security, permissions |
| Dependencies | No external integration | Database, API, third-party services |
| Risk | Read-only, non-critical | State changes, migrations, breaking changes |

**Escalation Levels:**
- **Simple** → Router (existing behavior)
- **Standard** → Team with Executor + 1 Critic
- **Complex** → Full team (Planner + Executors + 4 Critics)

**Configuration:**
```toml
[orchestration]
auto_detect = true
default_mode = "auto"  # "simple" | "standard" | "team" | "auto"
complexity_threshold = "medium"

[orchestration.signals]
multi_file_threshold = 3
security_keywords = ["auth", "crypto", "token", "password"]
architectural_keywords = ["design", "refactor", "migrate", "scale"]
```

#### 2. Team Orchestrator (`src/orch/orchestration/team.py`)

**Purpose:** Coordinates the team-of-rivals workflow with state management

**Workflow Phases:**

```python
async def orchestrate(user_prompt: str) -> OrchestrationResult:
    # Phase 1: Planning
    plan = await planner.propose(task_state)
    checkpoint("plan_complete")

    # Phase 2: Execution loop (max iterations)
    for iteration in range(max_iterations):
        # Execute plan steps
        results = await execute_plan(plan)
        checkpoint(f"execution_{iteration}")

        # Critics review
        critique = await critique_phase(results)
        checkpoint(f"critique_{iteration}")

        # Decision
        if critique.decision == "accept":
            break
        elif critique.decision == "reject":
            plan = await replan(plan, critique)
        else:  # needs_revision
            plan = revise(plan, critique)

    # Phase 3: Finalize
    artifact = create_artifact(results)
    checkpoint("complete")

    return OrchestrationResult(artifact, trace, metrics)
```

**State Management:**

```python
@dataclass
class OrchestrationSession:
    id: str
    user_prompt: str
    complexity_level: str
    workspace_root: Path
    started_at: datetime
    state: str  # "planning" | "executing" | "critiquing" | "complete"
    iteration: int
    trace: SessionTrace  # Audit log of all agent interactions
    metrics: SessionMetrics  # Performance metrics
    checkpoints: list[Checkpoint]  # Recovery points
    checkpoint_dir: Path
```

**Checkpoint Strategy:**
- Automatic snapshots after each major phase
- Stored in `~/.config/orch/sessions/<session-id>/`
- Recovery: `orch orchestrate resume <session-id> --from <phase>`

#### 3. Agent Role Protocol (`src/orch/agents/roles/protocol.py`)

**Base Interface:**

```python
class RoleAgent(ABC):
    @property
    @abstractmethod
    def role_name(self) -> str:
        """planner, executor, critic, expert"""

    @property
    @abstractmethod
    def goal(self) -> str:
        """What this agent optimizes for"""

    @abstractmethod
    async def propose(self, task_state: TaskState) -> AgentMessage:
        """Generate proposal based on task state"""

    @abstractmethod
    async def review(self, task_state: TaskState, artifact: Artifact) -> ReviewFeedback:
        """Review output and provide feedback"""
```

**Structured State (not raw text):**

```python
@dataclass
class TaskState:
    user_prompt: str
    complexity_level: str
    workspace_context: WorkspaceContext
    current_phase: str
    iteration: int
    metadata: dict

@dataclass
class AgentMessage:
    role: str
    content: str
    structured_data: dict  # Parsed plans, execution requests
    confidence: float
    timestamp: datetime

@dataclass
class ReviewFeedback:
    critic_type: str
    decision: str  # "accept", "reject", "needs_revision"
    issues: list[Issue]
    severity_score: int  # 0-100
    confidence: float
```

#### 4. Role Implementations

**PlannerAgent** (`src/orch/agents/roles/planner.py`)
- Goal: "Create clear, step-by-step implementation plans"
- Uses LLM to analyze prompt → structured plan
- Output: Plan with steps, dependencies, risk assessment

**ExecutorAgent** (`src/orch/agents/roles/executor.py`)
- Goal: "Implement tasks correctly and efficiently"
- Creates ExecutionRequest objects
- Delegates to hybrid execution layer
- Returns summarized results (context hygiene)

**CriticAgents** (`src/orch/agents/roles/critic.py`)

Four specialized critics with different veto powers:

| Critic | Veto Power | Focus | Blocks On |
|--------|-----------|-------|-----------|
| SecurityCritic | Absolute | Auth, crypto, injection | Any critical security issue |
| CorrectnessCritic | Strong | Logic, tests, edge cases | Test failures, logic errors |
| PerformanceCritic | Weak | Efficiency, scalability | Egregious algorithmic issues |
| StyleCritic | None (suggestion) | Code quality, maintainability | Never blocks |

**ExpertAgent** (`src/orch/agents/roles/expert.py`)
- Optional domain specialists (database, frontend, ML)
- Dynamically selected based on task type

#### 5. Hybrid Execution Layer (`src/orch/execution/`)

**ExecutionRouter** - Routes requests to appropriate executor

**SubprocessExecutor** (`subprocess_executor.py`)
- Fast, sandboxed execution for simple tasks
- Handles: pytest, ruff, mypy, bandit, jsonschema
- Returns structured summaries (not full output)

**AgentDelegator** (`agent_delegator.py`)
- Delegates complex tasks to AI CLIs
- Builds rich prompts with context
- Summarizes agent output to prevent context overflow

**Execution Request Structure:**

```python
@dataclass
class ExecutionRequest:
    # What needs to be done
    task: str
    task_type: str  # "implementation", "testing", "exploration"

    # Rich context for autonomous decision-making
    workspace_context: WorkspaceContext
    previous_attempts: list[ExecutionResult]
    constraints: dict  # Time, budget, sandbox

    # Suggestions (agent can ignore)
    suggested_agent: str | None
    suggested_approach: str | None
    available_skills: dict[str, list[NativeSkill]]
```

**Agent Autonomy:**
- Orchestrator provides suggestions, not commands
- Agents decide whether to use native skills/plugins
- Prompts include context + suggestions, not directives

#### 6. Context Propagation (`src/orch/orchestration/context.py`)

**AgentContext** - Rich context for independent operation:

```python
@dataclass
class AgentContext:
    # Workspace
    workspace_root: Path
    project_instructions: str | None  # CLAUDE.md contents
    available_skills: list[SkillMetadata]  # Claude Code skills
    git_context: GitContext  # Branch, commits, status

    # Task
    task_state: TaskState
    previous_messages: list[AgentMessage]
    execution_history: list[ExecutionResult]

    # Agent-specific
    role_instructions: str  # Loaded from instructions/<role>.md
    tools_available: list[str]
    constraints: dict

    # Codebase intelligence
    file_index: FileIndex | None
    dependency_graph: DependencyGraph | None
```

**ContextBuilder** automatically assembles context by:
- Loading CLAUDE.md
- Scanning .claude/skills/
- Getting git status
- Building file index
- Loading role-specific instructions

#### 7. Critic Aggregation (`src/orch/orchestration/critic_aggregator.py`)

**Veto Hierarchy + Weighted Scoring:**

```python
def aggregate(reviews: list[ReviewFeedback]) -> CritiqueDecision:
    # Step 1: Check absolute vetoes (security)
    if has_security_veto(reviews):
        return reject_for_security(reviews)

    # Step 2: Check strong vetoes (correctness)
    if has_correctness_veto(reviews):
        return reject_for_correctness(reviews)

    # Step 3: Weighted scoring for ambiguous cases
    if has_ambiguous_reviews(reviews):
        return weighted_scoring_decision(reviews)

    # Step 4: Aggregate non-critical issues
    if has_major_issues(reviews):
        return needs_revision(reviews)

    return accept(reviews)
```

**Weighted Scoring (for ambiguous cases):**
- Weight by critic priority and confidence
- Score threshold: >80 accept, 60-80 revise, <60 reject
- Used when weak critics disagree or confidence is low

#### 8. Failure Analytics (`src/orch/evaluation/tracker.py`)

**Logged Events:**

1. **Escalation Failures**
   - Wrong complexity detection
   - Tracks: prompt, detected vs actual mode needed

2. **Critic Interceptions**
   - Errors caught by critics
   - Tracks: critic type, issue category, severity

3. **Execution Failures**
   - Subprocess/agent failures
   - Tracks: executor type, error type, retry count

4. **Orchestration Metrics**
   - Rounds, time per phase, final outcome

**Storage:**
```
~/.config/orch/analytics/
├── failures/
│   ├── escalation-*.json
│   ├── critic-*.json
│   └── execution-*.json
├── sessions/
│   └── <session-id>/
│       ├── checkpoint-*.json
│       └── trace.json
└── metrics.db (SQLite)
```

**Analytics Commands:**
```bash
orch analytics failures --last 7d
orch analytics escalation --accuracy
orch analytics critics --effectiveness
orch analytics export --format csv
```

### CLI Integration

**New Commands:**

```bash
# Team orchestration
orch orchestrate run "implement auth system"
orch orchestrate run --mode simple "quick fix"  # Force simple
orch orchestrate resume <session-id>

# Session management
orch session list --active
orch session status <session-id>
orch session trace <session-id> --format markdown

# Analytics
orch analytics failures --last 7d
orch analytics escalation
orch analytics critics
```

**Backward Compatibility:**

```bash
# Existing commands work unchanged
orch ask "fix bug"              # Auto-escalates if complex
orch gemini "explain code"      # Direct agent access
orch compare "implement feature"

# Auto-escalation in ask command
orch ask "refactor auth system"
# → Detects complexity
# → Auto-escalates: "Auto-escalating to team orchestration..."
# → Runs full workflow
```

### Enhanced Skill (`.claude/skills/using-orch/SKILL.md`)

**Key Additions:**

1. **When orchestration auto-escalates** - Teach Claude the signals
2. **How to observe workflow** - Interpret planning/execution/critique phases
3. **Session management** - Resume, review traces
4. **Analytics usage** - Learn from failures
5. **Integration patterns** - Combine with TDD, git-worktrees, etc.

**Claude's Role:**
- Manager observing the team
- Explains decisions to user
- Highlights critic catches (security/correctness wins)
- Reviews traces to understand trade-offs
- Provides high-level guidance

## Implementation Roadmap

### Phase 1: Foundation (MVP)
- [ ] Core data structures (TaskState, AgentMessage, etc.)
- [ ] TeamOrchestrator with basic workflow
- [ ] Simple PlannerAgent (LLM-based)
- [ ] ExecutorAgent with delegation
- [ ] Hybrid execution layer (subprocess + agent)
- [ ] Basic critics (security, correctness)
- [ ] Checkpoint system
- [ ] CLI commands (orchestrate run, session list)
- [ ] Failure logging

### Phase 2: Enhanced Intelligence
- [ ] ComplexityAnalyzer with auto-detection
- [ ] CriticAggregator with veto hierarchy
- [ ] Weighted scoring for ambiguous cases
- [ ] Performance and Style critics
- [ ] Context propagation system
- [ ] Enhanced result summarization
- [ ] Analytics commands
- [ ] Updated using-orch skill

### Phase 3: Advanced Features
- [ ] Parallel execution of independent steps
- [ ] Dynamic critic selection
- [ ] Expert agents
- [ ] Real-time streaming progress
- [ ] LLM fine-tuning for roles
- [ ] Integration with external tools

## Extension Points

```python
# Pluggable components for future iteration
class ComplexityAnalyzer(ABC): ...
class CustomCritic(CriticAgent): ...
class CustomExecutor(RemoteCodeExecutor): ...
class LLMClient(ABC): ...
class AnalyticsPlugin(ABC): ...
```

## Configuration Schema

```toml
[orchestration]
auto_detect = true
default_mode = "auto"
max_iterations = 5
complexity_threshold = "medium"

[orchestration.signals]
multi_file_threshold = 3
security_keywords = ["auth", "crypto", "token"]
architectural_keywords = ["design", "refactor", "migrate"]

[orchestration.critics]
enabled = ["security", "correctness", "performance", "style"]

[orchestration.critics.security]
veto_power = "absolute"
check_for = ["auth", "crypto", "injection", "secrets"]

[orchestration.critics.correctness]
veto_power = "strong"
require_tests = true
min_test_coverage = 0.8

[orchestration.analytics]
enabled = true
log_failures = true
retention_days = 90
```

## Key Design Principles

1. **Automatic escalation** - No manual mode selection needed
2. **Agent autonomy** - Suggestions, not commands
3. **Context hygiene** - Summaries prevent overflow
4. **Adversarial review** - Critics catch errors through opposition
5. **Resilience** - Checkpoints enable recovery
6. **Continuous learning** - Failures improve future decisions
7. **Backward compatibility** - Existing workflows unchanged
8. **Iterative improvement** - Extension points for evolution

## Success Metrics

Track these to measure effectiveness:
- **Escalation accuracy** - % correct complexity detection
- **Critic effectiveness** - Issues found / false positives
- **Iteration efficiency** - Average rounds to completion
- **Error interception rate** - % of errors caught by critics
- **Time per phase** - Identify bottlenecks
- **User satisfaction** - Explicit feedback
- **Cost metrics** - Tokens, execution time

## Open Questions for Iteration

1. How to handle partially-failed executions? (some steps succeed, others fail)
2. Should experts be invoked proactively or on-demand?
3. Optimal max_iterations default? (currently 5)
4. When to suggest config changes to users?
5. How to handle conflicting expert opinions?

## References

- Paper: "If You Want Coherence, Orchestrate a Team of Rivals" (arXiv:2601.14351)
- Existing orch codebase: Router, AgentRegistry, existing CLI
- Claude Code skills: TDD, git-worktrees, code-review
