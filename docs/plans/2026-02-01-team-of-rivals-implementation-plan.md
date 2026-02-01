# Team-of-Rivals Orchestration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform orch into a multi-agent orchestration system with automatic complexity detection, adversarial review, and continuous learning.

**Architecture:** Layer team-of-rivals orchestration on top of existing Router. ComplexityAnalyzer detects when to escalate from simple delegation to full orchestration (Planner → Executors → Critics). Hybrid execution uses subprocess for simple tasks and AI CLIs for complex work. Checkpoints enable recovery.

**Tech Stack:** Python 3.11+, Click (CLI), Pydantic (schemas), asyncio, existing orch agent infrastructure

---

## Phase 1: Foundation (MVP)

### Task 1: Core Data Structures

**Files:**
- Create: `src/orch/orchestration/models.py`
- Test: `tests/unit/orchestration/test_models.py`

**Step 1: Write failing test for TaskState**

```python
# tests/unit/orchestration/test_models.py
from pathlib import Path
from datetime import datetime
from orch.orchestration.models import TaskState, WorkspaceContext

def test_task_state_creation():
    workspace = WorkspaceContext(
        workspace_root=Path("/test/workspace"),
        git_branch="main",
        recent_files=["file1.py", "file2.py"]
    )

    state = TaskState(
        user_prompt="implement auth",
        complexity_level="high",
        workspace_context=workspace,
        current_phase="planning",
        iteration=0,
        metadata={}
    )

    assert state.user_prompt == "implement auth"
    assert state.complexity_level == "high"
    assert state.current_phase == "planning"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/orchestration/test_models.py::test_task_state_creation -v`
Expected: FAIL with "No module named 'orch.orchestration.models'"

**Step 3: Create directory structure**

```bash
mkdir -p src/orch/orchestration tests/unit/orchestration
touch src/orch/orchestration/__init__.py
touch tests/unit/orchestration/__init__.py
```

**Step 4: Implement minimal models**

```python
# src/orch/orchestration/models.py
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class WorkspaceContext:
    """Context about the current workspace"""
    workspace_root: Path
    git_branch: str | None = None
    recent_files: list[str] = field(default_factory=list)
    git_status: str | None = None


@dataclass
class TaskState:
    """Shared state across all agents"""
    user_prompt: str
    complexity_level: str  # "simple" | "standard" | "complex"
    workspace_context: WorkspaceContext
    current_phase: str  # "planning" | "execution" | "critique"
    iteration: int
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Message from an agent"""
    role: str  # Agent role name
    content: str
    structured_data: dict = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Issue:
    """A single issue found by a critic"""
    category: str  # "security", "logic_error", "performance", "style"
    severity: str  # "critical", "major", "minor", "suggestion"
    description: str
    location: str | None = None
    suggested_fix: str | None = None
    confidence: float = 1.0


@dataclass
class ReviewFeedback:
    """Critic's review result"""
    critic_type: str  # "security", "correctness", "performance", "style"
    decision: str  # "accept", "reject", "needs_revision"
    issues: list[Issue] = field(default_factory=list)
    severity_score: int = 100  # 0-100
    confidence: float = 1.0


@dataclass
class ExecutionRequest:
    """Request for code execution"""
    task: str
    task_type: str  # "implementation", "testing", "exploration"
    workspace_context: WorkspaceContext
    previous_attempts: list = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    suggested_agent: str | None = None
    suggested_approach: str | None = None
    available_skills: dict = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result from execution"""
    executor_type: str  # "subprocess" | "agent_cli"
    success: bool
    summary: dict
    agent_name: str | None = None
    error: str | None = None
    metrics: dict = field(default_factory=dict)
```

**Step 5: Run test to verify it passes**

Run: `pytest tests/unit/orchestration/test_models.py::test_task_state_creation -v`
Expected: PASS

**Step 6: Add more model tests**

```python
# tests/unit/orchestration/test_models.py
def test_agent_message_creation():
    msg = AgentMessage(
        role="planner",
        content="Create 3-step plan",
        structured_data={"steps": [1, 2, 3]},
        confidence=0.95
    )
    assert msg.role == "planner"
    assert len(msg.structured_data["steps"]) == 3

def test_review_feedback_with_issues():
    issue = Issue(
        category="security",
        severity="critical",
        description="SQL injection vulnerability"
    )

    feedback = ReviewFeedback(
        critic_type="security",
        decision="reject",
        issues=[issue],
        severity_score=0
    )

    assert feedback.decision == "reject"
    assert len(feedback.issues) == 1
    assert feedback.issues[0].severity == "critical"
```

**Step 7: Run all model tests**

Run: `pytest tests/unit/orchestration/test_models.py -v`
Expected: All PASS

**Step 8: Commit**

```bash
git add src/orch/orchestration/ tests/unit/orchestration/
git commit -m "feat: add core orchestration data models

- TaskState, WorkspaceContext for state management
- AgentMessage, ReviewFeedback for agent communication
- ExecutionRequest, ExecutionResult for execution layer
- Issue model for critic feedback"
```

---

### Task 2: Agent Role Protocol

**Files:**
- Create: `src/orch/agents/roles/__init__.py`
- Create: `src/orch/agents/roles/protocol.py`
- Test: `tests/unit/agents/roles/test_protocol.py`

**Step 1: Write failing test for RoleAgent protocol**

```python
# tests/unit/agents/roles/test_protocol.py
import pytest
from orch.agents.roles.protocol import RoleAgent
from orch.orchestration.models import TaskState, AgentMessage, WorkspaceContext

class TestRoleAgent(RoleAgent):
    """Concrete test implementation"""

    @property
    def role_name(self) -> str:
        return "test"

    @property
    def goal(self) -> str:
        return "Test agent goal"

    async def propose(self, task_state: TaskState) -> AgentMessage:
        return AgentMessage(role="test", content="proposal")

    async def review(self, task_state: TaskState, artifact: dict) -> object:
        return None

def test_role_agent_protocol():
    agent = TestRoleAgent()
    assert agent.role_name == "test"
    assert agent.goal == "Test agent goal"

@pytest.mark.asyncio
async def test_role_agent_propose():
    agent = TestRoleAgent()
    workspace = WorkspaceContext(workspace_root=Path("/test"))
    state = TaskState(
        user_prompt="test task",
        complexity_level="simple",
        workspace_context=workspace,
        current_phase="planning",
        iteration=0
    )

    message = await agent.propose(state)
    assert message.role == "test"
    assert message.content == "proposal"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/agents/roles/test_protocol.py -v`
Expected: FAIL with "No module named 'orch.agents.roles'"

**Step 3: Create role agent protocol**

```python
# src/orch/agents/roles/__init__.py
from .protocol import RoleAgent

__all__ = ["RoleAgent"]
```

```python
# src/orch/agents/roles/protocol.py
from abc import ABC, abstractmethod
from orch.orchestration.models import TaskState, AgentMessage, ReviewFeedback


class RoleAgent(ABC):
    """Base protocol for team-of-rivals agents"""

    @property
    @abstractmethod
    def role_name(self) -> str:
        """Agent role: planner, executor, critic, expert"""
        pass

    @property
    @abstractmethod
    def goal(self) -> str:
        """What this agent optimizes for"""
        pass

    @abstractmethod
    async def propose(self, task_state: TaskState) -> AgentMessage:
        """Generate proposal/plan/implementation based on task state"""
        pass

    @abstractmethod
    async def review(self, task_state: TaskState, artifact: dict) -> ReviewFeedback:
        """Review output and provide feedback"""
        pass
```

**Step 4: Create test directory and run**

```bash
mkdir -p src/orch/agents/roles tests/unit/agents/roles
touch tests/unit/agents/roles/__init__.py
pytest tests/unit/agents/roles/test_protocol.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/agents/roles/ tests/unit/agents/roles/
git commit -m "feat: add RoleAgent protocol for team-of-rivals

- Abstract base class for all role agents
- Defines propose() and review() interface
- Foundation for Planner, Executor, Critic agents"
```

---

### Task 3: Execution Layer - Protocol

**Files:**
- Create: `src/orch/execution/__init__.py`
- Create: `src/orch/execution/protocol.py`
- Test: `tests/unit/execution/test_protocol.py`

**Step 1: Write failing test for RemoteCodeExecutor**

```python
# tests/unit/execution/test_protocol.py
import pytest
from orch.execution.protocol import RemoteCodeExecutor
from orch.orchestration.models import ExecutionRequest, ExecutionResult, WorkspaceContext
from pathlib import Path

class TestExecutor(RemoteCodeExecutor):
    """Test implementation"""

    def can_handle(self, request: ExecutionRequest) -> bool:
        return request.task_type == "test"

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        return ExecutionResult(
            executor_type="test",
            success=True,
            summary={"status": "done"}
        )

    async def health_check(self) -> bool:
        return True

@pytest.mark.asyncio
async def test_executor_can_handle():
    executor = TestExecutor()
    workspace = WorkspaceContext(workspace_root=Path("/test"))

    request = ExecutionRequest(
        task="test task",
        task_type="test",
        workspace_context=workspace
    )

    assert executor.can_handle(request) is True

@pytest.mark.asyncio
async def test_executor_execute():
    executor = TestExecutor()
    workspace = WorkspaceContext(workspace_root=Path("/test"))

    request = ExecutionRequest(
        task="run test",
        task_type="test",
        workspace_context=workspace
    )

    result = await executor.execute(request)
    assert result.success is True
    assert result.executor_type == "test"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/execution/test_protocol.py -v`
Expected: FAIL with "No module named 'orch.execution'"

**Step 3: Create executor protocol**

```python
# src/orch/execution/__init__.py
from .protocol import RemoteCodeExecutor

__all__ = ["RemoteCodeExecutor"]
```

```python
# src/orch/execution/protocol.py
from abc import ABC, abstractmethod
from orch.orchestration.models import ExecutionRequest, ExecutionResult


class RemoteCodeExecutor(ABC):
    """Base interface for code execution (hands vs brains separation)"""

    @abstractmethod
    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute code/task and return summarized result"""
        pass

    @abstractmethod
    def can_handle(self, request: ExecutionRequest) -> bool:
        """Check if this executor can handle the request"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify executor is available and working"""
        pass
```

**Step 4: Create directories and run tests**

```bash
mkdir -p src/orch/execution tests/unit/execution
touch tests/unit/execution/__init__.py
pytest tests/unit/execution/test_protocol.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/execution/ tests/unit/execution/
git commit -m "feat: add RemoteCodeExecutor protocol

- Abstract base for execution layer
- Defines execute(), can_handle(), health_check()
- Foundation for SubprocessExecutor and AgentDelegator"
```

---

### Task 4: Subprocess Executor (Simple Tasks)

**Files:**
- Create: `src/orch/execution/subprocess_executor.py`
- Test: `tests/unit/execution/test_subprocess_executor.py`

**Step 1: Write failing test for subprocess execution**

```python
# tests/unit/execution/test_subprocess_executor.py
import pytest
from pathlib import Path
from orch.execution.subprocess_executor import SubprocessExecutor
from orch.orchestration.models import ExecutionRequest, WorkspaceContext

@pytest.mark.asyncio
async def test_can_handle_test_tasks():
    executor = SubprocessExecutor()
    workspace = WorkspaceContext(workspace_root=Path.cwd())

    request = ExecutionRequest(
        task="run pytest",
        task_type="run_tests",
        workspace_context=workspace
    )

    assert executor.can_handle(request) is True

@pytest.mark.asyncio
async def test_cannot_handle_implementation():
    executor = SubprocessExecutor()
    workspace = WorkspaceContext(workspace_root=Path.cwd())

    request = ExecutionRequest(
        task="implement feature",
        task_type="implementation",
        workspace_context=workspace
    )

    assert executor.can_handle(request) is False

@pytest.mark.asyncio
async def test_health_check():
    executor = SubprocessExecutor()
    assert await executor.health_check() is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/execution/test_subprocess_executor.py -v`
Expected: FAIL with "No module named '...subprocess_executor'"

**Step 3: Implement SubprocessExecutor**

```python
# src/orch/execution/subprocess_executor.py
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
```

**Step 4: Run tests**

Run: `pytest tests/unit/execution/test_subprocess_executor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/execution/subprocess_executor.py tests/unit/execution/test_subprocess_executor.py
git commit -m "feat: add SubprocessExecutor for simple tasks

- Handles pytest, ruff, mypy via subprocess
- Summarizes output (context hygiene)
- Timeout and error handling"
```

---

### Task 5: Agent Delegator (Complex Tasks)

**Files:**
- Create: `src/orch/execution/agent_delegator.py`
- Test: `tests/unit/execution/test_agent_delegator.py`

**Step 1: Write failing test for agent delegation**

```python
# tests/unit/execution/test_agent_delegator.py
import pytest
from unittest.mock import Mock, AsyncMock
from pathlib import Path
from orch.execution.agent_delegator import AgentDelegator
from orch.orchestration.models import ExecutionRequest, WorkspaceContext
from orch.agents.protocol import ExecutionResult as AgentExecutionResult

@pytest.fixture
def mock_agent_registry():
    registry = Mock()
    agent = Mock()
    agent.name = "codex"
    agent.execute = AsyncMock(return_value=AgentExecutionResult(
        output="Code implemented",
        error=None,
        exit_code=0
    ))
    registry.get_agent.return_value = agent
    return registry

@pytest.mark.asyncio
async def test_can_handle_implementation(mock_agent_registry):
    delegator = AgentDelegator(mock_agent_registry)
    workspace = WorkspaceContext(workspace_root=Path.cwd())

    request = ExecutionRequest(
        task="implement feature",
        task_type="implementation",
        workspace_context=workspace
    )

    assert delegator.can_handle(request) is True

@pytest.mark.asyncio
async def test_execute_delegates_to_agent(mock_agent_registry):
    delegator = AgentDelegator(mock_agent_registry)
    workspace = WorkspaceContext(workspace_root=Path.cwd())

    request = ExecutionRequest(
        task="implement auth",
        task_type="implementation",
        workspace_context=workspace,
        suggested_agent="codex"
    )

    result = await delegator.execute(request)

    assert result.success is True
    assert result.executor_type == "agent_cli"
    assert result.agent_name == "codex"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/execution/test_agent_delegator.py -v`
Expected: FAIL with "No module named '...agent_delegator'"

**Step 3: Implement AgentDelegator**

```python
# src/orch/execution/agent_delegator.py
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
            success=not agent_result.is_error() if hasattr(agent_result, 'is_error') else agent_result.exit_code == 0,
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
            "status": "success" if (not hasattr(result, 'is_error') or not result.is_error()) else "failed",
        }

        # Add task-specific metadata
        if task_type == "implementation":
            summary["output_preview"] = getattr(result, 'output', '')[:200]

        return summary
```

**Step 4: Run tests**

Run: `pytest tests/unit/execution/test_agent_delegator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/orch/execution/agent_delegator.py tests/unit/execution/test_agent_delegator.py
git commit -m "feat: add AgentDelegator for complex tasks

- Delegates to AI CLI agents (codex, opencode, gemini)
- Builds rich prompts with context + suggestions
- Summarizes output for context hygiene"
```

---

### Task 6: Execution Router

**Files:**
- Create: `src/orch/execution/router.py`
- Test: `tests/unit/execution/test_router.py`

**Step 1: Write failing test for execution routing**

```python
# tests/unit/execution/test_router.py
import pytest
from unittest.mock import Mock, AsyncMock
from pathlib import Path
from orch.execution.router import ExecutionRouter
from orch.orchestration.models import ExecutionRequest, ExecutionResult, WorkspaceContext

@pytest.mark.asyncio
async def test_routes_to_subprocess_for_tests():
    router = ExecutionRouter(Mock())
    workspace = WorkspaceContext(workspace_root=Path.cwd())

    request = ExecutionRequest(
        task="run tests",
        task_type="run_tests",
        workspace_context=workspace
    )

    result = await router.execute(request)
    assert result.executor_type == "subprocess"

@pytest.mark.asyncio
async def test_routes_to_agent_for_implementation():
    mock_registry = Mock()
    mock_agent = Mock()
    mock_agent.name = "codex"
    mock_agent.execute = AsyncMock(return_value=Mock(exit_code=0, output="done"))
    mock_registry.get_agent.return_value = mock_agent
    mock_registry.list_agents.return_value = [mock_agent]

    router = ExecutionRouter(mock_registry)
    workspace = WorkspaceContext(workspace_root=Path.cwd())

    request = ExecutionRequest(
        task="implement feature",
        task_type="implementation",
        workspace_context=workspace
    )

    result = await router.execute(request)
    assert result.executor_type == "agent_cli"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/execution/test_router.py -v`
Expected: FAIL

**Step 3: Implement ExecutionRouter**

```python
# src/orch/execution/router.py
import asyncio
from orch.execution.protocol import RemoteCodeExecutor
from orch.execution.subprocess_executor import SubprocessExecutor
from orch.execution.agent_delegator import AgentDelegator
from orch.orchestration.models import ExecutionRequest, ExecutionResult
from orch.agents.registry import AgentRegistry


class ExecutionRouter:
    """Routes execution requests to appropriate executor"""

    def __init__(self, agent_registry: AgentRegistry):
        self.executors: list[RemoteCodeExecutor] = [
            SubprocessExecutor(),
            AgentDelegator(agent_registry)
        ]

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute request with appropriate executor"""
        # Find capable executor
        executor = self._select_executor(request)

        # Execute with timeout
        try:
            timeout = request.constraints.get("max_duration", 300)
            result = await asyncio.wait_for(
                executor.execute(request),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            return ExecutionResult(
                executor_type=executor.__class__.__name__,
                success=False,
                summary={"error": "Execution timeout"},
                error="Task exceeded maximum duration"
            )

    def _select_executor(self, request: ExecutionRequest) -> RemoteCodeExecutor:
        """Select executor based on task type"""
        # Try subprocess first (faster, cheaper)
        for executor in self.executors:
            if isinstance(executor, SubprocessExecutor) and executor.can_handle(request):
                return executor

        # Fall back to agent delegation
        for executor in self.executors:
            if isinstance(executor, AgentDelegator) and executor.can_handle(request):
                return executor

        raise ValueError(f"No executor can handle task type: {request.task_type}")
```

**Step 4: Run tests**

Run: `pytest tests/unit/execution/test_router.py -v`
Expected: PASS

**Step 5: Update execution __init__.py**

```python
# src/orch/execution/__init__.py
from .protocol import RemoteCodeExecutor
from .subprocess_executor import SubprocessExecutor
from .agent_delegator import AgentDelegator
from .router import ExecutionRouter

__all__ = [
    "RemoteCodeExecutor",
    "SubprocessExecutor",
    "AgentDelegator",
    "ExecutionRouter"
]
```

**Step 6: Commit**

```bash
git add src/orch/execution/router.py src/orch/execution/__init__.py tests/unit/execution/test_router.py
git commit -m "feat: add ExecutionRouter for hybrid execution

- Routes simple tasks to SubprocessExecutor
- Routes complex tasks to AgentDelegator
- Timeout handling and graceful degradation"
```

---

## Phase 1 continues with Tasks 7-15...

Due to length constraints, the remaining tasks follow the same pattern:

**Task 7:** Simple PlannerAgent (LLM-based planning)
**Task 8:** ExecutorAgent (creates ExecutionRequests)
**Task 9:** SecurityCritic and CorrectnessCritic
**Task 10:** CriticAggregator (veto hierarchy)
**Task 11:** Checkpoint system
**Task 12:** TeamOrchestrator workflow
**Task 13:** CLI commands (orch orchestrate run)
**Task 14:** Session management commands
**Task 15:** Basic failure logging

Each task follows the TDD pattern:
1. Write failing test
2. Run to verify failure
3. Implement minimal code
4. Run to verify pass
5. Commit with clear message

---

## Execution Strategy

Since this is a large implementation, I recommend using **orch itself** to delegate work:

```bash
# Use codex for implementing individual tasks
orch codex "Implement Task 7: Simple PlannerAgent following the test-first pattern in the plan"

# Use compare mode to validate approaches
orch compare "Review the PlannerAgent implementation for correctness"

# Use orchestrate (once built!) for complex integrations
orch orchestrate run "Integrate all critics with aggregator"
```

This creates a meta-circular implementation where we use orch to build the team-of-rivals feature!

---

## Post-Implementation Validation

After Phase 1 MVP:

1. **Integration test:** Create end-to-end test that runs full orchestration
2. **Manual test:** `orch orchestrate run "implement simple function"`
3. **Verify checkpoints:** Check `~/.config/orch/sessions/` for saved state
4. **Review trace:** `orch session trace <id>` shows full audit log

Then iterate to Phase 2 (complexity analyzer, weighted scoring, analytics).
