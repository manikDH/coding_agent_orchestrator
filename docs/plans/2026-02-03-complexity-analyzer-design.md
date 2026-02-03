# ComplexityAnalyzer Design (Phase 2)

**Date:** 2026-02-03
**Status:** Design Complete - Ready for Implementation
**Phase:** 2 - Enhanced Intelligence

## Overview

The ComplexityAnalyzer is the intelligence layer that automatically detects task complexity and routes orchestration to appropriate model tiers. This transforms orch from requiring manual complexity specification to intelligent auto-detection and model selection.

**Key Innovation:** LLM-based complexity detection with hybrid role-based + task-type model scaling, using agent-agnostic tier system that resolves to model + reasoning level (for codex) or best model (for claude/gemini).

## Problem Statement

Phase 1 MVP requires users to manually specify complexity:
```bash
orch orchestrate run --complexity complex "refactor auth"
```

This is problematic because:
1. Users don't know what counts as "complex" vs "standard"
2. Manual specification is error-prone (under/over-provision models)
3. No automatic model tier selection based on task characteristics
4. Wastes expensive models on simple tasks, cheap models on complex tasks

## Solution: Intelligent Complexity Detection

### High-Level Architecture

```
User Request: "refactor authentication to use OAuth2"
    ↓
ComplexityAnalyzer (LLM-based detection)
    ├─ Analyzes: prompt + workspace context
    ├─ Calls: Claude Haiku (fast, cheap)
    └─ Returns: ComplexityResult
        ├─ complexity_level: "complex"
        ├─ task_types: ["security_sensitive", "architectural"]
        ├─ reasoning: "Auth refactoring is security-critical..."
        ├─ confidence: 0.95
        └─ recommended_models: {
              "planner": "high",
              "executor": "highest",
              "security_critic": "highest",
              "correctness_critic": "highest"
            }
    ↓
Agent Adapters Resolve Tiers
    ├─ Codex: highest → gpt-5.2-codex --reasoning xhigh
    ├─ Claude: highest → claude-opus-4-5-20251101
    └─ Gemini: highest → gemini-2.0-flash-thinking
    ↓
TeamOrchestrator runs with selected models
```

## Component Design

### 1. ComplexityAnalyzer Class

**Location:** `src/orch/orchestration/complexity.py`

```python
from dataclasses import dataclass
from typing import Literal
import json

@dataclass
class ComplexityResult:
    """Result of complexity analysis."""
    complexity_level: Literal["simple", "standard", "complex"]
    task_types: list[str]  # ["security_sensitive", "architectural", ...]
    reasoning: str  # Why this classification
    confidence: float  # 0.0-1.0
    recommended_models: dict[str, str]  # role → tier mapping

    def to_dict(self) -> dict:
        """Serialize for logging/checkpoints."""
        return {
            "complexity_level": self.complexity_level,
            "task_types": self.task_types,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "recommended_models": self.recommended_models
        }


class ComplexityAnalyzer:
    """Analyzes task complexity using LLM and routes to appropriate models."""

    def __init__(self, llm_client: LLMClient, config: OrchConfig):
        self.llm_client = llm_client
        self.config = config

    async def analyze(
        self,
        user_prompt: str,
        workspace_context: WorkspaceContext
    ) -> ComplexityResult:
        """
        Analyzes task complexity using LLM.

        Process:
        1. Build context from workspace
        2. Call LLM (haiku) with structured prompt
        3. Parse JSON response
        4. Map to model tier recommendations
        5. Return ComplexityResult
        """
        # Build context
        context = self._build_context(workspace_context)

        # Call LLM
        prompt = self._build_detection_prompt(user_prompt, context)
        response = await self.llm_client.complete(
            prompt=prompt,
            model=self.config.orchestration.detection_model,
            max_tokens=500,
            temperature=0.0  # Deterministic
        )

        # Parse response
        detection = json.loads(response.content)

        # Map to model recommendations
        recommended_models = self._get_model_recommendations(
            detection["complexity_level"],
            detection["task_types"]
        )

        return ComplexityResult(
            complexity_level=detection["complexity_level"],
            task_types=detection["task_types"],
            reasoning=detection["reasoning"],
            confidence=detection["confidence"],
            recommended_models=recommended_models
        )

    def _build_context(self, workspace_context: WorkspaceContext) -> dict:
        """Build context dict for LLM prompt."""
        return {
            "file_count": len(workspace_context.relevant_files),
            "recent_files": workspace_context.recent_changes[:5],
            "project_type": workspace_context.project_type,
            "has_tests": workspace_context.has_tests,
        }

    def _build_detection_prompt(self, user_prompt: str, context: dict) -> str:
        """Build structured prompt for LLM classification."""
        return f"""Analyze this software development task and classify its complexity.

Task: {user_prompt}

Workspace Context:
- Files in workspace: {context['file_count']}
- Recent changes: {', '.join(context['recent_files']) if context['recent_files'] else 'none'}
- Project type: {context['project_type']}
- Has tests: {context['has_tests']}

Classify the task as:
- "simple": Single file, clear requirements, no edge cases (e.g., "add docstring", "fix typo")
- "standard": Multiple files or moderate complexity (e.g., "add validation", "implement feature")
- "complex": Architectural changes, security-sensitive, or high risk (e.g., "refactor auth", "migrate database")

Also identify task types (can be multiple):
- security_sensitive: Involves auth, crypto, tokens, permissions, secrets
- architectural: Refactoring, redesign, migration, major restructuring
- performance_critical: Optimization, scaling, caching, database performance
- data_sensitive: Database schema, migrations, data transformations
- testing_required: Needs comprehensive test coverage

Return JSON only:
{{
  "complexity_level": "simple|standard|complex",
  "task_types": ["type1", "type2"],
  "reasoning": "Brief explanation of classification",
  "confidence": 0.0-1.0
}}"""

    def _get_model_recommendations(
        self,
        complexity_level: str,
        task_types: list[str]
    ) -> dict[str, str]:
        """
        Maps complexity and task types to TIER LEVELS for each role.

        Tier levels: "low" | "medium" | "high" | "highest"

        Strategy:
        - Planner: Scales with base complexity (planning quality matters!)
        - Executor: Base tier + task-type boost
        - Critics: Scale with complexity and task type

        Returns agent-agnostic tier levels that each adapter resolves:
        - Codex: tier → model + reasoning level
        - Claude: tier → best model for tier
        - Gemini: tier → best model for tier
        """
        recommendations = {}

        # Planner: Scales with base complexity
        recommendations["planner"] = {
            "simple": "low",
            "standard": "medium",
            "complex": "high"  # Complex planning needs quality
        }[complexity_level]

        # Executor: Base tier + task-type boost
        base_executor_tier = {
            "simple": "low",
            "standard": "medium",
            "complex": "high"
        }[complexity_level]

        # Task-type scaling for executor
        if "security_sensitive" in task_types or "architectural" in task_types:
            executor_tier = "highest"  # Critical tasks get best model
        elif "performance_critical" in task_types or "data_sensitive" in task_types:
            executor_tier = "high"
        else:
            executor_tier = base_executor_tier

        recommendations["executor"] = executor_tier

        # Critics: Scale with complexity and task type
        if complexity_level == "complex" or "security_sensitive" in task_types:
            recommendations["security_critic"] = "highest"
            recommendations["correctness_critic"] = "highest"
        elif complexity_level == "standard":
            recommendations["security_critic"] = "high"
            recommendations["correctness_critic"] = "high"
        else:  # simple
            recommendations["security_critic"] = "medium"
            recommendations["correctness_critic"] = "medium"

        return recommendations
```

### 2. LLM Client Interface

**Location:** `src/orch/llm/client.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    tokens_used: int

class LLMClient(ABC):
    """Abstract LLM client for complexity detection."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.0
    ) -> LLMResponse:
        """Send prompt to LLM and get response."""
        pass

class AnthropicLLMClient(LLMClient):
    """Anthropic API client for complexity detection."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 500,
        temperature: float = 0.0
    ) -> LLMResponse:
        """Call Anthropic API."""
        response = await self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        return LLMResponse(
            content=response.content[0].text,
            model=model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens
        )
```

### 3. Agent Adapter Tier Resolution

**Codex Adapter:**

```python
# In src/orch/agents/codex.py

class CodexAgent(BaseAgent):

    def get_model_for_tier(self, tier: str) -> dict:
        """
        Resolves tier to model config.
        Returns dict with model + reasoning level.
        """
        tier_config = self.config.tiers.get(tier, self.config.tiers["medium"])

        # Codex tiers are dicts: {model: "...", reasoning: "..."}
        if isinstance(tier_config, dict):
            return tier_config
        else:
            # Fallback to simple model string
            return {"model": tier_config}

    def build_command(self, prompt: str, model_config: dict = None, **kwargs):
        """Build codex command with tier-specific model + reasoning."""
        model_config = model_config or self.get_model_for_tier("medium")

        cmd = ["codex", "exec"]
        cmd.extend(["-m", model_config["model"]])

        if "reasoning" in model_config:
            cmd.extend(["--reasoning", model_config["reasoning"]])

        # ... rest of command building
        return cmd
```

**Claude Adapter:**

```python
# In src/orch/agents/claude.py

class ClaudeAgent(BaseAgent):

    def get_model_for_tier(self, tier: str) -> dict:
        """
        Resolves tier to model.
        Returns dict with model only (Claude doesn't have reasoning levels).
        """
        tier_config = self.config.tiers.get(tier, self.config.tiers["medium"])

        if isinstance(tier_config, str):
            return {"model": tier_config}
        else:
            return tier_config
```

### 4. TeamOrchestrator Integration

**Location:** `src/orch/orchestration/team.py`

```python
class TeamOrchestrator:

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.session: OrchestrationSession | None = None
        self.llm_client = self._create_llm_client()

    def _create_llm_client(self) -> LLMClient:
        """Create LLM client for complexity detection."""
        config = ConfigManager.get_config()

        # Use Anthropic for detection (haiku is fast/cheap)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return AnthropicLLMClient(api_key)

    async def orchestrate(
        self,
        user_prompt: str,
        options: dict | None = None
    ) -> OrchestrationResult:
        options = options or {}

        # Initialize session
        session = self._create_session(user_prompt, options)
        self.session = session

        # Create checkpoint manager
        checkpoint_mgr = CheckpointManager(session.checkpoint_dir)

        try:
            # Checkpoint: init
            await self._checkpoint(checkpoint_mgr, "init")

            # === Auto-detect complexity ===
            if not options.get("complexity") or options.get("complexity") == "auto":
                session.state = "analyzing_complexity"

                analyzer = ComplexityAnalyzer(
                    self.llm_client,
                    ConfigManager.get_config()
                )

                complexity_result = await analyzer.analyze(
                    user_prompt,
                    session.workspace_context
                )

                # Store in session for traceability
                session.complexity_result = complexity_result
                session.complexity_level = complexity_result.complexity_level
                session.metadata["task_types"] = complexity_result.task_types
                session.metadata["recommended_models"] = complexity_result.recommended_models

                # Checkpoint the detection
                await self._checkpoint(
                    checkpoint_mgr,
                    "complexity_detected",
                    complexity=complexity_result.to_dict()
                )
            else:
                # Manual complexity specified
                session.complexity_level = options["complexity"]
                session.metadata["recommended_models"] = self._default_model_tiers(
                    options["complexity"]
                )

            # === Create agents with tier-specific models ===
            planner = PlannerAgent(
                model_tier=session.metadata["recommended_models"]["planner"]
            )

            executor = ExecutorAgent(
                execution_router=self.execution_router,
                model_tier=session.metadata["recommended_models"]["executor"]
            )

            security_critic = SecurityCritic(
                model_tier=session.metadata["recommended_models"]["security_critic"]
            )

            correctness_critic = CorrectnessCritic(
                model_tier=session.metadata["recommended_models"]["correctness_critic"]
            )

            # === Proceed with orchestration workflow ===
            # Phase 1: Planning
            session.state = "planning"
            # ... rest of orchestration

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
```

### 5. Agent Role Updates

**Location:** `src/orch/agents/roles/planner.py`, etc.

```python
class PlannerAgent(RoleAgent):
    """Create structured implementation plans."""

    def __init__(self, model_tier: str = "medium"):
        self.model_tier = model_tier

    async def propose(self, task_state: TaskState) -> AgentMessage:
        """Generate plan using tier-specific model."""
        # Pass model_tier to execution layer
        # Execution layer resolves tier to actual model
        ...
```

## Configuration Schema

**Location:** `~/.config/orch/config.toml`

```toml
[orchestration]
auto_detect = true  # Enable automatic complexity detection
default_complexity = "auto"  # "auto" | "simple" | "standard" | "complex"
detection_model = "claude-3-haiku-20240307"  # Fast, cheap model for detection

[orchestration.complexity]
# Confidence threshold - if LLM confidence < threshold, ask user
confidence_threshold = 0.7
# Cache detection results for similar prompts (optional Phase 3)
cache_enabled = false
cache_ttl_seconds = 3600

# ===== Agent Tier Configurations =====

[agents.codex.tiers]
low = { model = "gpt-4o-mini", reasoning = "low" }
medium = { model = "gpt-5.2-codex", reasoning = "medium" }
high = { model = "gpt-5.2-codex", reasoning = "high" }
highest = { model = "gpt-5.2-codex", reasoning = "xhigh" }

[agents.claude.tiers]
low = "claude-3-5-haiku-20241022"
medium = "claude-3-5-sonnet-20241022"
high = "claude-3-5-sonnet-20241022"
highest = "claude-opus-4-5-20251101"

[agents.gemini.tiers]
low = "gemini-1.5-flash"
medium = "gemini-2.0-flash"
high = "gemini-2.0-flash-thinking"
highest = "gemini-2.0-flash-thinking"

[agents.opencode.tiers]
low = "opencode/glm-4.7-free"
medium = "opencode/grok-code"
high = "opencode/grok-code"
highest = "opencode/minimax-m2.1-free"
```

## CLI Updates

**Location:** `src/orch/cli/main.py`

### Updated `orchestrate run` Command

```python
@orchestrate.command("run")
@click.argument("prompt", nargs=-1, required=True)
@click.option(
    "--complexity",
    type=click.Choice(["auto", "simple", "standard", "complex"]),
    default="auto",
    help="Task complexity (auto-detects if not specified)"
)
@click.option("--no-auto-detect", is_flag=True, help="Disable auto-detection, use default")
@click.option("--show-detection", is_flag=True, help="Show complexity detection reasoning")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def orchestrate_run(
    prompt: tuple[str, ...],
    complexity: str,
    no_auto_detect: bool,
    show_detection: bool,
    output_json: bool
) -> None:
    """Run team-of-rivals orchestration with auto-detection."""
    prompt_text = " ".join(prompt)

    async def _run():
        orchestrator = TeamOrchestrator()

        options = {}
        if complexity != "auto" or no_auto_detect:
            options["complexity"] = complexity if complexity != "auto" else "standard"

        result = await orchestrator.orchestrate(prompt_text, options)

        formatter = get_formatter()

        # Show detection reasoning if requested
        if show_detection and hasattr(result, 'complexity_result'):
            formatter.console.print("\n[bold cyan]Complexity Detection:[/bold cyan]")
            formatter.console.print(f"  Level: [yellow]{result.complexity_result.complexity_level}[/yellow]")
            formatter.console.print(f"  Task types: {', '.join(result.complexity_result.task_types)}")
            formatter.console.print(f"  Reasoning: {result.complexity_result.reasoning}")
            formatter.console.print(f"  Confidence: {result.complexity_result.confidence:.2f}")
            formatter.console.print(f"\n[bold cyan]Model Selection:[/bold cyan]")
            for role, tier in result.complexity_result.recommended_models.items():
                formatter.console.print(f"  {role}: [green]{tier}[/green]")
            formatter.console.print()

        # Standard output
        if output_json:
            import json
            output = {
                "session_id": result.session_id,
                "success": result.success,
                "artifact": result.artifact,
                "error": result.error
            }
            if hasattr(result, 'complexity_result'):
                output["complexity"] = result.complexity_result.to_dict()
            formatter.console.print_json(json.dumps(output))
        else:
            if result.success:
                formatter.print_success(f"Orchestration completed successfully!")
                formatter.console.print(f"\nSession ID: {result.session_id}")

                # Show complexity if auto-detected
                if hasattr(result, 'complexity_result') and not show_detection:
                    formatter.console.print(
                        f"Complexity: [yellow]{result.complexity_result.complexity_level}[/yellow] "
                        f"(use --show-detection for details)"
                    )

                formatter.console.print(f"\nStatus: {result.artifact.get('status')}")

                if result.metrics:
                    formatter.console.print(f"\nMetrics:")
                    formatter.console.print(f"  Executions: {result.metrics.executions_count}")
                    formatter.console.print(f"  Critique rounds: {result.metrics.critique_rounds}")
            else:
                formatter.print_error(f"Orchestration failed: {result.error}")
                formatter.console.print(f"\nSession ID: {result.session_id}")

    asyncio.run(_run())
```

### New Analytics Commands

```python
@cli.group()
def analytics() -> None:
    """View orchestration analytics and insights."""
    pass


@analytics.command("complexity")
@click.option("--limit", default=20, help="Number of recent detections to show")
def analytics_complexity(limit: int) -> None:
    """Show complexity detection patterns and accuracy."""
    from orch.config.schema import get_sessions_dir
    from orch.orchestration.checkpoint import CheckpointManager

    formatter = get_formatter()
    sessions_dir = get_sessions_dir()

    if not sessions_dir.exists():
        formatter.print_warning("No sessions found")
        return

    # Collect complexity detections from sessions
    detections = []
    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        checkpoint_mgr = CheckpointManager(session_dir)
        complexity_checkpoint = checkpoint_mgr.load_checkpoint("complexity_detected")

        if complexity_checkpoint:
            detections.append(complexity_checkpoint.data.get("complexity"))

    if not detections:
        formatter.print_warning("No complexity detections found")
        return

    # Analyze patterns
    complexity_counts = {"simple": 0, "standard": 0, "complex": 0}
    task_type_counts = {}
    avg_confidence = 0

    for detection in detections[-limit:]:
        if not detection:
            continue
        complexity_counts[detection["complexity_level"]] += 1
        avg_confidence += detection["confidence"]

        for task_type in detection.get("task_types", []):
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

    avg_confidence /= len(detections[-limit:])

    # Display results
    formatter.console.print(f"\n[bold]Complexity Detection Analytics[/bold] (last {limit}):\n")

    formatter.console.print("[bold cyan]Distribution:[/bold cyan]")
    for level, count in complexity_counts.items():
        pct = (count / sum(complexity_counts.values())) * 100
        formatter.console.print(f"  {level}: {count} ({pct:.1f}%)")

    formatter.console.print(f"\n[bold cyan]Average Confidence:[/bold cyan] {avg_confidence:.2f}")

    if task_type_counts:
        formatter.console.print(f"\n[bold cyan]Common Task Types:[/bold cyan]")
        for task_type, count in sorted(task_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            formatter.console.print(f"  {task_type}: {count}")


@analytics.command("models")
def analytics_models() -> None:
    """Show model tier usage and patterns."""
    from orch.config.schema import get_sessions_dir
    from orch.orchestration.checkpoint import CheckpointManager

    formatter = get_formatter()
    sessions_dir = get_sessions_dir()

    if not sessions_dir.exists():
        formatter.print_warning("No sessions found")
        return

    # Collect model tier usage
    tier_usage = {"low": 0, "medium": 0, "high": 0, "highest": 0}
    role_tiers = {}

    for session_dir in sessions_dir.iterdir():
        if not session_dir.is_dir():
            continue

        checkpoint_mgr = CheckpointManager(session_dir)
        complexity_checkpoint = checkpoint_mgr.load_checkpoint("complexity_detected")

        if complexity_checkpoint:
            models = complexity_checkpoint.data.get("complexity", {}).get("recommended_models", {})
            for role, tier in models.items():
                tier_usage[tier] = tier_usage.get(tier, 0) + 1

                if role not in role_tiers:
                    role_tiers[role] = {"low": 0, "medium": 0, "high": 0, "highest": 0}
                role_tiers[role][tier] = role_tiers[role].get(tier, 0) + 1

    # Display results
    formatter.console.print(f"\n[bold]Model Tier Usage Analytics[/bold]:\n")

    formatter.console.print("[bold cyan]Overall Tier Distribution:[/bold cyan]")
    total = sum(tier_usage.values())
    for tier, count in tier_usage.items():
        if total > 0:
            pct = (count / total) * 100
            formatter.console.print(f"  {tier}: {count} ({pct:.1f}%)")

    if role_tiers:
        formatter.console.print(f"\n[bold cyan]Tier Usage by Role:[/bold cyan]")
        for role, tiers in role_tiers.items():
            formatter.console.print(f"\n  {role}:")
            role_total = sum(tiers.values())
            for tier, count in tiers.items():
                if count > 0:
                    pct = (count / role_total) * 100
                    formatter.console.print(f"    {tier}: {count} ({pct:.1f}%)")
```

## Usage Examples

### Basic Auto-Detection

```bash
# Auto-detect and run (default behavior)
orch orchestrate run "refactor authentication system"

# Output:
# Analyzing complexity...
# Complexity: complex (use --show-detection for details)
#
# [Orchestration proceeds with:
#   planner: high tier
#   executor: highest tier (security_sensitive + architectural)
#   critics: highest tier]
```

### Show Detection Reasoning

```bash
orch orchestrate run --show-detection "add validation to login form"

# Output:
# Complexity Detection:
#   Level: standard
#   Task types: security_sensitive, testing_required
#   Reasoning: Login form validation involves security (input sanitization) and requires tests
#   Confidence: 0.92
#
# Model Selection:
#   planner: medium
#   executor: highest (security_sensitive boost)
#   security_critic: highest
#   correctness_critic: highest
```

### Override Auto-Detection

```bash
# Force simple complexity
orch orchestrate run --complexity simple "fix typo in README"

# Disable auto-detection entirely
orch orchestrate run --no-auto-detect "implement feature"
```

### Analytics Commands

```bash
# View complexity detection patterns
orch analytics complexity
# Shows: distribution, avg confidence, common task types

# View model tier usage
orch analytics models
# Shows: overall tier distribution, tier usage by role
```

## Data Flow

```
1. User runs: orch orchestrate run "refactor auth to OAuth2"
   ↓
2. TeamOrchestrator checks if complexity specified
   - If not specified or "auto" → proceed to detection
   ↓
3. ComplexityAnalyzer.analyze()
   - Build context: files, recent changes, project type
   - Build LLM prompt with structured format
   - Call Claude Haiku (fast, cheap)
   - Parse JSON response
   ↓
4. LLM Response:
   {
     "complexity_level": "complex",
     "task_types": ["security_sensitive", "architectural"],
     "reasoning": "OAuth2 refactoring is security-critical and architectural",
     "confidence": 0.95
   }
   ↓
5. Map to tier recommendations:
   _get_model_recommendations("complex", ["security_sensitive", "architectural"])
   →
   {
     "planner": "high",           # Complex needs good planning
     "executor": "highest",        # Security + architectural → xhigh
     "security_critic": "highest", # Complex + security → xhigh
     "correctness_critic": "highest"
   }
   ↓
6. Store in session:
   - session.complexity_result = ComplexityResult(...)
   - session.metadata["recommended_models"] = {...}
   ↓
7. Checkpoint: complexity_detected
   ↓
8. Create agents with tiers:
   - PlannerAgent(model_tier="high")
   - ExecutorAgent(model_tier="highest")
   - SecurityCritic(model_tier="highest")
   ↓
9. Agent adapters resolve tiers:
   - Codex: "highest" → {model: "gpt-5.2-codex", reasoning: "xhigh"}
   - Claude: "highest" → {model: "claude-opus-4-5-20251101"}
   ↓
10. Orchestration proceeds with selected models
```

## Testing Strategy

### Unit Tests

```python
# tests/unit/orchestration/test_complexity_analyzer.py

async def test_simple_task_detection():
    """Test detection of simple task."""
    analyzer = ComplexityAnalyzer(mock_llm_client, config)

    result = await analyzer.analyze(
        "fix typo in README",
        workspace_context
    )

    assert result.complexity_level == "simple"
    assert result.recommended_models["planner"] == "low"
    assert result.recommended_models["executor"] == "low"


async def test_complex_security_task():
    """Test detection of complex security-sensitive task."""
    result = await analyzer.analyze(
        "refactor authentication to use OAuth2",
        workspace_context
    )

    assert result.complexity_level == "complex"
    assert "security_sensitive" in result.task_types
    assert "architectural" in result.task_types
    assert result.recommended_models["executor"] == "highest"
    assert result.recommended_models["security_critic"] == "highest"


def test_model_tier_mapping():
    """Test tier recommendation logic."""
    analyzer = ComplexityAnalyzer(mock_llm_client, config)

    # Complex + security_sensitive → highest for executor
    models = analyzer._get_model_recommendations(
        "complex",
        ["security_sensitive"]
    )

    assert models["executor"] == "highest"
    assert models["security_critic"] == "highest"

    # Standard + performance_critical → high for executor
    models = analyzer._get_model_recommendations(
        "standard",
        ["performance_critical"]
    )

    assert models["executor"] == "high"
```

### Integration Tests

```python
# tests/integration/test_complexity_integration.py

async def test_end_to_end_complexity_detection():
    """Test full orchestration with auto-detection."""
    orchestrator = TeamOrchestrator()

    result = await orchestrator.orchestrate(
        "implement rate limiting for API",
        options={"complexity": "auto"}
    )

    assert result.success
    assert hasattr(result, 'complexity_result')
    assert result.complexity_result.complexity_level in ["standard", "complex"]
    assert "performance_critical" in result.complexity_result.task_types
```

## Success Metrics

1. **Detection Accuracy**: 90%+ correct complexity classification
2. **Model Efficiency**: 30%+ cost reduction vs always using highest tier
3. **User Experience**: Users prefer auto-detection vs manual specification
4. **Response Time**: <500ms for complexity detection
5. **Confidence**: 85%+ average confidence score

## Future Enhancements (Phase 3)

1. **Detection Cache**: Cache similar prompts to avoid redundant LLM calls
2. **Learning Loop**: Learn from failures to improve detection rules
3. **Hybrid Detection**: Add rule-based fast-path for 80% of cases
4. **Custom Task Types**: User-defined task types with custom tier mappings
5. **Multi-Model Detection**: Ensemble of LLMs for higher accuracy

## Implementation Checklist

- [ ] Create `LLMClient` interface and `AnthropicLLMClient` implementation
- [ ] Implement `ComplexityAnalyzer` class with detection logic
- [ ] Add tier configuration to `OrchConfig` schema
- [ ] Update agent adapters (`CodexAgent`, `ClaudeAgent`) with `get_model_for_tier()`
- [ ] Integrate `ComplexityAnalyzer` into `TeamOrchestrator`
- [ ] Update role agents to accept `model_tier` parameter
- [ ] Update CLI with `--show-detection` and `--no-auto-detect` options
- [ ] Add `analytics complexity` command
- [ ] Add `analytics models` command
- [ ] Write unit tests for `ComplexityAnalyzer`
- [ ] Write integration tests for end-to-end detection
- [ ] Update documentation and examples

## Conclusion

The ComplexityAnalyzer transforms orch into an intelligent orchestration system that automatically detects task complexity, identifies task types, and routes to appropriate model tiers. This provides:

- **Better Quality**: Complex tasks get powerful models automatically
- **Cost Efficiency**: Simple tasks use cheaper models
- **User Experience**: No manual complexity specification needed
- **Transparency**: Clear reasoning for model selection
- **Flexibility**: Agent-agnostic tier system supports any coding agent

Ready for implementation and review! 🚀
