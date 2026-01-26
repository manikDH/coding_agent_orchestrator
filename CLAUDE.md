# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Orch is a CLI tool that orchestrates multiple AI agent CLIs (Gemini, Codex, Claude, OpenCode) with smart routing, competition mode, and parallel tmux execution. It acts as a unified interface where Claude Code serves as an intelligent manager and AI CLIs serve as workers.

## Installation

### For Users

```bash
# Install from PyPI (when published)
pip install orch

# Or install directly from GitHub
pip install git+https://github.com/manikDH/coding_agent_orchestrator.git

# Verify installation
orch --help
```

### For Developers

```bash
# Clone the repository
git clone https://github.com/manikDH/coding_agent_orchestrator.git
cd coding_agent_orchestrator

# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

## Development Commands

```bash
# Run the CLI
orch --help
python -m orch --help

# Run all tests
pytest

# Run single test file
pytest tests/unit/test_router.py

# Run test with specific name pattern
pytest -k "test_routing"

# Type checking
mypy src/orch

# Linting
ruff check src/orch

# Format check
ruff format --check src/orch
```

## Architecture

### Core Layers

```
src/orch/
├── cli/           # Click-based CLI entry point
│   ├── main.py    # Commands: ask, compare, agent, config, tmux
│   └── commands/  # Subcommand modules
├── agents/        # Agent adapters (protocol + implementations)
│   ├── protocol.py   # AgentAdapter ABC + AgentCapabilities + ExecutionResult
│   ├── base.py       # BaseAgent with common async subprocess logic
│   ├── registry.py   # AgentRegistry - discovers built-in, entry point, and plugin agents
│   ├── gemini.py     # GeminiAgent
│   ├── codex.py      # CodexAgent
│   ├── claude.py     # ClaudeAgent
│   └── opencode.py   # OpenCodeAgent (free models only)
├── orchestration/ # Smart routing logic
│   └── router.py     # Router + StrengthBasedRouter for task classification
├── config/        # Pydantic configuration schema
│   ├── schema.py     # OrchConfig, AgentConfig, RoutingConfig, etc.
│   └── manager.py    # ConfigManager for loading/saving TOML config
├── tmux/          # tmux integration for parallel execution
│   └── manager.py    # TmuxManager using libtmux
└── output/        # Rich-based output formatting
    └── formatter.py
```

### Key Design Patterns

**Agent Adapter Protocol**: All agents implement `AgentAdapter` (in `protocol.py`). The abstract base provides:
- `build_command()` - construct CLI arguments
- `parse_output()` - normalize stdout/stderr into `ExecutionResult`
- `execute()` / `stream_execute()` - async subprocess execution

**Agent Registry**: Singleton pattern that discovers agents from three sources:
1. Built-in agents (gemini, codex, claude, opencode)
2. Entry points (`orch.agents` group in pyproject.toml)
3. User plugins (`~/.config/orch/plugins/*.py`)

**Smart Routing**: The `Router` class in `orchestration/router.py` classifies prompts by keyword matching and routes to the best available agent based on configured preferences.

**Dynamic CLI Commands**: `AliasedGroup` in `cli/main.py` dynamically creates Click commands for each registered agent, so `orch gemini "..."` works without explicit command definitions.

### Configuration

Config lives at `~/.config/orch/config.toml`. Schema defined in `config/schema.py`:
- `OrchConfig` is the root model containing:
  - `global_` (aliased as "global") - default agent, output format
  - `routing` - keyword rules and agent preferences per task type
  - `competition` - default agents for compare mode
  - `tmux` - session name, layout
  - `agents` - per-agent model settings, tiers, approval modes

### Adding New Agents

Create an agent class inheriting from `BaseAgent`:
1. Implement `name`, `display_name`, `cli_name` properties
2. Implement `get_capabilities()` returning `AgentCapabilities`
3. Implement `build_command()` to construct CLI args
4. Implement `parse_output()` to normalize results

Register via:
- Entry point in pyproject.toml: `[project.entry-points."orch.agents"]`
- Or drop a `.py` file in `~/.config/orch/plugins/`

### Test Structure

Tests use pytest with pytest-asyncio. The `conftest.py` fixture `reset_registry` clears the singleton registry before each test. Mock fixtures like `mock_gemini_available` patch `shutil.which` to simulate CLI availability.

## Agent Capability Differences

| Agent | File Creation | Shell Commands | Sessions | Streaming |
|-------|--------------|----------------|----------|-----------|
| Claude | Full | Full | No | Yes |
| Codex | Full | Sandboxed | No | Yes |
| Gemini | Limited | Limited | Yes | Yes |
| OpenCode | Full | Full | Yes | No |

- **Gemini**: Best for explanations/analysis
- **Claude/Codex**: Best for implementation tasks requiring file/shell access
- **OpenCode**: Good for coding tasks using free models

## OpenCode Agent

OpenCode is integrated with **free models only** to provide zero-cost AI assistance.

### Available Free Models

| Model | Description | Best For |
|-------|-------------|----------|
| `opencode/grok-code` | Grok Fast Code (default) | Fast coding tasks, implementation |
| `opencode/glm-4.7-free` | GLM 4.7 free tier | General coding, lightweight tasks |
| `opencode/minimax-m2.1-free` | Minimax M2.1 free tier | Analysis, complex reasoning |

### OpenCode CLI Options

The OpenCode agent uses `opencode run` for non-interactive execution:

```bash
# Basic usage via orch
orch opencode "implement a binary search function"

# With specific model
orch opencode -m opencode/minimax-m2.1-free "analyze this algorithm"
```

### OpenCode Agents (Internal)

OpenCode has built-in agents for different task types:

| Agent | Purpose | Permissions |
|-------|---------|-------------|
| `build` | Implementation tasks (default) | Full read/write/execute |
| `plan` | Planning and architecture | Read + plan file editing |
| `explore` | Codebase exploration | Read-only + search tools |
| `general` | General purpose | Full tools except todo |

Configure the agent in `~/.config/orch/config.toml`:

```toml
[agents.opencode]
model = "opencode/grok-code"
extra_args = { agent = "build" }
```

### Model Tiers for OpenCode

| Complexity | Model |
|------------|-------|
| low | `opencode/glm-4.7-free` |
| medium | `opencode/grok-code` |
| high | `opencode/minimax-m2.1-free` |

### Task Strengths

OpenCode excels at:
- `code` - Implementation and coding tasks
- `implementation` - Building features
- `exploration` - Codebase understanding
- `planning` - Architectural planning

### Usage Examples

```bash
# Use OpenCode directly
orch opencode "refactor this function to use async/await"

# Smart routing may select OpenCode for code tasks
orch ask "implement a caching layer"

# Compare OpenCode with other agents
orch compare -a opencode -a gemini "explain this algorithm"

# Use explore agent for codebase analysis
orch opencode --extra-args '{"agent": "explore"}' "find all API endpoints"
```

### Restrictions

- **Free models only**: The adapter validates that only free models are used
- Non-free models will raise a `ValueError` with available options
- No streaming support (uses batch execution)
