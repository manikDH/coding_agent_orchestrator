# Orch - Universal AI Agent Orchestrator

Claude Code as manager, AI CLIs as workers.

## Overview

Orch is a CLI tool that orchestrates multiple AI agent CLIs (Gemini, Codex, and more) with Claude Code acting as an intelligent manager. It supports:

- **Direct Agent Access**: Use any AI CLI with a unified interface
- **Smart Routing**: Automatically pick the best agent based on task type
- **Competition Mode**: Compare responses from multiple agents
- **Parallel Execution**: Run agents in tmux panes for visual monitoring
- **Extensible**: Easy to add new AI CLIs

## Installation

```bash
# Clone and install with pipx
pipx install ~/Projects/orch

# Or install in development mode
cd ~/Projects/orch
pip install -e .
```

## Requirements

- Python 3.11+
- At least one AI CLI installed:
  - `gemini` - Google Gemini CLI
  - `codex` - OpenAI Codex CLI
- `tmux` (optional, for parallel execution)

## Quick Start

```bash
# Direct agent access
orch gemini "explain recursion"
orch codex "implement binary search"

# Smart routing (auto-picks best agent)
orch "what does this error mean?"      # Routes to Gemini
orch "fix this null pointer exception" # Routes to Codex

# Compare agents
orch compare "implement a sorting algorithm"

# Run competition in tmux
orch tmux compete "implement caching layer"
```

## Commands

```
orch [PROMPT]                     # Execute with smart routing
orch <agent> [PROMPT]             # Use specific agent
orch compare [PROMPT]             # Compare multiple agents
orch tmux compete [PROMPT]        # Competition in tmux panes
orch agent list                   # List available agents
orch agent info <name>            # Show agent details
orch config show                  # Show configuration
```

## Configuration

Configuration file: `~/.config/orch/config.toml`

```toml
[global]
default_agent = "auto"    # "auto" for smart routing
verbose = false

[routing]
enabled = true

[routing.rules]
code = ["codex", "gemini"]
explain = ["gemini", "codex"]
debug = ["codex", "gemini"]

[agents.gemini]
model = "gemini-2.0-flash"
approval_mode = "auto_edit"

[agents.codex]
model = "gpt-5.2-codex"
sandbox = "workspace-write"
```

## Adding New Agents

Create a Python file in `~/.config/orch/plugins/`:

```python
from orch.agents.base import BaseAgent
from orch.agents.protocol import AgentCapabilities

class MyAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "myagent"

    @property
    def display_name(self) -> str:
        return "My Custom Agent"

    @property
    def cli_name(self) -> str:
        return "myagent"

    def get_capabilities(self) -> AgentCapabilities:
        return AgentCapabilities(
            supports_streaming=True,
            task_strengths=["general"],
        )

    def build_command(self, prompt, **kwargs) -> list[str]:
        return [self.executable, prompt]

    def parse_output(self, stdout, stderr, return_code):
        # Parse and return ExecutionResult
        ...
```

## License

MIT
