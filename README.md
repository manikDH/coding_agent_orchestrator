# Orch - Universal AI Agent Orchestrator

Claude Code as manager, AI CLIs as workers.

## Overview

Orch is a CLI tool that orchestrates multiple AI agent CLIs (Gemini, Codex, OpenCode, and more) with Claude Code acting as an intelligent manager. It supports:

- **Direct Agent Access**: Use any AI CLI with a unified interface
- **Smart Routing**: Automatically pick the best agent based on task type
- **Competition Mode**: Compare responses from multiple agents
- **Parallel Execution**: Run agents in tmux panes for visual monitoring
- **Extensible**: Easy to add new AI CLIs

## Installation

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

## Requirements

- Python 3.11+
- At least one AI CLI installed:
  - `claude` - Anthropic Claude Code CLI
  - `gemini` - Google Gemini CLI
  - `codex` - OpenAI Codex CLI
  - `opencode` - OpenCode CLI (free models)
- `tmux` (optional, for parallel execution)

## Quick Start

```bash
# Direct agent access
orch claude "refactor this function"
orch gemini "explain recursion"
orch codex "implement binary search"
orch opencode "write a utility function"  # Uses free models

# Smart routing (auto-picks best agent)
orch ask "what does this error mean?"      # Routes to Gemini
orch ask "fix this null pointer exception" # Routes to Codex

# With complexity-based model selection
orch ask --complexity high "design a distributed system"

# Compare agents side-by-side
orch compare "implement a sorting algorithm"

# Run competition in tmux
orch tmux compete "implement caching layer"
```

## Commands

```
orch ask [PROMPT]                 # Execute with smart routing
orch <agent> [PROMPT]             # Use specific agent (gemini, codex)
orch compare [PROMPT]             # Compare multiple agents
orch tmux compete [PROMPT]        # Competition in tmux panes
orch agent list                   # List available agents
orch agent info <name>            # Show agent details
orch config show                  # Show configuration
```

### Options

```
--complexity [low|medium|high]    # Select model tier based on task complexity
-m, --model TEXT                  # Override model selection
-s, --stream                      # Stream output in real-time
--json                            # Output in JSON format
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
code = ["codex", "opencode", "gemini"]
explain = ["gemini", "opencode", "codex"]
debug = ["codex", "opencode", "gemini"]

[agents.claude]
model = "sonnet"
model_tiers = { low = "haiku", medium = "sonnet", high = "opus" }

[agents.gemini]
model = "gemini-2.0-flash"
approval_mode = "auto_edit"

[agents.codex]
model = "gpt-5.2-codex"
sandbox = "workspace-write"

[agents.opencode]
model = "opencode/grok-code"
model_tiers = { low = "opencode/glm-4.7-free", medium = "opencode/grok-code", high = "opencode/minimax-m2.1-free" }
extra_args = { agent = "build" }
```

## Agent Capabilities & Limitations

Each agent has different capabilities when invoked via orch. Understanding these helps you choose the right agent for your task.

### Capability Matrix

| Capability | Claude | Codex | Gemini | OpenCode |
|------------|--------|-------|--------|----------|
| **File Creation** | ✅ Full | ✅ Full | ❌ Limited | ✅ Full |
| **File Editing** | ✅ Full | ✅ Full | ❌ Limited | ✅ Full |
| **Shell Commands** | ✅ Full | ✅ Full | ❌ Limited | ✅ Full |
| **Streaming Output** | ✅ | ✅ | ✅ | ❌ |
| **Session Persistence** | ❌ | ❌ | ✅ | ✅ |
| **Code Generation** | ✅ Excellent | ✅ Excellent | ✅ Good | ✅ Good |
| **Explanations** | ✅ Good | ✅ Good | ✅ Excellent | ✅ Good |
| **Cost** | Paid | Paid | Paid | Free |

### Detailed Limitations

#### Gemini CLI
- **No filesystem access**: Cannot create, edit, or delete files directly
- **No shell execution**: Cannot run shell commands like `write_file` or `run_shell_command`
- **Best for**: Explanations, analysis, research, answering questions
- **Workaround**: Use Gemini to generate code, then manually create files or pipe output

#### Codex CLI
- **Full filesystem access**: Can create and edit files in the workspace
- **Shell execution**: Can run commands within sandbox boundaries
- **Best for**: Implementation tasks, refactoring, file creation, debugging
- **Note**: Operates within sandbox permissions (`workspace-write` by default)

#### Claude CLI
- **Full filesystem access**: Can create, edit, and manage files
- **Shell execution**: Can run shell commands with appropriate permissions
- **Best for**: Complex coding tasks, architecture, refactoring, debugging
- **Note**: Use `--dangerously-skip-permissions` for unrestricted access (use with caution)

#### OpenCode CLI
- **Free models only**: Uses grok-code, glm-4.7-free, or minimax-m2.1-free
- **Full filesystem access**: Can create and edit files
- **Shell execution**: Can run shell commands
- **No streaming**: Uses batch execution mode
- **Best for**: Coding tasks when you want zero-cost AI assistance
- **Available models**:
  - `opencode/grok-code` (default) - Fast coding tasks
  - `opencode/glm-4.7-free` - General coding, lightweight tasks
  - `opencode/minimax-m2.1-free` - Analysis, complex reasoning

### Choosing the Right Agent

| Task Type | Recommended Agent | Reason |
|-----------|-------------------|--------|
| Implement a feature | `codex` or `claude` | Need file creation |
| Explain code | `gemini` | Strong at analysis |
| Debug an error | `codex` or `claude` | Need to edit files |
| Generate documentation | `gemini` | Good at writing |
| Refactor codebase | `claude` | Understands architecture |
| Quick code snippet | Any | All handle this well |
| Zero-cost coding | `opencode` | Free models only |

### Competition Mode Insights

When running `orch tmux compete`, keep in mind:
- Gemini will provide code output but cannot write files - you'll need to copy the output
- Codex, Claude, and OpenCode will directly create/modify files in your workspace
- OpenCode provides a free alternative for comparison without API costs
- Use competition mode to compare approaches, then cherry-pick the best implementation

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
