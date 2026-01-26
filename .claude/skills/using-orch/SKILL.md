---
name: using-orch
description: Use when delegating tasks to AI CLI agents (Gemini, Codex, OpenCode), comparing agent responses, running agents in parallel, or needing smart routing between multiple AI backends
---

# Using Orch - Universal AI Agent Orchestrator

## Skill Setup

### Installation

Copy this skill to your Claude Code skills directory:

```bash
# Create skills directory if it doesn't exist
mkdir -p ~/.claude/skills

# Copy from repo
cp -r .claude/skills/using-orch ~/.claude/skills/
```

Or symlink for automatic updates:

```bash
ln -s /path/to/coding_agent_orchestrator/.claude/skills/using-orch ~/.claude/skills/using-orch
```

### Usage

Invoke this skill in Claude Code by:

1. **Slash command**: Type `/using-orch` in your conversation
2. **Automatic**: Claude Code will detect when you're delegating to AI agents and load this skill

### When This Skill Activates

Claude Code will use this skill when you:
- Ask to delegate tasks to other AI agents
- Want to compare responses from multiple agents
- Need to run agents in parallel
- Ask about smart routing between AI backends
- Mention `orch`, `gemini`, `codex`, or `opencode` commands

## Overview

`orch` orchestrates multiple AI CLI agents with Claude Code as manager. Delegate work, compare implementations, run agents in parallel via tmux.

## When to Use

- Delegate coding tasks to Gemini, Codex, or OpenCode
- Compare implementations from multiple agents
- Run agents in parallel for visual monitoring
- Smart-route prompts to best agent by task type
- Use complexity-based model selection
- Use OpenCode for zero-cost AI assistance

## Quick Reference

| Command | Purpose |
|---------|---------|
| `orch ask "prompt"` | Smart routing to best agent |
| `orch gemini "prompt"` | Direct Gemini access |
| `orch codex "prompt"` | Direct Codex access |
| `orch opencode "prompt"` | Direct OpenCode access (free) |
| `orch compare "prompt"` | Side-by-side comparison |
| `orch tmux compete "prompt"` | Parallel in tmux panes |
| `orch agent list` | List available agents |
| `orch agent info <name>` | Agent capabilities |
| `orch config show` | Show configuration |

## Key Options

```bash
# Complexity-based model selection
orch ask --complexity low "quick question"    # Fast, cheap model
orch ask --complexity high "design system"    # Most capable model

# Streaming output
orch gemini -s "explain recursion"

# JSON output
orch codex --json "implement sorting"

# Specify backend with smart routing
orch ask -b codex "fix this bug"

# Use free models with OpenCode
orch opencode "implement a utility function"
```

## Comparison Mode

```bash
# Compare default agents (gemini, codex)
orch compare "implement binary search"

# Compare specific agents
orch compare -a gemini -a codex "sorting algorithm"

# Include OpenCode for free comparison
orch compare -a opencode -a codex "caching layer"

# Run in tmux for visual monitoring
orch tmux compete "implement caching"
```

## Smart Routing

Routes based on task keywords:
- **code**: implement, write, function, class, refactor → prefers Codex, OpenCode
- **explain**: explain, what is, how does, describe → prefers Gemini
- **debug**: error, bug, fix, traceback, exception → prefers Codex, OpenCode

## Agent Capabilities

| Agent | File Access | Shell Commands | Streaming | Cost |
|-------|-------------|----------------|-----------|------|
| Claude | Full | Full | Yes | Paid |
| Codex | Full | Sandboxed | Yes | Paid |
| Gemini | Limited | Limited | Yes | Paid |
| OpenCode | Full | Full | No | Free |

## OpenCode Free Models

| Model | Best For |
|-------|----------|
| `opencode/grok-code` (default) | Fast coding tasks |
| `opencode/glm-4.7-free` | General coding, lightweight |
| `opencode/minimax-m2.1-free` | Analysis, complex reasoning |

## Configuration

Config file: `~/.config/orch/config.toml`

```toml
[agents.gemini]
model = "gemini-2.5-flash"
model_tiers = { low = "gemini-2.5-flash-lite", medium = "gemini-2.5-flash", high = "gemini-2.5-pro" }

[agents.codex]
model = "gpt-5.2-codex"
model_tiers = { low = "gpt-5.1-codex-mini", medium = "gpt-5.2-codex", high = "gpt-5.1-codex-max" }

[agents.opencode]
model = "opencode/grok-code"
model_tiers = { low = "opencode/glm-4.7-free", medium = "opencode/grok-code", high = "opencode/minimax-m2.1-free" }

[routing.rules]
code = ["codex", "opencode", "gemini"]
explain = ["gemini", "opencode", "codex"]
```

## Common Patterns

**Delegate and review:**
```bash
orch codex "implement user authentication"
# Review output, provide feedback
```

**Zero-cost coding:**
```bash
orch opencode "refactor this function"
# Free AI assistance with full file access
```

**Competition for best solution:**
```bash
orch compare "implement rate limiter"
# Compare both implementations, pick best
```

**Parallel monitoring:**
```bash
orch tmux compete "refactor auth module"
# Watch both agents work in tmux panes
```

## Adding Custom Agents

Create `~/.config/orch/plugins/myagent.py`:
```python
from orch.agents.base import BaseAgent

class MyAgent(BaseAgent):
    @property
    def name(self) -> str: return "myagent"
    @property
    def cli_name(self) -> str: return "myagent"
    # ... implement required methods
```
