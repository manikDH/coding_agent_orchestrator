"""Configuration schema using Pydantic."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ModelTiers(BaseModel):
    """Model tiers for complexity-based selection."""

    low: str | None = None  # Fast, cheap
    medium: str | None = None  # Balanced
    high: str | None = None  # Most capable


class AgentConfig(BaseModel):
    """Configuration for a specific agent."""

    enabled: bool = True
    model: str | None = None
    model_tiers: ModelTiers | None = None
    approval_mode: str | None = None
    sandbox: str | None = None
    extra_args: dict[str, Any] = Field(default_factory=dict)


class RoutingConfig(BaseModel):
    """Smart routing configuration."""

    enabled: bool = True
    rules: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "code": ["codex", "opencode", "gemini"],
            "explain": ["gemini", "opencode", "codex"],
            "debug": ["codex", "opencode", "gemini"],
            "general": ["gemini", "opencode", "codex"],
        }
    )
    keywords: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "code": ["implement", "write code", "function", "class", "refactor", "create"],
            "explain": ["explain", "what is", "how does", "why", "describe"],
            "debug": ["error", "bug", "fix", "traceback", "exception", "failing"],
        }
    )


class CompetitionConfig(BaseModel):
    """Competition mode configuration."""

    default_agents: list[str] = Field(default_factory=lambda: ["gemini", "codex"])
    evaluation_criteria: list[str] = Field(
        default_factory=lambda: ["correctness", "efficiency", "cleanliness", "architecture"]
    )
    parallel: bool = True
    show_metadata: bool = False


class ReviewConfig(BaseModel):
    """Review configuration for delegation."""

    auto_review: bool = True
    strictness: str = "medium"  # low, medium, high
    max_iterations: int = 3


class TmuxConfig(BaseModel):
    """tmux integration configuration."""

    session_name: str = "orch-session"
    layout: str = "tiled"  # tiled, even-horizontal, even-vertical
    auto_attach: bool = True


class GlobalConfig(BaseModel):
    """Global orch configuration."""

    default_agent: str = "auto"  # "auto" means smart routing
    output_format: str = "text"  # text, json, stream
    color: bool = True
    verbose: bool = False
    stream_by_default: bool = False


class OrchConfig(BaseModel):
    """Root configuration model for orch."""

    global_: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    competition: CompetitionConfig = Field(default_factory=CompetitionConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    tmux: TmuxConfig = Field(default_factory=TmuxConfig)
    agents: dict[str, AgentConfig] = Field(default_factory=dict)

    class Config:
        populate_by_name = True

    @classmethod
    def default(cls) -> "OrchConfig":
        """Create default configuration."""
        return cls(
            agents={
                "gemini": AgentConfig(
                    model="gemini-2.0-flash",
                    model_tiers=ModelTiers(
                        low="gemini-1.5-flash",
                        medium="gemini-2.0-flash",
                        high="gemini-2.0-pro",
                    ),
                    approval_mode="auto_edit",
                ),
                "codex": AgentConfig(
                    model="gpt-5.2-codex",
                    model_tiers=ModelTiers(
                        low="gpt-4o-mini",
                        medium="gpt-4.1",
                        high="o3",
                    ),
                    approval_mode="on-request",
                    sandbox="workspace-write",
                ),
                "claude": AgentConfig(
                    model="sonnet",
                    model_tiers=ModelTiers(
                        low="haiku",
                        medium="sonnet",
                        high="opus",
                    ),
                ),
                "opencode": AgentConfig(
                    model="opencode/grok-code",
                    model_tiers=ModelTiers(
                        low="opencode/glm-4.7-free",
                        medium="opencode/grok-code",
                        high="opencode/minimax-m2.1-free",
                    ),
                    extra_args={"agent": "build"},
                ),
            }
        )

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get configuration for a specific agent."""
        return self.agents.get(agent_name, AgentConfig())


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    config_dir = Path.home() / ".config" / "orch"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_config_file() -> Path:
    """Get the main configuration file path."""
    return get_config_dir() / "config.toml"


def get_sessions_dir() -> Path:
    """Get the sessions storage directory."""
    sessions_dir = get_config_dir() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def get_plugins_dir() -> Path:
    """Get the plugins directory path."""
    plugins_dir = get_config_dir() / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    return plugins_dir
