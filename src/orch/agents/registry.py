"""Agent registry for discovering and managing agent backends."""

import importlib.util
import logging
from importlib.metadata import entry_points
from pathlib import Path
from typing import Type

from orch.agents.protocol import AgentAdapter
from orch.config.schema import get_plugins_dir

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Discovers, loads, and manages agent adapters.

    Agents are discovered from:
    1. Built-in agents (gemini, codex)
    2. Entry points (third-party packages)
    3. User plugin directory (~/.config/orch/plugins/)
    """

    _backends: dict[str, AgentAdapter] = {}
    _backend_classes: dict[str, Type[AgentAdapter]] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry by discovering all agents."""
        if cls._initialized:
            return

        cls._backends.clear()
        cls._backend_classes.clear()

        # 1. Load built-in agents
        cls._load_builtin_agents()

        # 2. Load entry point agents
        cls._load_entrypoint_agents()

        # 3. Load user plugins
        cls._load_user_plugins()

        cls._initialized = True

    @classmethod
    def _load_builtin_agents(cls) -> None:
        """Load the built-in agents."""
        from orch.agents.claude import ClaudeAgent
        from orch.agents.codex import CodexAgent
        from orch.agents.gemini import GeminiAgent
        from orch.agents.opencode import OpenCodeAgent

        cls._register_class(ClaudeAgent)
        cls._register_class(GeminiAgent)
        cls._register_class(CodexAgent)
        cls._register_class(OpenCodeAgent)

    @classmethod
    def _load_entrypoint_agents(cls) -> None:
        """Load agents from setuptools entry points."""
        try:
            eps = entry_points(group="orch.agents")
            for ep in eps:
                try:
                    agent_class = ep.load()
                    cls._register_class(agent_class)
                except Exception as e:
                    # Log but don't fail on bad plugins
                    logger.warning("Failed to load agent plugin %s: %s", ep.name, e)
        except Exception:
            # entry_points() might not work in all environments
            pass

    @classmethod
    def _load_user_plugins(cls) -> None:
        """Load agents from user plugin directory."""
        plugin_dir = get_plugins_dir()
        if not plugin_dir.exists():
            return

        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            try:
                cls._load_plugin_file(plugin_file)
            except Exception as e:
                logger.warning("Failed to load plugin %s: %s", plugin_file.name, e)

    @classmethod
    def _load_plugin_file(cls, plugin_file: Path) -> None:
        """Load a single plugin file."""
        spec = importlib.util.spec_from_file_location(
            plugin_file.stem, plugin_file
        )
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Look for AgentAdapter subclasses
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, AgentAdapter)
                and attr is not AgentAdapter
            ):
                cls._register_class(attr)

    @classmethod
    def _register_class(cls, agent_class: Type[AgentAdapter]) -> None:
        """Register an agent class."""
        instance = agent_class()
        cls._backend_classes[instance.name] = agent_class
        cls._backends[instance.name] = instance

    @classmethod
    def get(cls, name: str) -> AgentAdapter | None:
        """Get an agent by name."""
        cls._ensure_initialized()
        return cls._backends.get(name)

    @classmethod
    def get_available(cls) -> list[AgentAdapter]:
        """Get all available (installed) agents."""
        cls._ensure_initialized()
        return [b for b in cls._backends.values() if b.is_available()]

    @classmethod
    def get_all(cls) -> list[AgentAdapter]:
        """Get all registered agents."""
        cls._ensure_initialized()
        return list(cls._backends.values())

    @classmethod
    def get_names(cls) -> list[str]:
        """Get names of all registered agents."""
        cls._ensure_initialized()
        return list(cls._backends.keys())

    @classmethod
    def get_available_names(cls) -> list[str]:
        """Get names of all available agents."""
        return [a.name for a in cls.get_available()]

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure registry is initialized."""
        if not cls._initialized:
            cls.initialize()

    @classmethod
    def reload(cls) -> None:
        """Force reload of all agents."""
        cls._initialized = False
        cls.initialize()

    @classmethod
    def register(cls, agent_class: Type[AgentAdapter]) -> None:
        """Manually register an agent class.

        Useful for testing or programmatic registration.
        """
        cls._ensure_initialized()
        cls._register_class(agent_class)
