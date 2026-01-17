"""Configuration manager for loading and merging configs."""

from pathlib import Path
from typing import Any

import toml

from orch.config.schema import OrchConfig, get_config_file


class ConfigManager:
    """Manages configuration loading, merging, and access."""

    _instance: "ConfigManager | None" = None
    _config: OrchConfig | None = None

    def __new__(cls) -> "ConfigManager":
        """Singleton pattern for config manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_config(cls) -> OrchConfig:
        """Get the current configuration, loading if necessary."""
        if cls._config is None:
            cls._config = cls.load_config()
        return cls._config

    @classmethod
    def load_config(cls) -> OrchConfig:
        """Load configuration from all sources.

        Priority (highest to lowest):
        1. Project-level config (.orch.toml in cwd or parents)
        2. User config (~/.config/orch/config.toml)
        3. Default config
        """
        # Start with defaults
        config_dict: dict[str, Any] = {}

        # Load user config
        user_config_file = get_config_file()
        if user_config_file.exists():
            user_config = toml.load(user_config_file)
            config_dict = cls._deep_merge(config_dict, user_config)

        # Load project config (search up from cwd)
        project_config_file = cls._find_project_config()
        if project_config_file and project_config_file.exists():
            project_config = toml.load(project_config_file)
            config_dict = cls._deep_merge(config_dict, project_config)

        # Create config object
        if config_dict:
            return OrchConfig.model_validate(config_dict)
        return OrchConfig.default()

    @classmethod
    def reload(cls) -> OrchConfig:
        """Force reload configuration from disk."""
        cls._config = cls.load_config()
        return cls._config

    @classmethod
    def _find_project_config(cls) -> Path | None:
        """Find project-level config file by searching up from cwd."""
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            config_file = parent / ".orch.toml"
            if config_file.exists():
                return config_file
            # Stop at home directory
            if parent == Path.home():
                break
        return None

    @classmethod
    def _deep_merge(
        cls, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @classmethod
    def save_user_config(cls, config: OrchConfig) -> None:
        """Save configuration to user config file."""
        config_file = get_config_file()
        config_dict = config.model_dump(by_alias=True, exclude_none=True)
        with open(config_file, "w") as f:
            toml.dump(config_dict, f)

    @classmethod
    def set_value(cls, key_path: str, value: Any) -> None:
        """Set a configuration value by dot-separated path.

        Example: set_value("global.default_agent", "codex")
        """
        config = cls.get_config()
        config_dict = config.model_dump(by_alias=True)

        # Navigate to the right place
        keys = key_path.split(".")
        current = config_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

        # Reload
        cls._config = OrchConfig.model_validate(config_dict)
        cls.save_user_config(cls._config)

    @classmethod
    def get_value(cls, key_path: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated path."""
        config = cls.get_config()
        config_dict = config.model_dump(by_alias=True)

        keys = key_path.split(".")
        current = config_dict
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
