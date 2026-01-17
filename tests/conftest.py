"""Pytest configuration and fixtures."""

import pytest

from orch.agents.registry import AgentRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the agent registry before each test."""
    AgentRegistry._initialized = False
    AgentRegistry._backends.clear()
    AgentRegistry._backend_classes.clear()
    yield


@pytest.fixture
def mock_gemini_available(monkeypatch):
    """Mock gemini as available."""
    import shutil
    original = shutil.which

    def mock_which(name):
        if name == "gemini":
            return "/opt/homebrew/bin/gemini"
        return original(name)

    monkeypatch.setattr(shutil, "which", mock_which)


@pytest.fixture
def mock_codex_available(monkeypatch):
    """Mock codex as available."""
    import shutil
    original = shutil.which

    def mock_which(name):
        if name == "codex":
            return "/opt/homebrew/bin/codex"
        return original(name)

    monkeypatch.setattr(shutil, "which", mock_which)
