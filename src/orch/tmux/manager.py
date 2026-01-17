"""tmux session management for parallel agent execution."""

import shlex
import shutil
from typing import Any

import libtmux

from orch.config.schema import TmuxConfig


class TmuxManager:
    """Manages tmux sessions for parallel agent execution."""

    def __init__(self, config: TmuxConfig | None = None) -> None:
        self.config = config or TmuxConfig()
        self._server: libtmux.Server | None = None
        self._session: libtmux.Session | None = None

    @staticmethod
    def is_available() -> bool:
        """Check if tmux is available on the system."""
        return shutil.which("tmux") is not None

    @property
    def server(self) -> libtmux.Server:
        """Get or create the tmux server connection."""
        if self._server is None:
            self._server = libtmux.Server()
        return self._server

    def get_session(self, create: bool = True) -> libtmux.Session | None:
        """Get the orch tmux session.

        Args:
            create: If True, create session if it doesn't exist.

        Returns:
            The tmux session or None if not found and create=False.
        """
        session_name = self.config.session_name

        # Check if session exists
        try:
            session = self.server.sessions.get(session_name=session_name)
            if session:
                self._session = session
                return session
        except Exception:
            pass

        if create:
            return self.create_session()

        return None

    def create_session(self) -> libtmux.Session:
        """Create a new orch tmux session."""
        session_name = self.config.session_name

        # Kill existing session if any
        try:
            existing = self.server.sessions.get(session_name=session_name)
            if existing:
                existing.kill()
        except Exception:
            pass

        # Create new session
        self._session = self.server.new_session(
            session_name=session_name,
            attach=False,
        )

        return self._session

    def create_agent_panes(
        self,
        agent_names: list[str],
        prompt: str,
        working_dir: str | None = None,
    ) -> dict[str, libtmux.Pane]:
        """Create panes for each agent in the session.

        Args:
            agent_names: List of agent names to create panes for.
            prompt: The prompt to execute.
            working_dir: Working directory for the agents.

        Returns:
            Dictionary mapping agent names to their panes.
        """
        session = self.get_session(create=True)
        if session is None:
            raise RuntimeError("Failed to create tmux session")

        window = session.active_window
        panes: dict[str, libtmux.Pane] = {}

        # First pane is already created with the window
        first_pane = window.active_pane
        if first_pane and agent_names:
            panes[agent_names[0]] = first_pane

        # Create additional panes for remaining agents
        for agent_name in agent_names[1:]:
            pane = window.split()
            panes[agent_name] = pane

        # Apply layout
        window.select_layout(self.config.layout)

        return panes

    def run_agent_in_pane(
        self,
        pane: libtmux.Pane,
        agent_name: str,
        prompt: str,
        working_dir: str | None = None,
    ) -> None:
        """Run an agent command in a tmux pane.

        Args:
            pane: The tmux pane to run in.
            agent_name: Name of the agent.
            prompt: The prompt to execute.
            working_dir: Working directory.
        """
        # Use shlex.quote for proper shell escaping (handles newlines, quotes, etc.)
        escaped_prompt = shlex.quote(prompt)

        if working_dir:
            escaped_dir = shlex.quote(working_dir)
            pane.send_keys(f"cd {escaped_dir}", enter=True)

        # Build the orch command for this agent
        cmd = f"orch {agent_name} {escaped_prompt}"
        pane.send_keys(cmd, enter=True)

    def run_parallel_agents(
        self,
        agent_names: list[str],
        prompt: str,
        working_dir: str | None = None,
    ) -> None:
        """Run multiple agents in parallel tmux panes.

        Args:
            agent_names: Agents to run.
            prompt: The prompt to execute.
            working_dir: Working directory.
        """
        panes = self.create_agent_panes(agent_names, prompt, working_dir)

        for agent_name, pane in panes.items():
            self.run_agent_in_pane(pane, agent_name, prompt, working_dir)

    def add_manager_pane(self) -> libtmux.Pane:
        """Add a manager pane at the bottom for orchestration output."""
        session = self.get_session(create=False)
        if session is None:
            raise RuntimeError("No active session")

        window = session.active_window

        # Split horizontally to create bottom pane
        manager_pane = window.split(vertical=False, percent=30)

        # Re-apply layout
        window.select_layout(self.config.layout)

        return manager_pane

    def attach(self) -> None:
        """Attach to the orch tmux session."""
        import subprocess

        session = self.get_session(create=False)
        if session is None:
            raise RuntimeError("No orch session to attach to")

        subprocess.run(["tmux", "attach-session", "-t", self.config.session_name])

    def kill_session(self) -> None:
        """Kill the orch tmux session."""
        session = self.get_session(create=False)
        if session:
            session.kill()
            self._session = None

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all tmux sessions."""
        sessions = []
        for session in self.server.sessions:
            sessions.append({
                "name": session.name,
                "windows": len(session.windows),
                "attached": session.attached,
            })
        return sessions


def run_competition_in_tmux(
    agents: list[str],
    prompt: str,
    working_dir: str | None = None,
    auto_attach: bool = True,
) -> None:
    """Run a competition in tmux with agents in separate panes.

    Args:
        agents: List of agent names.
        prompt: The prompt to compete on.
        working_dir: Working directory.
        auto_attach: Whether to attach to session after creation.
    """
    from orch.config.manager import ConfigManager

    config = ConfigManager.get_config()
    manager = TmuxManager(config.tmux)

    if not manager.is_available():
        raise RuntimeError("tmux is not installed")

    # Create session with agent panes
    manager.run_parallel_agents(agents, prompt, working_dir)

    if auto_attach:
        manager.attach()
