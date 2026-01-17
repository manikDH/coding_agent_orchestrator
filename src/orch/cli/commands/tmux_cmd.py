"""tmux-related CLI commands."""

import os

import click

from orch.agents.registry import AgentRegistry
from orch.config.manager import ConfigManager
from orch.output.formatter import get_formatter


@click.group()
def tmux() -> None:
    """tmux integration for parallel agent execution."""
    pass


@tmux.command("compete")
@click.argument("prompt", nargs=-1, required=True)
@click.option("-a", "--agents", multiple=True, help="Agents to compete")
@click.option("--no-attach", is_flag=True, help="Don't attach to session")
@click.option("-C", "--cwd", "working_dir", help="Working directory for agents")
def tmux_compete(
    prompt: tuple[str, ...],
    agents: tuple[str, ...],
    no_attach: bool,
    working_dir: str | None,
) -> None:
    """Run competition in tmux with agents in separate panes."""
    from orch.tmux.manager import TmuxManager, run_competition_in_tmux

    formatter = get_formatter()

    if not TmuxManager.is_available():
        formatter.print_error("tmux is not installed. Install with: brew install tmux")
        raise SystemExit(1)

    config = ConfigManager.get_config()

    # Determine agents to use
    if agents:
        agent_names = list(agents)
    else:
        agent_names = config.competition.default_agents

    # Validate agents
    AgentRegistry.initialize()
    available_agents = []
    for name in agent_names:
        agent = AgentRegistry.get(name)
        if agent and agent.is_available():
            available_agents.append(name)
        else:
            formatter.print_warning(f"Agent '{name}' not available, skipping")

    if len(available_agents) < 2:
        formatter.print_error("Need at least 2 available agents for competition")
        raise SystemExit(1)

    prompt_text = " ".join(prompt)
    cwd = working_dir or os.getcwd()

    formatter.print_info(f"Starting tmux competition: {', '.join(available_agents)}")
    formatter.print_info(f"Working directory: {cwd}")

    run_competition_in_tmux(
        agents=available_agents,
        prompt=prompt_text,
        working_dir=cwd,
        auto_attach=not no_attach,
    )


@tmux.command("attach")
def tmux_attach() -> None:
    """Attach to the orch tmux session."""
    from orch.tmux.manager import TmuxManager

    formatter = get_formatter()
    manager = TmuxManager()

    if not manager.is_available():
        formatter.print_error("tmux is not installed")
        raise SystemExit(1)

    session = manager.get_session(create=False)
    if session is None:
        formatter.print_error("No orch tmux session running")
        raise SystemExit(1)

    manager.attach()


@tmux.command("kill")
def tmux_kill() -> None:
    """Kill the orch tmux session."""
    from orch.tmux.manager import TmuxManager

    formatter = get_formatter()
    manager = TmuxManager()

    if not manager.is_available():
        formatter.print_error("tmux is not installed")
        raise SystemExit(1)

    session = manager.get_session(create=False)
    if session is None:
        formatter.print_info("No orch tmux session running")
        return

    manager.kill_session()
    formatter.print_success("tmux session killed")


@tmux.command("list")
def tmux_list() -> None:
    """List all tmux sessions."""
    from orch.tmux.manager import TmuxManager

    from rich.table import Table

    formatter = get_formatter()
    manager = TmuxManager()

    if not manager.is_available():
        formatter.print_error("tmux is not installed")
        raise SystemExit(1)

    sessions = manager.list_sessions()

    if not sessions:
        formatter.print_info("No tmux sessions running")
        return

    table = Table(title="tmux Sessions")
    table.add_column("Name", style="cyan")
    table.add_column("Windows", justify="right")
    table.add_column("Attached")

    for session in sessions:
        attached = "[green]Yes[/green]" if session["attached"] else "No"
        table.add_row(session["name"], str(session["windows"]), attached)

    formatter.console.print(table)
