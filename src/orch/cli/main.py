"""Main CLI entry point for orch."""

import asyncio
from typing import Any

import click

from orch.agents.registry import AgentRegistry
from orch.config.manager import ConfigManager
from orch.output.formatter import get_formatter


class AliasedGroup(click.Group):
    """Click group that supports dynamic agent commands and aliases."""

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        # First check built-in commands
        rv = super().get_command(ctx, cmd_name)
        if rv is not None:
            return rv

        # Check if it's a registered agent
        AgentRegistry.initialize()
        agent = AgentRegistry.get(cmd_name)
        if agent:
            return _create_agent_command(cmd_name)

        return None

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List all commands including dynamic agent commands."""
        commands = super().list_commands(ctx)

        # Add available agents as commands
        AgentRegistry.initialize()
        for agent_name in AgentRegistry.get_available_names():
            if agent_name not in commands:
                commands.append(agent_name)

        return sorted(commands)


def _create_agent_command(agent_name: str) -> click.Command:
    """Dynamically create a command for a specific agent."""

    @click.command(name=agent_name, help=f"Execute prompt using {agent_name}")
    @click.argument("prompt", nargs=-1, required=True)
    @click.option("-m", "--model", help="Model to use")
    @click.option("-s", "--stream", is_flag=True, help="Stream output")
    @click.option("--json", "output_json", is_flag=True, help="JSON output")
    @click.pass_context
    def agent_cmd(
        ctx: click.Context,
        prompt: tuple[str, ...],
        model: str | None,
        stream: bool,
        output_json: bool,
    ) -> None:
        prompt_text = " ".join(prompt)
        asyncio.run(_execute_agent(agent_name, prompt_text, model, stream, output_json))

    return agent_cmd


def _get_model_for_complexity(
    agent_config: Any,
    complexity: str | None,
) -> str | None:
    """Select appropriate model based on task complexity.

    Args:
        agent_config: The agent's configuration (AgentConfig).
        complexity: Task complexity level (low, medium, high).

    Returns:
        Model name to use.
    """
    if complexity is None:
        return agent_config.model

    # Get model from configured tiers
    if agent_config.model_tiers:
        tier_model = getattr(agent_config.model_tiers, complexity, None)
        if tier_model:
            return tier_model

    # Fall back to default model
    return agent_config.model


async def _execute_agent(
    agent_name: str,
    prompt: str,
    model: str | None = None,
    stream: bool = False,
    output_json: bool = False,
    complexity: str | None = None,
) -> None:
    """Execute a prompt using a specific agent."""
    from orch.agents.protocol import OutputFormat

    formatter = get_formatter()
    agent = AgentRegistry.get(agent_name)

    if agent is None:
        formatter.print_error(f"Unknown agent: {agent_name}")
        raise SystemExit(1)

    if not agent.is_available():
        formatter.print_error(f"Agent '{agent_name}' is not available. Is {agent.cli_name} installed?")
        raise SystemExit(1)

    output_format = OutputFormat.JSON if output_json else OutputFormat.TEXT

    # Get agent config
    config = ConfigManager.get_config()
    agent_config = config.get_agent_config(agent_name)

    # Select model based on complexity
    selected_model = model or _get_model_for_complexity(agent_config, complexity)

    extra_args: dict[str, Any] = {}
    if agent_config.approval_mode:
        extra_args["approval_mode"] = agent_config.approval_mode
    if agent_config.sandbox:
        extra_args["sandbox"] = agent_config.sandbox
    extra_args.update(agent_config.extra_args)

    if complexity:
        formatter.print_info(f"Using {selected_model} for {complexity} complexity task")

    if stream and agent.get_capabilities().supports_streaming:
        async for chunk in agent.stream_execute(
            prompt,
            output_format=output_format,
            model=selected_model,
            extra_args=extra_args,
        ):
            formatter.print_streaming(chunk)
        print()  # Final newline
    else:
        result = await agent.execute(
            prompt,
            output_format=output_format,
            model=selected_model,
            extra_args=extra_args,
        )
        formatter.print_result(result, show_metadata=config.global_.verbose)


@click.group(cls=AliasedGroup)
@click.option("-v", "--verbose", is_flag=True, help="Verbose output")
@click.option("--no-color", is_flag=True, help="Disable colors")
@click.version_option(package_name="orch")
@click.pass_context
def cli(
    ctx: click.Context,
    verbose: bool,
    no_color: bool,
) -> None:
    """Orch - Universal AI Agent Orchestrator.

    Claude Code as manager, AI CLIs as workers.

    \b
    Examples:
        orch ask "explain this error"       # Smart routing
        orch gemini "explain recursion"     # Use Gemini
        orch codex "implement binary search" # Use Codex
        orch compare "implement sorting"    # Compare agents
        orch agent list                     # List agents
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["no_color"] = no_color

    # Initialize formatter
    get_formatter(color=not no_color, verbose=verbose)


@cli.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("-b", "--backend", "agent_name", help="Agent to use")
@click.option("-m", "--model", help="Model to use")
@click.option("-s", "--stream", is_flag=True, help="Stream output")
@click.option("--json", "output_json", is_flag=True, help="JSON output")
@click.option("--complexity", type=click.Choice(["low", "medium", "high"]), help="Task complexity for model selection")
def ask(
    prompt: tuple[str, ...],
    agent_name: str | None,
    model: str | None,
    stream: bool,
    output_json: bool,
    complexity: str | None,
) -> None:
    """Execute a prompt with smart routing.

    Uses smart routing to pick the best agent, or specify one with --backend.
    """
    prompt_text = " ".join(prompt)

    if agent_name:
        asyncio.run(_execute_agent(agent_name, prompt_text, model, stream, output_json, complexity))
    else:
        asyncio.run(_execute_with_routing(prompt_text, model, stream, output_json, complexity))


async def _execute_with_routing(
    prompt: str,
    model: str | None = None,
    stream: bool = False,
    output_json: bool = False,
    complexity: str | None = None,
) -> None:
    """Execute with smart routing to determine best agent."""
    from orch.orchestration.router import Router

    config = ConfigManager.get_config()
    router = Router(config.routing)

    # Determine best agent
    agent_name = router.route(prompt)
    formatter = get_formatter()
    formatter.print_info(f"Routing to: {agent_name}")

    await _execute_agent(agent_name, prompt, model, stream, output_json, complexity)


# --- Subcommands ---


@cli.group()
def agent() -> None:
    """Manage agent backends."""
    pass


@agent.command("list")
@click.option("--all", "show_all", is_flag=True, help="Show unavailable agents too")
def agent_list(show_all: bool) -> None:
    """List registered agents."""
    AgentRegistry.initialize()
    formatter = get_formatter()

    if show_all:
        agents = [(a.name, a.display_name, a.is_available()) for a in AgentRegistry.get_all()]
    else:
        agents = [(a.name, a.display_name, True) for a in AgentRegistry.get_available()]

    if not agents:
        formatter.print_warning("No agents registered")
        return

    formatter.print_agent_list(agents)


@agent.command("info")
@click.argument("name")
def agent_info(name: str) -> None:
    """Show detailed info about an agent."""
    AgentRegistry.initialize()
    formatter = get_formatter()

    agent = AgentRegistry.get(name)
    if agent is None:
        formatter.print_error(f"Unknown agent: {name}")
        raise SystemExit(1)

    from rich.table import Table

    table = Table(title=f"{agent.display_name} ({agent.name})")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Executable", agent.executable)
    table.add_row("Available", "Yes" if agent.is_available() else "No")

    caps = agent.get_capabilities()
    table.add_row("Streaming", "Yes" if caps.supports_streaming else "No")
    table.add_row("Sessions", "Yes" if caps.supports_sessions else "No")
    table.add_row("Images", "Yes" if caps.supports_images else "No")
    table.add_row("Sandbox", "Yes" if caps.supports_sandbox else "No")
    table.add_row("Strengths", ", ".join(caps.task_strengths))

    version = agent.get_version()
    if version:
        table.add_row("Version", version)

    formatter.console.print(table)


@agent.command("test")
@click.argument("name")
def agent_test(name: str) -> None:
    """Test connectivity to an agent."""
    AgentRegistry.initialize()
    formatter = get_formatter()

    agent = AgentRegistry.get(name)
    if agent is None:
        formatter.print_error(f"Unknown agent: {name}")
        raise SystemExit(1)

    if not agent.is_available():
        formatter.print_error(f"Agent '{name}' is not available")
        raise SystemExit(1)

    formatter.print_info(f"Testing {agent.display_name}...")

    async def test() -> None:
        result = await agent.execute("Say 'Hello from orch test!' in exactly those words.")
        if result.success:
            formatter.print_success(f"Agent responded: {result.content[:100]}...")
        else:
            formatter.print_error(f"Agent test failed: {result.error}")

    asyncio.run(test())


@cli.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("-a", "--agents", multiple=True, help="Agents to compare")
@click.option("--parallel/--sequential", default=True, help="Run in parallel")
@click.option("--metadata", is_flag=True, help="Show metadata")
@click.pass_context
def compare(
    ctx: click.Context,
    prompt: tuple[str, ...],
    agents: tuple[str, ...],
    parallel: bool,
    metadata: bool,
) -> None:
    """Compare responses from multiple agents."""
    prompt_text = " ".join(prompt)
    asyncio.run(_run_comparison(prompt_text, list(agents) if agents else None, parallel, metadata))


async def _run_comparison(
    prompt: str,
    agent_names: list[str] | None,
    parallel: bool,
    show_metadata: bool,
) -> None:
    """Run comparison between multiple agents."""
    from orch.agents.protocol import OutputFormat

    config = ConfigManager.get_config()
    formatter = get_formatter()

    # Get agents to compare
    if agent_names:
        agents_to_use = agent_names
    else:
        agents_to_use = config.competition.default_agents

    # Filter to available agents
    AgentRegistry.initialize()
    available_agents = []
    for name in agents_to_use:
        agent = AgentRegistry.get(name)
        if agent and agent.is_available():
            available_agents.append(agent)
        else:
            formatter.print_warning(f"Agent '{name}' not available, skipping")

    if len(available_agents) < 2:
        formatter.print_error("Need at least 2 available agents for comparison")
        raise SystemExit(1)

    formatter.print_info(f"Comparing: {', '.join(a.name for a in available_agents)}")

    # Execute on all agents
    if parallel:
        tasks = [
            agent.execute(prompt, output_format=OutputFormat.TEXT)
            for agent in available_agents
        ]
        results = await asyncio.gather(*tasks)
    else:
        results = []
        for agent in available_agents:
            result = await agent.execute(prompt, output_format=OutputFormat.TEXT)
            results.append(result)

    # Display comparison
    formatter.print_comparison(list(results), show_metadata=show_metadata)


@cli.group()
def config() -> None:
    """Manage configuration."""
    pass


@config.command("show")
def config_show() -> None:
    """Show current configuration."""
    config = ConfigManager.get_config()
    formatter = get_formatter()

    import json
    config_dict = config.model_dump(by_alias=True)
    formatter.console.print_json(json.dumps(config_dict, indent=2))


@config.command("edit")
def config_edit() -> None:
    """Open config file in editor."""
    import os
    import subprocess

    from orch.config.schema import get_config_file

    config_file = get_config_file()

    # Create default config if it doesn't exist
    if not config_file.exists():
        config = ConfigManager.get_config()
        ConfigManager.save_user_config(config)

    editor = os.environ.get("EDITOR", "vim")
    subprocess.run([editor, str(config_file)])


# Register tmux commands
from orch.cli.commands.tmux_cmd import tmux

cli.add_command(tmux)


if __name__ == "__main__":
    cli()
