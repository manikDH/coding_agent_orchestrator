"""Output formatting using Rich for beautiful terminal output."""

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from orch.agents.protocol import ExecutionResult

# Custom theme for orch
ORCH_THEME = Theme(
    {
        "agent.gemini": "cyan",
        "agent.codex": "green",
        "agent.default": "blue",
        "success": "green",
        "error": "red bold",
        "warning": "yellow",
        "info": "blue",
        "prompt": "magenta",
        "metadata": "dim",
    }
)


class OutputFormatter:
    """Handles all output formatting for orch."""

    def __init__(self, color: bool = True, verbose: bool = False) -> None:
        self.console = Console(theme=ORCH_THEME, force_terminal=color)
        self.verbose = verbose

    def print_result(self, result: ExecutionResult, show_metadata: bool = False) -> None:
        """Print an execution result."""
        if result.is_error:
            self.print_error(result.error or "Unknown error", result.agent_name)
            return

        # Print content as markdown if it looks like markdown
        if self._looks_like_markdown(result.content):
            self.console.print(Markdown(result.content))
        else:
            self.console.print(result.content)

        if show_metadata and result.metadata:
            self._print_metadata(result.metadata)

    def print_error(self, message: str, agent_name: str | None = None) -> None:
        """Print an error message."""
        prefix = f"[{agent_name}] " if agent_name else ""
        self.console.print(f"[error]{prefix}Error: {message}[/error]")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"[success]{message}[/success]")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        self.console.print(f"[info]{message}[/info]")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[warning]{message}[/warning]")

    def print_agent_header(self, agent_name: str, display_name: str) -> None:
        """Print a header for an agent's output."""
        style = f"agent.{agent_name}" if agent_name in ("gemini", "codex") else "agent.default"
        self.console.print(f"\n[{style}]━━━ {display_name} ━━━[/{style}]")

    def print_comparison(
        self,
        results: list[ExecutionResult],
        show_metadata: bool = False,
    ) -> None:
        """Print comparison of multiple agent results."""
        for result in results:
            agent_style = f"agent.{result.agent_name}"
            panel = Panel(
                Markdown(result.content) if self._looks_like_markdown(result.content) else result.content,
                title=f"[{agent_style}]{result.agent_name.upper()}[/{agent_style}]",
                border_style=agent_style,
            )
            self.console.print(panel)

            if show_metadata and result.metadata:
                self._print_metadata(result.metadata)

    def print_competition_result(
        self,
        winner: str,
        scores: dict[str, dict[str, float]],
        explanation: str,
    ) -> None:
        """Print competition result with winner and scores."""
        self.console.print()
        self.console.print(Panel(
            f"[success bold]Winner: {winner.upper()}[/success bold]",
            title="Competition Result",
            border_style="success",
        ))

        # Scores table
        if scores:
            table = Table(title="Evaluation Scores")
            table.add_column("Agent", style="cyan")
            table.add_column("Correctness", justify="right")
            table.add_column("Efficiency", justify="right")
            table.add_column("Cleanliness", justify="right")
            table.add_column("Architecture", justify="right")
            table.add_column("Total", justify="right", style="bold")

            for agent, agent_scores in scores.items():
                total = sum(agent_scores.values())
                table.add_row(
                    agent,
                    f"{agent_scores.get('correctness', 0):.1f}",
                    f"{agent_scores.get('efficiency', 0):.1f}",
                    f"{agent_scores.get('cleanliness', 0):.1f}",
                    f"{agent_scores.get('architecture', 0):.1f}",
                    f"{total:.1f}",
                )

            self.console.print(table)

        if explanation:
            self.console.print()
            self.console.print(Panel(
                Markdown(explanation),
                title="Analysis",
                border_style="info",
            ))

    def print_agent_list(self, agents: list[tuple[str, str, bool]]) -> None:
        """Print list of agents.

        Args:
            agents: List of (name, display_name, is_available) tuples.
        """
        table = Table(title="Registered Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Display Name")
        table.add_column("Status", justify="center")

        for name, display_name, is_available in agents:
            status = "[success]available[/success]" if is_available else "[error]unavailable[/error]"
            table.add_row(name, display_name, status)

        self.console.print(table)

    def print_streaming(self, chunk: str) -> None:
        """Print a streaming chunk without newline."""
        self.console.print(chunk, end="")

    def _print_metadata(self, metadata: dict[str, Any]) -> None:
        """Print metadata in a dimmed style."""
        parts = [f"{k}={v}" for k, v in metadata.items()]
        self.console.print(f"[metadata]({', '.join(parts)})[/metadata]")

    def _looks_like_markdown(self, text: str) -> bool:
        """Check if text appears to be markdown."""
        markdown_indicators = ["```", "##", "**", "- ", "1. ", "> ", "| "]
        return any(indicator in text for indicator in markdown_indicators)


# Global formatter instance
_formatter: OutputFormatter | None = None


def get_formatter(color: bool = True, verbose: bool = False) -> OutputFormatter:
    """Get or create the global formatter instance."""
    global _formatter
    if _formatter is None:
        _formatter = OutputFormatter(color=color, verbose=verbose)
    return _formatter
