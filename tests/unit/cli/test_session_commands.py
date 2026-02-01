"""Tests for session CLI commands"""
import pytest
from click.testing import CliRunner
from pathlib import Path
from unittest.mock import Mock, patch
from orch.cli.main import cli
from datetime import datetime


@pytest.fixture
def runner():
    """Create Click test runner"""
    return CliRunner()


@pytest.fixture
def mock_sessions_dir(tmp_path):
    """Create mock sessions directory"""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create a few mock session directories
    for session_id in ["abc123", "def456", "ghi789"]:
        session_dir = sessions_dir / session_id
        session_dir.mkdir()

        # Create a checkpoint file
        checkpoint_file = session_dir / "init_2024-01-01T10:00:00.json"
        checkpoint_file.write_text('{"session_id": "' + session_id + '", "phase": "init"}')

    return sessions_dir


def test_session_list(runner, mock_sessions_dir):
    """Test session list command"""
    with patch("orch.config.schema.get_sessions_dir", return_value=mock_sessions_dir):
        result = runner.invoke(cli, ["session", "list"])

        assert result.exit_code == 0
        # Should show session IDs
        assert "abc123" in result.output or "def456" in result.output or "ghi789" in result.output


def test_session_list_empty(runner, tmp_path):
    """Test session list with no sessions"""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with patch("orch.config.schema.get_sessions_dir", return_value=empty_dir):
        result = runner.invoke(cli, ["session", "list"])

        assert result.exit_code == 0
        assert "No sessions found" in result.output or "no sessions" in result.output.lower()


def test_session_status(runner, mock_sessions_dir):
    """Test session status command"""
    session_id = "abc123"
    session_dir = mock_sessions_dir / session_id

    # Create a more complete checkpoint
    checkpoint_file = session_dir / "complete_2024-01-01T10:05:00.json"
    checkpoint_file.write_text('''{
        "session_id": "abc123",
        "phase": "complete",
        "timestamp": "2024-01-01T10:05:00",
        "state_snapshot": {
            "id": "abc123",
            "state": "complete",
            "iteration": 1,
            "metrics": {
                "executions_count": 2,
                "critique_rounds": 1
            }
        },
        "data": {}
    }''')

    with patch("orch.config.schema.get_sessions_dir", return_value=mock_sessions_dir):
        result = runner.invoke(cli, ["session", "status", session_id])

        assert result.exit_code == 0
        assert session_id in result.output
        assert "complete" in result.output.lower()


def test_session_status_not_found(runner, tmp_path):
    """Test session status with non-existent session"""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with patch("orch.config.schema.get_sessions_dir", return_value=empty_dir):
        result = runner.invoke(cli, ["session", "status", "nonexistent"])

        # Should show error
        assert "not found" in result.output.lower() or "error" in result.output.lower()


def test_session_trace(runner, mock_sessions_dir):
    """Test session trace command"""
    session_id = "abc123"
    session_dir = mock_sessions_dir / session_id

    # Create checkpoints for a trace
    checkpoints = [
        ("init", "2024-01-01T10:00:00", {"phase": "init"}),
        ("plan_complete", "2024-01-01T10:01:00", {"plan": {"steps": ["test"]}}),
        ("execution_0", "2024-01-01T10:02:00", {"results": {"status": "done"}}),
    ]

    for phase, timestamp, data in checkpoints:
        checkpoint_file = session_dir / f"{phase}_{timestamp}.json"
        checkpoint_file.write_text(f'''{{
            "session_id": "{session_id}",
            "phase": "{phase}",
            "timestamp": "{timestamp}",
            "state_snapshot": {{}},
            "data": {data}
        }}'''.replace("'", '"'))

    with patch("orch.config.schema.get_sessions_dir", return_value=mock_sessions_dir):
        result = runner.invoke(cli, ["session", "trace", session_id])

        assert result.exit_code == 0
        assert session_id in result.output
        # Should show checkpoint phases
        assert "init" in result.output or "plan" in result.output
