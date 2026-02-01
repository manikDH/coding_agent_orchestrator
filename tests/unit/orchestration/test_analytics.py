"""Tests for analytics/failure logging"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from orch.orchestration.analytics import AnalyticsLogger, FailureRecord


@pytest.fixture
def temp_analytics_dir(tmp_path):
    """Create temporary analytics directory"""
    analytics_dir = tmp_path / "analytics"
    analytics_dir.mkdir()
    return analytics_dir


def test_log_failure(temp_analytics_dir):
    """Test logging a failure"""
    logger = AnalyticsLogger(temp_analytics_dir)

    failure = FailureRecord(
        session_id="test123",
        phase="execution",
        error_type="timeout",
        error_message="Agent timed out",
        context={"agent": "codex", "task": "implement feature"}
    )

    logger.log_failure(failure)

    # Verify file was created
    log_file = temp_analytics_dir / "failures.jsonl"
    assert log_file.exists()

    # Verify content
    content = log_file.read_text()
    assert "test123" in content
    assert "timeout" in content


def test_log_multiple_failures(temp_analytics_dir):
    """Test logging multiple failures"""
    logger = AnalyticsLogger(temp_analytics_dir)

    failures = [
        FailureRecord(
            session_id="session1",
            phase="planning",
            error_type="validation",
            error_message="Invalid plan",
            context={}
        ),
        FailureRecord(
            session_id="session2",
            phase="critique",
            error_type="security",
            error_message="Security issue",
            context={}
        )
    ]

    for failure in failures:
        logger.log_failure(failure)

    # Verify both logged
    log_file = temp_analytics_dir / "failures.jsonl"
    lines = log_file.read_text().strip().split("\n")
    assert len(lines) == 2


def test_get_failure_stats(temp_analytics_dir):
    """Test retrieving failure statistics"""
    logger = AnalyticsLogger(temp_analytics_dir)

    # Log some failures
    failures = [
        FailureRecord("s1", "execution", "timeout", "msg1", {}),
        FailureRecord("s2", "execution", "timeout", "msg2", {}),
        FailureRecord("s3", "critique", "security", "msg3", {}),
    ]

    for failure in failures:
        logger.log_failure(failure)

    # Get stats
    stats = logger.get_failure_stats()

    assert stats["total_failures"] == 3
    assert stats["by_phase"]["execution"] == 2
    assert stats["by_phase"]["critique"] == 1
    assert stats["by_error_type"]["timeout"] == 2
    assert stats["by_error_type"]["security"] == 1


def test_get_recent_failures(temp_analytics_dir):
    """Test getting recent failures"""
    logger = AnalyticsLogger(temp_analytics_dir)

    # Log failures
    for i in range(5):
        failure = FailureRecord(
            session_id=f"session{i}",
            phase="execution",
            error_type="error",
            error_message=f"Error {i}",
            context={}
        )
        logger.log_failure(failure)

    # Get recent (limit 3)
    recent = logger.get_recent_failures(limit=3)

    assert len(recent) == 3
    # Should be in reverse order (most recent first)
    assert "session4" in recent[0]["session_id"]


def test_failure_record_serialization():
    """Test FailureRecord can be serialized"""
    failure = FailureRecord(
        session_id="test",
        phase="execution",
        error_type="timeout",
        error_message="Test error",
        context={"key": "value"}
    )

    # Should be serializable to dict
    data = failure.to_dict()
    assert data["session_id"] == "test"
    assert data["phase"] == "execution"
    assert "timestamp" in data
