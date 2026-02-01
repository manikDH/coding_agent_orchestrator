"""Tests for checkpoint system"""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from orch.orchestration.checkpoint import Checkpoint, CheckpointManager


def test_checkpoint_serialization():
    """Test checkpoint can be serialized and deserialized"""
    checkpoint = Checkpoint(
        session_id="test123",
        phase="plan_complete",
        timestamp=datetime.now(),
        state_snapshot={"iteration": 0, "phase": "planning"},
        data={"plan": {"steps": [1, 2, 3]}}
    )

    # Convert to dict
    checkpoint_dict = checkpoint.to_dict()
    assert checkpoint_dict["session_id"] == "test123"
    assert checkpoint_dict["phase"] == "plan_complete"

    # Load from dict
    restored = Checkpoint.from_dict(checkpoint_dict)
    assert restored.session_id == "test123"
    assert restored.phase == "plan_complete"


def test_checkpoint_manager_save_load():
    """Test saving and loading checkpoints"""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        manager = CheckpointManager(checkpoint_dir)

        # Create and save checkpoint
        checkpoint = Checkpoint(
            session_id="test123",
            phase="execution_0",
            timestamp=datetime.now(),
            state_snapshot={"iteration": 0},
            data={"results": ["done"]}
        )

        filepath = manager.save_checkpoint(checkpoint)
        assert filepath.exists()

        # Load it back
        loaded = manager.load_checkpoint("execution_0")
        assert loaded is not None
        assert loaded.session_id == "test123"
        assert loaded.phase == "execution_0"


def test_checkpoint_manager_latest():
    """Test loading latest checkpoint"""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        manager = CheckpointManager(checkpoint_dir)

        # Save multiple checkpoints
        for i in range(3):
            checkpoint = Checkpoint(
                session_id="test123",
                phase=f"phase_{i}",
                timestamp=datetime.now(),
                state_snapshot={},
                data={}
            )
            manager.save_checkpoint(checkpoint)

        # Load latest
        latest = manager.load_checkpoint()
        assert latest is not None
        assert "phase_" in latest.phase


def test_checkpoint_manager_list():
    """Test listing all checkpoints"""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        manager = CheckpointManager(checkpoint_dir)

        # Save some checkpoints
        for i in range(2):
            checkpoint = Checkpoint(
                session_id="test123",
                phase=f"phase_{i}",
                timestamp=datetime.now(),
                state_snapshot={},
                data={}
            )
            manager.save_checkpoint(checkpoint)

        # List all
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2
