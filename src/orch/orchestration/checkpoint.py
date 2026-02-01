"""Checkpoint system for orchestration resilience"""
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Checkpoint:
    """Snapshot of orchestration state at a specific phase"""
    session_id: str
    phase: str  # "init", "plan_complete", "execution_0", "critique_0", "complete"
    timestamp: datetime
    state_snapshot: dict[str, Any]
    data: dict[str, Any]  # Phase-specific data (plan, results, etc.)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization"""
        return {
            "session_id": self.session_id,
            "phase": self.phase,
            "timestamp": self.timestamp.isoformat(),
            "state_snapshot": self.state_snapshot,
            "data": self.data
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        """Load from dict"""
        return cls(
            session_id=data["session_id"],
            phase=data["phase"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state_snapshot=data["state_snapshot"],
            data=data["data"]
        )

    @classmethod
    def from_file(cls, path: Path) -> "Checkpoint":
        """Load checkpoint from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class CheckpointManager:
    """Manages checkpoint creation and recovery"""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, checkpoint: Checkpoint) -> Path:
        """Save checkpoint to disk"""
        filename = f"{checkpoint.phase}_{checkpoint.timestamp.isoformat()}.json"
        filepath = self.checkpoint_dir / filename

        with open(filepath, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        return filepath

    def load_checkpoint(self, phase: str | None = None) -> Checkpoint | None:
        """Load checkpoint for specific phase or latest"""
        if phase:
            checkpoints = list(self.checkpoint_dir.glob(f"{phase}_*.json"))
        else:
            checkpoints = list(self.checkpoint_dir.glob("*.json"))

        if not checkpoints:
            return None

        # Return most recent
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return Checkpoint.from_file(latest)

    def list_checkpoints(self) -> list[Checkpoint]:
        """List all checkpoints for this session"""
        checkpoints = []
        for filepath in sorted(self.checkpoint_dir.glob("*.json")):
            checkpoints.append(Checkpoint.from_file(filepath))
        return checkpoints

    def get_checkpoint_path(self, phase: str) -> Path | None:
        """Get path to specific checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob(f"{phase}_*.json"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
