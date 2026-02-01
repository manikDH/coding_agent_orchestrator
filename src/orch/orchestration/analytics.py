"""Analytics and failure logging for continuous learning"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import defaultdict


@dataclass
class FailureRecord:
    """Record of an orchestration failure"""
    session_id: str
    phase: str  # "planning" | "execution" | "critique"
    error_type: str  # "timeout" | "security" | "validation" | etc.
    error_message: str
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "FailureRecord":
        """Load from dict"""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class AnalyticsLogger:
    """Logs failures and provides analytics"""

    def __init__(self, analytics_dir: Path | None = None):
        if analytics_dir is None:
            from orch.config.schema import get_analytics_dir
            analytics_dir = get_analytics_dir()

        self.analytics_dir = analytics_dir
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        self.failures_log = self.analytics_dir / "failures.jsonl"

    def log_failure(self, failure: FailureRecord) -> None:
        """Log a failure to the analytics file"""
        with open(self.failures_log, 'a') as f:
            json.dump(failure.to_dict(), f)
            f.write('\n')

    def get_failure_stats(self) -> dict[str, Any]:
        """Get aggregated failure statistics"""
        if not self.failures_log.exists():
            return {
                "total_failures": 0,
                "by_phase": {},
                "by_error_type": {},
            }

        by_phase = defaultdict(int)
        by_error_type = defaultdict(int)
        total = 0

        with open(self.failures_log, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                failure_data = json.loads(line)
                total += 1
                by_phase[failure_data["phase"]] += 1
                by_error_type[failure_data["error_type"]] += 1

        return {
            "total_failures": total,
            "by_phase": dict(by_phase),
            "by_error_type": dict(by_error_type),
        }

    def get_recent_failures(self, limit: int = 10) -> list[dict]:
        """Get most recent failures"""
        if not self.failures_log.exists():
            return []

        failures = []
        with open(self.failures_log, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                failures.append(json.loads(line))

        # Return most recent (last N lines)
        return failures[-limit:][::-1]  # Reverse to show newest first

    def get_failures_by_session(self, session_id: str) -> list[FailureRecord]:
        """Get all failures for a specific session"""
        if not self.failures_log.exists():
            return []

        failures = []
        with open(self.failures_log, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                failure_data = json.loads(line)
                if failure_data["session_id"] == session_id:
                    failures.append(FailureRecord.from_dict(failure_data))

        return failures
