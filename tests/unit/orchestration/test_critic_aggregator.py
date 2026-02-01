"""Tests for CriticAggregator"""
import pytest
from orch.orchestration.critic_aggregator import CriticAggregator, CritiqueDecision
from orch.orchestration.models import ReviewFeedback, Issue


def test_security_veto():
    """Security critic absolute veto blocks everything"""
    aggregator = CriticAggregator()

    security_issue = Issue(
        category="security",
        severity="critical",
        description="SQL injection vulnerability"
    )

    reviews = [
        ReviewFeedback(
            critic_type="security",
            decision="reject",
            issues=[security_issue],
            severity_score=0
        ),
        ReviewFeedback(
            critic_type="correctness",
            decision="accept",
            issues=[],
            severity_score=100
        )
    ]

    decision = aggregator.aggregate(reviews)

    assert decision.decision == "reject"
    assert "security" in decision.reason.lower()
    assert decision.requires_replanning is True


def test_correctness_veto():
    """Correctness critic strong veto blocks on critical issues"""
    aggregator = CriticAggregator()

    correctness_issue = Issue(
        category="logic_error",
        severity="critical",
        description="Test failures"
    )

    reviews = [
        ReviewFeedback(
            critic_type="correctness",
            decision="reject",
            issues=[correctness_issue],
            severity_score=0
        )
    ]

    decision = aggregator.aggregate(reviews)

    assert decision.decision == "reject"
    assert "correctness" in decision.reason.lower()
    assert decision.requires_replanning is False


def test_all_approved():
    """All critics approve - accept"""
    aggregator = CriticAggregator()

    reviews = [
        ReviewFeedback(
            critic_type="security",
            decision="accept",
            issues=[],
            severity_score=100
        ),
        ReviewFeedback(
            critic_type="correctness",
            decision="accept",
            issues=[],
            severity_score=100
        )
    ]

    decision = aggregator.aggregate(reviews)

    assert decision.decision == "accept"
    assert len(decision.blocking_issues) == 0


def test_major_issues_need_revision():
    """Major issues trigger needs_revision"""
    aggregator = CriticAggregator()

    major_issue = Issue(
        category="performance",
        severity="major",
        description="O(n^2) algorithm"
    )

    reviews = [
        ReviewFeedback(
            critic_type="performance",
            decision="needs_revision",
            issues=[major_issue],
            severity_score=60
        )
    ]

    decision = aggregator.aggregate(reviews)

    assert decision.decision == "needs_revision"
    assert len(decision.blocking_issues) == 1


def test_minor_issues_accept_with_suggestions():
    """Minor issues accepted with suggestions"""
    aggregator = CriticAggregator()

    minor_issue = Issue(
        category="style",
        severity="minor",
        description="Missing docstring"
    )

    reviews = [
        ReviewFeedback(
            critic_type="style",
            decision="accept",
            issues=[minor_issue],
            severity_score=90
        )
    ]

    decision = aggregator.aggregate(reviews)

    assert decision.decision == "accept_with_suggestions"
    assert len(decision.suggestions) == 1
