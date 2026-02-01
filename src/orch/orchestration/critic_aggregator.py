"""Aggregates multiple critic reviews into a single decision"""
from dataclasses import dataclass, field
from orch.orchestration.models import ReviewFeedback, Issue


@dataclass
class CritiqueDecision:
    """Aggregated decision from all critics"""
    decision: str  # "accept" | "accept_with_suggestions" | "needs_revision" | "reject"
    reason: str
    blocking_issues: list[Issue] = field(default_factory=list)
    suggestions: list[Issue] = field(default_factory=list)
    suggested_fixes: dict[str, str] = field(default_factory=dict)
    aggregated_reviews: list[ReviewFeedback] = field(default_factory=list)
    requires_replanning: bool = False


class CriticAggregator:
    """Aggregates multiple critic reviews using veto hierarchy + weighted scoring"""

    # Veto power hierarchy
    VETO_HIERARCHY = {
        "security": {"power": "absolute", "priority": 1},
        "correctness": {"power": "strong", "priority": 2},
        "performance": {"power": "weak", "priority": 3},
        "style": {"power": "suggestion", "priority": 4},
    }

    def aggregate(self, reviews: list[ReviewFeedback]) -> CritiqueDecision:
        """Aggregate reviews using veto hierarchy + weighted scoring"""

        # Step 1: Check for absolute vetoes (security)
        security_reviews = [r for r in reviews if r.critic_type == "security"]
        for review in security_reviews:
            if review.decision == "reject":
                return CritiqueDecision(
                    decision="reject",
                    reason="Security veto - critical security issues found",
                    blocking_issues=review.issues,
                    aggregated_reviews=reviews,
                    requires_replanning=True
                )

        # Step 2: Check for strong vetoes (correctness)
        correctness_reviews = [r for r in reviews if r.critic_type == "correctness"]
        critical_correctness = self._has_critical_issues(correctness_reviews)
        if critical_correctness:
            return CritiqueDecision(
                decision="reject",
                reason="Correctness veto - functional errors detected",
                blocking_issues=critical_correctness,
                aggregated_reviews=reviews,
                requires_replanning=False  # Can fix without replanning
            )

        # Step 3: Weighted scoring for ambiguous cases
        ambiguous_reviews = self._identify_ambiguous(reviews)
        if ambiguous_reviews:
            decision = self._weighted_scoring(ambiguous_reviews, reviews)
            if decision:
                return decision

        # Step 4: Aggregate non-critical issues
        all_issues = self._collect_all_issues(reviews)
        major_issues = [i for i in all_issues if i.severity in ["critical", "major"]]
        minor_issues = [i for i in all_issues if i.severity in ["minor", "suggestion"]]

        if major_issues:
            return CritiqueDecision(
                decision="needs_revision",
                reason=f"Found {len(major_issues)} major issues requiring fixes",
                blocking_issues=major_issues,
                suggested_fixes=self._suggest_fixes(major_issues),
                aggregated_reviews=reviews,
                requires_replanning=False
            )
        elif minor_issues:
            return CritiqueDecision(
                decision="accept_with_suggestions",
                reason="Acceptable with minor improvement suggestions",
                blocking_issues=[],
                suggestions=minor_issues,
                aggregated_reviews=reviews,
                requires_replanning=False
            )
        else:
            return CritiqueDecision(
                decision="accept",
                reason="All critics approved",
                blocking_issues=[],
                aggregated_reviews=reviews,
                requires_replanning=False
            )

    def _has_critical_issues(self, reviews: list[ReviewFeedback]) -> list[Issue]:
        """Check if any reviews have critical severity issues"""
        critical = []
        for review in reviews:
            critical.extend([i for i in review.issues if i.severity == "critical"])
        return critical

    def _identify_ambiguous(self, reviews: list[ReviewFeedback]) -> list[ReviewFeedback]:
        """Find reviews with conflicting opinions or low confidence"""
        ambiguous = []

        # Conflicting decisions
        decisions = [r.decision for r in reviews]
        if len(set(decisions)) > 1:  # Not unanimous
            # Check if it's weak veto critics disagreeing
            weak_critics = [
                r for r in reviews
                if self.VETO_HIERARCHY.get(r.critic_type, {}).get("power") in ["weak", "suggestion"]
            ]
            if weak_critics:
                ambiguous.extend(weak_critics)

        # Low confidence reviews
        low_confidence = [r for r in reviews if r.confidence < 0.7]
        ambiguous.extend(low_confidence)

        return ambiguous

    def _weighted_scoring(
        self,
        ambiguous_reviews: list[ReviewFeedback],
        all_reviews: list[ReviewFeedback]
    ) -> CritiqueDecision | None:
        """Use weighted scoring for ambiguous cases"""

        # Calculate weighted score
        total_score = 0
        total_weight = 0

        for review in ambiguous_reviews:
            # Weight by critic priority and confidence
            priority = self.VETO_HIERARCHY.get(review.critic_type, {}).get("priority", 5)
            weight = (6 - priority) * review.confidence  # Higher priority = more weight

            # Score: 100 (accept) to 0 (reject)
            score = review.severity_score

            total_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return None

        avg_score = total_score / total_weight

        # Decision thresholds
        if avg_score >= 80:
            return CritiqueDecision(
                decision="accept",
                reason=f"Weighted score: {avg_score:.1f}/100 - acceptable quality",
                blocking_issues=[],
                aggregated_reviews=all_reviews,
                requires_replanning=False
            )
        elif avg_score >= 60:
            return CritiqueDecision(
                decision="needs_revision",
                reason=f"Weighted score: {avg_score:.1f}/100 - needs improvement",
                blocking_issues=self._extract_issues(ambiguous_reviews),
                aggregated_reviews=all_reviews,
                requires_replanning=False
            )
        else:
            return CritiqueDecision(
                decision="reject",
                reason=f"Weighted score: {avg_score:.1f}/100 - quality below threshold",
                blocking_issues=self._extract_issues(ambiguous_reviews),
                aggregated_reviews=all_reviews,
                requires_replanning=True
            )

    def _collect_all_issues(self, reviews: list[ReviewFeedback]) -> list[Issue]:
        """Collect all issues from all reviews"""
        all_issues = []
        for review in reviews:
            all_issues.extend(review.issues)
        return all_issues

    def _extract_issues(self, reviews: list[ReviewFeedback]) -> list[Issue]:
        """Extract issues from reviews"""
        issues = []
        for review in reviews:
            issues.extend(review.issues)
        return issues

    def _suggest_fixes(self, issues: list[Issue]) -> dict[str, str]:
        """Extract suggested fixes from issues"""
        fixes = {}
        for i, issue in enumerate(issues):
            if issue.suggested_fix:
                fixes[f"issue_{i}"] = issue.suggested_fix
        return fixes
