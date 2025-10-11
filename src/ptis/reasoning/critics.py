"""Critics for branch evaluation."""

from __future__ import annotations

from typing import Iterable, List

from .types import BranchEvaluation, BranchState


class SelfCritic:
    """Evaluates a branch output for internal consistency."""

    def evaluate(self, branch: BranchState, response_text: str) -> BranchEvaluation:
        fatal_flaws: List[str] = []
        minor_issues: List[str] = []
        suggestions: List[str] = []
        if "TODO" in response_text.upper():
            fatal_flaws.append("存在未完成的步骤")
        if "?" in response_text:
            minor_issues.append("回答存在疑问，需要补充证据")
        if "根据检索" not in response_text:
            suggestions.append("补充检索证据确保可验证性")
        score = max(0.1, min(1.0, len(response_text) / 500))
        return BranchEvaluation(
            branch_id=branch.branch_id,
            score=score,
            fatal_flaws=fatal_flaws,
            minor_issues=minor_issues,
            suggestions=suggestions,
        )


class CrossCritic:
    """Performs cross-branch comparisons to encourage diversity and evidence."""

    def evaluate(self, branches: Iterable[BranchState]) -> List[BranchEvaluation]:
        evaluations: List[BranchEvaluation] = []
        for branch in branches:
            overlap = sum(1 for msg in branch.messages if isinstance(msg.content, str) and "证据" in msg.content)
            bonus = 0.1 * overlap
            evaluations.append(
                BranchEvaluation(
                    branch_id=branch.branch_id,
                    score=min(1.0, branch.score + bonus),
                    suggestions=["确保证据引用与结论一致"],
                )
            )
        return evaluations
