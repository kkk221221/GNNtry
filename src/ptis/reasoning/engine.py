"""Parallel reasoning engine coordinating branches, critics and judges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

from ..config import BudgetConfig, InferRequest
from ..planner import TaskPlan
from .branch import BranchGenerator
from .budget import BudgetManager
from .critics import CrossCritic, SelfCritic
from .types import AggregatedDecision, BranchEvaluation, BranchState


@dataclass
class ToolResult:
    tool_name: str
    output: str


class ParallelReasoningEngine:
    """Core engine implementing generation, critique, revision and aggregation."""

    def __init__(self, budget_config: BudgetConfig) -> None:
        self._branch_generator = BranchGenerator()
        self._self_critic = SelfCritic()
        self._cross_critic = CrossCritic()
        self._budget_manager = BudgetManager(budget_config)

    def execute(
        self,
        request: InferRequest,
        plan: TaskPlan,
        call_model: Callable[[BranchState], Tuple[str, Dict[str, List[ToolResult]]]],
    ) -> AggregatedDecision:
        branches = self._branch_generator.generate(request, plan)
        token_budget = self._budget_manager.allocate_tokens(plan.rounds, len(branches))
        for round_index in range(plan.rounds):
            for branch in branches:
                if not branch.alive:
                    continue
                response_text, tool_outputs = call_model(branch)
                branch.add_message("assistant", response_text)
                self._apply_tool_outputs(branch, tool_outputs)
                evaluation = self._self_critic.evaluate(branch, response_text)
                branch.score = evaluation.score
                branch.issues = evaluation.fatal_flaws + evaluation.minor_issues
            cross_evaluations = self._cross_critic.evaluate(b for b in branches if b.alive)
            self._apply_cross_feedback(branches, cross_evaluations)
            self._budget_manager.prune(branches, round_index)
        best_branch = self._select_best_branch(branches)
        alt_branches = [b for b in branches if b is not best_branch and b.alive]
        justification = self._build_justification(best_branch, alt_branches)
        return AggregatedDecision(best_branch=best_branch, justification=justification, alternative_branches=alt_branches)

    def _apply_tool_outputs(
        self, branch: BranchState, tool_outputs: Dict[str, List[ToolResult]]
    ) -> None:
        for tool_name, results in tool_outputs.items():
            for result in results:
                evidence_text = f"工具[{tool_name}] -> {result.output}"
                branch.evidence.append(evidence_text)
                branch.add_message("tool", evidence_text)

    def _apply_cross_feedback(
        self, branches: Iterable[BranchState], evaluations: Iterable[BranchEvaluation]
    ) -> None:
        eval_map = {evaluation.branch_id: evaluation for evaluation in evaluations}
        for branch in branches:
            evaluation = eval_map.get(branch.branch_id)
            if not evaluation:
                continue
            branch.score = max(branch.score, evaluation.score)
            branch.issues.extend(evaluation.suggestions)

    def _select_best_branch(self, branches: Iterable[BranchState]) -> BranchState:
        alive_branches = [b for b in branches if b.alive]
        if not alive_branches:
            alive_branches = list(branches)
        return max(alive_branches, key=lambda b: b.score)

    def _build_justification(self, best: BranchState, alternatives: List[BranchState]) -> str:
        justification_parts = [
            f"选择分支 {best.branch_id}，得分 {best.score:.2f}，证据数 {len(best.evidence)}。",
        ]
        if best.issues:
            justification_parts.append("仍需关注: " + "; ".join(best.issues))
        if alternatives:
            alt_summary = ", ".join(f"#{b.branch_id}: {b.score:.2f}" for b in alternatives)
            justification_parts.append(f"备选分支评分: {alt_summary}")
        return " ".join(justification_parts)
