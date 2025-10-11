"""Budget management strategies."""

from __future__ import annotations

from typing import List

from ..config import BudgetConfig
from .types import BranchState


class BudgetManager:
    """Implements successive halving inspired pruning of branches."""

    def __init__(self, config: BudgetConfig) -> None:
        self._config = config

    def allocate_tokens(self, rounds: int, branches: int) -> List[int]:
        base = max(1, self._config.token_budget // max(1, branches * rounds))
        return [base for _ in range(branches)]

    def prune(self, branches: List[BranchState], round_index: int) -> None:
        if not branches:
            return
        live_branches = [b for b in branches if b.alive]
        if len(live_branches) <= 1:
            return
        threshold = sorted((b.score for b in live_branches), reverse=True)
        cutoff_index = max(1, int(len(threshold) * self._config.halving_ratio)) - 1
        cutoff = threshold[cutoff_index]
        for branch in live_branches:
            if branch.score < cutoff:
                branch.alive = False

    def exploration_needed(self, round_index: int) -> bool:
        return round_index == 0 and self._config.exploration_ratio > 0
