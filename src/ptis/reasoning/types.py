"""Shared reasoning types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..adapters.qwen import ChatMessage


@dataclass(slots=True)
class BranchState:
    branch_id: int
    temperature: float
    messages: List[ChatMessage]
    score: float = 0.0
    alive: bool = True
    evidence: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))


@dataclass(slots=True)
class BranchEvaluation:
    branch_id: int
    score: float
    fatal_flaws: List[str] = field(default_factory=list)
    minor_issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass(slots=True)
class AggregatedDecision:
    best_branch: BranchState
    justification: str
    alternative_branches: List[BranchState] = field(default_factory=list)
