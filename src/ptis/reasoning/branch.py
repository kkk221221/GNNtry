"""Branch generation utilities."""

from __future__ import annotations

from typing import List

from ..adapters.qwen import ChatMessage
from ..config import InferRequest
from ..planner import TaskPlan
from .types import BranchState


class BranchGenerator:
    """Creates initial branch states based on the task plan."""

    def generate(self, request: InferRequest, plan: TaskPlan) -> List[BranchState]:
        branches: List[BranchState] = []
        for idx, temperature in enumerate(plan.temperature_schedule):
            messages = [
                ChatMessage(
                    role="system",
                    content=(
                        "你是严谨的推理专家。先计划再执行，必要时调用工具。"
                        f" 分支温度: {temperature}."
                    ),
                ),
                ChatMessage(role="user", content=request.prompt),
            ]
            branches.append(BranchState(branch_id=idx, temperature=temperature, messages=messages))
        return branches
