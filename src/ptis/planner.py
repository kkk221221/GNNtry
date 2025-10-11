"""Task planner and router for PTIS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import InferRequest, PlannerConfig, RiskLevel


@dataclass(slots=True)
class ToolPlan:
    tools: List[str] = field(default_factory=list)
    reasoning_notes: str = ""


@dataclass(slots=True)
class TaskPlan:
    branches: int
    rounds: int
    model_candidates: List[str]
    temperature_schedule: List[float]
    tool_plan: ToolPlan


class Planner:
    """Planner producing a concrete execution plan for a request."""

    DEFAULT_MODEL_BY_TASK: Dict[str, List[str]] = {
        "math": ["qwen-math-plus", "qwq-plus", "qwen-plus"],
        "code": ["qwen-coder-plus", "qwen-plus"],
        "vision": ["qwen-vl-plus", "qwen-plus"],
        "longform": ["qwen-long", "qwen-plus"],
    }

    def __init__(self, config: PlannerConfig) -> None:
        self._config = config

    def plan(self, request: InferRequest) -> TaskPlan:
        branches = min(max(2, self._config.default_branches), request.parallelism_max)
        rounds = self._config.default_rounds
        if request.risk_level is RiskLevel.HIGH:
            branches = min(request.parallelism_max, branches + 2)
            rounds += 1
        elif request.risk_level is RiskLevel.LOW:
            branches = max(1, branches - 1)

        model_candidates = self._select_models(request)
        temperature_schedule = self._temperature_schedule(branches)
        tool_plan = self._build_tool_plan(request)
        return TaskPlan(
            branches=branches,
            rounds=rounds,
            model_candidates=model_candidates,
            temperature_schedule=temperature_schedule,
            tool_plan=tool_plan,
        )

    def _select_models(self, request: InferRequest) -> List[str]:
        if request.models_preferred:
            return request.models_preferred
        task_lower = request.task_type.lower()
        for key, models in self.DEFAULT_MODEL_BY_TASK.items():
            if key in task_lower:
                return models
        return ["qwen-plus", "qwen-turbo"]

    def _temperature_schedule(self, branches: int) -> List[float]:
        if branches <= 1:
            return [self._config.min_temperature]
        step = 0
        schedule = []
        for idx in range(branches):
            ratio = idx / max(1, branches - 1)
            temperature = self._config.min_temperature + (
                (self._config.max_temperature - self._config.min_temperature) * ratio
            )
            schedule.append(round(temperature, 2))
        return schedule

    def _build_tool_plan(self, request: InferRequest) -> ToolPlan:
        tools = list(dict.fromkeys(request.tools_allowed))
        notes: List[str] = []
        if "rag" in tools:
            notes.append("启用检索增强，确保证据可追溯。")
        if "python" in tools:
            notes.append("复杂计算交给 Python 沙箱执行。")
        if request.risk_level is RiskLevel.HIGH:
            notes.append("高风险任务需要额外审计。")
        reasoning_notes = " ".join(notes)
        return ToolPlan(tools=tools, reasoning_notes=reasoning_notes)
