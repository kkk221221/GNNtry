"""Safety and governance utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .config import SafetyConfig


@dataclass(slots=True)
class SafetyEvent:
    stage: str
    message: str


@dataclass
class SafetyCenter:
    config: SafetyConfig
    events: List[SafetyEvent] = field(default_factory=list)

    def pre_screen(self, text: str) -> bool:
        if not self.config.enable_input_filter:
            return True
        if any(keyword in text for keyword in ("违法", "暴力")):
            self.events.append(SafetyEvent(stage="input", message="检测到敏感关键词"))
            return False
        return True

    def post_screen(self, text: str) -> bool:
        if not self.config.enable_output_filter:
            return True
        if "免责声明" in text:
            return True
        if any(keyword in text for keyword in ("杀伤", "恶意")):
            self.events.append(SafetyEvent(stage="output", message="输出包含风险内容"))
            return False
        return True

    def record_thinking(self, thinking: str) -> None:
        if self.config.enable_thinking_storage:
            self.events.append(SafetyEvent(stage="thinking", message=thinking[:200]))
