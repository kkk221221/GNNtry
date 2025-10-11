"""Configuration and request/response models for PTIS without third-party deps."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExplainLevel(str, Enum):
    NONE = "none"
    BRIEF = "brief"
    TRACE_PTRS = "trace_ptrs"


@dataclass(slots=True)
class InferRequest:
    task_type: str
    prompt: str
    latency_budget_ms: int
    cost_cap_usd: float
    risk_level: RiskLevel = RiskLevel.MEDIUM
    parallelism_max: int = 4
    region: str = "cn"
    models_preferred: List[str] = field(default_factory=list)
    tools_allowed: List[str] = field(default_factory=list)
    explain_level: ExplainLevel = ExplainLevel.BRIEF

    def __post_init__(self) -> None:
        if not self.task_type:
            raise ValueError("task_type must not be empty")
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")
        if self.cost_cap_usd <= 0:
            raise ValueError("cost_cap_usd must be positive")
        if self.parallelism_max <= 0:
            raise ValueError("parallelism_max must be positive")
        if self.region not in {"cn", "intl", "finance"}:
            raise ValueError("region must be one of cn, intl, finance")
        if not isinstance(self.risk_level, RiskLevel):
            raise TypeError("risk_level must be a RiskLevel")
        if not isinstance(self.explain_level, ExplainLevel):
            raise TypeError("explain_level must be an ExplainLevel")


@dataclass(slots=True)
class InferResponse:
    answer: str
    confidence: float
    trace_pointers: List[str]
    cost_actual_usd: float
    latency_ms: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.cost_actual_usd < 0:
            raise ValueError("cost_actual_usd must be non-negative")
        if self.latency_ms <= 0:
            raise ValueError("latency_ms must be positive")


@dataclass(slots=True)
class QwenProviderConfig:
    api_key: str
    base_url_cn: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    base_url_intl: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    base_url_finance: str = "https://dashscope-finance.aliyuncs.com/compatible-mode/v1"
    timeout_s: float = 30.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("api_key must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    def resolve_base_url(self, region: str) -> str:
        if region == "cn":
            return self.base_url_cn
        if region == "intl":
            return self.base_url_intl
        if region == "finance":
            return self.base_url_finance
        raise ValueError(f"Unsupported region: {region}")


@dataclass(slots=True)
class PlannerConfig:
    default_branches: int = 4
    default_rounds: int = 2
    min_temperature: float = 0.4
    max_temperature: float = 0.9

    def __post_init__(self) -> None:
        if self.default_branches <= 0:
            raise ValueError("default_branches must be positive")
        if self.default_rounds <= 0:
            raise ValueError("default_rounds must be positive")
        if not (0 <= self.min_temperature <= 2):
            raise ValueError("min_temperature must be between 0 and 2")
        if not (0 <= self.max_temperature <= 2):
            raise ValueError("max_temperature must be between 0 and 2")
        if self.min_temperature > self.max_temperature:
            raise ValueError("min_temperature cannot exceed max_temperature")


@dataclass(slots=True)
class BudgetConfig:
    token_budget: int = 4096
    exploration_ratio: float = 0.4
    halving_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if not 0 <= self.exploration_ratio <= 1:
            raise ValueError("exploration_ratio must be between 0 and 1")
        if not 0 < self.halving_ratio <= 1:
            raise ValueError("halving_ratio must be between 0 and 1")


@dataclass(slots=True)
class SafetyConfig:
    enable_input_filter: bool = True
    enable_output_filter: bool = True
    enable_thinking_storage: bool = True


@dataclass(slots=True)
class ObservabilityConfig:
    enable_tracing: bool = True
    enable_metrics: bool = True


@dataclass(slots=True)
class CacheConfig:
    enabled: bool = True
    ttl_seconds: int = 3600
    max_entries: int = 1024

    def __post_init__(self) -> None:
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if self.max_entries <= 0:
            raise ValueError("max_entries must be positive")


@dataclass(slots=True)
class PTISConfig:
    qwen: QwenProviderConfig
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
