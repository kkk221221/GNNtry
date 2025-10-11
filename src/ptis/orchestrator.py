"""Orchestrator connecting planner, PRE, safety, and providers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .adapters.qwen import ChatCompletionRequest, QwenAdapter
from .cache import TTLCache
from .config import InferRequest, InferResponse, PTISConfig
from .observability import MetricsCollector, Tracer
from .planner import Planner
from .reasoning.engine import ParallelReasoningEngine, ToolResult
from .reasoning.types import BranchState
from .safety import SafetyCenter
from .state_store import StateStore


@dataclass(slots=True)
class InferenceContext:
    trace_id: str
    start_time: float


class PTISOrchestrator:
    def __init__(self, config: PTISConfig) -> None:
        self._config = config
        self._planner = Planner(config.planner)
        self._engine = ParallelReasoningEngine(config.budget)
        self._adapter = QwenAdapter(config.qwen)
        self._safety = SafetyCenter(config.safety)
        self._tracer = Tracer(config.observability)
        self._metrics = MetricsCollector(config.observability)
        self._state_store = StateStore()
        self._cache = TTLCache(config.cache)
        self._call_model_override: Optional[
            Callable[[BranchState], Tuple[str, Dict[str, List[ToolResult]]]]
        ] = None

    def infer(
        self,
        request: InferRequest,
        call_model_fn: Optional[Callable[[BranchState], Tuple[str, Dict[str, List[ToolResult]]]]] = None,
    ) -> Tuple[InferResponse, str]:
        if not self._safety.pre_screen(request.prompt):
            raise ValueError("输入未通过安全审查")
        context = InferenceContext(trace_id=self._state_store.new_trace(), start_time=time.time())
        plan = self._planner.plan(request)
        with self._tracer.start_span("infer", parent_id=None) as span:
            if span is not None:
                span.add_event("plan_ready")
            self._metrics.incr("requests")
            model_caller = call_model_fn or self._call_model_override or self._build_model_caller(request, plan)
            decision = self._engine.execute(request, plan, model_caller)
            self._state_store.record_branches(context.trace_id, [decision.best_branch] + decision.alternative_branches)
            answer = self._extract_answer(decision.best_branch)
            if not self._safety.post_screen(answer):
                raise ValueError("输出未通过安全审查")
            latency_ms = int((time.time() - context.start_time) * 1000)
            response = InferResponse(
                answer=answer,
                confidence=min(1.0, decision.best_branch.score),
                trace_pointers=[context.trace_id],
                cost_actual_usd=0.0,
                latency_ms=max(1, latency_ms),
            )
            if span is not None:
                span.add_event("response_ready")
        return response, context.trace_id

    def get_trace(self, trace_id: str):  # pragma: no cover - simple passthrough
        return self._state_store.get_trace(trace_id)

    def set_call_model_override(
        self, override: Optional[Callable[[BranchState], Tuple[str, Dict[str, List[ToolResult]]]]]
    ) -> None:
        self._call_model_override = override

    def _extract_answer(self, branch: BranchState) -> str:
        for message in reversed(branch.messages):
            if message.role == "assistant" and isinstance(message.content, str):
                return message.content
        return ""

    def _build_model_caller(
        self, request: InferRequest, plan
    ) -> Callable[[BranchState], Tuple[str, Dict[str, List[ToolResult]]]]:
        def call(branch: BranchState) -> Tuple[str, Dict[str, List[ToolResult]]]:
            cache_key = f"{request.task_type}:{branch.temperature}:{len(branch.messages)}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
            chat_request = ChatCompletionRequest(
                model=plan.model_candidates[0],
                messages=branch.messages,
                temperature=branch.temperature,
            )
            payload = self._adapter.chat(chat_request, region=request.region)
            response_text = self._parse_response_text(payload)
            result = (response_text, {})
            self._cache.set(cache_key, result)
            return result

        return call

    def _parse_response_text(self, payload: Dict) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [part.get("text", "") for part in content if isinstance(part, dict)]
            return "".join(texts)
        return ""
