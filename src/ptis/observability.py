"""Observability utilities with lightweight tracing and metrics."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import ObservabilityConfig


@dataclass(slots=True)
class TraceEvent:
    timestamp: float
    message: str


@dataclass(slots=True)
class TraceSpan:
    span_id: str
    name: str
    parent_id: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    events: List[TraceEvent] = field(default_factory=list)

    def end(self) -> None:
        self.end_time = time.time()

    def add_event(self, message: str) -> None:
        self.events.append(TraceEvent(timestamp=time.time(), message=message))


class Tracer:
    def __init__(self, config: ObservabilityConfig) -> None:
        self._config = config
        self._spans: Dict[str, TraceSpan] = {}

    @contextmanager
    def start_span(self, name: str, parent_id: Optional[str] = None):
        if not self._config.enable_tracing:
            yield None
            return
        span_id = uuid.uuid4().hex
        span = TraceSpan(span_id=span_id, name=name, parent_id=parent_id, start_time=time.time())
        self._spans[span_id] = span
        try:
            yield span
        finally:
            span.end()

    def get_spans(self) -> List[TraceSpan]:
        return list(self._spans.values())


class MetricsCollector:
    def __init__(self, config: ObservabilityConfig) -> None:
        self._config = config
        self._counters: Dict[str, float] = {}
        self._timers: Dict[str, List[float]] = {}

    def incr(self, name: str, value: float = 1.0) -> None:
        if not self._config.enable_metrics:
            return
        self._counters[name] = self._counters.get(name, 0.0) + value

    @contextmanager
    def timer(self, name: str):
        if not self._config.enable_metrics:
            yield None
            return
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self._timers.setdefault(name, []).append(duration)

    def counters(self) -> Dict[str, float]:
        return dict(self._counters)

    def timers(self) -> Dict[str, List[float]]:
        return {name: list(values) for name, values in self._timers.items()}
