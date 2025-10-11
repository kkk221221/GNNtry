"""State store to persist reasoning artifacts."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List

from .reasoning.types import BranchState


@dataclass(slots=True)
class TraceRecord:
    trace_id: str
    timestamp: float
    branches: List[BranchState] = field(default_factory=list)


class StateStore:
    def __init__(self) -> None:
        self._records: Dict[str, TraceRecord] = {}

    def new_trace(self) -> str:
        trace_id = uuid.uuid4().hex
        self._records[trace_id] = TraceRecord(trace_id=trace_id, timestamp=time.time())
        return trace_id

    def record_branches(self, trace_id: str, branches: List[BranchState]) -> None:
        record = self._records.setdefault(trace_id, TraceRecord(trace_id=trace_id, timestamp=time.time()))
        record.branches = [branch for branch in branches]

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self._records.get(trace_id)
