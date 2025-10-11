"""HTTP API exposing PTIS capabilities."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from .config import ExplainLevel, InferRequest, InferResponse, PTISConfig, RiskLevel
from .orchestrator import PTISOrchestrator


class PTISRequestHandler(BaseHTTPRequestHandler):
    orchestrator: PTISOrchestrator = None  # type: ignore[assignment]

    def do_POST(self) -> None:  # pragma: no cover - exercised via integration test
        if self.path != "/v1/infer":
            self.send_error(404, "Not Found")
            return
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw_body)
            request = self._parse_request(payload)
            response, _ = self.orchestrator.infer(request)
            self._write_json(200, infer_response_to_dict(response))
        except (ValueError, KeyError) as exc:
            self._write_json(400, {"error": str(exc)})

    def do_GET(self) -> None:  # pragma: no cover - exercised via integration test
        parsed = urlparse(self.path)
        if parsed.path.startswith("/v1/traces/"):
            trace_id = parsed.path.rsplit("/", 1)[-1]
            record = self.orchestrator.get_trace(trace_id)
            if record is None:
                self._write_json(404, {"error": "trace not found"})
                return
            branches = []
            for branch in record.branches:
                branches.append(
                    {
                        "branch_id": branch.branch_id,
                        "score": branch.score,
                        "messages": [msg.to_json() for msg in branch.messages],
                        "evidence": list(branch.evidence),
                    }
                )
            self._write_json(200, {"trace_id": record.trace_id, "branches": branches})
            return
        self.send_error(404, "Not Found")

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover - silence default logging
        return

    def _parse_request(self, payload: Dict[str, Any]) -> InferRequest:
        try:
            risk_level = RiskLevel(payload.get("risk_level", "medium"))
        except ValueError as exc:
            raise ValueError("invalid risk_level") from exc
        explain = ExplainLevel(payload.get("explain_level", "brief"))
        return InferRequest(
            task_type=payload["task_type"],
            prompt=payload["prompt"],
            latency_budget_ms=int(payload["latency_budget_ms"]),
            cost_cap_usd=float(payload["cost_cap_usd"]),
            risk_level=risk_level,
            parallelism_max=int(payload.get("parallelism_max", 4)),
            region=payload.get("region", "cn"),
            models_preferred=list(payload.get("models_preferred", [])),
            tools_allowed=list(payload.get("tools_allowed", [])),
            explain_level=explain,
        )

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def infer_response_to_dict(response: InferResponse) -> Dict[str, Any]:
    return {
        "answer": response.answer,
        "confidence": response.confidence,
        "trace_pointers": response.trace_pointers,
        "cost_actual_usd": response.cost_actual_usd,
        "latency_ms": response.latency_ms,
    }


def run_server(config: PTISConfig, host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    orchestrator = PTISOrchestrator(config)
    PTISRequestHandler.orchestrator = orchestrator
    server = ThreadingHTTPServer((host, port), PTISRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
