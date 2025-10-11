"""使用真实 DashScope Qwen 服务执行一次端到端推理。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from ptis.config import ExplainLevel, InferRequest, RiskLevel
from ptis.orchestrator import PTISOrchestrator
from ptis.runtime import load_env_config


def main() -> None:
    config = load_env_config()
    orchestrator = PTISOrchestrator(config)

    request = InferRequest(
        task_type="general_reasoning",
        prompt="使用简洁中文解释地球为什么有四季更替。",
        latency_budget_ms=12000,
        cost_cap_usd=0.5,
        risk_level=RiskLevel.MEDIUM,
        parallelism_max=2,
        models_preferred=[os.getenv("PTIS_MODEL", "qwen-plus")],
        tools_allowed=[],
        explain_level=ExplainLevel.BRIEF,
    )

    response, trace_id = orchestrator.infer(request)
    output = asdict(response)
    output["trace_id"] = trace_id
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
