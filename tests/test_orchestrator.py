import unittest

from ptis.config import (
    PTISConfig,
    QwenProviderConfig,
    InferRequest,
    RiskLevel,
)
from ptis.orchestrator import PTISOrchestrator
from ptis.reasoning.engine import ToolResult


class OrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PTISConfig(qwen=QwenProviderConfig(api_key="test"))
        self.orchestrator = PTISOrchestrator(self.config)

    def test_infer_with_stubbed_model(self) -> None:
        request = InferRequest(
            task_type="math",
            prompt="1+1=?",
            latency_budget_ms=2000,
            cost_cap_usd=0.5,
            risk_level=RiskLevel.MEDIUM,
            parallelism_max=2,
            tools_allowed=["python"],
        )

        def call_model(branch):
            return ("根据检索，答案为2", {"python": [ToolResult("python", "2")]})

        response, trace_id = self.orchestrator.infer(request, call_model_fn=call_model)
        self.assertEqual(response.answer, "根据检索，答案为2")
        self.assertTrue(trace_id)
        self.assertEqual(response.trace_pointers[0], trace_id)


if __name__ == "__main__":
    unittest.main()
