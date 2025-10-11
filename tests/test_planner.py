import unittest

from ptis.config import InferRequest, PlannerConfig, RiskLevel
from ptis.planner import Planner


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = Planner(PlannerConfig())

    def test_plan_with_high_risk(self) -> None:
        request = InferRequest(
            task_type="math_proof",
            prompt="证明",
            latency_budget_ms=12000,
            cost_cap_usd=0.5,
            risk_level=RiskLevel.HIGH,
            parallelism_max=6,
            tools_allowed=["rag", "python"],
        )
        plan = self.planner.plan(request)
        self.assertGreaterEqual(plan.branches, 3)
        self.assertIn("qwen-math-plus", plan.model_candidates)
        self.assertIn("启用检索增强", plan.tool_plan.reasoning_notes)

    def test_plan_with_custom_models(self) -> None:
        request = InferRequest(
            task_type="general",
            prompt="hi",
            latency_budget_ms=5000,
            cost_cap_usd=0.2,
            risk_level=RiskLevel.LOW,
            parallelism_max=2,
            models_preferred=["custom-model"],
        )
        plan = self.planner.plan(request)
        self.assertEqual(plan.model_candidates, ["custom-model"])
        self.assertEqual(len(plan.temperature_schedule), plan.branches)


if __name__ == "__main__":
    unittest.main()
