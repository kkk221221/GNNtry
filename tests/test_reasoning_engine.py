import unittest

from ptis.config import BudgetConfig, InferRequest, PlannerConfig, RiskLevel
from ptis.planner import Planner
from ptis.reasoning.engine import ParallelReasoningEngine, ToolResult


class ReasoningEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = InferRequest(
            task_type="math",
            prompt="计算1+1",
            latency_budget_ms=1000,
            cost_cap_usd=0.1,
            risk_level=RiskLevel.MEDIUM,
            parallelism_max=3,
            tools_allowed=["python"],
        )
        self.planner = Planner(PlannerConfig(default_branches=3, default_rounds=2))
        self.plan = self.planner.plan(self.request)
        self.engine = ParallelReasoningEngine(BudgetConfig(token_budget=600, halving_ratio=0.5))

    def test_engine_selects_best_branch(self) -> None:
        responses = {
            0: [
                ("根据检索，答案为2", {"python": [ToolResult("python", "print(1+1)=2")]})
            ],
            1: [
                ("TODO 补充", {}),
            ],
            2: [
                ("根据检索，答案可能为3?", {}),
            ],
        }
        call_counts = {0: 0, 1: 0, 2: 0}

        def call_model(branch):
            index = call_counts[branch.branch_id]
            options = responses[branch.branch_id]
            if index < len(options):
                response = options[index]
            else:
                response = options[-1]
            call_counts[branch.branch_id] = index + 1
            return response

        decision = self.engine.execute(self.request, self.plan, call_model)
        self.assertEqual(decision.best_branch.branch_id, 0)
        self.assertTrue(decision.best_branch.evidence)
        self.assertIn("print(1+1)=2", decision.best_branch.evidence[0])


if __name__ == "__main__":
    unittest.main()
