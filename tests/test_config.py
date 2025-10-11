import unittest

from ptis.config import (
    PTISConfig,
    QwenProviderConfig,
    PlannerConfig,
    BudgetConfig,
    SafetyConfig,
    ObservabilityConfig,
    CacheConfig,
    InferRequest,
    RiskLevel,
    ExplainLevel,
)


class ConfigTests(unittest.TestCase):
    def test_ptis_config_defaults(self) -> None:
        cfg = PTISConfig(qwen=QwenProviderConfig(api_key="test"))
        self.assertEqual(cfg.planner.default_branches, 4)
        self.assertEqual(cfg.budget.token_budget, 4096)
        self.assertTrue(cfg.safety.enable_output_filter)
        self.assertTrue(cfg.observability.enable_tracing)
        self.assertTrue(cfg.cache.enabled)

    def test_infer_request_validation(self) -> None:
        req = InferRequest(
            task_type="math_proof",
            prompt="证明...",
            latency_budget_ms=12000,
            cost_cap_usd=0.35,
            risk_level=RiskLevel.HIGH,
            parallelism_max=6,
            region="intl",
            models_preferred=["qwen-math-plus"],
            tools_allowed=["python"],
            explain_level=ExplainLevel.TRACE_PTRS,
        )
        self.assertEqual(req.models_preferred, ["qwen-math-plus"])
        self.assertIs(req.explain_level, ExplainLevel.TRACE_PTRS)

    def test_infer_request_cost_cap_validation(self) -> None:
        with self.assertRaises(ValueError):
            InferRequest(
                task_type="test",
                prompt="test",
                latency_budget_ms=1000,
                cost_cap_usd=0,
                risk_level=RiskLevel.LOW,
            )


if __name__ == "__main__":
    unittest.main()
