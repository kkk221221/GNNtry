import http.client
import json
import time
import unittest

from ptis.api import PTISRequestHandler, run_server
from ptis.config import PTISConfig, QwenProviderConfig
from ptis.reasoning.engine import ToolResult


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PTISConfig(qwen=QwenProviderConfig(api_key="test"))
        self.server = run_server(self.config, host="127.0.0.1", port=0)
        orchestrator = PTISRequestHandler.orchestrator
        orchestrator.set_call_model_override(
            lambda branch: ("根据检索，答案为2", {"python": [ToolResult("python", "2")]})
        )
        self.port = self.server.server_address[1]
        time.sleep(0.05)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def test_infer_endpoint(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        payload = {
            "task_type": "math",
            "prompt": "1+1?",
            "latency_budget_ms": 1000,
            "cost_cap_usd": 0.2,
            "risk_level": "medium",
            "tools_allowed": ["python"],
        }
        conn.request("POST", "/v1/infer", body=json.dumps(payload), headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        data = json.loads(body)
        self.assertEqual(response.status, 200)
        self.assertIn("trace_pointers", data)
        trace_id = data["trace_pointers"][0]
        conn.request("GET", f"/v1/traces/{trace_id}")
        trace_resp = conn.getresponse()
        trace_body = json.loads(trace_resp.read().decode("utf-8"))
        self.assertEqual(trace_resp.status, 200)
        self.assertEqual(trace_body["trace_id"], trace_id)
        conn.close()


if __name__ == "__main__":
    unittest.main()
