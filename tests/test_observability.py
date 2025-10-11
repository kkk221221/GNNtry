import time
import unittest

from ptis.config import ObservabilityConfig
from ptis.observability import MetricsCollector, Tracer


class ObservabilityTests(unittest.TestCase):
    def test_tracer_records_spans(self) -> None:
        tracer = Tracer(ObservabilityConfig())
        with tracer.start_span("root") as span:
            self.assertIsNotNone(span)
            span.add_event("start")
            time.sleep(0.001)
        spans = tracer.get_spans()
        self.assertEqual(len(spans), 1)
        self.assertGreater(spans[0].end_time - spans[0].start_time, 0)

    def test_metrics_collector(self) -> None:
        metrics = MetricsCollector(ObservabilityConfig())
        metrics.incr("requests")
        with metrics.timer("latency"):
            time.sleep(0.001)
        self.assertEqual(metrics.counters()["requests"], 1.0)
        self.assertTrue(metrics.timers()["latency"][0] > 0)


if __name__ == "__main__":
    unittest.main()
