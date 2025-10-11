import unittest

from ptis.config import SafetyConfig
from ptis.safety import SafetyCenter


class SafetyTests(unittest.TestCase):
    def test_pre_and_post_screen(self) -> None:
        safety = SafetyCenter(SafetyConfig())
        self.assertFalse(safety.pre_screen("这是违法内容"))
        self.assertTrue(safety.post_screen("正常回答"))
        self.assertFalse(safety.post_screen("涉及杀伤行为"))
        self.assertTrue(any(event.stage == "output" for event in safety.events))

    def test_record_thinking(self) -> None:
        safety = SafetyCenter(SafetyConfig(enable_thinking_storage=True))
        safety.record_thinking("internal reasoning")
        self.assertTrue(safety.events)


if __name__ == "__main__":
    unittest.main()
