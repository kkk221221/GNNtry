import os
import unittest

from ptis.runtime import load_env_config


class RuntimeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = os.environ.copy()

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_missing_api_key_raises(self) -> None:
        if "DASHSCOPE_API_KEY" in os.environ:
            del os.environ["DASHSCOPE_API_KEY"]
        with self.assertRaises(RuntimeError):
            load_env_config()

    def test_loads_expected_defaults(self) -> None:
        os.environ["DASHSCOPE_API_KEY"] = "secret"
        config = load_env_config()
        self.assertEqual(config.qwen.api_key, "secret")
        self.assertFalse(config.observability.enable_tracing)
        self.assertFalse(config.observability.enable_metrics)
        self.assertFalse(config.cache.enabled)

    def test_invalid_timeout(self) -> None:
        os.environ["DASHSCOPE_API_KEY"] = "secret"
        os.environ["DASHSCOPE_TIMEOUT_S"] = "abc"
        with self.assertRaises(RuntimeError):
            load_env_config()


if __name__ == "__main__":
    unittest.main()
