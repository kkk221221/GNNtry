# -*- coding: utf-8 -*-
"""
LLM 网关重试机制 (_execute_with_retry) 相关的单元测试。
"""
import unittest
import os
import time
from unittest.mock import patch, MagicMock, call

from aibiowflow.llm_gateway.gateway import LLMGateway, RETRYABLE_STATUS_CODES
from aibiowflow.llm_gateway.exceptions import LLMAPIError
from aibiowflow.llm_gateway.providers.base_provider import ProviderResponse


class TestRetryMechanism(unittest.TestCase):
    """测试 LLMGateway 的重试逻辑 _execute_with_retry。"""

    def setUp(self):
        """测试前准备。"""
        self.temp_dir = "temp_test_retries_dir"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.config_path = os.path.join(self.temp_dir, "retries_config.yaml")
        self.prompts_path = os.path.join(self.temp_dir, "dummy_prompts_retries.toml")

        with open(self.prompts_path, 'w', encoding='utf-8') as f:
            f.write("[dummy]\ntemplate=\"dummy\"\n")

        # Mock provider for _init_providers, though not directly used by _execute_with_retry tests here
        self.mock_provider_instance = MagicMock()

    def tearDown(self):
        """测试后清理。"""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.prompts_path):
            os.remove(self.prompts_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def _create_gateway_with_retry_policy(self, retry_policy: dict) -> LLMGateway:
        """辅助函数：创建具有特定重试策略的 LLMGateway 实例。"""
        config_data = {
            "api_keys": {"mock": "key"}, # Needed for basic init
            "models": [{"name":"m", "provider":"mock", "aliases":["a"]}], # Needed for basic init
            "retry_policy": retry_policy
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            import yaml
            yaml.dump(config_data, f)

        # Mock _init_providers to avoid actual provider setup for these isolated retry tests
        # We are testing _execute_with_retry directly or via a simple call path
        with patch('aibiowflow.llm_gateway.gateway.LLMGateway._init_providers',
                   return_value={"mock": self.mock_provider_instance}):
            gateway = LLMGateway(self.config_path, self.prompts_path)
        return gateway

    @patch('time.sleep') # 模拟 time.sleep 以避免实际等待并可验证其调用
    def test_retry_on_retryable_error_succeeds_eventually(self, mock_sleep: MagicMock):
        """测试当动作在前几次失败（可重试错误）后最终成功时，重试机制的行为。"""
        retry_policy = {"max_retries": 3, "initial_wait_seconds": 0.01, "backoff_factor": 1.5}
        gateway = self._create_gateway_with_retry_policy(retry_policy)

        mock_action = MagicMock()
        # 模拟动作：前两次失败，第三次成功
        expected_response = ProviderResponse(text_content="成功！", model_name="test")
        mock_action.side_effect = [
            LLMAPIError("第一次失败", status_code=RETRYABLE_STATUS_CODES[0]),
            LLMAPIError("第二次失败", status_code=RETRYABLE_STATUS_CODES[1]),
            expected_response
        ]

        action_kwargs = {"arg1": "val1"}
        result = gateway._execute_with_retry(mock_action, "test_action_succeeds", **action_kwargs)

        self.assertEqual(result, expected_response)
        self.assertEqual(mock_action.call_count, 3) # 1次初试 + 2次重试
        mock_action.assert_has_calls([call(**action_kwargs)] * 3) # 确保参数传递正确

        # 验证 time.sleep 的调用次数和大致等待时间
        self.assertEqual(mock_sleep.call_count, 2) # 两次失败后的等待
        self.assertAlmostEqual(mock_sleep.call_args_list[0][0][0], 0.01) # 第一次等待
        self.assertAlmostEqual(mock_sleep.call_args_list[1][0][0], 0.01 * 1.5) # 第二次等待 (指数退避)


    @patch('time.sleep')
    def test_retry_exhausts_retries_and_raises_last_error(self, mock_sleep: MagicMock):
        """测试当所有重试都因可重试错误失败时，抛出最后的 LLMAPIError。"""
        max_r = 2
        retry_policy = {"max_retries": max_r, "initial_wait_seconds": 0.01}
        gateway = self._create_gateway_with_retry_policy(retry_policy)

        mock_action = MagicMock()
        final_error = LLMAPIError("最终失败", status_code=RETRYABLE_STATUS_CODES[0])
        mock_action.side_effect = [
            LLMAPIError("第一次失败", status_code=RETRYABLE_STATUS_CODES[0]),
            LLMAPIError("第二次失败", status_code=RETRYABLE_STATUS_CODES[1]),
            final_error # 第三次（1初试 + 2重试）仍然失败
        ]

        with self.assertRaises(LLMAPIError) as cm:
            gateway._execute_with_retry(mock_action, "test_action_fails_all_retries")

        self.assertIs(cm.exception, final_error) # 确保抛出的是最后一个异常实例
        self.assertEqual(mock_action.call_count, max_r + 1)
        self.assertEqual(mock_sleep.call_count, max_r) # 在每次失败后等待 (除了最后一次失败)

    def test_no_retry_on_non_retryable_error(self):
        """测试当发生不可重试的 LLMAPIError 时，不进行重试。"""
        retry_policy = {"max_retries": 3} # 即使允许多次重试
        gateway = self._create_gateway_with_retry_policy(retry_policy)

        mock_action = MagicMock()
        non_retryable_error = LLMAPIError("不可重试的错误", status_code=400) # 400 通常不可重试
        mock_action.side_effect = non_retryable_error

        with self.assertRaises(LLMAPIError) as cm:
            gateway._execute_with_retry(mock_action, "test_action_non_retryable")

        self.assertIs(cm.exception, non_retryable_error)
        self.assertEqual(mock_action.call_count, 1) # 只调用一次

    def test_no_retry_on_other_exceptions(self):
        """测试当发生非 LLMAPIError 的其他异常时，不进行重试。"""
        retry_policy = {"max_retries": 3}
        gateway = self._create_gateway_with_retry_policy(retry_policy)

        mock_action = MagicMock()
        other_exception = ValueError("这是一个其他类型的错误")
        mock_action.side_effect = other_exception

        with self.assertRaises(ValueError) as cm: # 捕获原始异常类型
            gateway._execute_with_retry(mock_action, "test_action_other_exception")

        self.assertIs(cm.exception, other_exception)
        self.assertEqual(mock_action.call_count, 1)

    @patch('time.sleep')
    def test_backoff_factor_and_max_wait(self, mock_sleep: MagicMock):
        """测试退避因子和最大等待时间是否按预期工作。"""
        retry_policy = {
            "max_retries": 3,
            "initial_wait_seconds": 1,
            "backoff_factor": 2,
            "max_wait_seconds": 2.5 # 设置一个较低的最大等待时间以测试上限
        }
        gateway = self._create_gateway_with_retry_policy(retry_policy)

        mock_action = MagicMock(side_effect=LLMAPIError("失败", status_code=500))

        with self.assertRaises(LLMAPIError):
            gateway._execute_with_retry(mock_action, "test_backoff_max_wait")

        self.assertEqual(mock_sleep.call_count, 3) # 3 次重试失败
        # 第一次等待: initial_wait = 1s
        self.assertAlmostEqual(mock_sleep.call_args_list[0][0][0], 1.0)
        # 第二次等待: 1 * 2^1 = 2s
        self.assertAlmostEqual(mock_sleep.call_args_list[1][0][0], 2.0)
        # 第三次等待: 1 * 2^2 = 4s, 但应被 max_wait_seconds (2.5s) 限制
        self.assertAlmostEqual(mock_sleep.call_args_list[2][0][0], 2.5)

    def test_action_succeeds_on_first_try(self):
        """测试当动作第一次就成功时，不进行重试。"""
        retry_policy = {"max_retries": 3}
        gateway = self._create_gateway_with_retry_policy(retry_policy)

        mock_action = MagicMock()
        expected_response = ProviderResponse(text_content="首次成功", model_name="test")
        mock_action.return_value = expected_response

        result = gateway._execute_with_retry(mock_action, "test_action_first_success")

        self.assertEqual(result, expected_response)
        self.assertEqual(mock_action.call_count, 1)


if __name__ == '__main__':
    unittest.main()
