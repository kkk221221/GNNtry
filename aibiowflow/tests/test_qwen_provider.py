# -*- coding: utf-8 -*-
"""
针对 QwenProvider 的单元测试。
"""
import unittest
import os
from unittest.mock import patch, MagicMock, ANY

# 模拟 openai 在 QwenProvider 导入之前，以确保即使 openai 未安装，测试文件也能被加载
# 这主要用于测试 QwenProvider 在 openai 未安装时的 ConfigurationError
mock_openai_sdk = MagicMock()
# 如果 openai 模块中还有其他在 QwenProvider 中直接访问的属性或类，也需要模拟
# 例如: mock_openai_sdk.OpenAI, mock_openai_sdk.APIError 等

# 在真正导入 QwenProvider 之前，根据需要决定是否 patch 'openai' 模块
# For tests that assume openai is installed, we don't need to patch it here.
# For tests that check for ConfigurationError if openai is not installed, we would.

from aibiowflow.llm_gateway.providers.qwen_provider import QwenProvider, DASHSCOPE_API_KEY_ENV_VAR, DEFAULT_QWEN_COMPATIBLE_BASE_URL
from aibiowflow.llm_gateway.providers.base_provider import ProviderResponse
from aibiowflow.llm_gateway.exceptions import LLMAPIError, ConfigurationError

# 尝试导入真实的 OpenAI 异常，如果导入失败，则使用 MagicMock 代替
try:
    from openai import APIError, APITimeoutError, APIStatusError
except ImportError:
    APIError = MagicMock
    APITimeoutError = MagicMock
    APIStatusError = MagicMock


class TestQwenProvider(unittest.TestCase):
    """测试 QwenProvider 的功能。"""

    def setUp(self):
        """测试前的准备工作。"""
        self.original_api_key_env = os.environ.get(DASHSCOPE_API_KEY_ENV_VAR)
        os.environ[DASHSCOPE_API_KEY_ENV_VAR] = "test_qwen_api_key"

        # Mock OpenAI client for most tests
        self.mock_openai_client_instance = MagicMock()
        self.mock_chat_completions_create = MagicMock()
        self.mock_openai_client_instance.chat.completions.create = self.mock_chat_completions_create

        self.openai_patcher = patch('openai.OpenAI', return_value=self.mock_openai_client_instance)
        self.mock_openai_constructor = self.openai_patcher.start()


    def tearDown(self):
        """测试后的清理工作。"""
        self.openai_patcher.stop()
        if self.original_api_key_env is None:
            if DASHSCOPE_API_KEY_ENV_VAR in os.environ:
                del os.environ[DASHSCOPE_API_KEY_ENV_VAR]
        else:
            os.environ[DASHSCOPE_API_KEY_ENV_VAR] = self.original_api_key_env

    def test_initialization_success_with_env_var(self):
        """测试当 API 密钥通过环境变量提供时，Provider 初始化成功。"""
        provider = QwenProvider()
        self.assertIsNotNone(provider.client)
        self.mock_openai_constructor.assert_called_once_with(
            api_key="test_qwen_api_key",
            base_url=DEFAULT_QWEN_COMPATIBLE_BASE_URL
        )

    def test_initialization_success_with_direct_api_key(self):
        """测试当 API 密钥直接传递时，Provider 初始化成功。"""
        del os.environ[DASHSCOPE_API_KEY_ENV_VAR] # 确保环境变量不存在
        provider = QwenProvider(api_key="direct_api_key")
        self.assertIsNotNone(provider.client)
        self.mock_openai_constructor.assert_called_once_with(
            api_key="direct_api_key",
            base_url=DEFAULT_QWEN_COMPATIBLE_BASE_URL
        )

    def test_initialization_failure_no_api_key(self):
        """测试当 API 密钥既未直接提供也未在环境变量中时，初始化失败。"""
        if DASHSCOPE_API_KEY_ENV_VAR in os.environ:
            del os.environ[DASHSCOPE_API_KEY_ENV_VAR]
        with self.assertRaisesRegex(LLMAPIError, "Qwen API 密钥未提供"):
            QwenProvider()

    def test_initialization_with_custom_config(self):
        """测试使用自定义 provider_config 初始化 Provider。"""
        custom_base_url = "https://custom.qwen.api/v1"
        custom_default_kwargs = {"top_k": 50}
        provider_config = {
            "base_url": custom_base_url,
            "default_model_kwargs": custom_default_kwargs
        }
        provider = QwenProvider(provider_config=provider_config)
        self.assertEqual(provider.base_url, custom_base_url)
        self.assertEqual(provider.default_model_kwargs, custom_default_kwargs)
        self.mock_openai_constructor.assert_called_with(
            api_key="test_qwen_api_key", # From env var
            base_url=custom_base_url
        )

    def test_generate_text_success(self):
        """测试 generate_text 方法成功调用。"""
        provider = QwenProvider()
        mock_completion_response = MagicMock()
        mock_completion_response.choices = [MagicMock(message=MagicMock(content="你好，世界"))]
        mock_completion_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_completion_response.model_dump.return_value = {"id": "fake_id", "choices": [{"message": {"content": "你好，世界"}}], "usage": {"prompt_tokens":10, "completion_tokens":5}}


        self.mock_chat_completions_create.return_value = mock_completion_response

        response = provider.generate_text(
            prompt="打个招呼",
            model_name="qwen-plus",
            temperature=0.5,
            max_tokens=50,
            system_prompt="你是助手"
        )

        self.assertIsInstance(response, ProviderResponse)
        self.assertEqual(response.text_content, "你好，世界")
        self.assertEqual(response.input_tokens, 10)
        self.assertEqual(response.output_tokens, 5)
        self.assertEqual(response.model_name, "qwen-plus")
        self.assertIsNotNone(response.raw_response)

        self.mock_chat_completions_create.assert_called_once_with(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是助手"},
                {"role": "user", "content": "打个招呼"}
            ],
            temperature=0.5,
            max_tokens=50
        )

    def test_generate_text_with_messages_override(self):
        """测试 generate_text 使用 messages_override 参数。"""
        provider = QwenProvider()
        custom_messages = [
            {"role": "system", "content": "Custom System"},
            {"role": "user", "content": "Custom User Prompt"}
        ]
        mock_completion_response = MagicMock()
        mock_completion_response.choices = [MagicMock(message=MagicMock(content="自定义响应"))]
        mock_completion_response.usage = None
        mock_completion_response.model_dump.return_value = {}
        self.mock_chat_completions_create.return_value = mock_completion_response

        provider.generate_text(
            prompt="这个会被忽略", # Should be ignored
            model_name="qwen-turbo",
            messages_override=custom_messages
        )
        self.mock_chat_completions_create.assert_called_once_with(
            model="qwen-turbo",
            messages=custom_messages,
            temperature=ANY # Default or from provider_config
        )


    def test_generate_structured_text_success(self):
        """测试 generate_structured_text 方法成功调用。"""
        provider = QwenProvider()
        expected_json_str = '{"key": "value", "number": 123}'
        mock_completion_response = MagicMock()
        mock_completion_response.choices = [MagicMock(message=MagicMock(content=expected_json_str))]
        mock_completion_response.usage = None #  Assume no usage info for simplicity here
        mock_completion_response.model_dump.return_value = {}

        self.mock_chat_completions_create.return_value = mock_completion_response

        class MySchema(BaseModel): # type: ignore
            key: str
            number: int

        response = provider.generate_structured_text(
            prompt="生成JSON",
            model_name="qwen-max",
            output_schema=MySchema,
            temperature=0.1
        )
        self.assertEqual(response.text_content, expected_json_str)
        self.mock_chat_completions_create.assert_called_once_with(
            model="qwen-max",
            messages=[{"role": "user", "content": "生成JSON"}],
            temperature=0.1,
            response_format={"type": "json_object"} # Crucial for structured output
        )

    def test_api_error_handling(self):
        """测试不同 OpenAI API 错误被正确包装为 LLMAPIError。"""
        provider = QwenProvider()

        test_cases = [
            (APIStatusError("服务内部错误", response=MagicMock(status_code=500), body=None), 500, "HTTP Status: 500"),
            (APITimeoutError("请求超时"), None, "超时"),
            (APIConnectionError("连接错误"), None, "连接失败"),
            (APIError("未知API错误"), None, "未知API错误") # Generic APIError
        ]

        for original_exception, expected_status_code, part_of_message in test_cases:
            self.mock_chat_completions_create.side_effect = original_exception
            with self.subTest(exception_type=type(original_exception).__name__):
                with self.assertRaises(LLMAPIError) as cm:
                    provider.generate_text(prompt="test", model_name="qwen-plus")

                self.assertIsInstance(cm.exception, LLMAPIError)
                self.assertEqual(cm.exception.status_code, expected_status_code)
                self.assertIn(part_of_message, str(cm.exception))
                self.assertIs(cm.exception.original_exception, original_exception)
            self.mock_chat_completions_create.reset_mock() # Reset side_effect for next subtest

    def test_provider_name(self):
        """测试 provider_name 属性是否正确。"""
        provider = QwenProvider()
        self.assertEqual(provider.provider_name, "qwen")

    # 测试openai SDK未安装的情况
    @patch('builtins.__import__', side_effect=ImportError("No module named 'openai'"))
    def test_initialization_openai_not_installed(self, mock_import):
        """
        测试当 openai SDK 未安装时，QwenProvider 初始化应抛出 ConfigurationError。
        注意：这个测试需要确保 QwenProvider 的导入发生在 patch 之后。
        为了确保这一点，我们可能需要将 QwenProvider 的导入移到测试方法内部或使用更复杂的导入劫持。
        一个简单的方法是重新加载模块，但更干净的是在模块级别进行patch，或者确保测试运行器隔离导入。
        这里我们假设测试执行顺序和patch能正确工作，或者在项目结构上，这个测试文件可能不会直接导入QwenProvider顶层。
        """
        # 卸载已加载的 QwenProvider (如果存在)，以便重新导入时触发 ImportError
        import sys
        if 'aibiowflow.llm_gateway.providers.qwen_provider' in sys.modules:
            del sys.modules['aibiowflow.llm_gateway.providers.qwen_provider']

        with self.assertRaisesRegex(ConfigurationError, "openai 包未安装"):
            from aibiowflow.llm_gateway.providers.qwen_provider import QwenProvider as QwenProviderLocal
            QwenProviderLocal()

        # 恢复正常的导入行为，以免影响其他测试
        mock_import.side_effect = __import__
        if 'aibiowflow.llm_gateway.providers.qwen_provider' in sys.modules:
             del sys.modules['aibiowflow.llm_gateway.providers.qwen_provider']


if __name__ == '__main__':
    unittest.main()
