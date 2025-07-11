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

from pydantic import BaseModel # Added for integration tests requiring schema definition
from aibiowflow.llm_gateway.providers.qwen_provider import QwenProvider, DASHSCOPE_API_KEY_ENV_VAR, DEFAULT_QWEN_COMPATIBLE_BASE_URL
from aibiowflow.llm_gateway.providers.base_provider import ProviderResponse
from aibiowflow.llm_gateway.exceptions import LLMAPIError, ConfigurationError

# 尝试导入真实的 OpenAI 异常，如果导入失败，则使用 MagicMock 代替
try:
    from openai import APIError, APITimeoutError, APIStatusError, APIConnectionError, RateLimitError
except ImportError:
    APIError = MagicMock # type: ignore
    APITimeoutError = MagicMock # type: ignore
    APIStatusError = MagicMock # type: ignore
    APIConnectionError = MagicMock # type: ignore
    RateLimitError = MagicMock # type: ignore

# Determine if actual OpenAI SDK is available for certain tests
OPENAI_SDK_AVAILABLE = False
try:
    import openai # Try importing the actual SDK
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    pass


class TestQwenProvider(unittest.TestCase):
    """测试 QwenProvider 的功能 (主要使用 Mock)。"""

    def setUp(self):
        """测试前的准备工作。"""
        self.original_api_key_env = os.environ.get(DASHSCOPE_API_KEY_ENV_VAR)
        os.environ[DASHSCOPE_API_KEY_ENV_VAR] = "test_qwen_api_key"

        # Mock OpenAI client for most tests
        self.mock_openai_client_instance = MagicMock()
        self.mock_chat_completions_create = MagicMock()
        self.mock_openai_client_instance.chat.completions.create = self.mock_chat_completions_create

        # Patch where OpenAI is looked up by the QwenProvider module
        self.openai_patcher = patch('aibiowflow.llm_gateway.providers.qwen_provider.OpenAI',
                                    return_value=self.mock_openai_client_instance)
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

        # 创建一个模拟的 request 对象，某些异常需要它
        mock_request = MagicMock()
        if OPENAI_SDK_AVAILABLE: # If real SDK is there, create a real (dummy) request
            try:
                import httpx
                mock_request = httpx.Request(method="POST", url="http://dummy.url/api")
            except ImportError: # Should not happen if openai is installed, as httpx is a dependency
                pass

        # 创建一个模拟的 response 对象，APIStatusError 和 RateLimitError 需要它
        mock_response_500 = MagicMock()
        mock_response_500.status_code = 500 # For APIStatusError

        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429 # For RateLimitError


        # 定义测试用例，确保为每个异常提供必要的参数
        # Note: Actual constructors for openai errors might vary slightly or evolve.
        # This setup aims for compatibility with common patterns.
        if OPENAI_SDK_AVAILABLE:
            timeout_error_instance = APITimeoutError(request=mock_request)
            # The str(APITimeoutError) is "Request timed out."
            # _handle_api_exception uses this str(e) in its message.
            timeout_message_check = "Request timed out"
        else:
            timeout_error_instance = APITimeoutError("请求超时") # MagicMock takes message
            timeout_message_check = "超时" # This is part of "Qwen API 调用超时"

        connection_error_message_check = "连接错误" # This is str(APIConnectionError(...)) and used in _handle_api_exception

        test_cases = [
            (APIStatusError("服务内部错误", response=mock_response_500, body=None), 500, "HTTP Status: 500"),
            (timeout_error_instance, None, timeout_message_check),
            (APIConnectionError(message="连接错误", request=mock_request), None, connection_error_message_check),
            (RateLimitError("速率限制", response=mock_response_429, body=None), 429, "速率限制"), # Check "速率限制" from provider's message
            (APIError("未知API错误", request=mock_request, body=None), None, "未知API错误") # Check "未知API错误" from provider's message
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
    def test_config_error_on_missing_openai_dependency(self):
        """
        测试当 openai SDK 未安装时 (即模块内的 OpenAI 变量为 None)，
        QwenProvider 初始化应抛出 ConfigurationError。
        """
        # Directly manipulate the OpenAI symbol in the imported qwen_provider module
        import aibiowflow.llm_gateway.providers.qwen_provider as qwen_provider_module

        original_openai_symbol_in_module = getattr(qwen_provider_module, 'OpenAI', 'AttributeNotPresent')
        qwen_provider_module.OpenAI = None # Set to None for this test

        original_env_val = os.environ.pop(DASHSCOPE_API_KEY_ENV_VAR, None)

        try:
            with self.assertRaisesRegex(ConfigurationError, "openai 包未安装"):
                # QwenProvider will be instantiated using the qwen_provider_module.OpenAI which is now None
                QwenProvider()
        finally:
            if original_env_val is not None:
                os.environ[DASHSCOPE_API_KEY_ENV_VAR] = original_env_val

            # Restore the OpenAI symbol in the qwen_provider module
            if original_openai_symbol_in_module != 'AttributeNotPresent':
                qwen_provider_module.OpenAI = original_openai_symbol_in_module
            else: # If it wasn't there before (should not happen if module loaded)
                if hasattr(qwen_provider_module, 'OpenAI'):
                    delattr(qwen_provider_module, 'OpenAI')

            # Optional: reload to be absolutely sure, though direct restoration should be enough
            # import importlib
            # importlib.reload(qwen_provider_module)


# --- Integration Tests ---
# These tests will make actual API calls to Qwen.
# They will only run if DASHSCOPE_API_KEY is set in the environment
# and RUN_QWEN_INTEGRATION_TESTS is set to "true".

RUN_INTEGRATION_TESTS = os.getenv('RUN_QWEN_INTEGRATION_TESTS', 'false').lower() == 'true'
API_KEY_IS_SET = DASHSCOPE_API_KEY_ENV_VAR in os.environ

@unittest.skipUnless(RUN_INTEGRATION_TESTS and API_KEY_IS_SET and OPENAI_SDK_AVAILABLE,
                     f"Skipping Qwen integration tests. Requirements: RUN_QWEN_INTEGRATION_TESTS=true, "
                     f"{DASHSCOPE_API_KEY_ENV_VAR} set, and openai SDK installed.")
class TestQwenProviderIntegration(unittest.TestCase):
    """
    Integration tests for QwenProvider.
    These tests will make actual API calls to the Qwen service.
    Ensure DASHSCOPE_API_KEY environment variable is set with a valid key
    and RUN_QWEN_INTEGRATION_TESTS is set to 'true' to run these tests.
    Example:
    DASHSCOPE_API_KEY="sk-yourkey" RUN_QWEN_INTEGRATION_TESTS="true" python -m unittest aibiowflow/tests/test_qwen_provider.py
    """
    def setUp(self):
        """Set up for integration tests."""
        self.api_key = os.getenv(DASHSCOPE_API_KEY_ENV_VAR)
        if not self.api_key: # Should be caught by skipUnless, but as a safeguard
            self.fail(f"{DASHSCOPE_API_KEY_ENV_VAR} not set for integration tests.")

        # It's good practice to ensure the API key being used is the one provided by the user
        # For this task, the user provided sk-f0a749a993ba42769278527d8d5159ec
        # We can add a check or assume the environment is set correctly by the user running the test.
        # For now, we'll assume it's correctly set.

        try:
            self.provider = QwenProvider(api_key=self.api_key)
        except ConfigurationError as e:
            self.fail(f"Failed to initialize QwenProvider for integration tests: {e}."
                      "Ensure 'openai' SDK is installed.")
        except LLMAPIError as e:
            self.fail(f"Failed to initialize QwenProvider due to API key or connection issue: {e}")
        self.default_model = "qwen-plus" # A common model for testing
        self.turbo_model = "qwen-turbo"

    def test_generate_text_simple_call(self):
        """Test a simple generate_text call with a common model."""
        try:
            response = self.provider.generate_text(
                prompt="你好，通义千问！请写一句关于AI的短语。",
                model_name=self.default_model,
                temperature=0.7,
                max_tokens=50
            )
            self.assertIsInstance(response, ProviderResponse)
            self.assertIsNotNone(response.text_content)
            self.assertTrue(len(response.text_content.strip()) > 0)
            self.assertEqual(response.model_name, self.default_model)
            # print(f"\n[Integration Test] Simple Call Response: {response.text_content}")
            # print(f"[Integration Test] Usage: In={response.input_tokens}, Out={response.output_tokens}")
        except LLMAPIError as e:
            self.fail(f"generate_text_simple_call failed with LLMAPIError: {e}")
        except Exception as e_gen: # Catch any other unexpected errors during the API call
            self.fail(f"generate_text_simple_call failed with an unexpected exception: {e_gen}")


    def test_generate_text_with_system_prompt(self):
        """Test generate_text with a system prompt."""
        try:
            response = self.provider.generate_text(
                prompt="Describe quantum entanglement in one sentence.",
                model_name=self.default_model,
                system_prompt="You are a helpful physics assistant. Explain complex topics simply.",
                max_tokens=100
            )
            self.assertIsInstance(response, ProviderResponse)
            self.assertTrue(len(response.text_content.strip()) > 0)
            # print(f"\n[Integration Test] System Prompt Response: {response.text_content}")
        except LLMAPIError as e:
            self.fail(f"test_generate_text_with_system_prompt failed: {e}")

    def test_generate_text_with_max_tokens(self):
        """Test generate_text respects max_tokens (approximately)."""
        # Note: Token count is not character count. This is an approximate check.
        # A more precise check would involve a tokenizer, which is out of scope for the provider.
        try:
            response = self.provider.generate_text(
                prompt="Tell me a very long story about a robot.",
                model_name=self.turbo_model, # Use a faster model
                max_tokens=10 # Very small max_tokens
            )
            self.assertIsInstance(response, ProviderResponse)
            self.assertTrue(len(response.text_content.strip()) > 0)
            # A rough check: the number of words should be small.
            # This isn't foolproof as tokens != words.
            self.assertTrue(len(response.text_content.split()) < 30) # Heuristic
            if response.output_tokens: # If API returns output_tokens
                self.assertLessEqual(response.output_tokens, 10 + 5) # Allow some leeway for token counting differences
            # print(f"\n[Integration Test] Max Tokens Response (limit 10): {response.text_content}")
        except LLMAPIError as e:
            self.fail(f"test_generate_text_with_max_tokens failed: {e}")

    def test_generate_text_with_message_override(self):
        """Test generate_text with messages_override."""
        messages = [
            {"role": "system", "content": "You are a friendly chatbot."},
            {"role": "user", "content": "Hello! What's your name?"},
            {"role": "assistant", "content": "I am a friendly chatbot created by OpenAI (compatible API)."},
            {"role": "user", "content": "What can you do?"}
        ]
        try:
            response = self.provider.generate_text(
                prompt="This prompt is ignored.", # Should be ignored
                model_name=self.turbo_model,
                messages_override=messages,
                max_tokens=50
            )
            self.assertIsInstance(response, ProviderResponse)
            self.assertTrue(len(response.text_content.strip()) > 0)
            # print(f"\n[Integration Test] Message Override Response: {response.text_content}")
        except LLMAPIError as e:
            self.fail(f"test_generate_text_with_message_override failed: {e}")


    def test_generate_text_token_usage(self):
        """Test that token usage is populated."""
        try:
            response = self.provider.generate_text(
                prompt="What is the capital of France?",
                model_name=self.turbo_model,
                max_tokens=20
            )
            self.assertIsInstance(response, ProviderResponse)
            self.assertTrue(len(response.text_content.strip()) > 0)
            self.assertIsNotNone(response.input_tokens, "Input tokens should be populated if usage is returned.")
            self.assertIsNotNone(response.output_tokens, "Output tokens should be populated if usage is returned.")
            self.assertTrue(response.input_tokens > 0)
            self.assertTrue(response.output_tokens > 0)
            # print(f"\n[Integration Test] Token Usage: Input={response.input_tokens}, Output={response.output_tokens}")
        except LLMAPIError as e:
            self.fail(f"test_generate_text_token_usage failed: {e}")


    def test_generate_structured_text_simple_json(self):
        """Test generate_structured_text for simple JSON output."""
        class UserProfile(BaseModel): # type: ignore
            name: str
            age: int
            city: str

        # It is important to use a model that is good at following JSON instructions.
        # qwen-plus and above are generally better for this.
        # The prompt must explicitly ask for JSON.
        prompt_text = f"""
        Generate a JSON object for a user profile with the following details:
        Name: Alice
        Age: 30
        City: New York

        The JSON object must conform to the following Pydantic schema:
        {UserProfile.model_json_schema()}
        Ensure the output is only the JSON object itself, with no surrounding text.
        """
        try:
            response = self.provider.generate_structured_text(
                prompt=prompt_text,
                model_name=self.default_model, # qwen-plus or better recommended
                output_schema=UserProfile,
                temperature=0.1, # Low temperature for more deterministic JSON
                max_tokens=150
            )
            self.assertIsInstance(response, ProviderResponse)
            self.assertTrue(len(response.text_content.strip()) > 0)
            # print(f"\n[Integration Test] Structured Text (JSON) Raw Response: {response.text_content}")

            try:
                import json
                json_output = json.loads(response.text_content)
                self.assertIsInstance(json_output, dict)
                self.assertIn("name", json_output)
                self.assertIn("age", json_output)
                self.assertIn("city", json_output)
                self.assertEqual(json_output["name"], "Alice")
                self.assertEqual(json_output["age"], 30)
                # City can sometimes vary slightly if model is not perfectly constrained.
                # For stricter test, one might need to validate against the Pydantic model directly.
                # validated_data = UserProfile(**json_output)
            except json.JSONDecodeError:
                self.fail(f"Failed to parse JSON output: {response.text_content}")
            except Exception as p_exc: # Catch Pydantic validation error if we use it
                self.fail(f"Pydantic validation failed for JSON output: {p_exc}. Output was: {response.text_content}")

        except LLMAPIError as e:
            self.fail(f"test_generate_structured_text_simple_json failed: {e}")

    def test_error_non_existent_model(self):
        """Test API error for a non-existent model name."""
        with self.assertRaises(LLMAPIError) as context:
            self.provider.generate_text(
                prompt="This should fail.",
                model_name="qwen-non-existent-model-xyz"
            )
        # Check for specific error details if possible, e.g., status code or message part
        # Qwen/Dashscope might return a 400 or 404 for invalid model, or a specific error code in body
        # Example: if it's an APIStatusError from openai sdk, it might have a status_code
        if hasattr(context.exception, 'status_code') and context.exception.status_code:
             self.assertIn(context.exception.status_code, [400, 404, 422]) # Common error codes
        self.assertIn("model", str(context.exception).lower()) # Error message should mention the model
        # print(f"\n[Integration Test] Non-existent model error: {context.exception}")


    def test_generate_text_qwen_turbo(self):
        """Test a simple call with qwen-turbo."""
        try:
            response = self.provider.generate_text(
                prompt="What is 1 + 1?",
                model_name=self.turbo_model,
                max_tokens=10
            )
            self.assertIsInstance(response, ProviderResponse)
            self.assertTrue(len(response.text_content.strip()) > 0)
            self.assertIn("2", response.text_content)
            # print(f"\n[Integration Test] qwen-turbo response: {response.text_content}")
        except LLMAPIError as e:
            self.fail(f"test_generate_text_qwen_turbo failed: {e}")

    @unittest.skipIf(os.getenv("QWEN_SKIP_MAX_MODEL_TEST", "false").lower() == "true",
                     "Skipping qwen-max model test as QWEN_SKIP_MAX_MODEL_TEST is set.")
    def test_generate_text_qwen_max_if_available(self):
        """
        Test a simple call with qwen-max.
        This test can be skipped by setting QWEN_SKIP_MAX_MODEL_TEST=true if qwen-max is too slow/costly.
        """
        qwen_max_model = "qwen-max" # or "qwen-max-longcontext" if specifically testing that
        try:
            response = self.provider.generate_text(
                prompt="Briefly, what is the concept of general relativity?",
                model_name=qwen_max_model,
                max_tokens=100
            )
            self.assertIsInstance(response, ProviderResponse)
            self.assertTrue(len(response.text_content.strip()) > 0)
            # print(f"\n[Integration Test] {qwen_max_model} response: {response.text_content}")
        except LLMAPIError as e:
            # Qwen-max might not be available to all keys or might have stricter quotas
            # So, we check for specific error messages that might indicate this.
            # The Dashscope error for "Permission Denied" or "charge type is postpaid" is often http 400 with code "InvalidParameter"
            # or http 403 with "NoPermission"
            if e.status_code == 400 and "InvalidParameter" in str(e.original_exception): # type: ignore
                self.skipTest(f"{qwen_max_model} may not be enabled or available for this API key (InvalidParameter): {e}")
            elif e.status_code == 403 and "NoPermission" in str(e.original_exception): # type: ignore
                 self.skipTest(f"{qwen_max_model} may not be enabled or available for this API key (NoPermission): {e}")
            elif "model not found" in str(e).lower() or \
                 (hasattr(e, 'original_exception') and e.original_exception and "Resource not found" in str(e.original_exception)): # type: ignore
                 self.skipTest(f"{qwen_max_model} not found or not accessible with this key: {e}")
            else:
                self.fail(f"test_generate_text_{qwen_max_model} failed: {e}")
        except Exception as e_gen:
            self.fail(f"test_generate_text_{qwen_max_model} failed with an unexpected exception: {e_gen}")


if __name__ == '__main__':
    # This allows running tests directly, e.g.,
    # python -m unittest aibiowflow/tests/test_qwen_provider.py
    # To run integration tests:
    # DASHSCOPE_API_KEY="your_key" RUN_QWEN_INTEGRATION_TESTS="true" python -m unittest aibiowflow/tests/test_qwen_provider.py
    unittest.main()
