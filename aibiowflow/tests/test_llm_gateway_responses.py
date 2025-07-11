# -*- coding: utf-8 -*-
"""
LLM 网关获取文本和结构化响应的单元测试。
包括 Provider 交互模拟和 Pydantic 验证。
"""
import unittest
import os
import time
from unittest.mock import patch, MagicMock, call
from pydantic import BaseModel, ValidationError as PydanticValidationError # 重命名以避免与本地 ValidationError 冲突

from aibiowflow.llm_gateway.gateway import LLMGateway
from aibiowflow.llm_gateway.exceptions import LLMAPIError, LLMOutputValidationError, PromptTemplateError
from aibiowflow.llm_gateway.providers.base_provider import BaseProvider, ProviderResponse

# --- 定义用于测试的 Pydantic 模型 ---
class SimpleSchema(BaseModel):
    name: str
    age: int

class ComplexSchema(BaseModel):
    item_id: str
    details: SimpleSchema
    tags: list[str]

class TestGatewayResponses(unittest.TestCase):
    """测试 LLMGateway 的 get_text_response 和 get_structured_response 方法。"""

    def setUp(self):
        """测试前准备。"""
        self.temp_dir = "temp_test_responses_dir"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.config_path = os.path.join(self.temp_dir, "responses_config.yaml")
        self.prompts_path = os.path.join(self.temp_dir, "responses_prompts.toml")

        # 模拟 Provider
        self.mock_provider = MagicMock(spec=BaseProvider)
        self.mock_provider.provider_name = "mock_provider"

        # 默认配置和 prompts
        self.config_data = {
            "api_keys": {"mock_provider": "dummy_key"},
            "models": [{
                "name": "mock-model-default", "provider": "mock_provider", "aliases": ["default_alias"]
            },{
                "name": "mock-model-text", "provider": "mock_provider", "aliases": ["text_alias"]
            },{
                "name": "mock-model-struct", "provider": "mock_provider", "aliases": ["struct_alias"]
            }],
            "retry_policy": {"max_retries": 1, "initial_wait_seconds": 0.01} # 快速测试重试
        }
        self.prompts_data = {
            "get_greeting": {"template": "你好, {name}!"},
            "get_simple_data": {"template": "获取关于 {item} 的简单数据。"},
            "get_complex_data": {"template": "获取关于 {id} 的复杂数据，包含 {sub_item}。"}
        }
        self._write_config(self.config_data)
        self._write_prompts(self.prompts_data)

        # Patch _init_providers to return our mock provider
        self.init_providers_patcher = patch(
            'aibiowflow.llm_gateway.gateway.LLMGateway._init_providers',
            return_value={"mock_provider": self.mock_provider}
        )
        self.mock_init_providers = self.init_providers_patcher.start()

        self.gateway = LLMGateway(self.config_path, self.prompts_path)


    def tearDown(self):
        """测试后清理。"""
        self.init_providers_patcher.stop() # 停止 patch
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.prompts_path):
            os.remove(self.prompts_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def _write_config(self, data: dict):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            import yaml
            yaml.dump(data, f)

    def _write_prompts(self, data: dict):
        with open(self.prompts_path, 'w', encoding='utf-8') as f:
            import toml
            toml.dump(data, f)

    # --- 测试 get_text_response ---
    def test_get_text_response_success(self):
        """测试 get_text_response 成功路径。"""
        expected_text = "模拟的文本响应"
        self.mock_provider.generate_text.return_value = ProviderResponse(
            text_content=expected_text, input_tokens=10, output_tokens=20, model_name="mock-model-text"
        )

        response = self.gateway.get_text_response(
            prompt_name="get_greeting",
            context={"name": "测试用户"},
            model_alias="text_alias",
            temperature=0.5 # provider_kwarg
        )
        self.assertEqual(response, expected_text)
        self.mock_provider.generate_text.assert_called_once_with(
            prompt="你好, 测试用户!",
            model_name="mock-model-text",
            temperature=0.5 # 检查 provider_kwarg 是否传递
        )

    def test_get_text_response_api_error_after_retries(self):
        """测试 get_text_response 在多次重试后仍发生 LLMAPIError。"""
        self.mock_provider.generate_text.side_effect = LLMAPIError("模拟API错误", status_code=500)

        with self.assertRaisesRegex(LLMAPIError, "模拟API错误"):
            self.gateway.get_text_response(
                prompt_name="get_greeting",
                context={"name": "错误测试"},
                model_alias="text_alias"
            )
        # 默认 max_retries=1 (来自setUp) + 1 初次尝试 = 2 次调用
        self.assertEqual(self.mock_provider.generate_text.call_count, 1 + self.gateway.max_retries)

    def test_get_text_response_prompt_template_error(self):
        """测试 get_text_response 中因 prompt 格式化错误抛出 PromptTemplateError。"""
        with self.assertRaises(PromptTemplateError):
            self.gateway.get_text_response(
                prompt_name="get_greeting", # 需要 'name'
                context={}, # 'name' 缺失
                model_alias="text_alias"
            )

    # --- 测试 get_structured_response ---
    def test_get_structured_response_success(self):
        """测试 get_structured_response 成功路径，包括 Pydantic 验证。"""
        valid_json_str = '{"name": "Alice", "age": 30}'
        self.mock_provider.generate_structured_text.return_value = ProviderResponse(
            text_content=valid_json_str, input_tokens=15, output_tokens=25, model_name="mock-model-struct"
        )

        response_obj = self.gateway.get_structured_response(
            prompt_name="get_simple_data",
            context={"item": "Alice"},
            output_schema=SimpleSchema,
            model_alias="struct_alias",
            temperature=0.1 # provider_kwarg
        )
        self.assertIsInstance(response_obj, SimpleSchema)
        self.assertEqual(response_obj.name, "Alice")
        self.assertEqual(response_obj.age, 30)
        self.mock_provider.generate_structured_text.assert_called_once_with(
            prompt="获取关于 Alice 的简单数据。",
            model_name="mock-model-struct",
            output_schema=SimpleSchema,
            temperature=0.1
        )

    def test_get_structured_response_invalid_json_from_provider(self):
        """测试当 provider 返回无效 JSON 时抛出 LLMOutputValidationError。"""
        invalid_json_str = '{"name": "Bob", "age": "not_an_int"' # age 应该是 int
        self.mock_provider.generate_structured_text.return_value = ProviderResponse(
            text_content=invalid_json_str, model_name="mock-model-struct"
        )

        with self.assertRaises(LLMOutputValidationError) as cm:
            self.gateway.get_structured_response(
                prompt_name="get_simple_data",
                context={"item": "Bob"},
                output_schema=SimpleSchema,
                model_alias="struct_alias"
            )
        # 检查异常是否包含 Pydantic 的验证错误信息
        self.assertIn("Input should be a valid integer", str(cm.exception.validation_errors)) # Pydantic v2
        # 对于 Pydantic v1, 错误信息可能不同，例如 "value is not a valid integer"

    def test_get_structured_response_json_not_matching_schema(self):
        """测试当 provider 返回的 JSON 结构不匹配 schema 时抛出 LLMOutputValidationError。"""
        json_wrong_schema = '{"full_name": "Charlie", "years": 40}' # 字段名不匹配 SimpleSchema
        self.mock_provider.generate_structured_text.return_value = ProviderResponse(
            text_content=json_wrong_schema, model_name="mock-model-struct"
        )
        with self.assertRaises(LLMOutputValidationError) as cm:
            self.gateway.get_structured_response(
                prompt_name="get_simple_data",
                context={"item": "Charlie"},
                output_schema=SimpleSchema,
                model_alias="struct_alias"
            )
        # Pydantic v2: "Field required" for 'name' and 'age', "Unexpected keyword argument" for 'full_name', 'years'
        # Pydantic v1: "field required" for 'name' and 'age'
        error_str = str(cm.exception.validation_errors).lower() # 使检查不区分大小写
        self.assertTrue("name" in error_str and "field required" in error_str or "missing" in error_str)
        self.assertTrue("age" in error_str and "field required" in error_str or "missing" in error_str)


    def test_get_structured_response_provider_returns_non_json_string(self):
        """测试当 provider 返回的不是 JSON 字符串时抛出 LLMOutputValidationError。"""
        non_json_string = "这完全不是一个 JSON 对象。"
        self.mock_provider.generate_structured_text.return_value = ProviderResponse(
            text_content=non_json_string, model_name="mock-model-struct"
        )
        with self.assertRaisesRegex(LLMOutputValidationError, "LLM 输出不是有效的 JSON。"):
            self.gateway.get_structured_response(
                prompt_name="get_simple_data",
                context={"item": "NonJson"},
                output_schema=SimpleSchema,
                model_alias="struct_alias"
            )

    def test_get_structured_response_cleans_markdown_json_block(self):
        """测试结构化响应是否能清理常见的 markdown JSON 代码块。"""
        json_with_markdown = '```json\n{"name": "Dave", "age": 25}\n```'
        expected_obj = SimpleSchema(name="Dave", age=25)
        self.mock_provider.generate_structured_text.return_value = ProviderResponse(
            text_content=json_with_markdown, model_name="mock-model-struct"
        )

        response_obj = self.gateway.get_structured_response(
            prompt_name="get_simple_data",
            context={"item": "Dave"},
            output_schema=SimpleSchema,
            model_alias="struct_alias"
        )
        self.assertEqual(response_obj, expected_obj)

    @patch('aibiowflow.llm_gateway.gateway.logger') # 模拟 logger 以检查日志输出
    def test_logging_on_success_and_failure(self, mock_logger):
        """测试成功和失败时的日志记录。"""
        # 成功情况 (文本)
        self.mock_provider.generate_text.return_value = ProviderResponse("成功文本", 5, 10, 0.001, "mock-model-text")
        self.gateway.get_text_response("get_greeting", {"name": "日志测试"}, "text_alias")

        # 检查 mock_logger.info 和 mock_logger.debug 是否被调用 (具体内容检查较复杂，可抽样)
        # _log_request_details 会调用 logger.info (成功) 或 logger.error (失败)
        # 以及 logger.debug (详细JSON)
        info_calls = [c for c in mock_logger.method_calls if c[0] == 'info']
        debug_calls = [c for c in mock_logger.method_calls if c[0] == 'debug']

        self.assertTrue(any("LLM请求" in str(call_args) and "成功" in str(call_args) for call_args in info_calls))
        self.assertTrue(any("详细日志条目" in str(call_args) for call_args in debug_calls))

        mock_logger.reset_mock() # 为下一次测试重置 mock

        # 失败情况 (结构化, API 错误)
        self.mock_provider.generate_structured_text.side_effect = LLMAPIError("模拟API故障", status_code=503)
        with self.assertRaises(LLMAPIError):
            self.gateway.get_structured_response(
                "get_simple_data", {"item": "错误日志"}, SimpleSchema, "struct_alias"
            )

        error_calls = [c for c in mock_logger.method_calls if c[0] == 'error']
        debug_calls_failure = [c for c in mock_logger.method_calls if c[0] == 'debug']

        self.assertTrue(any("LLM请求" in str(call_args) and "失败 (LLMAPIError)" in str(call_args) for call_args in error_calls))
        self.assertTrue(any("详细日志条目" in str(call_args) and '"status": "failure"' in str(call_args) for call_args in debug_calls_failure))


if __name__ == '__main__':
    unittest.main()
