# -*- coding: utf-8 -*-
"""
LLM 网关配置加载相关的单元测试。

这些测试验证 `LLMGateway` 类是否能够正确加载和解析
`config.yaml` 和 `prompts.toml` 文件，包括处理各种有效和无效的场景。
"""
import unittest
import os
import re # For escaping paths in regex
import yaml # 用于写入测试用的 YAML 文件
import toml # 用于写入测试用的 TOML 文件
from unittest.mock import patch # 用于模拟 gateway._init_providers

from aibiowflow.llm_gateway.gateway import LLMGateway
from aibiowflow.llm_gateway.exceptions import ConfigurationError

# 如果测试中需要用到 Pydantic 模型（例如，测试与 schema 相关的配置项），则导入
# from pydantic import BaseModel

class TestConfigLoading(unittest.TestCase):
    """
    测试 `LLMGateway` 的配置加载功能，包括：
    - 成功加载有效的配置文件和 Prompt 文件。
    - 正确处理环境变量替换（用于 API 密钥）。
    - 模型定义和别名映射的正确性。
    - 重试策略参数的解析。
    - 对缺失或格式错误的配置文件的错误处理。
    """

    def setUp(self):
        """
        每个测试方法运行前的准备工作。
        创建临时的目录和文件路径，用于存放测试用的配置文件。
        同时清理可能由其他测试遗留的环境变量。
        """
        self.temp_dir = "temp_test_config_dir_for_gateway" # 使用更独特的目录名
        os.makedirs(self.temp_dir, exist_ok=True)
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.prompts_path = os.path.join(self.temp_dir, "test_prompts.toml")

        # 清理可能由先前测试设置的环境变量，确保测试的隔离性
        self.test_env_var_name = "TEST_API_KEY_CONFIG_GW" # 定义一个测试专用的环境变量名
        if self.test_env_var_name in os.environ:
            del os.environ[self.test_env_var_name]

    def tearDown(self):
        """
        每个测试方法运行后的清理工作。
        删除创建的临时文件和目录，并恢复/清理环境变量。
        """
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.prompts_path):
            os.remove(self.prompts_path)
        if self.test_env_var_name in os.environ: # 确保环境变量被清理
            del os.environ[self.test_env_var_name]
        if os.path.exists(self.temp_dir):
            # 确保目录为空后才能删除，或者使用 shutil.rmtree
            try:
                os.rmdir(self.temp_dir)
            except OSError: # 如果目录非空（例如其他文件意外创建），打印警告并强制删除
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                print(f"警告: 测试目录 {self.temp_dir} 在 tearDown 时非空，已强制删除。")

    def _write_config(self, data: dict):
        """辅助函数：将给定的字典数据写入临时的 YAML 配置文件。"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)

    def _write_prompts(self, data: dict):
        """辅助函数：将给定的字典数据写入临时的 TOML prompts 文件。"""
        with open(self.prompts_path, 'w', encoding='utf-8') as f:
            toml.dump(data, f)

    def test_successful_config_loading(self):
        """测试场景：成功加载一个结构完整且有效的配置文件和 prompts 文件。"""
        config_data = {
            "api_keys": {"google": f"env:{self.test_env_var_name}"}, # API 密钥从环境变量读取
            "models": [{
                "name": "test-model-gemini",
                "provider": "google", # 假设 'google' 是 GeminiProvider.PROVIDER_ID
                "aliases": ["test_alias_gw", "default_gw_model"]
            }],
            "retry_policy": {"max_retries": 5, "backoff_factor": 1.5, "initial_wait_seconds": 0.1}
        }
        prompts_data = {"sample_prompt": {"template": "你好, {user}!"}}

        # 设置测试用的环境变量
        os.environ[self.test_env_var_name] = "actual_test_api_key_value"
        self._write_config(config_data)
        self._write_prompts(prompts_data)

        # 关键：模拟 LLMGateway._init_providers 方法，因为我们在此处不测试 Provider 的实际初始化逻辑，
        # 只测试配置是否被正确加载和解析。让它返回一个空字典，表示没有 Provider 被初始化。
        # _build_model_alias_map 会因此跳过模型别名映射（因为provider未初始化），这也是预期的。
        # 如果要测试模型别名映射，_init_providers 需要返回包含预期provider的mock。
        with patch('aibiowflow.llm_gateway.gateway.LLMGateway._init_providers', return_value={}) as mock_init_providers_call:
            gateway = LLMGateway(self.config_path, self.prompts_path)

            # 验证 API 密钥是否从环境变量正确加载
            self.assertEqual(gateway.config["api_keys"]["google"], "actual_test_api_key_value")
            # 验证模型定义是否被读取
            self.assertEqual(gateway.config["models"][0]["name"], "test-model-gemini")
            # 验证重试策略参数
            self.assertEqual(gateway.max_retries, 5)
            self.assertEqual(gateway.backoff_factor, 1.5)
            self.assertEqual(gateway.initial_wait_seconds, 0.1)
            # 验证 Prompt 模板是否被读取
            self.assertIn("sample_prompt", gateway.prompts)
            self.assertEqual(gateway.prompts["sample_prompt"]["template"], "你好, {user}!")

            mock_init_providers_call.assert_called_once() # 确保 _init_providers 被调用过一次

    def test_missing_config_file(self):
        """测试场景：当主配置文件 (`config.yaml`) 不存在时，应抛出 ConfigurationError。"""
        self._write_prompts({"some_prompt": {"template": "Test"}}) # Prompts 文件存在以隔离问题

        # 预期的错误消息应包含缺失的文件路径
        # 使用 os.path.abspath 来确保路径在不同系统上的一致性，以匹配错误消息中的路径
        # expected_error_regex = f"配置文件未找到: '{os.path.abspath(self.config_path)}'"
        # Adjusted regex to match ConfigurationError.__str__ format and use relative paths
        escaped_relative_config_path = re.escape(self.config_path)
        expected_error_regex = (
            f"Configuration Error \\(File: {escaped_relative_config_path}\\): "
            f"配置文件未找到: '{escaped_relative_config_path}'。"
        )
        with self.assertRaisesRegex(ConfigurationError, expected_error_regex):
            LLMGateway(self.config_path, self.prompts_path)

    def test_malformed_config_file(self):
        """测试场景：当主配置文件 (`config.yaml`) 包含无效 YAML 格式时，应抛出 ConfigurationError。"""
        # 写入一个格式错误的 YAML 内容
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write("api_keys: \n  google: key\nthis_is_not_valid_yaml: [missing_bracket") # 无效YAML
        self._write_prompts({"some_prompt": {"template": "Test"}}) # 保证 prompts 文件有效

        escaped_relative_config_path = re.escape(self.config_path)
        expected_error_regex = (
            f"Configuration Error \\(File: {escaped_relative_config_path}\\): "
            f"解析 YAML 配置文件 '{escaped_relative_config_path}' 失败。请检查文件格式。"
        )
        with self.assertRaisesRegex(ConfigurationError, expected_error_regex):
            LLMGateway(self.config_path, self.prompts_path)

    def test_missing_prompts_file(self):
        """测试场景：当 prompts 文件 (`prompts.toml`) 不存在时，应抛出 ConfigurationError。"""
        self._write_config({"api_keys": {}, "models": []}) # 主配置文件存在且基本有效

        escaped_relative_prompts_path = re.escape(self.prompts_path)
        expected_error_regex = (
            f"Configuration Error \\(File: {escaped_relative_prompts_path}\\): "
            f"Prompt 文件未找到: '{escaped_relative_prompts_path}'。"
        )
        with self.assertRaisesRegex(ConfigurationError, expected_error_regex):
            LLMGateway(self.config_path, self.prompts_path)

    def test_malformed_prompts_file(self):
        """测试场景：当 prompts 文件 (`prompts.toml`) 包含无效 TOML 格式时，应抛出 ConfigurationError。"""
        # 写入一个格式错误的 TOML 内容
        with open(self.prompts_path, 'w', encoding='utf-8') as f:
            f.write("[prompt_one]\ntemplate = 'valid'\n[prompt_two # 缺少右方括号导致错误") # 无效TOML
        self._write_config({"api_keys": {}, "models": []}) # 主配置文件有效

        escaped_relative_prompts_path = re.escape(self.prompts_path)
        expected_error_regex = (
            f"Configuration Error \\(File: {escaped_relative_prompts_path}\\): "
            f"解析 TOML Prompt 文件 '{escaped_relative_prompts_path}' 失败。请检查文件格式。"
        )
        with self.assertRaisesRegex(ConfigurationError, expected_error_regex):
            LLMGateway(self.config_path, self.prompts_path)

    def test_api_key_env_var_not_set(self):
        """测试场景：当 API 密钥配置为从环境变量读取，但该环境变量未设置时，对应密钥值应为 None。"""
        config_data = {"api_keys": {"google": f"env:{self.test_env_var_name}_NON_EXISTENT"}} # 使用不存在的环境变量
        self._write_config(config_data)
        self._write_prompts({}) # 空但有效的 prompts 文件

        # 模拟 _init_providers，因为我们关注的是 config 加载本身
        with patch('aibiowflow.llm_gateway.gateway.LLMGateway._init_providers', return_value={}):
            gateway = LLMGateway(self.config_path, self.prompts_path)
            # 验证 gateway.config 中对应的 API 密钥是否为 None
            self.assertIsNone(gateway.config["api_keys"]["google"],
                              "当环境变量未设置时，从 'env:' 读取的 API 密钥应为 None")

    def test_model_alias_mapping_with_initialized_provider(self):
        """测试场景：模型别名是否正确映射，特别是当其 Provider 成功初始化时。"""
        config_data = {
            "api_keys": {"google": "dummy_key_for_init_test"}, # 提供一个key，使得 'google' provider可以被“初始化”
            "models": [
                {"name": "gemini-model-1", "provider": "google", "aliases": ["alias_g1", "default_google"]},
                {"name": "other-model", "provider": "non_existent_provider", "aliases": ["alias_other"]},
            ]
        }
        self._write_config(config_data)
        self._write_prompts({})

        # 模拟 _init_providers，使其返回一个已初始化的 'google' provider 的 mock 对象。
        # 这使得 'google' provider 的模型能够被正确映射。
        mock_google_provider = unittest.mock.Mock(spec=True) # spec=True 确保模拟对象行为更像被模拟的类
        mock_google_provider.provider_name = "google" # 假设 GeminiProvider.PROVIDER_ID 是 'google'

        initialized_providers_map = {"google": mock_google_provider} # 模拟 "google" provider 已初始化

        with patch('aibiowflow.llm_gateway.gateway.LLMGateway._init_providers', return_value=initialized_providers_map):
            gateway = LLMGateway(self.config_path, self.prompts_path)

            # 验证 'google' provider 的模型别名
            self.assertIn("alias_g1", gateway.model_alias_map, "别名 'alias_g1' 应存在")
            self.assertEqual(gateway.model_alias_map["alias_g1"]["model_name"], "gemini-model-1")
            self.assertEqual(gateway.model_alias_map["alias_g1"]["provider"], "google")
            self.assertIn("default_google", gateway.model_alias_map, "别名 'default_google' 应存在")

            # 验证 'non_existent_provider' 的模型别名是否被忽略（因为该 provider 未初始化）
            self.assertNotIn("alias_other", gateway.model_alias_map,
                             "别名 'alias_other' 不应存在，因为其 Provider 'non_existent_provider' 未被初始化。")

    def test_invalid_model_config_missing_name(self):
        """测试场景：当模型定义缺少 'name' 字段时，应在初始化时抛出 ConfigurationError。"""
        config_data = {"models": [{"provider": "google", "aliases": ["alias_a"]}]} # 'name' 字段缺失
        self._write_config(config_data)
        self._write_prompts({})
        with self.assertRaisesRegex(ConfigurationError, "缺少有效的 'name' 字符串字段"):
            LLMGateway(self.config_path, self.prompts_path)

    def test_invalid_model_config_missing_provider(self):
        """测试场景：当模型定义缺少 'provider' 字段时，应在初始化时抛出 ConfigurationError。"""
        config_data = {"models": [{"name": "model_x", "aliases": ["alias_b"]}]} # 'provider' 字段缺失
        self._write_config(config_data)
        self._write_prompts({})
        with self.assertRaisesRegex(ConfigurationError, "缺少有效的 'provider' 字符串字段"):
            LLMGateway(self.config_path, self.prompts_path)

    def test_invalid_prompt_template_format_missing_template_key(self):
        """测试场景：当 Prompt 定义缺少 'template' 键时，应在初始化时抛出 ConfigurationError。"""
        prompts_data = {"invalid_prompt_def": {"description": "这是一个没有 template 键的 prompt 定义"}}
        self._write_prompts(prompts_data)
        self._write_config({"models": []}) # 提供一个有效但空的模型配置

        # 错误消息来自 _load_prompts 中的验证逻辑
        expected_error_regex = "必须是一个包含 'template' 键的字典"
        with self.assertRaisesRegex(ConfigurationError, expected_error_regex):
            LLMGateway(self.config_path, self.prompts_path)

    def test_retry_policy_parsing_and_defaults(self):
        """测试场景：验证重试策略参数是否从配置中正确解析，以及当部分或全部参数缺失时是否应用了正确的默认值。"""
        # 场景 1: 提供完整的重试策略配置
        config_full_retry = {
            "retry_policy": {
                "max_retries": 5, "backoff_factor": 1.5,
                "initial_wait_seconds": 0.5, "max_wait_seconds": 30.0
            }
        }
        self._write_config(config_full_retry)
        self._write_prompts({}) # 空 prompts 文件
        with patch('aibiowflow.llm_gateway.gateway.LLMGateway._init_providers', return_value={}): # 模拟 providers
            gateway_full = LLMGateway(self.config_path, self.prompts_path)
            self.assertEqual(gateway_full.max_retries, 5, "max_retries 未从配置中正确加载")
            self.assertEqual(gateway_full.backoff_factor, 1.5, "backoff_factor 未从配置中正确加载")
            self.assertEqual(gateway_full.initial_wait_seconds, 0.5, "initial_wait_seconds 未从配置中正确加载")
            self.assertEqual(gateway_full.max_wait_seconds, 30.0, "max_wait_seconds 未从配置中正确加载")

        # 场景 2: 提供部分重试策略配置，其余应使用默认值
        config_partial_retry = {"retry_policy": {"max_retries": 2, "initial_wait_seconds": 0.2}}
        self._write_config(config_partial_retry)
        with patch('aibiowflow.llm_gateway.gateway.LLMGateway._init_providers', return_value={}):
            gateway_partial = LLMGateway(self.config_path, self.prompts_path)
            self.assertEqual(gateway_partial.max_retries, 2, "max_retries (部分配置) 未正确加载")
            self.assertEqual(gateway_partial.initial_wait_seconds, 0.2, "initial_wait_seconds (部分配置) 未正确加载")
            self.assertEqual(gateway_partial.backoff_factor, 2.0, "backoff_factor 应为默认值") # 默认值来自 gateway.py
            self.assertEqual(gateway_partial.max_wait_seconds, 60.0, "max_wait_seconds 应为默认值") # 默认值

        # 场景 3: 未提供 retry_policy 部分，所有参数应使用默认值
        config_no_retry = {} # 空配置，或不含 retry_policy
        self._write_config(config_no_retry)
        with patch('aibiowflow.llm_gateway.gateway.LLMGateway._init_providers', return_value={}):
            gateway_no_retry = LLMGateway(self.config_path, self.prompts_path)
            self.assertEqual(gateway_no_retry.max_retries, 3, "max_retries (无配置) 应为默认值")
            self.assertEqual(gateway_no_retry.backoff_factor, 2.0, "backoff_factor (无配置) 应为默认值")
            self.assertEqual(gateway_no_retry.initial_wait_seconds, 1.0, "initial_wait_seconds (无配置) 应为默认值")
            self.assertEqual(gateway_no_retry.max_wait_seconds, 60.0, "max_wait_seconds (无配置) 应为默认值")

if __name__ == '__main__':
    unittest.main() # 运行测试
