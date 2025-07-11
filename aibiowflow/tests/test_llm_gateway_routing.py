# -*- coding: utf-8 -*-
"""
LLM 网关模型别名路由相关的单元测试 (_get_provider_and_model)。

这些测试验证 `LLMGateway` 是否能够根据提供的模型别名，
正确地解析出对应的 Provider 实例和实际的底层模型名称。
测试覆盖了成功路由、别名未定义、以及别名对应的 Provider 未初始化等场景。
"""
import unittest
import os
import yaml # 用于写入测试配置文件
from unittest.mock import patch, MagicMock # 用于模拟 Provider 实例和 LLMGateway 内部方法

from aibiowflow.llm_gateway.gateway import LLMGateway
from aibiowflow.llm_gateway.exceptions import LLMAPIError, ConfigurationError # ConfigurationError 可能在设置阶段抛出
from aibiowflow.llm_gateway.providers.base_provider import BaseProvider


class TestModelRouting(unittest.TestCase):
    """
    测试 `LLMGateway` 的模型路由逻辑，主要是其内部方法 `_get_provider_and_model`。
    这些测试依赖于对 `_init_providers` 方法的模拟，以便精确控制哪些 Provider 被认为是“已初始化的”。
    """

    def setUp(self):
        """
        每个测试方法运行前的准备工作。
        创建临时的目录和文件路径，并准备模拟的 Provider 实例。
        """
        self.temp_dir = "temp_test_routing_dir_for_gateway" # 更独特的目录名
        os.makedirs(self.temp_dir, exist_ok=True)
        self.config_path = os.path.join(self.temp_dir, "routing_test_config.yaml")
        self.prompts_path = os.path.join(self.temp_dir, "dummy_prompts_for_routing_test.toml")

        # 创建一个最简化的虚拟 prompts 文件，因为 LLMGateway 初始化需要它。
        with open(self.prompts_path, 'w', encoding='utf-8') as f:
            f.write("[dummy_prompt]\ntemplate=\"这是个虚拟的Prompt。\"\n")

        # 创建模拟的 Provider 实例，用于注入到 LLMGateway 中
        self.mock_provider_google = MagicMock(spec=BaseProvider)
        self.mock_provider_google.provider_name = "google" # 确保 provider_name 属性存在且正确

        self.mock_provider_anthropic = MagicMock(spec=BaseProvider)
        self.mock_provider_anthropic.provider_name = "anthropic"


    def tearDown(self):
        """
        每个测试方法运行后的清理工作。
        删除创建的临时文件和目录。
        """
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        if os.path.exists(self.prompts_path):
            os.remove(self.prompts_path)
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except OSError:
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                print(f"警告: 测试目录 {self.temp_dir} 在 tearDown 时非空，已强制删除。")


    def _create_gateway_with_config(self, config_data: dict,
                                    initialized_providers_map: dict[str, BaseProvider]) -> LLMGateway:
        """
        辅助函数：根据提供的 `config_data` 创建一个临时的配置文件，
        并使用模拟的 `initialized_providers_map` 来初始化 `LLMGateway` 实例。
        这允许我们精确控制哪些 Provider 被认为是“已初始化的”，从而测试路由逻辑。

        :param config_data: 要写入临时配置文件的字典数据。
        :param initialized_providers_map: 一个字典，模拟 `_init_providers` 方法的返回值，
                                          键是 provider_id，值是模拟的 Provider 实例。
        :return: 配置好的 `LLMGateway` 实例。
        """
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)

        # 关键：使用 patch 来模拟 LLMGateway._init_providers 方法，
        # 使其返回我们预设的、包含 mock provider 实例的字典。
        with patch('aibiowflow.llm_gateway.gateway.LLMGateway._init_providers',
                   return_value=initialized_providers_map):
            gateway = LLMGateway(self.config_path, self.prompts_path)
        return gateway

    def test_route_to_correct_provider_and_model(self):
        """测试场景：当提供有效的模型别名时，能够成功路由到正确的 Provider 实例和实际的模型名称。"""
        config_data = {
            "api_keys": { # 这些 API 密钥的实际值不重要，因为 _init_providers 被模拟了
                "google": "dummy_google_key",
                "anthropic": "dummy_anthropic_key"
            },
            "models": [
                {"name": "gemini-1.5-pro", "provider": "google", "aliases": ["smart_alias", "google_default_model"]},
                {"name": "claude-3-opus", "provider": "anthropic", "aliases": ["opus_alias", "anthropic_powerful_model"]},
            ]
        }
        # 假设 'google' 和 'anthropic' 两个 provider 都已成功初始化
        initialized_providers = {
            "google": self.mock_provider_google,
            "anthropic": self.mock_provider_anthropic
        }
        gateway = self._create_gateway_with_config(config_data, initialized_providers)

        # 测试路由到 Google Provider
        provider_instance_google, model_name_google = gateway._get_provider_and_model("smart_alias")
        self.assertIs(provider_instance_google, self.mock_provider_google, "未能正确路由到 Google Provider 实例。")
        self.assertEqual(model_name_google, "gemini-1.5-pro", "未能正确解析到 Gemini 模型的实际名称。")

        # 测试路由到 Anthropic Provider
        provider_instance_anthropic, model_name_anthropic = gateway._get_provider_and_model("opus_alias")
        self.assertIs(provider_instance_anthropic, self.mock_provider_anthropic, "未能正确路由到 Anthropic Provider 实例。")
        self.assertEqual(model_name_anthropic, "claude-3-opus", "未能正确解析到 Claude 模型的实际名称。")

    def test_undefined_model_alias(self):
        """测试场景：当请求的模��别名未在配置文件中定义时，应抛出 ValueError。"""
        config_data = {
            "api_keys": {"google": "dummy_key"},
            "models": [{"name": "gemini-1.0-pro", "provider": "google", "aliases": ["defined_alias"]}]
        }
        # 只初始化 'google' provider
        gateway = self._create_gateway_with_config(config_data, {"google": self.mock_provider_google})

        with self.assertRaisesRegex(ValueError, "模型别名 'non_existent_test_alias' 未在配置中定义"):
            gateway._get_provider_and_model("non_existent_test_alias")

    def test_alias_for_uninitialized_provider(self):
        """
        测试场景：当一个模型别名在配置中定义，但其对应的 Provider 未能成功初始化时，
        该别名不应出现在有效的 `model_alias_map` 中。尝试使用此别名应导致 ValueError
        (因为 `_build_model_alias_map` 会过滤掉这类模型的别名)。
        """
        config_data = {
            "api_keys": {"google": "dummy_key"}, # 假设 'anthropic' 的 API 密钥缺失或无效，导致其 provider 未初始化
            "models": [
                {"name": "gemini-model", "provider": "google", "aliases": ["google_model_alias"]},
                {"name": "claude-model", "provider": "anthropic", "aliases": ["anthropic_model_alias"]},
            ]
        }
        # 在此场景中，我们只模拟 'google' provider 的成功初始化
        initialized_providers = {"google": self.mock_provider_google} # 'anthropic' provider 未在此列出
        gateway = self._create_gateway_with_config(config_data, initialized_providers)

        # 验证 'anthropic_model_alias' 是否因为其 provider ('anthropic') 未初始化而不存在于最终的映射中
        self.assertNotIn("anthropic_model_alias", gateway.model_alias_map,
                         "别名 'anthropic_model_alias' 不应存在于映射中，因为其 Provider 'anthropic' 未被初始化。")

        # 尝试获取这个实际上无效的别名，应该抛出 ValueError (因为它不在 model_alias_map 中)
        with self.assertRaisesRegex(ValueError, "模型别名 'anthropic_model_alias' 未在配置中定义"):
            gateway._get_provider_and_model("anthropic_model_alias")

    def test_no_providers_initialized_at_all(self):
        """
        测试场景：当配置文件中定义了模型，但没有任何 Provider 成功初始化时
        （例如，所有 API 密钥都缺失或无效），`model_alias_map` 应为空。
        尝试获取任何别名都应导致 ValueError。
        """
        config_data = {
            "api_keys": {}, # 没有提供任何 API 密钥
            "models": [{"name": "any-llm-model", "provider": "google", "aliases": ["some_random_alias"]}]
        }
        # 模拟 _init_providers 返回一个空字典，表示没有任何 Provider 成功初始化
        gateway = self._create_gateway_with_config(config_data, {})

        self.assertEqual(len(gateway.providers), 0, "不应有任何 Provider 被初始化。")
        # 由于 'google' provider 未初始化，'some_random_alias' 不会被加入到 model_alias_map
        self.assertEqual(len(gateway.model_alias_map), 0, "模型别名映射应为空，因为没有 Provider 被初始化。")

        # 尝试获取别名，应失败并提示别名未定义（因为它不在过滤后的 model_alias_map 中）
        with self.assertRaisesRegex(ValueError, "模型别名 'some_random_alias' 未在配置中定义"):
            gateway._get_provider_and_model("some_random_alias")


if __name__ == '__main__':
    unittest.main() # 运行测试
