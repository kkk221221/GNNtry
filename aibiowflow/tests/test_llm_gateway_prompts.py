# -*- coding: utf-8 -*-
"""
LLM 网关 Prompt 处理相关的单元测试。

这些测试专注于 `LLMGateway` 类中 `_format_prompt` 方法的功能，
确保它能够正确地根据上下文填充 Prompt 模板，并处理各种边界情况和错误。
"""
import unittest
import os
import toml # 用于写入测试用的 TOML prompts 文件
from unittest.mock import patch # 用于模拟 LLMGateway 的部分内部方法

from aibiowflow.llm_gateway.gateway import LLMGateway
from aibiowflow.llm_gateway.exceptions import PromptTemplateError

class TestPromptFormatting(unittest.TestCase):
    """
    测试 `LLMGateway` 的 Prompt 格式化逻辑 (`_format_prompt` 方法)。
    这些测试验证：
    - 使用有效的上下文成功格式化 Prompt。
    - 当 Prompt 名称在模板文件中不存在时，正确抛出 `PromptTemplateError`。
    - 当上下文中缺少 Prompt 模板所需的键时，正确抛出 `PromptTemplateError`。
    - 即使上下文中包含模板不需要的额外键，格式化也能成功。
    - 对于没有变量的模板，使用空上下文也能正确格式化。
    - 正确处理模板字符串为空的情况。
    """

    def setUp(self):
        """
        每个测试方法运行前的准备工作。
        创建临时的目录和文件路径，用于存放测试用的虚拟配置文件和 Prompt 文件。
        """
        self.temp_dir = "temp_test_prompts_dir_for_gateway" # 更独特的目录名
        os.makedirs(self.temp_dir, exist_ok=True)
        self.config_path = os.path.join(self.temp_dir, "dummy_config_for_prompts_test.yaml")
        self.prompts_path = os.path.join(self.temp_dir, "test_prompts_for_formatting.toml")

        # 创建一个最简化的虚拟 config 文件，因为 LLMGateway 初始化时需要它。
        # 测试的重点是 Prompt 格式化，因此 config 内容不重要。
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write("models: []\napi_keys: {}\n")

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
            except OSError: # 如果目录非空
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                print(f"警告: 测试目录 {self.temp_dir} 在 tearDown 时非空，已强制删除。")


    def _create_gateway_with_prompts(self, prompts_data: dict) -> LLMGateway:
        """
        辅助函数：根据提供的 `prompts_data` 创建一个临时的 `prompts.toml` 文件，
        并初始化一个 `LLMGateway` 实例。
        为了隔离测试 `_format_prompt` 方法，`_init_providers` 和 `_build_model_alias_map`
        这两个通常在 `LLMGateway` 初始化时调用的方法会被模拟掉。

        :param prompts_data: 一个字典，其内容将被写入临时的 `prompts.toml` 文件。
        :return: 配置了指定 Prompt 的 `LLMGateway` 实例。
        """
        with open(self.prompts_path, 'w', encoding='utf-8') as f:
            toml.dump(prompts_data, f)

        # 模拟 _init_providers 和 _build_model_alias_map,
        # 因为这些测试只关注 _format_prompt 的逻辑，与 Provider 和模型映射无关。
        with patch('aibiowflow.llm_gateway.gateway.LLMGateway._init_providers', return_value={}), \
             patch('aibiowflow.llm_gateway.gateway.LLMGateway._build_model_alias_map', return_value={}):
            gateway = LLMGateway(self.config_path, self.prompts_path)
        return gateway

    def test_format_prompt_successfully(self):
        """测试场景：使用有效的上下文数据成功格式化包含占位符的 Prompt 模板。"""
        prompts_data = {
            "greeting_tpl": {"template": "你好, {name}! 今天是 {day}。"}, # 模板1
            "query_tpl": {"template": "用户的查询是: '{query_text}'"}      # 模板2
        }
        gateway = self._create_gateway_with_prompts(prompts_data)

        # 测试模板1
        context1 = {"name": "AI开发者", "day": "美好的一天"}
        expected_formatted_prompt1 = "你好, AI开发者! 今天是 美好的一天。"
        self.assertEqual(gateway._format_prompt("greeting_tpl", context1), expected_formatted_prompt1,
                         "使用完整上下文格式化 Prompt 失败。")

        # 测试模板2
        context2 = {"query_text": "LLM 网关如何工作？"}
        expected_formatted_prompt2 = "用户的查询是: 'LLM 网关如何工作？'"
        self.assertEqual(gateway._format_prompt("query_tpl", context2), expected_formatted_prompt2,
                         "使用不同上下文格式化另一个 Prompt 失败。")

    def test_format_prompt_missing_template_name(self):
        """测试场景：当请求格式化的 Prompt 名称在 Prompt 文件中不存在时，应抛出 PromptTemplateError。"""
        prompts_data = {"existing_prompt_key": {"template": "这是一个已存在的模板内容。"}}
        gateway = self._create_gateway_with_prompts(prompts_data)

        # 尝试格式化一个不存在的 Prompt 名称
        with self.assertRaisesRegex(PromptTemplateError,
                                     "Prompt 模板 'non_existent_template_name' 未在 Prompt 文件中定义。"):
            gateway._format_prompt("non_existent_template_name", {})

    def test_format_prompt_missing_key_in_context(self):
        """测试场景：当提供的上下文字典中缺少 Prompt 模板所必需的键（占位符）时，应抛出 PromptTemplateError。"""
        prompts_data = {"user_specific_greeting": {"template": "欢迎回来, {username}！您的积分为: {points}。"}}
        gateway = self._create_gateway_with_prompts(prompts_data)

        # 上下文中缺少 'points' 键
        context_missing_points = {"username": "test_user"}
        with self.assertRaisesRegex(PromptTemplateError,
                                     "填充 Prompt 模板 'user_specific_greeting' 时，上下文中缺少必要的键: 'points'"):
            gateway._format_prompt("user_specific_greeting", context_missing_points)

    def test_format_prompt_extra_keys_in_context(self):
        """测试场景：当提供的上下文字典中包含模板中未定义的额外键时，格式化应仍然成功，忽略这些额外键。"""
        prompts_data = {"simple_value_prompt": {"template": "当前值为: {value}"}}
        gateway = self._create_gateway_with_prompts(prompts_data)

        # 上下文中包含额外的 'unused_key'
        context_with_extra_keys = {"value": "42", "unused_key": "此键不应影响格式化"}
        expected_formatted_prompt = "当前值为: 42"
        self.assertEqual(gateway._format_prompt("simple_value_prompt", context_with_extra_keys),
                         expected_formatted_prompt, "包含额外键的上下文未能正确格式化 Prompt。")

    def test_format_prompt_empty_context_for_template_with_no_variables(self):
        """测试场景：当 Prompt 模板本身不包含任何变量（占位符）时，即使提供空的上下文，也应能正确返回模板原文。"""
        prompts_data = {"static_info_prompt": {"template": "系统版本：v1.0。无动态内容。"}}
        gateway = self._create_gateway_with_prompts(prompts_data)

        expected_static_prompt = "系统版本：v1.0。无动态内容。"
        # 使用空字典作为上下文
        self.assertEqual(gateway._format_prompt("static_info_prompt", {}), expected_static_prompt,
                         "对于无变量模板和空上下文，未能正确返回模板原文。")

    def test_format_prompt_empty_template_string(self):
        """测试场景：当 Prompt 模板中的 'template' 字符串本身为空时，格式化结果也应为空字符串。"""
        prompts_data = {"empty_template_str_prompt": {"template": ""}} # template 键的值是空字符串
        gateway = self._create_gateway_with_prompts(prompts_data)

        expected_empty_string = ""
        # 提供任意上下文，对于空模板字符串，结果都应为空
        self.assertEqual(gateway._format_prompt("empty_template_str_prompt", {"any_key": "any_value"}),
                         expected_empty_string, "空模板字符串未能正确格式化为空结果。")


if __name__ == '__main__':
    unittest.main() # 运行测试
