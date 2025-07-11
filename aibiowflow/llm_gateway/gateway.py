# -*- coding: utf-8 -*-
"""
LLM 网关核心实现。

包含 LLMGateway 类，用于管理与 LLM 的交互，包括配置加载、Prompt 管理、
Provider 路由、API 调用重试、结构化输出验证和日志记录。
"""

from pydantic import BaseModel # type: ignore
import pydantic # 用于 ValidationError
from typing import Type, Dict, Any, Callable, Tuple, Optional # 增加了 Optional
import yaml
import toml
import os
import time
import logging
import json # 用于结构化日志

from .exceptions import LLMAPIError, LLMOutputValidationError, ConfigurationError, PromptTemplateError
from .providers.base_provider import BaseProvider, ProviderResponse
from .providers import GeminiProvider # 显式导入以备后用, 其他 provider 类似

# 获取模块级 logger 实例
logger = logging.getLogger(__name__)


# 定义可重试的 HTTP 状态码列表
# 这些通常表示临时的服务器端问题或速率限制。
RETRYABLE_STATUS_CODES = [
    429, # Too Many Requests (速率限制)
    500, # Internal Server Error
    503, # Service Unavailable
    504, # Gateway Timeout
]


class LLMGateway:
    """
    LLM 网关服务 (LLM Gateway Service)。

    作为系统中与外部大型语言模型 (LLM) API 交互的唯一、专用入口。
    它抽象了底层 LLM Provider 的具体实现，提供了统一的接口，
    并集中处理通用关注点，如配置管理、Prompt 模板、API 重试、
    结构化输出验证和详细的审计日志。
    """
    def __init__(self, config_path: str, prompt_path: str):
        """
        初始化 LLM 网关。

        此构造函数负责加载和解析配置文件 (`config.yaml`) 和 Prompt 模板文件 (`prompts.toml`)。
        它还会基于配置初始化所有支持的 LLM Provider，并构建模型别名到具体模型的映射。
        重试策略参数也会在此处从配置中加载。

        :param config_path: 网关配置文件的路径 (通常是 `config.yaml`)。
                            此文件定义了 API 密钥、模型及其别名、重试策略等。
        :param prompt_path: Prompt 模板文件的路径 (通常是 `prompts.toml`)。
                            此文件存储了所有预定义的 Prompt 模板。
        :raises ConfigurationError: 如果配置文件或 Prompt 文件未找到、格式不正确，
                                    或包含无效的配置项 (例如，模型定义不完整)。
        """
        logger.info(f"正在初始化 LLMGateway，配置文件路径: '{config_path}', Prompt 文件路径: '{prompt_path}'")

        # 加载主配置文件和 Prompt 模板
        self.config: Dict[str, Any] = self._load_config(config_path)
        self.prompts: Dict[str, Any] = self._load_prompts(prompt_path)

        # 初始化所有在配置中定义的、且有有效 API 密钥的 Provider
        self.providers: Dict[str, BaseProvider] = self._init_providers()

        # 构建模型别名到 (provider, 实际模型名) 的映射
        # 这个映射只包含那些其 Provider 已成功初始化的模型
        self.model_alias_map: Dict[str, Dict[str, Any]] = self._build_model_alias_map()

        # 解析重试策略配置，提供默认值
        retry_policy_config: Dict[str, Any] = self.config.get('retry_policy', {})
        self.max_retries: int = int(retry_policy_config.get('max_retries', 3))
        self.backoff_factor: float = float(retry_policy_config.get('backoff_factor', 2.0))
        self.initial_wait_seconds: float = float(retry_policy_config.get('initial_wait_seconds', 1.0))
        self.max_wait_seconds: float = float(retry_policy_config.get('max_wait_seconds', 60.0))

        logger.info("LLMGateway 初始化成功。")
        logger.debug(f"已成功初始化的 Provider: {list(self.providers.keys())}")
        logger.debug(f"可用的模型别名映射: {self.model_alias_map}")
        logger.debug(
            f"重试策略配置: max_retries={self.max_retries}, backoff_factor={self.backoff_factor}, "
            f"initial_wait_seconds={self.initial_wait_seconds}, max_wait_seconds={self.max_wait_seconds}"
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        私有辅助方法：加载并解析 YAML 配置文件。
        同时处理对环境变量的引用 (例如 `env:MY_API_KEY`) 来安全地加载 API 密钥。

        :param config_path: YAML 配置文件的路径。
        :return: 解析后的配置数据字典。
        :raises ConfigurationError: 如果文件未找到、不是有效的 YAML 或 YAML 内容不是字典。
        """
        logger.debug(f"开始加载配置文件: '{config_path}'")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            if not isinstance(config_data, dict): # 配置文件顶层必须是字典
                msg = f"配置文件 '{config_path}' 的顶层内容必须是一个有效的 YAML 字典结构。"
                logger.error(msg)
                raise ConfigurationError(msg, config_file_path=config_path)
        except FileNotFoundError as e:
            msg = f"配置文件未找到: '{config_path}'。"
            logger.error(msg, exc_info=True) # exc_info=True 会记录异常堆栈
            raise ConfigurationError(msg, config_file_path=config_path, original_exception=e) from e
        except yaml.YAMLError as e:
            msg = f"解析 YAML 配置文件 '{config_path}' 失败。请检查文件格式。"
            logger.error(msg, exc_info=True)
            raise ConfigurationError(msg, config_file_path=config_path, original_exception=e) from e

        # 处理 API 密钥中的环境变量引用
        api_keys_section = config_data.get('api_keys', {})
        if not isinstance(api_keys_section, dict):
            logger.warning(
                f"配置文件 '{config_path}' 中的 'api_keys' 部分不是预期的字典格式，将被忽略。 "
                f"收到的类型: {type(api_keys_section)}。"
            )
            config_data['api_keys'] = {} # 确保 api_keys 字段存在且为字典
        else:
            for key_name, key_value in api_keys_section.items():
                if isinstance(key_value, str) and key_value.startswith("env:"):
                    # 如果值以 "env:" 开头，则尝试从环境变量加载
                    env_var_name = key_value.split("env:", 1)[1]
                    env_var_value = os.environ.get(env_var_name)
                    if not env_var_value:
                        logger.warning(
                            f"配置的 API 密钥 '{key_name}' 指向的环境变量 '{env_var_name}' 未设置或为空。 "
                            f"对应的 Provider 可能无法初始化。"
                        )
                        config_data['api_keys'][key_name] = None # 将其值设为 None
                    else:
                        logger.info(f"已成功从环境变量 '{env_var_name}' 加载 API 密钥用于 '{key_name}'。")
                        config_data['api_keys'][key_name] = env_var_value
                elif key_value is not None and not isinstance(key_value, str):
                    # 如果值不是字符串，也不是 None (允许显式设置密钥为 None 来禁用 provider)
                    logger.warning(
                        f"API 密钥 '{key_name}' 的值 '{key_value}' 不是字符串或 'env:' 指令，也不是 None。 "
                        "将尝试按原样使用，但这可能导致 Provider 初始化失败。"
                    )

        logger.info(f"配置文件 '{config_path}' 加载并初步处理完毕。")
        return config_data


    def _load_prompts(self, prompt_path: str) -> Dict[str, Any]:
        """
        私有辅助方法：加载并解析 TOML Prompt 模板文件。
        同时验证每个 Prompt 条目是否包含 'template' 键且其值为字符串。

        :param prompt_path: TOML Prompt 文件的路径。
        :return: 解析后的 Prompt 数据字典。
        :raises ConfigurationError: 如果文件未找到、不是有效的 TOML、TOML 内容不是字典，
                                    或者 Prompt 条目格式不正确。
        """
        logger.debug(f"开始加载 Prompt 文件: '{prompt_path}'")
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompts_data = toml.load(f)
            if not isinstance(prompts_data, dict): # Prompt 文件顶层必须是字典
                msg = f"Prompt 文件 '{prompt_path}' 的顶层内容必须是一个有效的 TOML 字典结构。"
                logger.error(msg)
                raise ConfigurationError(msg, config_file_path=prompt_path)
        except FileNotFoundError as e:
            msg = f"Prompt 文件未找到: '{prompt_path}'。"
            logger.error(msg, exc_info=True)
            raise ConfigurationError(msg, config_file_path=prompt_path, original_exception=e) from e
        except toml.TomlDecodeError as e:
            msg = f"解析 TOML Prompt 文件 '{prompt_path}' 失败。请检查文件格式。"
            logger.error(msg, exc_info=True)
            raise ConfigurationError(msg, config_file_path=prompt_path, original_exception=e) from e

        # 验证每个 Prompt 模板的结构
        for prompt_name, prompt_data in prompts_data.items():
            if not isinstance(prompt_data, dict) or 'template' not in prompt_data:
                msg = (f"Prompt 模板 '{prompt_name}' 在文件 '{prompt_path}' 中的定义不正确： "
                       "它必须是一个包含 'template' 键的字典。")
                logger.error(msg)
                raise ConfigurationError(msg, config_file_path=prompt_path)
            if not isinstance(prompt_data['template'], str):
                msg = (f"Prompt 模板 '{prompt_name}' 在文件 '{prompt_path}' 中的 'template' 键的值必须是字符串。 "
                       f"当前类型: {type(prompt_data['template'])}。")
                logger.error(msg)
                raise ConfigurationError(msg, config_file_path=prompt_path)

        logger.info(f"Prompt 文件 '{prompt_path}' 加载并验证完毕，共加载 {len(prompts_data)} 个模板。")
        return prompts_data

    def _init_providers(self) -> Dict[str, BaseProvider]:
        """
        私有辅助方法：根据主配置文件中的 `api_keys` 和 `provider_specific_configs` 部分，
        初始化所有已配置且具有有效 API 密钥的 LLM Provider 实例。

        目前硬编码支持 `GeminiProvider`。添加新的 Provider 时需要在此处扩展。

        :return: 一个字典，键是 Provider 的唯一 ID (例如 "google")，值是对应的 Provider 实例。
        """
        providers: Dict[str, BaseProvider] = {}
        api_keys: Dict[str, Any] = self.config.get('api_keys', {})
        provider_specific_configs: Dict[str, Any] = self.config.get('provider_specific_configs', {})

        # --- Google Gemini Provider 初始化逻辑 ---
        gemini_provider_id = GeminiProvider.PROVIDER_ID # 通常是 "google"

        # 检查 'api_keys' 部分是否包含此 provider_id 的条目
        if gemini_provider_id in api_keys:
            gemini_api_key = api_keys.get(gemini_provider_id) # 获取实际密钥值 (可能为 None)

            # 仅当 API 密钥存在且不为空时才尝试初始化 Provider
            # (允许在配置中将密钥设为 None 来显式禁用 Provider)
            if gemini_api_key:
                try:
                    # 获取特定于此 Provider 的配置 (例如，Gemini 的 safety_settings)
                    gemini_specific_config = provider_specific_configs.get(gemini_provider_id, {})
                    providers[gemini_provider_id] = GeminiProvider(
                        api_key=gemini_api_key,
                        provider_config=gemini_specific_config
                    )
                    logger.info(f"{gemini_provider_id.capitalize()} Provider 初始化成功。")
                except LLMAPIError as e_api: # Provider 初始化时可能因无效密钥抛出
                    logger.error(
                        f"初始化 {gemini_provider_id.capitalize()} Provider 失败 (API 相关问题): {e_api}。"
                        "请检查 API 密钥是否有效以及网络连接。"
                    )
                except ConfigurationError as e_conf: # Provider 初始化时可能因库缺失等配置问题抛出
                     logger.error(
                        f"初始化 {gemini_provider_id.capitalize()} Provider 失败 (配置相关问题): {e_conf}。"
                        "请确保相关依赖已安装且配置正确。"
                    )
                except Exception as e_unknown: # pylint: disable=broad-except
                    # 捕获其他所有未预料的异常
                    logger.error(
                        f"初始化 {gemini_provider_id.capitalize()} Provider 时发生未知错误: {e_unknown}",
                        exc_info=True # 记录完整堆栈
                    )
            else: # api_keys 中有 gemini_provider_id 但其值为 None 或空
                logger.info(
                    f"{gemini_provider_id.capitalize()} Provider 的 API 密钥未在配置中提供 (或为空)。"
                    "该 Provider 将不会被初始化。"
                )
        else: # api_keys 中完全没有 gemini_provider_id 条目
            logger.info(
                f"配置文件中未包含针对 '{gemini_provider_id}' Provider 的 API 密钥配置。"
                "如果需要使用此 Provider，请在 'api_keys' 部分添加相关条目。"
            )

        # --- 其他 Provider 初始化逻辑 (未来扩展点) ---
        # 例如，为 Anthropic Claude 添加初始化:
        # anthropic_provider_id = "anthropic" # 假设 AnthropicProvider.PROVIDER_ID
        # if anthropic_provider_id in api_keys:
        #     anthropic_api_key = api_keys.get(anthropic_provider_id)
        #     if anthropic_api_key:
        #         try:
        #             # from .providers import AnthropicProvider # 确保已导入
        #             # anthropic_config = provider_specific_configs.get(anthropic_provider_id, {})
        #             # providers[anthropic_provider_id] = AnthropicProvider(api_key=anthropic_api_key, provider_config=anthropic_config)
        #             # logger.info(f"{anthropic_provider_id.capitalize()} Provider 初始化成功。")
        #             pass # 占位
        #         except Exception as e: # ... 类似的错误处理 ...
        #             logger.error(f"初始化 {anthropic_provider_id.capitalize()} Provider 失败: {e}")
        #     # ... else 日志 ...
        # # ... else 日志 ...

        if not providers: # 如果没有任何 Provider 成功初始化
            logger.warning(
                "系统中没有成功初始化任何 LLM Provider。LLM 网关可能无法处理实际的 LLM 请求。"
                "请检查配置文件中的 'api_keys' 和环境变量设置。"
            )

        return providers

    def _build_model_alias_map(self) -> Dict[str, Dict[str, Any]]:
        """
        私有辅助方法：根据主配置文件中的 `models` 部分，构建模型别名到
        (provider_id, 实际模型名称) 的映射。

        此方法只为那些其 Provider 已在 `_init_providers` 中成功初始化的模型创建别名。
        如果一个模型定义的 Provider 未初始化，则其别名将被忽略。

        :return: 一个字典，键是模型别名 (字符串)，值是包含 "provider" (字符串) 和
                 "model_name" (字符串) 的字典。
        :raises ConfigurationError: 如果 `models` 配置部分格式不正确或模型条目无效。
        """
        logger.debug("开始构建模型别名映射...")
        alias_map: Dict[str, Dict[str, Any]] = {}
        models_config: list = self.config.get('models', []) # models 应该是列表

        if not isinstance(models_config, list):
            msg = "配置文件中的 'models' 部分必须是一个列表 (list of model definitions)。"
            logger.error(msg)
            raise ConfigurationError(msg, config_file_path=self.config_path) # 使用 self.config_path

        if not models_config:
            logger.warning("配置文件中 'models' 部分为空或未定义，将无法通过别名使用任何模型。")
            return alias_map

        for i, model_entry in enumerate(models_config):
            # 验证每个模型条目的基本结构
            if not isinstance(model_entry, dict):
                msg = f"模型定义列表中的条目 #{i+1} 不是一个有效的字典。每个模型定义都应为字典格式。"
                logger.error(msg)
                raise ConfigurationError(msg, config_file_path=self.config_path)

            model_name = model_entry.get('name')
            provider_id = model_entry.get('provider') # Provider ID, e.g., "google"
            aliases: list = model_entry.get('aliases', [])

            # 验证必要字段及其类型
            if not model_name or not isinstance(model_name, str):
                msg = f"模型定义条目 #{i+1} (内容: {model_entry}) 缺少有效的 'name' 字符串字段。"
                logger.error(msg)
                raise ConfigurationError(msg, config_file_path=self.config_path)
            if not provider_id or not isinstance(provider_id, str):
                msg = f"模型 '{model_name}' (条目 #{i+1}) 缺少有效的 'provider' 字符串字段。"
                logger.error(msg)
                raise ConfigurationError(msg, config_file_path=self.config_path)
            if not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
                msg = f"模型 '{model_name}' (条目 #{i+1}) 的 'aliases' 字段必须是一个字符串列表。"
                logger.error(msg)
                raise ConfigurationError(msg, config_file_path=self.config_path)

            if not aliases: # 模型没有别名
                logger.warning(
                    f"模型 '{model_name}' (提供商: {provider_id}) 未定义任何别名 ('aliases' 列表为空)。"
                    "此模型将无法通过别名进行调用。"
                )

            # 关键检查：此模型定义的 Provider 是否已成功初始化？
            if provider_id not in self.providers:
                logger.warning(
                    f"模型 '{model_name}' (条目 #{i+1}) 配置的提供商 '{provider_id}' 未能成功初始化 "
                    "(可能由于 API 密钥问题或 Provider 实现错误)。与此模型相关的别名将被忽略。"
                )
                continue # 跳过此模型的别名映射

            # 为此模型的所有别名创建映射
            for alias in aliases:
                if alias in alias_map: # 检查别名是否已被其他模型使用
                    logger.warning(
                        f"模型别名 '{alias}' 被重复定义。原映射指向: {alias_map[alias]['provider']}/{alias_map[alias]['model_name']}。 "
                        f"新映射将指向: {provider_id}/{model_name}。后者将覆盖前者。"
                    )
                alias_map[alias] = {"model_name": model_name, "provider": provider_id}
                logger.debug(f"已成功映射别名 '{alias}' -> 模型 '{model_name}' (提供商: '{provider_id}')")

        if not alias_map and models_config: # 有模型配置，但没有一个别名成功映射
            logger.warning(
                "配置文件中定义了模型，但由于其提供商未初始化或其他配置问题，"
                "最终没有构建任何有效的、可用的模型别名映射。"
            )
        elif alias_map:
             logger.info(f"成功构建模型别名映射，共 {len(alias_map)} 个可用别名。")
        # 如果 models_config 为空，则 alias_map 也为空，之前已有警告，此处无需额外日志

        return alias_map


    def _get_provider_and_model(self, model_alias: str) -> Tuple[BaseProvider, str]:
        """
        私有辅助方法：根据给定的模型别名，查找并返回对应的 Provider 实例和实际的模型名称。

        :param model_alias: 要查找的模型的逻辑别名 (在配置文件中定义)。
        :return: 一个元组，包含:
                 - `BaseProvider`: 对应此别名的 Provider 实例。
                 - `str`: 此别名实际指向的底层模型名称 (例如 "gemini-1.5-pro-latest")。
        :raises ValueError: 如果提供的 `model_alias` 未在配置中定义，或者其对应的 Provider 未成功初始化。
        :raises LLMAPIError: (理论上不应从此方法直接抛出，但在极端内部不一致情况下可能)
                             如果 Provider 在映射中存在但实例丢失。
        """
        logger.debug(f"正在为别名 '{model_alias}' 解析 Provider 和实际模型名称...")

        model_info = self.model_alias_map.get(model_alias)
        if not model_info: # 别名不存在于有效映射中
            msg = (f"模型别名 '{model_alias}' 未在配置中定义，或者其声明的 Provider 未能成功初始化。"
                   "请检查配置文件中的 'models' 部分和 Provider 初始化日志。")
            logger.error(msg)
            raise ValueError(msg) # 根据规范，与配置相关的查找失败应为 ValueError

        provider_id = model_info['provider']
        actual_model_name = model_info['model_name']

        # self.providers 字典现在应该只包含已成功初始化的 Provider 实例
        provider_instance = self.providers.get(provider_id)
        if not provider_instance:
            # 这个情况理论上不应该发生，因为 _build_model_alias_map 在构建映射时
            # 应该已经过滤掉了那些其 Provider 未初始化的模型。
            # 但作为防御性编程，保留此检查。
            msg = (f"内部逻辑错误：别名 '{model_alias}' 指向的提供商 '{provider_id}' 在模型映射中存在，"
                   "但该 Provider 的实例未在已初始化的提供商列表中找到。"
                   "这可能表示 _build_model_alias_map 和 _init_providers 的逻辑存在不一致。")
            logger.critical(msg) # 这是一个严重的内部问题
            raise LLMAPIError(msg) # 标记为 API 错误，因为这意味着网关无法提供服务

        logger.debug(f"别名 '{model_alias}' 成功解析为 -> 提供商: '{provider_id}', 实际模型: '{actual_model_name}'")
        return provider_instance, actual_model_name


    def _format_prompt(self, prompt_name: str, context: Dict[str, Any]) -> str:
        """
        私有辅助方法：根据提供的 Prompt 名称从加载的模板中获取模板字符串，
        并使用给定的上下文（字典）来填充模板中的占位符。

        :param prompt_name: 在 `prompts.toml` 文件中定义的 Prompt 的键名。
        :param context: 一个字典，包含用于填充模板中变量（占位符）的值。
        :return: 格式化（填充）完成后的 Prompt 字符串。
        :raises PromptTemplateError: 如果名为 `prompt_name` 的模板未找到，
                                    或者在格式化过程中 `context` 缺少模板所需的键。
        """
        logger.debug(f"开始格式化 Prompt 模板: '{prompt_name}'，使用上下文键: {list(context.keys())}")
        prompt_entry = self.prompts.get(prompt_name)

        # _load_prompts 方法已经验证过每个条目是字典且包含 'template' 字符串键
        if not prompt_entry:
            msg = f"名为 '{prompt_name}' 的 Prompt 模板未在 Prompt 文件中定义。"
            logger.error(msg)
            raise PromptTemplateError(msg, template_name=prompt_name)

        template_string = prompt_entry['template'] # 已验证为字符串
        try:
            # 使用字典解包 (**) 将上下文变量填充到模板字符串中
            filled_prompt = template_string.format(**context)
            logger.debug(f"Prompt 模板 '{prompt_name}' 格式化成功。")
            return filled_prompt
        except KeyError as e_key: # 当模板需要一个变量而 context 中没有提供时发生
            msg = f"填充 Prompt 模板 '{prompt_name}' 时，上下文中缺少必要的键: '{e_key.args[0]}'"
            logger.error(msg, exc_info=True) # 记录原始 KeyError 的堆栈
            raise PromptTemplateError(msg, template_name=prompt_name, original_exception=e_key) from e_key
        except Exception as e_format: # 捕获其他可能的 .format() 错误
            msg = f"格式化 Prompt 模板 '{prompt_name}' 时发生未知错误: {e_format}"
            logger.error(msg, exc_info=True)
            raise PromptTemplateError(msg, template_name=prompt_name, original_exception=e_format) from e_format

    def _execute_with_retry(
        self,
        action: Callable[..., ProviderResponse], # Provider 的方法，例如 provider.generate_text
        action_description: str, # 用于日志记录的操作描述，例如 "generate_text_call"
        **kwargs: Any # 传递给 action (Provider 方法) 的参数
    ) -> ProviderResponse:
        """
        私有辅助方法：执行一个指定的操作（通常是调用 Provider 的方法），
        并在发生可重试的 `LLMAPIError` 时根据配置的策略进行重试。

        重试逻辑包括指数退避 (exponential backoff) 和最大等待时间限制。

        :param action: 要执行的函数或方法，它应该返回一个 `ProviderResponse` 对象。
        :param action_description: 对正在执行的操作的简短描述，用于日志输出。
        :param kwargs: 将作为关键字参数传递给 `action` 函数的参数。
        :return: `action` 函数成功执行后的 `ProviderResponse` 对象。
        :raises LLMAPIError: 如果所有重试尝试均失败，或发生了不可重试的 API 错误。
        :raises Exception: 如果 `action` 抛出了非 `LLMAPIError` 类型的其他异常 (这些异常不会被重试)。
        """
        last_api_exception: Optional[LLMAPIError] = None # 存储最后一次捕获到的 LLMAPIError

        # 总尝试次数 = 1 (初次尝试) + self.max_retries (配置的重试次数)
        for attempt_num in range(self.max_retries + 1):
            try:
                logger.debug(
                    f"尝试执行操作 '{action_description}', 第 {attempt_num + 1} 次尝试。 "
                    f"传递的参数 (部分): { {k:v for k,v in kwargs.items() if k != 'prompt'} }" # 不记录完整 prompt
                )
                # 执行实际的 Provider 调用
                response = action(**kwargs)
                logger.info(f"操作 '{action_description}' 在第 {attempt_num + 1} 次尝试时成功。")
                return response # 成功，返回结果

            except LLMAPIError as e_api: # 捕获来自 Provider 的 API 错误
                last_api_exception = e_api # 保存此异常，以备所有重试失败后重新抛出

                # 检查错误是否可重试 (基于 HTTP 状态码)
                is_retryable = e_api.status_code is not None and e_api.status_code in RETRYABLE_STATUS_CODES

                # 如果错误可重试且尚未达到最大重试次数
                if is_retryable and attempt_num < self.max_retries:
                    # 计算等待时间，采用指数退避策略
                    wait_time = self.initial_wait_seconds * (self.backoff_factor ** attempt_num)
                    # 确保等待时间不超过配置的最大等待时间
                    wait_time = min(wait_time, self.max_wait_seconds)

                    logger.warning(
                        f"操作 '{action_description}' 第 {attempt_num + 1} 次尝试失败 (可重试错误: "
                        f"状态码={e_api.status_code}, 消息='{e_api}'). "
                        f"将在 {wait_time:.2f} 秒后进行下一次重试..."
                    )
                    time.sleep(wait_time) # 等待后继续下一次循环 (重试)
                else: # 如果错误不可重试，或已达到最大重试次数
                    log_message_suffix = "已达到最大重试次数。" if attempt_num >= self.max_retries else "错误不可重试。"
                    logger.error(
                        f"操作 '{action_description}' 最终失败 (错误: "
                        f"状态码={e_api.status_code if e_api.status_code else 'N/A'}, 消息='{e_api}'). "
                        f"{log_message_suffix}"
                    )
                    raise # 重新抛出捕获到的 (可能是最后的) LLMAPIError

            except Exception as e_other: # 捕获 Provider 方法可能抛出的其他非 LLMAPIError 异常
                logger.error(
                    f"操作 '{action_description}' 执行期间发生未预料的非 API 错误: {type(e_other).__name__} - {e_other}",
                    exc_info=True # 记录完整堆栈
                )
                raise # 直接重新抛出这些异常，它们通常表示代码错误或非网络问题，不应重试

        # 此代码路径理论上不应到达，因为循环要么成功返回，要么在最后一次尝试失败时抛出异常。
        # 但为代码完整性和静态分析器的满意，添加一个保险措施。
        if last_api_exception: # 如果循环结束但有保存的异常 (意味着所有重试都失败了)
            raise last_api_exception # 重新抛出最后的 API 错误
        else:
            # 这是一个不太可能发生的极端情况，例如 max_retries < 0 (构造函数已转为int，所以不会)
            # 或者 action 从未被调用。
            critical_msg = (f"重试逻辑在操作 '{action_description}' 中异常结束，没有最终的 API 异常，"
                            "也没有成功返回。请检查重试逻辑和 Provider 实现。")
            logger.critical(critical_msg)
            raise LLMAPIError(critical_msg) # 抛出一个通用的 API 错误


    def _log_request_details(self,
                             log_type: str,          # "text_response" 或 "structured_response"
                             prompt_name: str,       # Prompt 模板名称
                             model_alias: str,       # 用户请求的模型别名
                             actual_model_name: str, # 实际调用的模型名称
                             filled_prompt: str,     # 填充后的完整 Prompt (将被屏蔽)
                             provider_response: ProviderResponse | None, # 来自 Provider 的响应对象 (如果成功)
                             error: Exception | None, # 发生的异常 (如果失败)
                             duration_ms: float,     # 请求总耗时 (毫秒)
                             output_schema_name: str | None = None # 仅用于 structured_response
                            ):
        """
        私有辅助方法，用于以结构化和可读的方式记录 LLM 请求的详细信息。
        包括请求参数、响应元数据（如 token 计数）、耗时以及任何发生的错误。

        日志会同时尝试输出一个人类可读的摘要行 (INFO/ERROR级别) 和一个详细的 JSON 对象 (DEBUG级别)。
        """
        log_entry: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()), # ISO 8601 时间戳
            "log_type": log_type,
            "prompt_name": prompt_name,
            "model_alias": model_alias,
            "actual_model_name": actual_model_name,
            "duration_ms": round(duration_ms, 2),
            "masked_prompt": self._partially_mask_prompt(filled_prompt), # 屏蔽敏感信息
            # "full_prompt_hash": hashlib.sha256(filled_prompt.encode()).hexdigest(), # 可选：记录完整 Prompt 的哈希值
        }
        if output_schema_name: # 如果是结构化响应，记录目标 schema 名称
            log_entry["output_schema"] = output_schema_name

        if provider_response: # 如果 Provider 调用成功并返回了响应
            log_entry["status"] = "success" # 此处的 "success" 指 API 调用成功，后续验证可能仍会失败
            log_entry["input_tokens"] = provider_response.input_tokens
            log_entry["output_tokens"] = provider_response.output_tokens
            log_entry["estimated_cost_usd"] = provider_response.cost # 可能为 None
            # 可以考虑记录部分响应内容进行预览，但需注意Pydantic对象的序列化和日志大小
            # log_entry["response_preview"] = self._partially_mask_prompt(provider_response.text_content, 50)

        if error: # 如果在处理过程中发生了任何错误
            log_entry["status"] = "failure" # 标记总体状态为失败
            log_entry["error_type"] = type(error).__name__ # 记录异常类型
            log_entry["error_message"] = str(error) # 记录异常消息
            if isinstance(error, LLMAPIError) and error.status_code:
                log_entry["error_status_code"] = error.status_code # 如果是 API 错误，记录状态码
            if isinstance(error, LLMOutputValidationError) and error.validation_errors:
                # Pydantic 的 errors() 方法返回的列表/字典可能非常大，记录时需谨慎
                # 可以考虑转换为字符串并截断，或仅记录错误数量/类型
                log_entry["validation_error_summary"] = f"共 {len(error.validation_errors)} 个验证错误"
                # log_entry["validation_details_preview"] = str(error.validation_errors)[:500] + "..."

        # --- 生成人类可读的日志行 ---
        status_message_part = "成功" if not error else f"失败 ({type(error).__name__})"
        readable_log_message = (
            f"LLM请求处理完毕: 类型='{log_type}', Prompt='{prompt_name}', 别名='{model_alias}', "
            f"实际模型='{actual_model_name}', 耗时={log_entry['duration_ms']:.2f}ms, 状态='{status_message_part}'"
        )
        if provider_response: # 附加 token 和成本信息 (如果可用)
            readable_log_message += (
                f", 输入Tokens={provider_response.input_tokens if provider_response.input_tokens is not None else 'N/A'}, "
                f"输出Tokens={provider_response.output_tokens if provider_response.output_tokens is not None else 'N/A'}, "
                f"估算成本USD={provider_response.cost if provider_response.cost is not None else 'N/A'}"
            )
        if error: # 附加错误消息摘要
             readable_log_message += f", 错误信息='{str(error)[:200]}...'" # 截断长错误消息

        # 根据是否有错误，选择日志级别 (ERROR 或 INFO)
        if error:
            logger.error(readable_log_message, extra={"llm_request_details": log_entry})
        else:
            logger.info(readable_log_message, extra={"llm_request_details": log_entry})

        # 始终以 DEBUG 级别记录完整的 JSON 结构化日志条目，以便详细分析
        # ensure_ascii=False 允许中文字符直接出现在 JSON 中，而不是被转义
        logger.debug(f"详细的 LLM 请求日志条目: {json.dumps(log_entry, indent=2, ensure_ascii=False)}")


    def get_text_response(self,
                          prompt_name: str,
                          context: Dict[str, Any],
                          model_alias: str = "fast_model", # 默认使用 "fast_model" 别名
                          **provider_kwargs: Any # 允许传递额外的关键字参数给 Provider
                          ) -> str:
        """
        获取通用的文本响应。

        此方法处理整个流程：格式化 Prompt、选择 Provider 和模型、
        执行 API 调用（带重试）、记录详细审计日志，并返回最终的文本结果。

        :param prompt_name: 在 `prompts.toml` 中定义的 Prompt 模板的名称。
        :param context: 一个字典，用于填充 Prompt 模板中的变量。
        :param model_alias: 在 `config.yaml` 中定义的模型别名，用于选择要调用的 LLM。
                            默认为 "fast_model"。
        :param provider_kwargs: 传递给底层 Provider 的 `generate_text` 方法的额外参数
                                (例如 `temperature`, `max_tokens`, `top_p`, `top_k` 等)。
                                这些参数会影响 LLM 的生成行为。
        :return: LLM 生成的文本字符串。
        :raises LLMAPIError: 如果与 LLM API 的通信失败（包括所有重试尝试之后）。
        :raises PromptTemplateError: 如果指定的 `prompt_name` 未找到或 `context` 无法正确格式化模板。
        :raises ValueError: 如果提供的 `model_alias` 无效或未在配置中定义。
        :raises ConfigurationError: 如果网关或其依赖的 Provider 未能正确配置。
        """
        request_start_time = time.perf_counter() # 用于精确计时
        provider: Optional[BaseProvider] = None   # 初始化为 None，以便在 finally 中安全访问
        actual_model_name: str = "N/A"            # 同上
        filled_prompt: str = ""                   # 同上
        provider_api_response: Optional[ProviderResponse] = None # Provider 返回的原始响应对象
        error_occurred: Optional[Exception] = None # 记录过程中发生的任何异常

        try:
            logger.info(
                f"开始处理文本响应请求: prompt_name='{prompt_name}', model_alias='{model_alias}', "
                f"附带的 provider_kwargs: {provider_kwargs}"
            )
            # 1. 解析模型别名，获取 Provider 实例和实际模型名称
            provider, actual_model_name = self._get_provider_and_model(model_alias)

            # 2. 格式化 Prompt 模板
            filled_prompt = self._format_prompt(prompt_name, context)

            logger.debug(
                f"准备向 Provider '{provider.provider_name}' 的模型 '{actual_model_name}' 发送请求。"
            )
            # 记录部分屏蔽后的 Prompt，避免日志中出现过多或敏感的 Prompt 内容
            logger.debug(f"格式化后的 Prompt (部分屏蔽): {self._partially_mask_prompt(filled_prompt)}")

            # 3. 定义要通过重试机制执行的操作 (即调用 Provider 的方法)
            # 使用 lambda 延迟执行，并捕获当前作用域的变量
            action_to_execute = lambda: provider.generate_text( # type: ignore[union-attr] (mypy 可能抱怨 provider 是 Optional)
                prompt=filled_prompt,
                model_name=actual_model_name,
                **provider_kwargs # 将 get_text_response 接收到的额外参数透传给 Provider
            )

            # 4. 执行操作（带重试）
            provider_api_response = self._execute_with_retry(
                action=action_to_execute,
                action_description=f"文本生成 (prompt: '{prompt_name}', alias: '{model_alias}')",
            )

            # 5. 验证 Provider 的响应是否符合预期
            if not provider_api_response or not isinstance(provider_api_response.text_content, str):
                # Provider 的实现应确保 text_content 始终是字符串
                err_msg = "LLM Provider 未能按预期返回有效的文本内容 (text_content)。"
                logger.error(
                    err_msg + f" 收到的响应类型: "
                    f"{type(provider_api_response.text_content if provider_api_response else None)}"
                )
                # 这通常表示 Provider 实现有问题，或者 API 返回了意外的空结构
                raise LLMAPIError(err_msg) # 标记为 API 错误，因为它涉及与 Provider 的契约

            # 6. 返回文本内容
            return provider_api_response.text_content

        except Exception as e: # 捕获所有可能的异常 (LLMAPIError, PromptTemplateError, ValueError, ConfigurationError等)
            error_occurred = e # 保存异常以供日志记录
            # 日志已在 _execute_with_retry 或其他辅助方法中记录了具体错误，这里记录总体处理失败
            logger.error(f"处理文本响应请求 (prompt: '{prompt_name}', alias: '{model_alias}') 失败: {type(e).__name__} - {e}")
            raise # 将原始异常重新抛出，让调用者处理

        finally: # 无论成功或失败，都执行日志记录
            duration_ms = (time.perf_counter() - request_start_time) * 1000.0 # 计算总耗时
            self._log_request_details(
                log_type="text_response",
                prompt_name=prompt_name,
                model_alias=model_alias,
                actual_model_name=actual_model_name, # 可能在出错时仍为 "N/A"
                filled_prompt=filled_prompt,         # 可能在出错时仍为空字符串
                provider_response=provider_api_response, # 可能为 None
                error=error_occurred,                # 可能为 None
                duration_ms=duration_ms
            )


    def get_structured_response(self,
                                prompt_name: str,
                                context: Dict[str, Any],
                                output_schema: Type[BaseModel],
                                model_alias: str = "smart_model",
                                **provider_kwargs: Any # 例如 temperature, max_tokens 等
                                ) -> BaseModel:
        """
        获取经过验证的结构化数据响应。
        集成了 Provider 调用、重试、Pydantic 验证和详细日志记录。

        :param prompt_name: 模板名称。
        :param context: 填充模板的上下文。
        :param output_schema: 用于验证输出的Pydantic模型类。
        :param model_alias: 模型别名。
        :param provider_kwargs: 传递给底层 provider 的额外参数。
        :return: 一个实例化且经过验证的Pydantic数据对象。
        :raises LLMOutputValidationError: 如果输出不符合schema，或多次尝试修正后仍不符合。
        :raises LLMAPIError: 如果API调用（包括重试后）失败。
        :raises PromptTemplateError: 如果Prompt格式化失败。
        :raises ValueError: 如果 model_alias 无效。
        :raises ConfigurationError: 如果Provider未正确配置。
        """
        request_start_time = time.perf_counter()
        provider: Optional[BaseProvider] = None
        actual_model_name: str = "N/A"
        filled_prompt: str = ""
        provider_response_obj: Optional[ProviderResponse] = None # 重命名以区分 provider_response 变量
        error_occurred: Optional[Exception] = None
        validated_output: Optional[BaseModel] = None


        # TODO: 实现可选的“修正请求”逻辑，如果首次验证失败
        # num_correction_attempts = int(self.config.get('structured_output_correction_attempts', 0))

        try:
            logger.info(
                f"开始处理结构化响应请求: prompt_name='{prompt_name}', model_alias='{model_alias}', "
                f"output_schema='{output_schema.__name__}', provider_kwargs={provider_kwargs}"
            )
            provider, actual_model_name = self._get_provider_and_model(model_alias)
            # 结构化输出的 prompt 可能需要包含 schema 的描述或 JSON 格式指令
            # 例如: context['json_schema_description'] = output_schema.model_json_schema(indent=2)
            # 然后在 prompts.toml 中引用 {json_schema_description}
            filled_prompt = self._format_prompt(prompt_name, context)

            logger.debug(f"向 Provider '{provider.provider_name}' 模型 '{actual_model_name}' 发送结构化请求。")
            logger.debug(f"完整 Prompt (部分屏蔽): {self._partially_mask_prompt(filled_prompt)}")

            action_to_execute = lambda: provider.generate_structured_text( # type: ignore[union-attr]
                prompt=filled_prompt,
                model_name=actual_model_name,
                output_schema=output_schema, # Provider 可能会使用它来调整内部提示或参数
                **provider_kwargs
            )

            provider_response_obj = self._execute_with_retry(
                action=action_to_execute,
                action_description=f"generate_structured_text for {prompt_name} via {model_alias}"
            )

            if not provider_response_obj or not isinstance(provider_response_obj.text_content, str):
                err_msg = "Provider未能返回有效的文本内容以进行结构化解析。"
                logger.error(err_msg + f" 响应类型: {type(provider_response_obj.text_content if provider_response_obj else None)}")
                raise LLMAPIError(err_msg) # 或者 LLMOutputValidationError? 倾向于 APIError 如果内容都没有

            raw_json_output = provider_response_obj.text_content.strip()
            # 尝试清理常见的 LLM 在 JSON 前后添加的 markdown 代码块标记
            if raw_json_output.startswith("```json"):
                raw_json_output = raw_json_output[7:]
            if raw_json_output.endswith("```"):
                raw_json_output = raw_json_output[:-3]
            raw_json_output = raw_json_output.strip()


            logger.debug(f"从 Provider 收到的原始结构化文本 (清理后): '{raw_json_output[:500]}...'")

            try:
                # 使用 Pydantic 进行解析和验证
                # Pydantic v2 使用 model_validate_json, v1 使用 parse_raw_as(List[Model], ...) or parse_obj_as
                if hasattr(output_schema, 'model_validate_json'): # Pydantic v2+
                    validated_output = output_schema.model_validate_json(raw_json_output)
                else: # Pydantic v1 compatibility (approximate)
                    # For Pydantic v1, if output_schema is a list of models, this is more complex.
                    # Assuming output_schema is a single model for simplicity here.
                    # validated_output = output_schema.parse_raw(raw_json_output) # type: ignore
                    # A more robust v1 approach might be:
                    parsed_dict = json.loads(raw_json_output)
                    validated_output = output_schema(**parsed_dict) # type: ignore

                logger.info(f"LLM 输出成功通过 Pydantic schema '{output_schema.__name__}' 验证。")
                return validated_output # type: ignore

            except pydantic.ValidationError as e_val:
                # Pydantic 验证失败
                validation_error_details = e_val.errors() if hasattr(e_val, 'errors') else str(e_val)
                log_msg = (
                    f"LLM 输出未能通过 Pydantic schema '{output_schema.__name__}' 验证. "
                    f"错误: {validation_error_details}. "
                    f"原始输出 (部分): '{raw_json_output[:500]}...'"
                )
                logger.error(log_msg)
                # 抛出自定义的验证错误，包含 Pydantic 的错误信息
                # 将原始的 Pydantic ValidationError 作为 original_exception 传递
                raise LLMOutputValidationError(
                    message=f"LLM 输出不符合 Pydantic schema '{output_schema.__name__}'.",
                    validation_errors=validation_error_details, # type: ignore
                    original_exception=e_val
                ) from e_val
            except json.JSONDecodeError as e_json:
                # 如果 LLM 返回的不是合法的 JSON
                log_msg = (
                    f"LLM 输出不是有效的 JSON 格式，无法使用 Pydantic schema '{output_schema.__name__}' 解析. "
                    f"JSON 解析错误: {e_json}. "
                    f"原始输出 (部分): '{raw_json_output[:500]}...'"
                )
                logger.error(log_msg)
                raise LLMOutputValidationError(
                    message="LLM 输出不是有效的 JSON。",
                    original_exception=e_json
                ) from e_json


        except Exception as e: # 包括 LLMAPIError, PromptTemplateError, ValueError, ConfigurationError, LLMOutputValidationError
            error_occurred = e
            logger.error(f"处理结构化响应请求失败: {type(e).__name__} - {e}", exc_info=True)
            raise
        finally:
            duration_ms = (time.perf_counter() - request_start_time) * 1000
            self._log_request_details(
                log_type="structured_response",
                prompt_name=prompt_name,
                model_alias=model_alias,
                actual_model_name=actual_model_name if provider else "N/A",
                filled_prompt=filled_prompt,
                provider_response=provider_response_obj, # 使用重命名后的变量
                error=error_occurred,
                duration_ms=duration_ms,
                output_schema_name=output_schema.__name__
            )


    def _partially_mask_prompt(self, prompt: str, visible_chars_each_side: int = 100) -> str:
        # (此方法实现已在步骤3中完善，保持不变)
        if not isinstance(prompt, str):
            return "<非字符串类型>"
        prompt_len = len(prompt)
        if prompt_len <= visible_chars_each_side * 2:
            return prompt

        ellipsis = "...[内容已屏蔽]..."
        return f"{prompt[:visible_chars_each_side]}{ellipsis}{prompt[prompt_len-visible_chars_each_side:]}"

# 示例 Pydantic 模型 (用于测试)
class SuggestedDatasets(BaseModel):
    analysis_type: str
    datasets: list[dict]

# 主函数块，用于基本测试和演示。后续将用单元测试替代。
if __name__ == '__main__':
    # 为了能运行这个示例，需要创建临时的配置文件和 prompt 文件
    # 或者确保当前目录下有符合预期的 config.yaml 和 prompts.toml

    TEMP_CONFIG_PATH = "temp_llm_gateway_config.yaml"
    TEMP_PROMPTS_PATH = "temp_llm_gateway_prompts.toml"

    # 1. 创建示例 config.yaml
    sample_config_content = """
api_keys:
  google_gemini: "env:GEMINI_API_KEY_DUMMY"
  # anthropic_claude: "env:ANTHROPIC_API_KEY_DUMMY"

models:
  - name: "gemini-1.5-pro-latest"
    provider: "google" # 对应 api_keys 中的 'google_gemini'
    aliases: ["smart_model", "default_text_model"]
  - name: "gemini-1.5-flash-latest"
    provider: "google"
    aliases: ["fast_model"]
  # - name: "claude-3-opus-20240229"
  #   provider: "anthropic"
  #   aliases: ["claude_opus", "advanced_reasoning"]

retry_policy:
  max_retries: 2
  backoff_factor: 1.5
  initial_wait_seconds: 0.5
  max_wait_seconds: 30

# 可选: Provider 特定配置
# provider_specific_configs:
#   google:
#     default_temperature: 0.6
#     safety_settings: # 覆盖 GeminiProvider 中的默认安全设置
#       HARM_CATEGORY_HARASSMENT: "BLOCK_NONE"
#       # 注意: 这里的键需要和 google.generativeai.types.HarmCategory 的枚举成员名称字符串匹配
#       # 或者在 GeminiProvider 中处理字符串到枚举的转换
"""
    with open(TEMP_CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(sample_config_content)
    logger.info(f"创建了临时配置文件: {TEMP_CONFIG_PATH}")

    # 2. 创建示例 prompts.toml
    sample_prompts_content = """
[propose_strategy]
template = '''
将以下科研问题转化为搜索关键词: "{user_query}"
请以JSON格式返回，包含'pubmed_query'和'analysis_type'字段。
'''

[generate_samplesheet]
template = """为文件列表 {file_list} 生成 samplesheet。"""

[summarize]
template = "请总结以下文本：{text_to_summarize}"
"""
    with open(TEMP_PROMPTS_PATH, "w", encoding="utf-8") as f:
        f.write(sample_prompts_content)
    logger.info(f"创建了临时 Prompt 文件: {TEMP_PROMPTS_PATH}")

    # 设置一个虚拟的环境变量用于测试 API key 加载
    os.environ["GEMINI_API_KEY_DUMMY"] = "dummy_gemini_api_key_for_testing_12345"
    logger.info("设置了虚拟环境变量 GEMINI_API_KEY_DUMMY。")

    # 将日志级别设置为 DEBUG 以查看详细的加载过程
    logger.setLevel(logging.DEBUG)
    # 如果其他模块也使用 logging，可能需要更精细地控制 logger
    # logging.getLogger("aibiowflow.llm_gateway.gateway").setLevel(logging.DEBUG)


    try:
        logger.info("--- 开始 LLMGateway 初始化测试 ---")
        # 注意：此时 _init_providers 尚未完全实现，所以 providers 字典会是空的
        gateway = LLMGateway(config_path=TEMP_CONFIG_PATH, prompt_path=TEMP_PROMPTS_PATH)
        logger.info("--- LLMGateway 初始化成功 ---")

        # 验证配置是否按预期加载
        assert gateway.config['api_keys']['google_gemini'] == "dummy_gemini_api_key_for_testing_12345"
        assert "propose_strategy" in gateway.prompts
        assert gateway.model_alias_map["fast_model"]["model_name"] == "gemini-1.5-flash-latest"
        assert gateway.max_retries == 2

        logger.info("\n--- 测试 get_text_response (fast_model) ---")
        # 由于 providers 为空，这将使用占位符逻辑并可能记录警告
        text_resp = gateway.get_text_response(
            prompt_name="summarize",
            context={"text_to_summarize": "这是一个长长的文本，需要被总结。它包含很多细节。"},
            model_alias="fast_model"
        )
        logger.info(f"文本响应 (模拟): {text_resp}")
        assert "gemini-1.5-flash-latest" in text_resp # 检查模型名称是否在模拟响应中

        logger.info("\n--- 测试 get_structured_response (smart_model) ---")
        # 同样，这将使用占位符逻辑
        structured_resp = gateway.get_structured_response(
            prompt_name="propose_strategy", # 这个 prompt 暗示了 JSON 输出
            context={"user_query": "寻找关于特定蛋白质的公共数据集"},
            output_schema=SuggestedDatasets,
            model_alias="smart_model"
        )
        logger.info(f"结构化响应 (模拟): {structured_resp.model_dump_json(indent=2)}")
        assert isinstance(structured_resp, SuggestedDatasets)
        assert structured_resp.analysis_type is not None # 检查模拟数据是否已填充

        logger.info("\n--- 测试无效的 model_alias ---")
        try:
            gateway.get_text_response("summarize", {"text_to_summarize":"..." }, "non_existent_model_alias_123")
        except ValueError as e:
            logger.info(f"成功捕获预期的错误: {e}")
            assert "non_existent_model_alias_123" in str(e)
        else:
            assert False, "未捕获到无效 model_alias 的 ValueError"


        logger.info("\n--- 测试无效的 prompt_name ---")
        try:
            gateway.get_text_response("non_existent_prompt_name_456", {"text":"..."}, "fast_model")
        except PromptTemplateError as e:
            logger.info(f"成功捕获预期的错误: {e}")
            assert "non_existent_prompt_name_456" in str(e)
        else:
            assert False, "未捕获到无效 prompt_name 的 PromptTemplateError"

        logger.info("\n--- 测试 Prompt 格式化错误 (缺少键) ---")
        try:
            # 'summarize' prompt 需要 'text_to_summarize'
            gateway.get_text_response("summarize", {"wrong_key": "some text"}, "fast_model")
        except PromptTemplateError as e:
            logger.info(f"成功捕获预期的 Prompt 格式化错误: {e}")
            assert "text_to_summarize" in str(e) # 错误信息应提示缺少的键
        else:
            assert False, "未捕获到 Prompt 格式化错误"


    except ConfigurationError as e:
        logger.error(f"测试过程中发生配置错误: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"测试过程中发生未预料的错误: {e}", exc_info=True)
    finally:
        # 清理临时文件
        if os.path.exists(TEMP_CONFIG_PATH):
            os.remove(TEMP_CONFIG_PATH)
            logger.info(f"已删除临时配置文件: {TEMP_CONFIG_PATH}")
        if os.path.exists(TEMP_PROMPTS_PATH):
            os.remove(TEMP_PROMPTS_PATH)
            logger.info(f"已删除临时 Prompt 文件: {TEMP_PROMPTS_PATH}")

        # 清理环境变量 (如果之前未设置)
        if "GEMINI_API_KEY_DUMMY_PREVIOUS_VALUE" in os.environ: # 假设我们保存了原始值
            os.environ["GEMINI_API_KEY_DUMMY"] = os.environ["GEMINI_API_KEY_DUMMY_PREVIOUS_VALUE"]
            del os.environ["GEMINI_API_KEY_DUMMY_PREVIOUS_VALUE"]
        elif "GEMINI_API_KEY_DUMMY" in os.environ and os.environ["GEMINI_API_KEY_DUMMY"] == "dummy_gemini_api_key_for_testing_12345":
             del os.environ["GEMINI_API_KEY_DUMMY"]
        logger.info("环境变量 GEMINI_API_KEY_DUMMY 已清理/恢复。")

        logger.info("\n--- LLMGateway 配置加载及基础功能测试完成 ---")

"""
# 占位符，实际的 Provider 实现将在这里进行。
# 例如，对于 Gemini:
# from .providers import GeminiProvider (或者直接导入类)
# ... 在 _init_providers 中 ...
# self.providers["google"] = GeminiProvider(api_key=...)
"""
