# -*- coding: utf-8 -*-
"""
Google Gemini LLM 提供商适配器实现。

此类实现了 BaseProvider 接口，用于与 Google Gemini API 进行交互。
"""

import logging
import json # 用于结构化输出的辅助（如果需要手动构造）
from pydantic import BaseModel
from typing import Type, Any, Dict, cast, Optional

# 尝试导入 google.generativeai 相关组件
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold # type: ignore
    from google.generativeai.types import GenerateContentResponse # 用于类型提示
    from google.generativeai.types.generation_types import BlockedPromptException, StopCandidateException # 更具体的异常
    from google.api_core.exceptions import GoogleAPIError # Google 通用 API 错误
except ImportError:
    genai = None # type: ignore # 标记为 None，以便后续检查
    GenerationConfig = None # type: ignore
    HarmCategory = None # type: ignore
    HarmBlockThreshold = None # type: ignore
    GenerateContentResponse = None # type: ignore
    BlockedPromptException = None # type: ignore
    StopCandidateException = None # type: ignore
    GoogleAPIError = None # type: ignore


from ..exceptions import LLMAPIError, ConfigurationError
from .base_provider import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__) # 获取模块特定的 logger

# 默认的安全设置类别，可以根据需要调整或从配置加载
# 这些键需要是 HarmCategory 枚举成员的字符串表示，以便在配置中易于定义
DEFAULT_HARM_CATEGORIES_STR = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]

def get_default_safety_settings() -> Optional[Dict[HarmCategory, HarmBlockThreshold]]:
    """
    获取默认安全设置。
    仅当 HarmCategory 和 HarmBlockThreshold 从 google.generativeai 成功导入时才构造设置。
    :return: 包含默认安全规则的字典，或在无法导入所需类型时返回 None。
    """
    if HarmCategory is None or HarmBlockThreshold is None: # 检查类型是否成功导入
        return None

    # 将字符串列表转换为 HarmCategory 枚举成员
    harm_categories_enum = []
    for cat_str in DEFAULT_HARM_CATEGORIES_STR:
        try:
            harm_categories_enum.append(HarmCategory[cat_str]) # type: ignore
        except KeyError:
            logger.warning(f"默认安全类别 '{cat_str}' 不是有效的 HarmCategory 枚举成员。将被忽略。")

    return {
        category: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE # type: ignore
        for category in harm_categories_enum
    }


class GeminiProvider(BaseProvider):
    """
    Google Gemini LLM 提供商适配器。
    此类封装了与 Google Gemini API 交互的所有逻辑，包括认证、请求构建、
    响应解析和错误处理。
    """
    PROVIDER_ID = "google" # 提供商的唯一标识符

    def __init__(self, api_key: str | None, provider_config: Dict[str, Any] | None = None):
        """
        初始化 Gemini 提供商适配器。

        :param api_key: Google Gemini API 密钥。如果为 None，库将尝试其他认证方法（例如，ADC）。
        :param provider_config: 特定于此提供商的额外配置项（来自网关的主配置文件）。
                                 支持的键包括:
                                 - `safety_settings` (Dict[str, str]): 一个字典，键是 HarmCategory 枚举的字符串名称
                                   (例如 "HARM_CATEGORY_HARASSMENT")，值是 HarmBlockThreshold 枚举的字符串名称
                                   (例如 "BLOCK_NONE")。
                                 - `default_model_kwargs` (Dict[str, Any]): 将作为默认参数传递给
                                   `GenerationConfig` 的键值对 (例如 `{"top_p": 0.9}` )。
        :raises ConfigurationError: 如果 `google-generativeai` 库未安装。
        :raises LLMAPIError: 如果 API 密钥配置失败或认证过程中发生严重错误。
        """
        if genai is None: # 检查 google.generativeai 是否成功导入
            msg = ("google-generativeai 包未安装。请运行 'pip install google-generativeai' 来安装。")
            logger.error(msg)
            raise ConfigurationError(msg) # 抛出配置错误，因为这是运行时的先决条件

        # 配置 Gemini API 客户端
        if not api_key:
            # 当 api_key 为 None 或空字符串时，genai.configure() 不应被调用，
            # 库会尝试使用应用程序默认凭据 (ADC) 或其他已配置的环境。
            logger.info("Gemini API 密钥未在配置中直接提供。将依赖 google-generativeai 库的默认认证机制 (如 ADC 或环境变量 GOOGLE_API_KEY)。")
            # 探测性调用以确保认证已设置（可选，但有助于早期失败）
            # try:
            #    if not list(genai.list_models()): # 尝试列出模型，如果认证失败会抛异常
            #        logger.warning("Gemini: genai.list_models() 返回空列表，可能表示认证问题或无可用模型。")
            # except Exception as e:
            #    msg = f"Gemini: 探测性API调用失败，可能由于认证未正确配置: {e}"
            #    logger.error(msg, exc_info=True)
            #    raise LLMAPIError(msg, original_exception=e) from e
        else:
            # 如果提供了 API 密钥，则使用它配置库
            try:
                genai.configure(api_key=api_key)
                logger.info("Gemini API 密钥已成功配置。")
            except Exception as e: # pylint: disable=broad-except
                # genai.configure 可能会因无效密钥等原因抛出各种异常
                msg = f"使用提供的 API 密钥配置 Google Gemini API 失败: {e}"
                logger.error(msg, exc_info=True)
                raise LLMAPIError(msg, original_exception=e) from e # 包装为 LLMAPIError

        self.client = genai # 存储配置好的 genai 模块引用
        self.provider_config = provider_config or {} # 确保 provider_config 是一个字典

        # 解析来自 provider_config 的安全设置
        # `_parse_safety_settings` 会处理字符串到枚举的转换和默认值
        self.safety_settings = self._parse_safety_settings(
            self.provider_config.get("safety_settings")
        )

        # 获取特定于模型的默认参数，例如 GenerationConfig 的参数
        self.default_model_kwargs = self.provider_config.get("default_model_kwargs", {})

        logger.info(
            f"GeminiProvider 初始化完成。将使用的安全设置: "
            f"{ {k.name: v.name for k,v in self.safety_settings.items()} if self.safety_settings else 'API 默认'}"
        )

    def _parse_safety_settings(self, custom_settings_dict: Any) -> Optional[Dict[HarmCategory, HarmBlockThreshold]]:
        """
        从配置字典解析安全设置。
        配置中的键和值应该是 HarmCategory 和 HarmBlockThreshold 枚举成员的字符串名称。

        :param custom_settings_dict: 从配置文件传入的 safety_settings 字典。
        :return: 解析后的安全设置字典 (枚举键和值)，或在无法导入类型时返回 None，或在配置无效时返回默认设置。
        """
        if HarmCategory is None or HarmBlockThreshold is None: # 检查类型是否已加载
            logger.warning("无法解析安全设置，因为 google.generativeai.types 未完全加载。将不使用显式安全设置。")
            return None # Provider 将使用 API 的默认安全设置

        default_settings = get_default_safety_settings() # 获取预定义的默认值

        if not custom_settings_dict: # 如果配置中没有提供 safety_settings
            logger.debug("未提供自定义安全设置，使用默认安全设置。")
            return default_settings

        if not isinstance(custom_settings_dict, dict):
            logger.warning(
                f"自定义安全设置格式不正确（应为字典），将使用默认安全设置。收到类型: {type(custom_settings_dict)}."
            )
            return default_settings

        parsed_settings: Dict[HarmCategory, HarmBlockThreshold] = {}
        # 创建从字符串名称到枚举成员的映射，以便查找
        # type: ignore 用于告诉 MyPy/Pyright HarmCategory[...] 和 HarmBlockThreshold[...] 是有效的
        valid_harm_categories_map = {hc.name: hc for hc in HarmCategory} # type: ignore
        valid_block_thresholds_map = {hbt.name: hbt for hbt in HarmBlockThreshold} # type: ignore

        for key_str, value_str in custom_settings_dict.items():
            category = valid_harm_categories_map.get(key_str)
            threshold = valid_block_thresholds_map.get(str(value_str)) # 确保 value_str 是字符串

            if category and threshold:
                parsed_settings[category] = threshold
            else:
                logger.warning(
                    f"无法识别的安全设置键 '{key_str}' (应为 HarmCategory 名称) 或值 '{value_str}' "
                    f"(应为 HarmBlockThreshold 名称)。将忽略此条目。"
                )

        if not parsed_settings and custom_settings_dict: # 如果有自定义设置但所有条目都无效
            logger.warning("所有自定义安全设置条目均无效，将使用默认安全设置。")
            return default_settings

        logger.info(f"成功解析并应用自定义安全设置: {{ {', '.join([f'{k.name}: {v.name}' for k,v in parsed_settings.items()])} }}")
        return parsed_settings


    @property
    def provider_name(self) -> str:
        """返回此提供商的唯一标识符。"""
        return self.PROVIDER_ID

    def _prepare_generation_config(
        self,
        temperature: float | None = None,
        max_tokens: int | None = None, # Gemini API 使用 'max_output_tokens'
        stop_sequences: list[str] | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        model_kwargs: Dict[str, Any] | None = None # 其他特定于模型的参数 (来自调用者)
    ) -> GenerationConfig | None:
        """
        辅助函数，用于创建 Gemini API 的 GenerationConfig 对象。
        它合并了默认参数、方法调用参数和特定模型参数。

        :param temperature: 控制生成文本的随机性。
        :param max_tokens: 生成文本的最大 token 数量 (在 Gemini 中为 `max_output_tokens`)。
        :param stop_sequences: 遇到这些序列时停止生成。
        :param top_p: Top-p (nucleus) 采样参数。
        :param top_k: Top-k 采样参数。
        :param model_kwargs: 从 `get_text_response` 或 `get_structured_response` 传入的额外参数。
        :return: 配置好的 `GenerationConfig` 实例，如果没有指定任何参数则返回 `None`。
        """
        if GenerationConfig is None: # 检查 GenerationConfig 是否已成功导入
            logger.warning("GenerationConfig 类型未加载，无法准备生成配置。将不使用显式 GenerationConfig。")
            return None

        config_params: Dict[str, Any] = {}

        # 优先使用方法调用时明确传入的参数
        if temperature is not None:
            config_params["temperature"] = temperature
        if max_tokens is not None:
            config_params["max_output_tokens"] = max_tokens # Gemini API 特定的参数名
        if stop_sequences: # 注意: stop_sequences 可以是 None 或空列表
            config_params["stop_sequences"] = stop_sequences
        if top_p is not None:
            config_params["top_p"] = top_p
        if top_k is not None:
            config_params["top_k"] = top_k

        # 合并来自 provider_config 的默认模型参数 (`self.default_model_kwargs`)
        # 和来自方法调用的 `model_kwargs`。
        # `model_kwargs` (来自方法调用) 具有更高优先级。
        # `self.default_model_kwargs` (来自配置文件) 作为备选。
        final_model_kwargs = {**self.default_model_kwargs, **(model_kwargs or {})}
        config_params.update(final_model_kwargs)

        # 过滤掉不是 GenerationConfig 有效字段的参数，以避免 API 错误
        # 获取 GenerationConfig 所有有效字段的名称集合
        # GenerationConfig.__annotations__ 包含所有已类型注解的字段
        valid_gc_keys = set(GenerationConfig.__annotations__.keys())
        filtered_config_params = {k: v for k, v in config_params.items() if k in valid_gc_keys}

        if not filtered_config_params: # 如果没有指定任何有效的配置参数
            return None # API 将使用其内部默认值

        logger.debug(f"准备的 GenerationConfig 参数: {filtered_config_params}")
        return GenerationConfig(**filtered_config_params) # type: ignore (参数已过滤)


    def _handle_gemini_exception(self, e: Exception, model_name: str, prompt_length_for_log: int = 200) -> LLMAPIError:
        """
        统一处理来自 Gemini API 调用的异常，将其包装为 `LLMAPIError`。

        :param e: 捕获到的原始异常。
        :param model_name: 发生错误的模型名称。
        :param prompt_length_for_log: (未使用，但保留以备将来可能的 prompt 日志记录)
        :return: 一个包含原始异常信息的 `LLMAPIError` 实例。
        """
        # 检查是否是 GoogleAPIError (如果已导入且可用)
        if GoogleAPIError and isinstance(e, GoogleAPIError):
            status_code = e.code if hasattr(e, 'code') else None # 获取 HTTP 状态码 (如果存在)
            error_message = f"Gemini API 调用失败 (模型: {model_name}, GoogleAPIError): {e.message}"
            logger.error(error_message, exc_info=True) # 记录完整堆栈跟踪
            return LLMAPIError(error_message, status_code=status_code, original_exception=e)

        # 检查是否是特定于 Gemini 内容生成的异常 (如果已导入且可用)
        if BlockedPromptException and isinstance(e, BlockedPromptException):
            # 当 prompt 或响应因安全设置而被阻止时发生
            # e.response 对象可能包含更多关于阻塞原因的信息
            feedback_info = ""
            if hasattr(e, 'response') and e.response and hasattr(e.response, 'prompt_feedback'): # type: ignore
                 feedback_info = f" Prompt Feedback: {e.response.prompt_feedback}." # type: ignore
            error_message = f"Gemini API 调用失败 (模型: {model_name}): Prompt 内容或响应被安全策略阻止.{feedback_info}"
            logger.warning(error_message) # 通常不需要完整堆栈跟踪，因为这是预期的 API 行为
            return LLMAPIError(error_message, original_exception=e) # status_code 可能不适用

        if StopCandidateException and isinstance(e, StopCandidateException):
            # 当所有候选内容都因非错误原因停止时发生（例如，达到 max_tokens 或遇到 stop_sequence）
            # 这通常不被视为一个“错误”，而是正常的生成终止。
            # 但如果意外发生，或者 finish_reason 是 RECITATION 或 SAFETY，则可能需要关注。
            finish_reason_info = ""
            if hasattr(e, 'response') and e.response and e.response.candidates: # type: ignore
                finish_reason_info = f" Finish Reason: {e.response.candidates[0].finish_reason}." # type: ignore
            error_message = f"Gemini API 内容生成提前停止 (模型: {model_name}){finish_reason_info} Raw exception: {e}."
            logger.warning(error_message) # 通常是警告级别
            # 根据具体情况，这可能不应该被视为一个需要重试的 API 错误，
            # 但为了统一接口，我们仍将其包装。调用者可能需要检查原始异常类型。
            return LLMAPIError(error_message, original_exception=e)

        # 通用异常处理：捕获所有其他类型的异常
        logger.error(f"与 Gemini API (模型: {model_name}) 通信时发生未知错误: {e}", exc_info=True)
        return LLMAPIError(f"与 Gemini API (模型: {model_name}) 通信时发生错误: {e}", original_exception=e)


    def generate_text(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.7, # 默认值与 BaseProvider 一致
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any # 包含 top_p, top_k 等，将传递给 _prepare_generation_config
    ) -> ProviderResponse:
        """
        使用 Gemini API 生成通用文本响应。

        :param prompt: 发送给 LLM 的 Prompt 字符串。
        :param model_name: 要使用的 Gemini 模型名称 (例如 "gemini-1.5-pro-latest")。
        :param temperature: 控制生成文本的随机性。
        :param max_tokens: 生成文本的最大 token 数量 (对应 Gemini 的 `max_output_tokens`)。
        :param stop_sequences: 遇到这些序列时停止生成。
        :param kwargs: 其他特定于提供商的参数 (例如 `top_p`, `top_k`)，将传递给 `GenerationConfig`。
        :return: 一个 `ProviderResponse` 对象，包含生成的文本和元数据。
        :raises LLMAPIError: 如果 API 调用（包括重试后）失败。
        """
        logger.debug(
            f"GeminiProvider: 调用 generate_text, model='{model_name}', temp={temperature}, "
            f"max_tokens={max_tokens}, stop_sequences={stop_sequences is not None}, kwargs={kwargs}"
        )

        try:
            # 初始化 Gemini 模型客户端
            # safety_settings 从 __init__ 中获取，可以是 None (API 默认) 或解析后的字典
            model = self.client.GenerativeModel(
                model_name=model_name,
                safety_settings=self.safety_settings # type: ignore
            )
        except Exception as e_model_init: # pylint: disable=broad-except
            # 例如，如果模型名称无效或不受支持
            raise self._handle_gemini_exception(e_model_init, model_name)

        # 准备 GenerationConfig
        generation_config = self._prepare_generation_config(
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            top_p=kwargs.pop('top_p', None), # 从 kwargs 中提取已知参数以传递给 helper
            top_k=kwargs.pop('top_k', None),
            model_kwargs=kwargs # 剩余的 kwargs 作为特定模型参数传递
        )

        try:
            # 调用 Gemini API 的 generate_content 方法
            # type: ignore 用于 GenerateContentResponse，因为它可能未从 try-except 块导入
            response: GenerateContentResponse = model.generate_content( # type: ignore
                contents=prompt, # API 需要 `contents` 参数
                generation_config=generation_config # 可以是 None
            )

            # 检查响应的有效性：是否包含任何候选内容
            if not response.candidates:
                # 如果没有候选内容，检查 prompt_feedback 是否有阻塞信息
                block_reason_msg = ""
                # response.prompt_feedback 可能为 None
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason_msg = f"原因: {response.prompt_feedback.block_reason.name}." # type: ignore

                finish_reason_msg = ""
                # 即使没有 candidates，有时 finish_reason 会在 prompt_feedback 中
                # 或者如果 candidates 列表存在但为空，可能在第一个 candidate (如果存在)
                # 这里主要关注 prompt_feedback

                error_msg = f"Gemini API (模型: {model_name}) 返回了没有候选内容的响应。{block_reason_msg}"
                logger.warning(error_msg + f" 原始响应: {response}")
                # 可以考虑创建一个更具体的 ContentBlockedError 或基于 block_reason 的错误
                raise LLMAPIError(error_msg, raw_response=response) # 包含原始响应以供调试

            # Gemini API 通常返回一个包含 parts 的列表，response.text 会自动拼接这些 parts
            text_content = response.text

            input_tokens: int | None = None
            output_tokens: int | None = None

            # 尝试从 usage_metadata 获取 token 计数 (如果可用)
            # usage_metadata 的可用性可能因模型和 API 版本而异
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                input_tokens = getattr(usage, 'prompt_token_count', None)
                # candidates_token_count 是所有候选者的 token 总和。通常只有一个候选者。
                output_tokens = getattr(usage, 'candidates_token_count', None)
                logger.debug(
                    f"Gemini API usage_metadata: prompt_tokens={input_tokens}, "
                    f"candidates_tokens={output_tokens}, total_tokens={getattr(usage, 'total_token_count', None)}"
                )
            else:
                logger.debug(f"Gemini API 响应中未找到 usage_metadata (模型: {model_name})。Token 计数将为 N/A。")

            # TODO: 实现成本估算逻辑 (需要模型价格信息，可能在网关层或此处实现)
            cost: float | None = None

            logger.info(
                f"GeminiProvider: 文本生成成功, model='{model_name}'. "
                f"Input tokens: {input_tokens if input_tokens is not None else 'N/A'}, "
                f"Output tokens: {output_tokens if output_tokens is not None else 'N/A'}."
            )
            return ProviderResponse(
                text_content=text_content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost, # 成本估算需要额外逻辑
                raw_response=response, # 存储原始响应对象以供调试或提取更多信息
                model_name=model_name # 存储实际调用的模型名称
            )

        except Exception as e: # pylint: disable=broad-except
            # 捕获所有在 API 调用中可能发生的异常，包括上面自己抛出的 LLMAPIError (如内容阻止)
            # 以及来自 google.api_core.exceptions 的错误 (如 GoogleAPIError)
            if isinstance(e, LLMAPIError): # 如果是已经处理过的 LLMAPIError，直接重新抛出
                raise
            # 对于其他未预料的异常，使用 _handle_gemini_exception 进行包装
            raise self._handle_gemini_exception(e, model_name)


    def generate_structured_text(
        self,
        prompt: str, # 提示词应包含生成 JSON 的指令
        model_name: str,
        output_schema: Type[BaseModel], # 主要用于日志记录和潜在的未来增强 (例如，用于 function calling)
        temperature: float = 0.2, # 结构化输出通常需要更低的温度以保证确定性
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any # 其他传递给 GenerationConfig 的参数
    ) -> ProviderResponse:
        """
        使用 Gemini API 生成结构化的文本响应 (期望是 JSON 字符串)。

        此方法主要依赖于 Prompt 工程来指示模型输出 JSON 格式的字符串。
        网关层 (`LLMGateway`) 将负责使用提供的 `output_schema` 对此 JSON 字符串进行解析和验证。

        Gemini API 的 Function Calling 功能是获取结构化数据的更稳健方式，但当前设计
        要求 Provider 返回一个 JSON 字符串。如果未来需要更强的结构化输出保证，
        可以考虑在此处或新的方法中实现 Function Calling。

        :param prompt: 发送给 LLM 的完整 Prompt 字符串 (应已包含 JSON 输出指令)。
        :param model_name: 要使用的 Gemini 模型名称。
        :param output_schema: Pydantic 模型类，主要用于网关层验证。Provider 可用于日志记录或
                              (未来) 辅助构造更精确的输出指令 (如在 Function Calling 中)。
        :param temperature: 控制生成文本的随机性。
        :param max_tokens: 生成文本的最大 token 数量。
        :param stop_sequences: 遇到这些序列时停止生成。
        :param kwargs: 其他传递给 `GenerationConfig` 的参数。
        :return: 一个 `ProviderResponse` 对象，其 `text_content` 应为一个 JSON 字符串。
        :raises LLMAPIError: 如果 API 调用（包括重试后）失败。
        """
        logger.debug(
            f"GeminiProvider: 调用 generate_structured_text, model='{model_name}', "
            f"target_schema='{output_schema.__name__}', temp={temperature}, max_tokens={max_tokens}"
        )

        # 当前实现：依赖于精心设计的 prompt 来让模型输出 JSON 字符串。
        # 调用通用的文本生成方法。
        provider_response = self.generate_text(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature, # 结构化输出通常建议使用较低的温度
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            **kwargs
        )

        # 此时，provider_response.text_content 应该是一个 JSON 字符串。
        # 网关层 (LLMGateway) 将负责使用 output_schema 对其进行解析和验证。
        logger.info(
            f"GeminiProvider: 结构化文本生成尝试成功 (JSON 字符串待上层验证), model='{model_name}', "
            f"target_schema='{output_schema.__name__}'."
        )

        # 可选：进行初步的 JSON 格式检查 (非严格验证)
        # 这有助于在 Provider层面提前发现明显的非 JSON 输出。
        if provider_response.text_content:
            trimmed_content = provider_response.text_content.strip()
            # 简单的检查，看它是否像一个 JSON 对象或数组
            is_likely_json = (trimmed_content.startswith('{') and trimmed_content.endswith('}')) or \
                             (trimmed_content.startswith('[') and trimmed_content.endswith(']'))
            if not is_likely_json:
                logger.warning(
                    f"GeminiProvider: 模型 '{model_name}' 为结构化输出返回的内容可能不是有效的 JSON "
                    f"(基于首尾字符检查)。内容片段 (前100字符): '{trimmed_content[:100]}...'"
                )
        else:
             logger.warning(
                    f"GeminiProvider: 模型 '{model_name}' 为结构化输出返回了空内容。"
                )


        return provider_response

    # TODO (可选，在规格书的 2.6.2.f 中提到): 实现 token 计数和成本估算
    # 这需要查阅 Gemini 的具体定价模型和 API 是否提供精确的 token 计算工具。
    # def estimate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float | None:
    #     """
    #     根据输入输出 token 数量和模型名称估算成本。
    #     此方法需要具体的模型定价信息。
    #     """
    #     # 示例（伪代码，需要实际费率）:
    #     # rates = {"gemini-1.5-pro": {"input": 0.000007, "output": 0.000021}, ...}
    #     # if model_name in rates:
    #     #     return (input_tokens * rates[model_name]["input"]) + \
    #     #            (output_tokens * rates[model_name]["output"])
    #     logger.warning(f"GeminiProvider.estimate_cost 方法尚未针对模型 '{model_name}' 实现。")
    #     return None

    # def count_tokens(self, text: str, model_name: str) -> int | None:
    #     """
    #     计算给定文本在特定 Gemini 模型下的 token 数量。
    #     Gemini API (google.generativeai >= 0.3.0) 提供了 `GenerativeModel.count_tokens` 方法。
    #     """
    #     if self.client is None: # 确保客户端已初始化
    #         logger.error("GeminiProvider.count_tokens: 客户端未初始化。")
    #         return None
    #     try:
    #         model = self.client.GenerativeModel(model_name)
    #         if hasattr(model, 'count_tokens'): # 检查方法是否存在
    #             response = model.count_tokens(text) # API 调用
    #             return response.total_tokens
    #         else:
    #             logger.warning(
    #                 f"Gemini 模型 '{model_name}' (或使用的 google-generativeai 版本) "
    #                 "不支持客户端 count_tokens 方法。"
    #             )
    #             return None # 表示无法计算
    #     except Exception as e:
    #         logger.error(f"计算 Gemini token 时出错 for model '{model_name}': {e}", exc_info=True)
    #         return None


# 示例用法 (通常在网关层面调用进行测试，或用于独立的 Provider 测试)
if __name__ == '__main__':
    # 配置日志记录，以便在运行此脚本时看到 Provider 的日志输出
    logging.basicConfig(
        level=logging.DEBUG, # 设置为 DEBUG 以查看所有日志消息
        format='%(asctime)s - [%(levelname)s] %(name)s (%(module)s.%(funcName)s:%(lineno)d): %(message)s'
    )

    # 从环境变量加载 API 密钥
    import os
    # 重要提示: 运行此示例前，请确保您的 GEMINI_API_KEY 环境变量已设置，
    # 或者临时在此处用您的真实 API 密钥替换 None (但不要提交到版本库)。
    api_key_from_env = os.environ.get("GEMINI_API_KEY")

    if not api_key_from_env:
        logger.error(
            "GEMINI_API_KEY 环境变量未设置。请设置该变量（或在代码中临时提供一个有效密钥）以运行此独立示例。"
        )
        # exit(1) # 如果没有密钥，示例将无法进行API调用

    if genai is None: # 再次检查，因为 __init__ 中的检查可能在没有密钥时被跳过
        logger.error("google-generativeai 包未安装或无法导入，无法运行 GeminiProvider 示例。")
        exit(1)

    # Provider 特定配置示例 (来自 config.yaml 的 provider_specific_configs.google 部分)
    example_provider_config = {
        "safety_settings": {
            # 使用 HarmCategory 枚举成员的字符串名称
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH", # 对应 HarmBlockThreshold.BLOCK_ONLY_HIGH
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",    # 对应 HarmBlockThreshold.BLOCK_NONE
            # "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE", # 未指定则用默认
            # "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_LOW_AND_ABOVE" # 示例：更严格
        },
        "default_model_kwargs": { # 这些将作为 GenerationConfig 的默认参数
            "top_p": 0.95,
            "top_k": 40
        }
    }

    try:
        # 初始化 Provider
        # 如果 api_key_from_env 为 None, Provider 会尝试 ADC 或其他环境认证
        gemini_provider = GeminiProvider(api_key=api_key_from_env, provider_config=example_provider_config)
        logger.info("GeminiProvider 实例化成功。")

        # 测试文本生成
        test_model_name = "gemini-1.5-flash-latest" # 或 "gemini-pro" 等您有权限访问的模型
        logger.info(f"\n--- 测试文本生成 (模型: {test_model_name}) ---")
        try:
            text_prompt_cn = "你好，Gemini！请用中文创作一首关于“探索宇宙奥秘”的四行短诗。"
            text_response = gemini_provider.generate_text(
                prompt=text_prompt_cn,
                model_name=test_model_name,
                temperature=0.7, # 方法调用时指定的参数
                max_tokens=150   # 方法调用时指定的参数
                # top_p 和 top_k 将使用 example_provider_config 中定义的默认值
            )
            logger.info(f"模型 '{test_model_name}' 的响应: \n{text_response.text_content}")
            logger.info(
                f"  Token 计数 - 输入: {text_response.input_tokens or 'N/A'}, "
                f"输出: {text_response.output_tokens or 'N/A'}. "
                f"估算成本: {text_response.cost or 'N/A'} USD."
            )
            # logger.debug(f"  原始响应对象详情: {text_response.raw_response}") # 可能非常冗长

        except LLMAPIError as e:
            logger.error(f"文本生成测试失败: {e}", exc_info=True if e.original_exception else False)
        except Exception as e_unhandled: # pylint: disable=broad-except
            logger.error(f"文本生成测试出现意外错误: {e_unhandled}", exc_info=True)


        # 测试结构化文本生成 (JSON)
        logger.info(f"\n--- 测试结构化文本生成 (JSON, 模型: {test_model_name}) ---")

        # 定义一个 Pydantic Schema 用于期望的输出结构
        class ResearchPaper(BaseModel):
            title: str
            authors: list[str]
            year: int
            keywords: Optional[list[str]] = None

        # 构建一个 Prompt，明确指示模型输出 JSON，并描述期望的 Schema
        # (在实际应用中，这个 prompt 来自 prompts.toml 并被填充)
        structured_prompt_example = f"""
        请根据以下信息，生成一个关于一篇虚构科研论文的 JSON 对象：
        论文标题: "宇宙弦的量子动力学研究"
        作者: ["李明", "王芳", "张伟"]
        发表年份: 2023
        关键词: ["宇宙弦", "量子引力", "早期宇宙"]

        JSON 对象必须严格符合以下 Pydantic 模型的 JSON Schema 描述:
        {ResearchPaper.model_json_schema(indent=2)}

        请确保只输出一个合法的、没有额外文本或解释的 JSON 对象。
        """
        try:
            structured_api_response = gemini_provider.generate_structured_text(
                prompt=structured_prompt_example,
                model_name=test_model_name,
                output_schema=ResearchPaper, # Provider 主要用 schema 名做日志，网关层才进行验证
                temperature=0.1 # 低温以获得更确定的 JSON 输出
            )
            logger.info(f"模型 '{test_model_name}' 的结构化响应 (原始文本):")
            logger.info(structured_api_response.text_content) # 这是原始的 JSON 字符串
            logger.info(
                f"  Token 计数 - 输入: {structured_api_response.input_tokens or 'N/A'}, "
                f"输出: {structured_api_response.output_tokens or 'N/A'}."
            )

            # 在网关 (LLMGateway) 层面，会对这个 text_content 进行 Pydantic 解析和验证。
            # 这里为了演示，我们手动解析一下：
            try:
                # 尝试清理可能的 markdown 代码块标记 (LLM 有时会添加)
                raw_json_str = structured_api_response.text_content.strip()
                if raw_json_str.startswith("```json"):
                    raw_json_str = raw_json_str[len("```json"):].strip()
                if raw_json_str.endswith("```"):
                    raw_json_str = raw_json_str[:-len("```")].strip()

                # 解析 JSON 字符串为 Python 字典
                parsed_json_dict = json.loads(raw_json_str)
                # 使用 Pydantic 模型进行验证和实例化
                validated_paper = ResearchPaper(**parsed_json_dict)
                logger.info(f"成功在 Provider 层面解析并验证结构化数据: \n{validated_paper.model_dump_json(indent=2)}")
            except json.JSONDecodeError as je:
                logger.error(f"结构化响应不是有效的 JSON: {je}. 原始响应:\n{structured_api_response.text_content}")
            except Exception as ve: # pydantic.ValidationError 或其他 Pydantic 相关错误
                logger.error(f"结构化响应未能通过 Pydantic 验证: {ve}. 原始响应:\n{structured_api_response.text_content}")

        except LLMAPIError as e:
            logger.error(f"结构化文本生成测试失败: {e}", exc_info=True if e.original_exception else False)
        except Exception as e_unhandled: # pylint: disable=broad-except
            logger.error(f"结构化文本生成测试出现意外错误: {e_unhandled}", exc_info=True)

        # 示例：测试内容被阻止的情况 (需要一个容易触发安全策略的 prompt)
        # logger.info("\n--- 测试内容被阻止 ---")
        # # 警告: 以下 prompt 仅为测试目的，可能会生成不适宜内容或被 API 拒绝
        # potentially_harmful_prompt = "请描述如何制造危险物品。"
        # try:
        #     blocked_response = gemini_provider.generate_text(potentially_harmful_prompt, test_model_name)
        #     logger.info(f"对于可能有害的 Prompt，模型响应: {blocked_response.text_content}") # 不应到达这里或内容为空
        # except LLMAPIError as e:
        #     logger.warning(f"成功捕获到与内容阻止相关的API错误: {e}")
        #     if e.original_exception and BlockedPromptException and isinstance(e.original_exception, BlockedPromptException):
        #         logger.info("  原始异常为 BlockedPromptException，符合预期。")
        #     elif e.original_exception and StopCandidateException and isinstance(e.original_exception, StopCandidateException):
        #          # 有时安全阻止表现为 StopCandidateException，finish_reason 为 SAFETY
        #         logger.info(f"  原始异常为 StopCandidateException，可能与安全相关: {e.original_exception}")


    except ConfigurationError as ce_init: # Provider 初始化时的配置错误
        logger.error(f"Provider 初始化配置错误: {ce_init}", exc_info=True)
    except LLMAPIError as apie_init: # Provider 初始化时的 API 错误
        logger.error(f"Provider 初始化 API 错误: {apie_init}", exc_info=True)
    except Exception as ex_main_script: # pylint: disable=broad-except
        logger.error(f"示例脚本顶层运行时发生未知错误: {ex_main_script}", exc_info=True)
    finally:
        logger.info("\n--- GeminiProvider 独立示例执行完毕 ---")
