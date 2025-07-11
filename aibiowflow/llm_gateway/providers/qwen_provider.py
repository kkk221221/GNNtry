# -*- coding: utf-8 -*-
"""
阿里云通义千问 (Qwen) LLM 提供商适配器实现。

此类实现了 BaseProvider 接口，用于与阿里云通义千问模型进行交互，
优先使用其 OpenAI 兼容 API 模式。
"""

import logging
import os
from pydantic import BaseModel # type: ignore
from typing import Type, Any, Dict, Optional

# 尝试导入 openai SDK
try:
    from openai import OpenAI, APIError, APITimeoutError, APIConnectionError, APIStatusError # type: ignore
    # 更多具体的 OpenAI 异常类型可以按需导入
except ImportError:
    OpenAI = None # type: ignore
    APIError = None # type: ignore
    APITimeoutError = None # type: ignore
    APIConnectionError = None # type: ignore
    APIStatusError = None # type: ignore


from ..exceptions import LLMAPIError, ConfigurationError
from .base_provider import BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)

# 通义千问 OpenAI 兼容 API 的默认 Base URL
DEFAULT_QWEN_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# 从环境变量读取 API Key 的名称
DASHSCOPE_API_KEY_ENV_VAR = "DASHSCOPE_API_KEY"


class QwenProvider(BaseProvider):
    """
    阿里云通义千问 (Qwen) LLM 提供商适配器。

    使用 OpenAI 兼容模式与 Qwen API 进行交互。
    """
    PROVIDER_ID = "qwen" # 提供商的唯一标识符

    def __init__(self, api_key: str | None = None, provider_config: Dict[str, Any] | None = None):
        """
        初始化 Qwen 提供商适配器。

        :param api_key: 阿里云 DashScope API 密钥。如果为 None，则尝试从环境变量 DASHSCOPE_API_KEY 读取。
        :param provider_config: 特定于此提供商的额外配置项（来自网关的主配置文件）。
                                 支持的键包括:
                                 - `base_url` (str): 覆盖默认的 OpenAI 兼容 API 的 base_url。
                                 - `default_model_kwargs` (Dict[str, Any]): 将作为默认参数传递给
                                   OpenAI client.chat.completions.create 方法的额外参数。
        :raises ConfigurationError: 如果 `openai` SDK 未安装。
        :raises LLMAPIError: 如果 API 密钥未提供且无法从环境变量加载。
        """
        if OpenAI is None:
            msg = ("openai 包未安装。请运行 'pip install openai' 来安装，以便使用 QwenProvider。")
            logger.error(msg)
            raise ConfigurationError(msg)

        self.provider_config = provider_config or {}
        resolved_api_key = api_key or os.getenv(DASHSCOPE_API_KEY_ENV_VAR)

        if not resolved_api_key:
            msg = (f"Qwen API 密钥未提供，且无法从环境变量 {DASHSCOPE_API_KEY_ENV_VAR} 加载。")
            logger.error(msg)
            raise LLMAPIError(msg) # 或者 ConfigurationError，但 LLMAPIError 更符合密钥问题

        self.base_url = self.provider_config.get("base_url", DEFAULT_QWEN_COMPATIBLE_BASE_URL)
        self.default_model_kwargs = self.provider_config.get("default_model_kwargs", {})

        try:
            self.client = OpenAI(
                api_key=resolved_api_key,
                base_url=self.base_url
            )
            logger.info(f"QwenProvider 初始化成功。Base URL: {self.base_url}")
        except Exception as e: # pylint: disable=broad-except
            msg = f"初始化 OpenAI 客户端以用于 Qwen Provider 失败: {e}"
            logger.error(msg, exc_info=True)
            # 根据 OpenAI SDK 的具体异常类型，可能需要更细致的处理
            raise LLMAPIError(msg, original_exception=e) from e


    @property
    def provider_name(self) -> str:
        """返回此提供商的唯一标识符。"""
        return self.PROVIDER_ID

    def _handle_api_exception(self, e: Exception, model_name: str) -> LLMAPIError:
        """
        统一处理来自 OpenAI SDK (用于 Qwen) 调用的异常，将其包装为 LLMAPIError。
        """
        status_code: Optional[int] = None
        if isinstance(e, APIStatusError): # APIStatusError 是 APIError 的子类，包含 status_code
            status_code = e.status_code
            error_message = f"Qwen API 调用失败 (模型: {model_name}, HTTP Status: {status_code}): {e.message or str(e)}"
        elif isinstance(e, APIError): # 其他 APIError，可能没有 status_code
            error_message = f"Qwen API 调用遭遇问题 (模型: {model_name}): {e.message or str(e)}"
        elif isinstance(e, APITimeoutError):
            error_message = f"Qwen API 调用超时 (模型: {model_name}): {str(e)}"
        elif isinstance(e, APIConnectionError):
            error_message = f"Qwen API 连接失败 (模型: {model_name}): {str(e)}"
        else: # 其他未知异常
            error_message = f"与 Qwen API (模型: {model_name}) 通信时发生未知错误: {str(e)}"

        logger.error(error_message, exc_info=True)
        return LLMAPIError(error_message, status_code=status_code, original_exception=e)

    def generate_text(
        self,
        prompt: str, # 在 OpenAI 兼容模式下，这通常是 user message content
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        system_prompt: str | None = None, # 新增: 允许传入 system prompt
        messages_override: list[Dict[str, str]] | None = None, # 新增: 允许完全覆盖 messages
        **kwargs: Any
    ) -> ProviderResponse:
        """
        使用 Qwen API (OpenAI 兼容模式) 生成通用文本响应。

        :param prompt: 用户的主要输入/提示。如果提供了 `messages_override`，此参数将被忽略。
                       如果未提供 `messages_override` 但提供了 `system_prompt`，此参数将作为 user message。
                       如果均未提供，此参数将作为唯一的 user message。
        :param model_name: 要使用的 Qwen 模型名称 (例如 "qwen-plus")。
        :param temperature: 控制生成文本的随机性。
        :param max_tokens: 生成文本的最大 token 数量。
        :param stop_sequences: 遇到这些序列时停止生成 (OpenAI SDK 中的 `stop` 参数)。
        :param system_prompt: (可选) 系统级指令。如果提供，将作为 messages 列表的第一个元素。
        :param messages_override: (可选) 完全自定义的 messages 列表，用于多轮对话或复杂场景。
                                  如果提供，`prompt` 和 `system_prompt` 参数将被忽略。
                                  格式应为 `[{"role": "system/user/assistant", "content": "..."}, ...]`
        :param kwargs: 其他传递给 OpenAI `chat.completions.create` 方法的参数 (如 `top_p`)。
        :return: 一个 ProviderResponse 对象，包含生成的文本和元数据。
        :raises LLMAPIError: 如果 API 调用失败。
        """
        logger.debug(
            f"QwenProvider: 调用 generate_text, model='{model_name}', temp={temperature}, "
            f"max_tokens={max_tokens}, stop_sequences={stop_sequences is not None}, "
            f"system_prompt_present={system_prompt is not None}, "
            f"messages_override_present={messages_override is not None}, kwargs={kwargs}"
        )

        if messages_override:
            messages = messages_override
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        request_params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            **self.default_model_kwargs, # 合并来自 provider_config 的默认参数
            **kwargs # 合并来自方法调用的运行时参数
        }
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        if stop_sequences is not None: # OpenAI SDK 的 'stop' 参数可以是 str, list[str], or None
            request_params["stop"] = stop_sequences

        try:
            completion = self.client.chat.completions.create(**request_params)

            text_content = ""
            if completion.choices and completion.choices[0].message:
                text_content = completion.choices[0].message.content or ""

            input_tokens: Optional[int] = None
            output_tokens: Optional[int] = None
            if completion.usage:
                input_tokens = completion.usage.prompt_tokens
                output_tokens = completion.usage.completion_tokens
                logger.debug(
                    f"Qwen API usage: prompt_tokens={input_tokens}, "
                    f"completion_tokens={output_tokens}, total_tokens={completion.usage.total_tokens}"
                )
            else:
                logger.debug(f"Qwen API 响应中未找到 usage 信息 (模型: {model_name})。Token 计数将为 N/A。")

            # TODO: 实现成本估算 (基于模型列表和 token 数)
            cost: Optional[float] = None

            logger.info(
                f"QwenProvider: 文本生成成功, model='{model_name}'. "
                f"Input tokens: {input_tokens or 'N/A'}, Output tokens: {output_tokens or 'N/A'}."
            )
            return ProviderResponse(
                text_content=text_content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                raw_response=completion.model_dump(), # 存储原始响应的字典表示
                model_name=model_name
            )

        except Exception as e: # pylint: disable=broad-except
            if isinstance(e, LLMAPIError): # 如果是已经处理过的 LLMAPIError，直接重新抛出
                raise
            raise self._handle_api_exception(e, model_name)


    def generate_structured_text(
        self,
        prompt: str,
        model_name: str,
        output_schema: Type[BaseModel],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        system_prompt: str | None = None,
        messages_override: list[Dict[str, str]] | None = None,
        **kwargs: Any
    ) -> ProviderResponse:
        """
        使用 Qwen API (OpenAI 兼容模式) 生成结构化的文本响应 (期望是 JSON 字符串)。

        此方法通过设置 `response_format={"type": "json_object"}` 来指示模型输出 JSON。
        Prompt 工程也应配合，明确指示模型遵循 JSON 格式和 schema。

        :param prompt: 用户的主要输入/提示。应包含生成 JSON 的明确指令。
        :param model_name: 要使用的 Qwen 模型名称。
        :param output_schema: Pydantic 模型类，用于指导 LLM 生成的结构 (主要用于上层验证)。
        :param temperature: 控制生成文本的随机性。
        :param max_tokens: 生成文本的最大 token 数量。
        :param stop_sequences: 遇到这些序列时停止生成。
        :param system_prompt: (可选) 系统级指令。
        :param messages_override: (可选) 完全自定义的 messages 列表。
        :param kwargs: 其他传递给 OpenAI `chat.completions.create` 方法的参数。
        :return: 一个 ProviderResponse 对象，其 text_content 应为一个 JSON 字符串。
        :raises LLMAPIError: 如果 API 调用失败。
        """
        logger.debug(
            f"QwenProvider: 调用 generate_structured_text, model='{model_name}', "
            f"target_schema='{output_schema.__name__}', temp={temperature}, max_tokens={max_tokens}"
        )

        # 确保 prompt 中包含生成 JSON 的指令
        # 例如："请根据以下描述生成一个 JSON 对象... 必须严格符合指定的 JSON Schema。"
        # (这个职责通常在调用者构建 prompt 时完成)

        # 为结构化输出设置 response_format
        # 将其合并到 kwargs 中，如果 kwargs 中已存在，则 kwargs 中的优先
        current_kwargs = {"response_format": {"type": "json_object"}, **kwargs}

        provider_response = self.generate_text(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            system_prompt=system_prompt,
            messages_override=messages_override,
            **current_kwargs
        )

        logger.info(
            f"QwenProvider: 结构化文本生成尝试成功 (JSON 字符串待上层验证), "
            f"model='{model_name}', target_schema='{output_schema.__name__}'."
        )
        # 初步的 JSON 格式检查 (可选, 已在 GeminiProvider 中存在类似逻辑)
        if provider_response.text_content:
            trimmed_content = provider_response.text_content.strip()
            is_likely_json = (trimmed_content.startswith('{') and trimmed_content.endswith('}')) or \
                             (trimmed_content.startswith('[') and trimmed_content.endswith(']'))
            if not is_likely_json:
                logger.warning(
                    f"QwenProvider: 模型 '{model_name}' 为结构化输出返回的内容可能不是有效的 JSON "
                    f"(基于首尾字符检查)。内容片段 (前100字符): '{trimmed_content[:100]}...'"
                )
        else:
             logger.warning(
                    f"QwenProvider: 模型 '{model_name}' 为结构化输出返回了空内容。"
                )

        return provider_response

    # TODO: (可选) 实现 token 计数 (count_tokens) 和成本估算 (estimate_cost)
    # count_tokens 可以尝试使用 OpenAI 的 tiktoken 库，但需要 Qwen 兼容的 tokenizer 名称
    # estimate_cost 需要模型价格表

# 示例用法 (通常在网关层面调用进行测试，或用于独立的 Provider 测试)
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - [%(levelname)s] %(name)s (%(module)s.%(funcName)s:%(lineno)d): %(message)s'
    )

    # 确保设置了 DASHSCOPE_API_KEY 环境变量
    if not os.getenv(DASHSCOPE_API_KEY_ENV_VAR):
        logger.error(
            f"{DASHSCOPE_API_KEY_ENV_VAR} 环境变量未设置。请设置该变量以运行此独立示例。"
        )
        exit(1)

    if OpenAI is None:
        logger.error("OpenAI SDK 未安装，无法运行 QwenProvider 示例。")
        exit(1)

    try:
        qwen_provider = QwenProvider() # 使用环境变量中的 API Key
        logger.info("QwenProvider 实例化成功。")

        # --- 测试文本生成 ---
        test_model = "qwen-plus" # 或其他可用的 Qwen 模型，如 qwen-turbo, qwen-max
        logger.info(f"\n--- 测试文本生成 (模型: {test_model}) ---")
        try:
            text_prompt = "你好，通义千问！请用中文写一首关于春天的五言绝句。"
            text_response = qwen_provider.generate_text(
                prompt=text_prompt,
                model_name=test_model,
                temperature=0.7,
                max_tokens=100
            )
            logger.info(f"模型 '{test_model}' 的文本响应: \n{text_response.text_content}")
            logger.info(
                f"  Token 计数 - 输入: {text_response.input_tokens or 'N/A'}, "
                f"输出: {text_response.output_tokens or 'N/A'}."
            )
            # logger.debug(f"  原始响应对象详情: {text_response.raw_response}")

        except LLMAPIError as e:
            logger.error(f"文本生成测试失败: {e}", exc_info=True if e.original_exception else False)
        except Exception as e_unhandled:
            logger.error(f"文本生成测试出现意外错误: {e_unhandled}", exc_info=True)

        # --- 测试结构化文本生成 (JSON) ---
        logger.info(f"\n--- 测试结构化文本生成 (JSON, 模型: {test_model}) ---")

        class Book(BaseModel):
            title: str
            author: str
            year: int
            genres: list[str]

        # Prompt 中应包含生成 JSON 的指令
        structured_prompt = f"""
        请根据以下信息，生成一个关于一本书的 JSON 对象：
        书名: "三体"
        作者: "刘慈欣"
        出版年份: 2008
        类型: ["科幻", "小说"]

        JSON 对象必须严格符合以下 Pydantic 模型的 JSON Schema 描述:
        {Book.model_json_schema(indent=2)}

        请确保只输出一个合法的、没有额外文本或解释的 JSON 对象。
        """
        try:
            structured_response_obj = qwen_provider.generate_structured_text(
                prompt=structured_prompt,
                model_name=test_model,
                output_schema=Book,
                temperature=0.1
            )
            logger.info(f"模型 '{test_model}' 的结构化响应 (原始文本):")
            logger.info(structured_response_obj.text_content)
            logger.info(
                f"  Token 计数 - 输入: {structured_response_obj.input_tokens or 'N/A'}, "
                f"输出: {structured_response_obj.output_tokens or 'N/A'}."
            )

            # 尝试在 Provider 层面解析 (通常这步在 Gateway 完成)
            try:
                import json
                parsed_data = json.loads(structured_response_obj.text_content)
                validated_book = Book(**parsed_data)
                logger.info(f"成功在 Provider 层面解析并验证结构化数据: \n{validated_book.model_dump_json(indent=2)}")
            except Exception as parse_exc:
                logger.error(f"在 Provider 层面解析或验证 JSON 失败: {parse_exc}")

        except LLMAPIError as e:
            logger.error(f"结构化文本生成测试失败: {e}", exc_info=True if e.original_exception else False)
        except Exception as e_unhandled:
            logger.error(f"结构化文本生成测试出现意外错误: {e_unhandled}", exc_info=True)

    except LLMAPIError as apie_init:
        logger.error(f"QwenProvider 初始化 API 错误: {apie_init}", exc_info=True)
    except ConfigurationError as ce_init:
        logger.error(f"QwenProvider 初始化配置错误: {ce_init}", exc_info=True)
    except Exception as ex_main_script:
        logger.error(f"示例脚本顶层运行时发生未知错误: {ex_main_script}", exc_info=True)
    finally:
        logger.info("\n--- QwenProvider 独立示例执行完毕 ---")
