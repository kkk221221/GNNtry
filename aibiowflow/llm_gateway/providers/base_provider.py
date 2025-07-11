# -*- coding: utf-8 -*-
"""
LLM 提供商适配器的抽象基类。

定义了所有具体 LLM 提供商适配器必须实现的通用接口。
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Type, Any, Dict, Tuple, AsyncGenerator, Generator

class ProviderResponse:
    """
    封装来自 LLM 提供商的响应信息。
    """
    def __init__(self,
                 text_content: str,
                 input_tokens: int | None = None,
                 output_tokens: int | None = None,
                 cost: float | None = None,
                 raw_response: Any | None = None,
                 model_name: str | None = None):
        self.text_content = text_content # LLM 生成的主要文本内容
        self.input_tokens = input_tokens # 输入消耗的 token 数 (如果可用)
        self.output_tokens = output_tokens # 输出消耗的 token 数 (如果可用)
        self.cost = cost # 本次调用的估算成本 (如果可用)
        self.raw_response = raw_response # 提供商返回的原始响应对象 (用于调试或特定用途)
        self.model_name = model_name # 实际调用的模型名称

class BaseProvider(ABC):
    """
    抽象基类，定义了 LLM 提供商适配器的标准接口。
    每个具体的 LLM 提供商（如 Gemini, OpenAI, Anthropic）都应创建一个
    继承自此类并实现其抽象方法的适配器。
    """

    @abstractmethod
    def __init__(self, api_key: str, provider_config: Dict[str, Any] | None = None):
        """
        初始化提供商适配器。

        :param api_key: 用于认证的 API 密钥。
        :param provider_config: 特定于此提供商的额外配置项 (来自网关主配置文件)。
        """
        pass

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any
    ) -> ProviderResponse:
        """
        生成通用文本响应。

        :param prompt: 发送给 LLM 的完整 Prompt 字符串。
        :param model_name: 要使用的具体模型名称 (例如 "gemini-1.5-pro")。
        :param temperature: 控制生成文本的随机性，值越高越随机。
        :param max_tokens: 生成文本的最大 token 数量。
        :param stop_sequences: 遇到这些序列时停止生成。
        :param kwargs: 其他特定于提供商的参数。
        :return: 一个 ProviderResponse 对象，包含生成的文本和元数据。
        :raises LLMAPIError: 如果 API 调用失败。
        """
        pass

    @abstractmethod
    def generate_structured_text(
        self,
        prompt: str,
        model_name: str,
        output_schema: Type[BaseModel],
        temperature: float = 0.2, # 结构化输出通常需要较低的 temperature
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any
    ) -> ProviderResponse:
        """
        生成结构化文本响应 (期望是 JSON，后续由网关用 Pydantic 解析)。

        提供商实现应尽力确保 LLM 返回符合 schema 描述的 JSON。
        这可能通过特定的 API 功能 (如 OpenAI 的 JSON mode) 或通过 Prompt 工程实现。

        :param prompt: 发送给 LLM 的完整 Prompt 字符串 (可能已包含 JSON 输出指令)。
        :param model_name: 要使用的具体模型名称。
        :param output_schema: Pydantic 模型类，用于指导 LLM 生成的结构。
                              提供商可以用它来构造更精确的输出指令。
        :param temperature: 控制生成文本的随机性。
        :param max_tokens: 生成文本的最大 token 数量。
        :param stop_sequences: 遇到这些序列时停止生成。
        :param kwargs: 其他特定于提供商的参数。
        :return: 一个 ProviderResponse 对象，其 text_content 应为 JSON 字符串。
        :raises LLMAPIError: 如果 API 调用失败。
        """
        pass

    # 可选: 流式响应接口
    # @abstractmethod
    # def stream_text(
    #     self,
    #     prompt: str,
    #     model_name: str,
    #     temperature: float = 0.7,
    #     max_tokens: int | None = None,
    #     stop_sequences: list[str] | None = None,
    #     **kwargs: Any
    # ) -> Generator[ProviderResponse, None, None]: # 或者 AsyncGenerator
    #     """
    #     以流式方式生成文本响应。
    #
    #     每次产出 (yield) 一个 ProviderResponse 对象，其中 text_content 是当前块的文本。
    #     最后的 ProviderResponse 可能包含累积的 token 计数和成本。
    #
    #     :return: 一个生成器，逐步产出文本块。
    #     :raises LLMAPIError: 如果 API 调用失败。
    #     """
    #     pass

    # 可选: 获取 token 数量和成本估算的方法
    # @abstractmethod
    # def estimate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float | None:
    #     """
    #     根据输入输出 token 数量和模型名称估算成本。
    #
    #     :param input_tokens: 输入 token 数量。
    #     :param output_tokens: 输出 token 数量。
    #     :param model_name: 模型名称。
    #     :return: 估算的成本（例如美元），如果无法估算则返回 None。
    #     """
    #     pass
    #
    # @abstractmethod
    # def count_tokens(self, text: str, model_name: str) -> int | None:
    #     """
    #     计算给定文本在特定模型下的 token 数量。
    #
    #     :param text: 要计算 token 的文本。
    #     :param model_name: 模型名称。
    #     :return: token 数量，如果无法计算则返回 None。
    #     """
    #     pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        返回提供商的唯一名称/标识符 (例如 "google", "openai", "anthropic")。
        """
        pass
