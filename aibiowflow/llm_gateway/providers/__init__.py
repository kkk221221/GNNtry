# -*- coding: utf-8 -*-
"""
LLM 提供商适配器子包。

该包包含与具体 LLM 服务提供商（如 Google Gemini, Anthropic Claude 等）
进行交互的模块化代码。每个提供商都有其自己的实现，但都遵循一个共同的
`BaseProvider` 接口。
"""

from .base_provider import BaseProvider
from .gemini_provider import GeminiProvider
# 当添加更多 providers 时，在这里导入它们
# from .anthropic_provider import AnthropicProvider
# from .openai_provider import OpenAIProvider

__all__ = [
    "BaseProvider",
    "GeminiProvider",
    # "AnthropicProvider",
    # "OpenAIProvider",
]
