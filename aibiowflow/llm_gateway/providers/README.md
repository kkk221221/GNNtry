# LLM 提供商适配器 (providers)

本目录包含用于与不同大型语言模型 (LLM) 提供商 API 进行交互的适配器类。

## 设计目标

*   **抽象化**: 每个提供商的特定实现细节都被封装在其各自的适配器类中。
*   **可扩展性**: 添加对新 LLM 提供商的支持，只需在此目录中创建一个新的适配器类，该类继承自 `BaseProvider` 并实现其抽象方法。
*   **统一接口**: 所有适配器都遵循 `BaseProvider` 定义的统一接口，使得网关核心逻辑可以以一致的方式与它们交互。

## 文件说明

*   `__init__.py`: Python 子包初始化文件。
*   `base_provider.py`: 定义了 `BaseProvider` 抽象基类，所有具体的 LLM 提供商适配器都必须从此类继承。
*   `gemini_provider.py`: 针对 Google Gemini 系列模型的提供商适配器实现。
*   `anthropic_provider.py` (示例/未来添加): 针对 Anthropic Claude 系列模型的提供商适配器实现。
*   `openai_provider.py` (示例/未来添加): 针对 OpenAI GPT 系列模型的提供商适配器实现。

## 完成状态

- [ ] 定义 `BaseProvider` 抽象基类 (`base_provider.py`)。
- [ ] 实现 `GeminiProvider` (`gemini_provider.py`)。
    - [ ] 实现文本生成方法。
    - [ ] 实现结构化数据生成支持（如果 API 直接支持，或通过 prompt 工程）。
    - [ ] 实现错误处理和映射到网关的自定义异常。
    - [ ] 实现 token 计数和成本估算（如果 API 支持）。
- [ ] (可选) 为其他提供商（如 Anthropic, OpenAI）实现适配器。

## 注意事项

*   适配器应负责处理其对应 LLM API 的所有通信细节，包括认证、请求构建、响应解析和错误处理。
*   API 密钥等敏感信息不应硬编码在适配器中，而应通过构造函数从网关配置中传入。
*   所有代码注释需使用中文。
