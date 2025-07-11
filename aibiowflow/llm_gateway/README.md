# LLM 网关模块 (llm_gateway)

本目录包含 LLM 网关服务的所有代码和配置文件。

## 文件和子目录说明

*   `__init__.py`: Python 包初始化文件。
*   `gateway.py`: 包含 `LLMGateway` 类的核心实现。
*   `prompts.toml`: 存储 Prompt 模板。
*   `config.yaml`: 网关的配置文件，包括模型定义、API 密钥引用和重试策略。
*   `exceptions.py`: 定义模块特定的自定义异常。
*   `providers/`: 包含与不同 LLM 提供商交互的适配器。
    *   `providers/README.md`: `providers` 子目录的说明文件。
    *   `providers/__init__.py`: `providers` 子包初始化文件。
    *   `providers/base_provider.py`: 定义 LLM 提供商适配器的基类。
    *   `providers/gemini_provider.py`: Google Gemini LLM 提供商的实现。
    *   `providers/qwen_provider.py`: 阿里云通义千问 (Qwen) LLM 提供商的实现。

## 完成状态

- [x] 创建目录结构和基础文件。
- [x] 定义自定义异常 (`exceptions.py`)。
- [x] 实现配置加载 (`config.yaml`, `prompts.toml`)。
- [x] 实现 `BaseProvider` 抽象类。
- [x] 实现 `GeminiProvider` 类。
- [x] 实现 `LLMGateway` 核心逻辑。
- [x] 实现 `get_text_response` 方法。
- [x] 实现 `get_structured_response` 方法。
- [x] 编写单元测试 (覆盖核心功能，如配置、prompts、响应、重试、路由)。
- [x] 添加完整的代码注释和文档 (主要模块和类已有较详细中文注释)。

## 注意事项

所有代码注释需使用中文。
