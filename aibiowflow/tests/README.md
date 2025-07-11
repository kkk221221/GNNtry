# 测试目录 (tests)

本目录包含 AIBioFlow 项目的所有单元测试和集成测试。

## 子目录和文件说明

*   `__init__.py`: 将此目录标记为 Python 包。
*   `test_llm_gateway_config.py`: LLM 网关配置加载相关的单元测试。
*   `test_llm_gateway_prompts.py`: LLM 网关 Prompt 处理相关的单元测试。
*   `test_llm_gateway_routing.py`: LLM 网关模型别名路由逻辑的单元测试。
*   `test_llm_gateway_retries.py`: LLM 网关重试机制相关的单元测试。
*   `test_llm_gateway_responses.py`: LLM 网关获取文本和结构化响应（包括 Provider 交互和 Pydantic 验证）的单元测试。
*   `test_gemini_provider.py`: (可选/如果需要) 针对 Gemini Provider 的特定单元测试（可能需要更复杂的模拟）。
*   `test_qwen_provider.py`: 针对 Qwen Provider 的单元测试。

## 运行测试

可以使用 Python 的 `unittest` 模块来运行测试。在项目根目录下执行：

```bash
python -m unittest discover -s tests -p "test_*.py"
```
或者，如果使用像 `pytest` 这样的测试运行器：
```bash
pytest tests/
```

## 完成状态

- [x] 为配置加载编写测试。
- [x] 为 Prompt 模板渲染编写测试。
- [x] 为模型别名路由编写测试。
- [x] 为 `get_text_response` (模拟 Provider) 编写测试。
- [x] 为 `get_structured_response` (模拟 Provider 和 Pydantic 验证) 编写测试。
- [x] 为错误处理 (`LLMOutputValidationError`, `LLMAPIError`) 编写测试。
- [x] 为重试机制编写测试。

## 注意事项

*   测试应尽可能独立，避免依赖外部服务。使用 `unittest.mock` 来模拟外部依赖（如 LLM API 调用）。
*   测试文件名应以 `test_` 开头。
*   测试方法名也应以 `test_` 开头。
*   所有代码注释和文档字符串应使用中文。
